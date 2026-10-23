//! The tenants a node lately served, kept in a list in its cache directory
//! across a restart, so that it opens copies of them as it starts, before
//! anyone reads them (burst plan F7).
//!
//! The list holds tenants' names, and whether this node wrote for each,
//! nothing else. A tenant it wrote for is not warmed: a copy beside the
//! writer it opens when the tenant is asked for would hold the tenant in
//! memory twice. Warming keeps to a share of the memory budget, and a copy
//! it opens counts as not used, the coldest the node holds, until someone
//! reads it.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use vellev_core::TenantId;

use crate::{Inner, Node};

/// Of the memory budget, how much warming may fill, as a fraction: the
/// rest is room for the tenants the node is asked for while it warms, and
/// for the copies it warmed to grow as they follow their writers.
const WARM_UP_TO: (usize, usize) = (3, 4);

/// The most bytes a list of a tenant may take: its name, at most 63
/// characters (see [`TenantId`]), and what it is written in.
const BYTES_A_TENANT: u64 = 63 + 32;

/// One tenant on the list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    tenant: TenantId,
    /// Whether this node wrote for it.
    #[serde(default)]
    wrote: bool,
}

/// A tenant as the list holds it: as this release writes it, or as a name
/// alone, as the release before wrote it, read as a copy.
#[derive(Deserialize)]
#[serde(untagged)]
enum Listed {
    Entry(Entry),
    Named(TenantId),
}

impl From<Listed> for Entry {
    fn from(listed: Listed) -> Self {
        match listed {
            Listed::Entry(entry) => entry,
            Listed::Named(tenant) => Entry {
                tenant,
                wrote: false,
            },
        }
    }
}

/// A node's warm list, and the warming that reads it.
#[derive(Debug, Default)]
pub(crate) struct Warm {
    /// Held while the list is written, so that writes at once take turns.
    writing: Mutex<()>,
    /// The warming, and the keeping of the list after it.
    task: Mutex<Option<JoinHandle<()>>>,
    /// Whether warming is opening copies now.
    opening: AtomicBool,
    /// What the list named that warming has yet to come to: kept on the
    /// list should the node stop first.
    unwarmed: Mutex<Vec<Entry>>,
}

impl Node {
    /// Open copies of the tenants the node lately served, as its warm list
    /// names them, before anyone reads them; and keep that list as the
    /// node goes on (see [`crate::Settings::warm`]). Copies need no lease,
    /// so a node warms tenants another node writes for as safely as any.
    /// Passed over: a tenant this node does not serve, one it wrote for,
    /// one gone since, and, once warming has filled its share of the
    /// memory budget, the rest.
    pub fn warming(&self) {
        let Some(path) = self.0.settings.warm.clone() else {
            return;
        };
        let node = self.clone();
        let task = tokio::spawn(async move {
            let most = node.0.settings.warm_most;
            let opening = Opening::begun(&node);
            let listed = read_list(path, most).await;
            *node.0.warm.unwarmed.lock().expect("warm lock") = listed.clone();
            node.warm(listed).await;
            node.0.warm.unwarmed.lock().expect("warm lock").clear();
            drop(opening);
            let every = node.0.sweep_every();
            loop {
                tokio::time::sleep(every).await;
                if node.0.opened().stopped {
                    return;
                }
                node.remember_warm().await;
            }
        });
        if let Some(before) = self.0.warm.task.lock().expect("warm lock").replace(task) {
            before.abort();
        }
    }

    /// Whether the node is warming now: opening the copies its list names.
    pub fn warming_now(&self) -> bool {
        self.0.warm.opening.load(Ordering::SeqCst)
    }

    /// Stop warming, and wait until it has.
    pub(crate) async fn stop_warming(&self) {
        let task = self.0.warm.task.lock().expect("warm lock").take();
        if let Some(task) = task {
            task.abort();
            let _ = task.await;
        }
    }

    async fn warm(&self, listed: Vec<Entry>) {
        let budget = self
            .0
            .settings
            .memory
            .map(|memory| memory / WARM_UP_TO.1 * WARM_UP_TO.0);
        // The largest copy warmed yet: what the next is taken to need.
        let mut largest = 0;
        for (rank, entry) in listed.iter().enumerate() {
            let tenant = &entry.tenant;
            if self.0.opened().stopped {
                return;
            }
            if let Some(budget) = budget {
                if self.resident().saturating_add(largest) > budget {
                    tracing::info!(budget, "warming stopped at its share of the memory budget");
                    return;
                }
            }
            // Off the list of what is yet to be warmed once it has been
            // tried, so that a node stopping while it opens keeps it.
            let tried = || {
                let mut unwarmed = self.0.warm.unwarmed.lock().expect("warm lock");
                unwarmed.retain(|e| e != entry);
            };
            if entry.wrote || !self.serves(tenant) || self.holds(tenant) {
                tried();
                continue;
            }
            let copied = self.copy(tenant, Some(rank)).await;
            tried();
            match copied {
                Ok((copy, opened)) => {
                    let size = copy.resident();
                    largest = largest.max(size);
                    let over = self
                        .0
                        .settings
                        .memory
                        .is_some_and(|memory| self.resident() > memory);
                    if over {
                        if opened {
                            self.0.forget_follower(tenant, &copy);
                        }
                        tracing::info!(%tenant, size, "warming stopped: a copy over the memory budget let go");
                        return;
                    }
                }
                Err(error) => {
                    tracing::info!(%tenant, %error, "a tenant on the warm list was not opened");
                }
            }
        }
    }

    /// Whether the node holds `tenant` now, written for or copied.
    fn holds(&self, tenant: &TenantId) -> bool {
        let open = self.0.opened();
        open.tenants.contains_key(tenant) || open.followers.contains_key(tenant)
    }

    /// Write the warm list now: the tenants this node holds, written for
    /// or copied for reads, the most lately used first; then the copies
    /// warming opened that nobody has read, as they were listed; then
    /// what warming has yet to come to; as many as it keeps.
    pub async fn remember_warm(&self) {
        let Some(path) = self.0.settings.warm.clone() else {
            return;
        };
        let kept = self.0.warm_list();
        let inner = self.0.clone();
        let written = tokio::task::spawn_blocking(move || {
            let _turn = inner.warm.writing.lock().expect("warm lock");
            write_list(&path, &kept)
        })
        .await
        .unwrap_or_else(|error| Err(std::io::Error::other(error)));
        if let Err(error) = written {
            tracing::warn!(%error, "the warm list could not be written");
        }
    }
}

/// Warming under way, until this is dropped: as it ends, or is stopped.
struct Opening(Node);

impl Opening {
    fn begun(node: &Node) -> Self {
        node.0.warm.opening.store(true, Ordering::SeqCst);
        Self(node.clone())
    }
}

impl Drop for Opening {
    fn drop(&mut self) {
        self.0 .0.warm.opening.store(false, Ordering::SeqCst);
    }
}

impl Inner {
    fn warm_list(&self) -> Vec<Entry> {
        // Each tenant by when it was last used, or where warming found it.
        let mut used: HashMap<TenantId, (Option<Instant>, usize, bool)> = HashMap::new();
        {
            let open = self.opened();
            for (tenant, opening) in &open.tenants {
                if let Some(writer) = opening.get() {
                    used.insert(tenant.clone(), (Some(writer.used_at()), 0, true));
                }
            }
            for (tenant, following) in open.followers.iter() {
                let Some(copy) = following.get() else {
                    continue;
                };
                let (at, rank) = match copy.warmed() {
                    Some(rank) => (None, rank),
                    None => (Some(copy.used_at()), 0),
                };
                used.entry(tenant.clone())
                    .and_modify(|(was, _, _)| *was = (*was).max(at))
                    .or_insert((at, rank, false));
            }
        }
        let mut used: Vec<_> = used.into_iter().collect();
        // The used, most lately first; then the warmed, as they were listed.
        used.sort_by_key(|(_, (at, rank, _))| (at.is_none(), std::cmp::Reverse(*at), *rank));
        let mut kept: Vec<Entry> = used
            .into_iter()
            .map(|(tenant, (_, _, wrote))| Entry { tenant, wrote })
            .collect();
        for entry in self.warm.unwarmed.lock().expect("warm lock").iter() {
            if !kept.iter().any(|kept| kept.tenant == entry.tenant) {
                kept.push(entry.clone());
            }
        }
        kept.truncate(self.settings.warm_most);
        kept
    }
}

/// The list at `path`, at most `most` of it: none if there is none, or it
/// is larger than a list of that many can be, or unreadable.
async fn read_list(path: PathBuf, most: usize) -> Vec<Entry> {
    let limit = BYTES_A_TENANT * most as u64 + 2;
    let read = tokio::task::spawn_blocking(move || -> std::io::Result<Option<Vec<u8>>> {
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let size = file.metadata()?.len();
        if size > limit {
            return Err(std::io::Error::other(format!(
                "{size} bytes, more than a list of {most} can be"
            )));
        }
        let mut bytes = Vec::new();
        file.take(limit + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit {
            return Err(std::io::Error::other("it grew as it was read"));
        }
        Ok(Some(bytes))
    })
    .await
    .unwrap_or_else(|error| Err(std::io::Error::other(error)));
    let bytes = match read {
        Ok(Some(bytes)) => bytes,
        Ok(None) => return Vec::new(),
        Err(error) => {
            tracing::warn!(%error, "the warm list is unreadable; warming nothing");
            return Vec::new();
        }
    };
    match serde_json::from_slice::<Vec<Listed>>(&bytes) {
        Ok(listed) => listed.into_iter().take(most).map(Entry::from).collect(),
        Err(error) => {
            tracing::warn!(%error, "the warm list is unreadable; warming nothing");
            Vec::new()
        }
    }
}

/// Write `kept` to `path` whole or not at all: to a file beside it, made
/// durable, then put in its place, and the place made durable too.
fn write_list(path: &Path, kept: &[Entry]) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(kept).map_err(std::io::Error::other)?;
    let partial = path.with_extension("partial");
    let mut file = std::fs::File::create(&partial)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, path)?;
    if let Some(dir) = path.parent() {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_the_release_before_wrote_is_read_as_copies() {
        let listed: Vec<Listed> =
            serde_json::from_str(r#"["acme", {"tenant": "beta", "wrote": true}]"#).unwrap();
        let entries: Vec<Entry> = listed.into_iter().map(Entry::from).collect();
        assert_eq!(
            entries,
            [
                Entry {
                    tenant: TenantId::new("acme").unwrap(),
                    wrote: false
                },
                Entry {
                    tenant: TenantId::new("beta").unwrap(),
                    wrote: true
                },
            ]
        );
    }
}
