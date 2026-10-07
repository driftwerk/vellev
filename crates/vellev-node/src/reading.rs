//! Answering reads on any node: from the writer where this node holds the
//! tenant, and otherwise from a copy that follows what the writer commits.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use tokio::sync::OnceCell;
use vellev_core::{CollectionName, TenantId};
use vellev_embed::Embedders;
use vellev_storage::coordinator::{Committed, CoordinatorError, Incarnation, LogPosition};
use vellev_storage::{Follower, State, StorageError};

use crate::{key, Asked, Inner, Node, NodeError, Result, Tenant};

/// How fresh a read must be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Whatever the node has.
    Eventual,
    /// At least this far into the tenant's history.
    After(Incarnation, LogPosition),
    /// Everything committed when the read arrived.
    Strong,
}

/// What a read is answered from.
#[derive(Debug, Clone)]
pub enum Reader {
    Writer(Arc<Tenant>),
    Follower(Arc<Followed>),
}

impl Reader {
    /// Run `f` against the data the reader has.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        match self {
            Reader::Writer(tenant) => tenant.read(f),
            Reader::Follower(followed) => {
                followed.touch();
                followed.follower.partition().read(f)
            }
        }
    }

    /// The vector to search `collection` with for `text`, or `None` if the
    /// text is blank or the collection's clients supply its vectors.
    pub async fn query_vector(
        &self,
        collection: &CollectionName,
        text: &str,
    ) -> Result<Option<Vec<f32>>> {
        match self {
            Reader::Writer(tenant) => tenant.query_vector(collection, text).await,
            Reader::Follower(followed) => {
                let embedding = followed.follower.partition().read(|state| {
                    state
                        .collection(collection)
                        .map(|schema| schema.embedding.clone())
                        .ok_or_else(|| StorageError::UnknownCollection(collection.clone()))
                })?;
                query_vector(&followed.embedders, &embedding, text).await
            }
        }
    }
}

/// The vector `embedders` make for `text` as a query against data embedded
/// as `embedding` says.
pub(crate) async fn query_vector(
    embedders: &Embedders,
    embedding: &vellev_core::Embedding,
    text: &str,
) -> Result<Option<Vec<f32>>> {
    match embedders.embedder(embedding)? {
        Some(embedder) => Ok(embedder.query(text).await?),
        None => Ok(None),
    }
}

/// A copy of a tenant this node follows for reads, kept up to date while
/// it is used.
#[derive(Debug)]
pub struct Followed {
    follower: Follower,
    embedders: Arc<Embedders>,
    used: Mutex<Instant>,
}

impl Followed {
    fn touch(&self) {
        *self.used.lock().expect("use lock") = Instant::now();
    }

    pub(crate) fn idle(&self) -> std::time::Duration {
        self.used.lock().expect("use lock").elapsed()
    }

    /// About how many bytes the copy takes in memory.
    pub(crate) fn resident(&self) -> usize {
        self.follower.partition().read(State::resident)
    }

    fn reached(&self, incarnation: Incarnation, position: LogPosition) -> bool {
        self.follower.partition().reached(incarnation, position)
    }
}

pub(crate) type Following = Arc<OnceCell<Arc<Followed>>>;

/// Whether a reading the node acted on found the copy unusable, so that it
/// must be opened afresh from the committed state.
fn outgrown(error: &StorageError) -> bool {
    matches!(error.root(), StorageError::Diverged { .. })
}

impl Node {
    /// Where to answer a read of `tenant` that must be `fresh`. A node that
    /// holds the tenant answers from its writer, which has everything
    /// committed; any other answers from a copy it follows, brought as far
    /// as `fresh` asks. A read never takes the tenant's lease.
    pub async fn reader(&self, tenant: &TenantId, fresh: Freshness) -> Result<Reader> {
        self.0.admit(tenant, Asked::Query, 1)?;
        if let Some(writer) = self.0.held(tenant) {
            writer.touch();
            let partition = writer.writer.partition();
            let current = match fresh {
                Freshness::Eventual => true,
                Freshness::After(incarnation, position) => partition.reached(incarnation, position),
                Freshness::Strong => {
                    self.committed(tenant).await?.incarnation == partition.incarnation()
                }
            };
            if current {
                return Ok(Reader::Writer(writer));
            }
        }
        let followed = self.copy_of(tenant).await?;
        let (incarnation, position) = match fresh {
            Freshness::Eventual => return Ok(Reader::Follower(followed)),
            Freshness::Strong => {
                let (caught_up, _) = self.caught_up(tenant, followed).await?;
                return Ok(Reader::Follower(caught_up));
            }
            Freshness::After(incarnation, position) => (incarnation, position),
        };
        if followed.reached(incarnation, position) {
            return Ok(Reader::Follower(followed));
        }
        let (caught_up, committed) = self.caught_up(tenant, followed).await?;
        if caught_up.reached(incarnation, position) {
            Ok(Reader::Follower(caught_up))
        } else if committed.incarnation == incarnation {
            Err(NodeError::Ahead(tenant.clone()))
        } else {
            Err(NodeError::Behind(tenant.clone()))
        }
    }

    /// How many tenants this node keeps copies of for reads.
    pub fn copies(&self) -> usize {
        self.0.opened().followers.len()
    }

    async fn committed(&self, tenant: &TenantId) -> Result<Committed> {
        self.0
            .coordinator
            .committed(&key(tenant))
            .await?
            .ok_or_else(|| NodeError::UnknownTenant(tenant.clone()))
    }

    /// The copy of `tenant` this node follows, opened if this is the first
    /// read of it. Callers arriving together share one opening, which runs
    /// to its end and cleans up after itself even if they all stop waiting.
    async fn copy_of(&self, tenant: &TenantId) -> Result<Arc<Followed>> {
        let following = {
            let mut open = self.0.opened();
            if open.stopped {
                return Err(NodeError::Stopped);
            }
            open.followers.entry(tenant.clone()).or_default().clone()
        };
        let (inner, id) = (self.0.clone(), tenant.clone());
        let followed = tokio::spawn(async move {
            let opened = following.get_or_try_init(|| inner.following(&id)).await;
            if opened.is_err() {
                inner.unfollow(&id, &following);
            }
            opened.cloned()
        })
        .await
        .unwrap_or_else(|stopped| match stopped.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            Err(_) => Err(NodeError::Stopped),
        })?;
        followed.touch();
        self.bound();
        Ok(followed)
    }

    /// `followed` brought up to what is committed now, or a copy opened
    /// afresh when it cannot follow there, with the committed state it
    /// was brought to.
    async fn caught_up(
        &self,
        tenant: &TenantId,
        followed: Arc<Followed>,
    ) -> Result<(Arc<Followed>, Committed)> {
        let committed = self.committed(tenant).await?;
        match followed.follower.catch_up(&committed).await {
            Ok(_) => Ok((followed, committed)),
            Err(error) if outgrown(&error) => {
                self.0.forget_follower(tenant, &followed);
                Ok((self.copy_of(tenant).await?, committed))
            }
            Err(error) => Err(error.into()),
        }
    }
}

impl Inner {
    /// The writer, if this node holds the tenant now.
    fn held(&self, tenant: &TenantId) -> Option<Arc<Tenant>> {
        let open = self.opened();
        let writer = open.tenants.get(tenant)?.get()?;
        writer.held().then(|| writer.clone())
    }

    async fn following(self: &Arc<Self>, tenant: &TenantId) -> Result<Arc<Followed>> {
        let committed = self
            .coordinator
            .committed(&key(tenant))
            .await?
            .ok_or_else(|| NodeError::UnknownTenant(tenant.clone()))?;
        let follower = Follower::open(key(tenant), self.blobs.clone(), &committed).await?;
        let followed = Arc::new(Followed {
            follower,
            embedders: self.embedders.clone(),
            used: Mutex::new(Instant::now()),
        });
        self.bound_followers();
        self.polling(tenant.clone(), Arc::downgrade(&followed));
        Ok(followed)
    }

    /// Keep `followed` current while it is used, and let it go once it has
    /// been idle for the configured time.
    fn polling(self: &Arc<Self>, tenant: TenantId, followed: Weak<Followed>) {
        let (every, idle) = (self.settings.follow_every, self.settings.follow_idle);
        let inner = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let (Some(inner), Some(copy)) = (inner.upgrade(), followed.upgrade()) else {
                    return;
                };
                if !inner.follows(&tenant, &copy) {
                    return;
                }
                if copy.idle() > idle {
                    inner.forget_follower(&tenant, &copy);
                    return;
                }
                let committed = match inner.coordinator.committed(&key(&tenant)).await {
                    Ok(Some(committed)) => committed,
                    Ok(None) | Err(CoordinatorError::UnknownPartition(_)) => {
                        inner.forget_follower(&tenant, &copy);
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(%tenant, %error, "reading the committed state failed");
                        continue;
                    }
                };
                match copy.follower.catch_up(&committed).await {
                    Ok(_) => {}
                    Err(error) if outgrown(&error) => {
                        inner.forget_follower(&tenant, &copy);
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(%tenant, %error, "following the tenant failed")
                    }
                }
            }
        });
    }

    /// Whether `followed` is still the copy of `tenant` the node keeps.
    fn follows(&self, tenant: &TenantId, followed: &Arc<Followed>) -> bool {
        self.opened()
            .followers
            .get(tenant)
            .and_then(|following| following.get())
            .is_some_and(|held| Arc::ptr_eq(held, followed))
    }

    /// Let go of the copy of `tenant` if it is still `followed`, so that the
    /// next read opens one afresh.
    pub(crate) fn forget_follower(&self, tenant: &TenantId, followed: &Arc<Followed>) {
        if self.follows(tenant, followed) {
            self.opened().followers.remove(tenant);
        }
    }

    /// Let go of an opening of a copy that failed, if it is still the one
    /// in the map.
    fn unfollow(&self, tenant: &TenantId, following: &Following) {
        let mut open = self.opened();
        let unopened = open
            .followers
            .get(tenant)
            .is_some_and(|held| Arc::ptr_eq(held, following) && held.get().is_none());
        if unopened {
            open.followers.remove(tenant);
        }
    }

    /// Let go of the least recently used copies beyond the configured
    /// number, so that the copies a node keeps fit in its memory.
    fn bound_followers(&self) {
        let mut open = self.opened();
        let most = self.settings.followers;
        let mut opened: Vec<(TenantId, std::time::Duration)> = open
            .followers
            .iter()
            .filter_map(|(tenant, following)| Some((tenant.clone(), following.get()?.idle())))
            .collect();
        let over = open.followers.len().saturating_sub(most);
        opened.sort_by_key(|(_, idle)| std::cmp::Reverse(*idle));
        for (tenant, _) in opened.into_iter().take(over) {
            open.followers.remove(&tenant);
        }
    }
}

pub(crate) type Followers = HashMap<TenantId, Following>;
