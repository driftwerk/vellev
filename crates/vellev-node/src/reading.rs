//! Answering reads on any node: from the writer where this node holds the
//! tenant, and otherwise from a copy that follows what the writer commits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use tokio::time::Instant;

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

/// What a read is answered from, and whether anything costly has been
/// done for the read so far.
#[derive(Debug)]
pub struct Reader {
    from: From,
    /// Set by whatever the read itself has the node do that costs the
    /// node, or what the node depends on, something: a call to the
    /// coordinator, the opening of a copy, a text sent to a model. What
    /// another read had begun and this one only waited for is that
    /// read's, and is not set here: each piece of work is one read's, so
    /// that what a tenant is counted for is what was done for it. A read
    /// turned away having had none of it done was no query to count; see
    /// [`Node::unread`].
    worked: AtomicBool,
}

#[derive(Debug)]
enum From {
    Writer(Arc<Tenant>),
    Follower(Arc<Followed>),
}

impl Reader {
    fn new(from: From, worked: bool) -> Self {
        Self {
            from,
            worked: AtomicBool::new(worked),
        }
    }

    /// Whether the read is answered from a copy this node follows, and
    /// not from the tenant's writer.
    pub fn follows(&self) -> bool {
        matches!(self.from, From::Follower(_))
    }

    /// Whether the read itself had anything costly done: the coordinator
    /// asked, a copy opened, or a text sent to a model. Not what it only
    /// waited for another read to have done.
    pub fn worked(&self) -> bool {
        self.worked.load(Ordering::Relaxed)
    }

    /// Run `f` against the data the reader has.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        match &self.from {
            From::Writer(tenant) => tenant.read(f),
            From::Follower(followed) => {
                followed.touch();
                followed.follower.partition().read(f)
            }
        }
    }

    /// The vector to search `collection` with for `text`, or `None` if the
    /// text is blank or the collection's clients supply its vectors. A
    /// text this read sends to the model is work done for it; one whose
    /// vector is kept, or is being made for another read, or that a
    /// model with a fault that stands is not asked for, is not.
    pub async fn query_vector(
        &self,
        collection: &CollectionName,
        text: &str,
    ) -> Result<Option<Vec<f32>>> {
        let embedding = self.read(|state| {
            state
                .collection(collection)
                .map(|schema| schema.embedding.clone())
                .ok_or_else(|| StorageError::UnknownCollection(collection.clone()))
        })?;
        let (embedders, tenant) = match &self.from {
            From::Writer(tenant) => (&tenant.embedders, tenant.id()),
            From::Follower(followed) => (&followed.embedders, &followed.tenant),
        };
        let Some(embedder) = embedders.embedder(&embedding)? else {
            return Ok(None);
        };
        // In the tenant's name, by which a model shared among tenants
        // gives each its turn. Told by the embedder at the moment it
        // sends the text, so that it is known however the asking ends,
        // and whoever stops waiting.
        Ok(embedder
            .query_noting_for(tenant.as_str(), text, &self.worked)
            .await?)
    }
}

/// A copy of a tenant this node follows for reads, kept up to date while
/// it is used.
#[derive(Debug)]
pub struct Followed {
    tenant: TenantId,
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

/// A copy being opened or open, and how many reads are asking for it
/// now: each from when it takes this from the node's copies until its
/// asking is over, counted and uncounted with the copies held, so that
/// what is seen there of one is seen together with the other.
#[derive(Debug, Default)]
pub(crate) struct Opening {
    copy: OnceCell<Arc<Followed>>,
    asking: std::sync::atomic::AtomicUsize,
}

impl Opening {
    /// The copy, once it is open.
    pub(crate) fn get(&self) -> Option<&Arc<Followed>> {
        self.copy.get()
    }
}

pub(crate) type Following = Arc<Opening>;

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
    ///
    /// The read counts as one query against what the tenant may ask a
    /// second, and the reader keeps whether anything costly was done to
    /// answer it: see [`Self::unread`].
    pub async fn reader(&self, tenant: &TenantId, fresh: Freshness) -> Result<Reader> {
        self.0.admit(tenant, Asked::Query, 1)?;
        // Whether the coordinator was asked, here or below.
        let mut asked = false;
        if let Some(writer) = self.0.held(tenant) {
            writer.touch();
            let partition = writer.writer.partition();
            let current = match fresh {
                Freshness::Eventual => true,
                Freshness::After(incarnation, position) => partition.reached(incarnation, position),
                // Only an open writer whose lease no other node has taken
                // since, and which holds all that is committed, has it
                // all: one whose lease ran out, however its clock sees
                // it, may have been written past, and a closed one may
                // have committed what it never applied.
                Freshness::Strong => {
                    asked = true;
                    let committed = self.committed(tenant).await?;
                    !writer.writer.is_closed()
                        && committed.generation == writer.writer.lease().generation
                        && committed.incarnation == partition.incarnation()
                        && partition.read(State::position) >= committed.log_position
                }
            };
            if current {
                return Ok(Reader::new(From::Writer(writer), asked));
            }
        }
        let (followed, opened) = self.copy_of(tenant).await?;
        let worked = asked || opened;
        let (incarnation, position) = match fresh {
            Freshness::Eventual => return Ok(Reader::new(From::Follower(followed), worked)),
            Freshness::Strong => {
                let (caught_up, _) = self.caught_up(tenant, followed).await?;
                return Ok(Reader::new(From::Follower(caught_up), true));
            }
            Freshness::After(incarnation, position) => (incarnation, position),
        };
        if followed.reached(incarnation, position) {
            return Ok(Reader::new(From::Follower(followed), worked));
        }
        let (caught_up, committed) = self.caught_up(tenant, followed).await?;
        if caught_up.reached(incarnation, position) {
            Ok(Reader::new(From::Follower(caught_up), true))
        } else if committed.incarnation == incarnation {
            Err(NodeError::Ahead(tenant.clone()))
        } else {
            Err(NodeError::Behind(tenant.clone()))
        }
    }

    /// Tell the node that the read of `tenant` it gave `reader` for was
    /// turned away unanswered, for want of a thread to read on. If
    /// the read had nothing costly done, it is counted as never asked,
    /// and is no query against what the tenant may ask a second. If it
    /// had the coordinator asked, a copy opened or a text sent to a
    /// model, that was done at the tenant's asking whatever came of it,
    /// and it stays counted: else a tenant turned away could have the
    /// node do any amount of it for nothing. A read that only waited for
    /// what another had begun, the same copy opened or the same text
    /// embedded, is not counted for it: the other is.
    pub fn unread(&self, tenant: &TenantId, reader: &Reader) {
        if !reader.worked() {
            self.0.limits.give_back(tenant, Asked::Query, 1);
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
    /// read of it, and whether it was this call that opened it. Callers
    /// arriving together share one opening, which runs to its end and
    /// cleans up after itself even if they all stop waiting: it is the
    /// first's, and the others only wait for it.
    async fn copy_of(&self, tenant: &TenantId) -> Result<(Arc<Followed>, bool)> {
        let following = {
            let mut open = self.0.opened();
            if open.stopped {
                return Err(NodeError::Stopped);
            }
            let following = open.followers.entry(tenant.clone()).or_default().clone();
            following.asking.fetch_add(1, Ordering::Relaxed);
            following
        };
        let opened = Arc::new(AtomicBool::new(false));
        let (inner, id, opening) = (self.0.clone(), tenant.clone(), opened.clone());
        let followed = tokio::spawn(async move {
            let open = || {
                opening.store(true, Ordering::Relaxed);
                inner.following(&id)
            };
            let opened = following.copy.get_or_try_init(open).await.cloned();
            inner.asked(&id, &following);
            opened
        })
        .await
        .unwrap_or_else(|stopped| match stopped.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            Err(_) => Err(NodeError::Stopped),
        })?;
        followed.touch();
        self.bound();
        Ok((followed, opened.load(Ordering::Relaxed)))
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
                Ok((self.copy_of(tenant).await?.0, committed))
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
            tenant: tenant.clone(),
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

    /// Count one read as no longer asking for the copy of `tenant` that
    /// `following` is or was to be, and let go of an opening that failed
    /// if it is still the one kept and no other read is asking for it:
    /// one that is, opens the copy in its turn, and its copy must be the
    /// one kept.
    fn asked(&self, tenant: &TenantId, following: &Following) {
        let mut open = self.opened();
        let still_asked = following.asking.fetch_sub(1, Ordering::Relaxed) > 1;
        let unopened = open
            .followers
            .get(tenant)
            .is_some_and(|held| Arc::ptr_eq(held, following) && held.get().is_none());
        if unopened && !still_asked {
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
