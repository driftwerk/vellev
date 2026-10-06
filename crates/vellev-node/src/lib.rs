//! One node's part in the database: the tenants it writes for, each opened
//! when first asked for and written to through one shared queue.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::future::join_all;
use tokio::sync::{mpsc, OnceCell};
use vellev_core::{CollectionName, Embedding, NodeId, Partition, TenantId};
use vellev_embed::{EmbedError, Embedders};
use vellev_storage::blobs::Blobs;
use vellev_storage::coordinator::{
    Acquired, Coordinator, CoordinatorError, Lease, LogPosition, PartitionKey, Versions,
};
use vellev_storage::format::{DEFAULT_MAX_FORMAT, FORMAT_VERSION};
use vellev_storage::{
    Batch, Embedded, GroupCommitter, Mutation, Receipt, State, StorageError, Vectorize, Writer,
};

mod reading;

pub use reading::{Freshness, Reader};

/// The node-to-node protocol this release speaks: what a request one node
/// sends on to another may hold. A release that lets requests hold
/// something new raises it, and a node only sends that on once every node
/// announces the new one.
pub const PROTOCOL: u16 = 1;

/// A key nothing is kept under, asked for to tell that object storage
/// answers: that it is missing is the answer.
const READINESS: &str = "vellev-readiness";

/// How long object storage having answered counts for readiness, so that
/// health checks do not ask it every few seconds.
const STORE_ANSWERED: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct Settings {
    pub node: NodeId,
    /// How long the node's hold on a tenant lasts without being renewed.
    /// It is renewed every third of this while the tenant is open.
    pub lease: Duration,
    /// The most callers' writes that share one commit.
    pub group: usize,
    /// The most writes that wait for a commit before callers are held back.
    pub queue: usize,
    /// A tenant with this many rows written since its last checkpoint is
    /// flushed.
    pub flush_rows: usize,
    /// How long a file nothing reaches any more is kept before it is
    /// deleted. Files are looked for about this often while a tenant is
    /// open, at most once a minute.
    pub grace: Duration,
    /// How often a copy of a tenant kept for reads looks for new commits.
    pub follow_every: Duration,
    /// How long a copy of a tenant kept for reads is kept unused.
    pub follow_idle: Duration,
    /// The most copies of tenants kept for reads at once; the least
    /// recently used go first.
    pub followers: usize,
    /// The newest file format this node may write, whatever every node
    /// reads. A release that adds a format keeps this at the one before,
    /// so that rolling back stays possible until an operator raises it.
    pub max_format: u16,
}

impl Settings {
    pub fn new(node: NodeId) -> Self {
        Self {
            node,
            lease: Duration::from_secs(30),
            group: 64,
            queue: 1_024,
            flush_rows: 50_000,
            grace: Duration::from_secs(24 * 60 * 60),
            follow_every: Duration::from_secs(1),
            follow_idle: Duration::from_secs(5 * 60),
            followers: 64,
            max_format: DEFAULT_MAX_FORMAT,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("tenant {0} does not exist")]
    UnknownTenant(TenantId),
    #[error("tenant {tenant} is written by {holder}")]
    Elsewhere { tenant: TenantId, holder: NodeId },
    #[error("the node is stopping")]
    Stopped,
    #[error("the hold on tenant {0} ran out while it was being opened")]
    Lost(TenantId),
    #[error("this node has not caught up with tenant {0} as far as asked")]
    Behind(TenantId),
    #[error("the point asked for is past anything committed to tenant {0}")]
    Ahead(TenantId),
    #[error("the cluster writes file format {cluster}, and this release reads only up to {reads}")]
    Outdated { cluster: u16, reads: u16 },
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Embed(#[from] EmbedError),
}

impl From<CoordinatorError> for NodeError {
    fn from(error: CoordinatorError) -> Self {
        Self::Storage(error.into())
    }
}

pub type Result<T> = std::result::Result<T, NodeError>;

fn key(tenant: &TenantId) -> PartitionKey {
    PartitionKey {
        tenant: tenant.clone(),
        partition: Partition::DEFAULT,
    }
}

/// The node's embedders, as a writer asks for vectors.
#[derive(Debug)]
struct Documents(Arc<Embedders>);

#[async_trait]
impl Vectorize for Documents {
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let embedder = self
            .0
            .embedder(embedding)
            .map_err(|error| error.to_string())?
            .ok_or("its collection's clients supply the vectors")?;
        let vectors = embedder
            .documents(&texts)
            .await
            .map_err(|error| error.to_string())?;
        vectors
            .into_iter()
            .collect::<Option<_>>()
            .ok_or_else(|| "no vector for a text that is not blank".to_string())
    }
}

/// A tenant this node writes for.
#[derive(Debug)]
pub struct Tenant {
    id: TenantId,
    writer: Arc<Writer>,
    commits: GroupCommitter,
    embedders: Arc<Embedders>,
    flush_rows: usize,
    flushes: mpsc::Sender<()>,
    /// Set once the writer can no longer write, so that the node opens the
    /// tenant afresh for the next caller.
    broken: Arc<AtomicBool>,
    /// When the hold on the tenant runs out unless renewed, by this node's
    /// own clock.
    deadline: Arc<Mutex<Instant>>,
}

impl Tenant {
    pub fn id(&self) -> &TenantId {
        &self.id
    }

    /// Whether this node still holds the tenant, as far as it can tell
    /// without asking: its hold has not run out and nothing has failed.
    fn held(&self) -> bool {
        !self.broken.load(Ordering::SeqCst) && Instant::now() < *self.deadline()
    }

    fn deadline(&self) -> std::sync::MutexGuard<'_, Instant> {
        self.deadline.lock().expect("deadline lock")
    }

    /// Run `f` against the tenant's data as of its last commit.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        self.writer.partition().read(f)
    }

    /// The vectors `mutations` will need made, for as many as can be made
    /// now. Callers embed side by side here; whatever is not made here,
    /// for any reason, is made or refused by the commit, one commit at a
    /// time.
    async fn embedded(&self, mutations: &[Mutation]) -> Embedded {
        let mut alike: HashMap<Embedding, Vec<String>> = HashMap::new();
        for (embedding, text) in self.writer.wanted(mutations) {
            alike.entry(embedding).or_default().push(text);
        }
        let made = alike.into_iter().map(|(embedding, texts)| async move {
            let embedder = self.embedders.embedder(&embedding).ok()??;
            let vectors = embedder.documents(&texts).await.ok()?;
            Some((embedding, texts, vectors))
        });
        join_all(made)
            .await
            .into_iter()
            .flatten()
            .flat_map(|(embedding, texts, vectors)| {
                texts
                    .into_iter()
                    .zip(vectors)
                    .filter_map(move |(text, vector)| Some(((embedding.clone(), text), vector?)))
            })
            .collect()
    }

    /// Commit a batch as one atomic step, sharing the commit with whoever
    /// else is writing. A batch naming a request answered lately is
    /// answered as it was, and not made again.
    pub async fn write(&self, batch: impl Into<Batch>) -> Result<Receipt> {
        let batch = batch.into();
        // A request answered already is answered without its vectors.
        let answered = batch
            .request
            .is_some_and(|request| self.read(|state| state.answered(request)));
        let made = match answered {
            true => Embedded::new(),
            false => self.embedded(&batch.mutations).await,
        };
        let receipt = self
            .commits
            .submit_batch(batch, made)
            .await
            .map_err(|error| self.lost(error))?;
        if self.read(State::unflushed) >= self.flush_rows {
            let _ = self.flushes.try_send(());
        }
        Ok(receipt)
    }

    /// Move what has been written since the last checkpoint out of memory.
    pub async fn flush(&self) -> Result<LogPosition> {
        self.writer
            .flush()
            .await
            .map_err(|error| self.failed(error))
    }

    /// Merge segments until a merge finds nothing to do, and then delete
    /// the files nothing needs any more, as [`Writer::clean`] with `grace`.
    pub async fn compact(&self, grace: Duration) -> Result<usize> {
        while self
            .writer
            .merge()
            .await
            .map_err(|error| self.failed(error))?
            .is_some()
        {}
        self.writer.clean(grace).await.map_err(|error| match error {
            StorageError::Closed => self.failed(error),
            other => other.into(),
        })
    }

    /// Note that the writer failed at something that leaves it closed.
    fn failed(&self, error: StorageError) -> NodeError {
        self.broken.store(true, Ordering::SeqCst);
        error.into()
    }

    /// A write's failure as its caller should see it. A failure of the
    /// commit itself, rather than of one caller's batch, leaves the writer
    /// closed or without its lease.
    fn lost(&self, error: StorageError) -> NodeError {
        if matches!(error, StorageError::Shared(_) | StorageError::Closed) {
            self.broken.store(true, Ordering::SeqCst);
        }
        match error.root() {
            StorageError::Coordinator(CoordinatorError::UnknownPartition(_)) => {
                NodeError::UnknownTenant(self.id.clone())
            }
            _ => error.into(),
        }
    }

    /// The vector to search `collection` with for `text`, or `None` if the
    /// text is blank or the collection's clients supply its vectors.
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
        reading::query_vector(&self.embedders, &embedding, text).await
    }
}

type Opening = Arc<OnceCell<Arc<Tenant>>>;

#[derive(Debug, Default)]
struct Open {
    tenants: HashMap<TenantId, Opening>,
    followers: reading::Followers,
    stopped: bool,
}

#[derive(Debug)]
struct Inner {
    coordinator: Arc<dyn Coordinator>,
    blobs: Arc<dyn Blobs>,
    embedders: Arc<Embedders>,
    settings: Settings,
    open: Mutex<Open>,
    /// When object storage last answered a readiness check.
    store_answered: Mutex<Option<Instant>>,
}

#[derive(Debug, Clone)]
pub struct Node(Arc<Inner>);

impl Node {
    pub fn new(
        coordinator: Arc<dyn Coordinator>,
        blobs: Arc<dyn Blobs>,
        embedders: Arc<Embedders>,
        settings: Settings,
    ) -> Self {
        Self(Arc::new(Inner {
            coordinator,
            blobs,
            embedders,
            settings,
            open: Mutex::default(),
            store_answered: Mutex::default(),
        }))
    }

    /// Whether the node can reach Postgres and object storage and can read
    /// what the cluster writes.
    pub async fn ready(&self) -> Result<()> {
        self.store_answers().await?;
        let cluster = self.0.coordinator.format().await?;
        match cluster <= FORMAT_VERSION {
            true => Ok(()),
            false => Err(NodeError::Outdated {
                cluster,
                reads: FORMAT_VERSION,
            }),
        }
    }

    /// Whether object storage answered within [`STORE_ANSWERED`], asking
    /// it again if not.
    async fn store_answers(&self) -> Result<()> {
        let fresh = |at: &Option<Instant>| at.is_some_and(|at| at.elapsed() < STORE_ANSWERED);
        if fresh(&self.0.store_answered.lock().expect("readiness lock")) {
            return Ok(());
        }
        match self.0.blobs.get(READINESS).await {
            Ok(_) | Err(StorageError::Missing { .. }) => {
                *self.0.store_answered.lock().expect("readiness lock") = Some(Instant::now());
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    /// What this node reads.
    pub fn versions(&self) -> Versions {
        Versions {
            format: FORMAT_VERSION,
            protocol: PROTOCOL,
        }
    }

    /// Register this node as running and reading what it reads, now and
    /// every third of a lease until it stops, and each time raise the file
    /// format the cluster writes as far as this node may write, which only
    /// happens once every node registered reads it.
    pub fn announcing(&self) {
        let every = (self.0.settings.lease / 3).max(Duration::from_secs(1));
        let (inner, versions) = (Arc::downgrade(&self.0), self.versions());
        let writes = versions.format.min(self.0.settings.max_format);
        tokio::spawn(async move {
            loop {
                let Some(inner) = inner.upgrade() else { return };
                if inner.opened().stopped {
                    return;
                }
                let node = &inner.settings.node;
                let announced = async {
                    inner.coordinator.announce(node, versions).await?;
                    inner.coordinator.raise_format(writes).await
                };
                if let Err(error) = announced.await {
                    tracing::warn!(%error, "announcing the node failed");
                }
                drop(inner);
                tokio::time::sleep(every).await;
            }
        });
    }

    /// What the cluster reads and writes: the file format it writes, and
    /// the oldest protocol any node registered speaks.
    pub async fn cluster(&self) -> Result<Versions> {
        let format = self.0.coordinator.format().await?;
        let protocol = self
            .0
            .coordinator
            .oldest()
            .await?
            .map_or(PROTOCOL, |oldest| oldest.protocol.min(PROTOCOL));
        Ok(Versions { format, protocol })
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create_tenant(&self, tenant: &TenantId) -> Result<()> {
        Ok(self.0.coordinator.register(&key(tenant)).await?)
    }

    /// Forget the tenant. Returns whether it existed. Its files stay in
    /// object storage, where nothing refers to them any more. A node that
    /// holds the tenant notices at its next renewal, within a third of the
    /// lease.
    pub async fn drop_tenant(&self, tenant: &TenantId) -> Result<bool> {
        let existed = self.0.coordinator.remove(&key(tenant)).await?;
        self.0.opened().followers.remove(tenant);
        let opening = self.0.opened().tenants.remove(tenant);
        if let Some(opened) = opening.as_ref().and_then(|opening| opening.get()) {
            opened.broken.store(true, Ordering::SeqCst);
        }
        Ok(existed)
    }

    /// Up to `limit` tenants after `after`, in id order; fewer where a
    /// tenant has several partitions.
    pub async fn tenants(&self, after: Option<&TenantId>, limit: usize) -> Result<Vec<TenantId>> {
        let after = after.map(|tenant| PartitionKey {
            tenant: tenant.clone(),
            partition: Partition(u32::MAX),
        });
        let keys = self.0.coordinator.partitions(after.as_ref(), limit).await?;
        let mut tenants: Vec<TenantId> = keys.into_iter().map(|key| key.tenant).collect();
        tenants.dedup();
        Ok(tenants)
    }

    /// The tenant, opened if this is the first call for it or its writer
    /// was lost. Callers arriving together share one opening, which runs
    /// to its end even if they all stop waiting for it.
    pub async fn tenant(&self, tenant: &TenantId) -> Result<Arc<Tenant>> {
        let opening = {
            let mut open = self.0.opened();
            if open.stopped {
                return Err(NodeError::Stopped);
            }
            let lost = open
                .tenants
                .get(tenant)
                .and_then(|opening| opening.get())
                .is_some_and(|tenant| !tenant.held());
            if lost {
                open.tenants.remove(tenant);
            }
            open.tenants.entry(tenant.clone()).or_default().clone()
        };
        let (inner, id, shared) = (self.0.clone(), tenant.clone(), opening.clone());
        let opened = tokio::spawn(async move {
            let opened = shared.get_or_try_init(|| inner.opening(&id)).await;
            if opened.is_ok() {
                inner.opened().followers.remove(&id);
            }
            opened.cloned()
        })
        .await
        .unwrap_or_else(|stopped| match stopped.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            Err(_) => Err(NodeError::Stopped),
        });
        if opened.is_err() {
            let mut open = self.0.opened();
            let unopened = open
                .tenants
                .get(tenant)
                .is_some_and(|held| Arc::ptr_eq(held, &opening) && held.get().is_none());
            if unopened {
                open.tenants.remove(tenant);
            }
        }
        opened
    }

    /// Give up every tenant, so that another node can write for them at
    /// once rather than when the leases run out, and open no more.
    pub async fn shutdown(&self) {
        let openings: Vec<Opening> = {
            let mut open = self.0.opened();
            open.stopped = true;
            open.followers.clear();
            open.tenants.drain().map(|(_, opening)| opening).collect()
        };
        for opening in openings {
            let unopened = || async { Err(NodeError::Stopped) };
            if let Ok(tenant) = opening.get_or_try_init(unopened).await {
                tenant.broken.store(true, Ordering::SeqCst);
                self.0.release(tenant.writer.lease()).await;
            }
        }
        if let Err(error) = self.0.coordinator.depart(&self.0.settings.node).await {
            tracing::warn!(%error, "the node could not take itself off the register");
        }
    }
}

impl Inner {
    fn opened(&self) -> std::sync::MutexGuard<'_, Open> {
        self.open.lock().expect("tenants lock")
    }

    async fn release(&self, lease: &Lease) {
        if let Err(error) = self.coordinator.release(lease).await {
            tracing::warn!(partition = %lease.key, %error, "lease not released");
        }
    }

    async fn opening(&self, tenant: &TenantId) -> Result<Arc<Tenant>> {
        let Settings {
            node,
            lease,
            group,
            queue,
            flush_rows,
            ..
        } = &self.settings;
        let (lease, committed) = match self.coordinator.acquire(&key(tenant), node, *lease).await {
            Ok(Acquired::Lease { lease, committed }) => (lease, committed),
            Ok(Acquired::Held { holder, .. }) => {
                return Err(NodeError::Elsewhere {
                    tenant: tenant.clone(),
                    holder,
                })
            }
            Err(CoordinatorError::UnknownPartition(_)) => {
                return Err(NodeError::UnknownTenant(tenant.clone()))
            }
            Err(error) => return Err(error.into()),
        };
        let broken = Arc::new(AtomicBool::new(false));
        let deadline = Arc::new(Mutex::new(lease.deadline()));
        self.renewing(lease.clone(), broken.clone(), &deadline);
        let opened = Writer::open(
            self.blobs.clone(),
            self.coordinator.clone(),
            lease.clone(),
            &committed,
        )
        .await;
        let still_held = !broken.load(Ordering::SeqCst)
            && Instant::now() < *deadline.lock().expect("deadline lock");
        let writer = match opened {
            Ok(writer) if !self.opened().stopped && still_held => {
                Arc::new(writer.embedding_with(Arc::new(Documents(self.embedders.clone()))))
            }
            Ok(_) if self.opened().stopped => {
                self.release(&lease).await;
                return Err(NodeError::Stopped);
            }
            Ok(_) => {
                self.release(&lease).await;
                return Err(NodeError::Lost(tenant.clone()));
            }
            Err(error) => {
                self.release(&lease).await;
                return Err(error.into());
            }
        };
        let (flushes, wanted) = mpsc::channel(1);
        let opened = Arc::new(Tenant {
            id: tenant.clone(),
            commits: GroupCommitter::spawn(writer.clone(), *group, *queue),
            writer,
            embedders: self.embedders.clone(),
            flush_rows: *flush_rows,
            flushes,
            broken,
            deadline,
        });
        self.maintaining(&opened, wanted);
        Ok(opened)
    }

    /// Flush the tenant when asked to and it is still due, and merge and
    /// clean up its files after each flush and now and then regardless,
    /// for as long as the tenant is open and held. The task holds no
    /// reference of its own, so dropping the tenant ends it.
    fn maintaining(&self, tenant: &Arc<Tenant>, mut wanted: mpsc::Receiver<()>) {
        let grace = self.settings.grace;
        let every = (grace / 24).max(Duration::from_secs(60));
        let tenant = Arc::downgrade(tenant);
        tokio::spawn(async move {
            loop {
                let flush = match tokio::time::timeout(every, wanted.recv()).await {
                    Ok(Some(())) => true,
                    Ok(None) => return,
                    Err(_) => false,
                };
                let Some(tenant) = tenant.upgrade() else {
                    return;
                };
                if !tenant.held() {
                    return;
                }
                if flush && tenant.read(State::unflushed) >= tenant.flush_rows {
                    if let Err(error) = tenant.flush().await {
                        tracing::warn!(tenant = %tenant.id, %error, "flush failed");
                        continue;
                    }
                }
                match tenant.compact(grace).await {
                    Ok(0) => {}
                    Ok(deleted) => tracing::info!(tenant = %tenant.id, deleted, "cleaned up"),
                    Err(error) => tracing::warn!(tenant = %tenant.id, %error, "compaction failed"),
                }
            }
        });
    }

    /// Keep the hold the tenant was taken with for as long as `deadline`
    /// is held, from the opening on, and note when the hold is lost. Each
    /// renewal is timed from when the one before was asked for, so a slow
    /// answer does not let the hold run out.
    fn renewing(&self, lease: Lease, broken: Arc<AtomicBool>, deadline: &Arc<Mutex<Instant>>) {
        let coordinator = self.coordinator.clone();
        let ttl = self.settings.lease;
        let every = ttl / 3;
        let deadline = Arc::downgrade(deadline);
        tokio::spawn(async move {
            let mut next = lease.requested_at + every;
            loop {
                tokio::time::sleep_until(tokio::time::Instant::from_std(next)).await;
                let Some(deadline) = deadline.upgrade() else {
                    return;
                };
                if broken.load(Ordering::SeqCst) {
                    return;
                }
                let asked = Instant::now();
                match tokio::time::timeout(every, coordinator.renew(&lease, ttl)).await {
                    Ok(Ok(renewed)) => {
                        *deadline.lock().expect("deadline lock") = renewed.deadline();
                        next = renewed.requested_at + every;
                    }
                    Ok(Err(error)) if !error.is_retryable() => {
                        tracing::warn!(partition = %lease.key, %error, "hold on the tenant lost");
                        broken.store(true, Ordering::SeqCst);
                        return;
                    }
                    Ok(Err(error)) => {
                        tracing::warn!(partition = %lease.key, %error, "renewing the hold failed");
                        next = asked + (every / 4);
                    }
                    Err(_) => {
                        tracing::warn!(partition = %lease.key, "renewing the hold took too long");
                        next = asked + (every / 4);
                    }
                }
            }
        });
    }
}
