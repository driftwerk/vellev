//! One node's part in the database: the tenants it writes for, each opened
//! when first asked for and written to through one shared queue.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::join_all;
use tokio::sync::{mpsc, OnceCell};
use vellev_core::{CollectionName, Embedding, NodeId, Partition, TenantId};
use vellev_embed::{EmbedError, Embedders};
use vellev_storage::blobs::Blobs;
use vellev_storage::coordinator::{
    Acquired, Coordinator, CoordinatorError, Lease, LogPosition, PartitionKey,
};
use vellev_storage::{
    Embedded, GroupCommitter, Mutation, Receipt, State, StorageError, Vectorize, Writer,
};

#[derive(Debug, Clone)]
pub struct Settings {
    pub node: NodeId,
    /// How long the node's hold on a tenant lasts without being renewed.
    pub lease: Duration,
    /// The most callers' writes that share one commit.
    pub group: usize,
    /// The most writes that wait for a commit before callers are held back.
    pub queue: usize,
    /// A tenant with this many rows written since its last checkpoint is
    /// flushed.
    pub flush_rows: usize,
}

impl Settings {
    pub fn new(node: NodeId) -> Self {
        Self {
            node,
            lease: Duration::from_secs(30),
            group: 64,
            queue: 1_024,
            flush_rows: 50_000,
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
}

impl Tenant {
    pub fn id(&self) -> &TenantId {
        &self.id
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

    /// Commit `mutations` as one atomic step, sharing the commit with
    /// whoever else is writing.
    pub async fn write(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        let made = self.embedded(&mutations).await;
        let receipt = self
            .commits
            .submit_with(mutations, made)
            .await
            .map_err(|error| self.lost(error))?;
        if self.read(State::unflushed) >= self.flush_rows {
            let _ = self.flushes.try_send(());
        }
        Ok(receipt)
    }

    /// Move what has been written since the last checkpoint out of memory.
    pub async fn flush(&self) -> Result<LogPosition> {
        self.writer.flush().await.map_err(|error| {
            self.broken.store(true, Ordering::SeqCst);
            error.into()
        })
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
        match self.embedders.embedder(&embedding)? {
            Some(embedder) => Ok(embedder.query(text).await?),
            None => Ok(None),
        }
    }
}

type Opening = Arc<OnceCell<Arc<Tenant>>>;

#[derive(Debug, Default)]
struct Open {
    tenants: HashMap<TenantId, Opening>,
    stopped: bool,
}

#[derive(Debug)]
struct Inner {
    coordinator: Arc<dyn Coordinator>,
    blobs: Arc<dyn Blobs>,
    embedders: Arc<Embedders>,
    settings: Settings,
    open: Mutex<Open>,
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
        }))
    }

    /// Whether the node can reach what it needs to serve.
    pub async fn ready(&self) -> Result<()> {
        Ok(self.0.coordinator.ping().await?)
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create_tenant(&self, tenant: &TenantId) -> Result<()> {
        Ok(self.0.coordinator.register(&key(tenant)).await?)
    }

    /// Forget the tenant. Returns whether it existed. Its files stay in
    /// object storage, where nothing refers to them any more.
    pub async fn drop_tenant(&self, tenant: &TenantId) -> Result<bool> {
        let existed = self.0.coordinator.remove(&key(tenant)).await?;
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
                .is_some_and(|tenant| tenant.broken.load(Ordering::SeqCst));
            if lost {
                open.tenants.remove(tenant);
            }
            open.tenants.entry(tenant.clone()).or_default().clone()
        };
        let (inner, id, shared) = (self.0.clone(), tenant.clone(), opening.clone());
        let opened = tokio::spawn(async move {
            let opened = shared.get_or_try_init(|| inner.opening(&id)).await;
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
            open.tenants.drain().map(|(_, opening)| opening).collect()
        };
        for opening in openings {
            let unopened = || async { Err(NodeError::Stopped) };
            if let Ok(tenant) = opening.get_or_try_init(unopened).await {
                tenant.broken.store(true, Ordering::SeqCst);
                self.0.release(tenant.writer.lease()).await;
            }
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
        let opened = Writer::open(
            self.blobs.clone(),
            self.coordinator.clone(),
            lease.clone(),
            &committed,
        )
        .await;
        let writer = match opened {
            Ok(writer) if !self.opened().stopped => {
                Arc::new(writer.embedding_with(Arc::new(Documents(self.embedders.clone()))))
            }
            unusable => {
                self.release(&lease).await;
                return Err(unusable.err().map_or(NodeError::Stopped, Into::into));
            }
        };
        let broken = Arc::new(AtomicBool::new(false));
        let (flushes, mut wanted) = mpsc::channel(1);
        let (flushing, failed, flushed, rows) =
            (writer.clone(), broken.clone(), tenant.clone(), *flush_rows);
        tokio::spawn(async move {
            while wanted.recv().await.is_some() {
                if flushing.partition().read(State::unflushed) < rows {
                    continue;
                }
                if let Err(error) = flushing.flush().await {
                    tracing::warn!(tenant = %flushed, %error, "flush failed");
                    failed.store(true, Ordering::SeqCst);
                }
            }
        });
        Ok(Arc::new(Tenant {
            id: tenant.clone(),
            commits: GroupCommitter::spawn(writer.clone(), *group, *queue),
            writer,
            embedders: self.embedders.clone(),
            flush_rows: *flush_rows,
            flushes,
            broken,
        }))
    }
}
