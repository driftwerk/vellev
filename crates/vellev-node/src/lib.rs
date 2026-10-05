//! One node's part in the database: the tenants it writes for, each opened
//! when first asked for and written to through one shared queue.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, OnceCell};
use vellev_core::{
    CollectionName, Embedding, EmbeddingProvider, NodeId, Object, Partition, TenantId,
};
use vellev_embed::{EmbedError, Embedders};
use vellev_storage::blobs::Blobs;
use vellev_storage::coordinator::{
    Acquired, Coordinator, CoordinatorError, LogPosition, PartitionKey,
};
use vellev_storage::{GroupCommitter, Mutation, Receipt, State, StorageError, Vectorize, Writer};

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
    broken: AtomicBool,
}

impl Tenant {
    pub fn id(&self) -> &TenantId {
        &self.id
    }

    /// Run `f` against the tenant's data as of its last commit.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        self.writer.partition().read(f)
    }

    /// `mutations` with vectors made for the objects that are written
    /// whole, to a collection that embeds them, with text that has no
    /// vector stored. Callers embed side by side here; whatever is not
    /// made here, for any reason, is made or refused by the commit, one
    /// commit at a time.
    async fn embedded(&self, mut mutations: Vec<Mutation>) -> Vec<Mutation> {
        let wanted: Vec<(usize, Embedding, String)> = self.read(|state| {
            mutations
                .iter()
                .enumerate()
                .filter_map(|(at, mutation)| {
                    let (Mutation::Upsert {
                        collection,
                        id,
                        properties,
                        vector: None,
                    }
                    | Mutation::Insert {
                        collection,
                        id,
                        properties,
                        vector: None,
                    }) = mutation
                    else {
                        return None;
                    };
                    let schema = state.collection(collection)?;
                    let embedding = &schema.embedding;
                    let text = Object::from_json(schema, *id, properties, None)
                        .ok()?
                        .embedded_text(embedding);
                    let stored = state
                        .get(collection, *id)
                        .is_some_and(|stored| stored.embedded_text(embedding) == text);
                    let wanted = embedding.provider != EmbeddingProvider::None
                        && !stored
                        && !text.trim().is_empty();
                    wanted.then(|| (at, embedding.clone(), text))
                })
                .collect()
        });
        let mut alike: HashMap<Embedding, Vec<(usize, String)>> = HashMap::new();
        for (at, embedding, text) in wanted {
            alike.entry(embedding).or_default().push((at, text));
        }
        for (embedding, texts) in alike {
            let (at, texts): (Vec<usize>, Vec<String>) = texts.into_iter().unzip();
            let Ok(Some(embedder)) = self.embedders.embedder(&embedding) else {
                continue;
            };
            let Ok(vectors) = embedder.documents(&texts).await else {
                continue;
            };
            for (at, made) in at.into_iter().zip(vectors) {
                if let Mutation::Upsert { vector, .. } | Mutation::Insert { vector, .. } =
                    &mut mutations[at]
                {
                    *vector = made;
                }
            }
        }
        mutations
    }

    /// Commit `mutations` as one atomic step, sharing the commit with
    /// whoever else is writing.
    pub async fn write(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        let mutations = self.embedded(mutations).await;
        let receipt = self.commits.submit(mutations).await.inspect_err(|error| {
            let lost = matches!(
                error.root(),
                StorageError::Closed
                    | StorageError::Coordinator(
                        CoordinatorError::Fenced { .. } | CoordinatorError::UnknownPartition(_)
                    )
            );
            if lost {
                self.broken.store(true, Ordering::SeqCst);
            }
        })?;
        if self.read(State::unflushed) >= self.flush_rows {
            let _ = self.flushes.try_send(());
        }
        Ok(receipt)
    }

    /// Move what has been written since the last checkpoint out of memory.
    pub async fn flush(&self) -> Result<LogPosition> {
        Ok(self.writer.flush().await?)
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

#[derive(Debug)]
pub struct Node {
    coordinator: Arc<dyn Coordinator>,
    blobs: Arc<dyn Blobs>,
    embedders: Arc<Embedders>,
    settings: Settings,
    open: Mutex<HashMap<TenantId, Opening>>,
}

impl Node {
    pub fn new(
        coordinator: Arc<dyn Coordinator>,
        blobs: Arc<dyn Blobs>,
        embedders: Arc<Embedders>,
        settings: Settings,
    ) -> Self {
        Self {
            coordinator,
            blobs,
            embedders,
            settings,
            open: Mutex::default(),
        }
    }

    /// Whether the node can reach what it needs to serve.
    pub async fn ready(&self) -> Result<()> {
        Ok(self.coordinator.ping().await?)
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create_tenant(&self, tenant: &TenantId) -> Result<()> {
        Ok(self.coordinator.register(&key(tenant)).await?)
    }

    /// Forget the tenant and everything in it. Returns whether it existed.
    pub async fn drop_tenant(&self, tenant: &TenantId) -> Result<bool> {
        let existed = self.coordinator.remove(&key(tenant)).await?;
        self.opened().remove(tenant);
        Ok(existed)
    }

    /// Up to `limit` tenants after `after`, in id order.
    pub async fn tenants(&self, after: Option<&TenantId>, limit: usize) -> Result<Vec<TenantId>> {
        let after = after.map(|tenant| PartitionKey {
            tenant: tenant.clone(),
            partition: Partition(u32::MAX),
        });
        let keys = self.coordinator.partitions(after.as_ref(), limit).await?;
        Ok(keys.into_iter().map(|key| key.tenant).collect())
    }

    fn opened(&self) -> std::sync::MutexGuard<'_, HashMap<TenantId, Opening>> {
        self.open.lock().expect("tenants lock")
    }

    /// The tenant, opened if this is the first call for it or its writer
    /// was lost. Callers arriving together share one opening.
    pub async fn tenant(&self, tenant: &TenantId) -> Result<Arc<Tenant>> {
        let opening = {
            let mut open = self.opened();
            let lost = open
                .get(tenant)
                .and_then(|opening| opening.get())
                .is_some_and(|tenant| tenant.broken.load(Ordering::SeqCst));
            if lost {
                open.remove(tenant);
            }
            open.entry(tenant.clone()).or_default().clone()
        };
        opening
            .get_or_try_init(|| self.opening(tenant))
            .await
            .cloned()
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
        let writer = Writer::open(
            self.blobs.clone(),
            self.coordinator.clone(),
            lease,
            &committed,
        )
        .await?
        .embedding_with(Arc::new(Documents(self.embedders.clone())));
        let writer = Arc::new(writer);
        let (flushes, mut wanted) = mpsc::channel(1);
        let flushing = writer.clone();
        let flushed = tenant.clone();
        tokio::spawn(async move {
            while wanted.recv().await.is_some() {
                if let Err(error) = flushing.flush().await {
                    tracing::warn!(tenant = %flushed, %error, "flush failed");
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
            broken: AtomicBool::new(false),
        }))
    }

    /// Give up every tenant, so that another node can write for them at
    /// once rather than when the leases run out.
    pub async fn shutdown(&self) {
        let open: Vec<Arc<Tenant>> = self
            .opened()
            .drain()
            .filter_map(|(_, opening)| opening.get().cloned())
            .collect();
        for tenant in open {
            tenant.broken.store(true, Ordering::SeqCst);
            if let Err(error) = self.coordinator.release(tenant.writer.lease()).await {
                tracing::warn!(tenant = %tenant.id, %error, "lease not released");
            }
        }
    }
}
