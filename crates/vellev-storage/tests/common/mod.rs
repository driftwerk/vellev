#![allow(dead_code, unused_imports)]

pub use std::collections::{BTreeMap, BTreeSet};
pub use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
pub use std::sync::{Arc, Mutex};
pub use std::time::Duration;

pub use async_trait::async_trait;
pub use bytes::Bytes;
pub use chrono::{TimeZone, Utc};
pub use futures::future::join_all;
pub use object_store::memory::InMemory;
pub use proptest::prelude::*;
pub use serde_json::json;
pub use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition as PartitionNumber, Property, RequestId, TenantId, Value,
};
pub use vellev_storage::blobs::{Author, Blobs, CachedBlobs, Layout, Listed, Local, ObjectBlobs};
pub use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Forgotten, Generation, Incarnation,
    KeyHash, Lease, LogPosition, ManifestRef, ManualClock, MemoryCoordinator, PartitionKey, Point,
    Prior, RestorePoint, Restored, TenantKey, Versions,
};
pub use vellev_storage::format::{
    self, CollectionId, Kind, LogSegment, Manifest, Row, SegmentFile, SegmentIndex, SegmentMeta,
};
pub use vellev_storage::{Filter, Follower, Mutation, State, StorageError, Writer};

pub const TTL: Duration = Duration::from_secs(30);

pub fn name(collection: &str) -> CollectionName {
    CollectionName::new(collection).unwrap()
}

pub fn id(n: u8) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
}

pub fn schema(collection: &str) -> Collection {
    Collection {
        name: name(collection),
        properties: vec![
            Property::new("title", DataType::Text).searchable(),
            Property::new("n", DataType::Int),
            Property::new("score", DataType::Number),
            Property::new("tags", DataType::TextArray),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "test".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

pub fn create(collection: &str) -> Mutation {
    Mutation::CreateCollection(Box::new(schema(collection)))
}

pub fn upsert(collection: &str, object: u8, n: i64) -> Mutation {
    Mutation::Upsert {
        collection: name(collection),
        id: id(object),
        properties: json!({ "title": format!("row {object}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![n as f32, 0.5]),
        only_if_version: None,
    }
}

pub fn delete(collection: &str, object: u8) -> Mutation {
    Mutation::Delete {
        collection: name(collection),
        id: id(object),
        only_if_version: None,
    }
}

pub fn n_of(state: &State, collection: &str, object: u8) -> Option<i64> {
    state
        .get(&name(collection), id(object))
        .map(|found| match found.object.properties["n"] {
            Value::Int(n) => n,
            ref other => panic!("{other:?}"),
        })
}

#[derive(Debug)]
pub struct Observed {
    pub inner: Arc<dyn Blobs>,
    pub gets: Mutex<Vec<String>>,
    pub fail_puts: AtomicBool,
}

impl Observed {
    pub fn fetched(&self) -> Vec<String> {
        std::mem::take(&mut self.gets.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Observed {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.fail_puts.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.gets.lock().unwrap().push(key.to_string());
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// One partition's shared infrastructure: object storage and a coordinator.
pub struct Cluster {
    pub key: PartitionKey,
    pub blobs: Arc<dyn Blobs>,
    pub clock: Arc<ManualClock>,
    pub coordinator: Arc<MemoryCoordinator>,
}

impl Cluster {
    pub async fn new() -> Self {
        let clock = Arc::new(ManualClock::starting_at(
            Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
        ));
        let coordinator = Arc::new(MemoryCoordinator::new(clock.clone()));
        let key = PartitionKey {
            tenant: TenantId::new("tenant-a").unwrap(),
            partition: PartitionNumber::DEFAULT,
        };
        coordinator.register(&key).await.unwrap();
        Self {
            key,
            blobs: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
            clock,
            coordinator,
        }
    }

    pub async fn committed(&self) -> Committed {
        self.coordinator
            .committed(&self.key)
            .await
            .unwrap()
            .unwrap()
    }

    /// A new node takes the lease, waiting out whoever holds it.
    pub async fn lease(&self) -> (Lease, Committed) {
        let holder = NodeId::generate("node");
        if let Acquired::Lease { lease, committed } = self
            .coordinator
            .acquire(&self.key, &holder, TTL)
            .await
            .unwrap()
        {
            return (lease, committed);
        }
        self.clock.advance(TTL);
        match self
            .coordinator
            .acquire(&self.key, &holder, TTL)
            .await
            .unwrap()
        {
            Acquired::Lease { lease, committed } => (lease, committed),
            held => panic!("{held:?}"),
        }
    }

    pub async fn writer(&self) -> Writer {
        self.writer_on(self.coordinator.clone()).await
    }

    pub async fn writer_on(&self, coordinator: Arc<dyn Coordinator>) -> Writer {
        let (lease, committed) = self.lease().await;
        Writer::open(self.blobs.clone(), coordinator, lease, &committed)
            .await
            .unwrap()
    }

    pub async fn open(&self) -> Follower {
        self.open_on(self.blobs.clone()).await.unwrap()
    }

    pub async fn open_on(&self, blobs: Arc<dyn Blobs>) -> vellev_storage::Result<Follower> {
        Follower::open(self.key.clone(), blobs, &self.committed().await).await
    }

    /// The store wrapped so that reads are recorded and writes can be failed.
    pub fn observed(&self) -> Arc<Observed> {
        Arc::new(Observed {
            inner: self.blobs.clone(),
            gets: Mutex::default(),
            fail_puts: AtomicBool::new(false),
        })
    }
}

/// A small deterministic generator of values in -0.5..0.5.
pub struct Random(pub u64);

impl Random {
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    pub fn vector(&mut self, dimensions: usize) -> Vec<f32> {
        (0..dimensions).map(|_| self.unit()).collect()
    }
}

/// Blobs that, once armed, report each file of one kind they have stored
/// and then hold the upload until released, and report each listing taken.
#[derive(Debug)]
pub struct Paused {
    inner: Arc<dyn Blobs>,
    kind: &'static str,
    pub armed: AtomicBool,
    events: tokio::sync::mpsc::UnboundedSender<&'static str>,
    pub release: tokio::sync::Notify,
}

impl Paused {
    pub fn new(
        cluster: &mut Cluster,
        kind: &'static str,
    ) -> (
        Arc<Self>,
        tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    ) {
        let (events, seen) = tokio::sync::mpsc::unbounded_channel();
        let paused = Arc::new(Self {
            inner: cluster.blobs.clone(),
            kind,
            armed: AtomicBool::new(false),
            events,
            release: tokio::sync::Notify::new(),
        });
        cluster.blobs = paused.clone();
        (paused, seen)
    }
}

#[async_trait]
impl Blobs for Paused {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await?;
        if self.armed.load(Ordering::SeqCst) && key.contains(self.kind) {
            self.events.send("stored").unwrap();
            self.release.notified().await;
        }
        Ok(())
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        let listed = self.inner.list(prefix).await?;
        self.events.send("listed").unwrap();
        Ok(listed)
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// Purge a removed tenant as a node would, so that it may be made again.
pub async fn purge(coordinator: &dyn Coordinator, tenant: &TenantId) {
    let claim = coordinator
        .claim_purge(tenant, Duration::from_secs(60))
        .await
        .unwrap()
        .expect("a purge to claim");
    coordinator
        .purged(tenant, claim, Duration::from_secs(24 * 60 * 60))
        .await
        .unwrap();
}

/// A coordinator whose commit responses can be lost after the commit landed,
/// which can stop accepting commits altogether, answer them slowly, or
/// never answer them at all.
#[derive(Debug)]
pub struct Unreliable {
    pub inner: Arc<MemoryCoordinator>,
    pub lose_responses: AtomicUsize,
    pub refuse_commits: AtomicUsize,
    pub hang_after_commit: AtomicBool,
    /// Commits are never answered, and never land.
    pub unanswered: AtomicBool,
    /// How long each commit takes to be answered.
    pub slowly: Mutex<Duration>,
}

impl Unreliable {
    /// Over `inner`, faultless until told otherwise.
    pub fn over(inner: Arc<MemoryCoordinator>) -> Self {
        Self {
            inner,
            lose_responses: AtomicUsize::new(0),
            refuse_commits: AtomicUsize::new(0),
            hang_after_commit: AtomicBool::new(false),
            unanswered: AtomicBool::new(false),
            slowly: Mutex::new(Duration::ZERO),
        }
    }
}

fn lost() -> CoordinatorError {
    CoordinatorError::backend(std::io::Error::other("connection reset"))
}

#[async_trait]
impl Coordinator for Unreliable {
    async fn register(&self, key: &PartitionKey) -> Result<(), CoordinatorError> {
        self.inner.register(key).await
    }
    async fn remove(&self, key: &PartitionKey) -> Result<bool, CoordinatorError> {
        self.inner.remove(key).await
    }
    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired, CoordinatorError> {
        self.inner.acquire(key, holder, ttl).await
    }
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease, CoordinatorError> {
        self.inner.renew(lease, ttl).await
    }
    async fn release(&self, lease: &Lease) -> Result<(), CoordinatorError> {
        self.inner.release(lease).await
    }
    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed, CoordinatorError> {
        let take = |counter: &AtomicUsize| {
            counter
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        };
        if self.unanswered.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        let slowly = *self.slowly.lock().unwrap();
        tokio::time::sleep(slowly).await;
        if take(&self.refuse_commits) {
            return Err(lost());
        }
        let committed = self.inner.commit(lease, commit).await?;
        if self.hang_after_commit.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        if take(&self.lose_responses) {
            return Err(lost());
        }
        Ok(committed)
    }
    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>, CoordinatorError> {
        self.inner.committed(key).await
    }
    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>, CoordinatorError> {
        self.inner.partitions(after, limit).await
    }
    async fn ping(&self) -> Result<(), CoordinatorError> {
        self.inner.ping().await
    }
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<(), CoordinatorError> {
        self.inner.announce(node, versions).await
    }
    async fn depart(&self, node: &NodeId) -> Result<(), CoordinatorError> {
        self.inner.depart(node).await
    }
    async fn oldest(&self) -> Result<Option<Versions>, CoordinatorError> {
        self.inner.oldest().await
    }
    async fn format(&self) -> Result<u16, CoordinatorError> {
        self.inner.format().await
    }
    async fn raise_format(&self, format: u16) -> Result<u16, CoordinatorError> {
        self.inner.raise_format(format).await
    }

    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: uuid::Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>, CoordinatorError> {
        self.inner.issue_key(tenant, id, hash).await
    }

    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>, CoordinatorError> {
        self.inner.keys(tenant).await
    }

    async fn revoke_key(
        &self,
        tenant: &TenantId,
        id: uuid::Uuid,
    ) -> Result<bool, CoordinatorError> {
        self.inner.revoke_key(tenant, id).await
    }

    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>, CoordinatorError> {
        self.inner.key_tenant(hash).await
    }

    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>, CoordinatorError> {
        self.inner.restore_point(key, point, window).await
    }

    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>, CoordinatorError> {
        self.inner.restore_points(key, window).await
    }

    async fn forget_history(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<u64, CoordinatorError> {
        self.inner.forget_history(key, window).await
    }

    async fn restored(
        &self,
        key: &PartitionKey,
        request: RequestId,
    ) -> Result<Option<Restored>, CoordinatorError> {
        self.inner.restored(key, request).await
    }

    async fn forget_removed(
        &self,
        key: &PartitionKey,
        after: Option<Forgotten>,
        limit: usize,
    ) -> Result<Option<Forgotten>, CoordinatorError> {
        self.inner.forget_removed(key, after, limit).await
    }

    async fn forget_restores(&self) -> Result<u64, CoordinatorError> {
        self.inner.forget_restores().await
    }

    async fn restore_window(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<Duration>, CoordinatorError> {
        self.inner.restore_window(tenant).await
    }

    async fn purges(
        &self,
        due: Duration,
        lately: Duration,
    ) -> Result<vellev_storage::coordinator::Purges, CoordinatorError> {
        self.inner.purges(due, lately).await
    }

    async fn claim_purge(
        &self,
        tenant: &TenantId,
        hold: Duration,
    ) -> Result<Option<uuid::Uuid>, CoordinatorError> {
        self.inner.claim_purge(tenant, hold).await
    }

    async fn purged(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        forget_after: Duration,
    ) -> Result<(), CoordinatorError> {
        self.inner.purged(tenant, claim, forget_after).await
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<Option<uuid::Uuid>, CoordinatorError> {
        self.inner.claim_backup(tenant, due, hold).await
    }

    async fn release_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        made: bool,
    ) -> Result<(), CoordinatorError> {
        self.inner.release_backup(tenant, claim, made).await
    }

    async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool, CoordinatorError> {
        self.inner.set_restore_window(tenant, window).await
    }
}
