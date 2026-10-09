//! Measure the write path against a store with simulated latency.
//!
//!   cargo run --release -p vellev-storage --example write_throughput
//!
//! Environment: OBJECTS (150000), BATCH (1000), DIMS (768), PUT_MS (50),
//! COMMIT_MS (5), CALLERS (256).

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, RequestId, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Forgotten, KeyHash, Lease,
    MemoryCoordinator, PartitionKey, Point, RestorePoint, Restored, TenantKey, Versions,
};
use vellev_storage::{GroupCommitter, Mutation, Writer};

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Debug)]
struct SlowBlobs {
    inner: ObjectBlobs,
    put: Duration,
}

#[async_trait]
impl Blobs for SlowBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        tokio::time::sleep(self.put).await;
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }

    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[derive(Debug)]
struct SlowCoordinator {
    inner: MemoryCoordinator,
    commit: Duration,
}

#[async_trait]
impl Coordinator for SlowCoordinator {
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
        tokio::time::sleep(self.commit).await;
        self.inner.commit(lease, commit).await
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
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<u16, CoordinatorError> {
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

    async fn identity(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<vellev_storage::coordinator::Identity>, CoordinatorError> {
        self.inner.identity(tenant).await
    }

    async fn cluster(&self) -> Result<vellev_storage::coordinator::ClusterId, CoordinatorError> {
        self.inner.cluster().await
    }

    async fn adopt_on_commit(
        &self,
        key: &PartitionKey,
        generation: vellev_storage::coordinator::Generation,
        adoption: &vellev_storage::coordinator::Adoption,
    ) -> Result<bool, CoordinatorError> {
        self.inner.adopt_on_commit(key, generation, adoption).await
    }

    async fn adopt_no_more(&self, key: &PartitionKey) -> Result<(), CoordinatorError> {
        self.inner.adopt_no_more(key).await
    }

    async fn claim_purge(
        &self,
        tenant: &TenantId,
        due: Duration,
        hold: Duration,
    ) -> Result<Option<vellev_storage::coordinator::PurgeClaim>, CoordinatorError> {
        self.inner.claim_purge(tenant, due, hold).await
    }

    async fn renew_purge(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.inner.renew_purge(tenant, claim, hold).await
    }

    async fn look_overs(
        &self,
        lately: Duration,
        every: Duration,
        limit: usize,
    ) -> Result<Vec<vellev_storage::coordinator::Removed>, CoordinatorError> {
        self.inner.look_overs(lately, every, limit).await
    }

    async fn purged(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        forget_after: Duration,
    ) -> Result<(), CoordinatorError> {
        self.inner.purged(tenant, claim, forget_after).await
    }

    async fn reopen_purge(&self, tenant: &TenantId) -> Result<bool, CoordinatorError> {
        self.inner.reopen_purge(tenant).await
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.inner.claim_backup(tenant, claim, due, hold).await
    }

    async fn renew_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.inner.renew_backup(tenant, claim, hold).await
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

fn collection(dimensions: u32) -> Collection {
    Collection {
        name: CollectionName::new("Entities").expect("valid name"),
        properties: vec![
            Property::new("content", DataType::Text).searchable(),
            Property::new("repoId", DataType::Text),
            Property::new("line", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "supplied".into(),
            dimensions,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

fn entity(collection: &CollectionName, n: u64, dimensions: u32) -> Mutation {
    let properties = json!({
        "content": format!("fn entity_{n}() {{ /* a few hundred bytes of source text */ }}").repeat(4),
        "repoId": format!("repo-{}", n % 8),
        "line": n,
    });
    Mutation::Upsert {
        collection: collection.clone(),
        id: ObjectId::generate(),
        properties: properties.as_object().expect("an object").clone(),
        vector: Some(
            (0..dimensions)
                .map(|d| ((n + u64::from(d)) % 97) as f32 / 97.0)
                .collect(),
        ),
        only_if_version: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let objects = setting("OBJECTS", 150_000);
    let batch = setting("BATCH", 1_000);
    let dimensions = setting("DIMS", 768) as u32;
    let callers = setting("CALLERS", 256);
    let put = Duration::from_millis(setting("PUT_MS", 50));
    let commit = Duration::from_millis(setting("COMMIT_MS", 5));

    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(SlowCoordinator {
        inner: MemoryCoordinator::default(),
        commit,
    });
    coordinator.register(&key).await?;
    let blobs = Arc::new(SlowBlobs {
        inner: ObjectBlobs::new(Arc::new(InMemory::new())),
        put,
    });
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("bench"), Duration::from_secs(3600))
        .await?
    else {
        return Err("the lease is held".into());
    };
    let writer = Arc::new(Writer::open(blobs, coordinator.clone(), lease, &committed).await?);
    let schema = collection(dimensions);
    let name = schema.name.clone();
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(schema))])
        .await?;
    println!("simulated latency: {put:?} per upload, {commit:?} per coordinator commit\n");

    let started = Instant::now();
    let mut written = 0;
    while written < objects {
        let size = batch.min(objects - written);
        let mutations = (written..written + size)
            .map(|n| entity(&name, n, dimensions))
            .collect();
        writer.commit(mutations).await?;
        written += size;
    }
    let elapsed = started.elapsed();
    println!(
        "bulk import: {objects} objects of {dimensions} dims in batches of {batch}: {:.1}s, {:.0} objects/s, {} commits",
        elapsed.as_secs_f64(),
        objects as f64 / elapsed.as_secs_f64(),
        objects.div_ceil(batch),
    );

    let started = Instant::now();
    writer.flush().await?;
    println!(
        "checkpoint of all {objects} objects: {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let before = coordinator
        .committed(&key)
        .await?
        .map_or(0, |c| c.log_position.0);
    let committer = GroupCommitter::spawn(writer.clone(), 4096, 4096);
    let started = Instant::now();
    let tasks: Vec<_> = (0..callers)
        .map(|caller| {
            let (committer, name) = (committer.clone(), name.clone());
            tokio::spawn(async move {
                let mut slowest = Duration::ZERO;
                for n in 0..8 {
                    let asked = Instant::now();
                    committer
                        .submit(vec![entity(&name, caller * 8 + n, dimensions)])
                        .await?;
                    slowest = slowest.max(asked.elapsed());
                }
                Ok::<_, vellev_storage::StorageError>(slowest)
            })
        })
        .collect();
    let mut slowest = Duration::ZERO;
    for task in tasks {
        slowest = slowest.max(task.await??);
    }
    let elapsed = started.elapsed();
    let commits = coordinator
        .committed(&key)
        .await?
        .map_or(0, |c| c.log_position.0)
        - before;
    let writes = callers * 8;
    println!(
        "single writes: {callers} callers x 8 writes: {:.1}s, {:.0} writes/s, {commits} commits ({:.0} writes per commit), slowest write {slowest:?}",
        elapsed.as_secs_f64(),
        writes as f64 / elapsed.as_secs_f64(),
        writes as f64 / commits as f64,
    );
    println!(
        "without grouping the same writes would take about {:.0}s",
        writes as f64 * (put + commit).as_secs_f64()
    );
    Ok(())
}
