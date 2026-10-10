//! Measure how long a writer is held while it adopts a flush and a merge,
//! and while it commits the largest group a committer makes, on a copy of
//! a real tenant; and how much opening the tenant takes on the heap, held
//! and at the most, which with OPEN set is all it measures.
//!
//!   STORE=/path/to/store TENANT=acme HEAD=acme/0/manifest/….manifest \
//!     cargo run --release -p vellev-storage --example apply_timing
//!
//! STORE is a `file://` store's directory, which is only read: whatever
//! the writer writes is kept in memory. HEAD is the key of the tenant's
//! committed head, a manifest. ROWS (50000) rows are written before the
//! flush. The writer says how long it took at debug level, shown here.

use std::alloc::{GlobalAlloc, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use vellev_core::{NodeId, Partition, TenantId};
use vellev_storage::blobs::{Blobs, Listed, Local, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Coordinator, LogPosition, ManifestRef, MemoryCoordinator, PartitionKey,
};
use vellev_storage::{Batch, Merging, Mutation, State, StorageError, Weight, Writer, GROUP_MOST};

/// Counts the bytes allocated and not yet freed, and the most there were,
/// so that what opening a tenant takes on the heap is told apart from the
/// files it reads in place, which the system counts with it.
struct Counting;

static HELD: AtomicIsize = AtomicIsize::new(0);
static MOST: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every call is passed on to the system's allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, memory: std::alloc::Layout) -> *mut u8 {
        let held = HELD.fetch_add(memory.size() as isize, Ordering::Relaxed);
        MOST.fetch_max(held + memory.size() as isize, Ordering::Relaxed);
        // SAFETY: as the caller of this guarantees of `memory`.
        unsafe { System.alloc(memory) }
    }

    unsafe fn dealloc(&self, at: *mut u8, memory: std::alloc::Layout) {
        HELD.fetch_sub(memory.size() as isize, Ordering::Relaxed);
        // SAFETY: `at` was allocated here, by the system's allocator.
        unsafe { System.dealloc(at, memory) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// A store read from `under`, with whatever is written kept in `over`.
#[derive(Debug)]
struct Overlaid {
    under: ObjectBlobs,
    over: ObjectBlobs,
}

#[async_trait]
impl Blobs for Overlaid {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.over.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        match self.over.get(key).await {
            Err(StorageError::Missing { .. }) => self.under.get(key).await,
            found => found,
        }
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        let (mut listed, over) = (
            self.under.list(prefix).await?,
            self.over.list(prefix).await?,
        );
        listed.extend(over);
        Ok(listed)
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.over.delete(key).await
    }
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        self.under.local(key).await
    }
}

fn setting(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("vellev_storage=debug")
        .init();
    let (store, tenant, head) = (setting("STORE"), setting("TENANT"), setting("HEAD"));
    let rows: usize = std::env::var("ROWS")
        .ok()
        .and_then(|rows| rows.parse().ok())
        .unwrap_or(50_000);
    let position: u64 = head
        .rsplit('/')
        .next()
        .and_then(|name| name.get(..20)?.parse().ok())
        .expect("HEAD names a file written at a position");
    let key = PartitionKey {
        tenant: TenantId::new(tenant).expect("a tenant id"),
        partition: Partition::DEFAULT,
    };
    let under = LocalFileSystem::new_with_prefix(&store).expect("a store directory");
    let blobs: Arc<dyn Blobs> = Arc::new(Overlaid {
        under: ObjectBlobs::on_disk(Arc::new(under), &store),
        over: ObjectBlobs::new(Arc::new(InMemory::new())),
    });
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await.unwrap();
    let ttl = Duration::from_secs(3_600);
    let holder = NodeId::generate("timing");
    let Acquired::Lease { lease, .. } = coordinator.acquire(&key, &holder, ttl).await.unwrap()
    else {
        panic!("a fresh partition is free to take");
    };
    let committed = Commit {
        after: None,
        log_position: LogPosition(position),
        manifest: Some(ManifestRef(head)),
        restore: None,
        adopting: None,
    };
    let committed = coordinator.commit(&lease, committed).await.unwrap();

    let opening = Instant::now();
    let writer = Writer::open(blobs, coordinator, lease, &committed)
        .await
        .unwrap()
        .merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
    let shape = |state: &State| (state.rows(), state.segments().count());
    let (live, segments) = writer.partition().read(shape);
    println!(
        "opened in {:?}: {live} live rows in {segments} segments, {} MiB resident",
        opening.elapsed(),
        writer.partition().read(State::resident) / (1024 * 1024)
    );

    let mib = |bytes: &AtomicIsize| bytes.load(Ordering::Relaxed) / (1024 * 1024);
    println!(
        "on the heap: {} MiB held, {} MiB at the most",
        mib(&HELD),
        mib(&MOST)
    );
    // With OPEN set that is all: what opening the tenant takes, alone.
    if std::env::var_os("OPEN").is_some() {
        return;
    }

    // Rows like the tenant's own, written again under new ids.
    let copies: Vec<Mutation> = writer.partition().read(|state| {
        let collections: Vec<_> = state
            .collections()
            .map(|schema| schema.name.clone())
            .collect();
        collections
            .iter()
            .flat_map(|collection| {
                state.scan(collection).map(move |found| {
                    let object = found.to_object();
                    Mutation::Upsert {
                        collection: collection.clone(),
                        id: vellev_core::ObjectId(uuid::Uuid::new_v4()),
                        properties: object.properties_json(),
                        vector: object.vector,
                        only_if_version: None,
                    }
                })
            })
            .take(rows)
            .collect()
    });
    println!("writing {} rows", copies.len());
    for batch in copies.chunks(1_000) {
        writer.commit(batch.to_vec()).await.unwrap();
    }

    let flushing = Instant::now();
    writer.flush().await.unwrap();
    println!("flushed in {:?}", flushing.elapsed());
    for batch in copies.chunks(1_000).take(5) {
        writer.commit(batch.to_vec()).await.unwrap();
    }
    let merging = Instant::now();
    let merged = writer.merge().await.unwrap();
    println!("merged ({merged:?}) in {:?}", merging.elapsed());
    let (live, segments) = writer.partition().read(shape);
    println!("{live} live rows in {segments} segments");

    // The largest group a committer makes of such rows, in batches of ten:
    // as many as stay within its bound, written under new ids each time.
    let renamed = |mutation: &Mutation| match mutation.clone() {
        Mutation::Upsert {
            collection,
            properties,
            vector,
            ..
        } => Mutation::Upsert {
            collection,
            id: vellev_core::ObjectId(uuid::Uuid::new_v4()),
            properties,
            vector,
            only_if_version: None,
        },
        other => other,
    };
    // Once of the first rows written and twice of the last, which need not
    // be of a size.
    let last: Vec<Mutation> = copies.iter().rev().cloned().collect();
    for (round, rows) in [&copies, &last, &last].into_iter().enumerate() {
        let mut weight = Weight::default();
        let group: Vec<Batch> = rows
            .chunks(10)
            .map(|batch| Batch::from(batch.iter().map(renamed).collect::<Vec<_>>()))
            .take_while(|batch| {
                weight = weight + batch.weight();
                weight.within(&GROUP_MOST)
            })
            .collect();
        let weight: Weight = group.iter().map(Batch::weight).sum();
        let committing = Instant::now();
        let committed = writer.commit_batches(group.clone(), Default::default());
        committed.await.unwrap();
        println!(
            "group {round} of {} batches, {weight:?}, committed in {:?}",
            group.len(),
            committing.elapsed()
        );
    }

    // And the most mutations a group takes, each as small as one can be:
    // the rows written at first, deleted.
    let deletes: Vec<Mutation> = copies
        .iter()
        .filter_map(|mutation| match mutation {
            Mutation::Upsert { collection, id, .. } => Some(Mutation::Delete {
                collection: collection.clone(),
                id: *id,
                only_if_version: None,
            }),
            _ => None,
        })
        .take(GROUP_MOST.rows)
        .collect();
    let group: Vec<Batch> = deletes
        .chunks(10)
        .map(|batch| Batch::from(batch.to_vec()))
        .collect();
    let weight: Weight = group.iter().map(Batch::weight).sum();
    let committing = Instant::now();
    let committed = writer.commit_batches(group, Default::default());
    committed.await.unwrap();
    println!(
        "group of deletes, {weight:?}, committed in {:?}",
        committing.elapsed()
    );
}
