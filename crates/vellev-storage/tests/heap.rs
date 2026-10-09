//! How much memory a node takes to open a partition whose vectors it has
//! no copy of yet. The one test here has a binary to itself, since what it
//! measures is everything the process allocates meanwhile.

mod common;

use std::alloc::{GlobalAlloc, Layout as Memory, System};
use std::sync::atomic::AtomicIsize;

use common::*;
use vellev_storage::Transfers;

/// Counts the bytes allocated and not yet freed, and the most there were.
struct Counting;

static HELD: AtomicIsize = AtomicIsize::new(0);
static MOST: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every call is passed on to the system's allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, memory: Memory) -> *mut u8 {
        let held = HELD.fetch_add(memory.size() as isize, Ordering::Relaxed);
        MOST.fetch_max(held + memory.size() as isize, Ordering::Relaxed);
        // SAFETY: as the caller of this guarantees of `memory`.
        unsafe { System.alloc(memory) }
    }

    unsafe fn dealloc(&self, at: *mut u8, memory: Memory) {
        HELD.fetch_sub(memory.size() as isize, Ordering::Relaxed);
        // SAFETY: `at` was allocated here, by the system's allocator.
        unsafe { System.dealloc(at, memory) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const DIMENSIONS: usize = 4_096;
const ROWS: u16 = 1_024;
const MIB: isize = 1024 * 1024;

fn wide(n: u16) -> Mutation {
    Mutation::Upsert {
        collection: name("wide"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "n": n }).as_object().unwrap().clone(),
        vector: Some(vec![f32::from(n) + 1.0; DIMENSIONS]),
        only_if_version: None,
    }
}

#[tokio::test]
async fn vectors_with_no_copy_yet_are_fetched_and_read_without_being_held_in_memory() {
    // Sixteen mebibytes of vectors in one segment, in a store read a
    // quarter of a mebibyte at a time.
    let mut cluster = Cluster::new().await;
    let parts = Transfers {
        part_bytes: 256 * 1024,
        ..Transfers::default()
    };
    let store = ObjectBlobs::new(Arc::new(InMemory::new())).transferring(parts);
    cluster.blobs = Arc::new(store);
    let mut collection = schema("wide");
    collection.embedding.dimensions = DIMENSIONS as u32;
    let writer = cluster.writer().await;
    let create = Mutation::CreateCollection(Box::new(collection));
    writer.commit(vec![create]).await.unwrap();
    for batch in (0..ROWS).collect::<Vec<_>>().chunks(128) {
        let rows = batch.iter().map(|n| wide(*n)).collect();
        writer.commit(rows).await.unwrap();
    }
    writer.flush().await.unwrap();
    drop(writer);
    let vectors = (DIMENSIONS * usize::from(ROWS) * 4) as isize;
    assert_eq!(vectors, 16 * MIB);

    let dir = tempfile::tempdir().unwrap();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path()));
    let before = HELD.load(Ordering::Relaxed);
    MOST.store(before, Ordering::Relaxed);
    let opened = cluster.open_on(cached).await.unwrap();
    let (most, after) = (MOST.load(Ordering::Relaxed), HELD.load(Ordering::Relaxed));

    let read = |state: &State| {
        let last = state.get(&name("wide"), ObjectId(uuid::Uuid::from_u128(1_023)));
        last.unwrap().vector.map(|vector| vector[DIMENSIONS - 1])
    };
    assert_eq!(opened.partition().read(read), Some(1_024.0));
    // A part or two of the file at a time, and the rows without their
    // vectors: a small share of the file, never the whole of it.
    assert!(
        most - before < 4 * MIB,
        "{} bytes at the most",
        most - before
    );
    assert!(after - before < 2 * MIB, "{} bytes kept", after - before);
}
