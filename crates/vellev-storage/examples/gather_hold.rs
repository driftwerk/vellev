//! How long a merge's gather keeps a flush and a commit of its tenant
//! waiting: each is timed from just after a merge's first chunk of rows is
//! copied, with the merge stopped once both are done, and with no merge
//! under way, to tell what it costs to wait from what each takes anyway.
//! The building lock and the state's are all a gather can keep them behind.
//!
//!   cargo run --release -p vellev-storage --example gather_hold
//!
//! Environment: ROWS (100000, a segment), SEGMENTS (4), DIMS (768),
//! RUNS (5).

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{MergeBuilds, Merging, Mutation, Writer};

const PER_COMMIT: usize = 500;

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn notes(dimensions: u32) -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: CollectionName::new("notes").unwrap(),
        properties: vec![Property::new("text", DataType::Text).searchable()],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "stand-in".into(),
            dimensions,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

/// A note of about 700 bytes of text, as the bench's passages are.
fn note(n: u64, dims: usize, random: &mut Random) -> Mutation {
    let text: Vec<String> = (0..80)
        .map(|_| format!("w{:06}", random.next() % 200_000))
        .collect();
    Mutation::Upsert {
        collection: CollectionName::new("notes").unwrap(),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1)),
        properties: json!({ "text": text.join(" ") })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some((0..dims).map(|_| random.unit()).collect()),
        only_if_version: None,
    }
}

/// A flush and a commit, side by side, each timed.
async fn flush_and_commit(writer: &Writer, n: u64, dims: usize) -> (Duration, Duration) {
    let asked = Instant::now();
    let flushing = async {
        writer.flush().await.unwrap();
        asked.elapsed()
    };
    let committing = async {
        let mut random = Random(n | 1);
        let written = note(n, dims, &mut random);
        writer.commit(vec![written]).await.unwrap();
        asked.elapsed()
    };
    tokio::join!(flushing, committing)
}

#[tokio::main]
async fn main() {
    let (rows, segments, dims, runs) = (
        setting("ROWS", 100_000),
        setting("SEGMENTS", 4),
        setting("DIMS", 768),
        setting("RUNS", 5),
    );
    let coordinator = Arc::new(MemoryCoordinator::default());
    let key = PartitionKey {
        tenant: TenantId::new("t").unwrap(),
        partition: vellev_core::Partition::DEFAULT,
    };
    coordinator.register(&key).await.unwrap();
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let hour = Duration::from_secs(3_600);
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("n"), hour)
        .await
        .unwrap()
    else {
        panic!("a fresh partition is free to take")
    };
    let merging = Merging {
        fanout: segments,
        ..Merging::default()
    };
    let builds = MergeBuilds::new(1);
    let writer = Writer::open(blobs, coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_with(merging)
        .merging_among(builds.clone());
    let writer = Arc::new(writer);
    writer.commit(vec![notes(dims as u32)]).await.unwrap();
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let began = Instant::now();
    for segment in 0..segments {
        for from in (0..rows).step_by(PER_COMMIT) {
            let batch = (from..(from + PER_COMMIT).min(rows))
                .map(|n| note((segment * rows + n) as u64, dims, &mut random))
                .collect();
            writer.commit(batch).await.unwrap();
        }
        writer.flush().await.unwrap();
        println!("segment {segment} flushed at {:?}", began.elapsed());
    }

    let mut n = (segments * rows) as u64;
    let mut timed = |writer: &Arc<Writer>| {
        n += 1;
        let writer = writer.clone();
        let at = n;
        async move { flush_and_commit(&writer, at, dims).await }
    };
    for run in 0..runs {
        let (flush_alone, commit_alone) = timed(&writer).await;
        // What the last gather came to, until this one is seen to begin.
        let last = builds.last_gathered();
        let merge = tokio::spawn({
            let writer = writer.clone();
            async move { writer.merge().await }
        });
        while builds.last_gathered() == last {
            tokio::time::sleep(Duration::from_micros(50)).await;
        }
        while builds.last_gathered() == 0 {
            tokio::time::sleep(Duration::from_micros(50)).await;
        }
        let (flush, commit) = timed(&writer).await;
        writer.stop_merges();
        let merged = merge.await.unwrap().unwrap();
        assert!(merged.is_none(), "a stopped merge commits nothing");
        println!(
            "run {run}: with a gather under way, flush {flush:?}, commit {commit:?}; \
             with none, flush {flush_alone:?}, commit {commit_alone:?}"
        );
    }
}
