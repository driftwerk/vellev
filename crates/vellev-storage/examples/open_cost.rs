//! What opening a partition costs in CPU: rows written since the last
//! checkpoint replayed from their logs, or a checkpoint's segment loaded.
//!
//!   cargo run --release -p vellev-storage --example open_cost
//!
//! Environment: ROWS (50000), PER_COMMIT (500), DIMS (768), OPENS (5),
//! FLUSHED (0: replay the logs; 1: load the segment).

use std::sync::Arc;
use std::time::Instant;

use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Follower, Mutation, Writer};

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const WORDS: &[&str] = &[
    "deploy", "cluster", "memory", "agent", "recall", "vector", "search", "latency", "postgres",
    "lease", "tenant", "hub", "edge", "note", "meeting", "user", "prefers", "coffee", "python",
    "rust", "the", "a", "of", "to", "and", "in", "is", "it", "that", "for",
];

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

#[tokio::main]
async fn main() {
    let (rows, per_commit, dims, opens, flushed) = (
        setting("ROWS", 50_000),
        setting("PER_COMMIT", 500),
        setting("DIMS", 768),
        setting("OPENS", 5),
        setting("FLUSHED", 0) == 1,
    );
    let coordinator = Arc::new(MemoryCoordinator::default());
    let key = PartitionKey {
        tenant: TenantId::new("t").unwrap(),
        partition: vellev_core::Partition::DEFAULT,
    };
    coordinator.register(&key).await.unwrap();
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(
            &key,
            &NodeId::generate("n"),
            std::time::Duration::from_secs(3600),
        )
        .await
        .unwrap()
    else {
        panic!("lease")
    };
    let writer = Writer::open(blobs.clone(), coordinator.clone(), lease, &committed)
        .await
        .unwrap();
    let props = [
        "content",
        "memoryId",
        "layer",
        "owner",
        "visibility",
        "source",
        "identityId",
    ];
    let mut properties: Vec<Property> = props
        .iter()
        .map(|name| match *name {
            "content" => Property {
                filterable: false,
                ..Property::new(name, DataType::Text).searchable()
            },
            _ => Property::new(name, DataType::Text),
        })
        .collect();
    properties.push(Property::new("importance", DataType::Number));
    properties.push(Property::new("deleted", DataType::Bool));
    properties.push(Property::new("tags", DataType::TextArray));
    properties.push(Property::new("createdAt", DataType::Date));
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(Collection {
            name: CollectionName::new("memories").unwrap(),
            properties,
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "m".into(),
                dimensions: dims as u32,
                query_prefix: None,
                document_prefix: None,
                properties: vec![],
            },
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        }))])
        .await
        .unwrap();
    let mut random = Random(7);
    let mut written = 0u128;
    while (written as usize) < rows {
        let batch: Vec<Mutation> = (0..per_commit)
            .map(|_| {
                written += 1;
                let words = 8 + (random.next() % 50) as usize;
                let content: Vec<&str> = (0..words)
                    .map(|_| WORDS[(random.next() % WORDS.len() as u64) as usize])
                    .collect();
                Mutation::Upsert {
                    collection: CollectionName::new("memories").unwrap(),
                    id: ObjectId(uuid::Uuid::from_u128(written)),
                    properties: json!({
                        "content": content.join(" "),
                        "memoryId": uuid::Uuid::from_u128(written).to_string(),
                        "layer": if written.is_multiple_of(2) { "episodic" } else { "semantic" },
                        "owner": format!("agent-{}", written % 8),
                        "visibility": "private",
                        "source": "chat",
                        "identityId": format!("user-{}", written % 20),
                        "importance": (written % 100) as f64 / 100.0,
                        "deleted": written.is_multiple_of(20),
                        "tags": ["a", "b"],
                        "createdAt": "2026-10-01T10:00:00Z",
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    vector: Some((0..dims).map(|_| random.unit()).collect()),
                    only_if_version: None,
                }
            })
            .collect();
        writer.commit(batch).await.unwrap();
    }
    if flushed {
        writer.flush().await.unwrap();
    }
    let committed = coordinator.committed(&key).await.unwrap().unwrap();
    eprintln!("opening");
    let mut took = Vec::new();
    for _ in 0..opens {
        let began = Instant::now();
        let copy =
            Follower::open_through(key.clone(), blobs.clone(), coordinator.clone(), &committed)
                .await
                .unwrap();
        took.push(began.elapsed());
        drop(copy);
    }
    took.sort();
    println!(
        "{rows} rows of {dims} dims, {} : open {:?} (median of {opens}: {:?})",
        if flushed {
            "one segment"
        } else {
            "replayed from logs"
        },
        took[took.len() / 2],
        took
    );
}
