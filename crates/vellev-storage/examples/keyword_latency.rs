//! Measure keyword search over one flushed segment of generated text.
//!
//!   cargo run --release -p vellev-storage --example keyword_latency
//!
//! Environment: OBJECTS (200000), WORDS (120), VOCABULARY (50000),
//! QUERIES (200), TERMS (3). Words are drawn so that a few are very common
//! and most are rare, as in real text.

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, TenantId,
};
use vellev_storage::blobs::ObjectBlobs;
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Keywords, Mutation, Writer};

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

struct Words {
    state: u64,
    vocabulary: f64,
}

impl Words {
    fn next(&mut self) -> String {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let unit = (self.state >> 11) as f64 / (1u64 << 53) as f64;
        format!("w{}", self.vocabulary.powf(unit) as u64)
    }

    fn text(&mut self, words: u64) -> String {
        (0..words)
            .map(|_| self.next())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn collection(name: &CollectionName) -> Collection {
    Collection {
        name: name.clone(),
        properties: vec![Property::new("content", DataType::Text).searchable()],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "none".into(),
            dimensions: 1,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

#[tokio::main]
async fn main() {
    let objects = setting("OBJECTS", 200_000);
    let per_object = setting("WORDS", 120);
    let name = CollectionName::new("memories").unwrap();
    let mut words = Words {
        state: 7,
        vocabulary: setting("VOCABULARY", 50_000) as f64,
    };

    let key = PartitionKey {
        tenant: TenantId::new("bench").unwrap(),
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await.unwrap();
    let holder = NodeId::generate("bench");
    let ttl = Duration::from_secs(3600);
    let Acquired::Lease { lease, committed } =
        coordinator.acquire(&key, &holder, ttl).await.unwrap()
    else {
        unreachable!("nothing else holds the lease")
    };
    let blobs = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let writer = Writer::open(blobs, coordinator, lease, &committed)
        .await
        .unwrap();
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(collection(
            &name,
        )))])
        .await
        .unwrap();

    let started = Instant::now();
    for batch in (0..objects).collect::<Vec<_>>().chunks(5_000) {
        let puts = batch.iter().map(|n| Mutation::Upsert {
            collection: name.clone(),
            id: ObjectId(uuid::Uuid::from_u128(u128::from(*n))),
            properties: json!({ "content": words.text(per_object) })
                .as_object()
                .unwrap()
                .clone(),
            vector: None,
        });
        writer.commit(puts.collect()).await.unwrap();
    }
    println!(
        "{objects} objects of {per_object} words written in {:.1?}",
        started.elapsed()
    );
    let started = Instant::now();
    writer.flush().await.unwrap();
    println!("flushed to one segment in {:.1?}", started.elapsed());

    let terms = setting("TERMS", 3);
    let queries: Vec<String> = (0..setting("QUERIES", 200))
        .map(|_| words.text(terms))
        .collect();
    writer.partition().read(|state| {
        let mut taken: Vec<Duration> = queries
            .iter()
            .map(|query| {
                let started = Instant::now();
                let found = state
                    .keyword(&name, &Keywords::matching(query, 10))
                    .unwrap();
                std::hint::black_box(found);
                started.elapsed()
            })
            .collect();
        taken.sort_unstable();
        let at = |quantile: f64| taken[((taken.len() - 1) as f64 * quantile) as usize];
        println!(
            "{} queries of {terms} words: median {:.2?}, p99 {:.2?}, slowest {:.2?}",
            taken.len(),
            at(0.5),
            at(0.99),
            at(1.0)
        );
    });
}
