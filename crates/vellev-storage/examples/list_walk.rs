//! Measure what a page of a collection costs, and a walk of the whole of
//! it, in process: no store latency, no HTTP.
//!
//!   cargo run --release -p vellev-storage --example list_walk
//!
//! Environment: ROWS (1000000) rows are written in commits of 8192, with a
//! flush every FLUSH (500000) rows so the collection lies in several
//! segments and a memtable; ids are random, so id order is not write order.
//! Vectors have DIMS (2) dimensions, which only what a version costs
//! depends on. PAGE (100) is the page size of the full walk, made only with WALK set:
//! on a release before this one it costs the collection's size per page.
//! Each timing is the best of three.

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Filter, Follower, Live, Merging, Mutation, Query, State, Writer};

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn collection(dimensions: u32) -> Collection {
    Collection {
        name: CollectionName::new("Walked").expect("valid name"),
        properties: vec![
            Property::new("n", DataType::Int),
            Property::new("k", DataType::Text),
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

/// One in ten thousand rows has `k == "x"`.
const SPARSE: u64 = 10_000;

fn row(collection: &CollectionName, n: u64, dimensions: u32) -> Mutation {
    let k = if n.is_multiple_of(SPARSE) { "x" } else { "y" };
    let mut vector = vec![1.0; dimensions as usize];
    vector[0] = n as f32;
    Mutation::Upsert {
        collection: collection.clone(),
        id: ObjectId::generate(),
        properties: json!({ "n": n, "k": k })
            .as_object()
            .expect("an object")
            .clone(),
        vector: Some(vector),
        only_if_version: None,
    }
}

fn best_of_three(mut f: impl FnMut()) -> Duration {
    (0..3)
        .map(|_| {
            let started = Instant::now();
            f();
            started.elapsed()
        })
        .min()
        .expect("three runs")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = setting("ROWS", 1_000_000);
    let flush_every = setting("FLUSH", 500_000);
    let page = setting("PAGE", 100) as usize;
    let dimensions = setting("DIMS", 2) as u32;
    let walk = std::env::var_os("WALK").is_some();

    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await?;
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("bench"), Duration::from_secs(3600))
        .await?
    else {
        return Err("the lease is held".into());
    };
    let writer = Writer::open(blobs.clone(), coordinator.clone(), lease, &committed)
        .await?
        .merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
    let schema = collection(dimensions);
    let name = schema.name.clone();
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(schema))])
        .await?;

    let writing = Instant::now();
    let mut written = 0;
    let mut since_flush = 0;
    while written < rows {
        let size = 8_192.min(rows - written);
        let mutations = (written..written + size)
            .map(|n| row(&name, n, dimensions))
            .collect();
        writer.commit(mutations).await?;
        written += size;
        since_flush += size;
        if since_flush >= flush_every && written < rows {
            writer.flush().await?;
            since_flush = 0;
        }
    }
    let segments = writer.partition().read(|state| state.segments().count());
    println!(
        "{rows} rows written in {:.1}s, {segments} segments and a memtable of {} rows",
        writing.elapsed().as_secs_f64(),
        writer.partition().read(State::unflushed),
    );

    // What a node opening the tenant pays, with everything flushed: the
    // segments read and the live rows found among them.
    writer.flush().await?;
    let committed = coordinator.committed(&key).await?.expect("committed");
    let opening = Instant::now();
    let follower = Follower::open(key.clone(), blobs.clone(), &committed).await?;
    println!(
        "opened from {} segments in {:?}",
        follower.partition().read(|state| state.segments().count()),
        opening.elapsed()
    );
    drop(follower);

    let partition = writer.partition().clone();
    let ids: Vec<ObjectId> = partition.read(|state| {
        let mut ids: Vec<ObjectId> = state.scan(&name).map(|found| found.id()).collect();
        ids.sort_unstable();
        ids
    });
    let middle = ids[ids.len() / 2];
    let dense = Filter::Gte {
        property: "n".into(),
        value: json!(rows / 2),
    };
    let sparse = Filter::Eq {
        property: "k".into(),
        value: json!("x"),
    };
    let listed = |filter: Option<&Filter>, after: Option<ObjectId>, limit: usize| {
        partition.read(|state| {
            state
                .list(&name, filter, after, limit)
                .expect("a known collection")
                .iter()
                .map(Live::id)
                .collect::<Vec<_>>()
        })
    };
    let timed = |what: &str, filter: Option<&Filter>, after: Option<ObjectId>, limit: usize| {
        let mut found = 0;
        let took = best_of_three(|| found = listed(filter, after, limit).len());
        println!("{what}: {found} rows in {took:?}");
    };
    let got = best_of_three(|| {
        partition.read(|state| state.get(&name, middle).map(|found| found.to_object()));
    });
    println!("get: {got:?}");
    let versioned = best_of_three(|| {
        partition.read(|state| state.get(&name, middle).map(|found| found.version()));
    });
    println!("get, and its version: {versioned:?}");
    let versions = best_of_three(|| {
        partition.read(|state| {
            state
                .list(&name, None, None, 1_000)
                .expect("a known collection")
                .iter()
                .map(Live::version)
                .count()
        });
    });
    println!("page 1000 from the start, with versions: {versions:?}");
    // What answering such a page costs besides, roughly: each object's
    // properties and vector written as JSON, as the API writes them.
    let written = best_of_three(|| {
        partition.read(|state| {
            state
                .list(&name, None, None, 1_000)
                .expect("a known collection")
                .iter()
                .map(|found| {
                    let object = (found.object.properties_json(), found.vector);
                    serde_json::to_vec(&object).expect("JSON").len()
                })
                .sum::<usize>()
        });
    });
    println!("page 1000 from the start, written as JSON with vectors: {written:?}");
    let near = vec![0.5; dimensions as usize];
    let searched = best_of_three(|| {
        let query = Query::nearest(&near, 100);
        partition.read(|state| {
            state
                .search(&name, &query)
                .expect("a known collection")
                .len()
        });
    });
    println!("nearest top 100: {searched:?}");
    timed("page 50 from the start", None, None, 50);
    timed("page 1000 from the start", None, None, 1_000);
    timed("page 50 from the middle", None, Some(middle), 50);
    timed("page 50 past the end", None, Some(ids[ids.len() - 1]), 50);
    timed(
        "page 50, dense filter (n >= rows/2)",
        Some(&dense),
        None,
        50,
    );
    timed(
        "page 50, dense filter, from the middle",
        Some(&dense),
        Some(middle),
        50,
    );
    timed(
        "page 50, sparse filter (1 in 10000)",
        Some(&sparse),
        None,
        50,
    );
    timed(
        "page 50, sparse filter, from the middle",
        Some(&sparse),
        Some(middle),
        50,
    );

    if walk {
        for (what, filter) in [
            ("whole walk", None),
            ("whole walk, dense filter", Some(&dense)),
        ] {
            let started = Instant::now();
            let mut after = None;
            let mut pages = 0;
            let mut seen = 0;
            loop {
                let found = listed(filter, after, page);
                pages += 1;
                seen += found.len();
                if found.len() < page {
                    break;
                }
                after = found.last().copied();
            }
            let took = started.elapsed();
            println!(
                "{what} at page {page}: {seen} rows in {pages} pages, {took:?} ({:?} per page)",
                took / pages
            );
        }
    }
    Ok(())
}
