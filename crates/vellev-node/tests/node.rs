use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::extract::State as Shared;
use axum::routing::post;
use axum::{Json, Router};
use bytes::Bytes;
use chrono::{TimeZone, Utc};
use futures::future::join_all;
use object_store::memory::InMemory;
use serde_json::{json, Value};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, HttpApi,
    NodeId, ObjectId, Partition, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, NodeError, Settings};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Lease, ManualClock,
    MemoryCoordinator, PartitionKey,
};
use vellev_storage::{Mutation, State, StorageError};

const LEASE: Duration = Duration::from_secs(30);

/// Object storage whose writes can be made to fail.
#[derive(Debug)]
struct Store {
    inner: ObjectBlobs,
    failing: AtomicBool,
    unreadable: AtomicBool,
    /// How long each read takes, when reads are to be slow.
    slow: std::sync::Mutex<Option<Duration>>,
}

#[async_trait]
impl Blobs for Store {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        let delay = *self.slow.lock().unwrap();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        if self.unreadable.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// What nodes of one cluster share.
struct Cluster {
    clock: Arc<ManualClock>,
    coordinator: Arc<MemoryCoordinator>,
    store: Arc<Store>,
}

impl Cluster {
    fn new() -> Self {
        let clock = Arc::new(ManualClock::starting_at(
            Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
        ));
        Self {
            coordinator: Arc::new(MemoryCoordinator::new(clock.clone())),
            clock,
            store: Arc::new(Store {
                inner: ObjectBlobs::new(Arc::new(InMemory::new())),
                failing: AtomicBool::new(false),
                unreadable: AtomicBool::new(false),
                slow: std::sync::Mutex::new(None),
            }),
        }
    }

    fn node_with(&self, flush_rows: usize) -> Node {
        let settings = Settings {
            lease: LEASE,
            flush_rows,
            grace: Duration::ZERO,
            ..Settings::new(NodeId::generate("node"))
        };
        Node::new(
            self.coordinator.clone(),
            self.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            settings,
        )
    }

    fn node(&self) -> Node {
        self.node_with(50_000)
    }
}

fn tenant(id: &str) -> TenantId {
    TenantId::new(id).unwrap()
}

fn notes() -> CollectionName {
    CollectionName::new("notes").unwrap()
}

fn schema(provider: EmbeddingProvider) -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: notes(),
        properties: vec![
            Property::new("title", DataType::Text).searchable(),
            Property::new("n", DataType::Int),
        ],
        embedding: Embedding {
            provider,
            model: "stand-in".into(),
            dimensions: 2,
            query_prefix: Some("find: ".into()),
            document_prefix: None,
            properties: vec!["title".into()],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

fn id(n: u64) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
}

fn note(n: u64, title: &str, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Upsert {
        collection: notes(),
        id: id(n),
        properties: json!({ "title": title, "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector,
    }
}

fn count(state: &State) -> usize {
    state.count(&notes())
}

#[tokio::test]
async fn tenants_are_created_listed_a_page_at_a_time_and_dropped() {
    let cluster = Cluster::new();
    let node = cluster.node();
    node.ready().await.unwrap();
    assert!(matches!(
        node.tenant(&tenant("acme")).await,
        Err(NodeError::UnknownTenant(unknown)) if unknown == tenant("acme")
    ));
    for name in ["zeta", "acme", "acme", "mid"] {
        node.create_tenant(&tenant(name)).await.unwrap();
    }
    assert_eq!(
        node.tenants(None, 2).await.unwrap(),
        [tenant("acme"), tenant("mid")]
    );
    assert_eq!(
        node.tenants(Some(&tenant("mid")), 2).await.unwrap(),
        [tenant("zeta")]
    );
    assert_eq!(node.tenants(Some(&tenant("zeta")), 2).await.unwrap(), []);
    let second = PartitionKey {
        tenant: tenant("mid"),
        partition: Partition(1),
    };
    cluster.coordinator.register(&second).await.unwrap();
    assert_eq!(
        node.tenants(None, 10).await.unwrap(),
        [tenant("acme"), tenant("mid"), tenant("zeta")]
    );
    assert_eq!(
        node.tenants(Some(&tenant("mid")), 10).await.unwrap(),
        [tenant("zeta")]
    );
    cluster.coordinator.remove(&second).await.unwrap();

    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    assert_eq!(acme.read(count), 1);
    assert_eq!(node.tenant(&tenant("mid")).await.unwrap().read(count), 0);

    assert!(node.drop_tenant(&tenant("acme")).await.unwrap());
    assert!(!node.drop_tenant(&tenant("acme")).await.unwrap());
    assert!(matches!(
        node.tenant(&tenant("acme")).await,
        Err(NodeError::UnknownTenant(_))
    ));
    let orphaned = acme.write(vec![note(2, "hen", None)]).await.unwrap_err();
    assert!(
        matches!(&orphaned, NodeError::UnknownTenant(id) if *id == tenant("acme")),
        "{orphaned}"
    );
    for n in 0..50 {
        let ghost = tenant(&format!("ghost-{n}"));
        assert!(matches!(
            node.tenant(&ghost).await,
            Err(NodeError::UnknownTenant(_))
        ));
    }
    assert!(
        !format!("{node:?}").contains("ghost-"),
        "unopened tenants are not kept"
    );

    node.create_tenant(&tenant("acme")).await.unwrap();
    let anew = node.tenant(&tenant("acme")).await.unwrap();
    assert!(anew.read(|state| state.collection(&notes()).is_none()));
}

#[tokio::test]
async fn callers_arriving_together_share_one_opening_and_few_commits() {
    let cluster = Cluster::new();
    let node = Arc::new(cluster.node());
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = tenant("acme");
    let opened = join_all((0..16).map(|_| node.tenant(&acme))).await;
    let first = opened[0].as_ref().unwrap();
    assert!(opened
        .iter()
        .all(|other| Arc::ptr_eq(other.as_ref().unwrap(), first)));

    first
        .write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    let before = first.read(State::position);
    let writes = (0..200).map(|n| {
        let node = node.clone();
        tokio::spawn(async move {
            let acme = node.tenant(&tenant("acme")).await.unwrap();
            acme.write(vec![note(n, "fox", None)])
                .await
                .unwrap()
                .position
        })
    });
    let positions: Vec<_> = join_all(writes)
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let after = first.read(State::position);
    assert_eq!(first.read(count), 200);
    assert!(positions
        .iter()
        .all(|position| *position > before && *position <= after));
    assert!(
        after.0 - before.0 < 200,
        "{} commits for 200 writes",
        after.0 - before.0
    );
}

#[tokio::test]
async fn a_tenant_is_written_by_one_node_until_it_lets_go_or_its_hold_runs_out() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.node(), cluster.node());
    first.create_tenant(&tenant("acme")).await.unwrap();
    let held = first.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    assert!(matches!(
        second.tenant(&tenant("acme")).await,
        Err(NodeError::Elsewhere { tenant: t, .. }) if t == tenant("acme")
    ));

    first.shutdown().await;
    let taken = second.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(taken.read(count), 1);
    taken.write(vec![note(2, "hen", None)]).await.unwrap();
    let fenced = held.write(vec![note(3, "owl", None)]).await.unwrap_err();
    assert!(
        matches!(
            &fenced,
            NodeError::Storage(error)
                if matches!(error.root(), StorageError::Coordinator(CoordinatorError::Fenced { .. }))
        ),
        "{fenced}"
    );
    assert!(matches!(
        first.tenant(&tenant("acme")).await,
        Err(NodeError::Stopped)
    ));
    let first = cluster.node();
    assert!(matches!(
        first.tenant(&tenant("acme")).await,
        Err(NodeError::Elsewhere { .. })
    ));

    cluster.clock.advance(LEASE);
    let back = first.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(back.read(count), 2);
    assert!(taken.write(vec![note(4, "elk", None)]).await.is_err());
    assert!(matches!(
        second.tenant(&tenant("acme")).await,
        Err(NodeError::Elsewhere { .. })
    ));
    back.write(vec![note(5, "ant", None)]).await.unwrap();
    assert_eq!(back.read(count), 3);
}

#[tokio::test]
async fn a_writer_lost_to_a_failed_write_is_replaced_on_the_next_call() {
    let cluster = Cluster::new();
    let node = cluster.node();
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    cluster.store.failing.store(true, Ordering::SeqCst);
    assert!(acme.write(vec![note(2, "hen", None)]).await.is_err());
    cluster.store.failing.store(false, Ordering::SeqCst);
    let closed = acme.write(vec![note(2, "hen", None)]).await.unwrap_err();
    assert!(
        matches!(&closed, NodeError::Storage(error) if matches!(error.root(), StorageError::Closed)),
        "{closed}"
    );

    let reopened = node.tenant(&tenant("acme")).await.unwrap();
    assert!(!Arc::ptr_eq(&acme, &reopened));
    reopened.write(vec![note(2, "hen", None)]).await.unwrap();
    assert_eq!(reopened.read(count), 2);
    assert!(Arc::ptr_eq(
        &reopened,
        &node.tenant(&tenant("acme")).await.unwrap()
    ));

    let refused = reopened
        .write(vec![Mutation::DropCollection(
            CollectionName::new("nope").unwrap(),
        )])
        .await
        .unwrap_err();
    assert!(matches!(
        refused,
        NodeError::Storage(StorageError::UnknownCollection(_))
    ));
    assert!(Arc::ptr_eq(
        &reopened,
        &node.tenant(&tenant("acme")).await.unwrap()
    ));
}

#[tokio::test]
async fn a_tenant_is_flushed_once_enough_has_been_written() {
    let cluster = Cluster::new();
    let node = cluster.node_with(10);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    for n in 0..9 {
        acme.write(vec![note(n, "fox", None)]).await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(acme.read(State::unflushed), 9);
    acme.write(vec![note(9, "fox", None)]).await.unwrap();
    for _ in 0..200 {
        if acme.read(State::unflushed) == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(acme.read(State::unflushed), 0);
    assert_eq!(acme.read(count), 10);

    acme.write(vec![note(10, "hen", None)]).await.unwrap();
    assert_eq!(acme.read(State::unflushed), 1);
    acme.flush().await.unwrap();
    assert_eq!(acme.read(State::unflushed), 0);
}

#[tokio::test]
async fn small_segments_are_merged_after_a_flush_and_the_old_files_cleaned_up() {
    let cluster = Cluster::new();
    let node = cluster.node_with(10);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    for n in 0..40 {
        acme.write(vec![note(n, "fox", None)]).await.unwrap();
        if n % 10 == 9 {
            for _ in 0..500 {
                if acme.read(State::unflushed) == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(acme.read(State::unflushed), 0, "flushed ten at a time");
        }
    }
    let stored = || async {
        let listed = cluster.store.list("acme/").await.unwrap();
        listed
            .into_iter()
            .filter(|listed| listed.key.ends_with(".segment"))
            .count()
    };
    for _ in 0..500 {
        if acme.read(|state| state.segments().count()) == 1 && stored().await == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let merged: Vec<(u64, u64)> = acme.read(|state| {
        state
            .segments()
            .map(|segment| (segment.rows, segment.tombstones))
            .collect()
    });
    assert_eq!(merged, vec![(40, 0)]);
    assert_eq!(stored().await, 1, "the merged segments are gone");
    assert_eq!(acme.read(count), 40);
    assert!(acme.read(|state| state.get(&notes(), id(7)).is_some()));
}

/// A stand-in embedding server. Under `/lengths` a text's vector is its
/// length and its number of words; under `/swapped`, the other way round.
async fn embedding_server(calls: Arc<AtomicUsize>) -> String {
    async fn embed(calls: Arc<AtomicUsize>, body: Value, swapped: bool) -> Json<Value> {
        calls.fetch_add(1, Ordering::SeqCst);
        let vectors: Vec<[usize; 2]> = body["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| text.as_str().unwrap())
            .map(|text| [text.len(), text.split_whitespace().count()])
            .map(|[length, words]| {
                if swapped {
                    [words, length]
                } else {
                    [length, words]
                }
            })
            .collect();
        Json(json!(vectors))
    }
    let lengths = |Shared(calls): Shared<Arc<AtomicUsize>>, Json(body): Json<Value>| {
        embed(calls, body, false)
    };
    let swapped =
        |Shared(calls): Shared<Arc<AtomicUsize>>, Json(body): Json<Value>| embed(calls, body, true);
    let router = Router::new()
        .route("/lengths/embed", post(lengths))
        .route("/swapped/embed", post(swapped))
        .with_state(calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

fn served_by(url: &str, flavour: &str) -> EmbeddingProvider {
    EmbeddingProvider::Http {
        url: format!("{url}/{flavour}"),
        api: HttpApi::Tei,
    }
}

#[tokio::test]
async fn a_collection_that_embeds_its_objects_is_embedded_on_write_and_on_query() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let calls = Arc::new(AtomicUsize::new(0));
    let url = embedding_server(calls.clone()).await;
    let embedded = served_by(&url, "lengths");
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![
        schema(embedded),
        note(1, "a red fox", None),
        note(2, "supplied", Some(vec![1.0, 0.0])),
    ])
    .await
    .unwrap();
    let vector = |n: u64| acme.read(|state| state.get(&notes(), id(n)).unwrap().vector.clone());
    assert_eq!(vector(1), Some(vec![9.0, 3.0]));
    assert_eq!(vector(2), Some(vec![1.0, 0.0]));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let query = acme.query_vector(&notes(), "red fox").await.unwrap();
    assert_eq!(query, Some(vec![13.0, 3.0]));
    assert_eq!(acme.query_vector(&notes(), "red fox").await.unwrap(), query);
    assert_eq!(acme.query_vector(&notes(), "  ").await.unwrap(), None);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        acme.query_vector(&CollectionName::new("nope").unwrap(), "fox")
            .await,
        Err(NodeError::Storage(StorageError::UnknownCollection(_)))
    ));

    acme.write(vec![note(1, "a red fox", None), note(3, "hen", None)])
        .await
        .unwrap();
    assert_eq!(vector(3), Some(vec![3.0, 1.0]));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "only the new text is embedded"
    );
    acme.write(vec![note(1, "a red fox", None)]).await.unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "stored text is not embedded again"
    );
    let before = acme.read(State::position);
    let together =
        join_all((10..40).map(|n| acme.write(vec![note(n, &format!("note {n}"), None)])));
    assert!(together.await.iter().all(Result::is_ok));
    let commits = acme.read(State::position).0 - before.0;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        33,
        "each caller embeds for itself"
    );
    assert!(
        commits < 30,
        "{commits} commits for 30 callers embedding side by side"
    );
    assert!((10..40).all(|n| vector(n) == Some(vec![7.0, 2.0])));
    let calls_so_far = calls.load(Ordering::SeqCst);

    node.create_tenant(&tenant("plain")).await.unwrap();
    let plain = node.tenant(&tenant("plain")).await.unwrap();
    plain
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    assert_eq!(plain.query_vector(&notes(), "fox").await.unwrap(), None);
    assert_eq!(
        plain.read(|state| state.get(&notes(), id(1)).unwrap().vector.clone()),
        None
    );
    assert_eq!(calls.load(Ordering::SeqCst), calls_so_far);
}

#[tokio::test]
async fn a_vector_is_made_the_way_the_collection_embeds_when_the_write_lands() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let url = embedding_server(Arc::new(AtomicUsize::new(0))).await;
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![
        schema(served_by(&url, "lengths")),
        note(1, "a red fox", None),
    ])
    .await
    .unwrap();
    let vector = |n: u64| acme.read(|state| state.get(&notes(), id(n)).unwrap().vector.clone());
    assert_eq!(vector(1), Some(vec![9.0, 3.0]));

    acme.write(vec![
        Mutation::DropCollection(notes()),
        schema(served_by(&url, "swapped")),
        note(1, "a red fox", None),
    ])
    .await
    .unwrap();
    assert_eq!(vector(1), Some(vec![3.0, 9.0]));

    acme.write(vec![
        Mutation::DropCollection(notes()),
        schema(EmbeddingProvider::None),
        note(1, "a red fox", None),
    ])
    .await
    .unwrap();
    assert_eq!(vector(1), None);
}

#[tokio::test]
async fn an_opening_that_fails_lets_go_of_the_tenant_and_a_stopped_node_opens_nothing() {
    let cluster = Cluster::new();
    let (first, second, third) = (cluster.node(), cluster.node(), cluster.node());
    first.create_tenant(&tenant("acme")).await.unwrap();
    let acme = first.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    first.shutdown().await;

    cluster.store.unreadable.store(true, Ordering::SeqCst);
    assert!(matches!(
        second.tenant(&tenant("acme")).await,
        Err(NodeError::Storage(_))
    ));
    cluster.store.unreadable.store(false, Ordering::SeqCst);
    assert_eq!(third.tenant(&tenant("acme")).await.unwrap().read(count), 1);

    third.shutdown().await;
    assert!(matches!(
        third.tenant(&tenant("acme")).await,
        Err(NodeError::Stopped)
    ));
    assert_eq!(second.tenant(&tenant("acme")).await.unwrap().read(count), 1);
}

#[tokio::test]
async fn a_writer_lost_to_a_failed_flush_is_replaced_on_the_next_call() {
    let cluster = Cluster::new();
    let node = cluster.node();
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    cluster.store.failing.store(true, Ordering::SeqCst);
    assert!(acme.flush().await.is_err());
    cluster.store.failing.store(false, Ordering::SeqCst);

    let reopened = node.tenant(&tenant("acme")).await.unwrap();
    assert!(!Arc::ptr_eq(&acme, &reopened));
    reopened.write(vec![note(2, "hen", None)]).await.unwrap();
    assert_eq!(reopened.read(count), 2);
}

#[tokio::test]
async fn a_tenant_is_held_while_its_node_lives_and_passes_to_another_when_it_dies() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(300);
    let node_with_lease = || {
        let settings = Settings {
            lease,
            ..Settings::new(NodeId::generate("node"))
        };
        Node::new(
            cluster.coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            settings,
        )
    };
    let (first, second) = (node_with_lease(), node_with_lease());
    first.create_tenant(&tenant("acme")).await.unwrap();
    let acme = first.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    for n in 0..5 {
        acme.write(vec![note(n, "fox", None)]).await.unwrap();
        cluster.clock.advance(lease * 2 / 3);
        tokio::time::sleep(lease / 2).await;
        assert!(
            matches!(
                second.tenant(&tenant("acme")).await,
                Err(NodeError::Elsewhere { .. })
            ),
            "renewed, so still held after {n} rounds"
        );
    }

    drop(acme);
    drop(first);
    cluster.clock.advance(lease * 2);
    let taken = second.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(taken.read(count), 5, "every acknowledged write is there");
    taken.write(vec![note(9, "hen", None)]).await.unwrap();
}

#[tokio::test]
async fn a_node_notices_that_another_took_its_tenant() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(1_500);
    let node_with_lease = || {
        let settings = Settings {
            lease,
            ..Settings::new(NodeId::generate("node"))
        };
        Node::new(
            cluster.coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            settings,
        )
    };
    let (first, second) = (node_with_lease(), node_with_lease());
    first.create_tenant(&tenant("acme")).await.unwrap();
    let held = first.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    cluster.clock.advance(lease * 2);
    let taken = second.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(taken.read(count), 1);

    let started = std::time::Instant::now();
    let mut noticed = None;
    for _ in 0..40 {
        if matches!(
            first.tenant(&tenant("acme")).await,
            Err(NodeError::Elsewhere { .. })
        ) {
            noticed = Some(started.elapsed());
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let noticed = noticed.expect("the first node still hands out the tenant it lost");
    assert!(
        noticed < lease * 2 / 3,
        "noticed at the next renewal, not when the hold ran out: {noticed:?}"
    );
    assert!(held.write(vec![note(2, "hen", None)]).await.is_err());
}

/// A coordinator whose renewals can be made to fail as a backend would.
#[derive(Debug)]
struct Unrenewing {
    inner: Arc<MemoryCoordinator>,
    failing: AtomicBool,
    /// How long each renewal takes to be answered.
    slowly: Duration,
}

#[async_trait]
impl Coordinator for Unrenewing {
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
        if self.failing.load(Ordering::SeqCst) {
            return Err(CoordinatorError::Backend("connection reset".into()));
        }
        tokio::time::sleep(self.slowly).await;
        self.inner.renew(lease, ttl).await
    }
    async fn release(&self, lease: &Lease) -> Result<(), CoordinatorError> {
        self.inner.release(lease).await
    }
    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed, CoordinatorError> {
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
}

#[tokio::test]
async fn a_tenant_whose_hold_could_not_be_renewed_in_time_is_opened_afresh() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(300);
    let coordinator = Arc::new(Unrenewing {
        inner: cluster.coordinator.clone(),
        failing: AtomicBool::new(false),
        slowly: Duration::ZERO,
    });
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            lease,
            ..Settings::new(NodeId::generate("node"))
        },
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    tokio::time::sleep(lease).await;
    let same = node.tenant(&tenant("acme")).await.unwrap();
    assert!(Arc::ptr_eq(&acme, &same), "renewed meanwhile");

    coordinator.failing.store(true, Ordering::SeqCst);
    tokio::time::sleep(lease + lease / 2).await;
    let afresh = node.tenant(&tenant("acme")).await.unwrap();
    assert!(
        !Arc::ptr_eq(&acme, &afresh),
        "its hold ran out, so it was opened again"
    );
    assert_eq!(afresh.read(count), 1);
}

fn node_leasing(cluster: &Cluster, coordinator: Arc<dyn Coordinator>, lease: Duration) -> Node {
    Node::new(
        coordinator,
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            lease,
            ..Settings::new(NodeId::generate("node"))
        },
    )
}

#[tokio::test]
async fn a_tenant_slow_to_load_is_held_throughout_and_not_handed_out_once_lost() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(300);
    let first = node_leasing(&cluster, cluster.coordinator.clone(), lease);
    first.create_tenant(&tenant("acme")).await.unwrap();
    let acme = first.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    first.shutdown().await;

    *cluster.store.slow.lock().unwrap() = Some(lease / 2);
    let second = node_leasing(&cluster, cluster.coordinator.clone(), lease);
    let opened = second.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(opened.read(count), 1);
    let third = node_leasing(&cluster, cluster.coordinator.clone(), lease);
    assert!(
        matches!(
            third.tenant(&tenant("acme")).await,
            Err(NodeError::Elsewhere { .. })
        ),
        "renewed while it loaded, so still held"
    );
    assert!(Arc::ptr_eq(
        &opened,
        &second.tenant(&tenant("acme")).await.unwrap()
    ));
}

#[tokio::test]
async fn a_renewal_answered_slowly_still_keeps_the_hold() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(600);
    let coordinator = Arc::new(Unrenewing {
        inner: cluster.coordinator.clone(),
        failing: AtomicBool::new(false),
        slowly: lease / 3,
    });
    let node = node_leasing(&cluster, coordinator, lease);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    tokio::time::sleep(lease * 2).await;
    let same = node.tenant(&tenant("acme")).await.unwrap();
    assert!(
        Arc::ptr_eq(&acme, &same),
        "each renewal is timed from when it was asked for"
    );
}
