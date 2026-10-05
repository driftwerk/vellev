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
    NodeId, ObjectId, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, NodeError, Settings};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{CoordinatorError, ManualClock, MemoryCoordinator};
use vellev_storage::{Mutation, State, StorageError};

const LEASE: Duration = Duration::from_secs(30);

/// Object storage whose writes can be made to fail.
#[derive(Debug)]
struct Store {
    inner: ObjectBlobs,
    failing: AtomicBool,
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
        self.inner.get(key).await
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
            }),
        }
    }

    fn node_with(&self, flush_rows: usize) -> Node {
        let settings = Settings {
            lease: LEASE,
            flush_rows,
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
        matches!(
            &orphaned,
            NodeError::Storage(error) if matches!(
                error.root(),
                StorageError::Coordinator(CoordinatorError::UnknownPartition(_))
            )
        ),
        "{orphaned}"
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

/// A stand-in embedding server: a text's vector is its length and its
/// number of words.
async fn embedding_server(calls: Arc<AtomicUsize>) -> String {
    let embed = |Shared(calls): Shared<Arc<AtomicUsize>>, Json(body): Json<Value>| async move {
        calls.fetch_add(1, Ordering::SeqCst);
        let vectors: Vec<[usize; 2]> = body["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| text.as_str().unwrap())
            .map(|text| [text.len(), text.split_whitespace().count()])
            .collect();
        Json(json!(vectors))
    };
    let router = Router::new().route("/embed", post(embed)).with_state(calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

#[tokio::test]
async fn a_collection_that_embeds_its_objects_is_embedded_on_write_and_on_query() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let calls = Arc::new(AtomicUsize::new(0));
    let embedded = EmbeddingProvider::Http {
        url: embedding_server(calls.clone()).await,
        api: HttpApi::Tei,
    };
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
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
