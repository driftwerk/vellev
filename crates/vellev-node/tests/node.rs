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
    NodeId, ObjectId, Partition, Property, RequestId, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Freshness, Node, NodeError, Reader, Settings};
use vellev_storage::backups::Backups;
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Forgotten, KeyHash, Lease,
    LogPosition, ManualClock, MemoryCoordinator, PartitionKey, Point, RestorePoint, Restored,
    TenantKey, Versions,
};
use vellev_storage::{Batch, Mutation, Outcome, Refused, State, StorageError, Transfers, Writer};

const LEASE: Duration = Duration::from_secs(30);

/// What goes wrong with writes of one kind of file.
#[derive(Debug, Clone, Copy)]
enum Fault {
    Fails,
    /// The next write fails, and then they go right again.
    FailsOnce,
    NeverAnswers,
}

/// Object storage whose writes can be made to fail.
#[derive(Debug)]
struct Store {
    inner: ObjectBlobs,
    failing: AtomicBool,
    /// The kind of file, as its extension names it, whose writes go wrong,
    /// and how.
    faulty: std::sync::Mutex<Option<(&'static str, Fault)>>,
    unreadable: AtomicBool,
    /// Set for the next read to fail, and no other.
    failing_once: AtomicBool,
    /// How long each read takes, when reads are to be slow.
    slow: std::sync::Mutex<Option<Duration>>,
    /// The prefixes whose cached copies were dropped.
    forgotten: std::sync::Mutex<Vec<String>>,
}

impl Store {
    /// Have writes of `kind` of file go wrong as `fault` says, or with
    /// none go right again.
    fn fault(&self, faulty: Option<(&'static str, Fault)>) {
        *self.faulty.lock().unwrap() = faulty;
    }
}

#[async_trait]
impl Blobs for Store {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        let fault = {
            let mut faulty = self.faulty.lock().unwrap();
            let fault = faulty
                .filter(|(kind, _)| key.ends_with(&format!(".{kind}")))
                .map(|(_, fault)| fault);
            if let Some(Fault::FailsOnce) = fault {
                *faulty = None;
            }
            fault
        };
        if let Some(Fault::NeverAnswers) = fault {
            std::future::pending::<()>().await;
        }
        if self.failing.load(Ordering::SeqCst) || fault.is_some() {
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
        let once = self.failing_once.swap(false, Ordering::SeqCst);
        if self.unreadable.load(Ordering::SeqCst) || once {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        if self.unreadable.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: prefix.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
    async fn forget_local(&self, prefix: &str) -> vellev_storage::Result<()> {
        self.forgotten.lock().unwrap().push(prefix.to_string());
        Ok(())
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
                faulty: std::sync::Mutex::new(None),
                unreadable: AtomicBool::new(false),
                failing_once: AtomicBool::new(false),
                slow: std::sync::Mutex::new(None),
                forgotten: std::sync::Mutex::default(),
            }),
        }
    }

    fn node_with(&self, flush_rows: usize) -> Node {
        self.node_from(Settings {
            flush_rows,
            ..self.settings()
        })
    }

    fn settings(&self) -> Settings {
        Settings {
            lease: LEASE,
            restore_window: Duration::ZERO,
            grace: Duration::ZERO,
            purge_after: Duration::ZERO,
            ..Settings::new(NodeId::generate("node"))
        }
    }

    /// A node whose copies of tenants look for new commits only when a
    /// read asks them to.
    fn reading_node(&self) -> Node {
        self.node_from(Settings {
            follow_every: Duration::from_secs(3_600),
            ..self.settings()
        })
    }

    fn node_from(&self, settings: Settings) -> Node {
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

fn key(id: &str) -> PartitionKey {
    PartitionKey {
        tenant: tenant(id),
        partition: Partition::DEFAULT,
    }
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

/// A collection embedded by the model the node runs itself.
fn own_model() -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: notes(),
        properties: vec![
            Property::new("title", DataType::Text).searchable(),
            Property::new("n", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::Local,
            model: "BAAI/bge-base-en-v1.5".into(),
            dimensions: 768,
            query_prefix: Some("Represent this sentence for searching relevant passages: ".into()),
            document_prefix: None,
            properties: vec!["title".into()],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

/// Two tenants' writes on the model the node runs itself take turns at
/// it: a tenant's one object is embedded beside another tenant's bulk
/// write, not after it, and a search of the one is answered meanwhile.
/// The model is fetched and run, so this is run by hand:
///
///   VELLEV_MODEL_CACHE=... cargo test --release -p vellev-node --test node take_turns -- --ignored --nocapture
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn two_tenants_writes_take_turns_on_the_model_the_node_runs_itself() {
    let cache = std::env::var("VELLEV_MODEL_CACHE")
        .map(Into::into)
        .unwrap_or_else(|_| std::env::temp_dir().join("vellev-models"));
    let cluster = Cluster::new();
    let node = Node::new(
        cluster.coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings {
            cache,
            ..EmbedSettings::default()
        })),
        cluster.settings(),
    );
    let passage = "The quarterly review covered hiring, the migration off the old cluster, \
        an incident in the billing service, and plans for the next release.";
    for id in ["bulk", "small"] {
        node.create_tenant(&tenant(id)).await.unwrap();
        let opened = node.tenant(&tenant(id)).await.unwrap();
        opened
            .write(vec![own_model(), note(1, passage, None)])
            .await
            .unwrap();
    }
    // The bulk tenant writes sixty-four objects at once; half a second
    // in, the small tenant writes one and searches.
    let bulk = tokio::spawn({
        let bulk = node.tenant(&tenant("bulk")).await.unwrap();
        async move {
            let began = std::time::Instant::now();
            let notes: Vec<Mutation> = (2..66)
                .map(|n| note(n, &format!("{n}. {passage}"), None))
                .collect();
            (bulk.write(notes).await, began.elapsed())
        }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let small = node.tenant(&tenant("small")).await.unwrap();
    let began = std::time::Instant::now();
    let written = small
        .write(vec![note(2, "a small write meanwhile", None)])
        .await;
    let wrote_in = began.elapsed();
    let began = std::time::Instant::now();
    let reader = node
        .reader(&tenant("small"), Freshness::Eventual)
        .await
        .unwrap();
    let searched = reader
        .query_vector(&notes(), "what did the review cover?")
        .await;
    let searched_in = began.elapsed();
    let (bulk, bulk_took) = bulk.await.unwrap();
    println!(
        "beside a bulk write of 64 that took {bulk_took:.2?}: a 1-object write in {wrote_in:.2?}, a search in {searched_in:.2?}"
    );
    written.unwrap();
    assert!(searched.unwrap().is_some());
    bulk.unwrap();
    assert!(wrote_in < Duration::from_secs(5), "{wrote_in:?}");
    assert!(searched_in < Duration::from_secs(5), "{searched_in:?}");
    assert!(
        wrote_in + searched_in < bulk_took,
        "beside it, not after it"
    );
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
        only_if_version: None,
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
    // What still holds the tenant as it was writes to it no more.
    let orphaned = acme.write(vec![note(2, "hen", None)]).await.unwrap_err();
    assert!(
        matches!(&orphaned, NodeError::Storage(StorageError::Closed)),
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

    assert!(
        matches!(
            node.create_tenant(&tenant("acme")).await,
            Err(NodeError::Storage(
                vellev_storage::StorageError::Coordinator(CoordinatorError::Purging(_))
            ))
        ),
        "not made again until what was kept of it is purged"
    );
    node.purge_due().await.unwrap();
    assert!(cluster.store.list("acme/").await.unwrap().is_empty());
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
    // The node that let it go writes to it no more, whoever still holds
    // the tenant as it was open there.
    let let_go = held.write(vec![note(3, "owl", None)]).await.unwrap_err();
    assert!(
        matches!(&let_go, NodeError::Storage(StorageError::Closed)),
        "{let_go}"
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
    let vector = |n: u64| {
        acme.read(|state| {
            state
                .get(&notes(), id(n))
                .unwrap()
                .vector
                .map(<[f32]>::to_vec)
        })
    };
    assert_eq!(vector(1), Some(vec![9.0, 3.0]));
    assert_eq!(vector(2), Some(vec![1.0, 0.0]));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let searching = node
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    let query = searching.query_vector(&notes(), "red fox").await.unwrap();
    assert_eq!(query, Some(vec![13.0, 3.0]));
    let again = searching.query_vector(&notes(), "red fox").await.unwrap();
    assert_eq!(again, query);
    let blank = searching.query_vector(&notes(), "  ").await.unwrap();
    assert_eq!(blank, None);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        searching
            .query_vector(&CollectionName::new("nope").unwrap(), "fox")
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
    let searching = node
        .reader(&tenant("plain"), Freshness::Eventual)
        .await
        .unwrap();
    let supplied = searching.query_vector(&notes(), "fox").await.unwrap();
    assert_eq!(supplied, None);
    assert_eq!(
        plain.read(|state| state
            .get(&notes(), id(1))
            .unwrap()
            .vector
            .map(<[f32]>::to_vec)),
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
    let vector = |n: u64| {
        acme.read(|state| {
            state
                .get(&notes(), id(n))
                .unwrap()
                .vector
                .map(<[f32]>::to_vec)
        })
    };
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
    // A segment that could not be uploaded was never to be committed: the
    // writer carries on.
    cluster.store.failing.store(true, Ordering::SeqCst);
    assert!(acme.flush().await.is_err());
    cluster.store.failing.store(false, Ordering::SeqCst);
    assert!(Arc::ptr_eq(
        &acme,
        &node.tenant(&tenant("acme")).await.unwrap()
    ));

    cluster.store.fault(Some(("manifest", Fault::Fails)));
    assert!(acme.flush().await.is_err());
    cluster.store.fault(None);

    let reopened = node.tenant(&tenant("acme")).await.unwrap();
    assert!(!Arc::ptr_eq(&acme, &reopened));
    reopened.write(vec![note(2, "hen", None)]).await.unwrap();
    assert_eq!(reopened.read(count), 2);
}

#[tokio::test]
async fn a_request_repeated_in_one_commit_is_refused_alike_and_the_tenant_kept() {
    let cluster = Cluster::new();
    let node = cluster.node();
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();

    // Both are queued before the committer runs, so they share a commit.
    let batch = Batch {
        request: Some(RequestId::generate()),
        mutations: vec![Mutation::DropCollection(
            CollectionName::new("nope").unwrap(),
        )],
    };
    let refused = join_all([acme.write(batch.clone()), acme.write(batch)]).await;
    for refused in refused {
        let refused = refused.unwrap_err();
        let NodeError::Storage(error) = &refused else {
            panic!("{refused}");
        };
        assert!(!matches!(error, StorageError::Shared(_)), "{error:?}");
        assert_eq!(
            error.refusal().map(|refusal| refusal.kind),
            Some(Refused::NotFound),
            "{error:?}"
        );
    }
    assert!(Arc::ptr_eq(
        &acme,
        &node.tenant(&tenant("acme")).await.unwrap()
    ));
}

#[tokio::test(start_paused = true)]
async fn writes_while_a_restore_runs_are_turned_away_and_the_restore_completes() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    assert!(node
        .set_restore_window(&acme, Some(Duration::from_secs(60 * 60)))
        .await
        .unwrap());
    let opened = node.tenant(&acme).await.unwrap();
    let kept = opened
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    opened.flush().await.unwrap();
    opened.write(vec![note(2, "hen", None)]).await.unwrap();

    // Reading the state gone back to takes a while, and writes go on.
    *cluster.store.slow.lock().unwrap() = Some(Duration::from_millis(25));
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move {
            node.restore(&acme, Point::Position(kept.position), RequestId::generate())
                .await
        }
    });
    let mut turned_away = 0;
    let mut n = 3;
    while !restoring.is_finished() {
        let written = node.tenant(&acme).await.unwrap();
        match written.write(vec![note(n, "owl", None)]).await {
            Ok(_) => n += 1,
            Err(NodeError::Storage(StorageError::Restoring)) => turned_away += 1,
            Err(other) => panic!("{other}"),
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    *cluster.store.slow.lock().unwrap() = None;
    restoring.await.unwrap().unwrap();
    assert!(turned_away > 0, "some writes came while it ran");
    assert!(
        Arc::ptr_eq(&opened, &node.tenant(&acme).await.unwrap()),
        "the tenant was not opened again"
    );
    opened.write(vec![note(n, "owl", None)]).await.unwrap();
}

/// A tenant on `node` that may be restored for an hour back, with notes 1
/// and 2 written one after the other; returns the receipt of the first.
async fn restorable(node: &Node, acme: &TenantId) -> vellev_storage::Receipt {
    node.create_tenant(acme).await.unwrap();
    assert!(node
        .set_restore_window(acme, Some(Duration::from_secs(60 * 60)))
        .await
        .unwrap());
    let opened = node.tenant(acme).await.unwrap();
    let kept = opened
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    opened.write(vec![note(2, "hen", None)]).await.unwrap();
    kept
}

#[tokio::test]
async fn a_restore_asked_for_again_is_answered_as_it_was_on_any_node_and_undoes_nothing_since() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.node(), cluster.node());
    let acme = tenant("acme");
    let kept = restorable(&first, &acme).await;
    let (point, request) = (Point::Position(kept.position), RequestId::generate());

    let restored = first.restore(&acme, point, request).await.unwrap();
    assert_eq!(restored.restore.from, kept.position);
    let opened = first.tenant(&acme).await.unwrap();
    assert_eq!(opened.read(count), 1);
    opened.write(vec![note(3, "owl", None)]).await.unwrap();

    assert_eq!(
        first.restore(&acme, point, request).await.unwrap(),
        restored
    );
    assert_eq!(opened.read(count), 2, "the write since stays");

    first.shutdown().await;
    assert_eq!(
        second.restore(&acme, point, request).await.unwrap(),
        restored
    );
    assert_eq!(second.tenant(&acme).await.unwrap().read(count), 2);

    let again = second
        .restore(&acme, point, RequestId::generate())
        .await
        .unwrap();
    assert!(again.position > restored.position);
    assert_eq!(second.tenant(&acme).await.unwrap().read(count), 1);
}

#[tokio::test(start_paused = true)]
async fn a_restore_asked_for_while_another_runs_is_refused_and_the_same_one_waits_for_it() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    node.tenant(&acme).await.unwrap().flush().await.unwrap();
    let point = Point::Position(kept.position);
    let request = RequestId::generate();

    // Reading the state gone back to takes a while.
    *cluster.store.slow.lock().unwrap() = Some(Duration::from_millis(25));
    let restoring = |request| {
        let (node, acme) = (node.clone(), acme.clone());
        tokio::spawn(async move { node.restore(&acme, point, request).await })
    };
    let running = restoring(request);
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(matches!(
        node.restore(&acme, point, RequestId::generate()).await,
        Err(NodeError::Restoring(_))
    ));
    let waiting = restoring(request);
    let (ran, waited) = (running.await.unwrap(), waiting.await.unwrap());
    *cluster.store.slow.lock().unwrap() = None;
    let ran = ran.unwrap();
    assert_eq!(waited.unwrap(), ran);
    let (_, latest) = node
        .restore_points(&acme)
        .await
        .unwrap()
        .unwrap()
        .1
        .unwrap();
    assert_eq!(latest.position, ran.position, "it was made once");
}

#[tokio::test(start_paused = true)]
async fn a_restore_asked_for_again_after_a_try_that_failed_is_made_on_the_tenant_opened_afresh() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let opened = node.tenant(&acme).await.unwrap();
    let point = Point::Position(kept.position);
    let request = RequestId::generate();

    // The first try reads slowly and then fails as it commits, closing the
    // writer, while the second waits for it.
    *cluster.store.slow.lock().unwrap() = Some(Duration::from_millis(25));
    cluster.store.fault(Some(("manifest", Fault::FailsOnce)));
    let restoring = || {
        let (node, acme) = (node.clone(), acme.clone());
        tokio::spawn(async move { node.restore(&acme, point, request).await })
    };
    let first = restoring();
    tokio::time::sleep(Duration::from_millis(5)).await;
    let second = restoring();
    assert!(first.await.unwrap().is_err());
    let restored = second.await.unwrap().unwrap();
    *cluster.store.slow.lock().unwrap() = None;
    let reopened = node.tenant(&acme).await.unwrap();
    assert!(!Arc::ptr_eq(&opened, &reopened));
    assert_eq!(reopened.read(State::position), restored.position);
    assert_eq!(reopened.read(count), 1);
}

#[tokio::test]
async fn a_restore_whose_answer_was_lost_is_found_recorded_and_its_position_never_reused() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = node_leasing(&cluster, coordinator.clone(), LEASE);
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let point = Point::Position(kept.position);
    let request = RequestId::generate();

    // Every try of the commit lands, and every answer is lost: it is
    // looked for before its caller is told anything.
    coordinator.lose_answers.store(3, Ordering::SeqCst);
    let restored = node.restore(&acme, point, request).await.unwrap();
    let landed = cluster
        .coordinator
        .committed(&key("acme"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(restored.position, landed.log_position, "found as recorded");
    assert_eq!(node.restore(&acme, point, request).await.unwrap(), restored);
    let (_, latest) = node
        .restore_points(&acme)
        .await
        .unwrap()
        .unwrap()
        .1
        .unwrap();
    assert_eq!(latest.position, restored.position, "it was made once");
    let opened = node.tenant(&acme).await.unwrap();
    assert_eq!(opened.read(count), 1);
    let written = opened.write(vec![note(3, "owl", None)]).await.unwrap();
    assert_eq!(written.position, restored.position.next().unwrap());
}

#[tokio::test]
async fn a_request_asking_for_another_restore_than_it_did_is_refused() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let request = RequestId::generate();
    let restored = node
        .restore(&acme, Point::Position(kept.position), request)
        .await
        .unwrap();

    let elsewhere = Point::Position(restored.position);
    assert!(matches!(
        node.restore(&acme, elsewhere, request).await,
        Err(NodeError::Reused { .. })
    ));
    assert_eq!(
        node.restore(&acme, Point::Position(kept.position), request)
            .await
            .unwrap(),
        restored
    );
}

#[tokio::test]
async fn a_restore_that_fails_as_it_commits_has_the_tenant_opened_afresh() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    assert!(node
        .set_restore_window(&acme, Some(Duration::from_secs(60 * 60)))
        .await
        .unwrap());
    let opened = node.tenant(&acme).await.unwrap();
    let kept = opened
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    opened.write(vec![note(2, "hen", None)]).await.unwrap();

    cluster.store.fault(Some(("manifest", Fault::Fails)));
    assert!(node
        .restore(&acme, Point::Position(kept.position), RequestId::generate())
        .await
        .is_err());
    cluster.store.fault(None);

    let reopened = node.tenant(&acme).await.unwrap();
    assert!(!Arc::ptr_eq(&opened, &reopened));
    reopened.write(vec![note(3, "owl", None)]).await.unwrap();
    assert_eq!(reopened.read(count), 3);
}

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

    let started = tokio::time::Instant::now();
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

/// A coordinator whose renewals can be made to fail as a backend would,
/// and whose commits can be made never to be answered, as over a
/// connection that stopped answering without closing.
#[derive(Debug)]
struct Unrenewing {
    inner: Arc<MemoryCoordinator>,
    failing: AtomicBool,
    /// How long each renewal takes to be answered.
    slowly: Duration,
    unanswered_commits: AtomicBool,
    /// Commits land and then are never answered.
    hang_after_commit: AtomicBool,
    /// Told when a commit that is never answered has landed.
    landed: tokio::sync::Notify,
    /// Forgetting what is kept of a removed tenant fails.
    failing_forgets: AtomicBool,
    /// How many times forgetting it was asked for.
    forgets: AtomicUsize,
    /// How many commits land with their answers lost.
    lose_answers: AtomicUsize,
    /// How many renewals were asked for.
    renewals: AtomicUsize,
    /// Commits that make a restore are refused, as by a lease since taken.
    refused_restores: AtomicBool,
}

impl Unrenewing {
    fn new(inner: Arc<MemoryCoordinator>, slowly: Duration) -> Self {
        Self {
            inner,
            failing: AtomicBool::new(false),
            slowly,
            unanswered_commits: AtomicBool::new(false),
            hang_after_commit: AtomicBool::new(false),
            landed: tokio::sync::Notify::new(),
            failing_forgets: AtomicBool::new(false),
            forgets: AtomicUsize::new(0),
            lose_answers: AtomicUsize::new(0),
            renewals: AtomicUsize::new(0),
            refused_restores: AtomicBool::new(false),
        }
    }
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
        self.renewals.fetch_add(1, Ordering::SeqCst);
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
        if self.unanswered_commits.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        if commit.restore.is_some() && self.refused_restores.load(Ordering::SeqCst) {
            return Err(CoordinatorError::Fenced {
                key: lease.key.clone(),
                held: lease.generation,
                current: lease.generation,
            });
        }
        let committed = self.inner.commit(lease, commit).await?;
        if self.hang_after_commit.load(Ordering::SeqCst) {
            self.landed.notify_one();
            std::future::pending::<()>().await;
        }
        let lost = self
            .lose_answers
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
        match lost {
            true => Err(CoordinatorError::Backend("connection reset".into())),
            false => Ok(committed),
        }
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
        self.forgets.fetch_add(1, Ordering::SeqCst);
        if self.failing_forgets.load(Ordering::SeqCst) {
            return Err(CoordinatorError::Backend("connection reset".into()));
        }
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
        // As over a network: whoever asks waits for the answer.
        tokio::task::yield_now().await;
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

#[tokio::test(start_paused = true)]
async fn a_tenant_whose_hold_could_not_be_renewed_in_time_is_opened_afresh() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(300);
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
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

#[tokio::test(start_paused = true)]
async fn a_renewal_answered_slowly_still_keeps_the_hold() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(600);
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), lease / 3));
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

#[tokio::test(start_paused = true)]
async fn a_commit_left_unanswered_closes_the_writer_within_a_lease_and_another_node_takes_over() {
    let cluster = Cluster::new();
    let lease = Duration::from_secs(2);
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let first = node_leasing(&cluster, coordinator.clone(), lease);
    let second = node_leasing(&cluster, cluster.coordinator.clone(), lease);
    first.create_tenant(&tenant("acme")).await.unwrap();
    let held = first.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    coordinator.unanswered_commits.store(true, Ordering::SeqCst);
    let started = tokio::time::Instant::now();
    let failed = tokio::time::timeout(lease * 2, held.write(vec![note(2, "hen", None)]))
        .await
        .expect("the commit is given up")
        .unwrap_err();
    // Three tries of a quarter of the lease each, and the waits between.
    let tries = lease / 4 * 3 + Duration::from_millis(50 + 100);
    assert_eq!(started.elapsed(), tries);
    assert!(
        matches!(&failed, NodeError::Storage(error) if matches!(error.root(), StorageError::Coordinator(error) if error.is_retryable())),
        "{failed}"
    );
    assert!(!Arc::ptr_eq(
        &held,
        &first.tenant(&tenant("acme")).await.unwrap()
    ));

    cluster.clock.advance(lease * 2);
    let taken = second.tenant(&tenant("acme")).await.unwrap();
    taken.write(vec![note(3, "owl", None)]).await.unwrap();
    assert_eq!(taken.read(count), 2);
}

#[tokio::test(start_paused = true)]
async fn a_writer_closed_by_a_flush_given_up_part_way_is_read_from_no_more_nor_renewed() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(600);
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = node_leasing(&cluster, coordinator.clone(), lease);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let held = node.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    // The checkpoint's upload never ends, and its caller stops waiting.
    cluster.store.fault(Some(("manifest", Fault::NeverAnswers)));
    assert!(tokio::time::timeout(lease / 6, held.flush()).await.is_err());
    cluster.store.fault(None);
    let strong = node
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(strong.follows(), "a closed writer answers no strong read");

    tokio::time::sleep(lease).await;
    let renewed = coordinator.renewals.load(Ordering::SeqCst);
    tokio::time::sleep(lease).await;
    assert_eq!(
        coordinator.renewals.load(Ordering::SeqCst),
        renewed,
        "the hold is no longer renewed"
    );
    let reopened = node.tenant(&tenant("acme")).await.unwrap();
    assert!(!Arc::ptr_eq(&held, &reopened));
    reopened.write(vec![note(2, "hen", None)]).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_writer_publishing_past_every_bound_on_its_steps_is_renewed_no_more() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(600);
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let transfers = Transfers {
        part_bytes: 1024 * 1024,
        request_within: Duration::from_millis(100),
    };
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            lease,
            transfers,
            ..Settings::new(NodeId::generate("node"))
        },
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    let held = node.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    // The log's upload is never answered, and nothing gives it up: the
    // writer stays open, publishing for ever.
    cluster.store.fault(Some(("log", Fault::NeverAnswers)));
    let stuck = tokio::spawn({
        let held = held.clone();
        async move { held.write(vec![note(2, "hen", None)]).await }
    });
    // Well past every bound on what it began.
    tokio::time::sleep(Duration::from_secs(60 * 60)).await;
    let renewed = coordinator.renewals.load(Ordering::SeqCst);
    tokio::time::sleep(lease * 2).await;
    assert_eq!(
        coordinator.renewals.load(Ordering::SeqCst),
        renewed,
        "the hold is no longer renewed"
    );
    // The write is given up where it was, whether it was made unknown.
    let given_up = soon(stuck).await.unwrap().unwrap_err();
    assert!(
        matches!(&given_up, NodeError::Storage(StorageError::Shared(shared)) if matches!(**shared, StorageError::Closed)),
        "{given_up}"
    );
    cluster.store.fault(None);
    let reopened = node.tenant(&tenant("acme")).await.unwrap();
    assert!(!Arc::ptr_eq(&held, &reopened));
}

/// What `work` came to, failing the test rather than letting it hang if
/// that took an hour, which on a paused clock means it would never end.
async fn soon<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(60 * 60), work)
        .await
        .expect("done within the hour")
}

#[tokio::test(start_paused = true)]
async fn a_restore_under_way_when_the_hold_is_lost_is_stopped_and_its_caller_told() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let data = Arc::downgrade(node.tenant(&acme).await.unwrap().partition());

    // The upload of the segment gone back to is never answered.
    cluster.store.fault(Some(("segment", Fault::NeverAnswers)));
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move {
            node.restore(&acme, Point::Position(kept.position), RequestId::generate())
                .await
        }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!restoring.is_finished());
    node.shutdown().await;
    let stopped = soon(restoring).await.unwrap();
    assert!(matches!(stopped, Err(NodeError::Lost(_))), "{stopped:?}");
    assert!(data.upgrade().is_none(), "nothing holds the tenant's data");
}

#[tokio::test(start_paused = true)]
async fn a_tenant_given_up_again_and_again_leaves_none_of_its_data_in_memory() {
    let cluster = Cluster::new();
    let lease = Duration::from_millis(600);
    let node = node_leasing(&cluster, cluster.coordinator.clone(), lease);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let mut given_up = Vec::new();
    for round in 0..3 {
        let held = node.tenant(&tenant("acme")).await.unwrap();
        if round == 0 {
            held.write(vec![schema(EmbeddingProvider::None)])
                .await
                .unwrap();
        }
        // A write whose upload is never answered and never given up holds
        // the writer for good.
        cluster.store.fault(Some(("log", Fault::NeverAnswers)));
        let stuck = tokio::spawn({
            let held = held.clone();
            async move { held.write(vec![note(round, "owl", None)]).await }
        });
        tokio::time::sleep(Duration::from_secs(60 * 60)).await;
        assert!(soon(stuck).await.unwrap().is_err());
        cluster.store.fault(None);
        given_up.push(Arc::downgrade(held.partition()));
    }
    node.tenant(&tenant("acme")).await.unwrap();
    let live = given_up
        .iter()
        .filter(|data| data.upgrade().is_some())
        .count();
    assert_eq!(live, 0, "of {} given up", given_up.len());
}

#[tokio::test]
async fn a_strong_read_on_a_holder_whose_lease_another_took_reads_from_a_copy_before_any_write() {
    let cluster = Cluster::new();
    let (holding, other) = (cluster.node(), cluster.node());
    holding.create_tenant(&tenant("acme")).await.unwrap();
    let acme = holding.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    // Taken over, with nothing written since: only the generation says
    // the holder's lease is no longer its own.
    cluster.clock.advance(LEASE * 2);
    other.tenant(&tenant("acme")).await.unwrap();
    let strong = holding
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(strong.follows());
}

#[tokio::test]
async fn a_tenant_dropped_while_its_history_cannot_be_forgotten_is_dropped_and_its_purge_forgets_it(
) {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        cluster.settings(),
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    node.tenant(&tenant("acme"))
        .await
        .unwrap()
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    coordinator.failing_forgets.store(true, Ordering::SeqCst);
    assert!(node.drop_tenant(&tenant("acme")).await.unwrap());
    assert!(coordinator.forgets.load(Ordering::SeqCst) > 0);
    coordinator.failing_forgets.store(false, Ordering::SeqCst);

    let asked = coordinator.forgets.load(Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert!(coordinator.forgets.load(Ordering::SeqCst) > asked);
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_strong_read_on_a_holder_behind_its_own_commit_reads_it_from_a_copy() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = node_leasing(&cluster, coordinator.clone(), LEASE);
    node.create_tenant(&tenant("acme")).await.unwrap();
    let held = node.tenant(&tenant("acme")).await.unwrap();
    held.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    // The commit lands, and the writer waits on for its answer.
    coordinator.hang_after_commit.store(true, Ordering::SeqCst);
    let writing = tokio::spawn({
        let held = held.clone();
        async move { held.write(vec![note(2, "hen", None)]).await }
    });
    coordinator.landed.notified().await;
    let strong = node
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(strong.follows());
    assert_eq!(counted(&strong), 2);
    writing.abort();
}

fn after(receipt: &vellev_storage::Receipt) -> Freshness {
    Freshness::After(receipt.incarnation, receipt.position)
}

fn counted(reader: &Reader) -> usize {
    reader.read(count)
}

#[tokio::test]
async fn a_reader_says_whether_anything_costly_was_done_to_give_it() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    let calls = Arc::new(AtomicUsize::new(0));
    let url = embedding_server(calls.clone()).await;
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    let first = acme
        .write(vec![
            schema(served_by(&url, "lengths")),
            note(1, "fox", None),
        ])
        .await
        .unwrap();
    let reader = |node: &Node, fresh: Freshness| {
        let node = node.clone();
        async move { node.reader(&tenant("acme"), fresh).await.unwrap() }
    };

    // On the node that holds the tenant, only a strong read asks the
    // coordinator anything.
    assert!(!reader(&writing, Freshness::Eventual).await.worked());
    assert!(!reader(&writing, after(&first)).await.worked());
    assert!(reader(&writing, Freshness::Strong).await.worked());

    // On another, the first read opens a copy; one after it finds the
    // copy there, and asks nothing while the copy is as fresh as asked.
    // Of two that come while there is none, the opening is one's, and
    // the other only waits for it.
    *cluster.store.slow.lock().unwrap() = Some(Duration::from_millis(50));
    let (opening, waiting) = tokio::join!(
        reader(&reading, Freshness::Eventual),
        reader(&reading, Freshness::Eventual)
    );
    *cluster.store.slow.lock().unwrap() = None;
    assert!(opening.follows() && waiting.follows());
    assert!(opening.worked() != waiting.worked(), "one of the two");
    assert!(!reader(&reading, Freshness::Eventual).await.worked());
    assert!(!reader(&reading, after(&first)).await.worked());
    assert!(reader(&reading, Freshness::Strong).await.worked());
    let second = acme.write(vec![note(2, "hen", None)]).await.unwrap();
    let caught_up = reader(&reading, after(&second)).await;
    assert!(
        caught_up.worked(),
        "the coordinator was asked how far to go"
    );
    assert!(!reader(&reading, after(&second)).await.worked());

    // A text the model is asked for is work; the same text again, whose
    // vector is kept, is not, nor is a blank one, which no model is asked.
    let embedded = calls.load(Ordering::SeqCst);
    let asking = reader(&writing, Freshness::Eventual).await;
    assert_eq!(asking.query_vector(&notes(), "  ").await.unwrap(), None);
    assert!(!asking.worked());
    asking.query_vector(&notes(), "red fox").await.unwrap();
    assert!(asking.worked());
    assert_eq!(calls.load(Ordering::SeqCst), embedded + 1);
    let again = reader(&writing, Freshness::Eventual).await;
    again.query_vector(&notes(), "red fox").await.unwrap();
    assert!(!again.worked());
    assert_eq!(calls.load(Ordering::SeqCst), embedded + 1);
    // One too long to be kept is asked for every time.
    let long = "x".repeat(2_000);
    for _ in 0..2 {
        let asking = reader(&writing, Freshness::Eventual).await;
        asking.query_vector(&notes(), &long).await.unwrap();
        assert!(asking.worked());
    }
}

#[tokio::test]
async fn a_copy_one_read_failed_to_open_is_opened_and_kept_by_the_read_waiting_on_it() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let read = |node: &Node| {
        let node = node.clone();
        async move { node.reader(&tenant("acme"), Freshness::Eventual).await }
    };

    // Two reads come for a tenant not yet open, and the store fails the
    // first thing it is asked for: one read's opening fails, and the
    // other, which waited on it, opens the copy itself.
    *cluster.store.slow.lock().unwrap() = Some(Duration::from_millis(50));
    cluster.store.failing_once.store(true, Ordering::SeqCst);
    let (one, other) = tokio::join!(read(&reading), read(&reading));
    *cluster.store.slow.lock().unwrap() = None;
    let (failed, opened) = match (one, other) {
        (Err(failed), Ok(opened)) | (Ok(opened), Err(failed)) => (failed, opened),
        (one, other) => panic!("one of the two fails: {one:?}, {other:?}"),
    };
    assert!(matches!(failed, NodeError::Storage(_)), "{failed}");
    assert!(opened.worked(), "the opening is its own, taken over or not");
    assert_eq!(counted(&opened), 1);

    // The copy it opened is the one the node keeps: the next read finds
    // it there, and opens nothing.
    assert_eq!(reading.copies(), 1);
    let next = read(&reading).await.unwrap();
    assert!(!next.worked());
    assert_eq!(reading.copies(), 1);

    // An opening that fails with nobody waiting on it is let go of, and
    // the read after it opens afresh.
    let cold = cluster.reading_node();
    cluster.store.failing_once.store(true, Ordering::SeqCst);
    assert!(read(&cold).await.is_err());
    assert_eq!(cold.copies(), 0);
    assert!(read(&cold).await.unwrap().worked());
    assert_eq!(cold.copies(), 1);
}

#[tokio::test(start_paused = true)]
async fn tenants_due_a_backup_are_looked_for_unasked_unless_the_node_is_set_not_to() {
    let cluster = Cluster::new();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let backing_up = |scheduled: bool| {
        cluster.node_from(Settings {
            backups: Some(store.clone()),
            backups_scheduled: scheduled,
            ..cluster.settings()
        })
    };
    let (asked_only, unasked) = (backing_up(false), backing_up(true));
    asked_only.create_tenant(&tenant("acme")).await.unwrap();
    let backups = || async { asked_only.backups(&tenant("acme")).await.unwrap().len() };

    // A node set not to looks at no moment of its own, however long it
    // runs, and backs up what is due when it is asked to.
    asked_only.backing_up();
    tokio::time::sleep(Duration::from_secs(60 * 60)).await;
    assert_eq!(backups().await, 0);
    asked_only.back_up_due().await.unwrap();
    assert_eq!(backups().await, 1);

    // As a node is set up unless told otherwise, it looks within ten
    // minutes of starting, and finds the tenant due again a day on.
    cluster.clock.advance(Duration::from_secs(25 * 60 * 60));
    unasked.backing_up();
    tokio::time::sleep(Duration::from_secs(11 * 60)).await;
    assert_eq!(backups().await, 2);
    assert!(Settings::new(NodeId::generate("node")).backups_scheduled);
}

#[tokio::test]
async fn another_node_answers_reads_as_fresh_as_asked_and_never_takes_the_tenant() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    let first = reading
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    assert!(first.follows());
    assert_eq!(
        counted(&first),
        1,
        "a copy opened now has what is committed"
    );

    let receipt = acme.write(vec![note(2, "hen", None)]).await.unwrap();
    let eventual = reading
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    assert_eq!(counted(&eventual), 1, "eventual answers with what it has");
    let session = reading
        .reader(&tenant("acme"), after(&receipt))
        .await
        .unwrap();
    assert_eq!(counted(&session), 2, "session catches up to the write");

    acme.write(vec![note(3, "owl", None)]).await.unwrap();
    let older = reading
        .reader(&tenant("acme"), after(&receipt))
        .await
        .unwrap();
    assert_eq!(
        counted(&older),
        2,
        "a point already reached asks nothing more"
    );
    let strong = reading
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert_eq!(counted(&strong), 3, "strong has everything committed");

    let own = writing
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(!own.follows(), "the holder reads its own");
    acme.write(vec![note(4, "elk", None)]).await.unwrap();
    assert!(matches!(
        reading.tenant(&tenant("acme")).await,
        Err(NodeError::Elsewhere { .. })
    ));
    assert!(matches!(
        reading.reader(&tenant("gone"), Freshness::Strong).await,
        Err(NodeError::UnknownTenant(_))
    ));
}

#[tokio::test]
async fn a_tenant_dropped_and_created_again_never_holds_a_read_up() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    let mut old = acme.write(vec![note(1, "fox", None)]).await.unwrap();
    for n in 2..6 {
        old = acme.write(vec![note(n, "fox", None)]).await.unwrap();
    }
    let followed = reading.reader(&tenant("acme"), after(&old)).await.unwrap();
    assert_eq!(counted(&followed), 5);

    writing.drop_tenant(&tenant("acme")).await.unwrap();
    writing.purge_due().await.unwrap();
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    let new = acme
        .write(vec![schema(EmbeddingProvider::None), note(9, "hen", None)])
        .await
        .unwrap();
    assert!(new.incarnation > old.incarnation);
    assert!(new.position < old.position);

    let fresh = cluster.reading_node();
    let answered = fresh.reader(&tenant("acme"), after(&old)).await.unwrap();
    assert_eq!(
        counted(&answered),
        1,
        "an older incarnation's point is behind the new one"
    );
    let moved = reading.reader(&tenant("acme"), after(&new)).await.unwrap();
    assert_eq!(counted(&moved), 1, "a copy of the old tenant is let go");
    assert!(moved.read(|state| state.get(&notes(), id(9)).is_some()));
}

#[tokio::test]
async fn a_node_whose_hold_ran_out_answers_a_session_read_from_the_new_writer() {
    let cluster = Cluster::new();
    let (stale, taking) = (cluster.node(), cluster.node());
    stale.create_tenant(&tenant("acme")).await.unwrap();
    let acme = stale.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    cluster.clock.advance(LEASE);
    let taken = taking.tenant(&tenant("acme")).await.unwrap();
    let receipt = taken.write(vec![note(2, "hen", None)]).await.unwrap();

    let read = stale
        .reader(&tenant("acme"), after(&receipt))
        .await
        .unwrap();
    assert!(read.follows(), "a writer behind the point is not trusted");
    assert_eq!(counted(&read), 2);
}

#[tokio::test]
async fn a_copy_left_unread_catches_up_through_merges_and_cleanup() {
    let cluster = Cluster::new();
    let writing = cluster.node_with(5);
    let reading = cluster.reading_node();
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(0, "ant", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    let copy = reading
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    assert_eq!(counted(&copy), 1);

    for n in 1..=40 {
        acme.write(vec![note(n, "fox", None)]).await.unwrap();
        if n % 5 == 0 {
            acme.flush().await.unwrap();
        }
    }
    acme.write(vec![Mutation::Delete {
        collection: notes(),
        id: id(0),
        only_if_version: None,
    }])
    .await
    .unwrap();
    acme.compact(Duration::ZERO).await.unwrap();

    let strong = reading
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert_eq!(counted(&strong), 40);
    assert!(strong.read(|state| state.get(&notes(), id(0)).is_none()));
}

#[tokio::test]
async fn a_point_past_anything_committed_is_refused_rather_than_waited_for() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    let receipt = acme
        .write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    let beyond = Freshness::After(receipt.incarnation, LogPosition(receipt.position.0 + 5));
    assert!(matches!(
        reading.reader(&tenant("acme"), beyond).await,
        Err(NodeError::Ahead(_))
    ));
    assert!(matches!(
        writing.reader(&tenant("acme"), beyond).await,
        Err(NodeError::Ahead(_))
    ));
}

#[tokio::test]
async fn a_node_lets_its_copy_go_once_it_writes_the_tenant_itself() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    reading
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    assert_eq!(reading.copies(), 1);
    cluster.clock.advance(LEASE);
    let taken = reading.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(reading.copies(), 0);
    assert_eq!(taken.read(count), 1);
    let strong = reading.reader(&tenant("acme"), Freshness::Strong).await;
    assert!(!strong.unwrap().follows());
}

#[tokio::test]
async fn copies_follow_on_their_own_and_are_let_go_when_idle_or_too_many() {
    let cluster = Cluster::new();
    let writing = cluster.node();
    let reading = cluster.node_from(Settings {
        follow_every: Duration::from_millis(10),
        follow_idle: Duration::from_millis(300),
        followers: 2,
        ..cluster.settings()
    });
    for name in ["a", "b", "c"] {
        writing.create_tenant(&tenant(name)).await.unwrap();
        let opened = writing.tenant(&tenant(name)).await.unwrap();
        opened
            .write(vec![schema(EmbeddingProvider::None)])
            .await
            .unwrap();
    }
    let a = reading
        .reader(&tenant("a"), Freshness::Eventual)
        .await
        .unwrap();
    assert_eq!(counted(&a), 0);
    writing
        .tenant(&tenant("a"))
        .await
        .unwrap()
        .write(vec![note(1, "fox", None)])
        .await
        .unwrap();
    let mut waited = Duration::ZERO;
    while counted(&a) == 0 && waited < Duration::from_secs(2) {
        tokio::time::sleep(Duration::from_millis(10)).await;
        waited += Duration::from_millis(10);
    }
    assert_eq!(
        counted(&a),
        1,
        "an eventual read sees a commit without asking"
    );

    for name in ["b", "c"] {
        reading
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    assert_eq!(reading.copies(), 2, "the least recently used goes first");
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert_eq!(reading.copies(), 0, "idle copies are let go");
}

#[tokio::test]
async fn a_strong_read_on_a_holder_of_a_tenant_since_recreated_reads_the_new_one() {
    let cluster = Cluster::new();
    let (holding, other) = (cluster.node(), cluster.node());
    holding.create_tenant(&tenant("acme")).await.unwrap();
    let acme = holding.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    other.drop_tenant(&tenant("acme")).await.unwrap();
    other.purge_due().await.unwrap();
    other.create_tenant(&tenant("acme")).await.unwrap();
    let eventual = holding
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
    assert!(!eventual.follows(), "eventual takes what it has");
    let strong = holding
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(strong.follows());
    assert!(strong.read(|state| state.collection(&notes()).is_none()));
}

#[tokio::test]
async fn a_strong_read_on_a_holder_whose_lease_was_taken_over_reads_the_new_writes() {
    let cluster = Cluster::new();
    let (holding, other) = (cluster.node(), cluster.node());
    holding.create_tenant(&tenant("acme")).await.unwrap();
    let acme = holding.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();

    // The database's clock runs ahead of the holder's: the lease is over
    // there while the holder, by its own clock, still has it.
    cluster.clock.advance(LEASE * 2);
    let taken = other.tenant(&tenant("acme")).await.unwrap();
    taken.write(vec![note(2, "hen", None)]).await.unwrap();

    let strong = holding
        .reader(&tenant("acme"), Freshness::Strong)
        .await
        .unwrap();
    assert!(strong.follows());
    assert_eq!(counted(&strong), 2);
}

#[tokio::test]
async fn a_node_registers_while_it_runs_and_will_not_serve_a_format_it_cannot_read() {
    let cluster = Cluster::new();
    let node = cluster.node();
    let ours = node.versions();
    assert_eq!(node.cluster().await.unwrap(), ours);
    node.ready().await.unwrap();

    node.announcing();
    let mut waited = Duration::ZERO;
    while cluster.coordinator.oldest().await.unwrap().is_none() && waited < Duration::from_secs(2) {
        tokio::time::sleep(Duration::from_millis(10)).await;
        waited += Duration::from_millis(10);
    }
    assert_eq!(cluster.coordinator.oldest().await.unwrap(), Some(ours));
    let older = Versions {
        protocol: ours.protocol - 1,
        ..ours
    };
    let other = NodeId::generate("older");
    cluster.coordinator.announce(&other, older).await.unwrap();
    assert_eq!(node.cluster().await.unwrap().protocol, older.protocol);
    cluster.coordinator.depart(&other).await.unwrap();

    node.shutdown().await;
    assert_eq!(
        cluster.coordinator.oldest().await.unwrap(),
        None,
        "a node that stops takes itself off"
    );

    let ahead = Cluster::new();
    let newest = ahead
        .coordinator
        .raise_format(ours.format + 1)
        .await
        .unwrap();
    assert_eq!(newest, ours.format + 1);
    assert!(matches!(
        ahead.node().ready().await,
        Err(NodeError::Outdated { .. })
    ));
}

#[tokio::test]
async fn a_node_is_not_ready_until_object_storage_answers_and_does_not_ask_it_every_time() {
    let cluster = Cluster::new();
    let node = cluster.node();
    cluster.store.unreadable.store(true, Ordering::SeqCst);
    assert!(matches!(node.ready().await, Err(NodeError::Storage(_))));
    cluster.store.unreadable.store(false, Ordering::SeqCst);
    node.ready().await.unwrap();
    cluster.store.unreadable.store(true, Ordering::SeqCst);
    node.ready().await.unwrap();
}

#[tokio::test]
async fn an_idle_tenant_is_let_go_for_any_node_to_take_and_one_in_use_is_kept() {
    let cluster = Cluster::new();
    let first = cluster.node_from(Settings {
        tenant_idle: Duration::from_millis(100),
        ..cluster.settings()
    });
    first.sweeping();
    let second = cluster.node();
    for name in ["idle", "busy"] {
        first.create_tenant(&tenant(name)).await.unwrap();
        let opened = first.tenant(&tenant(name)).await.unwrap();
        opened
            .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
            .await
            .unwrap();
    }
    let in_use = first.tenant(&tenant("busy")).await.unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let taken = loop {
        match second.tenant(&tenant("idle")).await {
            Ok(taken) => break taken,
            Err(NodeError::Elsewhere { .. }) if std::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(error) => panic!("the idle tenant was not let go: {error}"),
        }
    };
    assert_eq!(taken.read(count), 1);
    assert_eq!(
        taken.read(State::unflushed),
        0,
        "flushed before it was let go"
    );
    assert!(
        matches!(
            second.tenant(&tenant("busy")).await,
            Err(NodeError::Elsewhere { .. })
        ),
        "a tenant someone is using is kept"
    );
    in_use.write(vec![note(2, "hen", None)]).await.unwrap();
}

#[tokio::test]
async fn a_tenant_asking_too_much_is_turned_away_and_others_are_not() {
    let cluster = Cluster::new();
    let node = cluster.node_from(Settings {
        queries_per_second: Some(5.0),
        writes_per_second: Some(10.0),
        ..cluster.settings()
    });
    for name in ["busy", "quiet"] {
        node.create_tenant(&tenant(name)).await.unwrap();
        let opened = node.tenant(&tenant(name)).await.unwrap();
        opened
            .write(vec![schema(EmbeddingProvider::None)])
            .await
            .unwrap();
    }
    for _ in 0..5 {
        node.reader(&tenant("busy"), Freshness::Eventual)
            .await
            .unwrap();
    }
    let limited = node
        .reader(&tenant("busy"), Freshness::Eventual)
        .await
        .unwrap_err();
    assert!(
        matches!(&limited, NodeError::Limited { wait, .. } if *wait <= Duration::from_millis(200)),
        "{limited}"
    );
    node.reader(&tenant("quiet"), Freshness::Eventual)
        .await
        .unwrap();

    let busy = node.tenant(&tenant("busy")).await.unwrap();
    busy.write((1..10).map(|n| note(n, "fox", None)).collect::<Vec<_>>())
        .await
        .unwrap();
    assert!(matches!(
        busy.write(vec![note(10, "hen", None)]).await,
        Err(NodeError::Limited { .. })
    ));
    tokio::time::sleep(Duration::from_millis(200)).await;
    busy.write(vec![note(10, "hen", None)]).await.unwrap();
}

#[tokio::test]
async fn past_its_memory_budget_a_node_lets_go_of_the_least_recently_used() {
    let cluster = Cluster::new();
    let measure = cluster.node();
    measure.create_tenant(&tenant("sized")).await.unwrap();
    let sized = measure.tenant(&tenant("sized")).await.unwrap();
    let writes = || {
        let rows = (0..200).map(|n| note(n, &"a long title ".repeat(20), None));
        std::iter::once(schema(EmbeddingProvider::None))
            .chain(rows)
            .collect::<Vec<_>>()
    };
    sized.write(writes()).await.unwrap();
    let one = sized.read(State::resident);
    assert!(one > 100_000, "{one}");
    drop(sized);
    measure.shutdown().await;

    let node = cluster.node_from(Settings {
        memory: Some(one * 3 / 2),
        ..cluster.settings()
    });
    node.sweeping();
    let second = cluster.node();
    for name in ["oldest", "middle", "newest"] {
        node.create_tenant(&tenant(name)).await.unwrap();
        let opened = node.tenant(&tenant(name)).await.unwrap();
        opened.write(writes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let taken = |name: &'static str| {
        let second = &second;
        async move { second.tenant(&tenant(name)).await.is_ok() }
    };
    while !(taken("oldest").await && taken("middle").await) {
        assert!(std::time::Instant::now() < deadline, "nothing was let go");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        matches!(
            second.tenant(&tenant("newest")).await,
            Err(NodeError::Elsewhere { .. })
        ),
        "the most recently used is kept"
    );
}

#[tokio::test]
async fn many_tenants_are_held_within_the_memory_budget() {
    let cluster = Cluster::new();
    let rows = || {
        let rows = (0..100).map(|n| note(n, &"a long title ".repeat(20), None));
        std::iter::once(schema(EmbeddingProvider::None))
            .chain(rows)
            .collect::<Vec<_>>()
    };
    let sizing = cluster.node();
    sizing.create_tenant(&tenant("sizing")).await.unwrap();
    let opened = sizing.tenant(&tenant("sizing")).await.unwrap();
    opened.write(rows()).await.unwrap();
    let one = opened.read(State::resident);
    drop(opened);
    sizing.shutdown().await;

    let budget = one * 4;
    let node = cluster.node_from(Settings {
        memory: Some(budget),
        ..cluster.settings()
    });
    for n in 0..12 {
        let name = format!("t{n}");
        node.create_tenant(&tenant(&name)).await.unwrap();
        node.tenant(&tenant(&name))
            .await
            .unwrap()
            .write(rows())
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    node.create_tenant(&tenant("last")).await.unwrap();
    node.tenant(&tenant("last")).await.unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while node.resident() > budget {
        assert!(
            std::time::Instant::now() < deadline,
            "{} held over a budget of {budget}",
            node.resident()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn a_tenant_flooding_a_node_is_turned_away_while_a_quiet_one_is_answered() {
    let cluster = Cluster::new();
    let node = cluster.node_from(Settings {
        queries_per_second: Some(20.0),
        ..cluster.settings()
    });
    for name in ["busy", "quiet"] {
        node.create_tenant(&tenant(name)).await.unwrap();
        let opened = node.tenant(&tenant(name)).await.unwrap();
        opened
            .write(vec![schema(EmbeddingProvider::None)])
            .await
            .unwrap();
    }
    let flood = (0..500).map(|_| {
        let node = node.clone();
        async move {
            node.reader(&tenant("busy"), Freshness::Eventual)
                .await
                .is_ok()
        }
    });
    let quiet = (0..10).map(|_| {
        let node = node.clone();
        async move {
            node.reader(&tenant("quiet"), Freshness::Eventual)
                .await
                .is_ok()
        }
    });
    let (flooded, answered) = tokio::join!(join_all(flood), join_all(quiet));
    assert!(
        flooded.iter().filter(|ok| **ok).count() <= 25,
        "the flood is held to its rate"
    );
    assert!(
        answered.iter().all(|ok| *ok),
        "the quiet tenant is answered"
    );
}

#[tokio::test]
async fn a_tenant_is_restored_to_a_point_in_its_window_and_readers_follow() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.node(), cluster.reading_node());
    let acme = tenant("acme");
    writing.create_tenant(&acme).await.unwrap();
    let day = Duration::from_secs(24 * 60 * 60);
    assert!(writing.set_restore_window(&acme, Some(day)).await.unwrap());
    let opened = writing.tenant(&acme).await.unwrap();
    opened
        .write(vec![
            schema(EmbeddingProvider::None),
            note(1, "fox", None),
            note(2, "hen", None),
        ])
        .await
        .unwrap();
    let (window, points) = writing.restore_points(&acme).await.unwrap().unwrap();
    let (_, kept) = points.unwrap();
    assert_eq!(window, day);

    cluster.clock.advance(Duration::from_secs(60 * 60));
    let removed = opened
        .write(vec![Mutation::Delete {
            collection: notes(),
            id: id(1),
            only_if_version: None,
        }])
        .await
        .unwrap();
    assert_eq!(opened.read(count), 1);

    let half_an_hour_on = kept.at + chrono::Duration::minutes(30);
    let restored = writing
        .restore(&acme, Point::At(half_an_hour_on), RequestId::generate())
        .await
        .unwrap();
    assert_eq!(
        (restored.restore.from, restored.restore.from_at),
        (kept.position, kept.at)
    );
    assert!(restored.position > removed.position);
    assert_eq!(restored.incarnation, removed.incarnation);
    assert_eq!(opened.read(count), 2);
    let followed = reading
        .reader(
            &acme,
            Freshness::After(restored.incarnation, restored.position),
        )
        .await
        .unwrap();
    assert_eq!(
        counted(&followed),
        2,
        "a copy elsewhere follows the restore"
    );

    cluster.clock.advance(2 * day);
    assert!(matches!(
        writing
            .restore(&acme, Point::At(half_an_hour_on), RequestId::generate())
            .await,
        Err(NodeError::NoRestorePoint(_))
    ));
    let (_, latest) = writing
        .restore_points(&acme)
        .await
        .unwrap()
        .unwrap()
        .1
        .unwrap();
    let back_to = writing
        .restore(
            &acme,
            Point::Position(latest.position),
            RequestId::generate(),
        )
        .await
        .unwrap();
    assert_eq!(
        (back_to.restore.from, back_to.restore.from_at),
        (latest.position, latest.at),
        "how it stood when the window began is in reach"
    );
    assert_eq!(
        writing.restore_points(&tenant("nobody")).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn a_restore_from_a_backup_pruned_since_is_answered_as_it_was() {
    let cluster = Cluster::new();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        ..cluster.settings()
    });
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    let opened = node.tenant(&acme).await.unwrap();
    opened
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let backup = node.back_up(&acme).await.unwrap();
    opened.write(vec![note(2, "hen", None)]).await.unwrap();

    let request = RequestId::generate();
    let restored = node
        .restore_backup(&acme, backup.id, request)
        .await
        .unwrap();
    Backups::new(store).discard(&backup).await.unwrap();
    assert_eq!(
        node.restore_backup(&acme, backup.id, request)
            .await
            .unwrap(),
        restored
    );
    assert!(matches!(
        node.restore_backup(&acme, backup.id, RequestId::generate())
            .await,
        Err(NodeError::UnknownBackup(..))
    ));
}

#[tokio::test]
async fn every_tenant_is_backed_up_once_a_day_held_or_not() {
    let cluster = Cluster::new();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let node = cluster.node_from(Settings {
        backups: Some(store),
        ..cluster.settings()
    });
    for name in ["acme", "beta"] {
        node.create_tenant(&tenant(name)).await.unwrap();
    }
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let counts = || async {
        let mut counts = Vec::new();
        for name in ["acme", "beta"] {
            counts.push(node.backups(&tenant(name)).await.unwrap().len());
        }
        counts
    };

    node.back_up_due().await.unwrap();
    assert_eq!(counts().await, [1, 1], "beta, never opened, too");
    node.back_up_due().await.unwrap();
    assert_eq!(counts().await, [1, 1], "not due again yet");
    cluster.clock.advance(Duration::from_secs(25 * 60 * 60));
    node.back_up_due().await.unwrap();
    assert_eq!(counts().await, [2, 2]);
    assert!(matches!(
        cluster.node().back_up(&tenant("acme")).await,
        Err(NodeError::NoBackups)
    ));
}

#[tokio::test]
async fn every_node_drops_what_it_cached_of_a_purged_tenant_not_made_again() {
    let cluster = Cluster::new();
    let (purging, other) = (cluster.node(), cluster.node());
    for name in ["gone", "back"] {
        purging.create_tenant(&tenant(name)).await.unwrap();
        purging.drop_tenant(&tenant(name)).await.unwrap();
    }
    purging.purge_due().await.unwrap();
    purging.create_tenant(&tenant("back")).await.unwrap();
    cluster.store.forgotten.lock().unwrap().clear();

    other.purge_due().await.unwrap();
    other.purge_due().await.unwrap();
    assert_eq!(
        *cluster.store.forgotten.lock().unwrap(),
        ["gone/"],
        "once, and not of the tenant made again"
    );
}

#[tokio::test(start_paused = true)]
async fn a_restore_stopped_after_its_commit_landed_is_answered_as_made() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = node_leasing(&cluster, coordinator.clone(), LEASE);
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let point = Point::Position(kept.position);
    let committed = || async {
        let committed = cluster.coordinator.committed(&key("acme")).await;
        committed.unwrap().unwrap().log_position
    };
    let before = committed().await;

    // The restore's commit lands and its answer never comes; the node lets
    // the tenant go meanwhile.
    coordinator.hang_after_commit.store(true, Ordering::SeqCst);
    let request = RequestId::generate();
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move { node.restore(&acme, point, request).await }
    });
    soon(coordinator.landed.notified()).await;
    let landed = committed().await;
    assert!(landed > before, "the restore is committed");
    node.shutdown().await;
    let stopped = soon(restoring).await.unwrap().unwrap();
    assert_eq!(stopped.position, landed, "told as it was made");

    // Another node finds the tenant put back, and the request answered.
    let second = cluster.node();
    let opened = second.tenant(&acme).await.unwrap();
    assert_eq!(opened.read(count), 1);
    opened.write(vec![note(3, "owl", None)]).await.unwrap();
    assert_eq!(
        second.restore(&acme, point, request).await.unwrap(),
        stopped
    );
    assert_eq!(second.tenant(&acme).await.unwrap().read(count), 2);
}

/// Whether `failure` tells its caller that what it asked for may or may
/// not have been made.
fn unknown(failure: &NodeError) -> bool {
    matches!(failure, NodeError::Storage(StorageError::Shared(_)))
}

#[tokio::test(start_paused = true)]
async fn a_restore_whose_commit_is_never_answered_is_told_as_unknown_not_as_unmade() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let node = node_leasing(&cluster, coordinator.clone(), LEASE);
    let acme = tenant("acme");
    let kept = restorable(&node, &acme).await;
    let point = Point::Position(kept.position);
    let request = RequestId::generate();

    // Given up after every try of it: it was not made, which the node
    // cannot know of a commit it has no answer to.
    coordinator.unanswered_commits.store(true, Ordering::SeqCst);
    let given_up = soon(node.restore(&acme, point, request)).await.unwrap_err();
    assert!(unknown(&given_up), "{given_up:?}");
    assert!(matches!(
        &given_up,
        NodeError::Storage(failure)
            if matches!(failure.root(), StorageError::Coordinator(CoordinatorError::Backend(_)))
    ));

    // And stopped while it waits for the answer: unknown no less.
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move { node.restore(&acme, point, request).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!restoring.is_finished());
    node.shutdown().await;
    let stopped = soon(restoring).await.unwrap().unwrap_err();
    assert!(unknown(&stopped), "{stopped:?}");

    // Asked again by its name it is made, once.
    let second = cluster.node();
    let restored = second.restore(&acme, point, request).await.unwrap();
    assert_eq!(
        second.restore(&acme, point, request).await.unwrap(),
        restored
    );
    assert_eq!(second.tenant(&acme).await.unwrap().read(count), 1);
}

/// A node that backs up to a store of its own, over a coordinator that
/// can be made to go wrong, with tenant acme backed up when it held one
/// note and holding two since.
async fn backed_up(cluster: &Cluster) -> (Node, Arc<Unrenewing>, Settings, uuid::Uuid) {
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone(), Duration::ZERO));
    let settings = Settings {
        backups: Some(store),
        ..cluster.settings()
    };
    let embedders = Arc::new(Embedders::new(EmbedSettings::default()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        embedders,
        settings.clone(),
    );
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    let opened = node.tenant(&acme).await.unwrap();
    opened
        .write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let backup = node.back_up(&acme).await.unwrap();
    opened.write(vec![note(2, "hen", None)]).await.unwrap();
    (node, coordinator, settings, backup.id)
}

#[tokio::test]
async fn a_restore_from_a_backup_whose_commit_is_refused_says_so_and_lets_its_claim_go() {
    let cluster = Cluster::new();
    let (node, coordinator, _, backup) = backed_up(&cluster).await;
    let acme = tenant("acme");
    let request = RequestId::generate();

    coordinator.refused_restores.store(true, Ordering::SeqCst);
    let refused = node.restore_backup(&acme, backup, request).await;
    let refused = refused.unwrap_err();
    assert!(
        matches!(
            &refused,
            NodeError::Storage(StorageError::Coordinator(CoordinatorError::Fenced { .. }))
        ),
        "told as it was refused: {refused:?}"
    );

    // Nothing holds the tenant's backups: it is restored at once when
    // asked again, and may be backed up.
    coordinator.refused_restores.store(false, Ordering::SeqCst);
    node.restore_backup(&acme, backup, request).await.unwrap();
    assert_eq!(node.tenant(&acme).await.unwrap().read(count), 1);
    node.back_up(&acme).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_restore_from_a_backup_stopped_under_way_lets_its_claim_go() {
    let cluster = Cluster::new();
    let (node, coordinator, settings, backup) = backed_up(&cluster).await;
    let acme = tenant("acme");
    let request = RequestId::generate();

    // The restore's commit is never answered, and the node lets the tenant
    // go while it waits.
    coordinator.unanswered_commits.store(true, Ordering::SeqCst);
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move { node.restore_backup(&acme, backup, request).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!restoring.is_finished());
    node.shutdown().await;
    assert!(soon(restoring).await.unwrap().is_err());
    coordinator
        .unanswered_commits
        .store(false, Ordering::SeqCst);

    // The claim went with it, not hours later: another node restores the
    // tenant as soon as it is asked.
    let second = cluster.node_from(Settings {
        node: NodeId::generate("node"),
        ..settings
    });
    soon(second.restore_backup(&acme, backup, request))
        .await
        .unwrap();
    assert_eq!(second.tenant(&acme).await.unwrap().read(count), 1);
}

#[tokio::test(start_paused = true)]
async fn a_delete_by_filter_is_charged_for_every_object_it_deleted() {
    let cluster = Cluster::new();
    let node = cluster.node_from(Settings {
        writes_per_second: Some(100.0),
        ..cluster.settings()
    });
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    let opened = node.tenant(&acme).await.unwrap();
    opened
        .write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    let foxes: Vec<_> = (0..40).map(|n| note(n, "fox", None)).collect();
    opened.write(foxes).await.unwrap();

    // Let through as one write, of the 59 it may still make this second,
    // and then owing one for each of the forty it deleted.
    let every_fox = Mutation::DeleteWhere {
        limit: None,
        collection: notes(),
        filter: vellev_storage::Filter::Eq {
            property: "title".into(),
            value: json!("fox"),
        },
    };
    let deleted = opened.write(vec![every_fox]).await.unwrap();
    assert_eq!(deleted.outcomes, vec![vellev_storage::Outcome::Deleted(40)]);
    let hens: Vec<_> = (0..30).map(|n| note(n, "hen", None)).collect();
    let turned_away = opened.write(hens.clone()).await;
    assert!(
        matches!(turned_away, Err(NodeError::Limited { .. })),
        "{turned_away:?}"
    );
    opened.write(hens[..10].to_vec()).await.unwrap();
}

#[tokio::test]
async fn a_collection_the_node_cannot_embed_for_is_not_made_but_one_there_already_is_left_be() {
    let cluster = Cluster::new();
    let node = cluster.node();
    node.create_tenant(&tenant("acme")).await.unwrap();
    // A collection this build cannot run the model of, as a node of
    // another build would have made it: written straight to the store.
    let unrunnable = || schema(EmbeddingProvider::Local);
    let key = PartitionKey {
        tenant: tenant("acme"),
        partition: Partition::DEFAULT,
    };
    let other_build = NodeId::generate("node");
    let acquired = cluster.coordinator.acquire(&key, &other_build, LEASE);
    let Acquired::Lease { lease, committed } = acquired.await.unwrap() else {
        panic!("nobody holds the tenant yet");
    };
    let coordinator: Arc<dyn Coordinator> = cluster.coordinator.clone();
    let writer = Writer::open(cluster.store.clone(), coordinator, lease, &committed)
        .await
        .unwrap();
    writer.commit(vec![unrunnable()]).await.unwrap();
    cluster.clock.advance(LEASE);

    let acme = node.tenant(&tenant("acme")).await.unwrap();
    let again = acme.write(vec![unrunnable()]).await.unwrap();
    assert_eq!(
        again.outcomes,
        [Outcome::Unchanged],
        "made again as it is, it changes nothing"
    );
    let no_such_model = |refused: Result<_, NodeError>| matches!(refused, Err(NodeError::NoSuchModel(reason)) if reason == "stand-in cannot be run locally");
    let Mutation::CreateCollection(mut changed) = unrunnable() else {
        unreachable!("a schema makes a collection");
    };
    changed.embedding.query_prefix = None;
    let clash = acme.write(vec![Mutation::CreateCollection(changed)]).await;
    assert!(
        matches!(
            &clash,
            Err(NodeError::Storage(error))
                if matches!(error.root(), StorageError::CollectionExists(_))
        ),
        "made otherwise, it clashes with what is stored, whatever its model: {clash:?}"
    );
    let dropped_and_made = vec![Mutation::DropCollection(notes()), unrunnable()];
    assert!(
        no_such_model(acme.write(dropped_and_made).await),
        "dropped and made in one batch, it would be made, and is not"
    );
    let Mutation::CreateCollection(mut elsewhere) = unrunnable() else {
        unreachable!("a schema makes a collection");
    };
    elsewhere.name = CollectionName::new("others").unwrap();
    assert!(no_such_model(
        acme.write(vec![Mutation::CreateCollection(elsewhere.clone())])
            .await
    ));
    let others = CollectionName::new("others").unwrap();
    assert!(acme.read(|state| state.collection(&others).is_none()));
    assert!(acme.read(|state| state.collection(&notes()).is_some()));

    // In the collection that is there, it is this build that cannot run
    // the model, not the writer of an object who is at fault.
    let unembedded = acme.write(vec![note(1, "fox", None)]).await;
    assert!(
        matches!(
            &unembedded,
            Err(NodeError::Storage(error)) if matches!(
                error.root(),
                StorageError::Embedding { lasting: Some(vellev_core::ModelFault::Build), .. }
            )
        ),
        "{unembedded:?}"
    );

    // A request answered already is answered as it was, and what it is
    // sent with the second time is not looked at.
    let request = Some(vellev_core::RequestId::generate());
    let mut supplied = elsewhere.clone();
    supplied.embedding.provider = EmbeddingProvider::None;
    let made = acme
        .write(Batch {
            request,
            mutations: vec![Mutation::CreateCollection(supplied)],
        })
        .await
        .unwrap();
    assert_eq!(made.outcomes, [Outcome::Applied]);
    // Not the collection just made, which would clash, but one that is
    // not there and could not be made.
    elsewhere.name = CollectionName::new("third").unwrap();
    let again = acme
        .write(Batch {
            request,
            mutations: vec![Mutation::CreateCollection(elsewhere.clone())],
        })
        .await
        .unwrap();
    assert!(no_such_model(
        acme.write(vec![Mutation::CreateCollection(elsewhere)])
            .await
    ));
    assert_eq!(
        (again.position, again.outcomes),
        (made.position, made.outcomes)
    );
}
