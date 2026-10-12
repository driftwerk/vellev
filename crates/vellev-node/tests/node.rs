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
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use serde_json::{json, Value};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, HttpApi,
    NodeId, ObjectId, Partition, Property, RequestId, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{
    ClaimClock, Freshness, Node, NodeError, Reader, RequestBounds, Settings, LOOKED_OVER_EVERY,
    REMEMBERED,
};
use vellev_storage::backups::{Backup, Backups, DELETED_AT_ONCE};
use vellev_storage::blobs::{Blobs, CachedBlobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Adoption, Clock, Commit, Committed, Coordinator, CoordinatorError, Forgotten,
    KeyHash, Lease, LogPosition, ManualClock, MemoryCoordinator, PartitionKey, Point, RestorePoint,
    Restored, TenantKey, Versions, MAX_BACKUP_HOLD,
};
use vellev_storage::format::{decode_head, Head};
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
            claim_clock: Arc::new(RuntimeClock::new()),
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
            Err(NodeError::Purging { .. })
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    /// How many renewals of leases were asked for.
    lease_renewals: AtomicUsize,
    /// Commits that make a restore are refused, as by a lease since taken.
    refused_restores: AtomicBool,
    /// Whether the next claim on a backup is taken but its answer lost.
    losing_claim: AtomicBool,
    /// How long each claim on a backup takes to be answered once taken.
    claims_slowly: Duration,
    /// Told each time a claim on a backup is taken, before it is answered.
    claimed: tokio::sync::Notify,
    /// Whether renewals of claims on backups are answered as taken over.
    disowning: AtomicBool,
    /// How many claims on backups were renewed.
    renewals: AtomicUsize,
    /// Whether the tenant whose keys are next asked for, as a backup does
    /// before it copies, is first removed.
    removing: AtomicBool,
    /// Whether a tenant removed so is purged and made again.
    made_again: bool,
    /// The format announcing answers the cluster writes, if not the one it
    /// does, as when another node raised it just before.
    announced_format: Option<u16>,
    /// How many more renewals of claims on backups are answered; the rest
    /// go unanswered.
    renewals_answered: AtomicUsize,
    /// How many renewals of claims on backups are next to fail as a
    /// backend would.
    renewals_failing: AtomicUsize,
    /// Whether renewals of claims on backups are refused outright.
    renewals_refused: AtomicBool,
    /// How many renewals of a claim are answered before the rest are
    /// refused, if any are to be.
    renewals_refused_after: std::sync::Mutex<Option<usize>>,
    /// Whether each renewal and release of a claim on a backup first looks
    /// for the claim having lapsed meanwhile, by taking it with an id of
    /// no one's.
    watching: AtomicBool,
    /// Whether a claim on a backup was found lapsed while still in use.
    lapsed: AtomicBool,
    /// Whether the next commit makes renewals answer as if the claim on
    /// its tenant's backups were taken over, and takes `commits_slowly`.
    disowning_on_commit: AtomicBool,
    commits_slowly: Duration,
    /// What happens once, just after an adoption is next noted for a
    /// tenant, as a restore asked to adopt has it be before its commit.
    adoption_noted: Hook,
    /// Whether two adoptions of one tenant are raced: then how many were
    /// noted is counted, a signal is given once a second is, and a commit
    /// by the lease that noted it waits, as over a slow link, until an
    /// adoption is next given up, which is signalled too.
    adoptions_raced: AtomicBool,
    adoptions_noted: AtomicUsize,
    second_adoption_noted: tokio::sync::Notify,
    commit_waits_for_giving_up: std::sync::Mutex<Option<vellev_storage::coordinator::Generation>>,
    adoption_given_up: tokio::sync::Notify,
    /// How many claims to purge were let go.
    purge_releases: AtomicUsize,
    /// Whether every claim on a backup is taken and the answer saying so
    /// lost, however often it is asked for.
    claims_unheard: AtomicBool,
    /// Whether letting a claim go, on a backup or to purge, waits, once it
    /// is asked for, which is signalled, until it is let through.
    releases_wait: AtomicBool,
    release_asked: tokio::sync::Notify,
    release_let_through: tokio::sync::Notify,
    /// How long each renewal of a claim on a backup or a purge that is
    /// answered takes to be.
    claim_renewals_slowly: Duration,
    /// Whether commits fail as a backend would, none of them landing.
    commits_failing: AtomicBool,
    /// What happens once, just after a claim on a backup is next taken.
    backup_claimed: Hook,
    /// How many claims on backups were asked for.
    claims: AtomicUsize,
}

impl Unrenewing {
    /// One whose renewals each take `slowly` to be answered.
    fn slow(inner: Arc<MemoryCoordinator>, slowly: Duration) -> Self {
        Self {
            slowly,
            ..Self::new(inner)
        }
    }

    fn new(inner: Arc<MemoryCoordinator>) -> Self {
        Self {
            inner,
            failing: AtomicBool::new(false),
            slowly: Duration::ZERO,
            losing_claim: AtomicBool::new(false),
            claims_slowly: Duration::ZERO,
            claimed: tokio::sync::Notify::new(),
            disowning: AtomicBool::new(false),
            renewals: AtomicUsize::new(0),
            removing: AtomicBool::new(false),
            made_again: false,
            announced_format: None,
            renewals_answered: AtomicUsize::new(usize::MAX),
            renewals_failing: AtomicUsize::new(0),
            renewals_refused: AtomicBool::new(false),
            renewals_refused_after: std::sync::Mutex::new(None),
            watching: AtomicBool::new(false),
            lapsed: AtomicBool::new(false),
            disowning_on_commit: AtomicBool::new(false),
            commits_slowly: Duration::ZERO,
            adoption_noted: Hook::default(),
            adoptions_raced: AtomicBool::new(false),
            adoptions_noted: AtomicUsize::new(0),
            second_adoption_noted: tokio::sync::Notify::new(),
            commit_waits_for_giving_up: std::sync::Mutex::new(None),
            adoption_given_up: tokio::sync::Notify::new(),
            purge_releases: AtomicUsize::new(0),
            claims_unheard: AtomicBool::new(false),
            releases_wait: AtomicBool::new(false),
            release_asked: tokio::sync::Notify::new(),
            release_let_through: tokio::sync::Notify::new(),
            claim_renewals_slowly: Duration::ZERO,
            commits_failing: AtomicBool::new(false),
            backup_claimed: Hook::default(),
            claims: AtomicUsize::new(0),
            unanswered_commits: AtomicBool::new(false),
            hang_after_commit: AtomicBool::new(false),
            landed: tokio::sync::Notify::new(),
            failing_forgets: AtomicBool::new(false),
            forgets: AtomicUsize::new(0),
            lose_answers: AtomicUsize::new(0),
            refused_restores: AtomicBool::new(false),
            lease_renewals: AtomicUsize::new(0),
        }
    }

    /// What a renewal of a claim on a backup or a purge meets before it
    /// reaches the coordinator: failing, going unanswered, being refused or
    /// taking its time, as the knobs say.
    async fn renewal_let_through(&self) -> Result<(), CoordinatorError> {
        self.renewals.fetch_add(1, Ordering::SeqCst);
        let failing = self
            .renewals_failing
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        if failing.is_ok() {
            return Err(CoordinatorError::Backend("connection reset".into()));
        }
        let answered =
            self.renewals_answered
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        if answered.is_err() {
            tokio::time::sleep(Duration::from_secs(60 * 60)).await;
        }
        let asked = self.renewals.load(Ordering::SeqCst);
        let refused = self.renewals_refused.load(Ordering::SeqCst)
            || (*self.renewals_refused_after.lock().unwrap()).is_some_and(|after| asked > after);
        if refused {
            return Err(CoordinatorError::InvalidArgument("refused".into()));
        }
        tokio::time::sleep(self.claim_renewals_slowly).await;
        Ok(())
    }

    /// Note whether the claim on backing `tenant` up lapsed, if watching.
    async fn watch_lapse(&self, tenant: &TenantId) -> Result<(), CoordinatorError> {
        if !self.watching.load(Ordering::SeqCst) {
            return Ok(());
        }
        let probe = uuid::Uuid::new_v4();
        if self.inner.claim_backup(tenant, probe, None, LEASE).await? {
            self.lapsed.store(true, Ordering::SeqCst);
            self.inner.release_backup(tenant, probe, false).await?;
        }
        Ok(())
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
        self.lease_renewals.fetch_add(1, Ordering::SeqCst);
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
        let waits = *self.commit_waits_for_giving_up.lock().unwrap() == Some(lease.generation);
        if waits {
            let given_up = self.adoption_given_up.notified();
            tokio::time::timeout(Duration::from_secs(60), given_up)
                .await
                .expect("an adoption is given up");
        }
        if self.disowning_on_commit.swap(false, Ordering::SeqCst) {
            self.disowning.store(true, Ordering::SeqCst);
            tokio::time::sleep(self.commits_slowly).await;
        }
        if self.commits_failing.load(Ordering::SeqCst) {
            return Err(CoordinatorError::Backend("connection reset".into()));
        }
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
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<u16, CoordinatorError> {
        let format = self.inner.announce(node, versions).await?;
        Ok(self.announced_format.unwrap_or(format))
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
        if self.removing.swap(false, Ordering::SeqCst) {
            let key = PartitionKey {
                tenant: tenant.clone(),
                partition: Partition::DEFAULT,
            };
            self.inner.remove(&key).await?;
            if self.made_again {
                let purge = self
                    .inner
                    .claim_purge(tenant, Duration::ZERO, LEASE)
                    .await?;
                let purge = purge.expect("the tenant removed is to be purged");
                self.inner.purged(tenant, purge.id, REMEMBERED).await?;
                self.inner.register(&key).await?;
            }
        }
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
        let noted = self
            .inner
            .adopt_on_commit(key, generation, adoption)
            .await?;
        let raced = noted && self.adoptions_raced.load(Ordering::SeqCst);
        if raced && self.adoptions_noted.fetch_add(1, Ordering::SeqCst) == 1 {
            *self.commit_waits_for_giving_up.lock().unwrap() = Some(generation);
            self.second_adoption_noted.notify_one();
        }
        self.adoption_noted.run().await;
        Ok(noted)
    }

    async fn adopt_no_more(
        &self,
        key: &PartitionKey,
        generation: vellev_storage::coordinator::Generation,
    ) -> Result<(), CoordinatorError> {
        let given_up = self.inner.adopt_no_more(key, generation).await;
        self.adoption_given_up.notify_one();
        given_up
    }

    async fn claim_purge(
        &self,
        tenant: &TenantId,
        due: Duration,
        hold: Duration,
    ) -> Result<Option<vellev_storage::coordinator::PurgeClaim>, CoordinatorError> {
        let claimed = self.inner.claim_purge(tenant, due, hold).await?;
        tokio::time::sleep(self.claims_slowly).await;
        Ok(claimed)
    }

    async fn renew_purge(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.renewal_let_through().await?;
        self.inner.renew_purge(tenant, claim, hold).await
    }

    async fn release_purge(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
    ) -> Result<(), CoordinatorError> {
        if self.releases_wait.load(Ordering::SeqCst) {
            self.release_asked.notify_one();
            self.release_let_through.notified().await;
        }
        self.purge_releases.fetch_add(1, Ordering::SeqCst);
        self.inner.release_purge(tenant, claim).await
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
        self.claims.fetch_add(1, Ordering::SeqCst);
        let taken = self.inner.claim_backup(tenant, claim, due, hold).await?;
        if taken {
            self.backup_claimed.run().await;
        }
        self.claimed.notify_one();
        tokio::time::sleep(self.claims_slowly).await;
        let unheard = self.claims_unheard.load(Ordering::SeqCst);
        if self.losing_claim.swap(false, Ordering::SeqCst) || unheard {
            return Err(CoordinatorError::Backend("connection reset".into()));
        }
        Ok(taken)
    }

    async fn renew_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.watch_lapse(tenant).await?;
        self.renewal_let_through().await?;
        let renewed = self.inner.renew_backup(tenant, claim, hold).await?;
        Ok(renewed && !self.disowning.load(Ordering::SeqCst))
    }

    async fn release_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        made: bool,
    ) -> Result<(), CoordinatorError> {
        // As over a network: whoever asks waits for the answer.
        tokio::task::yield_now().await;
        if self.releases_wait.load(Ordering::SeqCst) {
            self.release_asked.notify_one();
            self.release_let_through.notified().await;
        }
        self.watch_lapse(tenant).await?;
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let coordinator = Arc::new(Unrenewing::slow(cluster.coordinator.clone(), lease / 3));
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let renewed = coordinator.lease_renewals.load(Ordering::SeqCst);
    tokio::time::sleep(lease).await;
    assert_eq!(
        coordinator.lease_renewals.load(Ordering::SeqCst),
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let renewed = coordinator.lease_renewals.load(Ordering::SeqCst);
    tokio::time::sleep(lease * 2).await;
    assert_eq!(
        coordinator.lease_renewals.load(Ordering::SeqCst),
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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

    // Its history is the purge's to forget, under the claim that purges
    // it: a purge that cannot forget it is not done, and is taken up
    // again once its claim has run out.
    coordinator.failing_forgets.store(true, Ordering::SeqCst);
    assert!(node.drop_tenant(&tenant("acme")).await.unwrap());
    assert_eq!(coordinator.forgets.load(Ordering::SeqCst), 0);
    node.purge_due().await.unwrap();
    let asked = coordinator.forgets.load(Ordering::SeqCst);
    assert!(asked > 0);
    assert!(
        node.create_tenant(&tenant("acme")).await.is_err(),
        "not purged while its history cannot be forgotten"
    );
    // Its files were deleted under the claim before it failed, so the
    // claim is left to run out, and nothing is asked again until it has.
    coordinator.failing_forgets.store(false, Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert_eq!(coordinator.forgets.load(Ordering::SeqCst), asked);
    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(coordinator.forgets.load(Ordering::SeqCst) > asked);
    assert_eq!(
        coordinator.purge_releases.load(Ordering::SeqCst),
        0,
        "left to run out when it failed, and ended by the coordinator when done"
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_strong_read_on_a_holder_behind_its_own_commit_reads_it_from_a_copy() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    noted(&asked_only, 1, "fox").await;
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
async fn how_old_a_tenants_files_are_is_told_by_the_clock_the_node_is_given() {
    let cluster = Cluster::new();
    // Not the coordinator's clock, by which the tenant's lease runs out.
    let hour = Duration::from_secs(60 * 60);
    let clock = Arc::new(ManualClock::starting_at(Utc::now()));
    let node = cluster.node_from(Settings {
        clock: clock.clone(),
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(0, "ant", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    acme.write(vec![note(1, "fox", None)]).await.unwrap();
    acme.flush().await.unwrap();

    assert_eq!(
        acme.compact(hour).await.unwrap(),
        0,
        "every file was written within the hour"
    );
    clock.advance(2 * hour);
    assert_eq!(
        acme.compact(hour).await.unwrap(),
        3,
        "both logs and the first checkpoint are two hours old by the node's clock"
    );
}

#[tokio::test]
async fn a_node_looking_after_a_tenant_keeps_its_files_for_the_grace_it_is_set_to() {
    let cluster = Cluster::new();
    let minutes = |n: u64| Duration::from_secs(n * 60);
    // The tenant is flushed at every write, and looked after once each
    // flush is made; nothing may be restored, so the grace is all its
    // files are kept for: thirty minutes, twice the least a node takes
    // (`LEAST_GRACE`), so that files kept for that least instead are gone
    // too soon. Not the coordinator's clock, by which the lease runs out.
    let clock = Arc::new(ManualClock::starting_at(Utc::now()));
    let node = cluster.node_from(Settings {
        flush_rows: 1,
        grace: minutes(30),
        clock: clock.clone(),
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None)])
        .await
        .unwrap();
    // Two writes, each flushed: the node looks the tenant over after each
    // flush and before the next, so by the second flush it has looked it
    // over as it stood at the first.
    let looked_after = |from: u64| {
        let acme = acme.clone();
        async move {
            for n in from..from + 2 {
                acme.write(vec![note(n, "fox", None)]).await.unwrap();
                for _ in 0..500 {
                    if acme.read(State::unflushed) == 0 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                assert_eq!(acme.read(State::unflushed), 0, "flushed at every write");
            }
        }
    };
    let heads = || async {
        let mut keys = keys_under(cluster.store.as_ref(), "acme/").await;
        keys.retain(|key| key.ends_with(".log") || key.ends_with(".manifest"));
        keys
    };
    looked_after(0).await;
    let first = heads().await;
    assert!(first.len() >= 4, "{first:?}");

    clock.advance(minutes(20));
    looked_after(2).await;
    let kept = heads().await;
    assert!(
        first.iter().all(|key| kept.contains(key)),
        "twenty minutes old, and kept thirty: {first:?} of {kept:?}"
    );

    clock.advance(minutes(20));
    looked_after(4).await;
    let kept = heads().await;
    assert!(
        !first.iter().any(|key| kept.contains(key)),
        "forty minutes old, and nothing reaches them: {first:?} of {kept:?}"
    );
    assert_eq!(acme.read(count), 6);
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
async fn a_node_that_announces_into_a_newer_format_is_no_longer_ready() {
    let cluster = Cluster::new();
    let reads = cluster.node().versions().format;
    let coordinator = Arc::new(Unrenewing {
        announced_format: Some(reads + 1),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = node_leasing(&cluster, coordinator, LEASE);
    node.ready().await.unwrap();

    node.announcing();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match node.ready().await {
            Err(NodeError::Outdated { cluster, .. }) => {
                assert_eq!(cluster, reads + 1);
                break;
            }
            other => assert!(
                std::time::Instant::now() < deadline,
                "still {other:?}, though the format read asking was newer"
            ),
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    node.shutdown().await;
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
    Backups::new(store).forget(&backup).await.unwrap();
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
    for name in ["acme", "beta", "gamma"] {
        node.create_tenant(&tenant(name)).await.unwrap();
    }
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let elsewhere = cluster.node();
    let beta = elsewhere.tenant(&tenant("beta")).await.unwrap();
    beta.write(vec![schema(EmbeddingProvider::None), note(1, "owl", None)])
        .await
        .unwrap();
    drop(beta);
    elsewhere.shutdown().await;
    let counts = || async {
        let mut counts = Vec::new();
        for name in ["acme", "beta", "gamma"] {
            counts.push(node.backups(&tenant(name)).await.unwrap().len());
        }
        counts
    };

    node.back_up_due().await.unwrap();
    assert_eq!(
        counts().await,
        [1, 1, 0],
        "beta, held by no node, too; gamma, holding nothing, not"
    );
    node.back_up_due().await.unwrap();
    assert_eq!(counts().await, [1, 1, 0], "not due again yet");
    cluster.clock.advance(Duration::from_secs(25 * 60 * 60));
    node.back_up_due().await.unwrap();
    assert_eq!(counts().await, [2, 2, 0]);
    assert!(matches!(
        node.back_up(&tenant("gamma")).await,
        Err(NodeError::Empty(_))
    ));
    assert!(matches!(
        cluster.node().back_up(&tenant("acme")).await,
        Err(NodeError::NoBackups)
    ));
}

#[tokio::test]
async fn the_daily_backups_claim_no_tenant_that_never_committed() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(Arc::new(SlowStore::new(Duration::ZERO))),
            ..cluster.settings()
        },
    );
    node.create_tenant(&tenant("gamma")).await.unwrap();
    node.back_up_due().await.unwrap();
    node.back_up_due().await.unwrap();
    assert_eq!(coordinator.claims.load(Ordering::SeqCst), 0);
    assert_eq!(node.backups(&tenant("gamma")).await.unwrap(), []);
}

/// A store each write to which takes `slowly`, and each read of a backed
/// up file `file_reads`; the next such read once told to `stalls` the
/// whole runtime first, as a paused VM would.
#[derive(Debug)]
struct SlowStore {
    inner: ObjectBlobs,
    slowly: Duration,
    file_reads: std::sync::Mutex<Duration>,
    stalls: AtomicBool,
    /// How long each delete of a backup's complete record takes.
    deletes: std::sync::Mutex<Duration>,
}

impl SlowStore {
    fn new(slowly: Duration) -> Self {
        Self {
            inner: ObjectBlobs::new(Arc::new(InMemory::new())),
            slowly,
            file_reads: std::sync::Mutex::new(Duration::ZERO),
            stalls: AtomicBool::new(false),
            deletes: std::sync::Mutex::new(Duration::ZERO),
        }
    }
}

#[async_trait]
impl Blobs for SlowStore {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        tokio::time::sleep(self.slowly).await;
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.contains("/data/") {
            let delay = *self.file_reads.lock().unwrap();
            tokio::time::sleep(delay).await;
            if self.stalls.swap(false, Ordering::SeqCst) {
                std::thread::sleep(STALL);
            }
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        if key.ends_with(".backup.json") {
            let delay = *self.deletes.lock().unwrap();
            tokio::time::sleep(delay).await;
        }
        self.inner.delete(key).await
    }
}

/// A node that backs tenants up to `backups` through `coordinator`, and
/// tenant acme made and written to by another node, since stopped, so
/// that none holds it open: a test may then pause the runtime's clock. The
/// node waits a day for the backup store, so that one slow to write, as a
/// test makes it to have work outlast a claim, is not given up on.
async fn backing_up_through(
    cluster: &Cluster,
    coordinator: Arc<Unrenewing>,
    backups: Arc<dyn Blobs>,
) -> Node {
    let node = |settings: Settings| {
        Node::new(
            coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            settings,
        )
    };
    let writing = node(cluster.settings());
    writing.create_tenant(&tenant("acme")).await.unwrap();
    let acme = writing.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    drop(acme);
    writing.shutdown().await;
    node(Settings {
        backups: Some(backups),
        request_bounds: RequestBounds {
            store: Duration::from_secs(24 * 60 * 60),
            ..RequestBounds::default()
        },
        ..cluster.settings()
    })
}

/// Whether nobody holds a claim on backing `tenant` up, found by taking one
/// and letting it go.
async fn unclaimed(coordinator: &MemoryCoordinator, tenant: &TenantId) -> bool {
    let probe = uuid::Uuid::new_v4();
    let taken = coordinator
        .claim_backup(tenant, probe, None, LEASE)
        .await
        .unwrap();
    coordinator
        .release_backup(tenant, probe, false)
        .await
        .unwrap();
    taken
}

#[tokio::test]
async fn a_backup_whose_claim_was_taken_unheard_is_made_and_lets_the_claim_go() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        losing_claim: AtomicBool::new(true),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = backing_up_through(
        &cluster,
        coordinator.clone(),
        Arc::new(SlowStore::new(Duration::ZERO)),
    )
    .await;
    let backup = node.back_up(&tenant("acme")).await.unwrap();
    assert!(
        !coordinator.losing_claim.load(Ordering::SeqCst),
        "the claim was taken, and the answer saying so lost"
    );
    assert_eq!(node.backups(&tenant("acme")).await.unwrap(), [backup]);
    assert!(unclaimed(&cluster.coordinator, &tenant("acme")).await);
}

#[tokio::test]
async fn a_claim_taken_for_a_caller_who_stopped_waiting_is_let_go() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        claims_slowly: Duration::from_millis(200),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = backing_up_through(
        &cluster,
        coordinator.clone(),
        Arc::new(SlowStore::new(Duration::ZERO)),
    )
    .await;
    let backing = tokio::spawn({
        let node = node.clone();
        async move { node.back_up(&tenant("acme")).await }
    });
    coordinator.claimed.notified().await;
    backing.abort();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !unclaimed(&cluster.coordinator, &tenant("acme")).await {
        assert!(
            std::time::Instant::now() < deadline,
            "the claim is still held"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap().len(),
        1,
        "the backup asked for was made all the same"
    );
}

/// Time that passes as the runtime's does, paused or not, and further by
/// however long the machine is made to sleep through: as a coordinator's
/// clock, and as a claim's.
#[derive(Debug)]
struct RuntimeClock {
    start: chrono::DateTime<Utc>,
    origin: tokio::time::Instant,
    slept: std::sync::Mutex<Duration>,
}

impl RuntimeClock {
    fn new() -> Self {
        Self {
            start: Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
            origin: tokio::time::Instant::now(),
            slept: std::sync::Mutex::new(Duration::ZERO),
        }
    }

    fn elapsed(&self) -> Duration {
        self.origin.elapsed() + *self.slept.lock().unwrap()
    }

    /// Move on by `slept`, as a machine suspended that long finds on waking.
    fn sleep_through(&self, slept: Duration) {
        *self.slept.lock().unwrap() += slept;
    }
}

impl Clock for RuntimeClock {
    fn now(&self) -> chrono::DateTime<Utc> {
        self.start + chrono::Duration::from_std(self.elapsed()).unwrap()
    }
}

impl ClaimClock for RuntimeClock {
    fn now(&self) -> Duration {
        self.elapsed()
    }
}

/// A coordinator whose claims run out by `clock`.
fn ticking_by(clock: Arc<RuntimeClock>) -> Arc<MemoryCoordinator> {
    Arc::new(MemoryCoordinator::new(clock))
}

/// A coordinator whose claims run out in the runtime's time.
fn ticking() -> Arc<MemoryCoordinator> {
    ticking_by(Arc::new(RuntimeClock::new()))
}

/// How long a node stalls in the tests that stall one: longer than the
/// claims they hold.
const STALL: Duration = Duration::from_millis(500);

/// Something a store does once, at a point a test picks.
#[derive(Default)]
struct Hook(std::sync::Mutex<Option<futures::future::BoxFuture<'static, ()>>>);

impl Hook {
    fn set(&self, hooked: impl std::future::Future<Output = ()> + Send + 'static) {
        *self.0.lock().unwrap() = Some(Box::pin(hooked));
    }

    fn is_set(&self) -> bool {
        self.0.lock().unwrap().is_some()
    }

    /// Do it, if it is to be done, once.
    async fn run(&self) {
        let hooked = self.0.lock().unwrap().take();
        if let Some(hooked) = hooked {
            hooked.await;
        }
    }
}

impl std::fmt::Debug for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hook")
    }
}

/// A backup store that counts the writes and deletes begun once another
/// could take the claim on acme's backups, and runs a hook before the next
/// delete of a complete record, as pruning makes, and another after the
/// next write of one lands; and, once told to, loses every backed up file
/// just after the next complete record is written, as a node that stalled
/// past its claim would delete those a newer backup reuses.
#[derive(Debug)]
struct Watched {
    inner: ObjectBlobs,
    coordinator: Arc<MemoryCoordinator>,
    late: AtomicUsize,
    deleting: Hook,
    written: Hook,
    losing_files: AtomicBool,
}

impl Watched {
    fn new(coordinator: Arc<MemoryCoordinator>) -> Self {
        Self {
            inner: ObjectBlobs::new(Arc::new(InMemory::new())),
            coordinator,
            late: AtomicUsize::new(0),
            deleting: Hook::default(),
            written: Hook::default(),
            losing_files: AtomicBool::new(false),
        }
    }

    /// Count a write or delete begun if another could take the claim now.
    async fn watched(&self) {
        if unclaimed(&self.coordinator, &tenant("acme")).await {
            self.late.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[async_trait]
impl Blobs for Watched {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.watched().await;
        self.inner.put(key, bytes).await?;
        if key.ends_with(".backup.json") {
            self.written.run().await;
            if self.losing_files.swap(false, Ordering::SeqCst) {
                let kept = self.inner.list("_backups/acme/").await?;
                for file in kept.iter().filter(|kept| kept.key.contains("/data/")) {
                    self.inner.delete(&file.key).await?;
                }
            }
        }
        Ok(())
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.watched().await;
        if key.ends_with(".backup.json") {
            self.deleting.run().await;
        }
        self.inner.delete(key).await
    }
}

/// Two backups of acme kept by one node, and another node that prunes
/// them, and with them all its backups, once it takes a third: both
/// backing up to `watched` through `coordinator`, under claims measured by
/// `claims` and `claimed` long.
async fn pruned_after(
    cluster: &Cluster,
    coordinator: Arc<MemoryCoordinator>,
    watched: Arc<Watched>,
    claims: Arc<RuntimeClock>,
    claimed: Duration,
) -> (Node, Backup, Backup) {
    let node = |backups_kept| {
        Node::new(
            coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            Settings {
                backups: Some(watched.clone()),
                backups_kept,
                backup_claim: claimed,
                claim_clock: claims.clone(),
                ..cluster.settings()
            },
        )
    };
    let (keeping, pruning) = (node(Duration::from_secs(60 * 60)), node(Duration::ZERO));
    keeping.create_tenant(&tenant("acme")).await.unwrap();
    let acme = keeping.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let first = keeping.back_up(&tenant("acme")).await.unwrap();
    let second = keeping.back_up(&tenant("acme")).await.unwrap();
    (pruning, first, second)
}

#[tokio::test]
async fn a_node_that_stalls_past_its_claim_deletes_nothing_more() {
    let cluster = Cluster::new();
    let coordinator = ticking();
    let watched = Arc::new(Watched::new(coordinator.clone()));
    let claims = Arc::new(RuntimeClock::new());
    let short = Duration::from_millis(300);
    let (pruning, first, second) =
        pruned_after(&cluster, coordinator, watched.clone(), claims, short).await;

    // Pruning forgets the first two; the node stalls as it deletes the
    // first, and starts no other delete once it could be another's.
    watched.deleting.set(async { std::thread::sleep(STALL) });
    let third = pruning.back_up(&tenant("acme")).await.unwrap();
    assert!(!watched.deleting.is_set(), "a delete stalled");
    assert_eq!(
        watched.late.load(Ordering::SeqCst),
        0,
        "no write or delete begins once another could take the claim"
    );
    let left = pruning.backups(&tenant("acme")).await.unwrap();
    assert_eq!(left, [second, third], "{first:?} went as the node stalled");
}

#[tokio::test]
async fn a_node_that_sleeps_past_its_claim_deletes_nothing_more() {
    let cluster = Cluster::new();
    let (wall, claims) = (Arc::new(RuntimeClock::new()), Arc::new(RuntimeClock::new()));
    let coordinator = ticking_by(wall.clone());
    let watched = Arc::new(Watched::new(coordinator.clone()));
    let (pruning, first, second) = pruned_after(
        &cluster,
        coordinator,
        watched.clone(),
        claims.clone(),
        hold(),
    )
    .await;

    // The machine is suspended as the first is deleted: the claim's clock
    // finds the claim past when it is taken for lost, and the coordinator,
    // which went on longer, past when it ran out.
    watched.deleting.set(async move {
        claims.sleep_through(hold() - hold() / 3);
        wall.sleep_through(hold() * 2);
    });
    let third = pruning.back_up(&tenant("acme")).await.unwrap();
    assert!(!watched.deleting.is_set(), "the machine slept");
    assert_eq!(
        watched.late.load(Ordering::SeqCst),
        0,
        "no write or delete begins once another could take the claim"
    );
    let left = pruning.backups(&tenant("acme")).await.unwrap();
    assert_eq!(left, [second, third], "{first:?} went as the machine slept");
}

#[tokio::test]
async fn a_backup_whose_files_another_deleted_once_it_was_written_is_made_whole() {
    let cluster = Cluster::new();
    let coordinator = ticking();
    let watched = Arc::new(Watched::new(coordinator.clone()));
    let claims = Arc::new(RuntimeClock::new());
    let (node, ..) = pruned_after(&cluster, coordinator, watched.clone(), claims, hold()).await;

    // A node that slept past its claim deletes, as unlisted when it looked,
    // every file the newest backup reuses, just after it is written.
    watched.losing_files.store(true, Ordering::SeqCst);
    let newest = node.back_up(&tenant("acme")).await.unwrap();
    assert!(
        !watched.losing_files.load(Ordering::SeqCst),
        "the files went"
    );
    for file in newest.files() {
        let kept = format!("{}{file}", files_of(&newest));
        watched.get(&kept).await.unwrap();
    }
    let elsewhere = Cluster::new();
    let restoring = elsewhere.node_from(Settings {
        backups: Some(watched.clone()),
        ..elsewhere.settings()
    });
    restoring
        .adopt_backup(&tenant("acme"), newest.id, true, RequestId::generate())
        .await
        .unwrap();
    let opened = restoring.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(opened.read(count), 1, "the newest backup restores");
}

#[tokio::test]
async fn a_backup_that_cannot_be_made_whole_is_forgotten_and_its_run_fails() {
    let cluster = Cluster::new();
    let coordinator = ticking();
    let watched = Arc::new(Watched::new(coordinator.clone()));
    let claims = Arc::new(RuntimeClock::new());
    let (node, first, second) =
        pruned_after(&cluster, coordinator, watched.clone(), claims, hold()).await;

    // Every file the newest backup reuses is deleted just after it is
    // written, and the tenant's own store has lost them by then too.
    let primary = cluster.store.clone();
    watched.written.set(async move {
        for file in primary.list("acme/").await.unwrap() {
            primary.delete(&file.key).await.unwrap();
        }
    });
    watched.losing_files.store(true, Ordering::SeqCst);
    let failed = node.back_up(&tenant("acme")).await;
    assert!(
        !watched.losing_files.load(Ordering::SeqCst),
        "the files went"
    );
    assert!(
        matches!(
            failed,
            Err(NodeError::Storage(StorageError::Missing { .. }))
        ),
        "{failed:?}"
    );
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap(),
        [first, second],
        "a backup lacking files is not listed, and nothing is pruned by it"
    );
}

#[tokio::test]
async fn a_backup_whose_claim_is_lost_once_it_is_written_is_made_or_forgotten_with_its_tenant() {
    let cluster = Cluster::new();
    let claims = Arc::new(RuntimeClock::new());
    let coordinator = ticking();
    let watched = Arc::new(Watched::new(coordinator.clone()));
    let (node, first, second) = pruned_after(
        &cluster,
        coordinator.clone(),
        watched.clone(),
        claims.clone(),
        hold(),
    )
    .await;

    // The claim is lost just after the complete record lands: the backup
    // is made all the same.
    let losing = claims.clone();
    watched.written.set(async move {
        losing.sleep_through(hold());
        tokio::task::yield_now().await;
    });
    let made = node.back_up(&tenant("acme")).await.unwrap();
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap(),
        [first, second, made],
        "made, though, its claim lost, nothing is pruned"
    );

    // The tenant is removed, and the claim lost, just after another lands:
    // that one is forgotten all the same.
    let removing = (claims, coordinator);
    watched.written.set(async move {
        removing.1.remove(&acme()).await.unwrap();
        removing.0.sleep_through(hold());
        tokio::task::yield_now().await;
    });
    let forgotten = node.back_up(&tenant("acme")).await;
    assert!(
        matches!(forgotten, Err(NodeError::UnknownTenant(_))),
        "{forgotten:?}"
    );
    let records = watched.inner.list("_backups/acme/").await.unwrap();
    let records = records.iter().filter(|kept| kept.key.ends_with(".json"));
    assert_eq!(
        records.count(),
        3,
        "the backup of the tenant removed is forgotten"
    );
}

/// The ids of `keys`, in order.
fn key_ids(keys: &[TenantKey]) -> Vec<uuid::Uuid> {
    let mut ids: Vec<uuid::Uuid> = keys.iter().map(|key| key.id).collect();
    ids.sort();
    ids
}

/// Where the files of `backup`, and of the others its cluster took of its
/// tenant, lie.
fn files_of(backup: &Backup) -> String {
    let Backup {
        tenant,
        lineage,
        cluster,
        ..
    } = backup;
    format!("_backups/{tenant}/data/{lineage}/{cluster}/")
}

/// Whether `key` is of what the cluster that took `backup` keeps of its
/// tenant.
fn kept_with(backup: &Backup, key: &str) -> bool {
    let Backup {
        tenant,
        lineage,
        cluster,
        ..
    } = backup;
    let records = format!("_backups/{tenant}/records/{lineage}/{cluster}/");
    key.starts_with(&records) || key.starts_with(&files_of(backup))
}

/// The default claim on a backup, as a node holds it unless told
/// otherwise.
fn hold() -> Duration {
    Settings::new(NodeId::generate("node")).backup_claim
}

/// That `took`, on a clock paused part way into a test, is `expected` but
/// for less than a second, far less than any step a claim takes.
fn took_about(took: Duration, expected: Duration, why: &str) {
    assert!(
        took >= expected && took < expected + Duration::from_secs(1),
        "{why}: took {took:?}, not {expected:?}"
    );
}

/// A coordinator whose claims run out in the runtime's time, watching for
/// one lapsing while in use, with `knobs` turned; a node backing up to a
/// store each write to which takes as long as two claims last, and the
/// time a backup of acme took and what it came to.
async fn backed_up_slowly(
    knobs: impl FnOnce(&mut Unrenewing),
) -> (Arc<Unrenewing>, Duration, Result<(), NodeError>) {
    let cluster = Cluster::new();
    let mut coordinator = Unrenewing {
        watching: AtomicBool::new(true),
        ..Unrenewing::new(ticking())
    };
    knobs(&mut coordinator);
    let coordinator = Arc::new(coordinator);
    let slowly = Arc::new(SlowStore::new(hold() * 2));
    let node = backing_up_through(&cluster, coordinator.clone(), slowly).await;
    tokio::time::pause();
    let started = tokio::time::Instant::now();
    let backed_up = node.back_up(&tenant("acme")).await.map(drop);
    (coordinator, started.elapsed(), backed_up)
}

#[tokio::test]
async fn a_backup_that_outlasts_its_claim_holds_it_throughout() {
    let (coordinator, took, backed_up) = backed_up_slowly(|_| {}).await;
    backed_up.unwrap();
    assert!(took >= hold() * 4, "took {took:?}");
    assert!(coordinator.renewals.load(Ordering::SeqCst) >= 9);
    assert!(
        !coordinator.lapsed.load(Ordering::SeqCst),
        "the claim was held until it was let go"
    );
}

#[tokio::test]
async fn a_renewal_that_failed_once_and_then_succeeded_keeps_the_claim() {
    let (coordinator, _, backed_up) =
        backed_up_slowly(|knobs| knobs.renewals_failing = AtomicUsize::new(1)).await;
    backed_up.unwrap();
    assert_eq!(coordinator.renewals_failing.load(Ordering::SeqCst), 0);
    assert!(!coordinator.lapsed.load(Ordering::SeqCst));
}

/// Whether a backup that `knobs` make lose its claim after `renewed`
/// renewals, each a third of a hold after the one before, stopped a third
/// of a hold before the claim would run out after the last of them, and so
/// before it did.
async fn stops_in_time(renewed: u32, knobs: impl FnOnce(&mut Unrenewing)) {
    let (coordinator, took, backed_up) = backed_up_slowly(knobs).await;
    assert!(
        matches!(backed_up, Err(NodeError::ClaimLost(_))),
        "{backed_up:?}"
    );
    let last_asked = hold() / 3 * renewed;
    took_about(
        took,
        last_asked + hold() - hold() / 3,
        "it stops at once, with a margin",
    );
    assert!(
        !coordinator.lapsed.load(Ordering::SeqCst),
        "the claim was let go before it ran out"
    );
}

#[tokio::test]
async fn a_backup_whose_renewals_go_unanswered_stops_before_its_claim_runs_out() {
    stops_in_time(0, |knobs| {
        // The renewal asked for again after the failure would still be
        // waited for past the time to stop, were it waited for in full.
        knobs.renewals_failing = AtomicUsize::new(1);
        knobs.renewals_answered = AtomicUsize::new(0);
    })
    .await;
}

#[tokio::test]
async fn a_backup_whose_renewals_go_unanswered_after_some_were_stops_by_the_last() {
    stops_in_time(2, |knobs| knobs.renewals_answered = AtomicUsize::new(2)).await;
}

#[tokio::test]
async fn a_backup_whose_renewals_keep_failing_stops_before_its_claim_runs_out() {
    stops_in_time(0, |knobs| {
        knobs.renewals_failing = AtomicUsize::new(usize::MAX)
    })
    .await;
}

#[tokio::test]
async fn a_backup_whose_renewal_is_refused_stops_at_once() {
    let (_, took, backed_up) =
        backed_up_slowly(|knobs| knobs.renewals_refused = AtomicBool::new(true)).await;
    assert!(
        matches!(backed_up, Err(NodeError::ClaimLost(_))),
        "{backed_up:?}"
    );
    took_about(took, hold() / 3, "at the first renewal");
}

#[tokio::test]
async fn a_backup_whose_claim_is_taken_over_stops_where_it_is() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        disowning: AtomicBool::new(true),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let slowly = Arc::new(SlowStore::new(hold() * 2));
    let node = backing_up_through(&cluster, coordinator.clone(), slowly).await;
    tokio::time::pause();
    let started = tokio::time::Instant::now();
    let stopped = node.back_up(&tenant("acme")).await;
    assert!(
        matches!(stopped, Err(NodeError::ClaimLost(_))),
        "{stopped:?}"
    );
    took_about(
        started.elapsed(),
        hold() / 3,
        "at the first renewal, which found it gone",
    );
    assert_eq!(node.backups(&tenant("acme")).await.unwrap(), []);
    assert!(unclaimed(&cluster.coordinator, &tenant("acme")).await);
}

#[tokio::test]
async fn pruning_stops_once_a_renewal_finds_the_claim_lost() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(Arc::new(SlowStore::new(Duration::ZERO))),
            backups_kept: Duration::ZERO,
            ..cluster.settings()
        },
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    let first = node.back_up(&tenant("acme")).await.unwrap();
    coordinator.disowning.store(true, Ordering::SeqCst);
    let second = node.back_up(&tenant("acme")).await.unwrap();
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap(),
        [first, second],
        "the older backup is not pruned without the claim"
    );
}

#[test]
fn a_grace_below_fifteen_minutes_or_three_requests_to_the_store_is_refused() {
    let seconds = Duration::from_secs;
    // A backup claim long enough for every store here, so that nothing
    // but the grace is refused.
    let graced = |grace, store| {
        Settings {
            grace,
            backup_claim: seconds(6 * 60 * 60),
            request_bounds: RequestBounds {
                store,
                coordinator: seconds(10),
            },
            ..Settings::new(NodeId::generate("node"))
        }
        .check()
    };
    let refusal = |checked: Result<(), NodeError>| match checked {
        Err(NodeError::Settings(problem)) => problem,
        other => panic!("not refused for its settings: {other:?}"),
    };
    let just_under = |floor: Duration| floor - Duration::from_millis(1);
    let unset = Settings::new(NodeId::generate("node"));
    assert_eq!(unset.grace, seconds(60 * 60));
    unset.check().unwrap();

    // As a node is set up unless told otherwise, three requests to the
    // store take less than fifteen minutes, and fifteen minutes it is: to
    // the millisecond.
    let floor = vellev_node::LEAST_GRACE;
    assert_eq!(floor, seconds(15 * 60));
    let store = RequestBounds::default().store;
    assert_eq!(store, seconds(187));
    assert_eq!(RequestBounds::default().least_grace(), floor);
    graced(floor, store).unwrap();
    graced(floor + seconds(1), store).unwrap();
    let refused = refusal(graced(just_under(floor), store));
    assert!(refused.contains("a grace of 899.999s"), "{refused}");
    assert!(refused.contains("at least 900s"), "{refused}");
    assert!(refusal(graced(floor - seconds(1), store)).contains("at least 900s"));
    assert!(refusal(graced(Duration::ZERO, store)).contains("at least 900s"));
    // However quick the store, no less.
    graced(floor, Duration::ZERO).unwrap();
    assert!(refusal(graced(just_under(floor), Duration::ZERO)).contains("at least 900s"));

    // A store given longer for a request moves the floor up with it, to
    // three times as long.
    graced(floor, seconds(300)).unwrap();
    assert!(refusal(graced(just_under(floor), seconds(300))).contains("at least 900s"));
    graced(seconds(903), seconds(301)).unwrap();
    let refused = refusal(graced(just_under(seconds(903)), seconds(301)));
    assert!(refused.contains("at least 903s"), "{refused}");
    assert!(refused.contains("object storage, 301s"), "{refused}");
    graced(seconds(1_200), seconds(400)).unwrap();
    let refused = refusal(graced(just_under(seconds(1_200)), seconds(400)));
    assert!(refused.contains("a grace of 1199.999s"), "{refused}");
    assert!(refused.contains("at least 1200s"), "{refused}");
    assert!(refusal(graced(floor, seconds(400))).contains("at least 1200s"));
    assert!(
        refusal(graced(seconds(60 * 60), seconds(1_201))).contains("at least 3603s"),
        "the grace a node has unless told otherwise is refused too, under a store that slow"
    );

    // With a backup claim that cannot be held either, it is the grace
    // that is spoken of, as by the binary.
    let both = Settings {
        grace: Duration::ZERO,
        backup_claim: seconds(60),
        ..Settings::new(NodeId::generate("node"))
    };
    let claim = both.request_bounds.claim_problem(both.backup_claim);
    assert!(claim.unwrap().contains("a backup claim of 60s"));
    let refused = refusal(both.check());
    assert!(refused.contains("a grace of 0ns"), "{refused}");
    assert!(!refused.contains("backup claim"), "{refused}");
}

#[test]
fn a_backup_claim_must_outlast_a_request_and_leave_time_to_renew_it() {
    let seconds = Duration::from_secs;
    let claimed = |backup_claim, store, coordinator| {
        Settings {
            backup_claim,
            request_bounds: RequestBounds { store, coordinator },
            ..Settings::new(NodeId::generate("node"))
        }
        .check()
    };
    let refused = |checked: Result<(), NodeError>| matches!(checked, Err(NodeError::Settings(_)));
    Settings::new(NodeId::generate("node")).check().unwrap();

    // Its margin, a third of it, must outlast a request to the store and
    // what the store is given to apply one the node gave up: to the
    // millisecond, with nothing to spare at the least.
    let landed = vellev_node::LANDED_AFTER;
    assert_eq!(landed, seconds(30));
    claimed(seconds(720), seconds(210), seconds(10)).unwrap();
    let just_under = seconds(720) - Duration::from_millis(3);
    assert!(refused(claimed(just_under, seconds(210), seconds(10))));
    assert!(refused(claimed(seconds(630), seconds(210), seconds(10))));
    // However quick the store, it is given that long.
    claimed(seconds(90), Duration::ZERO, Duration::from_millis(100)).unwrap();
    assert!(refused(claimed(
        Duration::from_millis(89_997),
        Duration::ZERO,
        Duration::from_millis(100)
    )));
    // The rest must leave time for every attempt to take or renew it:
    // three of a minute, and the waits between them.
    claimed(seconds(272), seconds(1), seconds(60)).unwrap();
    assert!(refused(claimed(seconds(270), seconds(1), seconds(60))));
    assert!(refused(claimed(
        MAX_BACKUP_HOLD + seconds(3),
        seconds(1),
        seconds(1)
    )));
}

/// A backup of acme, written to, in a store of its own that another
/// cluster made.
async fn backed_up_elsewhere() -> (Arc<SlowStore>, Backup) {
    let elsewhere = Cluster::new();
    let store = Arc::new(SlowStore::new(Duration::ZERO));
    let node = elsewhere.node_from(Settings {
        backups: Some(store.clone()),
        ..elsewhere.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    for _ in 0..2 {
        node.issue_key(&tenant("acme")).await.unwrap().unwrap();
    }
    let window = Some(Duration::from_secs(2 * 60 * 60));
    assert!(node
        .set_restore_window(&tenant("acme"), window)
        .await
        .unwrap());
    let backup = node.back_up(&tenant("acme")).await.unwrap();
    assert_eq!(backup.restore_window, window);
    (store, backup)
}

/// A node of `cluster` restoring from `backups` through `coordinator`,
/// under claims a third of a second long.
fn restoring_node(
    cluster: &Cluster,
    coordinator: Arc<Unrenewing>,
    backups: Arc<SlowStore>,
) -> Node {
    Node::new(
        coordinator,
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(backups),
            backup_claim: Duration::from_millis(300),
            ..cluster.settings()
        },
    )
}

/// That `kept`, the only backup of acme in `store`, a restore of which
/// `node` of `cluster` just failed, survives daily backups and pruning for
/// days, files and all, and restores once `hidden`, a file the restore
/// lacked, if any, is back.
async fn outlives_a_failed_restore(
    cluster: &Cluster,
    node: &Node,
    store: &Arc<SlowStore>,
    kept: &Backup,
    hidden: Option<(String, Bytes)>,
) {
    node.back_up_due().await.unwrap();
    for _ in 0..3 {
        cluster.clock.advance(Duration::from_secs(25 * 60 * 60));
        node.back_up_due().await.unwrap();
    }
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap(),
        std::slice::from_ref(kept),
        "no backup is taken of the tenant the restore left empty"
    );
    if let Some((hidden, bytes)) = hidden {
        store.put(&hidden, bytes).await.unwrap();
    }
    for file in kept.files() {
        let file = format!("{}{file}", files_of(kept));
        store.get(&file).await.unwrap();
    }
    node.adopt_backup(&tenant("acme"), kept.id, true, RequestId::generate())
        .await
        .unwrap();
    let opened = node.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(opened.read(count), 1);
    let keys = cluster.coordinator.keys(&tenant("acme")).await.unwrap();
    assert_eq!(
        key_ids(&keys),
        key_ids(&kept.keys),
        "asked again, it has the keys it had"
    );
}

/// A backup of acme elsewhere, with one of its files taken out of the
/// store, and what was taken out.
async fn backed_up_elsewhere_lacking_a_file() -> (Arc<SlowStore>, Backup, (String, Bytes)) {
    let (store, kept) = backed_up_elsewhere().await;
    let hidden = format!("{}{}", files_of(&kept), kept.files().next().unwrap());
    let bytes = store.get(&hidden).await.unwrap();
    store.delete(&hidden).await.unwrap();
    (store, kept, (hidden, bytes))
}

#[tokio::test]
async fn a_backup_outlives_a_restore_that_failed_and_the_days_after_it() {
    let (store, kept, hidden) = backed_up_elsewhere_lacking_a_file().await;
    let cluster = Cluster::new();
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        backups_kept: Duration::ZERO,
        ..cluster.settings()
    });
    let failed = node
        .adopt_backup(&tenant("acme"), kept.id, true, RequestId::generate())
        .await;
    assert!(
        matches!(
            failed,
            Err(NodeError::Storage(StorageError::BackupLacks { .. }))
        ),
        "{failed:?}"
    );
    outlives_a_failed_restore(&cluster, &node, &store, &kept, Some(hidden)).await;
}

#[tokio::test]
async fn a_backup_outlives_a_restore_the_coordinator_failed() {
    let (store, kept) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store.clone()),
            backups_kept: Duration::ZERO,
            ..cluster.settings()
        },
    );
    // Everything is fetched and the adoption noted, and the commit that
    // was to make it does not land: nothing of the backup is the tenant's.
    coordinator.commits_failing.store(true, Ordering::SeqCst);
    let failed = node
        .adopt_backup(&tenant("acme"), kept.id, true, RequestId::generate())
        .await;
    assert!(failed.is_err(), "{failed:?}");
    coordinator.commits_failing.store(false, Ordering::SeqCst);
    let now = cluster.coordinator.identity(&tenant("acme")).await.unwrap();
    let now = now.unwrap();
    assert!(!now.committed && now.lineage != kept.lineage, "{now:?}");
    assert_eq!(cluster.coordinator.keys(&tenant("acme")).await.unwrap(), []);
    let window = cluster.coordinator.restore_window(&tenant("acme")).await;
    assert_eq!(window.unwrap(), None);
    outlives_a_failed_restore(&cluster, &node, &store, &kept, None).await;
}

fn acme() -> PartitionKey {
    PartitionKey {
        tenant: tenant("acme"),
        partition: Partition::DEFAULT,
    }
}

#[tokio::test]
async fn a_restore_whose_claim_is_lost_while_it_fetches_files_stops_before_its_commit() {
    let (store, backup) = backed_up_elsewhere().await;
    *store.file_reads.lock().unwrap() = Duration::from_secs(2);
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        disowning: AtomicBool::new(true),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = restoring_node(&cluster, coordinator, store);
    let stopped = node
        .adopt_backup(&tenant("acme"), backup.id, true, RequestId::generate())
        .await;
    assert!(
        matches!(stopped, Err(NodeError::ClaimLost(_))),
        "{stopped:?}"
    );
    let committed = cluster.coordinator.committed(&acme()).await.unwrap();
    assert_eq!(
        committed.and_then(|committed| committed.manifest),
        None,
        "nothing was committed"
    );
    let probe = uuid::Uuid::new_v4();
    let day = Some(Duration::from_secs(24 * 60 * 60));
    assert!(
        cluster
            .coordinator
            .claim_backup(&tenant("acme"), probe, day, LEASE)
            .await
            .unwrap(),
        "a restore that failed is not noted as a backup of the tenant's own data"
    );
}

#[tokio::test]
async fn a_restore_whose_claim_runs_out_as_its_last_file_arrives_makes_no_commit() {
    let (store, backup) = backed_up_elsewhere().await;
    store.stalls.store(true, Ordering::SeqCst);
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = restoring_node(&cluster, coordinator, store.clone());
    let stopped = node
        .adopt_backup(&tenant("acme"), backup.id, true, RequestId::generate())
        .await;
    assert!(!store.stalls.load(Ordering::SeqCst), "the fetch stalled");
    assert!(
        matches!(stopped, Err(NodeError::ClaimLost(_))),
        "{stopped:?}"
    );
    let committed = cluster.coordinator.committed(&acme()).await.unwrap();
    assert_eq!(
        committed.and_then(|committed| committed.manifest),
        None,
        "nothing was committed"
    );
}

#[tokio::test]
async fn a_request_to_the_backup_store_is_given_up_once_it_took_as_long_as_one_may() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let bound = Duration::from_secs(2);
    let slowly: Arc<dyn Blobs> = Arc::new(SlowStore::new(bound * 30));
    backing_up_through(&cluster, coordinator, slowly.clone()).await;
    let node = cluster.node_from(Settings {
        backups: Some(slowly),
        request_bounds: RequestBounds {
            store: bound,
            ..RequestBounds::default()
        },
        ..cluster.settings()
    });
    tokio::time::pause();
    let started = tokio::time::Instant::now();
    let given_up = node.back_up(&tenant("acme")).await;
    assert!(
        matches!(given_up, Err(NodeError::Storage(StorageError::Blob { .. }))),
        "{given_up:?}"
    );
    took_about(started.elapsed(), bound, "the first write is given up");
    assert!(unclaimed(&cluster.coordinator, &tenant("acme")).await);
}

#[tokio::test]
async fn a_restore_whose_claim_is_lost_once_its_commit_began_runs_to_its_end() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        disowning_on_commit: AtomicBool::new(true),
        commits_slowly: Duration::from_secs(1),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = restoring_node(&cluster, coordinator.clone(), store);
    let restored = node
        .adopt_backup(&tenant("acme"), backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(restored.restore.from, backup.position());
    let position = restored.position;
    assert!(
        coordinator.disowning.load(Ordering::SeqCst),
        "the claim was lost while the commit ran"
    );
    let committed = cluster
        .coordinator
        .committed(&acme())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(committed.log_position, position);
    let opened = node.tenant(&tenant("acme")).await.unwrap();
    assert_eq!(opened.read(count), 1);
}

/// The keys under `prefix` in `store`.
async fn listed(store: &Arc<dyn Blobs>, prefix: &str) -> Vec<String> {
    let listed = store.list(prefix).await.unwrap();
    listed.into_iter().map(|listed| listed.key).collect()
}

/// What backing up acme, written to, leaves in the backup store when the
/// tenant is removed, and with `made_again` made again, while it copies.
async fn backed_up_while_removed(made_again: bool) -> (Cluster, Node, Arc<dyn Blobs>) {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        made_again,
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let store: Arc<dyn Blobs> = Arc::new(SlowStore::new(Duration::ZERO));
    let node = backing_up_through(&cluster, coordinator.clone(), store.clone()).await;
    coordinator.removing.store(true, Ordering::SeqCst);
    let refused = node.back_up(&tenant("acme")).await;
    assert!(
        matches!(refused, Err(NodeError::UnknownTenant(_))),
        "{refused:?}"
    );
    assert_eq!(node.backups(&tenant("acme")).await.unwrap(), []);
    let left = listed(&store, "_backups/acme/").await;
    assert!(
        !left.is_empty() && left.iter().all(|key| key.contains("/data/")),
        "its record is forgotten, and its files left where they lie: {left:?}"
    );
    (cluster, node, store)
}

#[tokio::test]
async fn a_backup_of_a_tenant_made_again_while_it_copied_leaves_its_files_to_the_look_over() {
    let (cluster, node, store) = backed_up_while_removed(true).await;
    assert!(unclaimed(&cluster.coordinator, &tenant("acme")).await);

    // The tenant made since is backed up beside what was left, and the
    // look over the one purged takes that and nothing of this one's.
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(2, "hen", None)])
        .await
        .unwrap();
    let own = node.back_up(&tenant("acme")).await.unwrap();
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    let left = listed(&store, "_backups/acme/").await;
    assert!(
        left.iter().all(|key| kept_with(&own, key)),
        "only its own is left: {left:?}"
    );
    assert_eq!(node.backups(&tenant("acme")).await.unwrap(), [own]);
}

#[tokio::test]
async fn a_backup_of_a_tenant_removed_while_it_copied_leaves_its_files_to_the_purge() {
    let (cluster, node, store) = backed_up_while_removed(false).await;
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    assert_eq!(listed(&store, "_backups/acme/").await, [] as [String; 0]);
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
    let coordinator = Arc::new(Unrenewing::slow(
        cluster.coordinator.clone(),
        Duration::ZERO,
    ));
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
/// A backup store that runs a hook once just before the first backed up
/// file lands, its bytes read already, as a put in flight is, and another
/// once just before the first delete.
#[derive(Debug)]
struct Hooked {
    inner: ObjectBlobs,
    before_file: Hook,
    before_delete: Hook,
}

impl Hooked {
    fn new() -> Self {
        Self {
            inner: ObjectBlobs::new(Arc::new(InMemory::new())),
            before_file: Hook::default(),
            before_delete: Hook::default(),
        }
    }
}

#[async_trait]
impl Blobs for Hooked {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.contains("/data/") {
            self.before_file.run().await;
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.before_delete.run().await;
        self.inner.delete(key).await
    }
}

/// What is left in the backup store of a node reading tenants through
/// `blobs`, of acme: removed, and purged by another node once a backup
/// claim has run out, all while the node has the first file of a backup of
/// it on the way to the backup store, a stall that long its own clock does
/// not see. Returns the node and the backup store too.
async fn left_by_a_backup_outliving_the_purge(
    cluster: &Cluster,
    blobs: Arc<dyn Blobs>,
) -> (Node, Arc<dyn Blobs>, Vec<String>) {
    let store = Arc::new(Hooked::new());
    let settings = || Settings {
        backups: Some(store.clone()),
        ..cluster.settings()
    };
    let node = Node::new(
        cluster.coordinator.clone(),
        blobs,
        Arc::new(Embedders::new(EmbedSettings::default())),
        settings(),
    );
    let other = cluster.node_from(settings());
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    acme.write(vec![note(2, "hen", None)]).await.unwrap();
    drop(acme);

    let clock = cluster.clock.clone();
    store.before_file.set(async move {
        other.drop_tenant(&tenant("acme")).await.unwrap();
        clock.advance(hold());
        other.purge_due().await.unwrap();
    });
    let backed_up = node.back_up(&tenant("acme")).await;
    assert!(!store.before_file.is_set(), "the purge ran mid-backup");
    assert!(backed_up.is_err(), "{backed_up:?}");
    assert!(
        cluster.store.list("acme/").await.unwrap().is_empty(),
        "the tenant's files were purged"
    );
    let store: Arc<dyn Blobs> = store;
    let left = listed(&store, "_backups/acme/").await;
    (node, store, left)
}

/// That a pass of `node`'s purging deletes what a backup left of acme in
/// `store` after its purge, and that acme can then be made again.
async fn is_purged_again(cluster: &Cluster, node: &Node, store: &Arc<dyn Blobs>) {
    node.purge_due().await.unwrap();
    assert!(
        !listed(store, "_backups/acme/").await.is_empty(),
        "purged a moment ago, it is not looked over yet"
    );
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(
        listed(store, "_backups/acme/").await,
        [] as [String; 0],
        "removing a tenant removes everything kept of it"
    );
    assert!(cluster.store.list("acme/").await.unwrap().is_empty());
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_backup_that_outlives_its_tenants_purge_leaves_nothing_for_long() {
    let cluster = Cluster::new();
    let (node, store, left) =
        left_by_a_backup_outliving_the_purge(&cluster, cluster.store.clone()).await;
    assert_eq!(
        left.len(),
        1,
        "the file on its way when the purge ran: {left:?}"
    );
    is_purged_again(&cluster, &node, &store).await;
}

#[tokio::test]
async fn a_backup_fed_by_its_nodes_cache_after_its_tenants_purge_leaves_nothing_for_long() {
    let cluster = Cluster::new();
    let cache = tempfile::tempdir().unwrap();
    let cached = Arc::new(CachedBlobs::new(cluster.store.clone(), cache.path()));
    let (node, store, left) = left_by_a_backup_outliving_the_purge(&cluster, cached).await;
    assert!(
        left.len() > 1,
        "every file, read from the cache though the store had lost it: {left:?}"
    );
    is_purged_again(&cluster, &node, &store).await;
}

#[tokio::test]
async fn a_purge_is_done_only_once_it_finds_nothing_left() {
    let cluster = Cluster::new();
    let store = Arc::new(Hooked::new());
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(1, "fox", None)])
        .await
        .unwrap();
    drop(acme);
    let backup = node.back_up(&tenant("acme")).await.unwrap();
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    // A file lands while the purge deletes what it found.
    let late = (store.clone(), format!("{}late", files_of(&backup)));
    store.before_delete.set(async move {
        late.0
            .inner
            .put(&late.1, Bytes::from("late"))
            .await
            .unwrap();
    });
    node.purge_due().await.unwrap();
    assert!(!store.before_delete.is_set(), "a file landed mid-purge");
    let store: Arc<dyn Blobs> = store;
    assert_eq!(listed(&store, "_backups/acme/").await, [] as [String; 0]);
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_tenant_is_not_purged_while_a_backup_of_it_could_hold_its_claim() {
    let cluster = Cluster::new();
    let store: Arc<dyn Blobs> = Arc::new(Hooked::new());
    let node = cluster.node_from(Settings {
        backups: Some(store),
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    node.drop_tenant(&tenant("acme")).await.unwrap();

    cluster.clock.advance(hold() - Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(
        node.create_tenant(&tenant("acme")).await.is_err(),
        "not purged yet"
    );
    cluster.clock.advance(Duration::from_secs(1));
    node.purge_due().await.unwrap();
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_purged_tenant_is_looked_over_again_now_and_then_while_it_is_not_made_again() {
    let cluster = Cluster::new();
    let node = cluster.node();
    for name in ["gone", "back"] {
        node.create_tenant(&tenant(name)).await.unwrap();
        node.drop_tenant(&tenant(name)).await.unwrap();
    }
    node.purge_due().await.unwrap();
    node.create_tenant(&tenant("back")).await.unwrap();
    node.purge_due().await.unwrap();

    for name in ["gone", "back"] {
        cluster
            .store
            .put(&format!("{name}/0/log/late"), Bytes::from("late"))
            .await
            .unwrap();
    }
    let left = || async {
        let mut left = Vec::new();
        for name in ["gone", "back"] {
            let files = cluster.store.list(&format!("{name}/")).await.unwrap();
            left.push(!files.is_empty());
        }
        left
    };
    node.purge_due().await.unwrap();
    assert_eq!(left().await, [true, true], "looked over a moment ago");

    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(
        left().await,
        [false, true],
        "what is left of the tenant made again is its own"
    );
    node.purge_due().await.unwrap();
    node.create_tenant(&tenant("gone")).await.unwrap();
}

/// A store of backups, in memory.
fn backup_store() -> Arc<dyn Blobs> {
    Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))
}

/// A node of `cluster` backing tenants up to `store`.
fn backing_up_to(cluster: &Cluster, store: Arc<dyn Blobs>) -> Node {
    cluster.node_from(Settings {
        backups: Some(store),
        ..cluster.settings()
    })
}

/// Note `n` written, as `title`, to acme as it is now on `node`, by a
/// schema made with it.
async fn noted(node: &Node, n: u64, title: &str) {
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(n, title, None)])
        .await
        .unwrap();
}

/// Which of notes 1 and 2 acme holds on `node`.
async fn notes_held(node: &Node) -> [bool; 2] {
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.read(|state| [1, 2].map(|n| state.get(&notes(), id(n)).is_some()))
}

#[tokio::test]
async fn a_tenant_made_again_under_a_name_neither_sees_restores_nor_inherits_the_one_before() {
    let cluster = Cluster::new();
    let store = backup_store();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    // The acme before: one customer's.
    node.create_tenant(&acme).await.unwrap();
    noted(&node, 1, "old secret").await;
    let old_key = node.issue_key(&acme).await.unwrap().unwrap();
    let old = node.back_up(&acme).await.unwrap();
    // What a backup's writes, given up on or sent by a guest then paused,
    // land later: all of it, at worst.
    let mut late = Vec::new();
    for key in listed(&store, "_backups/acme/").await {
        late.push((key.clone(), store.get(&key).await.unwrap()));
    }
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    assert_eq!(listed(&store, "_backups/acme/").await, [] as [String; 0]);

    // The acme after: another customer's.
    node.create_tenant(&acme).await.unwrap();
    for (key, bytes) in late {
        store.put(&key, bytes).await.unwrap();
    }
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&old),
        "never written to, it is shown what whoever runs the cluster may have it take up"
    );
    let refused = node
        .restore_backup(&acme, old.id, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "nor restored: {refused:?}"
    );
    assert_eq!(notes_held(&node).await, [false, false]);
    assert_eq!(node.keys(&acme).await.unwrap(), [], "nor are its keys");
    assert_eq!(node.key_tenant(&old_key.secret).await.unwrap(), None);

    // Written to, it is not taken for the one before whoever asks, and is
    // shown none but its own.
    noted(&node, 2, "new").await;
    assert_eq!(node.backups(&acme).await.unwrap(), []);
    let refused = node
        .adopt_backup(&acme, old.id, true, RequestId::generate())
        .await;
    assert!(matches!(refused, Err(NodeError::NotLost(_))), "{refused:?}");
    assert_eq!(notes_held(&node).await, [false, true]);
    assert_eq!(node.keys(&acme).await.unwrap(), []);
    let own = node.back_up(&acme).await.unwrap();
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&own)
    );

    // And what landed late of the one before goes when it is looked over,
    // with nothing of this one's.
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    let left = listed(&store, "_backups/acme/").await;
    assert!(left.iter().all(|key| kept_with(&own, key)), "{left:?}");
    node.restore_backup(&acme, own.id, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(notes_held(&node).await, [false, true]);
}

#[tokio::test]
async fn a_cluster_that_lost_a_tenant_takes_it_up_again_from_a_backup_only_when_asked_to_adopt() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    let identity = || async { cluster.coordinator.identity(&acme).await.unwrap() };
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&backup),
        "what is kept under the name of a tenant it does not have"
    );
    let refused = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownTenant(_))),
        "{refused:?}"
    );
    let refused = node
        .adopt_backup(&acme, uuid::Uuid::new_v4(), true, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "{refused:?}"
    );
    assert_eq!(identity().await, None, "no tenant is made unasked");

    let adopted = node
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(adopted.restore.from, backup.position());
    let taken_up = identity().await.unwrap();
    assert_eq!(taken_up.lineage, backup.lineage);
    assert_eq!(notes_held(&node).await, [true, false]);
    let keys = node.keys(&acme).await.unwrap();
    assert_eq!(key_ids(&keys), key_ids(&backup.keys));
    assert_eq!(
        cluster.coordinator.restore_window(&acme).await.unwrap(),
        backup.restore_window
    );

    // It goes on as the tenant it was: it backs up under its lineage and
    // this cluster's id, and what either cluster took of it is its own.
    let next = node.back_up(&acme).await.unwrap();
    assert_eq!(next.lineage, backup.lineage);
    assert_ne!(next.cluster, backup.cluster);
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        [backup.clone(), next.clone()]
    );
    node.restore_backup(&acme, backup.id, RequestId::generate())
        .await
        .unwrap();
    // Written to, it takes nothing up again, its own or not: a restore of
    // its own keeps the keys it has now.
    let refused = node
        .adopt_backup(&acme, next.id, true, RequestId::generate())
        .await;
    assert!(matches!(refused, Err(NodeError::NotLost(_))), "{refused:?}");
}

#[tokio::test]
async fn a_tenant_never_written_to_takes_up_a_backup_when_asked_and_not_while_its_name_is_purged() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    // A tenant of the name here, another's, removed: its purge takes
    // nothing of the backup, which is of another lineage.
    node.create_tenant(&acme).await.unwrap();
    node.drop_tenant(&acme).await.unwrap();
    let refused = node
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await;
    let Err(NodeError::Purging { wait, .. }) = refused else {
        panic!("{refused:?}");
    };
    assert_eq!(
        wait,
        hold() + cluster.settings().purges_looked_for_every,
        "until its purge is due, and a node next looks for purges"
    );
    cluster.clock.advance(hold() / 3);
    let refused = node.create_tenant(&acme).await;
    let Err(NodeError::Purging { wait, .. }) = refused else {
        panic!("{refused:?}");
    };
    assert_eq!(
        wait,
        hold() - hold() / 3 + cluster.settings().purges_looked_for_every
    );
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&backup)
    );

    // Made, and never written to, it is shown what it may take up, and
    // takes it up only when asked to adopt it.
    node.create_tenant(&acme).await.unwrap();
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&backup)
    );
    let refused = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "{refused:?}"
    );
    assert_eq!(node.keys(&acme).await.unwrap(), []);
    node.adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(notes_held(&node).await, [true, false]);
    assert_eq!(
        key_ids(&node.keys(&acme).await.unwrap()),
        key_ids(&backup.keys)
    );
    // Looked over again, it is the tenant purged no longer: nothing goes.
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(node.backups(&acme).await.unwrap(), [backup]);
}

/// A written acme of a cluster of its own, with one key, backed up to
/// `store`: the node, the key and the backup.
async fn another_acme_backed_up_to(
    store: &Arc<dyn Blobs>,
    n: u64,
    title: &str,
) -> (Node, vellev_node::Issued, Backup) {
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(&node, n, title).await;
    let key = node.issue_key(&tenant("acme")).await.unwrap().unwrap();
    let backup = node.back_up(&tenant("acme")).await.unwrap();
    (node, key, backup)
}

#[tokio::test]
async fn a_tenant_that_takes_up_another_backup_after_one_failed_has_nothing_of_the_first() {
    let store = backup_store();
    let (_, _, first) = another_acme_backed_up_to(&store, 1, "customer one").await;
    let (_, _, second) = another_acme_backed_up_to(&store, 2, "customer two").await;
    assert_ne!(first.lineage, second.lineage);
    // The first lacks a file: adopting it fails once all else is fetched.
    let hidden = format!("{}{}", files_of(&first), first.files().next().unwrap());
    store.delete(&hidden).await.unwrap();

    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    let failed = node
        .adopt_backup(&acme, first.id, true, RequestId::generate())
        .await;
    assert!(
        matches!(
            failed,
            Err(NodeError::Storage(StorageError::BackupLacks { .. }))
        ),
        "{failed:?}"
    );
    let made = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert!(!made.committed, "{made:?}");
    assert!(![first.lineage, second.lineage].contains(&made.lineage));
    assert_eq!(node.keys(&acme).await.unwrap(), [], "no key of the first");
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        [first.clone(), second.clone()],
        "it is shown what it may still take up"
    );

    node.adopt_backup(&acme, second.id, true, RequestId::generate())
        .await
        .unwrap();
    let now = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert_eq!(now.lineage, second.lineage);
    assert_eq!(notes_held(&node).await, [false, true]);
    assert_eq!(
        key_ids(&node.keys(&acme).await.unwrap()),
        key_ids(&second.keys),
        "the keys of the one it took up, and of no other"
    );
}

#[tokio::test]
async fn removing_a_tenant_whose_adoption_failed_leaves_the_backups_it_was_to_take_up() {
    let (store, kept, _) = backed_up_elsewhere_lacking_a_file().await;
    let cluster = Cluster::new();
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        ..cluster.settings()
    });
    let acme = tenant("acme");
    assert!(node
        .adopt_backup(&acme, kept.id, true, RequestId::generate())
        .await
        .is_err());
    let before = keys_under(&store.inner, "_backups/acme/").await;

    // Removed, to start over: it is purged as the empty tenant it was.
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(keys_under(&store.inner, "_backups/acme/").await, before);
    assert_eq!(node.backups(&acme).await.unwrap(), [kept]);
}

#[tokio::test]
async fn a_cluster_that_adopted_a_tenant_and_removes_it_deletes_nothing_another_cluster_took() {
    let store = backup_store();
    let (live, _, taken_live) = another_acme_backed_up_to(&store, 1, "live").await;
    let theirs = listed(&store, "_backups/acme/").await;
    // A drill: another cluster, given the live one's backup store, adopts
    // the tenant, backs it up, and removes it again.
    let drill = Cluster::new();
    let node = backing_up_to(&drill, store.clone());
    let acme = tenant("acme");
    node.adopt_backup(&acme, taken_live.id, false, RequestId::generate())
        .await
        .unwrap();
    let acme_open = node.tenant(&acme).await.unwrap();
    acme_open.write(vec![note(2, "drill", None)]).await.unwrap();
    drop(acme_open);
    let taken_drill = node.back_up(&acme).await.unwrap();
    assert_eq!(taken_drill.lineage, taken_live.lineage);
    let all = listed(&store, "_backups/acme/").await;
    let ours = all.iter().filter(|key| !theirs.contains(key));
    assert!(ours.clone().count() > 1);
    assert!(
        ours.clone().all(|key| kept_with(&taken_drill, key)),
        "what it takes lies under its own id"
    );
    assert_eq!(
        live.backups(&acme).await.unwrap(),
        [taken_live.clone(), taken_drill]
    );

    node.drop_tenant(&acme).await.unwrap();
    drill.clock.advance(hold());
    node.purge_due().await.unwrap();
    drill.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(
        listed(&store, "_backups/acme/").await,
        theirs,
        "its own went, and nothing of the cluster's it adopted the tenant from"
    );
    assert_eq!(live.backups(&acme).await.unwrap(), [taken_live]);
}

#[tokio::test]
async fn a_backup_gives_the_tenant_that_adopts_it_its_keys_only_if_asked() {
    let store = backup_store();
    let (live, kept_key, backup) = another_acme_backed_up_to(&store, 1, "fox").await;
    let acme = tenant("acme");
    // Revoked since the backup was taken, which kept it.
    assert!(live.revoke_key(&acme, kept_key.key.id).await.unwrap());
    assert_eq!(key_ids(&backup.keys), [kept_key.key.id]);

    let without = backing_up_to(&Cluster::new(), store.clone());
    without
        .adopt_backup(&acme, backup.id, false, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(notes_held(&without).await, [true, false]);
    assert_eq!(without.keys(&acme).await.unwrap(), [], "none unasked");
    assert_eq!(without.key_tenant(&kept_key.secret).await.unwrap(), None);

    let with = backing_up_to(&Cluster::new(), store.clone());
    with.adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(key_ids(&with.keys(&acme).await.unwrap()), [kept_key.key.id]);
    assert_eq!(
        with.key_tenant(&kept_key.secret).await.unwrap(),
        Some(acme),
        "asked for, the keys the backup kept reach the tenant, one revoked since too"
    );
}

#[tokio::test]
async fn a_tenant_with_keys_of_its_own_takes_no_backup_up_until_they_are_revoked() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    let own = node.issue_key(&acme).await.unwrap().unwrap();
    for with_keys in [false, true] {
        let refused = node
            .adopt_backup(&acme, backup.id, with_keys, RequestId::generate())
            .await;
        assert!(matches!(refused, Err(NodeError::Keyed(_))), "{refused:?}");
    }
    assert_eq!(notes_held(&node).await, [false, false]);
    assert_eq!(key_ids(&node.keys(&acme).await.unwrap()), [own.key.id]);

    assert!(node.revoke_key(&acme, own.key.id).await.unwrap());
    node.adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(
        key_ids(&node.keys(&acme).await.unwrap()),
        key_ids(&backup.keys),
        "exactly the keys asked for"
    );
    assert_eq!(node.key_tenant(&own.secret).await.unwrap(), None);
}

#[tokio::test]
async fn a_record_that_says_no_lineage_is_no_tenants_to_list_restore_take_up_or_purge() {
    let store = backup_store();
    let (_, _, backup) = another_acme_backed_up_to(&store, 1, "customer one").await;
    // As a release before tenants had lineages kept it: under the name
    // alone, saying neither lineage nor cluster.
    let mut before = serde_json::to_value(&backup).unwrap();
    for unsaid in ["lineage", "cluster"] {
        before.as_object_mut().unwrap().remove(unsaid);
    }
    for key in listed(&store, "_backups/acme/").await {
        store.delete(&key).await.unwrap();
    }
    let record = "_backups/acme/backups/00000000000000000001-before.backup.json";
    let before = Bytes::from(serde_json::to_vec(&before).unwrap());
    store.put(record, before).await.unwrap();

    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    assert_eq!(node.backups(&acme).await.unwrap(), []);
    let refused = node
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "{refused:?}"
    );
    node.create_tenant(&acme).await.unwrap();
    noted(&node, 2, "customer two").await;
    assert_eq!(node.backups(&acme).await.unwrap(), []);
    let refused = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await;
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "{refused:?}"
    );
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    assert_eq!(listed(&store, "_backups/acme/").await, [record]);
}

/// The keys under `prefix` in `store`.
async fn keys_under(store: &dyn Blobs, prefix: &str) -> Vec<String> {
    let listed = store.list(prefix).await.unwrap();
    listed.into_iter().map(|listed| listed.key).collect()
}

/// Acme made on `node`, written to in a segment and since, and backed up.
async fn written_and_backed_up(node: &Node) -> Backup {
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(node, 1, "fox").await;
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.flush().await.unwrap();
    acme.write(vec![note(2, "hen", None)]).await.unwrap();
    drop(acme);
    node.back_up(&tenant("acme")).await.unwrap()
}

/// That a node purging acme which stalls, unseen by its own clock, past
/// its claim as it deletes from `stalled`, its tenants' store or its backup
/// store, while another purges the tenant and it is made again, written to
/// and backed up, deletes nothing of the tenant made again once it goes
/// on, and lists nothing more.
async fn outlasts_its_claim_stalled_in(stalled: &str) {
    let cluster = Cluster::new();
    let store = backup_store();
    let files = Listing::of(cluster.store.clone(), Duration::ZERO);
    let backups = Listing::of(store.clone(), Duration::ZERO);
    let node = Node::new(
        cluster.coordinator.clone(),
        files.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(backups.clone()),
            ..cluster.settings()
        },
    );
    let other = backing_up_to(&cluster, store.clone());
    written_and_backed_up(&node).await;
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    let made = Arc::new(std::sync::Mutex::new(None));
    let (clock, primary, kept_in) = (cluster.clock.clone(), cluster.store.clone(), store.clone());
    let (noting, listings) = (made.clone(), (files.clone(), backups.clone()));
    let deleting = match stalled {
        "files" => &files.deleting,
        _ => &backups.deleting,
    };
    deleting.set(async move {
        clock.advance(hold() + Duration::from_secs(1));
        other.purge_due().await.unwrap();
        written_and_backed_up(&other).await;
        other.shutdown().await;
        let live = keys_under(primary.as_ref(), "acme/").await;
        let kept = keys_under(kept_in.as_ref(), "_backups/acme/").await;
        *noting.lock().unwrap() = Some((live, kept));
        // What it listed before it stalled is no matter.
        for listing in [listings.0, listings.1] {
            listing.listed();
        }
    });
    node.purge_due().await.unwrap();
    assert!(!deleting.is_set(), "{stalled}: the purge stalled");
    let (live, kept) = made.lock().unwrap().take().unwrap();
    assert!(!live.is_empty() && !kept.is_empty());
    assert_eq!(
        keys_under(cluster.store.as_ref(), "acme/").await,
        live,
        "{stalled}: none of the files of the tenant made again went"
    );
    assert_eq!(
        keys_under(store.as_ref(), "_backups/acme/").await,
        kept,
        "{stalled}: nor any of its backups"
    );
    assert_eq!(
        (files.listed(), backups.listed()),
        (Vec::new(), Vec::new()),
        "{stalled}: its claim lost, the purge looked for nothing more to delete"
    );
    assert_eq!(notes_held(&cluster.node()).await, [true, true]);
}

#[tokio::test]
async fn a_purge_that_outlasts_its_claim_deletes_nothing_of_the_tenant_made_again() {
    outlasts_its_claim_stalled_in("files").await;
    outlasts_its_claim_stalled_in("backups").await;
}

#[tokio::test]
async fn a_look_over_stops_deleting_a_lineage_once_the_tenant_of_its_name_takes_it_up() {
    let cluster = Cluster::new();
    let store = Arc::new(Hooked::new());
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        ..cluster.settings()
    });
    let old = written_and_backed_up(&node).await;
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    assert!(keys_under(&store.inner, "_backups/acme/").await.is_empty());

    // A tenant of the name is made, and more than a batch of files lands
    // late for the one purged; as the first of them is deleted, the tenant
    // takes up the purged one's lineage, as one adopting its backup does.
    node.create_tenant(&tenant("acme")).await.unwrap();
    for n in 0..DELETED_AT_ONCE + 4 {
        let late = format!("{}late/{n}", files_of(&old));
        store.inner.put(&late, Bytes::from("late")).await.unwrap();
    }
    cluster.clock.advance(LOOKED_OVER_EVERY);
    let (coordinator, lineage) = (cluster.coordinator.clone(), old.lineage);
    store.before_delete.set(async move {
        let holder = NodeId::generate("adopting");
        let leased = coordinator.acquire(&acme(), &holder, LEASE).await.unwrap();
        let Acquired::Lease { lease, .. } = leased else {
            panic!("{leased:?}");
        };
        let adoption = Adoption {
            lineage,
            at: LogPosition(1),
            keys: Vec::new(),
            restore_window: None,
        };
        let key = acme();
        let noted = coordinator.adopt_on_commit(&key, lease.generation, &adoption);
        assert!(noted.await.unwrap());
        let restored = Commit {
            after: None,
            log_position: LogPosition(1),
            manifest: Some(vellev_storage::coordinator::ManifestRef("restored".into())),
            restore: None,
            adopting: Some(lineage),
        };
        coordinator.commit(&lease, restored).await.unwrap();
        coordinator.release(&lease).await.unwrap();
    });
    node.purge_due().await.unwrap();
    assert!(!store.before_delete.is_set(), "the lineage was looked over");
    assert_eq!(
        keys_under(&store.inner, "_backups/acme/").await.len(),
        4,
        "the batch it was told it might delete went, and nothing after"
    );
}

#[tokio::test]
async fn a_purge_whose_machine_sleeps_past_its_claim_deletes_nothing_more_until_claimed_again() {
    let cluster = Cluster::new();
    let store = Arc::new(Hooked::new());
    let claims = Arc::new(RuntimeClock::new());
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        claim_clock: claims.clone(),
        ..cluster.settings()
    });
    written_and_backed_up(&node).await;
    let kept = keys_under(&store.inner, "_backups/acme/").await.len();
    assert!(kept > 2);
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    // Suspended as its first delete in the backup store is on its way: by
    // the claim's clock, woken, the claim is past when it is taken for
    // lost, though the coordinator would renew it still.
    store
        .before_delete
        .set(async move { claims.sleep_through(hold()) });
    node.purge_due().await.unwrap();
    assert!(!store.before_delete.is_set(), "the machine slept");
    assert_eq!(
        keys_under(&store.inner, "_backups/acme/").await.len(),
        kept - 1,
        "the delete on its way when it slept, and no other"
    );
    assert!(
        node.create_tenant(&tenant("acme")).await.is_err(),
        "its purge is not done"
    );

    // The claim, lost to the node, is not let go with the coordinator,
    // which holds it still: a delete was on its way when the machine
    // slept, and may land yet. Nobody purges the tenant until the claim
    // has run out, which is no sooner than that delete can land.
    node.purge_due().await.unwrap();
    assert_eq!(
        keys_under(&store.inner, "_backups/acme/").await.len(),
        kept - 1,
        "not taken up again at once"
    );
    assert!(node.create_tenant(&tenant("acme")).await.is_err());
    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(keys_under(&store.inner, "_backups/acme/").await.is_empty());
    node.create_tenant(&tenant("acme")).await.unwrap();
}

/// A node backing up to `store`, which it waits a day for, through
/// `through`, which asks `coordinator`, under claims `backup_claim` long;
/// and acme, made and written to by another node since stopped, backed up
/// and removed a default claim ago, with the runtime's clock paused, which
/// is the coordinator's and the claims'.
async fn about_to_purge(
    coordinator: Arc<MemoryCoordinator>,
    through: Arc<Unrenewing>,
    store: Arc<SlowStore>,
    backup_claim: Duration,
) -> Node {
    let cluster = Cluster::new();
    let plainly = Arc::new(Unrenewing::new(coordinator));
    let before = backing_up_through(&cluster, plainly, store.clone()).await;
    before.back_up(&tenant("acme")).await.unwrap();
    before.drop_tenant(&tenant("acme")).await.unwrap();
    tokio::time::pause();
    tokio::time::advance(hold()).await;
    Node::new(
        through,
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store),
            backup_claim,
            request_bounds: RequestBounds {
                store: Duration::from_secs(24 * 60 * 60),
                ..RequestBounds::default()
            },
            ..cluster.settings()
        },
    )
}

#[tokio::test]
async fn a_purge_slower_than_its_claim_holds_it_throughout_and_is_done() {
    let coordinator = ticking();
    let store = Arc::new(SlowStore::new(Duration::ZERO));
    let through = Arc::new(Unrenewing::new(coordinator.clone()));
    let purging = about_to_purge(coordinator.clone(), through.clone(), store.clone(), hold());
    let node = purging.await;
    *store.deletes.lock().unwrap() = hold();

    // Another node asks to purge it all the while, and never may.
    let lapsed = Arc::new(AtomicBool::new(false));
    let asking = tokio::spawn({
        let (coordinator, lapsed) = (coordinator.clone(), lapsed.clone());
        async move {
            loop {
                tokio::time::sleep(hold() / 10).await;
                let acme = tenant("acme");
                let claimed = coordinator.claim_purge(&acme, Duration::ZERO, hold());
                if claimed.await.unwrap().is_some() {
                    lapsed.store(true, Ordering::SeqCst);
                }
            }
        }
    });
    let started = tokio::time::Instant::now();
    node.purge_due().await.unwrap();
    asking.abort();
    assert!(started.elapsed() >= hold(), "{:?}", started.elapsed());
    assert!(through.renewals.load(Ordering::SeqCst) >= 3);
    assert!(
        !lapsed.load(Ordering::SeqCst),
        "the claim was held until the purge was done"
    );
    assert!(keys_under(&store.inner, "_backups/acme/").await.is_empty());
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_purge_whose_claim_was_answered_slowly_and_is_not_renewed_stops_by_when_it_asked() {
    let coordinator = ticking();
    let store = Arc::new(SlowStore::new(Duration::ZERO));
    let through = Arc::new(Unrenewing {
        claims_slowly: Duration::from_secs(5),
        renewals_answered: AtomicUsize::new(0),
        ..Unrenewing::new(coordinator.clone())
    });
    // A claim so short that asking to renew it, again and again, outlasts
    // it: its renewal before it lists anything is waited for until the
    // claim is taken for lost, a margin before it runs out from when it
    // was asked for, not from when that was answered five seconds on.
    let short = Duration::from_secs(30);
    let node = about_to_purge(coordinator, through, store.clone(), short).await;
    let kept = keys_under(&store.inner, "_backups/acme/").await.len();
    let started = tokio::time::Instant::now();
    node.purge_due().await.unwrap();
    took_about(
        started.elapsed(),
        short - short / 3,
        "it stops a margin before the claim runs out",
    );
    assert_eq!(
        keys_under(&store.inner, "_backups/acme/").await.len(),
        kept,
        "nothing was deleted without the claim"
    );
}

#[tokio::test]
async fn a_claim_answered_slowly_is_held_from_when_it_was_asked_for() {
    stops_in_time(0, |knobs| {
        knobs.claims_slowly = Duration::from_secs(5);
        knobs.renewals_answered = AtomicUsize::new(0);
    })
    .await;
}

#[tokio::test]
async fn a_claim_whose_renewal_is_answered_slowly_is_held_from_when_that_was_asked_for() {
    stops_in_time(1, |knobs| {
        knobs.claim_renewals_slowly = Duration::from_secs(5);
        knobs.renewals_answered = AtomicUsize::new(1);
    })
    .await;
}

#[tokio::test]
async fn a_claim_renewed_slowly_before_a_delete_is_held_from_when_that_was_asked_for() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        claim_renewals_slowly: Duration::from_secs(5),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let store = Arc::new(SlowStore::new(Duration::ZERO));
    backing_up_through(&cluster, coordinator.clone(), store.clone()).await;
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store.clone()),
            backups_kept: Duration::ZERO,
            request_bounds: RequestBounds {
                store: Duration::from_secs(24 * 60 * 60),
                ..RequestBounds::default()
            },
            ..cluster.settings()
        },
    );
    let first = node.back_up(&tenant("acme")).await.unwrap();

    // The next backup prunes the first: the claim is renewed before the
    // delete, answered five seconds after it was asked, and never again;
    // and the delete takes longer than the claim lasts.
    tokio::time::pause();
    *store.deletes.lock().unwrap() = hold() * 2;
    coordinator.renewals_answered.store(1, Ordering::SeqCst);
    let started = tokio::time::Instant::now();
    let second = node.back_up(&tenant("acme")).await.unwrap();
    took_about(
        started.elapsed(),
        hold() - hold() / 3,
        "pruning stops a margin before the claim, renewed when asked, runs out",
    );
    *store.deletes.lock().unwrap() = Duration::ZERO;
    assert_eq!(
        node.backups(&tenant("acme")).await.unwrap(),
        [first, second]
    );
}

/// A store one kind of request to which, once told, takes `long`.
#[derive(Debug)]
struct Unanswering {
    inner: Arc<dyn Blobs>,
    request: &'static str,
    long: Duration,
    told: AtomicBool,
}

impl Unanswering {
    fn to(request: &'static str, inner: Arc<dyn Blobs>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            request,
            long: hold() * 10,
            told: AtomicBool::new(false),
        })
    }

    async fn asked(&self, request: &str) {
        if request == self.request && self.told.load(Ordering::SeqCst) {
            tokio::time::sleep(self.long).await;
        }
    }
}

#[async_trait]
impl Blobs for Unanswering {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.asked("put").await;
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.asked("get").await;
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.asked("list").await;
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.asked("delete").await;
        self.inner.delete(key).await
    }
}

/// How long a node waits for object storage in the tests that have it give
/// a request up.
const GIVEN_UP: Duration = Duration::from_secs(2);

/// The settings of a node of `cluster` that gives a request to object
/// storage up after [`GIVEN_UP`], backing up to `backups`.
fn giving_up(cluster: &Cluster, backups: Arc<dyn Blobs>) -> Settings {
    Settings {
        backups: Some(backups),
        backups_kept: Duration::ZERO,
        request_bounds: RequestBounds {
            store: GIVEN_UP,
            ..RequestBounds::default()
        },
        ..cluster.settings()
    }
}

/// Acme made on a node of `cluster` and written to, by a node since
/// stopped, so that the runtime's clock may be paused.
async fn written_by_a_node_since_stopped(cluster: &Cluster) {
    let writing = cluster.node();
    writing.create_tenant(&tenant("acme")).await.unwrap();
    noted(&writing, 1, "fox").await;
    writing.shutdown().await;
}

fn unanswered<T: std::fmt::Debug>(asked: &Result<T, NodeError>) -> bool {
    matches!(asked, Err(NodeError::Storage(StorageError::Blob { .. })))
}

#[tokio::test]
async fn a_read_a_listing_and_a_delete_in_the_backup_store_are_each_given_up_as_a_write_is() {
    for request in ["get", "list", "delete"] {
        let cluster = Cluster::new();
        written_by_a_node_since_stopped(&cluster).await;
        let store = Unanswering::to(request, backup_store());
        let node = cluster.node_from(giving_up(&cluster, store.clone()));
        node.back_up(&tenant("acme")).await.unwrap();

        tokio::time::pause();
        store.told.store(true, Ordering::SeqCst);
        let started = tokio::time::Instant::now();
        if request == "delete" {
            // Two deletes, each given up: of the pending record, and of
            // the first backup's, pruned.
            node.back_up(&tenant("acme")).await.unwrap();
            took_about(started.elapsed(), GIVEN_UP * 2, request);
        } else {
            let listed = node.backups(&tenant("acme")).await;
            assert!(unanswered(&listed), "{request}: {listed:?}");
            took_about(started.elapsed(), GIVEN_UP, request);
        }
        tokio::time::resume();
    }
}

#[tokio::test]
async fn a_purge_gives_a_request_to_the_tenants_store_up_as_one_to_the_backup_store() {
    for request in ["list", "delete"] {
        let cluster = Cluster::new();
        written_by_a_node_since_stopped(&cluster).await;
        let primary = Unanswering::to(request, cluster.store.clone());
        let node = Node::new(
            cluster.coordinator.clone(),
            primary.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            giving_up(&cluster, backup_store()),
        );
        node.drop_tenant(&tenant("acme")).await.unwrap();
        cluster.clock.advance(hold());

        tokio::time::pause();
        primary.told.store(true, Ordering::SeqCst);
        let started = tokio::time::Instant::now();
        node.purge_due().await.unwrap();
        took_about(started.elapsed(), GIVEN_UP, request);
        assert!(
            !cluster.store.list("acme/").await.unwrap().is_empty(),
            "{request}: the purge failed, and is left to a later pass"
        );
        tokio::time::resume();
    }
}

#[tokio::test]
async fn a_restore_gives_a_write_to_the_tenants_store_up_as_one_to_the_backup_store() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let primary = Arc::new(Unanswering {
        inner: cluster.store.clone(),
        request: "put",
        long: Duration::from_secs(5),
        told: AtomicBool::new(true),
    });
    let node = Node::new(
        cluster.coordinator.clone(),
        primary,
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            request_bounds: RequestBounds {
                store: Duration::from_millis(100),
                ..RequestBounds::default()
            },
            ..giving_up(&cluster, store)
        },
    );
    let given_up = node
        .adopt_backup(&tenant("acme"), backup.id, true, RequestId::generate())
        .await;
    assert!(unanswered(&given_up), "{given_up:?}");
    let committed = cluster.coordinator.committed(&acme()).await.unwrap();
    assert_eq!(
        committed.and_then(|committed| committed.manifest),
        None,
        "nothing was committed"
    );
}

#[tokio::test]
async fn a_restore_whose_machine_sleeps_as_its_adoption_is_noted_commits_and_takes_nothing() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let claims = Arc::new(RuntimeClock::new());
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store),
            claim_clock: claims.clone(),
            ..cluster.settings()
        },
    );
    let acme = tenant("acme");
    coordinator
        .adoption_noted
        .set(async move { claims.sleep_through(hold()) });
    let stopped = node
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await;
    assert!(!coordinator.adoption_noted.is_set(), "the machine slept");
    assert!(
        matches!(stopped, Err(NodeError::ClaimLost(_))),
        "{stopped:?}"
    );
    let own = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert!(!own.committed && own.lineage != backup.lineage, "{own:?}");
    assert_eq!(cluster.coordinator.keys(&acme).await.unwrap(), []);
    assert_eq!(
        cluster.coordinator.restore_window(&acme).await.unwrap(),
        None
    );

    // The adoption is given up with the restore: written to after, the
    // tenant is itself.
    noted(&node, 2, "its own").await;
    let now = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert_eq!((now.lineage, now.committed), (own.lineage, true));
    assert_eq!(cluster.coordinator.keys(&acme).await.unwrap(), []);
}

#[tokio::test]
async fn a_node_backs_tenants_up_and_purges_those_removed_now_and_then_once_told_to() {
    let cluster = Cluster::new();
    let often = Duration::from_millis(20);
    let node = cluster.node_from(Settings {
        backups: Some(backup_store()),
        backups_looked_for_every: often,
        purges_looked_for_every: often,
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(&node, 1, "fox").await;
    node.create_tenant(&tenant("gone")).await.unwrap();
    node.drop_tenant(&tenant("gone")).await.unwrap();
    cluster.clock.advance(hold());

    node.backing_up();
    node.purging();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let backed_up = node.backups(&tenant("acme")).await.unwrap().len() == 1;
        let purged = cluster
            .coordinator
            .purges(Duration::ZERO, REMEMBERED)
            .await
            .unwrap();
        if backed_up && purged.due.is_empty() && purged.lately == [tenant("gone")] {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "backed up: {backed_up}; purges: {purged:?}"
        );
        tokio::time::sleep(often).await;
    }
    node.shutdown().await;
}

#[tokio::test]
async fn a_listing_of_the_backup_store_is_given_up_by_how_long_an_answer_takes_not_the_whole() {
    let cluster = Cluster::new();
    written_by_a_node_since_stopped(&cluster).await;
    let slow = Arc::new(ThrottledStore::new(
        InMemory::new(),
        ThrottleConfig::default(),
    ));
    let node = cluster.node_from(Settings {
        backups_kept: Duration::from_secs(24 * 60 * 60),
        ..giving_up(&cluster, Arc::new(ObjectBlobs::new(slow.clone())))
    });
    for _ in 0..3 {
        node.back_up(&tenant("acme")).await.unwrap();
    }

    // Each of its three records is listed sooner than a request is given
    // up, and all of them later.
    tokio::time::pause();
    slow.config_mut(|config| config.wait_list_per_entry = GIVEN_UP * 2 / 3);
    let started = tokio::time::Instant::now();
    assert_eq!(node.backups(&tenant("acme")).await.unwrap().len(), 3);
    assert!(started.elapsed() >= GIVEN_UP * 2, "{:?}", started.elapsed());
    slow.config_mut(|config| config.wait_list_per_entry = GIVEN_UP * 2);
    let listed = node.backups(&tenant("acme")).await;
    assert!(unanswered(&listed), "{listed:?}");
}

/// Object storage that notes each listing asked of it, and takes `latency`
/// to answer one; and does what a test has it do before a delete.
#[derive(Debug)]
struct Listing {
    inner: Arc<dyn Blobs>,
    latency: Duration,
    listed: std::sync::Mutex<Vec<String>>,
    under_way: AtomicUsize,
    most_under_way: AtomicUsize,
    /// What happens once, just before the next delete.
    deleting: Hook,
}

impl Listing {
    fn of(inner: Arc<dyn Blobs>, latency: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner,
            latency,
            listed: std::sync::Mutex::default(),
            under_way: AtomicUsize::new(0),
            most_under_way: AtomicUsize::new(0),
            deleting: Hook::default(),
        })
    }

    /// The prefixes listed since last asked.
    fn listed(&self) -> Vec<String> {
        std::mem::take(&mut self.listed.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Listing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.listed.lock().unwrap().push(prefix.to_string());
        let under_way = self.under_way.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_under_way.fetch_max(under_way, Ordering::SeqCst);
        tokio::time::sleep(self.latency).await;
        self.under_way.fetch_sub(1, Ordering::SeqCst);
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.deleting.run().await;
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn thousands_of_purged_tenants_are_looked_over_once_each_between_nodes_and_many_together() {
    const PURGED: usize = 5_000;
    const LATENCY: Duration = Duration::from_millis(50);
    /// A listing of a tenant's files, and two of what is kept of it.
    const LISTINGS: usize = 3;
    let cluster = Cluster::new();
    let coordinator = &cluster.coordinator;
    let purged: Vec<PartitionKey> = (0..PURGED)
        .map(|n| PartitionKey {
            tenant: tenant(&format!("purged-{n:04}")),
            partition: Partition::DEFAULT,
        })
        .collect();
    for key in &purged {
        coordinator.register(key).await.unwrap();
        coordinator.remove(key).await.unwrap();
        let claimed = coordinator.claim_purge(&key.tenant, Duration::ZERO, LEASE);
        let claim = claimed.await.unwrap().unwrap();
        let done = coordinator.purged(&key.tenant, claim.id, REMEMBERED);
        done.await.unwrap();
    }
    let primary = Listing::of(cluster.store.clone(), LATENCY);
    let backups = Listing::of(backup_store(), LATENCY);
    let node = || {
        Node::new(
            cluster.coordinator.clone(),
            primary.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            Settings {
                backups: Some(backups.clone()),
                ..cluster.settings()
            },
        )
    };
    let (one, other) = (node(), node());
    tokio::time::pause();
    let pass = || async {
        let started = tokio::time::Instant::now();
        let (one, other) = tokio::join!(one.purge_due(), other.purge_due());
        one.and(other).unwrap();
        (started.elapsed(), primary.listed(), backups.listed().len())
    };

    let (took, ..) = pass().await;
    assert_eq!(took, Duration::ZERO, "purged a moment ago, none is due");
    cluster.clock.advance(LOOKED_OVER_EVERY);
    let (took, mut files, kept) = pass().await;
    files.sort();
    let each: Vec<String> = purged
        .iter()
        .map(|key| format!("{}/", key.tenant))
        .collect();
    assert_eq!(files, each, "each looked over by one node, once");
    assert_eq!(kept, PURGED * (LISTINGS - 1));
    let one_at_a_time = LATENCY * (PURGED * LISTINGS) as u32;
    let together = primary.most_under_way.load(Ordering::SeqCst);
    eprintln!(
        "{PURGED} purged tenants looked over by two nodes: {} listings in {took:?} at \
         {LATENCY:?} each, {together} tenants at once at most; {one_at_a_time:?} one at a time",
        PURGED * LISTINGS
    );
    assert!(took < one_at_a_time / 20, "{took:?}");
    assert!((2..=64).contains(&together), "{together}");

    let (took, files, kept) = pass().await;
    assert_eq!((took, files.len(), kept), (Duration::ZERO, 0, 0));
}

/// A tenants' store that does what a test has it do once, just after the
/// file under `key` is written.
#[derive(Debug)]
struct Written {
    inner: Arc<dyn Blobs>,
    key: String,
    then: Hook,
}

#[async_trait]
impl Blobs for Written {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await?;
        if key == self.key {
            self.then.run().await;
        }
        Ok(())
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_restore_whose_machine_sleeps_once_its_last_file_is_written_makes_no_commit() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let claims = Arc::new(RuntimeClock::new());
    let primary = Arc::new(Written {
        inner: cluster.store.clone(),
        key: backup.files().last().unwrap().clone(),
        then: Hook::default(),
    });
    let node = Node::new(
        cluster.coordinator.clone(),
        primary.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store),
            claim_clock: claims.clone(),
            ..cluster.settings()
        },
    );
    primary
        .then
        .set(async move { claims.sleep_through(hold()) });
    let stopped = node
        .adopt_backup(&tenant("acme"), backup.id, true, RequestId::generate())
        .await;
    assert!(!primary.then.is_set(), "the machine slept");
    assert!(
        matches!(stopped, Err(NodeError::ClaimLost(_))),
        "{stopped:?}"
    );
    let committed = cluster.coordinator.committed(&acme()).await.unwrap();
    assert_eq!(
        committed.and_then(|committed| committed.manifest),
        None,
        "nothing was committed"
    );
}

#[tokio::test]
async fn a_restore_asked_of_a_tenant_since_made_again_restores_nothing_into_the_new_one() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(backup_store()),
            ..cluster.settings()
        },
    );
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    noted(&node, 1, "the one before").await;
    let backup = node.back_up(&acme).await.unwrap();

    // The backup is found to be the tenant's own; before the restore holds
    // its claim the tenant is removed, purged and made again.
    let inner = cluster.coordinator.clone();
    coordinator.backup_claimed.set(async move {
        let key = self::acme();
        assert!(inner.remove(&key).await.unwrap());
        let purge = inner.claim_purge(&key.tenant, Duration::ZERO, LEASE).await;
        let purge = purge.unwrap().unwrap();
        inner
            .purged(&key.tenant, purge.id, REMEMBERED)
            .await
            .unwrap();
        inner.register(&key).await.unwrap();
    });
    let refused = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await;
    assert!(!coordinator.backup_claimed.is_set(), "it was made again");
    assert!(
        matches!(refused, Err(NodeError::UnknownBackup(..))),
        "{refused:?}"
    );
    let now = cluster.coordinator.committed(&self::acme()).await.unwrap();
    assert_eq!(now.and_then(|now| now.manifest), None, "nothing restored");
}

#[tokio::test]
async fn an_adoption_given_up_by_a_node_whose_lease_ran_out_leaves_anothers_to_be_made() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node_through = || {
        Node::new(
            coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            Settings {
                backups: Some(store.clone()),
                ..cluster.settings()
            },
        )
    };
    let (stalled, other) = (node_through(), node_through());
    let acme = tenant("acme");
    // The first node notes its adoption, then stalls past its lease and
    // its claim by the coordinator's clock. Meanwhile the other takes the
    // tenant, notes an adoption of its own under the new lease, and is
    // about to commit; its commit is slow to arrive, and lands only once
    // the first node, going on, has had its restore refused and given up
    // the adoption it noted. That is nothing to the other's.
    coordinator.adoptions_raced.store(true, Ordering::SeqCst);
    let (sending, adopting_elsewhere) = tokio::sync::oneshot::channel();
    let (clock, through) = (cluster.clock.clone(), coordinator.clone());
    let (elsewhere, taken) = (other.clone(), backup.id);
    coordinator.adoption_noted.set(async move {
        clock.advance(hold() + Duration::from_secs(1));
        let adopting = tokio::spawn(async move {
            let request = RequestId::generate();
            elsewhere
                .adopt_backup(&tenant("acme"), taken, true, request)
                .await
        });
        let second = through.second_adoption_noted.notified();
        tokio::time::timeout(Duration::from_secs(60), second)
            .await
            .expect("the other node notes its adoption");
        assert!(sending.send(adopting).is_ok());
    });
    let stale = stalled
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await;
    assert!(stale.is_err(), "{stale:?}");
    assert!(!coordinator.adoption_noted.is_set());
    let adopted = adopting_elsewhere.await.unwrap().await.unwrap().unwrap();
    assert_eq!(adopted.restore.from, backup.position());
    let now = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert!(now.committed, "{now:?}");
    assert_eq!(
        now.lineage, backup.lineage,
        "the tenant is the one the backup is of"
    );
    let keys = cluster.coordinator.keys(&acme).await.unwrap();
    assert_eq!(
        key_ids(&keys),
        key_ids(&backup.keys),
        "with the keys asked for"
    );
    assert_eq!(
        cluster.coordinator.restore_window(&acme).await.unwrap(),
        backup.restore_window,
        "and its window"
    );
    assert_eq!(notes_held(&other).await, [true, false]);
}

#[tokio::test]
async fn a_request_to_adopt_made_again_the_other_way_about_keys_is_told_it_is_reused() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    let request = RequestId::generate();
    let adopted = node
        .adopt_backup(&acme, backup.id, false, request)
        .await
        .unwrap();
    let again = node.adopt_backup(&acme, backup.id, false, request).await;
    assert_eq!(again.unwrap(), adopted, "as it was");
    let other_way = node.adopt_backup(&acme, backup.id, true, request).await;
    assert!(
        matches!(other_way, Err(NodeError::Reused { .. })),
        "{other_way:?}"
    );
    assert_eq!(node.keys(&acme).await.unwrap(), [], "and it gives no key");
}

#[tokio::test]
async fn a_purge_forgets_a_tenants_history_only_under_a_claim_renewed_just_before() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = Node::new(
        coordinator.clone(),
        cluster.store.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        cluster.settings(),
    );
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(&node, 1, "fox").await;
    let files = keys_under(cluster.store.as_ref(), "acme/").await.len();
    assert!(files > 0);
    node.drop_tenant(&tenant("acme")).await.unwrap();

    // The purge renews its claim before it lists the tenant's files, before
    // each batch it deletes, before it lists again and finds none, and
    // then before the first batch of history it forgets: that renewal is
    // refused, and nothing is forgotten.
    let before_forgetting = 2 + files.div_ceil(DELETED_AT_ONCE);
    coordinator.renewals.store(0, Ordering::SeqCst);
    *coordinator.renewals_refused_after.lock().unwrap() = Some(before_forgetting);
    node.purge_due().await.unwrap();
    assert_eq!(
        coordinator.renewals.load(Ordering::SeqCst),
        before_forgetting + 1,
        "the renewal just before forgetting was asked for and refused"
    );
    assert_eq!(coordinator.forgets.load(Ordering::SeqCst), 0);
    assert!(
        node.create_tenant(&tenant("acme")).await.is_err(),
        "the purge is not done"
    );
    assert_eq!(
        coordinator.purge_releases.load(Ordering::SeqCst),
        0,
        "deletes were sent under the claim: it is left to run out"
    );

    // Renewed, once the claim has run out and is taken again, it forgets,
    // and the purge is done.
    *coordinator.renewals_refused_after.lock().unwrap() = None;
    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(coordinator.forgets.load(Ordering::SeqCst) > 0);
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_look_over_of_a_tenant_that_took_its_purged_lineage_up_again_lists_nothing() {
    let cluster = Cluster::new();
    let backups = Listing::of(backup_store(), Duration::ZERO);
    let node = cluster.node_from(Settings {
        backups: Some(backups.clone()),
        ..cluster.settings()
    });
    let old = written_and_backed_up(&node).await;
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());
    node.purge_due().await.unwrap();
    assert!(keys_under(backups.inner.as_ref(), "_backups/acme/")
        .await
        .is_empty());

    // A tenant of the name is made and takes the purged lineage up, as one
    // adopting a backup of it another cluster took would; then a file
    // lands late under that lineage, as one of a backup this cluster began
    // before the purge might. It is the tenant's own now, and the look
    // over leaves it be without so much as a listing.
    node.create_tenant(&tenant("acme")).await.unwrap();
    let holder = NodeId::generate("adopting");
    let leased = cluster
        .coordinator
        .acquire(&acme(), &holder, LEASE)
        .await
        .unwrap();
    let Acquired::Lease { lease, .. } = leased else {
        panic!("{leased:?}");
    };
    let adoption = Adoption {
        lineage: old.lineage,
        at: LogPosition(1),
        keys: Vec::new(),
        restore_window: None,
    };
    let key = acme();
    let noted = cluster
        .coordinator
        .adopt_on_commit(&key, lease.generation, &adoption);
    assert!(noted.await.unwrap());
    let restored = Commit {
        after: None,
        log_position: LogPosition(1),
        manifest: Some(vellev_storage::coordinator::ManifestRef("restored".into())),
        restore: None,
        adopting: Some(old.lineage),
    };
    cluster.coordinator.commit(&lease, restored).await.unwrap();
    cluster.coordinator.release(&lease).await.unwrap();
    let late = format!("{}late", files_of(&old));
    backups.inner.put(&late, Bytes::from("late")).await.unwrap();
    backups.listed();

    cluster.clock.advance(LOOKED_OVER_EVERY);
    node.purge_due().await.unwrap();
    assert_eq!(backups.listed(), [] as [String; 0]);
    assert_eq!(
        keys_under(backups.inner.as_ref(), "_backups/acme/").await,
        [late]
    );
}

#[tokio::test]
async fn a_tenant_restored_or_taken_up_is_as_due_a_backup_as_before() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let acme = tenant("acme");
    node.adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        std::slice::from_ref(&backup)
    );

    // Taken up, the tenant has no backup by this cluster: it is backed up
    // the next time backups due are looked for, not a day on.
    node.back_up_due().await.unwrap();
    let kept = node.backups(&acme).await.unwrap();
    assert_eq!(kept.len(), 2, "{kept:?}");
    let own = &kept[1];
    assert_ne!(own.cluster, backup.cluster);

    // Restored from one of its own just before its next backup is due, it
    // is due it still: a restore is no backup.
    let every = Settings::new(NodeId::generate("node")).backup_every;
    cluster.clock.advance(every - Duration::from_secs(60));
    node.restore_backup(&acme, own.id, RequestId::generate())
        .await
        .unwrap();
    cluster.clock.advance(Duration::from_secs(61));
    node.back_up_due().await.unwrap();
    assert_eq!(node.backups(&acme).await.unwrap().len(), 3);
}

/// A database made for one test, dropped with it however the test ends:
/// from a runtime of its own on another thread, as the test's own cannot
/// be blocked on from within it.
struct ScratchDatabase {
    admin: String,
    name: String,
}

impl ScratchDatabase {
    /// One made through `admin`, the URL of a database that is there.
    async fn made_through(admin: &str) -> Self {
        let name = format!(
            "vellev_scratch_{}_{}",
            Utc::now().timestamp(),
            uuid::Uuid::new_v4().simple()
        );
        let making = sqlx::PgPool::connect(admin).await.unwrap();
        sqlx::query(&format!("CREATE DATABASE {name}"))
            .execute(&making)
            .await
            .unwrap();
        making.close().await;
        Self {
            admin: admin.to_string(),
            name,
        }
    }

    fn url(&self) -> String {
        let (head, _) = self.admin.rsplit_once('/').unwrap();
        format!("{head}/{}", self.name)
    }
}

impl Drop for ScratchDatabase {
    fn drop(&mut self) {
        let (admin, name) = (self.admin.clone(), self.name.clone());
        let dropped = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime to drop the database with");
            runtime.block_on(async {
                let admin = sqlx::PgPool::connect(&admin).await?;
                sqlx::query(&format!("DROP DATABASE IF EXISTS {name} WITH (FORCE)"))
                    .execute(&admin)
                    .await
                    .map(|_| ())
            })
        })
        .join();
        if !matches!(dropped, Ok(Ok(()))) {
            eprintln!("could not drop the scratch database {}", self.name);
        }
    }
}

/// The adoption path end to end on the Postgres coordinator, as the node
/// tests otherwise run it on the memory one alone: adopt with keys, the
/// request made again answered as it was and the other way refused,
/// back up under the lineage, restore its own, refuse a second adoption,
/// and purge only what this cluster took. Skipped without
/// `VELLEV_TEST_DATABASE_URL`, on a scratch database of its own.
#[tokio::test]
async fn adoption_runs_end_to_end_on_postgres() {
    let Ok(url) = std::env::var("VELLEV_TEST_DATABASE_URL") else {
        eprintln!("VELLEV_TEST_DATABASE_URL is not set; skipping a Postgres node test");
        return;
    };
    // Dropped with the test, passed or failed, once all that follows is.
    let database = ScratchDatabase::made_through(&url).await;
    let scratch = database.url();
    let (store, backup) = backed_up_elsewhere().await;
    let coordinator = Arc::new(
        vellev_storage::coordinator::PostgresCoordinator::connect(
            &scratch,
            8,
            Duration::from_secs(5),
        )
        .await
        .unwrap(),
    );
    let node = Node::new(
        coordinator.clone(),
        Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store.clone()),
            lease: LEASE,
            restore_window: Duration::ZERO,
            grace: Duration::ZERO,
            purge_after: Duration::ZERO,
            ..Settings::new(NodeId::generate("pg"))
        },
    );
    let acme = tenant("acme");
    let request = RequestId::generate();
    let adopted = node
        .adopt_backup(&acme, backup.id, true, request)
        .await
        .unwrap();
    assert_eq!(adopted.restore.from, backup.position());
    let now = coordinator.identity(&acme).await.unwrap().unwrap();
    assert_eq!((now.lineage, now.committed), (backup.lineage, true));
    let keys = node.keys(&acme).await.unwrap();
    assert_eq!(key_ids(&keys), key_ids(&backup.keys));
    assert_eq!(
        coordinator.restore_window(&acme).await.unwrap(),
        backup.restore_window
    );
    assert_eq!(notes_held(&node).await, [true, false]);

    // Asked again: as it was; the other way: reused.
    let again = node.adopt_backup(&acme, backup.id, true, request).await;
    assert_eq!(again.unwrap(), adopted);
    let other = node.adopt_backup(&acme, backup.id, false, request).await;
    assert!(matches!(other, Err(NodeError::Reused { .. })), "{other:?}");

    // A second adoption is refused; its own backups and restores work.
    let refused = node
        .adopt_backup(&acme, backup.id, false, RequestId::generate())
        .await;
    assert!(matches!(refused, Err(NodeError::NotLost(_))), "{refused:?}");
    let next = node.back_up(&acme).await.unwrap();
    assert_eq!(next.lineage, backup.lineage);
    assert_ne!(next.cluster, backup.cluster);
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        [backup.clone(), next.clone()]
    );
    noted(&node, 2, "later").await;
    node.restore_backup(&acme, backup.id, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(notes_held(&node).await, [true, false]);

    // Removed and purged, once the purge is due: what this cluster took
    // goes, and what the other took stays.
    node.drop_tenant(&acme).await.unwrap();
    let on_scratch = sqlx::PgPool::connect(&scratch).await.unwrap();
    sqlx::query("UPDATE vellev.purges SET requested_at = requested_at - interval '1 day'")
        .execute(&on_scratch)
        .await
        .unwrap();
    on_scratch.close().await;
    node.purge_due().await.unwrap();
    let left = node.backups(&acme).await.unwrap();
    assert_eq!(left, std::slice::from_ref(&backup), "{left:?}");
    node.shutdown().await;
}

#[tokio::test]
async fn a_node_whose_lease_went_before_it_noted_its_adoption_notes_nothing_for_the_next() {
    let (store, backup) = backed_up_elsewhere().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node_through = || {
        Node::new(
            coordinator.clone(),
            cluster.store.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            Settings {
                backups: Some(store.clone()),
                ..cluster.settings()
            },
        )
    };
    let fetching = Arc::new(Fetching {
        inner: cluster.store.clone(),
        before_put: Hook::default(),
    });
    let stalled = Node::new(
        coordinator.clone(),
        fetching.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(store.clone()),
            ..cluster.settings()
        },
    );
    let other = node_through();
    let acme = tenant("acme");
    // The node stalls past its lease as it fetches the backup's files, with
    // the adoption still to be noted; another node takes the tenant
    // meanwhile. The adoption is asked for under the lease the node holds,
    // which is no more, and so refused: not under the lease current now,
    // which is the other node's and asked for nothing.
    let (clock, elsewhere) = (cluster.clock.clone(), other.clone());
    fetching.before_put.set(async move {
        clock.advance(LEASE + Duration::from_secs(1));
        drop(elsewhere.tenant(&tenant("acme")).await.unwrap());
    });
    let lost = stalled
        .adopt_backup(&acme, backup.id, true, RequestId::generate())
        .await;
    assert!(!fetching.before_put.is_set(), "the machine slept");
    assert!(matches!(lost, Err(NodeError::Lost(_))), "{lost:?}");

    // The other node writes the tenant as the tenant it is: nothing noted
    // stands in its way, and nothing of the backup is taken.
    noted(&other, 1, "its own").await;
    let now = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert!(now.committed && now.lineage != backup.lineage, "{now:?}");
    assert_eq!(cluster.coordinator.keys(&acme).await.unwrap(), []);
}

/// A tenant's store that does what a test has it do once, just before the
/// next write into it: as a restore fetching a backup's files back makes.
#[derive(Debug)]
struct Fetching {
    inner: Arc<Store>,
    before_put: Hook,
}

#[async_trait]
impl Blobs for Fetching {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.before_put.run().await;
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_restore_says_when_what_it_went_back_to_was_as_its_source_says_on_a_clock_however_fine() {
    let cluster = Cluster::new();
    // A clock to the nanosecond, as a Linux machine's is.
    cluster.clock.advance(Duration::from_nanos(123_456_789));
    let node = backing_up_to(&cluster, backup_store());
    let acme = tenant("acme");
    restorable(&node, &acme).await;
    let (_, points) = node.restore_points(&acme).await.unwrap().unwrap();
    let (_, latest) = points.unwrap();
    let request = RequestId::generate();
    let restored = node
        .restore(&acme, Point::Position(latest.position), request)
        .await
        .unwrap();
    assert_eq!(
        (restored.restore.from, restored.restore.from_at),
        (latest.position, latest.at),
        "the time the restore point says"
    );
    let again = node
        .restore(&acme, Point::Position(latest.position), request)
        .await
        .unwrap();
    assert_eq!(again, restored, "and the same asked again, from the record");

    let backup = node.back_up(&acme).await.unwrap();
    let from_backup = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(from_backup.restore.from_at, backup.taken_at);
}

/// A tenants' store that goes wrong as a test has it: the delete of one
/// file given up unanswered, once, and remembered, to be landed late; a file
/// written with every delete, as by a node still writing; listings that
/// fail; a delete that never answers, once it is said to have been sent,
/// and remembered, to be landed late; a file that goes just after the next
/// listing; and one that goes as soon as it is written, every time. It
/// remembers every key it was given.
#[derive(Debug)]
struct Troubled {
    inner: Arc<Store>,
    giving_up_on: std::sync::Mutex<Option<String>>,
    given_up: std::sync::Mutex<Vec<String>>,
    refilling: AtomicBool,
    refilled: AtomicUsize,
    lists_fail: AtomicBool,
    deletes_hang: AtomicBool,
    delete_sent: tokio::sync::Notify,
    hung: std::sync::Mutex<Vec<String>>,
    lost_after_listing: std::sync::Mutex<Option<String>>,
    lost_as_written: std::sync::Mutex<Option<String>>,
    put: std::sync::Mutex<Vec<String>>,
}

impl Troubled {
    fn over(inner: Arc<Store>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            giving_up_on: std::sync::Mutex::new(None),
            given_up: std::sync::Mutex::default(),
            refilling: AtomicBool::new(false),
            refilled: AtomicUsize::new(0),
            lists_fail: AtomicBool::new(false),
            deletes_hang: AtomicBool::new(false),
            delete_sent: tokio::sync::Notify::new(),
            hung: std::sync::Mutex::default(),
            lost_after_listing: std::sync::Mutex::new(None),
            lost_as_written: std::sync::Mutex::new(None),
            put: std::sync::Mutex::default(),
        })
    }
}

#[async_trait]
impl Blobs for Troubled {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.put.lock().unwrap().push(key.to_string());
        self.inner.put(key, bytes).await?;
        let lost = self.lost_as_written.lock().unwrap().as_deref() == Some(key);
        if lost {
            self.inner.delete(key).await?;
        }
        Ok(())
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        if self.lists_fail.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: prefix.to_string(),
                source: "connection reset".into(),
            });
        }
        let listed = self.inner.list(prefix).await?;
        let lost = self.lost_after_listing.lock().unwrap().take();
        if let Some(lost) = lost {
            self.inner.delete(&lost).await?;
        }
        Ok(listed)
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        let given_up = {
            let mut giving_up_on = self.giving_up_on.lock().unwrap();
            let it = giving_up_on.as_deref() == Some(key);
            it.then(|| giving_up_on.take()).is_some()
        };
        if given_up {
            self.given_up.lock().unwrap().push(key.to_string());
            return Err(vellev_storage::blobs::unanswered(
                key,
                Duration::from_secs(187),
            ));
        }
        if self.deletes_hang.load(Ordering::SeqCst) {
            self.hung.lock().unwrap().push(key.to_string());
            self.delete_sent.notify_one();
            std::future::pending::<()>().await;
        }
        if self.refilling.load(Ordering::SeqCst) {
            let n = self.refilled.fetch_add(1, Ordering::SeqCst);
            let late = format!("acme/late/{n}");
            self.inner.put(&late, Bytes::from("late")).await?;
        }
        self.inner.delete(key).await
    }
}

/// Acme with a segment and more, backed up by another cluster to a store
/// of backups, which is returned with the backup.
async fn backed_up_elsewhere_with_a_segment() -> (Arc<dyn Blobs>, Backup) {
    let elsewhere = Cluster::new();
    let store = backup_store();
    let source = backing_up_to(&elsewhere, store.clone());
    let backup = written_and_backed_up(&source).await;
    assert!(backup.files().any(|key| key.contains("/segment/")));
    (store, backup)
}

/// A node of `cluster` through `coordinator`, keeping tenants in `blobs`
/// and backing up to `backups`.
fn node_over(
    cluster: &Cluster,
    coordinator: Arc<Unrenewing>,
    blobs: Arc<dyn Blobs>,
    backups: Arc<dyn Blobs>,
) -> Node {
    Node::new(
        coordinator,
        blobs,
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(backups),
            ..cluster.settings()
        },
    )
}

#[tokio::test]
async fn a_delete_given_up_by_a_purge_that_failed_lands_before_the_name_can_be_taken_again() {
    let (store, backup) = backed_up_elsewhere_with_a_segment().await;
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let blobs = Troubled::over(cluster.store.clone());
    let node = node_over(&cluster, coordinator.clone(), blobs.clone(), store.clone());
    let acme = tenant("acme");
    let adopt = || async {
        node.adopt_backup(&acme, backup.id, false, RequestId::generate())
            .await
    };
    adopt().await.unwrap();
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());

    // The purge's delete of a file the backup lists is given up
    // unanswered: the purge fails, with that delete's fate unknown.
    let segment = backup.files().find(|key| key.contains("/segment/"));
    *blobs.giving_up_on.lock().unwrap() = segment.cloned();
    node.purge_due().await.unwrap();
    let given_up = blobs.given_up.lock().unwrap().clone();
    assert_eq!(given_up.len(), 1, "one delete was given up");
    assert!(keys_under(cluster.store.as_ref(), "acme/")
        .await
        .contains(&given_up[0]));

    // Its claim is not let go, so nothing purges the tenant, and the name
    // cannot be taken, by a tenant made afresh or one taken up again, for
    // as long as that delete may still land.
    node.purge_due().await.unwrap();
    assert_eq!(coordinator.purge_releases.load(Ordering::SeqCst), 0);
    assert!(keys_under(cluster.store.as_ref(), "acme/")
        .await
        .contains(&given_up[0]));
    assert!(node.create_tenant(&acme).await.is_err());
    let refused = adopt().await;
    assert!(
        matches!(refused, Err(NodeError::Purging { .. })),
        "{refused:?}"
    );

    // It lands. Only once the claim has run out is the tenant purged, and
    // then its lineage taken up again from the same backup, which writes
    // the same keys: nothing is on its way to take them from under it.
    cluster.store.delete(&given_up[0]).await.unwrap();
    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    adopt().await.unwrap();
    assert!(
        keys_under(cluster.store.as_ref(), "acme/")
            .await
            .contains(&given_up[0]),
        "the key the delete given up named is the tenant's again"
    );
    node.shutdown().await;
    cluster.clock.advance(LEASE + Duration::from_secs(1));
    let fresh = backing_up_to(&cluster, store.clone());
    assert_eq!(notes_held(&fresh).await, [true, true]);
    fresh.back_up(&acme).await.unwrap();
}

#[tokio::test]
async fn a_purge_that_gave_up_on_a_tenant_still_written_to_lets_its_claim_go_at_once() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let blobs = Troubled::over(cluster.store.clone());
    let node = node_over(&cluster, coordinator.clone(), blobs.clone(), backup_store());
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(&node, 1, "fox").await;
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    // A file lands with every delete, so every pass finds more: the purge
    // gives up for now, every delete it sent answered.
    blobs.refilling.store(true, Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert!(blobs.refilled.load(Ordering::SeqCst) > 0);
    assert!(!keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    assert!(node.create_tenant(&tenant("acme")).await.is_err());
    assert_eq!(
        coordinator.purge_releases.load(Ordering::SeqCst),
        1,
        "let go, for the next pass to go on"
    );
    blobs.refilling.store(false, Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert!(keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    node.create_tenant(&tenant("acme")).await.unwrap();
    assert_eq!(
        coordinator.purge_releases.load(Ordering::SeqCst),
        1,
        "done, its claim was ended by the coordinator: there was none to let go"
    );
}

#[tokio::test]
async fn a_purge_that_failed_before_it_sent_any_delete_lets_its_claim_go_at_once() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let blobs = Troubled::over(cluster.store.clone());
    let node = node_over(&cluster, coordinator.clone(), blobs.clone(), backup_store());
    node.create_tenant(&tenant("acme")).await.unwrap();
    noted(&node, 1, "fox").await;
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    blobs.lists_fail.store(true, Ordering::SeqCst);
    let kept = keys_under(cluster.store.as_ref(), "acme/").await;
    node.purge_due().await.unwrap();
    assert_eq!(keys_under(cluster.store.as_ref(), "acme/").await, kept);
    // The claim is let go on a task of its own, which is given its turn.
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(coordinator.purge_releases.load(Ordering::SeqCst), 1);
    blobs.lists_fail.store(false, Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert!(keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    node.create_tenant(&tenant("acme")).await.unwrap();
}

#[tokio::test]
async fn a_purge_stopped_with_a_delete_on_its_way_leaves_its_claim_to_run_out() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let blobs = Troubled::over(cluster.store.clone());
    let node = node_over(&cluster, coordinator.clone(), blobs.clone(), backup_store());
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    noted(&node, 1, "fox").await;
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());

    // Whoever runs the purge is stopped, as a node shutting down stops its
    // tasks, once a delete has been sent and before it is answered.
    blobs.deletes_hang.store(true, Ordering::SeqCst);
    tokio::select! {
        biased;
        () = blobs.delete_sent.notified() => {}
        purged = node.purge_due() => panic!("{purged:?}"),
    }
    blobs.deletes_hang.store(false, Ordering::SeqCst);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(coordinator.purge_releases.load(Ordering::SeqCst), 0);
    let claimed = cluster
        .coordinator
        .claim_purge(&acme, Duration::ZERO, hold());
    assert!(claimed.await.unwrap().is_none(), "held still");
    node.purge_due().await.unwrap();
    assert!(node.create_tenant(&acme).await.is_err());

    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    node.create_tenant(&acme).await.unwrap();
}

#[tokio::test]
async fn a_restore_looks_again_that_every_file_is_there_just_before_it_commits() {
    let cluster = Cluster::new();
    let store = backup_store();
    let blobs = Troubled::over(cluster.store.clone());
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = node_over(&cluster, coordinator, blobs.clone(), store.clone());
    let acme = tenant("acme");
    let backup = written_and_backed_up(&node).await;
    let segment = backup
        .files()
        .find(|key| key.contains("/segment/"))
        .unwrap()
        .clone();
    noted(&node, 3, "owl").await;

    // The node holds the segment loaded, and loading the state to restore
    // reads nothing it holds: a segment that goes from the store once the
    // restore has found every file there, as by a delete landing late,
    // would be committed to unseen. It is looked for again, found gone,
    // and fetched back.
    *blobs.lost_after_listing.lock().unwrap() = Some(segment.clone());
    node.restore_backup(&acme, backup.id, RequestId::generate())
        .await
        .unwrap();
    assert!(blobs.lost_after_listing.lock().unwrap().is_none());
    assert!(
        keys_under(cluster.store.as_ref(), "acme/")
            .await
            .contains(&segment),
        "fetched back before the commit"
    );
    node.shutdown().await;
    cluster.clock.advance(LEASE + Duration::from_secs(1));
    let fresh = backing_up_to(&cluster, store);
    assert_eq!(notes_held(&fresh).await, [true, true]);
}

/// Acme on a node over a troubled store: notes 1 and 2 written, with
/// `vector` if any, note 1 in a segment, and backed up; then notes 3 to 5
/// written, a checkpoint made after each, so that once the tenant is
/// compacted, which merges its four segments first, every file the backup
/// lists lies below its newest checkpoint, which lists none of them, and
/// goes. The node is returned with
/// its cluster, its tenants' store and the backup.
async fn backed_up_and_merged_past(
    vector: Option<Vec<f32>>,
) -> (Cluster, Arc<dyn Blobs>, Arc<Troubled>, Node, Backup) {
    let cluster = Cluster::new();
    let store = backup_store();
    let blobs = Troubled::over(cluster.store.clone());
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = node_over(&cluster, coordinator, blobs.clone(), store.clone());
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    let tenant = node.tenant(&acme).await.unwrap();
    let written = |n, title| note(n, title, vector.clone());
    tenant
        .write(vec![schema(EmbeddingProvider::None), written(1, "fox")])
        .await
        .unwrap();
    tenant.flush().await.unwrap();
    tenant.write(vec![written(2, "hen")]).await.unwrap();
    let backup = node.back_up(&acme).await.unwrap();
    for (n, title) in [(3, "owl"), (4, "elk"), (5, "bee")] {
        tenant.write(vec![written(n, title)]).await.unwrap();
        tenant.flush().await.unwrap();
    }
    drop(tenant);
    (cluster, store, blobs, node, backup)
}

/// The one file of `kind` the backup lists, by the directory it lies in.
fn listed_of_kind(backup: &Backup, kind: &str) -> String {
    let mut listed = backup
        .files()
        .filter(|key| key.contains(&format!("/{kind}/")));
    let file = listed.next().cloned();
    assert!(listed.next().is_none(), "the backup lists one {kind} file");
    file.unwrap_or_else(|| panic!("the backup lists no {kind} file"))
}

/// Every segment, index and vectors file the checkpoint acme is committed
/// at names, and the logs it replays.
async fn named_by_the_head(cluster: &Cluster) -> Vec<String> {
    let committed = cluster.coordinator.committed(&key("acme")).await.unwrap();
    let head = committed.and_then(|committed| committed.manifest).unwrap();
    let frame = cluster.store.get(&head.0).await.unwrap();
    let Ok(Head::Manifest(checkpoint)) = decode_head(&head.0, &frame) else {
        panic!("a restore commits a checkpoint, and {head:?} is none");
    };
    let segments = checkpoint.segments.into_iter();
    let named = segments.flat_map(|segment| [Some(segment.key), segment.index, segment.vectors]);
    named.flatten().chain(checkpoint.replay).collect()
}

/// Acme restored from `backup` on `node`, which must succeed, and seen to
/// have put no key the backup lists and to name none: DW-680's rule, that
/// nothing comes in from outside a tenant's store under the key it had.
/// It does both today, and a delete of one of those keys sent before the
/// restore, and landing after it, takes it from under the head.
async fn restored_under_no_key_the_backup_lists(
    cluster: &Cluster,
    blobs: &Troubled,
    node: &Node,
    backup: &Backup,
) {
    let had = blobs.put.lock().unwrap().len();
    node.restore_backup(&tenant("acme"), backup.id, RequestId::generate())
        .await
        .unwrap();
    let listed: Vec<&String> = backup.files().collect();
    let put = blobs.put.lock().unwrap()[had..].to_vec();
    let put: Vec<&String> = put.iter().filter(|key| listed.contains(key)).collect();
    let named = named_by_the_head(cluster).await;
    let named: Vec<&String> = named.iter().filter(|key| listed.contains(key)).collect();
    assert!(
        put.is_empty() && named.is_empty(),
        "DW-680 K4: a restore from a backup put {put:?} and its head names {named:?}, \
         keys the backup lists"
    );
}

/// DW-680, T1: a clean gives up the delete of the one file of `kind` the
/// backup lists; the tenant is restored from that backup, which it took
/// itself, and the delete lands. A node with nothing of its own opens the
/// tenant and holds every note the backup held, and the tenant is backed
/// up again. With `rule`, the restore is first seen to keep DW-680's rule.
async fn a_delete_given_up_lands_after_a_restore_from_the_tenants_own_backup(
    kind: &str,
    vector: Option<Vec<f32>>,
    rule: bool,
) {
    let (cluster, store, blobs, node, backup) = backed_up_and_merged_past(vector).await;
    let acme = tenant("acme");
    let file = listed_of_kind(&backup, kind);
    *blobs.giving_up_on.lock().unwrap() = Some(file.clone());
    let cleaned = node.tenant(&acme).await.unwrap();
    let cleaned = cleaned.compact(Duration::ZERO).await;
    assert!(cleaned.is_err(), "the clean was told a delete was lost");
    assert_eq!(*blobs.given_up.lock().unwrap(), std::slice::from_ref(&file));
    assert!(keys_under(cluster.store.as_ref(), "acme/")
        .await
        .contains(&file));

    if rule {
        restored_under_no_key_the_backup_lists(&cluster, &blobs, &node, &backup).await;
    } else {
        node.restore_backup(&acme, backup.id, RequestId::generate())
            .await
            .unwrap();
    }
    cluster.store.delete(&file).await.unwrap();
    node.shutdown().await;
    cluster.clock.advance(LEASE + Duration::from_secs(1));
    let fresh = backing_up_to(&cluster, store);
    assert_eq!(notes_held(&fresh).await, [true, true]);
    let held = fresh.tenant(&acme).await.unwrap();
    assert!(held.read(|state| state.get(&notes(), id(3)).is_none()));
    fresh.back_up(&acme).await.unwrap();
}

#[tokio::test]
async fn a_late_delete_of_a_checkpoint_a_backup_lists_spares_the_tenant_restored_from_it() {
    a_delete_given_up_lands_after_a_restore_from_the_tenants_own_backup("manifest", None, false)
        .await;
}

#[tokio::test]
async fn a_late_delete_of_a_log_a_backup_lists_spares_the_tenant_restored_from_it() {
    a_delete_given_up_lands_after_a_restore_from_the_tenants_own_backup("log", None, false).await;
}

/// The reported bug: the restore finds the segment there still, writes
/// nothing, and commits a checkpoint that names it; the delete then takes
/// it, and no node with nothing of its own can open the tenant.
#[tokio::test]
#[should_panic(expected = "DW-680 K4: a restore from a backup put")]
async fn a_late_delete_of_a_segment_a_backup_lists_spares_the_tenant_restored_from_it() {
    a_delete_given_up_lands_after_a_restore_from_the_tenants_own_backup("segment", None, true)
        .await;
}

#[tokio::test]
#[should_panic(expected = "DW-680 K4: a restore from a backup put")]
async fn a_late_delete_of_a_vectors_file_a_backup_lists_spares_the_tenant_restored_from_it() {
    let vector = Some(vec![1.0, 0.5]);
    a_delete_given_up_lands_after_a_restore_from_the_tenants_own_backup("vectors", vector, true)
        .await;
}

/// DW-680, T2: a clean's minute runs out with its deletes sent and none
/// answered, and it returns as if all were well. The tenant is restored
/// from its own backup, and the deletes land.
#[tokio::test(start_paused = true)]
#[should_panic(expected = "DW-680 K4: a restore from a backup put")]
async fn deletes_a_clean_stopped_waiting_for_land_after_a_restore_from_the_tenants_own_backup() {
    let (cluster, store, blobs, node, backup) = backed_up_and_merged_past(None).await;
    let acme = tenant("acme");
    blobs.deletes_hang.store(true, Ordering::SeqCst);
    let cleaned = node.tenant(&acme).await.unwrap();
    let cleaned = cleaned.compact(Duration::ZERO).await;
    assert_eq!(cleaned.unwrap(), 0, "a pass that ran out of time");
    blobs.deletes_hang.store(false, Ordering::SeqCst);
    let hung = blobs.hung.lock().unwrap().clone();
    let segment = listed_of_kind(&backup, "segment");
    assert!(hung.contains(&segment), "{hung:?}");
    let kept = keys_under(cluster.store.as_ref(), "acme/").await;
    assert!(hung.iter().all(|key| kept.contains(key)), "{hung:?}");

    restored_under_no_key_the_backup_lists(&cluster, &blobs, &node, &backup).await;
    for key in &hung {
        cluster.store.delete(key).await.unwrap();
    }
    node.shutdown().await;
    cluster.clock.advance(LEASE + Duration::from_secs(1));
    let fresh = backing_up_to(&cluster, store);
    assert_eq!(notes_held(&fresh).await, [true, true]);
    fresh.back_up(&acme).await.unwrap();
}

#[tokio::test]
async fn a_claim_none_of_whose_answers_was_heard_is_let_go() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing {
        claims_unheard: AtomicBool::new(true),
        ..Unrenewing::new(cluster.coordinator.clone())
    });
    let node = backing_up_through(&cluster, coordinator.clone(), backup_store()).await;
    let failed = node.back_up(&tenant("acme")).await;
    assert!(failed.is_err(), "{failed:?}");
    assert!(coordinator.claims.load(Ordering::SeqCst) > 1, "asked again");
    // Taken, each time, for all the node heard: so it is let go, and
    // another may take it at once rather than a claim later.
    assert!(unclaimed(&cluster.coordinator, &tenant("acme")).await);
}

#[tokio::test]
async fn a_claim_whose_holder_stops_waiting_as_it_is_let_go_is_let_go_all_the_same() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let blobs = Troubled::over(cluster.store.clone());
    let node = node_over(&cluster, coordinator.clone(), blobs.clone(), backup_store());
    let acme = tenant("acme");
    node.create_tenant(&acme).await.unwrap();
    noted(&node, 1, "fox").await;
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());

    // A purge gives up on a tenant still written to and lets its claim go,
    // the coordinator slow to answer; whoever ran the purge is stopped
    // just then, as a node shutting down stops its tasks.
    blobs.refilling.store(true, Ordering::SeqCst);
    coordinator.releases_wait.store(true, Ordering::SeqCst);
    tokio::select! {
        biased;
        () = coordinator.release_asked.notified() => {}
        purged = node.purge_due() => panic!("{purged:?}"),
    }
    let claimed = || async {
        let claimed = cluster
            .coordinator
            .claim_purge(&acme, Duration::ZERO, hold());
        claimed.await.unwrap()
    };
    assert!(claimed().await.is_none(), "held until it is let go");
    coordinator.releases_wait.store(false, Ordering::SeqCst);
    coordinator.release_let_through.notify_one();
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(
        claimed().await.is_some(),
        "let go on a task of its own, which ran to its end"
    );
}

#[tokio::test]
async fn a_purge_that_failed_with_only_a_backups_delete_sent_leaves_its_claim_to_run_out_too() {
    let cluster = Cluster::new();
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let backups = Troubled::over(Cluster::new().store.clone());
    let node = node_over(
        &cluster,
        coordinator.clone(),
        cluster.store.clone(),
        backups.clone(),
    );
    let acme = tenant("acme");
    written_and_backed_up(&node).await;
    node.drop_tenant(&acme).await.unwrap();
    cluster.clock.advance(hold());

    // A first purge deletes the tenant's files and fails listing its
    // backups; its claim, files deleted under it, runs out.
    backups.lists_fail.store(true, Ordering::SeqCst);
    node.purge_due().await.unwrap();
    assert!(keys_under(cluster.store.as_ref(), "acme/").await.is_empty());
    backups.lists_fail.store(false, Ordering::SeqCst);
    cluster.clock.advance(hold() + Duration::from_secs(1));

    // The next finds no file of the tenant's to delete, and has a delete
    // of one of its backups' given up: sent under the claim all the same,
    // to a store whose files a tenant of the lineage would take the names
    // of again.
    let kept = keys_under(backups.inner.as_ref(), "_backups/acme/").await;
    *backups.giving_up_on.lock().unwrap() = kept.first().cloned();
    node.purge_due().await.unwrap();
    assert_eq!(backups.given_up.lock().unwrap().len(), 1);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(coordinator.purge_releases.load(Ordering::SeqCst), 0);
    node.purge_due().await.unwrap();
    assert!(node.create_tenant(&acme).await.is_err());

    cluster.clock.advance(hold() + Duration::from_secs(1));
    node.purge_due().await.unwrap();
    assert!(keys_under(backups.inner.as_ref(), "_backups/acme/")
        .await
        .is_empty());
    node.create_tenant(&acme).await.unwrap();
}

/// A store each delete of which, of a key holding `only` if that is given,
/// waits out a backoff on the runtime's timers before the request goes, as
/// the store's client does between two tries. The first such runs a hook
/// before it waits, and every delete sent once that hook has run is
/// counted.
#[derive(Debug)]
struct Backoff {
    inner: Arc<dyn Blobs>,
    only: Option<&'static str>,
    before_backoff: Hook,
    hooked: AtomicBool,
    sent_since: AtomicUsize,
}

impl Backoff {
    fn over(inner: Arc<dyn Blobs>, only: Option<&'static str>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            only,
            before_backoff: Hook::default(),
            hooked: AtomicBool::new(false),
            sent_since: AtomicUsize::new(0),
        })
    }

    /// Have the machine be suspended for a claim's hold, as `claims` counts
    /// it, while the next delete is parked on its backoff: by another task,
    /// which on the one thread these tests run on has its turn only once
    /// the delete has parked. So the clock moves between two polls of the
    /// delete, as it does over a suspend, and not within one.
    fn suspended_while_parked(&self, claims: Arc<RuntimeClock>) {
        self.before_backoff.set(async move {
            tokio::spawn(async move { claims.sleep_through(hold()) });
        });
    }
}

#[async_trait]
impl Blobs for Backoff {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        if self.only.is_none_or(|only| key.contains(only)) {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if self.before_backoff.is_set() {
                self.before_backoff.run().await;
                self.hooked.store(true, Ordering::SeqCst);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        if self.hooked.load(Ordering::SeqCst) {
            self.sent_since.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_machine_suspended_between_two_tries_of_a_delete_sends_nothing_once_woken() {
    let cluster = Cluster::new();
    let inner = backup_store();
    let store = Backoff::over(inner.clone(), None);
    let claims = Arc::new(RuntimeClock::new());
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        claim_clock: claims.clone(),
        ..cluster.settings()
    });
    written_and_backed_up(&node).await;
    let kept = keys_under(inner.as_ref(), "_backups/acme/").await.len();
    assert!(kept > 2);
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    // The machine is suspended past the claim while a purge's delete in
    // the backups waits out a backoff inside the store's client. Woken,
    // the client's own timer fires and takes the delete up again: the
    // claim's clock is asked before it is, and nothing is sent.
    store.suspended_while_parked(claims);
    node.purge_due().await.unwrap();
    assert!(!store.before_backoff.is_set(), "the machine slept");
    assert_eq!(store.sent_since.load(Ordering::SeqCst), 0);
    assert_eq!(
        keys_under(inner.as_ref(), "_backups/acme/").await.len(),
        kept,
        "nothing was deleted once the claim had run out by its own clock"
    );
    assert!(node.create_tenant(&tenant("acme")).await.is_err());
}

#[tokio::test]
async fn a_machine_suspended_between_two_tries_of_a_tenants_file_delete_sends_nothing_once_woken() {
    let cluster = Cluster::new();
    let blobs = Backoff::over(cluster.store.clone(), None);
    let claims = Arc::new(RuntimeClock::new());
    let node = Node::new(
        cluster.coordinator.clone(),
        blobs.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            backups: Some(backup_store()),
            claim_clock: claims.clone(),
            ..cluster.settings()
        },
    );
    written_and_backed_up(&node).await;
    let kept = keys_under(cluster.store.as_ref(), "acme/").await.len();
    assert!(kept > 2);
    node.drop_tenant(&tenant("acme")).await.unwrap();
    cluster.clock.advance(hold());

    // The same, in the tenants' store, where a file deleted late is not
    // put back by anything.
    blobs.suspended_while_parked(claims);
    node.purge_due().await.unwrap();
    assert!(!blobs.before_backoff.is_set(), "the machine slept");
    assert_eq!(blobs.sent_since.load(Ordering::SeqCst), 0);
    assert_eq!(
        keys_under(cluster.store.as_ref(), "acme/").await.len(),
        kept
    );
    assert!(node.create_tenant(&tenant("acme")).await.is_err());
}

#[tokio::test]
async fn a_machine_suspended_between_two_tries_of_a_prune_sends_nothing_once_woken() {
    let cluster = Cluster::new();
    let inner = backup_store();
    let day = Duration::from_secs(24 * 60 * 60);
    let claims = Arc::new(RuntimeClock::new());
    let plain = cluster.node_from(Settings {
        backups: Some(inner.clone()),
        backups_kept: day,
        clock: cluster.clock.clone(),
        ..cluster.settings()
    });
    let acme = tenant("acme");
    let first = written_and_backed_up(&plain).await;
    cluster.clock.advance(Duration::from_secs(60 * 60));
    noted(&plain, 3, "owl").await;
    let second = plain.back_up(&acme).await.unwrap();
    drop(plain);
    cluster.clock.advance(LEASE + Duration::from_secs(1));

    // Under a claim on a tenant's backups, in pruning: the machine is
    // suspended past the claim while the delete of an old backup's record
    // waits between two tries. Nothing is pruned once it wakes, and the
    // backup made is the run's outcome all the same.
    let store = Backoff::over(inner.clone(), Some(".backup.json"));
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        backups_kept: day,
        clock: cluster.clock.clone(),
        claim_clock: claims.clone(),
        ..cluster.settings()
    });
    cluster.clock.advance(2 * day);
    noted(&node, 4, "elk").await;
    let before = keys_under(inner.as_ref(), "_backups/acme/").await;
    store.suspended_while_parked(claims);
    let third = node.back_up(&acme).await.unwrap();
    assert!(
        !store.before_backoff.is_set(),
        "the machine slept in pruning"
    );
    assert_eq!(store.sent_since.load(Ordering::SeqCst), 0);
    let after = keys_under(inner.as_ref(), "_backups/acme/").await;
    let gone: Vec<_> = before.iter().filter(|key| !after.contains(key)).collect();
    assert!(gone.is_empty(), "{gone:?}");
    assert_eq!(node.backups(&acme).await.unwrap(), [first, second, third]);
}

#[tokio::test]
async fn backups_are_taken_and_pruned_by_the_clock_the_node_is_given() {
    let cluster = Cluster::new();
    let store = backup_store();
    let day = Duration::from_secs(24 * 60 * 60);
    let node = cluster.node_from(Settings {
        backups: Some(store.clone()),
        backups_kept: day,
        clock: cluster.clock.clone(),
        ..cluster.settings()
    });
    let acme = tenant("acme");
    let first = written_and_backed_up(&node).await;
    assert_eq!(first.taken_at, cluster.clock.now(), "by the node's clock");

    // An hour on by that clock, whatever the machine's says, which is days
    // past it: the first is kept, a day not being up.
    cluster.clock.advance(Duration::from_secs(60 * 60));
    noted(&node, 3, "owl").await;
    let second = node.back_up(&acme).await.unwrap();
    assert_eq!(
        node.backups(&acme).await.unwrap(),
        [first.clone(), second.clone()]
    );

    // Two days on, both are older than a day, and the newest of them is
    // kept all the same, with the one taken now.
    cluster.clock.advance(2 * day);
    noted(&node, 4, "elk").await;
    let third = node.back_up(&acme).await.unwrap();
    assert_eq!(node.backups(&acme).await.unwrap(), [third]);
}

#[tokio::test]
async fn a_backup_lacking_a_file_stays_listed_and_says_which_when_it_is_restored_from() {
    let elsewhere = Cluster::new();
    let store = backup_store();
    let source = backing_up_to(&elsewhere, store.clone());
    let backup = written_and_backed_up(&source).await;
    let acme = tenant("acme");
    let kept = keys_under(store.as_ref(), "_backups/acme/").await;
    let lost = kept
        .iter()
        .find(|key| key.contains("/data/") && key.contains("/segment/"))
        .unwrap()
        .clone();
    store.delete(&lost).await.unwrap();

    // Nothing looks whether a backup's files are there until it is
    // restored from: it is listed still.
    assert_eq!(
        source.backups(&acme).await.unwrap(),
        std::slice::from_ref(&backup)
    );

    // A cluster without the file is told which backup lacks which file,
    // and takes nothing of it.
    let cluster = Cluster::new();
    let node = backing_up_to(&cluster, store.clone());
    let refused = node
        .adopt_backup(&acme, backup.id, false, RequestId::generate())
        .await;
    let Err(NodeError::Storage(StorageError::BackupLacks {
        tenant,
        backup: lacking,
        file,
    })) = refused
    else {
        panic!("{refused:?}");
    };
    assert_eq!((tenant, lacking), (acme.clone(), backup.id));
    assert!(lost.ends_with(&file), "{file} of {lost}");
    let now = cluster.coordinator.identity(&acme).await.unwrap().unwrap();
    assert!(!now.committed && now.lineage != backup.lineage, "{now:?}");

    // The cluster that took it has the file, restores from it all the
    // same, and puts the file back with its next backup, which lists it.
    source
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await
        .unwrap();
    source.back_up(&acme).await.unwrap();
    assert!(keys_under(store.as_ref(), "_backups/acme/")
        .await
        .contains(&lost));
    node.adopt_backup(&acme, backup.id, false, RequestId::generate())
        .await
        .unwrap();
    assert_eq!(notes_held(&node).await, [true, true]);
}

#[tokio::test]
async fn a_restore_whose_file_goes_each_time_it_is_fetched_back_is_refused_saying_which() {
    let cluster = Cluster::new();
    let store = backup_store();
    let blobs = Troubled::over(cluster.store.clone());
    let coordinator = Arc::new(Unrenewing::new(cluster.coordinator.clone()));
    let node = node_over(&cluster, coordinator, blobs.clone(), store);
    let acme = tenant("acme");
    let backup = written_and_backed_up(&node).await;
    let segment = backup.files().find(|key| key.contains("/segment/"));
    let segment = segment.unwrap().clone();
    noted(&node, 3, "owl").await;
    let before = cluster.coordinator.committed(&self::acme()).await.unwrap();

    // The segment is gone from the tenant's store, and goes again as soon
    // as it is fetched back, each time.
    cluster.store.delete(&segment).await.unwrap();
    *blobs.lost_as_written.lock().unwrap() = Some(segment.clone());
    let refused = node
        .restore_backup(&acme, backup.id, RequestId::generate())
        .await;
    let Err(NodeError::Storage(StorageError::Corrupt { key, reason })) = refused else {
        panic!("{refused:?}");
    };
    assert_eq!(key, segment);
    assert!(reason.contains(&backup.id.to_string()), "{reason}");
    assert!(reason.contains("each of the 3 times"), "{reason}");
    let after = cluster.coordinator.committed(&self::acme()).await.unwrap();
    assert_eq!(after, before, "nothing was committed");
}

/// DW-681 trigger D through a node, F2b: a tenant's restore window made
/// longer. A node cleans with the window and its grace on top; the first
/// clean, with one hour in all, deletes below the newest checkpoint, the
/// one before it among them. Once the window is longer, the next clean
/// keeps more: it finds files young by four hours below that checkpoint,
/// walks back past it to the one deleted, and is to succeed all the same.
/// Today it fails, and every round after it, until every file is older
/// than four hours.
#[tokio::test]
#[should_panic(expected = "DW-681: a clean failed with nothing in its way")]
async fn a_node_cleans_after_the_tenants_window_was_made_longer() {
    let cluster = Cluster::new();
    let hour = Duration::from_secs(60 * 60);
    let clock = Arc::new(ManualClock::starting_at(Utc::now()));
    let node = cluster.node_from(Settings {
        clock: clock.clone(),
        ..cluster.settings()
    });
    node.create_tenant(&tenant("acme")).await.unwrap();
    let acme = node.tenant(&tenant("acme")).await.unwrap();
    acme.write(vec![schema(EmbeddingProvider::None), note(0, "ant", None)])
        .await
        .unwrap();
    acme.flush().await.unwrap();
    acme.write(vec![note(1, "fox", None)]).await.unwrap();
    acme.flush().await.unwrap();
    clock.advance(2 * hour);
    assert!(acme.compact(hour).await.unwrap() > 0);
    if let Err(error) = acme.compact(4 * hour).await {
        panic!("DW-681: a clean failed with nothing in its way: {error}");
    }
}
