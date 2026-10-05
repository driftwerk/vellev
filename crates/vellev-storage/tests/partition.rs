use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{TimeZone, Utc};
use object_store::memory::InMemory;
use proptest::prelude::*;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition as PartitionNumber, Property, TenantId, Value,
};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Lease, LogPosition, ManualClock,
    MemoryCoordinator, PartitionKey,
};
use vellev_storage::{Mutation, Partition, State, StorageError, Writer};

const TTL: Duration = Duration::from_secs(30);

fn name(collection: &str) -> CollectionName {
    CollectionName::new(collection).unwrap()
}

fn id(n: u8) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
}

fn schema(collection: &str) -> Collection {
    Collection {
        name: name(collection),
        properties: vec![
            Property::new("title", DataType::Text).searchable(),
            Property::new("n", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "test".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

fn create(collection: &str) -> Mutation {
    Mutation::CreateCollection(Box::new(schema(collection)))
}

fn upsert(collection: &str, object: u8, n: i64) -> Mutation {
    Mutation::Upsert {
        collection: name(collection),
        id: id(object),
        properties: json!({ "title": format!("row {object}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![n as f32, 0.5]),
    }
}

fn delete(collection: &str, object: u8) -> Mutation {
    Mutation::Delete {
        collection: name(collection),
        id: id(object),
    }
}

fn n_of(state: &State, collection: &str, object: u8) -> Option<i64> {
    state
        .get(&name(collection), id(object))
        .map(|object| match object.properties["n"] {
            Value::Int(n) => n,
            ref other => panic!("{other:?}"),
        })
}

/// One partition's shared infrastructure: object storage and a coordinator.
struct Cluster {
    key: PartitionKey,
    blobs: Arc<dyn Blobs>,
    clock: Arc<ManualClock>,
    coordinator: Arc<MemoryCoordinator>,
}

impl Cluster {
    async fn new() -> Self {
        let clock = Arc::new(ManualClock::starting_at(
            Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
        ));
        let coordinator = Arc::new(MemoryCoordinator::new(clock.clone()));
        let key = PartitionKey {
            tenant: TenantId::new("tenant-a").unwrap(),
            partition: PartitionNumber::DEFAULT,
        };
        coordinator.register(&key).await.unwrap();
        Self {
            key,
            blobs: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
            clock,
            coordinator,
        }
    }

    async fn committed(&self) -> Committed {
        self.coordinator
            .committed(&self.key)
            .await
            .unwrap()
            .unwrap()
    }

    /// A new node takes the lease, waiting out whoever holds it.
    async fn lease(&self) -> (Lease, Committed) {
        let holder = NodeId::generate("node");
        if let Acquired::Lease { lease, committed } = self
            .coordinator
            .acquire(&self.key, &holder, TTL)
            .await
            .unwrap()
        {
            return (lease, committed);
        }
        self.clock.advance(TTL);
        match self
            .coordinator
            .acquire(&self.key, &holder, TTL)
            .await
            .unwrap()
        {
            Acquired::Lease { lease, committed } => (lease, committed),
            held => panic!("{held:?}"),
        }
    }

    async fn writer(&self) -> Writer {
        self.writer_on(self.coordinator.clone()).await
    }

    async fn writer_on(&self, coordinator: Arc<dyn Coordinator>) -> Writer {
        let (lease, committed) = self.lease().await;
        Writer::open(self.blobs.clone(), coordinator, lease, &committed)
            .await
            .unwrap()
    }

    async fn open(&self) -> Partition {
        Partition::open(
            self.key.clone(),
            self.blobs.clone(),
            &self.committed().await,
        )
        .await
        .unwrap()
    }
}

#[tokio::test]
async fn data_survives_a_restart_with_nothing_held_locally() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 10),
            upsert("notes", 2, 20),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let position = writer
        .commit(vec![upsert("notes", 2, 21), upsert("notes", 3, 30)])
        .await
        .unwrap();
    drop(writer);

    let reopened = cluster.open().await;
    reopened.read(|state| {
        assert_eq!(state.position(), position);
        assert_eq!(n_of(state, "notes", 1), Some(10));
        assert_eq!(n_of(state, "notes", 2), Some(21));
        assert_eq!(n_of(state, "notes", 3), Some(30));
        assert_eq!(state.count(&name("notes")), 3);
        assert_eq!(
            state.get(&name("notes"), id(2)).unwrap().vector,
            Some(vec![21.0, 0.5])
        );
        assert_eq!(state.collection(&name("notes")), Some(&schema("notes")));
    });
}

#[tokio::test]
async fn a_rejected_mutation_fails_the_whole_batch_and_changes_nothing() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let position = writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();

    let bad_type = Mutation::Upsert {
        collection: name("notes"),
        id: id(2),
        properties: json!({ "n": "two" }).as_object().unwrap().clone(),
        vector: None,
    };
    let wrong_vector = Mutation::Upsert {
        collection: name("notes"),
        id: id(2),
        properties: Default::default(),
        vector: Some(vec![1.0; 3]),
    };
    type Expected = fn(&StorageError) -> bool;
    let rejected: Vec<(Vec<Mutation>, Expected)> = vec![
        (vec![upsert("notes", 2, 2), bad_type], |e| {
            matches!(e, StorageError::Object(_))
        }),
        (vec![wrong_vector], |e| matches!(e, StorageError::Object(_))),
        (vec![upsert("notes", 2, 2), upsert("nope", 1, 1)], |e| {
            matches!(e, StorageError::UnknownCollection(_))
        }),
        (vec![create("notes")], |e| {
            matches!(e, StorageError::CollectionExists(_))
        }),
        (vec![delete("nope", 1)], |e| {
            matches!(e, StorageError::UnknownCollection(_))
        }),
        (vec![Mutation::DropCollection(name("nope"))], |e| {
            matches!(e, StorageError::UnknownCollection(_))
        }),
    ];
    for (batch, expected) in rejected {
        let error = writer.commit(batch).await.unwrap_err();
        assert!(expected(&error), "{error:?}");
    }
    assert_eq!(cluster.committed().await.log_position, position);
    writer.partition().read(|state| {
        assert_eq!(state.position(), position);
        assert_eq!(state.count(&name("notes")), 1);
    });
    assert_eq!(
        writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap(),
        LogPosition(position.0 + 1)
    );
}

#[tokio::test]
async fn a_commit_or_flush_with_nothing_to_do_writes_nothing() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    assert_eq!(writer.flush().await.unwrap(), LogPosition(0));
    assert_eq!(writer.commit(vec![]).await.unwrap(), LogPosition(0));
    assert_eq!(cluster.committed().await.manifest, None);

    let position = writer.commit(vec![create("notes")]).await.unwrap();
    let unchanged = Mutation::AddProperty {
        collection: name("notes"),
        property: Property::new("n", DataType::Int),
    };
    assert_eq!(writer.commit(vec![unchanged]).await.unwrap(), position);
    let flushed = writer.flush().await.unwrap();
    let head = cluster.committed().await;
    assert_eq!(writer.flush().await.unwrap(), flushed);
    assert_eq!(cluster.committed().await, head);
}

#[tokio::test]
async fn schema_changes_are_durable() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let added = Mutation::AddProperty {
        collection: name("notes"),
        property: Property::new("reviewed", DataType::Bool),
    };
    writer.commit(vec![create("notes"), added]).await.unwrap();
    let reviewed = Mutation::Upsert {
        collection: name("notes"),
        id: id(1),
        properties: json!({ "reviewed": true }).as_object().unwrap().clone(),
        vector: None,
    };
    writer.commit(vec![reviewed]).await.unwrap();
    for flush in [false, true] {
        if flush {
            writer.flush().await.unwrap();
        }
        cluster.open().await.read(|state| {
            assert!(state
                .collection(&name("notes"))
                .unwrap()
                .property("reviewed")
                .is_some());
            assert_eq!(
                state.get(&name("notes"), id(1)).unwrap().properties["reviewed"],
                Value::Bool(true)
            );
        });
    }
}

#[tokio::test]
async fn deletes_and_overwrites_hold_across_checkpoints() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 2),
            upsert("notes", 3, 3),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .commit(vec![delete("notes", 1), upsert("notes", 2, 22)])
        .await
        .unwrap();
    writer
        .commit(vec![upsert("notes", 3, 33), delete("notes", 3)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 1, 111)]).await.unwrap();
    writer.commit(vec![delete("notes", 9)]).await.unwrap();

    let expect = |state: &State| {
        assert_eq!(n_of(state, "notes", 1), Some(111));
        assert_eq!(n_of(state, "notes", 2), Some(22));
        assert_eq!(n_of(state, "notes", 3), None);
        assert_eq!(state.count(&name("notes")), 2);
    };
    writer.partition().read(expect);
    cluster.open().await.read(expect);
    writer.flush().await.unwrap();
    cluster.open().await.read(expect);
}

#[tokio::test]
async fn a_dropped_collection_does_not_come_back_with_its_rows() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer
        .commit(vec![
            Mutation::DropCollection(name("notes")),
            create("notes"),
            upsert("notes", 3, 3),
        ])
        .await
        .unwrap();

    let expect = |state: &State| {
        assert_eq!(n_of(state, "notes", 1), None);
        assert_eq!(n_of(state, "notes", 2), None);
        assert_eq!(n_of(state, "notes", 3), Some(3));
        assert_eq!(state.count(&name("notes")), 1);
    };
    writer.partition().read(expect);
    cluster.open().await.read(expect);
    writer.flush().await.unwrap();
    cluster.open().await.read(expect);
}

#[tokio::test]
async fn a_follower_catches_up_across_commits_and_checkpoints() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let follower = cluster.open().await;
    assert_eq!(follower.read(State::position), LogPosition(0));

    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    follower.read(|state| assert_eq!(state.count(&name("notes")), 2));

    writer.flush().await.unwrap();
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    writer.flush().await.unwrap();
    let position = writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    assert_eq!(
        follower.catch_up(&cluster.committed().await).await.unwrap(),
        position
    );
    follower.read(|state| {
        assert_eq!(n_of(state, "notes", 1), None);
        assert_eq!(n_of(state, "notes", 2), Some(2));
        assert_eq!(n_of(state, "notes", 3), Some(3));
    });
    assert_eq!(
        follower.catch_up(&cluster.committed().await).await.unwrap(),
        position
    );
}

#[tokio::test]
async fn a_fenced_writer_changes_nothing_and_its_file_is_never_read() {
    let cluster = Cluster::new().await;
    let stalled = cluster.writer().await;
    stalled
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();

    let successor = cluster.writer().await;
    successor
        .commit(vec![upsert("notes", 1, 100)])
        .await
        .unwrap();

    let fenced = stalled
        .commit(vec![upsert("notes", 1, -1), upsert("notes", 2, -2)])
        .await;
    assert!(matches!(
        fenced,
        Err(StorageError::Coordinator(CoordinatorError::Fenced { .. }))
    ));
    assert!(matches!(
        stalled.commit(vec![upsert("notes", 3, 3)]).await,
        Err(StorageError::Closed)
    ));
    assert!(matches!(stalled.flush().await, Err(StorageError::Closed)));
    stalled
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));

    let position = successor.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    cluster.open().await.read(|state| {
        assert_eq!(state.position(), position);
        assert_eq!(n_of(state, "notes", 1), Some(100));
        assert_eq!(n_of(state, "notes", 2), None);
        assert_eq!(n_of(state, "notes", 4), Some(4));
    });
}

/// A coordinator whose commit responses can be lost after the commit landed,
/// or which can stop accepting commits altogether.
#[derive(Debug)]
struct Unreliable {
    inner: Arc<MemoryCoordinator>,
    lose_responses: AtomicUsize,
    refuse_commits: AtomicUsize,
}

fn lost() -> CoordinatorError {
    CoordinatorError::backend(std::io::Error::other("connection reset"))
}

#[async_trait]
impl Coordinator for Unreliable {
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
        self.inner.renew(lease, ttl).await
    }
    async fn release(&self, lease: &Lease) -> Result<(), CoordinatorError> {
        self.inner.release(lease).await
    }
    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed, CoordinatorError> {
        let take = |counter: &AtomicUsize| {
            counter
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        };
        if take(&self.refuse_commits) {
            return Err(lost());
        }
        let committed = self.inner.commit(lease, commit).await?;
        if take(&self.lose_responses) {
            return Err(lost());
        }
        Ok(committed)
    }
    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>, CoordinatorError> {
        self.inner.committed(key).await
    }
    async fn ping(&self) -> Result<(), CoordinatorError> {
        self.inner.ping().await
    }
}

#[tokio::test]
async fn a_lost_commit_response_is_retried_and_an_unknown_outcome_closes_the_writer() {
    let cluster = Cluster::new().await;
    let unreliable = Arc::new(Unreliable {
        inner: cluster.coordinator.clone(),
        lose_responses: AtomicUsize::new(2),
        refuse_commits: AtomicUsize::new(0),
    });
    let writer = cluster.writer_on(unreliable.clone()).await;

    let position = writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    assert_eq!(cluster.committed().await.log_position, position);
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));

    unreliable
        .refuse_commits
        .store(usize::MAX, Ordering::SeqCst);
    let unknown = writer.commit(vec![upsert("notes", 2, 2)]).await;
    assert!(matches!(
        unknown,
        Err(StorageError::Coordinator(CoordinatorError::Backend(_)))
    ));
    unreliable.refuse_commits.store(0, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(vec![upsert("notes", 3, 3)]).await,
        Err(StorageError::Closed)
    ));

    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 2), None));
    assert_eq!(cluster.committed().await.log_position, position);
    let reopened = cluster.writer().await;
    reopened.commit(vec![upsert("notes", 2, 20)]).await.unwrap();
    cluster
        .open()
        .await
        .read(|state| assert_eq!(n_of(state, "notes", 2), Some(20)));
}

#[derive(Debug)]
struct Counting {
    inner: Arc<dyn Blobs>,
    gets: AtomicUsize,
}

#[async_trait]
impl Blobs for Counting {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        self.inner.get(key).await
    }
}

#[tokio::test]
async fn files_are_read_from_the_store_once_and_then_from_local_disk() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Counting {
        inner: cluster.blobs.clone(),
        gets: AtomicUsize::new(0),
    });
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), dir.path()));
    let committed = cluster.committed().await;

    let first = Partition::open(cluster.key.clone(), cached.clone(), &committed)
        .await
        .unwrap();
    let fetched = store.gets.load(Ordering::SeqCst);
    assert_eq!(fetched, 3, "a log segment, a manifest and a segment");
    let second = Partition::open(cluster.key.clone(), cached, &committed)
        .await
        .unwrap();
    assert_eq!(store.gets.load(Ordering::SeqCst), fetched);
    for partition in [first, second] {
        partition.read(|state| assert_eq!(state.count(&name("notes")), 2));
    }
}

#[tokio::test]
async fn a_damaged_or_missing_file_stops_recovery_with_its_name() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let committed = cluster.committed().await;
    let head = committed.manifest.clone().unwrap().0;

    let mut damaged = cluster.blobs.get(&head).await.unwrap().to_vec();
    *damaged.last_mut().unwrap() ^= 0xff;
    let broken: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    broken.put(&head, Bytes::from(damaged)).await.unwrap();
    assert!(matches!(
        Partition::open(cluster.key.clone(), broken, &committed).await,
        Err(StorageError::Corrupt { key, .. }) if key == head
    ));

    let empty: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    assert!(matches!(
        Partition::open(cluster.key.clone(), empty, &committed).await,
        Err(StorageError::Missing { key }) if key == head
    ));

    let behind = Committed {
        log_position: LogPosition(committed.log_position.0 + 1),
        ..committed
    };
    assert!(matches!(
        Partition::open(cluster.key.clone(), cluster.blobs.clone(), &behind).await,
        Err(StorageError::Corrupt { .. })
    ));
}

const COLLECTIONS: [&str; 2] = ["a", "b"];
const IDS: u8 = 5;

#[derive(Debug, Clone)]
enum Change {
    Create(usize),
    Drop(usize),
    Upsert(usize, u8, i64),
    Delete(usize, u8),
}

#[derive(Debug, Clone)]
enum Action {
    Commit(Vec<Change>),
    Flush,
    Restart,
    Takeover,
    Follow,
}

type Model = BTreeMap<&'static str, BTreeMap<u8, i64>>;

impl Change {
    fn mutation(&self) -> Mutation {
        match *self {
            Change::Create(c) => create(COLLECTIONS[c]),
            Change::Drop(c) => Mutation::DropCollection(name(COLLECTIONS[c])),
            Change::Upsert(c, object, n) => upsert(COLLECTIONS[c], object, n),
            Change::Delete(c, object) => delete(COLLECTIONS[c], object),
        }
    }

    /// The change applied to the model, or `None` when the storage layer must reject it.
    fn applied(&self, mut model: Model) -> Option<Model> {
        match *self {
            Change::Create(c) => model
                .insert(COLLECTIONS[c], BTreeMap::new())
                .is_none()
                .then_some(model),
            Change::Drop(c) => model.remove(COLLECTIONS[c]).map(|_| model),
            Change::Upsert(c, object, n) => {
                model.get_mut(COLLECTIONS[c])?.insert(object, n);
                Some(model)
            }
            Change::Delete(c, object) => {
                model.get_mut(COLLECTIONS[c])?.remove(&object);
                Some(model)
            }
        }
    }
}

fn change() -> impl Strategy<Value = Change> {
    let collection = 0..COLLECTIONS.len();
    prop_oneof![
        1 => collection.clone().prop_map(Change::Create),
        1 => collection.clone().prop_map(Change::Drop),
        6 => (collection.clone(), 0..IDS, -3i64..4).prop_map(|(c, o, n)| Change::Upsert(c, o, n)),
        3 => (collection, 0..IDS).prop_map(|(c, o)| Change::Delete(c, o)),
    ]
}

fn action() -> impl Strategy<Value = Action> {
    prop_oneof![
        8 => prop::collection::vec(change(), 0..5).prop_map(Action::Commit),
        3 => Just(Action::Flush),
        2 => Just(Action::Restart),
        2 => Just(Action::Takeover),
        2 => Just(Action::Follow),
    ]
}

fn assert_matches_model(state: &State, model: &Model, context: &str) {
    for collection in COLLECTIONS {
        let expected = model.get(collection);
        assert_eq!(
            state.collection(&name(collection)).is_some(),
            expected.is_some(),
            "{context}"
        );
        let empty = BTreeMap::new();
        let expected = expected.unwrap_or(&empty);
        for object in 0..IDS {
            assert_eq!(
                n_of(state, collection, object),
                expected.get(&object).copied(),
                "{context}"
            );
        }
        let scanned: BTreeSet<ObjectId> = state.scan(&name(collection)).map(|o| o.id).collect();
        assert_eq!(
            scanned,
            expected.keys().map(|object| id(*object)).collect(),
            "{context}"
        );
        assert_eq!(state.count(&name(collection)), expected.len(), "{context}");
    }
}

async fn run(actions: Vec<Action>) {
    let cluster = Cluster::new().await;
    let mut writer = cluster.writer().await;
    let follower = cluster.open().await;
    let mut model = Model::new();

    for (step, action) in actions.into_iter().enumerate() {
        let context = format!("step {step}: {action:?}");
        match action {
            Action::Commit(changes) => {
                let next = changes
                    .iter()
                    .try_fold(model.clone(), |m, change| change.applied(m));
                let outcome = writer
                    .commit(changes.iter().map(Change::mutation).collect())
                    .await;
                match next {
                    Some(next) => {
                        outcome.unwrap_or_else(|e| panic!("{context}: {e}"));
                        model = next;
                    }
                    None => assert!(outcome.is_err(), "{context}"),
                }
            }
            Action::Flush => {
                writer.flush().await.unwrap();
            }
            Action::Restart => writer = cluster.writer().await,
            Action::Takeover => {
                let stalled = std::mem::replace(&mut writer, cluster.writer().await);
                let attempt = stalled
                    .commit(vec![create("a"), Mutation::DropCollection(name("a"))])
                    .await;
                let attempt = match attempt {
                    Err(StorageError::CollectionExists(_)) => {
                        stalled
                            .commit(vec![Mutation::DropCollection(name("a")), create("a")])
                            .await
                    }
                    other => other,
                };
                assert!(
                    matches!(
                        attempt,
                        Err(StorageError::Coordinator(CoordinatorError::Fenced { .. }))
                    ),
                    "{context}: {attempt:?}"
                );
            }
            Action::Follow => {
                follower.catch_up(&cluster.committed().await).await.unwrap();
                follower.read(|state| assert_matches_model(state, &model, &context));
            }
        }
        writer
            .partition()
            .read(|state| assert_matches_model(state, &model, &context));
        assert_eq!(
            writer.partition().read(State::position),
            cluster.committed().await.log_position
        );
    }

    cluster
        .open()
        .await
        .read(|state| assert_matches_model(state, &model, "reopened at the end"));
    follower.catch_up(&cluster.committed().await).await.unwrap();
    follower.read(|state| assert_matches_model(state, &model, "follower at the end"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn any_history_of_commits_flushes_restarts_and_takeovers_matches_a_plain_map(
        actions in prop::collection::vec(action(), 1..40)
    ) {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(run(actions));
    }
}
