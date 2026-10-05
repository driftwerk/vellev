use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
use vellev_storage::blobs::{Blobs, CachedBlobs, Layout, ObjectBlobs};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Lease, LogPosition, ManualClock,
    MemoryCoordinator, PartitionKey,
};
use vellev_storage::format::{
    self, CollectionId, Kind, LogSegment, Manifest, Row, SegmentFile, SegmentMeta,
};
use vellev_storage::{Follower, Mutation, State, StorageError, Writer};

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

#[derive(Debug)]
struct Observed {
    inner: Arc<dyn Blobs>,
    gets: Mutex<Vec<String>>,
    fail_puts: AtomicBool,
}

impl Observed {
    fn fetched(&self) -> Vec<String> {
        std::mem::take(&mut self.gets.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Observed {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.fail_puts.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.gets.lock().unwrap().push(key.to_string());
        self.inner.get(key).await
    }
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

    async fn open(&self) -> Follower {
        self.open_on(self.blobs.clone()).await.unwrap()
    }

    async fn open_on(&self, blobs: Arc<dyn Blobs>) -> vellev_storage::Result<Follower> {
        Follower::open(self.key.clone(), blobs, &self.committed().await).await
    }

    /// The store wrapped so that reads are recorded and writes can be failed.
    fn observed(&self) -> Arc<Observed> {
        Arc::new(Observed {
            inner: self.blobs.clone(),
            gets: Mutex::default(),
            fail_puts: AtomicBool::new(false),
        })
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
    reopened.partition().read(|state| {
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
        cluster.open().await.partition().read(|state| {
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
    cluster.open().await.partition().read(expect);
    writer.flush().await.unwrap();
    cluster.open().await.partition().read(expect);
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
    cluster.open().await.partition().read(expect);
    writer.flush().await.unwrap();
    cluster.open().await.partition().read(expect);
}

#[tokio::test]
async fn a_follower_catches_up_across_commits_and_checkpoints() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let follower = cluster.open().await;
    assert_eq!(follower.partition().read(State::position), LogPosition(0));

    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 2));

    writer.flush().await.unwrap();
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    writer.flush().await.unwrap();
    let position = writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    assert_eq!(
        follower.catch_up(&cluster.committed().await).await.unwrap(),
        position
    );
    follower.partition().read(|state| {
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
    cluster.open().await.partition().read(|state| {
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
    hang_after_commit: AtomicBool,
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
        if self.hang_after_commit.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
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
        hang_after_commit: AtomicBool::new(false),
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
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 2), Some(20)));
}

#[tokio::test]
async fn a_write_the_caller_stopped_waiting_for_closes_the_writer() {
    let cluster = Cluster::new().await;
    let unreliable = Arc::new(Unreliable {
        inner: cluster.coordinator.clone(),
        lose_responses: AtomicUsize::new(0),
        refuse_commits: AtomicUsize::new(0),
        hang_after_commit: AtomicBool::new(false),
    });
    let writer = cluster.writer_on(unreliable.clone()).await;
    writer.commit(vec![create("notes")]).await.unwrap();

    unreliable.hang_after_commit.store(true, Ordering::SeqCst);
    let abandoned = tokio::time::timeout(
        Duration::from_millis(50),
        writer.commit(vec![upsert("notes", 1, 1)]),
    )
    .await;
    assert!(abandoned.is_err());
    unreliable.hang_after_commit.store(false, Ordering::SeqCst);

    let head = cluster.committed().await;
    for batch in [vec![upsert("notes", 2, 2)], vec![], vec![create("notes")]] {
        assert!(matches!(
            writer.commit(batch).await,
            Err(StorageError::Closed)
        ));
    }
    assert!(matches!(writer.flush().await, Err(StorageError::Closed)));
    assert_eq!(cluster.committed().await, head);
    cluster.open().await.partition().read(|state| {
        assert_eq!(n_of(state, "notes", 1), Some(1));
        assert_eq!(n_of(state, "notes", 2), None);
    });
}

#[tokio::test]
async fn a_failed_upload_closes_the_writer() {
    let cluster = Cluster::new().await;
    let store = cluster.observed();
    let (lease, committed) = cluster.lease().await;
    let writer = Writer::open(
        store.clone(),
        cluster.coordinator.clone(),
        lease,
        &committed,
    )
    .await
    .unwrap();
    writer.commit(vec![create("notes")]).await.unwrap();

    store.fail_puts.store(true, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(vec![upsert("notes", 1, 1)]).await,
        Err(StorageError::Blob { .. })
    ));
    store.fail_puts.store(false, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(vec![upsert("notes", 2, 2)]).await,
        Err(StorageError::Closed)
    ));
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 0));
}

#[tokio::test]
async fn a_fenced_writer_cannot_checkpoint() {
    let cluster = Cluster::new().await;
    let stalled = cluster.writer().await;
    stalled
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let successor = cluster.writer().await;
    let head = cluster.committed().await;

    assert!(matches!(
        stalled.flush().await,
        Err(StorageError::Coordinator(CoordinatorError::Fenced { .. }))
    ));
    assert_eq!(cluster.committed().await, head);
    successor.flush().await.unwrap();
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}

#[tokio::test]
async fn an_orphan_at_a_committed_position_is_never_read() {
    let cluster = Cluster::new().await;
    let stalled = cluster.writer().await;
    stalled.commit(vec![create("notes")]).await.unwrap();
    let successor = cluster.writer().await;
    let position = successor.commit(vec![upsert("notes", 1, 1)]).await.unwrap();
    stalled
        .commit(vec![upsert("notes", 1, -1)])
        .await
        .unwrap_err();

    let orphan = Layout(&cluster.key).log(position, stalled.lease().generation);
    cluster
        .blobs
        .get(&orphan)
        .await
        .expect("the fenced writer did upload its file");

    let store = cluster.observed();
    let reopened = cluster.open_on(store.clone()).await.unwrap();
    assert!(!store.fetched().contains(&orphan));
    reopened
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}

#[tokio::test]
async fn a_follower_fetches_only_what_it_does_not_already_hold() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();

    let store = cluster.observed();
    let follower = cluster.open_on(store.clone()).await.unwrap();
    assert_eq!(store.fetched().len(), 2, "a manifest and its segment");

    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    assert_eq!(store.fetched().len(), 1, "one log segment");

    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    writer.flush().await.unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    assert_eq!(
        store.fetched().len(),
        3,
        "the newest manifest and the two segments it added"
    );
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 3));
}

#[tokio::test]
async fn an_older_reading_of_the_committed_state_is_ignored() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let before_checkpoint = cluster.committed().await;
    writer.flush().await.unwrap();
    let older = cluster.committed().await;
    let position = writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

    let follower = cluster.open().await;
    for stale in [&older, &before_checkpoint] {
        assert_eq!(follower.catch_up(stale).await.unwrap(), position);
        follower.partition().read(|state| {
            assert_eq!(state.position(), position);
            assert_eq!(n_of(state, "notes", 2), Some(2));
        });
    }
}

#[tokio::test]
async fn a_follower_of_a_replaced_partition_is_told_to_reopen() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let follower = cluster.open().await;

    cluster.coordinator.remove(&cluster.key).await.unwrap();
    cluster.coordinator.register(&cluster.key).await.unwrap();
    let replacement = cluster.writer().await;
    assert!(matches!(
        follower.catch_up(&cluster.committed().await).await,
        Err(StorageError::Diverged { .. })
    ));
    replacement.commit(vec![create("other")]).await.unwrap();
    assert!(matches!(
        follower.catch_up(&cluster.committed().await).await,
        Err(StorageError::Diverged { .. })
    ));
    follower
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}

#[tokio::test]
async fn a_checkpoint_is_written_for_schema_only_history_and_holds_no_empty_segment() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    writer.commit(vec![upsert("notes", 1, 1)]).await.unwrap();
    writer
        .commit(vec![
            Mutation::DropCollection(name("notes")),
            create("other"),
        ])
        .await
        .unwrap();
    let position = writer.flush().await.unwrap();

    let head = cluster.committed().await.manifest.unwrap().0;
    let manifest: Manifest = format::decode(
        &head,
        Kind::Manifest,
        &cluster.blobs.get(&head).await.unwrap(),
    )
    .unwrap();
    assert_eq!(manifest.position, position);
    assert_eq!(manifest.segments, vec![]);
    assert_eq!(manifest.collections.len(), 1);
    cluster.open().await.partition().read(|state| {
        assert!(state.collection(&name("notes")).is_none());
        assert!(state.collection(&name("other")).is_some());
    });
}

#[tokio::test]
async fn files_are_read_from_the_store_once_and_a_damaged_cache_entry_is_refetched() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let store = cluster.observed();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), dir.path()));
    let count = |follower: Follower| {
        follower
            .partition()
            .read(|state| state.count(&name("notes")))
    };

    assert_eq!(count(cluster.open_on(cached.clone()).await.unwrap()), 2);
    assert_eq!(
        store.fetched().len(),
        3,
        "a log segment, a manifest and a segment"
    );
    assert_eq!(count(cluster.open_on(cached.clone()).await.unwrap()), 2);
    assert_eq!(store.fetched(), Vec::<String>::new());

    let head = cluster.committed().await.manifest.unwrap().0;
    std::fs::write(dir.path().join(&head), b"").unwrap();
    assert_eq!(count(cluster.open_on(cached.clone()).await.unwrap()), 2);
    assert_eq!(store.fetched(), vec![head]);
    assert_eq!(count(cluster.open_on(cached).await.unwrap()), 2);
    assert_eq!(store.fetched(), Vec::<String>::new());
}

#[tokio::test]
async fn the_cache_never_reads_or_writes_outside_its_directory() {
    let outside = tempfile::tempdir().unwrap();
    let dir = outside.path().join("cache");
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let cached = CachedBlobs::new(store.clone(), &dir);
    let frame = format::encode(
        Kind::Segment,
        &SegmentFile {
            position: LogPosition(1),
            rows: vec![],
            tombstones: vec![],
        },
    )
    .unwrap();
    std::fs::write(outside.path().join("secret"), &frame).unwrap();

    assert!(matches!(
        cached.get("../secret").await,
        Err(StorageError::Missing { .. })
    ));
    let absolute = outside.path().join("secret");
    assert!(cached.get(absolute.to_str().unwrap()).await.is_err());
    cached.put("a/../../planted", frame).await.unwrap();
    assert!(!outside.path().join("planted").exists());
}

/// A partition whose committed head is a hand-made file.
async fn forged(
    cluster: &Cluster,
    files: Vec<(String, Bytes)>,
    position: u64,
) -> vellev_storage::Result<Follower> {
    for (key, frame) in &files {
        cluster.blobs.put(key, frame.clone()).await.unwrap();
    }
    let committed = Committed {
        log_position: LogPosition(position),
        manifest: files
            .last()
            .map(|(key, _)| vellev_storage::coordinator::ManifestRef(key.clone())),
        ..cluster.committed().await
    };
    Follower::open(cluster.key.clone(), cluster.blobs.clone(), &committed).await
}

fn manifest_of(segments: Vec<SegmentMeta>) -> Manifest {
    Manifest {
        position: LogPosition(1),
        next_collection_id: 1,
        collections: vec![(CollectionId(0), schema("notes"))],
        segments,
    }
}

#[tokio::test]
async fn a_history_that_loops_or_skips_a_position_is_refused() {
    let cluster = Cluster::new().await;
    let key = |position| {
        Layout(&cluster.key).log(
            LogPosition(position),
            vellev_storage::coordinator::Generation(1),
        )
    };
    let log = |position, prev: Option<String>| {
        format::encode(
            Kind::Log,
            &LogSegment {
                position: LogPosition(position),
                prev,
                ops: vec![],
            },
        )
        .unwrap()
    };

    let looping = vec![(key(1), log(1, Some(key(1))))];
    assert!(matches!(
        forged(&cluster, looping, 1).await,
        Err(StorageError::Diverged { .. })
    ));

    let skipping = vec![(key(1), log(1, None)), (key(3), log(3, Some(key(1))))];
    assert!(matches!(
        forged(&cluster, skipping, 3).await,
        Err(StorageError::Corrupt { key: k, .. }) if k == key(1)
    ));

    let mislabelled = vec![(key(2), log(2, None))];
    assert!(matches!(
        forged(&cluster, mislabelled, 5).await,
        Err(StorageError::Corrupt { key: k, .. }) if k == key(2)
    ));
}

#[tokio::test]
async fn a_checkpoint_that_contradicts_itself_is_refused() {
    let cluster = Cluster::new().await;
    let layout = Layout(&cluster.key);
    let generation = vellev_storage::coordinator::Generation(1);
    let segment_key = layout.segment(LogPosition(1), generation);
    let manifest_key = layout.manifest(LogPosition(1), generation);
    let row = |object: u8| Row {
        collection: CollectionId(0),
        object: vellev_core::Object {
            id: id(object),
            properties: Default::default(),
            vector: None,
        },
    };
    let segment = |rows: Vec<Row>| {
        format::encode(
            Kind::Segment,
            &SegmentFile {
                position: LogPosition(1),
                rows,
                tombstones: vec![],
            },
        )
        .unwrap()
    };
    let listing = |key: &str, rows: u64| {
        format::encode(
            Kind::Manifest,
            &manifest_of(vec![SegmentMeta {
                key: key.into(),
                rows,
                tombstones: 0,
            }]),
        )
        .unwrap()
    };
    let refused = |files: Vec<(String, Bytes)>| async {
        match forged(&cluster, files, 1).await {
            Err(StorageError::Corrupt { key, .. }) => key,
            other => panic!("{other:?}"),
        }
    };

    let repeated = vec![
        (segment_key.clone(), segment(vec![row(1), row(1)])),
        (manifest_key.clone(), listing(&segment_key, 2)),
    ];
    assert_eq!(refused(repeated).await, segment_key);

    let miscounted = vec![
        (segment_key.clone(), segment(vec![row(1)])),
        (manifest_key.clone(), listing(&segment_key, 2)),
    ];
    assert_eq!(refused(miscounted).await, segment_key);

    let foreign = "tenant-b/0/segment/1.segment";
    let trespassing = vec![(manifest_key.clone(), listing(foreign, 0))];
    assert_eq!(refused(trespassing).await, foreign);

    let mut duplicate_ids = manifest_of(vec![]);
    duplicate_ids
        .collections
        .push((CollectionId(0), schema("other")));
    let duplicated = vec![(
        manifest_key.clone(),
        format::encode(Kind::Manifest, &duplicate_ids).unwrap(),
    )];
    assert_eq!(refused(duplicated).await, manifest_key);

    let mut reusable_id = manifest_of(vec![]);
    reusable_id.next_collection_id = 0;
    let reusable = vec![(
        manifest_key.clone(),
        format::encode(Kind::Manifest, &reusable_id).unwrap(),
    )];
    assert_eq!(refused(reusable).await, manifest_key);
}

#[tokio::test]
async fn a_damaged_or_missing_file_stops_recovery_with_its_name() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();

    let store = cluster.observed();
    cluster.open_on(store.clone()).await.unwrap();
    let files = store.fetched();
    assert_eq!(files.len(), 4, "two log segments, a manifest and a segment");

    for file in &files {
        let healthy = cluster.blobs.get(file).await.unwrap();
        let mut damaged = healthy.to_vec();
        *damaged.last_mut().unwrap() ^= 0xff;
        cluster.blobs.put(file, Bytes::from(damaged)).await.unwrap();
        assert!(
            matches!(cluster.open_on(cluster.blobs.clone()).await, Err(StorageError::Corrupt { key, .. }) if key == *file),
            "{file}"
        );
        cluster.blobs.put(file, healthy).await.unwrap();
    }
    cluster.open().await;

    let empty: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    assert!(matches!(
        cluster.open_on(empty).await,
        Err(StorageError::Missing { key }) if key == files[0]
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
    Rejected(Vec<Change>),
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

    fn collection(&self) -> usize {
        match *self {
            Change::Create(c) | Change::Drop(c) | Change::Upsert(c, ..) | Change::Delete(c, _) => c,
        }
    }

    fn applied(&self, model: &mut Model) {
        match *self {
            Change::Create(c) => {
                model.insert(COLLECTIONS[c], BTreeMap::new());
            }
            Change::Drop(c) => {
                model.remove(COLLECTIONS[c]);
            }
            Change::Upsert(c, object, n) => {
                model.get_mut(COLLECTIONS[c]).unwrap().insert(object, n);
            }
            Change::Delete(c, object) => {
                model.get_mut(COLLECTIONS[c]).unwrap().remove(&object);
            }
        }
    }
}

/// Make every change valid for the model it meets, by first creating a
/// missing collection or dropping one that is in the way, so that generated
/// histories spend their time writing rows rather than being rejected.
fn valid_batch(changes: Vec<Change>, model: &mut Model) -> Vec<Change> {
    changes
        .into_iter()
        .flat_map(|change| {
            let c = change.collection();
            let exists = model.contains_key(COLLECTIONS[c]);
            let prelude = match (&change, exists) {
                (Change::Create(_), true) => Some(Change::Drop(c)),
                (Change::Create(_), false) => None,
                (_, false) => Some(Change::Create(c)),
                (_, true) => None,
            };
            let steps: Vec<Change> = prelude.into_iter().chain([change]).collect();
            steps.iter().for_each(|step| step.applied(model));
            steps
        })
        .collect()
}

fn change() -> impl Strategy<Value = Change> {
    let collection = 0..COLLECTIONS.len();
    prop_oneof![
        1 => collection.clone().prop_map(Change::Create),
        1 => collection.clone().prop_map(Change::Drop),
        12 => (collection.clone(), 0..IDS, -3i64..4).prop_map(|(c, o, n)| Change::Upsert(c, o, n)),
        5 => (collection, 0..IDS).prop_map(|(c, o)| Change::Delete(c, o)),
    ]
}

fn action() -> impl Strategy<Value = Action> {
    let changes = || prop::collection::vec(change(), 1..5);
    prop_oneof![
        10 => changes().prop_map(Action::Commit),
        1 => changes().prop_map(Action::Rejected),
        4 => Just(Action::Flush),
        2 => Just(Action::Restart),
        2 => Just(Action::Takeover),
        3 => Just(Action::Follow),
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
    assert_eq!(state.collections().count(), model.len(), "{context}");
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
                let batch = valid_batch(changes, &mut model);
                let position = writer.partition().read(State::position);
                let committed = writer
                    .commit(batch.iter().map(Change::mutation).collect())
                    .await;
                assert_eq!(committed.unwrap(), position.next().unwrap(), "{context}");
            }
            Action::Rejected(changes) => {
                let batch = valid_batch(changes, &mut model.clone());
                let mutations = batch
                    .iter()
                    .map(Change::mutation)
                    .chain([upsert("missing", 0, 0)]);
                assert!(
                    matches!(
                        writer.commit(mutations.collect()).await,
                        Err(StorageError::UnknownCollection(_))
                    ),
                    "{context}"
                );
            }
            Action::Flush => {
                writer.flush().await.unwrap();
            }
            Action::Restart => writer = cluster.writer().await,
            Action::Takeover => {
                let stalled = std::mem::replace(&mut writer, cluster.writer().await);
                let attempt = stalled.commit(vec![create("fence_probe")]).await;
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
                follower
                    .partition()
                    .read(|state| assert_matches_model(state, &model, &context));
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
        .partition()
        .read(|state| assert_matches_model(state, &model, "reopened at the end"));
    follower.catch_up(&cluster.committed().await).await.unwrap();
    follower
        .partition()
        .read(|state| assert_matches_model(state, &model, "follower at the end"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn any_history_of_commits_flushes_restarts_and_takeovers_matches_a_plain_map(
        actions in prop::collection::vec(action(), 1..60)
    ) {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(run(actions));
    }
}
