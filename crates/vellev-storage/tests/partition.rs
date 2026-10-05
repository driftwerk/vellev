mod common;

use common::*;

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
        .unwrap()
        .position;
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
        .unwrap()
        .position;

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
    let mut resized = schema("notes");
    resized.embedding.dimensions = 3;
    type Expected = fn(&StorageError) -> bool;
    let rejected: Vec<(Vec<Mutation>, Expected)> = vec![
        (vec![upsert("notes", 2, 2), bad_type], |e| {
            matches!(e, StorageError::Object(_))
        }),
        (vec![wrong_vector], |e| matches!(e, StorageError::Object(_))),
        (vec![upsert("notes", 2, 2), upsert("nope", 1, 1)], |e| {
            matches!(e, StorageError::UnknownCollection(_))
        }),
        (vec![Mutation::CreateCollection(Box::new(resized))], |e| {
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
        writer
            .commit(vec![upsert("notes", 2, 2)])
            .await
            .unwrap()
            .position,
        LogPosition(position.0 + 1)
    );
}

#[tokio::test]
async fn a_commit_or_flush_with_nothing_to_do_writes_nothing() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    assert_eq!(writer.flush().await.unwrap(), LogPosition(0));
    assert_eq!(
        writer.commit(vec![]).await.unwrap().position,
        LogPosition(0)
    );
    assert_eq!(cluster.committed().await.manifest, None);

    let position = writer.commit(vec![create("notes")]).await.unwrap().position;
    let unchanged = Mutation::AddProperty {
        collection: name("notes"),
        property: Property::new("n", DataType::Int),
    };
    assert_eq!(
        writer.commit(vec![unchanged]).await.unwrap().position,
        position
    );
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
    let position = writer
        .commit(vec![upsert("notes", 3, 3)])
        .await
        .unwrap()
        .position;
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

    let position = successor
        .commit(vec![upsert("notes", 4, 4)])
        .await
        .unwrap()
        .position;
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
        .unwrap()
        .position;
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
    let position = successor
        .commit(vec![upsert("notes", 1, 1)])
        .await
        .unwrap()
        .position;
    stalled
        .commit(vec![upsert("notes", 1, -1)])
        .await
        .unwrap_err();

    let orphan = Layout(&cluster.key).log(position, stalled.author());
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
    let position = writer
        .commit(vec![upsert("notes", 2, 2)])
        .await
        .unwrap()
        .position;

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
    cached.put("a/../../planted", frame.clone()).await.unwrap();
    assert!(!outside.path().join("planted").exists());

    cached.put("a/b", frame).await.unwrap();
    for alias in ["a/./b", "a//b", "a/b/", "./a/b"] {
        let through_cache = cached.get(alias).await.is_ok();
        assert_eq!(through_cache, store.get(alias).await.is_ok(), "{alias}");
    }
}

#[tokio::test]
async fn a_writer_reopened_on_a_live_lease_cannot_replace_a_commit_that_landed_late() {
    let cluster = Cluster::new().await;
    let unreliable = Arc::new(Unreliable {
        inner: cluster.coordinator.clone(),
        lose_responses: AtomicUsize::new(0),
        refuse_commits: AtomicUsize::new(0),
        hang_after_commit: AtomicBool::new(false),
    });
    let (lease, committed) = cluster.lease().await;
    let first = Writer::open(
        cluster.blobs.clone(),
        unreliable.clone(),
        lease.clone(),
        &committed,
    )
    .await
    .unwrap();
    first.commit(vec![create("notes")]).await.unwrap();

    unreliable
        .refuse_commits
        .store(usize::MAX, Ordering::SeqCst);
    first
        .commit(vec![upsert("notes", 1, 111)])
        .await
        .unwrap_err();
    unreliable.refuse_commits.store(0, Ordering::SeqCst);

    let before = cluster.committed().await;
    let position = before.log_position.next().unwrap();
    let late = Layout(&cluster.key).log(position, first.author());
    let second = Writer::open(
        cluster.blobs.clone(),
        unreliable.clone(),
        lease.clone(),
        &before,
    )
    .await
    .unwrap();
    assert_ne!(Layout(&cluster.key).log(position, second.author()), late);

    let landed = Commit {
        log_position: position,
        manifest: Some(ManifestRef(late.clone())),
    };
    cluster.coordinator.commit(&lease, landed).await.unwrap();
    let follower = cluster.open().await;

    assert!(matches!(
        second
            .commit(vec![upsert("notes", 1, 222), upsert("notes", 2, 2)])
            .await,
        Err(StorageError::Coordinator(
            CoordinatorError::Regressed { .. }
        ))
    ));
    assert_eq!(cluster.committed().await.manifest, Some(ManifestRef(late)));
    for copy in [follower, cluster.open().await] {
        copy.partition().read(|state| {
            assert_eq!(n_of(state, "notes", 1), Some(111));
            assert_eq!(n_of(state, "notes", 2), None);
        });
    }
}

#[tokio::test]
async fn a_reading_from_a_removed_history_is_ignored_even_when_it_is_further_along() {
    let cluster = Cluster::new().await;
    let removed = cluster.writer().await;
    removed
        .commit(vec![create("old"), upsert("old", 1, 1)])
        .await
        .unwrap();
    removed.flush().await.unwrap();
    removed.commit(vec![upsert("old", 2, 2)]).await.unwrap();
    let old = cluster.committed().await;

    cluster.coordinator.remove(&cluster.key).await.unwrap();
    cluster.coordinator.register(&cluster.key).await.unwrap();
    let writer = cluster.writer().await;
    let position = writer.commit(vec![create("notes")]).await.unwrap().position;
    assert!(position < old.log_position);

    let follower = cluster.open().await;
    assert_eq!(follower.catch_up(&old).await.unwrap(), position);
    follower
        .partition()
        .read(|state| assert!(state.collection(&name("old")).is_none()));

    let next = writer
        .commit(vec![upsert("notes", 1, 1)])
        .await
        .unwrap()
        .position;
    assert_eq!(
        follower.catch_up(&cluster.committed().await).await.unwrap(),
        next
    );
    follower
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}

#[tokio::test]
async fn a_removed_history_is_never_adopted_before_anyone_takes_a_new_lease() {
    let cluster = Cluster::new().await;
    let removed = cluster.writer().await;
    removed
        .commit(vec![create("old"), upsert("old", 1, 1)])
        .await
        .unwrap();
    removed.commit(vec![upsert("old", 2, 2)]).await.unwrap();
    let old = cluster.committed().await;
    let holding_old = cluster.open().await;

    cluster.coordinator.remove(&cluster.key).await.unwrap();
    cluster.coordinator.register(&cluster.key).await.unwrap();
    let registered = cluster.committed().await;
    assert!(registered.generation > old.generation);

    let fresh = cluster.open().await;
    assert_eq!(fresh.catch_up(&old).await.unwrap(), LogPosition(0));
    fresh
        .partition()
        .read(|state| assert!(state.collection(&name("old")).is_none()));

    assert!(matches!(
        holding_old.catch_up(&registered).await,
        Err(StorageError::Diverged { .. })
    ));
}

#[tokio::test]
async fn a_log_that_reuses_a_dropped_collection_id_is_refused() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .commit(vec![Mutation::DropCollection(name("notes"))])
        .await
        .unwrap();

    let head = cluster.committed().await;
    let position = head.log_position.next().unwrap();
    let key = Layout(&cluster.key).log(position, &forger());
    let reusing = LogSegment {
        position,
        prev: head.manifest.map(|head| head.0),
        ops: vec![format::Op::CreateCollection {
            id: CollectionId(0),
            schema: Box::new(schema("notes")),
        }],
    };
    let forgery = vec![(key.clone(), format::encode(Kind::Log, &reusing).unwrap())];
    assert!(matches!(
        forged(&cluster, forgery, position.0).await,
        Err(StorageError::Corrupt { key: k, .. }) if k == key
    ));
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
        manifest: files.last().map(|(key, _)| ManifestRef(key.clone())),
        ..cluster.committed().await
    };
    Follower::open(cluster.key.clone(), cluster.blobs.clone(), &committed).await
}

fn forger() -> Author {
    Author {
        generation: Generation(1),
        writer: uuid::Uuid::nil(),
    }
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
    let key = |position| Layout(&cluster.key).log(LogPosition(position), &forger());
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
        Err(StorageError::Corrupt { key: k, .. }) if k == key(1)
    ));

    let ending_early = vec![(key(3), log(3, None))];
    assert!(matches!(
        forged(&cluster, ending_early, 3).await,
        Err(StorageError::Corrupt { key: k, .. }) if k == key(3)
    ));

    let headless = Committed {
        log_position: LogPosition(4),
        ..cluster.committed().await
    };
    assert!(matches!(
        Follower::open(cluster.key.clone(), cluster.blobs.clone(), &headless).await,
        Err(StorageError::Corrupt { .. })
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
    let segment_key = layout.segment(LogPosition(1), &forger());
    let manifest_key = layout.manifest(LogPosition(1), &forger());
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

    for foreign in [
        "tenant-b/0/segment/1.segment",
        "tenant-a/0/../../tenant-b/0/segment/1.segment",
        "tenant-a/0//x",
        "tenant-a/0/./x",
    ] {
        let trespassing = vec![(manifest_key.clone(), listing(foreign, 0))];
        assert_eq!(refused(trespassing).await, foreign);
    }

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
    Insert(usize, u8, i64),
    Increment(usize, u8, i64),
    Delete(usize, u8),
    /// Delete every row whose `n` is at least this.
    DeleteFrom(usize, i64),
}

#[derive(Debug, Clone)]
enum Action {
    Commit(Vec<Change>),
    Rejected(Vec<Change>),
    Flush,
    Restart,
    Takeover,
    Follow,
    FollowStale(usize),
}

/// Per collection, each row's `n` and the `n` its vector was written with:
/// an increment changes the first and leaves the second.
type Model = BTreeMap<&'static str, BTreeMap<u8, (i64, i64)>>;

impl Change {
    fn mutation(&self) -> Mutation {
        match *self {
            Change::Create(c) => create(COLLECTIONS[c]),
            Change::Drop(c) => Mutation::DropCollection(name(COLLECTIONS[c])),
            Change::Upsert(c, object, n) => upsert(COLLECTIONS[c], object, n),
            Change::Insert(c, object, n) => match upsert(COLLECTIONS[c], object, n) {
                Mutation::Upsert {
                    collection,
                    id,
                    properties,
                    vector,
                } => Mutation::Insert {
                    collection,
                    id,
                    properties,
                    vector,
                },
                other => other,
            },
            Change::Increment(c, object, by) => Mutation::Increment {
                collection: name(COLLECTIONS[c]),
                id: id(object),
                property: "n".into(),
                by: by as f64,
            },
            Change::Delete(c, object) => delete(COLLECTIONS[c], object),
            Change::DeleteFrom(c, least) => Mutation::DeleteWhere {
                collection: name(COLLECTIONS[c]),
                filter: Filter::Gte {
                    property: "n".into(),
                    value: json!(least),
                },
            },
        }
    }

    /// The change that has to come first for this one to be valid, if any.
    fn prerequisite(&self, model: &Model) -> Option<Change> {
        let rows = |c: usize| model.get(COLLECTIONS[c]);
        match *self {
            Change::Create(c) => rows(c).map(|_| Change::Drop(c)),
            Change::Drop(c)
            | Change::Upsert(c, ..)
            | Change::Delete(c, _)
            | Change::DeleteFrom(c, _) => rows(c).is_none().then_some(Change::Create(c)),
            Change::Insert(c, object, _) => match rows(c) {
                None => Some(Change::Create(c)),
                Some(rows) => rows
                    .contains_key(&object)
                    .then_some(Change::Delete(c, object)),
            },
            Change::Increment(c, object, _) => match rows(c) {
                None => Some(Change::Create(c)),
                Some(rows) => (!rows.contains_key(&object)).then_some(Change::Upsert(c, object, 0)),
            },
        }
    }

    /// Apply the change to the model and say whether storage logs anything
    /// for it: a change that leaves the row as it was is valid and not logged.
    fn applied(&self, model: &mut Model) -> bool {
        match *self {
            Change::Create(c) => {
                model.insert(COLLECTIONS[c], BTreeMap::new());
                true
            }
            Change::Drop(c) => {
                model.remove(COLLECTIONS[c]);
                true
            }
            Change::Upsert(c, object, n) | Change::Insert(c, object, n) => {
                model
                    .get_mut(COLLECTIONS[c])
                    .unwrap()
                    .insert(object, (n, n))
                    != Some((n, n))
            }
            Change::Increment(c, object, by) => {
                let rows = model.get_mut(COLLECTIONS[c]).unwrap();
                rows.get_mut(&object).unwrap().0 += by;
                by != 0
            }
            Change::Delete(c, object) => model
                .get_mut(COLLECTIONS[c])
                .unwrap()
                .remove(&object)
                .is_some(),
            Change::DeleteFrom(c, least) => {
                let rows = model.get_mut(COLLECTIONS[c]).unwrap();
                let before = rows.len();
                rows.retain(|_, (n, _)| *n < least);
                rows.len() < before
            }
        }
    }
}

/// Make every change valid for the model it meets by putting what it needs
/// in front of it, so that generated histories spend their time writing rows
/// rather than being rejected. Returns the batch and whether it logs anything.
fn valid_batch(changes: Vec<Change>, model: &mut Model) -> (Vec<Change>, bool) {
    let mut logged = false;
    let mut batch = Vec::new();
    for change in changes {
        while let Some(first) = change.prerequisite(model) {
            logged |= first.applied(model);
            batch.push(first);
        }
        logged |= change.applied(model);
        batch.push(change);
    }
    (batch, logged)
}

fn change() -> impl Strategy<Value = Change> {
    let collection = || 0..COLLECTIONS.len();
    prop_oneof![
        1 => collection().prop_map(Change::Create),
        1 => collection().prop_map(Change::Drop),
        8 => (collection(), 0..IDS, -3i64..4).prop_map(|(c, o, n)| Change::Upsert(c, o, n)),
        3 => (collection(), 0..IDS, -3i64..4).prop_map(|(c, o, n)| Change::Insert(c, o, n)),
        4 => (collection(), 0..IDS, -3i64..4).prop_map(|(c, o, n)| Change::Increment(c, o, n)),
        5 => (collection(), 0..IDS).prop_map(|(c, o)| Change::Delete(c, o)),
        2 => (collection(), -3i64..5).prop_map(|(c, least)| Change::DeleteFrom(c, least)),
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
        1 => any::<usize>().prop_map(Action::FollowStale),
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
                expected.get(&object).map(|(n, _)| *n),
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
    for (collection, rows) in model {
        for wanted in -3i64..4 {
            let filter = Filter::Gte {
                property: "n".into(),
                value: json!(wanted),
            };
            let mut selected: Vec<ObjectId> = state
                .select(&name(collection), &filter)
                .unwrap()
                .iter()
                .map(|o| o.id)
                .collect();
            selected.sort_unstable();
            let expected: Vec<ObjectId> = rows
                .iter()
                .filter(|(_, (n, _))| *n >= wanted)
                .map(|(object, _)| id(*object))
                .collect();
            assert_eq!(
                selected, expected,
                "{context}: n >= {wanted} in {collection}"
            );
        }
    }
}

async fn run(actions: Vec<Action>) {
    let cluster = Cluster::new().await;
    let mut writer = cluster.writer().await;
    let follower = cluster.open().await;
    let mut model = Model::new();
    let mut readings: Vec<Committed> = Vec::new();

    for (step, action) in actions.into_iter().enumerate() {
        let context = format!("step {step}: {action:?}");
        match action {
            Action::Commit(changes) => {
                let (batch, logged) = valid_batch(changes, &mut model);
                let before = writer.partition().read(State::position);
                let expected = if logged {
                    before.next().unwrap()
                } else {
                    before
                };
                let receipt = writer
                    .commit(batch.iter().map(Change::mutation).collect())
                    .await;
                assert_eq!(receipt.unwrap().position, expected, "{context}");
            }
            Action::Rejected(changes) => {
                let (batch, _) = valid_batch(changes, &mut model.clone());
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
                let committed = cluster.committed().await;
                follower.catch_up(&committed).await.unwrap();
                follower
                    .partition()
                    .read(|state| assert_matches_model(state, &model, &context));
                readings.push(committed);
            }
            Action::FollowStale(pick) => {
                if let Some(stale) = readings.get(pick % readings.len().max(1)) {
                    let position = follower.partition().read(State::position);
                    assert_eq!(
                        follower.catch_up(stale).await.unwrap(),
                        position,
                        "{context}"
                    );
                }
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
