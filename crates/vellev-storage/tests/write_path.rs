mod common;

use common::*;
use serde_json::Value as Json;
use vellev_storage::{Condition, GroupCommitter, Outcome, Receipt};

fn object(properties: Json) -> serde_json::Map<String, Json> {
    properties.as_object().unwrap().clone()
}

fn insert(object_id: u8, properties: Json) -> Mutation {
    Mutation::Insert {
        collection: name("notes"),
        id: id(object_id),
        properties: object(properties),
        vector: None,
    }
}

fn patch(object_id: u8, properties: Json, only_if: Option<(&str, Json)>) -> Mutation {
    Mutation::Patch {
        collection: name("notes"),
        id: id(object_id),
        properties: object(properties),
        vector: None,
        only_if: only_if.map(|(property, equals)| Condition {
            property: property.into(),
            equals,
        }),
    }
}

fn increment(object_id: u8, property: &str, by: f64) -> Mutation {
    Mutation::Increment {
        collection: name("notes"),
        id: id(object_id),
        property: property.into(),
        by,
    }
}

fn append(object_id: u8, values: &[&str], unique: bool) -> Mutation {
    Mutation::Append {
        collection: name("notes"),
        id: id(object_id),
        property: "tags".into(),
        values: values.iter().map(|v| v.to_string()).collect(),
        unique,
    }
}

fn property(state: &State, object_id: u8, property: &str) -> Option<Value> {
    state
        .get(&name("notes"), id(object_id))?
        .properties
        .get(property)
        .cloned()
}

fn tags(values: &[&str]) -> Option<Value> {
    Some(Value::TextArray(
        values.iter().map(|v| v.to_string()).collect(),
    ))
}

async fn notes() -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 10)])
        .await
        .unwrap();
    (cluster, writer)
}

/// The batch is refused with an error `expected` accepts, and nothing changes.
async fn refused(
    cluster: &Cluster,
    writer: &Writer,
    batch: Vec<Mutation>,
    expected: fn(&StorageError) -> bool,
) {
    let before = cluster.committed().await;
    let error = writer.commit(batch).await.unwrap_err();
    assert!(expected(&error), "{error:?}");
    assert_eq!(cluster.committed().await, before);
    assert_eq!(
        writer.partition().read(State::position),
        before.log_position
    );
}

#[tokio::test]
async fn insert_writes_only_when_the_object_is_absent() {
    let (cluster, writer) = notes().await;
    writer
        .commit(vec![insert(2, json!({ "n": 2 }))])
        .await
        .unwrap();
    refused(
        &cluster,
        &writer,
        vec![insert(2, json!({ "n": 99 }))],
        |e| matches!(e, StorageError::ObjectExists { .. }),
    )
    .await;
    refused(
        &cluster,
        &writer,
        vec![insert(3, json!({})), insert(3, json!({}))],
        |e| matches!(e, StorageError::ObjectExists { .. }),
    )
    .await;
    writer
        .commit(vec![delete("notes", 2), insert(2, json!({ "n": 3 }))])
        .await
        .unwrap();
    cluster.open().await.partition().read(|state| {
        assert_eq!(n_of(state, "notes", 2), Some(3));
        assert_eq!(n_of(state, "notes", 3), None);
    });
}

#[tokio::test]
async fn patch_merges_removes_and_keeps_the_rest() {
    let (cluster, writer) = notes().await;
    let receipt = writer
        .commit(vec![patch(
            1,
            json!({ "title": null, "score": 0.5, "tags": ["a"] }),
            None,
        )])
        .await
        .unwrap();
    assert_eq!(receipt.outcomes, vec![Outcome::Applied]);
    cluster.open().await.partition().read(|state| {
        assert_eq!(property(state, 1, "title"), None);
        assert_eq!(property(state, 1, "score"), Some(Value::Number(0.5)));
        assert_eq!(property(state, 1, "tags"), tags(&["a"]));
        assert_eq!(n_of(state, "notes", 1), Some(10));
        assert_eq!(
            state.get(&name("notes"), id(1)).unwrap().vector,
            Some(vec![10.0, 0.5])
        );
    });

    let revector = Mutation::Patch {
        collection: name("notes"),
        id: id(1),
        properties: Default::default(),
        vector: Some(vec![1.0, 2.0]),
        only_if: None,
    };
    writer.commit(vec![revector]).await.unwrap();
    writer.partition().read(|state| {
        assert_eq!(
            state.get(&name("notes"), id(1)).unwrap().vector,
            Some(vec![1.0, 2.0])
        );
    });

    refused(
        &cluster,
        &writer,
        vec![patch(9, json!({ "n": 1 }), None)],
        |e| matches!(e, StorageError::UnknownObject { .. }),
    )
    .await;
    refused(
        &cluster,
        &writer,
        vec![patch(1, json!({ "n": "ten" }), None)],
        |e| matches!(e, StorageError::Object(_)),
    )
    .await;
}

#[tokio::test]
async fn a_mutation_that_changes_nothing_is_not_logged() {
    let (cluster, writer) = notes().await;
    writer.commit(vec![append(1, &["a"], true)]).await.unwrap();
    let before = cluster.committed().await;
    let stored = writer
        .partition()
        .read(|state| state.get(&name("notes"), id(1)).cloned().unwrap());
    let identical = Mutation::Upsert {
        collection: name("notes"),
        id: id(1),
        properties: stored.properties_json(),
        vector: stored.vector.clone(),
    };
    let unchanged = vec![
        identical,
        patch(1, json!({ "n": 10 }), None),
        patch(1, json!({ "score": null }), None),
        append(1, &["a"], true),
        append(1, &[], false),
        delete("notes", 7),
        increment(1, "n", 0.0),
        increment(1, "score", -0.0),
    ];
    let receipt = writer.commit(unchanged).await.unwrap();
    assert_eq!(receipt.position, before.log_position);
    assert_eq!(receipt.outcomes[..6], vec![Outcome::Unchanged; 6]);
    assert_eq!(
        receipt.outcomes[6..],
        [
            Outcome::Value(Value::Int(10)),
            Outcome::Value(Value::Number(0.0))
        ]
    );
    assert_eq!(cluster.committed().await, before);
    writer.partition().read(|state| {
        assert_eq!(state.get(&name("notes"), id(1)), Some(&stored));
    });
}

#[tokio::test]
async fn a_conditional_patch_applies_only_while_the_condition_holds() {
    let (cluster, writer) = notes().await;
    writer
        .commit(vec![patch(1, json!({ "n": 11 }), Some(("n", json!(10))))])
        .await
        .unwrap();
    writer
        .commit(vec![patch(
            1,
            json!({ "score": 1.0 }),
            Some(("score", Json::Null)),
        )])
        .await
        .unwrap();
    let failed: fn(&StorageError) -> bool = |e| matches!(e, StorageError::ConditionFailed { .. });
    refused(
        &cluster,
        &writer,
        vec![patch(1, json!({ "n": 12 }), Some(("n", json!(10))))],
        failed,
    )
    .await;
    refused(
        &cluster,
        &writer,
        vec![patch(1, json!({ "n": 12 }), Some(("score", Json::Null)))],
        failed,
    )
    .await;
    refused(
        &cluster,
        &writer,
        vec![patch(1, json!({ "n": 12 }), Some(("n", json!("11"))))],
        |e| matches!(e, StorageError::Object(_)),
    )
    .await;
    refused(
        &cluster,
        &writer,
        vec![patch(1, json!({ "n": 12 }), Some(("nope", json!(1))))],
        |e| matches!(e, StorageError::Object(_)),
    )
    .await;
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(11)));
}

#[tokio::test]
async fn increment_adds_to_a_number_and_returns_the_new_value() {
    let (cluster, writer) = notes().await;
    let receipt = writer
        .commit(vec![
            increment(1, "n", 5.0),
            increment(1, "n", -20.0),
            increment(1, "score", 0.25),
            increment(1, "score", 0.5),
        ])
        .await
        .unwrap();
    assert_eq!(
        receipt.outcomes,
        vec![
            Outcome::Value(Value::Int(15)),
            Outcome::Value(Value::Int(-5)),
            Outcome::Value(Value::Number(0.25)),
            Outcome::Value(Value::Number(0.75)),
        ]
    );
    cluster.open().await.partition().read(|state| {
        assert_eq!(n_of(state, "notes", 1), Some(-5));
        assert_eq!(property(state, 1, "score"), Some(Value::Number(0.75)));
    });

    let invalid: fn(&StorageError) -> bool = |e| matches!(e, StorageError::InvalidMutation(_));
    writer
        .commit(vec![patch(1, json!({ "n": i64::MAX }), None)])
        .await
        .unwrap();
    for bad in [
        increment(1, "n", 0.5),
        increment(1, "n", 1e30),
        increment(1, "n", f64::NAN),
        increment(1, "score", f64::INFINITY),
        increment(1, "title", 1.0),
        increment(1, "tags", 1.0),
    ] {
        refused(&cluster, &writer, vec![bad], invalid).await;
    }
    refused(&cluster, &writer, vec![increment(1, "n", 1.0)], |e| {
        matches!(e, StorageError::InvalidMutation(reason) if reason.contains("leaves its range"))
    })
    .await;
    refused(&cluster, &writer, vec![increment(1, "nope", 1.0)], |e| {
        matches!(e, StorageError::Object(_))
    })
    .await;
    refused(&cluster, &writer, vec![increment(9, "n", 1.0)], |e| {
        matches!(e, StorageError::UnknownObject { .. })
    })
    .await;
}

#[tokio::test]
async fn append_extends_a_text_array() {
    let (cluster, writer) = notes().await;
    writer
        .commit(vec![append(1, &["a", "b"], false)])
        .await
        .unwrap();
    writer
        .commit(vec![append(1, &["b", "c", "c"], true)])
        .await
        .unwrap();
    writer.commit(vec![append(1, &["a"], false)]).await.unwrap();
    cluster.open().await.partition().read(|state| {
        assert_eq!(property(state, 1, "tags"), tags(&["a", "b", "c", "a"]));
    });

    let not_an_array = Mutation::Append {
        collection: name("notes"),
        id: id(1),
        property: "title".into(),
        values: vec!["x".into()],
        unique: false,
    };
    refused(&cluster, &writer, vec![not_an_array], |e| {
        matches!(e, StorageError::InvalidMutation(_))
    })
    .await;
    refused(&cluster, &writer, vec![append(9, &["x"], false)], |e| {
        matches!(e, StorageError::UnknownObject { .. })
    })
    .await;
}

#[tokio::test]
async fn several_updates_to_one_object_are_one_atomic_step() {
    let (cluster, writer) = notes().await;
    let recurrence = || {
        vec![
            append(1, &["message-1", "message-2"], true),
            increment(1, "n", 1.0),
            patch(1, json!({ "title": "seen again" }), None),
        ]
    };
    let receipt = writer.commit(recurrence()).await.unwrap();
    assert_eq!(receipt.outcomes[1], Outcome::Value(Value::Int(11)));

    let mut failing = recurrence();
    failing.push(patch(1, json!({}), Some(("title", json!("never")))));
    refused(&cluster, &writer, failing, |e| {
        matches!(e, StorageError::ConditionFailed { .. })
    })
    .await;

    let follower = cluster.open().await;
    for copy in [writer.partition().clone(), follower.partition().clone()] {
        copy.read(|state| {
            assert_eq!(n_of(state, "notes", 1), Some(11));
            assert_eq!(
                property(state, 1, "tags"),
                tags(&["message-1", "message-2"])
            );
            assert_eq!(
                property(state, 1, "title"),
                Some(Value::Text("seen again".into()))
            );
        });
    }
}

#[tokio::test]
async fn a_group_commits_once_and_a_rejected_batch_leaves_no_trace() {
    let (cluster, writer) = notes().await;
    let before = cluster.committed().await.log_position;
    let results = writer
        .commit_group(vec![
            vec![increment(1, "n", 1.0), insert(2, json!({ "n": 1 }))],
            vec![increment(1, "n", 100.0), insert(2, json!({ "n": 2 }))],
            vec![increment(1, "n", 1.0), increment(2, "n", 1.0)],
            vec![],
            vec![upsert("missing", 1, 1)],
        ])
        .await
        .unwrap();

    let position = before.next().unwrap();
    assert_eq!(cluster.committed().await.log_position, position);
    assert_eq!(
        results[0].as_ref().unwrap(),
        &Receipt {
            position,
            outcomes: vec![Outcome::Value(Value::Int(11)), Outcome::Applied]
        }
    );
    assert!(matches!(results[1], Err(StorageError::ObjectExists { .. })));
    assert_eq!(
        results[2].as_ref().unwrap().outcomes,
        vec![
            Outcome::Value(Value::Int(12)),
            Outcome::Value(Value::Int(2))
        ]
    );
    assert_eq!(
        results[3].as_ref().unwrap(),
        &Receipt {
            position,
            outcomes: vec![]
        }
    );
    assert!(matches!(
        results[4],
        Err(StorageError::UnknownCollection(_))
    ));
    cluster.open().await.partition().read(|state| {
        assert_eq!(n_of(state, "notes", 1), Some(12));
        assert_eq!(n_of(state, "notes", 2), Some(2));
    });

    let nothing = writer
        .commit_group(vec![vec![insert(2, json!({}))], vec![]])
        .await
        .unwrap();
    assert!(nothing[0].is_err());
    assert_eq!(nothing[1].as_ref().unwrap().position, position);
    assert_eq!(cluster.committed().await.log_position, position);
    assert_eq!(writer.commit_group(vec![]).await.unwrap().len(), 0);
}

#[tokio::test]
async fn a_schema_change_and_writes_to_it_can_share_a_group() {
    let (cluster, writer) = notes().await;
    let reviewed = Mutation::AddProperty {
        collection: name("notes"),
        property: Property::new("reviewed", DataType::Bool),
    };
    let results = writer
        .commit_group(vec![
            vec![patch(1, json!({ "reviewed": true }), None)],
            vec![reviewed, create("other")],
            vec![
                patch(1, json!({ "reviewed": true }), None),
                upsert("other", 1, 1),
            ],
            vec![Mutation::DropCollection(name("other"))],
            vec![upsert("other", 2, 2)],
        ])
        .await
        .unwrap();
    assert!(matches!(results[0], Err(StorageError::Object(_))));
    assert!(results[1].is_ok() && results[2].is_ok() && results[3].is_ok());
    assert!(matches!(
        results[4],
        Err(StorageError::UnknownCollection(_))
    ));
    cluster.open().await.partition().read(|state| {
        assert_eq!(property(state, 1, "reviewed"), Some(Value::Bool(true)));
        assert!(state.collection(&name("other")).is_none());
    });
}

/// A store that takes a while to accept a file, as a remote one does.
#[derive(Debug)]
struct Slow(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for Slow {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        tokio::time::sleep(Duration::from_millis(15)).await;
        self.0.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.0.get(key).await
    }

    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        self.0.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.0.delete(key).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_callers_share_commits_and_lose_no_update() {
    const CALLERS: u64 = 64;
    let cluster = Cluster::new().await;
    let (lease, committed) = cluster.lease().await;
    let slow: Arc<dyn Blobs> = Arc::new(Slow(cluster.blobs.clone()));
    let writer = Arc::new(
        Writer::open(slow, cluster.coordinator.clone(), lease, &committed)
            .await
            .unwrap(),
    );
    let committer = GroupCommitter::spawn(writer.clone(), 1024, 1024);
    committer
        .submit(vec![create("notes"), upsert("notes", 1, 0)])
        .await
        .unwrap();
    let start = cluster.committed().await.log_position;

    let callers = (0..CALLERS).map(|_| {
        let committer = committer.clone();
        tokio::spawn(async move {
            committer
                .submit(vec![increment(1, "n", 1.0)])
                .await
                .unwrap()
        })
    });
    let receipts: Vec<Receipt> = join_all(callers)
        .await
        .into_iter()
        .map(|joined| joined.unwrap())
        .collect();

    let mut seen: Vec<i64> = receipts
        .iter()
        .map(|receipt| match receipt.outcomes[..] {
            [Outcome::Value(Value::Int(n))] => n,
            ref other => panic!("{other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (1..=CALLERS as i64).collect::<Vec<_>>());

    let end = cluster.committed().await.log_position;
    let commits = end.0 - start.0;
    assert!(
        commits < CALLERS / 2,
        "{commits} commits for {CALLERS} callers"
    );
    assert!(receipts
        .iter()
        .all(|receipt| receipt.position > start && receipt.position <= end));
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(CALLERS as i64)));
}

#[tokio::test]
async fn a_writer_failure_reaches_every_caller_in_the_group_and_after_it() {
    let cluster = Cluster::new().await;
    let stalled = Arc::new(cluster.writer().await);
    stalled
        .commit(vec![create("notes"), upsert("notes", 1, 0)])
        .await
        .unwrap();
    let committer = GroupCommitter::spawn(stalled.clone(), 16, 16);
    let _successor = cluster.writer().await;

    let fenced = join_all((0..3).map(|_| committer.submit(vec![increment(1, "n", 1.0)]))).await;
    let roots: Vec<&StorageError> = fenced
        .iter()
        .map(|r| r.as_ref().unwrap_err().root())
        .collect();
    assert!(roots.iter().any(|e| matches!(
        e,
        StorageError::Coordinator(CoordinatorError::Fenced { .. })
    )));
    assert!(roots.iter().all(|e| {
        matches!(
            e,
            StorageError::Coordinator(CoordinatorError::Fenced { .. }) | StorageError::Closed
        )
    }));
    let after = committer
        .submit(vec![increment(1, "n", 1.0)])
        .await
        .unwrap_err();
    assert!(matches!(after.root(), StorageError::Closed));
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(0)));
}

#[tokio::test]
async fn updates_to_rows_in_a_segment_and_a_recreated_collection_replay_identically() {
    let (cluster, writer) = notes().await;
    writer.flush().await.unwrap();
    let recreate = || [Mutation::DropCollection(name("notes")), create("notes")];
    let results = writer
        .commit_group(vec![
            vec![
                increment(1, "n", 1.0),
                patch(1, json!({ "title": "from a segment" }), None),
            ],
            recreate()
                .into_iter()
                .chain([insert(1, json!({ "n": 5 })), increment(1, "n", 1.0)])
                .collect(),
            recreate()
                .into_iter()
                .chain([increment(1, "n", 1.0)])
                .collect(),
            vec![increment(1, "n", 1.0)],
        ])
        .await
        .unwrap();
    assert_eq!(
        results[0].as_ref().unwrap().outcomes[0],
        Outcome::Value(Value::Int(11))
    );
    assert_eq!(
        results[1].as_ref().unwrap().outcomes[3],
        Outcome::Value(Value::Int(6))
    );
    assert!(matches!(
        results[2],
        Err(StorageError::UnknownObject { .. })
    ));
    assert_eq!(
        results[3].as_ref().unwrap().outcomes[0],
        Outcome::Value(Value::Int(7))
    );

    let follower = cluster.open().await;
    writer.flush().await.unwrap();
    let reopened = cluster.open().await;
    for copy in [
        writer.partition(),
        follower.partition(),
        reopened.partition(),
    ] {
        copy.read(|state| {
            assert_eq!(n_of(state, "notes", 1), Some(7));
            assert_eq!(property(state, 1, "title"), None);
            assert_eq!(state.count(&name("notes")), 1);
        });
    }
}

#[tokio::test]
async fn a_caller_that_gave_up_before_its_turn_is_skipped() {
    let cluster = Cluster::new().await;
    let (lease, committed) = cluster.lease().await;
    let slow: Arc<dyn Blobs> = Arc::new(Slow(cluster.blobs.clone()));
    let writer = Arc::new(
        Writer::open(slow, cluster.coordinator.clone(), lease, &committed)
            .await
            .unwrap(),
    );
    let committer = GroupCommitter::spawn(writer, 16, 16);
    committer
        .submit(vec![create("notes"), upsert("notes", 1, 0)])
        .await
        .unwrap();

    let in_flight = tokio::spawn({
        let committer = committer.clone();
        async move { committer.submit(vec![increment(1, "n", 1.0)]).await }
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    let impatient = committer.submit(vec![increment(1, "n", 100.0)]);
    assert!(tokio::time::timeout(Duration::from_millis(1), impatient)
        .await
        .is_err());
    in_flight.await.unwrap().unwrap();

    let receipt = committer
        .submit(vec![increment(1, "n", 1.0)])
        .await
        .unwrap();
    assert_eq!(receipt.outcomes, vec![Outcome::Value(Value::Int(2))]);
}
