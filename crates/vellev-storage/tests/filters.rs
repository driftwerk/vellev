mod common;

use common::*;
use serde_json::Value as Json;
use vellev_storage::{FilterError, Outcome};

fn filter(json: Json) -> Filter {
    serde_json::from_value(json).unwrap()
}

fn n(operator: &str, value: i64) -> Filter {
    filter(json!({ operator: { "property": "n", "value": value } }))
}

fn tagged(object: u8, n: i64, tags: &[&str]) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: id(object),
        properties: json!({ "n": n, "tags": tags }).as_object().unwrap().clone(),
        vector: None,
    }
}

fn ids(found: Vec<vellev_storage::Live<'_>>) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = found.into_iter().map(|found| found.id()).collect();
    ids.sort_unstable();
    ids
}

fn selected(state: &State, filter: &Filter) -> Vec<ObjectId> {
    ids(state.select(&name("notes"), filter).unwrap())
}

fn these(objects: &[u8]) -> Vec<ObjectId> {
    objects.iter().map(|object| id(*object)).collect()
}

/// Every copy of the partition: the writer's, a follower's and a fresh open.
async fn copies(cluster: &Cluster, writer: &Writer) -> Vec<Arc<vellev_storage::Partition>> {
    vec![
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ]
}

#[tokio::test]
async fn a_filter_sees_each_row_once_at_its_newest_value_across_layers() {
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
        .commit(vec![
            upsert("notes", 2, 20),
            delete("notes", 3),
            upsert("notes", 4, 4),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .commit(vec![upsert("notes", 1, 10), upsert("notes", 5, 2)])
        .await
        .unwrap();

    for copy in copies(&cluster, &writer).await {
        copy.read(|state| {
            assert_eq!(selected(state, &n("eq", 2)), these(&[5]));
            assert_eq!(selected(state, &n("eq", 1)), these(&[]));
            assert_eq!(selected(state, &n("eq", 3)), these(&[]));
            assert_eq!(selected(state, &n("gte", 10)), these(&[1, 2]));
            assert_eq!(selected(state, &n("lt", 10)), these(&[4, 5]));
            assert_eq!(
                selected(state, &filter(json!({ "and": [] }))),
                these(&[1, 2, 4, 5])
            );
            assert_eq!(
                selected(
                    state,
                    &filter(json!({ "not": { "eq": { "property": "n", "value": 2 } } }))
                ),
                these(&[1, 2, 4])
            );
        });
    }
}

#[tokio::test]
async fn tags_match_exactly_and_a_property_added_later_is_absent_on_older_rows() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            tagged(1, 1, &["date:2026-07-05"]),
            tagged(2, 2, &["date:2026-05-07", "lesson"]),
            upsert("notes", 3, 3),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let reviewed = Mutation::AddProperty {
        collection: name("notes"),
        property: Property::new("reviewed", DataType::Bool),
    };
    let mark = Mutation::Patch {
        collection: name("notes"),
        id: id(2),
        properties: json!({ "reviewed": true }).as_object().unwrap().clone(),
        vector: None,
        only_if: None,
    };
    writer.commit(vec![reviewed, mark]).await.unwrap();

    let tag = |operator: &str, value: &str| {
        filter(json!({ operator: { "property": "tags", "value": value } }))
    };
    for copy in copies(&cluster, &writer).await {
        copy.read(|state| {
            assert_eq!(selected(state, &tag("eq", "date:2026-07-05")), these(&[1]));
            assert_eq!(selected(state, &tag("eq", "date:2026-05-07")), these(&[2]));
            assert_eq!(selected(state, &tag("eq", "date")), these(&[]));
            assert_eq!(selected(state, &tag("ne", "lesson")), these(&[1, 3]));
            assert_eq!(
                selected(state, &filter(json!({ "is_null": { "property": "tags" } }))),
                these(&[3])
            );
            assert_eq!(
                selected(
                    state,
                    &filter(json!({ "eq": { "property": "reviewed", "value": true } }))
                ),
                these(&[2])
            );
            assert_eq!(
                selected(
                    state,
                    &filter(json!({ "is_null": { "property": "reviewed" } }))
                ),
                these(&[1, 3])
            );
            assert_eq!(
                selected(
                    state,
                    &filter(json!({ "ne": { "property": "reviewed", "value": true } }))
                ),
                these(&[1, 3])
            );
        });
    }
}

#[tokio::test]
async fn a_list_pages_in_id_order_and_stays_correct_while_rows_change() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let initial = (1..=7).map(|object| upsert("notes", object, i64::from(object % 2)));
    writer
        .commit([create("notes")].into_iter().chain(initial).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();

    let page = |after: Option<u8>, filter: Option<&Filter>| {
        writer.partition().read(|state| {
            let page = state
                .list(&name("notes"), filter, after.map(id), 3)
                .unwrap();
            page.into_iter().map(|found| found.id()).collect::<Vec<_>>()
        })
    };
    assert_eq!(page(None, None), these(&[1, 2, 3]));
    writer
        .commit(vec![
            delete("notes", 4),
            upsert("notes", 0, 1),
            upsert("notes", 8, 1),
        ])
        .await
        .unwrap();
    assert_eq!(page(Some(3), None), these(&[5, 6, 7]));
    assert_eq!(page(Some(7), None), these(&[8]));
    assert_eq!(page(Some(8), None), these(&[]));

    let odd = n("eq", 1);
    assert_eq!(page(None, Some(&odd)), these(&[0, 1, 3]));
    assert_eq!(page(Some(3), Some(&odd)), these(&[5, 7, 8]));
    writer.partition().read(|state| {
        assert_eq!(state.list(&name("notes"), None, None, 0).unwrap().len(), 0);
        assert_eq!(
            state
                .list(&name("notes"), None, None, usize::MAX)
                .unwrap()
                .len(),
            8
        );
    });
}

#[tokio::test]
async fn delete_where_removes_what_the_batch_sees_and_replays_identically() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 1),
            upsert("notes", 3, 2),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 4, 1)]).await.unwrap();

    let where_n = |value: i64| Mutation::DeleteWhere {
        collection: name("notes"),
        filter: n("eq", value),
    };
    let receipt = writer
        .commit(vec![
            upsert("notes", 5, 1),
            upsert("notes", 2, 7),
            delete("notes", 4),
            upsert("notes", 3, 1),
            where_n(1),
            upsert("notes", 6, 1),
            where_n(99),
        ])
        .await
        .unwrap();
    assert_eq!(
        receipt.outcomes[4],
        Outcome::Deleted(3),
        "rows 1, 3 and 5; 2 was changed and 4 already deleted"
    );
    assert_eq!(receipt.outcomes[6], Outcome::Deleted(0));

    for copy in copies(&cluster, &writer).await {
        copy.read(|state| {
            assert_eq!(
                selected(state, &filter(json!({ "and": [] }))),
                these(&[2, 6])
            );
            assert_eq!(n_of(state, "notes", 2), Some(7));
        });
    }

    let before = cluster.committed().await;
    let nothing = writer.commit(vec![where_n(99)]).await.unwrap();
    assert_eq!(
        (nothing.position, cluster.committed().await),
        (before.log_position, before)
    );
}

#[tokio::test]
async fn delete_where_is_atomic_with_its_batch_and_its_group() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 1),
        ])
        .await
        .unwrap();
    let everything = || Mutation::DeleteWhere {
        collection: name("notes"),
        filter: n("gte", -1000),
    };

    let failed = writer
        .commit(vec![everything(), upsert("missing", 1, 1)])
        .await;
    assert!(matches!(failed, Err(StorageError::UnknownCollection(_))));
    writer
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 2));

    let results = writer
        .commit_group(vec![
            vec![upsert("notes", 3, 1)],
            vec![everything()],
            vec![upsert("notes", 4, 1)],
        ])
        .await
        .unwrap();
    assert_eq!(
        results[1].as_ref().unwrap().outcomes,
        vec![Outcome::Deleted(3)]
    );
    for copy in copies(&cluster, &writer).await {
        copy.read(|state| assert_eq!(selected(state, &filter(json!({ "and": [] }))), these(&[4])));
    }
}

#[tokio::test]
async fn a_recreated_collection_does_not_match_its_predecessors_rows() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 1)]).await.unwrap();
    writer
        .commit(vec![
            Mutation::DropCollection(name("notes")),
            create("notes"),
            upsert("notes", 3, 1),
        ])
        .await
        .unwrap();

    for flush in [false, true] {
        if flush {
            writer.flush().await.unwrap();
        }
        for copy in copies(&cluster, &writer).await {
            copy.read(|state| assert_eq!(selected(state, &n("eq", 1)), these(&[3])));
        }
    }
}

#[tokio::test]
async fn a_filter_the_schema_cannot_answer_is_an_error_everywhere() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let unknown = filter(json!({ "eq": { "property": "nope", "value": 1 } }));
    let untyped = filter(json!({ "gt": { "property": "n", "value": "one" } }));
    let unfilterable = |e: &StorageError| {
        matches!(
            e,
            StorageError::Filter(FilterError::UnknownProperty(_) | FilterError::Type { .. })
        )
    };

    writer.partition().read(|state| {
        for bad in [&unknown, &untyped] {
            assert!(unfilterable(
                &state.select(&name("notes"), bad).unwrap_err()
            ));
            assert!(unfilterable(
                &state.list(&name("notes"), Some(bad), None, 10).unwrap_err()
            ));
        }
        assert!(matches!(
            state.select(&name("missing"), &n("eq", 1)),
            Err(StorageError::UnknownCollection(_))
        ));
        assert!(matches!(
            state.list(&name("missing"), None, None, 10),
            Err(StorageError::UnknownCollection(_))
        ));
    });

    let before = cluster.committed().await;
    let refused = writer
        .commit(vec![
            upsert("notes", 2, 2),
            Mutation::DeleteWhere {
                collection: name("notes"),
                filter: unknown,
            },
        ])
        .await;
    assert!(unfilterable(&refused.unwrap_err()));
    assert_eq!(cluster.committed().await, before);
}

#[tokio::test]
async fn delete_where_sees_a_later_batch_over_an_earlier_one_in_the_same_group() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 1),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let ones = || Mutation::DeleteWhere {
        collection: name("notes"),
        filter: n("eq", 1),
    };

    let results = writer
        .commit_group(vec![
            vec![
                upsert("notes", 1, 5),
                upsert("notes", 3, 1),
                upsert("notes", 4, 1),
            ],
            vec![
                upsert("notes", 3, 5),
                upsert("notes", 1, 1),
                delete("notes", 4),
                ones(),
            ],
        ])
        .await
        .unwrap();
    assert_eq!(
        results[1].as_ref().unwrap().outcomes[3],
        Outcome::Deleted(2),
        "rows 1 and 2: the batch set 3 to five and deleted 4 itself"
    );
    for copy in copies(&cluster, &writer).await {
        copy.read(|state| {
            assert_eq!(selected(state, &n("gte", -1000)), these(&[3]));
            assert_eq!(n_of(state, "notes", 3), Some(5));
        });
    }
}

#[tokio::test]
async fn a_delete_with_no_condition_is_refused() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let before = cluster.committed().await;
    let unconditional = Mutation::DeleteWhere {
        collection: name("notes"),
        filter: filter(json!({ "and": [] })),
    };
    assert!(matches!(
        writer.commit(vec![unconditional]).await,
        Err(StorageError::InvalidMutation(reason)) if reason.contains("needs a condition")
    ));
    assert_eq!(cluster.committed().await, before);
    writer
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 1));
}
