use std::collections::BTreeSet;

use proptest::prelude::*;
use serde_json::{json, Value as Json};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, Object,
    ObjectId, Property,
};
use vellev_index::filter::{Filter, FilterError, RowIndex, MAX_DEPTH};

fn collection() -> Collection {
    let mut content = Property::new("content", DataType::Text).searchable();
    content.filterable = false;
    Collection {
        name: CollectionName::new("Memory").unwrap(),
        properties: vec![
            content,
            Property::new("owner", DataType::Text),
            Property::new("tags", DataType::TextArray),
            Property::new("deleted", DataType::Bool),
            Property::new("count", DataType::Int),
            Property::new("importance", DataType::Number),
            Property::new("createdAt", DataType::Date),
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

fn object(n: u32, properties: Json) -> Object {
    let id = ObjectId(uuid::Uuid::from_u128(u128::from(n)));
    Object::from_json(&collection(), id, properties.as_object().unwrap(), None).unwrap()
}

fn filter(json: Json) -> Filter {
    serde_json::from_value(json).unwrap()
}

/// The rows passing `filter`, by the index and by asking each object; the
/// two must agree.
fn passing(objects: &[Object], filter: &Filter) -> BTreeSet<u32> {
    let schema = collection();
    let predicate = filter.compile(&schema).unwrap();
    let index = objects
        .iter()
        .enumerate()
        .fold(RowIndex::default(), |mut index, (row, object)| {
            index.insert(&schema, row as u32, object);
            index
        });
    let indexed: BTreeSet<u32> = predicate.select(&index).into_iter().collect();
    let asked: BTreeSet<u32> = (0..objects.len() as u32)
        .filter(|row| predicate.matches(&objects[*row as usize]))
        .collect();
    assert_eq!(indexed, asked, "index and predicate disagree on {filter:?}");
    indexed
}

fn memories() -> Vec<Object> {
    vec![
        object(
            0,
            json!({ "owner": "ada", "tags": ["date:2026-07-05", "lesson"], "deleted": false, "count": 3, "importance": 0.9, "createdAt": "2026-07-05T09:00:00Z" }),
        ),
        object(
            1,
            json!({ "owner": "ada", "tags": ["date:2026-05-07"], "deleted": true, "count": 0, "importance": 0.2, "createdAt": "2026-05-07T09:00:00Z" }),
        ),
        object(
            2,
            json!({ "owner": "bob", "tags": [], "deleted": false, "count": -1, "importance": 0.5, "createdAt": "2026-07-05T10:00:00+01:00" }),
        ),
        object(3, json!({ "owner": "team:core", "deleted": false })),
        object(4, json!({ "content": "only content" })),
    ]
}

fn rows(filter_json: Json) -> Vec<u32> {
    passing(&memories(), &filter(filter_json))
        .into_iter()
        .collect()
}

#[test]
fn text_matches_the_whole_value_and_arrays_match_by_element() {
    assert_eq!(
        rows(json!({ "eq": { "property": "owner", "value": "ada" } })),
        [0, 1]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "owner", "value": "team" } })),
        [] as [u32; 0]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "owner", "value": "team:core" } })),
        [3]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "tags", "value": "date:2026-07-05" } })),
        [0]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "tags", "value": "date:2026-05-07" } })),
        [1]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "tags", "value": "date" } })),
        [] as [u32; 0]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "deleted", "value": false } })),
        [0, 2, 3]
    );
}

#[test]
fn not_equal_includes_rows_with_no_value_and_ranges_exclude_them() {
    assert_eq!(
        rows(json!({ "ne": { "property": "tags", "value": "lesson" } })),
        [1, 2, 3, 4]
    );
    assert_eq!(
        rows(json!({ "ne": { "property": "owner", "value": "ada" } })),
        [2, 3, 4]
    );
    assert_eq!(rows(json!({ "is_null": { "property": "tags" } })), [3, 4]);
    assert_eq!(
        rows(json!({ "not": { "is_null": { "property": "tags" } } })),
        [0, 1, 2]
    );
    let both = json!({ "and": [
        { "is_null": { "property": "importance" } },
        { "ne": { "property": "owner", "value": "ada" } }
    ] });
    assert_eq!(rows(both), [3, 4]);
    assert_eq!(
        rows(json!({ "gte": { "property": "count", "value": -5 } })),
        [0, 1, 2]
    );
    assert_eq!(
        rows(json!({ "not": { "gte": { "property": "count", "value": -5 } } })),
        [3, 4]
    );
}

#[test]
fn ranges_compare_numbers_and_instants() {
    assert_eq!(
        rows(json!({ "gt": { "property": "count", "value": 0 } })),
        [0]
    );
    assert_eq!(
        rows(json!({ "gte": { "property": "count", "value": 0 } })),
        [0, 1]
    );
    assert_eq!(
        rows(json!({ "lt": { "property": "count", "value": 0 } })),
        [2]
    );
    assert_eq!(
        rows(json!({ "lte": { "property": "importance", "value": 0.5 } })),
        [1, 2]
    );
    assert_eq!(
        rows(json!({ "gte": { "property": "importance", "value": 1 } })),
        [] as [u32; 0]
    );
    assert_eq!(
        rows(json!({ "eq": { "property": "importance", "value": 0.5 } })),
        [2]
    );
    let same_instant =
        json!({ "eq": { "property": "createdAt", "value": "2026-07-05T11:00:00+02:00" } });
    assert_eq!(rows(same_instant), [0, 2]);
    let window = json!({ "and": [
        { "gte": { "property": "createdAt", "value": "2026-06-01T00:00:00Z" } },
        { "lte": { "property": "createdAt", "value": "2026-07-05T09:00:00Z" } }
    ] });
    assert_eq!(rows(window), [0, 2]);
}

#[test]
fn the_filters_workforce_builds_select_what_it_expects() {
    let visible_to_ada = json!({ "and": [
        { "eq": { "property": "deleted", "value": false } },
        { "or": [
            { "eq": { "property": "owner", "value": "ada" } },
            { "eq": { "property": "owner", "value": "team:core" } }
        ] },
        { "or": [
            { "eq": { "property": "tags", "value": "lesson" } },
            { "eq": { "property": "tags", "value": "date:2026-05-07" } }
        ] },
        { "ne": { "property": "tags", "value": "archived" } },
        { "gte": { "property": "importance", "value": 0.4 } }
    ] });
    assert_eq!(rows(visible_to_ada), [0]);
    assert_eq!(rows(json!({ "and": [] })), [0, 1, 2, 3, 4]);
    assert_eq!(rows(json!({ "or": [] })), [] as [u32; 0]);
    assert_eq!(rows(json!({ "not": { "or": [] } })), [0, 1, 2, 3, 4]);
}

#[test]
fn a_filter_the_schema_cannot_answer_is_refused_with_the_reason() {
    let refused = |json: Json| filter(json).compile(&collection()).unwrap_err();
    assert_eq!(
        refused(json!({ "eq": { "property": "nope", "value": 1 } })),
        FilterError::UnknownProperty("nope".into())
    );
    assert_eq!(
        refused(json!({ "eq": { "property": "content", "value": "x" } })),
        FilterError::NotFilterable("content".into())
    );
    assert_eq!(
        refused(json!({ "is_null": { "property": "content" } })),
        FilterError::NotFilterable("content".into())
    );
    for (property, value, expected) in [
        ("count", json!("3"), DataType::Int),
        ("count", json!(1.5), DataType::Int),
        ("importance", json!("high"), DataType::Number),
        ("deleted", json!("false"), DataType::Bool),
        ("createdAt", json!("yesterday"), DataType::Date),
        ("owner", json!(7), DataType::Text),
        ("tags", json!(["lesson"]), DataType::TextArray),
        ("tags", json!(7), DataType::TextArray),
    ] {
        assert_eq!(
            refused(json!({ "eq": { "property": property, "value": value } })),
            FilterError::Type {
                property: property.into(),
                expected
            }
        );
    }
    assert_eq!(
        refused(json!({ "ne": { "property": "owner", "value": null } })),
        FilterError::Null {
            property: "owner".into()
        }
    );
    for (property, data_type) in [
        ("owner", DataType::Text),
        ("tags", DataType::TextArray),
        ("deleted", DataType::Bool),
    ] {
        assert_eq!(
            refused(json!({ "gt": { "property": property, "value": "a" } })),
            FilterError::Unordered {
                property: property.into(),
                data_type
            }
        );
    }
}

#[test]
fn nesting_is_bounded() {
    let nested = |depth: usize| {
        (0..depth).fold(
            Filter::IsNull {
                property: "tags".into(),
            },
            |inner, _| Filter::Not(Box::new(inner)),
        )
    };
    assert!(nested(MAX_DEPTH - 1).compile(&collection()).is_ok());
    assert_eq!(
        nested(MAX_DEPTH).compile(&collection()),
        Err(FilterError::TooDeep)
    );
}

const OWNERS: [&str; 3] = ["ada", "bob", "team:core"];
const TAGS: [&str; 4] = ["lesson", "date:2026-07-05", "date:2026-05-07", "x"];
const DATES: [&str; 3] = [
    "2026-01-01T00:00:00Z",
    "2026-07-05T09:00:00Z",
    "2026-07-05T10:00:00+01:00",
];

fn any_object() -> impl Strategy<Value = Json> {
    (
        prop::option::of(0..OWNERS.len()),
        prop::option::of(prop::collection::vec(0..TAGS.len(), 0..3)),
        prop::option::of(any::<bool>()),
        prop::option::of(-2i64..3),
        prop::option::of(prop_oneof![Just(-0.0), Just(0.0), Just(0.5), Just(1.0)]),
        prop::option::of(0..DATES.len()),
    )
        .prop_map(|(owner, tags, deleted, count, importance, created)| {
            let tags = tags.map(|tags| tags.into_iter().map(|t| TAGS[t]).collect::<Vec<_>>());
            json!({
                "owner": owner.map(|o| OWNERS[o]),
                "tags": tags,
                "deleted": deleted,
                "count": count,
                "importance": importance,
                "createdAt": created.map(|d| DATES[d]),
            })
        })
}

fn any_comparison() -> impl Strategy<Value = Filter> {
    let scalar = prop_oneof![
        (0..OWNERS.len()).prop_map(|o| ("owner", json!(OWNERS[o]), false)),
        (0..TAGS.len()).prop_map(|t| ("tags", json!(TAGS[t]), false)),
        any::<bool>().prop_map(|b| ("deleted", json!(b), false)),
        (-2i64..3).prop_map(|n| ("count", json!(n), true)),
        prop_oneof![Just(-0.0), Just(0.0), Just(0.5), Just(1.0)].prop_map(|n| (
            "importance",
            json!(n),
            true
        )),
        (0..DATES.len()).prop_map(|d| ("createdAt", json!(DATES[d]), true)),
    ];
    (scalar, 0..7usize).prop_map(|((property, value, ordered), operator)| {
        let property = property.to_string();
        match (operator, ordered) {
            (0, _) => Filter::Eq { property, value },
            (1, _) => Filter::Ne { property, value },
            (2, _) => Filter::IsNull { property },
            (3, true) => Filter::Gt { property, value },
            (4, true) => Filter::Gte { property, value },
            (5, true) => Filter::Lt { property, value },
            (6, true) => Filter::Lte { property, value },
            (_, false) => Filter::Eq { property, value },
            _ => unreachable!("operators are drawn from 0..7"),
        }
    })
}

fn any_filter() -> impl Strategy<Value = Filter> {
    any_comparison().prop_recursive(4, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Filter::And),
            prop::collection::vec(inner.clone(), 0..4).prop_map(Filter::Or),
            inner.prop_map(|filter| Filter::Not(Box::new(filter))),
        ]
    })
}

proptest! {
    #[test]
    fn the_index_selects_exactly_the_rows_the_predicate_accepts(
        objects in prop::collection::vec(any_object(), 0..24),
        filter in any_filter(),
    ) {
        let objects: Vec<Object> =
            objects.into_iter().enumerate().map(|(n, json)| object(n as u32, json)).collect();
        passing(&objects, &filter);
    }

    #[test]
    fn a_filter_survives_a_json_round_trip(filter in any_filter()) {
        let json = serde_json::to_value(&filter).unwrap();
        prop_assert_eq!(serde_json::from_value::<Filter>(json).unwrap(), filter);
    }
}
