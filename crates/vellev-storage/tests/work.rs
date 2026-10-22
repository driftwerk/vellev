//! What a search reports it did: the rows it scored against the query's
//! vector, by layer, the rows a clustered index estimated, the postings
//! and rows of the keyword leg, and how much of what it searched its
//! filter allowed.

mod common;

use common::*;
use vellev_storage::{Filter, Hybrid, Keywords, Merging, Query, Work};

const NOTES: &str = "notes";
const EAST: [f32; 2] = [1.0, 0.0];

fn note(object: u128, title: &str, n: i64, vector: Option<[f32; 2]>) -> Mutation {
    Mutation::Upsert {
        collection: name(NOTES),
        id: ObjectId(uuid::Uuid::from_u128(object)),
        properties: json!({ "title": title, "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: vector.map(Vec::from),
        only_if_version: None,
    }
}

/// Four notes, three with a vector; "fox" in 1, 2 and 4; `n` = 1 for 2
/// and 3.
async fn notes() -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create(NOTES),
            note(1, "fox", 0, Some([1.0, 0.0])),
            note(2, "fox fox den", 1, Some([0.0, 1.0])),
            note(3, "hen", 1, Some([-1.0, 0.0])),
            note(4, "a fox and a hen", 0, None),
        ])
        .await
        .unwrap();
    (cluster, writer)
}

fn n_is(n: i64) -> Filter {
    Filter::Eq {
        property: "n".into(),
        value: json!(n),
    }
}

fn searched(state: &State, query: &Query<'_>) -> Work {
    let mut work = Work::default();
    state
        .search_counted(&name(NOTES), query, &mut work)
        .unwrap();
    work
}

fn worded(state: &State, query: &Keywords<'_>) -> Work {
    let mut work = Work::default();
    state
        .keyword_counted(&name(NOTES), query, &mut work)
        .unwrap();
    work
}

#[tokio::test]
async fn every_memtable_row_with_a_vector_is_scored_exactly_and_counted_as_the_memtables() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let work = searched(state, &Query::nearest(&EAST, 1));
        assert_eq!(work.exact_memtable, 3, "every vector, whatever the limit");
        assert_eq!((work.exact_segments, work.estimated), (0, 0));
        assert_eq!(work.selectivity(), None, "no filter");
        let filter = n_is(1);
        let work = searched(state, &Query::nearest(&EAST, 1).filtered(&filter));
        assert_eq!(work.exact_memtable, 2, "only the allowed rows");
        assert_eq!((work.filtered, work.allowed), (4, 2));
        assert_eq!(work.selectivity(), Some(0.5));
    });
}

#[tokio::test]
async fn rows_flushed_to_a_segment_are_counted_as_the_segments() {
    let (_cluster, writer) = notes().await;
    writer.flush().await.unwrap();
    writer.partition().read(|state| {
        let work = searched(state, &Query::nearest(&EAST, 1));
        assert_eq!((work.exact_memtable, work.exact_segments), (0, 3));
    });
}

#[tokio::test]
async fn a_large_segment_scores_only_what_its_index_offers() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut random = Random(7);
    let rows: u128 = 6_000;
    let mut batch = vec![create(NOTES)];
    for object in 1..=rows {
        let vector = random.vector(2);
        batch.push(note(object, "fox", 0, Some([vector[0], vector[1]])));
    }
    writer.commit(batch).await.unwrap();
    writer.flush().await.unwrap();
    writer.partition().read(|state| {
        let query = Query {
            oversample: 4,
            ..Query::nearest(&EAST, 10)
        };
        let work = searched(state, &query);
        assert_eq!(work.exact_memtable, 0);
        assert!(work.exact_segments >= 10, "{work:?}");
        assert!(
            (work.exact_segments as u128) < rows,
            "the index shortlists: {work:?}"
        );
    });
}

#[tokio::test]
async fn the_keyword_leg_counts_the_rows_it_scores_and_the_postings_it_reads() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let work = worded(state, &Keywords::matching("fox", 10));
        assert_eq!(work.keyword_rows, 3, "notes 1, 2 and 4 hold fox");
        assert!(work.postings >= 3, "{work:?}");
        let filter = n_is(0);
        let work = worded(
            state,
            &Keywords {
                filter: Some(&filter),
                ..Keywords::matching("fox", 10)
            },
        );
        assert_eq!(work.keyword_rows, 2, "note 2 is filtered out");
        assert_eq!((work.filtered, work.allowed), (4, 2));
        assert_eq!(
            (work.exact_memtable, work.exact_segments, work.estimated),
            (0, 0, 0),
            "no vector was scored"
        );
    });
}

#[tokio::test]
async fn a_hybrid_search_counts_both_its_legs() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let filter = n_is(1);
        let query = Hybrid {
            filter: Some(&filter),
            ..Hybrid::matching("fox", Some(&EAST), 10)
        };
        let mut work = Work::default();
        state
            .hybrid_counted(&name(NOTES), &query, &mut work)
            .unwrap();
        let legs = searched(
            state,
            &Query {
                oversample: query.oversample,
                ..Query::nearest(&EAST, query.candidates).filtered(&filter)
            },
        );
        let words = worded(
            state,
            &Keywords {
                filter: Some(&filter),
                ..Keywords::matching("fox", query.candidates)
            },
        );
        assert_eq!(work.exact_memtable, legs.exact_memtable);
        assert_eq!(work.keyword_rows, words.keyword_rows);
        assert_eq!(work.postings, words.postings);
        assert_eq!(work.selectivity(), Some(0.5), "both legs, the same share");
        let mut words_only = Work::default();
        let query = Hybrid {
            alpha: 0.0,
            ..query
        };
        state
            .hybrid_counted(&name(NOTES), &query, &mut words_only)
            .unwrap();
        assert_eq!(words_only.exact_memtable, 0, "alpha 0 scores no vector");
        assert_eq!(words_only.keyword_rows, 1);
    });
}

#[tokio::test]
async fn a_clustered_segment_counts_the_rows_it_estimated_and_scores_fewer() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    let mut random = Random(11);
    let mut batch = vec![create(NOTES)];
    for object in 1..=6_000u128 {
        let vector = random.vector(2);
        batch.push(note(object, "fox", 0, Some([vector[0], vector[1]])));
    }
    writer.commit(batch).await.unwrap();
    writer.flush().await.unwrap();
    writer.partition().read(|state| {
        let query = Query {
            oversample: 4,
            ..Query::nearest(&EAST, 10)
        };
        let work = searched(state, &query);
        assert_eq!(work.exact_memtable, 0);
        assert!(work.exact_segments >= 10, "{work:?}");
        assert!(
            work.estimated >= work.exact_segments,
            "every row scored was first estimated, as each has a vector: {work:?}"
        );
        assert!(work.estimated < 6_000, "{work:?}");
    });
}

#[tokio::test]
async fn the_keyword_leg_counts_each_posting_it_reads_exactly() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let work = worded(state, &Keywords::matching("fox", 10));
        assert_eq!(
            work.postings, 3,
            "one posting of fox in each of notes 1, 2 and 4, and none read to weigh it while nothing is dead"
        );
    });
    // Note 1 again, after a flush: its row in the segment dies, and its
    // posting of fox there with it.
    writer.flush().await.unwrap();
    writer
        .commit(vec![note(1, "a fox", 0, Some([1.0, 0.0]))])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let work = worded(state, &Keywords::matching("fox", 10));
        assert_eq!(work.keyword_rows, 3, "{work:?}");
        assert_eq!(
            work.postings,
            3 + 3 + 1,
            "the segment's three postings of fox, all read to count the two live ones as it has a dead row, then read again where each is scored or passed over; and the memtable's one, read where it is scored"
        );
    });
}
