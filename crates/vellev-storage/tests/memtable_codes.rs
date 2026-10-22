//! The memtable's vector codes (burst plan F2b), written before them.
//!
//! A collection's rows written since the last checkpoint are coded with
//! one bit a dimension, about a centre fixed from the first `CODE_FROM`
//! versions of its rows with a vector applied since that checkpoint, in
//! log order, dead or not; and a search shortlists them by their codes as
//! it does a segment's. The rules:
//!
//! - T1: a coded memtable of many rows is scored exactly no more than a
//!   segment is, the default budget, besides the first `CODE_FROM` rows,
//!   which are never coded and always scored exactly: so that reaching
//!   the threshold computes a centre and codes nothing, holding the
//!   partition's lock, which readers wait on, for well under a millisecond.
//! - T4: coding is per collection, by its own applied versions, and once
//!   coded a memtable stays coded however many of its rows die.
//! - T9: a writer, a copy that followed it, and a copy opened afresh code
//!   alike, to the bit, and answer alike, with rows replaced and deleted
//!   beside a flush and among the first versions the centre is taken from.
//! - T5: a memtable coded but with few live rows is still scored whole and
//!   exactly: dead rows do not count toward shortlisting.
//! - T2: recall holds when the memtable drifts far from its centre.
//! - Memory: the codes are counted as resident, and let go at a flush.

mod common;

use common::*;
use vellev_storage::{default_oversample, Hybrid, Query, Work, DEFAULT_CANDIDATES};

const NOTES: &str = "notes";
const OTHER: &str = "other";
const DIMENSIONS: usize = 128;
/// From how many versions applied a collection's memtable is coded.
const CODE_FROM: usize = 1_024;

fn wide(collection: &str) -> Mutation {
    let mut schema = schema(collection);
    schema.embedding.dimensions = DIMENSIONS as u32;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(collection: &str, n: u32, vector: Vec<f32>) -> Mutation {
    let words = ["fox", "hen", "den", "owl", "elk"];
    Mutation::Upsert {
        collection: name(collection),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": words[n as usize % 5], "n": n % 100 })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector),
        only_if_version: None,
    }
}

fn gone(collection: &str, n: u32) -> Mutation {
    Mutation::Delete {
        collection: name(collection),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        only_if_version: None,
    }
}

/// A vector about one of `centres`, `spread` from it.
fn about(random: &mut Random, centres: &[Vec<f32>], n: usize, spread: f32) -> Vec<f32> {
    let jitter = random.vector(DIMENSIONS);
    centres[n % centres.len()]
        .iter()
        .zip(jitter)
        .map(|(c, e)| c + spread * e)
        .collect()
}

fn centres(random: &mut Random, count: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| random.vector(DIMENSIONS)).collect()
}

async fn commit(writer: &Writer, rows: Vec<Mutation>) {
    for chunk in rows.chunks(2_000) {
        writer.commit(chunk.to_vec()).await.unwrap();
    }
}

fn hybrid_work(state: &State, collection: &str, query: &Hybrid<'_>) -> Work {
    let mut work = Work::default();
    state
        .hybrid_counted(&name(collection), query, &mut work)
        .unwrap();
    work
}

#[tokio::test]
async fn t1_a_coded_memtable_is_scored_exactly_no_more_than_its_budget() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide(NOTES)]).await.unwrap();
    let mut random = Random(3);
    let groups = centres(&mut random, 30);
    let all: Vec<Vec<f32>> = (0..20_000)
        .map(|n| about(&mut random, &groups, n, 0.35))
        .collect();
    commit(
        &writer,
        all.iter()
            .enumerate()
            .map(|(n, v)| row(NOTES, n as u32, v.clone()))
            .collect(),
    )
    .await;
    writer.partition().read(|state| {
        for limit in [10, 50] {
            let query = Hybrid {
                oversample: default_oversample(limit),
                ..Hybrid::matching("fox", Some(&all[7]), limit)
            };
            let budget = (limit * default_oversample(limit)).max(DEFAULT_CANDIDATES) as u64;
            let bound = budget + CODE_FROM as u64;
            let work = hybrid_work(state, NOTES, &query);
            assert!(
                work.exact_memtable <= bound,
                "limit {limit}: {} memtable rows scored exactly; the budget, {budget}, and the \
                 first {CODE_FROM} rows, never coded, come to {bound}",
                work.exact_memtable
            );
            assert_eq!(
                work.coded_memtable,
                20_000 - CODE_FROM as u64,
                "every live row after the first CODE_FROM ranked by its code"
            );
        }
    });
}

#[tokio::test]
async fn t4_a_collection_is_coded_by_its_own_versions_and_stays_coded() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide(NOTES), wide(OTHER)]).await.unwrap();
    let mut random = Random(5);
    let groups = centres(&mut random, 10);
    let notes: Vec<Mutation> = (0..CODE_FROM as u32 + 76)
        .map(|n| row(NOTES, n, about(&mut random, &groups, n as usize, 0.3)))
        .collect();
    let other: Vec<Mutation> = (0..900)
        .map(|n| {
            row(
                OTHER,
                100_000 + n,
                about(&mut random, &groups, n as usize, 0.3),
            )
        })
        .collect();
    commit(&writer, notes).await;
    commit(&writer, other).await;
    writer.partition().read(|state| {
        let coding = state.memtable_coding(&name(NOTES)).expect("notes is coded");
        assert_eq!(
            (coding.applied, coding.coded),
            (CODE_FROM + 76, 76),
            "the first CODE_FROM versions are scored exactly, uncoded"
        );
        assert_eq!(coding.centre.len() % 64, 0, "a centre in the rotated space");
        assert_eq!(
            state.memtable_coding(&name(OTHER)),
            None,
            "900 of its own: not coded, whatever notes has"
        );
    });
    // Deaths bring the live rows under the threshold; coding stays.
    commit(&writer, (0..700).map(|n| gone(NOTES, n)).collect()).await;
    writer.partition().read(|state| {
        let coding = state.memtable_coding(&name(NOTES)).expect("still coded");
        assert_eq!(coding.coded, 76, "dead rows keep their codes");
    });
}

/// Every hit of `query` in `collection` as (id, distance), for a reader.
fn answers(state: &State, queries: &[Vec<f32>]) -> Vec<Vec<(ObjectId, u32)>> {
    queries
        .iter()
        .map(|query| {
            let mut found: Vec<(ObjectId, u32)> = state
                .hybrid(&name(NOTES), &Hybrid::matching("fox", Some(query), 20))
                .unwrap()
                .iter()
                .map(|fused| (fused.object.id, fused.score.to_bits()))
                .collect();
            found.extend(
                state
                    .search(&name(NOTES), &Query::nearest(query, 20))
                    .unwrap()
                    .iter()
                    .map(|hit| (hit.object.id, hit.distance.to_bits())),
            );
            found
        })
        .collect()
}

#[tokio::test]
async fn t9_a_writer_and_its_copies_code_and_answer_alike_with_changes_beside_a_flush() {
    let mut cluster = Cluster::new().await;
    let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
    let writer = Arc::new(cluster.writer().await);
    let following = cluster.open().await;
    writer.commit(vec![wide(NOTES)]).await.unwrap();
    let mut random = Random(9);
    let groups = centres(&mut random, 20);
    let mut vector = |n: usize| about(&mut random, &groups, n, 0.3);
    commit(
        &writer,
        (0..600)
            .map(|n| row(NOTES, n, vector(n as usize)))
            .collect(),
    )
    .await;
    // A flush held while rows are replaced, deleted and written beside it:
    // the first versions after its checkpoint, which the centre is taken
    // from, include dead ones.
    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(stored.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);
    // Beside the flush: flushed rows replaced and deleted, then more than
    // CODE_FROM new rows, some of them replaced and deleted beside it too,
    // so that the flush keeps more than CODE_FROM versions, dead ones
    // among the first of them, and recodes them.
    let mut beside: Vec<Mutation> = (0..100)
        .map(|n| row(NOTES, n, vector(n as usize + 7)))
        .collect();
    beside.extend((100..150).map(|n| gone(NOTES, n)));
    beside.extend((600..1_800).map(|n| row(NOTES, n, vector(n as usize))));
    beside.extend((600..700).map(|n| row(NOTES, n, vector(n as usize + 11))));
    beside.extend((700..750).map(|n| gone(NOTES, n)));
    commit(&writer, beside).await;
    paused.release.notify_one();
    flushing.await.unwrap().unwrap();
    // After the flush: rows kept from beside it replaced and deleted, and
    // many more written.
    let mut after: Vec<Mutation> = (750..800)
        .map(|n| row(NOTES, n, vector(n as usize + 3)))
        .collect();
    after.extend((800..820).map(|n| gone(NOTES, n)));
    after.extend((1_800..7_000).map(|n| row(NOTES, n, vector(n as usize))));
    commit(&writer, after).await;
    let committed = cluster.committed().await;
    following.catch_up(&committed).await.unwrap();
    let afresh = cluster.open().await;
    let queries: Vec<Vec<f32>> = (0..30).map(|n| vector(n * 211)).collect();
    let coding = |state: &State| state.memtable_coding(&name(NOTES)).expect("coded");
    let (written, followed, opened) = (
        writer
            .partition()
            .read(|state| (coding(state), answers(state, &queries))),
        following
            .partition()
            .read(|state| (coding(state), answers(state, &queries))),
        afresh
            .partition()
            .read(|state| (coding(state), answers(state, &queries))),
    );
    let bits = |coding: &vellev_storage::MemtableCoding| -> Vec<u32> {
        coding.centre.iter().map(|value| value.to_bits()).collect()
    };
    assert_eq!(
        bits(&written.0),
        bits(&followed.0),
        "a follower's centre, to the bit"
    );
    assert_eq!(
        bits(&written.0),
        bits(&opened.0),
        "a reopened copy's centre, to the bit"
    );
    assert_eq!(written.0, followed.0);
    assert_eq!(written.0, opened.0);
    assert_eq!(written.1, followed.1, "the follower answers alike");
    assert_eq!(written.1, opened.1, "the reopened copy answers alike");
}

/// The exact nearest `limit` live rows, worked out apart from the search.
fn nearest_by_hand(state: &State, query: &[f32], limit: usize) -> Vec<(ObjectId, f32)> {
    let mut all: Vec<(ObjectId, f32)> = state
        .list(&name(NOTES), None, None, usize::MAX)
        .unwrap()
        .into_iter()
        .filter_map(|live| {
            let vector = live.vector?;
            let distance = vellev_index::vector::distance(Distance::Cosine, query, vector);
            Some((live.id(), distance))
        })
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(limit);
    all
}

#[tokio::test]
async fn t5_a_memtable_with_few_live_rows_is_scored_whole_however_many_died() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide(NOTES)]).await.unwrap();
    let mut random = Random(13);
    let groups = centres(&mut random, 20);
    // 3,000 rows, each written three times: 9,000 versions, 3,000 live.
    for round in 0..3 {
        commit(
            &writer,
            (0..3_000)
                .map(|n| {
                    row(
                        NOTES,
                        n,
                        about(&mut random, &groups, n as usize + round, 0.3),
                    )
                })
                .collect(),
        )
        .await;
    }
    writer.partition().read(|state| {
        for (q, limit) in [(0usize, 10usize), (1, 50), (2, 300)] {
            let query = about(&mut Random(100 + q as u64), &groups, q, 0.3);
            // An oversample of one: were the memtable shortlisted, only as
            // many rows as asked for would be scored, and the nearest
            // missed; scored whole, every one is.
            let asked = Query {
                oversample: 1,
                ..Query::nearest(&query, limit)
            };
            let found: Vec<(ObjectId, f32)> = state
                .search(&name(NOTES), &asked)
                .unwrap()
                .into_iter()
                .map(|hit| (hit.object.id, hit.distance))
                .collect();
            assert_eq!(
                found,
                nearest_by_hand(state, &query, limit),
                "limit {limit}"
            );
        }
    });
}

#[tokio::test]
async fn t2_recall_holds_when_the_memtable_drifts_far_from_its_centre() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide(NOTES)]).await.unwrap();
    let mut random = Random(17);
    // The first versions, which the centre is taken from, about one family;
    // every later row about another, far from it.
    let early = centres(&mut random, 5);
    let late: Vec<Vec<f32>> = centres(&mut random, 25)
        .into_iter()
        .map(|centre| {
            centre
                .iter()
                .map(|value| -4.0 * value.abs() - 1.0)
                .collect()
        })
        .collect();
    let mut all: Vec<Vec<f32>> = (0..CODE_FROM)
        .map(|n| about(&mut random, &early, n, 0.3))
        .collect();
    all.extend((0..11_000).map(|n| about(&mut random, &late, n, 0.35)));
    commit(
        &writer,
        all.iter()
            .enumerate()
            .map(|(n, v)| row(NOTES, n as u32, v.clone()))
            .collect(),
    )
    .await;
    writer.partition().read(|state| {
        let ids = |query: &Hybrid<'_>| -> Vec<ObjectId> {
            state
                .hybrid(&name(NOTES), query)
                .unwrap()
                .iter()
                .map(|fused| fused.object.id)
                .collect()
        };
        for limit in [10, 50] {
            let (mut shares, mut firsts, mut asked) = (0.0, 0, 0);
            for (q, near) in all[CODE_FROM..].iter().step_by(180).enumerate() {
                let query: Vec<f32> = near
                    .iter()
                    .zip(Random(500 + q as u64).vector(DIMENSIONS))
                    .map(|(v, e)| v + 0.05 * e)
                    .collect();
                let default = Hybrid {
                    oversample: default_oversample(limit),
                    ..Hybrid::matching(["fox", "hen den"][q % 2], Some(&query), limit)
                };
                let exact = ids(&Hybrid {
                    oversample: usize::MAX,
                    ..default.clone()
                });
                let found = ids(&default);
                shares += exact.iter().filter(|id| found.contains(id)).count() as f64
                    / exact.len().max(1) as f64;
                firsts += usize::from(found.first() == exact.first());
                asked += 1;
            }
            let (mean, first) = (shares / asked as f64, firsts as f64 / asked as f64);
            assert!(
                mean >= 0.99 && first >= 0.99,
                "limit {limit}: overlap {mean:.3}, same top {first:.3}"
            );
        }
    });
}

/// `count` rows with a vector, spread over `collections` collections in
/// turn, all alike but for the collection they are in.
async fn spread(writer: &Writer, collections: &[&str], count: u32) {
    let mut random = Random(19);
    let groups = centres(&mut random, 10);
    writer
        .commit(collections.iter().map(|c| wide(c)).collect())
        .await
        .unwrap();
    commit(
        writer,
        (0..count)
            .map(|n| {
                let collection = collections[n as usize % collections.len()];
                row(collection, n, about(&mut random, &groups, n as usize, 0.3))
            })
            .collect(),
    )
    .await;
}

#[tokio::test]
async fn the_codes_are_counted_as_resident_and_let_go_at_a_flush() {
    // The same rows, in one collection that is coded and in three that are
    // not: what the first holds more is its codes.
    let rows = CODE_FROM as u32 + 2_000;
    let coded = Cluster::new().await;
    let coded = coded.writer().await;
    spread(&coded, &[NOTES], rows).await;
    let plain = Cluster::new().await;
    let plain = plain.writer().await;
    spread(&plain, &[NOTES, OTHER, "third"], rows).await;
    let resident = |writer: &Writer| writer.partition().read(State::resident);
    let codes = 2_000 * DIMENSIONS / 8;
    let more = resident(&coded) as i64 - resident(&plain) as i64;
    assert!(
        more >= codes as i64,
        "{more} bytes more with 2,000 codes of {codes} bytes in all"
    );
    coded.flush().await.unwrap();
    plain.flush().await.unwrap();
    let more = resident(&coded) as i64 - resident(&plain) as i64;
    assert!(
        more < codes as i64 / 4,
        "the codes let go at the flush: {more} bytes more"
    );
}

/// The apply that brings a collection to `CODE_FROM` versions codes no
/// row: it computes the centre alone. Rows after it are coded one by one.
#[tokio::test]
#[ignore = "F2b: the memtable is coded (burst plan F2b, PR 3)"]
async fn reaching_the_threshold_computes_a_centre_and_codes_no_row() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide(NOTES)]).await.unwrap();
    let mut random = Random(23);
    let groups = centres(&mut random, 10);
    let mut vector = |n: usize| about(&mut random, &groups, n, 0.3);
    commit(
        &writer,
        (0..CODE_FROM as u32 - 1)
            .map(|n| row(NOTES, n, vector(n as usize)))
            .collect(),
    )
    .await;
    let coding = |writer: &Writer| {
        writer
            .partition()
            .read(|state| state.memtable_coding(&name(NOTES)))
    };
    assert_eq!(coding(&writer), None);
    writer
        .commit(vec![row(NOTES, CODE_FROM as u32, vector(1))])
        .await
        .unwrap();
    let reached = coding(&writer).expect("a centre at the threshold");
    assert_eq!(
        (reached.applied, reached.coded),
        (CODE_FROM, 0),
        "no row coded"
    );
    writer
        .commit(vec![row(NOTES, CODE_FROM as u32 + 1, vector(2))])
        .await
        .unwrap();
    assert_eq!(
        coding(&writer).unwrap().coded,
        1,
        "the next row is coded as it comes"
    );
}

/// How long readers waited, at the most, for the partition while
/// `commit` ran.
async fn longest_wait(writer: &Writer, commit: Vec<Mutation>) -> std::time::Duration {
    let partition = writer.partition().clone();
    let stop = Arc::new(AtomicBool::new(false));
    let reading = std::thread::spawn({
        let stop = stop.clone();
        move || {
            let mut longest = std::time::Duration::ZERO;
            while !stop.load(Ordering::SeqCst) {
                let began = std::time::Instant::now();
                partition.read(|_| ());
                longest = longest.max(began.elapsed());
            }
            longest
        }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    writer.commit(commit).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    stop.store(true, Ordering::SeqCst);
    reading.join().unwrap()
}

/// No apply holds the partition's lock, which readers wait on, for more
/// than about a millisecond when it brings a collection to `CODE_FROM`
/// versions of 768 dimensions. Timed in an optimised build only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(debug_assertions, ignore = "timed: run with --release")]
async fn reaching_the_threshold_keeps_readers_waiting_a_millisecond_at_most() {
    let mut waits = Vec::new();
    for attempt in 0..5u64 {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await;
        let mut schema = schema(NOTES);
        schema.embedding.dimensions = 768;
        writer
            .commit(vec![Mutation::CreateCollection(Box::new(schema))])
            .await
            .unwrap();
        let mut random = Random(29 + attempt);
        let rows: Vec<Mutation> = (0..CODE_FROM as u32)
            .map(|n| row(NOTES, n, random.vector(768)))
            .collect();
        let (before, last) = rows.split_at(CODE_FROM - 1);
        commit(&writer, before.to_vec()).await;
        waits.push(longest_wait(&writer, last.to_vec()).await);
    }
    waits.sort();
    let median = waits[waits.len() / 2];
    assert!(
        median <= Duration::from_millis(1),
        "readers waited up to {median:?} (median of {waits:?})"
    );
}
