//! Opening a partition splits its rows' text into tokens on several
//! threads, as a segment is loaded and as commits are replayed (burst
//! plan, open CPU). A test binary of its own, so that its pool of threads
//! is made with as many as it needs before anything else makes it: on a
//! runner with one CPU, the pool rayon would make is of one thread, and
//! nothing would be split on several.

mod common;

use common::*;

const NOTES: &str = "notes";

/// The threads the global pool is made with, more than one however many
/// CPUs the runner has.
const POOL: usize = 4;

/// `rows` notes of text in turn, a commit of 100 at a time, and flushed if
/// `flushed`.
async fn texts(rows: u32, flushed: bool) -> Cluster {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create(NOTES)]).await.unwrap();
    let words = ["fox", "hen", "den", "owl", "elk", "the", "deploys", "ÉCOLE"];
    for batch in 0..rows / 100 {
        let rows: Vec<Mutation> = (0..100u32)
            .map(|n| {
                let at = batch * 100 + n;
                let title: Vec<&str> = (0..20)
                    .map(|k| words[((at * 3 + k) % 8) as usize])
                    .collect();
                Mutation::Upsert {
                    collection: name(NOTES),
                    id: ObjectId(uuid::Uuid::from_u128(u128::from(at))),
                    properties: json!({ "title": title.join(" "), "n": at })
                        .as_object()
                        .unwrap()
                        .clone(),
                    vector: None,
                    only_if_version: None,
                }
            })
            .collect();
        writer.commit(rows).await.unwrap();
    }
    if flushed {
        writer.flush().await.unwrap();
    }
    cluster
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_segment_loaded_and_commits_replayed_split_their_text_on_several_threads() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(POOL)
        .build_global()
        .expect("the pool, made before anything else made it");
    for flushed in [true, false] {
        let cluster = texts(4_000, flushed).await;
        let copy = Follower::open_through(
            cluster.key.clone(),
            cluster.blobs.clone(),
            cluster.coordinator.clone(),
            &cluster.committed().await,
        )
        .await
        .unwrap();
        let replayed = copy.partition().replayed();
        assert!(
            replayed.tokenized_on_threads > 1,
            "flushed {flushed}: {replayed:?}"
        );
        // What it finds is what the writer finds.
        let found = |state: &State| {
            state
                .keyword(
                    &name(NOTES),
                    &vellev_storage::Keywords::matching("fox owl", 50),
                )
                .unwrap()
                .iter()
                .map(|scored| (scored.object.id, scored.score.to_bits()))
                .collect::<Vec<_>>()
        };
        let writer = cluster.writer().await;
        assert_eq!(
            copy.partition().read(found),
            writer.partition().read(found),
            "flushed {flushed}"
        );
    }
}
