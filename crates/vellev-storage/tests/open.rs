//! Opening a partition whose memtable holds many commits: the logs it
//! replays are fetched together, not one after another (burst plan F7).
//! Against an object store, each fetch is a round trip; a hundred logs
//! fetched in turn take a hundred of them before a read is answered.

mod common;

use std::time::Instant;

use common::*;

const NOTES: &str = "notes";
const COMMITS: u8 = 100;
/// How long each fetch from the store takes.
const ROUND_TRIP: Duration = Duration::from_millis(20);

/// A store each fetch from which takes [`ROUND_TRIP`], and which counts how
/// many fetches were under way at once, at the most.
#[derive(Debug)]
struct Distant {
    inner: Arc<dyn Blobs>,
    now: AtomicUsize,
    most: AtomicUsize,
}

impl Distant {
    fn over(inner: Arc<dyn Blobs>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            now: AtomicUsize::new(0),
            most: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl Blobs for Distant {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(ROUND_TRIP).await;
        let got = self.inner.get(key).await;
        self.now.fetch_sub(1, Ordering::SeqCst);
        got
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// `COMMITS` commits of a note each, never flushed.
async fn committed_one_by_one() -> Cluster {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create(NOTES)]).await.unwrap();
    for n in 1..=COMMITS {
        writer
            .commit(vec![upsert(NOTES, n, i64::from(n))])
            .await
            .unwrap();
    }
    cluster
}

fn every_note(state: &State) -> Vec<Option<i64>> {
    (1..=COMMITS).map(|n| n_of(state, NOTES, n)).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_copy_opened_fetches_the_logs_it_replays_together() {
    let cluster = committed_one_by_one().await;
    let distant = Distant::over(cluster.blobs.clone());
    let began = Instant::now();
    let copy = Follower::open_through(
        cluster.key.clone(),
        distant.clone(),
        cluster.coordinator.clone(),
        &cluster.committed().await,
    )
    .await
    .unwrap();
    let took = began.elapsed();
    let most = distant.most.load(Ordering::SeqCst);
    assert!(most >= 8, "at most {most} fetches at once");
    let serial = ROUND_TRIP * u32::from(COMMITS);
    assert!(
        took < serial / 4,
        "opened in {took:?}; one after another takes {serial:?}"
    );
    let expected: Vec<Option<i64>> = (1..=COMMITS).map(|n| Some(i64::from(n))).collect();
    assert_eq!(copy.partition().read(every_note), expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_writer_opened_fetches_the_logs_it_replays_together() {
    let cluster = committed_one_by_one().await;
    let distant = Distant::over(cluster.blobs.clone());
    let (lease, committed) = cluster.lease().await;
    let began = Instant::now();
    let writer = Writer::open(
        distant.clone(),
        cluster.coordinator.clone(),
        lease,
        &committed,
    )
    .await
    .unwrap();
    let took = began.elapsed();
    let most = distant.most.load(Ordering::SeqCst);
    assert!(most >= 8, "at most {most} fetches at once");
    let serial = ROUND_TRIP * u32::from(COMMITS);
    assert!(
        took < serial / 4,
        "opened in {took:?}; one after another takes {serial:?}"
    );
    let expected: Vec<Option<i64>> = (1..=COMMITS).map(|n| Some(i64::from(n))).collect();
    assert_eq!(writer.partition().read(every_note), expected);
}

/// A partition replaying many commits to a collection coded in memory codes
/// their rows on several threads before applying them: how many it codes
/// at once, at the most, shows in what the load reports.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partition_replaying_coded_rows_codes_them_several_at_once() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut schema = schema(NOTES);
    schema.embedding.dimensions = 64;
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(schema))])
        .await
        .unwrap();
    let mut random = Random(61);
    for batch in 0..40u32 {
        let rows: Vec<Mutation> = (0..100u32)
            .map(|n| Mutation::Upsert {
                collection: name(NOTES),
                id: ObjectId(uuid::Uuid::from_u128(u128::from(batch * 100 + n))),
                properties: json!({ "title": "fox", "n": n })
                    .as_object()
                    .unwrap()
                    .clone(),
                vector: Some(random.vector(64)),
                only_if_version: None,
            })
            .collect();
        writer.commit(rows).await.unwrap();
    }
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
        replayed.coded_on_threads > 1,
        "{replayed:?}: the rows past the threshold coded on one thread"
    );
    assert_eq!(
        copy.partition()
            .read(|state| state.memtable_coding(&name(NOTES)).unwrap().coded),
        4_000 - vellev_storage::CODE_FROM
    );
}

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
