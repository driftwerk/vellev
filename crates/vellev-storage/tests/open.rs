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
    ); // Coded on several threads, a run at a time, the codes are those the
       // writer made applying each commit as it was made, bit for bit.
    let digest = |partition: &vellev_storage::Partition| {
        partition.read(|state| state.memtable_coding(&name(NOTES)).unwrap().digest)
    };
    assert_eq!(digest(copy.partition()), digest(writer.partition()));
}

/// A store that counts the logs and checkpoints fetched from it.
#[derive(Debug)]
struct Counting {
    inner: Arc<dyn Blobs>,
    heads: AtomicUsize,
}

#[async_trait]
impl Blobs for Counting {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.ends_with(".log") || key.ends_with(".manifest") {
            self.heads.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// The heads a copy opened cold fetches, `after` commits made since the
/// writer last checkpointed a long history.
async fn heads_fetched_opening(after: u8) -> usize {
    let cluster = committed_one_by_one().await;
    let writer = cluster.writer().await;
    writer.flush().await.unwrap();
    for n in 1..=after {
        writer
            .commit(vec![upsert(NOTES, n, -i64::from(n))])
            .await
            .unwrap();
    }
    let counting = Arc::new(Counting {
        inner: cluster.blobs.clone(),
        heads: AtomicUsize::new(0),
    });
    let copy = Follower::open_through(
        cluster.key.clone(),
        counting.clone(),
        cluster.coordinator.clone(),
        &cluster.committed().await,
    )
    .await
    .unwrap();
    for n in 1..=after {
        assert_eq!(
            copy.partition().read(|state| n_of(state, NOTES, n)),
            Some(-i64::from(n))
        );
    }
    counting.heads.load(Ordering::SeqCst)
}

#[tokio::test]
async fn a_cold_open_fetches_no_log_from_before_the_newest_checkpoint() {
    // The logs since the checkpoint, and the checkpoint: none before it,
    // though the coordinator lists a hundred commits before it.
    assert_eq!(heads_fetched_opening(3).await, 4);
    // A checkpoint at the head: that alone.
    assert_eq!(heads_fetched_opening(0).await, 1);
}

/// A property added searchable among commits replayed together: the rows
/// after it are found by its words too, in a copy as in the writer that
/// applied them one by one, though the commits were split into tokens
/// together before the property was there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_property_made_searchable_mid_replay_is_searched_from_then_on() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create(NOTES)]).await.unwrap();
    let note = |n: u32| {
        let mut properties = json!({ "title": "hen", "n": n });
        if (4..100).contains(&n) {
            properties["body"] = json!("otter");
        }
        Mutation::Upsert {
            collection: name(NOTES),
            id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
            properties: properties.as_object().unwrap().clone(),
            vector: None,
            only_if_version: None,
        }
    };
    // Enough commits first that the ones around the property are split
    // into tokens together with the collection there, as it was before it.
    for n in 100..116u32 {
        writer.commit(vec![note(n)]).await.unwrap();
    }
    for n in 0..4u32 {
        writer.commit(vec![note(n)]).await.unwrap();
    }
    writer
        .commit(vec![Mutation::AddProperty {
            collection: name(NOTES),
            property: Property::new("body", DataType::Text).searchable(),
        }])
        .await
        .unwrap();
    for n in 4..8u32 {
        writer.commit(vec![note(n)]).await.unwrap();
    }
    let copy = Follower::open_through(
        cluster.key.clone(),
        cluster.blobs.clone(),
        cluster.coordinator.clone(),
        &cluster.committed().await,
    )
    .await
    .unwrap();
    let found = |state: &State| {
        let mut ids: Vec<u128> = state
            .keyword(
                &name(NOTES),
                &vellev_storage::Keywords::matching("otter", 50),
            )
            .unwrap()
            .iter()
            .map(|scored| scored.object.id.0.as_u128())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(writer.partition().read(found), [4, 5, 6, 7]);
    assert_eq!(copy.partition().read(found), [4, 5, 6, 7]);
}

/// Text split into tokens before the state is locked is what is indexed:
/// it is not split again under the lock, by the writer applying its own
/// commits or by a copy replaying them, while the searchable properties
/// it was split by are still the collection's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tokens_prepared_before_the_lock_are_used_under_it() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create(NOTES)]).await.unwrap();
    // A flush, so that the copy finds the collection in its segment's
    // catalog and replays only the commits after it.
    writer.commit(vec![upsert(NOTES, 0, 0)]).await.unwrap();
    writer.flush().await.unwrap();
    let before = writer.partition().read(|state| state.texts_applied());
    let note = |n: u32| Mutation::Upsert {
        collection: name(NOTES),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": format!("fox {n} owl"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: None,
        only_if_version: None,
    };
    for batch in 0..20u32 {
        writer
            .commit((1..=10).map(|n| note(batch * 10 + n)).collect())
            .await
            .unwrap();
    }
    let after = writer.partition().read(|state| state.texts_applied());
    assert_eq!(
        (
            after.prepared - before.prepared,
            after.under_lock - before.under_lock
        ),
        (200, 0),
        "the writer: {before:?} then {after:?}"
    );
    let copy = Follower::open_through(
        cluster.key.clone(),
        cluster.blobs.clone(),
        cluster.coordinator.clone(),
        &cluster.committed().await,
    )
    .await
    .unwrap();
    let replayed = copy.partition().read(|state| state.texts_applied());
    assert_eq!(
        (replayed.prepared, replayed.under_lock),
        (200, 0),
        "the copy: {replayed:?}"
    );
}
