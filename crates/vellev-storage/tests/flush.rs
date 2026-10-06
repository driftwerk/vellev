mod common;

use common::*;

/// A collection as a reader sees it: its name, schema, and every live
/// object's `n`.
type Seen = (String, Collection, Vec<(ObjectId, Option<i64>)>);

/// Every collection of `state` as a reader sees it.
fn seen(state: &State) -> Vec<Seen> {
    let mut seen: Vec<_> = state
        .collections()
        .map(|schema| {
            let mut objects: Vec<_> = state
                .scan(&schema.name)
                .map(|live| {
                    let n = match live.object.properties.get("n") {
                        Some(Value::Int(n)) => Some(*n),
                        _ => None,
                    };
                    (live.id(), n)
                })
                .collect();
            objects.sort();
            (schema.name.to_string(), schema.clone(), objects)
        })
        .collect();
    seen.sort_by(|a, b| a.0.cmp(&b.0));
    seen
}

#[tokio::test]
async fn commits_made_while_a_flush_builds_are_kept_by_the_writer_and_every_reader() {
    let mut cluster = Cluster::new().await;
    let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![
            create("notes"),
            create("spare"),
            upsert("spare", 1, 1),
        ])
        .await
        .unwrap();
    for n in 1..=6 {
        writer
            .commit(vec![upsert("notes", n, i64::from(n))])
            .await
            .unwrap();
    }
    let before = writer.partition().read(State::position);
    paused.armed.store(true, Ordering::SeqCst);

    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(stored.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);

    let started = std::time::Instant::now();
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a commit does not wait for the flush"
    );
    writer.commit(vec![upsert("notes", 2, 20)]).await.unwrap();
    let midway = cluster.open().await;
    writer
        .commit(vec![delete("notes", 3), upsert("notes", 3, 30)])
        .await
        .unwrap();
    writer
        .commit(vec![
            Mutation::DropCollection(name("spare")),
            create("spare"),
            upsert("spare", 2, 2),
        ])
        .await
        .unwrap();
    writer
        .commit(vec![Mutation::AddProperty {
            collection: name("notes"),
            property: Property::new("reviewed", DataType::Bool),
        }])
        .await
        .unwrap();
    writer.commit(vec![upsert("notes", 7, 7)]).await.unwrap();

    paused.release.notify_one();
    let flushed = flushing.await.unwrap().unwrap();
    assert!(flushed > before);
    let committed = cluster.committed().await;
    assert_eq!(committed.log_position, flushed);
    let head = committed.manifest.clone().unwrap().0;
    let manifest: Manifest = format::decode(
        &head,
        Kind::Manifest,
        &cluster.blobs.get(&head).await.unwrap(),
    )
    .unwrap();
    assert_eq!(manifest.replay.len(), 6, "the commits made while it built");
    assert_eq!(manifest.segments.len(), 1);

    let expected = writer.partition().read(seen);
    let notes = &expected
        .iter()
        .find(|(name, ..)| name == "notes")
        .unwrap()
        .2;
    assert_eq!(
        notes.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
        [Some(20), Some(30), Some(4), Some(5), Some(6), Some(7)]
    );
    assert_eq!(
        writer.partition().read(State::unflushed),
        4,
        "the rows written while it built, still live"
    );

    midway.catch_up(&committed).await.unwrap();
    for (name, copy) in [
        ("a copy caught up from midway", midway.partition().clone()),
        (
            "a copy opened now",
            cluster.open().await.partition().clone(),
        ),
    ] {
        assert_eq!(copy.read(seen), expected, "{name}");
        assert_eq!(copy.read(State::position), flushed, "{name}");
    }

    writer.commit(vec![delete("notes", 4)]).await.unwrap();
    writer.flush().await.unwrap();
    assert_eq!(writer.partition().read(State::unflushed), 0);
    writer.clean(Duration::ZERO).await.unwrap();
    let expected = writer.partition().read(seen);
    assert_eq!(cluster.open().await.partition().read(seen), expected);
}

#[tokio::test]
async fn a_flush_overtaken_by_a_checkpoint_leaves_it_be() {
    let mut cluster = Cluster::new().await;
    let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    paused.armed.store(true, Ordering::SeqCst);
    let first = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(stored.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);
    let second = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    tokio::task::yield_now().await;
    paused.release.notify_one();
    let (first, second) = (
        first.await.unwrap().unwrap(),
        second.await.unwrap().unwrap(),
    );
    assert_eq!(first, second, "the second found nothing more to flush");
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}
