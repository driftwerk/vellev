//! What a partition publishes of itself as it changes (CI on #165, the
//! 100k gate): read without its lock, by a node for its metrics and its
//! memory budget, it is always what its state comes to after the last
//! change, whatever made it.

mod common;

use common::*;
use vellev_storage::Figures;

/// The published figures are the state's own, as it is now.
fn published(partition: &vellev_storage::Partition) -> (Figures, Figures) {
    (partition.figures(), partition.read(State::figures))
}

/// After each commit, flush, merge and restore, and as a copy is opened
/// and catches up, what is published is what the state comes to.
#[tokio::test]
async fn what_a_partition_publishes_is_its_state_after_every_change() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let same = |what: &str, writer: &Writer| {
        let (figures, state) = published(writer.partition());
        assert_eq!(figures, state, "after {what}");
    };
    writer.commit(vec![create("notes")]).await.unwrap();
    same("creating a collection", &writer);
    for round in 0..4u8 {
        let writes = (0..10).map(|n| upsert("notes", round * 10 + n, i64::from(round)));
        writer.commit(writes.collect()).await.unwrap();
        same("a commit", &writer);
        assert!(writer.partition().figures().unflushed > 0);
        writer.flush().await.unwrap();
        same("a flush", &writer);
        assert_eq!(writer.partition().figures().unflushed, 0);
    }
    assert_eq!(writer.partition().figures().plain, 4);
    let committed = cluster.committed().await;
    let restore_to = (committed.manifest.unwrap(), committed.log_position);
    writer
        .merge()
        .await
        .unwrap()
        .expect("four segments to merge");
    same("a merge", &writer);
    assert_eq!(writer.partition().figures().plain, 1);
    assert_eq!(writer.partition().figures().plain_rows, 40);

    let copy = cluster.open().await;
    let (figures, state) = published(copy.partition());
    assert_eq!(figures, state, "as a copy is opened");
    assert_eq!(figures.plain_rows, 40);
    writer.commit(vec![upsert("notes", 100, 1)]).await.unwrap();
    copy.catch_up(&cluster.committed().await).await.unwrap();
    let (figures, state) = published(copy.partition());
    assert_eq!(figures, state, "as a copy catches up");
    assert_eq!(figures.unflushed, 1);
    // Past a checkpoint made since: the copy loads it afresh.
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 101, 1)]).await.unwrap();
    copy.catch_up(&cluster.committed().await).await.unwrap();
    let (figures, state) = published(copy.partition());
    assert_eq!(figures, state, "as a copy catches up past a checkpoint");
    assert_eq!((figures.plain, figures.unflushed), (2, 1));

    writer
        .restore(Some(&restore_to.0), restore_to.1)
        .await
        .unwrap();
    same("a restore", &writer);
    assert_eq!(writer.partition().figures().plain, 4, "as before the merge");
}
