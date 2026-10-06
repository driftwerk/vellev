mod common;

use common::*;
use vellev_core::RequestId;
use vellev_storage::{Batch, Embedded, Outcome, Receipt};

fn increment(object: u8, by: f64) -> Mutation {
    Mutation::Increment {
        collection: name("notes"),
        id: id(object),
        property: "n".into(),
        by,
    }
}

async fn made(writer: &Writer, batches: Vec<Batch>) -> Vec<vellev_storage::Result<Receipt>> {
    writer
        .commit_batches(batches, Embedded::new())
        .await
        .unwrap()
}

async fn requested(writer: &Writer, request: RequestId, mutations: Vec<Mutation>) -> Receipt {
    let batch = Batch {
        request: Some(request),
        mutations,
    };
    made(writer, vec![batch]).await.pop().unwrap().unwrap()
}

/// What happens between a request's first answer and its try again.
#[derive(Debug, Clone, Copy)]
enum Between {
    Nothing,
    Flush,
    Merge,
    Takeover,
    FlushAndTakeover,
}

#[tokio::test]
async fn a_request_tried_again_is_answered_as_it_was_and_a_later_write_survives() {
    for between in [
        Between::Nothing,
        Between::Flush,
        Between::Merge,
        Between::Takeover,
        Between::FlushAndTakeover,
    ] {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await;
        writer
            .commit(vec![create("notes"), upsert("notes", 1, 0)])
            .await
            .unwrap();
        let request = RequestId::generate();
        let first = requested(&writer, request, vec![increment(1, 5.0)]).await;
        // Another caller writes the same object before the first tries
        // again, its answer having been lost.
        writer.commit(vec![upsert("notes", 1, 100)]).await.unwrap();
        let writer = match between {
            Between::Nothing => writer,
            Between::Flush => {
                writer.flush().await.unwrap();
                writer
            }
            Between::Merge => {
                writer.flush().await.unwrap();
                writer.commit(vec![upsert("notes", 2, 0)]).await.unwrap();
                writer.flush().await.unwrap();
                while writer.merge().await.unwrap().is_some() {}
                writer
            }
            Between::Takeover => cluster.writer().await,
            Between::FlushAndTakeover => {
                writer.flush().await.unwrap();
                cluster.writer().await
            }
        };
        let again = requested(&writer, request, vec![increment(1, 5.0)]).await;
        assert_eq!(again, first, "{between:?}");
        writer
            .partition()
            .read(|state| assert_eq!(n_of(state, "notes", 1), Some(100), "{between:?}"));
    }
}

#[tokio::test]
async fn a_request_twice_in_one_commit_is_made_once() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 0)])
        .await
        .unwrap();
    let request = RequestId::generate();
    let batch = |by| Batch {
        request: Some(request),
        mutations: vec![increment(1, by)],
    };
    let results = made(
        &writer,
        vec![batch(1.0), vec![increment(1, 10.0)].into(), batch(1.0)],
    )
    .await;
    let results: Vec<Receipt> = results.into_iter().map(Result::unwrap).collect();
    assert_eq!(results[2], results[0]);
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(11)));

    let request = RequestId::generate();
    let refused = || Batch {
        request: Some(request),
        mutations: vec![upsert("missing", 1, 0)],
    };
    let results = made(&writer, vec![refused(), refused()]).await;
    assert!(results.iter().all(Result::is_err), "{results:?}");
}

#[tokio::test]
async fn a_refused_request_is_refused_again_as_it_was() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let request = RequestId::generate();
    let batch = || Batch {
        request: Some(request),
        mutations: vec![upsert("notes", 1, 7)],
    };
    let refused = made(&writer, vec![batch()]).await.pop().unwrap();
    let refusal = refused.unwrap_err().refusal().unwrap();
    writer.commit(vec![create("notes")]).await.unwrap();
    let again = made(&writer, vec![batch()]).await.pop().unwrap();
    assert_eq!(again.unwrap_err().refusal(), Some(refusal));
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), None));
}

#[tokio::test]
async fn a_request_that_changed_nothing_does_not_undo_a_later_write_when_it_comes_late() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    let request = RequestId::generate();
    let delete = || Batch {
        request: Some(request),
        mutations: vec![delete("notes", 1)],
    };
    let first = made(&writer, vec![delete()]).await.pop().unwrap().unwrap();
    assert_eq!(first.outcomes, [Outcome::Unchanged]);
    writer.commit(vec![upsert("notes", 1, 5)]).await.unwrap();
    let late = made(&writer, vec![delete()]).await.pop().unwrap().unwrap();
    assert_eq!(late, first);
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(5)));
}

#[tokio::test]
async fn a_request_made_while_a_flush_builds_is_known_to_whoever_takes_over_and_every_copy() {
    let mut cluster = Cluster::new().await;
    let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 0)])
        .await
        .unwrap();
    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(stored.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);
    let request = RequestId::generate();
    let first = requested(&writer, request, vec![increment(1, 5.0)]).await;
    paused.release.notify_one();
    flushing.await.unwrap().unwrap();
    writer.commit(vec![upsert("notes", 1, 100)]).await.unwrap();

    let copy = cluster.open().await;
    assert!(copy.partition().read(|state| state.answered(request)));
    let taken = cluster.writer().await;
    let again = requested(&taken, request, vec![increment(1, 5.0)]).await;
    assert_eq!(again, first);
    taken
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(100)));
}
