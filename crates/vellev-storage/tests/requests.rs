mod common;

use common::*;
use vellev_core::RequestId;
use vellev_storage::{Batch, Embedded, Receipt};

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
async fn a_refused_request_is_not_remembered() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let request = RequestId::generate();
    let batch = || Batch {
        request: Some(request),
        mutations: vec![upsert("notes", 1, 7)],
    };
    let refused = made(&writer, vec![batch()]).await.pop().unwrap();
    assert!(refused.is_err());
    writer.commit(vec![create("notes")]).await.unwrap();
    made(&writer, vec![batch()]).await.pop().unwrap().unwrap();
    writer
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(7)));
}
