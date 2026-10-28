//! Which work is counted as a build, during which reads do not split
//! (vellev_index::pool): the heavy parallel sections, and not an ordinary
//! flush. A binary of its own, one test, so that nothing else builds
//! while it counts.

mod common;

use common::*;
use vellev_storage::Merging;

const DIMENSIONS: usize = 8;

fn wide() -> Mutation {
    let mut schema = schema("notes");
    schema.embedding.dimensions = DIMENSIONS as u32;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(n: u32, random: &mut Random) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": format!("row {n}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(random.vector(DIMENSIONS)),
        only_if_version: None,
    }
}

async fn written(writer: &Writer, from: u32, rows: u32, random: &mut Random) {
    for start in (from..from + rows).step_by(1_000) {
        let writes = (start..(start + 1_000).min(from + rows)).map(|n| row(n, random));
        writer.commit(writes.collect()).await.unwrap();
    }
}

#[tokio::test]
async fn heavy_sections_are_counted_as_builds_and_an_ordinary_flush_is_not() {
    let begun = vellev_index::pool::builds_begun;
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer.commit(vec![wide()]).await.unwrap();
    let mut random = Random(41);

    // An ordinary flush, of a few rows: not a build.
    written(&writer, 0, 1_200, &mut random).await;
    let before = begun();
    writer.flush().await.unwrap();
    assert_eq!(begun(), before, "a flush of 1,200 rows is not counted");

    // A merge that builds a clustered index: a build.
    for flush in 1..4 {
        written(&writer, flush * 1_200, 1_200, &mut random).await;
        writer.flush().await.unwrap();
    }
    let before = begun();
    writer.merge().await.unwrap().expect("four segments merge");
    assert!(begun() > before, "a merge that clusters is counted");

    // A reload of a segment of many rows: its indexing is a build.
    written(
        &writer,
        10_000,
        vellev_storage::HEAVY_ROWS as u32,
        &mut random,
    )
    .await;
    writer.flush().await.unwrap();
    drop(writer);
    let before = begun();
    let opened = cluster.open().await;
    assert!(
        begun() > before,
        "indexing a large segment on open is counted"
    );
    drop(opened);
}
