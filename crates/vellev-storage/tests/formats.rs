//! Files written by released formats, read by this one. A released
//! format's fixtures are never written again: they stand for what is in
//! object storage somewhere. `write_format_1` wrote `fixtures/format-1`
//! and is kept to show how; run it only to make the fixtures of a format
//! not yet released.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use object_store::local::LocalFileSystem;
use vellev_storage::format::FORMAT_VERSION;
use vellev_storage::{Keywords, Merging, Query};

const ROWS: u32 = 4_100;

fn fixtures(format: u16) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("format-{format}"))
}

fn object(n: u32) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
}

/// Row `n`'s vector: one of a few directions, a little apart.
fn vector(n: u32) -> Vec<f32> {
    let angle = (n % 8) as f32 * std::f32::consts::FRAC_PI_4 + (n % 97) as f32 * 1e-3;
    vec![angle.cos(), angle.sin()]
}

fn row(n: u32) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: object(n),
        properties:
            json!({ "title": format!("row {n} {}", ["fox", "hen", "owl"][n as usize % 3]), "n": n })
                .as_object()
                .unwrap()
                .clone(),
        vector: Some(vector(n)),
    }
}

fn gone(n: u32) -> Mutation {
    Mutation::Delete {
        collection: name("notes"),
        id: object(n),
    }
}

/// A partition with every kind of file: a large segment with its vectors
/// and clustered index, a later segment with tombstones and a manifest
/// naming the one before, and commits in the log since.
#[tokio::test]
#[ignore = "writes the fixtures of a format not yet released"]
async fn write_format_1() {
    let dir = fixtures(1);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut cluster = Cluster::new().await;
    let store = LocalFileSystem::new_with_prefix(&dir).unwrap();
    cluster.blobs = Arc::new(ObjectBlobs::on_disk(Arc::new(store), &dir));
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer.commit(vec![create("notes")]).await.unwrap();
    for rows in (0..ROWS).collect::<Vec<_>>().chunks(500) {
        writer
            .commit(rows.iter().map(|n| row(*n)).collect())
            .await
            .unwrap();
    }
    writer.flush().await.unwrap();
    writer
        .commit((0..10).map(gone).chain([row(ROWS)]).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![gone(10), row(ROWS + 1)]).await.unwrap();
    writer
        .commit(vec![Mutation::Patch {
            collection: name("notes"),
            id: object(11),
            properties: json!({ "n": 1_011 }).as_object().unwrap().clone(),
            vector: None,
            only_if: None,
        }])
        .await
        .unwrap();

    let committed = cluster.committed().await;
    let described = json!({
        "generation": committed.generation.0,
        "incarnation": committed.incarnation.0,
        "log_position": committed.log_position.0,
        "manifest": committed.manifest.map(|head| head.0),
    });
    std::fs::write(
        dir.join("committed.json"),
        serde_json::to_vec_pretty(&described).unwrap(),
    )
    .unwrap();
}

/// The committed state the fixtures were written under.
fn committed(dir: &Path) -> Committed {
    let text = std::fs::read(dir.join("committed.json")).unwrap();
    let described: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let number = |name: &str| described[name].as_u64().unwrap();
    Committed {
        generation: Generation(number("generation")),
        incarnation: Incarnation(number("incarnation")),
        log_position: LogPosition(number("log_position")),
        manifest: described["manifest"]
            .as_str()
            .map(|head| ManifestRef(head.to_string())),
    }
}

async fn opened(format: u16) -> Follower {
    let dir = fixtures(format);
    let store = LocalFileSystem::new_with_prefix(&dir).unwrap();
    let blobs = Arc::new(ObjectBlobs::on_disk(Arc::new(store), &dir));
    let key = PartitionKey {
        tenant: TenantId::new("tenant-a").unwrap(),
        partition: PartitionNumber::DEFAULT,
    };
    Follower::open(key, blobs, &committed(&dir)).await.unwrap()
}

#[tokio::test]
async fn files_of_format_1_read_as_they_were_written() {
    let follower = opened(1).await;
    follower.partition().read(|state| {
        let notes = name("notes");
        assert_eq!(state.count(&notes), ROWS as usize + 2 - 11);
        for n in 0..=10 {
            assert!(state.get(&notes, object(n)).is_none(), "{n} was deleted");
        }
        let eleven = state.get(&notes, object(11)).unwrap();
        assert_eq!(eleven.object.properties["n"], Value::Int(1_011));
        assert_eq!(eleven.vector, Some(&vector(11)[..]));
        for n in [12, 2_000, ROWS - 1, ROWS, ROWS + 1] {
            let found = state.get(&notes, object(n)).unwrap();
            assert_eq!(found.object.properties["n"], Value::Int(i64::from(n)));
            assert_eq!(found.vector, Some(&vector(n)[..]), "{n}");
        }
        let segments: Vec<_> = state.segments().cloned().collect();
        assert_eq!(segments.len(), 2);
        assert!(segments[0].index.is_some() && segments[0].vectors.is_some());
        assert!(segments[1].tombstones > 0);

        let nearest = state
            .search(&notes, &Query::nearest(&vector(2_003), 5))
            .unwrap();
        assert_eq!(nearest.len(), 5);
        assert!(nearest.iter().all(|hit| hit.distance < 0.01), "{nearest:?}");
        let words = state
            .keyword(&notes, &Keywords::matching("owl", 3))
            .unwrap();
        assert_eq!(words.len(), 3);
    });
}

#[test]
fn every_format_this_release_reads_has_fixtures() {
    for format in 1..=FORMAT_VERSION {
        let dir = fixtures(format);
        assert!(
            dir.join("committed.json").is_file(),
            "{} holds no fixtures",
            dir.display()
        );
    }
}
