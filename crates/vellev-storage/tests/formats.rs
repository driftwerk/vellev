//! Files written by released formats, read by this one. A released
//! format's fixtures are never written again: they stand for what is in
//! object storage somewhere. `write_fixtures` writes those of the newest
//! format, and only into a directory that does not exist yet, when
//! `VELLEV_WRITE_FIXTURES` is set.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use object_store::local::LocalFileSystem;
use vellev_core::AccessRule;
use vellev_storage::format::{SegmentIndex, FORMAT_VERSION};
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
    let properties = json!({
        "title": format!("row {n} {}", ["fox", "hen", "owl"][n as usize % 3]),
        "n": n,
        "score": f64::from(n) / 4.0,
        "tags": [format!("t{}", n % 5), "all"],
        "seen": format!("2026-10-{:02}T09:00:00Z", 1 + n % 28),
        "team": format!("team:{}", n % 3),
        "reviewed": n.is_multiple_of(2),
    });
    Mutation::Upsert {
        collection: name("notes"),
        id: object(n),
        properties: properties.as_object().unwrap().clone(),
        vector: Some(vector(n)),
    }
}

fn gone(n: u32) -> Mutation {
    Mutation::Delete {
        collection: name("notes"),
        id: object(n),
    }
}

/// The schema the fixtures are written with: every kind of property, an
/// access rule, and parameters other than the defaults.
fn notes() -> Collection {
    let base = schema("notes");
    Collection {
        properties: [
            base.properties.clone(),
            vec![
                Property::new("seen", DataType::Date),
                Property::new("team", DataType::Text),
            ],
        ]
        .concat(),
        distance: Distance::Dot,
        bm25: Bm25 { k1: 1.5, b: 0.5 },
        access_rule: Some(AccessRule {
            owner: "title".into(),
            team: Some("team".into()),
            team_prefix: "team:".into(),
        }),
        ..base
    }
}

/// A partition with every kind of file and op: a large segment with its
/// vectors and clustered index, a later segment with tombstones under a
/// manifest naming the one before, and commits in the log since.
#[tokio::test]
async fn write_fixtures() {
    if std::env::var_os("VELLEV_WRITE_FIXTURES").is_none() {
        return;
    }
    let dir = fixtures(FORMAT_VERSION);
    assert!(
        !dir.exists(),
        "{} exists; a released format's fixtures are never written again",
        dir.display()
    );
    std::fs::create_dir_all(&dir).unwrap();
    let mut cluster = Cluster::new().await;
    let store = LocalFileSystem::new_with_prefix(&dir).unwrap();
    cluster.blobs = Arc::new(ObjectBlobs::on_disk(Arc::new(store), &dir));
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer
        .commit(vec![
            Mutation::CreateCollection(Box::new(notes())),
            create("scratch"),
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
    writer
        .commit(vec![
            gone(10),
            row(ROWS + 1),
            Mutation::DropCollection(name("scratch")),
        ])
        .await
        .unwrap();
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
async fn the_fixtures_of_every_format_read_as_they_were_written() {
    for format in 1..=FORMAT_VERSION {
        let follower = opened(format).await;
        follower.partition().read(|state| {
            let notes = name("notes");
            let mut schema = self::notes();
            schema
                .properties
                .push(Property::new("reviewed", DataType::Bool));
            assert_eq!(state.collection(&notes), Some(&schema));
            assert!(state.collection(&name("scratch")).is_none());
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
            let twelve = &state.get(&notes, object(12)).unwrap().object.properties;
            assert_eq!(
                twelve["tags"],
                Value::TextArray(vec!["t2".into(), "all".into()])
            );
            assert_eq!(twelve["reviewed"], Value::Bool(true));
            let reviewed = Filter::Eq {
                property: "reviewed".into(),
                value: json!(true),
            };
            let even = (11..=ROWS + 1).filter(|n| n.is_multiple_of(2)).count();
            assert_eq!(state.select(&notes, &reviewed).unwrap().len(), even);

            let segments: Vec<_> = state.segments().cloned().collect();
            assert_eq!(segments.len(), 2);
            assert!(segments[0].index.is_some() && segments[0].vectors.is_some());
            assert!(segments[1].tombstones > 0);
            let nearest = state
                .search(&notes, &Query::nearest(&vector(2_003), 5))
                .unwrap();
            assert_eq!(nearest.len(), 5);
            let words = state
                .keyword(&notes, &Keywords::matching("owl", 3))
                .unwrap();
            assert_eq!(words.len(), 3);
        });
    }
}

/// Every file of every format's fixtures decodes as its kind and is
/// stamped with the format, and the manifests chain back through the
/// checkpoint before.
#[test]
fn every_fixture_file_decodes_as_what_it_is() {
    for format in 1..=FORMAT_VERSION {
        let partition = fixtures(format).join("tenant-a/0");
        let mut manifests = Vec::new();
        for kind in ["log", "manifest", "segment", "vectors", "index"] {
            let mut files: Vec<PathBuf> = std::fs::read_dir(partition.join(kind))
                .unwrap_or_else(|_| panic!("format {format} has no {kind} files"))
                .map(|entry| entry.unwrap().path())
                .collect();
            files.sort();
            assert!(!files.is_empty(), "format {format} has no {kind} files");
            for path in files {
                let name = path.file_name().unwrap().to_string_lossy();
                let key = format!("tenant-a/0/{kind}/{name}");
                let frame = std::fs::read(&path).unwrap();
                assert_eq!(
                    u16::from_le_bytes([frame[5], frame[6]]),
                    format,
                    "{key} is stamped with another format"
                );
                match kind {
                    "log" => {
                        format::decode::<LogSegment>(&key, Kind::Log, &frame).unwrap();
                    }
                    "manifest" => {
                        let manifest: Manifest =
                            format::decode(&key, Kind::Manifest, &frame).unwrap();
                        manifests.push((key, manifest));
                    }
                    "segment" => {
                        format::decode::<SegmentFile>(&key, Kind::Segment, &frame).unwrap();
                    }
                    "index" => {
                        let index: SegmentIndex =
                            format::decode(&key, Kind::Index, &frame).unwrap();
                        assert!(!index.vectors.is_empty());
                    }
                    _ => format::verify(&key, &frame).unwrap(),
                }
            }
        }
        let (first, newest) = (&manifests[0], &manifests[manifests.len() - 1]);
        assert_eq!(first.1.prev, None);
        assert_eq!(newest.1.prev.as_deref(), Some(first.0.as_str()));
    }
}
