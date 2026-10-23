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
use vellev_core::RequestId;
use vellev_storage::format::{SegmentIndex, FORMAT_VERSION};
use vellev_storage::{Batch, Embedded, Keywords, Merging, Query};

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
        only_if_version: None,
    }
}

fn gone(n: u32) -> Mutation {
    Mutation::Delete {
        collection: name("notes"),
        id: object(n),
        only_if_version: None,
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

/// From format 2: a collection embedding a property it does not store.
fn pages() -> Collection {
    Collection {
        name: name("pages"),
        properties: vec![
            Property::new("title", DataType::Text),
            Property::new("body", DataType::Text).unstored(),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::Local,
            model: "fixture".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec!["title".into(), "body".into()],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

fn page(n: u32) -> Mutation {
    Mutation::Upsert {
        collection: name("pages"),
        id: object(n),
        properties: json!({ "title": format!("page {n}"), "body": format!("body {n}") })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector(n)),
        only_if_version: None,
    }
}

/// A partition with every kind of file and op: a large segment with its
/// vectors and clustered index, then a later segment with tombstones
/// flushed while commits went on, under a manifest naming the checkpoint
/// before and the logs to replay over it, and requests answered both before
/// the manifest and in a log it replays.
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
    let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
    let writing = Arc::new(std::sync::atomic::AtomicU16::new(FORMAT_VERSION));
    let writer = Arc::new(
        cluster
            .writer()
            .await
            .writing_format(writing)
            .merging_with(Merging {
                clustered_above: 4_000,
                ..Merging::default()
            }),
    );
    writer
        .commit(vec![
            Mutation::CreateCollection(Box::new(notes())),
            create("scratch"),
        ])
        .await
        .unwrap();
    // What format 2 added, from the first commit after the schemas.
    writer
        .commit(vec![
            Mutation::CreateCollection(Box::new(pages())),
            page(1),
            page(2),
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
    let named = |n: u128, mutations| Batch {
        request: Some(RequestId(uuid::Uuid::from_u128(n))),
        mutations,
    };
    let made = writer
        .commit_batches(
            vec![named(1, (0..10).map(gone).chain([row(ROWS)]).collect())],
            Embedded::new(),
        )
        .await
        .unwrap();
    assert!(made.iter().all(Result::is_ok));
    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(stored.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);
    writer
        .commit(vec![
            gone(10),
            row(ROWS + 1),
            Mutation::DropCollection(name("scratch")),
        ])
        .await
        .unwrap();
    let patch = Mutation::Patch {
        collection: name("notes"),
        id: object(11),
        properties: json!({ "n": 1_011 }).as_object().unwrap().clone(),
        vector: None,
        only_if: None,
        only_if_version: None,
    };
    let made = writer
        .commit_batches(vec![named(2, vec![patch])], Embedded::new())
        .await
        .unwrap();
    assert!(made.iter().all(Result::is_ok));
    paused.release.notify_one();
    flushing.await.unwrap().unwrap();

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
            if format >= 2 {
                let pages = name("pages");
                assert_eq!(state.collection(&pages), Some(&self::pages()));
                for n in [1, 2] {
                    let found = state.get(&pages, object(n)).unwrap();
                    let title = Value::Text(format!("page {n}"));
                    assert_eq!(found.object.properties.get("title"), Some(&title));
                    assert!(!found.object.properties.contains_key("body"));
                    assert_eq!(found.vector, Some(&vector(n)[..]));
                }
            }
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
        assert_eq!(
            newest.1.replay.len(),
            2,
            "logs committed while it was built"
        );
        for log in &newest.1.replay {
            assert!(partition
                .join(log.trim_start_matches("tenant-a/0/"))
                .is_file());
        }
        let request = |n: u128| RequestId(uuid::Uuid::from_u128(n));
        let kept: Vec<RequestId> = newest
            .1
            .answered
            .iter()
            .map(|(_, answered)| answered.request)
            .collect();
        assert!(kept.contains(&request(1)), "{kept:?}");
        let replayed = newest.1.replay.last().unwrap();
        let log: LogSegment = format::decode(
            replayed,
            Kind::Log,
            &std::fs::read(partition.join(replayed.trim_start_matches("tenant-a/0/"))).unwrap(),
        )
        .unwrap();
        assert_eq!(log.answered[0].request, request(2));
    }
}

/// The newest checkpoint of `format`'s fixtures and its key, with the
/// frame it is kept as.
fn newest_checkpoint(format: u16) -> (String, Bytes, Manifest) {
    let directory = fixtures(format).join("tenant-a/0/manifest");
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    let path = files.last().unwrap();
    let name = path.file_name().unwrap().to_string_lossy();
    let key = format!("tenant-a/0/manifest/{name}");
    let frame = Bytes::from(std::fs::read(path).unwrap());
    let manifest = format::decode(&key, Kind::Manifest, &frame).unwrap();
    (key, frame, manifest)
}

fn holds(frame: &[u8], key: &str) -> bool {
    frame
        .windows(key.len())
        .any(|window| window == key.as_bytes())
}

/// A checkpoint a release wrote, with segments, an index, vectors, logs
/// to replay and a checkpoint before it: remapped, nothing in it is the
/// key of a file as that release wrote it; remapped to the same keys, it
/// is written byte for byte as it was.
#[test]
fn a_released_checkpoint_is_remapped_whole_and_written_as_it_was_under_the_same_keys() {
    let (key, frame, manifest) = newest_checkpoint(FORMAT_VERSION);
    assert!(manifest.prev.is_some() && !manifest.replay.is_empty());
    assert!(manifest
        .segments
        .iter()
        .any(|segment| segment.index.is_some() && segment.vectors.is_some()));
    // Every file there is but the checkpoint itself, which nothing in it
    // names, and so every one it could name.
    let partition = fixtures(FORMAT_VERSION);
    let mut names = Vec::new();
    for kind in ["log", "manifest", "segment", "vectors", "index"] {
        for entry in std::fs::read_dir(partition.join("tenant-a/0").join(kind)).unwrap() {
            let name = entry.unwrap().file_name();
            names.push(format!("tenant-a/0/{kind}/{}", name.to_string_lossy()));
        }
    }
    names.retain(|name| *name != key);
    let named: Vec<&String> = names.iter().filter(|name| holds(&frame, name)).collect();
    assert_eq!(named.len(), 1 + 2 + 2 + 1 + 2, "{named:?}");

    let same = names.iter().map(|name| (name.clone(), name.clone()));
    let unchanged = manifest.clone().remapped(&key, &same.collect()).unwrap();
    assert_eq!(format::encode(Kind::Manifest, &unchanged).unwrap(), frame);

    let fresh = names
        .iter()
        .map(|name| (name.clone(), name.replace("tenant-a/0/", "tenant-b/3/")));
    let remapped = manifest.remapped(&key, &fresh.collect()).unwrap();
    let written = format::encode(Kind::Manifest, &remapped).unwrap();
    assert!(!holds(&written, "tenant-a/0/"));
    let read: Manifest = format::decode(&key, Kind::Manifest, &written).unwrap();
    assert_eq!(read, remapped);
}

/// An index file a release wrote says its segment's key. Re-keyed, it
/// says the new one, and holds the indexes it held.
#[test]
fn a_released_index_file_is_rekeyed_whole() {
    let (_, _, manifest) = newest_checkpoint(FORMAT_VERSION);
    let segment = manifest
        .segments
        .iter()
        .find(|segment| segment.index.is_some())
        .unwrap();
    let key = segment.index.clone().unwrap();
    let frame = std::fs::read(fixtures(FORMAT_VERSION).join(&key)).unwrap();
    let was: SegmentIndex = format::decode(&key, Kind::Index, &frame).unwrap();
    assert_eq!(was.segment, segment.key);

    let to = segment.key.replace("tenant-a/0/", "tenant-b/3/");
    let rekeyed = format::rekeyed_index(&key, &frame, &segment.key, &to).unwrap();
    assert!(!holds(&rekeyed, &segment.key));
    let now: SegmentIndex = format::decode(&key, Kind::Index, &rekeyed).unwrap();
    assert_eq!(now.segment, to);
    assert_eq!(now.vectors.len(), was.vectors.len());
    let of = |index: &SegmentIndex| -> Vec<_> { index.vectors.iter().map(|(of, _)| *of).collect() };
    assert!(!was.vectors.is_empty());
    assert_eq!(of(&now), of(&was));
    // Given its key back it is the file it was, to the byte: nothing of
    // the indexes was touched.
    assert_eq!(
        format::rekeyed_index(&key, &rekeyed, &to, &segment.key).unwrap(),
        frame
    );
    // One of the other segment of the checkpoint is left alone.
    let other = &manifest
        .segments
        .iter()
        .find(|other| other.key != segment.key);
    assert!(matches!(
        format::rekeyed_index(&key, &frame, &other.unwrap().key, &to),
        Err(StorageError::Corrupt { .. })
    ));
}
