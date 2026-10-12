//! Links written by released link versions, read by this one. A released
//! version's fixtures are never written again: they stand for what is in a
//! backup store somewhere. `write_link_fixtures` writes those of the newest
//! version, and only into a directory that does not exist yet, when
//! `VELLEV_WRITE_FIXTURES` is set.

use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use uuid::Uuid;
use vellev_core::{Partition, TenantId};
use vellev_storage::coordinator::{ClusterId, Generation, Incarnation, Lineage, LogPosition};
use vellev_storage::format::{self, Kind, LogSegment};
use vellev_storage::link::{self, Before, Body, Head, Header, Needed, Sum, Timeline, LINK_VERSION};

const WRITER: &str = "8560994bc5f64972b4d9de1d3358753b";

fn fixtures(version: u16) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("link-{version}"))
}

fn key(kind: &str, position: u64) -> String {
    format!("tenant-a/0/{kind}/{position:020}-00000000000000000002-{WRITER}.{kind}")
}

/// The frame of the log at position 7.
fn frame() -> bytes::Bytes {
    let log = LogSegment {
        position: LogPosition(7),
        prev: Some(key("log", 6)),
        ops: Vec::new(),
        answered: Vec::new(),
    };
    format::encode(Kind::Log, &log).unwrap()
}

/// The header of the log at position 7, inline.
fn inline() -> Header {
    let frame = frame();
    Header {
        tenant: TenantId::new("tenant-a").unwrap(),
        lineage: Lineage(Uuid::from_u128(1)),
        cluster: ClusterId(Uuid::from_u128(2)),
        incarnation: Incarnation(3),
        partition: Partition(0),
        position: LogPosition(7),
        key: key("log", 7),
        head: Head::Log,
        generation: Generation(2),
        committed_at: Utc.with_ymd_and_hms(2026, 10, 12, 9, 0, 0).unwrap()
            + chrono::Duration::microseconds(123_456),
        sum: Sum::of(&frame),
        size: frame.len() as u64,
        before: Some(Before {
            position: LogPosition(6),
            sum: Sum::of(b"the frame at six"),
        }),
        files: Vec::new(),
        body: Body::Inline,
    }
}

/// The header of a manifest at position 8, spilled.
fn spilled() -> Header {
    let needed = |kind: &str| Needed {
        key: key(kind, 8),
        size: 10,
        sum: Sum::of(kind.as_bytes()),
    };
    Header {
        position: LogPosition(8),
        key: key("manifest", 8),
        head: Head::Manifest,
        sum: Sum::of(b"the manifest at eight"),
        size: 21,
        before: Some(Before {
            position: LogPosition(7),
            sum: Sum::of(&frame()),
        }),
        files: vec![needed("segment"), needed("index"), needed("vectors")],
        body: Body::Spilled,
        ..inline()
    }
}

#[test]
fn write_link_fixtures() {
    if std::env::var_os("VELLEV_WRITE_FIXTURES").is_none() {
        return;
    }
    let dir = fixtures(LINK_VERSION);
    assert!(
        !dir.exists(),
        "{} exists; a released version's fixtures are never written again",
        dir.display()
    );
    std::fs::create_dir_all(&dir).unwrap();
    let inline = link::encode(&inline(), Some(&frame())).unwrap();
    std::fs::write(dir.join("inline.link"), inline).unwrap();
    std::fs::write(
        dir.join("spilled.link"),
        link::encode(&spilled(), None).unwrap(),
    )
    .unwrap();
}

#[test]
fn links_of_version_1_are_read_as_they_were_written() {
    let dir = fixtures(1);
    let header = inline();
    let at = Timeline::of(&header);

    let bytes = std::fs::read(dir.join("inline.link")).unwrap();
    let read = link::decode(&at.link(LogPosition(7)), &bytes).unwrap();
    assert_eq!(read.header, header);
    assert_eq!(read.frame.as_deref(), Some(&frame()[..]));
    assert_eq!(read.header.committed_at.timestamp_subsec_micros(), 123_456);
    // This release writes it byte for byte as it was.
    assert_eq!(link::encode(&header, Some(&frame())).unwrap(), bytes);

    let bytes = std::fs::read(dir.join("spilled.link")).unwrap();
    let read = link::decode(&at.link(LogPosition(8)), &bytes).unwrap();
    assert_eq!(read.header, spilled());
    assert_eq!(read.frame, None);
    assert_eq!(link::encode(&spilled(), None).unwrap(), bytes);
}
