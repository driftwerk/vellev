//! The three kinds of file a partition is made of, and their framing.
//!
//! A frame is `magic, kind, version, body length, CRC, body`, where the CRC
//! covers everything between the magic and itself as well as the body. The
//! body is self-describing, so a newer writer can add fields an older reader
//! skips.

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use vellev_core::{Collection, Object, ObjectId, Property};

use crate::coordinator::LogPosition;
use crate::{Result, StorageError};

const MAGIC: [u8; 4] = *b"VLLV";
const HEADER: usize = 4 + 1 + 2 + 8 + 4;
pub const FORMAT_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Log = 1,
    Manifest = 2,
    Segment = 3,
}

impl Kind {
    fn from_byte(byte: u8) -> Option<Self> {
        [Kind::Log, Kind::Manifest, Kind::Segment]
            .into_iter()
            .find(|kind| *kind as u8 == byte)
    }
}

/// A collection's identity for the life of the partition. A dropped
/// collection's id is never given out again, so recreating a collection under
/// the same name cannot bring its old rows back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CollectionId(pub u64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    CreateCollection {
        id: CollectionId,
        schema: Box<Collection>,
    },
    AddProperty {
        collection: CollectionId,
        property: Property,
    },
    DropCollection {
        collection: CollectionId,
    },
    Upsert {
        collection: CollectionId,
        object: Object,
    },
    Delete {
        collection: CollectionId,
        id: ObjectId,
    },
}

/// One commit. `prev` is the head this commit was made on top of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogSegment {
    pub position: LogPosition,
    pub prev: Option<String>,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub collection: CollectionId,
    pub object: Object,
}

/// What a checkpoint flushed from memory: the rows written since the last
/// checkpoint, and the keys deleted since then that an older segment may hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentFile {
    pub position: LogPosition,
    pub rows: Vec<Row>,
    pub tombstones: Vec<(CollectionId, ObjectId)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub key: String,
    pub rows: u64,
    pub tombstones: u64,
}

/// A checkpoint: everything needed to rebuild the partition as of `position`
/// without reading any log before it. Segments are listed oldest first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub position: LogPosition,
    pub next_collection_id: u64,
    pub collections: Vec<(CollectionId, Collection)>,
    pub segments: Vec<SegmentMeta>,
}

/// A file a head reference can point at.
#[derive(Debug, Clone, PartialEq)]
pub enum Head {
    Log(LogSegment),
    Manifest(Manifest),
}

pub fn encode<T: Serialize>(kind: Kind, value: &T) -> Result<Bytes> {
    let body = rmp_serde::to_vec_named(value).map_err(|e| StorageError::Encode(e.to_string()))?;
    let mut frame = Vec::with_capacity(HEADER + body.len());
    frame.extend_from_slice(&MAGIC);
    frame.push(kind as u8);
    frame.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
    frame.extend_from_slice(&checksum(&frame[MAGIC.len()..], &body).to_le_bytes());
    frame.extend_from_slice(&body);
    Ok(Bytes::from(frame))
}

pub fn decode<T: DeserializeOwned>(key: &str, kind: Kind, frame: &[u8]) -> Result<T> {
    let (found, body) = open(key, frame)?;
    if found != kind {
        return Err(StorageError::corrupt(
            key,
            format!("expected {kind:?}, found {found:?}"),
        ));
    }
    parse(key, body)
}

pub fn decode_head(key: &str, frame: &[u8]) -> Result<Head> {
    match open(key, frame)? {
        (Kind::Log, body) => parse(key, body).map(Head::Log),
        (Kind::Manifest, body) => parse(key, body).map(Head::Manifest),
        (Kind::Segment, _) => Err(StorageError::corrupt(key, "a segment cannot be a head")),
    }
}

fn checksum(described: &[u8], body: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(described);
    hasher.update(body);
    hasher.finalize()
}

fn parse<T: DeserializeOwned>(key: &str, body: &[u8]) -> Result<T> {
    rmp_serde::from_slice(body).map_err(|e| StorageError::corrupt(key, e.to_string()))
}

fn open<'a>(key: &str, frame: &'a [u8]) -> Result<(Kind, &'a [u8])> {
    let corrupt = |reason: &str| StorageError::corrupt(key, reason);
    let (header, body) = frame
        .split_at_checked(HEADER)
        .ok_or_else(|| corrupt("shorter than a frame header"))?;
    if header[..4] != MAGIC {
        return Err(corrupt("not a vellev file"));
    }
    let kind = Kind::from_byte(header[4]).ok_or_else(|| corrupt("unknown file kind"))?;
    let version = u16::from_le_bytes([header[5], header[6]]);
    if version == 0 {
        return Err(corrupt("format version zero"));
    }
    if version > FORMAT_VERSION {
        return Err(StorageError::UnsupportedVersion {
            key: key.to_string(),
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let length = u64::from_le_bytes(header[7..15].try_into().expect("eight bytes"));
    let crc = u32::from_le_bytes(header[15..19].try_into().expect("four bytes"));
    if length != body.len() as u64 {
        return Err(corrupt("body length does not match the header"));
    }
    if crc != checksum(&header[MAGIC.len()..15], body) {
        return Err(corrupt("checksum mismatch"));
    }
    Ok((kind, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> LogSegment {
        LogSegment {
            position: LogPosition(7),
            prev: Some("t/0/log/6.log".into()),
            ops: vec![Op::Delete {
                collection: CollectionId(1),
                id: ObjectId(uuid::Uuid::from_u128(7)),
            }],
        }
    }

    #[test]
    fn a_file_survives_a_round_trip() {
        let frame = encode(Kind::Log, &log()).unwrap();
        assert_eq!(decode::<LogSegment>("k", Kind::Log, &frame).unwrap(), log());
        assert_eq!(decode_head("k", &frame).unwrap(), Head::Log(log()));
    }

    #[test]
    fn any_damaged_byte_is_detected() {
        let frame = encode(Kind::Log, &log()).unwrap();
        for index in 0..frame.len() {
            let mut damaged = frame.to_vec();
            damaged[index] ^= 0x01;
            assert!(
                decode::<LogSegment>("k", Kind::Log, &damaged).is_err(),
                "byte {index}"
            );
        }
        for length in [0, HEADER - 1, HEADER, frame.len() - 1] {
            assert!(
                decode::<LogSegment>("k", Kind::Log, &frame[..length]).is_err(),
                "{length}"
            );
        }
    }

    #[test]
    fn a_newer_format_version_is_refused_by_name() {
        let mut frame = encode(Kind::Log, &log()).unwrap().to_vec();
        frame[5..7].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let crc = checksum(&frame[MAGIC.len()..15], &frame[HEADER..]);
        frame[15..19].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            decode::<LogSegment>("k", Kind::Log, &frame),
            Err(StorageError::UnsupportedVersion { found, supported, .. })
                if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
        ));
    }

    #[test]
    fn a_file_of_the_wrong_kind_is_refused() {
        let segment = SegmentFile {
            position: LogPosition(1),
            rows: vec![],
            tombstones: vec![],
        };
        let frame = encode(Kind::Segment, &segment).unwrap();
        assert!(matches!(
            decode::<LogSegment>("k", Kind::Log, &frame),
            Err(StorageError::Corrupt { .. })
        ));
        assert!(matches!(
            decode_head("k", &frame),
            Err(StorageError::Corrupt { .. })
        ));
    }
}
