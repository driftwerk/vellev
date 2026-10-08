//! The three kinds of file a partition is made of, and their framing.
//!
//! A frame is `magic, kind, version, body length, CRC, body`, where the CRC
//! covers everything between the magic and itself as well as the body. The
//! header layout never changes between versions, so damage is told apart from
//! a newer format before the version is interpreted.
//!
//! The body is self-describing: a reader skips struct fields it does not
//! know. A field may be added without a new version only if an older node
//! may both ignore it and drop it when it rewrites the file; anything else,
//! and any new enum variant, needs a new `FORMAT_VERSION`.

use std::sync::Arc;

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use vellev_core::{Collection, Object, ObjectId, Property, RequestId};
use vellev_index::clustered::ClusteredIndex;

use crate::coordinator::LogPosition;
use crate::mutation::Outcome;
use crate::{Refusal, Result, StorageError};

const MAGIC: [u8; 4] = *b"VLLV";
const HEADER: usize = 4 + 1 + 2 + 8 + 4;
/// The newest format this release reads and can write. Every writer stamps
/// the files it writes with it today; a release that adds a format must
/// write the one the cluster writes instead (see docs/compatibility.md).
pub const FORMAT_VERSION: u16 = 1;

/// The newest format a node writes unless an operator allows a newer one:
/// the format before the newest when a release adds one, so that the
/// release before can read everything written while both run.
pub const DEFAULT_MAX_FORMAT: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Log = 1,
    Manifest = 2,
    Segment = 3,
    Index = 4,
    Vectors = 5,
}

impl Kind {
    fn from_byte(byte: u8) -> Option<Self> {
        [
            Kind::Log,
            Kind::Manifest,
            Kind::Segment,
            Kind::Index,
            Kind::Vectors,
        ]
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
    /// The requests, named by their callers, whose changes this commit
    /// made.
    #[serde(default)]
    pub answered: Vec<Answered>,
}

/// A request a commit answered: how, and when the commit was made, in
/// milliseconds since the Unix epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Answered {
    pub request: RequestId,
    pub at: u64,
    pub answer: Answer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Answer {
    /// The request's changes were made, or changed nothing, with these
    /// outcomes.
    Made(Vec<Outcome>),
    /// The request was refused, and none of it made.
    Refused(Refusal),
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
    /// The file holding the segment's clustered vector indexes, if it is
    /// large enough to have any.
    #[serde(default)]
    pub index: Option<String>,
    /// The file holding the rows' vectors, if any row has one; the rows in
    /// the segment file then carry none.
    #[serde(default)]
    pub vectors: Option<String>,
}

/// The vector indexes of a segment that took long enough to build to be
/// kept: one per collection with enough vectors in the segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentIndex {
    /// The key of the segment the indexes are of.
    pub segment: String,
    pub vectors: Vec<(CollectionId, Arc<ClusteredIndex>)>,
}

/// A checkpoint: everything needed to rebuild the partition as of `position`
/// without reading any log before it. Segments are listed oldest first.
///
/// A flush builds its segment from the partition as it was at an earlier
/// position while commits go on; its manifest then lists, in `replay`, the
/// logs committed since, which are applied after the segments, and the
/// collections are those of that earlier position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub position: LogPosition,
    /// The checkpoint before this one, if any.
    #[serde(default)]
    pub prev: Option<String>,
    pub next_collection_id: u64,
    pub collections: Vec<(CollectionId, Collection)>,
    pub segments: Vec<SegmentMeta>,
    /// The logs to apply after the segments, oldest first.
    #[serde(default)]
    pub replay: Vec<String>,
    /// The position the segments hold, just before the first log to
    /// replay; unused when there is none.
    #[serde(default)]
    pub replay_after: LogPosition,
    /// The requests answered lately, oldest first, each with the position
    /// it became visible at.
    #[serde(default)]
    pub answered: Vec<(LogPosition, Answered)>,
}

/// A file a head reference can point at.
#[derive(Debug, Clone, PartialEq)]
pub enum Head {
    Log(LogSegment),
    Manifest(Manifest),
}

pub fn encode<T: Serialize>(kind: Kind, value: &T) -> Result<Bytes> {
    rmp_serde::to_vec_named(value)
        .map(|body| Bytes::from(frame(kind, FORMAT_VERSION, &body)))
        .map_err(|e| StorageError::Encode(e.to_string()))
}

fn frame(kind: Kind, version: u16, body: &[u8]) -> Vec<u8> {
    framed(kind, version, body.len(), |frame| {
        frame.extend_from_slice(body)
    })
}

/// A frame whose body `fill` writes in place, for one too large to build
/// first and copy. `fill` must append exactly `length` bytes.
pub(crate) fn framed(
    kind: Kind,
    version: u16,
    length: usize,
    fill: impl FnOnce(&mut Vec<u8>),
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(HEADER + length);
    frame.extend_from_slice(&MAGIC);
    frame.push(kind as u8);
    frame.extend_from_slice(&version.to_le_bytes());
    frame.extend_from_slice(&(length as u64).to_le_bytes());
    frame.extend_from_slice(&[0; 4]);
    fill(&mut frame);
    debug_assert_eq!(frame.len(), HEADER + length);
    let (header, body) = frame.split_at(HEADER);
    let checksum = checksum(&header[MAGIC.len()..HEADER - 4], body).to_le_bytes();
    frame[HEADER - 4..HEADER].copy_from_slice(&checksum);
    frame
}

/// Where a frame's body starts.
pub(crate) const BODY: usize = HEADER;

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
        (Kind::Segment | Kind::Index | Kind::Vectors, _) => Err(StorageError::corrupt(
            key,
            "only a log or a manifest can be a head",
        )),
    }
}

fn checksum(described: &[u8], body: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(described);
    hasher.update(body);
    hasher.finalize()
}

/// Check a frame's integrity without decoding its body.
pub fn verify(key: &str, frame: &[u8]) -> Result<()> {
    open(key, frame).map(|_| ())
}

/// Decoded through a cursor so that bytes left over after the value, which
/// `from_slice` would accept, are noticed.
fn parse<T: DeserializeOwned>(key: &str, body: &[u8]) -> Result<T> {
    let mut reader = std::io::Cursor::new(body);
    let value =
        rmp_serde::from_read(&mut reader).map_err(|e| StorageError::corrupt(key, e.to_string()))?;
    if reader.position() == body.len() as u64 {
        Ok(value)
    } else {
        Err(StorageError::corrupt(key, "bytes follow the encoded value"))
    }
}

pub(crate) fn open<'a>(key: &str, frame: &'a [u8]) -> Result<(Kind, &'a [u8])> {
    let corrupt = |reason: &str| StorageError::corrupt(key, reason);
    let (header, body) = frame
        .split_at_checked(HEADER)
        .ok_or_else(|| corrupt("shorter than a frame header"))?;
    if header[..4] != MAGIC {
        return Err(corrupt("not a vellev file"));
    }
    let length = u64::from_le_bytes(header[7..15].try_into().expect("eight bytes"));
    let crc = u32::from_le_bytes(header[15..19].try_into().expect("four bytes"));
    if length != body.len() as u64 {
        return Err(corrupt("body length does not match the header"));
    }
    if crc != checksum(&header[MAGIC.len()..15], body) {
        return Err(corrupt("checksum mismatch"));
    }
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
    let kind = Kind::from_byte(header[4]).ok_or_else(|| corrupt("unknown file kind"))?;
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
            answered: vec![
                Answered {
                    request: RequestId(uuid::Uuid::from_u128(9)),
                    at: 1_791_300_000_000,
                    answer: Answer::Made(vec![
                        Outcome::Applied,
                        Outcome::Unchanged,
                        Outcome::Value(vellev_core::Value::Int(3)),
                        Outcome::Deleted(2),
                    ]),
                },
                Answered {
                    request: RequestId(uuid::Uuid::from_u128(10)),
                    at: 1_791_300_000_000,
                    answer: Answer::Refused(Refusal {
                        kind: crate::Refused::Conflict,
                        message: "object 7 already exists in notes".into(),
                        lacking: None,
                    }),
                },
            ],
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
        let body = rmp_serde::to_vec_named(&log()).unwrap();
        for (version, supported) in [(FORMAT_VERSION, true), (FORMAT_VERSION + 1, false)] {
            let outcome = decode::<LogSegment>("k", Kind::Log, &frame(Kind::Log, version, &body));
            assert_eq!(outcome.is_ok(), supported);
            assert_eq!(
                matches!(outcome, Err(StorageError::UnsupportedVersion { found, .. }) if found == version),
                !supported
            );
        }
        assert!(matches!(
            decode::<LogSegment>("k", Kind::Log, &frame(Kind::Log, 0, &body)),
            Err(StorageError::Corrupt { .. })
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

    #[test]
    fn bytes_after_the_encoded_value_are_refused() {
        let mut body = rmp_serde::to_vec_named(&log()).unwrap();
        assert_eq!(
            decode::<LogSegment>("k", Kind::Log, &frame(Kind::Log, FORMAT_VERSION, &body)).unwrap(),
            log()
        );
        body.push(0xc0);
        assert!(matches!(
            decode::<LogSegment>("k", Kind::Log, &frame(Kind::Log, FORMAT_VERSION, &body)),
            Err(StorageError::Corrupt { .. })
        ));
    }

    #[test]
    fn a_field_added_by_a_newer_writer_is_skipped() {
        #[derive(Serialize)]
        struct NewerLogSegment {
            position: LogPosition,
            prev: Option<String>,
            ops: Vec<Op>,
            answered: Vec<Answered>,
            written_by: String,
        }
        let LogSegment {
            position,
            prev,
            ops,
            answered,
        } = log();
        let newer = NewerLogSegment {
            position,
            prev,
            ops,
            answered,
            written_by: "vellev 9".into(),
        };
        let body = rmp_serde::to_vec_named(&newer).unwrap();
        assert_eq!(
            decode::<LogSegment>("k", Kind::Log, &frame(Kind::Log, FORMAT_VERSION, &body)).unwrap(),
            log()
        );
    }

    #[test]
    fn what_a_refusal_found_missing_is_kept_and_a_release_from_before_it_reads_past_it() {
        /// A log as the release before `lacking` was kept reads and
        /// writes it.
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct OlderLog {
            position: LogPosition,
            prev: Option<String>,
            ops: Vec<Op>,
            answered: Vec<OlderAnswered>,
        }
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct OlderAnswered {
            request: RequestId,
            at: u64,
            answer: OlderAnswer,
        }
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum OlderAnswer {
            Made(Vec<Outcome>),
            Refused(OlderRefusal),
        }
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct OlderRefusal {
            kind: crate::Refused,
            message: String,
        }
        let message = "collection notes does not exist";
        let request = RequestId(uuid::Uuid::from_u128(11));
        let refused = |lacking| LogSegment {
            answered: vec![Answered {
                request,
                at: 1_791_300_000_000,
                answer: Answer::Refused(Refusal {
                    kind: crate::Refused::NotFound,
                    message: message.into(),
                    lacking,
                }),
            }],
            ..log()
        };
        let older = || {
            let LogSegment {
                position,
                prev,
                ops,
                ..
            } = log();
            OlderLog {
                position,
                prev,
                ops,
                answered: vec![OlderAnswered {
                    request,
                    at: 1_791_300_000_000,
                    answer: OlderAnswer::Refused(OlderRefusal {
                        kind: crate::Refused::NotFound,
                        message: message.into(),
                    }),
                }],
            }
        };
        let lacking = refused(Some(crate::Lacking::Collection));
        let kept = encode(Kind::Log, &lacking).unwrap();
        assert_eq!(
            decode::<LogSegment>("k", Kind::Log, &kept).unwrap(),
            lacking
        );
        assert_eq!(
            decode::<OlderLog>("k", Kind::Log, &kept).unwrap(),
            older(),
            "the release before reads past what it does not know"
        );
        let before = encode(Kind::Log, &older()).unwrap();
        assert_eq!(
            decode::<LogSegment>("k", Kind::Log, &before).unwrap(),
            refused(None),
            "and what it wrote reads as a refusal that kept nothing"
        );
        assert_eq!(
            encode(Kind::Log, &refused(None)).unwrap(),
            before,
            "a refusal that found nothing missing is written as it always was"
        );
        assert_ne!(kept, before);

        // A release after this one may name another kind of thing: read
        // here, after a rollback, its refusal is one that kept nothing.
        #[derive(Debug, Serialize)]
        enum LaterLacking {
            Property,
        }
        #[derive(Debug, Serialize)]
        struct LaterRefusal {
            kind: crate::Refused,
            message: String,
            lacking: Option<LaterLacking>,
        }
        #[derive(Debug, Serialize)]
        enum LaterAnswer {
            Refused(LaterRefusal),
        }
        #[derive(Debug, Serialize)]
        struct LaterAnswered {
            request: RequestId,
            at: u64,
            answer: LaterAnswer,
        }
        #[derive(Debug, Serialize)]
        struct LaterLog {
            position: LogPosition,
            prev: Option<String>,
            ops: Vec<Op>,
            answered: Vec<LaterAnswered>,
        }
        let LogSegment {
            position,
            prev,
            ops,
            ..
        } = log();
        let later = LaterLog {
            position,
            prev,
            ops,
            answered: vec![LaterAnswered {
                request,
                at: 1_791_300_000_000,
                answer: LaterAnswer::Refused(LaterRefusal {
                    kind: crate::Refused::NotFound,
                    message: message.into(),
                    lacking: Some(LaterLacking::Property),
                }),
            }],
        };
        let from_later = encode(Kind::Log, &later).unwrap();
        assert_eq!(
            decode::<LogSegment>("k", Kind::Log, &from_later).unwrap(),
            refused(None)
        );
    }

    #[test]
    fn every_embedding_provider_survives_the_file_format() {
        use vellev_core::{
            Bm25, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, HttpApi,
            Property,
        };
        let providers = [
            EmbeddingProvider::Local,
            EmbeddingProvider::Bedrock {
                region: "eu-west-2".into(),
            },
            EmbeddingProvider::Http {
                url: "http://embeddings:80".into(),
                api: HttpApi::Tei,
            },
            EmbeddingProvider::Http {
                url: "https://api.example.test".into(),
                api: HttpApi::Openai,
            },
            EmbeddingProvider::None,
        ];
        for provider in providers {
            let schema = Collection {
                name: CollectionName::new("Notes").unwrap(),
                properties: vec![Property::new("body", DataType::Text).searchable()],
                embedding: Embedding {
                    provider,
                    model: "m".into(),
                    dimensions: 4,
                    query_prefix: Some("q: ".into()),
                    document_prefix: None,
                    properties: vec!["body".into()],
                },
                distance: Distance::Dot,
                bm25: Bm25::default(),
                access_rule: None,
            };
            let manifest = Manifest {
                position: LogPosition(3),
                prev: None,
                next_collection_id: 2,
                collections: vec![(CollectionId(1), schema)],
                segments: vec![],
                replay: vec!["a/0/log/x.log".into()],
                replay_after: LogPosition(2),
                answered: Vec::new(),
            };
            let frame = encode(Kind::Manifest, &manifest).unwrap();
            assert_eq!(
                decode::<Manifest>("k", Kind::Manifest, &frame).unwrap(),
                manifest
            );
        }
    }
}
