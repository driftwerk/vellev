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

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use vellev_core::{Collection, Object, ObjectId, Property, RequestId};
use vellev_index::clustered::ClusteredIndex;

use crate::blobs::is_plain;
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

impl Manifest {
    /// This checkpoint, read from under `key`, naming every file by the
    /// key `fresh` gives for the one it names it by now: each segment, its
    /// index and its vectors, the logs to replay and the checkpoint before.
    /// Nothing else of it changes.
    ///
    /// No file passes by accident. One `fresh` gives no key for fails it,
    /// and so does a key `fresh` gives that would make two files one or
    /// name nothing: one given for two files, one another file of the
    /// checkpoint is named by now, an empty one, or one that is not a
    /// plain path. A file given the key it has keeps it, which is the
    /// caller's to want or not: the remap does not know which keys are
    /// new.
    ///
    /// A restore from a backup makes with it the checkpoint it loads what
    /// it copied through, kept in memory and never committed, having first
    /// cleared `prev` and taken `replay` out, to put it back after: the
    /// logs are read under the keys they had, and no key is given for
    /// them, so that either left in fails here.
    pub fn remapped(self, key: &str, fresh: &HashMap<String, String>) -> Result<Self> {
        // Taken apart in full, so that a field added to either does not
        // compile here until it is said whether it names a file.
        let Self {
            position,
            prev,
            next_collection_id,
            collections,
            segments,
            replay,
            replay_after,
            answered,
        } = self;
        let refused = |reason: String| StorageError::corrupt(key, reason);
        // The key each file had, by the one it was given.
        let mut given: HashMap<String, String> = HashMap::new();
        let mut renamed = |old: String| {
            let new = fresh
                .get(&old)
                .ok_or_else(|| refused(format!("it names {old}, which was given no new key")))?;
            // An empty key is no plain one either.
            if !is_plain(new) {
                let reason = format!("it names {old}, which was given {new:?}, no plain key");
                return Err(refused(reason));
            }
            match given.get(new) {
                Some(other) if *other != old => Err(refused(format!(
                    "it names {other} and {old}, which were both given {new}"
                ))),
                Some(_) => Ok(new.clone()),
                None => {
                    given.insert(new.clone(), old);
                    Ok(new.clone())
                }
            }
        };
        let mut renamed_segments = Vec::with_capacity(segments.len());
        for segment in segments {
            let SegmentMeta {
                key,
                rows,
                tombstones,
                index,
                vectors,
            } = segment;
            renamed_segments.push(SegmentMeta {
                key: renamed(key)?,
                rows,
                tombstones,
                index: index.map(&mut renamed).transpose()?,
                vectors: vectors.map(&mut renamed).transpose()?,
            });
        }
        let remapped = Self {
            position,
            prev: prev.map(&mut renamed).transpose()?,
            next_collection_id,
            collections,
            segments: renamed_segments,
            replay: replay
                .into_iter()
                .map(&mut renamed)
                .collect::<Result<_>>()?,
            replay_after,
            answered,
        };
        // Every key it named went through `renamed`, so `given` holds each
        // of them: a new key that is another file's old one would leave
        // that file's name meaning two files, before and after.
        let had: std::collections::HashSet<&String> = given.values().collect();
        match given
            .iter()
            .find(|(new, old)| new != old && had.contains(new))
        {
            Some((new, old)) => Err(refused(format!(
                "it names {old}, which was given {new}, the key of another file it names"
            ))),
            None => Ok(remapped),
        }
    }
}

/// The index file `frame`, read from under `key`, of the segment that was
/// under `from`, as the index file of that segment under `to`: the same
/// indexes, saying the key the segment now has. One that says another
/// segment than `from` is refused. Loaded, it is passed over as another
/// segment's; given the new key, it would be taken for this segment's and
/// searched by.
pub fn rekeyed_index(key: &str, frame: &[u8], from: &str, to: &str) -> Result<Bytes> {
    let SegmentIndex { segment, vectors } = decode(key, Kind::Index, frame)?;
    if segment != from {
        return Err(StorageError::corrupt(key, "it indexes another segment"));
    }
    let rekeyed = SegmentIndex {
        segment: to.to_string(),
        vectors,
    };
    encode(Kind::Index, &rekeyed)
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

/// How much of a body is summed between one word of progress and the next.
pub(crate) const SUMMED_AT_ONCE: usize = 4 * 1024 * 1024;

/// What a frame's checksum must come to, and of what besides its body.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sum<'a> {
    described: &'a [u8],
    expected: u32,
}

impl Sum<'_> {
    /// Whether `body` sums to what the frame says, telling `summed` of
    /// each piece of it as it is done, so that whoever waits on a body of
    /// gigabytes on a slow disk can tell slow from stopped.
    pub fn holds(&self, body: &[u8], mut summed: impl FnMut(usize)) -> bool {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(self.described);
        for piece in body.chunks(SUMMED_AT_ONCE) {
            hasher.update(piece);
            summed(piece.len());
        }
        hasher.finalize() == self.expected
    }
}

/// A frame's header and body, seen to be a vellev file of the length it
/// says, with what its checksum must come to: not yet summed.
fn parts<'a>(key: &str, frame: &'a [u8]) -> Result<(&'a [u8], &'a [u8], Sum<'a>)> {
    let corrupt = |reason: &str| StorageError::corrupt(key, reason);
    let (header, body) = frame
        .split_at_checked(HEADER)
        .ok_or_else(|| corrupt("shorter than a frame header"))?;
    if header[..4] != MAGIC {
        return Err(corrupt("not a vellev file"));
    }
    let length = u64::from_le_bytes(header[7..15].try_into().expect("eight bytes"));
    if length != body.len() as u64 {
        return Err(corrupt("body length does not match the header"));
    }
    let sum = Sum {
        described: &header[MAGIC.len()..15],
        expected: u32::from_le_bytes(header[15..19].try_into().expect("four bytes")),
    };
    Ok((header, body, sum))
}

/// The kind of file a header says it is, if it is of a format this
/// release reads.
fn kind(key: &str, header: &[u8]) -> Result<Kind> {
    let corrupt = |reason: &str| StorageError::corrupt(key, reason);
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
    Kind::from_byte(header[4]).ok_or_else(|| corrupt("unknown file kind"))
}

/// As [`open`], leaving the checksum to whoever asked: for a file read in
/// place, too large to sum before anyone may be told how it goes. Until
/// the sum is seen to hold, the body may be anything.
pub(crate) fn open_unsummed<'a>(key: &str, frame: &'a [u8]) -> Result<(Kind, &'a [u8], Sum<'a>)> {
    let (header, body, sum) = parts(key, frame)?;
    Ok((kind(key, header)?, body, sum))
}

/// Check a frame's integrity without decoding its body.
pub fn verify(key: &str, frame: &[u8]) -> Result<()> {
    open(key, frame).map(|_| ())
}

/// Decoded through a cursor so that bytes left over after the value, which
/// `from_slice` would accept, are noticed.
pub(crate) fn parse<T: DeserializeOwned>(key: &str, body: &[u8]) -> Result<T> {
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
    let (header, body, sum) = parts(key, frame)?;
    if !sum.holds(body, |_| {}) {
        return Err(StorageError::corrupt(key, "checksum mismatch"));
    }
    Ok((kind(key, header)?, body))
}

/// What a file came to as it went by a part at a time: how long it was,
/// and the CRC-32 of all of it. Two files that come to the same are taken
/// for the same file: what is guarded against is a file damaged in moving
/// or lying in a store, not one made to look like another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Came {
    length: u64,
    crc: u32,
}

/// [`Came`], taken as a file goes by.
#[derive(Debug, Default)]
pub(crate) struct Coming {
    length: u64,
    crc: crc32fast::Hasher,
}

impl Coming {
    pub fn update(&mut self, part: &[u8]) {
        self.length += part.len() as u64;
        self.crc.update(part);
    }

    pub fn came(self) -> Came {
        Came {
            length: self.length,
            crc: self.crc.finalize(),
        }
    }
}

/// [`Came`] of a file taken as it goes by, and whether it is one whole
/// frame of a kind: the frame's checksum is summed as the file goes, and
/// nothing of it is held but its header.
#[derive(Debug, Default)]
pub(crate) struct Framed {
    whole: Coming,
    header: Vec<u8>,
    body: crc32fast::Hasher,
}

impl Framed {
    pub fn update(&mut self, mut part: &[u8]) {
        self.whole.update(part);
        if self.header.len() < HEADER {
            let taken = (HEADER - self.header.len()).min(part.len());
            self.header.extend_from_slice(&part[..taken]);
            part = &part[taken..];
            if self.header.len() == HEADER {
                self.body.update(&self.header[MAGIC.len()..15]);
            }
        }
        self.body.update(part);
    }

    /// What the file came to, if it is one whole frame of `expected`, of
    /// a format this release reads; a failure saying how it is not
    /// otherwise.
    pub fn came(self, key: &str, expected: Kind) -> Result<Came> {
        let corrupt = |reason: &str| StorageError::corrupt(key, reason);
        let Self {
            whole,
            header,
            body,
        } = self;
        if header.len() < HEADER {
            return Err(corrupt("shorter than a frame header"));
        }
        if header[..4] != MAGIC {
            return Err(corrupt("not a vellev file"));
        }
        if kind(key, &header)? != expected {
            return Err(corrupt("not the kind of file it is named for"));
        }
        let came = whole.came();
        let length = u64::from_le_bytes(header[7..15].try_into().expect("eight bytes"));
        if Some(length) != came.length.checked_sub(HEADER as u64) {
            return Err(corrupt("body length does not match the header"));
        }
        let crc = u32::from_le_bytes(header[15..19].try_into().expect("four bytes"));
        if body.finalize() != crc {
            return Err(corrupt("checksum mismatch"));
        }
        Ok(came)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `file` taken as it goes by in parts of `each` bytes.
    fn framed_by(file: &[u8], each: usize) -> Framed {
        let mut framed = Framed::default();
        file.chunks(each).for_each(|part| framed.update(part));
        framed
    }

    /// A file taken as it goes by, in parts of any size, comes to what it
    /// comes to whole, and is seen to be the frame it is.
    #[test]
    fn a_frame_going_by_in_parts_is_seen_whole_and_comes_to_what_it_is() {
        let frame = encode(Kind::Index, &(7u64, "seven")).unwrap();
        let mut whole = Coming::default();
        whole.update(&frame);
        let whole = whole.came();
        for each in [1, 3, HEADER - 1, HEADER, HEADER + 1, frame.len()] {
            let framed = framed_by(&frame, each);
            assert_eq!(framed.came("k", Kind::Index).unwrap(), whole, "{each}");
        }
    }

    /// A file that is not one whole frame of the kind it is named for is
    /// refused, whatever it comes to; and two files that differ come to
    /// different things.
    #[test]
    fn a_file_going_by_that_is_not_a_whole_frame_of_its_kind_is_refused() {
        let frame = encode(Kind::Index, &(7u64, "seven")).unwrap();
        let mut flipped = frame.to_vec();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        let mut long = frame.to_vec();
        long.push(0);
        let mut unnamed = frame.to_vec();
        unnamed[0] = b'X';
        let refused: [(&str, &[u8], Kind); 6] = [
            ("damaged", &flipped, Kind::Index),
            ("cut short", &frame[..frame.len() - 1], Kind::Index),
            ("too long", &long, Kind::Index),
            ("no header", &frame[..HEADER - 1], Kind::Index),
            ("no vellev file", &unnamed, Kind::Index),
            ("another kind", &frame, Kind::Segment),
        ];
        for (what, file, kind) in refused {
            let came = framed_by(file, 5).came("k", kind);
            assert!(
                matches!(came, Err(StorageError::Corrupt { .. })),
                "{what}: {came:?}"
            );
        }
        let other = encode(Kind::Index, &(8u64, "seven")).unwrap();
        let (one, two) = (
            framed_by(&frame, 4).came("k", Kind::Index),
            framed_by(&other, 4).came("k", Kind::Index),
        );
        assert_ne!(one.unwrap(), two.unwrap());
    }

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
                        over: None,
                        version: None,
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
                    over: None,
                    version: None,
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
    fn what_a_refusal_found_asked_too_much_of_is_kept_and_other_releases_read_past_it() {
        use crate::{Capped, Over, Refused};
        /// A refusal as the release before `over` was kept reads and
        /// writes it.
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct OlderRefusal {
            kind: Refused,
            message: String,
        }
        let refused = |over| Refusal {
            kind: Refused::Invalid,
            message: "a delete takes a limit of 1 to 8192".into(),
            lacking: None,
            over,
            version: None,
        };
        let older = || OlderRefusal {
            kind: Refused::Invalid,
            message: "a delete takes a limit of 1 to 8192".into(),
        };
        fn pack<T: Serialize>(kept: &T) -> Vec<u8> {
            rmp_serde::to_vec_named(kept).unwrap()
        }
        let over = refused(Some(Over {
            capped: Capped::Deleted,
            most: 8_192,
        }));
        let kept = pack(&over);
        assert_eq!(rmp_serde::from_slice::<Refusal>(&kept).unwrap(), over);
        assert_eq!(
            rmp_serde::from_slice::<OlderRefusal>(&kept).unwrap(),
            older(),
            "the release before reads past what it does not know"
        );
        let before = pack(&older());
        assert_eq!(
            rmp_serde::from_slice::<Refusal>(&before).unwrap(),
            refused(None),
            "and what it wrote reads as a refusal that kept nothing"
        );
        assert_eq!(pack(&refused(None)), before, "written as it always was");

        // A release after this one may cap another thing: read here,
        // after a rollback, its refusal is one that kept nothing.
        #[derive(Debug, Serialize)]
        enum LaterCapped {
            Written,
        }
        #[derive(Debug, Serialize)]
        struct LaterOver {
            capped: LaterCapped,
            most: u64,
        }
        #[derive(Debug, Serialize)]
        struct LaterRefusal {
            kind: Refused,
            message: String,
            over: Option<LaterOver>,
        }
        let later = LaterRefusal {
            kind: Refused::Invalid,
            message: "a delete takes a limit of 1 to 8192".into(),
            over: Some(LaterOver {
                capped: LaterCapped::Written,
                most: 5,
            }),
        };
        assert_eq!(
            rmp_serde::from_slice::<Refusal>(&pack(&later)).unwrap(),
            refused(None)
        );
    }

    #[test]
    fn the_version_a_refusal_was_at_is_kept_and_other_releases_read_past_it() {
        use crate::Refused;
        use vellev_core::Version;
        /// A refusal as the release before `version` was kept reads and
        /// writes it.
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct OlderRefusal {
            kind: Refused,
            message: String,
        }
        let at: Version = "0123456789abcdef0123456789abcdef".parse().unwrap();
        let refused = |version| Refusal {
            kind: Refused::Conflict,
            message: "object 7 in notes is not at the version given".into(),
            lacking: None,
            over: None,
            version,
        };
        let older = || OlderRefusal {
            kind: Refused::Conflict,
            message: "object 7 in notes is not at the version given".into(),
        };
        fn pack<T: Serialize>(kept: &T) -> Vec<u8> {
            rmp_serde::to_vec_named(kept).unwrap()
        }
        let kept = pack(&refused(Some(at)));
        assert_eq!(
            rmp_serde::from_slice::<Refusal>(&kept).unwrap(),
            refused(Some(at))
        );
        assert_eq!(
            rmp_serde::from_slice::<OlderRefusal>(&kept).unwrap(),
            older(),
            "the release before reads past what it does not know"
        );
        let before = pack(&older());
        assert_eq!(
            rmp_serde::from_slice::<Refusal>(&before).unwrap(),
            refused(None),
            "and what it wrote reads as a refusal that kept no version"
        );
        assert_eq!(pack(&refused(None)), before, "written as it always was");

        // A release after this one may keep a version written otherwise:
        // read here, after a rollback, it is a refusal that kept none.
        #[derive(Debug, Serialize)]
        struct LaterRefusal {
            kind: Refused,
            message: String,
            version: Vec<u8>,
        }
        let later = LaterRefusal {
            kind: Refused::Conflict,
            message: "object 7 in notes is not at the version given".into(),
            version: vec![1, 2, 3],
        };
        assert_eq!(
            rmp_serde::from_slice::<Refusal>(&pack(&later)).unwrap(),
            refused(None)
        );
    }

    /// A key as a file had it where a checkpoint was read from.
    fn old(kind: &str, name: &str) -> String {
        format!("tenant-a/0/{kind}/{name}.{kind}")
    }

    /// The key a restore would give the file that was under `old`: one of
    /// its own, holding nothing of the old one.
    fn new(old: &str) -> String {
        let name = old.rsplit('/').next().unwrap();
        format!("tenant-b/7/restored/{}", name.replace('0', "9"))
    }

    fn mapping(keys: &[String]) -> HashMap<String, String> {
        keys.iter().map(|key| (key.clone(), new(key))).collect()
    }

    fn checkpoint(
        prev: Option<String>,
        segments: Vec<SegmentMeta>,
        replay: Vec<String>,
    ) -> Manifest {
        Manifest {
            position: LogPosition(16),
            prev,
            next_collection_id: 3,
            collections: Vec::new(),
            replay_after: LogPosition(if replay.is_empty() { 0 } else { 14 }),
            segments,
            replay,
            answered: vec![(LogPosition(15), log().answered[0].clone())],
        }
    }

    fn segment(key: String, index: Option<String>, vectors: Option<String>) -> SegmentMeta {
        SegmentMeta {
            key,
            rows: 4,
            tombstones: 1,
            index,
            vectors,
        }
    }

    /// Checkpoints of each kind there is, each with every key it names,
    /// written out here and not gathered from its fields.
    fn checkpoints() -> Vec<(&'static str, Manifest, Vec<String>)> {
        let all = |name: &str| {
            let (key, index, vectors) = (
                old("segment", name),
                old("index", name),
                old("vectors", name),
            );
            let names = vec![key.clone(), index.clone(), vectors.clone()];
            (segment(key, Some(index), Some(vectors)), names)
        };
        let before = old("manifest", "00000000000000000012-w");
        let logs = vec![
            old("log", "00000000000000000015-w"),
            old("log", "00000000000000000016-w"),
        ];
        let (large, large_names) = all("00000000000000000011-w-1f");
        let (merged, merged_names) = all("00000000000000000016-w-2e-merged");
        let (kept, kept_names) = all("00000000000000000009-w-3d");
        let unindexed = old("segment", "00000000000000000014-w-4c");
        let unindexed_vectors = old("vectors", "00000000000000000014-w-4c");
        let bare = old("segment", "00000000000000000016-w-5b");
        vec![
            (
                "the first flush, of rows without vectors",
                checkpoint(None, vec![segment(bare.clone(), None, None)], Vec::new()),
                vec![bare.clone()],
            ),
            (
                "a flush that replays the logs committed while it was built",
                checkpoint(
                    Some(before.clone()),
                    vec![
                        large,
                        segment(unindexed.clone(), None, Some(unindexed_vectors.clone())),
                    ],
                    logs.clone(),
                ),
                [
                    vec![before.clone()],
                    large_names,
                    vec![unindexed, unindexed_vectors],
                    logs,
                ]
                .concat(),
            ),
            (
                "a merge",
                checkpoint(Some(before.clone()), vec![merged], Vec::new()),
                [vec![before.clone()], merged_names].concat(),
            ),
            (
                "a restore: the segments of the point gone back to, and what it held in memory",
                checkpoint(
                    Some(before.clone()),
                    vec![kept, segment(bare.clone(), None, None)],
                    Vec::new(),
                ),
                [vec![before], kept_names, vec![bare]].concat(),
            ),
        ]
    }

    fn holds(frame: &[u8], key: &str) -> bool {
        frame
            .windows(key.len())
            .any(|window| window == key.as_bytes())
    }

    #[test]
    fn a_remapped_checkpoint_names_no_file_by_the_key_it_had() {
        for (case, manifest, names) in checkpoints() {
            let fresh = mapping(&names);
            for (old, new) in &fresh {
                assert!(names.iter().all(|name| !new.contains(name)), "{old} {new}");
            }
            let was = encode(Kind::Manifest, &manifest).unwrap();
            for name in &names {
                assert!(holds(&was, name), "{case}: {name} is not one it names");
            }
            let remapped = manifest.clone().remapped("k", &fresh).unwrap();
            let frame = encode(Kind::Manifest, &remapped).unwrap();
            for name in &names {
                assert!(!holds(&frame, name), "{case}: {name} is still named");
                assert!(holds(&frame, &fresh[name]), "{case}: {name} was dropped");
            }
            assert_eq!(
                decode::<Manifest>("k", Kind::Manifest, &frame).unwrap(),
                remapped,
                "{case}"
            );
            // Mapped back, it is the checkpoint it was: every key went to
            // the one given for it, and nothing else changed.
            let back: HashMap<String, String> = fresh
                .iter()
                .map(|(old, new)| (new.clone(), old.clone()))
                .collect();
            assert_eq!(remapped.remapped("k", &back).unwrap(), manifest, "{case}");
        }
    }

    #[test]
    fn a_file_given_no_new_key_fails_the_remap_whichever_field_names_it() {
        for (case, manifest, names) in checkpoints() {
            for lacking in &names {
                let mut fresh = mapping(&names);
                fresh.remove(lacking);
                match manifest.clone().remapped("k", &fresh) {
                    Err(StorageError::Corrupt { key, reason }) => {
                        assert_eq!(key, "k");
                        assert!(reason.contains(lacking.as_str()), "{case}: {reason}");
                    }
                    other => panic!("{case}: without {lacking}: {other:?}"),
                }
            }
        }
    }

    /// `mapping`, with the keys `to` gives for some of them instead.
    fn mapping_but(keys: &[String], to: &[(&String, &str)]) -> HashMap<String, String> {
        let mut fresh = mapping(keys);
        for (old, new) in to {
            assert!(fresh.insert((*old).clone(), new.to_string()).is_some());
        }
        fresh
    }

    /// That `outcome` is a refusal of the checkpoint under "k" that says
    /// each of `saying`.
    fn refused(case: &str, outcome: Result<Manifest>, saying: &[&str]) {
        match outcome {
            Err(StorageError::Corrupt { key, reason }) => {
                assert_eq!(key, "k");
                for said in saying {
                    assert!(reason.contains(said), "{case}: {reason}");
                }
            }
            other => panic!("{case}: {other:?}"),
        }
    }

    #[test]
    fn a_remap_that_gives_two_files_one_key_is_refused_whichever_fields_name_them() {
        for (case, manifest, names) in checkpoints() {
            // Every pair of files it names, in both orders: two segments,
            // a segment and its own index or vectors, a log and the
            // checkpoint before, and the rest.
            for (one, other) in names
                .iter()
                .flat_map(|one| names.iter().map(move |o| (one, o)))
            {
                if one == other {
                    continue;
                }
                let both = new(one);
                let fresh = mapping_but(&names, &[(other, &both)]);
                let outcome = manifest.clone().remapped("k", &fresh);
                refused(case, outcome, &[one.as_str(), other.as_str(), &both]);
            }
        }
        // A segment, its index and its vectors all given the one key.
        let (_, manifest, names) = checkpoints().remove(2);
        let fresh = mapping_but(
            &names,
            &[(&names[2], &new(&names[1])), (&names[3], &new(&names[1]))],
        );
        refused(
            "a merge",
            manifest.remapped("k", &fresh),
            &[&new(&names[1])],
        );
    }

    #[test]
    fn a_remap_that_gives_a_file_the_key_another_had_is_refused() {
        for (case, manifest, names) in checkpoints() {
            for (one, other) in names
                .iter()
                .flat_map(|one| names.iter().map(move |o| (one, o)))
            {
                if one == other {
                    continue;
                }
                // One file moved onto the key another had, which moved on.
                let chained = mapping_but(&names, &[(one, other.as_str())]);
                let outcome = manifest.clone().remapped("k", &chained);
                refused(case, outcome, &[one.as_str(), other.as_str()]);
                // And two files that change keys with each other.
                let swapped = mapping_but(&names, &[(one, other.as_str()), (other, one.as_str())]);
                let outcome = manifest.clone().remapped("k", &swapped);
                refused(case, outcome, &[one.as_str(), other.as_str()]);
            }
        }
    }

    #[test]
    fn a_remap_to_a_key_that_is_empty_or_not_a_plain_path_is_refused() {
        for (case, manifest, names) in checkpoints() {
            for old in &names {
                for unfit in [
                    "",
                    "tenant-b//x",
                    "tenant-b/./x",
                    "tenant-b/../x",
                    "tenant-b\\x",
                    "/x",
                    "x/",
                ] {
                    let fresh = mapping_but(&names, &[(old, unfit)]);
                    let outcome = manifest.clone().remapped("k", &fresh);
                    refused(case, outcome, &[old.as_str(), "no plain key"]);
                }
            }
        }
    }

    /// A checkpoint that names one file twice, as no writer makes one, is
    /// not refused for it: both names are given the file's one new key. A
    /// restore that is to refuse such a checkpoint must say so itself.
    #[test]
    fn a_file_a_checkpoint_names_twice_is_given_its_one_new_key_both_times() {
        let log = old("log", "00000000000000000015-w");
        let bare = old("segment", "00000000000000000016-w-5b");
        let twice = checkpoint(
            None,
            vec![segment(bare.clone(), None, None); 2],
            vec![log.clone(), log.clone()],
        );
        let fresh = mapping(&[bare.clone(), log.clone()]);
        let remapped = twice.remapped("k", &fresh).unwrap();
        let keys: Vec<&String> = remapped.segments.iter().map(|s| &s.key).collect();
        assert_eq!(keys, [&fresh[&bare], &fresh[&bare]]);
        assert_eq!(remapped.replay, [fresh[&log].clone(), fresh[&log].clone()]);
    }

    #[test]
    fn a_checkpoint_remapped_to_the_keys_it_had_is_written_as_it_was() {
        for (case, manifest, names) in checkpoints() {
            let same: HashMap<String, String> =
                names.iter().map(|key| (key.clone(), key.clone())).collect();
            let frame = encode(Kind::Manifest, &manifest).unwrap();
            let read: Manifest = decode("k", Kind::Manifest, &frame).unwrap();
            let remapped = read.remapped("k", &same).unwrap();
            assert_eq!(encode(Kind::Manifest, &remapped).unwrap(), frame, "{case}");
        }
    }

    fn index_of(segment: &str) -> Bytes {
        let index = SegmentIndex {
            segment: segment.to_string(),
            vectors: Vec::new(),
        };
        encode(Kind::Index, &index).unwrap()
    }

    #[test]
    fn a_rekeyed_index_says_the_key_its_segment_now_has() {
        let (from, to) = (old("segment", "11-w-1f"), new(&old("segment", "11-w-1f")));
        let rekeyed = rekeyed_index("k", &index_of(&from), &from, &to).unwrap();
        assert!(!holds(&rekeyed, &from), "the key the segment had is kept");
        let read: SegmentIndex = decode("k", Kind::Index, &rekeyed).unwrap();
        assert_eq!(read.segment, to);
        assert_eq!(rekeyed, index_of(&to));
    }

    #[test]
    fn an_index_of_another_segment_is_not_rekeyed() {
        let (from, to) = (old("segment", "11-w-1f"), new(&old("segment", "11-w-1f")));
        // Another segment's, and one that says the new key already: it
        // is not the index of the segment that was under `from`.
        for anothers in [index_of(&old("segment", "13-w-2e")), index_of(&to)] {
            assert!(matches!(
                rekeyed_index("k", &anothers, &from, &to),
                Err(StorageError::Corrupt { key, reason })
                    if key == "k" && reason == "it indexes another segment"
            ));
        }
    }

    #[test]
    fn an_index_that_cannot_be_read_is_not_rekeyed() {
        let (from, to) = (old("segment", "11-w-1f"), new(&old("segment", "11-w-1f")));
        let whole = index_of(&from);
        let mut damaged = whole.to_vec();
        *damaged.last_mut().unwrap() ^= 0x01;
        let manifest = encode(Kind::Manifest, &checkpoints().remove(0).1).unwrap();
        for unreadable in [&damaged[..], &whole[..whole.len() - 1], &manifest[..], &[]] {
            assert!(matches!(
                rekeyed_index("k", unreadable, &from, &to),
                Err(StorageError::Corrupt { .. })
            ));
        }
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
