//! Links: what continuous backup ships for each committed head, one per
//! position, to the backup store (DW-668, amendment 1).
//!
//! A link is created once, under its position alone, by
//! [`crate::blobs::Blobs::put_new`], at
//! `_backups/{tenant}/ship/{lineage}/{cluster}/{incarnation}/{partition}/links/{position:020}.link`.
//! It is a header, which says what was committed there, and the head's
//! frame, byte for byte as the primary holds it:
//!
//! - **inline**, after the header in the link itself, when both fit in
//!   [`inline_most`] (every ordinary commit);
//! - or **spilled**: shipped first, as an ordinary file under
//!   `frames/{original key}` beside `links/`, which the link then names
//!   by the key in its header. A log has no bound on its size (a patch
//!   logs the whole object it patched), so a frame of any size is shipped.
//!
//! A link is framed by its own magic, [`LINK_VERSION`] and a CRC32 over
//! the header, not as a file of `format::Kind`: it is read only by backup
//! code, and its version is not the cluster's file format (see
//! docs/compatibility.md).
//!
//! Two links of one position are of the same commit when every field of
//! their headers but `body` is the same ([`Header::differs`]); one shipped
//! inline by one node and spilled by another, whose stores create files
//! of different sizes at most, is one commit. Anything else at a position
//! is a fork.
//!
//! A manifest's header lists its checkpoint files only: its segments,
//! their indexes and vectors files. The logs it replays are the positions
//! just after `replay_after` up to just before its own, each with a link
//! of its own.
//!
//! `before` names the position and frame sum of the commit before. A link
//! with none has no predecessor in its timeline: what that may be (the
//! first commit, or a base the timeline starts from) is for the reader of
//! a chain to judge, not for a link alone.

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vellev_core::{Partition as PartitionNumber, TenantId};

use crate::blobs::Layout;
use crate::coordinator::{ClusterId, Generation, Incarnation, Lineage, LogPosition, PartitionKey};
use crate::{Result, StorageError};

/// The newest form of link this release reads and writes.
pub const LINK_VERSION: u16 = 1;

/// The most a link with its frame inline may be, whatever its store
/// creates.
pub const INLINE_MOST: usize = 8 * 1024 * 1024;

const MAGIC: [u8; 4] = *b"VLNK";
/// Magic, version, the header's length and the CRC32 over the version,
/// the length and the header.
const PREAMBLE: usize = 4 + 2 + 4 + 4;

/// A SHA-256, as lowercase hex.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sum(pub String);

impl Sum {
    /// The SHA-256 of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        Self(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
}

/// What kind of head a link ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Head {
    Log,
    Manifest,
}

impl Head {
    /// The directory of a partition its files lie in.
    fn kind(self) -> &'static str {
        match self {
            Head::Log => "log",
            Head::Manifest => "manifest",
        }
    }
}

/// The commit before a link's: its position and its frame's sum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Before {
    pub position: LogPosition,
    pub sum: Sum,
}

/// A checkpoint file a manifest needs: a segment, its index or its
/// vectors file, by its key in the primary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needed {
    pub key: String,
    pub size: u64,
    pub sum: Sum,
}

/// Where a link's frame is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Body {
    /// After the header, in the link.
    Inline,
    /// Under `frames/{key}` beside the link, shipped before it.
    Spilled,
}

/// What a link says was committed at its position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub tenant: TenantId,
    pub lineage: Lineage,
    pub cluster: ClusterId,
    pub incarnation: Incarnation,
    pub partition: PartitionNumber,
    pub position: LogPosition,
    /// The head's key in the primary.
    pub key: String,
    pub head: Head,
    pub generation: Generation,
    /// When the commit was made, by the database's clock, as
    /// `vellev.commits` has it.
    pub committed_at: DateTime<Utc>,
    /// The frame's SHA-256 and size.
    pub sum: Sum,
    pub size: u64,
    pub before: Option<Before>,
    /// A manifest's checkpoint files; none for a log.
    pub files: Vec<Needed>,
    pub body: Body,
}

impl Header {
    /// The fields in which `other` says another commit was made than this
    /// one: every field but `body`, by name. None if they are of one
    /// commit, however each was shipped. A field added to the header is
    /// compared unless it is named here as not: the fields are taken
    /// apart whole, so that one cannot be left out by forgetting it.
    pub fn differs(&self, other: &Header) -> Vec<&'static str> {
        let Header {
            tenant,
            lineage,
            cluster,
            incarnation,
            partition,
            position,
            key,
            head,
            generation,
            committed_at,
            sum,
            size,
            before,
            files,
            body: _,
        } = self;
        [
            ("tenant", *tenant == other.tenant),
            ("lineage", *lineage == other.lineage),
            ("cluster", *cluster == other.cluster),
            ("incarnation", *incarnation == other.incarnation),
            ("partition", *partition == other.partition),
            ("position", *position == other.position),
            ("key", *key == other.key),
            ("head", *head == other.head),
            ("generation", *generation == other.generation),
            ("committed_at", *committed_at == other.committed_at),
            ("sum", *sum == other.sum),
            ("size", *size == other.size),
            ("before", *before == other.before),
            ("files", *files == other.files),
        ]
        .into_iter()
        .filter(|(_, same)| !same)
        .map(|(field, _)| field)
        .collect()
    }

    fn encoded(&self) -> Vec<u8> {
        rmp_serde::to_vec_named(self).expect("a header is strings, numbers and lists of them")
    }
}

/// Where one timeline of a partition's history is shipped: one cluster's
/// history of one incarnation of one tenant of one lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timeline {
    pub tenant: TenantId,
    pub lineage: Lineage,
    pub cluster: ClusterId,
    pub incarnation: Incarnation,
    pub partition: PartitionNumber,
}

impl Timeline {
    /// The timeline a header says its link is of.
    pub fn of(header: &Header) -> Self {
        Self {
            tenant: header.tenant.clone(),
            lineage: header.lineage,
            cluster: header.cluster,
            incarnation: header.incarnation,
            partition: header.partition,
        }
    }

    pub fn prefix(&self) -> String {
        let Self {
            tenant,
            lineage,
            cluster,
            incarnation,
            partition,
        } = self;
        format!(
            "_backups/{tenant}/ship/{lineage}/{cluster}/{}/{}/",
            incarnation.0, partition.0
        )
    }

    /// Where the link of `position` is.
    pub fn link(&self, position: LogPosition) -> String {
        format!("{}links/{:020}.link", self.prefix(), position.0)
    }

    /// Where the checkpoint file under `key` in the primary is copied to.
    pub fn file(&self, key: &str) -> String {
        format!("{}files/{key}", self.prefix())
    }

    /// Where the frame of the head under `key` in the primary is spilled.
    pub fn frame(&self, key: &str) -> String {
        format!("{}frames/{key}", self.prefix())
    }
}

/// The most a link with its frame inline may be in a store that creates
/// files of `creates_at_most` bytes at most.
pub fn inline_most(creates_at_most: usize) -> usize {
    INLINE_MOST.min(creates_at_most)
}

/// Why a link cannot be made for a head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unlinkable {
    /// The store creates no file only if none is there.
    NoCreate,
    /// Even with its frame spilled, the header is larger than the store
    /// creates.
    HeaderTooLarge { bytes: usize, most: usize },
}

/// How the head `header` describes is shipped to a store that creates
/// files of `creates_at_most` bytes at most: inline if the link with its
/// frame fits [`inline_most`], else spilled. Its `body` is not read.
pub fn shape(
    header: &Header,
    creates_at_most: Option<usize>,
) -> std::result::Result<Body, Unlinkable> {
    let most = creates_at_most.ok_or(Unlinkable::NoCreate)?;
    let link = |body| {
        let header = Header {
            body,
            ..header.clone()
        };
        PREAMBLE + header.encoded().len()
    };
    let size = usize::try_from(header.size).unwrap_or(usize::MAX);
    if link(Body::Inline).saturating_add(size) <= inline_most(most) {
        return Ok(Body::Inline);
    }
    match link(Body::Spilled) {
        bytes if bytes <= most => Ok(Body::Spilled),
        bytes => Err(Unlinkable::HeaderTooLarge { bytes, most }),
    }
}

/// A link: its header and, if inline, its frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub header: Header,
    pub frame: Option<Bytes>,
}

/// Whether `frame` is the one `header` sums.
fn summed(header: &Header, frame: &[u8]) -> bool {
    frame.len() as u64 == header.size && Sum::of(frame) == header.sum
}

/// The link of `header`, with `frame` after it if its body is inline. The
/// frame must be the one the header sums, and given exactly when inline.
pub fn encode(header: &Header, frame: Option<&[u8]>) -> Result<Bytes> {
    let refused =
        |reason: &str| StorageError::Encode(format!("the link of {}: {reason}", header.key));
    let frame = match (header.body, frame) {
        (Body::Inline, Some(frame)) if summed(header, frame) => frame,
        (Body::Inline, Some(_)) => return Err(refused("the frame is not the one its header sums")),
        (Body::Inline, None) => return Err(refused("an inline link was given no frame")),
        (Body::Spilled, None) => &[],
        (Body::Spilled, Some(_)) => return Err(refused("a spilled link was given its frame")),
    };
    let encoded = header.encoded();
    let length = u32::try_from(encoded.len()).map_err(|_| refused("its header is too large"))?;
    let mut link = Vec::with_capacity(PREAMBLE + encoded.len() + frame.len());
    link.extend_from_slice(&MAGIC);
    link.extend_from_slice(&LINK_VERSION.to_le_bytes());
    link.extend_from_slice(&length.to_le_bytes());
    let mut crc = crc32fast::Hasher::new();
    crc.update(&link[MAGIC.len()..]);
    crc.update(&encoded);
    link.extend_from_slice(&crc.finalize().to_le_bytes());
    link.extend_from_slice(&encoded);
    link.extend_from_slice(frame);
    Ok(Bytes::from(link))
}

/// The link read from under `key`, trusted only so far: refused unless it
/// is whole, of a version this release reads, lies where its header says
/// it does, names a head of its own partition at its own position, lists
/// only checkpoint files of that partition (none for a log), and, if
/// inline, carries the frame its header sums.
///
/// The version is read before the CRC is checked, as a later version may
/// frame its header otherwise: so a link whose version bytes are damaged
/// into a later version is refused as `UnsupportedVersion`, one this
/// release does not read, rather than as corrupt.
pub fn decode(key: &str, bytes: &[u8]) -> Result<Link> {
    let corrupt = |reason: String| StorageError::corrupt(key, reason);
    let Some(preamble) = bytes.get(..PREAMBLE) else {
        return Err(corrupt("shorter than a link's preamble".into()));
    };
    if preamble[..4] != MAGIC {
        return Err(corrupt("not a link".into()));
    }
    let version = u16::from_le_bytes([preamble[4], preamble[5]]);
    if version == 0 {
        return Err(corrupt("link version zero".into()));
    }
    if version > LINK_VERSION {
        return Err(StorageError::UnsupportedVersion {
            key: key.to_string(),
            found: version,
            supported: LINK_VERSION,
        });
    }
    let length = u32::from_le_bytes(preamble[6..10].try_into().expect("four bytes")) as usize;
    let expected = u32::from_le_bytes(preamble[10..14].try_into().expect("four bytes"));
    let Some(encoded) = bytes.get(PREAMBLE..PREAMBLE.saturating_add(length)) else {
        return Err(corrupt("shorter than its header".into()));
    };
    let mut crc = crc32fast::Hasher::new();
    crc.update(&preamble[MAGIC.len()..10]);
    crc.update(encoded);
    if crc.finalize() != expected {
        return Err(corrupt("its header's checksum does not hold".into()));
    }
    let header: Header = crate::format::parse(key, encoded)?;
    let rest = &bytes[PREAMBLE + length..];
    let frame = match header.body {
        Body::Inline if summed(&header, rest) => Some(Bytes::copy_from_slice(rest)),
        Body::Inline => return Err(corrupt("its frame is not the one its header sums".into())),
        Body::Spilled if rest.is_empty() => None,
        Body::Spilled => return Err(corrupt("bytes follow a spilled link's header".into())),
    };
    trusted(&header).map_err(corrupt)?;
    if Timeline::of(&header).link(header.position) != key {
        return Err(corrupt(format!(
            "it says it is the link of {} at position {}",
            Timeline::of(&header).prefix(),
            header.position.0
        )));
    }
    Ok(Link { header, frame })
}

/// Why `header` names what no link of its partition may name, if it does:
/// keys are read exactly as the partition lays them out, as restore puts
/// the frame back under the head's own key.
fn trusted(header: &Header) -> std::result::Result<(), String> {
    let partition = PartitionKey {
        tenant: header.tenant.clone(),
        partition: header.partition,
    };
    let layout = Layout(&partition);
    let named = |key| layout.named_by(key);
    let head = named(&header.key);
    let laid_out = head.is_some_and(|named| {
        named.kind == header.head.kind()
            && named.position == header.position
            && named.generation == header.generation
    });
    if !laid_out {
        return Err(format!(
            "it names {} as its partition's {} at position {} of generation {}",
            header.key,
            header.head.kind(),
            header.position.0,
            header.generation.0
        ));
    }
    if let Some(before) = &header.before {
        if before.position >= header.position {
            return Err(format!(
                "it says the commit before it is at position {}",
                before.position.0
            ));
        }
    }
    if header.head == Head::Log && !header.files.is_empty() {
        return Err("it is a log's, and lists files".into());
    }
    let checkpoint =
        |key| named(key).is_some_and(|named| matches!(named.kind, "segment" | "index" | "vectors"));
    match header.files.iter().find(|file| !checkpoint(&file.key)) {
        Some(file) => Err(format!(
            "it lists {}, no checkpoint file of its partition",
            file.key
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use uuid::Uuid;

    use super::*;
    use crate::format::{self, Kind, LogSegment};

    const WRITER: &str = "8560994bc5f64972b4d9de1d3358753b";

    fn log_key(position: u64) -> String {
        format!("tenant-a/0/log/{position:020}-00000000000000000002-{WRITER}.log")
    }

    fn frame(position: u64) -> Bytes {
        let log = LogSegment {
            position: LogPosition(position),
            prev: Some(log_key(position - 1)),
            ops: Vec::new(),
            answered: Vec::new(),
        };
        format::encode(Kind::Log, &log).unwrap()
    }

    /// The header of the log at position 7 whose frame is `frame`.
    fn header(frame: &[u8]) -> Header {
        Header {
            tenant: TenantId::new("tenant-a").unwrap(),
            lineage: Lineage(Uuid::from_u128(1)),
            cluster: ClusterId(Uuid::from_u128(2)),
            incarnation: Incarnation(3),
            partition: PartitionNumber(0),
            position: LogPosition(7),
            key: log_key(7),
            head: Head::Log,
            generation: Generation(2),
            committed_at: Utc.with_ymd_and_hms(2026, 10, 12, 9, 0, 0).unwrap()
                + chrono::Duration::microseconds(123_456),
            sum: Sum::of(frame),
            size: frame.len() as u64,
            before: Some(Before {
                position: LogPosition(6),
                sum: Sum::of(b"the frame at six"),
            }),
            files: Vec::new(),
            body: Body::Inline,
        }
    }

    fn manifest_header() -> Header {
        let prefix = "tenant-a/0";
        let at = format!("00000000000000000007-00000000000000000002-{WRITER}");
        let needed = |kind: &str| Needed {
            key: format!("{prefix}/{kind}/{at}.{kind}"),
            size: 10,
            sum: Sum::of(kind.as_bytes()),
        };
        Header {
            key: format!("{prefix}/manifest/{at}.manifest"),
            head: Head::Manifest,
            files: vec![needed("segment"), needed("index"), needed("vectors")],
            ..header(b"a manifest")
        }
    }

    #[test]
    fn a_sum_is_the_sha256_of_the_bytes_in_lowercase_hex() {
        assert_eq!(
            Sum::of(b"abc").0,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_timeline_lies_under_the_tenant_by_lineage_cluster_incarnation_and_partition() {
        let timeline = Timeline::of(&header(b"x"));
        let at = format!(
            "_backups/tenant-a/ship/{}/{}/3/0/",
            Uuid::from_u128(1).simple(),
            Uuid::from_u128(2).simple()
        );
        assert_eq!(timeline.prefix(), at);
        assert_eq!(
            timeline.link(LogPosition(7)),
            format!("{at}links/00000000000000000007.link")
        );
        assert_eq!(
            timeline.frame(&log_key(7)),
            format!("{at}frames/{}", log_key(7))
        );
    }

    #[test]
    fn a_link_is_inline_up_to_what_the_store_creates_and_never_more_than_inline_most() {
        assert_eq!(inline_most(16 * 1024 * 1024), INLINE_MOST);
        assert_eq!(inline_most(5 * 1024 * 1024), 5 * 1024 * 1024);
        assert_eq!(inline_most(INLINE_MOST), INLINE_MOST);
    }

    #[test]
    fn a_frame_goes_inline_exactly_while_the_link_fits_and_spills_one_byte_over() {
        let frame = Bytes::from(vec![7u8; 1000]);
        let mut header = header(&frame);
        // The link of a 1000-byte frame inline, its exact length.
        let inline_len = encode(&header, Some(&frame)).unwrap().len();
        // A store that creates exactly that much takes it inline.
        assert_eq!(shape(&header, Some(inline_len)), Ok(Body::Inline));
        // One byte less, and the frame is spilled.
        assert_eq!(shape(&header, Some(inline_len - 1)), Ok(Body::Spilled));
        // A store that creates more is held to INLINE_MOST.
        header.size = INLINE_MOST as u64;
        assert_eq!(shape(&header, Some(16 * 1024 * 1024)), Ok(Body::Spilled));
        // A store that creates less than INLINE_MOST holds it to that.
        header.size = 4 * 1024 * 1024;
        assert_eq!(shape(&header, Some(16 * 1024 * 1024)), Ok(Body::Inline));
        assert_eq!(shape(&header, Some(4 * 1024 * 1024)), Ok(Body::Spilled));
    }

    #[test]
    fn a_store_that_creates_nothing_or_too_little_for_the_header_takes_no_link() {
        let header = manifest_header();
        assert_eq!(shape(&header, None), Err(Unlinkable::NoCreate));
        let spilled = Header {
            body: Body::Spilled,
            ..header.clone()
        };
        let bytes = encode(&spilled, None).unwrap().len();
        assert_eq!(shape(&header, Some(bytes)), Ok(Body::Spilled));
        assert_eq!(
            shape(&header, Some(bytes - 1)),
            Err(Unlinkable::HeaderTooLarge {
                bytes,
                most: bytes - 1
            })
        );
    }

    #[test]
    fn two_headers_are_of_one_commit_when_every_field_but_the_body_is_the_same() {
        let one = manifest_header();
        let spilled = Header {
            body: Body::Spilled,
            ..one.clone()
        };
        assert_eq!(one.differs(&spilled), Vec::<&str>::new());
        assert_eq!(one.differs(&one.clone()), Vec::<&str>::new());

        let other = Uuid::from_u128(9);
        let changed: Vec<(&str, Header)> = vec![
            (
                "tenant",
                Header {
                    tenant: TenantId::new("tenant-b").unwrap(),
                    ..one.clone()
                },
            ),
            (
                "lineage",
                Header {
                    lineage: Lineage(other),
                    ..one.clone()
                },
            ),
            (
                "cluster",
                Header {
                    cluster: ClusterId(other),
                    ..one.clone()
                },
            ),
            (
                "incarnation",
                Header {
                    incarnation: Incarnation(4),
                    ..one.clone()
                },
            ),
            (
                "partition",
                Header {
                    partition: PartitionNumber(1),
                    ..one.clone()
                },
            ),
            (
                "position",
                Header {
                    position: LogPosition(8),
                    ..one.clone()
                },
            ),
            (
                "key",
                Header {
                    key: format!("{}x", one.key),
                    ..one.clone()
                },
            ),
            (
                "head",
                Header {
                    head: Head::Log,
                    ..one.clone()
                },
            ),
            (
                "generation",
                Header {
                    generation: Generation(5),
                    ..one.clone()
                },
            ),
            (
                "committed_at",
                Header {
                    committed_at: one.committed_at + chrono::Duration::microseconds(1),
                    ..one.clone()
                },
            ),
            (
                "sum",
                Header {
                    sum: Sum::of(b"another frame"),
                    ..one.clone()
                },
            ),
            (
                "size",
                Header {
                    size: one.size + 1,
                    ..one.clone()
                },
            ),
            (
                "before",
                Header {
                    before: None,
                    ..one.clone()
                },
            ),
            (
                "files",
                Header {
                    files: one.files[1..].to_vec(),
                    ..one.clone()
                },
            ),
        ];
        for (field, header) in changed {
            assert_eq!(one.differs(&header), vec![field], "{field}");
            assert_eq!(header.differs(&one), vec![field], "{field}");
        }
    }

    #[test]
    fn a_link_reads_back_as_it_was_written_inline_and_spilled() {
        let frame = frame(7);
        let inline = header(&frame);
        let key = Timeline::of(&inline).link(LogPosition(7));
        let bytes = encode(&inline, Some(&frame)).unwrap();
        let read = decode(&key, &bytes).unwrap();
        assert_eq!(read.header, inline);
        assert_eq!(read.frame.as_deref(), Some(&frame[..]));
        // Its time comes back to the microsecond.
        assert_eq!(read.header.committed_at.timestamp_subsec_micros(), 123_456);

        let spilled = Header {
            body: Body::Spilled,
            ..manifest_header()
        };
        let bytes = encode(&spilled, None).unwrap();
        let read = decode(&key, &bytes).unwrap();
        assert_eq!(read.header, spilled);
        assert_eq!(read.frame, None);
    }

    #[test]
    fn a_frame_is_given_with_a_link_exactly_when_inline_and_must_be_the_one_summed() {
        let frame = frame(7);
        let inline = header(&frame);
        assert!(encode(&inline, None).is_err());
        assert!(encode(&inline, Some(b"another frame")).is_err());
        let spilled = Header {
            body: Body::Spilled,
            ..inline
        };
        assert!(encode(&spilled, Some(&frame)).is_err());
    }

    /// What `decode` says of `bytes` read from under the link key of
    /// position 7.
    fn read(bytes: &[u8]) -> Result<Link> {
        let key = Timeline::of(&header(b"x")).link(LogPosition(7));
        decode(&key, bytes)
    }

    fn corrupt(result: Result<Link>, what: &str) {
        assert!(
            matches!(result, Err(StorageError::Corrupt { .. })),
            "{what}: {result:?}"
        );
    }

    /// The link of `header` with its preamble's CRC made right again, so
    /// that only what was changed in it is wrong.
    fn framed(header: &Header, frame: Option<&[u8]>) -> Vec<u8> {
        let body = rmp_serde::to_vec_named(header).unwrap();
        let mut link = Vec::new();
        link.extend_from_slice(&MAGIC);
        link.extend_from_slice(&LINK_VERSION.to_le_bytes());
        link.extend_from_slice(&(body.len() as u32).to_le_bytes());
        let mut crc = crc32fast::Hasher::new();
        crc.update(&link[4..10]);
        crc.update(&body);
        link.extend_from_slice(&crc.finalize().to_le_bytes());
        link.extend_from_slice(&body);
        link.extend_from_slice(frame.unwrap_or_default());
        link
    }

    #[test]
    fn a_link_as_this_test_frames_it_is_read_so_the_refusals_below_are_of_one_fault_each() {
        let frame = frame(7);
        let header = header(&frame);
        assert_eq!(
            framed(&header, Some(&frame)),
            encode(&header, Some(&frame)).unwrap()
        );
        assert!(read(&framed(&header, Some(&frame))).is_ok());
    }

    /// `link` with its CRC made right again for whatever was changed in
    /// its version, length or header.
    fn resummed(mut link: Vec<u8>) -> Vec<u8> {
        let length = u32::from_le_bytes(link[6..10].try_into().unwrap()) as usize;
        let mut crc = crc32fast::Hasher::new();
        crc.update(&link[4..10]);
        crc.update(&link[PREAMBLE..PREAMBLE + length]);
        link[10..14].copy_from_slice(&crc.finalize().to_le_bytes());
        link
    }

    #[test]
    fn a_damaged_or_unknown_link_is_refused() {
        let frame = frame(7);
        let good = encode(&header(&frame), Some(&frame)).unwrap().to_vec();

        let mut magic = good.clone();
        magic[0] ^= 1;
        corrupt(read(&magic), "magic");

        let mut flipped = good.clone();
        flipped[PREAMBLE + 3] ^= 1;
        corrupt(read(&flipped), "a header byte");

        // A change that leaves the header readable and trusted, which only
        // its CRC tells: a digit of the sum it names of the commit before.
        let before = Sum::of(b"the frame at six").0;
        let at = good
            .windows(before.len())
            .position(|window| window == before.as_bytes())
            .unwrap();
        let mut other = good.clone();
        other[at] = if other[at] == b'0' { b'1' } else { b'0' };
        corrupt(read(&other), "another sum of the commit before");
        let read_so = read(&resummed(other)).unwrap();
        assert_ne!(read_so.header.before, header(&frame).before);

        corrupt(read(&good[..PREAMBLE - 1]), "shorter than a preamble");
        corrupt(read(&good[..PREAMBLE + 5]), "shorter than its header");

        let mut zero = good.clone();
        zero[4..6].copy_from_slice(&0u16.to_le_bytes());
        corrupt(read(&resummed(zero)), "version zero");

        // Its CRC is left as it was, now wrong: a later version is told
        // before the CRC is, as `decode` says.
        let mut newer = good.clone();
        newer[4..6].copy_from_slice(&(LINK_VERSION + 1).to_le_bytes());
        assert!(
            matches!(read(&newer), Err(StorageError::UnsupportedVersion { found, supported: LINK_VERSION, .. }) if found == LINK_VERSION + 1),
            "{:?}",
            read(&newer)
        );
    }

    #[test]
    fn an_inline_link_must_carry_exactly_the_frame_its_header_sums() {
        let frame = frame(7);
        let header = header(&frame);
        let mut longer = framed(&header, Some(&frame));
        longer.push(0);
        corrupt(read(&longer), "a byte after the frame");
        let wrong_size = Header {
            size: header.size + 1,
            ..header.clone()
        };
        corrupt(
            read(&framed(&wrong_size, Some(&frame))),
            "the frame it sums, of another size than it says",
        );
        corrupt(
            read(&framed(&header, Some(&frame[..frame.len() - 1]))),
            "a byte short",
        );
        let mut other = frame.to_vec();
        let last = other.len() - 1;
        other[last] ^= 1;
        corrupt(
            read(&framed(&header, Some(&other))),
            "another frame of its size",
        );
        let spilled = Header {
            body: Body::Spilled,
            ..header
        };
        corrupt(
            read(&framed(&spilled, Some(b"x"))),
            "a byte after a spilled header",
        );
    }

    #[test]
    fn a_link_must_lie_where_its_header_says_and_name_a_head_of_its_own_partition() {
        let frame = frame(7);
        let good = header(&frame);
        let refused =
            |header: Header, what: &str| corrupt(read(&framed(&header, Some(&frame))), what);
        refused(
            Header {
                position: LogPosition(8),
                key: log_key(8),
                ..good.clone()
            },
            "under another position's key",
        );
        refused(
            Header {
                incarnation: Incarnation(4),
                ..good.clone()
            },
            "of another timeline",
        );
        refused(
            Header {
                key: log_key(6),
                ..good.clone()
            },
            "a key of another position",
        );
        refused(
            Header {
                key: good.key.replace("tenant-a/0/", "tenant-b/0/"),
                ..good.clone()
            },
            "a key of another tenant",
        );
        refused(
            Header {
                key: good.key.replace("tenant-a/0/", "tenant-a/1/"),
                ..good.clone()
            },
            "a key of another partition",
        );
        refused(
            Header {
                key: format!("{}/..", good.key),
                ..good.clone()
            },
            "a key that is not plain",
        );
        refused(
            Header {
                head: Head::Manifest,
                ..good.clone()
            },
            "a log's key named as a manifest",
        );
        refused(
            Header {
                key: good.key.replace("/log/", "/segment/"),
                ..good.clone()
            },
            "a segment named as a log",
        );
        let needed = manifest_header().files;
        refused(
            Header {
                files: needed.clone(),
                ..good.clone()
            },
            "a log listing files",
        );
    }

    #[test]
    fn a_manifest_lists_only_checkpoint_files_of_its_own_partition() {
        let good = Header {
            body: Body::Spilled,
            ..manifest_header()
        };
        assert!(read(&framed(&good, None)).is_ok());
        let with = |key: String| {
            let mut files = good.files.clone();
            files[0].key = key;
            Header {
                files,
                ..good.clone()
            }
        };
        corrupt(
            read(&framed(&with(log_key(6)), None)),
            "a log among its files",
        );
        let elsewhere = good.files[0].key.replace("tenant-a/0/", "tenant-a/1/");
        corrupt(
            read(&framed(&with(elsewhere), None)),
            "a file of another partition",
        );
        let manifest = good.key.clone();
        corrupt(
            read(&framed(&with(manifest), None)),
            "a manifest among its files",
        );
        let unplain = format!("{}//x", good.files[0].key);
        corrupt(
            read(&framed(&with(unplain), None)),
            "a key that is not plain",
        );
    }

    #[test]
    fn a_head_key_must_be_laid_out_exactly_and_of_the_generation_its_header_says() {
        let frame = frame(7);
        let good = header(&frame);
        let refused = |header: Header, what: &str| {
            corrupt(read(&framed(&header, Some(&frame))), what);
        };
        refused(
            Header {
                key: "tenant-a/0/log/00000000000000000007-x/y".into(),
                ..good.clone()
            },
            "a key read by its first twenty characters alone",
        );
        refused(
            Header {
                key: good
                    .key
                    .replace("-00000000000000000002-", "-00000000000000000003-"),
                ..good.clone()
            },
            "a key of another generation than the header's",
        );
        refused(
            Header {
                generation: Generation(3),
                ..good.clone()
            },
            "a header of another generation than its key's",
        );
        refused(
            Header {
                key: good.key.replace(".log", "-merged.log"),
                ..good.clone()
            },
            "a merged log",
        );
        assert!(read(&framed(&good, Some(&frame))).is_ok());
    }

    #[test]
    fn a_checkpoint_file_must_be_laid_out_exactly() {
        let good = Header {
            body: Body::Spilled,
            ..manifest_header()
        };
        let with = |key: String| {
            let mut files = good.files.clone();
            files[0].key = key;
            Header {
                files,
                ..good.clone()
            }
        };
        let merged = good.files[0].key.replace(".segment", "-merged.segment");
        assert!(read(&framed(&with(merged), None)).is_ok());
        let loose = "tenant-a/0/segment/00000000000000000007-x/y".to_string();
        corrupt(read(&framed(&with(loose), None)), "a loosely laid out file");
        let other = good.files[0].key.replace(".segment", ".index");
        corrupt(
            read(&framed(&with(other), None)),
            "another kind as its extension",
        );
    }

    #[test]
    fn the_commit_before_a_link_is_at_an_earlier_position() {
        let frame = frame(7);
        let good = header(&frame);
        for position in [7, 8, u64::MAX] {
            let before = Header {
                before: Some(Before {
                    position: LogPosition(position),
                    sum: Sum::of(b"x"),
                }),
                ..good.clone()
            };
            corrupt(
                read(&framed(&before, Some(&frame))),
                "a commit before at or after it",
            );
        }
        let first = Header {
            before: None,
            ..good.clone()
        };
        assert!(read(&framed(&first, Some(&frame))).is_ok());
        assert!(read(&framed(&good, Some(&frame))).is_ok());
    }
}
