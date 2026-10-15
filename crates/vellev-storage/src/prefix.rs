//! The start of a log's frame, read as it streams past: the position the
//! log says it is at and the head it says it was made on top of, found
//! without holding the frame whole (DW-668, amendment 1, A3).
//!
//! A spilled log is shipped a part at a time, and its link is created
//! only once the log is seen to be the one its row of `vellev.commits`
//! names. So the log's `position` and `prev` are read from the start of
//! its body as the parts go by:
//!
//! - the frame's header must say a log, of a format this release reads;
//! - the body is a MessagePack map, read entry by entry, by key: an entry
//!   of a key not looked for is skipped whole, whatever it holds, as a
//!   newer writer may add one;
//! - reaching `ops`, or the end of the map, before both `position` and
//!   `prev` were found fails: the shipper stalls rather than guess, and
//!   never reads through `ops`, which is most of a large log;
//! - so does finding neither within [`PREFIX_MOST`] bytes of the body.
//!
//! Logs are encoded as maps with their fields named, in the order they
//! are declared (`position` first, then `prev`); nothing here relies on
//! the order. A key given twice fails, as a full read of the log does.
//!
//! What is found is only what the start of the frame says, read before
//! the rest has come: a [`Found`], as soon as it is said and as often as
//! it is said again, may be acted on only once the whole frame has come
//! and its CRC and SHA-256 hold (amendment 1, A2). A later part is neither
//! read nor checked here.
//!
//! The start is read again from its beginning with each part taken in, so
//! a frame fed a byte at a time costs the square of the start's length;
//! parts as the store gives them, of `part_bytes`, cost nothing to speak
//! of.

use crate::coordinator::LogPosition;
use crate::format::{kind_of_header, Kind, BODY};
use crate::{Result, StorageError};

/// The most of a body read for `position` and `prev`.
pub const PREFIX_MOST: usize = 1024 * 1024;

/// What the start of a log says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub position: LogPosition,
    pub prev: Option<String>,
}

/// The start of the log under `key`, as its frame comes a part at a time.
/// No more of it is held than its header and [`PREFIX_MOST`] bytes of its
/// body, and none once what it says is found.
#[derive(Debug)]
pub struct Prefix {
    key: String,
    held: Vec<u8>,
    found: Option<Found>,
}

impl Prefix {
    pub fn new(key: &str) -> Self {
        Self {
            key: key.to_string(),
            held: Vec::new(),
            found: None,
        }
    }

    /// Take in the next part of the frame, and say what its start says
    /// once that has come: `None` while more is needed. Once found, what
    /// was found is said again for every part after.
    pub fn feed(&mut self, part: &[u8]) -> Result<Option<Found>> {
        if let Some(found) = &self.found {
            return Ok(Some(found.clone()));
        }
        let most = BODY + PREFIX_MOST;
        let room = most.saturating_sub(self.held.len());
        self.held.extend_from_slice(&part[..part.len().min(room)]);
        match self.read()? {
            Some(found) => {
                self.held = Vec::new();
                self.found = Some(found.clone());
                Ok(Some(found))
            }
            None if self.held.len() >= most => Err(self.corrupt(format!(
                "its position and prev were not found within {PREFIX_MOST} bytes of its body"
            ))),
            None => Ok(None),
        }
    }

    /// The frame has all come: what its start says, or why it said none.
    pub fn finish(mut self) -> Result<Found> {
        if let Some(found) = self.found.take() {
            return Ok(found);
        }
        match self.read()? {
            Some(found) => Ok(found),
            None => Err(self.corrupt("the frame ended before its position and prev".into())),
        }
    }

    fn corrupt(&self, reason: String) -> StorageError {
        StorageError::corrupt(&self.key, reason)
    }

    /// What the bytes held so far say, `None` if more are needed for it.
    fn read(&self) -> Result<Option<Found>> {
        if self.held.len() < BODY {
            return Ok(None);
        }
        if kind_of_header(&self.key, &self.held)? != Kind::Log {
            return Err(self.corrupt("it is not a log".into()));
        }
        let mut body = Reader {
            bytes: &self.held[BODY..],
            at: 0,
        };
        match body.start() {
            Ok(found) => Ok(Some(found)),
            Err(Stop::More) => Ok(None),
            Err(Stop::Bad(reason)) => Err(self.corrupt(reason.into())),
        }
    }
}

/// Whether `key` is one of the two looked for, found already: a log's map
/// gives each key once, as a full read of it requires.
fn found_twice(key: &str, position: bool, prev: bool) -> bool {
    (key == "position" && position) || (key == "prev" && prev)
}

/// Why reading stopped: the bytes ran out before what was read ended, or
/// what was read is not what a log's start may be.
enum Stop {
    More,
    Bad(&'static str),
}

type Read<T> = std::result::Result<T, Stop>;

/// MessagePack, read from `at` on.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: u64) -> Read<&[u8]> {
        let count = usize::try_from(count).map_err(|_| Stop::Bad("a length past memory"))?;
        let end = self.at.checked_add(count).ok_or(Stop::More)?;
        let taken = self.bytes.get(self.at..end).ok_or(Stop::More)?;
        self.at = end;
        Ok(taken)
    }

    fn byte(&mut self) -> Read<u8> {
        Ok(self.take(1)?[0])
    }

    /// An unsigned big-endian number of `count` bytes.
    fn number(&mut self, count: u64) -> Read<u64> {
        let bytes = self.take(count)?;
        Ok(bytes
            .iter()
            .fold(0, |sum, byte| sum << 8 | u64::from(*byte)))
    }

    /// The `position` and `prev` of a log's body, a map read by key.
    fn start(&mut self) -> Read<Found> {
        let entries = match self.byte()? {
            marker @ 0x80..=0x8f => u64::from(marker & 0x0f),
            0xde => self.number(2)?,
            0xdf => self.number(4)?,
            _ => return Err(Stop::Bad("its body is not a map")),
        };
        let (mut position, mut prev) = (None, None);
        for _ in 0..entries {
            let marker = self.byte()?;
            let key = match self.text(marker)? {
                Some(key) => key,
                None => {
                    // A key of another kind than text, skipped with its value.
                    self.at -= 1;
                    self.skip(2)?;
                    continue;
                }
            };
            match key.as_str() {
                "position" | "prev" if found_twice(&key, position.is_some(), prev.is_some()) => {
                    return Err(Stop::Bad("it gives its position or prev twice"))
                }
                "position" => position = Some(LogPosition(self.unsigned()?)),
                "prev" => prev = Some(self.prev()?),
                "ops" => return Err(Stop::Bad("it reaches ops before its position and prev")),
                _ => self.skip(1)?,
            }
            if let (Some(position), Some(prev)) = (position, &prev) {
                return Ok(Found {
                    position,
                    prev: prev.clone(),
                });
            }
        }
        Err(Stop::Bad("its map ended before its position and prev"))
    }

    /// The text whose marker is `marker`, `None` if it marks something else.
    fn text(&mut self, marker: u8) -> Read<Option<String>> {
        let length = match marker {
            0xa0..=0xbf => u64::from(marker & 0x1f),
            0xd9 => self.number(1)?,
            0xda => self.number(2)?,
            0xdb => self.number(4)?,
            _ => return Ok(None),
        };
        let bytes = self.take(length)?;
        match std::str::from_utf8(bytes) {
            Ok(text) => Ok(Some(text.to_string())),
            Err(_) => Err(Stop::Bad("a key or prev that is not UTF-8")),
        }
    }

    fn unsigned(&mut self) -> Read<u64> {
        match self.byte()? {
            marker @ 0x00..=0x7f => Ok(u64::from(marker)),
            0xcc => self.number(1),
            0xcd => self.number(2),
            0xce => self.number(4),
            0xcf => self.number(8),
            _ => Err(Stop::Bad("its position is not an unsigned integer")),
        }
    }

    fn prev(&mut self) -> Read<Option<String>> {
        match self.byte()? {
            0xc0 => Ok(None),
            marker => match self.text(marker)? {
                Some(prev) => Ok(Some(prev)),
                None => Err(Stop::Bad("its prev is neither nil nor text")),
            },
        }
    }

    /// Skip `values` whole values, whatever each holds, counting what is
    /// left to skip rather than going down into each, so that no depth of
    /// nesting is too deep.
    fn skip(&mut self, values: u64) -> Read<()> {
        let mut left = values;
        while left > 0 {
            left -= 1;
            let marker = self.byte()?;
            let skipped = match marker {
                0x00..=0x7f | 0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => 0,
                0x80..=0x8f => {
                    left = left.saturating_add(2 * u64::from(marker & 0x0f));
                    0
                }
                0x90..=0x9f => {
                    left = left.saturating_add(u64::from(marker & 0x0f));
                    0
                }
                0xa0..=0xbf => u64::from(marker & 0x1f),
                0xc4 | 0xd9 => self.number(1)?,
                0xc5 | 0xda => self.number(2)?,
                0xc6 | 0xdb => self.number(4)?,
                0xc7 => self.number(1)? + 1,
                0xc8 => self.number(2)? + 1,
                0xc9 => self.number(4)? + 1,
                0xca => 4,
                0xcb => 8,
                0xcc | 0xd0 => 1,
                0xcd | 0xd1 => 2,
                0xce | 0xd2 => 4,
                0xcf | 0xd3 => 8,
                0xd4 => 2,
                0xd5 => 3,
                0xd6 => 5,
                0xd7 => 9,
                0xd8 => 17,
                0xdc => {
                    left = left.saturating_add(self.number(2)?);
                    0
                }
                0xdd => {
                    left = left.saturating_add(self.number(4)?);
                    0
                }
                0xde => {
                    left = left.saturating_add(2 * self.number(2)?);
                    0
                }
                0xdf => {
                    left = left.saturating_add(2 * self.number(4)?);
                    0
                }
                0xc1 => return Err(Stop::Bad("a byte MessagePack never uses")),
            };
            self.take(skipped)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Serialize;
    use vellev_core::{ObjectId, RequestId};

    use super::*;
    use crate::format::{
        self, Answer, Answered, CollectionId, Kind, LogSegment, Op, FORMAT_VERSION,
    };
    use crate::StorageError;

    const KEY: &str = "tenant-a/0/log/00000000000000000007-x.log";

    fn ops() -> Vec<Op> {
        vec![Op::Delete {
            collection: CollectionId(1),
            id: ObjectId(uuid::Uuid::from_u128(7)),
        }]
    }

    fn answered() -> Vec<Answered> {
        vec![Answered {
            request: RequestId(uuid::Uuid::from_u128(9)),
            at: 1_791_300_000_000,
            answer: Answer::Made(Vec::new()),
        }]
    }

    fn log(prev: Option<&str>) -> LogSegment {
        LogSegment {
            position: LogPosition(7),
            prev: prev.map(String::from),
            ops: ops(),
            answered: answered(),
        }
    }

    /// The frame of a log whose body is `value`, as `format::encode`
    /// writes every log: `rmp_serde::to_vec_named`.
    fn framed(value: &impl Serialize) -> Vec<u8> {
        format::encode(Kind::Log, value).unwrap().to_vec()
    }

    /// A log's frame around a body of any bytes. Its CRC is not made
    /// right, which the start of a frame is not read for.
    fn raw(body: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(b"VLLV");
        frame.push(Kind::Log as u8);
        frame.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&[0; 4]);
        frame.extend_from_slice(body);
        frame
    }

    /// What the start of `frame` says, given whole.
    fn found(frame: &[u8]) -> Result<Found> {
        let mut prefix = Prefix::new(KEY);
        prefix.feed(frame)?;
        prefix.finish()
    }

    fn found_at(position: u64, prev: Option<&str>) -> Found {
        Found {
            position: LogPosition(position),
            prev: prev.map(String::from),
        }
    }

    fn fails(result: Result<Found>, because: &str) {
        match result {
            Err(StorageError::Corrupt { key, reason }) => {
                assert_eq!(key, KEY);
                assert!(reason.contains(because), "{reason:?} is not of {because:?}");
            }
            other => panic!("{other:?} is not a failure of {because:?}"),
        }
    }

    #[test]
    fn a_log_says_where_it_is_and_what_it_was_made_on_at_its_start() {
        let prev = "tenant-a/0/log/00000000000000000006-x.log";
        assert_eq!(
            found(&framed(&log(Some(prev)))).unwrap(),
            found_at(7, Some(prev))
        );
        assert_eq!(found(&framed(&log(None))).unwrap(), found_at(7, None));
    }

    #[test]
    fn what_a_log_says_is_found_before_its_ops_come_and_said_again_after() {
        let frame = framed(&log(Some("p")));
        let ops_at = frame
            .windows(3)
            .position(|window| window == b"ops")
            .unwrap();
        let mut prefix = Prefix::new(KEY);
        let mut said = None;
        for (at, byte) in frame.iter().enumerate() {
            if let Some(found) = prefix.feed(std::slice::from_ref(byte)).unwrap() {
                said.get_or_insert((at, found));
            }
        }
        let (at, found) = said.expect("found");
        assert!(at < ops_at, "found at byte {at}, after ops at {ops_at}");
        assert_eq!(found, found_at(7, Some("p")));
        assert_eq!(prefix.feed(b"more").unwrap(), Some(found_at(7, Some("p"))));
        assert_eq!(prefix.finish().unwrap(), found_at(7, Some("p")));
    }

    #[test]
    fn a_log_is_read_by_key_whatever_the_order_of_its_entries_before_ops() {
        #[derive(Serialize)]
        struct PrevFirst {
            prev: Option<String>,
            position: LogPosition,
            ops: Vec<Op>,
        }
        let body = PrevFirst {
            prev: Some("p".into()),
            position: LogPosition(7),
            ops: ops(),
        };
        assert_eq!(found(&framed(&body)).unwrap(), found_at(7, Some("p")));
    }

    #[test]
    fn an_entry_of_a_key_not_looked_for_is_skipped_whatever_it_holds() {
        #[derive(Serialize)]
        struct Newer {
            map: BTreeMap<String, Vec<i64>>,
            float: f64,
            bytes: serde_bytes::ByteBuf,
            nested: Vec<Option<(String, bool)>>,
            negative: (i8, i16, i32, i64),
            unsigned: (u8, u16, u32, u64),
            long: String,
            unit: (),
            position: LogPosition,
            also: f32,
            prev: Option<String>,
            ops: Vec<Op>,
            answered: Vec<Answered>,
        }
        let body = Newer {
            map: BTreeMap::from([
                ("a".into(), vec![1, -2, 300, -70_000]),
                ("b".into(), Vec::new()),
            ]),
            float: 0.5,
            bytes: serde_bytes::ByteBuf::from(vec![7; 300]),
            nested: vec![Some(("x".repeat(40), true)), None],
            negative: (-5, -300, -70_000, -5_000_000_000),
            unsigned: (200, 60_000, 4_000_000_000, u64::MAX),
            long: "y".repeat(70_000),
            unit: (),
            position: LogPosition(u64::MAX - 1),
            also: 1.5,
            prev: Some("p".into()),
            ops: ops(),
            answered: answered(),
        };
        assert_eq!(
            found(&framed(&body)).unwrap(),
            found_at(u64::MAX - 1, Some("p"))
        );

        // What serde does not write here, written by hand: an extension
        // of every size, a fixed array and map, and binary of 16 and 32
        // bits of length, before both entries looked for.
        let mut body = vec![0x8a];
        let mut entry = |key: &str, value: &[u8]| {
            body.push(0xa0 | key.len() as u8);
            body.extend_from_slice(key.as_bytes());
            body.extend_from_slice(value);
        };
        entry("e1", &[0xd4, 1, 9]);
        entry("e2", &[0xd5, 1, 9, 9]);
        entry(
            "e16",
            &[0xd8, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        );
        entry("e8", &[0xc7, 2, 1, 9, 9]);
        entry("e32", &[0xc9, 0, 0, 0, 1, 1, 9]);
        entry("a", &[0x92, 0x81, 0xa1, b'k', 0xc3, 0xc2]);
        entry("b16", &[0xc5, 0, 2, 1, 2]);
        entry("b32", &[0xc6, 0, 0, 0, 1, 1]);
        entry("position", &[0x07]);
        entry("prev", &[0xc0]);
        assert_eq!(found(&raw(&body)).unwrap(), found_at(7, None));

        // Every other form a value may take, each with something after
        // it, so that skipping one too few or too many bytes is seen.
        // What a value holds is 0xc1, which MessagePack never uses, so a
        // misread lands on it and fails rather than reads on out of step.
        let poison = |count: usize| vec![0xc1; count];
        let mut body = vec![0x8f];
        let mut entry = |key: &str, value: &[u8]| {
            body.push(0xa0 | key.len() as u8);
            body.extend_from_slice(key.as_bytes());
            body.extend_from_slice(value);
        };
        entry("s16", &[0xda, 0, 1, b'z']);
        entry("a16", &[0xdc, 0, 2, 0xc0, 0x01]);
        entry("a32", &[0xdd, 0, 0, 0, 2, 0xc0, 0x01]);
        entry("m16", &[0xde, 0, 1, 0xa1, b'k', 0xc0]);
        entry("m32", &[0xdf, 0, 0, 0, 1, 0xa1, b'k', 0xc0]);
        entry("fm", &[0x81, 0xa1, b'k', 0x92, 0xc0, 0x01]);
        entry("x16", &[0xc8, 0, 1, 1, 9]);
        entry("f4", &[0xd6, 1, 9, 9, 9, 9]);
        entry("f8", &[0xd7, 1, 9, 9, 9, 9, 9, 9, 9, 9]);
        entry("float", &[0xca, 0x3f, 0x80, 0, 0]);
        entry("f64", &[[0xcb].as_slice(), &poison(8)].concat());
        entry("b8", &[[0xc4, 2].as_slice(), &poison(2)].concat());
        entry("fs", &[[0xb4].as_slice(), &poison(20)].concat());
        entry("position", &[0x07]);
        entry("prev", &[0xc0]);
        assert_eq!(found(&raw(&body)).unwrap(), found_at(7, None));

        // A map of 16 bits of length, and a key that is not text, which
        // is skipped with its value.
        let mut body = vec![0xde, 0, 3, 0xcd, 0, 5, 0xa1, 0xc1];
        body.extend_from_slice(&[0xa8]);
        body.extend_from_slice(b"position");
        body.extend_from_slice(&[0xcd, 0x01, 0x2c, 0xa4]);
        body.extend_from_slice(b"prev");
        body.extend_from_slice(&[0xa1, b'p']);
        assert_eq!(found(&raw(&body)).unwrap(), found_at(300, Some("p")));
    }

    #[test]
    fn reaching_ops_before_both_position_and_prev_fails() {
        #[derive(Serialize)]
        struct OpsBeforePrev {
            position: LogPosition,
            ops: Vec<Op>,
            prev: Option<String>,
        }
        #[derive(Serialize)]
        struct OpsFirst {
            ops: Vec<Op>,
            position: LogPosition,
            prev: Option<String>,
        }
        let before_prev = OpsBeforePrev {
            position: LogPosition(7),
            ops: ops(),
            prev: None,
        };
        fails(found(&framed(&before_prev)), "ops before");
        let first = OpsFirst {
            ops: ops(),
            position: LogPosition(7),
            prev: None,
        };
        fails(found(&framed(&first)), "ops before");
    }

    #[test]
    fn reaching_the_end_of_the_map_before_both_fails() {
        #[derive(Serialize)]
        struct NoPrev {
            position: LogPosition,
            answered: Vec<Answered>,
        }
        let body = NoPrev {
            position: LogPosition(7),
            answered: answered(),
        };
        fails(found(&framed(&body)), "map ended");
        // And so does an empty one.
        fails(found(&raw(&[0x80])), "map ended");
    }

    #[test]
    fn a_frame_that_ends_first_fails() {
        let frame = framed(&log(Some("p")));
        for cut in [0, 3, 18, 19, 20, 25] {
            fails(found(&frame[..cut]), "ended");
        }
    }

    #[test]
    fn a_frame_not_of_a_log_of_a_format_this_release_reads_fails() {
        let manifest = format::encode(Kind::Manifest, &log(None)).unwrap();
        fails(found(&manifest), "not a log");

        let mut newer = framed(&log(None));
        newer[5..7].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let mut prefix = Prefix::new(KEY);
        assert!(matches!(
            prefix.feed(&newer),
            Err(StorageError::UnsupportedVersion { found, supported: FORMAT_VERSION, .. }) if found == FORMAT_VERSION + 1
        ),);

        let mut zero = framed(&log(None));
        zero[5..7].copy_from_slice(&0u16.to_le_bytes());
        fails(found(&zero), "version zero");

        let mut magic = framed(&log(None));
        magic[0] ^= 1;
        fails(found(&magic), "not a vellev file");
    }

    #[test]
    fn a_position_is_read_at_every_width_messagepack_gives_it() {
        let u32_max = u64::from(u32::MAX);
        for position in [
            0x7f,
            0x80,
            0xff,
            0x100,
            0xffff,
            0x1_0000,
            u32_max,
            u32_max + 1,
            u64::MAX,
        ] {
            let log = LogSegment {
                position: LogPosition(position),
                ..log(Some("p"))
            };
            assert_eq!(
                found(&framed(&log)).unwrap(),
                found_at(position, Some("p")),
                "{position:#x}"
            );
        }
    }

    /// A body of the entries `position` and `prev` and then `ops`, each key
    /// and `prev`'s text marked as `marked` marks text of its length: as
    /// a writer that does not take the shortest form may write them, which
    /// MessagePack allows.
    fn marked_body(marked: impl Fn(&str) -> Vec<u8>, prev: &str) -> Vec<u8> {
        let mut body = vec![0x83];
        body.extend(marked("position"));
        body.extend([0xcd, 0x01, 0x2c]);
        body.extend(marked("prev"));
        body.extend(marked(prev));
        body.extend(marked("ops"));
        body.push(0x90);
        body
    }

    #[test]
    fn keys_and_prev_are_read_however_long_a_length_their_text_is_given() {
        let str16 = |text: &str| {
            let mut out = vec![0xda];
            out.extend((text.len() as u16).to_be_bytes());
            out.extend(text.as_bytes());
            out
        };
        let str32 = |text: &str| {
            let mut out = vec![0xdb];
            out.extend((text.len() as u32).to_be_bytes());
            out.extend(text.as_bytes());
            out
        };
        let prev = "tenant-a/0/log/00000000000000000299-x.log";
        assert_eq!(
            found(&raw(&marked_body(str16, prev))).unwrap(),
            found_at(300, Some(prev))
        );
        assert_eq!(
            found(&raw(&marked_body(str32, prev))).unwrap(),
            found_at(300, Some(prev))
        );
        // And a long one, which only str16 can say.
        let long = "p".repeat(300);
        assert_eq!(
            found(&raw(&marked_body(str16, &long))).unwrap(),
            found_at(300, Some(&long))
        );
    }

    #[test]
    fn a_byte_messagepack_never_uses_fails_wherever_it_is_skipped() {
        let mut body = vec![0x83, 0xa1, b'x', 0xc1];
        body.extend([0xa8]);
        body.extend(b"position");
        body.push(0x07);
        body.extend([0xa4]);
        body.extend(b"prev");
        body.push(0xc0);
        fails(found(&raw(&body)), "never uses");
    }

    #[test]
    fn a_key_given_twice_fails_as_a_full_read_of_the_log_would() {
        for twice in ["position", "prev"] {
            let mut body = vec![0x83];
            let mut entry = |key: &str, value: &[u8]| {
                body.push(0xa0 | key.len() as u8);
                body.extend(key.as_bytes());
                body.extend(value);
            };
            let value: &[u8] = if twice == "position" {
                &[0x07]
            } else {
                &[0xc0]
            };
            entry(twice, value);
            entry(twice, value);
            let other = if twice == "position" {
                "prev"
            } else {
                "position"
            };
            let other_value: &[u8] = if other == "position" {
                &[0x07]
            } else {
                &[0xc0]
            };
            entry(other, other_value);
            fails(found(&raw(&body)), "twice");
        }
        // As serde's full read does.
        let mut body = vec![0x83, 0xa8];
        body.extend(b"position");
        body.push(0x07);
        body.push(0xa8);
        body.extend(b"position");
        body.push(0x07);
        body.push(0xa3);
        body.extend(b"ops");
        body.push(0x90);
        assert!(rmp_serde::from_slice::<LogSegment>(&body).is_err());
    }

    #[test]
    fn a_body_that_is_not_a_map_fails() {
        // A log encoded with its fields in order and unnamed.
        let positional = rmp_serde::to_vec(&log(None)).unwrap();
        fails(found(&raw(&positional)), "not a map");
    }

    #[test]
    fn position_and_prev_must_be_of_the_kinds_a_log_holds() {
        #[derive(Serialize)]
        struct Wrong<P, Q> {
            position: P,
            prev: Q,
            ops: Vec<Op>,
        }
        let named = |position, prev| {
            framed(&Wrong {
                position,
                prev,
                ops: ops(),
            })
        };
        fails(
            found(&framed(&Wrong {
                position: "seven",
                prev: None::<String>,
                ops: ops(),
            })),
            "position",
        );
        fails(
            found(&framed(&Wrong {
                position: -7i64,
                prev: None::<String>,
                ops: ops(),
            })),
            "position",
        );
        fails(
            found(&framed(&Wrong {
                position: 7u64,
                prev: 6u64,
                ops: ops(),
            })),
            "prev",
        );
        assert_eq!(
            found(&named(7u64, Some("p".to_string()))).unwrap(),
            found_at(7, Some("p"))
        );
    }

    #[test]
    fn neither_found_within_the_most_read_fails() {
        #[derive(Serialize)]
        struct Large {
            large: String,
            position: LogPosition,
            prev: Option<String>,
        }
        let body = Large {
            large: "z".repeat(PREFIX_MOST),
            position: LogPosition(7),
            prev: None,
        };
        fails(found(&framed(&body)), "not found within");
        let fits = Large {
            large: "z".repeat(PREFIX_MOST - 64),
            position: LogPosition(7),
            prev: None,
        };
        assert_eq!(found(&framed(&fits)).unwrap(), found_at(7, None));
    }
}
