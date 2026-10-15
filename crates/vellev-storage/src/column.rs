//! A segment's vectors as one column: every row's vector end to end in a
//! file of its own, read where it lies on local disk rather than held in
//! memory, so that a node's memory is spent on codes and not on the
//! vectors the codes stand for.
//!
//! The file's body is a byte of padding, so that the values are aligned
//! for reading in place, then every vector's values as little-endian
//! floats, then where each row's vector starts, as one offset per row and
//! a last one past the end, and finally the row count.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use bytes::Bytes;

use crate::format::{self, framed, Kind, Row, BODY, FORMAT_VERSION};
use crate::{Result, StorageError};

const _: () = assert!(
    cfg!(target_endian = "little"),
    "vectors are read in place as little-endian"
);
const PADDING: usize = (4 - BODY % 4) % 4;

#[derive(Debug)]
pub(crate) struct Column {
    values: Values,
    /// Where each row's vector starts, in values, and one past the last.
    offsets: Vec<u64>,
}

#[derive(Debug)]
enum Values {
    Memory(Vec<f32>),
    Mapped(memmap2::Mmap, Range<usize>),
}

impl Column {
    /// The vectors of `rows`, which are left without them.
    pub fn taken_from(rows: &mut [Row]) -> Self {
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        for row in rows {
            if let Some(vector) = row.object.vector.take() {
                values.extend_from_slice(&vector);
            }
            offsets.push(values.len() as u64);
        }
        Self {
            values: Values::Memory(values),
            offsets,
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// About how many bytes the column holds in memory: its vectors if
    /// they are not read in place from a local file, and where each starts.
    pub fn resident(&self) -> usize {
        let values = match &self.values {
            Values::Memory(values) => values.len() * 4,
            Values::Mapped(..) => 0,
        };
        values + self.offsets.len() * 8
    }

    pub fn get(&self, row: usize) -> Option<&[f32]> {
        let (start, end) = (
            *self.offsets.get(row)? as usize,
            *self.offsets.get(row + 1)? as usize,
        );
        (start < end).then(|| &self.values()[start..end])
    }

    fn values(&self) -> &[f32] {
        match &self.values {
            Values::Memory(values) => values,
            Values::Mapped(map, range) => bytemuck::cast_slice(&map[range.clone()]),
        }
    }

    pub fn encode(&self) -> Bytes {
        let values = self.values();
        let length = PADDING + values.len() * 4 + self.offsets.len() * 8 + 8;
        let frame = framed(Kind::Vectors, FORMAT_VERSION, length, |frame| {
            frame.extend_from_slice(&[0; PADDING]);
            frame.extend_from_slice(bytemuck::cast_slice(values));
            for offset in &self.offsets {
                frame.extend_from_slice(&offset.to_le_bytes());
            }
            frame.extend_from_slice(&(self.len() as u64).to_le_bytes());
        });
        Bytes::from(frame)
    }

    /// The column held under `key` as `frame`, copied into memory.
    pub fn decode(key: &str, frame: &[u8]) -> Result<Self> {
        let (kind, body) = format::open(key, frame)?;
        let (values, offsets) = Self::parts(key, kind, body)?;
        let values = values
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect();
        Ok(Self {
            values: Values::Memory(values),
            offsets,
        })
    }

    /// The column in the file at `path`, holding `key`'s frame, read in
    /// place: laid out as a column is, which reads only its ends, and not
    /// yet seen to be whole, which [`Unsummed::summed`] sees by reading
    /// all of it. The file must never be written in place while the column
    /// is open: the store's cache only ever replaces an entry by renaming
    /// a whole new file over it or unlinks it, either of which leaves a
    /// mapping on the old file untouched. That holds on Unix alone, so
    /// elsewhere a column is never mapped.
    pub fn map(key: &str, path: &Path) -> Result<Unsummed> {
        if !cfg!(unix) {
            return Err(StorageError::corrupt(key, "read in place only on Unix"));
        }
        let io = |error: std::io::Error| StorageError::corrupt(key, error.to_string());
        let file = std::fs::File::open(path).map_err(io)?;
        // SAFETY: the file is a cache entry, replaced only by rename or
        // unlink and never written in place, so the mapping stays valid.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(io)?;
        let (kind, body, _) = format::open_unsummed(key, &map)?;
        let (values, offsets) = Self::parts(key, kind, body)?;
        let start = values.as_ptr() as usize - map.as_ptr() as usize;
        let range = start..start + values.len();
        if !start.is_multiple_of(4) {
            return Err(StorageError::corrupt(key, "values are not aligned"));
        }
        Ok(Unsummed(Self {
            values: Values::Mapped(map, range),
            offsets,
        }))
    }

    /// The values' bytes and the offsets of a frame's body.
    fn parts<'a>(key: &str, kind: Kind, body: &'a [u8]) -> Result<(&'a [u8], Vec<u64>)> {
        let corrupt = |reason: &str| StorageError::corrupt(key, reason);
        if kind != Kind::Vectors {
            return Err(corrupt("not a vectors file"));
        }
        let (rest, count) = body
            .split_last_chunk::<8>()
            .ok_or_else(|| corrupt("shorter than a row count"))?;
        let rows = usize::try_from(u64::from_le_bytes(*count))
            .ok()
            .and_then(|rows| rows.checked_add(1))
            .ok_or_else(|| corrupt("row count too large"))?;
        let table = rows
            .checked_mul(8)
            .ok_or_else(|| corrupt("row count too large"))?;
        let split = rest
            .len()
            .checked_sub(table)
            .ok_or_else(|| corrupt("shorter than its offsets"))?;
        let (values, table) = rest.split_at(split);
        let offsets: Vec<u64> = table
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
            .collect();
        let values = values
            .get(PADDING..)
            .ok_or_else(|| corrupt("shorter than its padding"))?;
        let ascending = offsets.windows(2).all(|pair| pair[0] <= pair[1]);
        if !ascending || offsets.last().copied() != Some(values.len() as u64 / 4) {
            return Err(corrupt("offsets do not fit the values"));
        }
        if !values.len().is_multiple_of(4) {
            return Err(corrupt("values are not whole floats"));
        }
        Ok((values, offsets))
    }
}

/// A file as it lay when it was looked at: where, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Seen {
    path: PathBuf,
    lay: Lay,
}

/// How a file lay: how large, when it was last written and, where the
/// system tells, which file of its disk it was and when anything of it
/// last changed. A cache entry is never written in place, only replaced
/// or removed, so one that is the same in all of these is the same file,
/// holding what it held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lay {
    size: u64,
    written: std::time::SystemTime,
    /// The disk and the file's number on it.
    file: Option<(u64, u64)>,
    /// When its contents or anything said of it last changed, in seconds
    /// and nanoseconds.
    changed: Option<(i64, i64)>,
}

impl Seen {
    /// Whether `other` is the very file this is, under whatever name:
    /// the same file of the same disk, where the system tells which.
    pub(crate) fn same_file(&self, other: &Seen) -> bool {
        self.lay.file.is_some() && self.lay.file == other.lay.file
    }
}

/// The file at `path` as it lies now.
pub(crate) fn seen(path: &Path) -> std::io::Result<Seen> {
    let meta = std::fs::metadata(path)?;
    #[cfg(unix)]
    let (file, changed) = {
        use std::os::unix::fs::MetadataExt;
        (
            Some((meta.dev(), meta.ino())),
            Some((meta.ctime(), meta.ctime_nsec())),
        )
    };
    #[cfg(not(unix))]
    let (file, changed) = (None, None);
    Ok(Seen {
        path: path.to_path_buf(),
        lay: Lay {
            size: meta.len(),
            written: meta.modified()?,
            file,
            changed,
        },
    })
}

/// How many files are remembered as summed at the most: more than a node
/// has on its disk at once in any use foreseen, and few enough that
/// remembering them all costs it a few mebibytes.
const SUMMED_KEPT: usize = 65_536;

/// The vectors files this process has summed, each as it lay then. One is
/// not summed again while it lies as it did: summing a tenant's files
/// takes as long as reading every one of them through, and a load that
/// failed part way, or a tenant opened again, would otherwise begin from
/// nothing each time and might never end.
///
/// A file is forgotten when its copy is removed or replaced, and no more
/// than `most` are remembered: past that the one remembered longest ago
/// is forgotten, which costs it one more summing and nothing else.
#[derive(Debug)]
struct Summed {
    most: usize,
    /// Each file with when it was remembered, as a number that grows.
    files: HashMap<PathBuf, (u64, Lay)>,
    /// The same files, by when each was remembered.
    order: std::collections::BTreeMap<u64, PathBuf>,
    remembered: u64,
}

impl Summed {
    fn keeping(most: usize) -> Self {
        Self {
            most,
            files: HashMap::new(),
            order: Default::default(),
            remembered: 0,
        }
    }

    /// Whether `file` was summed as it lies now.
    fn holds(&self, file: &Seen) -> bool {
        self.files
            .get(&file.path)
            .is_some_and(|(_, lay)| *lay == file.lay)
    }

    fn remember(&mut self, file: Seen) {
        self.forget(&file.path);
        while self.files.len() >= self.most {
            let Some((_, longest_ago)) = self.order.pop_first() else {
                return;
            };
            self.files.remove(&longest_ago);
        }
        self.remembered += 1;
        self.order.insert(self.remembered, file.path.clone());
        self.files.insert(file.path, (self.remembered, file.lay));
    }

    fn forget(&mut self, path: &Path) {
        if let Some((when, _)) = self.files.remove(path) {
            self.order.remove(&when);
        }
    }

    /// Forget every file in the directory `dir` or below it.
    fn forget_under(&mut self, dir: &Path) {
        let Self { files, order, .. } = self;
        files.retain(|path, (when, _)| {
            let kept = !path.starts_with(dir);
            if !kept {
                order.remove(when);
            }
            kept
        });
    }
}

fn summed() -> std::sync::MutexGuard<'static, Summed> {
    static SUMMED: std::sync::OnceLock<std::sync::Mutex<Summed>> = std::sync::OnceLock::new();
    SUMMED
        .get_or_init(|| std::sync::Mutex::new(Summed::keeping(SUMMED_KEPT)))
        .lock()
        .expect("summed lock")
}

/// Whether this process has summed `file` as it lies now.
pub(crate) fn summed_before(file: &Seen) -> bool {
    summed().holds(file)
}

/// Remember that `file` was summed as it lies: it is another name of a
/// file this process summed, and proved the same.
pub(crate) fn remember_summed(file: Seen) {
    summed().remember(file);
}

/// Remember no longer that the file at `path` was summed: its copy is
/// removed or replaced.
pub(crate) fn forget_summed(path: &Path) {
    summed().forget(path);
}

/// As [`forget_summed`], for every file in the directory `dir` or below
/// it.
pub(crate) fn forget_summed_under(dir: &Path) {
    summed().forget_under(dir);
}

/// A column read in place whose file has not been summed: nothing of it
/// may be used until it has, here or before.
#[derive(Debug)]
pub(crate) struct Unsummed(Column);

impl Unsummed {
    /// The column of a file this process summed before as it lies now
    /// (see [`summed_before`]), which need not be summed again.
    pub fn trusted_as_summed(self) -> Column {
        self.0
    }

    /// As [`Self::summed`], remembering the file as it lay when it was
    /// opened to be summed.
    pub fn summed_as(self, key: &str, file: Seen, told: impl FnMut(usize)) -> Result<Column> {
        let column = self.summed(key, told)?;
        summed().remember(file);
        Ok(column)
    }

    /// The column, once its whole file is seen to sum to what it says,
    /// telling `summed` of each piece of the file as it is read.
    pub fn summed(self, key: &str, summed: impl FnMut(usize)) -> Result<Column> {
        let Values::Mapped(map, _) = &self.0.values else {
            return Ok(self.0);
        };
        let (_, body, sum) = format::open_unsummed(key, map)?;
        match sum.holds(body, summed) {
            true => Ok(self.0),
            false => Err(StorageError::corrupt(key, "checksum mismatch")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::CollectionId;
    use vellev_core::{Object, ObjectId};

    fn row(vector: Option<Vec<f32>>) -> Row {
        Row {
            collection: CollectionId(1),
            object: Object {
                id: ObjectId::generate(),
                properties: Default::default(),
                vector,
            },
        }
    }

    #[test]
    fn a_column_round_trips_in_memory_and_from_a_file() {
        let mut rows = vec![
            row(Some(vec![1.0, 2.0])),
            row(None),
            row(Some(vec![3.0, 4.0, 5.0])),
            row(Some(vec![])),
        ];
        let column = Column::taken_from(&mut rows);
        assert!(rows.iter().all(|row| row.object.vector.is_none()));
        assert_eq!(column.len(), 4);
        let frame = column.encode();
        let decoded = Column::decode("k", &frame).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        std::fs::write(&path, &frame).unwrap();
        let mapped = Column::map("k", &path).unwrap();
        let mut read = 0;
        let mapped = mapped.summed("k", |bytes| read += bytes).unwrap();
        assert_eq!(read, frame.len() - BODY, "all of its body");
        for copy in [&column, &decoded, &mapped] {
            assert_eq!(copy.get(0), Some(&[1.0, 2.0][..]));
            assert_eq!(copy.get(1), None);
            assert_eq!(copy.get(2), Some(&[3.0, 4.0, 5.0][..]));
            assert_eq!(copy.get(3), None);
            assert_eq!(copy.get(4), None);
        }
        assert_eq!(Column::taken_from(&mut []).len(), 0);
    }

    fn lying(path: &str, size: u64) -> Seen {
        Seen {
            path: PathBuf::from(path),
            lay: Lay {
                size,
                written: std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000),
                file: Some((1, 7)),
                changed: Some((1_000, 5)),
            },
        }
    }

    #[test]
    fn a_file_is_known_as_summed_only_while_nothing_of_how_it_lies_has_changed() {
        let mut summed = Summed::keeping(8);
        let file = lying("cache/a/1.vectors", 4_096);
        assert!(!summed.holds(&file));
        summed.remember(file.clone());
        assert!(summed.holds(&file));

        // The same in all but one thing, whichever it is, it is another
        // file or the same one changed.
        let changed = |change: fn(&mut Lay)| {
            let mut other = file.clone();
            change(&mut other.lay);
            other
        };
        const LATER: std::time::Duration = std::time::Duration::from_nanos(1);
        for (what, other) in [
            ("its size", changed(|lay| lay.size += 1)),
            ("when it was written", changed(|lay| lay.written += LATER)),
            ("which file it is", changed(|lay| lay.file = Some((1, 8)))),
            (
                "which disk it is on",
                changed(|lay| lay.file = Some((2, 7))),
            ),
            (
                "when it changed",
                changed(|lay| lay.changed = Some((1_000, 6))),
            ),
            ("what the system tells", changed(|lay| lay.file = None)),
        ] {
            assert!(!summed.holds(&other), "{what}");
        }
        assert!(!summed.holds(&lying("cache/a/2.vectors", 4_096)));

        // Summed again as it lies now, it is known as that and no longer
        // as it lay.
        let grown = changed(|lay| lay.size += 1);
        summed.remember(grown.clone());
        assert!(summed.holds(&grown) && !summed.holds(&file));
        assert_eq!((summed.files.len(), summed.order.len()), (1, 1));
    }

    #[test]
    fn files_summed_are_forgotten_with_their_copies_and_no_more_are_remembered_than_may_be() {
        let mut summed = Summed::keeping(3);
        let files: Vec<Seen> = ["cache/a/0/1", "cache/a/0/2", "cache/ab/0/1", "cache/b/0/1"]
            .iter()
            .map(|path| lying(path, 1))
            .collect();
        // Past as many as are kept, the one remembered longest ago goes.
        for file in &files {
            summed.remember(file.clone());
        }
        let held = |summed: &Summed| -> Vec<bool> {
            files.iter().map(|file| summed.holds(file)).collect()
        };
        assert_eq!(held(&summed), [false, true, true, true]);
        // Remembered again, a file is the one remembered last.
        summed.remember(files[1].clone());
        summed.remember(files[0].clone());
        assert_eq!(held(&summed), [true, true, false, true]);

        // One removed is forgotten, and so is all below a directory
        // removed: that directory, and none whose name only begins as its
        // name does.
        summed.forget(Path::new("cache/b/0/1"));
        summed.forget(Path::new("cache/never/there"));
        assert_eq!(held(&summed), [true, true, false, false]);
        summed.remember(files[2].clone());
        summed.forget_under(Path::new("cache/a"));
        assert_eq!(held(&summed), [false, false, true, false]);
        assert_eq!((summed.files.len(), summed.order.len()), (1, 1));
        summed.forget_under(Path::new("cache"));
        assert!(summed.files.is_empty() && summed.order.is_empty());

        // However many are summed, no more are kept.
        for n in 0..1_000 {
            summed.remember(lying(&format!("cache/c/{n}"), n));
            assert!(summed.files.len() <= 3 && summed.order.len() == summed.files.len());
        }
        assert!(summed.holds(&lying("cache/c/999", 999)));
    }

    #[test]
    fn a_file_on_disk_is_seen_as_it_lies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        std::fs::write(&path, [0u8; 10]).unwrap();
        let first = seen(&path).unwrap();
        assert_eq!((first.path.as_path(), first.lay.size), (path.as_path(), 10));
        assert_eq!(seen(&path).unwrap(), first, "as long as it lies as it did");
        #[cfg(unix)]
        assert!(first.lay.file.is_some() && first.lay.changed.is_some());
        // Replaced by another of the same size, as a cache entry is, it
        // is another file.
        let other = dir.path().join("other");
        std::fs::write(&other, [1u8; 10]).unwrap();
        std::fs::rename(&other, &path).unwrap();
        #[cfg(unix)]
        assert_ne!(seen(&path).unwrap().lay.file, first.lay.file);
        assert!(seen(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn offsets_that_do_not_fit_the_values_are_refused() {
        let reframed = |offsets: &[u64], rows: u64| {
            let values = [1.0f32, 2.0, 3.0];
            let length = PADDING + values.len() * 4 + offsets.len() * 8 + 8;
            let frame = framed(Kind::Vectors, FORMAT_VERSION, length, |frame| {
                frame.extend_from_slice(&[0; PADDING]);
                frame.extend_from_slice(bytemuck::cast_slice(&values));
                offsets
                    .iter()
                    .for_each(|offset| frame.extend_from_slice(&offset.to_le_bytes()));
                frame.extend_from_slice(&rows.to_le_bytes());
            });
            Column::decode("k", &frame)
        };
        assert!(reframed(&[0, 2, 3], 2).is_ok());
        assert!(reframed(&[0, 3, 2], 2).is_err(), "not ascending");
        assert!(reframed(&[0, 2, 4], 2).is_err(), "past the values");
        assert!(
            reframed(&[0, 2, 3], u64::MAX).is_err(),
            "more rows than offsets"
        );
        assert!(reframed(&[0, 2, 3], 1).is_err(), "fewer rows than offsets");
    }

    #[test]
    fn a_column_read_in_place_whose_body_is_damaged_is_found_out_as_it_is_summed() {
        let column = Column::taken_from(&mut [row(Some(vec![1.0, 2.0])), row(Some(vec![3.0]))]);
        let mut frame = column.encode().to_vec();
        // A value changed, under a header and offsets left as they were.
        frame[BODY + PADDING] ^= 1;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        std::fs::write(&path, &frame).unwrap();
        let mapped = Column::map("k", &path).expect("laid out as a column is");
        let damaged = mapped.summed("k", |_| {}).unwrap_err();
        assert!(damaged.to_string().contains("checksum"), "{damaged}");
    }

    #[test]
    fn a_damaged_column_is_refused() {
        let column = Column::taken_from(&mut [row(Some(vec![1.0, 2.0])), row(None)]);
        let frame = column.encode();
        let mut flipped = frame.to_vec();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(Column::decode("k", &flipped).is_err());
        let mut short = frame.to_vec();
        short.truncate(BODY + 3);
        assert!(Column::decode("k", &short).is_err());
    }
}
