//! A segment's vectors as one column: every row's vector end to end in a
//! file of its own, read where it lies on local disk rather than held in
//! memory, so that a node's memory is spent on codes and not on the
//! vectors the codes stand for.
//!
//! The file's body is a byte of padding, so that the values are aligned
//! for reading in place, then every vector's values as little-endian
//! floats, then where each row's vector starts, as one offset per row and
//! a last one past the end, and finally the row count.

use std::ops::Range;
use std::path::Path;

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
        let (values, offsets) = Self::parts(key, frame)?;
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
    /// place. The file must not change while the column is open; the
    /// store's cache writes a file once and never again.
    pub fn map(key: &str, path: &Path) -> Result<Self> {
        let io = |error: std::io::Error| StorageError::corrupt(key, error.to_string());
        let file = std::fs::File::open(path).map_err(io)?;
        // SAFETY: the mapped file is a cache entry, written whole under
        // another name and renamed into place, and never written again.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(io)?;
        let (values, offsets) = Self::parts(key, &map)?;
        let start = values.as_ptr() as usize - map.as_ptr() as usize;
        let range = start..start + values.len();
        if !start.is_multiple_of(4) {
            return Err(StorageError::corrupt(key, "values are not aligned"));
        }
        Ok(Self {
            values: Values::Mapped(map, range),
            offsets,
        })
    }

    /// The values' bytes and the offsets of a frame.
    fn parts<'a>(key: &str, frame: &'a [u8]) -> Result<(&'a [u8], Vec<u64>)> {
        let corrupt = |reason: &str| StorageError::corrupt(key, reason);
        let (kind, body) = format::open(key, frame)?;
        if kind != Kind::Vectors {
            return Err(corrupt("not a vectors file"));
        }
        let (rest, count) = body
            .split_last_chunk::<8>()
            .ok_or_else(|| corrupt("shorter than a row count"))?;
        let rows = u64::from_le_bytes(*count) as usize;
        let table = (rows + 1) * 8;
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
        for copy in [&column, &decoded, &mapped] {
            assert_eq!(copy.get(0), Some(&[1.0, 2.0][..]));
            assert_eq!(copy.get(1), None);
            assert_eq!(copy.get(2), Some(&[3.0, 4.0, 5.0][..]));
            assert_eq!(copy.get(3), None);
            assert_eq!(copy.get(4), None);
        }
        assert_eq!(Column::taken_from(&mut []).len(), 0);
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
