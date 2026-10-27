//! The benchmark fixture as the release unpacks it (design r2, §3):
//!
//! ```text
//! DIR/text/passages.jsonl      {id, text, title, url, bucket, owner, deleted}
//! DIR/text/queries.jsonl       {id, text, source, title}, measured
//! DIR/text/warmup.jsonl        the same, never measured
//! DIR/vectors/passages/NNNNN.f32   768 little-endian f32s a row, rows in id order
//! DIR/vectors/queries/00000.f32
//! DIR/vectors/warmup/00000.f32
//! DIR/truth/truth.jsonl        {query, case, ids, scores}, the whole corpus
//! DIR/truth/truth-<N>.jsonl    the same over the first N passages
//! ```
//!
//! Passages are read as they are loaded, a shard at a time, so the
//! harness never holds the corpus while an engine holds it too.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use eyre::{bail, ensure, eyre, Result, WrapErr};
use serde::Deserialize;

pub const DIMENSIONS: usize = 768;

/// The filter cases (§3c), each with `deleted = false` as well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Case {
    /// Every row.
    Broad,
    /// `bucket < 10`.
    Mid,
    /// `bucket < 1`.
    Narrow,
    /// The rows of the owner of the article the query was asked of.
    Owner,
}

impl Case {
    pub const ALL: [Case; 4] = [Case::Broad, Case::Mid, Case::Narrow, Case::Owner];

    pub fn name(self) -> &'static str {
        match self {
            Case::Broad => "broad",
            Case::Mid => "mid",
            Case::Narrow => "narrow",
            Case::Owner => "owner",
        }
    }

    pub fn from_name(name: &str) -> Result<Case> {
        Case::ALL
            .into_iter()
            .find(|case| case.name() == name)
            .ok_or_else(|| eyre!("no filter case {name:?}"))
    }
}

/// A filter as both engines are asked it: `deleted = false`, and the
/// case's own condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    pub bucket_below: Option<i64>,
    pub owner: Option<i64>,
}

impl Filter {
    pub fn of(case: Case, query_owner: i64) -> Filter {
        match case {
            Case::Broad => Filter {
                bucket_below: None,
                owner: None,
            },
            Case::Mid => Filter {
                bucket_below: Some(10),
                owner: None,
            },
            Case::Narrow => Filter {
                bucket_below: Some(1),
                owner: None,
            },
            Case::Owner => Filter {
                bucket_below: None,
                owner: Some(query_owner),
            },
        }
    }
}

/// A passage as loaded into an engine.
#[derive(Debug, Clone)]
pub struct Row {
    pub id: u64,
    pub text: String,
    pub bucket: i64,
    pub owner: i64,
    pub deleted: bool,
    pub vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct PassageLine {
    id: u64,
    text: String,
    title: String,
    bucket: i64,
    owner: i64,
    deleted: bool,
}

#[derive(Debug, Deserialize)]
struct QueryLine {
    id: u64,
    text: String,
    title: String,
}

/// A question, with its vector and the owner of the article it was asked
/// of, for the owner case.
#[derive(Debug, Clone)]
pub struct Query {
    pub id: u64,
    pub text: String,
    pub owner: i64,
    pub vector: Vec<f32>,
}

/// The exact top 100 of each query under each case.
#[derive(Debug, Default)]
pub struct Truth(HashMap<(u64, Case), (Vec<u64>, Vec<f64>)>);

impl Truth {
    pub fn read(path: &Path) -> Result<Truth> {
        #[derive(Deserialize)]
        struct Line {
            query: u64,
            case: String,
            ids: Vec<u64>,
            scores: Vec<f64>,
        }
        let mut truth = HashMap::new();
        for line in lines(path)? {
            let line: Line = serde_json::from_str(&line?)?;
            ensure!(
                line.ids.len() == line.scores.len(),
                "query {}: ids and scores differ in length",
                line.query
            );
            truth.insert(
                (line.query, Case::from_name(&line.case)?),
                (line.ids, line.scores),
            );
        }
        Ok(Truth(truth))
    }

    /// The exact nearest ids, nearest first.
    pub fn of(&self, query: u64, case: Case) -> Result<&[u64]> {
        Ok(self.with_scores(query, case)?.0)
    }

    /// The exact nearest ids with their cosines, nearest first.
    pub fn with_scores(&self, query: u64, case: Case) -> Result<(&[u64], &[f64])> {
        self.0
            .get(&(query, case))
            .map(|(ids, scores)| (ids.as_slice(), scores.as_slice()))
            .ok_or_else(|| eyre!("no truth for query {query} under {}", case.name()))
    }
}

pub struct Fixture {
    dir: PathBuf,
}

fn lines(path: &Path) -> Result<impl Iterator<Item = std::io::Result<String>>> {
    let file = std::fs::File::open(path).wrap_err_with(|| path.display().to_string())?;
    Ok(BufReader::new(file).lines())
}

/// The `.f32` shards of `dir`, in order.
fn shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .wrap_err_with(|| dir.display().to_string())?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()?;
    shards.retain(|path| path.extension().is_some_and(|ext| ext == "f32"));
    shards.sort();
    Ok(shards)
}

fn vectors_of(bytes: &[u8]) -> Result<Vec<Vec<f32>>> {
    ensure!(
        bytes.len().is_multiple_of(DIMENSIONS * 4),
        "{} bytes is no whole number of rows",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(DIMENSIONS * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect()
        })
        .collect())
}

impl Fixture {
    pub fn at(dir: &Path) -> Fixture {
        Fixture {
            dir: dir.to_path_buf(),
        }
    }

    /// The owner of each article, by title, as its passages give it.
    fn owners(&self) -> Result<HashMap<String, i64>> {
        let mut owners = HashMap::new();
        for line in lines(&self.dir.join("text/passages.jsonl"))? {
            let row: PassageLine = serde_json::from_str(&line?)?;
            owners.entry(row.title).or_insert(row.owner);
        }
        Ok(owners)
    }

    /// The measured questions (`queries`) or the warm-up ones (`warmup`).
    pub fn queries(&self, set: &str) -> Result<Vec<Query>> {
        let owners = self.owners()?;
        let vectors: Vec<Vec<f32>> = shards(&self.dir.join("vectors").join(set))?
            .iter()
            .map(|shard| vectors_of(&std::fs::read(shard)?))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let mut queries = Vec::new();
        for (line, vector) in
            lines(&self.dir.join("text").join(format!("{set}.jsonl")))?.zip(vectors)
        {
            let line: QueryLine = serde_json::from_str(&line?)?;
            let owner = *owners
                .get(&line.title)
                .ok_or_else(|| eyre!("query {}: no passage of {:?}", line.id, line.title))?;
            queries.push(Query {
                id: line.id,
                text: line.text,
                owner,
                vector,
            });
        }
        Ok(queries)
    }

    /// The first `rows` passages, in batches of `batch`, each read when it
    /// is wanted.
    pub fn passages(&self, rows: u64, batch: usize) -> Result<Passages> {
        Ok(Passages {
            text: Box::new(lines(&self.dir.join("text/passages.jsonl"))?),
            shards: shards(&self.dir.join("vectors/passages"))?.into_iter(),
            vectors: Vec::new().into_iter(),
            left: rows,
            batch,
            next: 0,
        })
    }
}

/// Passages in batches, their text and vectors read as they go.
pub struct Passages {
    text: Box<dyn Iterator<Item = std::io::Result<String>> + Send>,
    shards: std::vec::IntoIter<PathBuf>,
    vectors: std::vec::IntoIter<Vec<f32>>,
    left: u64,
    batch: usize,
    next: u64,
}

impl Passages {
    fn vector(&mut self) -> Result<Vec<f32>> {
        loop {
            if let Some(vector) = self.vectors.next() {
                return Ok(vector);
            }
            let Some(shard) = self.shards.next() else {
                bail!("the vectors end before passage {}", self.next);
            };
            let bytes = std::fs::read(&shard).wrap_err_with(|| shard.display().to_string())?;
            self.vectors = vectors_of(&bytes)?.into_iter();
        }
    }

    fn row(&mut self) -> Result<Row> {
        let line = self
            .text
            .next()
            .ok_or_else(|| eyre!("the passages end before {}", self.next))??;
        let line: PassageLine = serde_json::from_str(&line)?;
        ensure!(line.id == self.next, "passage {} out of order", line.id);
        self.next += 1;
        Ok(Row {
            id: line.id,
            text: line.text,
            bucket: line.bucket,
            owner: line.owner,
            deleted: line.deleted,
            vector: self.vector()?,
        })
    }
}

impl Iterator for Passages {
    type Item = Result<Vec<Row>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            return None;
        }
        let take = (self.batch as u64).min(self.left);
        self.left -= take;
        Some((0..take).map(|_| self.row()).collect())
    }
}

/// A row id as both engines store it: a UUID whose byte order is the id's,
/// so that ties ordered by id are ordered as the truth orders them.
pub fn uuid_of(id: u64) -> uuid::Uuid {
    uuid::Uuid::from_u128(u128::from(id))
}

pub fn id_of(uuid: &str) -> Result<u64> {
    let uuid: uuid::Uuid = uuid.parse().wrap_err_with(|| uuid.to_string())?;
    u64::try_from(uuid.as_u128()).map_err(|_| eyre!("{uuid} is no row id"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_ids_keep_their_order_as_uuids() {
        let ids = [0u64, 1, 255, 256, 99_999, 1_000_000];
        let mut uuids: Vec<String> = ids.iter().map(|&id| uuid_of(id).to_string()).collect();
        let sorted = uuids.clone();
        uuids.sort();
        assert_eq!(uuids, sorted);
        for id in ids {
            assert_eq!(id_of(&uuid_of(id).to_string()).unwrap(), id);
        }
    }

    #[test]
    fn each_case_filters_as_the_truth_does() {
        assert_eq!(
            Filter::of(Case::Broad, 3),
            Filter {
                bucket_below: None,
                owner: None
            }
        );
        assert_eq!(Filter::of(Case::Mid, 3).bucket_below, Some(10));
        assert_eq!(Filter::of(Case::Narrow, 3).bucket_below, Some(1));
        assert_eq!(
            Filter::of(Case::Owner, 3),
            Filter {
                bucket_below: None,
                owner: Some(3)
            }
        );
    }
}
