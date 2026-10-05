use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::future::try_join_all;
use tokio::sync::Mutex;

use crate::blobs::{Blobs, Layout};
use crate::catalog::Mutation;
use crate::coordinator::{
    Commit, Committed, Coordinator, CoordinatorError, Lease, LogPosition, ManifestRef, PartitionKey,
};
use crate::format::{decode, decode_head, encode, Head, Kind, LogSegment, Manifest, SegmentFile};
use crate::state::State;
use crate::{Result, StorageError};

/// How many times a commit whose outcome is unknown is repeated before the
/// writer gives up. Repeating is safe because the coordinator accepts an
/// identical commit twice.
const COMMIT_ATTEMPTS: usize = 3;

/// A partition's committed data, readable on any node.
#[derive(Debug)]
pub struct Partition {
    key: PartitionKey,
    blobs: Arc<dyn Blobs>,
    state: RwLock<State>,
}

/// The committed commits between a state and a newer head, oldest first, and
/// the checkpoint they start from when the chain reaches one before it
/// reaches the state.
struct Tail {
    checkpoint: Option<(String, Manifest)>,
    logs: Vec<(String, LogSegment)>,
}

impl Partition {
    /// Load the partition exactly as committed.
    pub async fn open(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
    ) -> Result<Self> {
        let partition = Self {
            key,
            blobs,
            state: RwLock::new(State::default()),
        };
        partition.catch_up(committed).await?;
        Ok(partition)
    }

    pub fn key(&self) -> &PartitionKey {
        &self.key
    }

    /// Run `f` against a consistent view of the partition.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.read().expect("partition lock"))
    }

    /// Bring this copy up to `committed`, reading only files reachable from
    /// its head. Returns the position reached.
    pub async fn catch_up(&self, committed: &Committed) -> Result<LogPosition> {
        let (position, head) =
            self.read(|state| (state.position(), state.head().map(String::from)));
        let target = committed.manifest.as_ref().map(|head| head.0.as_str());
        if target == head.as_deref() && committed.log_position == position {
            return Ok(position);
        }
        let tail = self.tail(target, head.as_deref()).await?;
        let mut state = match tail.checkpoint {
            Some((key, manifest)) => self.restore(&key, manifest).await?,
            None => self.read(State::clone),
        };
        for (key, log) in &tail.logs {
            state.apply(key, log)?;
        }
        if state.position() != committed.log_position {
            return Err(StorageError::corrupt(
                target.unwrap_or("the empty head"),
                format!(
                    "head is at {:?}, and the coordinator committed {:?}",
                    state.position(),
                    committed.log_position
                ),
            ));
        }
        *self.state.write().expect("partition lock") = state;
        Ok(committed.log_position)
    }

    /// Walk back from `target` until reaching `known`, a checkpoint, or the
    /// start of the partition's history.
    async fn tail(&self, target: Option<&str>, known: Option<&str>) -> Result<Tail> {
        let mut logs = Vec::new();
        let mut cursor = target.map(String::from);
        let checkpoint = loop {
            match cursor {
                None if known.is_none() => break None,
                None => {
                    let lost = known.unwrap_or_default();
                    return Err(StorageError::corrupt(
                        lost,
                        "the committed history does not include it",
                    ));
                }
                Some(key) if Some(key.as_str()) == known => break None,
                Some(key) => match decode_head(&key, &self.blobs.get(&key).await?)? {
                    Head::Manifest(manifest) => break Some((key, manifest)),
                    Head::Log(log) => {
                        cursor = log.prev.clone();
                        logs.push((key, log));
                    }
                },
            }
        };
        logs.reverse();
        Ok(Tail { checkpoint, logs })
    }

    async fn restore(&self, key: &str, manifest: Manifest) -> Result<State> {
        let loaded: HashMap<String, Arc<SegmentFile>> = self.read(|state| {
            state
                .segment_files()
                .map(|(key, file)| (key.to_string(), file.clone()))
                .collect()
        });
        let files = try_join_all(manifest.segments.iter().map(|meta| {
            let loaded = loaded.get(&meta.key).cloned();
            async move {
                match loaded {
                    Some(file) => Ok(file),
                    None => decode(&meta.key, Kind::Segment, &self.blobs.get(&meta.key).await?)
                        .map(Arc::new),
                }
            }
        }))
        .await?;
        Ok(State::from_manifest(key, manifest, files))
    }
}

/// The one node allowed to change a partition, for as long as its lease's
/// generation is current. Nothing it writes is visible, here or anywhere,
/// until the coordinator has accepted the commit.
#[derive(Debug)]
pub struct Writer {
    partition: Arc<Partition>,
    coordinator: Arc<dyn Coordinator>,
    lease: Lease,
    open: Mutex<bool>,
}

impl Writer {
    /// Start writing from the state the lease was taken at.
    pub async fn open(
        blobs: Arc<dyn Blobs>,
        coordinator: Arc<dyn Coordinator>,
        lease: Lease,
        committed: &Committed,
    ) -> Result<Self> {
        let partition = Arc::new(Partition::open(lease.key.clone(), blobs, committed).await?);
        Ok(Self {
            partition,
            coordinator,
            lease,
            open: Mutex::new(true),
        })
    }

    pub fn partition(&self) -> &Arc<Partition> {
        &self.partition
    }

    pub fn lease(&self) -> &Lease {
        &self.lease
    }

    /// Commit `mutations` as one atomic step and return its position. A
    /// rejected mutation fails the whole batch and changes nothing.
    pub async fn commit(&self, mutations: Vec<Mutation>) -> Result<LogPosition> {
        let mut open = self.open.lock().await;
        let (position, prev, mut catalog) = self.partition.read(|state| {
            (
                state.position(),
                state.head().map(String::from),
                state.catalog().clone(),
            )
        });
        let ops = mutations
            .into_iter()
            .filter_map(|mutation| catalog.resolve(mutation).transpose())
            .collect::<Result<Vec<_>>>()?;
        if ops.is_empty() {
            return Ok(position);
        }
        let log = LogSegment {
            position: LogPosition(position.0 + 1),
            prev,
            ops,
        };
        let key = Layout(&self.lease.key).log(log.position, self.lease.generation);
        self.publish(&mut open, &key, encode(Kind::Log, &log)?, log.position)
            .await?;
        self.partition
            .state
            .write()
            .expect("partition lock")
            .apply(&key, &log)?;
        Ok(log.position)
    }

    /// Write a checkpoint: move the rows held in memory into a segment and
    /// commit a manifest, so recovery no longer needs the log before it.
    pub async fn flush(&self) -> Result<LogPosition> {
        let mut open = self.open.lock().await;
        let (position, unflushed) = self
            .partition
            .read(|state| (state.position(), state.has_unflushed_writes()));
        if !unflushed {
            return Ok(position);
        }
        let position = LogPosition(position.0 + 1);
        let layout = Layout(&self.lease.key);
        let segment_key = layout.segment(position, self.lease.generation);
        let (segment, manifest) = self
            .partition
            .read(|state| state.checkpoint(position, &segment_key));
        if let Some(segment) = &segment {
            self.put(&mut open, &segment_key, encode(Kind::Segment, segment)?)
                .await?;
        }
        let key = layout.manifest(position, self.lease.generation);
        self.publish(
            &mut open,
            &key,
            encode(Kind::Manifest, &manifest)?,
            position,
        )
        .await?;
        self.partition
            .state
            .write()
            .expect("partition lock")
            .checkpointed(&key, &manifest, segment);
        Ok(position)
    }

    async fn put(&self, open: &mut bool, key: &str, frame: bytes::Bytes) -> Result<()> {
        if !*open {
            return Err(StorageError::Closed);
        }
        self.partition.blobs.put(key, frame).await
    }

    /// Upload a head file and commit it. Once the upload has happened, any
    /// failure to commit leaves the outcome unknown, so the writer closes.
    async fn publish(
        &self,
        open: &mut bool,
        key: &str,
        frame: bytes::Bytes,
        position: LogPosition,
    ) -> Result<()> {
        self.put(open, key, frame).await?;
        let commit = Commit {
            log_position: position,
            manifest: Some(ManifestRef(key.to_string())),
        };
        let mut outcome = self.coordinator.commit(&self.lease, commit.clone()).await;
        for _ in 1..COMMIT_ATTEMPTS {
            if !matches!(&outcome, Err(error) if error.is_retryable()) {
                break;
            }
            outcome = self.coordinator.commit(&self.lease, commit.clone()).await;
        }
        outcome.map(|_| ()).map_err(|error: CoordinatorError| {
            *open = false;
            error.into()
        })
    }
}
