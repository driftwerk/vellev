use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::try_join_all;
use tokio::sync::Mutex;

use crate::blobs::{Blobs, Layout};
use crate::catalog::Mutation;
use crate::coordinator::{
    Commit, Committed, Coordinator, Generation, Lease, LogPosition, ManifestRef, PartitionKey,
};
use crate::format::{decode, decode_head, encode, Head, Kind, LogSegment, Manifest, SegmentFile};
use crate::state::State;
use crate::{Result, StorageError};

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;
const COMMIT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// A partition's committed data in memory. It only ever moves forward, and
/// only through its [`Follower`] or [`Writer`].
#[derive(Debug)]
pub struct Partition {
    key: PartitionKey,
    blobs: Arc<dyn Blobs>,
    state: RwLock<State>,
    /// Serialises catch-up, and remembers the newest coordinator generation
    /// seen so an older reading of the committed state is recognised as stale.
    seen: Mutex<Generation>,
}

/// The committed commits between a state and a newer head, oldest first, and
/// the checkpoint they start from when the chain reaches one before it
/// reaches the state.
struct Tail {
    checkpoint: Option<(String, Manifest)>,
    logs: Vec<(String, LogSegment)>,
}

impl Partition {
    pub fn key(&self) -> &PartitionKey {
        &self.key
    }

    /// Run `f` against a consistent view of the partition.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.read().expect("partition lock"))
    }

    async fn load(key: PartitionKey, blobs: Arc<dyn Blobs>, committed: &Committed) -> Result<Self> {
        let partition = Self {
            key,
            blobs,
            state: RwLock::new(State::default()),
            seen: Mutex::new(Generation(0)),
        };
        partition.advance(committed).await?;
        Ok(partition)
    }

    /// A file named by data rather than built by [`Layout`]: it must belong
    /// to this partition before it is fetched.
    async fn fetch(&self, key: &str) -> Result<Bytes> {
        if !key.starts_with(&Layout(&self.key).prefix()) {
            return Err(StorageError::corrupt(
                key,
                "the key belongs to another partition",
            ));
        }
        self.blobs.get(key).await
    }

    async fn advance(&self, committed: &Committed) -> Result<LogPosition> {
        let mut seen = self.seen.lock().await;
        let (position, head) =
            self.read(|state| (state.position(), state.head().map(String::from)));
        let target = committed.manifest.as_ref().map(|head| head.0.as_str());
        let stale = committed.generation <= *seen && committed.log_position < position;
        if stale || (committed.log_position == position && target == head.as_deref()) {
            return Ok(position);
        }
        if committed.log_position <= position {
            return Err(StorageError::Diverged { head });
        }
        let tail = self.tail(committed, position, head.as_deref()).await?;
        match tail.checkpoint {
            Some((key, manifest)) => {
                let mut state = self.restore(&key, manifest).await?;
                tail.logs
                    .iter()
                    .try_for_each(|(key, log)| state.apply(key, log))?;
                *self.state.write().expect("partition lock") = state;
            }
            None => {
                let mut state = self.state.write().expect("partition lock");
                tail.logs
                    .iter()
                    .try_for_each(|(key, log)| state.apply(key, log))?;
            }
        }
        *seen = committed.generation.max(*seen);
        Ok(committed.log_position)
    }

    /// Walk back from the committed head until reaching the head this copy
    /// is at, or a checkpoint. Each file must sit exactly one position below
    /// the one that named it, so the walk is bounded and cannot loop.
    async fn tail(
        &self,
        committed: &Committed,
        position: LogPosition,
        known: Option<&str>,
    ) -> Result<Tail> {
        let mut logs = Vec::new();
        let mut cursor = committed.manifest.as_ref().map(|head| head.0.clone());
        let mut expected = committed.log_position;
        let checkpoint = loop {
            let key = match cursor {
                reached if reached.as_deref() == known && expected == position => break None,
                Some(key) if expected > position => key,
                _ => {
                    return Err(StorageError::Diverged {
                        head: known.map(String::from),
                    })
                }
            };
            let head = decode_head(&key, &self.fetch(&key).await?)?;
            let found = match &head {
                Head::Log(log) => log.position,
                Head::Manifest(manifest) => manifest.position,
            };
            if found != expected {
                let reason = format!("it is at {found:?} where {expected:?} was committed");
                return Err(StorageError::corrupt(&key, reason));
            }
            match head {
                Head::Manifest(manifest) => break Some((key, manifest)),
                Head::Log(log) => {
                    expected = LogPosition(expected.0 - 1);
                    cursor = log.prev.clone();
                    logs.push((key, log));
                }
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
                    None => decode(&meta.key, Kind::Segment, &self.fetch(&meta.key).await?)
                        .map(Arc::new),
                }
            }
        }))
        .await?;
        State::from_manifest(key, manifest, files)
    }
}

/// A read-only copy of a partition that follows what the writer commits.
#[derive(Debug, Clone)]
pub struct Follower(Arc<Partition>);

impl Follower {
    /// Load the partition exactly as committed.
    pub async fn open(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
    ) -> Result<Self> {
        Partition::load(key, blobs, committed)
            .await
            .map(Arc::new)
            .map(Self)
    }

    pub fn partition(&self) -> &Arc<Partition> {
        &self.0
    }

    /// Bring this copy up to `committed`, reading only files reachable from
    /// its head, and return the position it is now at. A reading of the
    /// committed state older than one already applied is ignored.
    pub async fn catch_up(&self, committed: &Committed) -> Result<LogPosition> {
        self.0.advance(committed).await
    }
}

/// The one node allowed to change a partition, for as long as its lease's
/// generation is current. Nothing it writes is visible, here or anywhere,
/// until the coordinator has accepted the commit.
///
/// A file's name is fixed by its position and the lease's generation, so a
/// writer must never attempt the same position twice. From the first upload
/// of an attempt until it is applied in memory the writer is closed, and it
/// reopens only on success: an error, or a caller that stops waiting, leaves
/// it closed for good.
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
        let partition = Arc::new(Partition::load(lease.key.clone(), blobs, committed).await?);
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
        let (position, prev, mut catalog) = self.snapshot(&open, |state| {
            (
                state.position(),
                state.head().map(String::from),
                state.catalog().clone(),
            )
        })?;
        let ops = mutations
            .into_iter()
            .filter_map(|mutation| catalog.resolve(mutation).transpose())
            .collect::<Result<Vec<_>>>()?;
        if ops.is_empty() {
            return Ok(position);
        }
        let log = LogSegment {
            position: self.following(position)?,
            prev,
            ops,
        };
        let key = Layout(&self.lease.key).log(log.position, self.lease.generation);
        let frame = encode(Kind::Log, &log)?;
        attempt(&mut open, async {
            self.publish(&key, frame, log.position).await?;
            self.partition
                .state
                .write()
                .expect("partition lock")
                .apply(&key, &log)
        })
        .await?;
        Ok(log.position)
    }

    /// Write a checkpoint: move the rows held in memory into a segment and
    /// commit a manifest, so recovery no longer needs the log before it.
    pub async fn flush(&self) -> Result<LogPosition> {
        let mut open = self.open.lock().await;
        let (position, pending) = self.snapshot(&open, |state| {
            (state.position(), state.has_commits_since_checkpoint())
        })?;
        if !pending {
            return Ok(position);
        }
        let position = self.following(position)?;
        let layout = Layout(&self.lease.key);
        let segment_key = layout.segment(position, self.lease.generation);
        let key = layout.manifest(position, self.lease.generation);
        let (segment, manifest) = self
            .partition
            .read(|state| state.checkpoint(position, &segment_key));
        let segment_frame = segment
            .as_ref()
            .map(|s| encode(Kind::Segment, s))
            .transpose()?;
        let frame = encode(Kind::Manifest, &manifest)?;
        attempt(&mut open, async {
            if let Some(segment_frame) = segment_frame {
                self.partition
                    .blobs
                    .put(&segment_key, segment_frame)
                    .await?;
            }
            self.publish(&key, frame, position).await?;
            let mut state = self.partition.state.write().expect("partition lock");
            state.checkpointed(&key, &manifest, segment)
        })
        .await?;
        Ok(position)
    }

    fn snapshot<T>(&self, open: &bool, f: impl FnOnce(&State) -> T) -> Result<T> {
        open.then(|| self.partition.read(f))
            .ok_or(StorageError::Closed)
    }

    fn following(&self, position: LogPosition) -> Result<LogPosition> {
        position
            .next()
            .ok_or_else(|| StorageError::Encode(format!("{} is out of positions", self.lease.key)))
    }

    /// Upload a head file and commit it.
    async fn publish(&self, key: &str, frame: Bytes, position: LogPosition) -> Result<()> {
        self.partition.blobs.put(key, frame).await?;
        let commit = Commit {
            log_position: position,
            manifest: Some(ManifestRef(key.to_string())),
        };
        let mut attempt = 1;
        loop {
            match self.coordinator.commit(&self.lease, commit.clone()).await {
                Err(error) if error.is_retryable() && attempt < COMMIT_ATTEMPTS => {
                    tokio::time::sleep(COMMIT_RETRY_DELAY * attempt).await;
                    attempt += 1;
                }
                outcome => return outcome.map(|_| ()).map_err(Into::into),
            }
        }
    }
}

/// Run one write attempt with the writer closed, reopening it only if the
/// attempt ran to completion and succeeded.
async fn attempt<T>(open: &mut bool, work: impl Future<Output = Result<T>>) -> Result<T> {
    *open = false;
    let done = work.await?;
    *open = true;
    Ok(done)
}
