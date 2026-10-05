use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{join_all, try_join_all};
use tokio::sync::Mutex;
use vellev_core::Embedding;

use crate::blobs::{is_plain, Author, Blobs, Layout};
use crate::coordinator::{
    Commit, Committed, Coordinator, Generation, Lease, LogPosition, ManifestRef, PartitionKey,
};
use crate::draft::{Draft, Wanted};
use crate::format::{decode, decode_head, encode, Head, Kind, LogSegment, Manifest, Op};
use crate::mutation::{Mutation, Receipt, Vectorize};
use crate::state::{Loaded, State};
use crate::{Result, StorageError};

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;
const COMMIT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// How long a commit waits for the vectors it needs made.
const EMBEDDING_TIMEOUT: Duration = Duration::from_secs(300);

/// The vectors made so far for one commit, by what was embedded and how.
type Made = HashMap<(Embedding, String), Vec<f32>>;

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

    /// Run `f` against a consistent view of the partition. The lock is a
    /// blocking one and is never held across an await.
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

    /// Fetch a file named by data rather than built by [`Layout`], which must
    /// be a plain key of this partition's.
    async fn fetch(&self, key: &str) -> Result<Bytes> {
        if !(is_plain(key) && key.starts_with(&Layout(&self.key).prefix())) {
            return Err(StorageError::corrupt(
                key,
                "the key is not one of this partition's",
            ));
        }
        self.blobs.get(key).await
    }

    /// Generations only grow, so a reading from an older generation than one
    /// already applied is out of date whatever its position, and so is a
    /// lower position from the same generation. A reading that is not ahead
    /// and not stale means the history this copy holds was replaced.
    async fn advance(&self, committed: &Committed) -> Result<LogPosition> {
        let mut seen = self.seen.lock().await;
        let (position, head) = self.read(State::at);
        let target = committed.manifest.as_ref().map(|head| head.0.as_str());
        let stale = committed.generation < *seen
            || (committed.generation == *seen && committed.log_position < position);
        if stale {
            return Ok(position);
        }
        let lost = || match (&head, target) {
            (Some(_), _) => StorageError::Diverged { head: head.clone() },
            (None, Some(target)) => {
                StorageError::corrupt(target, "its history does not reach the start of the log")
            }
            (None, None) => StorageError::corrupt(
                &Layout(&self.key).prefix(),
                format!("{:?} is committed with no head", committed.log_position),
            ),
        };
        if committed.log_position == position && target == head.as_deref() {
            *seen = committed.generation;
            return Ok(position);
        }
        if committed.log_position <= position {
            return Err(lost());
        }
        let tail = self
            .tail(committed, position, head.as_deref())
            .await?
            .ok_or_else(lost)?;
        *seen = committed.generation;
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
        Ok(committed.log_position)
    }

    /// Walk back from the committed head until reaching the head this copy
    /// is at, or a checkpoint; `None` when the chain leads to neither. Each
    /// file must sit exactly one position below the one that named it, so the
    /// walk is bounded and cannot loop.
    async fn tail(
        &self,
        committed: &Committed,
        position: LogPosition,
        known: Option<&str>,
    ) -> Result<Option<Tail>> {
        let mut logs = Vec::new();
        let mut cursor = committed.manifest.as_ref().map(|head| head.0.clone());
        let mut expected = committed.log_position;
        let checkpoint = loop {
            let key = match cursor {
                reached if reached.as_deref() == known && expected == position => break None,
                Some(key) if expected > position => key,
                _ => return Ok(None),
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
        Ok(Some(Tail { checkpoint, logs }))
    }

    async fn restore(&self, key: &str, manifest: Manifest) -> Result<State> {
        let held: HashMap<String, Loaded> = self.read(|state| {
            state
                .loaded_segments()
                .map(|(key, loaded)| (key.to_string(), loaded))
                .collect()
        });
        let segments = try_join_all(manifest.segments.iter().map(|meta| {
            let held = held.get(&meta.key).cloned();
            async move {
                match held {
                    Some(loaded) => Ok(loaded),
                    None => decode(&meta.key, Kind::Segment, &self.fetch(&meta.key).await?)
                        .map(|file| (Arc::new(file), None)),
                }
            }
        }))
        .await?;
        State::from_manifest(key, manifest, segments)
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
/// A file is named by its position and its author, and is never rewritten.
/// The writer therefore never attempts a position twice: from the first
/// upload of an attempt until it is applied in memory the writer is not
/// ready, and it becomes ready again only on success. An error, or a caller
/// that stops waiting, leaves it closed for good.
#[derive(Debug)]
pub struct Writer {
    partition: Arc<Partition>,
    coordinator: Arc<dyn Coordinator>,
    lease: Lease,
    author: Author,
    vectorize: Option<Arc<dyn Vectorize>>,
    ready: Mutex<bool>,
}

impl Writer {
    /// Start writing from the state the lease was taken at: `committed` must
    /// be what the coordinator returned together with `lease`.
    pub async fn open(
        blobs: Arc<dyn Blobs>,
        coordinator: Arc<dyn Coordinator>,
        lease: Lease,
        committed: &Committed,
    ) -> Result<Self> {
        let partition = Arc::new(Partition::load(lease.key.clone(), blobs, committed).await?);
        let author = Author {
            generation: lease.generation,
            writer: uuid::Uuid::new_v4(),
        };
        Ok(Self {
            partition,
            coordinator,
            lease,
            author,
            vectorize: None,
            ready: Mutex::new(true),
        })
    }

    /// Embed with `vectorize` for the collections that embed their own
    /// objects. Without it, a write that needs a vector made is refused.
    pub fn embedding_with(self, vectorize: Arc<dyn Vectorize>) -> Self {
        Self {
            vectorize: Some(vectorize),
            ..self
        }
    }

    pub fn partition(&self) -> &Arc<Partition> {
        &self.partition
    }

    pub fn lease(&self) -> &Lease {
        &self.lease
    }

    pub fn author(&self) -> &Author {
        &self.author
    }

    /// Commit `mutations` as one atomic step. A rejected mutation fails the
    /// whole batch and changes nothing.
    pub async fn commit(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        self.commit_group(vec![mutations])
            .await?
            .pop()
            .expect("one result per batch")
    }

    /// Commit several callers' batches as one log segment. Each batch is
    /// atomic on its own and sees the batches before it; a rejected batch is
    /// left out with its own error and does not affect the others. Every
    /// accepted batch becomes visible at the same position.
    ///
    /// A batch that needs a vector which cannot be made is rejected like
    /// any other, and the rest are resolved again without it.
    pub async fn commit_group(&self, batches: Vec<Vec<Mutation>>) -> Result<Vec<Result<Receipt>>> {
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let mut unembedded: Vec<Option<(String, String)>> = vec![None; batches.len()];
        let mut made = Made::new();
        let ((position, prev), outcomes, ops) = loop {
            let (at, outcomes, (mut ops, wanted)) = self.partition.read(|state| {
                let mut draft = Draft::new(state);
                let outcomes: Vec<_> = batches
                    .iter()
                    .zip(&unembedded)
                    .map(|(batch, unembedded)| match unembedded {
                        Some(_) => draft.batch(Vec::new()),
                        None => draft.batch(batch.clone()),
                    })
                    .collect();
                (state.at(), outcomes, draft.finish())
            });
            let failed = self.make(&wanted, &mut made).await;
            if failed.is_empty() {
                for Wanted {
                    op,
                    embedding,
                    text,
                    ..
                } in wanted
                {
                    if let Op::Upsert { object, .. } = &mut ops[op] {
                        object.vector = made.get(&(embedding, text)).cloned();
                    }
                }
                break (at, outcomes, ops);
            }
            for wanted in &wanted {
                if let Some(reason) = failed.get(&wanted.embedding) {
                    unembedded[wanted.batch] =
                        Some((wanted.embedding.model.clone(), reason.clone()));
                }
            }
        };
        let position = if ops.is_empty() {
            position
        } else {
            let log = LogSegment {
                position: following(position)?,
                prev,
                ops,
            };
            let key = Layout(&self.lease.key).log(log.position, &self.author);
            let frame = encode(Kind::Log, &log)?;
            attempt(&mut ready, async {
                self.publish(&key, frame, log.position).await?;
                self.partition
                    .state
                    .write()
                    .expect("partition lock")
                    .apply(&key, &log)
            })
            .await?;
            log.position
        };
        Ok(outcomes
            .into_iter()
            .zip(unembedded)
            .map(|(outcomes, unembedded)| match unembedded {
                Some((model, reason)) => Err(StorageError::Embedding { model, reason }),
                None => outcomes.map(|outcomes| Receipt { position, outcomes }),
            })
            .collect())
    }

    /// Make the vectors `wanted` that are not `made` yet, each way of
    /// embedding at the same time as the others. Returns why, for each way
    /// that could not make all of its own.
    ///
    /// The writer is held meanwhile, so that no other write can change a
    /// text a vector is being made for.
    async fn make(&self, wanted: &[Wanted], made: &mut Made) -> HashMap<Embedding, String> {
        let mut missing: HashMap<&Embedding, Vec<&String>> = HashMap::new();
        for Wanted {
            embedding, text, ..
        } in wanted
        {
            let texts = missing.entry(embedding).or_default();
            if !texts.contains(&text) && !made.contains_key(&(embedding.clone(), text.clone())) {
                texts.push(text);
            }
        }
        let asked = missing
            .into_iter()
            .filter(|(_, texts)| !texts.is_empty())
            .map(|(embedding, texts)| async move {
                let texts: Vec<String> = texts.into_iter().cloned().collect();
                let vectors = self.vectors(embedding, texts.clone()).await;
                (embedding, vectors.map(|vectors| (texts, vectors)))
            });
        let mut failed = HashMap::new();
        for (embedding, outcome) in join_all(asked).await {
            match outcome {
                Ok((texts, vectors)) => made.extend(
                    texts
                        .into_iter()
                        .zip(vectors)
                        .map(|(text, vector)| ((embedding.clone(), text), vector)),
                ),
                Err(reason) => {
                    failed.insert(embedding.clone(), reason);
                }
            }
        }
        failed
    }

    /// A vector the collection can hold for each of `texts`, or why not.
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let vectorize = self
            .vectorize
            .as_ref()
            .ok_or("this writer has nothing to embed with")?;
        let asked = texts.len();
        let vectors = tokio::time::timeout(EMBEDDING_TIMEOUT, vectorize.vectors(embedding, texts))
            .await
            .map_err(|_| format!("no vectors within {EMBEDDING_TIMEOUT:?}"))??;
        if vectors.len() != asked {
            return Err(format!("{} vectors for {asked} texts", vectors.len()));
        }
        vectors
            .into_iter()
            .map(|vector| embedding.checked(vector).map_err(|error| error.to_string()))
            .collect()
    }

    /// Write a checkpoint: move the rows held in memory into a segment and
    /// commit a manifest, so recovery no longer needs the log before it.
    pub async fn flush(&self) -> Result<LogPosition> {
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let (position, pending) = self
            .partition
            .read(|state| (state.position(), state.has_commits_since_checkpoint()));
        if !pending {
            return Ok(position);
        }
        let position = following(position)?;
        let layout = Layout(&self.lease.key);
        let segment_key = layout.segment(position, &self.author);
        let key = layout.manifest(position, &self.author);
        let (segment, manifest) = self
            .partition
            .read(|state| state.checkpoint(position, &segment_key));
        let segment_frame = segment
            .as_ref()
            .map(|(_, file)| encode(Kind::Segment, file))
            .transpose()?;
        // Indexed here, before the attempt, so that readers are not held
        // behind the write lock while a large segment is indexed.
        let segment = segment.map(|(meta, file)| {
            let indexes = self.partition.read(|state| state.indexed(&file.rows));
            (meta, (Arc::new(file), Some(Arc::new(indexes))))
        });
        let frame = encode(Kind::Manifest, &manifest)?;
        attempt(&mut ready, async {
            if let Some(segment_frame) = segment_frame {
                self.partition
                    .blobs
                    .put(&segment_key, segment_frame)
                    .await?;
            }
            self.publish(&key, frame, position).await?;
            let mut state = self.partition.state.write().expect("partition lock");
            state.checkpointed(&key, position, segment)
        })
        .await?;
        Ok(position)
    }

    async fn publish(&self, key: &str, frame: Bytes, position: LogPosition) -> Result<()> {
        self.partition.blobs.put(key, frame).await?;
        let commit = Commit {
            log_position: position,
            manifest: Some(ManifestRef(key.to_string())),
        };
        let mut tries = 1;
        loop {
            match self.coordinator.commit(&self.lease, commit.clone()).await {
                Err(error) if error.is_retryable() && tries < COMMIT_ATTEMPTS => {
                    tokio::time::sleep(COMMIT_RETRY_DELAY * tries).await;
                    tries += 1;
                }
                outcome => return outcome.map(|_| ()).map_err(Into::into),
            }
        }
    }
}

fn following(position: LogPosition) -> Result<LogPosition> {
    position
        .next()
        .ok_or(StorageError::Exhausted("log positions"))
}

/// Run one write attempt with the writer not ready, making it ready again
/// only if the attempt ran to completion and succeeded.
async fn attempt<T>(ready: &mut bool, work: impl Future<Output = Result<T>>) -> Result<T> {
    *ready = false;
    let done = work.await?;
    *ready = true;
    Ok(done)
}
