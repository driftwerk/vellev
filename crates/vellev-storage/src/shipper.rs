//! Continuous backup's shipper (DW-668): what has been committed to a
//! partition, shipped to the backup store a batch at a time, from the rows
//! of `vellev.commits`, never from a listing.
//!
//! A batch, for the partition's incarnation:
//!
//! 1. reads the partition's shipping afresh (amendment 1, A8), and ships
//!    nothing unless it is of that incarnation, has a base, and has not
//!    stalled: so a stale shipper, of a run since re-based or of an
//!    incarnation since removed, stops;
//! 2. takes the commits after what the run has shipped (or from its base)
//!    in order, positions skipping where they do (a restore commits past
//!    the current position);
//! 3. ships each as a link: its frame read from the primary under the
//!    row's head, checked by its CRC and decoded to the row's position
//!    (and, for a log, made on the row before), its header made from the
//!    row (its time, by the database's clock) and the key (its
//!    generation); a checkpoint's files first, each copied to the backup
//!    store and summed on the way, and only then its link (I3);
//! 4. creates each link once (`put_new`): a link already there of the same
//!    commit, every field but `body` equal, is shipped; at the run's own
//!    base, one that differs only in `before`, its own `None`, is too
//!    (bootstrap note C); anything else is a fork, and stalls the run
//!    naming the fields that differ;
//! 5. notes the watermark at the last position it shipped.
//!
//! A frame too large to go inline stalls the run until spilling is built
//! (S3). Every other failure of the stores or the coordinator ends the
//! batch, after noting what it shipped, and is the caller's: the node's
//! scheduler (review 6.2 item 7) decides when to try again, when to
//! alarm, and whether a failure that persists stalls the run. The core
//! only retries by being called again.

use std::fmt;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use sha2::{Digest, Sha256};

use crate::blobs::{Blobs, Created, Layout};
use crate::coordinator::{
    ClusterId, Coordinator, Incarnation, Lineage, LogPosition, PartitionKey, RestorePoint, ShipBase,
};
use crate::format::{decode_head, Head as Stored, Manifest};
use crate::link::{self, Before, Body, Head, Header, Needed, Sum, Timeline, Unlinkable};
use crate::{Result, StorageError};

/// How many times a link the store said was there, and was not when read
/// back, is created again: another create of it was under way and failed.
const CREATE_TRIES: usize = 3;

/// Why a run of shipping stopped. Each is written as the run's stall,
/// for an operator: none passes by being asked again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stall {
    /// A link at `position` is of another commit than this one: the
    /// header fields named differ.
    Fork {
        position: LogPosition,
        fields: Vec<&'static str>,
    },
    /// No commit row is at the run's base, where a run's first link is.
    BaseRowMissing { position: LogPosition },
    /// The link at `position` cannot be created in the backup store.
    NotCreated { position: LogPosition, why: String },
    /// A committed head, or a file a checkpoint lists, is not in the
    /// primary.
    PrimaryMissing { key: String },
    /// A committed head's frame, or a link, does not hold together.
    Corrupt { key: String, why: String },
    /// The row at `position` names no head of this partition laid out as
    /// one, or its frame is of another position or made on another head.
    NotTheRowsHead { position: LogPosition },
    /// The commit at `position` was made before the one before it, by the
    /// database's clock (review R9).
    TimeWentBack { position: LogPosition },
    /// The link of what the run has shipped up to is not in the backup
    /// store.
    LinkMissing { position: LogPosition },
    /// The frame at `position` is too large to go inline, and spilling it
    /// is not built yet (S3).
    SpillNotBuilt { position: LogPosition },
}

impl fmt::Display for Stall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stall::Fork { position, fields } => write!(
                f,
                "fork: the link at position {} is of another commit, differing in {}",
                position.0,
                fields.join(", ")
            ),
            Stall::BaseRowMissing { position } => write!(
                f,
                "no commit row is at the run's base, position {}",
                position.0
            ),
            Stall::NotCreated { position, why } => write!(
                f,
                "the link at position {} cannot be created in the backup store: {why}",
                position.0
            ),
            Stall::PrimaryMissing { key } => {
                write!(f, "{key} is committed and not in the primary store")
            }
            Stall::Corrupt { key, why } => write!(f, "{key} is corrupt: {why}"),
            Stall::NotTheRowsHead { position } => write!(
                f,
                "the commit at position {} names no head of it made on the commit before",
                position.0
            ),
            Stall::TimeWentBack { position } => write!(
                f,
                "the commit at position {} was made before the commit before it",
                position.0
            ),
            Stall::LinkMissing { position } => write!(
                f,
                "the link at position {}, shipped, is not in the backup store",
                position.0
            ),
            Stall::SpillNotBuilt { position } => write!(
                f,
                "the frame at position {} is too large to go inline, and spill is not yet built",
                position.0
            ),
        }
    }
}

/// Why a batch shipped nothing without trying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Idle {
    /// No base is recorded: shipping has not begun.
    NotShipping,
    /// The partition is of another incarnation now.
    OtherIncarnation,
    /// The run has stalled, for this reason.
    Stalled(String),
}

/// What a batch did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Batch {
    /// Shipped nothing, without trying.
    Idle(Idle),
    /// Shipped up to `through`, if anything, and stalled the run if it
    /// says why.
    Shipped {
        through: Option<LogPosition>,
        stalled: Option<Stall>,
    },
}

/// Ships one partition's history to the backup store.
#[derive(Debug, Clone)]
pub struct Shipper {
    primary: Arc<dyn Blobs>,
    backup: Arc<dyn Blobs>,
    coordinator: Arc<dyn Coordinator>,
    key: PartitionKey,
    lineage: Lineage,
    cluster: ClusterId,
}

impl Shipper {
    /// The shipper of the partition `key`, of the tenant of `lineage`, in
    /// the cluster `cluster`, from `primary` to `backup`.
    pub fn new(
        primary: Arc<dyn Blobs>,
        backup: Arc<dyn Blobs>,
        coordinator: Arc<dyn Coordinator>,
        key: PartitionKey,
        lineage: Lineage,
        cluster: ClusterId,
    ) -> Self {
        Self {
            primary,
            backup,
            coordinator,
            key,
            lineage,
            cluster,
        }
    }

    /// The timeline of the partition's incarnation `incarnation`.
    pub fn timeline(&self, incarnation: Incarnation) -> Timeline {
        Timeline {
            tenant: self.key.tenant.clone(),
            lineage: self.lineage,
            cluster: self.cluster,
            incarnation,
            partition: self.key.partition,
        }
    }

    /// Ship up to `limit` commits of the partition's incarnation
    /// `incarnation`: see the module's doc. An error is a failure of the
    /// stores or the coordinator, after what was shipped is noted; the
    /// run is left to be shipped on by the next batch.
    pub async fn ship_batch(&self, incarnation: Incarnation, limit: usize) -> Result<Batch> {
        let shipping = self.coordinator.shipping(&self.key).await?;
        if shipping.incarnation != incarnation {
            return Ok(Batch::Idle(Idle::OtherIncarnation));
        }
        let Some(base) = shipping.base else {
            return Ok(Batch::Idle(Idle::NotShipping));
        };
        if let Some(why) = shipping.stalled {
            return Ok(Batch::Idle(Idle::Stalled(why)));
        }
        let timeline = self.timeline(incarnation);
        let mut last = match shipping.shipped {
            None => None,
            Some(shipped) => match self.shipped(&timeline, shipped).await? {
                Ok(made) => Some(made),
                Err(stall) => return self.stall(incarnation, &base, None, stall).await,
            },
        };
        let after = match &last {
            Some(made) => made.position,
            None => LogPosition(base.position.0.saturating_sub(1)),
        };
        let rows = self
            .coordinator
            .commits_after(&self.key, incarnation, after, limit)
            .await?;
        let mut through = None;
        for row in &rows {
            let made = match self.ship(&timeline, &base, last.as_ref(), row).await {
                Ok(made) => made,
                Err(error) => {
                    if let Err(unnoted) = self.note(incarnation, &base, through).await {
                        tracing::warn!(
                            partition = %self.key,
                            error = %unnoted,
                            "what a failed batch shipped was not noted"
                        );
                    }
                    return Err(error);
                }
            };
            match made {
                Ok(made) => {
                    through = Some(row.position);
                    last = Some(made);
                }
                Err(stall) => return self.stall(incarnation, &base, through, stall).await,
            }
        }
        self.note(incarnation, &base, through).await?;
        Ok(Batch::Shipped {
            through,
            stalled: None,
        })
    }

    /// Note the run shipped up to `through`, if anything.
    async fn note(
        &self,
        incarnation: Incarnation,
        base: &ShipBase,
        through: Option<LogPosition>,
    ) -> Result<()> {
        if let Some(position) = through {
            self.coordinator
                .note_shipped(&self.key, incarnation, base.position, position)
                .await?;
        }
        Ok(())
    }

    /// Note the run shipped up to `through`, if anything, and stalled.
    async fn stall(
        &self,
        incarnation: Incarnation,
        base: &ShipBase,
        through: Option<LogPosition>,
        stall: Stall,
    ) -> Result<Batch> {
        self.note(incarnation, base, through).await?;
        self.coordinator
            .stall_shipping(&self.key, incarnation, base.position, &stall.to_string())
            .await?;
        Ok(Batch::Shipped {
            through,
            stalled: Some(stall),
        })
    }

    /// The commit the run has shipped up to, at `position`, as its link in
    /// the backup store says.
    async fn shipped(
        &self,
        timeline: &Timeline,
        position: LogPosition,
    ) -> Result<std::result::Result<Made, Stall>> {
        let at = timeline.link(position);
        let bytes = match self.backup.get(&at).await {
            Ok(bytes) => bytes,
            Err(StorageError::Missing { .. }) => return Ok(Err(Stall::LinkMissing { position })),
            Err(error) => return Err(error),
        };
        Ok(match link::decode(&at, &bytes) {
            Ok(link) => Ok(Made::of(&link.header)),
            Err(error) => Err(Stall::Corrupt {
                key: at,
                why: why(error),
            }),
        })
    }

    /// Ship the commit of `row`, made on `last`, or at the run's base if
    /// there is none.
    async fn ship(
        &self,
        timeline: &Timeline,
        base: &ShipBase,
        last: Option<&Made>,
        row: &RestorePoint,
    ) -> Result<std::result::Result<Made, Stall>> {
        let position = row.position;
        let key = &row.head.0;
        let not_the_rows = Ok(Err(Stall::NotTheRowsHead { position }));
        if last.is_none() && position != base.position {
            return Ok(Err(Stall::BaseRowMissing {
                position: base.position,
            }));
        }
        let layout = Layout(&self.key);
        let Some(named) = layout.named_by(key) else {
            return not_the_rows;
        };
        let head = match named.kind {
            "log" => Head::Log,
            "manifest" => Head::Manifest,
            _ => return not_the_rows,
        };
        if named.position != position {
            return not_the_rows;
        }
        if last.is_some_and(|last| row.at < last.at) {
            return Ok(Err(Stall::TimeWentBack { position }));
        }
        let frame = match self.primary.get(key).await {
            Ok(frame) => frame,
            Err(StorageError::Missing { .. }) => {
                return Ok(Err(Stall::PrimaryMissing { key: key.clone() }))
            }
            Err(error) => return Err(error),
        };
        let stored = match decode_head(key, &frame) {
            Ok(stored) => stored,
            Err(error) => {
                return Ok(Err(Stall::Corrupt {
                    key: key.clone(),
                    why: why(error),
                }))
            }
        };
        let manifest = match &stored {
            Stored::Log(log) => {
                // A log follows the commit just before it, and names it.
                let follows = last.is_none_or(|last| {
                    last.position.next() == Some(position)
                        && log.prev.as_deref() == Some(last.key.as_str())
                });
                if log.position != position || !follows {
                    return not_the_rows;
                }
                None
            }
            Stored::Manifest(manifest) if manifest.position == position => Some(manifest),
            Stored::Manifest(_) => return not_the_rows,
        };
        let mut files = Vec::new();
        for file in manifest.map(checkpoint_files).unwrap_or_default() {
            match self.copy(timeline, file).await? {
                Ok(needed) => files.push(needed),
                Err(stall) => return Ok(Err(stall)),
            }
        }
        let header = Header {
            tenant: timeline.tenant.clone(),
            lineage: timeline.lineage,
            cluster: timeline.cluster,
            incarnation: timeline.incarnation,
            partition: timeline.partition,
            position,
            key: key.clone(),
            head,
            generation: named.generation,
            committed_at: row.at,
            sum: Sum::of(&frame),
            size: frame.len() as u64,
            before: last.map(|last| Before {
                position: last.position,
                sum: last.sum.clone(),
            }),
            files,
            body: Body::Inline,
        };
        match link::shape(&header, self.backup.creates_at_most()) {
            Ok(Body::Inline) => {}
            Ok(Body::Spilled) => return Ok(Err(Stall::SpillNotBuilt { position })),
            Err(Unlinkable::NoCreate) => {
                return Ok(Err(Stall::NotCreated {
                    position,
                    why: "the backup store does not create a file only if none is there".into(),
                }))
            }
            Err(Unlinkable::HeaderTooLarge { bytes, most }) => {
                return Ok(Err(Stall::NotCreated {
                    position,
                    why: format!("its header is {bytes} bytes; the store creates {most} at most"),
                }))
            }
        }
        let made = Made::of(&header);
        self.create(timeline, base, header, &frame)
            .await
            .map(|created| created.map(|()| made))
    }

    /// Create the link of `header`, with `frame` inline, once: a link
    /// already there passes only if it is of the same commit.
    async fn create(
        &self,
        timeline: &Timeline,
        base: &ShipBase,
        header: Header,
        frame: &[u8],
    ) -> Result<std::result::Result<(), Stall>> {
        let position = header.position;
        let at = timeline.link(position);
        let encoded = link::encode(&header, Some(frame))?;
        for _ in 0..CREATE_TRIES {
            match self.backup.put_new(&at, encoded.clone()).await {
                Ok(Created::Written) => return Ok(Ok(())),
                Ok(Created::AlreadyThere) => {}
                Err(StorageError::NotCreated { reason, .. }) => {
                    return Ok(Err(Stall::NotCreated {
                        position,
                        why: reason,
                    }))
                }
                Err(error) => return Err(error),
            }
            let there = match self.backup.get(&at).await {
                Ok(there) => there,
                // Another create of it was under way, and failed.
                Err(StorageError::Missing { .. }) => continue,
                Err(error) => return Err(error),
            };
            let theirs = match link::decode(&at, &there) {
                Ok(theirs) => theirs.header,
                Err(error) => {
                    return Ok(Err(Stall::Corrupt {
                        key: at,
                        why: why(error),
                    }))
                }
            };
            let fields = header.differs(&theirs);
            let at_base = position == base.position;
            // Every field but `before` comes from the commit itself, so any
            // other difference, at the base or not, is another commit.
            return Ok(match (fields.as_slice(), at_base) {
                ([], _) => Ok(()),
                // An older run's link of the same commit, which knew the
                // commit before it (bootstrap note C).
                (["before"], true) if header.before.is_none() => Ok(()),
                _ => Err(Stall::Fork { position, fields }),
            });
        }
        Err(StorageError::Blob {
            key: at,
            source: format!(
                "said to be there {CREATE_TRIES} times, and each time not there when read"
            )
            .into(),
        })
    }

    /// Copy the checkpoint file under `key` from the primary to the backup
    /// store, a part at a time, summing it on the way.
    async fn copy(
        &self,
        timeline: &Timeline,
        key: &str,
    ) -> Result<std::result::Result<Needed, Stall>> {
        let summed = Arc::new(Mutex::new(Summed::default()));
        let parts = {
            let summed = summed.clone();
            self.primary
                .parts(key)
                .map(move |part| {
                    let mut summed = summed.lock().expect("summing never panics");
                    match &part {
                        Ok(bytes) => {
                            summed.hasher.update(bytes);
                            summed.size += bytes.len() as u64;
                        }
                        Err(StorageError::Missing { .. }) => summed.missing = true,
                        Err(_) => {}
                    }
                    part
                })
                .boxed()
        };
        let put = self.backup.put_parts(&timeline.file(key), parts).await;
        let summed = std::mem::take(&mut *summed.lock().expect("summing never panics"));
        match put {
            Ok(()) => Ok(Ok(Needed {
                key: key.to_string(),
                size: summed.size,
                sum: Sum(hex(&summed.hasher.finalize())),
            })),
            Err(_) if summed.missing => Ok(Err(Stall::PrimaryMissing {
                key: key.to_string(),
            })),
            Err(error) => Err(error),
        }
    }
}

/// A commit shipped, as the next is made on it.
#[derive(Debug, Clone)]
struct Made {
    position: LogPosition,
    key: String,
    sum: Sum,
    at: chrono::DateTime<chrono::Utc>,
}

impl Made {
    fn of(header: &Header) -> Self {
        Self {
            position: header.position,
            key: header.key.clone(),
            sum: header.sum.clone(),
            at: header.committed_at,
        }
    }
}

/// A file summed as it goes by.
#[derive(Default)]
struct Summed {
    hasher: Sha256,
    size: u64,
    missing: bool,
}

/// Every checkpoint file `manifest` names: each segment, its index and its
/// vectors file, as [`crate::chain::needed`] requires its link to list.
fn checkpoint_files(manifest: &Manifest) -> Vec<&str> {
    manifest
        .segments
        .iter()
        .flat_map(|segment| {
            [
                Some(&segment.key),
                segment.index.as_ref(),
                segment.vectors.as_ref(),
            ]
        })
        .flatten()
        .map(String::as_str)
        .collect()
}

/// Bytes as lowercase hex, as [`Sum`] holds a SHA-256.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// What makes a file corrupt, without the key a stall names already.
fn why(error: StorageError) -> String {
    match error {
        StorageError::Corrupt { reason, .. } => reason,
        other => other.to_string(),
    }
}
