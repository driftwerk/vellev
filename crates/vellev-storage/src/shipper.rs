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
//! A frame too large to go inline, with its link's header, is spilled
//! (amendment 1, A2): put first under `frames/{key}` beside the links, and
//! only then linked, the link carrying no frame. A log is spilled as it
//! comes from the primary, a part at a time and never held whole, in one
//! pass that sums it, checks its CRC and reads its start for its position
//! and `prev` (A3); the upload is given up, and the run stalls, unless all
//! of them hold. A checkpoint, read whole for its files, is spilled from
//! what is held of it. A link whose header alone is larger than the store
//! creates stalls the run.
//!
//! Every other failure of the stores or the coordinator ends the
//! batch, after noting what it shipped, and is the caller's: the node's
//! scheduler (review 6.2 item 7) decides when to try again, when to
//! alarm, and whether a failure that persists stalls the run. The core
//! only retries by being called again.

use std::fmt;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use sha2::{Digest, Sha256};

use crate::blobs::{Blobs, Created, Layout};
use crate::coordinator::{
    ClusterId, Coordinator, Incarnation, Lineage, LogPosition, PartitionKey, RestorePoint, ShipBase,
};
use crate::format::{decode_head, Framed, Head as Stored, Kind, Manifest};
use crate::link::{self, Before, Body, Head, Header, Needed, Sum, Timeline, Unlinkable};
use crate::prefix::{Found, Prefix};
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
        // What the run had shipped as this batch began: its stall is taken
        // only if it still has (`Coordinator::stall_shipping`).
        let seen = shipping.shipped;
        let mut last = match shipping.shipped {
            None => None,
            Some(shipped) => match self.shipped(&timeline, shipped).await? {
                Ok(made) => Some(made),
                Err(stall) => return self.stall(incarnation, &base, seen, None, stall).await,
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
                Err(stall) => return self.stall(incarnation, &base, seen, through, stall).await,
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

    /// Note the run shipped up to `through`, if anything, and stalled, in
    /// one step, if it has shipped no further than `seen` since the batch
    /// began: a stale shipper's stall is refused, and the error is its.
    async fn stall(
        &self,
        incarnation: Incarnation,
        base: &ShipBase,
        seen: Option<LogPosition>,
        through: Option<LogPosition>,
        stall: Stall,
    ) -> Result<Batch> {
        let why = stall.to_string();
        self.coordinator
            .stall_shipping(&self.key, incarnation, base.position, seen, through, &why)
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
        let made_on = last.map(|last| Before {
            position: last.position,
            sum: last.sum.clone(),
        });
        let header = |sum: Sum, size: u64, files: Vec<Needed>, body: Body| Header {
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
            sum,
            size,
            before: made_on.clone(),
            files,
            body,
        };
        // A log is read a part at a time, and spilled as it comes if it is
        // too large to go inline (amendment 1, A2); a checkpoint, which is
        // decoded for its files, and a log that goes inline, are read whole.
        let frame = match head {
            Head::Manifest => match self.primary.get(key).await {
                Ok(frame) => frame,
                Err(StorageError::Missing { .. }) => {
                    return Ok(Err(Stall::PrimaryMissing { key: key.clone() }))
                }
                Err(error) => return Err(error),
            },
            Head::Log => {
                let mut parts = self.primary.parts(key);
                let mut come = Vec::new();
                let mut framed = Framed::default();
                // As much as says how long the frame is.
                while framed.declared().is_none() {
                    match parts.next().await {
                        None => break,
                        Some(Ok(part)) => {
                            framed.update(&part);
                            come.push(part);
                        }
                        Some(Err(StorageError::Missing { .. })) => {
                            return Ok(Err(Stall::PrimaryMissing { key: key.clone() }))
                        }
                        Some(Err(error)) => return Err(error),
                    }
                }
                let declared = framed.declared();
                // Its sum unknown yet, but of the length it will have.
                let unsummed = Sum("0".repeat(64));
                let shape = declared.map(|size| {
                    let unsummed = header(unsummed, size, Vec::new(), Body::Inline);
                    link::shape(&unsummed, self.backup.creates_at_most())
                });
                match shape {
                    Some(Err(unlinkable)) => return Ok(Err(not_linkable(position, unlinkable))),
                    Some(Ok(Body::Spilled)) => {
                        let expected = Expected {
                            key: key.clone(),
                            position,
                            last: last.map(|last| (last.position, last.key.clone())),
                        };
                        let rest = stream::iter(come.into_iter().map(Ok)).chain(parts).boxed();
                        let (sum, size) =
                            match self.spill_streamed(timeline, expected, rest).await? {
                                Ok(spilled) => spilled,
                                Err(stall) => return Ok(Err(stall)),
                            };
                        let header = header(sum, size, Vec::new(), Body::Spilled);
                        let made = Made::of(&header);
                        return self
                            .create(timeline, base, header, None)
                            .await
                            .map(|created| created.map(|()| made));
                    }
                    Some(Ok(Body::Inline)) | None => {}
                }
                // Whole, and no longer than it says it is.
                let most = declared.unwrap_or(u64::MAX);
                let mut length: u64 = come.iter().map(|part| part.len() as u64).sum();
                while length <= most {
                    match parts.next().await {
                        None => break,
                        Some(Ok(part)) => {
                            length += part.len() as u64;
                            come.push(part);
                        }
                        Some(Err(StorageError::Missing { .. })) => {
                            return Ok(Err(Stall::PrimaryMissing { key: key.clone() }))
                        }
                        Some(Err(error)) => return Err(error),
                    }
                }
                if length > most {
                    return Ok(Err(Stall::Corrupt {
                        key: key.clone(),
                        why: "longer than its header says".into(),
                    }));
                }
                Bytes::from(come.concat())
            }
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
                let found = Found {
                    position: log.position,
                    prev: log.prev.clone(),
                };
                if !follows(&found, position, last) {
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
        let mut header = header(Sum::of(&frame), frame.len() as u64, files, Body::Inline);
        header.body = match link::shape(&header, self.backup.creates_at_most()) {
            Ok(body) => body,
            Err(unlinkable) => return Ok(Err(not_linkable(position, unlinkable))),
        };
        let made = Made::of(&header);
        let inline = match header.body {
            Body::Inline => Some(&frame[..]),
            // Its frame first, from what is held of it, then its link.
            Body::Spilled => {
                let whole = stream::once(std::future::ready(Ok(frame.clone()))).boxed();
                self.backup.put_parts(&timeline.frame(key), whole).await?;
                None
            }
        };
        self.create(timeline, base, header, inline)
            .await
            .map(|created| created.map(|()| made))
    }

    /// Spill the log `expected` names, coming a part at a time in `parts`,
    /// to its frame's place in the backup store, in one pass: summed by
    /// SHA-256 as it goes, its CRC checked and its start read for its
    /// position and `prev` (amendment 1, A2 and A3). The upload is given
    /// up, and nothing put, unless all of them hold. Its sum and size, or
    /// why the run stalls.
    async fn spill_streamed(
        &self,
        timeline: &Timeline,
        expected: Expected,
        parts: BoxStream<'_, Result<Bytes>>,
    ) -> Result<std::result::Result<(Sum, u64), Stall>> {
        let key = expected.key.clone();
        let spilling = Arc::new(Mutex::new(Spilling {
            hasher: Sha256::new(),
            size: 0,
            framed: Framed::default(),
            prefix: Some(Prefix::new(&key)),
            missing: false,
            stall: None,
            checked: false,
        }));
        let given_up = |key: &str| StorageError::corrupt(key, "the spill was given up");
        let tapped = {
            let (spilling, expected) = (spilling.clone(), expected.clone());
            parts.map(move |part| {
                let mut spilling = spilling.lock().expect("spilling never panics");
                let part = match part {
                    Ok(part) => part,
                    Err(error) => {
                        spilling.missing |= matches!(error, StorageError::Missing { .. });
                        return Err(error);
                    }
                };
                spilling.hasher.update(&part);
                spilling.size += part.len() as u64;
                spilling.framed.update(&part);
                if let Some(prefix) = spilling.prefix.as_mut() {
                    match prefix.feed(&part) {
                        Ok(None) => {}
                        read => {
                            if let Some(stall) = expected.stall(read) {
                                spilling.stall = Some(stall);
                                return Err(given_up(&expected.key));
                            }
                            // Found, and right: nothing more of it is read.
                            spilling.prefix = None;
                        }
                    }
                }
                Ok(part)
            })
        };
        // Once all of it has come: its CRC, and its start if not yet read.
        let checked = {
            let (spilling, expected) = (spilling.clone(), expected.clone());
            stream::once(async move {
                let mut spilling = spilling.lock().expect("spilling never panics");
                let framed = std::mem::take(&mut spilling.framed);
                if let Err(error) = framed.came(&expected.key, Kind::Log) {
                    spilling.stall = Some(Stall::Corrupt {
                        key: expected.key.clone(),
                        why: why(error),
                    });
                    return Some(Err(given_up(&expected.key)));
                }
                if let Some(prefix) = spilling.prefix.take() {
                    if let Some(stall) = expected.stall(prefix.finish().map(Some)) {
                        spilling.stall = Some(stall);
                        return Some(Err(given_up(&expected.key)));
                    }
                }
                spilling.checked = true;
                None
            })
            .filter_map(std::future::ready)
        };
        let put = self
            .backup
            .put_parts(&timeline.frame(&key), tapped.chain(checked).boxed())
            .await;
        let spilling = std::mem::replace(
            &mut *spilling.lock().expect("spilling never panics"),
            Spilling::spent(),
        );
        if spilling.missing {
            return Ok(Err(Stall::PrimaryMissing { key }));
        }
        if let Some(stall) = spilling.stall {
            return Ok(Err(stall));
        }
        put?;
        if !spilling.checked {
            return Err(StorageError::Blob {
                key: timeline.frame(&key),
                source: "the spill was put before all of its frame had come".into(),
            });
        }
        Ok(Ok((Sum(hex(&spilling.hasher.finalize())), spilling.size)))
    }

    /// Create the link of `header`, with `frame` if it is inline, once: a link
    /// already there passes only if it is of the same commit.
    async fn create(
        &self,
        timeline: &Timeline,
        base: &ShipBase,
        header: Header,
        frame: Option<&[u8]>,
    ) -> Result<std::result::Result<(), Stall>> {
        let position = header.position;
        let at = timeline.link(position);
        let encoded = link::encode(&header, frame)?;
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

/// What a log being spilled must say of itself: it is the head of the row
/// at `position`, made on `last`, the commit shipped just before it, if
/// the run has shipped one.
#[derive(Debug, Clone)]
struct Expected {
    key: String,
    position: LogPosition,
    last: Option<(LogPosition, String)>,
}

impl Expected {
    /// Why the run stalls, given what reading the log's start came to, if
    /// it does: none while more of it is needed.
    fn stall(&self, read: Result<Option<Found>>) -> Option<Stall> {
        match read {
            Ok(None) => None,
            Ok(Some(found)) => {
                let follows = match &self.last {
                    None => true,
                    Some((at, key)) => {
                        at.next() == Some(self.position) && found.prev.as_deref() == Some(key)
                    }
                };
                (found.position != self.position || !follows).then_some(Stall::NotTheRowsHead {
                    position: self.position,
                })
            }
            Err(error) => Some(Stall::Corrupt {
                key: self.key.clone(),
                why: why(error),
            }),
        }
    }
}

/// A log's frame as it is spilled: what it has come to so far.
struct Spilling {
    hasher: Sha256,
    size: u64,
    framed: Framed,
    /// Its start, until what it says is found and seen to be right.
    prefix: Option<Prefix>,
    /// The primary said it was not there.
    missing: bool,
    /// Why the run stalls, the upload given up for it.
    stall: Option<Stall>,
    /// All of it came, and every check held.
    checked: bool,
}

impl Spilling {
    fn spent() -> Self {
        Self {
            hasher: Sha256::new(),
            size: 0,
            framed: Framed::default(),
            prefix: None,
            missing: false,
            stall: None,
            checked: false,
        }
    }
}

/// Whether a log that says `found` of itself is the head of the row at
/// `position`, made on `last`, the commit shipped just before it: one
/// position past it, naming it as `prev`.
fn follows(found: &Found, position: LogPosition, last: Option<&Made>) -> bool {
    let made_on = last.is_none_or(|last| {
        last.position.next() == Some(position) && found.prev.as_deref() == Some(last.key.as_str())
    });
    found.position == position && made_on
}

/// The stall of a link the backup store cannot take.
fn not_linkable(position: LogPosition, unlinkable: Unlinkable) -> Stall {
    let why = match unlinkable {
        Unlinkable::NoCreate => {
            "the backup store does not create a file only if none is there".into()
        }
        Unlinkable::HeaderTooLarge { bytes, most } => {
            format!("its header is {bytes} bytes; the store creates {most} at most")
        }
    };
    Stall::NotCreated { position, why }
}
