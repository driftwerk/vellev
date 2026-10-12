//! What a restore from shipped links needs, and the chain of links it
//! walks to know it (DW-668, amendment 1). Pure: it reads headers already
//! decoded, and the checkpoint's manifest its caller decoded.
//!
//! A link's `before` names the commit before it in its incarnation: the
//! previous row of `vellev.commits`, by position and frame sum. Positions
//! may skip (a restore commits at the position after the greater of the
//! current one and the one it restores), so a chain is walked by
//! `before`, never by counting positions.
//!
//! To restore the point `target`:
//!
//! - The checkpoint is the newest manifest link at or below `target`. Its
//!   manifest says which logs it replays: those just after `replay_after`
//!   up to just before itself, by key. The needed links run from the
//!   first of them (or the checkpoint, if it replays none) up to
//!   `target`, and the checkpoint's files are needed with them. Loading a
//!   checkpoint never reads the checkpoint before it.
//! - With no checkpoint at or below `target`, the needed links run from
//!   the very first commit, a log at position 1 with nothing before it,
//!   as only that can be replayed from an empty partition.
//!
//! The walk goes down from `target` by `before` and stops at the start it
//! needs, whatever that link's own `before` says: a timeline may begin
//! at a link whose commit before was never shipped. Each step must name
//! a link that is there, of the same timeline, at an earlier position,
//! of the frame sum it says; the walk may not pass below the start, nor
//! end above it, nor miss the checkpoint; and the logs it walks below the
//! checkpoint must be the ones the checkpoint replays, key by key.
//!
//! `before: None` is a commit with nothing before it in its incarnation.
//! The walk takes it as an end only where it may be one: at the start,
//! or, with no checkpoint, as a log at position 1. Whether a chain's
//! commit times run forward is for the shipper and for finding a
//! position by time; a restore by position does not depend on it.

use std::collections::BTreeMap;

use crate::coordinator::LogPosition;
use crate::format::Manifest;
use crate::link::{Head, Header, Needed, Timeline};

/// What restoring a point from shipped links needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restorable {
    /// The checkpoint loaded, if any.
    pub checkpoint: Option<LogPosition>,
    /// The links whose frames are needed, oldest first.
    pub links: Vec<LogPosition>,
    /// The checkpoint's files.
    pub files: Vec<Needed>,
}

/// Why a point cannot be restored from the links given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Broken {
    /// No link at `position`, which `named_by` says is before it, or the
    /// target if none.
    Missing {
        position: LogPosition,
        named_by: Option<LogPosition>,
    },
    /// The link given at `position` says it is at `says`.
    Misfiled {
        position: LogPosition,
        says: LogPosition,
    },
    /// The link at `position` is of another timeline than the target.
    OtherTimeline { position: LogPosition },
    /// The link at `position` names a commit before it that is not
    /// earlier.
    NotEarlier { position: LogPosition },
    /// The link at `position` names a commit before it whose frame sum is
    /// not the one that link has.
    OtherSum { position: LogPosition },
    /// The walk passed below `start` from the link at `position`.
    Passed {
        position: LogPosition,
        start: LogPosition,
    },
    /// The link at `position` has nothing before it, above `start`.
    BeginsAbove {
        position: LogPosition,
        start: LogPosition,
    },
    /// With no checkpoint, the walk ended at `position`, which is not the
    /// first commit: a log at position 1.
    NoBeginning { position: LogPosition },
    /// The walk from the target did not pass through the checkpoint.
    MissedCheckpoint { checkpoint: LogPosition },
    /// The manifest given is not the checkpoint's, or was given or not
    /// given against whether there is one.
    OtherCheckpoint { checkpoint: Option<LogPosition> },
    /// The checkpoint's replay does not end just before it.
    ReplayOutOfPlace { checkpoint: LogPosition },
    /// The log at `position` is not the one the checkpoint replays there.
    OtherReplay { position: LogPosition },
}

/// The link at `position`, as one of the target's timeline filed by its
/// own position; `named_by` is the link that named it, if any.
fn visit<'a>(
    links: &'a BTreeMap<LogPosition, Header>,
    timeline: &Timeline,
    position: LogPosition,
    named_by: Option<LogPosition>,
) -> Result<&'a Header, Broken> {
    let header = links
        .get(&position)
        .ok_or(Broken::Missing { position, named_by })?;
    if header.position != position {
        return Err(Broken::Misfiled {
            position,
            says: header.position,
        });
    }
    if Timeline::of(header) != *timeline {
        return Err(Broken::OtherTimeline { position });
    }
    Ok(header)
}

/// The target's link and its timeline.
fn target_of(
    links: &BTreeMap<LogPosition, Header>,
    target: LogPosition,
) -> Result<(&Header, Timeline), Broken> {
    let header = links.get(&target).ok_or(Broken::Missing {
        position: target,
        named_by: None,
    })?;
    let timeline = Timeline::of(header);
    Ok((visit(links, &timeline, target, None)?, timeline))
}

/// The newest manifest link at or below `target`: the checkpoint a
/// restore of it loads.
pub fn checkpoint_for(
    links: &BTreeMap<LogPosition, Header>,
    target: LogPosition,
) -> Result<Option<&Header>, Broken> {
    let (_, timeline) = target_of(links, target)?;
    let newest = links
        .range(..=target)
        .rev()
        .find(|(_, header)| header.head == Head::Manifest);
    match newest {
        Some((position, _)) => visit(links, &timeline, *position, None).map(Some),
        None => Ok(None),
    }
}

/// The links from `target` down to `start` by `before`, oldest first, or,
/// with no start, down to the first commit: a log at position 1 with
/// nothing before it. Every step goes to an earlier position, so the walk
/// ends.
fn walk(
    links: &BTreeMap<LogPosition, Header>,
    target: LogPosition,
    start: Option<LogPosition>,
) -> Result<Vec<LogPosition>, Broken> {
    let (mut header, timeline) = target_of(links, target)?;
    let mut walked = vec![target];
    loop {
        let position = header.position;
        if Some(position) == start {
            break;
        }
        let Some(before) = &header.before else {
            return Err(match start {
                Some(start) => Broken::BeginsAbove { position, start },
                // With no checkpoint at or below the target, every link
                // walked is a log.
                None if position == LogPosition(1) => break,
                None => Broken::NoBeginning { position },
            });
        };
        if before.position >= position {
            return Err(Broken::NotEarlier { position });
        }
        if let Some(start) = start.filter(|start| before.position < *start) {
            return Err(Broken::Passed { position, start });
        }
        let earlier = visit(links, &timeline, before.position, Some(position))?;
        if earlier.sum != before.sum {
            return Err(Broken::OtherSum { position });
        }
        header = earlier;
        walked.push(header.position);
    }
    walked.reverse();
    Ok(walked)
}

/// What restoring `target` from `links`, the links of one timeline by
/// position, needs; `checkpoint` is the manifest of the link
/// [`checkpoint_for`] gives, decoded, or none if it gives none.
pub fn needed(
    links: &BTreeMap<LogPosition, Header>,
    target: LogPosition,
    checkpoint: Option<&Manifest>,
) -> Result<Restorable, Broken> {
    let found = checkpoint_for(links, target)?;
    let (found, manifest) = match (found, checkpoint) {
        (None, None) => {
            return Ok(Restorable {
                checkpoint: None,
                links: walk(links, target, None)?,
                files: Vec::new(),
            })
        }
        (Some(found), Some(manifest)) if manifest.position == found.position => (found, manifest),
        (found, _) => {
            return Err(Broken::OtherCheckpoint {
                checkpoint: found.map(|found| found.position),
            })
        }
    };
    let at = found.position;
    let replayed = u64::try_from(manifest.replay.len()).unwrap_or(u64::MAX);
    let start = match replayed {
        0 => at,
        _ if manifest.replay_after.0.checked_add(replayed + 1) != Some(at.0) => {
            return Err(Broken::ReplayOutOfPlace { checkpoint: at });
        }
        _ => LogPosition(manifest.replay_after.0 + 1),
    };
    let walked = walk(links, target, Some(start))?;
    let Some(checkpoint_at) = walked.iter().position(|position| *position == at) else {
        return Err(Broken::MissedCheckpoint { checkpoint: at });
    };
    let replay = &walked[..checkpoint_at];
    if replay.len() != manifest.replay.len() {
        let position = replay.first().copied().unwrap_or(at);
        return Err(Broken::OtherReplay { position });
    }
    for (position, key) in replay.iter().zip(&manifest.replay) {
        if links[position].key != *key {
            return Err(Broken::OtherReplay {
                position: *position,
            });
        }
    }
    Ok(Restorable {
        checkpoint: Some(at),
        links: walked,
        files: found.files.clone(),
    })
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use uuid::Uuid;
    use vellev_core::{Partition as PartitionNumber, TenantId};

    use super::*;
    use crate::coordinator::{ClusterId, Generation, Incarnation, Lineage};
    use crate::link::{Before, Body, Sum};

    const WRITER: &str = "8560994bc5f64972b4d9de1d3358753b";

    fn key(head: Head, position: u64) -> String {
        let kind = match head {
            Head::Log => "log",
            Head::Manifest => "manifest",
        };
        format!("tenant-a/0/{kind}/{position:020}-00000000000000000002-{WRITER}.{kind}")
    }

    fn sum(position: u64) -> Sum {
        Sum::of(format!("the frame at {position}").as_bytes())
    }

    fn files(position: u64) -> Vec<Needed> {
        ["segment", "index", "vectors"]
            .map(|kind| Needed {
                key: format!(
                    "tenant-a/0/{kind}/{position:020}-00000000000000000002-{WRITER}.{kind}"
                ),
                size: 10,
                sum: Sum::of(kind.as_bytes()),
            })
            .to_vec()
    }

    /// The link at `position` of a `head`, made on the commit at `before`.
    fn link(position: u64, head: Head, before: Option<u64>) -> Header {
        Header {
            tenant: TenantId::new("tenant-a").unwrap(),
            lineage: Lineage(Uuid::from_u128(1)),
            cluster: ClusterId(Uuid::from_u128(2)),
            incarnation: Incarnation(3),
            partition: PartitionNumber(0),
            position: LogPosition(position),
            key: key(head, position),
            head,
            generation: Generation(2),
            committed_at: chrono::Utc.with_ymd_and_hms(2026, 10, 12, 9, 0, 0).unwrap(),
            sum: sum(position),
            size: 10,
            before: before.map(|before| Before {
                position: LogPosition(before),
                sum: sum(before),
            }),
            files: match head {
                Head::Log => Vec::new(),
                Head::Manifest => files(position),
            },
            body: Body::Inline,
        }
    }

    fn log(position: u64, before: Option<u64>) -> Header {
        link(position, Head::Log, before)
    }

    fn checkpoint(position: u64, before: Option<u64>) -> Header {
        link(position, Head::Manifest, before)
    }

    /// Logs from 1 to `last`, each on the one before.
    fn logs(last: u64) -> Vec<Header> {
        (1..=last)
            .map(|position| log(position, (position > 1).then(|| position - 1)))
            .collect()
    }

    fn chain(links: impl IntoIterator<Item = Header>) -> BTreeMap<LogPosition, Header> {
        links
            .into_iter()
            .map(|link| (link.position, link))
            .collect()
    }

    /// The manifest of the checkpoint at `position`, replaying the logs at
    /// `replay` by their keys.
    fn manifest(position: u64, replay: &[u64]) -> Manifest {
        Manifest {
            position: LogPosition(position),
            prev: None,
            next_collection_id: 0,
            collections: Vec::new(),
            segments: Vec::new(),
            replay: replay.iter().map(|at| key(Head::Log, *at)).collect(),
            replay_after: LogPosition(replay.first().map_or(0, |first| first - 1)),
            answered: Vec::new(),
        }
    }

    fn at(positions: &[u64]) -> Vec<LogPosition> {
        positions.iter().copied().map(LogPosition).collect()
    }

    fn restorable(checkpoint: Option<u64>, links: &[u64]) -> Restorable {
        Restorable {
            checkpoint: checkpoint.map(LogPosition),
            links: at(links),
            files: checkpoint.map(files).unwrap_or_default(),
        }
    }

    /// Logs 1 to 4, a checkpoint at 5 replaying 3 and 4, logs 6 and 7.
    fn replaying() -> Vec<Header> {
        let mut links = logs(4);
        links.extend([checkpoint(5, Some(4)), log(6, Some(5)), log(7, Some(6))]);
        links
    }

    fn with(links: Vec<Header>, changed: Header) -> BTreeMap<LogPosition, Header> {
        let mut links = chain(links);
        links.insert(changed.position, changed);
        links
    }

    #[test]
    fn the_checkpoint_is_the_newest_manifest_at_or_below_the_target() {
        let mut links = logs(2);
        links.extend([checkpoint(3, Some(2)), log(4, Some(3)), log(5, Some(4))]);
        links.extend([checkpoint(6, Some(5)), log(7, Some(6)), log(8, Some(7))]);
        let links = chain(links);
        let found = |target| {
            checkpoint_for(&links, LogPosition(target))
                .unwrap()
                .map(|header| header.position.0)
        };
        assert_eq!(found(8), Some(6));
        assert_eq!(found(6), Some(6));
        assert_eq!(found(5), Some(3));
        assert_eq!(found(2), None);
        let got = needed(&links, LogPosition(8), Some(&manifest(6, &[]))).unwrap();
        assert_eq!(got, restorable(Some(6), &[6, 7, 8]));
    }

    #[test]
    fn with_no_checkpoint_every_log_from_the_first_commit_is_needed() {
        let links = chain(logs(5));
        assert_eq!(
            needed(&links, LogPosition(5), None).unwrap(),
            restorable(None, &[1, 2, 3, 4, 5])
        );
        assert_eq!(
            needed(&links, LogPosition(3), None).unwrap(),
            restorable(None, &[1, 2, 3])
        );
    }

    #[test]
    fn a_chain_is_walked_by_before_and_may_skip_positions() {
        // A log at 5 made on the commit at 2.
        let links = chain([
            log(1, None),
            log(2, Some(1)),
            log(5, Some(2)),
            log(6, Some(5)),
        ]);
        assert_eq!(
            needed(&links, LogPosition(6), None).unwrap(),
            restorable(None, &[1, 2, 5, 6])
        );
        // A restore checkpoint at 7 made on the commit at 3, as a restore
        // to a later position commits.
        let mut links = logs(3);
        links.extend([checkpoint(7, Some(3)), log(8, Some(7))]);
        let got = needed(&chain(links), LogPosition(8), Some(&manifest(7, &[]))).unwrap();
        assert_eq!(got, restorable(Some(7), &[7, 8]));
    }

    #[test]
    fn a_checkpoint_needs_the_logs_it_replays_and_its_files() {
        let links = chain(replaying());
        let replay = manifest(5, &[3, 4]);
        assert_eq!(
            needed(&links, LogPosition(7), Some(&replay)).unwrap(),
            restorable(Some(5), &[3, 4, 5, 6, 7])
        );
        // The target the checkpoint itself.
        assert_eq!(
            needed(&links, LogPosition(5), Some(&replay)).unwrap(),
            restorable(Some(5), &[3, 4, 5])
        );
    }

    #[test]
    fn a_checkpoint_that_replays_nothing_needs_no_log_before_it() {
        let mut links = logs(4);
        links.extend([checkpoint(5, Some(4)), log(6, Some(5))]);
        let got = needed(&chain(links), LogPosition(6), Some(&manifest(5, &[]))).unwrap();
        assert_eq!(got, restorable(Some(5), &[5, 6]));
    }

    #[test]
    fn a_timeline_may_begin_where_its_restore_starts_whatever_came_before() {
        // An adopted tenant's first commit, a checkpoint with nothing before.
        let links = chain([checkpoint(11, None), log(12, Some(11))]);
        let got = needed(&links, LogPosition(12), Some(&manifest(11, &[]))).unwrap();
        assert_eq!(got, restorable(Some(11), &[11, 12]));
        // A base whose commit before was never shipped.
        let links = chain([log(4, Some(3)), checkpoint(5, Some(4)), log(6, Some(5))]);
        let got = needed(&links, LogPosition(6), Some(&manifest(5, &[4]))).unwrap();
        assert_eq!(got, restorable(Some(5), &[4, 5, 6]));
    }

    #[test]
    fn with_no_checkpoint_the_chain_must_reach_a_log_at_position_1() {
        let links = chain([log(4, None), log(5, Some(4))]);
        assert_eq!(
            needed(&links, LogPosition(5), None),
            Err(Broken::NoBeginning {
                position: LogPosition(4)
            })
        );
        let links = chain([log(4, Some(3)), log(5, Some(4))]);
        assert_eq!(
            needed(&links, LogPosition(5), None),
            Err(Broken::Missing {
                position: LogPosition(3),
                named_by: Some(LogPosition(4))
            })
        );
    }

    #[test]
    fn nothing_before_a_link_above_where_the_walk_must_go_is_refused() {
        let links = with(replaying(), log(4, None));
        assert_eq!(
            needed(&links, LogPosition(7), Some(&manifest(5, &[3, 4]))),
            Err(Broken::BeginsAbove {
                position: LogPosition(4),
                start: LogPosition(3)
            })
        );
        // A checkpoint with nothing before it that replays something.
        let links = chain([log(3, Some(2)), log(4, Some(3)), checkpoint(5, None)]);
        assert_eq!(
            needed(&links, LogPosition(5), Some(&manifest(5, &[3, 4]))),
            Err(Broken::BeginsAbove {
                position: LogPosition(5),
                start: LogPosition(3)
            })
        );
    }

    #[test]
    fn a_replay_that_starts_below_what_was_shipped_is_refused() {
        let links = chain([log(4, Some(3)), checkpoint(5, Some(4)), log(6, Some(5))]);
        assert_eq!(
            needed(&links, LogPosition(6), Some(&manifest(5, &[3, 4]))),
            Err(Broken::Missing {
                position: LogPosition(3),
                named_by: Some(LogPosition(4))
            })
        );
    }

    #[test]
    fn a_hole_is_refused_naming_what_is_missing_and_what_named_it() {
        let mut links = chain(logs(5));
        links.remove(&LogPosition(3));
        assert_eq!(
            needed(&links, LogPosition(5), None),
            Err(Broken::Missing {
                position: LogPosition(3),
                named_by: Some(LogPosition(4))
            })
        );
        assert_eq!(
            needed(&links, LogPosition(9), None),
            Err(Broken::Missing {
                position: LogPosition(9),
                named_by: None
            })
        );
    }

    #[test]
    fn a_commit_before_of_another_frame_or_not_earlier_is_refused() {
        let mut other = log(4, Some(3));
        other.before = Some(Before {
            position: LogPosition(3),
            sum: Sum::of(b"another frame"),
        });
        assert_eq!(
            needed(&with(logs(5), other), LogPosition(5), None),
            Err(Broken::OtherSum {
                position: LogPosition(4)
            })
        );
        for before in [4, 9] {
            let looped = with(logs(5), log(4, Some(before)));
            // Run apart and waited on for a while, so that a walk that
            // never ends fails rather than hangs.
            let (done, came) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = done.send(needed(&looped, LogPosition(5), None));
            });
            let walked = came
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the walk ends");
            assert_eq!(
                walked,
                Err(Broken::NotEarlier {
                    position: LogPosition(4)
                }),
                "{before}"
            );
        }
    }

    #[test]
    fn the_walk_may_neither_pass_below_its_start_nor_miss_the_checkpoint() {
        // The log at 4 made on 2, past the first log replayed, 3.
        let links = with(replaying(), log(4, Some(2)));
        assert_eq!(
            needed(&links, LogPosition(7), Some(&manifest(5, &[3, 4]))),
            Err(Broken::Passed {
                position: LogPosition(4),
                start: LogPosition(3)
            })
        );
        // The log at 6 made on 4, past the checkpoint at 5.
        let links = with(replaying(), log(6, Some(4)));
        assert_eq!(
            needed(&links, LogPosition(7), Some(&manifest(5, &[3, 4]))),
            Err(Broken::MissedCheckpoint {
                checkpoint: LogPosition(5)
            })
        );
        // Nor, replaying nothing, pass below the checkpoint itself.
        let mut links = logs(4);
        links.extend([checkpoint(5, Some(4)), log(6, Some(4))]);
        assert_eq!(
            needed(&chain(links), LogPosition(6), Some(&manifest(5, &[]))),
            Err(Broken::Passed {
                position: LogPosition(6),
                start: LogPosition(5)
            })
        );
    }

    #[test]
    fn the_logs_below_the_checkpoint_must_be_the_ones_it_replays() {
        let mut other = log(4, Some(3));
        other.key = other
            .key
            .replace(WRITER, "0123456789abcdef0123456789abcdef");
        let links = with(replaying(), other);
        assert_eq!(
            needed(&links, LogPosition(7), Some(&manifest(5, &[3, 4]))),
            Err(Broken::OtherReplay {
                position: LogPosition(4)
            })
        );
        // A replay that does not end just before the checkpoint.
        let mut out_of_place = manifest(5, &[3, 4]);
        out_of_place.replay_after = LogPosition(1);
        assert_eq!(
            needed(&chain(replaying()), LogPosition(7), Some(&out_of_place)),
            Err(Broken::ReplayOutOfPlace {
                checkpoint: LogPosition(5)
            })
        );
        let mut short = manifest(5, &[3, 4]);
        short.replay.pop();
        assert_eq!(
            needed(&chain(replaying()), LogPosition(7), Some(&short)),
            Err(Broken::ReplayOutOfPlace {
                checkpoint: LogPosition(5)
            })
        );
    }

    #[test]
    fn the_manifest_given_must_be_the_checkpoints() {
        let links = chain(replaying());
        assert_eq!(
            needed(&links, LogPosition(7), Some(&manifest(6, &[]))),
            Err(Broken::OtherCheckpoint {
                checkpoint: Some(LogPosition(5))
            })
        );
        assert_eq!(
            needed(&links, LogPosition(7), None),
            Err(Broken::OtherCheckpoint {
                checkpoint: Some(LogPosition(5))
            })
        );
        assert_eq!(
            needed(&chain(logs(3)), LogPosition(3), Some(&manifest(3, &[]))),
            Err(Broken::OtherCheckpoint { checkpoint: None })
        );
    }

    #[test]
    fn every_link_walked_or_taken_as_the_checkpoint_is_of_the_targets_timeline_and_filed_by_its_position(
    ) {
        let mut other = log(3, Some(2));
        other.incarnation = Incarnation(4);
        assert_eq!(
            needed(&with(logs(5), other), LogPosition(5), None),
            Err(Broken::OtherTimeline {
                position: LogPosition(3)
            })
        );
        let mut foreign = checkpoint(6, Some(5));
        foreign.lineage = Lineage(Uuid::from_u128(9));
        let mut links = chain(replaying());
        links.insert(LogPosition(6), foreign);
        assert_eq!(
            checkpoint_for(&links, LogPosition(7)),
            Err(Broken::OtherTimeline {
                position: LogPosition(6)
            })
        );

        let mut links = chain(logs(5));
        links.insert(LogPosition(3), log(2, Some(1)));
        assert_eq!(
            needed(&links, LogPosition(5), None),
            Err(Broken::Misfiled {
                position: LogPosition(3),
                says: LogPosition(2)
            })
        );
    }
}
