//! A merge asked to stop (merge note, step 3, §7): from its plan to the
//! end of loading what it built back, it asks its stop in each long stage,
//! and gives up, leaving nothing behind, once it is stopped.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use vellev_index::stop::{Stage, Stop};

use crate::state::TOKENIZED_AT_ONCE;

/// How many rows' text a build holds tokenized at once, at the most, as a
/// test asks: see [`tokenize_at_once`].
static ASKED_AT_ONCE: AtomicUsize = AtomicUsize::new(TOKENIZED_AT_ONCE);

/// Have every merge, flush and open from now on hold `rows` rows' text
/// tokenized at once, at the most. For rules that weigh a build tokenized in
/// chunks against one tokenized all at once, in a binary of their own, one
/// measurement at a time: it is the process's, as an open has no writer to
/// ask, and what it weighs, the allocator's count, is the process's too. Not
/// for use.
#[doc(hidden)]
pub fn tokenize_at_once(rows: usize) {
    ASKED_AT_ONCE.store(rows.max(1), Ordering::SeqCst);
}

/// The stages of a merge in which it asks its stop. Named for tests that
/// show each one asks; not for use.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MergeStage {
    /// Before it plans, once its turn to merge has come.
    Planning,
    /// Waiting for a turn among the node's merge builds, before it gathers,
    /// or, with one or without, for its partition's building lock.
    Queued,
    /// Waiting for a turn among the work done off the async threads.
    Turn,
    /// Copying the run's live rows out, before each chunk of them.
    Gathering,
    /// Indexing the rows' filterable values and text.
    Indexing,
    /// Indexing the vectors, in the index build's own stage.
    Vectors(Stage),
    /// Encoding the segment, its vectors and its index.
    Encoding,
    /// Uploading its files, before each and while each goes up.
    Uploading,
    /// Loading what it built back from local disk.
    Loading,
    /// Summing what it built, read back in place, before each piece.
    Summing,
}

/// How often a merge waiting on the store or the disk asks its stop.
const ASKS_EVERY: Duration = Duration::from_millis(20);

/// A build's stop, the stage a test has it stop itself in, and what else a
/// test asks of the build.
#[derive(Clone, Debug)]
pub(crate) struct Stopping {
    stop: Stop,
    at: Option<MergeStage>,
    /// Whether the build panics, as a test asks.
    panics: bool,
    /// How many rows' text the build holds tokenized at once, at the most:
    /// [`TOKENIZED_AT_ONCE`], unless a test asks otherwise. Rows, not bytes:
    /// a chunk of long documents holds what they come to.
    tokenized_at_once: usize,
    /// Its askings, counted as a test asks.
    #[cfg(test)]
    asked: Option<std::sync::Arc<Asked>>,
}

/// How often a build asked its stop in any stage, and the asking at which
/// the stop stops itself.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Asked {
    pub times: AtomicUsize,
    pub stops_at: usize,
}

impl Stopping {
    /// For a merge, stopped by whoever holds a clone of [`Self::stop`], or
    /// by itself at the first asking in `at`.
    pub(crate) fn merge(at: Option<MergeStage>, panics: bool) -> Self {
        let stop = match at {
            Some(MergeStage::Vectors(stage)) => Stop::at_first(stage),
            _ => Stop::new(),
        };
        Self {
            stop,
            at,
            panics,
            tokenized_at_once: ASKED_AT_ONCE.load(Ordering::SeqCst),
            #[cfg(test)]
            asked: None,
        }
    }

    /// For a flush or a restore, which nothing stops.
    pub(crate) fn never() -> Self {
        Self {
            stop: Stop::new(),
            at: None,
            panics: false,
            tokenized_at_once: ASKED_AT_ONCE.load(Ordering::SeqCst),
            #[cfg(test)]
            asked: None,
        }
    }

    /// As this, holding `rows` rows' text tokenized at once, at the most,
    /// and at least one.
    #[cfg(test)]
    pub(crate) fn tokenizing(self, rows: usize) -> Self {
        Self {
            tokenized_at_once: rows.max(1),
            ..self
        }
    }

    /// As this, its askings counted, and stopped at the `stops_at`th.
    #[cfg(test)]
    pub(crate) fn counted(self, stops_at: usize) -> (Self, std::sync::Arc<Asked>) {
        let asked = std::sync::Arc::new(Asked {
            times: AtomicUsize::new(0),
            stops_at,
        });
        let stopping = Self {
            asked: Some(asked.clone()),
            ..self
        };
        (stopping, asked)
    }

    /// The stop an index build is given.
    pub(crate) fn stop(&self) -> &Stop {
        &self.stop
    }

    /// Asked in `stage`: whether to give up.
    pub(crate) fn stopped(&self, stage: MergeStage) -> bool {
        if self.at == Some(stage) {
            self.stop.stop();
        }
        #[cfg(test)]
        if let Some(asked) = &self.asked {
            let times = asked.times.fetch_add(1, Ordering::SeqCst);
            if times + 1 >= asked.stops_at {
                self.stop.stop();
            }
        }
        self.stop.is_stopped()
    }

    /// How many rows' text the build holds tokenized at once, at the most.
    pub(crate) fn tokenized_at_once(&self) -> usize {
        self.tokenized_at_once
    }

    /// Whether the build is to panic, as a test asks.
    pub(crate) fn panics(&self) -> bool {
        self.panics
    }

    /// Whether it was stopped, asked by no stage.
    pub(crate) fn is_stopped(&self) -> bool {
        self.stop.is_stopped()
    }

    /// `work`'s answer, or `None` if it is stopped first, in `stage`: the
    /// work is then dropped where it waits.
    pub(crate) async fn unless_stopped<T>(
        &self,
        stage: MergeStage,
        work: impl std::future::Future<Output = T>,
    ) -> Option<T> {
        if self.stopped(stage) {
            return None;
        }
        let stopped = async {
            while !self.is_stopped() {
                tokio::time::sleep(ASKS_EVERY).await;
            }
        };
        // Unbiased: should both be ready at once, either way is sound. Work
        // that came to an end unseen, as an upload that landed, is let go
        // of only once whoever began it has noted it to be undone, as each
        // file is noted before it goes up.
        tokio::select! {
            done = work => Some(done),
            () = stopped => None,
        }
    }
}
