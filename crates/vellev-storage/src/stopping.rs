//! A merge asked to stop (merge note, step 3, §7): from its plan to the
//! end of loading what it built back, it asks its stop in each long stage,
//! and gives up, leaving nothing behind, once it is stopped.

use std::time::Duration;

use vellev_index::stop::{Stage, Stop};

/// The stages of a merge in which it asks its stop. Named for tests that
/// show each one asks; not for use.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MergeStage {
    /// Before it plans, once its turn to merge has come.
    Planning,
    /// Copying the run's live rows out, between its segments.
    Gathering,
    /// Waiting for a turn among the work done off the async threads.
    Turn,
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
}

/// How often a merge waiting on the store or the disk asks its stop.
const ASKS_EVERY: Duration = Duration::from_millis(20);

/// A build's stop, and the stage a test has it stop itself in.
#[derive(Clone, Debug)]
pub(crate) struct Stopping {
    stop: Stop,
    at: Option<MergeStage>,
    /// Whether the build panics, as a test asks.
    panics: bool,
}

impl Stopping {
    /// For a merge, stopped by whoever holds a clone of [`Self::stop`], or
    /// by itself at the first asking in `at`.
    pub(crate) fn merge(at: Option<MergeStage>, panics: bool) -> Self {
        let stop = match at {
            Some(MergeStage::Vectors(stage)) => Stop::at_first(stage),
            _ => Stop::new(),
        };
        Self { stop, at, panics }
    }

    /// For a flush or a restore, which nothing stops.
    pub(crate) fn never() -> Self {
        Self {
            stop: Stop::new(),
            at: None,
            panics: false,
        }
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
        self.stop.is_stopped()
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
