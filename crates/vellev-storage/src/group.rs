use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Notify};

use crate::mutation::{Batch, Mutation, Receipt, Weight};
use crate::partition::{Embedded, Writer};
use crate::{Result, StorageError};

struct Request {
    batch: Batch,
    made: Embedded,
    reply: oneshot::Sender<Result<Receipt>>,
}

/// The most one commit takes from the queue, in mutations and in their
/// bytes, however many batches that is. A group is one log file, written
/// in one request and then applied in memory while every other write, and
/// for the last of it every read, waits. This much makes a file that goes
/// up in one request, with room to spare under the 16 MiB a part of an
/// upload is by default ([`Transfers`](crate::blobs::Transfers)), and is
/// resolved, encoded and applied in a sixth of a second: 0.15s for 2,310
/// rows of 12 MiB and 0.16s for 5,310, of which the last 0.1s held up
/// reads, and 0.02s for 8,192 deletes, on a tenant of 4.3 million rows
/// (`examples/apply_timing.rs`).
///
/// A batch is never split, so one that is more than this on its own is
/// committed alone: what a single caller may ask for is for whoever takes
/// its request to bound, as the API does by the size of a request's body.
pub const GROUP_MOST: Weight = Weight {
    rows: 8_192,
    bytes: 12 * 1024 * 1024,
};

/// A queue in front of a writer that lets concurrent callers share commits.
/// While one commit is in flight, later submissions wait; the next commit
/// takes what is waiting, up to a limit, as one group. Nothing is delayed
/// to make a group bigger: a lone caller commits at once.
#[derive(Debug, Clone)]
pub struct GroupCommitter {
    queue: mpsc::Sender<Request>,
    stop: Arc<Notify>,
}

impl GroupCommitter {
    /// Start committing for `writer`. At most `max_group` batches share one
    /// commit, and no more of them than keep it within [`GROUP_MOST`], and
    /// at most `queue` wait before a submitter is held back. The task ends
    /// when the last clone of the committer is dropped, or when it is
    /// stopped.
    ///
    /// # Panics
    ///
    /// Outside a Tokio runtime, because the queue is drained by a spawned task.
    pub fn spawn(writer: Arc<Writer>, max_group: usize, queue: usize) -> Self {
        let (sender, mut waiting) = mpsc::channel::<Request>(queue.max(1));
        let stop = Arc::new(Notify::new());
        let stopped = stop.clone();
        tokio::spawn(async move {
            // The batch taken from the queue that the group before had no
            // room for, which the next begins with.
            let mut left_over: Option<Request> = None;
            loop {
                let first = tokio::select! {
                    biased;
                    () = stopped.notified() => return,
                    first = async {
                        match left_over.take() {
                            Some(first) => Some(first),
                            None => waiting.recv().await,
                        }
                    } => match first {
                        Some(first) => first,
                        None => return,
                    },
                };
                let mut weight = first.batch.weight();
                let mut group = vec![first];
                while group.len() < max_group.max(1) {
                    let Ok(next) = waiting.try_recv() else { break };
                    weight = weight + next.batch.weight();
                    if !weight.within(&GROUP_MOST) {
                        left_over = Some(next);
                        break;
                    }
                    group.push(next);
                }
                let mut made = Embedded::new();
                let (batches, replies): (Vec<_>, Vec<_>) = group
                    .into_iter()
                    .filter(|request| !request.reply.is_closed())
                    .map(|request| {
                        made.extend(request.made);
                        (request.batch, request.reply)
                    })
                    .unzip();
                let shared = |error: StorageError| {
                    let shared = Arc::new(error);
                    replies
                        .iter()
                        .map(|_| Err(StorageError::Shared(shared.clone())))
                        .collect::<Vec<_>>()
                };
                // The group fails as a whole only when its commit failed,
                // which closes the writer, so every caller shares the one
                // failure; a batch refused or not made fails on its own.
                // A commit given up where it is by a stop is such a
                // failure too: whether it was made is unknown.
                let (results, go_on) = tokio::select! {
                    biased;
                    () = stopped.notified() => (shared(StorageError::Closed), false),
                    committed = writer.commit_batches(batches, made) => match committed {
                        Ok(results) => (results, true),
                        Err(StorageError::Closed) => {
                            let closed = replies.iter().map(|_| Err(StorageError::Closed));
                            (closed.collect(), true)
                        }
                        Err(error) => (shared(error), true),
                    },
                };
                for (reply, result) in replies.into_iter().zip(results) {
                    let _ = reply.send(result);
                }
                if !go_on {
                    return;
                }
            }
        });
        Self {
            queue: sender,
            stop,
        }
    }

    /// Stop committing, and let go of the writer and all it holds. A commit
    /// under way is given up where it is, which leaves the writer closed if
    /// it had begun to be made, and its callers are told its outcome is
    /// unknown; whoever waits behind it, or comes later, is told the writer
    /// is closed, nothing of theirs having been made.
    pub fn stop(&self) {
        self.stop.notify_one();
    }

    /// Commit `mutations` atomically, together with whatever else is waiting.
    ///
    /// A caller that stops waiting before its batch is taken into a commit is
    /// skipped. One that stops waiting after that point cannot be: its batch
    /// is committed all the same, so a caller that retries after a timeout
    /// must name its request (see [`Self::submit_batch`]) or use mutations
    /// that are safe to repeat. A failure of the shared
    /// commit arrives as [`StorageError::Shared`]; see [`StorageError::root`].
    pub async fn submit(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        self.submit_with(mutations, Embedded::new()).await
    }

    /// As [`Self::submit`], with vectors made ahead for the commit to use;
    /// see [`Writer::commit_group_with`].
    pub async fn submit_with(&self, mutations: Vec<Mutation>, made: Embedded) -> Result<Receipt> {
        self.submit_batch(mutations.into(), made).await
    }

    /// As [`Self::submit_with`], for a batch that may name its request: see
    /// [`Writer::commit_batches`].
    pub async fn submit_batch(&self, batch: Batch, made: Embedded) -> Result<Receipt> {
        let (reply, outcome) = oneshot::channel();
        let request = Request { batch, made, reply };
        self.queue
            .send(request)
            .await
            .map_err(|_| StorageError::Closed)?;
        outcome.await.map_err(|_| StorageError::Closed)?
    }
}
