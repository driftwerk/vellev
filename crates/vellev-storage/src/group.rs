use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use crate::mutation::{Mutation, Receipt};
use crate::partition::Writer;
use crate::{Result, StorageError};

struct Request {
    mutations: Vec<Mutation>,
    reply: oneshot::Sender<Result<Receipt>>,
}

/// A queue in front of a writer that lets concurrent callers share commits.
/// While one commit is in flight, later submissions wait; the next commit
/// takes everything waiting, up to a limit, as one group. Nothing is delayed
/// to make a group bigger: a lone caller commits at once.
#[derive(Debug, Clone)]
pub struct GroupCommitter {
    queue: mpsc::Sender<Request>,
}

impl GroupCommitter {
    /// Start committing for `writer`. At most `max_group` batches, whatever
    /// their size, share one commit, and at most `queue` wait before a
    /// submitter is held back. The task ends when the last clone of the
    /// committer is dropped.
    ///
    /// # Panics
    ///
    /// Outside a Tokio runtime, because the queue is drained by a spawned task.
    pub fn spawn(writer: Arc<Writer>, max_group: usize, queue: usize) -> Self {
        let (sender, mut waiting) = mpsc::channel::<Request>(queue.max(1));
        tokio::spawn(async move {
            while let Some(first) = waiting.recv().await {
                let mut group = vec![first];
                while group.len() < max_group.max(1) {
                    match waiting.try_recv() {
                        Ok(next) => group.push(next),
                        Err(_) => break,
                    }
                }
                let (batches, replies): (Vec<_>, Vec<_>) = group
                    .into_iter()
                    .filter(|request| !request.reply.is_closed())
                    .map(|request| (request.mutations, request.reply))
                    .unzip();
                let results = match writer.commit_group(batches).await {
                    Ok(results) => results,
                    Err(StorageError::Closed) => {
                        replies.iter().map(|_| Err(StorageError::Closed)).collect()
                    }
                    Err(error) => {
                        let shared = Arc::new(error);
                        replies
                            .iter()
                            .map(|_| Err(StorageError::Shared(shared.clone())))
                            .collect()
                    }
                };
                for (reply, result) in replies.into_iter().zip(results) {
                    let _ = reply.send(result);
                }
            }
        });
        Self { queue: sender }
    }

    /// Commit `mutations` atomically, together with whatever else is waiting.
    ///
    /// A caller that stops waiting before its batch is taken into a commit is
    /// skipped. One that stops waiting after that point cannot be: its batch
    /// is committed all the same, so a caller that retries after a timeout
    /// must use mutations that are safe to repeat. A failure of the shared
    /// commit arrives as [`StorageError::Shared`]; see [`StorageError::root`].
    pub async fn submit(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        let (reply, outcome) = oneshot::channel();
        self.queue
            .send(Request { mutations, reply })
            .await
            .map_err(|_| StorageError::Closed)?;
        outcome.await.map_err(|_| StorageError::Closed)?
    }
}
