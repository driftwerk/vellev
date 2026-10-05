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
    /// Start committing for `writer`. At most `max_group` batches share one
    /// commit, and at most `queue` wait before a submitter is held back. The
    /// task ends when the last clone of the committer is dropped.
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
                    .map(|request| (request.mutations, request.reply))
                    .unzip();
                let results = match writer.commit_group(batches).await {
                    Ok(results) => results,
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
    pub async fn submit(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        let (reply, outcome) = oneshot::channel();
        self.queue
            .send(Request { mutations, reply })
            .await
            .map_err(|_| StorageError::Closed)?;
        outcome.await.map_err(|_| StorageError::Closed)?
    }
}
