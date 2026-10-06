//! The requests a partition has answered lately, so that a request tried
//! again is answered as it was the first time rather than made twice.
//!
//! What is kept, and for how long, follows from the committed files
//! alone, so that every copy of a partition, and a writer that takes it
//! over, keeps the same: each commit records the requests it answered and
//! when it was made, and a request is forgotten once a commit
//! [`KEPT_FOR`] newer has been made.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use vellev_core::RequestId;

use crate::coordinator::LogPosition;
use crate::format::Answered;
use crate::mutation::Outcome;

/// How long a request is answered as it was: far longer than a client
/// keeps trying one.
pub const KEPT_FOR: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Default)]
pub(crate) struct Answers {
    by_request: HashMap<RequestId, (LogPosition, Answered)>,
    /// In the order they were answered.
    order: VecDeque<RequestId>,
    /// When the newest commit was made, in milliseconds since the epoch.
    newest: u64,
}

impl Answers {
    /// Those a manifest kept, oldest first.
    pub fn of(kept: Vec<(LogPosition, Answered)>) -> Self {
        kept.into_iter()
            .fold(Self::default(), |mut answers, (position, answered)| {
                answers.remember(position, answered);
                answers
            })
    }

    /// Remember a request answered by the commit at `position`, and forget
    /// those answered long enough before it. Remembering one already
    /// remembered changes nothing.
    pub fn remember(&mut self, position: LogPosition, answered: Answered) {
        self.newest = self.newest.max(answered.at);
        if !self.by_request.contains_key(&answered.request) {
            self.order.push_back(answered.request);
            self.by_request
                .insert(answered.request, (position, answered));
        }
        let kept_for = KEPT_FOR.as_millis() as u64;
        while let Some(oldest) = self.order.front() {
            let (_, answered) = &self.by_request[oldest];
            if answered.at.saturating_add(kept_for) >= self.newest {
                break;
            }
            self.by_request.remove(oldest);
            self.order.pop_front();
        }
    }

    /// Where `request` became visible and what it did, if it was answered
    /// lately.
    pub fn get(&self, request: RequestId) -> Option<(LogPosition, &[Outcome])> {
        self.by_request
            .get(&request)
            .map(|(position, answered)| (*position, answered.outcomes.as_slice()))
    }

    /// Every one remembered, oldest first, for a manifest to keep.
    pub fn kept(&self) -> Vec<(LogPosition, Answered)> {
        self.order
            .iter()
            .map(|request| self.by_request[request].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answered(n: u128, at: u64) -> Answered {
        Answered {
            request: RequestId(uuid::Uuid::from_u128(n)),
            at,
            outcomes: vec![Outcome::Applied],
        }
    }

    fn known(answers: &Answers, n: u128) -> bool {
        answers.get(RequestId(uuid::Uuid::from_u128(n))).is_some()
    }

    #[test]
    fn a_request_is_forgotten_once_a_commit_long_enough_after_it_is_made() {
        let kept_for = KEPT_FOR.as_millis() as u64;
        let mut answers = Answers::default();
        answers.remember(LogPosition(1), answered(1, 1_000));
        answers.remember(LogPosition(2), answered(2, 2_000));
        answers.remember(LogPosition(3), answered(3, 1_000 + kept_for));
        assert!(known(&answers, 1) && known(&answers, 2));
        answers.remember(LogPosition(4), answered(4, 1_001 + kept_for));
        assert!(!known(&answers, 1) && known(&answers, 2));
        let kept: Vec<LogPosition> = answers.kept().into_iter().map(|(at, _)| at).collect();
        assert_eq!(kept, [LogPosition(2), LogPosition(3), LogPosition(4)]);
        assert_eq!(Answers::of(answers.kept()).kept(), answers.kept());
    }

    #[test]
    fn remembering_a_request_again_keeps_its_first_answer() {
        let mut answers = Answers::default();
        answers.remember(LogPosition(1), answered(1, 1_000));
        answers.remember(LogPosition(5), answered(1, 1_000));
        let (position, _) = answers.get(RequestId(uuid::Uuid::from_u128(1))).unwrap();
        assert_eq!(position, LogPosition(1));
        assert_eq!(answers.kept().len(), 1);
    }
}
