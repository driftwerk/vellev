//! The requests a partition has answered lately, so that a request tried
//! again is answered as it was the first time rather than made twice.
//!
//! What is kept, and for how long, follows from the committed files
//! alone, so that every copy of a partition, and a writer that takes it
//! over, keeps the same: each commit records the requests it answered and
//! when it was made, and a request is forgotten once a commit
//! [`KEPT_FOR`] newer has been made, or sooner once [`MOST_KEPT`] newer
//! ones are kept.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use vellev_core::RequestId;

use crate::coordinator::LogPosition;
use crate::format::{Answer, Answered};

/// How long a request is answered as it was: far longer than a client
/// keeps trying one. The API's description of `x-vellev-request` and
/// docs/compatibility.md say so too.
pub const KEPT_FOR: Duration = Duration::from_secs(15 * 60);

/// The most requests kept, so that the window, which every manifest
/// holds, stays small whatever the rate of writes; past it the oldest go
/// early.
pub const MOST_KEPT: usize = 100_000;

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
    /// remembered, or one already too old to keep, changes nothing.
    pub fn remember(&mut self, position: LogPosition, answered: Answered) {
        self.newest = self.newest.max(answered.at);
        let expired = |answered: &Answered| {
            answered.at.saturating_add(KEPT_FOR.as_millis() as u64) < self.newest
        };
        if !expired(&answered) && !self.by_request.contains_key(&answered.request) {
            self.order.push_back(answered.request);
            self.by_request
                .insert(answered.request, (position, answered));
        }
        while let Some(oldest) = self.order.front() {
            if !expired(&self.by_request[oldest].1) && self.order.len() <= MOST_KEPT {
                break;
            }
            self.by_request.remove(oldest);
            self.order.pop_front();
        }
    }

    /// When a commit made at `now` is to say it was made: never before the
    /// newest, so that a writer whose clock is behind another's cannot
    /// have its requests forgotten early, nor answers go out of order.
    pub fn stamp(&self, now: u64) -> u64 {
        now.max(self.newest)
    }

    /// Where `request` became visible and how it was answered, if it was
    /// answered lately.
    pub fn get(&self, request: RequestId) -> Option<(LogPosition, &Answer)> {
        self.by_request
            .get(&request)
            .map(|(position, answered)| (*position, &answered.answer))
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
            answer: Answer::Made(vec![crate::Outcome::Applied]),
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

    #[test]
    fn a_request_too_old_to_keep_is_not_remembered_and_the_newest_are_kept() {
        let kept_for = KEPT_FOR.as_millis() as u64;
        let mut answers = Answers::default();
        answers.remember(LogPosition(1), answered(1, 1_000 + kept_for + 1));
        answers.remember(LogPosition(2), answered(2, 1_000));
        assert!(!known(&answers, 2));
        assert_eq!(answers.stamp(5), 1_000 + kept_for + 1);

        let mut answers = Answers::default();
        for n in 0..=MOST_KEPT as u128 {
            answers.remember(LogPosition(n as u64), answered(n, 1_000));
        }
        assert!(!known(&answers, 0) && known(&answers, 1));
        assert_eq!(answers.kept().len(), MOST_KEPT);
    }
}
