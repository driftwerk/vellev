//! How much one tenant may ask of a node: queries and written objects a
//! second, each a bucket that refills at the rate and holds a second's
//! worth, so that a burst is taken and a flood is turned away. One tenant
//! asking too much is answered "too many", and others are not held up.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use tokio::time::Instant;

use vellev_core::TenantId;

use crate::{NodeError, Result};

/// What a request asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asked {
    Query,
    /// Writes, counted by the objects they change.
    Write,
}

#[derive(Debug, Default)]
pub(crate) struct Limits {
    queries: Option<f64>,
    writes: Option<f64>,
    buckets: Mutex<HashMap<(TenantId, Asked), Bucket>>,
}

#[derive(Debug, Clone, Copy)]
struct Bucket {
    tokens: f64,
    at: Instant,
}

impl Limits {
    pub fn new(queries: Option<f64>, writes: Option<f64>) -> Self {
        Self {
            queries: queries.filter(|rate| *rate > 0.0),
            writes: writes.filter(|rate| *rate > 0.0),
            buckets: Mutex::default(),
        }
    }

    fn rate(&self, asked: Asked) -> Option<f64> {
        match asked {
            Asked::Query => self.queries,
            Asked::Write => self.writes,
        }
    }

    /// Take `cost` of what `tenant` may ask for now, or say how long until
    /// it may. A request costing more than a bucket holds is let through
    /// once the bucket is full, and leaves it owing the rest, so that none
    /// is turned away for ever and the rate holds however requests are
    /// sized.
    pub fn admit(&self, tenant: &TenantId, asked: Asked, cost: usize) -> Result<()> {
        let Some(rate) = self.rate(asked) else {
            return Ok(());
        };
        let now = Instant::now();
        let mut buckets = self.buckets.lock().expect("limits lock");
        let bucket = buckets.entry((tenant.clone(), asked)).or_insert(Bucket {
            tokens: rate,
            at: now,
        });
        bucket.tokens =
            (bucket.tokens + rate * now.duration_since(bucket.at).as_secs_f64()).min(rate);
        bucket.at = now;
        let cost = cost as f64;
        if bucket.tokens >= cost.min(rate) {
            bucket.tokens -= cost;
            return Ok(());
        }
        let wait = Duration::from_secs_f64((cost.min(rate) - bucket.tokens) / rate);
        tracing::debug!(%tenant, ?asked, ?wait, "turned away");
        Err(NodeError::Limited {
            tenant: tenant.clone(),
            wait,
        })
    }

    /// Take `cost` more of what `tenant` may ask for, for something let
    /// through already that turned out to cost more than was known when
    /// it was, as a delete by filter does by each object it deletes. The
    /// bucket is left owing it.
    pub fn owe(&self, tenant: &TenantId, asked: Asked, cost: usize) {
        if cost == 0 {
            return;
        }
        let Some(rate) = self.rate(asked) else { return };
        // A bucket full again may have been forgotten meanwhile, which is
        // no reason to forget what is owed: it is made afresh, full.
        let mut buckets = self.buckets.lock().expect("limits lock");
        let bucket = buckets.entry((tenant.clone(), asked)).or_insert(Bucket {
            tokens: rate,
            at: Instant::now(),
        });
        bucket.tokens -= cost as f64;
    }

    /// Give `tenant` back `cost` it was let through for, for something
    /// that was then not done for it: a read turned away for want of a
    /// thread is no query. No bucket is left holding more than it holds
    /// full, and one forgotten meanwhile is full already.
    pub fn give_back(&self, tenant: &TenantId, asked: Asked, cost: usize) {
        let Some(rate) = self.rate(asked) else { return };
        let mut buckets = self.buckets.lock().expect("limits lock");
        if let Some(bucket) = buckets.get_mut(&(tenant.clone(), asked)) {
            bucket.tokens = (bucket.tokens + cost as f64).min(rate);
        }
    }

    /// Forget buckets full again, which cost nothing to make afresh.
    pub fn forget_idle(&self) {
        let now = Instant::now();
        self.buckets
            .lock()
            .expect("limits lock")
            .retain(|(_, asked), bucket| {
                let rate = self.rate(*asked).unwrap_or(0.0);
                bucket.tokens + rate * now.duration_since(bucket.at).as_secs_f64() < rate
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_burst_is_taken_a_flood_turned_away_and_others_not_held_up() {
        let limits = Limits::new(Some(10.0), Some(100.0));
        let (busy, quiet) = (
            TenantId::new("busy").unwrap(),
            TenantId::new("quiet").unwrap(),
        );
        for _ in 0..10 {
            limits.admit(&busy, Asked::Query, 1).unwrap();
        }
        let waited = |turned: Result<()>| match turned {
            Err(NodeError::Limited { wait, .. }) => wait,
            other => panic!("{other:?}"),
        };
        let wait = waited(limits.admit(&busy, Asked::Query, 1));
        assert!(
            wait > Duration::ZERO && wait <= Duration::from_millis(100),
            "{wait:?}"
        );
        limits.admit(&quiet, Asked::Query, 1).unwrap();
        let unlimited = Limits::new(None, None);
        for _ in 0..1_000 {
            unlimited.admit(&busy, Asked::Query, 1).unwrap();
        }
    }

    // On a clock that stands still, as the two after it are: a bucket
    // refills by its clock between one asking and the next, so on a
    // running one what is owed is less than was taken by however long the
    // test took to ask again.
    #[tokio::test(start_paused = true)]
    async fn a_batch_larger_than_a_bucket_is_let_through_and_owed() {
        let limits = Limits::new(None, Some(100.0));
        let busy = TenantId::new("busy").unwrap();
        limits.admit(&busy, Asked::Write, 500).unwrap();
        let owed = match limits.admit(&busy, Asked::Write, 1) {
            Err(NodeError::Limited { wait, .. }) => wait,
            other => panic!("{other:?}"),
        };
        // The 400 over, and the one asked for now, at 100 a second.
        assert_eq!(owed, Duration::from_secs_f64(401.0 / 100.0));
    }

    #[tokio::test(start_paused = true)]
    async fn what_a_request_turns_out_to_cost_is_owed_too() {
        let limits = Limits::new(None, Some(100.0));
        let busy = TenantId::new("busy").unwrap();
        limits.admit(&busy, Asked::Write, 1).unwrap();
        limits.owe(&busy, Asked::Write, 0);
        limits.admit(&busy, Asked::Write, 1).unwrap();
        limits.owe(&busy, Asked::Write, 498);
        let owed = match limits.admit(&busy, Asked::Write, 1) {
            Err(NodeError::Limited { wait, .. }) => wait,
            other => panic!("{other:?}"),
        };
        assert_eq!(owed, Duration::from_secs_f64(401.0 / 100.0));
        Limits::new(None, None).owe(&busy, Asked::Write, 1_000);
    }

    #[tokio::test(start_paused = true)]
    async fn what_is_given_back_may_be_asked_again_and_no_more_than_that() {
        let limits = Limits::new(Some(2.0), None);
        let busy = TenantId::new("busy").unwrap();
        let asks = || limits.admit(&busy, Asked::Query, 1);
        asks().unwrap();
        asks().unwrap();
        asks().unwrap_err();
        limits.give_back(&busy, Asked::Query, 1);
        asks().unwrap();
        asks().unwrap_err();
        // Given back more than was taken, a bucket holds what it holds
        // full and no more.
        limits.give_back(&busy, Asked::Query, 10);
        asks().unwrap();
        asks().unwrap();
        asks().unwrap_err();
        // Nor does the rest pay for what is owed after.
        limits.give_back(&busy, Asked::Query, 10);
        limits.owe(&busy, Asked::Query, 1);
        asks().unwrap();
        asks().unwrap_err();
        // One forgotten meanwhile is full already, and is not made over
        // full by it; nor is there anything to give back where nothing
        // is limited.
        tokio::time::advance(Duration::from_secs(1)).await;
        limits.forget_idle();
        limits.give_back(&busy, Asked::Query, 1);
        assert!(limits.buckets.lock().unwrap().is_empty());
        asks().unwrap();
        asks().unwrap();
        asks().unwrap_err();
        Limits::new(None, None).give_back(&busy, Asked::Query, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn what_is_owed_by_a_tenant_forgotten_meanwhile_is_owed_all_the_same() {
        let limits = Limits::new(None, Some(100.0));
        let busy = TenantId::new("busy").unwrap();
        // Let through as one write; by the time it is known to have cost
        // five hundred, its bucket is full again and forgotten.
        limits.admit(&busy, Asked::Write, 1).unwrap();
        tokio::time::advance(Duration::from_secs(1)).await;
        limits.forget_idle();
        limits.owe(&busy, Asked::Write, 500);
        let owed = match limits.admit(&busy, Asked::Write, 1) {
            Err(NodeError::Limited { wait, .. }) => wait,
            other => panic!("{other:?}"),
        };
        assert!(owed >= Duration::from_secs(3), "{owed:?}");
    }
}
