//! How much one tenant may ask of a node: queries and written objects a
//! second, each a bucket that refills at the rate and holds a second's
//! worth, so that a burst is taken and a flood is turned away. One tenant
//! asking too much is answered "too many", and others are not held up.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

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

    #[test]
    fn a_batch_larger_than_a_bucket_is_let_through_and_owed() {
        let limits = Limits::new(None, Some(100.0));
        let busy = TenantId::new("busy").unwrap();
        limits.admit(&busy, Asked::Write, 500).unwrap();
        let owed = match limits.admit(&busy, Asked::Write, 1) {
            Err(NodeError::Limited { wait, .. }) => wait,
            other => panic!("{other:?}"),
        };
        assert!(
            owed >= Duration::from_secs(4),
            "{owed:?}: the 400 over is owed"
        );
    }
}
