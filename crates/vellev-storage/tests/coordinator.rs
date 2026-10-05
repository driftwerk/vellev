use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use vellev_core::{NodeId, Partition, TenantId};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease, LogPosition,
    ManifestRef, ManualClock, MemoryCoordinator, PartitionKey, PostgresCoordinator,
};

const TTL: Duration = Duration::from_millis(200);

/// A coordinator under test and a way to let its clock move past a TTL.
#[async_trait]
trait Harness: Send + Sync {
    fn coordinator(&self) -> &dyn Coordinator;
    async fn pass(&self, duration: Duration);
}

struct Memory {
    clock: Arc<ManualClock>,
    coordinator: MemoryCoordinator,
}

impl Memory {
    fn new() -> Self {
        let clock = Arc::new(ManualClock::starting_at(
            Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
        ));
        Self {
            coordinator: MemoryCoordinator::new(clock.clone()),
            clock,
        }
    }
}

#[async_trait]
impl Harness for Memory {
    fn coordinator(&self) -> &dyn Coordinator {
        &self.coordinator
    }
    async fn pass(&self, duration: Duration) {
        self.clock.advance(duration);
    }
}

struct Pg(PostgresCoordinator);

impl Pg {
    /// `None` when no database is configured; CI always configures one.
    async fn connect() -> Option<Self> {
        let Ok(url) = std::env::var("VELLEV_TEST_DATABASE_URL") else {
            eprintln!(
                "VELLEV_TEST_DATABASE_URL is not set; skipping the Postgres coordinator test"
            );
            return None;
        };
        Some(Self(
            PostgresCoordinator::connect(&url, 4)
                .await
                .expect("connect to test database"),
        ))
    }
}

#[async_trait]
impl Harness for Pg {
    fn coordinator(&self) -> &dyn Coordinator {
        &self.0
    }
    async fn pass(&self, duration: Duration) {
        tokio::time::sleep(duration + Duration::from_millis(50)).await;
    }
}

fn key() -> PartitionKey {
    PartitionKey {
        tenant: TenantId::new(uuid::Uuid::new_v4().simple().to_string()).unwrap(),
        partition: Partition::DEFAULT,
    }
}

fn node(name: &str) -> NodeId {
    NodeId::from(format!("{name}-{}", uuid::Uuid::new_v4().simple()))
}

async fn registered(h: &dyn Harness) -> PartitionKey {
    let key = key();
    h.coordinator().register(&key).await.unwrap();
    key
}

async fn lease(h: &dyn Harness, key: &PartitionKey, holder: &NodeId) -> (Lease, Committed) {
    match h.coordinator().acquire(key, holder, TTL).await.unwrap() {
        Acquired::Lease { lease, committed } => (lease, committed),
        Acquired::Held { holder, .. } => panic!("expected to acquire {key}, held by {holder}"),
    }
}

fn commit(position: u64, manifest: &str) -> Commit {
    Commit {
        log_position: LogPosition(position),
        manifest: Some(ManifestRef(manifest.into())),
    }
}

async fn registering_is_idempotent_and_unregistered_partitions_are_unknown(h: &dyn Harness) {
    let c = h.coordinator();
    let key = key();
    assert_eq!(c.committed(&key).await.unwrap(), None);
    assert!(matches!(
        c.acquire(&key, &node("a"), TTL).await,
        Err(CoordinatorError::UnknownPartition(k)) if k == key
    ));

    c.register(&key).await.unwrap();
    c.register(&key).await.unwrap();
    assert_eq!(
        c.committed(&key).await.unwrap(),
        Some(Committed {
            generation: Generation(0),
            log_position: LogPosition(0),
            manifest: None
        })
    );
}

async fn the_first_lease_starts_generation_one_at_position_zero(h: &dyn Harness) {
    let key = registered(h).await;
    let holder = node("a");
    let (lease, committed) = lease(h, &key, &holder).await;
    assert_eq!(lease.key, key);
    assert_eq!(lease.holder, holder);
    assert_eq!(lease.generation, Generation(1));
    assert_eq!(lease.ttl, TTL);
    assert_eq!(committed.log_position, LogPosition(0));
    assert_eq!(committed.manifest, None);
}

async fn a_held_lease_is_reported_until_it_expires(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;

    let held = c.acquire(&key, &node("b"), TTL).await.unwrap();
    assert_eq!(
        held,
        Acquired::Held {
            holder: first.holder.clone(),
            expires_at: first.expires_at
        }
    );

    h.pass(TTL).await;
    let (second, _) = lease(h, &key, &node("b")).await;
    assert_eq!(second.generation, Generation(2));
}

async fn renewing_extends_the_lease_and_is_fenced_after_a_takeover(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;

    h.pass(TTL / 2).await;
    let renewed = c.renew(&first, TTL).await.unwrap();
    assert!(renewed.expires_at > first.expires_at);
    assert_eq!(renewed.generation, first.generation);
    assert!(matches!(
        c.acquire(&key, &node("b"), TTL).await.unwrap(),
        Acquired::Held { .. }
    ));

    h.pass(TTL).await;
    let renewed_late = c.renew(&renewed, TTL).await.unwrap();
    assert_eq!(renewed_late.generation, first.generation);

    h.pass(TTL).await;
    let (second, _) = lease(h, &key, &node("b")).await;
    assert!(matches!(
        c.renew(&renewed_late, TTL).await,
        Err(CoordinatorError::Fenced { held: Generation(1), current, .. }) if current == second.generation
    ));
}

async fn releasing_frees_the_lease_and_a_stale_release_is_harmless(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;

    c.release(&first).await.unwrap();
    let (second, _) = lease(h, &key, &node("b")).await;
    assert!(second.generation > first.generation);

    c.release(&first).await.unwrap();
    assert!(matches!(
        c.acquire(&key, &node("c"), TTL).await.unwrap(),
        Acquired::Held { .. }
    ));
    assert!(matches!(
        c.commit(&first, commit(1, "m1")).await,
        Err(CoordinatorError::Fenced { .. })
    ));
    c.commit(&second, commit(1, "m1")).await.unwrap();
}

async fn commits_advance_the_position_and_followers_see_them(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;

    let committed = c.commit(&lease, commit(10, "m10")).await.unwrap();
    assert_eq!(
        committed,
        Committed {
            generation: lease.generation,
            log_position: LogPosition(10),
            manifest: Some(ManifestRef("m10".into())),
        }
    );
    c.commit(
        &lease,
        Commit {
            log_position: LogPosition(11),
            manifest: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(
        c.committed(&key).await.unwrap(),
        Some(Committed {
            generation: lease.generation,
            log_position: LogPosition(11),
            manifest: None
        })
    );
}

async fn a_stalled_writer_is_fenced_and_changes_nothing(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (stalled, _) = lease(h, &key, &node("a")).await;
    c.commit(&stalled, commit(5, "m5")).await.unwrap();

    h.pass(TTL).await;
    let (successor, resumed) = lease(h, &key, &node("b")).await;
    assert_eq!(resumed.log_position, LogPosition(5));
    c.commit(&successor, commit(6, "m6")).await.unwrap();

    assert!(matches!(
        c.commit(&stalled, commit(7, "m7-stale")).await,
        Err(CoordinatorError::Fenced {
            held: Generation(1),
            current: Generation(2),
            ..
        })
    ));
    assert!(matches!(
        c.renew(&stalled, TTL).await,
        Err(CoordinatorError::Fenced { .. })
    ));
    assert_eq!(
        c.committed(&key).await.unwrap(),
        Some(Committed {
            generation: Generation(2),
            log_position: LogPosition(6),
            manifest: Some(ManifestRef("m6".into())),
        })
    );
}

async fn a_commit_that_does_not_advance_is_refused(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    c.commit(&lease, commit(3, "m3")).await.unwrap();

    for position in [3, 2, 0] {
        assert!(matches!(
            c.commit(&lease, commit(position, "stale")).await,
            Err(CoordinatorError::Regressed { committed: LogPosition(3), attempted, .. })
                if attempted == LogPosition(position)
        ));
    }
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().manifest,
        Some(ManifestRef("m3".into()))
    );
}

async fn removing_a_partition_makes_it_unknown(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;

    assert!(c.remove(&key).await.unwrap());
    assert!(!c.remove(&key).await.unwrap());
    assert_eq!(c.committed(&key).await.unwrap(), None);
    assert!(matches!(
        c.acquire(&key, &node("b"), TTL).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    assert!(matches!(
        c.commit(&lease, commit(1, "m1")).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    assert!(matches!(
        c.renew(&lease, TTL).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    c.release(&lease).await.unwrap();
}

async fn ping_succeeds(h: &dyn Harness) {
    h.coordinator().ping().await.unwrap();
}

macro_rules! conformance {
    ($($scenario:ident),* $(,)?) => {
        mod memory {
            $(
                #[tokio::test]
                async fn $scenario() {
                    super::$scenario(&super::Memory::new()).await;
                }
            )*
        }
        mod postgres {
            $(
                #[tokio::test]
                async fn $scenario() {
                    if let Some(pg) = super::Pg::connect().await {
                        super::$scenario(&pg).await;
                    }
                }
            )*
        }
    };
}

conformance! {
    registering_is_idempotent_and_unregistered_partitions_are_unknown,
    the_first_lease_starts_generation_one_at_position_zero,
    a_held_lease_is_reported_until_it_expires,
    renewing_extends_the_lease_and_is_fenced_after_a_takeover,
    releasing_frees_the_lease_and_a_stale_release_is_harmless,
    commits_advance_the_position_and_followers_see_them,
    a_stalled_writer_is_fenced_and_changes_nothing,
    a_commit_that_does_not_advance_is_refused,
    removing_a_partition_makes_it_unknown,
    ping_succeeds,
}

#[tokio::test]
async fn postgres_migrations_tolerate_nodes_starting_together() {
    let Ok(url) = std::env::var("VELLEV_TEST_DATABASE_URL") else {
        return;
    };
    let starts: Vec<_> = (0..6)
        .map(|_| {
            let url = url.clone();
            tokio::spawn(async move { PostgresCoordinator::connect(&url, 2).await })
        })
        .collect();
    for started in starts {
        started.await.unwrap().expect("concurrent migration");
    }
}
