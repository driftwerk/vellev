use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use futures::future::join_all;
use vellev_core::{NodeId, Partition, TenantId};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease, LogPosition,
    ManifestRef, ManualClock, MemoryCoordinator, PartitionKey, PostgresCoordinator, MAX_TTL,
};

const TTL: Duration = Duration::from_secs(30);

/// A coordinator under test and a way to move a partition's lease into the
/// past, as if that much time had gone by.
#[async_trait]
trait Harness: Send + Sync {
    fn coordinator(&self) -> &dyn Coordinator;
    async fn pass(&self, key: &PartitionKey, duration: Duration);
}

struct Memory {
    clock: Arc<ManualClock>,
    coordinator: MemoryCoordinator,
}

impl Memory {
    fn new() -> Self {
        let start = Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap();
        let clock = Arc::new(ManualClock::starting_at(start));
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
    async fn pass(&self, _: &PartitionKey, duration: Duration) {
        self.clock.advance(duration);
    }
}

struct Pg {
    coordinator: PostgresCoordinator,
    pool: sqlx::PgPool,
}

fn database_url() -> Option<String> {
    let url = std::env::var("VELLEV_TEST_DATABASE_URL").ok();
    if url.is_none() {
        eprintln!("VELLEV_TEST_DATABASE_URL is not set; skipping a Postgres coordinator test");
    }
    url
}

impl Pg {
    async fn connect() -> Option<Self> {
        let url = database_url()?;
        let pool = sqlx::PgPool::connect(&url)
            .await
            .expect("connect to test database");
        let coordinator = PostgresCoordinator::with_pool(pool.clone())
            .await
            .expect("migrate");
        Some(Self { coordinator, pool })
    }
}

#[async_trait]
impl Harness for Pg {
    fn coordinator(&self) -> &dyn Coordinator {
        &self.coordinator
    }
    async fn pass(&self, key: &PartitionKey, duration: Duration) {
        sqlx::query(
            "UPDATE vellev.partitions
             SET lease_expires_at = lease_expires_at - make_interval(secs => $3)
             WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(duration.as_secs_f64())
        .execute(&self.pool)
        .await
        .expect("rewind lease");
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

async fn held_by(h: &dyn Harness, key: &PartitionKey, holder: &NodeId) -> bool {
    matches!(
        h.coordinator().acquire(key, &node("probe"), TTL).await.unwrap(),
        Acquired::Held { holder: current, .. } if current == *holder
    )
}

async fn generation(h: &dyn Harness, key: &PartitionKey) -> Generation {
    h.coordinator()
        .committed(key)
        .await
        .unwrap()
        .unwrap()
        .generation
}

fn commit(position: u64, manifest: &str) -> Commit {
    Commit {
        log_position: LogPosition(position),
        manifest: Some(ManifestRef(manifest.into())),
    }
}

fn fenced(result: &Result<impl std::fmt::Debug, CoordinatorError>, current: Generation) -> bool {
    matches!(result, Err(CoordinatorError::Fenced { current: c, .. }) if *c == current)
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
    let registered = c.committed(&key).await.unwrap().unwrap();
    assert!(registered.generation > Generation(0));
    assert_eq!(
        (registered.log_position, &registered.manifest),
        (LogPosition(0), &None)
    );
    c.register(&key).await.unwrap();
    assert_eq!(c.committed(&key).await.unwrap(), Some(registered));
}

async fn the_first_lease_starts_a_fresh_generation_at_position_zero(h: &dyn Harness) {
    let key = registered(h).await;
    let holder = node("a");
    let (lease, committed) = lease(h, &key, &holder).await;
    assert_eq!(lease.key, key);
    assert_eq!(lease.holder, holder);
    assert!(lease.generation > Generation(0));
    assert_eq!(lease.ttl, TTL);
    assert_eq!(lease.deadline(), lease.requested_at + TTL);
    assert_eq!(committed.generation, lease.generation);
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

    h.pass(&key, TTL).await;
    let (second, _) = lease(h, &key, &node("b")).await;
    assert!(second.generation > first.generation);
}

async fn the_holder_can_take_its_own_lease_again_and_the_old_one_is_fenced(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let holder = node("a");
    let (first, _) = lease(h, &key, &holder).await;
    let (again, _) = lease(h, &key, &holder).await;
    assert!(again.generation > first.generation);
    assert!(fenced(
        &c.commit(&first, commit(1, "m1")).await,
        again.generation
    ));
    c.commit(&again, commit(1, "m1")).await.unwrap();
}

async fn a_zero_ttl_lease_is_expired_at_once(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    match c.acquire(&key, &node("a"), Duration::ZERO).await.unwrap() {
        Acquired::Lease { lease, .. } => assert_eq!(lease.deadline(), lease.requested_at),
        held => panic!("{held:?}"),
    }
    lease(h, &key, &node("b")).await;
}

async fn a_ttl_beyond_the_maximum_is_refused(h: &dyn Harness) {
    let c = h.coordinator();
    let too_long = [MAX_TTL + Duration::from_secs(1), Duration::MAX];
    let invalid = |r: Result<Acquired, CoordinatorError>| {
        matches!(r, Err(CoordinatorError::InvalidArgument(_)))
    };

    let unregistered = key();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    for ttl in too_long {
        assert!(invalid(c.acquire(&unregistered, &node("x"), ttl).await));
        assert!(invalid(c.acquire(&key, &node("x"), ttl).await));
        assert!(invalid(c.acquire(&key, &node("a"), ttl).await));
    }
    assert!(held_by(h, &key, &lease.holder).await);
    assert!(matches!(
        c.renew(&lease, Duration::MAX).await,
        Err(CoordinatorError::InvalidArgument(_))
    ));
    c.renew(&lease, MAX_TTL).await.unwrap();
}

async fn renewing_extends_the_lease_and_is_fenced_after_a_takeover(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;

    h.pass(&key, TTL / 2).await;
    let renewed = c.renew(&first, TTL).await.unwrap();
    assert!(renewed.expires_at > first.expires_at);
    assert_eq!(renewed.generation, first.generation);
    assert!(renewed.requested_at >= first.requested_at);
    assert!(held_by(h, &key, &first.holder).await);

    h.pass(&key, TTL).await;
    let renewed_late = c.renew(&renewed, TTL).await.unwrap();
    assert_eq!(renewed_late.generation, first.generation);

    h.pass(&key, TTL).await;
    let (second, _) = lease(h, &key, &node("b")).await;
    assert!(fenced(
        &c.renew(&renewed_late, TTL).await,
        second.generation
    ));
}

async fn an_expired_lease_nobody_took_over_can_still_commit(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    h.pass(&key, TTL * 2).await;
    c.commit(&lease, commit(1, "m1")).await.unwrap();
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().log_position,
        LogPosition(1)
    );
}

async fn releasing_moves_the_generation_and_a_stale_release_is_harmless(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;

    c.release(&first).await.unwrap();
    let released = generation(h, &key).await;
    assert!(released > first.generation);
    assert!(fenced(&c.renew(&first, TTL).await, released));
    assert!(fenced(&c.commit(&first, commit(1, "m1")).await, released));

    let (second, _) = lease(h, &key, &node("b")).await;
    assert!(second.generation > released);

    c.release(&first).await.unwrap();
    assert_eq!(generation(h, &key).await, second.generation);
    assert!(held_by(h, &key, &second.holder).await);
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

async fn repeating_a_commit_that_landed_succeeds(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    let first = c.commit(&lease, commit(4, "m4")).await.unwrap();
    assert_eq!(c.commit(&lease, commit(4, "m4")).await.unwrap(), first);
    assert!(matches!(
        c.commit(&lease, commit(4, "other")).await,
        Err(CoordinatorError::Regressed {
            committed: LogPosition(4),
            attempted: LogPosition(4),
            ..
        })
    ));
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().manifest,
        Some(ManifestRef("m4".into()))
    );
}

async fn a_stalled_writer_is_fenced_and_changes_nothing(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (stalled, _) = lease(h, &key, &node("a")).await;
    c.commit(&stalled, commit(5, "m5")).await.unwrap();

    h.pass(&key, TTL).await;
    let (successor, resumed) = lease(h, &key, &node("b")).await;
    assert_eq!(resumed.log_position, LogPosition(5));
    c.commit(&successor, commit(6, "m6")).await.unwrap();

    assert!(fenced(
        &c.commit(&stalled, commit(7, "m7-stale")).await,
        successor.generation
    ));
    assert!(fenced(&c.renew(&stalled, TTL).await, successor.generation));
    assert_eq!(
        c.committed(&key).await.unwrap(),
        Some(Committed {
            generation: successor.generation,
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

    for position in [2, 0] {
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

async fn a_generation_is_never_reused_after_remove_and_register(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (zombie, _) = lease(h, &key, &node("a")).await;

    assert!(c.remove(&key).await.unwrap());
    c.register(&key).await.unwrap();
    assert!(generation(h, &key).await > zombie.generation);
    let (fresh, _) = lease(h, &key, &node("b")).await;
    assert!(fresh.generation > zombie.generation);
    assert!(fenced(
        &c.commit(&zombie, commit(1, "zombie")).await,
        fresh.generation
    ));
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().log_position,
        LogPosition(0)
    );
}

async fn removing_a_partition_makes_it_unknown(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;

    assert!(c.remove(&key).await.unwrap());
    assert!(!c.remove(&key).await.unwrap());
    assert_eq!(c.committed(&key).await.unwrap(), None);
    let unknown = |r: &Result<_, CoordinatorError>| matches!(r, Err(CoordinatorError::UnknownPartition(k)) if *k == key);
    assert!(unknown(&c.acquire(&key, &node("b"), TTL).await.map(|_| ())));
    assert!(unknown(
        &c.commit(&lease, commit(1, "m1")).await.map(|_| ())
    ));
    assert!(unknown(&c.renew(&lease, TTL).await.map(|_| ())));
    c.release(&lease).await.unwrap();
}

async fn concurrent_acquires_hand_out_exactly_one_lease(h: &dyn Harness) {
    let key = registered(h).await;
    let holders: Vec<_> = (0..8).map(|i| node(&format!("n{i}"))).collect();
    let outcomes = join_all(
        holders
            .iter()
            .map(|holder| h.coordinator().acquire(&key, holder, TTL)),
    )
    .await;
    let leases = outcomes
        .iter()
        .filter(|o| matches!(o, Ok(Acquired::Lease { .. })))
        .count();
    assert_eq!(leases, 1, "{outcomes:?}");
    assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
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
    the_first_lease_starts_a_fresh_generation_at_position_zero,
    a_held_lease_is_reported_until_it_expires,
    the_holder_can_take_its_own_lease_again_and_the_old_one_is_fenced,
    a_zero_ttl_lease_is_expired_at_once,
    a_ttl_beyond_the_maximum_is_refused,
    renewing_extends_the_lease_and_is_fenced_after_a_takeover,
    an_expired_lease_nobody_took_over_can_still_commit,
    releasing_moves_the_generation_and_a_stale_release_is_harmless,
    commits_advance_the_position_and_followers_see_them,
    repeating_a_commit_that_landed_succeeds,
    a_stalled_writer_is_fenced_and_changes_nothing,
    a_commit_that_does_not_advance_is_refused,
    a_generation_is_never_reused_after_remove_and_register,
    removing_a_partition_makes_it_unknown,
    concurrent_acquires_hand_out_exactly_one_lease,
    ping_succeeds,
}

#[tokio::test]
async fn postgres_migrations_are_idempotent_under_concurrent_starts() {
    let Some(url) = database_url() else { return };
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
