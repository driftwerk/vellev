use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use futures::future::join_all;
use vellev_core::{NodeId, Partition, TenantId};
use vellev_storage::coordinator::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease, LogPosition,
    ManifestRef, ManualClock, MemoryCoordinator, PartitionKey, PostgresCoordinator, Versions,
    MAX_TTL,
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
    let (lease, leased) = lease(h, &key, &node("a")).await;

    let committed = c.commit(&lease, commit(10, "m10")).await.unwrap();
    assert_eq!(
        committed,
        Committed {
            generation: lease.generation,
            incarnation: leased.incarnation,
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
            incarnation: leased.incarnation,
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
    let (stalled, leased) = lease(h, &key, &node("a")).await;
    let registered_as = leased.incarnation;
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
            incarnation: registered_as,
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

async fn the_incarnation_stays_through_leases_and_grows_when_registered_again(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let incarnation = || async { c.committed(&key).await.unwrap().unwrap().incarnation };
    let first = incarnation().await;
    let (held, committed) = lease(h, &key, &node("a")).await;
    assert_eq!(committed.incarnation, first);
    let landed = c.commit(&held, commit(1, "m1")).await.unwrap();
    assert_eq!(landed.incarnation, first);
    c.release(&held).await.unwrap();
    let (_, committed) = lease(h, &key, &node("b")).await;
    assert_eq!(committed.incarnation, first);
    assert_eq!(incarnation().await, first);

    c.register(&key).await.unwrap();
    assert_eq!(
        incarnation().await,
        first,
        "registering again changes nothing"
    );
    assert!(c.remove(&key).await.unwrap());
    c.register(&key).await.unwrap();
    assert!(incarnation().await > first);
}

async fn the_oldest_versions_are_the_least_of_the_nodes_registered(h: &dyn Harness) {
    let c = h.coordinator();
    let at = |format, protocol| Versions { format, protocol };
    let (a, b) = (node("a"), node("b"));
    c.announce(&a, at(7, 0)).await.unwrap();
    c.announce(&b, at(0, 9)).await.unwrap();
    let oldest = c.oldest().await.unwrap().unwrap();
    assert_eq!(oldest, at(0, 0).least(oldest), "{oldest:?}");
    c.announce(&b, at(6, 11)).await.unwrap();
    c.depart(&a).await.unwrap();
    c.depart(&b).await.unwrap();
    assert_eq!(c.format().await.unwrap(), 1);
    assert_eq!(c.raise_format(1).await.unwrap(), 1);
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

/// Every partition, read `page` at a time. Other tests register and remove
/// their own meanwhile.
async fn listed(h: &dyn Harness, page: usize) -> Vec<PartitionKey> {
    let mut all: Vec<PartitionKey> = Vec::new();
    loop {
        let next = h.coordinator().partitions(all.last(), page).await.unwrap();
        assert!(next.len() <= page);
        if next.is_empty() {
            return all;
        }
        all.extend(next);
    }
}

async fn partitions_are_listed_in_key_order_a_page_at_a_time(h: &dyn Harness) {
    let c = h.coordinator();
    let shared = uuid::Uuid::new_v4().simple().to_string();
    let named = |tenant: &str, partition: u32| PartitionKey {
        tenant: TenantId::new(format!("{shared}{tenant}")).unwrap(),
        partition: Partition(partition),
    };
    let keys = [
        named("_", 0),
        named("a", 1),
        named("a", 0),
        named("-", 0),
        named("0", 0),
    ];
    for key in &keys {
        c.register(key).await.unwrap();
    }
    let mine = |all: Vec<PartitionKey>| -> Vec<PartitionKey> {
        all.into_iter().filter(|key| keys.contains(key)).collect()
    };
    let expected = [
        named("-", 0),
        named("0", 0),
        named("_", 0),
        named("a", 0),
        named("a", 1),
    ];
    for page in [1, 2, 1_000] {
        let all = listed(h, page).await;
        assert!(all.windows(2).all(|pair| pair[0] < pair[1]), "page {page}");
        assert_eq!(mine(all), expected, "page {page}");
    }
    assert_eq!(c.partitions(None, 0).await.unwrap(), []);
    assert_eq!(
        c.partitions(Some(&named("a", 0)), 1).await.unwrap(),
        [named("a", 1)]
    );

    c.remove(&named("a", 0)).await.unwrap();
    assert_eq!(mine(listed(h, 3).await).len(), 4);
}

async fn a_key_reaches_its_tenant_until_revoked_or_the_tenant_goes(h: &dyn Harness) {
    let c = h.coordinator();
    let (tenant, other) = (key(), key());
    let hash = |n: u8| [n; 32];
    let secret = |n: u8| {
        let mut hash = *uuid::Uuid::new_v4().as_bytes();
        hash[0] = n;
        [hash, hash].concat().try_into().unwrap()
    };
    let unregistered = c
        .issue_key(&tenant.tenant, uuid::Uuid::new_v4(), &hash(1))
        .await
        .unwrap();
    assert_eq!(
        unregistered, None,
        "no key for a tenant that does not exist"
    );

    c.register(&tenant).await.unwrap();
    c.register(&other).await.unwrap();
    let (first, second, theirs): ([u8; 32], [u8; 32], [u8; 32]) = (secret(1), secret(2), secret(3));
    let issued = c
        .issue_key(&tenant.tenant, uuid::Uuid::new_v4(), &first)
        .await
        .unwrap()
        .unwrap();
    let later = c
        .issue_key(&tenant.tenant, uuid::Uuid::new_v4(), &second)
        .await
        .unwrap()
        .unwrap();
    c.issue_key(&other.tenant, uuid::Uuid::new_v4(), &theirs)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        c.key_tenant(&first).await.unwrap(),
        Some(tenant.tenant.clone())
    );
    assert_eq!(
        c.key_tenant(&theirs).await.unwrap(),
        Some(other.tenant.clone())
    );
    assert_eq!(c.key_tenant(&secret(4)).await.unwrap(), None);
    let listed: Vec<uuid::Uuid> = c
        .keys(&tenant.tenant)
        .await
        .unwrap()
        .iter()
        .map(|key| key.id)
        .collect();
    assert_eq!(listed.len(), 2);
    assert!(listed.contains(&issued.id) && listed.contains(&later.id));

    assert!(
        !c.revoke_key(&other.tenant, issued.id).await.unwrap(),
        "only by its own tenant"
    );
    assert!(c.revoke_key(&tenant.tenant, issued.id).await.unwrap());
    assert_eq!(c.key_tenant(&first).await.unwrap(), None);
    assert!(!c.revoke_key(&tenant.tenant, issued.id).await.unwrap());

    c.remove(&tenant).await.unwrap();
    c.register(&tenant).await.unwrap();
    assert_eq!(
        c.key_tenant(&second).await.unwrap(),
        None,
        "a tenant made again under its id is not reached by its old keys"
    );
    assert_eq!(c.keys(&tenant.tenant).await.unwrap(), []);
    assert_eq!(c.key_tenant(&theirs).await.unwrap(), Some(other.tenant));
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
    the_incarnation_stays_through_leases_and_grows_when_registered_again,
    the_oldest_versions_are_the_least_of_the_nodes_registered,
    removing_a_partition_makes_it_unknown,
    concurrent_acquires_hand_out_exactly_one_lease,
    partitions_are_listed_in_key_order_a_page_at_a_time,
    a_key_reaches_its_tenant_until_revoked_or_the_tenant_goes,
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

#[tokio::test]
async fn a_node_starts_on_a_database_a_newer_release_has_migrated() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let ahead = 1_000_000 + i64::from(rand_suffix());
    sqlx::query("INSERT INTO vellev.schema_version (version, sql) VALUES ($1, 'SELECT 1')")
        .bind(ahead)
        .execute(&pg.pool)
        .await
        .unwrap();
    let started = PostgresCoordinator::connect(&database_url().unwrap(), 2).await;
    sqlx::query("DELETE FROM vellev.schema_version WHERE version = $1")
        .bind(ahead)
        .execute(&pg.pool)
        .await
        .unwrap();
    started.expect("a node of the older release starts");
}

fn rand_suffix() -> u16 {
    uuid::Uuid::new_v4().as_u128() as u16
}

#[tokio::test]
async fn a_row_inserted_as_before_incarnations_gets_its_generation_as_one() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let key = PartitionKey {
        tenant: TenantId::new(format!("older-{}", uuid::Uuid::new_v4().simple())).unwrap(),
        partition: Partition::DEFAULT,
    };
    sqlx::query(
        "INSERT INTO vellev.partitions (tenant, partition, generation)
         VALUES ($1, $2, nextval('vellev.generations'))",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    let committed = pg.coordinator.committed(&key).await.unwrap().unwrap();
    assert_eq!(committed.incarnation.0, committed.generation.0);
}

#[tokio::test]
async fn the_written_format_rises_only_when_every_node_reads_it_and_never_falls() {
    let c = MemoryCoordinator::default();
    let at = |format| Versions {
        format,
        protocol: 1,
    };
    let (newer, older) = (node("newer"), node("older"));
    c.announce(&newer, at(2)).await.unwrap();
    c.announce(&older, at(1)).await.unwrap();
    assert_eq!(
        c.raise_format(2).await.unwrap(),
        1,
        "a node still reads only 1"
    );
    assert_eq!(c.oldest().await.unwrap(), Some(at(1)));
    c.depart(&older).await.unwrap();
    assert_eq!(c.raise_format(2).await.unwrap(), 2);
    c.announce(&older, at(1)).await.unwrap();
    assert_eq!(c.raise_format(1).await.unwrap(), 2, "it never falls");
    assert_eq!(c.format().await.unwrap(), 2);
}
