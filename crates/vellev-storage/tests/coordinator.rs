use std::future::Future;
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use futures::future::join_all;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use vellev_core::{NodeId, Partition, RequestId, TenantId};
use vellev_storage::coordinator::{
    Acquired, Adoption, Bounded, Cleaned, Commit, Committed, Coordinator, CoordinatorError,
    Generation, Identity, Incarnation, Lease, Lineage, LogPosition, ManifestRef, ManualClock,
    MemoryCoordinator, PartitionKey, Point, PostgresCoordinator, Prior, PurgeClaim, Purges,
    Removed, Restore, RestorePoint, Restored, ShipBase, Shipping, Stale, TenantKey, Unshipped,
    Versions, MAX_BACKUP_HOLD, MAX_RESTORE_WINDOW, MAX_TTL, RESTORES_KEPT_FOR,
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

/// How many connections to Postgres the tests here hold between them at
/// once: well under what a server allows, a hundred unless it is told
/// otherwise, which other suites may be drawing on meanwhile. A test waits
/// its turn for its connections rather than be refused them.
const CONNECTIONS: u32 = 24;
/// How many connections a harness holds: one for the coordinator under
/// test, and one to move time with or look on with while it works.
const HARNESS: u32 = 2;
/// How long anything here is waited for before it is taken to have hung:
/// what stops a broken test, not what a slow one is held to.
const HUNG: Duration = Duration::from_secs(120);

static BUDGET: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(CONNECTIONS as usize)));

/// The right to hold `connections` connections to Postgres until it is
/// dropped: all a test needs, taken at once, so that no two tests wait on
/// connections the other holds.
async fn granted(connections: u32) -> OwnedSemaphorePermit {
    assert!(connections <= CONNECTIONS, "more than there are to grant");
    BUDGET
        .clone()
        .acquire_many_owned(connections)
        .await
        .expect("the budget is never closed")
}

/// A pool of at most `connections` by `options`, which must be granted.
async fn pool(options: PgConnectOptions, connections: u32) -> sqlx::PgPool {
    PgPoolOptions::new()
        .max_connections(connections)
        .acquire_timeout(HUNG)
        .connect_with(options)
        .await
        .expect("connect to test database")
}

/// A Postgres coordinator on a database of its own: the server is shared
/// with whatever else is being tested, of this release and of others, whose
/// migrations and tenants are not this test's.
struct Pg {
    coordinator: PostgresCoordinator,
    pool: sqlx::PgPool,
    scratch: ScratchDatabase,
    _granted: OwnedSemaphorePermit,
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
        Self::connect_beside(0).await
    }

    /// A harness, with `others` more connections granted for pools the
    /// test opens beside it.
    async fn connect_beside(others: u32) -> Option<Self> {
        let options = database_options()?;
        let granted = granted(HARNESS + others + 1).await;
        let scratch = ScratchDatabase::create(options).await;
        let pool = pool(scratch.options(), HARNESS).await;
        let coordinator = PostgresCoordinator::with_pool(pool.clone())
            .await
            .expect("migrate");
        Some(Self {
            coordinator,
            pool,
            scratch,
            _granted: granted,
        })
    }

    /// The options to connect to this harness's database by.
    fn options(&self) -> PgConnectOptions {
        self.scratch.options()
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
             SET lease_expires_at = lease_expires_at - make_interval(secs => $3),
                 backup_claimed_until = backup_claimed_until - make_interval(secs => $3),
                 backed_up_at = backed_up_at - make_interval(secs => $3)
             WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(duration.as_secs_f64())
        .execute(&self.pool)
        .await
        .expect("rewind lease and backups");
        sqlx::query(
            "UPDATE vellev.commits
             SET committed_at = committed_at - make_interval(secs => $3)
             WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(duration.as_secs_f64())
        .execute(&self.pool)
        .await
        .expect("rewind history");
        sqlx::query(
            "UPDATE vellev.restores
             SET made_at = made_at - make_interval(secs => $3)
             WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(duration.as_secs_f64())
        .execute(&self.pool)
        .await
        .expect("rewind restores");
        sqlx::query(
            "UPDATE vellev.purges
             SET requested_at = requested_at - make_interval(secs => $2),
                 purged_at = purged_at - make_interval(secs => $2),
                 claimed_until = claimed_until - make_interval(secs => $2),
                 looked_at = looked_at - make_interval(secs => $2)
             WHERE tenant = $1",
        )
        .bind(key.tenant.as_str())
        .bind(duration.as_secs_f64())
        .execute(&self.pool)
        .await
        .expect("rewind purges");
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
        after: None,
        log_position: LogPosition(position),
        manifest: Some(ManifestRef(manifest.into())),
        restore: None,
        adopting: None,
    }
}

/// The commit at `position` that takes up the backup of `lineage`, as a
/// restore asked to adopt makes it.
fn adopting_commit(position: u64, manifest: &str, lineage: Lineage) -> Commit {
    Commit {
        adopting: Some(lineage),
        ..commit(position, manifest)
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
            after: None,
            log_position: LogPosition(11),
            manifest: None,
            restore: None,
            adopting: None,
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

async fn a_commit_is_made_only_on_what_it_says_it_follows(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (lease, leased) = lease(h, &key, &node("a")).await;
    let after = |position: u64, head: Option<&str>, commit: Commit| Commit {
        after: Some(Prior {
            position: LogPosition(position),
            head: head.map(|head| ManifestRef(head.into())),
        }),
        ..commit
    };
    let start = leased.log_position.0;
    let first = after(start, None, commit(4, "m4"));
    let made = c.commit(&lease, first.clone()).await.unwrap();

    // Past what is committed and under the lease, but built on a position
    // or on a head that is not what is committed: refused, whichever.
    for stale in [
        after(start, None, commit(9, "m9")),
        after(4, Some("other"), commit(9, "m9")),
        after(4, None, commit(9, "m9")),
        after(3, Some("m4"), commit(9, "m9")),
    ] {
        let refused = c.commit(&lease, stale.clone()).await;
        assert!(
            matches!(
                refused,
                Err(CoordinatorError::Regressed {
                    committed: LogPosition(4),
                    attempted: LogPosition(9),
                    ..
                })
            ),
            "{stale:?}: {refused:?}"
        );
    }
    assert_eq!(c.committed(&key).await.unwrap(), Some(made.clone()));

    // The very commit made, asked again: made already, whatever it says it
    // followed, as its writer said before the answer was lost.
    assert_eq!(c.commit(&lease, first).await.unwrap(), made);
    // And one that follows it.
    let next = after(4, Some("m4"), commit(9, "m9"));
    let made = c.commit(&lease, next.clone()).await.unwrap();
    assert_eq!(made.log_position, LogPosition(9));
    assert_eq!(c.commit(&lease, next).await.unwrap(), made);
    // One that says nothing of what it follows is made on whatever is
    // committed, as releases before this one commit.
    c.commit(&lease, commit(12, "m12")).await.unwrap();
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
    purge(c, &key.tenant).await;
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
    purge(c, &key.tenant).await;
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
    purge(c, &tenant.tenant).await;
    c.register(&tenant).await.unwrap();
    assert_eq!(
        c.key_tenant(&second).await.unwrap(),
        None,
        "a tenant made again under its id is not reached by its old keys"
    );
    assert_eq!(c.keys(&tenant.tenant).await.unwrap(), []);
    assert_eq!(c.key_tenant(&theirs).await.unwrap(), Some(other.tenant));
}

const HOUR: Duration = Duration::from_secs(60 * 60);

/// The claim to purge `tenant`, removed however lately, if it is there to
/// take.
async fn purge_claimed(coordinator: &dyn Coordinator, tenant: &TenantId) -> Option<PurgeClaim> {
    coordinator
        .claim_purge(tenant, Duration::ZERO, HOUR)
        .await
        .unwrap()
}

/// Purge a removed tenant as a node would, so that it may be made again.
async fn purge(coordinator: &dyn Coordinator, tenant: &TenantId) {
    let claim = purge_claimed(coordinator, tenant).await.unwrap();
    coordinator
        .purged(tenant, claim.id, 24 * HOUR)
        .await
        .unwrap();
}

/// Which tenant `tenant` is now.
async fn identity(h: &dyn Harness, tenant: &TenantId) -> Option<Identity> {
    h.coordinator().identity(tenant).await.unwrap()
}

/// Another partition of the tenant of `key`.
fn beside(key: &PartitionKey, partition: u32) -> PartitionKey {
    PartitionKey {
        tenant: key.tenant.clone(),
        partition: Partition(partition),
    }
}

async fn a_tenant_keeps_its_lineage_until_it_is_removed_and_one_made_again_has_another(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    assert_eq!(identity(h, &key().tenant).await, None, "no tenant");
    let key = registered(h).await;
    let tenant = &key.tenant;
    let first = identity(h, tenant).await.unwrap();
    assert!(!first.committed);

    let other = beside(&key, 1);
    c.register(&other).await.unwrap();
    c.register(&key).await.unwrap();
    assert_eq!(
        identity(h, tenant).await,
        Some(first),
        "its partitions are of one"
    );
    let (held, _) = lease(h, &other, &node("a")).await;
    c.commit(&held, commit(1, "m1")).await.unwrap();
    c.release(&held).await.unwrap();
    let written = Identity {
        committed: true,
        ..first
    };
    assert_eq!(identity(h, tenant).await, Some(written));
    assert_ne!(
        identity(h, &registered(h).await.tenant)
            .await
            .unwrap()
            .lineage,
        first.lineage,
        "another tenant has its own"
    );

    assert!(c.remove(&key).await.unwrap());
    assert_eq!(
        identity(h, tenant).await.unwrap().lineage,
        first.lineage,
        "while any partition of it is left"
    );
    assert!(c.remove(&other).await.unwrap());
    assert_eq!(identity(h, tenant).await, None);
    let claim = purge_claimed(c, tenant).await.unwrap();
    assert_eq!(claim.removed.tenant, *tenant);
    assert_eq!(claim.removed.lineage, first.lineage);
    c.purged(tenant, claim.id, 24 * HOUR).await.unwrap();

    c.register(&key).await.unwrap();
    let again = identity(h, tenant).await.unwrap();
    assert_ne!(again.lineage, first.lineage, "made again, it is another");
    assert!(!again.committed);
}

async fn partitions_first_registered_together_are_of_one_lineage(h: &dyn Harness) {
    let c = h.coordinator();
    for _ in 0..20 {
        let key = key();
        let (other, third) = (beside(&key, 1), beside(&key, 2));
        let (a, b, d) = tokio::join!(c.register(&key), c.register(&other), c.register(&third));
        for registered in [a, b, d] {
            registered.unwrap();
        }
        let lineage = identity(h, &key.tenant).await.unwrap().lineage;
        for removed in [&key, &other] {
            assert!(c.remove(removed).await.unwrap());
            assert_eq!(
                identity(h, &key.tenant).await.unwrap().lineage,
                lineage,
                "every partition of {} is of one",
                key.tenant
            );
        }
    }
}

/// What a tenant is to become with the commit at `at`: of a lineage of
/// its own, with one key and a restore window of an hour.
fn adoption(at: u64) -> Adoption {
    let id = uuid::Uuid::new_v4();
    Adoption {
        lineage: Lineage(uuid::Uuid::new_v4()),
        at: LogPosition(at),
        keys: vec![TenantKey {
            id,
            created: Utc.with_ymd_and_hms(2026, 10, 1, 8, 0, 0).unwrap(),
            hash: hash_of(id),
        }],
        restore_window: Some(HOUR),
    }
}

/// Thirty-two bytes that differ with `id`, as the hashes of keys do.
fn hash_of(id: uuid::Uuid) -> [u8; 32] {
    let mut hash = [0; 32];
    hash[..16].copy_from_slice(id.as_bytes());
    hash
}

/// The lineage, key ids and restore window `tenant` has now.
async fn taken_up(
    h: &dyn Harness,
    tenant: &TenantId,
) -> (Lineage, Vec<uuid::Uuid>, Option<Duration>) {
    let c = h.coordinator();
    let keys = c.keys(tenant).await.unwrap();
    (
        identity(h, tenant).await.unwrap().lineage,
        keys.iter().map(|key| key.id).collect(),
        c.restore_window(tenant).await.unwrap(),
    )
}

fn adopting(refused: &Result<Committed, CoordinatorError>) -> bool {
    matches!(refused, Err(CoordinatorError::Adopting(_)))
}

fn unadopted(refused: &Result<Committed, CoordinatorError>) -> bool {
    matches!(refused, Err(CoordinatorError::Unadopted(_)))
}

async fn a_tenant_takes_a_backup_up_with_the_commit_that_restores_it_and_nothing_before(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = &key.tenant;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = taken_up(h, tenant).await;
    let to_be = adoption(5);
    assert!(c
        .adopt_on_commit(&key, held.generation, &to_be)
        .await
        .unwrap());
    assert_eq!(taken_up(h, tenant).await, own, "nothing is taken yet");

    // No commit but the one named is made meanwhile, and it takes nothing.
    assert!(adopting(&c.commit(&held, commit(1, "m1")).await));
    let emptied = Commit {
        after: None,
        log_position: LogPosition(5),
        manifest: None,
        restore: None,
        adopting: None,
    };
    assert!(adopting(&c.commit(&held, emptied).await));
    // Nor the commit at the position named, unless it says it adopts: a
    // restore meant to take the backup up never lands as a plain one.
    assert!(adopting(&c.commit(&held, commit(5, "restored")).await));
    assert_eq!(taken_up(h, tenant).await, own);
    assert!(!identity(h, tenant).await.unwrap().committed);

    let restoring = adopting_commit(5, "restored", to_be.lineage);
    let restored = c.commit(&held, restoring.clone()).await.unwrap();
    assert_eq!(restored.log_position, LogPosition(5));
    let key_ids = vec![to_be.keys[0].id];
    assert_eq!(
        taken_up(h, tenant).await,
        (to_be.lineage, key_ids, Some(HOUR)),
        "all of it, with the commit"
    );
    let kept = c.keys(tenant).await.unwrap();
    assert_eq!(kept, to_be.keys, "as the backup kept them");
    let reached = c.key_tenant(&to_be.keys[0].hash).await.unwrap();
    assert_eq!(reached.as_ref(), Some(tenant));
    // Asked again, as a writer whose answer was lost asks, it is answered
    // as it was: saying it adopts, or not.
    assert_eq!(c.commit(&held, restoring).await.unwrap(), restored);
    assert_eq!(
        c.commit(&held, commit(5, "restored")).await.unwrap(),
        restored
    );

    // Once and no more: what is changed after stays changed.
    assert!(c.revoke_key(tenant, to_be.keys[0].id).await.unwrap());
    c.commit(&held, commit(6, "m6")).await.unwrap();
    assert_eq!(
        taken_up(h, tenant).await,
        (to_be.lineage, Vec::new(), Some(HOUR))
    );
    assert!(!c
        .adopt_on_commit(&key, held.generation, &adoption(9))
        .await
        .unwrap());
}

async fn an_adoption_is_noted_only_for_an_empty_tenant_without_keys_by_the_lease_named(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let to_be = adoption(3);
    let noted = |key: PartitionKey, generation: Generation| {
        let to_be = to_be.clone();
        async move { c.adopt_on_commit(&key, generation, &to_be).await.unwrap() }
    };
    assert!(!noted(key(), Generation(1)).await, "no partition");
    let key = registered(h).await;
    let tenant = &key.tenant;
    let (held, _) = lease(h, &key, &node("a")).await;
    let moved = Generation(held.generation.0 + 1);
    assert!(!noted(key.clone(), moved).await, "not the lease held");

    let issued = c.issue_key(tenant, uuid::Uuid::new_v4(), &[9; 32]).await;
    let issued = issued.unwrap().unwrap();
    assert!(!noted(key.clone(), held.generation).await, "it has a key");
    assert!(c.revoke_key(tenant, issued.id).await.unwrap());

    let other = beside(&key, 1);
    c.register(&other).await.unwrap();
    let (elsewhere, _) = lease(h, &other, &node("a")).await;
    c.commit(&elsewhere, commit(1, "m1")).await.unwrap();
    assert!(
        !noted(key.clone(), held.generation).await,
        "another partition of it has committed"
    );
    assert!(c.remove(&other).await.unwrap());
    assert!(noted(key.clone(), held.generation).await);

    let restoring = adopting_commit(3, "restored", to_be.lineage);
    assert!(c.commit(&held, restoring).await.is_ok());
    assert_eq!(identity(h, tenant).await.unwrap().lineage, to_be.lineage);
    assert!(
        !noted(key.clone(), held.generation).await,
        "it has committed"
    );
}

async fn an_adoption_is_given_up_replaced_or_outlived_by_its_lease(h: &dyn Harness) {
    let c = h.coordinator();
    // Given up: the tenant commits as any other, and is as it was.
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = taken_up(h, &key.tenant).await;
    let to_be = adoption(4);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    c.adopt_no_more(&key, held.generation).await.unwrap();
    let restoring = adopting_commit(4, "m4", to_be.lineage);
    assert!(unadopted(&c.commit(&held, restoring).await));
    c.commit(&held, commit(4, "m4")).await.unwrap();
    assert_eq!(taken_up(h, &key.tenant).await, own);

    // Asked again, for another backup: only the last is made.
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    let (first, second) = (adoption(4), adoption(7));
    for to_be in [&first, &second] {
        let noted = c.adopt_on_commit(&key, held.generation, to_be);
        assert!(noted.await.unwrap());
    }
    let restoring = adopting_commit(4, "first", first.lineage);
    assert!(adopting(&c.commit(&held, restoring).await));
    let restoring = adopting_commit(7, "second", second.lineage);
    c.commit(&held, restoring).await.unwrap();
    let key_ids = vec![second.keys[0].id];
    assert_eq!(
        taken_up(h, &key.tenant).await,
        (second.lineage, key_ids, Some(HOUR)),
        "nothing of the first is stacked on it"
    );

    // Outlived: the lease it was noted for is let go, and whoever takes
    // the partition next commits as to any other, and adopts nothing it
    // did not note itself.
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = taken_up(h, &key.tenant).await;
    let to_be = adoption(4);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    c.release(&held).await.unwrap();
    let (next, _) = lease(h, &key, &node("b")).await;
    let restoring = adopting_commit(4, "m4", to_be.lineage);
    assert!(unadopted(&c.commit(&next, restoring).await));
    c.commit(&next, commit(4, "m4")).await.unwrap();
    assert_eq!(taken_up(h, &key.tenant).await, own);
    c.commit(&next, commit(5, "m5")).await.unwrap();
}

async fn an_adoption_is_given_up_only_by_the_lease_it_was_noted_for(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = &key.tenant;
    // A node notes an adoption, stalls past its lease, and another takes
    // the partition and notes its own. The first, going on, gives its
    // adoption up: that is nothing to the second's, which it makes.
    let (stale, _) = lease(h, &key, &node("a")).await;
    let first = adoption(4);
    let noted = c.adopt_on_commit(&key, stale.generation, &first);
    assert!(noted.await.unwrap());
    h.pass(&key, TTL + Duration::from_secs(1)).await;
    let (held, _) = lease(h, &key, &node("b")).await;
    assert_ne!(held.generation, stale.generation);
    let to_be = adoption(6);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    c.adopt_no_more(&key, stale.generation).await.unwrap();
    assert!(
        adopting(&c.commit(&held, commit(6, "plain")).await),
        "the second's adoption is noted still"
    );
    let restoring = adopting_commit(6, "restored", to_be.lineage);
    c.commit(&held, restoring).await.unwrap();
    let key_ids = vec![to_be.keys[0].id];
    assert_eq!(
        taken_up(h, tenant).await,
        (to_be.lineage, key_ids, Some(HOUR))
    );

    // Given up by the lease that noted it, it is gone.
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = taken_up(h, &key.tenant).await;
    let to_be = adoption(2);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    c.adopt_no_more(&key, held.generation).await.unwrap();
    let restoring = adopting_commit(2, "restored", to_be.lineage);
    assert!(unadopted(&c.commit(&held, restoring).await));
    assert_eq!(taken_up(h, &key.tenant).await, own);
}

async fn a_commit_that_says_it_adopts_is_refused_unless_that_adoption_is_noted_for_its_lease(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = &key.tenant;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = taken_up(h, tenant).await;
    // Nothing noted: refused, and nothing is committed.
    let unasked = Lineage(uuid::Uuid::new_v4());
    let restoring = adopting_commit(1, "m1", unasked);
    assert!(unadopted(&c.commit(&held, restoring).await));
    assert!(!identity(h, tenant).await.unwrap().committed);
    assert_eq!(taken_up(h, tenant).await, own);

    // Noted: made only by the commit that says that lineage, at that
    // position.
    let to_be = adoption(3);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    let restoring = adopting_commit(3, "m3", unasked);
    assert!(adopting(&c.commit(&held, restoring).await));
    // A commit that neither follows nor repeats what is committed, here
    // nothing, under a head, is told that and nothing of adoptions.
    for nowhere in [commit(0, "m"), adopting_commit(0, "m", to_be.lineage)] {
        let refused = c.commit(&held, nowhere).await;
        assert!(
            matches!(refused, Err(CoordinatorError::Regressed { .. })),
            "{refused:?}"
        );
    }
    let restoring = adopting_commit(2, "m2", to_be.lineage);
    assert!(adopting(&c.commit(&held, restoring).await));
    assert_eq!(taken_up(h, tenant).await, own);
    let restoring = adopting_commit(3, "m3", to_be.lineage);
    let made = c.commit(&held, restoring.clone()).await.unwrap();
    assert_eq!(taken_up(h, tenant).await.0, to_be.lineage);

    // Made, the very commit asked again is answered as it was; any other
    // that says it adopts is refused, the adoption being made already.
    assert_eq!(c.commit(&held, restoring).await.unwrap(), made);
    let restoring = adopting_commit(3, "m3", unasked);
    assert!(unadopted(&c.commit(&held, restoring).await));
    // Another position under the same head is another commit too; and
    // the same position under another head follows nothing, adopting or
    // not.
    for (position, head) in [(4, "m4"), (4, "m3")] {
        let restoring = adopting_commit(position, head, to_be.lineage);
        let refused = c.commit(&held, restoring).await;
        assert!(unadopted(&refused), "{position} {head}: {refused:?}");
    }
    let refused = c
        .commit(&held, adopting_commit(3, "other", to_be.lineage))
        .await;
    assert!(
        matches!(refused, Err(CoordinatorError::Regressed { .. })),
        "{refused:?}"
    );
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().log_position,
        LogPosition(3)
    );
    c.commit(&held, commit(4, "m4")).await.unwrap();
}

async fn a_lease_moved_on_is_told_so_whatever_its_commit_adopts_and_whatever_is_noted(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (stale, _) = lease(h, &key, &node("a")).await;
    let first = adoption(2);
    let noted = c.adopt_on_commit(&key, stale.generation, &first);
    assert!(noted.await.unwrap());
    h.pass(&key, TTL + Duration::from_secs(1)).await;
    let (held, _) = lease(h, &key, &node("b")).await;
    let told = |commit: Commit, what: &'static str| {
        let (stale, current) = (stale.clone(), held.generation);
        async move {
            let refused = c.commit(&stale, commit).await;
            assert!(fenced(&refused, current), "{what}: {refused:?}");
        }
    };
    // Nothing noted for the lease that took over.
    told(adopting_commit(2, "s", first.lineage), "adopting").await;
    told(commit(1, "p"), "plain").await;
    // An adoption noted for it: the stale lease's own, a plain commit, and
    // one saying the very lineage noted for the other.
    let to_be = adoption(2);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    told(adopting_commit(2, "s", first.lineage), "adopting its own").await;
    told(commit(1, "p"), "plain, another noted").await;
    told(
        adopting_commit(2, "s", to_be.lineage),
        "adopting the other's",
    )
    .await;

    // Nothing was committed, and the adoption noted is made by its lease.
    assert!(!identity(h, &key.tenant).await.unwrap().committed);
    let restoring = adopting_commit(2, "r", to_be.lineage);
    c.commit(&held, restoring).await.unwrap();
    assert_eq!(taken_up(h, &key.tenant).await.0, to_be.lineage);
}

/// Every answer to a commit, by what is committed, whose lease commits,
/// what adoption is noted and for whom, how the commit stands to what is
/// committed, and what it says of adopting: asked in one order whatever
/// the coordinator. The lease first: one moved on is told so. Then the
/// commit: one that neither follows what is committed, by what it says
/// it follows too, nor repeats it, is told that. Only then the adoption.
async fn a_commit_is_answered_by_its_lease_then_what_it_follows_then_the_adoption(h: &dyn Harness) {
    let c = h.coordinator();
    let named = |head: &str| Some(ManifestRef(head.into()));
    for committed in [false, true] {
        for stale in [false, true] {
            for noted in ["none", "for the stale lease", "for the current lease"] {
                for relation in [
                    "advances",
                    "repeats",
                    "neither",
                    "after another head",
                    "after another position",
                ] {
                    for says in ["nothing", "the lineage noted", "another lineage"] {
                        let cell = format!(
                            "committed {committed}, stale {stale}, noted {noted}, {relation}, says {says}"
                        );
                        let key = registered(h).await;
                        let (first, _) = lease(h, &key, &node("a")).await;
                        if committed {
                            c.commit(&first, commit(1, "base")).await.unwrap();
                        }
                        let own = identity(h, &key.tenant).await.unwrap().lineage;
                        let to_be = adoption(2);
                        if noted == "for the stale lease" {
                            let noted = c.adopt_on_commit(&key, first.generation, &to_be);
                            assert_eq!(noted.await.unwrap(), !committed, "{cell}");
                        }
                        h.pass(&key, TTL + Duration::from_secs(1)).await;
                        let (current, _) = lease(h, &key, &node("b")).await;
                        if noted == "for the current lease" {
                            let noted = c.adopt_on_commit(&key, current.generation, &to_be);
                            assert_eq!(noted.await.unwrap(), !committed, "{cell}");
                        }
                        let at = u64::from(committed);
                        let head = committed.then_some("base");
                        let (position, manifest) = match relation {
                            "repeats" => (at, head.and_then(named)),
                            "neither" => (at, named("another")),
                            _ => (2, named("m")),
                        };
                        let after = match relation {
                            "after another head" => Some(Prior {
                                position: LogPosition(at),
                                head: named("nope"),
                            }),
                            "after another position" => Some(Prior {
                                position: LogPosition(at + 1),
                                head: head.and_then(named),
                            }),
                            _ => None,
                        };
                        let asked = Commit {
                            after,
                            log_position: LogPosition(position),
                            manifest,
                            restore: None,
                            adopting: match says {
                                "nothing" => None,
                                "the lineage noted" => Some(to_be.lineage),
                                _ => Some(Lineage(uuid::Uuid::new_v4())),
                            },
                        };
                        let by = if stale { &first } else { &current };
                        let answered = c.commit(by, asked).await;

                        let adopting = !committed && noted == "for the current lease";
                        let follows = matches!(relation, "advances" | "repeats");
                        let adopts = relation == "advances" && says == "the lineage noted";
                        let expected = match () {
                            () if stale => "fenced",
                            () if !follows => "does not follow",
                            () if adopting && adopts => "adopted",
                            () if adopting => "adopting",
                            () if says != "nothing" => "unadopted",
                            () => "made",
                        };
                        let got = match &answered {
                            Ok(_) if adopting && adopts => "adopted",
                            Ok(_) => "made",
                            Err(CoordinatorError::Fenced { .. }) => "fenced",
                            Err(CoordinatorError::Regressed { .. }) => "does not follow",
                            Err(CoordinatorError::Adopting(_)) => "adopting",
                            Err(CoordinatorError::Unadopted(_)) => "unadopted",
                            Err(_) => "something else",
                        };
                        assert_eq!(got, expected, "{cell}: {answered:?}");

                        // And nothing is left changed by a commit refused.
                        let now = c.committed(&key).await.unwrap().unwrap();
                        let lineage = identity(h, &key.tenant).await.unwrap().lineage;
                        match expected {
                            "adopted" => assert_eq!(lineage, to_be.lineage, "{cell}"),
                            _ => assert_eq!(lineage, own, "{cell}"),
                        }
                        match expected {
                            "adopted" | "made" => {
                                assert_eq!(now.log_position, LogPosition(position), "{cell}")
                            }
                            _ => assert_eq!(now.log_position, LogPosition(at), "{cell}"),
                        }
                    }
                }
            }
        }
    }
}

async fn a_commit_repeated_does_not_record_its_restore_again(h: &dyn Harness) {
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    let c = h.coordinator();
    let restore = Restore {
        request: RequestId::generate(),
        asked: [1; 32],
        from: LogPosition(0),
        from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
    };
    let restoring = Commit {
        after: None,
        restore: Some(restore.clone()),
        ..commit(1, "head-1")
    };
    let made = c.commit(&lease, restoring.clone()).await.unwrap();
    let recorded = c.restored(&key, restore.request).await.unwrap();
    assert!(recorded.is_some());

    // Asked again nearly a day on, by a writer that never heard, it is
    // answered as made and makes nothing: the record is as old as the
    // restore is, and forgotten when the restore is a day old.
    h.pass(&key, RESTORES_KEPT_FOR - Duration::from_secs(60))
        .await;
    assert_eq!(c.commit(&lease, restoring.clone()).await.unwrap(), made);
    assert_eq!(c.restored(&key, restore.request).await.unwrap(), recorded);
    h.pass(&key, Duration::from_secs(120)).await;
    assert_eq!(c.restored(&key, restore.request).await.unwrap(), None);
    // Nor once it is forgotten: only a commit that advances records one.
    assert_eq!(c.commit(&lease, restoring).await.unwrap(), made);
    assert_eq!(c.restored(&key, restore.request).await.unwrap(), None);
}

async fn a_tenant_given_a_key_or_written_to_elsewhere_meanwhile_takes_nothing_up(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = &key.tenant;
    let other = beside(&key, 1);
    c.register(&other).await.unwrap();
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = identity(h, tenant).await.unwrap().lineage;
    let to_be = adoption(2);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());

    let issued = c.issue_key(tenant, uuid::Uuid::new_v4(), &[9; 32]).await;
    let issued = issued.unwrap().unwrap();
    let restoring = adopting_commit(2, "restored", to_be.lineage);
    assert!(adopting(&c.commit(&held, restoring.clone()).await));
    assert_eq!(identity(h, tenant).await.unwrap().lineage, own);
    assert!(c.revoke_key(tenant, issued.id).await.unwrap());

    let (elsewhere, _) = lease(h, &other, &node("a")).await;
    c.commit(&elsewhere, commit(1, "m1")).await.unwrap();
    assert!(adopting(&c.commit(&held, restoring.clone()).await));
    assert_eq!(
        taken_up(h, tenant).await,
        (own, Vec::new(), None),
        "written to, it is some tenant already"
    );
    c.adopt_no_more(&key, held.generation).await.unwrap();
    c.commit(&held, commit(2, "m2")).await.unwrap();
    assert_eq!(identity(h, tenant).await.unwrap().lineage, own);
}

async fn a_purge_claim_let_go_is_taken_again_at_once_and_any_other_is_left_be(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = key.tenant.clone();
    assert!(c.remove(&key).await.unwrap());
    let first = purge_claimed(c, &tenant).await.unwrap();
    assert!(purge_claimed(c, &tenant).await.is_none(), "held");

    // Let go by another id, it is held still; by its own, it is free.
    c.release_purge(&tenant, uuid::Uuid::new_v4())
        .await
        .unwrap();
    assert!(purge_claimed(c, &tenant).await.is_none());
    c.release_purge(&tenant, first.id).await.unwrap();
    let second = purge_claimed(c, &tenant).await.unwrap();
    assert_ne!(second.id, first.id);
    assert!(
        !c.renew_purge(&tenant, first.id, HOUR).await.unwrap(),
        "the one let go is not taken up again"
    );

    // Run out and taken over, the old holder's release leaves the new
    // claim be.
    h.pass(&key, HOUR + Duration::from_secs(1)).await;
    let third = purge_claimed(c, &tenant).await.unwrap();
    c.release_purge(&tenant, second.id).await.unwrap();
    assert!(c.renew_purge(&tenant, third.id, HOUR).await.unwrap());
    assert!(purge_claimed(c, &tenant).await.is_none());

    // Done, it is left be, and the tenant may be made again.
    c.purged(&tenant, third.id, 24 * HOUR).await.unwrap();
    c.release_purge(&tenant, third.id).await.unwrap();
    assert!(purge_claimed(c, &tenant).await.is_none());
    c.register(&key).await.unwrap();
}

async fn a_tenant_removed_before_it_took_a_backup_up_is_purged_as_the_tenant_it_was(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    let own = identity(h, &key.tenant).await.unwrap().lineage;
    let to_be = adoption(2);
    let noted = c.adopt_on_commit(&key, held.generation, &to_be);
    assert!(noted.await.unwrap());
    assert!(c.remove(&key).await.unwrap());
    let claim = purge_claimed(c, &key.tenant).await.unwrap();
    assert_eq!(claim.removed.lineage, own, "not the one it was to take up");
    c.purged(&key.tenant, claim.id, 24 * HOUR).await.unwrap();

    // Made again, nothing is left noted for it.
    c.register(&key).await.unwrap();
    let (held, _) = lease(h, &key, &node("a")).await;
    c.commit(&held, commit(2, "m2")).await.unwrap();
    assert_ne!(
        identity(h, &key.tenant).await.unwrap().lineage,
        to_be.lineage
    );
    assert_eq!(c.keys(&key.tenant).await.unwrap(), []);
}

async fn a_cluster_keeps_the_id_it_was_given(h: &dyn Harness) {
    let c = h.coordinator();
    let id = c.cluster().await.unwrap();
    registered(h).await;
    assert_eq!(c.cluster().await.unwrap(), id);
}

async fn a_tenant_still_being_purged_is_refused_saying_how_long_ago_that_was_asked(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    assert!(c.remove(&key).await.unwrap());
    h.pass(&key, HOUR).await;
    let refused = c.register(&key).await;
    let Err(CoordinatorError::Purging { tenant, asked }) = refused else {
        panic!("{refused:?}");
    };
    assert_eq!(tenant, key.tenant);
    assert!(
        asked >= HOUR && asked < HOUR + Duration::from_secs(60),
        "{asked:?}"
    );
}

async fn a_purge_is_claimed_only_once_due_and_renewed_only_while_it_is_the_claims_to_do(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let tenant = &key.tenant;
    let renewed =
        |claim: uuid::Uuid| async move { c.renew_purge(tenant, claim, HOUR).await.unwrap() };
    assert!(!renewed(uuid::Uuid::new_v4()).await, "never removed");
    assert!(c.remove(&key).await.unwrap());
    assert_eq!(
        c.claim_purge(tenant, HOUR, HOUR).await.unwrap(),
        None,
        "not removed an hour ago yet"
    );
    h.pass(&key, 2 * HOUR).await;
    let first = c.claim_purge(tenant, HOUR, HOUR).await.unwrap().unwrap();
    assert!(renewed(first.id).await);
    assert!(!renewed(uuid::Uuid::new_v4()).await, "not the claim held");

    h.pass(&key, HOUR / 2).await;
    assert!(renewed(first.id).await, "held on from when it is renewed");
    h.pass(&key, HOUR * 3 / 4).await;
    assert_eq!(purge_claimed(c, tenant).await, None, "it was renewed");
    h.pass(&key, HOUR).await;
    let second = purge_claimed(c, tenant).await.unwrap();
    assert_eq!(second.removed, first.removed);
    assert!(!renewed(first.id).await, "another took the purge");
    assert!(renewed(second.id).await);

    c.purged(tenant, second.id, 24 * HOUR).await.unwrap();
    assert!(!renewed(second.id).await, "the purge is done");
    c.register(&key).await.unwrap();
    assert!(!renewed(second.id).await, "and the tenant made again");
    assert!(!renewed(first.id).await);
    assert!(matches!(
        c.renew_purge(tenant, second.id, MAX_BACKUP_HOLD + Duration::from_secs(1))
            .await,
        Err(CoordinatorError::InvalidArgument(_))
    ));
}

async fn a_purged_tenant_is_looked_over_by_one_node_now_and_then_while_remembered(h: &dyn Harness) {
    const DAY: Duration = Duration::from_secs(24 * 60 * 60);
    let every = HOUR / 4;
    let c = h.coordinator();
    let looked = |limit: usize| async move { c.look_overs(DAY, every, limit).await.unwrap() };
    let mut removed = Vec::new();
    for _ in 0..3 {
        let key = registered(h).await;
        let lineage = identity(h, &key.tenant).await.unwrap().lineage;
        assert!(c.remove(&key).await.unwrap());
        removed.push((key, lineage));
    }
    assert_eq!(looked(8).await, [], "none is purged yet");
    for (key, _) in &removed[..2] {
        purge(c, &key.tenant).await;
    }
    assert_eq!(looked(8).await, [], "purging them looked them over");
    for (key, _) in &removed[..2] {
        h.pass(key, every).await;
    }
    let (first, second) = (looked(1).await, looked(8).await);
    assert_eq!((first.len(), second.len()), (1, 1), "as many as asked for");
    let mut seen: Vec<&Removed> = first.iter().chain(&second).collect();
    seen.sort_by(|a, b| a.tenant.cmp(&b.tenant));
    let mut purged: Vec<(TenantId, Lineage)> = removed[..2]
        .iter()
        .map(|(key, lineage)| (key.tenant.clone(), *lineage))
        .collect();
    purged.sort();
    assert_eq!(
        seen.iter()
            .map(|removed| (removed.tenant.clone(), removed.lineage))
            .collect::<Vec<_>>(),
        purged,
        "each is taken once, as the tenant it was"
    );
    assert_eq!(looked(8).await, [], "each was looked over just now");

    // Time is moved for the first alone, where it can be.
    let (first, _) = &removed[0];
    let due = |looked: Vec<Removed>| looked.iter().any(|removed| removed.tenant == first.tenant);
    h.pass(first, every / 2).await;
    assert_eq!(looked(8).await, []);
    h.pass(first, every).await;
    assert!(due(looked(8).await), "due again");
    assert!(!due(looked(8).await));

    // One made again is still looked over, for its backups.
    c.register(first).await.unwrap();
    h.pass(first, 2 * every).await;
    assert!(due(looked(8).await));
    h.pass(first, DAY).await;
    let left = looked(8).await;
    assert!(!due(left.clone()), "no longer remembered");
    let unpurged = &removed[2].0.tenant;
    assert!(left.iter().all(|removed| &removed.tenant != unpurged));
}

/// The positions of the earliest and latest states a restore reaching back
/// `window` may go to.
async fn reach(h: &dyn Harness, key: &PartitionKey, window: Duration) -> Option<(u64, u64)> {
    h.coordinator()
        .restore_points(key, window)
        .await
        .unwrap()
        .map(|(earliest, latest)| (earliest.position.0, latest.position.0))
}

async fn point(
    h: &dyn Harness,
    key: &PartitionKey,
    point: Point,
    window: Duration,
) -> Option<RestorePoint> {
    h.coordinator()
        .restore_point(key, point, window)
        .await
        .unwrap()
}

async fn every_commit_is_a_restore_point_once(h: &dyn Harness) {
    let key = registered(h).await;
    assert_eq!(reach(h, &key, HOUR).await, None, "nothing is committed yet");
    let (lease, _) = lease(h, &key, &node("a")).await;
    for position in 1..=3 {
        let head = format!("head-{position}");
        h.coordinator()
            .commit(&lease, commit(position, &head))
            .await
            .unwrap();
    }
    h.coordinator()
        .commit(&lease, commit(3, "head-3"))
        .await
        .unwrap();
    assert_eq!(reach(h, &key, HOUR).await, Some((1, 3)));
    let second = point(h, &key, Point::Position(LogPosition(2)), HOUR).await;
    assert_eq!(second.unwrap().head, ManifestRef("head-2".into()));
    assert_eq!(
        point(h, &key, Point::Position(LogPosition(4)), HOUR).await,
        None
    );
    assert_eq!(reach(h, &registered(h).await, HOUR).await, None);
}

async fn a_restore_reaches_back_through_its_window_and_to_how_it_then_stood(h: &dyn Harness) {
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    for position in 1..=3 {
        h.coordinator()
            .commit(&lease, commit(position, &format!("head-{position}")))
            .await
            .unwrap();
        h.pass(&key, 10 * HOUR).await;
    }
    // Committed 30, 20 and 10 hours ago.
    assert_eq!(reach(h, &key, 25 * HOUR).await, Some((1, 3)));
    assert_eq!(
        reach(h, &key, 15 * HOUR).await,
        Some((2, 3)),
        "the last commit before the window is how it stood when the window began"
    );
    let window = 15 * HOUR;
    let second = point(h, &key, Point::Position(LogPosition(2)), window)
        .await
        .unwrap();
    let third = point(h, &key, Point::Position(LogPosition(3)), window)
        .await
        .unwrap();
    assert_eq!(
        point(h, &key, Point::Position(LogPosition(1)), window).await,
        None
    );
    let later = |by: Duration| second.at + chrono::Duration::from_std(by).unwrap();
    for (at, expected) in [
        (later(6 * HOUR), Some(&second)),
        (later(11 * HOUR), Some(&third)),
        (later(19 * HOUR), Some(&third)),
        (later(4 * HOUR), None),
        (later(21 * HOUR), None),
    ] {
        assert_eq!(
            point(h, &key, Point::At(at), window).await.as_ref(),
            expected,
            "{at}"
        );
    }
    assert_eq!(
        h.coordinator().forget_history(&key, window).await.unwrap(),
        1
    );
    assert_eq!(reach(h, &key, 25 * HOUR).await, Some((2, 3)));
    assert_eq!(
        h.coordinator().forget_history(&key, window).await.unwrap(),
        0
    );
}

async fn removing_a_partition_forgets_its_history(h: &dyn Harness) {
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    h.coordinator()
        .commit(&lease, commit(1, "head-1"))
        .await
        .unwrap();
    h.coordinator().remove(&key).await.unwrap();
    assert_eq!(reach(h, &key, HOUR).await, None);
    forget_removed(h.coordinator(), &key, 1).await;
    purge(h.coordinator(), &key.tenant).await;
    h.coordinator().register(&key).await.unwrap();
    assert_eq!(
        reach(h, &key, HOUR).await,
        None,
        "nothing of before reaches the tenant made again"
    );
    assert_eq!(h.coordinator().forget_history(&key, HOUR).await.unwrap(), 0);
}

async fn a_restore_is_recorded_with_its_commit_for_a_day_and_for_its_incarnation(h: &dyn Harness) {
    let key = registered(h).await;
    let (lease, committed) = lease(h, &key, &node("a")).await;
    let coordinator = h.coordinator();
    coordinator
        .commit(&lease, commit(1, "head-1"))
        .await
        .unwrap();
    // A time to the microsecond, as fine as any a node asks with, is read
    // back as it was given.
    let restore = Restore {
        request: RequestId::generate(),
        asked: [1; 32],
        from: LogPosition(1),
        from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap()
            + chrono::Duration::microseconds(123_456),
    };
    let restoring = Commit {
        after: None,
        restore: Some(restore.clone()),
        ..commit(2, "head-2")
    };
    coordinator.commit(&lease, restoring.clone()).await.unwrap();
    coordinator.commit(&lease, restoring).await.unwrap();
    let restored = Restored {
        restore: restore.clone(),
        incarnation: committed.incarnation,
        position: LogPosition(2),
    };
    let asked = || coordinator.restored(&key, restore.request);
    assert_eq!(asked().await.unwrap(), Some(restored.clone()));
    assert_eq!(
        coordinator
            .restored(&key, RequestId::generate())
            .await
            .unwrap(),
        None
    );

    h.pass(&key, RESTORES_KEPT_FOR - Duration::from_secs(60))
        .await;
    coordinator.forget_history(&key, HOUR).await.unwrap();
    assert_eq!(asked().await.unwrap(), Some(restored));
    h.pass(&key, Duration::from_secs(120)).await;
    assert_eq!(asked().await.unwrap(), None, "a day old, it counts no more");
    coordinator.forget_history(&key, HOUR).await.unwrap();
    assert_eq!(asked().await.unwrap(), None, "and is forgotten");

    let again = Commit {
        after: None,
        restore: Some(restore.clone()),
        ..commit(3, "head-3")
    };
    coordinator.commit(&lease, again).await.unwrap();
    assert!(asked().await.unwrap().is_some());
    coordinator.remove(&key).await.unwrap();
    assert_eq!(asked().await.unwrap(), None, "the record goes with it");
    purge(coordinator, &key.tenant).await;
    coordinator.register(&key).await.unwrap();
    assert_eq!(asked().await.unwrap(), None);
}

async fn restores_a_day_old_are_forgotten_whether_or_not_their_partition_is_kept_up(
    h: &dyn Harness,
) {
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    let coordinator = h.coordinator();
    let request = RequestId::generate();
    let restoring = Commit {
        after: None,
        restore: Some(Restore {
            request,
            asked: [1; 32],
            from: LogPosition(0),
            from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
        }),
        ..commit(1, "head-1")
    };
    coordinator.commit(&lease, restoring).await.unwrap();
    h.pass(&key, RESTORES_KEPT_FOR + Duration::from_secs(60))
        .await;
    assert!(coordinator.forget_restores().await.unwrap() >= 1);
    assert_eq!(coordinator.restored(&key, request).await.unwrap(), None);
}

async fn a_tenant_keeps_its_own_restore_window_until_told_otherwise(h: &dyn Harness) {
    let key = registered(h).await;
    let tenant = &key.tenant;
    let coordinator = h.coordinator();
    assert_eq!(coordinator.restore_window(tenant).await.unwrap(), None);
    assert!(coordinator
        .set_restore_window(tenant, Some(3 * 24 * HOUR))
        .await
        .unwrap());
    assert_eq!(
        coordinator.restore_window(tenant).await.unwrap(),
        Some(3 * 24 * HOUR)
    );
    assert!(coordinator.set_restore_window(tenant, None).await.unwrap());
    assert_eq!(coordinator.restore_window(tenant).await.unwrap(), None);
    assert!(matches!(
        coordinator
            .set_restore_window(tenant, Some(MAX_RESTORE_WINDOW + HOUR))
            .await,
        Err(CoordinatorError::InvalidArgument(_))
    ));
    let absent = self::key().tenant;
    assert!(!coordinator
        .set_restore_window(&absent, Some(HOUR))
        .await
        .unwrap());
}

/// A new claim on backing `tenant` up, if it was taken.
async fn claimed(
    coordinator: &dyn Coordinator,
    tenant: &TenantId,
    due: Option<Duration>,
) -> Option<uuid::Uuid> {
    let claim = uuid::Uuid::new_v4();
    coordinator
        .claim_backup(tenant, claim, due, HOUR)
        .await
        .unwrap()
        .then_some(claim)
}

async fn one_node_at_a_time_backs_a_tenant_up_and_only_when_due(h: &dyn Harness) {
    let key = registered(h).await;
    let tenant = &key.tenant;
    let coordinator = h.coordinator();
    let day = Some(24 * HOUR);
    let first = claimed(coordinator, tenant, day).await;
    let first = first.expect("never backed up, so due");
    assert_eq!(claimed(coordinator, tenant, None).await, None, "held");
    coordinator
        .release_backup(tenant, first, true)
        .await
        .unwrap();
    assert_eq!(
        claimed(coordinator, tenant, day).await,
        None,
        "backed up just now"
    );
    let asked = claimed(coordinator, tenant, None).await;
    let asked = asked.expect("asked for, a backup need not be due");

    h.pass(&key, 2 * HOUR).await;
    let taken_over = claimed(coordinator, tenant, None).await;
    let taken_over = taken_over.expect("a claim that ran out is taken over");
    coordinator
        .release_backup(tenant, asked, false)
        .await
        .unwrap();
    assert_eq!(
        claimed(coordinator, tenant, None).await,
        None,
        "a stale release leaves the newer claim be"
    );
    coordinator
        .release_backup(tenant, taken_over, false)
        .await
        .unwrap();
    assert!(claimed(coordinator, tenant, None).await.is_some());
    assert_eq!(
        claimed(coordinator, &self::key().tenant, None).await,
        None,
        "no such tenant"
    );
}

async fn a_claim_asked_for_again_is_still_held_and_only_its_holder_renews_it(h: &dyn Harness) {
    let key = registered(h).await;
    let tenant = &key.tenant;
    let coordinator = h.coordinator();
    let day = Some(24 * HOUR);
    let (claim, other) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
    let take = |claim, due| coordinator.claim_backup(tenant, claim, due, HOUR);
    let renew = |claim| coordinator.renew_backup(tenant, claim, HOUR);
    assert!(take(claim, day).await.unwrap());
    assert!(
        take(claim, day).await.unwrap(),
        "asked for again, as when the answer was lost, it is still held"
    );
    assert!(!take(other, None).await.unwrap(), "held by the first");

    h.pass(&key, HOUR * 3 / 4).await;
    assert!(renew(claim).await.unwrap());
    assert!(!renew(other).await.unwrap(), "only the holder renews it");
    h.pass(&key, HOUR / 2).await;
    assert!(
        !take(other, None).await.unwrap(),
        "renewed, it has not run out"
    );
    h.pass(&key, HOUR).await;
    assert!(
        take(other, None).await.unwrap(),
        "run out, it is taken over"
    );
    assert!(
        !renew(claim).await.unwrap(),
        "one taken over is not renewed"
    );
    assert!(!take(claim, None).await.unwrap());
    coordinator
        .release_backup(tenant, other, true)
        .await
        .unwrap();
    assert!(
        !renew(other).await.unwrap(),
        "one let go is not taken again by renewing it"
    );
    assert!(!take(claim, day).await.unwrap(), "backed up just now");
    assert!(take(claim, None).await.unwrap());
    assert!(!coordinator
        .renew_backup(&self::key().tenant, claim, HOUR)
        .await
        .unwrap());
    assert!(matches!(
        coordinator
            .renew_backup(tenant, claim, MAX_BACKUP_HOLD + HOUR)
            .await,
        Err(CoordinatorError::InvalidArgument(_))
    ));
}

async fn a_removed_tenant_is_purged_before_it_is_made_again(h: &dyn Harness) {
    let key = registered(h).await;
    let tenant = &key.tenant;
    let coordinator = h.coordinator();
    let due = |purges: Purges| purges.due.contains(tenant);
    let lately = |purges: Purges| purges.lately.contains(tenant);
    assert!(!due(coordinator
        .purges(Duration::ZERO, HOUR)
        .await
        .unwrap()));
    assert_eq!(purge_claimed(coordinator, tenant).await, None);
    assert!(coordinator.remove(&key).await.unwrap());
    assert!(matches!(
        coordinator.register(&key).await,
        Err(CoordinatorError::Purging { .. })
    ));
    assert!(
        !due(coordinator.purges(HOUR, HOUR).await.unwrap()),
        "not yet"
    );
    assert!(due(coordinator.purges(Duration::ZERO, HOUR).await.unwrap()));

    let first = purge_claimed(coordinator, tenant).await.unwrap().id;
    assert_eq!(
        purge_claimed(coordinator, tenant).await,
        None,
        "one node purges at a time"
    );
    h.pass(&key, 2 * HOUR).await;
    let second = purge_claimed(coordinator, tenant).await.unwrap().id;
    coordinator.purged(tenant, first, 24 * HOUR).await.unwrap();
    assert!(
        due(coordinator.purges(Duration::ZERO, HOUR).await.unwrap()),
        "a claim that ran out marks nothing done"
    );
    coordinator.purged(tenant, second, 24 * HOUR).await.unwrap();
    let purges = coordinator.purges(Duration::ZERO, HOUR).await.unwrap();
    assert!(!due(purges.clone()) && lately(purges));
    assert_eq!(
        purge_claimed(coordinator, tenant).await,
        None,
        "a purge done is not claimed again"
    );
    coordinator.register(&key).await.unwrap();

    h.pass(&key, 2 * HOUR).await;
    let purges = coordinator.purges(Duration::ZERO, HOUR).await.unwrap();
    assert!(!lately(purges), "purged longer ago than lately");
    assert!(
        !coordinator.remove(&self::key()).await.unwrap(),
        "removing what was never there purges nothing"
    );
}

async fn a_purge_done_is_taken_up_again_only_while_its_tenant_is_not_made_again(h: &dyn Harness) {
    let key = registered(h).await;
    let tenant = &key.tenant;
    let coordinator = h.coordinator();
    let reopen = || coordinator.reopen_purge(tenant);
    let purge = || async {
        let claim = purge_claimed(coordinator, tenant).await;
        let claim = claim.expect("a purge to be done is claimed");
        coordinator
            .purged(tenant, claim.id, 24 * HOUR)
            .await
            .unwrap();
    };
    assert!(!reopen().await.unwrap(), "never removed");
    assert!(coordinator.remove(&key).await.unwrap());
    assert!(!reopen().await.unwrap(), "still to be done");
    h.pass(&key, 2 * HOUR).await;
    purge().await;

    assert!(
        reopen().await.unwrap(),
        "done, and the tenant not made again"
    );
    assert!(
        matches!(
            coordinator.register(&key).await,
            Err(CoordinatorError::Purging { .. })
        ),
        "not made again while it is purged again"
    );
    let purges = coordinator.purges(HOUR, HOUR).await.unwrap();
    assert!(
        purges.due.contains(tenant) && !purges.lately.contains(tenant),
        "to be purged at once: it has waited already"
    );
    purge().await;

    coordinator.register(&key).await.unwrap();
    assert!(!reopen().await.unwrap(), "made again, it is left be");
    let purges = coordinator.purges(Duration::ZERO, HOUR).await.unwrap();
    assert!(purges.lately.contains(tenant) && !purges.due.contains(tenant));
    assert!(!coordinator.reopen_purge(&self::key().tenant).await.unwrap());
}

async fn registering_and_removing_together_never_leave_a_tenant_to_be_purged(h: &dyn Harness) {
    let coordinator = h.coordinator();
    for _ in 0..20 {
        let key = registered(h).await;
        let (removed, registered) =
            tokio::join!(coordinator.remove(&key), coordinator.register(&key));
        assert!(removed.unwrap());
        let live = coordinator.committed(&key).await.unwrap().is_some();
        let purging = coordinator
            .purges(Duration::ZERO, HOUR)
            .await
            .unwrap()
            .due
            .contains(&key.tenant);
        assert!(!(live && purging), "{key} is live and to be purged");
        assert!(registered.is_ok() || purging);
    }
}

// DW-681: what a clean of a partition records it deletes below. A clean
// records its base before its first delete, and may take a checkpoint's
// missing predecessor for a clean's work only below the record; so the
// record must be written only by the lease that cleans, never go down, and
// never outlive the partition it was of.

fn base(position: u64) -> (ManifestRef, LogPosition) {
    (
        ManifestRef(format!("tenant/0/manifest/{position:020}.manifest")),
        LogPosition(position),
    )
}

fn cleaned_at(position: u64) -> Cleaned {
    let (base, position) = base(position);
    Cleaned { base, position }
}

async fn record(
    c: &dyn Coordinator,
    lease: &Lease,
    position: u64,
) -> Result<Cleaned, CoordinatorError> {
    let (base, position) = base(position);
    c.clean_below(lease, &base, position).await
}

async fn a_partition_never_cleaned_has_no_record_and_one_not_registered_is_unknown(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    assert_eq!(c.cleaned(&key).await.unwrap(), None);
    let (held, _) = lease(h, &key, &node("a")).await;
    c.commit(&held, commit(1, "m1")).await.unwrap();
    assert_eq!(
        c.cleaned(&key).await.unwrap(),
        None,
        "a commit records nothing"
    );
    let unknown = crate::key();
    assert!(matches!(
        c.cleaned(&unknown).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
}

async fn a_clean_is_recorded_only_by_the_lease_that_is_current(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let holder = node("a");
    let (first, _) = lease(h, &key, &holder).await;
    assert_eq!(record(c, &first, 5).await.unwrap(), cleaned_at(5));
    // Its own lease taken again, and then the lease taken over: either way
    // the earlier one records nothing, whether higher or not.
    let (again, _) = lease(h, &key, &holder).await;
    assert!(fenced(&record(c, &first, 9).await, again.generation));
    assert!(fenced(&record(c, &first, 3).await, again.generation));
    h.pass(&key, TTL + Duration::from_secs(1)).await;
    let (other, _) = lease(h, &key, &node("b")).await;
    assert!(fenced(&record(c, &again, 9).await, other.generation));
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(5)));
    // A lease released records nothing either.
    c.release(&other).await.unwrap();
    let now = generation(h, &key).await;
    assert!(fenced(&record(c, &other, 9).await, now));
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(5)));
}

async fn what_a_clean_recorded_is_only_ever_raised(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    assert_eq!(record(c, &held, 5).await.unwrap(), cleaned_at(5));
    // Lower: left as it is, and said, not refused.
    assert_eq!(record(c, &held, 3).await.unwrap(), cleaned_at(5));
    // As high, of another key: left as it is too.
    let other = ManifestRef("tenant/0/manifest/other.manifest".into());
    let kept = c.clean_below(&held, &other, LogPosition(5)).await.unwrap();
    assert_eq!(kept, cleaned_at(5));
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(5)));
    // Higher: raised.
    assert_eq!(record(c, &held, 8).await.unwrap(), cleaned_at(8));
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(8)));
}

async fn what_a_clean_recorded_stays_through_leases_and_commits(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;
    c.commit(&first, commit(6, "m6")).await.unwrap();
    record(c, &first, 4).await.unwrap();
    c.release(&first).await.unwrap();
    let (next, _) = lease(h, &key, &node("b")).await;
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(4)));
    c.commit(&next, commit(7, "m7")).await.unwrap();
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(4)));
    assert_eq!(record(c, &next, 2).await.unwrap(), cleaned_at(4));
}

async fn a_record_is_of_its_own_partition_alone(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (held, _) = lease(h, &key, &node("a")).await;
    record(c, &held, 6).await.unwrap();
    let sibling = beside(&key, 1);
    c.register(&sibling).await.unwrap();
    assert_eq!(c.cleaned(&sibling).await.unwrap(), None);
    let (other, _) = lease(h, &sibling, &node("b")).await;
    assert_eq!(record(c, &other, 2).await.unwrap(), cleaned_at(2));
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(6)));
}

async fn a_partition_removed_loses_its_record_and_one_made_again_starts_with_none(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let (zombie, _) = lease(h, &key, &node("a")).await;
    record(c, &zombie, 7).await.unwrap();
    assert!(c.remove(&key).await.unwrap());
    assert!(matches!(
        c.cleaned(&key).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    assert!(matches!(
        record(c, &zombie, 9).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    assert_eq!(
        c.cleaned(&key).await.unwrap(),
        None,
        "a new incarnation starts empty"
    );
    let (fresh, _) = lease(h, &key, &node("b")).await;
    assert!(fenced(&record(c, &zombie, 9).await, fresh.generation));
    assert_eq!(c.cleaned(&key).await.unwrap(), None);
    assert_eq!(record(c, &fresh, 2).await.unwrap(), cleaned_at(2));
}

// DW-668: how a partition's history is shipped. A hold keeps commit rows
// from the lowest still there and caps the cleaner's record; a base is
// recorded only where it stands, raised only; a run's own writes are
// fenced by its incarnation and base; and nothing outlives the partition.
// Nothing holds a lease to ship: the incarnation fences.

async fn incarnation_of(h: &dyn Harness, key: &PartitionKey) -> Incarnation {
    h.coordinator()
        .committed(key)
        .await
        .unwrap()
        .unwrap()
        .incarnation
}

fn not_shipped(incarnation: Incarnation) -> Shipping {
    Shipping {
        incarnation,
        hold_from: None,
        cap: None,
        base: None,
        shipped: None,
        stalled: None,
    }
}

fn held(incarnation: Incarnation, from: u64, cap: u64) -> Shipping {
    Shipping {
        hold_from: Some(LogPosition(from)),
        cap: Some(LogPosition(cap)),
        ..not_shipped(incarnation)
    }
}

fn ship_base(position: u64, checkpoint_at: u64, checkpoint: &str) -> ShipBase {
    ShipBase {
        position: LogPosition(position),
        checkpoint: ManifestRef(checkpoint.into()),
        checkpoint_at: LogPosition(checkpoint_at),
    }
}

/// Shipping held from `from`, with a base of `base`, its cap raised to its
/// checkpoint, and its run's own record.
fn based(incarnation: Incarnation, from: u64, base: &ShipBase) -> Shipping {
    Shipping {
        base: Some(base.clone()),
        ..held(incarnation, from, base.checkpoint_at.0)
    }
}

/// Commit, as the holder of a lease taken for it, each of `heads` at its
/// position, and let the lease go.
async fn committed_heads(h: &dyn Harness, key: &PartitionKey, heads: &[(u64, &str)]) {
    let c = h.coordinator();
    let (held, _) = lease(h, key, &node("committer")).await;
    for (position, head) in heads {
        c.commit(&held, commit(*position, head)).await.unwrap();
    }
    c.release(&held).await.unwrap();
}

/// Commit 1 to 3, an hour on, 4, and forget what the window of an hour
/// lets go: the rows at 3 and 4 are left. 3 is a checkpoint.
async fn forgotten_below_3(h: &dyn Harness) -> PartitionKey {
    let key = registered(h).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "l2"), (3, "m3")]).await;
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(4, "l4")]).await;
    assert_eq!(h.coordinator().forget_history(&key, HOUR).await.unwrap(), 2);
    key
}

fn stale(result: &Result<Shipping, CoordinatorError>, why: Stale) -> bool {
    matches!(result, Err(CoordinatorError::StaleShipping { why: w, .. }) if *w == why)
}

fn other_incarnation(result: &Result<Shipping, CoordinatorError>, current: Incarnation) -> bool {
    matches!(result, Err(CoordinatorError::OtherIncarnation { current: c, .. }) if *c == current)
}

fn other_base(result: &Result<Shipping, CoordinatorError>, recorded: Option<u64>) -> bool {
    matches!(
        result,
        Err(CoordinatorError::OtherBase { recorded: r, .. }) if *r == recorded.map(LogPosition)
    )
}

async fn a_partition_never_shipped_has_nothing_and_one_not_registered_is_unknown(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    assert_eq!(c.shipping(&key).await.unwrap(), not_shipped(now));
    committed_heads(h, &key, &[(1, "l1")]).await;
    assert_eq!(
        c.shipping(&key).await.unwrap(),
        not_shipped(now),
        "a commit records nothing"
    );
    let unknown = crate::key();
    let is_unknown = |result: Result<Shipping, CoordinatorError>| {
        matches!(result, Err(CoordinatorError::UnknownPartition(_)))
    };
    assert!(is_unknown(c.shipping(&unknown).await));
    assert!(is_unknown(
        c.hold_for_shipping(&unknown, now, LogPosition(0)).await
    ));
    assert!(is_unknown(
        c.record_base(&unknown, now, &ship_base(1, 1, "m1")).await
    ));
    assert!(is_unknown(
        c.note_shipped(&unknown, now, LogPosition(1), LogPosition(1))
            .await
    ));
    assert!(is_unknown(
        c.stall_shipping(&unknown, now, LogPosition(1), None, None, "why")
            .await
    ));
}

async fn a_hold_is_taken_from_the_lowest_row_still_there_once(h: &dyn Harness) {
    let c = h.coordinator();
    let key = forgotten_below_3(h).await;
    let now = incarnation_of(h, &key).await;
    let hold = c
        .hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    assert_eq!(hold, held(now, 3, 3));
    // Taken: left as it is, and said, not refused.
    assert_eq!(
        c.hold_for_shipping(&key, now, LogPosition(4))
            .await
            .unwrap(),
        held(now, 3, 3)
    );
    assert_eq!(c.shipping(&key).await.unwrap(), held(now, 3, 3));
    // With no row, from the position after the committed one, capped at
    // no checkpoint.
    let empty = registered(h).await;
    let theirs = incarnation_of(h, &empty).await;
    assert_eq!(
        c.hold_for_shipping(&empty, theirs, LogPosition(0))
            .await
            .unwrap(),
        held(theirs, 1, 0)
    );
}

async fn a_hold_capped_below_a_clean_or_above_the_commits_is_refused(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    let (lease_held, _) = lease(h, &key, &node("a")).await;
    for (position, head) in [(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4")] {
        c.commit(&lease_held, commit(position, head)).await.unwrap();
    }
    c.clean_below(&lease_held, &ManifestRef("m4".into()), LogPosition(4))
        .await
        .unwrap();
    assert!(stale(
        &c.hold_for_shipping(&key, now, LogPosition(2)).await,
        Stale::CleanedAbove
    ));
    assert!(stale(
        &c.hold_for_shipping(&key, now, LogPosition(5)).await,
        Stale::AboveCommitted
    ));
    let stale_one = Incarnation(now.0 + 1);
    assert!(other_incarnation(
        &c.hold_for_shipping(&key, stale_one, LogPosition(4)).await,
        now
    ));
    assert_eq!(c.shipping(&key).await.unwrap(), not_shipped(now));
    assert_eq!(
        c.hold_for_shipping(&key, now, LogPosition(4))
            .await
            .unwrap(),
        held(now, 1, 4)
    );
}

async fn a_base_is_recorded_only_where_it_stands(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "m5")],
    )
    .await;
    let base = ship_base(4, 5, "m5");
    assert!(stale(
        &c.record_base(&key, now, &base).await,
        Stale::NotHeld
    ));
    c.hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    // A checkpoint whose row names another head, or no row at all.
    assert!(stale(
        &c.record_base(&key, now, &ship_base(4, 5, "m-other")).await,
        Stale::CheckpointNotThere
    ));
    assert!(stale(
        &c.record_base(&key, now, &ship_base(4, 7, "m7")).await,
        Stale::CheckpointNotThere
    ));
    // Below the cap, it would lower it.
    assert!(stale(
        &c.record_base(&key, now, &ship_base(1, 2, "l2")).await,
        Stale::CapAbove
    ));
    let stale_one = Incarnation(now.0 + 1);
    assert!(other_incarnation(
        &c.record_base(&key, stale_one, &base).await,
        now
    ));
    for wrong in [ship_base(0, 5, "m5"), ship_base(6, 5, "m5")] {
        let refused = c.record_base(&key, now, &wrong).await;
        assert!(
            matches!(refused, Err(CoordinatorError::InvalidArgument(_))),
            "{refused:?}"
        );
    }
    assert_eq!(c.shipping(&key).await.unwrap(), held(now, 1, 3));
    assert_eq!(
        c.record_base(&key, now, &base).await.unwrap(),
        based(now, 1, &base)
    );
    assert_eq!(c.shipping(&key).await.unwrap(), based(now, 1, &base));
}

async fn a_base_below_the_hold_or_its_row_or_a_clean_is_refused(h: &dyn Harness) {
    let c = h.coordinator();
    // The hold is above the base asked for.
    let key = forgotten_below_3(h).await;
    let now = incarnation_of(h, &key).await;
    c.hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    assert!(stale(
        &c.record_base(&key, now, &ship_base(2, 3, "m3")).await,
        Stale::HoldAbove
    ));
    // The row at the base kept by forget_history since the hold, which it
    // honours: the base is recorded.
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2"), (3, "l3")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(4, "l4")]).await;
    c.forget_history(&key, HOUR).await.unwrap();
    let kept = ship_base(2, 2, "m2");
    assert_eq!(
        c.record_base(&key, now, &kept).await.unwrap(),
        based(now, 1, &kept)
    );
    // No row at the base: none was ever committed there, positions
    // skipping as a restore's do, the checkpoint's own row there.
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "l2"), (5, "m5")]).await;
    c.hold_for_shipping(&key, now, LogPosition(5))
        .await
        .unwrap();
    assert!(stale(
        &c.record_base(&key, now, &ship_base(3, 5, "m5")).await,
        Stale::NoRowAtBase
    ));
    // No row at the base: forgotten as the floor of a run that had shipped
    // past it allowed, before a base was raised to it.
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(1, 2, "m2"))
        .await
        .unwrap();
    c.note_shipped(&key, now, LogPosition(1), LogPosition(3))
        .await
        .unwrap();
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(5, "l5")]).await;
    c.forget_history(&key, HOUR).await.unwrap();
    assert!(stale(
        &c.record_base(&key, now, &ship_base(3, 4, "m4")).await,
        Stale::NoRowAtBase
    ));
    // A clean recorded above a checkpoint: held to the cap now (PR 6, H),
    // so only under a cap at or above it; a base at the checkpoint below
    // is then refused, for the cap above it first.
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    let (lease_held, _) = lease(h, &key, &node("a")).await;
    for (position, head) in [(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4")] {
        c.commit(&lease_held, commit(position, head)).await.unwrap();
    }
    c.hold_for_shipping(&key, now, LogPosition(4))
        .await
        .unwrap();
    c.clean_below(&lease_held, &ManifestRef("m4".into()), LogPosition(4))
        .await
        .unwrap();
    let refused = c.record_base(&key, now, &ship_base(1, 2, "m2")).await;
    assert!(
        stale(&refused, Stale::CapAbove) || stale(&refused, Stale::CleanedAbove),
        "{refused:?}"
    );
    let base = ship_base(3, 4, "m4");
    assert_eq!(
        c.record_base(&key, now, &base).await.unwrap(),
        based(now, 1, &base)
    );
}

async fn a_base_is_raised_only_and_a_new_run_starts_afresh(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "m5")],
    )
    .await;
    c.hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    let first = ship_base(4, 5, "m5");
    c.record_base(&key, now, &first).await.unwrap();
    c.note_shipped(&key, now, LogPosition(4), LogPosition(5))
        .await
        .unwrap();
    // Lower, or as high: left as it is, and given back.
    let standing = Shipping {
        shipped: Some(LogPosition(5)),
        ..based(now, 1, &first)
    };
    assert_eq!(
        c.record_base(&key, now, &ship_base(1, 3, "m3"))
            .await
            .unwrap(),
        standing
    );
    assert_eq!(
        c.record_base(&key, now, &ship_base(4, 5, "m5"))
            .await
            .unwrap(),
        standing
    );
    // Higher: raised, the cap with it, and its run starts with nothing
    // shipped or stalled.
    committed_heads(h, &key, &[(6, "l6"), (7, "m7")]).await;
    c.stall_shipping(
        &key,
        now,
        LogPosition(4),
        Some(LogPosition(5)),
        None,
        "the first run stopped",
    )
    .await
    .unwrap();
    let second = ship_base(6, 7, "m7");
    assert_eq!(
        c.record_base(&key, now, &second).await.unwrap(),
        based(now, 1, &second)
    );
}

async fn a_runs_writes_are_fenced_by_its_base_and_only_raise(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    let note = |base: u64, position: u64| {
        c.note_shipped(&key, now, LogPosition(base), LogPosition(position))
    };
    let stall = |base: u64, seen: Option<u64>, reason: &'static str| {
        let seen = seen.map(LogPosition);
        c.stall_shipping(&key, now, LogPosition(base), seen, None, reason)
    };
    assert!(other_base(&note(4, 5).await, None));
    assert!(other_base(&stall(4, None, "no run").await, None));
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "m5")],
    )
    .await;
    c.hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    let base = ship_base(4, 5, "m5");
    c.record_base(&key, now, &base).await.unwrap();
    let run = |shipped: Option<u64>, stalled: Option<&str>| Shipping {
        shipped: shipped.map(LogPosition),
        stalled: stalled.map(String::from),
        ..based(now, 1, &base)
    };
    assert_eq!(note(4, 5).await.unwrap(), run(Some(5), None));
    assert_eq!(
        note(4, 4).await.unwrap(),
        run(Some(5), None),
        "lower: left, said"
    );
    assert_eq!(
        note(4, 5).await.unwrap(),
        run(Some(5), None),
        "as high: left, said"
    );
    assert_eq!(note(4, 6).await.unwrap(), run(Some(6), None));
    let below = note(4, 3).await;
    assert!(
        matches!(below, Err(CoordinatorError::InvalidArgument(_))),
        "{below:?}"
    );
    assert!(other_base(&note(3, 7).await, Some(4)));
    assert!(other_base(
        &stall(5, Some(6), "another run's").await,
        Some(4)
    ));
    assert_eq!(
        stall(4, Some(6), "the first").await.unwrap(),
        run(Some(6), Some("the first"))
    );
    assert_eq!(
        stall(4, Some(6), "a later").await.unwrap(),
        run(Some(6), Some("the first")),
        "the first stands"
    );
    let stale_one = Incarnation(now.0 + 1);
    assert!(other_incarnation(
        &c.note_shipped(&key, stale_one, LogPosition(4), LogPosition(9))
            .await,
        now
    ));
    assert!(other_incarnation(
        &c.stall_shipping(
            &key,
            stale_one,
            LogPosition(4),
            Some(LogPosition(6)),
            None,
            "x"
        )
        .await,
        now
    ));
    // After a re-base, a stale run neither raises nor stops the new one.
    committed_heads(h, &key, &[(6, "l6"), (7, "m7")]).await;
    let second = ship_base(6, 7, "m7");
    c.record_base(&key, now, &second).await.unwrap();
    assert!(other_base(&note(4, 9).await, Some(6)));
    assert!(other_base(
        &stall(4, Some(6), "a stale run's").await,
        Some(6)
    ));
    assert_eq!(c.shipping(&key).await.unwrap(), based(now, 1, &second));
}

/// DW-668: the commits a run ships are read in one step with the run, by a
/// shipper that read it as it stands: what it shipped is `seen`, and its
/// base is the run's. A stale one, whose rows `forget_history` may have
/// taken since as the run shipped past them, is refused, as its stall is:
/// given what is left, it would link a checkpoint over the gap, on a
/// commit that is not the one before it.
async fn the_commits_a_run_ships_are_read_only_by_a_shipper_that_saw_the_run_as_it_stands(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "l4"), (5, "l5")],
    )
    .await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(1, 2, "m2"))
        .await
        .unwrap();
    let to_ship = |incarnation: Incarnation, base: u64, seen: Option<u64>, limit: usize| {
        c.commits_to_ship(
            &key,
            incarnation,
            LogPosition(base),
            seen.map(LogPosition),
            limit,
        )
    };
    let positions =
        |rows: Vec<RestorePoint>| -> Vec<u64> { rows.iter().map(|row| row.position.0).collect() };
    let shipped_since =
        |result: &Result<Vec<RestorePoint>, CoordinatorError>, was: Option<u64>, is: u64| {
            matches!(
                result,
                Err(CoordinatorError::ShippedSince { seen, shipped, .. })
                    if *seen == was.map(LogPosition) && *shipped == Some(LogPosition(is))
            )
        };
    // Nothing shipped: from the base.
    let rows = to_ship(now, 1, None, 10).await.unwrap();
    assert_eq!(positions(rows), vec![1, 2, 3, 4, 5]);
    c.note_shipped(&key, now, LogPosition(1), LogPosition(3))
        .await
        .unwrap();
    // After what the run has shipped, up to the limit.
    let rows = to_ship(now, 1, Some(3), 1).await.unwrap();
    assert_eq!(positions(rows), vec![4]);
    // A shipper that read the run before it shipped to 3: refused.
    let stale = to_ship(now, 1, None, 10).await;
    assert!(shipped_since(&stale, None, 3), "{stale:?}");
    let stale = to_ship(now, 1, Some(2), 10).await;
    assert!(shipped_since(&stale, Some(2), 3), "{stale:?}");
    // Rows forgotten below the run's floor, after 3: still refused, and
    // the run's own reads on from 3.
    h.pass(&key, 2 * HOUR).await;
    assert_eq!(c.forget_history(&key, HOUR).await.unwrap(), 3);
    let stale = to_ship(now, 1, Some(2), 10).await;
    assert!(shipped_since(&stale, Some(2), 3), "{stale:?}");
    let rows = to_ship(now, 1, Some(3), 10).await.unwrap();
    assert_eq!(positions(rows), vec![4, 5]);
    // Of another base, or another incarnation: refused.
    let other = to_ship(now, 2, Some(3), 10).await;
    assert!(
        matches!(
            other,
            Err(CoordinatorError::OtherBase {
                asked: LogPosition(2),
                recorded: Some(LogPosition(1)),
                ..
            })
        ),
        "{other:?}"
    );
    let other = to_ship(Incarnation(now.0 + 1), 1, Some(3), 10).await;
    assert!(
        matches!(other, Err(CoordinatorError::OtherIncarnation { .. })),
        "{other:?}"
    );
    // Shipped to the last: none, and no refusal.
    c.note_shipped(&key, now, LogPosition(1), LogPosition(5))
        .await
        .unwrap();
    assert_eq!(to_ship(now, 1, Some(5), 10).await.unwrap(), Vec::new());
}

/// DW-668, S5: a stall is decided on what a batch read of the run, and a
/// shipper of the same run that read it before the run moved on, a stale
/// one, is refused: the run under way is never stopped by it. The stall
/// notes what its own batch shipped in the same step, as `note_shipped`
/// would, raising only.
async fn a_stall_is_taken_only_from_a_run_that_saw_what_it_has_shipped_and_notes_it_too(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "l4"), (5, "l5")],
    )
    .await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(1, 2, "m2");
    c.record_base(&key, now, &base).await.unwrap();
    let run = |shipped: Option<u64>, stalled: Option<&str>| Shipping {
        shipped: shipped.map(LogPosition),
        stalled: stalled.map(String::from),
        ..based(now, 1, &base)
    };
    let stall = |seen: Option<u64>, through: Option<u64>, reason: &'static str| {
        let (seen, through) = (seen.map(LogPosition), through.map(LogPosition));
        c.stall_shipping(&key, now, LogPosition(1), seen, through, reason)
    };
    let shipped_since = |result: &Result<Shipping, CoordinatorError>, was: Option<u64>, is: u64| {
        matches!(
            result,
            Err(CoordinatorError::ShippedSince { seen, shipped, .. })
                if *seen == was.map(LogPosition) && *shipped == Some(LogPosition(is))
        )
    };
    c.note_shipped(&key, now, LogPosition(1), LogPosition(3))
        .await
        .unwrap();
    // A shipper that read the run before it shipped to 3: refused, and
    // nothing written, neither its stall nor what it shipped.
    let stale = stall(None, Some(2), "a stale shipper's").await;
    assert!(shipped_since(&stale, None, 3), "{stale:?}");
    let stale = stall(Some(2), Some(4), "a stale shipper's").await;
    assert!(shipped_since(&stale, Some(2), 3), "{stale:?}");
    assert_eq!(c.shipping(&key).await.unwrap(), run(Some(3), None));
    // One that read it as it stands: what it shipped noted, and stalled,
    // in one step.
    assert_eq!(
        stall(Some(3), Some(5), "the run's own").await.unwrap(),
        run(Some(5), Some("the run's own"))
    );
    // Stalled already: the first reason stands, whatever was seen.
    assert_eq!(
        stall(Some(5), None, "a later").await.unwrap(),
        run(Some(5), Some("the run's own"))
    );
    assert_eq!(
        stall(None, None, "a stale later").await.unwrap(),
        run(Some(5), Some("the run's own"))
    );
}

/// DW-668, S5: noting what a stall's batch shipped only raises, and one
/// below the base is refused as a note is.
async fn a_stall_raises_what_was_shipped_only(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "l4")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(2, 2, "m2");
    c.record_base(&key, now, &base).await.unwrap();
    let below = c
        .stall_shipping(&key, now, LogPosition(2), None, Some(LogPosition(1)), "x")
        .await;
    assert!(
        matches!(below, Err(CoordinatorError::InvalidArgument(_))),
        "{below:?}"
    );
    c.note_shipped(&key, now, LogPosition(2), LogPosition(4))
        .await
        .unwrap();
    let lower = c
        .stall_shipping(
            &key,
            now,
            LogPosition(2),
            Some(LogPosition(4)),
            Some(LogPosition(3)),
            "y",
        )
        .await
        .unwrap();
    assert_eq!(
        (lower.shipped, lower.stalled.as_deref()),
        (Some(LogPosition(4)), Some("y"))
    );
}

fn above_cap(result: &Result<Cleaned, CoordinatorError>, asked: u64, cap: u64) -> bool {
    matches!(
        result,
        Err(CoordinatorError::CleanAboveCap { asked: a, cap: c, .. })
            if *a == LogPosition(asked) && *c == LogPosition(cap)
    )
}

/// DW-668 PR 6 (bootstrap note H): while a partition is held for
/// shipping, a clean records nothing above the cap, checked on the record
/// it would write: one below the cap stands, and a record left as it is
/// is no record written.
async fn a_clean_is_held_to_the_shipping_cap_on_the_record_it_writes(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[
            (1, "l1"),
            (2, "m2"),
            (3, "l3"),
            (4, "m4"),
            (5, "l5"),
            (6, "m6"),
        ],
    )
    .await;
    let (held, _) = lease(h, &key, &node("a")).await;
    // No hold: no cap.
    assert_eq!(record(c, &held, 2).await.unwrap(), cleaned_at(2));
    c.hold_for_shipping(&key, now, LogPosition(4))
        .await
        .unwrap();
    // An old record below the cap, a new one above it: refused, the old
    // one standing.
    assert!(above_cap(&record(c, &held, 6).await, 6, 4), "above");
    assert_eq!(c.cleaned(&key).await.unwrap(), Some(cleaned_at(2)));
    // At the cap: recorded.
    assert_eq!(record(c, &held, 4).await.unwrap(), cleaned_at(4));
    // Lower than the record: left as it is, and said, under the cap or not.
    assert_eq!(record(c, &held, 3).await.unwrap(), cleaned_at(4));
    // The cap raised by a base: up to it, recorded.
    c.record_base(&key, now, &ship_base(5, 6, "m6"))
        .await
        .unwrap();
    assert_eq!(record(c, &held, 6).await.unwrap(), cleaned_at(6));
}

/// DW-668 PR 6 (the hold's lifecycle, 3): the cap follows what the run
/// ships, raised only, never past its watermark, and only by its own run.
async fn a_cap_is_raised_only_and_never_past_what_its_run_shipped(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    let raise =
        |base: u64, cap: u64| c.raise_ship_cap(&key, now, LogPosition(base), LogPosition(cap));
    committed_heads(
        h,
        &key,
        &[
            (1, "l1"),
            (2, "m2"),
            (3, "l3"),
            (4, "m4"),
            (5, "l5"),
            (6, "m6"),
            (7, "l7"),
        ],
    )
    .await;
    // No run: another base, none.
    assert!(other_base(&raise(2, 2).await, None));
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(2, 2, "m2");
    c.record_base(&key, now, &base).await.unwrap();
    // Past what was shipped, nothing yet: refused.
    let past = raise(2, 3).await;
    assert!(
        matches!(past, Err(CoordinatorError::InvalidArgument(_))),
        "{past:?}"
    );
    c.note_shipped(&key, now, LogPosition(2), LogPosition(5))
        .await
        .unwrap();
    let past = raise(2, 6).await;
    assert!(
        matches!(past, Err(CoordinatorError::InvalidArgument(_))),
        "{past:?}"
    );
    // Up to it: raised; lower: left, and said.
    assert_eq!(raise(2, 5).await.unwrap().cap, Some(LogPosition(5)));
    assert_eq!(raise(2, 4).await.unwrap().cap, Some(LogPosition(5)));
    assert_eq!(c.shipping(&key).await.unwrap().cap, Some(LogPosition(5)));
    // Another run's, or another incarnation's: refused.
    assert!(other_base(&raise(3, 5).await, Some(2)));
    let stale = c
        .raise_ship_cap(&key, Incarnation(now.0 + 1), LogPosition(2), LogPosition(5))
        .await;
    assert!(other_incarnation(&stale, now), "{stale:?}");
    // A clean is held to it.
    let (held, _) = lease(h, &key, &node("a")).await;
    assert!(above_cap(&record(c, &held, 6).await, 6, 5));
    assert_eq!(record(c, &held, 4).await.unwrap(), cleaned_at(4));
}

async fn shipping_stays_through_leases_and_is_of_its_own_partition_alone(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(1, 2, "m2");
    c.record_base(&key, now, &base).await.unwrap();
    c.note_shipped(&key, now, LogPosition(1), LogPosition(2))
        .await
        .unwrap();
    let standing = Shipping {
        shipped: Some(LogPosition(2)),
        ..based(now, 1, &base)
    };
    committed_heads(h, &key, &[(3, "l3")]).await;
    assert_eq!(c.shipping(&key).await.unwrap(), standing);
    let sibling = beside(&key, 1);
    c.register(&sibling).await.unwrap();
    let theirs = incarnation_of(h, &sibling).await;
    assert_eq!(c.shipping(&sibling).await.unwrap(), not_shipped(theirs));
    c.hold_for_shipping(&sibling, theirs, LogPosition(0))
        .await
        .unwrap();
    assert_eq!(c.shipping(&key).await.unwrap(), standing);
}

async fn a_partition_removed_loses_its_shipping_and_one_made_again_starts_with_none(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let old = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2")]).await;
    c.hold_for_shipping(&key, old, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, old, &ship_base(1, 2, "m2"))
        .await
        .unwrap();
    c.note_shipped(&key, old, LogPosition(1), LogPosition(2))
        .await
        .unwrap();
    assert!(c.remove(&key).await.unwrap());
    assert!(matches!(
        c.shipping(&key).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    let now = incarnation_of(h, &key).await;
    assert_ne!(now, old);
    assert_eq!(
        c.shipping(&key).await.unwrap(),
        not_shipped(now),
        "a new incarnation starts empty"
    );
    // A shipper of the one removed records nothing in the new one.
    assert!(other_incarnation(
        &c.hold_for_shipping(&key, old, LogPosition(0)).await,
        now
    ));
    assert!(other_incarnation(
        &c.record_base(&key, old, &ship_base(1, 2, "m2")).await,
        now
    ));
    assert!(other_incarnation(
        &c.note_shipped(&key, old, LogPosition(1), LogPosition(3))
            .await,
        now
    ));
    assert!(other_incarnation(
        &c.stall_shipping(&key, old, LogPosition(1), None, None, "x")
            .await,
        now
    ));
    assert_eq!(c.shipping(&key).await.unwrap(), not_shipped(now));
}

// DW-668, 3b: what a shipper reads. The commits of an incarnation above a
// position, from the rows the commits made; and the partitions whose run
// of shipping has commits it has yet to ship, in turn by key.

async fn commits_of(
    h: &dyn Harness,
    key: &PartitionKey,
    incarnation: Incarnation,
    after: u64,
    limit: usize,
) -> Result<Vec<u64>, CoordinatorError> {
    h.coordinator()
        .commits_after(key, incarnation, LogPosition(after), limit)
        .await
        .map(|points| points.into_iter().map(|point| point.position.0).collect())
}

async fn the_commits_after_a_position_are_read_from_their_rows_in_order(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    assert_eq!(
        commits_of(h, &key, now, 0, 10).await.unwrap(),
        Vec::<u64>::new()
    );
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "l5")],
    )
    .await;
    let all = c
        .commits_after(&key, now, LogPosition(0), 10)
        .await
        .unwrap();
    let heads: Vec<(u64, &str)> = all
        .iter()
        .map(|point| (point.position.0, point.head.0.as_str()))
        .collect();
    assert_eq!(
        heads,
        [(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "l5")]
    );
    // Each at the time its commit was made, as the history tells it.
    let (earliest, latest) = c.restore_points(&key, HOUR).await.unwrap().unwrap();
    assert_eq!(all.first(), Some(&earliest));
    assert_eq!(all.last(), Some(&latest));
    assert_eq!(commits_of(h, &key, now, 2, 2).await.unwrap(), [3, 4]);
    assert_eq!(commits_of(h, &key, now, 4, 10).await.unwrap(), [5]);
    assert_eq!(
        commits_of(h, &key, now, 5, 10).await.unwrap(),
        Vec::<u64>::new()
    );
    assert_eq!(
        commits_of(h, &key, now, 0, 0).await.unwrap(),
        Vec::<u64>::new()
    );
    let unknown = crate::key();
    assert!(matches!(
        commits_of(h, &unknown, now, 0, 10).await,
        Err(CoordinatorError::UnknownPartition(_))
    ));
}

async fn the_commits_read_skip_where_the_positions_do_and_lose_what_is_forgotten(h: &dyn Harness) {
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    // A restore commits after the greater of the current position and the
    // one it restores.
    committed_heads(h, &key, &[(1, "l1"), (2, "l2"), (3, "m3")]).await;
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(7, "m7"), (8, "l8")]).await;
    assert_eq!(
        commits_of(h, &key, now, 0, 10).await.unwrap(),
        [1, 2, 3, 7, 8]
    );
    assert_eq!(commits_of(h, &key, now, 3, 10).await.unwrap(), [7, 8]);
    h.coordinator().forget_history(&key, HOUR).await.unwrap();
    assert_eq!(commits_of(h, &key, now, 0, 10).await.unwrap(), [3, 7, 8]);
}

async fn the_commits_of_another_incarnation_are_refused(h: &dyn Harness) {
    let c = h.coordinator();
    let key = registered(h).await;
    let old = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "l2")]).await;
    let ahead = Incarnation(old.0 + 1);
    assert!(matches!(
        commits_of(h, &key, ahead, 0, 10).await,
        Err(CoordinatorError::OtherIncarnation { .. })
    ));
    assert!(c.remove(&key).await.unwrap());
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "n1")]).await;
    assert!(matches!(
        commits_of(h, &key, old, 0, 10).await,
        Err(CoordinatorError::OtherIncarnation { .. })
    ));
    let fresh = c
        .commits_after(&key, now, LogPosition(0), 10)
        .await
        .unwrap();
    assert_eq!(fresh.len(), 1);
    assert_eq!(fresh[0].head, ManifestRef("n1".into()));
}

/// A partition, held and based at 1 on its checkpoint at 2, with `more`
/// committed after.
async fn based_at_1(h: &dyn Harness, more: &[(u64, &str)]) -> (PartitionKey, Incarnation) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(1, 2, "m2"))
        .await
        .unwrap();
    if !more.is_empty() {
        committed_heads(h, &key, more).await;
    }
    (key, now)
}

fn behind(
    key: &PartitionKey,
    incarnation: Incarnation,
    shipped: Option<u64>,
    committed: u64,
) -> Unshipped {
    Unshipped {
        key: key.clone(),
        incarnation,
        base: LogPosition(1),
        shipped: shipped.map(LogPosition),
        committed: LogPosition(committed),
    }
}

async fn a_run_is_unshipped_while_commits_lie_above_what_it_shipped(h: &dyn Harness) {
    let c = h.coordinator();
    assert_eq!(c.unshipped(None, 10).await.unwrap(), Vec::new());
    // Held, with no base: nothing of it is a run's to ship yet.
    let held_only = registered(h).await;
    let theirs = incarnation_of(h, &held_only).await;
    committed_heads(h, &held_only, &[(1, "l1")]).await;
    c.hold_for_shipping(&held_only, theirs, LogPosition(0))
        .await
        .unwrap();
    assert_eq!(c.unshipped(None, 10).await.unwrap(), Vec::new());
    // Based, nothing shipped: unshipped from its base.
    let (key, now) = based_at_1(h, &[]).await;
    assert_eq!(
        c.unshipped(None, 10).await.unwrap(),
        [behind(&key, now, None, 2)]
    );
    // Shipped up to what is committed: not.
    c.note_shipped(&key, now, LogPosition(1), LogPosition(2))
        .await
        .unwrap();
    assert_eq!(c.unshipped(None, 10).await.unwrap(), Vec::new());
    // Committed past it: again.
    committed_heads(h, &key, &[(3, "l3")]).await;
    assert_eq!(
        c.unshipped(None, 10).await.unwrap(),
        [behind(&key, now, Some(2), 3)]
    );
    // Stalled: an operator's, not a shipper's.
    c.stall_shipping(
        &key,
        now,
        LogPosition(1),
        Some(LogPosition(2)),
        None,
        "a fork",
    )
    .await
    .unwrap();
    assert_eq!(c.unshipped(None, 10).await.unwrap(), Vec::new());
    // A base that is its own checkpoint and the head, nothing committed
    // past it nor shipped: the base itself is still to ship.
    let own = registered(h).await;
    let theirs = incarnation_of(h, &own).await;
    committed_heads(h, &own, &[(1, "l1"), (2, "m2")]).await;
    c.hold_for_shipping(&own, theirs, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&own, theirs, &ship_base(2, 2, "m2"))
        .await
        .unwrap();
    let base_alone = Unshipped {
        base: LogPosition(2),
        ..behind(&own, theirs, None, 2)
    };
    assert_eq!(c.unshipped(None, 10).await.unwrap(), [base_alone]);
}

async fn unshipped_partitions_are_given_in_turn_by_key(h: &dyn Harness) {
    let c = h.coordinator();
    let mut runs = Vec::new();
    for _ in 0..3 {
        let (key, now) = based_at_1(h, &[(3, "l3")]).await;
        runs.push(behind(&key, now, None, 3));
    }
    runs.sort_by(|one, other| one.key.cmp(&other.key));
    assert_eq!(c.unshipped(None, 10).await.unwrap(), runs);
    assert_eq!(c.unshipped(None, 2).await.unwrap(), runs[..2]);
    assert_eq!(c.unshipped(Some(&runs[1].key), 2).await.unwrap(), runs[2..]);
    assert_eq!(
        c.unshipped(Some(&runs[2].key), 2).await.unwrap(),
        Vec::new()
    );
    assert_eq!(c.unshipped(None, 0).await.unwrap(), Vec::new());
    // A partition removed is no one's to ship.
    assert!(c.remove(&runs[0].key).await.unwrap());
    assert_eq!(c.unshipped(None, 10).await.unwrap(), runs[1..]);
}

// DW-668, PR 5: forget_history keeps what shipping needs. From the floor
// of the current incarnation's run, the position after what it shipped,
// else its base, else its hold, no commit row is forgotten, whatever the
// restore window lets go; with no hold, it forgets as it always has.

/// What `forget_history` with a window of an hour leaves of the commits 1
/// to 5 (3 and 5 checkpoints) made two hours ago and 6 made since, once
/// `ship` has done what it does to shipping before it runs; and how many
/// it said it forgot.
async fn left_after_forgetting(
    h: &dyn Harness,
    ship: impl for<'a> FnOnce(
        &'a dyn Coordinator,
        &'a PartitionKey,
        Incarnation,
    ) -> futures::future::BoxFuture<'a, ()>,
) -> (Vec<u64>, u64) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(
        h,
        &key,
        &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4"), (5, "m5")],
    )
    .await;
    ship(c, &key, now).await;
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(6, "l6")]).await;
    let forgot = c.forget_history(&key, HOUR).await.unwrap();
    (commits_of(h, &key, now, 0, 100).await.unwrap(), forgot)
}

async fn forget_history_keeps_every_row_from_the_floor_shipping_needs(h: &dyn Harness) {
    use futures::FutureExt;
    // No hold: as ever, the newest commit before the window, and those in it.
    let none = left_after_forgetting(h, |_, _, _| async {}.boxed()).await;
    assert_eq!(none, (vec![5, 6], 4), "no hold");
    // Held, no base: from the hold, the lowest row there when it was taken.
    let held = left_after_forgetting(h, |c, key, now| {
        async move {
            c.hold_for_shipping(key, now, LogPosition(3)).await.unwrap();
        }
        .boxed()
    })
    .await;
    assert_eq!(held, (vec![1, 2, 3, 4, 5, 6], 0), "held, no base");
    // Based at 3, nothing shipped: from the base.
    let based = left_after_forgetting(h, |c, key, now| {
        async move {
            c.hold_for_shipping(key, now, LogPosition(3)).await.unwrap();
            c.record_base(key, now, &ship_base(3, 3, "m3"))
                .await
                .unwrap();
        }
        .boxed()
    })
    .await;
    assert_eq!(based, (vec![3, 4, 5, 6], 2), "based, nothing shipped");
    // Based at 3, shipped to 3: from the commit after it.
    let shipped = left_after_forgetting(h, |c, key, now| {
        async move {
            c.hold_for_shipping(key, now, LogPosition(3)).await.unwrap();
            c.record_base(key, now, &ship_base(3, 3, "m3"))
                .await
                .unwrap();
            c.note_shipped(key, now, LogPosition(3), LogPosition(3))
                .await
                .unwrap();
        }
        .boxed()
    })
    .await;
    assert_eq!(shipped, (vec![4, 5, 6], 3), "based and shipped");
    // Shipped past the newest commit before the window: as with no hold,
    // the window keeps what shipping no longer needs.
    let caught_up = left_after_forgetting(h, |c, key, now| {
        async move {
            c.hold_for_shipping(key, now, LogPosition(3)).await.unwrap();
            c.record_base(key, now, &ship_base(3, 3, "m3"))
                .await
                .unwrap();
            c.note_shipped(key, now, LogPosition(3), LogPosition(5))
                .await
                .unwrap();
        }
        .boxed()
    })
    .await;
    assert_eq!(caught_up, (vec![5, 6], 4), "shipped to the window");
    // A run reset by a raised base, nothing of it shipped: from its base,
    // not from what the run before shipped.
    let reset = left_after_forgetting(h, |c, key, now| {
        async move {
            c.hold_for_shipping(key, now, LogPosition(3)).await.unwrap();
            c.record_base(key, now, &ship_base(3, 3, "m3"))
                .await
                .unwrap();
            c.note_shipped(key, now, LogPosition(3), LogPosition(3))
                .await
                .unwrap();
            c.record_base(key, now, &ship_base(5, 5, "m5"))
                .await
                .unwrap();
        }
        .boxed()
    })
    .await;
    assert_eq!(reset, (vec![5, 6], 4), "a run reset by a raised base");
}

async fn what_forget_history_keeps_for_shipping_lets_the_floor_rise_with_what_is_shipped(
    h: &dyn Harness,
) {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    committed_heads(h, &key, &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "l4")]).await;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(1, 2, "m2"))
        .await
        .unwrap();
    h.pass(&key, 2 * HOUR).await;
    committed_heads(h, &key, &[(5, "l5")]).await;
    assert_eq!(c.forget_history(&key, HOUR).await.unwrap(), 0);
    assert_eq!(
        commits_of(h, &key, now, 0, 10).await.unwrap(),
        [1, 2, 3, 4, 5]
    );
    c.note_shipped(&key, now, LogPosition(1), LogPosition(2))
        .await
        .unwrap();
    assert_eq!(c.forget_history(&key, HOUR).await.unwrap(), 2);
    assert_eq!(commits_of(h, &key, now, 0, 10).await.unwrap(), [3, 4, 5]);
    c.note_shipped(&key, now, LogPosition(1), LogPosition(5))
        .await
        .unwrap();
    assert_eq!(c.forget_history(&key, HOUR).await.unwrap(), 1);
    assert_eq!(commits_of(h, &key, now, 0, 10).await.unwrap(), [4, 5]);
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
    every_commit_is_a_restore_point_once,
    a_commit_is_made_only_on_what_it_says_it_follows,
    a_restore_reaches_back_through_its_window_and_to_how_it_then_stood,
    removing_a_partition_forgets_its_history,
    a_partition_made_again_keeps_its_history_while_the_old_is_forgotten,
    a_restore_asked_for_again_after_its_record_lapsed_is_recorded_again,
    a_restore_is_recorded_with_its_commit_for_a_day_and_for_its_incarnation,
    restores_a_day_old_are_forgotten_whether_or_not_their_partition_is_kept_up,
    a_tenant_keeps_its_own_restore_window_until_told_otherwise,
    one_node_at_a_time_backs_a_tenant_up_and_only_when_due,
    a_claim_asked_for_again_is_still_held_and_only_its_holder_renews_it,
    a_removed_tenant_is_purged_before_it_is_made_again,
    a_purge_done_is_taken_up_again_only_while_its_tenant_is_not_made_again,
    registering_and_removing_together_never_leave_a_tenant_to_be_purged,
    a_tenant_keeps_its_lineage_until_it_is_removed_and_one_made_again_has_another,
    partitions_first_registered_together_are_of_one_lineage,
    a_tenant_takes_a_backup_up_with_the_commit_that_restores_it_and_nothing_before,
    an_adoption_is_noted_only_for_an_empty_tenant_without_keys_by_the_lease_named,
    an_adoption_is_given_up_replaced_or_outlived_by_its_lease,
    an_adoption_is_given_up_only_by_the_lease_it_was_noted_for,
    a_commit_that_says_it_adopts_is_refused_unless_that_adoption_is_noted_for_its_lease,
    a_lease_moved_on_is_told_so_whatever_its_commit_adopts_and_whatever_is_noted,
    a_commit_repeated_does_not_record_its_restore_again,
    a_commit_is_answered_by_its_lease_then_what_it_follows_then_the_adoption,
    a_tenant_given_a_key_or_written_to_elsewhere_meanwhile_takes_nothing_up,
    a_purge_claim_let_go_is_taken_again_at_once_and_any_other_is_left_be,
    a_tenant_removed_before_it_took_a_backup_up_is_purged_as_the_tenant_it_was,
    a_cluster_keeps_the_id_it_was_given,
    a_tenant_still_being_purged_is_refused_saying_how_long_ago_that_was_asked,
    a_purge_is_claimed_only_once_due_and_renewed_only_while_it_is_the_claims_to_do,
    a_purged_tenant_is_looked_over_by_one_node_now_and_then_while_remembered,
    ping_succeeds,
    a_partition_never_cleaned_has_no_record_and_one_not_registered_is_unknown,
    a_clean_is_recorded_only_by_the_lease_that_is_current,
    what_a_clean_recorded_is_only_ever_raised,
    what_a_clean_recorded_stays_through_leases_and_commits,
    a_partition_removed_loses_its_record_and_one_made_again_starts_with_none,
    a_record_is_of_its_own_partition_alone,
    a_partition_never_shipped_has_nothing_and_one_not_registered_is_unknown,
    a_hold_is_taken_from_the_lowest_row_still_there_once,
    a_hold_capped_below_a_clean_or_above_the_commits_is_refused,
    a_base_is_recorded_only_where_it_stands,
    a_base_below_the_hold_or_its_row_or_a_clean_is_refused,
    a_base_is_raised_only_and_a_new_run_starts_afresh,
    a_runs_writes_are_fenced_by_its_base_and_only_raise,
    a_clean_is_held_to_the_shipping_cap_on_the_record_it_writes,
    a_cap_is_raised_only_and_never_past_what_its_run_shipped,
    a_stall_is_taken_only_from_a_run_that_saw_what_it_has_shipped_and_notes_it_too,
    a_stall_raises_what_was_shipped_only,
    shipping_stays_through_leases_and_is_of_its_own_partition_alone,
    a_partition_removed_loses_its_shipping_and_one_made_again_starts_with_none,
    the_commits_after_a_position_are_read_from_their_rows_in_order,
    the_commits_a_run_ships_are_read_only_by_a_shipper_that_saw_the_run_as_it_stands,
    the_commits_read_skip_where_the_positions_do_and_lose_what_is_forgotten,
    the_commits_of_another_incarnation_are_refused,
    a_run_is_unshipped_while_commits_lie_above_what_it_shipped,
    unshipped_partitions_are_given_in_turn_by_key,
    forget_history_keeps_every_row_from_the_floor_shipping_needs,
    what_forget_history_keeps_for_shipping_lets_the_floor_rise_with_what_is_shipped,
}

/// The test database's connection options.
fn database_options() -> Option<PgConnectOptions> {
    database_url().map(|url| PgConnectOptions::from_str(&url).expect("database url"))
}

/// A pool of `connections` by `options`, each named `application`, so
/// that what they wait on can be told apart from what other tests do
/// meanwhile.
async fn named_pool(
    options: PgConnectOptions,
    application: &str,
    connections: u32,
) -> sqlx::PgPool {
    pool(options.application_name(application), connections).await
}

/// How many connections named `application` are waiting on a lock.
async fn waiting_on_a_lock(pool: &sqlx::PgPool, application: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT count(*) FROM pg_stat_activity
         WHERE application_name = $1 AND wait_event_type = 'Lock'",
    )
    .bind(application)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// Wait until `count` connections named `application` are waiting on a
/// lock, or until `done`.
async fn waiting_on_locks(
    pool: &sqlx::PgPool,
    application: &str,
    count: i64,
    done: impl Fn() -> bool,
) {
    let deadline = std::time::Instant::now() + HUNG;
    loop {
        let waiting = waiting_on_a_lock(pool, application).await;
        if waiting >= count || done() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{waiting} of {count} are waiting on a lock"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Do `work`, and say whether a connection named `application` was seen
/// waiting on a lock meanwhile. A wait may go unseen, if this was kept from
/// looking while it lasted; none is seen that was not there.
async fn seen_waiting_on_a_lock<T>(
    pool: &sqlx::PgPool,
    application: &str,
    work: impl Future<Output = T>,
) -> (T, bool) {
    tokio::pin!(work);
    let mut seen = false;
    loop {
        tokio::select! {
            done = &mut work => return (done, seen),
            () = tokio::time::sleep(Duration::from_millis(10)) => {
                seen |= waiting_on_a_lock(pool, application).await > 0;
            }
        }
    }
}

#[tokio::test]
async fn postgres_backup_claims_made_together_are_won_by_exactly_one() {
    const CLAIMS: usize = 8;
    let Some(pg) = Pg::connect_beside(CLAIMS as u32 + 2).await else {
        return;
    };
    let application = format!("claims-{}", uuid::Uuid::new_v4().simple());
    let claiming = PostgresCoordinator::with_pool(
        named_pool(pg.options(), &application, CLAIMS as u32 + 2).await,
    )
    .await
    .unwrap();
    for _ in 0..5 {
        let tenant = key().tenant;
        for partition in 0..3 {
            let key = PartitionKey {
                tenant: tenant.clone(),
                partition: Partition(partition),
            };
            pg.coordinator.register(&key).await.unwrap();
        }
        // Every claim starts while another transaction holds the rows, so
        // that all of them look before any has taken the tenant.
        let mut holding = pg.pool.begin().await.unwrap();
        sqlx::query("SELECT 1 FROM vellev.partitions WHERE tenant = $1 FOR UPDATE")
            .bind(tenant.as_str())
            .execute(&mut *holding)
            .await
            .unwrap();
        let claims: Vec<_> = (0..CLAIMS)
            .map(|_| {
                let (claiming, tenant, claim) =
                    (claiming.clone(), tenant.clone(), uuid::Uuid::new_v4());
                tokio::spawn(async move {
                    let won = claiming.claim_backup(&tenant, claim, None, HOUR).await;
                    won.map(|won| won.then_some(claim))
                })
            })
            .collect();
        waiting_on_locks(&pg.pool, &application, CLAIMS as i64, || false).await;
        holding.commit().await.unwrap();

        let mut won = Vec::new();
        for claim in join_all(claims).await {
            won.extend(claim.unwrap().unwrap());
        }
        assert_eq!(won.len(), 1, "claims won: {won:?}");
        let held: Vec<Option<uuid::Uuid>> = sqlx::query_scalar(
            "SELECT DISTINCT backup_claim FROM vellev.partitions WHERE tenant = $1",
        )
        .bind(tenant.as_str())
        .fetch_all(&pg.pool)
        .await
        .unwrap();
        assert_eq!(
            held,
            [Some(won[0])],
            "every partition is held by the one claim won"
        );
    }
}

#[tokio::test]
async fn postgres_a_late_renewal_and_a_takeover_never_both_hold_a_claim() {
    let Some(pg) = Pg::connect_beside(4).await else {
        return;
    };
    let application = format!("renewal-{}", uuid::Uuid::new_v4().simple());
    let claiming = PostgresCoordinator::with_pool(named_pool(pg.options(), &application, 4).await)
        .await
        .unwrap();
    for _ in 0..5 {
        let tenant = key().tenant;
        for partition in 0..3 {
            let key = PartitionKey {
                tenant: tenant.clone(),
                partition: Partition(partition),
            };
            pg.coordinator.register(&key).await.unwrap();
        }
        let (renewed, taking) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        assert!(pg
            .coordinator
            .claim_backup(&tenant, renewed, None, HOUR)
            .await
            .unwrap());
        sqlx::query(
            "UPDATE vellev.partitions SET backup_claimed_until = now() - interval '1 second'
             WHERE tenant = $1",
        )
        .bind(tenant.as_str())
        .execute(&pg.pool)
        .await
        .unwrap();
        // The renewal comes as the claim runs out, and the takeover while
        // it is under way: both wait on another transaction holding the
        // rows, the renewal first.
        let mut holding = pg.pool.begin().await.unwrap();
        sqlx::query("SELECT 1 FROM vellev.partitions WHERE tenant = $1 FOR UPDATE")
            .bind(tenant.as_str())
            .execute(&mut *holding)
            .await
            .unwrap();
        let renewal = {
            let (claiming, tenant) = (claiming.clone(), tenant.clone());
            tokio::spawn(async move { claiming.renew_backup(&tenant, renewed, HOUR).await })
        };
        waiting_on_locks(&pg.pool, &application, 1, || false).await;
        let takeover = {
            let (claiming, tenant) = (claiming.clone(), tenant.clone());
            tokio::spawn(async move { claiming.claim_backup(&tenant, taking, None, HOUR).await })
        };
        waiting_on_locks(&pg.pool, &application, 2, || false).await;
        holding.commit().await.unwrap();

        let renewal = renewal.await.unwrap().unwrap();
        let takeover = takeover.await.unwrap().unwrap();
        let held: Vec<Option<uuid::Uuid>> = sqlx::query_scalar(
            "SELECT DISTINCT backup_claim FROM vellev.partitions WHERE tenant = $1",
        )
        .bind(tenant.as_str())
        .fetch_all(&pg.pool)
        .await
        .unwrap();
        match (renewal, takeover) {
            (true, false) => assert_eq!(held, [Some(renewed)]),
            (false, true) => assert_eq!(held, [Some(taking)]),
            both => panic!("renewed and taken over: {both:?}, held by {held:?}"),
        }
    }
}

#[tokio::test]
async fn postgres_writes_across_a_tenants_partitions_made_together_all_succeed() {
    const WRITES: usize = 64;
    const AT_ONCE: u32 = 8;
    let Some(options) = database_options() else {
        return;
    };
    let _granted = granted(AT_ONCE + 1).await;
    let scratch = ScratchDatabase::create(options).await;
    let coordinator = PostgresCoordinator::with_pool(pool(scratch.options(), AT_ONCE).await)
        .await
        .unwrap();
    let tenant = key().tenant;
    for partition in [2, 0, 1] {
        let key = PartitionKey {
            tenant: tenant.clone(),
            partition: Partition(partition),
        };
        coordinator.register(&key).await.unwrap();
    }
    let claims: Vec<uuid::Uuid> = (0..4).map(|_| uuid::Uuid::new_v4()).collect();
    let writes = (0..WRITES).map(|write| {
        let (coordinator, tenant, claim) = (
            coordinator.clone(),
            tenant.clone(),
            claims[write % claims.len()],
        );
        tokio::spawn(async move {
            match write % 4 {
                0 => coordinator
                    .claim_backup(&tenant, claim, None, HOUR)
                    .await
                    .map(drop),
                1 => coordinator
                    .renew_backup(&tenant, claim, HOUR)
                    .await
                    .map(drop),
                2 => coordinator.release_backup(&tenant, claim, true).await,
                _ => coordinator
                    .set_restore_window(&tenant, Some(HOUR * write as u32))
                    .await
                    .map(drop),
            }
        })
    });
    for written in join_all(writes).await {
        written
            .unwrap()
            .expect("no write across the partitions fails, by deadlock or otherwise");
    }
}

/// How many connections a test with a database of its own is granted: a
/// pool of [`SCRATCH_POOL`] on it, and one to make it and to drop it with.
const SCRATCH: u32 = SCRATCH_POOL + 1;
const SCRATCH_POOL: u32 = 4;

/// A database of its own for one test, dropped however the test ends.
struct ScratchDatabase {
    admin: PgConnectOptions,
    name: String,
}

/// What every scratch database's name starts with, before the second it
/// was made in and an id.
const SCRATCH_NAMED: &str = "vellev_scratch_";
/// How long after it was made a scratch database nobody is connected to
/// is taken for left behind by a run that was killed: far longer than any
/// test here runs.
const SCRATCH_LEFT_AFTER: Duration = Duration::from_secs(60 * 60);

/// Drop the scratch databases left behind by runs that were killed, once
/// for this run: those made over [`SCRATCH_LEFT_AFTER`] ago that nobody is
/// connected to. Another run's are its own to drop, and are in use.
async fn sweep_scratch_databases(admin: &sqlx::PgPool) {
    static SWEPT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    let swept = || async {
        for name in scratch_databases_left_behind(admin).await {
            // Not forced: one in use after all is left to whoever uses it.
            let drop = format!("DROP DATABASE IF EXISTS {name}");
            if let Err(error) = sqlx::query(&drop).execute(admin).await {
                eprintln!("could not drop the scratch database {name}: {error}");
            }
        }
    };
    SWEPT.get_or_init(swept).await;
}

/// The scratch databases made over [`SCRATCH_LEFT_AFTER`] ago that nobody
/// is connected to.
async fn scratch_databases_left_behind(admin: &sqlx::PgPool) -> Vec<String> {
    let idle: Vec<String> = sqlx::query_scalar(
        "SELECT datname::text FROM pg_database d
         WHERE starts_with(datname::text, $1)
           AND NOT EXISTS (SELECT 1 FROM pg_stat_activity a WHERE a.datname = d.datname)",
    )
    .bind(SCRATCH_NAMED)
    .fetch_all(admin)
    .await
    .unwrap();
    let now = Utc::now().timestamp();
    let left = idle.into_iter().filter(|name| left_behind(name, now));
    left.collect()
}

/// Whether the scratch database `name` was made over
/// [`SCRATCH_LEFT_AFTER`] before `now`, by the second its name says.
fn left_behind(name: &str, now: i64) -> bool {
    let made = name
        .strip_prefix(SCRATCH_NAMED)
        .and_then(|rest| rest.split('_').next())
        .and_then(|secs| secs.parse::<i64>().ok());
    made.is_some_and(|made| now - made > SCRATCH_LEFT_AFTER.as_secs() as i64)
}

impl ScratchDatabase {
    async fn create(admin: PgConnectOptions) -> Self {
        let made = Utc::now().timestamp();
        let name = format!("{SCRATCH_NAMED}{made}_{}", uuid::Uuid::new_v4().simple());
        let creating = pool(admin.clone(), 1).await;
        sweep_scratch_databases(&creating).await;
        sqlx::query(&format!("CREATE DATABASE {name}"))
            .execute(&creating)
            .await
            .unwrap();
        creating.close().await;
        Self { admin, name }
    }

    fn options(&self) -> PgConnectOptions {
        self.admin.clone().database(&self.name)
    }
}

impl Drop for ScratchDatabase {
    fn drop(&mut self) {
        let (admin, name) = (self.admin.clone(), self.name.clone());
        // The test's own runtime cannot be blocked on from within it, so the
        // database is dropped from a runtime of its own on another thread.
        let dropped = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime to drop the database with");
            runtime.block_on(async {
                let admin = sqlx::PgPool::connect_with(admin).await?;
                sqlx::query(&format!("DROP DATABASE IF EXISTS {name} WITH (FORCE)"))
                    .execute(&admin)
                    .await
                    .map(|_| ())
            })
        })
        .join();
        if !matches!(dropped, Ok(Ok(()))) {
            eprintln!("could not drop the scratch database {}", self.name);
        }
    }
}

#[tokio::test]
async fn postgres_raises_no_format_past_a_node_announcing_meanwhile() {
    let Some(options) = database_options() else {
        return;
    };
    // The format is the cluster's, and never falls, so this raises it in a
    // database of its own.
    let _granted = granted(SCRATCH).await;
    let scratch = ScratchDatabase::create(options).await;
    let application = format!("raise-{}", uuid::Uuid::new_v4().simple());
    let pool = named_pool(scratch.options(), &application, SCRATCH_POOL).await;
    let coordinator = PostgresCoordinator::with_pool(pool.clone()).await.unwrap();

    let mut announcing = pool.begin().await.unwrap();
    sqlx::query(
        "INSERT INTO vellev.nodes (node, format, protocol, seen_at) VALUES ('older', 1, 1, now())",
    )
    .execute(&mut *announcing)
    .await
    .unwrap();
    let (unraised, waited) =
        seen_waiting_on_a_lock(&pool, &application, coordinator.raise_format(1)).await;
    assert_eq!(unraised.unwrap(), 1);
    assert!(!waited, "a format already written waits on no announce");
    let raising = {
        let coordinator = coordinator.clone();
        tokio::spawn(async move { coordinator.raise_format(2).await })
    };
    waiting_on_locks(&pool, &application, 1, || raising.is_finished()).await;
    announcing.commit().await.unwrap();
    let raised = raising.await.unwrap().unwrap();
    assert_eq!(
        (raised, coordinator.format().await.unwrap()),
        (1, 1),
        "a node that reads only 1 announced"
    );
    assert_eq!(
        coordinator
            .announce(
                &node("newer"),
                Versions {
                    format: 2,
                    protocol: 1
                }
            )
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn postgres_a_raise_waits_on_announces_only_briefly_and_not_at_all_for_one_it_cannot_make() {
    let Some(options) = database_options() else {
        return;
    };
    let _granted = granted(SCRATCH).await;
    let scratch = ScratchDatabase::create(options).await;
    let application = format!("raise-{}", uuid::Uuid::new_v4().simple());
    let pool = named_pool(scratch.options(), &application, SCRATCH_POOL).await;
    let coordinator = PostgresCoordinator::with_pool(pool.clone()).await.unwrap();
    let newer = Versions {
        format: 2,
        protocol: 1,
    };
    let announcing = || async {
        let mut announcing = pool.begin().await.unwrap();
        sqlx::query(
            "INSERT INTO vellev.nodes (node, format, protocol, seen_at)
             VALUES ($1, 2, 1, now())",
        )
        .bind(node("announcing").as_str())
        .execute(&mut *announcing)
        .await
        .unwrap();
        announcing
    };

    let held_up = announcing().await;
    let started = std::time::Instant::now();
    let raised = tokio::time::timeout(HUNG, coordinator.raise_format(2)).await;
    assert!(
        matches!(raised, Ok(Ok(1))),
        "a raise held up gives up, leaving the format: {raised:?}"
    );
    assert!(started.elapsed() >= Duration::from_millis(900), "it waited");
    held_up.commit().await.unwrap();

    let older = node("older");
    coordinator
        .announce(&older, Versions { format: 1, ..newer })
        .await
        .unwrap();
    let held_up = announcing().await;
    let (unraised, waited) =
        seen_waiting_on_a_lock(&pool, &application, coordinator.raise_format(2)).await;
    assert_eq!(unraised.unwrap(), 1);
    assert!(!waited, "a raise a node registered rules out takes no lock");
    held_up.commit().await.unwrap();

    coordinator.depart(&older).await.unwrap();
    assert_eq!(
        coordinator.raise_format(2).await.unwrap(),
        2,
        "asked again, it is raised"
    );
}

#[tokio::test]
async fn postgres_a_raise_held_up_by_another_raise_gives_up_as_briefly() {
    let Some(options) = database_options() else {
        return;
    };
    let _granted = granted(SCRATCH).await;
    let scratch = ScratchDatabase::create(options).await;
    let pool = pool(scratch.options(), SCRATCH_POOL).await;
    let coordinator = PostgresCoordinator::with_pool(pool.clone()).await.unwrap();

    // Another raise holds the format's row, as one under way does.
    let mut raising = pool.begin().await.unwrap();
    sqlx::query("SELECT format FROM vellev.cluster FOR UPDATE")
        .execute(&mut *raising)
        .await
        .unwrap();
    let raised = tokio::time::timeout(HUNG, coordinator.raise_format(2)).await;
    assert!(
        matches!(raised, Ok(Ok(1))),
        "the raise gives up, leaving the format: {raised:?}"
    );
    raising.commit().await.unwrap();
    assert_eq!(coordinator.raise_format(2).await.unwrap(), 2);
}

#[tokio::test]
async fn postgres_migrations_are_idempotent_under_concurrent_starts() {
    const STARTS: u32 = 6;
    let Some(options) = database_options() else {
        return;
    };
    let _granted = granted(STARTS * HARNESS + 1).await;
    let scratch = ScratchDatabase::create(options).await;
    let starts: Vec<_> = (0..STARTS)
        .map(|_| {
            let options = scratch.options();
            tokio::spawn(async move {
                PostgresCoordinator::with_pool(pool(options, HARNESS).await).await
            })
        })
        .collect();
    for started in starts {
        started.await.unwrap().expect("concurrent migration");
    }
}

#[tokio::test]
async fn a_node_starts_on_a_database_a_newer_release_has_migrated() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let ahead = 1_000_000 + i64::from(rand_suffix());
    sqlx::query("INSERT INTO vellev.schema_version (version, sql) VALUES ($1, 'SELECT 1')")
        .bind(ahead)
        .execute(&pg.pool)
        .await
        .unwrap();
    let started = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await).await;
    sqlx::query("DELETE FROM vellev.schema_version WHERE version = $1")
        .bind(ahead)
        .execute(&pg.pool)
        .await
        .unwrap();
    started.expect("a node of the older release starts");
}

/// One history of what cleans record, as each coordinator answers it.
async fn answers_to_cleans(h: &dyn Harness) -> Vec<String> {
    let c = h.coordinator();
    let key = registered(h).await;
    let mut said = vec![format!("{:?}", c.cleaned(&key).await.unwrap())];
    let holder = node("a");
    let (first, _) = lease(h, &key, &holder).await;
    for position in [4, 2, 4, 6] {
        said.push(format!("{:?}", record(c, &first, position).await));
    }
    said.push(format!("{:?}", c.cleaned(&key).await));
    let (again, _) = lease(h, &key, &holder).await;
    said.push(format!(
        "{:?}",
        record(c, &first, 9).await.map_err(|_| "fenced")
    ));
    said.push(format!("{:?}", record(c, &again, 9).await));
    c.remove(&key).await.unwrap();
    said.push(format!(
        "{:?}",
        c.cleaned(&key).await.map_err(|_| "unknown")
    ));
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    said.push(format!("{:?}", c.cleaned(&key).await));
    said
}

/// DW-681: the memory coordinator, which the simulation cleans against,
/// answers what cleans record just as Postgres does, call for call.
#[tokio::test]
async fn postgres_and_memory_answer_what_cleans_record_alike() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let memory = answers_to_cleans(&Memory::new()).await;
    assert_eq!(answers_to_cleans(&pg).await, memory);
    assert_eq!(memory.len(), 10, "{memory:?}");
}

/// DW-681: a coordinator bounded as a node bounds it answers what cleans
/// record as the one it wraps.
#[tokio::test]
async fn a_bounded_coordinator_answers_what_cleans_record_as_the_one_it_wraps() {
    struct BoundedMemory {
        memory: Memory,
        bounded: Bounded,
    }
    #[async_trait]
    impl Harness for BoundedMemory {
        fn coordinator(&self) -> &dyn Coordinator {
            &self.bounded
        }
        async fn pass(&self, key: &PartitionKey, duration: Duration) {
            self.memory.pass(key, duration).await;
        }
    }
    let memory = Memory::new();
    let inner = Arc::new(MemoryCoordinator::new(memory.clock.clone()));
    let bounded = BoundedMemory {
        bounded: Bounded::new(inner, Duration::from_secs(10)),
        memory,
    };
    assert_eq!(
        answers_to_cleans(&bounded).await,
        answers_to_cleans(&Memory::new()).await
    );
}

/// DW-681's migration, rolled forward over a database the release before
/// left: its partitions gain no record, a record can then be made, half a
/// record is refused, and a node of the release before still registers
/// partitions there, which then have none.
#[tokio::test]
async fn postgres_the_cleaned_record_rolls_forward_over_the_release_before() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let before = registered(&pg).await;
    // As the release before left it: no record anywhere, nor anything a
    // later release added.
    sqlx::raw_sql(
        "DROP TABLE vellev.shipping;
         ALTER TABLE vellev.partitions
             DROP CONSTRAINT ship_held,
             DROP CONSTRAINT ship_base_whole,
             DROP CONSTRAINT ship_base_held,
             DROP COLUMN ship_hold_from,
             DROP COLUMN ship_cap,
             DROP COLUMN ship_base,
             DROP COLUMN ship_base_checkpoint,
             DROP COLUMN ship_base_checkpoint_position;
         DELETE FROM vellev.schema_version WHERE version = 12;
         ALTER TABLE vellev.partitions
             DROP CONSTRAINT cleaned_whole,
             DROP COLUMN cleaned_below,
             DROP COLUMN cleaned_position;
         DELETE FROM vellev.schema_version WHERE version = 11;",
    )
    .execute(&pg.pool)
    .await
    .unwrap();

    let migrated = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    assert_eq!(migrated.cleaned(&before).await.unwrap(), None);
    let (held, _) = lease(&pg, &before, &node("a")).await;
    let (base, position) = base(3);
    assert_eq!(
        migrated.clean_below(&held, &base, position).await.unwrap(),
        cleaned_at(3)
    );
    let half = sqlx::query(
        "UPDATE vellev.partitions SET cleaned_position = NULL
         WHERE tenant = $1 AND partition = $2",
    )
    .bind(before.tenant.as_str())
    .bind(i64::from(before.partition.0))
    .execute(&pg.pool)
    .await;
    assert!(half.is_err(), "half a record is refused: {half:?}");
    // A node of the release before inserts its rows naming none of it.
    let older = crate::key();
    sqlx::query(
        "INSERT INTO vellev.partitions (tenant, partition, generation)
         VALUES ($1, $2, nextval('vellev.generations'))",
    )
    .bind(older.tenant.as_str())
    .bind(i64::from(older.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    assert_eq!(migrated.cleaned(&older).await.unwrap(), None);
}

/// The lock shipping's hold takes, and `forget_history` will: as
/// `PostgresCoordinator` takes it, on (tenant, partition).
const SHIP_LOCK: &str = "SELECT pg_advisory_xact_lock(hashtextextended($1 || '/' || $2::text, 0))";

async fn ship_locked(
    pool: &sqlx::PgPool,
    key: &PartitionKey,
) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut locked = pool.begin().await.unwrap();
    sqlx::query(SHIP_LOCK)
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *locked)
        .await
        .unwrap();
    locked
}

/// DW-668, #81 review B1: a base raised over a run that had shipped lowers
/// the floor, from what the run shipped down to the new base; so it waits
/// for a forget_history under way, and then, the rows it would base on
/// gone, is refused. Raised without the lock, it saw the rows the delete
/// had yet to commit, was recorded, and was left with none of them.
#[tokio::test]
async fn postgres_a_base_raised_while_history_is_forgotten_waits_and_is_refused_its_rows_gone() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(
        &pg,
        &key,
        &[
            (1, "l1"),
            (2, "m2"),
            (3, "l3"),
            (4, "m4"),
            (5, "l5"),
            (6, "l6"),
        ],
    )
    .await;
    pg.coordinator
        .hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let first = ship_base(1, 2, "m2");
    pg.coordinator.record_base(&key, now, &first).await.unwrap();
    pg.coordinator
        .note_shipped(&key, now, LogPosition(1), LogPosition(5))
        .await
        .unwrap();
    // forget_history under way, as it runs: the lock, then its delete of
    // what its floor of 6 lets go.
    let mut forgetting = ship_locked(&pg.pool, &key).await;
    sqlx::query("DELETE FROM vellev.commits WHERE tenant = $1 AND partition = $2 AND position < 6")
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *forgetting)
        .await
        .unwrap();
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let raising = tokio::spawn({
        let key = key.clone();
        async move { other.record_base(&key, now, &ship_base(3, 4, "m4")).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !raising.is_finished(),
        "the raise waits for the shipping lock"
    );
    forgetting.commit().await.unwrap();
    let refused = raising.await.unwrap();
    assert!(stale(&refused, Stale::NoRowAtBase), "{refused:?}");
    let shipping = pg.coordinator.shipping(&key).await.unwrap();
    assert_eq!(shipping.base, Some(first));
    assert_eq!(shipping.shipped, Some(LogPosition(5)));
}

/// DW-668, S5: the stale stall's race, on two connections. A stale
/// shipper decided its stall on the run before it shipped on; the run's
/// note is made, not yet committed, when the stall is asked for. The
/// stall is one statement whose condition is what the run has shipped, so
/// it waits for the note's row and is then refused as shipped since: a
/// stall read first and written after would stop the run under way.
#[tokio::test]
async fn postgres_a_stale_stall_waiting_on_a_note_is_refused_once_it_lands() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "m2"), (3, "l3")]).await;
    pg.coordinator
        .hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(1, 2, "m2");
    pg.coordinator.record_base(&key, now, &base).await.unwrap();
    // The run under way's note, made and not yet committed.
    let mut noting = pg.pool.begin().await.unwrap();
    sqlx::query("UPDATE vellev.shipping SET shipped = 3 WHERE tenant = $1 AND partition = $2")
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *noting)
        .await
        .unwrap();
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let stalling = tokio::spawn({
        let key = key.clone();
        async move {
            other
                .stall_shipping(
                    &key,
                    now,
                    LogPosition(1),
                    None,
                    Some(LogPosition(2)),
                    "stale",
                )
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !stalling.is_finished(),
        "the stall waits for the note's row"
    );
    noting.commit().await.unwrap();
    let refused = stalling.await.unwrap();
    assert!(
        matches!(
            refused,
            Err(CoordinatorError::ShippedSince {
                seen: None,
                shipped: Some(LogPosition(3)),
                ..
            })
        ),
        "{refused:?}"
    );
    let shipping = pg.coordinator.shipping(&key).await.unwrap();
    assert_eq!(
        (shipping.shipped, shipping.stalled),
        (Some(LogPosition(3)), None)
    );
}

/// A clean's record asked for on its own connection, its transaction
/// left open: the partition's row held until it commits.
async fn cleaned_uncommitted(
    pool: &sqlx::PgPool,
    key: &PartitionKey,
    position: u64,
) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut cleaning = pool.begin().await.unwrap();
    let (base, position) = base(position);
    sqlx::query(
        "UPDATE vellev.partitions SET cleaned_below = $3, cleaned_position = $4
         WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .bind(base.0.as_str())
    .bind(i64::try_from(position.0).unwrap())
    .execute(&mut *cleaning)
    .await
    .unwrap();
    cleaning
}

/// `asked`, which must be answered without waiting for another's lock:
/// a refusal on the cap read, not a wait for the row. A wait fails the
/// test, rather than hanging it while the lock's holder waits on this.
async fn at_once<T>(asked: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), asked)
        .await
        .expect("answered at once, not after waiting for the partition's row")
}

/// DW-668 PR 6, the interleavings 3a deferred: the cap against a clean's
/// record, on two connections, in each order. The second waits for the
/// partition's row, and is then refused on what the first wrote.
#[tokio::test]
async fn postgres_a_cap_and_a_clean_record_meet_on_the_partitions_row_in_either_order() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let other = || async {
        PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
            .await
            .unwrap()
    };
    for order in [
        "hold, then clean",
        "clean, then hold",
        "base, then clean",
        "clean, then base",
    ] {
        let key = registered(&pg).await;
        let now = incarnation_of(&pg, &key).await;
        committed_heads(
            &pg,
            &key,
            &[
                (1, "l1"),
                (2, "m2"),
                (3, "l3"),
                (4, "m4"),
                (5, "l5"),
                (6, "m6"),
            ],
        )
        .await;
        let (held, _) = lease(&pg, &key, &node("cleaner")).await;
        let second = other().await;
        match order {
            "hold, then clean" => {
                // The hold made, capped at 2, and not yet committed.
                let mut holding = pg.pool.begin().await.unwrap();
                sqlx::query("UPDATE vellev.partitions SET ship_hold_from = 1, ship_cap = 2 WHERE tenant = $1 AND partition = $2")
                    .bind(key.tenant.as_str())
                    .bind(i64::from(key.partition.0))
                    .execute(&mut *holding)
                    .await
                    .unwrap();
                let cleaning = tokio::spawn({
                    let held = held.clone();
                    async move { record(&second, &held, 4).await }
                });
                tokio::time::sleep(Duration::from_millis(300)).await;
                assert!(
                    !cleaning.is_finished(),
                    "{order}: the clean waits for the row"
                );
                holding.commit().await.unwrap();
                let refused = cleaning.await.unwrap();
                assert!(above_cap(&refused, 4, 2), "{order}: {refused:?}");
                assert_eq!(pg.coordinator.cleaned(&key).await.unwrap(), None, "{order}");
            }
            "clean, then hold" => {
                let cleaning = cleaned_uncommitted(&pg.pool, &key, 4).await;
                let holding = tokio::spawn({
                    let key = key.clone();
                    async move { second.hold_for_shipping(&key, now, LogPosition(2)).await }
                });
                tokio::time::sleep(Duration::from_millis(300)).await;
                assert!(
                    !holding.is_finished(),
                    "{order}: the hold waits for the row"
                );
                cleaning.commit().await.unwrap();
                let refused = holding.await.unwrap();
                assert!(stale(&refused, Stale::CleanedAbove), "{order}: {refused:?}");
            }
            "base, then clean" => {
                pg.coordinator
                    .hold_for_shipping(&key, now, LogPosition(2))
                    .await
                    .unwrap();
                // The base made at 4, raising the cap to it, and not yet
                // committed: a clean at 6 waits, and is refused at 4.
                let mut basing = pg.pool.begin().await.unwrap();
                sqlx::query("UPDATE vellev.partitions SET ship_base = 3, ship_base_checkpoint = 'm4', ship_base_checkpoint_position = 4, ship_cap = 4 WHERE tenant = $1 AND partition = $2")
                    .bind(key.tenant.as_str())
                    .bind(i64::from(key.partition.0))
                    .execute(&mut *basing)
                    .await
                    .unwrap();
                // A cap only rises, so a clean above the cap it reads is
                // refused on it at once, without waiting: never taken.
                let refused = at_once(record(&second, &held, 6)).await;
                assert!(above_cap(&refused, 6, 2), "{order}: {refused:?}");
                basing.commit().await.unwrap();
                let refused = record(&second, &held, 6).await;
                assert!(above_cap(&refused, 6, 4), "{order}: {refused:?}");
            }
            _ => {
                pg.coordinator
                    .hold_for_shipping(&key, now, LogPosition(2))
                    .await
                    .unwrap();
                // A record above the base's checkpoint, made as a clean
                // with no cap to hold it would make it (H keeps one from
                // being asked for now), and not yet committed: the base
                // waits, and is refused on its own direct check.
                let cleaning = cleaned_uncommitted(&pg.pool, &key, 6).await;
                let basing = tokio::spawn({
                    let key = key.clone();
                    async move { second.record_base(&key, now, &ship_base(3, 4, "m4")).await }
                });
                tokio::time::sleep(Duration::from_millis(300)).await;
                assert!(!basing.is_finished(), "{order}: the base waits for the row");
                cleaning.commit().await.unwrap();
                let refused = basing.await.unwrap();
                assert!(stale(&refused, Stale::CleanedAbove), "{order}: {refused:?}");
            }
        }
    }
}

/// DW-668 PR 6: a raise of the cap by a run re-based meanwhile, on two
/// connections. The re-base holds the partition's row; the raise waits for
/// it, and is then refused on the base it names, which is no longer the
/// run's: it never sets the new run's cap from the old run's watermark.
#[tokio::test]
async fn postgres_a_raise_by_a_run_re_based_meanwhile_waits_and_is_refused() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(
        &pg,
        &key,
        &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4"), (5, "l5")],
    )
    .await;
    let c = &pg.coordinator;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(2, 2, "m2"))
        .await
        .unwrap();
    c.note_shipped(&key, now, LogPosition(2), LogPosition(5))
        .await
        .unwrap();
    // A re-base at 3, its checkpoint and cap at 4, made and not committed:
    // the partition's row first, then the run's, as record_base writes.
    let mut basing = pg.pool.begin().await.unwrap();
    sqlx::query(
        "UPDATE vellev.partitions SET ship_base = 3, ship_base_checkpoint = 'm4',
             ship_base_checkpoint_position = 4, ship_cap = 4
         WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&mut *basing)
    .await
    .unwrap();
    sqlx::query(
        "UPDATE vellev.shipping SET base = 3, shipped = NULL WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&mut *basing)
    .await
    .unwrap();
    let second = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let raising = tokio::spawn({
        let key = key.clone();
        async move {
            second
                .raise_ship_cap(&key, now, LogPosition(2), LogPosition(5))
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!raising.is_finished(), "the raise waits for the row");
    basing.commit().await.unwrap();
    let refused = raising.await.unwrap();
    assert!(other_base(&refused, Some(3)), "{refused:?}");
    assert_eq!(c.shipping(&key).await.unwrap().cap, Some(LogPosition(4)));
}

/// DW-668 PR 6: a raise of the cap and a clean's record on two
/// connections. A raise only loosens: a clean above the cap it reads is
/// refused on it at once, and taken once the raise has committed.
#[tokio::test]
async fn postgres_a_clean_above_a_cap_being_raised_is_taken_once_the_raise_commits() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(
        &pg,
        &key,
        &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4"), (5, "l5")],
    )
    .await;
    let c = &pg.coordinator;
    c.hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    c.record_base(&key, now, &ship_base(2, 2, "m2"))
        .await
        .unwrap();
    c.note_shipped(&key, now, LogPosition(2), LogPosition(5))
        .await
        .unwrap();
    let (held, _) = lease(&pg, &key, &node("cleaner")).await;
    // The raise to 5 made, and not yet committed.
    let mut raising = pg.pool.begin().await.unwrap();
    sqlx::query("UPDATE vellev.partitions SET ship_cap = 5 WHERE tenant = $1 AND partition = $2")
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *raising)
        .await
        .unwrap();
    let second = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let refused = at_once(record(&second, &held, 4)).await;
    assert!(above_cap(&refused, 4, 2), "{refused:?}");
    raising.commit().await.unwrap();
    assert_eq!(record(&second, &held, 4).await.unwrap(), cleaned_at(4));
}

/// DW-668, #81 review: the floor is the current incarnation's alone. The
/// rows of one removed go as they always have, a hold on the one made
/// again keeping none of them.
#[tokio::test]
async fn postgres_a_hold_keeps_none_of_a_removed_incarnations_rows() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let c = &pg.coordinator;
    let key = registered(&pg).await;
    let old = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "l2"), (3, "l3")]).await;
    assert!(c.remove(&key).await.unwrap());
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "n1"), (2, "n2")]).await;
    c.hold_for_shipping(&key, now, LogPosition(0))
        .await
        .unwrap();
    let old_rows = || {
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM vellev.commits
             WHERE tenant = $1 AND partition = $2 AND incarnation = $3",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(i64::try_from(old.0).unwrap())
        .fetch_one(&pg.pool)
    };
    assert_eq!(
        old_rows().await.unwrap(),
        3,
        "the removed incarnation's rows are there"
    );
    c.forget_history(&key, HOUR).await.unwrap();
    assert_eq!(old_rows().await.unwrap(), 0);
    assert_eq!(commits_of(&pg, &key, now, 0, 10).await.unwrap(), [1, 2]);
}

/// DW-668, PR 5: forget_history waits for a hold being taken under the
/// shipping lock, and then keeps every row from it: its delete is a
/// statement after its own taking of the lock, which sees the hold.
#[tokio::test]
async fn postgres_forget_history_waits_for_a_hold_being_taken_and_keeps_what_it_holds() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "m2"), (3, "l3")]).await;
    pg.pass(&key, 2 * HOUR).await;
    committed_heads(&pg, &key, &[(4, "l4")]).await;
    // The hold, as hold_for_shipping takes it: the lock, then its update.
    let mut holding = ship_locked(&pg.pool, &key).await;
    sqlx::query(
        "UPDATE vellev.partitions SET ship_hold_from = 1, ship_cap = 2
         WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&mut *holding)
    .await
    .unwrap();
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let forgetting = tokio::spawn({
        let key = key.clone();
        async move { other.forget_history(&key, HOUR).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !forgetting.is_finished(),
        "forget_history waits for the lock"
    );
    holding.commit().await.unwrap();
    assert_eq!(forgetting.await.unwrap().unwrap(), 0);
    assert_eq!(
        commits_of(&pg, &key, now, 0, 10).await.unwrap(),
        [1, 2, 3, 4]
    );
}

/// DW-668, PR 5: a hold and forget_history made at once, each from its own
/// connection, in either order: every row from the hold is still there.
#[tokio::test]
async fn postgres_a_hold_and_forget_history_at_once_leave_every_row_from_the_hold() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    for _ in 0..10 {
        let key = registered(&pg).await;
        let now = incarnation_of(&pg, &key).await;
        committed_heads(&pg, &key, &[(1, "l1"), (2, "m2"), (3, "l3")]).await;
        pg.pass(&key, 2 * HOUR).await;
        committed_heads(&pg, &key, &[(4, "l4")]).await;
        let (held, forgot) = tokio::join!(
            pg.coordinator.hold_for_shipping(&key, now, LogPosition(2)),
            other.forget_history(&key, HOUR),
        );
        let (from, _) = (held.unwrap().hold_from.unwrap().0, forgot.unwrap());
        let left = commits_of(&pg, &key, now, 0, 10).await.unwrap();
        let expected: Vec<u64> = (from..=4).collect();
        assert_eq!(left, expected, "held from {from}");
        // Either the hold came first, from 1, or the forgetting did, and
        // the hold is from the lowest row it left.
        assert!(from == 1 || from == 3, "held from {from}");
    }
}

/// DW-668, PR 5: forget_removed takes no shipping lock: it deletes only rows
/// of incarnations no longer registered, which no hold covers.
#[tokio::test]
async fn postgres_forget_removed_takes_no_shipping_lock() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    committed_heads(&pg, &key, &[(1, "l1")]).await;
    let locked = ship_locked(&pg.pool, &key).await;
    let forgetting = pg.coordinator.forget_removed(&key, None, 1_000);
    tokio::time::timeout(Duration::from_secs(5), forgetting)
        .await
        .expect("forget_removed does not wait for the shipping lock")
        .unwrap();
    drop(locked);
}

/// DW-668, amendment I: a hold waits for whoever holds the shipping lock,
/// as `forget_history` will while it deletes, and reads the lowest row
/// only after: rows deleted under the lock are not held from.
#[tokio::test]
async fn postgres_a_hold_waits_for_the_shipping_lock_and_reads_rows_after_it() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "l2"), (3, "m3"), (4, "l4")]).await;
    let now = incarnation_of(&pg, &key).await;
    let mut locked = ship_locked(&pg.pool, &key).await;
    sqlx::query("DELETE FROM vellev.commits WHERE tenant = $1 AND partition = $2 AND position < 3")
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *locked)
        .await
        .unwrap();
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let holding = tokio::spawn({
        let key = key.clone();
        async move { other.hold_for_shipping(&key, now, LogPosition(3)).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!holding.is_finished(), "the hold waits for the lock");
    locked.commit().await.unwrap();
    assert_eq!(holding.await.unwrap().unwrap(), held(now, 3, 3));
}

/// DW-668, amendment I: a commit never waits on the shipping lock.
#[tokio::test]
async fn postgres_a_commit_is_never_held_up_by_the_shipping_lock() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let locked = ship_locked(&pg.pool, &key).await;
    let (held_lease, _) = lease(&pg, &key, &node("a")).await;
    let committing = pg.coordinator.commit(&held_lease, commit(1, "l1"));
    tokio::time::timeout(Duration::from_secs(5), committing)
        .await
        .expect("a commit does not wait for the shipping lock")
        .unwrap();
    drop(locked);
}

/// DW-668, amendment J: two holds at once, each from its own connection:
/// one is taken, and both are told the one that stands.
#[tokio::test]
async fn postgres_of_two_holds_at_once_one_stands_and_both_are_told_it() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    for _ in 0..10 {
        let key = registered(&pg).await;
        committed_heads(&pg, &key, &[(1, "l1"), (2, "m2"), (3, "m3")]).await;
        let now = incarnation_of(&pg, &key).await;
        let (one, two) = tokio::join!(
            pg.coordinator.hold_for_shipping(&key, now, LogPosition(2)),
            other.hold_for_shipping(&key, now, LogPosition(3)),
        );
        let (one, two) = (one.unwrap(), two.unwrap());
        assert_eq!(one, two);
        assert!([held(now, 1, 2), held(now, 1, 3)].contains(&one), "{one:?}");
        assert_eq!(pg.coordinator.shipping(&key).await.unwrap(), one);
    }
}

/// DW-668, amendment J: a partition removed while a base is being
/// recorded. The removal waits for the partition's row, and either takes
/// the run's record with it or leaves the base refused: no record of a
/// run outlives its partition.
/// DW-668, #77 review: a shipping write that ran out of tries tells
/// whoever asked to ask again, as a backend's failure does.
#[test]
fn a_shipping_write_changed_under_it_too_often_may_be_asked_again() {
    let contended = CoordinatorError::ShippingContended { key: crate::key() };
    assert!(contended.is_retryable());
    let stale = CoordinatorError::StaleShipping {
        key: crate::key(),
        why: Stale::NotHeld,
    };
    assert!(!stale.is_retryable());
}

/// Hold the partition's row in a transaction of its own, as a statement
/// of another node's does while it runs, until the transaction ends.
async fn row_held(
    pool: &sqlx::PgPool,
    key: &PartitionKey,
    sql: &str,
) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut held = pool.begin().await.unwrap();
    sqlx::query(sql)
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *held)
        .await
        .unwrap();
    held
}

/// DW-668, #77 review: a base recorded while a clean records itself above
/// the base's checkpoint, the clean's statement holding the partition's
/// row first, is refused: the condition is on the row, read again once
/// the clean is done.
#[tokio::test]
async fn postgres_a_base_waiting_on_a_clean_that_records_above_its_checkpoint_is_refused() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4")]).await;
    pg.coordinator
        .hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let clean = row_held(
        &pg.pool,
        &key,
        "UPDATE vellev.partitions SET cleaned_below = 'm4', cleaned_position = 4
         WHERE tenant = $1 AND partition = $2",
    )
    .await;
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let recording = tokio::spawn({
        let key = key.clone();
        async move { other.record_base(&key, now, &ship_base(1, 2, "m2")).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !recording.is_finished(),
        "the base waits for the partition's row"
    );
    clean.commit().await.unwrap();
    let refused = recording.await.unwrap();
    assert!(stale(&refused, Stale::CleanedAbove), "{refused:?}");
    assert_eq!(
        pg.coordinator.shipping(&key).await.unwrap(),
        held(now, 1, 2)
    );
}

/// The commit at `position` of `head`, as the commit statement makes it,
/// the partition's row first, held open in a transaction of its own.
async fn committing(
    pool: &sqlx::PgPool,
    key: &PartitionKey,
    incarnation: Incarnation,
    position: i64,
    head: &str,
) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut open = row_held(
        pool,
        key,
        &format!(
            "UPDATE vellev.partitions SET log_position = {position}, manifest = '{head}'
             WHERE tenant = $1 AND partition = $2"
        ),
    )
    .await;
    sqlx::query(
        "INSERT INTO vellev.commits (tenant, partition, incarnation, position, head)
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .bind(i64::try_from(incarnation.0).unwrap())
    .bind(position)
    .bind(head)
    .execute(&mut *open)
    .await
    .unwrap();
    open
}

/// DW-668, #77 review: a base whose checkpoint's commit has not yet landed
/// is refused at once, without waiting for the commit: none of its rows
/// is there to be seen, and the partition's row is not asked for. Asked
/// again once the commit has landed, it is recorded.
#[tokio::test]
async fn postgres_a_base_on_a_checkpoint_still_being_committed_is_refused_and_recorded_once_it_lands(
) {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "m2")]).await;
    pg.coordinator
        .hold_for_shipping(&key, now, LogPosition(2))
        .await
        .unwrap();
    let open = committing(&pg.pool, &key, now, 3, "m3").await;
    let base = ship_base(3, 3, "m3");
    let refused = tokio::time::timeout(
        Duration::from_secs(5),
        pg.coordinator.record_base(&key, now, &base),
    )
    .await
    .expect("refused at once, not waiting for the commit");
    assert!(stale(&refused, Stale::NoRowAtBase), "{refused:?}");
    open.commit().await.unwrap();
    assert_eq!(
        pg.coordinator.record_base(&key, now, &base).await.unwrap(),
        based(now, 1, &base)
    );
}

/// DW-668, #77 review: a base whose checkpoint is committed, recorded while
/// a later commit holds the partition's row, waits for it and is then
/// recorded: what it reads of the commit rows at the base and checkpoint,
/// read before the wait, still holds after it, as rows there go only by
/// forget_history, under the shipping lock.
#[tokio::test]
async fn postgres_a_base_waiting_on_a_later_commit_is_recorded_once_it_lands() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let key = registered(&pg).await;
    let now = incarnation_of(&pg, &key).await;
    committed_heads(&pg, &key, &[(1, "l1"), (2, "l2"), (3, "m3")]).await;
    pg.coordinator
        .hold_for_shipping(&key, now, LogPosition(3))
        .await
        .unwrap();
    let open = committing(&pg.pool, &key, now, 4, "l4").await;
    let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let base = ship_base(2, 3, "m3");
    let recording = tokio::spawn({
        let (key, base) = (key.clone(), base.clone());
        async move { other.record_base(&key, now, &base).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !recording.is_finished(),
        "the base waits for the partition's row"
    );
    open.commit().await.unwrap();
    assert_eq!(recording.await.unwrap().unwrap(), based(now, 1, &base));
}

#[tokio::test]
async fn postgres_a_removal_beside_a_base_being_recorded_leaves_no_run_behind() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let runs_of = |key: &PartitionKey| {
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM vellev.shipping WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str().to_string())
        .bind(i64::from(key.partition.0))
        .fetch_one(&pg.pool)
    };
    for removed_first in [true, false] {
        let key = registered(&pg).await;
        committed_heads(&pg, &key, &[(1, "l1"), (2, "m2")]).await;
        let now = incarnation_of(&pg, &key).await;
        pg.coordinator
            .hold_for_shipping(&key, now, LogPosition(2))
            .await
            .unwrap();
        let other = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
            .await
            .unwrap();
        // Hold the partition's row, as a removal under way holds it.
        let mut removing = pg.pool.begin().await.unwrap();
        sqlx::query(
            "SELECT 1 FROM vellev.partitions WHERE tenant = $1 AND partition = $2 FOR UPDATE",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *removing)
        .await
        .unwrap();
        let recording = tokio::spawn({
            let key = key.clone();
            async move { other.record_base(&key, now, &ship_base(1, 2, "m2")).await }
        });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            !recording.is_finished(),
            "the base waits for the partition's row"
        );
        if removed_first {
            sqlx::query("DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = $2")
                .bind(key.tenant.as_str())
                .bind(i64::from(key.partition.0))
                .execute(&mut *removing)
                .await
                .unwrap();
            removing.commit().await.unwrap();
            let refused = recording.await.unwrap();
            assert!(
                matches!(refused, Err(CoordinatorError::UnknownPartition(_))),
                "{refused:?}"
            );
        } else {
            removing.commit().await.unwrap();
            recording.await.unwrap().unwrap();
            assert_eq!(runs_of(&key).await.unwrap(), 1);
            assert!(pg.coordinator.remove(&key).await.unwrap());
        }
        assert_eq!(
            runs_of(&key).await.unwrap(),
            0,
            "removed first: {removed_first}"
        );
    }
}

/// One history of shipping, as each coordinator answers it: each answer
/// with the incarnation said as whether it is the partition's current
/// one, which the two coordinators number apart.
async fn answers_to_shipping(h: &dyn Harness) -> Vec<String> {
    let c = h.coordinator();
    let key = registered(h).await;
    let now = incarnation_of(h, &key).await;
    let said = |result: Result<Shipping, CoordinatorError>| match result {
        Ok(shipping) => format!(
            "current {} hold {:?} cap {:?} base {:?} shipped {:?} stalled {:?}",
            shipping.incarnation == now,
            shipping.hold_from.map(|at| at.0),
            shipping.cap.map(|at| at.0),
            shipping
                .base
                .map(|base| (base.position.0, base.checkpoint.0, base.checkpoint_at.0)),
            shipping.shipped.map(|at| at.0),
            shipping.stalled,
        ),
        Err(CoordinatorError::UnknownPartition(_)) => "unknown".into(),
        Err(CoordinatorError::OtherIncarnation { .. }) => "other incarnation".into(),
        Err(CoordinatorError::OtherBase { recorded, .. }) => format!("other base, {recorded:?}"),
        Err(CoordinatorError::StaleShipping { why, .. }) => format!("stale, {why:?}"),
        Err(CoordinatorError::InvalidArgument(_)) => "invalid".into(),
        Err(CoordinatorError::ShippedSince { seen, shipped, .. }) => {
            format!("shipped since, {seen:?} to {shipped:?}")
        }
        Err(other) => format!("{other:?}"),
    };
    let mut answers = vec![said(c.shipping(&key).await)];
    answers.push(said(c.record_base(&key, now, &ship_base(1, 2, "m2")).await));
    committed_heads(h, &key, &[(1, "l1"), (2, "m2"), (3, "l3"), (4, "m4")]).await;
    answers.push(said(c.hold_for_shipping(&key, now, LogPosition(5)).await));
    answers.push(said(
        c.hold_for_shipping(&key, Incarnation(now.0 + 1), LogPosition(2))
            .await,
    ));
    answers.push(said(c.hold_for_shipping(&key, now, LogPosition(2)).await));
    answers.push(said(c.hold_for_shipping(&key, now, LogPosition(4)).await));
    for base in [
        ship_base(1, 1, "l1"),
        ship_base(3, 4, "m-other"),
        ship_base(3, 4, "m4"),
        ship_base(1, 2, "m2"),
    ] {
        answers.push(said(c.record_base(&key, now, &base).await));
    }
    for (base, position) in [(3, 3), (3, 4), (3, 3), (2, 5), (3, 2)] {
        answers.push(said(
            c.note_shipped(&key, now, LogPosition(base), LogPosition(position))
                .await,
        ));
    }
    answers.push(said(
        c.stall_shipping(&key, now, LogPosition(3), None, None, "stale")
            .await,
    ));
    answers.push(said(
        c.stall_shipping(
            &key,
            now,
            LogPosition(3),
            Some(LogPosition(4)),
            None,
            "first",
        )
        .await,
    ));
    answers.push(said(
        c.stall_shipping(
            &key,
            now,
            LogPosition(3),
            Some(LogPosition(4)),
            None,
            "second",
        )
        .await,
    ));
    c.remove(&key).await.unwrap();
    answers.push(said(c.shipping(&key).await));
    purge(c, &key.tenant).await;
    c.register(&key).await.unwrap();
    answers.push(said(c.shipping(&key).await.map(|shipping| Shipping {
        incarnation: now,
        ..shipping
    })));
    answers.push(said(
        c.note_shipped(&key, now, LogPosition(3), LogPosition(7))
            .await,
    ));
    answers
}

/// DW-668: the memory coordinator, which the simulation ships against,
/// answers shipping just as Postgres does, call for call.
#[tokio::test]
async fn postgres_and_memory_answer_shipping_alike() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let memory = answers_to_shipping(&Memory::new()).await;
    assert_eq!(answers_to_shipping(&pg).await, memory);
    assert_eq!(memory.len(), 21, "{memory:?}");
}

/// DW-668: a coordinator bounded as a node bounds it answers shipping as
/// the one it wraps.
#[tokio::test]
async fn a_bounded_coordinator_answers_shipping_as_the_one_it_wraps() {
    struct BoundedMemory {
        memory: Memory,
        bounded: Bounded,
    }
    #[async_trait]
    impl Harness for BoundedMemory {
        fn coordinator(&self) -> &dyn Coordinator {
            &self.bounded
        }
        async fn pass(&self, key: &PartitionKey, duration: Duration) {
            self.memory.pass(key, duration).await;
        }
    }
    let memory = Memory::new();
    let inner = Arc::new(MemoryCoordinator::new(memory.clock.clone()));
    let bounded = BoundedMemory {
        bounded: Bounded::new(inner, Duration::from_secs(10)),
        memory,
    };
    assert_eq!(
        answers_to_shipping(&bounded).await,
        answers_to_shipping(&Memory::new()).await
    );
}

/// DW-668's migration, rolled forward over a database the release before
/// left: its partitions have no shipping, shipping can then be held and
/// based, the constraints refuse a base without its checkpoint or below
/// its hold, a node of the release before still registers partitions
/// there, and removing one takes its run with it.
#[tokio::test]
async fn postgres_shipping_rolls_forward_over_the_release_before() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let before = registered(&pg).await;
    committed_heads(&pg, &before, &[(1, "l1"), (2, "m2")]).await;
    // As the release before left it: no shipping anywhere.
    sqlx::raw_sql(
        "DROP TABLE vellev.shipping;
         ALTER TABLE vellev.partitions
             DROP CONSTRAINT ship_held,
             DROP CONSTRAINT ship_base_whole,
             DROP CONSTRAINT ship_base_held,
             DROP COLUMN ship_hold_from,
             DROP COLUMN ship_cap,
             DROP COLUMN ship_base,
             DROP COLUMN ship_base_checkpoint,
             DROP COLUMN ship_base_checkpoint_position;
         DELETE FROM vellev.schema_version WHERE version = 12;",
    )
    .execute(&pg.pool)
    .await
    .unwrap();

    let migrated = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let now = incarnation_of(&pg, &before).await;
    assert_eq!(migrated.shipping(&before).await.unwrap(), not_shipped(now));
    migrated
        .hold_for_shipping(&before, now, LogPosition(2))
        .await
        .unwrap();
    let base = ship_base(1, 2, "m2");
    assert_eq!(
        migrated.record_base(&before, now, &base).await.unwrap(),
        based(now, 1, &base)
    );
    for (wrong, sql) in [
        (
            "half a base",
            "UPDATE vellev.partitions SET ship_base_checkpoint = NULL",
        ),
        (
            "a base below its hold",
            "UPDATE vellev.partitions SET ship_hold_from = 2",
        ),
        (
            "a checkpoint above the cap",
            "UPDATE vellev.partitions SET ship_cap = 1",
        ),
        (
            "a hold without a cap",
            "UPDATE vellev.partitions SET ship_cap = NULL",
        ),
    ] {
        let refused = sqlx::query(&format!("{sql} WHERE tenant = $1 AND partition = $2"))
            .bind(before.tenant.as_str())
            .bind(i64::from(before.partition.0))
            .execute(&pg.pool)
            .await;
        assert!(refused.is_err(), "{wrong} is refused: {refused:?}");
    }
    let below =
        sqlx::query("UPDATE vellev.shipping SET shipped = 0 WHERE tenant = $1 AND partition = $2")
            .bind(before.tenant.as_str())
            .bind(i64::from(before.partition.0))
            .execute(&pg.pool)
            .await;
    assert!(
        below.is_err(),
        "a watermark below its base is refused: {below:?}"
    );
    // A node of the release before inserts its rows naming none of it, and
    // deletes them as it does, which takes their run with them.
    let older = crate::key();
    sqlx::query(
        "INSERT INTO vellev.partitions (tenant, partition, generation)
         VALUES ($1, $2, nextval('vellev.generations'))",
    )
    .bind(older.tenant.as_str())
    .bind(i64::from(older.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    let theirs = incarnation_of(&pg, &older).await;
    assert_eq!(
        migrated.shipping(&older).await.unwrap(),
        not_shipped(theirs)
    );
    sqlx::query("DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = $2")
        .bind(before.tenant.as_str())
        .bind(i64::from(before.partition.0))
        .execute(&pg.pool)
        .await
        .unwrap();
    let left: i64 = sqlx::query_scalar("SELECT count(*) FROM vellev.shipping WHERE tenant = $1")
        .bind(before.tenant.as_str())
        .fetch_one(&pg.pool)
        .await
        .unwrap();
    assert_eq!(left, 0);
}

/// One history of what a shipper reads, as each coordinator answers it.
async fn answers_to_reading(h: &dyn Harness) -> Vec<String> {
    let c = h.coordinator();
    let (key, now) = based_at_1(h, &[(3, "l3"), (6, "m6")]).await;
    let said_points = |points: Result<Vec<RestorePoint>, CoordinatorError>| match points {
        Ok(points) => format!(
            "{:?}",
            points
                .iter()
                .map(|point| (point.position.0, point.head.0.clone()))
                .collect::<Vec<_>>()
        ),
        Err(CoordinatorError::UnknownPartition(_)) => "unknown".into(),
        Err(CoordinatorError::OtherIncarnation { .. }) => "other incarnation".into(),
        Err(other) => format!("{other:?}"),
    };
    let said_runs = |runs: Result<Vec<Unshipped>, CoordinatorError>| match runs {
        Ok(runs) => format!(
            "{:?}",
            runs.iter()
                .map(|run| {
                    let ours = run.key == key && run.incarnation == now;
                    (
                        ours,
                        run.base.0,
                        run.shipped.map(|at| at.0),
                        run.committed.0,
                    )
                })
                .collect::<Vec<_>>()
        ),
        Err(other) => format!("{other:?}"),
    };
    let mut answers = Vec::new();
    for (after, limit) in [(0, 10), (2, 1), (3, 10), (6, 10)] {
        answers.push(said_points(
            c.commits_after(&key, now, LogPosition(after), limit).await,
        ));
    }
    let ahead = Incarnation(now.0 + 1);
    answers.push(said_points(
        c.commits_after(&key, ahead, LogPosition(0), 10).await,
    ));
    answers.push(said_runs(c.unshipped(None, 10).await));
    c.note_shipped(&key, now, LogPosition(1), LogPosition(3))
        .await
        .unwrap();
    answers.push(said_runs(c.unshipped(None, 10).await));
    c.note_shipped(&key, now, LogPosition(1), LogPosition(6))
        .await
        .unwrap();
    answers.push(said_runs(c.unshipped(None, 10).await));
    c.remove(&key).await.unwrap();
    answers.push(said_points(
        c.commits_after(&key, now, LogPosition(0), 10).await,
    ));
    answers.push(said_runs(c.unshipped(None, 10).await));
    answers
}

/// DW-668: the memory coordinator, which the simulation ships against,
/// answers what a shipper reads just as Postgres does, call for call.
#[tokio::test]
async fn postgres_and_memory_answer_what_a_shipper_reads_alike() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let memory = answers_to_reading(&Memory::new()).await;
    assert_eq!(answers_to_reading(&pg).await, memory);
    assert_eq!(memory.len(), 10, "{memory:?}");
}

/// DW-668: a coordinator bounded as a node bounds it answers what a
/// shipper reads as the one it wraps.
#[tokio::test]
async fn a_bounded_coordinator_answers_what_a_shipper_reads_as_the_one_it_wraps() {
    struct BoundedMemory {
        memory: Memory,
        bounded: Bounded,
    }
    #[async_trait]
    impl Harness for BoundedMemory {
        fn coordinator(&self) -> &dyn Coordinator {
            &self.bounded
        }
        async fn pass(&self, key: &PartitionKey, duration: Duration) {
            self.memory.pass(key, duration).await;
        }
    }
    let memory = Memory::new();
    let inner = Arc::new(MemoryCoordinator::new(memory.clock.clone()));
    let bounded = BoundedMemory {
        bounded: Bounded::new(inner, Duration::from_secs(10)),
        memory,
    };
    assert_eq!(
        answers_to_reading(&bounded).await,
        answers_to_reading(&Memory::new()).await
    );
}

#[tokio::test]
async fn postgres_tenants_there_before_lineages_are_each_given_one() {
    let Some(pg) = Pg::connect_beside(HARNESS).await else {
        return;
    };
    let (one, other) = (key(), key());
    let keys = [
        beside(&one, 0),
        beside(&one, 1),
        beside(&one, 2),
        other.clone(),
    ];
    for key in &keys {
        pg.coordinator.register(key).await.unwrap();
    }
    // As the release before left it: no lineage anywhere, nor anything a
    // later release added.
    sqlx::raw_sql(
        "DROP TABLE vellev.shipping;
         ALTER TABLE vellev.partitions
             DROP CONSTRAINT ship_held,
             DROP CONSTRAINT ship_base_whole,
             DROP CONSTRAINT ship_base_held,
             DROP COLUMN ship_hold_from,
             DROP COLUMN ship_cap,
             DROP COLUMN ship_base,
             DROP COLUMN ship_base_checkpoint,
             DROP COLUMN ship_base_checkpoint_position;
         DELETE FROM vellev.schema_version WHERE version = 12;
         ALTER TABLE vellev.partitions
             DROP CONSTRAINT cleaned_whole,
             DROP COLUMN cleaned_below,
             DROP COLUMN cleaned_position;
         DELETE FROM vellev.schema_version WHERE version = 11;
         DROP TRIGGER removed_tenant ON vellev.partitions;
         DROP FUNCTION vellev.removed_tenant();
         DROP TRIGGER registered_lineage ON vellev.partitions;
         DROP FUNCTION vellev.registered_lineage();
         ALTER TABLE vellev.partitions DROP COLUMN lineage, DROP COLUMN adopting;
         ALTER TABLE vellev.purges DROP COLUMN lineage, DROP COLUMN looked_at;
         ALTER TABLE vellev.cluster DROP COLUMN id;
         DELETE FROM vellev.schema_version WHERE version = 10;",
    )
    .execute(&pg.pool)
    .await
    .unwrap();

    let migrated = PostgresCoordinator::with_pool(pool(pg.options(), HARNESS).await)
        .await
        .unwrap();
    let lineages: Vec<(String, uuid::Uuid)> = sqlx::query_as(
        "SELECT tenant, lineage FROM vellev.partitions WHERE tenant = ANY($1) ORDER BY tenant",
    )
    .bind(vec![one.tenant.as_str(), other.tenant.as_str()])
    .fetch_all(&pg.pool)
    .await
    .unwrap();
    let of = |tenant: &TenantId| -> Vec<uuid::Uuid> {
        let of = lineages
            .iter()
            .filter(|(named, _)| named == tenant.as_str());
        of.map(|(_, lineage)| *lineage).collect()
    };
    let (ones, others) = (of(&one.tenant), of(&other.tenant));
    assert_eq!(ones.len(), 3);
    assert!(ones.iter().all(|lineage| *lineage == ones[0]), "{ones:?}");
    assert_ne!(ones[0], others[0], "each tenant has its own");
    assert_eq!(
        migrated
            .identity(&one.tenant)
            .await
            .unwrap()
            .unwrap()
            .lineage,
        Lineage(ones[0])
    );
}

#[tokio::test]
async fn postgres_a_tenant_another_node_is_taking_to_look_over_is_passed_over_not_waited_for() {
    const DAY: Duration = Duration::from_secs(24 * 60 * 60);
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let c = &pg.coordinator;
    let (taken, free) = (registered(&pg).await, registered(&pg).await);
    for key in [&taken, &free] {
        assert!(c.remove(key).await.unwrap());
        purge(c, &key.tenant).await;
        pg.pass(key, HOUR).await;
    }
    // Another node has the first: its row is locked until that one is done.
    let mut taking = pg.pool.begin().await.unwrap();
    sqlx::query("SELECT tenant FROM vellev.purges WHERE tenant = $1 FOR UPDATE")
        .bind(taken.tenant.as_str())
        .execute(&mut *taking)
        .await
        .unwrap();
    let looked = |limit: usize| async move {
        let looking = c.look_overs(DAY, HOUR / 4, limit);
        let looked = tokio::time::timeout(Duration::from_secs(10), looking).await;
        let looked = looked.expect("a tenant being taken is not waited for");
        let tenants = looked.unwrap().into_iter().map(|removed| removed.tenant);
        tenants.collect::<Vec<_>>()
    };
    assert_eq!(looked(8).await, std::slice::from_ref(&free.tenant));
    assert_eq!(looked(8).await, []);
    taking.commit().await.unwrap();
    assert_eq!(looked(8).await, std::slice::from_ref(&taken.tenant));
}

#[test]
fn a_scratch_database_is_left_behind_an_hour_after_its_name_says_it_was_made() {
    let now = 1_800_000_000;
    let named = |made: i64| format!("{SCRATCH_NAMED}{made}_0123abcd");
    assert!(left_behind(&named(now - 3_601), now));
    assert!(!left_behind(&named(now - 3_599), now));
    assert!(!left_behind(&named(now), now));
    assert!(!left_behind("vellev_scratch_0123abcd", now), "says no time");
    assert!(!left_behind("vellev", now));
}

#[tokio::test]
async fn postgres_a_scratch_database_left_behind_is_dropped_and_one_in_use_is_not() {
    let Some(options) = database_options() else {
        return;
    };
    let _granted = granted(3).await;
    let admin = pool(options.clone(), 1).await;
    let long_ago = Utc::now().timestamp() - 2 * SCRATCH_LEFT_AFTER.as_secs() as i64;
    let named = |suffix: &str| {
        let id = uuid::Uuid::new_v4().simple();
        format!("{SCRATCH_NAMED}{long_ago}_{id}{suffix}")
    };
    let (left, used) = (named("left"), named("used"));
    for name in [&left, &used] {
        let make = format!("CREATE DATABASE {name}");
        sqlx::query(&make).execute(&admin).await.unwrap();
    }
    let using = pool(options.clone().database(&used), 1).await;
    sqlx::query("SELECT 1").execute(&using).await.unwrap();

    // What the first harness of a run drops, whichever that was.
    let found = scratch_databases_left_behind(&admin).await;
    assert!(found.contains(&left), "{found:?}");
    assert!(!found.contains(&used), "one in use is not left behind");
    using.close().await;
    for name in [&left, &used] {
        let drop = format!("DROP DATABASE IF EXISTS {name} WITH (FORCE)");
        sqlx::query(&drop).execute(&admin).await.unwrap();
    }
}

#[tokio::test]
async fn postgres_a_purge_is_not_renewed_once_done_or_its_tenant_there_whatever_its_claim() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let c = &pg.coordinator;
    let renewed = |tenant: TenantId, claim: uuid::Uuid| async move {
        c.renew_purge(&tenant, claim, HOUR).await.unwrap()
    };
    // What registering refuses, done to the rows all the same: a partition
    // of the tenant there while its purge is claimed.
    let key = registered(&pg).await;
    assert!(c.remove(&key).await.unwrap());
    let claim = purge_claimed(c, &key.tenant).await.unwrap().id;
    assert!(renewed(key.tenant.clone(), claim).await);
    sqlx::query(
        "INSERT INTO vellev.partitions (tenant, partition, generation)
         VALUES ($1, 0, nextval('vellev.generations'))",
    )
    .bind(key.tenant.as_str())
    .execute(&pg.pool)
    .await
    .unwrap();
    assert!(
        !renewed(key.tenant.clone(), claim).await,
        "a tenant of the name is there"
    );

    // And a purge noted done with its claim left on it.
    let key = registered(&pg).await;
    assert!(c.remove(&key).await.unwrap());
    let claim = purge_claimed(c, &key.tenant).await.unwrap().id;
    sqlx::query("UPDATE vellev.purges SET purged_at = now() WHERE tenant = $1")
        .bind(key.tenant.as_str())
        .execute(&pg.pool)
        .await
        .unwrap();
    assert!(!renewed(key.tenant.clone(), claim).await, "it is done");
}

#[tokio::test]
async fn postgres_a_purge_is_claimed_in_turn_with_the_other_writes_to_its_tenant() {
    let Some(pg) = Pg::connect_beside(2).await else {
        return;
    };
    let application = format!("purging-{}", uuid::Uuid::new_v4().simple());
    let claiming = PostgresCoordinator::with_pool(named_pool(pg.options(), &application, 2).await)
        .await
        .unwrap();
    let key = registered(&pg).await;
    assert!(pg.coordinator.remove(&key).await.unwrap());

    // Another write to the tenant is under way: it holds the tenant's lock.
    let mut writing = pg.pool.begin().await.unwrap();
    sqlx::query("SELECT pg_advisory_xact_lock($1, hashtext($2))")
        .bind(0x766c)
        .bind(key.tenant.as_str())
        .execute(&mut *writing)
        .await
        .unwrap();
    let claim = {
        let tenant = key.tenant.clone();
        tokio::spawn(async move { claiming.claim_purge(&tenant, Duration::ZERO, HOUR).await })
    };
    waiting_on_locks(&pg.pool, &application, 1, || claim.is_finished()).await;
    assert!(!claim.is_finished(), "the claim waits its turn");
    writing.commit().await.unwrap();
    assert!(claim.await.unwrap().unwrap().is_some());
}

#[tokio::test]
async fn postgres_a_first_commit_that_waited_while_an_adoption_was_noted_does_not_slip_past() {
    let Some(pg) = Pg::connect_beside(2).await else {
        return;
    };
    let application = format!("adopting-{}", uuid::Uuid::new_v4().simple());
    let committing =
        PostgresCoordinator::with_pool(named_pool(pg.options(), &application, 2).await)
            .await
            .unwrap();
    let key = registered(&pg).await;
    let (held, _) = lease(&pg, &key, &node("a")).await;
    let own = identity(&pg, &key.tenant).await.unwrap().lineage;

    // An adoption is being noted: the row is held, and what is noted on
    // it not yet committed.
    let to_be = serde_json::json!({ "generation": held.generation, "adoption": adoption(7) });
    let mut noting = pg.pool.begin().await.unwrap();
    sqlx::query("UPDATE vellev.partitions SET adopting = $2 WHERE tenant = $1")
        .bind(key.tenant.as_str())
        .bind(to_be.to_string())
        .execute(&mut *noting)
        .await
        .unwrap();
    let racing = {
        let held = held.clone();
        tokio::spawn(async move { committing.commit(&held, commit(1, "m1")).await })
    };
    waiting_on_locks(&pg.pool, &application, 1, || racing.is_finished()).await;
    assert!(!racing.is_finished(), "the commit waits on the row");
    noting.commit().await.unwrap();

    let raced = racing.await.unwrap();
    assert!(adopting(&raced), "it finds the adoption noted: {raced:?}");
    let now = identity(&pg, &key.tenant).await.unwrap();
    assert_eq!((now.lineage, now.committed), (own, false));
}

/// Nothing writes an adoption onto a partition that has committed, nor
/// leaves one there past the commit that makes or outlives it; should one
/// be found there all the same, it is of a lease that is no more, and no
/// commit makes it. The memory coordinator's guard has the same test.
#[tokio::test]
async fn postgres_an_adoption_noted_on_a_partition_that_has_committed_is_never_made() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let c = &pg.coordinator;
    let key = registered(&pg).await;
    let (held, _) = lease(&pg, &key, &node("a")).await;
    let own = taken_up(&pg, &key.tenant).await;
    let to_be = adoption(2);
    assert!(c
        .adopt_on_commit(&key, held.generation, &to_be)
        .await
        .unwrap());
    sqlx::query(
        "UPDATE vellev.partitions SET log_position = 1 WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    let restoring = adopting_commit(2, "restored", to_be.lineage);
    assert!(unadopted(&c.commit(&held, restoring).await));
    assert_eq!(taken_up(&pg, &key.tenant).await, own);
    c.commit(&held, commit(2, "m2")).await.unwrap();
    assert_eq!(taken_up(&pg, &key.tenant).await, own);
    assert_eq!(
        c.committed(&key).await.unwrap().unwrap().log_position,
        LogPosition(2)
    );
}

#[tokio::test]
async fn postgres_registering_removing_and_adopting_together_leave_a_tenant_one_lineage() {
    const TASKS: u32 = 12;
    let Some(pg) = Pg::connect_beside(TASKS).await else {
        return;
    };
    let together = PostgresCoordinator::with_pool(pool(pg.options(), TASKS).await)
        .await
        .unwrap();
    let key = key();
    let tasks = (0..TASKS).map(|n| {
        let (c, key, looking) = (together.clone(), key.clone(), pg.pool.clone());
        tokio::spawn(async move {
            for round in 0..12 {
                let part = beside(&key, (n + round) % 3);
                match (n + round) % 4 {
                    0 | 1 => drop(c.register(&part).await),
                    2 => {
                        if let Ok(Acquired::Lease { lease, .. }) =
                            c.acquire(&part, &node("a"), TTL).await
                        {
                            let to_be = adoption(u64::from(round) + 2);
                            let noted = c.adopt_on_commit(&part, lease.generation, &to_be).await;
                            if noted.is_ok_and(|noted| noted) {
                                let at = u64::from(round) + 2;
                                let restoring = adopting_commit(at, "m", to_be.lineage);
                                drop(c.commit(&lease, restoring).await);
                            }
                            drop(c.release(&lease).await);
                        }
                    }
                    _ => {
                        if c.remove(&part).await.unwrap() {
                            let claimed = c.claim_purge(&key.tenant, Duration::ZERO, HOUR).await;
                            if let Some(claim) = claimed.unwrap() {
                                c.purged(&key.tenant, claim.id, 24 * HOUR).await.unwrap();
                            }
                        }
                    }
                }
                let lineages: i64 = sqlx::query_scalar(
                    "SELECT count(DISTINCT lineage) FROM vellev.partitions WHERE tenant = $1",
                )
                .bind(key.tenant.as_str())
                .fetch_one(&looking)
                .await
                .unwrap();
                assert!(lineages <= 1, "{lineages} lineages in one tenant");
            }
        })
    });
    for task in join_all(tasks).await {
        task.unwrap();
    }
}

/// What the release before does to remove a tenant's only partition: it
/// knows nothing of lineages, and notes the purge itself.
#[tokio::test]
async fn postgres_a_tenant_removed_by_the_release_before_is_purged_as_the_tenant_it_was() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let c = &pg.coordinator;
    let pool = &pg.pool;
    let remove_as_before = |key: PartitionKey| async move {
        let mut removing = pool.begin().await.unwrap();
        for sql in [
            "DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = 0",
            "INSERT INTO vellev.purges (tenant)
             SELECT $1 WHERE NOT EXISTS (SELECT 1 FROM vellev.partitions WHERE tenant = $1)
             ON CONFLICT (tenant) DO UPDATE SET requested_at = now(), purged_at = NULL",
        ] {
            sqlx::query(sql)
                .bind(key.tenant.as_str())
                .execute(&mut *removing)
                .await
                .unwrap();
        }
        removing.commit().await.unwrap();
    };
    let key = registered(&pg).await;
    let first = identity(&pg, &key.tenant).await.unwrap();
    remove_as_before(key.clone()).await;
    let claim = purge_claimed(c, &key.tenant).await.unwrap();
    assert_eq!(claim.removed.lineage, first.lineage);
    c.purged(&key.tenant, claim.id, 24 * HOUR).await.unwrap();

    // Made and removed again while its purge is remembered, it is purged
    // as the tenant it was then, not the one before.
    c.register(&key).await.unwrap();
    let second = identity(&pg, &key.tenant).await.unwrap();
    remove_as_before(key.clone()).await;
    let claim = purge_claimed(c, &key.tenant).await.unwrap();
    assert_eq!(claim.removed.lineage, second.lineage);
    assert_ne!(second.lineage, first.lineage);
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
async fn a_restore_left_behind_by_a_release_that_does_not_know_them_is_not_the_tenants_now() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let key = registered(&pg).await;
    let (lease, _) = lease(&pg, &key, &node("a")).await;
    let request = RequestId::generate();
    let restoring = Commit {
        after: None,
        restore: Some(Restore {
            request,
            asked: [1; 32],
            from: LogPosition(0),
            from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
        }),
        ..commit(1, "head-1")
    };
    pg.coordinator.commit(&lease, restoring).await.unwrap();
    assert!(pg
        .coordinator
        .restored(&key, request)
        .await
        .unwrap()
        .is_some());

    // Removed and made again by a release that does not know restores.
    sqlx::query(
        "UPDATE vellev.partitions SET incarnation = nextval('vellev.generations')
         WHERE tenant = $1 AND partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    assert_eq!(pg.coordinator.restored(&key, request).await.unwrap(), None);
}

/// Forget what is kept of the removed partition, `limit` commits a call,
/// as a node does. Returns how many calls forgot something.
async fn forget_removed(coordinator: &dyn Coordinator, key: &PartitionKey, limit: usize) -> usize {
    let (mut after, mut calls) = (None, 0);
    while let Some(got_to) = coordinator.forget_removed(key, after, limit).await.unwrap() {
        after = Some(got_to);
        calls += 1;
    }
    calls
}

/// How many rows of a removed partition's history are left.
async fn kept_of(pg: &Pg, key: &PartitionKey) -> (i64, i64) {
    sqlx::query_as(
        "SELECT (SELECT count(*) FROM vellev.commits WHERE tenant = $1 AND partition = $2),
                (SELECT count(*) FROM vellev.restores WHERE tenant = $1 AND partition = $2)",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .fetch_one(&pg.pool)
    .await
    .unwrap()
}

#[tokio::test]
async fn a_busy_partition_is_removed_and_its_history_forgotten_within_the_bound_on_a_call() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let key = registered(&pg).await;
    let (lease, _) = lease(&pg, &key, &node("a")).await;
    let restoring = Commit {
        after: None,
        restore: Some(Restore {
            request: RequestId::generate(),
            asked: [1; 32],
            from: LogPosition(0),
            from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
        }),
        ..commit(1, "head-1")
    };
    pg.coordinator.commit(&lease, restoring).await.unwrap();
    // A million commits, as a busy tenant with a long window keeps.
    sqlx::query(
        "INSERT INTO vellev.commits (tenant, partition, incarnation, position, head)
         SELECT $1, $2, p.incarnation, n, 'head-' || n
         FROM vellev.partitions p, generate_series(2, 1000001) n
         WHERE p.tenant = $1 AND p.partition = $2",
    )
    .bind(key.tenant.as_str())
    .bind(i64::from(key.partition.0))
    .execute(&pg.pool)
    .await
    .unwrap();
    // As autovacuum would have by the time a tenant grew so.
    sqlx::query("ANALYZE vellev.commits")
        .execute(&pg.pool)
        .await
        .unwrap();

    // Under a node's default bound on one call, removing touches none of
    // the history, which goes ten thousand rows a call after.
    let bounded = Bounded::new(Arc::new(pg.coordinator.clone()), Duration::from_secs(10));
    assert!(bounded.remove(&key).await.unwrap());
    assert_eq!(kept_of(&pg, &key).await, (1_000_001, 1));
    let batches = forget_removed(&bounded, &key, 10_000).await;
    assert_eq!(batches, 101);
    assert_eq!(kept_of(&pg, &key).await, (0, 0));
}

async fn a_partition_made_again_keeps_its_history_while_the_old_is_forgotten(h: &dyn Harness) {
    let key = registered(h).await;
    let (first, _) = lease(h, &key, &node("a")).await;
    let coordinator = h.coordinator();
    coordinator
        .commit(&first, commit(1, "head-1"))
        .await
        .unwrap();
    coordinator.remove(&key).await.unwrap();
    purge(coordinator, &key.tenant).await;
    coordinator.register(&key).await.unwrap();
    let (again, _) = lease(h, &key, &node("b")).await;
    let request = RequestId::generate();
    let restoring = Commit {
        after: None,
        restore: Some(Restore {
            request,
            asked: [1; 32],
            from: LogPosition(1),
            from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
        }),
        ..commit(1, "head-again")
    };
    coordinator.commit(&again, restoring).await.unwrap();

    forget_removed(coordinator, &key, 1).await;
    assert_eq!(
        reach(h, &key, HOUR).await,
        Some((1, 1)),
        "the new history stays"
    );
    assert!(coordinator.restored(&key, request).await.unwrap().is_some());
}

async fn a_restore_asked_for_again_after_its_record_lapsed_is_recorded_again(h: &dyn Harness) {
    let key = registered(h).await;
    let (lease, _) = lease(h, &key, &node("a")).await;
    let coordinator = h.coordinator();
    let request = RequestId::generate();
    let restore = |from: u64| Restore {
        request,
        asked: [1; 32],
        from: LogPosition(from),
        from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
    };
    let first = Commit {
        after: None,
        restore: Some(restore(0)),
        ..commit(1, "head-1")
    };
    coordinator.commit(&lease, first).await.unwrap();
    h.pass(&key, RESTORES_KEPT_FOR + Duration::from_secs(60))
        .await;
    assert_eq!(coordinator.restored(&key, request).await.unwrap(), None);

    let again = Commit {
        after: None,
        restore: Some(restore(1)),
        ..commit(2, "head-2")
    };
    coordinator.commit(&lease, again).await.unwrap();
    let found = coordinator.restored(&key, request).await.unwrap();
    assert_eq!(found.map(|done| done.position), Some(LogPosition(2)));
}

#[tokio::test]
async fn a_call_waiting_for_a_connection_gives_up_within_its_bound() {
    let Some(pg) = Pg::connect_beside(1).await else {
        return;
    };
    let key = registered(&pg).await;
    let within = Duration::from_millis(500);
    let named = format!("vellev-test-{}", uuid::Uuid::new_v4().simple());
    let one = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(within)
        .connect_with(pg.options().application_name(&named))
        .await
        .unwrap();
    let one = PostgresCoordinator::with_pool(one).await.unwrap();

    // Another session holds the partition's row, so removing it through
    // `one` waits, on the only connection `one` has.
    let mut holding = pg.pool.begin().await.unwrap();
    sqlx::query("SELECT 1 FROM vellev.partitions WHERE tenant = $1 AND partition = $2 FOR UPDATE")
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *holding)
        .await
        .unwrap();
    let removing = tokio::spawn({
        let (one, key) = (one.clone(), key.clone());
        async move { one.remove(&key).await }
    });
    // Once the removal waits on the row, ask for something else: it fails
    // well before the half minute sqlx otherwise waits for a connection.
    let waiting = async {
        while sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM pg_stat_activity
             WHERE application_name = $1 AND wait_event_type = 'Lock'",
        )
        .bind(&named)
        .fetch_one(&pg.pool)
        .await
        .unwrap()
            == 0
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(30), waiting)
        .await
        .expect("the removal waits on the row");
    let pinged = tokio::time::timeout(Duration::from_secs(15), one.ping()).await;
    assert!(matches!(pinged, Ok(Err(_))), "{pinged:?}");
    holding.rollback().await.unwrap();
    assert!(removing.await.unwrap().unwrap());
}

#[tokio::test]
async fn a_restore_of_a_tenant_never_seen_again_is_forgotten_after_a_day() {
    let Some(pg) = Pg::connect().await else {
        return;
    };
    let key = registered(&pg).await;
    let (lease, _) = lease(&pg, &key, &node("a")).await;
    let restoring = Commit {
        after: None,
        restore: Some(Restore {
            request: RequestId::generate(),
            asked: [1; 32],
            from: LogPosition(0),
            from_at: Utc.with_ymd_and_hms(2026, 10, 5, 8, 0, 0).unwrap(),
        }),
        ..commit(1, "head-1")
    };
    pg.coordinator.commit(&lease, restoring).await.unwrap();
    // Removed by a node that never forgot its history.
    pg.coordinator.remove(&key).await.unwrap();
    pg.pass(&key, RESTORES_KEPT_FOR + Duration::from_secs(60))
        .await;
    assert!(pg.coordinator.forget_restores().await.unwrap() >= 1);
    assert_eq!(kept_of(&pg, &key).await.1, 0);
}

#[tokio::test]
async fn the_written_format_rises_only_when_every_node_reads_it_and_never_falls() {
    let c = MemoryCoordinator::default();
    let at = |format| Versions {
        format,
        protocol: 1,
    };
    let (newer, older) = (node("newer"), node("older"));
    assert_eq!(c.announce(&newer, at(2)).await.unwrap(), 1);
    assert_eq!(c.announce(&older, at(1)).await.unwrap(), 1);
    assert_eq!(
        c.raise_format(2).await.unwrap(),
        1,
        "a node still reads only 1"
    );
    assert_eq!(c.oldest().await.unwrap(), Some(at(1)));
    c.depart(&older).await.unwrap();
    assert_eq!(c.raise_format(2).await.unwrap(), 2);
    assert_eq!(
        c.announce(&older, at(1)).await.unwrap(),
        2,
        "a node announcing learns the format was raised past it"
    );
    assert_eq!(c.raise_format(1).await.unwrap(), 2, "it never falls");
    assert_eq!(c.format().await.unwrap(), 2);
}
