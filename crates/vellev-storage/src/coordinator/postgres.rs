use std::time::Duration;

use tokio::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgArguments, PgPool, PgPoolOptions};
use sqlx::query::QueryAs;
use sqlx::{Executor, FromRow, Postgres};
use uuid::Uuid;
use vellev_core::{NodeId, Partition, RequestId, TenantId};

use super::{
    checked_hold, checked_ttl, checked_window, Acquired, Commit, Committed, Coordinator,
    CoordinatorError, Forgotten, Generation, Incarnation, KeyHash, Lease, LogPosition, ManifestRef,
    PartitionKey, Point, Purges, Restore, RestorePoint, Restored, Result, TenantKey, Versions,
    RESTORES_KEPT_FOR,
};

const MIGRATIONS: &[(i64, &str)] = &[
    (1, include_str!("../../migrations/0001_partitions.sql")),
    (2, include_str!("../../migrations/0002_tenant_order.sql")),
    (3, include_str!("../../migrations/0003_incarnation.sql")),
    (4, include_str!("../../migrations/0004_nodes.sql")),
    (5, include_str!("../../migrations/0005_tenant_keys.sql")),
    (6, include_str!("../../migrations/0006_commits.sql")),
    (7, include_str!("../../migrations/0007_backups.sql")),
    (8, include_str!("../../migrations/0008_purges.sql")),
    (9, include_str!("../../migrations/0009_restores.sql")),
];

/// Serialises migrations across nodes starting together. Transaction-scoped,
/// so it is released with the transaction and never pins an RDS Proxy session.
const MIGRATION_LOCK: i64 = 0x76656c_6c6576;
/// The class of the transaction locks that serialise registering and
/// removing a tenant, each taken with the hash of the tenant's id, so that
/// a tenant is never registered while its removal marks it to be purged.
const TENANT_LOCK: i32 = 0x766c;

/// How many times `acquire` re-reads a row that changed under it before
/// giving up; each retry needs a release to land between its two statements.
const ACQUIRE_ATTEMPTS: usize = 3;

/// The coordinator on Postgres. Every state change is one statement whose
/// predicate carries the fence, and a change that did not happen is explained
/// by a follow-up read. Expiry is compared in the database's clock.
#[derive(Debug, Clone)]
pub struct PostgresCoordinator {
    pool: PgPool,
}

impl PostgresCoordinator {
    /// Connect and bring the `vellev` schema up to date, waiting at most
    /// `acquire_within` for a connection from the pool. A call given up
    /// part way leaves its connection to be tested before it is used again,
    /// which on a connection that stopped answering waits for TCP to give
    /// up; with every connection held so, a call fails within this rather
    /// than waiting with them.
    pub async fn connect(
        url: &str,
        max_connections: u32,
        acquire_within: Duration,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .acquire_timeout(acquire_within)
            .connect(url)
            .await?;
        Self::with_pool(pool).await
    }

    pub async fn with_pool(pool: PgPool) -> Result<Self> {
        migrate(&pool).await?;
        Ok(Self { pool })
    }

    async fn row(&self, key: &PartitionKey) -> Result<Option<Snapshot>> {
        Ok(snapshot(
            "SELECT generation, incarnation, lease_holder, lease_expires_at, log_position, manifest
             FROM vellev.partitions WHERE tenant = $1 AND partition = $2",
            key,
        )
        .fetch_optional(&self.pool)
        .await?)
    }

    /// Why a fenced statement on `lease` changed no row. `Ok` means the
    /// generation is still current, so the statement failed on another clause.
    async fn refusal(&self, lease: &Lease) -> Result<Snapshot> {
        match self.row(&lease.key).await? {
            None => Err(CoordinatorError::UnknownPartition(lease.key.clone())),
            Some(row) if row.generation != lease.generation => Err(lease.fenced_by(row.generation)),
            Some(row) => Ok(row),
        }
    }
}

fn snapshot<'q>(
    sql: &'static str,
    key: &'q PartitionKey,
) -> QueryAs<'q, Postgres, Snapshot, PgArguments> {
    sqlx::query_as(sql)
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
}

async fn lock_tenant(
    transaction: &mut sqlx::Transaction<'_, Postgres>,
    tenant: &TenantId,
) -> Result<()> {
    sqlx::query("SELECT pg_advisory_xact_lock($1, hashtext($2))")
        .bind(TENANT_LOCK)
        .bind(tenant.as_str())
        .execute(&mut **transaction)
        .await?;
    Ok(())
}

fn bigint(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| CoordinatorError::InvalidArgument(format!("{value} exceeds the bigint range")))
}

fn invariant(message: String) -> CoordinatorError {
    CoordinatorError::Backend(message.into())
}

async fn migrate(pool: &PgPool) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(MIGRATION_LOCK)
        .execute(&mut *tx)
        .await?;
    tx.execute("CREATE SCHEMA IF NOT EXISTS vellev").await?;
    tx.execute(
        "CREATE TABLE IF NOT EXISTS vellev.schema_version (
             version bigint PRIMARY KEY,
             sql text NOT NULL,
             applied_at timestamptz NOT NULL DEFAULT now()
         )",
    )
    .await?;
    let applied: Vec<(i64, String)> =
        sqlx::query_as("SELECT version, sql FROM vellev.schema_version ORDER BY version")
            .fetch_all(&mut *tx)
            .await?;
    for (version, sql) in pending(MIGRATIONS, &applied).map_err(invariant)? {
        tx.execute(sqlx::raw_sql(sql)).await?;
        sqlx::query("INSERT INTO vellev.schema_version (version, sql) VALUES ($1, $2)")
            .bind(version)
            .bind(sql)
            .execute(&mut *tx)
            .await?;
    }
    Ok(tx.commit().await?)
}

/// The migrations of `known` still to apply over `applied`, oldest first.
/// Every applied migration this release knows must be as it knows it; one
/// it does not know must be newer than all it knows, applied by a newer
/// release, which leaves this one working (the database is ahead during a
/// rolling deploy and after a rollback); and none still to apply may be
/// older than one applied, or it would be skipped for good.
fn pending<'a>(
    known: &'a [(i64, &'a str)],
    applied: &[(i64, String)],
) -> std::result::Result<Vec<&'a (i64, &'a str)>, String> {
    let newest_known = known.last().map_or(0, |(version, _)| *version);
    for (version, sql) in applied {
        match known.iter().find(|(v, _)| v == version) {
            Some((_, expected)) if expected == sql => {}
            None if *version > newest_known => {}
            _ => {
                return Err(format!(
                    "migration {version} differs from the one applied; recreate the database"
                ))
            }
        }
    }
    let newest_applied = applied
        .iter()
        .map(|(version, _)| *version)
        .max()
        .unwrap_or(0);
    let pending: Vec<_> = known
        .iter()
        .filter(|(version, _)| applied.iter().all(|(done, _)| done != version))
        .collect();
    match pending.first() {
        Some((version, _)) if *version < newest_applied => Err(format!(
            "migration {version} was never applied, though {newest_applied} was"
        )),
        _ => Ok(pending),
    }
}

fn version(value: i32) -> Result<u16> {
    u16::try_from(value).map_err(|error| invariant(error.to_string()))
}

/// The commits of the partition as it is now that a restore reaching back
/// `$3` seconds may go to: those within the window, and the last before
/// it, found by time and then taken by position, which grows with it.
const REACHABLE: &str = "WITH current AS (
         SELECT incarnation, now() - make_interval(secs => $3) AS start
         FROM vellev.partitions WHERE tenant = $1 AND partition = $2
     ), boundary AS (
         SELECT current.incarnation, COALESCE((
             SELECT c.position FROM vellev.commits c
             WHERE c.tenant = $1 AND c.partition = $2 AND c.incarnation = current.incarnation
               AND c.committed_at < current.start
             ORDER BY c.committed_at DESC LIMIT 1
         ), 0) AS position
         FROM current
     ), reachable AS (
         SELECT c.position, c.head, c.committed_at, c.incarnation
         FROM vellev.commits c JOIN boundary USING (incarnation)
         WHERE c.tenant = $1 AND c.partition = $2 AND c.position >= boundary.position
     )";

#[derive(FromRow)]
struct Noted {
    position: i64,
    head: String,
    committed_at: DateTime<Utc>,
}

impl Noted {
    fn point(self) -> Result<RestorePoint> {
        let position = u64::try_from(self.position)
            .map_err(|_| invariant(format!("a commit is at position {}", self.position)))?;
        Ok(RestorePoint {
            position: LogPosition(position),
            head: ManifestRef(self.head),
            at: self.committed_at,
        })
    }
}

#[derive(FromRow)]
struct Snapshot {
    #[sqlx(try_from = "i64")]
    generation: Generation,
    #[sqlx(try_from = "i64")]
    incarnation: Incarnation,
    lease_holder: Option<String>,
    lease_expires_at: Option<DateTime<Utc>>,
    #[sqlx(try_from = "i64")]
    log_position: LogPosition,
    manifest: Option<String>,
}

impl Snapshot {
    fn committed(&self) -> Committed {
        Committed {
            generation: self.generation,
            incarnation: self.incarnation,
            log_position: self.log_position,
            manifest: self.manifest.clone().map(ManifestRef),
        }
    }

    fn lease(&self) -> Option<(NodeId, DateTime<Utc>)> {
        self.lease_holder
            .clone()
            .map(NodeId::from)
            .zip(self.lease_expires_at)
    }
}

/// A restore as recorded.
#[derive(FromRow)]
struct Recorded {
    request: Uuid,
    asked: Vec<u8>,
    #[sqlx(try_from = "i64")]
    incarnation: Incarnation,
    #[sqlx(try_from = "i64")]
    position: LogPosition,
    #[sqlx(try_from = "i64")]
    from_position: LogPosition,
    from_at: DateTime<Utc>,
}

impl TryFrom<Recorded> for Restored {
    type Error = CoordinatorError;
    fn try_from(recorded: Recorded) -> Result<Self> {
        let asked = recorded.asked.try_into().map_err(|asked: Vec<u8>| {
            invariant(format!(
                "a restore's digest has {} bytes, not 32",
                asked.len()
            ))
        })?;
        Ok(Restored {
            restore: Restore {
                request: RequestId(recorded.request),
                asked,
                from: recorded.from_position,
                from_at: recorded.from_at,
            },
            incarnation: recorded.incarnation,
            position: recorded.position,
        })
    }
}

impl TryFrom<i64> for Generation {
    type Error = std::num::TryFromIntError;
    fn try_from(value: i64) -> std::result::Result<Self, Self::Error> {
        u64::try_from(value).map(Generation)
    }
}

impl TryFrom<i64> for Incarnation {
    type Error = std::num::TryFromIntError;
    fn try_from(value: i64) -> std::result::Result<Self, Self::Error> {
        u64::try_from(value).map(Incarnation)
    }
}

impl TryFrom<i64> for LogPosition {
    type Error = std::num::TryFromIntError;
    fn try_from(value: i64) -> std::result::Result<Self, Self::Error> {
        u64::try_from(value).map(LogPosition)
    }
}

impl From<sqlx::Error> for CoordinatorError {
    fn from(error: sqlx::Error) -> Self {
        CoordinatorError::backend(error)
    }
}

#[async_trait]
impl Coordinator for PostgresCoordinator {
    async fn register(&self, key: &PartitionKey) -> Result<()> {
        let mut transaction = self.pool.begin().await?;
        lock_tenant(&mut transaction, &key.tenant).await?;
        let purging: Option<i32> = sqlx::query_scalar(
            "SELECT 1 FROM vellev.purges WHERE tenant = $1 AND purged_at IS NULL",
        )
        .bind(key.tenant.as_str())
        .fetch_optional(&mut *transaction)
        .await?;
        if purging.is_some() {
            return Err(CoordinatorError::Purging(key.tenant.clone()));
        }
        sqlx::query(
            "INSERT INTO vellev.partitions (tenant, partition, generation)
             VALUES ($1, $2, nextval('vellev.generations'))
             ON CONFLICT DO NOTHING",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(())
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        let mut transaction = self.pool.begin().await?;
        lock_tenant(&mut transaction, &key.tenant).await?;
        let done =
            sqlx::query("DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = $2")
                .bind(key.tenant.as_str())
                .bind(i64::from(key.partition.0))
                .execute(&mut *transaction)
                .await?;
        sqlx::query(
            "DELETE FROM vellev.tenant_keys WHERE tenant = $1
               AND NOT EXISTS (SELECT 1 FROM vellev.partitions WHERE tenant = $1)",
        )
        .bind(key.tenant.as_str())
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO vellev.purges (tenant)
             SELECT $1 WHERE $2 AND NOT EXISTS (SELECT 1 FROM vellev.partitions WHERE tenant = $1)
             ON CONFLICT (tenant) DO UPDATE SET requested_at = now(), purged_at = NULL",
        )
        .bind(key.tenant.as_str())
        .bind(done.rows_affected() > 0)
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(done.rows_affected() > 0)
    }

    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired> {
        let ttl = checked_ttl(ttl)?;
        for _ in 0..ACQUIRE_ATTEMPTS {
            let requested_at = Instant::now();
            let taken = snapshot(
                "UPDATE vellev.partitions
                 SET generation = nextval('vellev.generations'),
                     lease_holder = $3,
                     lease_expires_at = now() + make_interval(secs => $4)
                 WHERE tenant = $1 AND partition = $2
                   AND (lease_expires_at IS NULL OR lease_expires_at <= now() OR lease_holder = $3)
                 RETURNING generation, incarnation, lease_holder, lease_expires_at, log_position, manifest",
                key,
            )
            .bind(holder.as_str())
            .bind(ttl.as_secs_f64())
            .fetch_optional(&self.pool)
            .await?;
            if let Some(row) = taken {
                let (_, expires_at) = row.lease().ok_or_else(|| {
                    invariant(format!("lease on {key} was taken without an expiry"))
                })?;
                return Ok(Acquired::Lease {
                    lease: Lease {
                        key: key.clone(),
                        holder: holder.clone(),
                        generation: row.generation,
                        ttl,
                        expires_at,
                        requested_at,
                    },
                    committed: row.committed(),
                });
            }
            let row = self
                .row(key)
                .await?
                .ok_or_else(|| CoordinatorError::UnknownPartition(key.clone()))?;
            if let Some((holder, expires_at)) = row.lease() {
                return Ok(Acquired::Held { holder, expires_at });
            }
        }
        Err(invariant(format!(
            "lease on {key} kept changing across {ACQUIRE_ATTEMPTS} attempts"
        )))
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        let ttl = checked_ttl(ttl)?;
        let requested_at = Instant::now();
        let expires_at: Option<DateTime<Utc>> = sqlx::query_scalar(
            "UPDATE vellev.partitions
             SET lease_expires_at = now() + make_interval(secs => $4)
             WHERE tenant = $1 AND partition = $2 AND generation = $3
             RETURNING lease_expires_at",
        )
        .bind(lease.key.tenant.as_str())
        .bind(i64::from(lease.key.partition.0))
        .bind(bigint(lease.generation.0)?)
        .bind(ttl.as_secs_f64())
        .fetch_optional(&self.pool)
        .await?;
        match expires_at {
            Some(expires_at) => Ok(Lease {
                ttl,
                expires_at,
                requested_at,
                ..lease.clone()
            }),
            None => {
                self.refusal(lease).await?;
                Err(invariant(format!(
                    "renewing {} changed no row at a current generation",
                    lease.key
                )))
            }
        }
    }

    async fn release(&self, lease: &Lease) -> Result<()> {
        sqlx::query(
            "UPDATE vellev.partitions
             SET generation = nextval('vellev.generations'),
                 lease_holder = NULL,
                 lease_expires_at = NULL
             WHERE tenant = $1 AND partition = $2 AND generation = $3",
        )
        .bind(lease.key.tenant.as_str())
        .bind(i64::from(lease.key.partition.0))
        .bind(bigint(lease.generation.0)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed> {
        let done = snapshot(
            "WITH done AS (
                 UPDATE vellev.partitions
                 SET log_position = $4, manifest = $5
                 WHERE tenant = $1 AND partition = $2 AND generation = $3
                   AND ((log_position < $4
                         AND ($10::bigint IS NULL
                              OR (log_position = $10 AND manifest IS NOT DISTINCT FROM $11)))
                        OR (log_position = $4 AND manifest IS NOT DISTINCT FROM $5))
                 RETURNING generation, incarnation, lease_holder, lease_expires_at, log_position, manifest
             ), noted AS (
                 INSERT INTO vellev.commits (tenant, partition, incarnation, position, head)
                 SELECT $1, $2, incarnation, log_position, manifest FROM done
                 WHERE manifest IS NOT NULL
                 ON CONFLICT DO NOTHING
             ), restored AS (
                 INSERT INTO vellev.restores
                     (tenant, partition, request, asked, incarnation, position, from_position, from_at)
                 SELECT $1, $2, $6, $9, incarnation, log_position, $7, $8 FROM done
                 WHERE $6::uuid IS NOT NULL
                 ON CONFLICT (tenant, partition, incarnation, request) DO UPDATE
                 SET asked = EXCLUDED.asked, position = EXCLUDED.position,
                     from_position = EXCLUDED.from_position, from_at = EXCLUDED.from_at,
                     made_at = now()
             )
             SELECT * FROM done",
            &lease.key,
        )
        .bind(bigint(lease.generation.0)?)
        .bind(bigint(commit.log_position.0)?)
        .bind(commit.manifest.as_ref().map(|m| m.0.as_str()))
        .bind(commit.restore.as_ref().map(|restore| restore.request.0))
        .bind(
            commit
                .restore
                .as_ref()
                .map(|restore| bigint(restore.from.0))
                .transpose()?,
        )
        .bind(commit.restore.as_ref().map(|restore| restore.from_at))
        .bind(commit.restore.as_ref().map(|restore| restore.asked.to_vec()))
        .bind(
            commit
                .after
                .as_ref()
                .map(|after| bigint(after.position.0))
                .transpose()?,
        )
        .bind(
            commit
                .after
                .as_ref()
                .and_then(|after| after.head.as_ref().map(|head| head.0.as_str())),
        )
        .fetch_optional(&self.pool)
        .await?;
        match done {
            Some(row) => Ok(row.committed()),
            None => {
                let row = self.refusal(lease).await?;
                Err(CoordinatorError::Regressed {
                    key: lease.key.clone(),
                    committed: row.log_position,
                    attempted: commit.log_position,
                })
            }
        }
    }

    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>> {
        Ok(self.row(key).await?.as_ref().map(Snapshot::committed))
    }

    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>> {
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows: Vec<(String, i64)> = match after {
            None => {
                sqlx::query_as(
                    "SELECT tenant, partition FROM vellev.partitions
                     ORDER BY tenant, partition LIMIT $1",
                )
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
            Some(after) => {
                sqlx::query_as(
                    "SELECT tenant, partition FROM vellev.partitions
                     WHERE (tenant, partition) > ($1, $2)
                     ORDER BY tenant, partition LIMIT $3",
                )
                .bind(after.tenant.as_str())
                .bind(i64::from(after.partition.0))
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
        };
        rows.into_iter()
            .map(|(tenant, partition)| {
                let stored = || invariant(format!("{tenant}/{partition} is not a partition key"));
                Ok(PartitionKey {
                    tenant: TenantId::new(tenant.as_str()).map_err(|_| stored())?,
                    partition: u32::try_from(partition)
                        .map(Partition)
                        .map_err(|_| stored())?,
                })
            })
            .collect()
    }

    async fn ping(&self) -> Result<()> {
        sqlx::query("SELECT 1").fetch_one(&self.pool).await?;
        Ok(())
    }

    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<()> {
        sqlx::query(
            "INSERT INTO vellev.nodes (node, format, protocol, seen_at)
             VALUES ($1, $2, $3, now())
             ON CONFLICT (node) DO UPDATE
             SET format = EXCLUDED.format, protocol = EXCLUDED.protocol, seen_at = now()",
        )
        .bind(node.as_str())
        .bind(i32::from(versions.format))
        .bind(i32::from(versions.protocol))
        .execute(&self.pool)
        .await?;
        sqlx::query("DELETE FROM vellev.nodes WHERE seen_at < now() - interval '1 day'")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn depart(&self, node: &NodeId) -> Result<()> {
        sqlx::query("DELETE FROM vellev.nodes WHERE node = $1")
            .bind(node.as_str())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn oldest(&self) -> Result<Option<Versions>> {
        let row: (Option<i32>, Option<i32>) =
            sqlx::query_as("SELECT min(format), min(protocol) FROM vellev.nodes")
                .fetch_one(&self.pool)
                .await?;
        match row {
            (Some(format), Some(protocol)) => Ok(Some(Versions {
                format: version(format)?,
                protocol: version(protocol)?,
            })),
            _ => Ok(None),
        }
    }

    async fn format(&self) -> Result<u16> {
        let format: i32 = sqlx::query_scalar("SELECT format FROM vellev.cluster")
            .fetch_one(&self.pool)
            .await?;
        version(format)
    }

    async fn raise_format(&self, format: u16) -> Result<u16> {
        sqlx::query(
            "UPDATE vellev.cluster SET format = $1
             WHERE format < $1
               AND NOT EXISTS (SELECT 1 FROM vellev.nodes WHERE format < $1)",
        )
        .bind(i32::from(format))
        .execute(&self.pool)
        .await?;
        self.format().await
    }

    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>> {
        // The tenant's rows are held while the key is kept, so a removal
        // either comes first, and no key is kept, or waits and takes it.
        let mut transaction = self.pool.begin().await?;
        let registered: Option<i32> = sqlx::query_scalar(
            "SELECT 1 FROM vellev.partitions WHERE tenant = $1 LIMIT 1 FOR SHARE",
        )
        .bind(tenant.as_str())
        .fetch_optional(&mut *transaction)
        .await?;
        if registered.is_none() {
            return Ok(None);
        }
        let created: DateTime<Utc> = sqlx::query_scalar(
            "INSERT INTO vellev.tenant_keys (id, tenant, hash) VALUES ($1, $2, $3)
             RETURNING created_at",
        )
        .bind(id)
        .bind(tenant.as_str())
        .bind(hash.as_slice())
        .fetch_one(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(Some(TenantKey {
            id,
            created,
            hash: *hash,
        }))
    }

    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>> {
        let keys: Vec<(Uuid, DateTime<Utc>, Vec<u8>)> = sqlx::query_as(
            "SELECT id, created_at, hash FROM vellev.tenant_keys WHERE tenant = $1
             ORDER BY created_at, id",
        )
        .bind(tenant.as_str())
        .fetch_all(&self.pool)
        .await?;
        keys.into_iter()
            .map(|(id, created, hash)| {
                let hash = KeyHash::try_from(hash.as_slice()).map_err(|_| {
                    invariant(format!("key {id} has a hash of {} bytes", hash.len()))
                })?;
                Ok(TenantKey { id, created, hash })
            })
            .collect()
    }

    async fn revoke_key(&self, tenant: &TenantId, id: Uuid) -> Result<bool> {
        let done = sqlx::query("DELETE FROM vellev.tenant_keys WHERE tenant = $1 AND id = $2")
            .bind(tenant.as_str())
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>> {
        let tenant: Option<String> =
            sqlx::query_scalar("SELECT tenant FROM vellev.tenant_keys WHERE hash = $1")
                .bind(hash.as_slice())
                .fetch_optional(&self.pool)
                .await?;
        tenant
            .map(|tenant| TenantId::new(tenant).map_err(|error| invariant(error.to_string())))
            .transpose()
    }

    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>> {
        let (at, position) = match point {
            Point::At(at) => (Some(at), None),
            Point::Position(position) => (None, Some(bigint(position.0)?)),
        };
        let found: Option<Noted> = sqlx::query_as(&format!(
            "{REACHABLE}
             SELECT position, head, committed_at FROM reachable
             WHERE ($4::timestamptz IS NULL
                    OR ($4 <= now() AND $4 >= now() - make_interval(secs => $3)
                        AND committed_at <= $4))
               AND ($5::bigint IS NULL OR position = $5)
             ORDER BY position DESC LIMIT 1"
        ))
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(window.as_secs_f64())
        .bind(at)
        .bind(position)
        .fetch_optional(&self.pool)
        .await?;
        found.map(Noted::point).transpose()
    }

    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>> {
        let found: Vec<Noted> = sqlx::query_as(&format!(
            "{REACHABLE}
             (SELECT position, head, committed_at FROM reachable ORDER BY position LIMIT 1)
             UNION ALL
             (SELECT position, head, committed_at FROM reachable ORDER BY position DESC LIMIT 1)"
        ))
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(window.as_secs_f64())
        .fetch_all(&self.pool)
        .await?;
        let mut found = found.into_iter().map(Noted::point);
        Ok(found.next().transpose()?.zip(found.next().transpose()?))
    }

    async fn forget_history(&self, key: &PartitionKey, window: Duration) -> Result<u64> {
        let done = sqlx::query(&format!(
            "{REACHABLE}
             DELETE FROM vellev.commits c
             WHERE c.tenant = $1 AND c.partition = $2
               AND NOT EXISTS (SELECT 1 FROM boundary b
                               WHERE b.incarnation = c.incarnation AND c.position >= b.position)"
        ))
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(window.as_secs_f64())
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "DELETE FROM vellev.restores
             WHERE tenant = $1 AND partition = $2 AND made_at < now() - make_interval(secs => $3)",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(RESTORES_KEPT_FOR.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }

    async fn restored(&self, key: &PartitionKey, request: RequestId) -> Result<Option<Restored>> {
        let recorded: Option<Recorded> = sqlx::query_as(
            "SELECT r.request, r.asked, r.incarnation, r.position, r.from_position, r.from_at
             FROM vellev.restores r JOIN vellev.partitions p USING (tenant, partition, incarnation)
             WHERE r.tenant = $1 AND r.partition = $2 AND r.request = $3
               AND r.made_at >= now() - make_interval(secs => $4)",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(request.0)
        .bind(RESTORES_KEPT_FOR.as_secs_f64())
        .fetch_optional(&self.pool)
        .await?;
        recorded.map(Restored::try_from).transpose()
    }

    async fn forget_removed(
        &self,
        key: &PartitionKey,
        after: Option<Forgotten>,
        limit: usize,
    ) -> Result<Option<Forgotten>> {
        sqlx::query(
            "DELETE FROM vellev.restores r
             WHERE r.tenant = $1 AND r.partition = $2
               AND NOT EXISTS (SELECT 1 FROM vellev.partitions p
                               WHERE p.tenant = $1 AND p.partition = $2
                                 AND p.incarnation = r.incarnation)",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&self.pool)
        .await?;
        let (incarnation, position) = after.map_or((-1, -1), |after| {
            (
                i64::try_from(after.incarnation.0).unwrap_or(i64::MAX),
                i64::try_from(after.position.0).unwrap_or(i64::MAX),
            )
        });
        let last: Option<(i64, i64)> = sqlx::query_as(
            "WITH gone AS (
                 DELETE FROM vellev.commits
                 WHERE ctid = ANY(ARRAY(
                     SELECT kept.ctid FROM vellev.commits kept
                     WHERE kept.tenant = $1 AND kept.partition = $2
                       AND (kept.incarnation, kept.position) > ($3, $4)
                       AND NOT EXISTS (SELECT 1 FROM vellev.partitions p
                                       WHERE p.tenant = $1 AND p.partition = $2
                                         AND p.incarnation = kept.incarnation)
                     ORDER BY kept.incarnation, kept.position
                     LIMIT $5))
                 RETURNING incarnation, position
             )
             SELECT incarnation, position FROM gone
             ORDER BY incarnation DESC, position DESC LIMIT 1",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .bind(incarnation)
        .bind(position)
        .bind(i64::try_from(limit).unwrap_or(i64::MAX))
        .fetch_optional(&self.pool)
        .await?;
        last.map(|(incarnation, position)| {
            let stored = || invariant(format!("{key} has a commit at ({incarnation}, {position})"));
            Ok(Forgotten {
                incarnation: Incarnation(u64::try_from(incarnation).map_err(|_| stored())?),
                position: LogPosition(u64::try_from(position).map_err(|_| stored())?),
            })
        })
        .transpose()
    }

    async fn forget_restores(&self) -> Result<u64> {
        let done = sqlx::query(
            "DELETE FROM vellev.restores WHERE made_at < now() - make_interval(secs => $1)",
        )
        .bind(RESTORES_KEPT_FOR.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }

    async fn purges(&self, due: Duration, lately: Duration) -> Result<Purges> {
        let rows: Vec<(String, bool)> = sqlx::query_as(
            "SELECT tenant, purged_at IS NULL FROM vellev.purges
             WHERE (purged_at IS NULL AND requested_at <= now() - make_interval(secs => $1))
                OR purged_at >= now() - make_interval(secs => $2)
             ORDER BY tenant",
        )
        .bind(due.as_secs_f64())
        .bind(lately.as_secs_f64())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .try_fold(Purges::default(), |mut purges, (tenant, due)| {
                let tenant = TenantId::new(tenant).map_err(|error| invariant(error.to_string()))?;
                match due {
                    true => purges.due.push(tenant),
                    false => purges.lately.push(tenant),
                }
                Ok(purges)
            })
    }

    async fn claim_purge(&self, tenant: &TenantId, hold: Duration) -> Result<Option<Uuid>> {
        let claim = Uuid::new_v4();
        let done = sqlx::query(
            "UPDATE vellev.purges
             SET claim = $2, claimed_until = now() + make_interval(secs => $3)
             WHERE tenant = $1 AND purged_at IS NULL
               AND (claimed_until IS NULL OR claimed_until < now())",
        )
        .bind(tenant.as_str())
        .bind(claim)
        .bind(checked_hold(hold)?.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok((done.rows_affected() > 0).then_some(claim))
    }

    async fn purged(&self, tenant: &TenantId, claim: Uuid, forget_after: Duration) -> Result<()> {
        let mut transaction = self.pool.begin().await?;
        sqlx::query(
            "UPDATE vellev.purges SET purged_at = now(), claim = NULL, claimed_until = NULL
             WHERE tenant = $1 AND claim = $2 AND purged_at IS NULL",
        )
        .bind(tenant.as_str())
        .bind(claim)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "DELETE FROM vellev.purges WHERE purged_at < now() - make_interval(secs => $1)",
        )
        .bind(forget_after.as_secs_f64())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(())
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<Option<Uuid>> {
        let claim = Uuid::new_v4();
        let done = sqlx::query(
            "UPDATE vellev.partitions
             SET backup_claim = $2, backup_claimed_until = now() + make_interval(secs => $4)
             WHERE tenant = $1
               AND NOT EXISTS (
                   SELECT 1 FROM vellev.partitions
                   WHERE tenant = $1
                     AND NOT ((backup_claimed_until IS NULL OR backup_claimed_until < now())
                              AND ($3::float8 IS NULL OR backed_up_at IS NULL
                                   OR backed_up_at <= now() - make_interval(secs => $3))))",
        )
        .bind(tenant.as_str())
        .bind(claim)
        .bind(due.map(|due| due.as_secs_f64()))
        .bind(checked_hold(hold)?.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok((done.rows_affected() > 0).then_some(claim))
    }

    async fn release_backup(&self, tenant: &TenantId, claim: Uuid, made: bool) -> Result<()> {
        sqlx::query(
            "UPDATE vellev.partitions
             SET backup_claim = NULL, backup_claimed_until = NULL,
                 backed_up_at = CASE WHEN $3 THEN now() ELSE backed_up_at END
             WHERE tenant = $1 AND backup_claim = $2",
        )
        .bind(tenant.as_str())
        .bind(claim)
        .bind(made)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn restore_window(&self, tenant: &TenantId) -> Result<Option<Duration>> {
        let secs: Option<i64> = sqlx::query_scalar(
            "SELECT max(restore_window_secs) FROM vellev.partitions WHERE tenant = $1",
        )
        .bind(tenant.as_str())
        .fetch_one(&self.pool)
        .await?;
        secs.map(|secs| {
            u64::try_from(secs)
                .map(Duration::from_secs)
                .map_err(|_| invariant(format!("a restore window of {secs} seconds")))
        })
        .transpose()
    }

    async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool> {
        let secs = window
            .map(|window| bigint(checked_window(window)?.as_secs().max(1)))
            .transpose()?;
        let done =
            sqlx::query("UPDATE vellev.partitions SET restore_window_secs = $2 WHERE tenant = $1")
                .bind(tenant.as_str())
                .bind(secs)
                .execute(&self.pool)
                .await?;
        Ok(done.rows_affected() > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{pending, MIGRATIONS};

    #[test]
    fn migrations_newer_than_known_are_left_alone_and_none_is_skipped() {
        let known = [(1, "a"), (2, "b"), (3, "c")];
        let applied = |done: &[(i64, &str)]| -> Vec<(i64, String)> {
            done.iter().map(|(v, sql)| (*v, sql.to_string())).collect()
        };
        let versions = |pending: Vec<&(i64, &str)>| -> Vec<i64> {
            pending.into_iter().map(|(v, _)| *v).collect()
        };
        assert_eq!(versions(pending(&known, &[]).unwrap()), [1, 2, 3]);
        assert_eq!(
            versions(pending(&known, &applied(&[(1, "a")])).unwrap()),
            [2, 3]
        );
        assert_eq!(
            versions(pending(&known, &applied(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")])).unwrap()),
            [] as [i64; 0],
            "a newer release's migration is left alone"
        );
        assert!(pending(&known, &applied(&[(1, "x")])).is_err(), "changed");
        assert!(
            pending(&known, &applied(&[(1, "a"), (2, "b"), (5, "e")])).is_err(),
            "3 would never be applied"
        );
        assert!(
            pending(&known, &applied(&[(1, "a"), (3, "c")])).is_err(),
            "2 was skipped"
        );
        assert!(
            pending(&[(1, "a"), (3, "c")], &applied(&[(1, "a"), (2, "b")])).is_err(),
            "2 is unknown but older than what is known"
        );
    }

    #[test]
    fn migrations_are_in_strictly_increasing_order() {
        assert!(MIGRATIONS.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(MIGRATIONS.first().map(|(version, _)| *version), Some(1));
    }
}
