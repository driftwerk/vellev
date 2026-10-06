use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgArguments, PgPool, PgPoolOptions};
use sqlx::query::QueryAs;
use sqlx::{Executor, FromRow, Postgres};
use vellev_core::{NodeId, Partition, TenantId};

use super::{
    checked_ttl, Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation,
    Incarnation, Lease, LogPosition, ManifestRef, PartitionKey, Result, Versions,
};

const MIGRATIONS: &[(i64, &str)] = &[
    (1, include_str!("../../migrations/0001_partitions.sql")),
    (2, include_str!("../../migrations/0002_tenant_order.sql")),
    (3, include_str!("../../migrations/0003_incarnation.sql")),
    (4, include_str!("../../migrations/0004_nodes.sql")),
];

/// Serialises migrations across nodes starting together. Transaction-scoped,
/// so it is released with the transaction and never pins an RDS Proxy session.
const MIGRATION_LOCK: i64 = 0x76656c_6c6576;

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
    /// Connect and bring the `vellev` schema up to date.
    pub async fn connect(url: &str, max_connections: u32) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
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
        sqlx::query(
            "INSERT INTO vellev.partitions (tenant, partition, generation)
             VALUES ($1, $2, nextval('vellev.generations'))
             ON CONFLICT DO NOTHING",
        )
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        let done =
            sqlx::query("DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = $2")
                .bind(key.tenant.as_str())
                .bind(i64::from(key.partition.0))
                .execute(&self.pool)
                .await?;
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
            "UPDATE vellev.partitions
             SET log_position = $4, manifest = $5
             WHERE tenant = $1 AND partition = $2 AND generation = $3
               AND (log_position < $4 OR (log_position = $4 AND manifest IS NOT DISTINCT FROM $5))
             RETURNING generation, incarnation, lease_holder, lease_expires_at, log_position, manifest",
            &lease.key,
        )
        .bind(bigint(lease.generation.0)?)
        .bind(bigint(commit.log_position.0)?)
        .bind(commit.manifest.as_ref().map(|m| m.0.as_str()))
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

    async fn oldest(&self, within: Duration) -> Result<Option<Versions>> {
        let row: (Option<i32>, Option<i32>) = sqlx::query_as(
            "SELECT min(format), min(protocol) FROM vellev.nodes
             WHERE seen_at > now() - make_interval(secs => $1)",
        )
        .bind(within.as_secs_f64())
        .fetch_one(&self.pool)
        .await?;
        let version =
            |value: i32| u16::try_from(value).map_err(|error| invariant(error.to_string()));
        match row {
            (Some(format), Some(protocol)) => Ok(Some(Versions {
                format: version(format)?,
                protocol: version(protocol)?,
            })),
            _ => Ok(None),
        }
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
