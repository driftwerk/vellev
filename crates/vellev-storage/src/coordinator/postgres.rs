use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgArguments, PgPool, PgPoolOptions};
use sqlx::query::QueryAs;
use sqlx::{Executor, FromRow, Postgres};
use vellev_core::{NodeId, Partition, TenantId};

use super::{
    checked_ttl, Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease,
    LogPosition, ManifestRef, PartitionKey, Result,
};

const MIGRATIONS: &[(i64, &str)] = &[(1, include_str!("../../migrations/0001_partitions.sql"))];

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
            "SELECT generation, lease_holder, lease_expires_at, log_position, manifest
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
    for (version, sql) in &applied {
        match MIGRATIONS.iter().find(|(v, _)| v == version) {
            Some((_, expected)) if expected == sql => {}
            _ => {
                return Err(invariant(format!(
                    "migration {version} differs from the one applied; recreate the database"
                )))
            }
        }
    }
    let latest = applied.last().map_or(0, |(version, _)| *version);
    for (version, sql) in MIGRATIONS.iter().filter(|(version, _)| *version > latest) {
        tx.execute(sqlx::raw_sql(sql)).await?;
        sqlx::query("INSERT INTO vellev.schema_version (version, sql) VALUES ($1, $2)")
            .bind(version)
            .bind(sql)
            .execute(&mut *tx)
            .await?;
    }
    Ok(tx.commit().await?)
}

#[derive(FromRow)]
struct Snapshot {
    #[sqlx(try_from = "i64")]
    generation: Generation,
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
                 RETURNING generation, lease_holder, lease_expires_at, log_position, manifest",
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
             RETURNING generation, lease_holder, lease_expires_at, log_position, manifest",
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
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT tenant, partition FROM vellev.partitions
             WHERE $1::text IS NULL OR (tenant COLLATE \"C\", partition) > ($1 COLLATE \"C\", $2)
             ORDER BY tenant COLLATE \"C\", partition
             LIMIT $3",
        )
        .bind(after.map(|key| key.tenant.as_str()))
        .bind(after.map_or(0, |key| i64::from(key.partition.0)))
        .bind(i64::try_from(limit).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await?;
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
}

#[cfg(test)]
mod tests {
    use super::MIGRATIONS;

    #[test]
    fn migrations_are_in_strictly_increasing_order() {
        assert!(MIGRATIONS.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(MIGRATIONS.first().map(|(version, _)| *version), Some(1));
    }
}
