use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgArguments, PgPool, PgPoolOptions};
use sqlx::query::QueryAs;
use sqlx::{Executor, FromRow, Postgres};
use vellev_core::NodeId;

use super::{
    bigint, checked_ttl, Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation,
    Lease, LogPosition, ManifestRef, PartitionKey, Result,
};

const MIGRATIONS: &[(i64, &str)] = &[(1, include_str!("../../migrations/0001_partitions.sql"))];

/// Serialises migrations across nodes starting together. Transaction-scoped,
/// so it is released with the transaction and never pins an RDS Proxy session.
const MIGRATION_LOCK: i64 = 0x76656c_6c6576;

const COLUMNS: &str = "generation, lease_holder, lease_expires_at, log_position, manifest";

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
        Ok(keyed(
            &format!(
                "SELECT {COLUMNS} FROM vellev.partitions WHERE tenant = $1 AND partition = $2"
            ),
            key,
        )
        .fetch_optional(&self.pool)
        .await?)
    }

    /// Why a fenced statement on `lease` did nothing. `on_current` explains the
    /// case where the generation still matches, so the predicate failed for
    /// another reason.
    async fn refusal(
        &self,
        lease: &Lease,
        on_current: impl FnOnce(&Snapshot) -> CoordinatorError + Send,
    ) -> CoordinatorError {
        match self.row(&lease.key).await {
            Err(error) => error,
            Ok(None) => CoordinatorError::UnknownPartition(lease.key.clone()),
            Ok(Some(row)) if row.generation != lease.generation => lease.fenced_by(row.generation),
            Ok(Some(row)) => on_current(&row),
        }
    }
}

fn keyed<'q>(sql: &'q str, key: &'q PartitionKey) -> QueryAs<'q, Postgres, Snapshot, PgArguments> {
    sqlx::query_as(sql)
        .bind(key.tenant.as_str())
        .bind(i64::from(key.partition.0))
}

fn changed_underneath(_: &Snapshot) -> CoordinatorError {
    CoordinatorError::InvalidArgument("partition changed between two statements".into())
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
             applied_at timestamptz NOT NULL DEFAULT now()
         )",
    )
    .await?;
    let applied: i64 =
        sqlx::query_scalar("SELECT coalesce(max(version), 0) FROM vellev.schema_version")
            .fetch_one(&mut *tx)
            .await?;
    for (version, sql) in MIGRATIONS.iter().filter(|(version, _)| *version > applied) {
        tx.execute(sqlx::raw_sql(sql)).await?;
        sqlx::query("INSERT INTO vellev.schema_version (version) VALUES ($1)")
            .bind(version)
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
            "INSERT INTO vellev.partitions (tenant, partition) VALUES ($1, $2)
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
        let sql = format!(
            "UPDATE vellev.partitions
             SET generation = nextval('vellev.generations'),
                 lease_holder = $3,
                 lease_expires_at = now() + make_interval(secs => $4)
             WHERE tenant = $1 AND partition = $2
               AND (lease_expires_at IS NULL OR lease_expires_at <= now() OR lease_holder = $3)
             RETURNING {COLUMNS}"
        );
        for _ in 0..ACQUIRE_ATTEMPTS {
            let requested_at = Instant::now();
            let taken = keyed(&sql, key)
                .bind(holder.as_str())
                .bind(ttl.as_secs_f64())
                .fetch_optional(&self.pool)
                .await?;
            if let Some(row) = taken {
                let (_, expires_at) = row.lease().ok_or_else(|| {
                    CoordinatorError::InvalidArgument("acquired lease has no expiry".into())
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
        Err(CoordinatorError::InvalidArgument(format!(
            "lease on {key} kept changing across {ACQUIRE_ATTEMPTS} attempts"
        )))
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        let ttl = checked_ttl(ttl)?;
        let requested_at = Instant::now();
        let expires_at: Option<DateTime<Utc>> = sqlx::query_scalar(
            "UPDATE vellev.partitions
             SET lease_holder = $4, lease_expires_at = now() + make_interval(secs => $5)
             WHERE tenant = $1 AND partition = $2 AND generation = $3
             RETURNING lease_expires_at",
        )
        .bind(lease.key.tenant.as_str())
        .bind(i64::from(lease.key.partition.0))
        .bind(bigint(lease.generation.0)?)
        .bind(lease.holder.as_str())
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
            None => Err(self.refusal(lease, changed_underneath).await),
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
        let sql = format!(
            "UPDATE vellev.partitions
             SET log_position = $4, manifest = $5
             WHERE tenant = $1 AND partition = $2 AND generation = $3
               AND (log_position < $4 OR (log_position = $4 AND manifest IS NOT DISTINCT FROM $5))
             RETURNING {COLUMNS}"
        );
        let done = keyed(&sql, &lease.key)
            .bind(bigint(lease.generation.0)?)
            .bind(bigint(commit.log_position.0)?)
            .bind(commit.manifest.as_ref().map(|m| m.0.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        match done {
            Some(row) => Ok(row.committed()),
            None => Err(self
                .refusal(lease, |row| CoordinatorError::Regressed {
                    key: lease.key.clone(),
                    committed: row.log_position,
                    attempted: commit.log_position,
                })
                .await),
        }
    }

    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>> {
        Ok(self.row(key).await?.as_ref().map(Snapshot::committed))
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
