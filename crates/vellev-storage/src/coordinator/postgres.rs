use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::{Executor, FromRow, Postgres};
use vellev_core::{NodeId, Partition};

use super::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease, LogPosition,
    ManifestRef, PartitionKey, Result,
};

const MIGRATIONS: &[(i64, &str)] = &[(1, include_str!("../../migrations/0001_partitions.sql"))];

/// Serialises migrations across nodes starting together. Transaction-scoped,
/// so it is released with the transaction and safe behind RDS Proxy.
const MIGRATION_LOCK: i64 = 0x76656c_6c6576;

/// The coordinator on Postgres. Every operation is a single statement, and
/// every expiry comparison happens in the database's clock.
#[derive(Clone)]
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
        Ok(sqlx::query_as(
            "SELECT generation, lease_holder, lease_expires_at, log_position, manifest
             FROM vellev.partitions WHERE tenant = $1 AND partition = $2",
        )
        .bind(key.tenant.as_str())
        .bind(partition(key.partition)?)
        .fetch_optional(&self.pool)
        .await?)
    }

    async fn fenced(&self, lease: &Lease) -> CoordinatorError {
        match self.current(lease).await {
            Err(error) => error,
            Ok(row) => CoordinatorError::Fenced {
                key: lease.key.clone(),
                held: lease.generation,
                current: row.generation,
            },
        }
    }

    async fn current(&self, lease: &Lease) -> Result<Snapshot> {
        let row = self
            .row(&lease.key)
            .await?
            .ok_or_else(|| CoordinatorError::UnknownPartition(lease.key.clone()))?;
        if row.generation == lease.generation {
            Ok(row)
        } else {
            Err(CoordinatorError::Fenced {
                key: lease.key.clone(),
                held: lease.generation,
                current: row.generation,
            })
        }
    }
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

fn bigint(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|e| CoordinatorError::Backend(Box::new(e)))
}

fn partition(partition: Partition) -> Result<i32> {
    i32::try_from(partition.0).map_err(|e| CoordinatorError::Backend(Box::new(e)))
}

impl From<sqlx::Error> for CoordinatorError {
    fn from(error: sqlx::Error) -> Self {
        CoordinatorError::Backend(Box::new(error))
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
        .bind(partition(key.partition)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        let done =
            sqlx::query("DELETE FROM vellev.partitions WHERE tenant = $1 AND partition = $2")
                .bind(key.tenant.as_str())
                .bind(partition(key.partition)?)
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
        loop {
            let taken = sqlx::query_as::<Postgres, Snapshot>(
                "UPDATE vellev.partitions
                 SET generation = generation + 1,
                     lease_holder = $3,
                     lease_expires_at = now() + make_interval(secs => $4)
                 WHERE tenant = $1 AND partition = $2
                   AND (lease_expires_at IS NULL OR lease_expires_at <= now())
                 RETURNING generation, lease_holder, lease_expires_at, log_position, manifest",
            )
            .bind(key.tenant.as_str())
            .bind(partition(key.partition)?)
            .bind(holder.as_str())
            .bind(ttl.as_secs_f64())
            .fetch_optional(&self.pool)
            .await?;
            if let Some(row) = taken {
                let expires_at = row.lease_expires_at.unwrap_or_default();
                return Ok(Acquired::Lease {
                    lease: Lease {
                        key: key.clone(),
                        holder: holder.clone(),
                        generation: row.generation,
                        ttl,
                        expires_at,
                    },
                    committed: row.committed(),
                });
            }
            let row = self
                .row(key)
                .await?
                .ok_or_else(|| CoordinatorError::UnknownPartition(key.clone()))?;
            if let (Some(holder), Some(expires_at)) = (row.lease_holder, row.lease_expires_at) {
                return Ok(Acquired::Held {
                    holder: NodeId::from(holder),
                    expires_at,
                });
            }
        }
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        let expires_at: Option<DateTime<Utc>> = sqlx::query_scalar(
            "UPDATE vellev.partitions
             SET lease_holder = $4, lease_expires_at = now() + make_interval(secs => $5)
             WHERE tenant = $1 AND partition = $2 AND generation = $3
             RETURNING lease_expires_at",
        )
        .bind(lease.key.tenant.as_str())
        .bind(partition(lease.key.partition)?)
        .bind(bigint(lease.generation.0)?)
        .bind(lease.holder.as_str())
        .bind(ttl.as_secs_f64())
        .fetch_optional(&self.pool)
        .await?;
        match expires_at {
            Some(expires_at) => Ok(Lease {
                ttl,
                expires_at,
                ..lease.clone()
            }),
            None => Err(self.fenced(lease).await),
        }
    }

    async fn release(&self, lease: &Lease) -> Result<()> {
        sqlx::query(
            "UPDATE vellev.partitions
             SET generation = generation + 1, lease_holder = NULL, lease_expires_at = NULL
             WHERE tenant = $1 AND partition = $2 AND generation = $3",
        )
        .bind(lease.key.tenant.as_str())
        .bind(partition(lease.key.partition)?)
        .bind(bigint(lease.generation.0)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed> {
        let done = sqlx::query_as::<Postgres, Snapshot>(
            "UPDATE vellev.partitions
             SET log_position = $4, manifest = $5, committed_at = now()
             WHERE tenant = $1 AND partition = $2 AND generation = $3 AND log_position < $4
             RETURNING generation, lease_holder, lease_expires_at, log_position, manifest",
        )
        .bind(lease.key.tenant.as_str())
        .bind(partition(lease.key.partition)?)
        .bind(bigint(lease.generation.0)?)
        .bind(bigint(commit.log_position.0)?)
        .bind(commit.manifest.as_ref().map(|m| m.0.as_str()))
        .fetch_optional(&self.pool)
        .await?;
        match done {
            Some(row) => Ok(row.committed()),
            None => {
                let row = self.current(lease).await?;
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

    async fn ping(&self) -> Result<()> {
        sqlx::query("SELECT 1").fetch_one(&self.pool).await?;
        Ok(())
    }
}
