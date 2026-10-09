use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;
use vellev_core::{NodeId, RequestId, TenantId};

use super::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Forgotten, KeyHash, Lease,
    PartitionKey, Point, Purges, RestorePoint, Restored, Result, TenantKey, Versions,
};

/// A coordinator whose every call is given up once it has gone unanswered
/// for `within`, as a backend failure: its outcome is unknown, as when a
/// connection drops. A connection that stops answering without closing,
/// which nothing else notices for many minutes, then holds up nothing.
#[derive(Debug)]
pub struct Bounded {
    inner: Arc<dyn Coordinator>,
    within: Duration,
}

impl Bounded {
    pub fn new(inner: Arc<dyn Coordinator>, within: Duration) -> Self {
        Self { inner, within }
    }

    async fn bounded<T>(&self, call: impl Future<Output = Result<T>>) -> Result<T> {
        tokio::time::timeout(self.within, call)
            .await
            .unwrap_or_else(|_| {
                let reason = format!("the coordinator did not answer within {:?}", self.within);
                Err(CoordinatorError::backend(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    reason,
                )))
            })
    }
}

#[async_trait]
impl Coordinator for Bounded {
    async fn register(&self, key: &PartitionKey) -> Result<()> {
        self.bounded(self.inner.register(key)).await
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        self.bounded(self.inner.remove(key)).await
    }

    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired> {
        self.bounded(self.inner.acquire(key, holder, ttl)).await
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        self.bounded(self.inner.renew(lease, ttl)).await
    }

    async fn release(&self, lease: &Lease) -> Result<()> {
        self.bounded(self.inner.release(lease)).await
    }

    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed> {
        self.bounded(self.inner.commit(lease, commit)).await
    }

    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>> {
        self.bounded(self.inner.committed(key)).await
    }

    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>> {
        self.bounded(self.inner.partitions(after, limit)).await
    }

    async fn ping(&self) -> Result<()> {
        self.bounded(self.inner.ping()).await
    }

    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<()> {
        self.bounded(self.inner.announce(node, versions)).await
    }

    async fn depart(&self, node: &NodeId) -> Result<()> {
        self.bounded(self.inner.depart(node)).await
    }

    async fn oldest(&self) -> Result<Option<Versions>> {
        self.bounded(self.inner.oldest()).await
    }

    async fn format(&self) -> Result<u16> {
        self.bounded(self.inner.format()).await
    }

    async fn raise_format(&self, format: u16) -> Result<u16> {
        self.bounded(self.inner.raise_format(format)).await
    }

    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>> {
        self.bounded(self.inner.issue_key(tenant, id, hash)).await
    }

    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>> {
        self.bounded(self.inner.keys(tenant)).await
    }

    async fn revoke_key(&self, tenant: &TenantId, id: Uuid) -> Result<bool> {
        self.bounded(self.inner.revoke_key(tenant, id)).await
    }

    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>> {
        self.bounded(self.inner.key_tenant(hash)).await
    }

    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>> {
        self.bounded(self.inner.restore_point(key, point, window))
            .await
    }

    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>> {
        self.bounded(self.inner.restore_points(key, window)).await
    }

    async fn forget_history(&self, key: &PartitionKey, window: Duration) -> Result<u64> {
        self.bounded(self.inner.forget_history(key, window)).await
    }

    async fn restored(&self, key: &PartitionKey, request: RequestId) -> Result<Option<Restored>> {
        self.bounded(self.inner.restored(key, request)).await
    }

    async fn forget_removed(
        &self,
        key: &PartitionKey,
        after: Option<Forgotten>,
        limit: usize,
    ) -> Result<Option<Forgotten>> {
        self.bounded(self.inner.forget_removed(key, after, limit))
            .await
    }

    async fn forget_restores(&self) -> Result<u64> {
        self.bounded(self.inner.forget_restores()).await
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<Option<Uuid>> {
        self.bounded(self.inner.claim_backup(tenant, due, hold))
            .await
    }

    async fn release_backup(&self, tenant: &TenantId, claim: Uuid, made: bool) -> Result<()> {
        self.bounded(self.inner.release_backup(tenant, claim, made))
            .await
    }

    async fn purges(&self, due: Duration, lately: Duration) -> Result<Purges> {
        self.bounded(self.inner.purges(due, lately)).await
    }

    async fn claim_purge(&self, tenant: &TenantId, hold: Duration) -> Result<Option<Uuid>> {
        self.bounded(self.inner.claim_purge(tenant, hold)).await
    }

    async fn purged(&self, tenant: &TenantId, claim: Uuid, forget_after: Duration) -> Result<()> {
        self.bounded(self.inner.purged(tenant, claim, forget_after))
            .await
    }

    async fn restore_window(&self, tenant: &TenantId) -> Result<Option<Duration>> {
        self.bounded(self.inner.restore_window(tenant)).await
    }

    async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool> {
        self.bounded(self.inner.set_restore_window(tenant, window))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::{MemoryCoordinator, SystemClock};

    #[tokio::test(start_paused = true)]
    async fn a_call_left_unanswered_is_given_up_as_a_backend_failure() {
        let memory = Arc::new(MemoryCoordinator::new(Arc::new(SystemClock)));
        let key = PartitionKey {
            tenant: TenantId::new("acme").unwrap(),
            partition: vellev_core::Partition::DEFAULT,
        };
        let bounded = Bounded::new(memory, Duration::from_secs(5));
        bounded.register(&key).await.unwrap();
        assert!(bounded.committed(&key).await.unwrap().is_some());

        let started = tokio::time::Instant::now();
        let silent = bounded.bounded(std::future::pending::<Result<()>>());
        let given_up = silent.await.unwrap_err();
        assert!(given_up.is_retryable(), "{given_up}");
        assert_eq!(started.elapsed(), Duration::from_secs(5));
    }
}
