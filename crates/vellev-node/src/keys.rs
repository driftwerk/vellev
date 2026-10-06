//! Keys that reach one tenant each. A key's secret is shown once, when it
//! is issued; the coordinator keeps only its SHA-256, and a node keeps
//! what it has looked up for a little while, so that a request does not
//! wait on Postgres. A key revoked through one node stops working there at
//! once and everywhere within [`REMEMBERED_FOR`].

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::coordinator::{Coordinator, KeyHash, TenantKey};

use crate::Result;

/// How long a node takes a key it has looked up, found or not, to be as it
/// was.
pub const REMEMBERED_FOR: Duration = Duration::from_secs(10);

/// The most keys a node remembers; past it, it starts again.
const MOST_REMEMBERED: usize = 10_000;

/// What starts every key, so that one is recognisable wherever it leaks.
const PREFIX: &str = "vlv_";

/// A key just issued, with its secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issued {
    pub key: TenantKey,
    pub secret: String,
}

#[derive(Debug, Default)]
pub(crate) struct Keys {
    remembered: Mutex<HashMap<KeyHash, (Option<TenantId>, Instant)>>,
}

fn hashed(secret: &str) -> KeyHash {
    Sha256::digest(secret.as_bytes()).into()
}

impl Keys {
    pub async fn issue(
        &self,
        coordinator: &dyn Coordinator,
        tenant: &TenantId,
    ) -> Result<Option<Issued>> {
        let random = || Uuid::new_v4().simple().to_string();
        let secret = format!("{PREFIX}{}{}", random(), random());
        let key = coordinator
            .issue_key(tenant, Uuid::new_v4(), &hashed(&secret))
            .await?;
        Ok(key.map(|key| Issued { key, secret }))
    }

    /// The tenant `secret` reaches, if it is a key.
    pub async fn tenant(
        &self,
        coordinator: &dyn Coordinator,
        secret: &str,
    ) -> Result<Option<TenantId>> {
        if !secret.starts_with(PREFIX) {
            return Ok(None);
        }
        let hash = hashed(secret);
        let remembered = self.remembered().get(&hash).cloned();
        if let Some((tenant, _)) = remembered.filter(|(_, at)| at.elapsed() < REMEMBERED_FOR) {
            return Ok(tenant);
        }
        let tenant = coordinator.key_tenant(&hash).await?;
        let mut remembered = self.remembered();
        if remembered.len() >= MOST_REMEMBERED {
            remembered.clear();
        }
        remembered.insert(hash, (tenant.clone(), Instant::now()));
        Ok(tenant)
    }

    /// Forget what was looked up for `tenant`, whose keys have changed.
    pub fn forget(&self, tenant: &TenantId) {
        self.remembered()
            .retain(|_, (reaches, _)| reaches.as_ref() != Some(tenant));
    }

    fn remembered(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<KeyHash, (Option<TenantId>, Instant)>> {
        self.remembered.lock().expect("keys lock")
    }
}
