//! Keys that reach one tenant each. A key's secret is shown once, when it
//! is issued; the coordinator keeps only its SHA-256, and a node keeps
//! what it has looked up for a little while, so that a request does not
//! wait on Postgres. A key revoked, or a tenant dropped, through a node is
//! forgotten there at once, and by every node within [`REMEMBERED_FOR`]:
//! until then another node may still let the key through, to its tenant or
//! to one made again under the same id.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::coordinator::{Coordinator, KeyHash, TenantKey};

use crate::{NodeError, Result};

/// How long a node takes a key it has looked up, found or not, to be as it
/// was.
pub const REMEMBERED_FOR: Duration = Duration::from_secs(10);

/// The most keys a node remembers as found, and as not found. Each is kept
/// apart, so that a flood of made-up keys never pushes out real ones.
const MOST_FOUND: usize = 10_000;
const MOST_UNKNOWN: usize = 1_000;

/// How many lookups of keys not remembered may wait on the coordinator at
/// once; past it a request is turned away, to be tried again, rather than
/// queued.
const LOOKUPS: usize = 16;

/// What starts every key, so that one is recognisable wherever it leaks.
const PREFIX: &str = "vlv_";

/// A key just issued, with its secret.
#[derive(Clone, PartialEq, Eq)]
pub struct Issued {
    pub key: TenantKey,
    pub secret: String,
}

impl std::fmt::Debug for Issued {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Issued")
            .field("key", &self.key)
            .field("secret", &"…")
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct Keys {
    remembered: Mutex<Remembered>,
    lookups: Semaphore,
}

impl Default for Keys {
    fn default() -> Self {
        Self {
            remembered: Mutex::default(),
            lookups: Semaphore::new(LOOKUPS),
        }
    }
}

#[derive(Debug, Default)]
struct Remembered {
    found: HashMap<KeyHash, (TenantId, Instant)>,
    unknown: HashMap<KeyHash, Instant>,
    /// Moved on whenever keys are forgotten, so that a lookup made before
    /// is not remembered after.
    generation: u64,
}

impl Remembered {
    fn get(&self, hash: &KeyHash) -> Option<Option<TenantId>> {
        let fresh = |at: &Instant| at.elapsed() < REMEMBERED_FOR;
        if let Some((tenant, _)) = self.found.get(hash).filter(|(_, at)| fresh(at)) {
            return Some(Some(tenant.clone()));
        }
        self.unknown.get(hash).filter(|at| fresh(at)).map(|_| None)
    }

    fn insert(&mut self, hash: KeyHash, tenant: Option<TenantId>) {
        let now = Instant::now();
        match tenant {
            Some(tenant) => {
                bounded(&mut self.found, MOST_FOUND, |(_, at)| *at);
                self.found.insert(hash, (tenant, now));
            }
            None => {
                bounded(&mut self.unknown, MOST_UNKNOWN, |at| *at);
                self.unknown.insert(hash, now);
            }
        }
    }
}

/// Make room in `map` for one more: drop what has expired, and if it is
/// still full, the oldest half.
fn bounded<V>(map: &mut HashMap<KeyHash, V>, most: usize, at: impl Fn(&V) -> Instant) {
    if map.len() < most {
        return;
    }
    map.retain(|_, value| at(value).elapsed() < REMEMBERED_FOR);
    if map.len() >= most {
        let mut ages: Vec<Instant> = map.values().map(&at).collect();
        ages.sort_unstable();
        let cut = ages[ages.len() / 2];
        map.retain(|_, value| at(value) > cut);
    }
}

fn hashed(secret: &str) -> KeyHash {
    Sha256::digest(secret.as_bytes()).into()
}

/// Whether `secret` has the form every issued key has.
fn well_formed(secret: &str) -> bool {
    secret.strip_prefix(PREFIX).is_some_and(|rest| {
        rest.len() == 64 && rest.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
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
        if !well_formed(secret) {
            return Ok(None);
        }
        let hash = hashed(secret);
        let generation = {
            let remembered = self.remembered();
            if let Some(tenant) = remembered.get(&hash) {
                return Ok(tenant);
            }
            remembered.generation
        };
        let _looking = self.lookups.try_acquire().map_err(|_| NodeError::Busy)?;
        let tenant = coordinator.key_tenant(&hash).await?;
        let mut remembered = self.remembered();
        if remembered.generation == generation {
            remembered.insert(hash, tenant.clone());
        }
        Ok(tenant)
    }

    /// Forget what was looked up for `tenant`, whose keys have changed, and
    /// whatever is being looked up now.
    pub fn forget(&self, tenant: &TenantId) {
        let mut remembered = self.remembered();
        remembered.generation += 1;
        remembered.found.retain(|_, (reaches, _)| reaches != tenant);
    }

    fn remembered(&self) -> std::sync::MutexGuard<'_, Remembered> {
        self.remembered.lock().expect("keys lock")
    }
}

#[cfg(test)]
mod tests {
    use vellev_core::Partition;
    use vellev_storage::coordinator::{MemoryCoordinator, PartitionKey};

    use super::*;

    async fn tenant(coordinator: &MemoryCoordinator, name: &str) -> TenantId {
        let tenant = TenantId::new(name).unwrap();
        let key = PartitionKey {
            tenant: tenant.clone(),
            partition: Partition::DEFAULT,
        };
        coordinator.register(&key).await.unwrap();
        tenant
    }

    #[test]
    fn only_a_key_of_the_issued_form_is_looked_up() {
        let random = || Uuid::new_v4().simple().to_string();
        assert!(well_formed(&format!("{PREFIX}{}{}", random(), random())));
        for not in [
            "vlv_",
            "dev-key",
            &format!("{PREFIX}{}", "0".repeat(63)),
            &format!("{PREFIX}{}", "A".repeat(64)),
            &format!("{PREFIX}{}", "0".repeat(65)),
        ] {
            assert!(!well_formed(not), "{not}");
        }
    }

    #[tokio::test]
    async fn a_flood_of_made_up_keys_never_forgets_a_real_one() {
        let coordinator = MemoryCoordinator::default();
        let acme = tenant(&coordinator, "acme").await;
        let keys = Keys::default();
        let issued = keys.issue(&coordinator, &acme).await.unwrap().unwrap();
        assert_eq!(
            keys.tenant(&coordinator, &issued.secret).await.unwrap(),
            Some(acme.clone())
        );
        for _ in 0..MOST_UNKNOWN * 3 {
            let made_up = format!(
                "{PREFIX}{}{}",
                Uuid::new_v4().simple(),
                Uuid::new_v4().simple()
            );
            assert_eq!(keys.tenant(&coordinator, &made_up).await.unwrap(), None);
        }
        let remembered = keys.remembered();
        assert!(remembered.unknown.len() <= MOST_UNKNOWN);
        assert_eq!(remembered.get(&hashed(&issued.secret)), Some(Some(acme)));
    }
}
