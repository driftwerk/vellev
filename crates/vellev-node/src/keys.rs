//! Keys that reach one tenant each. A key's secret is shown once, when it
//! is issued; the coordinator keeps only its SHA-256, and a node keeps
//! what it has looked up for a little while, so that a request does not
//! wait on Postgres. A key revoked, or a tenant dropped, through a node is
//! forgotten there at once, and by every node within [`REMEMBERED_FOR`]:
//! until then another node may still let the key through, to its tenant or
//! to one made again under the same id.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use tokio::time::Instant;

use futures::future::BoxFuture;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::coordinator::{self, Coordinator, KeyHash, TenantKey};

use crate::{NodeError, Result};

/// How long a node takes a key it has looked up, found or not, to be as it
/// was.
pub const REMEMBERED_FOR: Duration = Duration::from_secs(10);

/// How long before a key in use is due to be looked up again that a node
/// looks it up, answering from what it remembers meanwhile.
pub const REFRESHED_BEFORE: Duration = Duration::from_secs(2);

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

/// Where a node looks up the tenant a key's hash reaches: the coordinator,
/// or in a test, a stand-in for it. The lookup is a future of its own,
/// owning what it needs, so that it can be run apart from whoever asked.
pub(crate) trait KeyLookup: Send + Sync + 'static {
    fn key_tenant(
        &self,
        hash: KeyHash,
    ) -> BoxFuture<'static, coordinator::Result<Option<TenantId>>>;
}

impl KeyLookup for std::sync::Arc<dyn Coordinator> {
    fn key_tenant(
        &self,
        hash: KeyHash,
    ) -> BoxFuture<'static, coordinator::Result<Option<TenantId>>> {
        let coordinator = self.clone();
        Box::pin(async move { Coordinator::key_tenant(coordinator.as_ref(), &hash).await })
    }
}

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
    counted: Counted,
}

/// How the keys presented to a node have been answered, counted from
/// its start: from what it remembered, by asking the coordinator, or
/// turned away for want of a lookup.
#[derive(Debug, Default)]
struct Counted {
    remembered: AtomicU64,
    looked_up: AtomicU64,
    busy: AtomicU64,
}

/// How many keys a node has answered from what it remembered, looked up
/// with the coordinator, and turned away as busy, since it started. A key
/// not of the issued form is none of these: it is refused unlooked at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyLookups {
    pub remembered: u64,
    pub looked_up: u64,
    pub busy: u64,
}

impl Default for Keys {
    fn default() -> Self {
        Self {
            remembered: Mutex::default(),
            lookups: Semaphore::new(LOOKUPS),
            counted: Counted::default(),
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
    pub async fn tenant(&self, lookup: &impl KeyLookup, secret: &str) -> Result<Option<TenantId>> {
        if !well_formed(secret) {
            return Ok(None);
        }
        let hash = hashed(secret);
        let generation = {
            let remembered = self.remembered();
            if let Some(tenant) = remembered.get(&hash) {
                self.counted.remembered.fetch_add(1, Ordering::Relaxed);
                return Ok(tenant);
            }
            remembered.generation
        };
        let Ok(_looking) = self.lookups.try_acquire() else {
            self.counted.busy.fetch_add(1, Ordering::Relaxed);
            return Err(NodeError::Busy);
        };
        self.counted.looked_up.fetch_add(1, Ordering::Relaxed);
        let tenant = lookup.key_tenant(hash).await?;
        let mut remembered = self.remembered();
        if remembered.generation == generation {
            remembered.insert(hash, tenant.clone());
        }
        Ok(tenant)
    }

    /// How the keys presented so far have been answered.
    pub fn lookups(&self) -> KeyLookups {
        let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
        KeyLookups {
            remembered: read(&self.counted.remembered),
            looked_up: read(&self.counted.looked_up),
            busy: read(&self.counted.busy),
        }
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

    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use super::*;

    fn memory() -> Arc<dyn Coordinator> {
        Arc::new(MemoryCoordinator::default())
    }

    async fn tenant(coordinator: &Arc<dyn Coordinator>, name: &str) -> TenantId {
        let tenant = TenantId::new(name).unwrap();
        let key = PartitionKey {
            tenant: tenant.clone(),
            partition: Partition::DEFAULT,
        };
        coordinator.register(&key).await.unwrap();
        tenant
    }

    #[tokio::test]
    async fn only_a_key_of_the_issued_form_is_looked_up() {
        let (coordinator, keys) = (memory(), Keys::default());
        assert_eq!(keys.tenant(&coordinator, "dev-key").await.unwrap(), None);
        assert!(keys.remembered().unknown.is_empty(), "not even remembered");
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
        let coordinator = memory();
        let acme = tenant(&coordinator, "acme").await;
        let keys = Keys::default();
        let issued = keys
            .issue(coordinator.as_ref(), &acme)
            .await
            .unwrap()
            .unwrap();
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

    /// A stand-in for the coordinator: it counts its lookups, holds each
    /// until let go, and answers as it is told to then.
    #[derive(Clone)]
    struct Stand {
        calls: Arc<AtomicUsize>,
        gate: Arc<Semaphore>,
        answer: Arc<Mutex<Option<Option<TenantId>>>>,
    }

    impl Stand {
        /// Answering `tenant`, or failing while the answer is `None`.
        fn new(tenant: Option<TenantId>) -> Self {
            Self {
                calls: Arc::default(),
                gate: Arc::new(Semaphore::new(0)),
                answer: Arc::new(Mutex::new(Some(tenant))),
            }
        }

        fn open(self) -> Self {
            self.gate.add_permits(Semaphore::MAX_PERMITS / 2);
            self
        }

        fn release(&self) {
            self.gate.add_permits(Semaphore::MAX_PERMITS / 2);
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn answering(&self, answer: Option<Option<TenantId>>) {
            *self.answer.lock().unwrap() = answer;
        }

        /// Once the stand-in has been asked `times` times; failing the test,
        /// rather than hanging it, if that is not soon.
        async fn called(&self, times: usize) {
            for _ in 0..1_000 {
                if self.calls() >= times {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            panic!("asked {} times, not {times}", self.calls());
        }
    }

    impl KeyLookup for Stand {
        fn key_tenant(
            &self,
            _: KeyHash,
        ) -> BoxFuture<'static, coordinator::Result<Option<TenantId>>> {
            let stand = self.clone();
            Box::pin(async move {
                stand.calls.fetch_add(1, Ordering::SeqCst);
                let _go = stand.gate.acquire().await.unwrap();
                let answer = stand.answer.lock().unwrap().clone();
                answer.ok_or_else(|| {
                    vellev_storage::coordinator::CoordinatorError::Backend("down".into())
                })
            })
        }
    }

    fn secret() -> String {
        format!(
            "{PREFIX}{}{}",
            Uuid::new_v4().simple(),
            Uuid::new_v4().simple()
        )
    }

    fn acme() -> TenantId {
        TenantId::new("acme").unwrap()
    }

    /// `askers` asks for `secret` at once, each on a task of its own.
    fn asking(
        keys: &Arc<Keys>,
        stand: &Stand,
        secret: &str,
        askers: usize,
    ) -> Vec<tokio::task::JoinHandle<Result<Option<TenantId>>>> {
        (0..askers)
            .map(|_| {
                let (keys, stand, secret) = (keys.clone(), stand.clone(), secret.to_string());
                tokio::spawn(async move { keys.tenant(&stand, &secret).await })
            })
            .collect()
    }

    /// T3: however many ask for a key not remembered at once, it is looked
    /// up once, and none of them is turned away as busy.
    #[tokio::test]
    async fn one_lookup_answers_every_concurrent_ask_for_a_key() {
        let (keys, stand, secret) = (
            Arc::new(Keys::default()),
            Stand::new(Some(acme())),
            secret(),
        );
        let asked = asking(&keys, &stand, &secret, 64);
        stand.called(1).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        stand.release();
        for answer in asked {
            assert_eq!(answer.await.unwrap().unwrap(), Some(acme()));
        }
        assert_eq!(stand.calls(), 1, "one lookup for all 64");
        let looked = keys.lookups();
        assert_eq!((looked.looked_up, looked.busy), (1, 0));
    }

    /// T11: a lookup goes on when whoever began it stops waiting, and those
    /// who joined it are answered by it.
    #[tokio::test]
    async fn a_lookup_goes_on_when_whoever_began_it_stops_waiting() {
        let (keys, stand, secret) = (
            Arc::new(Keys::default()),
            Stand::new(Some(acme())),
            secret(),
        );
        let first = asking(&keys, &stand, &secret, 1).remove(0);
        stand.called(1).await;
        let joined = asking(&keys, &stand, &secret, 1).remove(0);
        tokio::time::sleep(Duration::from_millis(50)).await;
        first.abort();
        stand.release();
        let answer = tokio::time::timeout(Duration::from_secs(5), joined)
            .await
            .expect("answered, not left waiting");
        assert_eq!(answer.unwrap().unwrap(), Some(acme()));
        assert_eq!(stand.calls(), 1);
    }

    /// A failed lookup is told to everyone who asked, once, and is not
    /// remembered: the next ask looks the key up again.
    #[tokio::test]
    async fn a_failed_lookup_is_told_to_each_asker_and_not_remembered() {
        let (keys, stand, secret) = (Arc::new(Keys::default()), Stand::new(None), secret());
        stand.answering(None);
        let asked = asking(&keys, &stand, &secret, 8);
        stand.called(1).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        stand.release();
        for answer in asked {
            assert!(
                answer.await.unwrap().is_err(),
                "the failure, to every asker"
            );
        }
        assert_eq!(stand.calls(), 1);
        stand.answering(Some(Some(acme())));
        assert_eq!(keys.tenant(&stand, &secret).await.unwrap(), Some(acme()));
        assert_eq!(stand.calls(), 2, "the failure was not remembered");
    }

    /// A key nearly due to be looked up again is looked up while what was
    /// remembered of it still answers: no ask waits on the coordinator for
    /// a key in use, and a revoked key is found out no later than it was.
    #[tokio::test(start_paused = true)]
    async fn a_key_nearly_due_is_looked_up_again_while_it_still_answers() {
        let (keys, stand, secret) = (
            Arc::new(Keys::default()),
            Stand::new(Some(acme())).open(),
            secret(),
        );
        assert_eq!(keys.tenant(&stand, &secret).await.unwrap(), Some(acme()));
        tokio::time::advance(REMEMBERED_FOR - REFRESHED_BEFORE + Duration::from_millis(1)).await;
        stand.answering(Some(None));
        assert_eq!(
            keys.tenant(&stand, &secret).await.unwrap(),
            Some(acme()),
            "answered from what is remembered"
        );
        stand.called(2).await;
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            keys.tenant(&stand, &secret).await.unwrap(),
            None,
            "the fresh answer"
        );
        assert_eq!(stand.calls(), 2);
    }
}
