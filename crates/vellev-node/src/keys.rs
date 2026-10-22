//! Keys that reach one tenant each. A key's secret is shown once, when it
//! is issued; the coordinator keeps only its SHA-256, and a node keeps
//! what it has looked up for a little while, so that a request does not
//! wait on Postgres. A key revoked, or a tenant dropped, through a node is
//! forgotten there at once, and by every node within [`REMEMBERED_FOR`]:
//! until then another node may still let the key through, to its tenant or
//! to one made again under the same id.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use futures::future::{BoxFuture, FutureExt, Shared};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::coordinator::{self, Coordinator, KeyHash, TenantKey};
use vellev_storage::StorageError;

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

impl KeyLookup for Arc<dyn Coordinator> {
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

/// What a node remembers of keys, and the lookups it has under way. A key
/// is looked up once however many ask for it at once: the first to find
/// it neither remembered nor being looked up starts a lookup on a task of
/// its own, which goes on whoever stops waiting for it, and every other
/// ask joins it. A key in use is looked up again a little before it is
/// due ([`REFRESHED_BEFORE`]), its asks answered from what is remembered
/// meanwhile; that never keeps an answer past [`REMEMBERED_FOR`] from the
/// lookup that gave it.
#[derive(Debug, Clone)]
pub(crate) struct Keys(Arc<Kept>);

#[derive(Debug)]
struct Kept {
    remembered: Mutex<Remembered>,
    lookups: Arc<Semaphore>,
    counted: Counted,
}

/// How a lookup ended, as each of those waiting on it is told.
type Looked = std::result::Result<Option<TenantId>, Failed>;

#[derive(Debug, Clone)]
enum Failed {
    /// The coordinator did not answer: every asker is told, and nothing is
    /// remembered, so the next ask looks again.
    Lookup(Arc<StorageError>),
    /// The node stopped before the lookup was done.
    Stopped,
}

/// A lookup under way, and which it is, so that the lookup takes out of
/// those under way only itself.
#[derive(Clone)]
struct Pending {
    id: u64,
    answer: Shared<BoxFuture<'static, Looked>>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").field("id", &self.id).finish()
    }
}

/// How the keys presented to a node have been answered, counted from
/// its start: from what it remembered, by asking the coordinator, or
/// turned away for want of a lookup.
#[derive(Debug, Default)]
struct Counted {
    remembered: AtomicU64,
    looked_up: AtomicU64,
    shared: AtomicU64,
    busy: AtomicU64,
}

/// How many asks for keys a node has answered from what it remembered,
/// by looking the key up with the coordinator, by joining a lookup already
/// under way, and how many it turned away as busy, since it started; and
/// lookups of keys in use made before they were due are among those
/// looked up. A key not of the issued form is none of these: it is
/// refused unlooked at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyLookups {
    pub remembered: u64,
    pub looked_up: u64,
    pub shared: u64,
    pub busy: u64,
}

impl Default for Keys {
    fn default() -> Self {
        Self(Arc::new(Kept {
            remembered: Mutex::default(),
            lookups: Arc::new(Semaphore::new(LOOKUPS)),
            counted: Counted::default(),
        }))
    }
}

#[derive(Debug, Default)]
struct Remembered {
    found: HashMap<KeyHash, (TenantId, Instant)>,
    unknown: HashMap<KeyHash, Instant>,
    /// Moved on whenever keys are forgotten, so that a lookup made before
    /// is not remembered after.
    generation: u64,
    /// The lookups under way, one at most for a key.
    pending: HashMap<KeyHash, Pending>,
    /// The id of the next lookup.
    next: u64,
}

impl Remembered {
    /// What is remembered of the key, if it is still fresh, and when it
    /// was looked up.
    fn get(&self, hash: &KeyHash) -> Option<(Option<TenantId>, Instant)> {
        let fresh = |at: &Instant| at.elapsed() < REMEMBERED_FOR;
        if let Some((tenant, at)) = self.found.get(hash).filter(|(_, at)| fresh(at)) {
            return Some((Some(tenant.clone()), *at));
        }
        self.unknown
            .get(hash)
            .filter(|at| fresh(at))
            .map(|at| (None, *at))
    }

    fn insert(&mut self, hash: KeyHash, tenant: Option<TenantId>) {
        let now = Instant::now();
        // A key looked up again may have changed either way: what was
        // remembered of it before goes.
        match tenant {
            Some(tenant) => {
                self.unknown.remove(&hash);
                bounded(&mut self.found, MOST_FOUND, |(_, at)| *at);
                self.found.insert(hash, (tenant, now));
            }
            None => {
                self.found.remove(&hash);
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
        let counted = &self.0.counted;
        let answer = {
            let mut remembered = self.remembered();
            if let Some((tenant, at)) = remembered.get(&hash) {
                counted.remembered.fetch_add(1, Ordering::Relaxed);
                let due = at.elapsed() >= REMEMBERED_FOR.saturating_sub(REFRESHED_BEFORE);
                if due && !remembered.pending.contains_key(&hash) {
                    // Looked up again now, if a lookup is free, and answered
                    // from what is remembered meanwhile. With none free, the
                    // next ask tries again.
                    let _ = self.look_up(&mut remembered, lookup, hash);
                }
                return Ok(tenant);
            }
            match remembered.pending.get(&hash) {
                Some(pending) => {
                    counted.shared.fetch_add(1, Ordering::Relaxed);
                    pending.answer.clone()
                }
                None => match self.look_up(&mut remembered, lookup, hash) {
                    Some(answer) => answer,
                    None => {
                        counted.busy.fetch_add(1, Ordering::Relaxed);
                        return Err(NodeError::Busy);
                    }
                },
            }
        };
        answer.await.map_err(|failed| match failed {
            Failed::Lookup(error) => NodeError::KeyLookup(error),
            Failed::Stopped => NodeError::Stopped,
        })
    }

    /// Start looking `hash` up, on a task of its own, and note it as under
    /// way: or `None`, if as many lookups are under way as may be. The
    /// lookup remembers what it finds unless keys were forgotten since it
    /// began, and takes itself out of those under way as it ends, both
    /// before anyone waiting on it is answered.
    fn look_up(
        &self,
        remembered: &mut Remembered,
        lookup: &impl KeyLookup,
        hash: KeyHash,
    ) -> Option<Shared<BoxFuture<'static, Looked>>> {
        let permit = self.0.lookups.clone().try_acquire_owned().ok()?;
        self.0.counted.looked_up.fetch_add(1, Ordering::Relaxed);
        let (kept, generation, id) = (self.0.clone(), remembered.generation, remembered.next);
        remembered.next += 1;
        let asked = lookup.key_tenant(hash);
        let task = tokio::spawn(async move {
            let _permit = permit;
            let looked = asked
                .await
                .map_err(|error| Failed::Lookup(Arc::new(error.into())));
            let mut remembered = kept.remembered.lock().expect("keys lock");
            if let Ok(tenant) = &looked {
                if remembered.generation == generation {
                    remembered.insert(hash, tenant.clone());
                }
            }
            if remembered
                .pending
                .get(&hash)
                .is_some_and(|pending| pending.id == id)
            {
                remembered.pending.remove(&hash);
            }
            looked
        });
        let answer = async move {
            match task.await {
                Ok(looked) => looked,
                Err(ended) => match ended.try_into_panic() {
                    Ok(panic) => std::panic::resume_unwind(panic),
                    Err(_) => Err(Failed::Stopped),
                },
            }
        }
        .boxed()
        .shared();
        remembered.pending.insert(
            hash,
            Pending {
                id,
                answer: answer.clone(),
            },
        );
        Some(answer)
    }

    /// How the keys presented so far have been answered.
    pub fn lookups(&self) -> KeyLookups {
        let read = |count: &AtomicU64| count.load(Ordering::Relaxed);
        let counted = &self.0.counted;
        KeyLookups {
            remembered: read(&counted.remembered),
            looked_up: read(&counted.looked_up),
            shared: read(&counted.shared),
            busy: read(&counted.busy),
        }
    }

    /// Forget what was looked up for `tenant`, whose keys have changed, and
    /// whatever is being looked up now: a lookup under way remembers
    /// nothing, and no ask from now on joins it.
    pub fn forget(&self, tenant: &TenantId) {
        let mut remembered = self.remembered();
        remembered.generation += 1;
        remembered.found.retain(|_, (reaches, _)| reaches != tenant);
        remembered.pending.clear();
    }

    fn remembered(&self) -> std::sync::MutexGuard<'_, Remembered> {
        self.0.remembered.lock().expect("keys lock")
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
        let (found, _) = remembered.get(&hashed(&issued.secret)).unwrap();
        assert_eq!(found, Some(acme));
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

    /// An ask made after a tenant's keys are forgotten does not join a
    /// lookup begun before: it looks the key up afresh.
    #[tokio::test]
    async fn an_ask_after_keys_are_forgotten_joins_no_lookup_begun_before() {
        let (keys, stand, secret) = (
            Arc::new(Keys::default()),
            Stand::new(Some(acme())),
            secret(),
        );
        let before = asking(&keys, &stand, &secret, 1).remove(0);
        stand.called(1).await;
        keys.forget(&acme());
        stand.answering(Some(None));
        let after = asking(&keys, &stand, &secret, 1).remove(0);
        stand.called(2).await;
        stand.release();
        assert_eq!(
            before.await.unwrap().unwrap(),
            None,
            "both see the answer given as they end"
        );
        assert_eq!(after.await.unwrap().unwrap(), None);
        assert_eq!(stand.calls(), 2, "the later ask looked the key up itself");
    }
}
