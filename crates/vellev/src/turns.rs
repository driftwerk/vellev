//! The turns a node has for answering reads, and who gets the next one.
//!
//! A node answers so many reads at once, and a tenant's share of them is
//! half. A free turn is given to whichever read asks for it first, with
//! one exception: while a read waits whose tenant has less than its
//! share, a tenant that has its share or more is given none. So a tenant
//! with the node to itself may have every turn; one that comes while
//! another has them all is given those that come free until it has its
//! share, in no order among several such tenants; and no turn is left
//! free while a read waits for one, since where every tenant waiting has
//! its share already, any of them may have it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use vellev_core::TenantId;

/// The turns of one node.
#[derive(Debug)]
pub(crate) struct Turns {
    most: usize,
    taken: Mutex<Taken>,
    /// Told whenever a turn may have become one a waiting read can take:
    /// one came free, or a read stopped waiting.
    changed: Notify,
}

#[derive(Debug, Default)]
struct Taken {
    /// How many turns each tenant has, and how many of its reads wait for
    /// one. A tenant with neither is not kept.
    held: HashMap<TenantId, usize>,
    waiting: HashMap<TenantId, usize>,
}

impl Taken {
    fn all(&self) -> usize {
        self.held.values().sum()
    }

    fn less(counted: &mut HashMap<TenantId, usize>, tenant: &TenantId) {
        if let Some(count) = counted.get_mut(tenant) {
            *count -= 1;
            if *count == 0 {
                counted.remove(tenant);
            }
        }
    }
}

/// One turn, given back when dropped.
#[derive(Debug)]
pub(crate) struct Turn {
    turns: Arc<Turns>,
    tenant: TenantId,
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut taken = self.turns.taken.lock().expect("turns lock");
        Taken::less(&mut taken.held, &self.tenant);
        drop(taken);
        self.turns.changed.notify_waiters();
    }
}

/// A read counted as waiting for a turn until this is dropped, whether
/// it got one or stopped waiting.
struct Waiting<'a> {
    turns: &'a Turns,
    tenant: &'a TenantId,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut taken = self.turns.taken.lock().expect("turns lock");
        Taken::less(&mut taken.waiting, self.tenant);
        drop(taken);
        // With this read gone, another tenant may no longer be held to
        // its share.
        self.turns.changed.notify_waiters();
    }
}

impl Turns {
    /// `most` turns, and one at least.
    pub fn new(most: usize) -> Arc<Self> {
        Arc::new(Self {
            most: most.max(1),
            taken: Mutex::default(),
            changed: Notify::new(),
        })
    }

    /// How many turns there are.
    pub fn most(&self) -> usize {
        self.most
    }

    /// How many turns a tenant may have while another tenant waits for
    /// one.
    pub fn share(&self) -> usize {
        (self.most / 2).max(1)
    }

    /// A turn for `tenant` if it may have one now: there is one free, and
    /// the tenant has less than its share, or no tenant with less than
    /// its share is waiting for one. A tenant that has its share is not
    /// such a tenant itself, so only others' waiting keeps it back.
    fn free(self: &Arc<Self>, tenant: &TenantId) -> Option<Turn> {
        let mut taken = self.taken.lock().expect("turns lock");
        let share = self.share();
        let has = |tenant: &TenantId| taken.held.get(tenant).copied().unwrap_or(0);
        let owed_to_another = taken.waiting.keys().any(|waiting| has(waiting) < share);
        if taken.all() >= self.most || (has(tenant) >= share && owed_to_another) {
            return None;
        }
        *taken.held.entry(tenant.clone()).or_default() += 1;
        Some(Turn {
            turns: self.clone(),
            tenant: tenant.clone(),
        })
    }

    /// A turn for `tenant`, waited for as long as it takes. Whoever stops
    /// waiting is no longer counted as waiting.
    pub async fn take(self: &Arc<Self>, tenant: &TenantId) -> Turn {
        if let Some(turn) = self.free(tenant) {
            return turn;
        }
        *self
            .taken
            .lock()
            .expect("turns lock")
            .waiting
            .entry(tenant.clone())
            .or_default() += 1;
        let _waiting = Waiting {
            turns: self,
            tenant,
        };
        loop {
            // Listening from before it looks, so that a turn that comes
            // free between the look and the wait is not missed.
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(turn) = self.free(tenant) {
                return turn;
            }
            changed.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn tenant(name: &str) -> TenantId {
        TenantId::new(name).unwrap()
    }

    /// Whether `tenant` is given a turn without anything coming free.
    async fn given(turns: &Arc<Turns>, tenant: &TenantId) -> Option<Turn> {
        tokio::time::timeout(Duration::from_secs(1), turns.take(tenant))
            .await
            .ok()
    }

    #[tokio::test(start_paused = true)]
    async fn a_tenant_alone_has_every_turn_and_no_more() {
        let turns = Turns::new(4);
        let acme = tenant("acme");
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(given(&turns, &acme).await.expect("a turn"));
        }
        assert!(given(&turns, &acme).await.is_none(), "there are four");
        // What it gives back it may take again.
        held.pop();
        held.push(given(&turns, &acme).await.expect("the one given back"));
    }

    #[tokio::test(start_paused = true)]
    async fn while_another_tenant_waits_none_is_given_more_than_its_share() {
        let turns = Turns::new(4);
        let (acme, beta) = (tenant("acme"), tenant("beta"));
        let mut acmes = Vec::new();
        for _ in 0..4 {
            acmes.push(given(&turns, &acme).await.expect("a turn"));
        }
        // Both wait: three reads of the tenant that has every turn, which
        // asked first, and three of another.
        let asking = |tenant: &TenantId| {
            let (turns, tenant) = (turns.clone(), tenant.clone());
            tokio::spawn(async move { turns.take(&tenant).await })
        };
        let more: Vec<_> = (0..3).map(|_| asking(&acme)).collect();
        tokio::task::yield_now().await;
        let betas: Vec<_> = (0..3).map(|_| asking(&beta)).collect();
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Each turn that comes free goes to the other tenant until it has
        // its share, and the first keeps no more than its own.
        acmes.pop();
        acmes.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        let had = |asked: &[tokio::task::JoinHandle<Turn>]| {
            asked.iter().filter(|asked| asked.is_finished()).count()
        };
        assert_eq!((had(&more), had(&betas)), (0, 2));
        // Nor does the other take more than its share while the first
        // waits.
        acmes.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!((had(&more), had(&betas)), (1, 2));
        assert_eq!(turns.taken.lock().unwrap().all(), 4);

        // Once nobody else waits, a tenant may go past its share again.
        for waiting in &more {
            waiting.abort();
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        acmes.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(had(&betas), 3, "the turn nobody else waits for");
    }

    #[tokio::test(start_paused = true)]
    async fn no_turn_is_left_free_while_a_read_waits_for_one() {
        // Five turns, and a share of two: two tenants with their shares
        // leave one over, which neither is owed and either may have.
        let turns = Turns::new(5);
        let (acme, beta) = (tenant("acme"), tenant("beta"));
        let mut held = Vec::new();
        for _ in 0..5 {
            held.push(given(&turns, &acme).await.expect("a turn"));
        }
        let asking = |tenant: &TenantId| {
            let (turns, tenant) = (turns.clone(), tenant.clone());
            tokio::spawn(async move { turns.take(&tenant).await })
        };
        // The first to come free goes to the tenant with none, though
        // the other asked first.
        let more = asking(&acme);
        tokio::task::yield_now().await;
        let other: Vec<_> = (0..3).map(|_| asking(&beta)).collect();
        tokio::time::sleep(Duration::from_millis(1)).await;
        let had = |asked: &[tokio::task::JoinHandle<Turn>]| {
            asked.iter().filter(|asked| asked.is_finished()).count()
        };
        held.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!((more.is_finished(), had(&other)), (false, 1));
        // So does the next, which gives that tenant its share. The one
        // after is owed to neither: both have their shares, both wait,
        // and it is not kept from both.
        held.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!((more.is_finished(), had(&other)), (false, 2));
        held.pop();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(usize::from(more.is_finished()) + had(&other), 3);
        assert_eq!(turns.taken.lock().unwrap().all(), 5);
    }

    #[tokio::test(start_paused = true)]
    async fn a_read_that_stops_waiting_is_counted_no_longer() {
        let turns = Turns::new(2);
        let (acme, beta) = (tenant("acme"), tenant("beta"));
        let first = given(&turns, &acme).await.expect("a turn");
        let second = given(&turns, &acme).await.expect("a turn");
        // Another tenant waits and gives up.
        assert!(given(&turns, &beta).await.is_none());
        {
            let taken = turns.taken.lock().unwrap();
            assert!(taken.waiting.is_empty(), "{taken:?}");
        }
        // So the first is held to no share: it takes back what it frees.
        drop(first);
        let again = given(&turns, &acme).await.expect("past its share of one");
        drop((second, again));
        let taken = turns.taken.lock().unwrap();
        assert!(taken.held.is_empty(), "{taken:?}");
    }

    /// Counts how often whoever polls with it is told to look again.
    #[derive(Default)]
    struct Told(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for Told {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn a_read_kept_back_for_one_that_then_stops_waiting_is_told_to_look_again() {
        use std::future::Future;
        use std::sync::atomic::Ordering;
        use std::task::{Context, Poll};
        // One tenant has both turns and asks for another; a second asks
        // too. Each is looked at by hand, so that nothing but the order
        // written here decides who looks when.
        let turns = Turns::new(2);
        let (acme, beta) = (tenant("acme"), tenant("beta"));
        let mut held: Vec<Turn> = (0..2).map(|_| turns.free(&acme).expect("a turn")).collect();
        let told = Arc::new(Told::default());
        let waker = std::task::Waker::from(told.clone());
        let mut looking = Context::from_waker(&waker);
        let mut more = Box::pin(turns.take(&acme));
        let mut other = Box::pin(turns.take(&beta));
        assert!(more.as_mut().poll(&mut looking).is_pending());
        assert!(other.as_mut().poll(&mut looking).is_pending());

        // A turn comes free. The first looks before the second does, and
        // is kept back on its account: the second has less than its
        // share.
        held.pop();
        assert!(more.as_mut().poll(&mut looking).is_pending());
        // The second stops waiting, never having looked. The turn is
        // free and nobody is owed it: the first must be told, for
        // nothing else will ever have it look again.
        let before = told.0.load(Ordering::SeqCst);
        drop(other);
        assert!(told.0.load(Ordering::SeqCst) > before, "told to look again");
        assert!(matches!(more.as_mut().poll(&mut looking), Poll::Ready(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn a_node_has_one_turn_at_least() {
        let turns = Turns::new(0);
        assert_eq!((turns.most(), turns.share()), (1, 1));
        assert!(given(&turns, &tenant("acme")).await.is_some());
    }
}
