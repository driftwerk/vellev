//! The clock that what waits on a model reads and sleeps by: tokio's,
//! unless a test hands in its own. A test that hands in a [`Manual`]
//! clock moves it only when it chooses, and can see what is asleep on it
//! first: so it can tell, and not guess from the time, that what it
//! tests has nothing left to do but wait, whatever is still on its way
//! over a socket or on another thread.

use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

/// A sleep on a [`Clock`].
pub type Sleep = Pin<Box<dyn Future<Output = ()> + Send>>;

/// What the time is, and a sleep until a time.
pub trait Clock: Debug + Send + Sync {
    fn now(&self) -> Instant;
    fn sleep_until(&self, at: Instant) -> Sleep;

    fn sleep(&self, wait: Duration) -> Sleep {
        let now = self.now();
        self.sleep_until(now.checked_add(wait).unwrap_or(now + crate::LONGEST))
    }
}

/// Tokio's clock: what everything runs by, but for tests that hand in
/// another.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tokio;

impl Clock for Tokio {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep_until(&self, at: Instant) -> Sleep {
        Box::pin(tokio::time::sleep_until(at))
    }
}

/// The clock everything runs by unless a test hands in its own.
pub fn tokio() -> Arc<dyn Clock> {
    Arc::new(Tokio)
}

/// A clock a test moves by hand. It starts where tokio's clock was when
/// it was made, and stands still until moved.
#[derive(Debug)]
pub struct Manual {
    started: Instant,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    gone: Duration,
    asleep: Vec<(Instant, tokio::sync::oneshot::Sender<()>)>,
}

impl Manual {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            started: Instant::now(),
            state: Mutex::new(State::default()),
        })
    }

    /// How far it has been moved.
    pub fn gone(&self) -> Duration {
        self.state.lock().expect("clock lock").gone
    }

    /// When each sleep still waited on ends, earliest first. A sleep no
    /// longer waited on, its waiter having gone on without it, is not
    /// counted.
    pub fn asleep(&self) -> Vec<Instant> {
        let mut state = self.state.lock().expect("clock lock");
        state.asleep.retain(|(_, woken)| !woken.is_closed());
        let mut ends: Vec<Instant> = state.asleep.iter().map(|(at, _)| *at).collect();
        ends.sort();
        ends
    }

    /// Moves the clock to `at`, if that is later, waking every sleep that
    /// ends by then.
    pub fn move_to(&self, at: Instant) {
        let mut state = self.state.lock().expect("clock lock");
        let gone = at.saturating_duration_since(self.started);
        state.gone = state.gone.max(gone);
        let now = self.started + state.gone;
        let (due, still): (Vec<_>, Vec<_>) = std::mem::take(&mut state.asleep)
            .into_iter()
            .partition(|(at, _)| *at <= now);
        state.asleep = still;
        drop(state);
        for (_, woken) in due {
            let _ = woken.send(());
        }
    }
}

impl Clock for Manual {
    fn now(&self) -> Instant {
        self.started + self.state.lock().expect("clock lock").gone
    }

    fn sleep_until(&self, at: Instant) -> Sleep {
        let mut state = self.state.lock().expect("clock lock");
        if at <= self.started + state.gone {
            return Box::pin(std::future::ready(()));
        }
        let (woken, wakes) = tokio::sync::oneshot::channel();
        state.asleep.push((at, woken));
        Box::pin(async move {
            let _ = wakes.await;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokio's clock, which everything runs by unless a test hands in its
    /// own, is tokio's own time: read and slept by exactly as before.
    #[tokio::test(start_paused = true)]
    async fn the_clock_everything_runs_by_is_tokios() {
        let clock = tokio();
        let began = Instant::now();
        assert_eq!(clock.now(), began);
        clock.sleep(Duration::from_millis(1_500)).await;
        assert_eq!(Instant::now() - began, Duration::from_millis(1_500));
        assert_eq!(clock.now(), Instant::now());
        clock.sleep_until(began).await;
        assert_eq!(Instant::now() - began, Duration::from_millis(1_500));
    }

    #[tokio::test]
    async fn a_manual_clock_moves_only_when_moved_and_wakes_what_is_due() {
        let clock = Manual::new();
        let began = clock.now();
        let short = clock.sleep(Duration::from_millis(500));
        let long = clock.sleep(Duration::from_secs(2));
        assert_eq!(
            clock.asleep(),
            [
                began + Duration::from_millis(500),
                began + Duration::from_secs(2)
            ]
        );
        let short = tokio::spawn(short);
        clock.move_to(began + Duration::from_secs(1));
        short.await.unwrap();
        assert_eq!(clock.gone(), Duration::from_secs(1));
        assert_eq!(clock.asleep(), [began + Duration::from_secs(2)]);
        drop(long);
        assert!(clock.asleep().is_empty(), "a sleep given up is not counted");
        // Not moved back.
        clock.move_to(began);
        assert_eq!(clock.now(), began + Duration::from_secs(1));
    }
}
