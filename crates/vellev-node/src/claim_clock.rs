//! The clock a backup claim is measured on.
//!
//! The coordinator lets another node take a claim once it has run out by
//! its own clock, which goes on while this machine is suspended or
//! hibernating. The clock the runtime and `Instant` keep does not:
//! CLOCK_MONOTONIC on Linux and CLOCK_UPTIME_RAW on macOS stop while the
//! system sleeps. A node that woke would take a claim long run out for
//! held. So a claim is measured on a clock that counts the time this
//! system was suspended: CLOCK_BOOTTIME on Linux, and CLOCK_MONOTONIC on
//! macOS, which counts the time asleep there.
//!
//! It counts only what the system knows passed. A guest its hypervisor
//! pauses, to snapshot or move it, or keeps from running, is not suspended
//! as far as it can tell, and often resumes with every clock where it was:
//! no clock of the machine is a fence against that. What fences it is the
//! coordinator, asked before each batch of deletes, and a backup looking
//! again that its files are there; the module that holds claims says what
//! that leaves.

use std::fmt;
use std::time::Duration;

use nix::time::{clock_gettime, ClockId};

/// Time as a backup claim is measured: since some fixed point, counting
/// any time the machine was suspended.
pub trait ClaimClock: fmt::Debug + Send + Sync {
    fn now(&self) -> Duration;
}

/// The clock that counts suspended time on the machine the node runs on.
#[derive(Debug, Default, Clone, Copy)]
pub struct BootClock;

#[cfg(any(target_os = "linux", target_os = "android"))]
const COUNTING_SUSPEND: ClockId = ClockId::CLOCK_BOOTTIME;
#[cfg(any(target_os = "macos", target_os = "ios"))]
const COUNTING_SUSPEND: ClockId = ClockId::CLOCK_MONOTONIC;
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
compile_error!("a backup claim needs a clock that counts the time the machine is suspended");

impl ClaimClock for BootClock {
    fn now(&self) -> Duration {
        // Reading a clock the system has cannot fail but for a clock id it
        // lacks, which the cfgs above rule out.
        clock_gettime(COUNTING_SUSPEND)
            .map(Duration::from)
            .expect("the clock that counts suspended time can be read")
    }
}

#[cfg(test)]
mod tests {
    use super::{BootClock, ClaimClock};

    #[test]
    fn the_boot_clock_moves_on() {
        let clock = BootClock;
        let before = clock.now();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(clock.now() > before);
    }
}
