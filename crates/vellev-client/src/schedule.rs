//! Which address a request is sent to next, and after what wait.
//!
//! The rules, for one request and the addresses a client knows:
//!
//! 1. The first address asked is the one that last answered for the
//!    tenant, and after it the addresses are gone round in order.
//! 2. An address that has said it cannot do the request as it is set up
//!    is passed over while any address has not said so. Once every address
//!    has, none is passed over: an address may be a load balancer with
//!    another node behind it next time.
//! 3. Every failure that may pass spends one of the request's tries. An
//!    address saying it cannot spends none while another has yet to say
//!    so, so that every address is reached however few the tries; after
//!    that it spends one like any other. A client of one address therefore
//!    sends as many times as it has tries, and one of `n` addresses at
//!    most `n - 1` more.
//! 4. No address is asked twice without a wait between: before an address
//!    already asked since the last wait is asked again, the backoff is
//!    waited, and the backoff doubles, up to [`LONGEST_BACKOFF`]. No wait
//!    comes before an address not asked since the last. An address that
//!    turned the request away for asking too much is asked again itself,
//!    after the wait it gave if that is longer.
//! 5. The request is given up when its tries are spent.

use std::time::Duration;

/// The longest wait between two tries.
pub(crate) const LONGEST_BACKOFF: Duration = Duration::from_secs(5);

/// How a try ended, where it did not end the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tried {
    /// It failed in a way that may pass, here or at another address.
    MayPass,
    /// The address turned it away for asking too much, to be asked again
    /// after this long.
    TurnedAway(Duration),
    /// The address cannot do it as it is set up; another may.
    NotByThisAddress,
}

/// What to do after a try.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Next {
    /// Ask this address, after this wait.
    Ask {
        address: usize,
        after: Duration,
    },
    GiveUp,
}

#[derive(Debug)]
pub(crate) struct Schedule {
    /// The address asked last.
    address: usize,
    tries_left: u32,
    backoff: Duration,
    /// The addresses that have said they cannot do the request.
    unable: Vec<bool>,
    /// The addresses asked since the last wait.
    asked: Vec<bool>,
}

impl Schedule {
    /// A request with `tries` to spend over `addresses`, of which `first`
    /// is asked now.
    pub fn new(addresses: usize, first: usize, tries: u32, backoff: Duration) -> Self {
        let mut asked = vec![false; addresses];
        asked[first] = true;
        Self {
            address: first,
            tries_left: tries,
            backoff,
            unable: vec![false; addresses],
            asked,
        }
    }

    /// What follows a try of the address asked last that ended as `tried`.
    pub fn after(&mut self, tried: Tried) -> Next {
        let free = match tried {
            Tried::NotByThisAddress => {
                self.unable[self.address] = true;
                self.unable.contains(&false)
            }
            Tried::MayPass | Tried::TurnedAway(_) => false,
        };
        if !free {
            self.tries_left = self.tries_left.saturating_sub(1);
            if self.tries_left == 0 {
                return Next::GiveUp;
            }
        }
        let (address, least) = match tried {
            Tried::TurnedAway(wait) => (self.address, wait),
            Tried::MayPass | Tried::NotByThisAddress => (self.following(), Duration::ZERO),
        };
        let after = match self.asked[address] {
            true => {
                let wait = self.backoff.max(least);
                self.backoff = self.backoff.saturating_mul(2).min(LONGEST_BACKOFF);
                self.asked.fill(false);
                wait
            }
            false => Duration::ZERO,
        };
        self.asked[address] = true;
        self.address = address;
        Next::Ask { address, after }
    }

    /// The address after the one asked last, passing over those that have
    /// said they cannot while any has not.
    fn following(&self) -> usize {
        let count = self.unable.len();
        let mut round = (1..=count).map(|step| (self.address + step) % count);
        let passed_over = self.unable.contains(&false);
        round
            .find(|&address| !(passed_over && self.unable[address]))
            .unwrap_or(self.address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKOFF: Duration = Duration::from_millis(100);
    const NONE: Duration = Duration::ZERO;

    /// The addresses asked and the wait before each, the first included,
    /// when each address answers every try as `answers` says, until the
    /// request is given up or an address answers `None`: that it is done.
    fn asked(
        answers: &[&[Option<Tried>]],
        first: usize,
        tries: u32,
    ) -> (Vec<(usize, Duration)>, bool) {
        let mut schedule = Schedule::new(answers.len(), first, tries, BACKOFF);
        let mut sent = vec![0; answers.len()];
        let mut asked = vec![(first, NONE)];
        let mut address = first;
        loop {
            let answer = answers[address][sent[address].min(answers[address].len() - 1)];
            sent[address] += 1;
            let Some(tried) = answer else {
                return (asked, true);
            };
            match schedule.after(tried) {
                Next::GiveUp => return (asked, false),
                Next::Ask {
                    address: next,
                    after,
                } => {
                    asked.push((next, after));
                    address = next;
                }
            }
        }
    }

    const AWAY: &[Option<Tried>] = &[Some(Tried::MayPass)];
    const UNABLE: &[Option<Tried>] = &[Some(Tried::NotByThisAddress)];
    const DONE: &[Option<Tried>] = &[None];
    const MS: fn(u64) -> Duration = Duration::from_millis;

    #[test]
    fn every_address_is_asked_before_any_wait_and_each_failure_that_may_pass_spends_a_try() {
        assert_eq!(
            asked(&[AWAY, AWAY], 0, 4),
            (vec![(0, NONE), (1, NONE), (0, MS(100)), (1, NONE)], false)
        );
        assert_eq!(
            asked(&[AWAY], 0, 3),
            (vec![(0, NONE), (0, MS(100)), (0, MS(200))], false)
        );
        assert_eq!(
            asked(&[AWAY, AWAY, DONE], 1, 4),
            (vec![(1, NONE), (2, NONE)], true),
            "starting from the address that last answered"
        );
        assert_eq!(asked(&[AWAY, DONE], 0, 1), (vec![(0, NONE)], false));
    }

    #[test]
    fn an_address_that_cannot_costs_no_try_while_another_has_yet_to_say_so() {
        assert_eq!(
            asked(&[UNABLE, UNABLE, DONE], 0, 1),
            (vec![(0, NONE), (1, NONE), (2, NONE)], true),
            "every address is reached with one try"
        );
        // Once every address has said so, each try is spent as any other,
        // with a wait before an address is asked again.
        assert_eq!(
            asked(&[UNABLE, UNABLE], 0, 3),
            (vec![(0, NONE), (1, NONE), (0, MS(100)), (1, NONE)], false),
            "one more send than tries, for two addresses"
        );
        assert_eq!(
            asked(&[UNABLE], 0, 4),
            (
                vec![(0, NONE), (0, MS(100)), (0, MS(200)), (0, MS(400))],
                false
            ),
            "one address is tried as often as for any failure that may pass"
        );
        let then_able: &[Option<Tried>] = &[Some(Tried::NotByThisAddress), None];
        assert_eq!(
            asked(&[then_able], 0, 4),
            (vec![(0, NONE), (0, MS(100))], true),
            "another node may be behind the one address next time"
        );
    }

    #[test]
    fn an_address_that_cannot_is_passed_over_while_another_may_yet_answer() {
        assert_eq!(
            asked(&[AWAY, UNABLE], 0, 3),
            (
                vec![(0, NONE), (1, NONE), (0, MS(100)), (0, MS(200))],
                false
            )
        );
        assert_eq!(
            asked(&[UNABLE, AWAY], 0, 3),
            (
                vec![(0, NONE), (1, NONE), (1, MS(100)), (1, MS(200))],
                false
            )
        );
        // Said in the middle of a round: the address that may yet answer
        // is not asked again without a wait.
        assert_eq!(
            asked(&[AWAY, UNABLE, AWAY], 0, 4),
            (
                vec![(0, NONE), (1, NONE), (2, NONE), (0, MS(100)), (2, NONE)],
                false
            )
        );
    }

    #[test]
    fn an_address_that_turns_the_request_away_is_asked_again_itself_after_the_longer_wait() {
        let limited = |wait| Some(Tried::TurnedAway(MS(wait)));
        let answers: &[Option<Tried>] = &[limited(1_000), limited(50), None];
        assert_eq!(
            asked(&[answers, DONE], 0, 4),
            (vec![(0, NONE), (0, MS(1_000)), (0, MS(200))], true)
        );
        assert_eq!(
            asked(&[answers], 0, 2),
            (vec![(0, NONE), (0, MS(1_000))], false)
        );
    }

    #[test]
    fn the_wait_doubles_no_further_than_the_longest() {
        let (asked, _) = asked(&[AWAY], 0, 9);
        let waits: Vec<Duration> = asked.into_iter().map(|(_, wait)| wait).collect();
        let secs = |secs| Duration::from_secs(secs);
        assert_eq!(
            waits,
            [
                NONE,
                MS(100),
                MS(200),
                MS(400),
                MS(800),
                MS(1_600),
                MS(3_200),
                secs(5),
                secs(5)
            ]
        );
    }

    /// Every way a request to up to three addresses can go for seven
    /// tries' answers: whatever the answers, the rules hold.
    #[test]
    fn whatever_the_answers_no_address_is_asked_twice_without_a_wait_nor_waited_for_unasked() {
        let kinds = [
            Tried::MayPass,
            Tried::NotByThisAddress,
            Tried::TurnedAway(MS(30)),
        ];
        let mut checked = 0;
        for addresses in 1..=3usize {
            for tries in 1..=4u32 {
                for first in 0..addresses {
                    for script in 0..kinds.len().pow(7) {
                        let answer = |sent: usize| kinds[script / kinds.len().pow(sent as u32) % 3];
                        let mut schedule = Schedule::new(addresses, first, tries, BACKOFF);
                        let mut since_wait = vec![false; addresses];
                        since_wait[first] = true;
                        let mut unable = vec![false; addresses];
                        let (mut address, mut sends, mut spent) = (first, 1, 0);
                        for sent in 0..7 {
                            let tried = answer(sent);
                            if tried == Tried::NotByThisAddress {
                                unable[address] = true;
                            }
                            let free = tried == Tried::NotByThisAddress && unable.contains(&false);
                            spent += u32::from(!free);
                            let next = schedule.after(tried);
                            let context = format!(
                                "{addresses} addresses, {tries} tries, from {first}, script {script}, send {sent}"
                            );
                            let Next::Ask { address: to, after } = next else {
                                assert_eq!(
                                    spent, tries,
                                    "given up only with the tries spent: {context}"
                                );
                                break;
                            };
                            assert!(
                                spent < tries,
                                "not given up with the tries spent: {context}"
                            );
                            if after > NONE {
                                assert!(
                                    since_wait[to],
                                    "a wait before an address not asked: {context}"
                                );
                                since_wait.fill(false);
                            }
                            assert!(!since_wait[to], "asked twice without a wait: {context}");
                            if let Tried::TurnedAway(wait) = tried {
                                assert!(to == address && after >= wait, "{context}");
                            }
                            let passed_over = unable[to] && unable.contains(&false);
                            assert!(
                                !passed_over,
                                "an address that cannot was asked while another may: {context}"
                            );
                            since_wait[to] = true;
                            address = to;
                            sends += 1;
                        }
                        assert!(
                            sends <= addresses - 1 + tries as usize,
                            "{sends} sends of {addresses} addresses with {tries} tries, script {script}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, (1 + 2 + 3) * 4 * 3usize.pow(7));
    }
}
