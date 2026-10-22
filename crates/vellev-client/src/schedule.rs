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
//!    after the wait it gave if that is longer, up to
//!    [`LONGEST_TURNED_AWAY`]. An address that could not answer just now
//!    and said when to ask again is not asked again sooner either, up to
//!    the same longest wait: the other addresses are asked meanwhile, and
//!    the wait that comes before any is asked a second time is no shorter
//!    than the one it gave.
//! 5. The request is given up when its tries are spent.
//! 6. An answer that is less than another address might give ends all of
//!    the above. Each address not yet asked for this request is asked
//!    once, at once, whatever the others answer, and nothing more is sent:
//!    the answer in hand is not kept back for a better one that may never
//!    come. A client of one address sends nothing more at all.
//!
//! So a request is sent at most as many times as it has tries and one
//! less than there are addresses.

use std::time::Duration;

/// The longest wait between two tries.
pub(crate) const LONGEST_BACKOFF: Duration = Duration::from_secs(5);

/// The longest an address that turns a request away can make its caller
/// wait before the next try, whatever wait it gives.
pub(crate) const LONGEST_TURNED_AWAY: Duration = Duration::from_secs(60);

/// How a try ended, where it did not end the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tried {
    /// It failed in a way that may pass, here or at another address.
    MayPass,
    /// It failed in a way that may pass, and the address said to ask it
    /// again no sooner than after this long.
    Busy(Duration),
    /// The address turned it away for asking too much, to be asked again
    /// after this long.
    TurnedAway(Duration),
    /// The address cannot do it as it is set up; another may.
    NotByThisAddress,
    /// The address does not serve the tenant, and never will in this
    /// request: it is not asked again, and once no address is left that
    /// may serve it, the request is given up.
    NotServedHere,
    /// The address answered with less than another might.
    Less,
}

/// What to do after a try.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Next {
    /// Ask this address, after this wait.
    Ask {
        address: usize,
        after: Duration,
    },
    /// Ask this address at once, and not for long, whether it can better
    /// the answer in hand.
    AskForBetter {
        address: usize,
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
    /// The addresses that said they do not serve the tenant.
    refused: Vec<bool>,
    /// The addresses asked since the last wait.
    asked: Vec<bool>,
    /// The addresses asked at all.
    sent: Vec<bool>,
    /// The longest any address asked since the last wait said to leave
    /// it before it is asked again.
    told_to_wait: Duration,
    /// Whether an answer is in hand that another address might better.
    bettering: bool,
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
            refused: vec![false; addresses],
            sent: asked.clone(),
            asked,
            told_to_wait: Duration::ZERO,
            bettering: false,
        }
    }

    /// What follows a try of the address asked last that ended as `tried`.
    pub fn after(&mut self, tried: Tried) -> Next {
        self.bettering |= tried == Tried::Less;
        if self.bettering {
            return match self.sent.iter().position(|sent| !sent) {
                Some(address) => {
                    self.sent[address] = true;
                    self.address = address;
                    Next::AskForBetter { address }
                }
                None => Next::GiveUp,
            };
        }
        let free = match tried {
            Tried::NotByThisAddress => {
                self.unable[self.address] = true;
                self.unable.contains(&false)
            }
            Tried::NotServedHere => {
                self.unable[self.address] = true;
                self.refused[self.address] = true;
                if !self.refused.contains(&false) {
                    return Next::GiveUp;
                }
                self.unable.contains(&false)
            }
            Tried::MayPass | Tried::Busy(_) | Tried::TurnedAway(_) | Tried::Less => false,
        };
        if !free {
            self.tries_left = self.tries_left.saturating_sub(1);
            if self.tries_left == 0 {
                return Next::GiveUp;
            }
        }
        let (address, least) = match tried {
            Tried::TurnedAway(wait) => (self.address, wait.min(LONGEST_TURNED_AWAY)),
            Tried::Busy(wait) => {
                self.told_to_wait = self.told_to_wait.max(wait.min(LONGEST_TURNED_AWAY));
                (self.following(), Duration::ZERO)
            }
            Tried::MayPass | Tried::NotByThisAddress | Tried::NotServedHere | Tried::Less => {
                (self.following(), Duration::ZERO)
            }
        };
        let after = match self.asked[address] {
            true => {
                let wait = self.backoff.max(least).max(self.told_to_wait);
                self.told_to_wait = Duration::ZERO;
                self.backoff = self.backoff.saturating_mul(2).min(LONGEST_BACKOFF);
                self.asked.fill(false);
                wait
            }
            false => Duration::ZERO,
        };
        self.asked[address] = true;
        self.sent[address] = true;
        self.address = address;
        Next::Ask { address, after }
    }

    /// The address after the one asked last, passing over those that have
    /// said they cannot while any has not.
    fn following(&self) -> usize {
        let count = self.unable.len();
        let round = || (1..=count).map(|step| (self.address + step) % count);
        let passed_over = self.unable.contains(&false);
        // An address that does not serve the tenant is never asked again;
        // `after` gives up before every address has said so.
        round()
            .find(|&address| !self.refused[address] && !(passed_over && self.unable[address]))
            .or_else(|| round().find(|&address| !self.refused[address]))
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
                Next::AskForBetter { address: next } => unreachable!("no answer was less: {next}"),
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
    fn an_address_that_does_not_serve_the_tenant_is_asked_once_and_none_left_gives_up() {
        const REFUSED: &[Option<Tried>] = &[Some(Tried::NotServedHere)];
        assert_eq!(asked(&[REFUSED], 0, 4), (vec![(0, NONE)], false));
        assert_eq!(
            asked(&[REFUSED, REFUSED], 1, 4),
            (vec![(1, NONE), (0, NONE)], false)
        );
        assert_eq!(
            asked(&[REFUSED, DONE], 0, 1),
            (vec![(0, NONE), (1, NONE)], true)
        );
        // A refusing address is passed over even once others failed.
        assert_eq!(
            asked(&[REFUSED, AWAY], 0, 3),
            (
                vec![(0, NONE), (1, NONE), (1, MS(100)), (1, MS(200))],
                false
            )
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
        // However long an address says to wait, it is not waited on past
        // the longest.
        let for_an_hour: &[Option<Tried>] = &[Some(Tried::TurnedAway(MS(3_600_000))), None];
        assert_eq!(
            asked(&[for_an_hour], 0, 2),
            (vec![(0, NONE), (0, LONGEST_TURNED_AWAY)], true)
        );
    }

    #[test]
    fn an_address_that_says_when_to_ask_again_is_left_that_long_while_the_others_are_asked() {
        let busy = |wait| Some(Tried::Busy(MS(wait)));
        // Alone, it is asked again after the wait it gave, the longer of
        // that and the backoff.
        let answers: &[Option<Tried>] = &[busy(1_000), busy(50), None];
        assert_eq!(
            asked(&[answers], 0, 4),
            (vec![(0, NONE), (0, MS(1_000)), (0, MS(200))], true)
        );
        // Beside another, the other is asked at once, and the wait before
        // either is asked a second time is the one it gave.
        assert_eq!(
            asked(&[answers, AWAY], 0, 4),
            (vec![(0, NONE), (1, NONE), (0, MS(1_000)), (1, NONE)], false)
        );
        assert_eq!(
            asked(&[AWAY, answers], 0, 4),
            (vec![(0, NONE), (1, NONE), (0, MS(1_000)), (1, NONE)], false)
        );
        // Once waited, it is not waited again.
        assert_eq!(
            asked(&[&[busy(1_000), Some(Tried::MayPass)]], 0, 4),
            (
                vec![(0, NONE), (0, MS(1_000)), (0, MS(200)), (0, MS(400))],
                false
            )
        );
        // However long an address says to wait, it is not waited on past
        // the longest.
        let for_an_hour: &[Option<Tried>] = &[busy(3_600_000), None];
        assert_eq!(
            asked(&[for_an_hour], 0, 2),
            (vec![(0, NONE), (0, LONGEST_TURNED_AWAY)], true)
        );
    }

    /// The steps that follow each of `tried` in turn, from a request to
    /// `addresses` starting at `first` with four tries.
    fn steps(addresses: usize, first: usize, tried: &[Tried]) -> Vec<Next> {
        let mut schedule = Schedule::new(addresses, first, 4, BACKOFF);
        tried.iter().map(|tried| schedule.after(*tried)).collect()
    }

    #[test]
    fn an_answer_that_is_less_is_bettered_only_by_addresses_not_yet_asked_each_once_at_once() {
        let better = |address| Next::AskForBetter { address };
        assert_eq!(
            steps(1, 0, &[Tried::Less]),
            [Next::GiveUp],
            "one address has nobody to better it"
        );
        // Whatever the others answer, each is asked once and no more.
        for others in [
            [Tried::Less, Tried::Less],
            [Tried::MayPass, Tried::NotByThisAddress],
            [Tried::TurnedAway(MS(1_000)), Tried::MayPass],
        ] {
            assert_eq!(
                steps(3, 1, &[Tried::Less, others[0], others[1]]),
                [better(0), better(2), Next::GiveUp],
                "{others:?}"
            );
        }
        // An address that failed before the lesser answer came is not
        // asked again for a better one, nor is one that could not.
        assert_eq!(
            steps(3, 0, &[Tried::MayPass, Tried::Less, Tried::MayPass]),
            [
                Next::Ask {
                    address: 1,
                    after: NONE
                },
                better(2),
                Next::GiveUp
            ]
        );
        assert_eq!(
            steps(2, 0, &[Tried::NotByThisAddress, Tried::Less]),
            [
                Next::Ask {
                    address: 1,
                    after: NONE
                },
                Next::GiveUp
            ]
        );
        // After tries spent on failures that may pass, with their waits,
        // a lesser answer is still the end of them.
        assert_eq!(
            steps(1, 0, &[Tried::MayPass, Tried::Less]),
            [
                Next::Ask {
                    address: 0,
                    after: MS(100)
                },
                Next::GiveUp
            ]
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

    /// Every way a request to up to three addresses can go for six tries'
    /// answers: whatever the answers, the rules hold. What is expected is
    /// worked out here from the rules as they are written above, not as
    /// the schedule works them out.
    #[test]
    fn whatever_the_answers_the_rules_hold() {
        let kinds = [
            Tried::MayPass,
            Tried::NotByThisAddress,
            // Shorter than the first backoff, longer than the longest
            // backoff, and longer than an address may make a caller wait.
            Tried::TurnedAway(MS(30)),
            Tried::TurnedAway(MS(10_000)),
            Tried::TurnedAway(MS(120_000)),
        ];
        const ANSWERS: u32 = 6;
        let mut checked = 0;
        for addresses in 1..=3usize {
            for tries in 1..=4u32 {
                for first in 0..addresses {
                    for script in 0..kinds.len().pow(ANSWERS) {
                        let mut schedule = Schedule::new(addresses, first, tries, BACKOFF);
                        // What the rules say, kept apart from the schedule.
                        let mut since_wait = vec![false; addresses];
                        since_wait[first] = true;
                        let mut said_cannot = vec![false; addresses];
                        let mut waits = 0u32;
                        let (mut address, mut sends) = (first, 1usize);
                        let mut given_up = false;
                        for answer in 0..ANSWERS {
                            let tried = kinds[script / kinds.len().pow(answer) % kinds.len()];
                            let context = format!(
                                "{addresses} addresses, {tries} tries, from {first}, \
                                 script {script}, answer {answer}: {tried:?}"
                            );
                            if tried == Tried::NotByThisAddress {
                                said_cannot[address] = true;
                            }
                            let cannot = said_cannot.iter().filter(|said| **said).count();
                            // Rule 3: a request is sent once for each try,
                            // and once more for each address that said it
                            // cannot while another had yet to.
                            let allowed = tries as usize + cannot.min(addresses - 1);
                            let next = schedule.after(tried);
                            let Next::Ask { address: to, after } = next else {
                                assert_eq!(next, Next::GiveUp, "{context}");
                                assert_eq!(
                                    sends, allowed,
                                    "given up too soon or too late: {context}"
                                );
                                given_up = true;
                                break;
                            };
                            assert!(sends < allowed, "sent past its tries: {context}");
                            // Rule 4: the wait that is due, if one is.
                            let backoff = (BACKOFF * 2u32.pow(waits)).min(LONGEST_BACKOFF);
                            let due = match tried {
                                Tried::TurnedAway(given) => {
                                    assert_eq!(to, address, "asked again itself: {context}");
                                    backoff.max(given.min(Duration::from_secs(60)))
                                }
                                _ => {
                                    // Rules 1 and 2: the next in order
                                    // that is not passed over.
                                    let all_cannot = cannot == addresses;
                                    let expected = (1..=addresses)
                                        .map(|step| (address + step) % addresses)
                                        .find(|next| all_cannot || !said_cannot[*next]);
                                    assert_eq!(Some(to), expected, "{context}");
                                    backoff
                                }
                            };
                            match since_wait[to] {
                                true => {
                                    assert_eq!(
                                        after, due,
                                        "asked twice without the wait due: {context}"
                                    );
                                    since_wait.fill(false);
                                    waits += 1;
                                }
                                false => assert_eq!(
                                    after, NONE,
                                    "a wait before an address not asked: {context}"
                                ),
                            }
                            since_wait[to] = true;
                            address = to;
                            sends += 1;
                        }
                        assert!(
                            given_up,
                            "never given up: {addresses} addresses, {tries} tries, script {script}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, (1 + 2 + 3) * 4 * kinds.len().pow(ANSWERS));
    }
}
