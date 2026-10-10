//! Many tenants at once against a stand-in local model, with the load
//! drawn by a seed, and what the README promises checked on every run.
//!
//! The stand-in costs what a model costs in shape: something for each
//! text, and something for each byte of the first few thousand, as a
//! model that reads so many tokens and no more does. A text that says
//! "fail" fails its run; one that says "panic" makes the model panic.
//!
//! The suite runs a few seeds. Many more are run by hand:
//!
//!   HARNESS_SEEDS=200 cargo test -p vellev-embed --lib harness -- --nocapture

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::line::{weight, LONGEST_TEXT};
use crate::local::{Load, Local, Model, Room, Times, RUN};
use crate::{EmbedError, Embedder, Purpose};

/// What the stand-in costs: for each text, and for each byte it reads,
/// which is the first so many of a text.
const EACH_TEXT: Duration = Duration::from_micros(40);
const EACH_BYTE: Duration = Duration::from_nanos(20);
const READS: usize = 4_096;
/// About how long a run is to take, and so about how long a search waits
/// behind writes.
const RUN_FOR: Duration = Duration::from_millis(10);
/// The most a run can cost: as many texts as a run can be of, each read
/// to the full.
const RUN_AT_MOST: Duration = Duration::from_micros(40 * 32 + 20 * 4_096 * 32 / 1_000);
/// How long a write may go without getting anywhere, and the longest it
/// is kept waiting for its turn, here: short, so that a seed is run in
/// seconds.
const STALLED: Duration = Duration::from_millis(400);
const HELD_AT_MOST: Duration = Duration::from_millis(600);
/// How long a seed's load runs for.
const RUNS_FOR: Duration = Duration::from_millis(1_500);
/// How long a search is given: well beyond the bound a search is held
/// to here, so that what is tested is the bound and not the deadline.
const QUERY_WITHIN: Duration = Duration::from_secs(3);
/// The longest a load takes here, which is drawn by the seed.
const LOAD_AT_MOST: Duration = Duration::from_millis(300);
/// What a turn costs besides the model: threads woken, lines kept.
const TURN_SLACK: Duration = Duration::from_millis(30);

/// A seeded source of numbers: the same seed, the same load.
struct Seeded(u64);

impl Seeded {
    fn next(&mut self) -> u64 {
        // xorshift64*, as Vigna gives it.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// From `low` to `high`, both in.
    fn between(&mut self, low: usize, high: usize) -> usize {
        low + (self.next() % (high - low + 1) as u64) as usize
    }

    fn one_in(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }
}

/// A run of writes: when it began, and the weight of each tenant's texts
/// in it, as (tenant, weight).
type WriteRun = (Instant, Vec<(usize, usize)>);

/// What the stand-in was asked for, in all.
#[derive(Default)]
struct Asked {
    texts: AtomicUsize,
    runs: AtomicUsize,
    loads: AtomicUsize,
    /// The longest run, as the stand-in took over it.
    longest_run: Mutex<Duration>,
    /// Each run of writes.
    write_runs: Mutex<Vec<WriteRun>>,
    /// Whether the model has stopped: it answers nothing from then on.
    stopped: AtomicBool,
}

fn stand_in(asked: Arc<Asked>, load_takes: Duration) -> Load {
    Box::new(move || {
        asked.loads.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(load_takes);
        let asked = asked.clone();
        Ok(Box::new(move |texts: Vec<&str>| {
            if asked.stopped.load(Ordering::SeqCst) {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
            let began = Instant::now();
            asked.runs.fetch_add(1, Ordering::SeqCst);
            asked.texts.fetch_add(texts.len(), Ordering::SeqCst);
            // A write's text names its tenant first, as "t3 ...".
            let of_writes: Vec<(usize, usize)> = texts
                .iter()
                .filter_map(|text| {
                    let tenant = text.strip_prefix('t')?.split(' ').next()?;
                    Some((tenant.parse().ok()?, weight(text)))
                })
                .collect();
            if !of_writes.is_empty() {
                asked.write_runs.lock().unwrap().push((began, of_writes));
            }
            assert!(texts
                .iter()
                .all(|text| text.chars().count() <= LONGEST_TEXT));
            let read: usize = texts.iter().map(|text| text.len().min(READS)).sum();
            let cost = EACH_TEXT * u32::try_from(texts.len()).unwrap()
                + EACH_BYTE * u32::try_from(read).unwrap();
            std::thread::sleep(cost);
            let took = began.elapsed();
            let mut longest = asked.longest_run.lock().unwrap();
            *longest = (*longest).max(took);
            drop(longest);
            assert!(
                !texts.iter().any(|text| text.starts_with("panic")),
                "told to"
            );
            if texts.iter().any(|text| text.starts_with("fail")) {
                return Err("told to".to_string());
            }
            Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
        }) as Model)
    })
}

/// What a tenant does for the length of a run.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Role {
    /// Searches, one after another, with so many searchers.
    Searches(usize),
    /// Writes in bulk: writes of so many texts, one after another.
    Bulk(usize),
    /// Writes long texts: writes of a few texts of so many kilobytes.
    Long(usize),
    /// Writes small writes, one to four texts, as fast as it can.
    Flood,
    /// Every write has a text the model fails on.
    Poison,
    /// Every write has a text the model panics on.
    Panics,
}

impl Role {
    fn healthy(self) -> bool {
        !matches!(self, Self::Poison | Self::Panics)
    }
}

/// One call a tenant made, and how it went.
#[derive(Debug)]
struct Call {
    tenant: usize,
    role: Role,
    purpose: Purpose,
    texts: usize,
    /// How many texts the tenant had waiting when this was made.
    ahead: usize,
    began: Instant,
    took: Duration,
    outcome: Result<(), EmbedError>,
}

/// Why a seed's run broke a promise, if it did.
type Broken = Vec<String>;

/// Run one seed: the calls made, what was asked of the model, and how
/// long the model takes over a load.
async fn run(seed: u64, stops: bool) -> (Vec<Call>, Arc<Asked>, HashMap<usize, Role>) {
    let mut seeded = Seeded(seed.max(1));
    let times = Times {
        run_for: RUN_FOR,
        held_at_most: HELD_AT_MOST,
        loading: Duration::from_secs(10),
        first_backoff: Duration::from_millis(50),
        longest_backoff: Duration::from_millis(400),
    };
    let asked = Arc::new(Asked::default());
    let load_takes =
        Duration::from_millis(seeded.between(0, LOAD_AT_MOST.as_millis() as usize) as u64);
    let room = Room::new(1, Duration::MAX, None, times);
    let local =
        Arc::new(Local::spawn("stand-in", stand_in(asked.clone(), load_takes), room).unwrap());
    let embedding = vellev_core::Embedding {
        provider: vellev_core::EmbeddingProvider::Local,
        model: "stand-in".into(),
        dimensions: 1,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    };
    let embedder = Embedder::new(local, embedding)
        .answering_within(QUERY_WITHIN, STALLED, Duration::from_millis(1))
        .holding_at_most(HELD_AT_MOST);
    let embedder = Arc::new(embedder);

    // Two to six tenants, each with a role; always at least one who
    // searches and one who writes in bulk.
    let tenants = seeded.between(2, 6);
    let mut roles = HashMap::new();
    roles.insert(0, Role::Searches(if seeded.one_in(2) { 8 } else { 40 }));
    roles.insert(1, Role::Bulk(seeded.between(1, 1_000)));
    for tenant in 2..tenants {
        let role = match seeded.between(0, 6) {
            0 => Role::Searches(seeded.between(1, 8)),
            1 => Role::Bulk(seeded.between(1, 300)),
            2 => Role::Long(seeded.between(10, 1_000)),
            3 => Role::Flood,
            4 => Role::Poison,
            5 => Role::Panics,
            _ => Role::Bulk(seeded.between(1, 30)),
        };
        roles.insert(tenant, role);
    }

    let calls: Arc<Mutex<Vec<Call>>> = Arc::default();
    let waiting: Arc<Mutex<HashMap<usize, usize>>> = Arc::default();
    let going = Arc::new(AtomicBool::new(true));
    let mut tasks = Vec::new();
    for (&tenant, &role) in &roles {
        let workers = match role {
            Role::Searches(searchers) => searchers,
            Role::Flood => 4,
            _ => 1,
        };
        for worker in 0..workers {
            let (embedder, calls, waiting, going) = (
                embedder.clone(),
                calls.clone(),
                waiting.clone(),
                going.clone(),
            );
            let mut seeded = Seeded(seed ^ ((tenant as u64) << 32) ^ (worker as u64 + 1));
            tasks.push(tokio::spawn(async move {
                let name = format!("tenant {tenant}");
                let mut nth = 0;
                while going.load(Ordering::SeqCst) {
                    nth += 1;
                    let (purpose, texts): (Purpose, Vec<String>) = match role {
                        Role::Searches(_) => (
                            Purpose::Query,
                            vec![format!("search {tenant} {worker} {nth}")],
                        ),
                        Role::Bulk(size) => (
                            Purpose::Document,
                            (0..size).map(|n| format!("t{tenant} w{nth} {n}")).collect(),
                        ),
                        Role::Long(kilobytes) => (
                            Purpose::Document,
                            (0..seeded.between(1, 4))
                                .map(|n| {
                                    format!("t{tenant} w{nth} {n} {}", "x".repeat(kilobytes * 1024))
                                })
                                .collect(),
                        ),
                        Role::Flood => (
                            Purpose::Document,
                            (0..seeded.between(1, 4))
                                .map(|n| format!("t{tenant} w{nth} {n}"))
                                .collect(),
                        ),
                        Role::Poison | Role::Panics => {
                            let word = if role == Role::Poison {
                                "fail"
                            } else {
                                "panic"
                            };
                            let mut texts: Vec<String> = (0..seeded.between(2, 12))
                                .map(|n| format!("t{tenant} w{nth} {n}"))
                                .collect();
                            texts[1] = format!("{word} {tenant} {nth}");
                            (Purpose::Document, texts)
                        }
                    };
                    let ahead = {
                        let mut waiting = waiting.lock().unwrap();
                        let ahead = *waiting.get(&tenant).unwrap_or(&0);
                        *waiting.entry(tenant).or_default() += texts.len();
                        ahead
                    };
                    let began = Instant::now();
                    let outcome = match purpose {
                        Purpose::Query => embedder.query_for(&name, &texts[0]).await.map(drop),
                        Purpose::Document => embedder.documents_for(&name, &texts).await.map(drop),
                    };
                    let took = began.elapsed();
                    *waiting.lock().unwrap().get_mut(&tenant).unwrap() -= texts.len();
                    calls.lock().unwrap().push(Call {
                        tenant,
                        role,
                        purpose,
                        texts: texts.len(),
                        ahead,
                        began,
                        took,
                        outcome,
                    });
                }
            }));
        }
    }
    tokio::time::sleep(RUNS_FOR).await;
    if stops {
        asked.stopped.store(true, Ordering::SeqCst);
        tokio::time::sleep(STALLED * 4 + HELD_AT_MOST).await;
    }
    going.store(false, Ordering::SeqCst);
    let done = futures::future::join_all(tasks);
    tokio::time::timeout(Duration::from_secs(30), done)
        .await
        .expect("every call is answered");
    let calls = std::mem::take(&mut *calls.lock().unwrap());
    (calls, asked, roles)
}

/// Whether a failure is the prompt refusal of a write the model would
/// not come to soon.
fn refused(outcome: &Result<(), EmbedError>) -> bool {
    matches!(outcome, Err(EmbedError::Failed { reason, .. }) if reason.contains("more waiting for it than it will come to soon"))
}

/// The promises, checked against what one seed's run came to.
fn check(seed: u64, calls: &[Call], asked: &Asked, roles: &HashMap<usize, Role>) -> Broken {
    let mut broken = Vec::new();
    let searches: Vec<&Call> = calls
        .iter()
        .filter(|call| call.purpose == Purpose::Query)
        .collect();
    let writes: Vec<&Call> = calls
        .iter()
        .filter(|call| call.purpose == Purpose::Document)
        .collect();

    // 1. No healthy write fails for others' load: a healthy write is
    //    made, or refused at once as one the model would not come to
    //    soon, and then within the time a write is held and a run or
    //    two.
    for call in writes.iter().filter(|call| call.role.healthy()) {
        match &call.outcome {
            Ok(()) => {}
            outcome if refused(outcome) => {
                let promptly = HELD_AT_MOST + RUN_AT_MOST * 4;
                if call.took > promptly {
                    broken.push(format!(
                        "{call:?}: refused, but not promptly ({promptly:?})"
                    ));
                }
            }
            Err(error) => broken.push(format!("{call:?}: a healthy write failed: {error}")),
        }
    }
    // 2. A healthy write is done within so many turns: its own tenant's
    //    texts ahead of it and its own, a tenant's share of a run at a
    //    time; each turn one run of documents and one of queries, and a
    //    model loaded again for each tenant whose texts make it panic;
    //    the stall allowance; and a little for the machine.
    let failing = u32::try_from(roles.values().filter(|role| !role.healthy()).count()).unwrap();
    for call in writes
        .iter()
        .filter(|call| call.role.healthy() && call.outcome.is_ok())
    {
        let own = call.ahead + call.texts;
        let share = (RUN / roles.len()).max(1);
        let turns = u32::try_from(own / share + 2).unwrap();
        let bound = (RUN_AT_MOST * 2 + LOAD_AT_MOST * failing + TURN_SLACK) * turns
            + STALLED
            + LOAD_AT_MOST
            + Duration::from_millis(60);
        if call.took > bound {
            broken.push(format!("{call:?}: done, but not within {bound:?}"));
        }
    }
    // 3. A search waits for the run under way, a run of documents, and
    //    its own run, and no more: three runs, with the model loaded in
    //    between, at first or again after a panic, which may be twice in
    //    a turn, once for the run a panicking tenant has alone and once
    //    after it. Its own tenant's searches ahead of it are its own:
    //    each run's worth of them is one more such turn. And a little
    //    for the machine.
    for call in &searches {
        if let Err(error) = &call.outcome {
            broken.push(format!("{call:?}: a search failed: {error}"));
        }
        let turns = u32::try_from(call.ahead / RUN + 2).unwrap();
        let bound =
            (RUN_AT_MOST * 2 + LOAD_AT_MOST * 2 + TURN_SLACK) * turns + Duration::from_millis(60);
        if call.took > bound {
            broken.push(format!("{call:?}: a search waited more than {bound:?}"));
        }
    }
    // 5. Nothing is made for nothing: the model was asked for each text
    //    of a write that was made once, and for failed runs' texts again
    //    by halves, at most seven askings a failing write.
    let made: usize = calls
        .iter()
        .filter(|call| call.outcome.is_ok())
        .map(|call| call.texts)
        .sum();
    let failing = writes.iter().filter(|call| !call.role.healthy()).count();
    let asked_texts = asked.texts.load(Ordering::SeqCst);
    if asked_texts < made || asked_texts > made + failing * RUN * 7 {
        broken.push(format!(
            "{asked_texts} texts asked of the model for {made} made and {failing} failing writes"
        ));
    }
    if !searches.is_empty()
        && writes
            .iter()
            .filter(|call| call.role.healthy())
            .all(|call| call.outcome.is_err())
    {
        broken.push("no healthy write was made at all".into());
    }
    // 6. Each tenant has as much of the model's time as any other that
    //    wants it: over any dozen runs of writes, a healthy tenant with a
    //    write waiting throughout had its share of the runs' weight, the
    //    weight shared equally among the tenants that had a write waiting
    //    at any time in them, give or take half.
    let write_runs = asked.write_runs.lock().unwrap();
    for window in write_runs.chunks_exact(12) {
        let (from, to) = (window[0].0, window[window.len() - 1].0);
        let total: usize = window.iter().flat_map(|(_, of)| of).map(|(_, w)| w).sum();
        let pending = |call: &&&Call, from: Instant, to: Instant| {
            call.role.healthy() && call.began <= from && call.began + call.took >= to
        };
        let present: HashSet<usize> = writes
            .iter()
            .filter(|call| pending(call, to, from))
            .map(|call| call.tenant)
            .collect();
        let throughout: HashSet<usize> = writes
            .iter()
            .filter(|call| pending(call, from, to))
            .map(|call| call.tenant)
            .collect();
        for tenant in throughout {
            let had: usize = window
                .iter()
                .flat_map(|(_, of)| of)
                .filter(|(of, _)| *of == tenant)
                .map(|(_, w)| w)
                .sum();
            let share = total / present.len().max(1);
            if had * 2 < share {
                broken.push(format!(
                    "tenant {tenant} had {had} of {total} weight over a dozen runs it waited \
                     through, with {} tenants present: less than half its share of {share}",
                    present.len()
                ));
            }
        }
    }
    drop(write_runs);
    if !broken.is_empty() {
        let mut by_tenant: Vec<(usize, usize, usize)> = roles
            .keys()
            .map(|tenant| {
                let of = calls.iter().filter(|call| call.tenant == *tenant);
                let (made, failed) = of.fold((0, 0), |(made, failed), call| match call.outcome {
                    Ok(()) => (made + 1, failed),
                    Err(_) => (made, failed + 1),
                });
                (*tenant, made, failed)
            })
            .collect();
        by_tenant.sort();
        broken.insert(
            0,
            format!("seed {seed}: {roles:?}; (tenant, made, failed): {by_tenant:?}"),
        );
    }
    broken
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tenants_at_once_are_kept_what_they_are_promised_whatever_the_seed() {
    let seeds: u64 = std::env::var("HARNESS_SEEDS")
        .ok()
        .and_then(|seeds| seeds.parse().ok())
        .unwrap_or(6);
    let mut broken_in = Vec::new();
    for seed in 1..=seeds {
        let (calls, asked, roles) = run(seed, false).await;
        let broken = check(seed, &calls, &asked, &roles);
        let searches = calls
            .iter()
            .filter(|call| call.purpose == Purpose::Query)
            .count();
        let longest = *asked.longest_run.lock().unwrap();
        println!(
            "seed {seed}: {} tenants, {} calls ({searches} searches), {} runs, longest {longest:?}, {} loads, {} broken",
            roles.len(),
            calls.len(),
            asked.runs.load(Ordering::SeqCst),
            asked.loads.load(Ordering::SeqCst),
            broken.len()
        );
        for line in &broken {
            println!("  {line}");
        }
        if !broken.is_empty() {
            broken_in.push(seed);
        }
    }
    assert!(
        broken_in.is_empty(),
        "promises broken for seeds {broken_in:?}"
    );
}

/// 4. A model that stops is given up on by every write and search within
///    a stated time: a search in its own time, a write within what it
///    was told it may wait and as long again as it may get nowhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_that_stops_is_given_up_on_by_everyone_within_the_stated_time() {
    let (calls, asked, roles) = run(7, true).await;
    assert!(asked.stopped.load(Ordering::SeqCst));
    // The calls still out when the model stopped are those that took
    // longest: none took longer than the time a write is held, the
    // reckoned wait four times over, and the stall, together.
    let longest = calls.iter().map(|call| call.took).max().unwrap();
    let bound = HELD_AT_MOST * 4 + STALLED * 2 + Duration::from_secs(1);
    assert!(longest <= bound, "{longest:?} > {bound:?}: {roles:?}");
    let given_up: Vec<&Call> = calls
        .iter()
        .filter(|call| {
            call.purpose == Purpose::Document
                && call.role.healthy()
                && call.outcome.is_err()
                && !refused(&call.outcome)
        })
        .collect();
    assert!(!given_up.is_empty(), "no write was given up on");
    for call in given_up {
        let said = format!("{:?}", call.outcome);
        assert!(
            said.contains("nothing from the model")
                || said.contains("did not come to it")
                || said.contains("no answer within"),
            "{said}"
        );
    }
}
