use std::collections::HashMap;
use std::future::Future;
use std::io::ErrorKind;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, ModelInfo, TextEmbedding};
use hf_hub::api::sync::ApiError as FetchError;
use tokio::sync::{mpsc, oneshot};

use crate::line::{cut, weight, Job, Line, Shares};
use crate::{EmbedError, Progress, Provider, Purpose, Result};

/// The most texts in one run of the model, which bounds the memory a run
/// takes.
pub(crate) const RUN: usize = 32;
/// About how long a run of documents is to take. A run cannot be
/// interrupted, so this is how long a query is kept waiting by
/// documents: it goes ahead of them all, but not of the run under way.
/// Thirty-two long texts take seconds on a slow machine, so a run is of
/// as many texts as the runs before it say take about this long.
pub(crate) const RUN_FOR: Duration = Duration::from_secs(1);
/// Texts in the first run of documents after a model is loaded, when
/// nothing is yet known of how long its runs take.
const FIRST_RUN: usize = 4;
/// Callers that can be waiting for the model before others wait to join
/// them.
const WAITING: usize = 256;

/// A loaded model: a vector for each text, or why not.
pub(crate) type Model = Box<dyn FnMut(Vec<&str>) -> std::result::Result<Vec<Vec<f32>>, String>>;
pub(crate) type Load = Box<dyn FnMut() -> Result<Model> + Send>;

/// The local models a node has loaded, each the better part of a
/// gigabyte while it runs. It has no more loaded at once than it is set
/// up to, and lets go of one that has not been used for as long as it is
/// set up to keep it.
///
/// So the models that have the places are those in use, as the node
/// counts use: asked for within that time. One more than there are
/// places for is refused for now, and not loaded in the place of one of
/// them: collections that between them embed with more models than the
/// node holds would otherwise have it load and let go of one after
/// another, for as long as they were written to and searched.
///
/// The node's own model, if it has one, is not among those that wait for
/// a place or lose one. A place is kept for it whatever else is loaded,
/// and once loaded it stays: the collections that embed with other
/// models can never keep those that embed with the node's own from it.
#[derive(Debug)]
pub(crate) struct Room {
    most: usize,
    idle: Duration,
    own: Option<String>,
    times: Times,
    loaded: Mutex<Vec<String>>,
}

/// The times a node's local models keep to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Times {
    /// About how long each run is to take.
    pub(crate) run_for: Duration,
    /// The longest a write is kept waiting for its turn: one the model
    /// is reckoned not to come to within this is refused at once.
    pub(crate) held_at_most: Duration,
    /// How long a model is given to load, fetched first if it must be.
    pub(crate) loading: Duration,
    /// How long a tenant whose texts the model failed on is kept from
    /// it: so long the first time, twice as long each time after, and
    /// no longer than the longest.
    pub(crate) first_backoff: Duration,
    pub(crate) longest_backoff: Duration,
}

/// How long a model is given to load. One not yet fetched is some four
/// hundred megabytes to fetch first, which at a megabyte a second is
/// seven minutes.
pub(crate) const LOADING: Duration = Duration::from_secs(10 * 60);

impl Default for Times {
    fn default() -> Self {
        Self {
            run_for: RUN_FOR,
            held_at_most: crate::DOCUMENTS_HELD_AT_MOST,
            loading: LOADING,
            first_backoff: Duration::from_secs(1),
            longest_backoff: Duration::from_secs(60),
        }
    }
}

/// A loaded model's place among those a node has loaded, kept until it
/// is let go.
#[derive(Debug)]
struct Place {
    room: Arc<Room>,
    model: String,
}

impl Room {
    /// Room for `most` models at once, and one at the least, each let go
    /// when it has not been used for `idle`: but for `own`, the node's
    /// own model, which has one of the places kept for it and is not let
    /// go. Its models keep to `times`.
    pub(crate) fn new(most: usize, idle: Duration, own: Option<String>, times: Times) -> Arc<Self> {
        Arc::new(Self {
            most: most.max(1),
            idle,
            own,
            times,
            loaded: Mutex::default(),
        })
    }

    fn own(&self, model: &str) -> bool {
        self.own.as_deref() == Some(model)
    }

    /// How long `model` is kept loaded with nothing asked of it.
    fn kept_idle(&self, model: &str) -> Duration {
        match self.own(model) {
            true => Duration::MAX,
            false => self.idle,
        }
    }

    /// A place for `model`, or that there is none for now: which passes
    /// when one of those loaded has gone unused long enough. The node's
    /// own model always has its place.
    fn place(self: &Arc<Self>, model: &str) -> Result<Place> {
        let mut loaded = self.loaded.lock().expect("room lock");
        if !self.own(model) {
            let others: Vec<&str> = loaded
                .iter()
                .map(String::as_str)
                .filter(|loaded| !self.own(loaded))
                .collect();
            let places = self.most - usize::from(self.own.is_some());
            let beside = match &self.own {
                Some(own) => format!(" beside its own, {own}"),
                None => String::new(),
            };
            if places == 0 {
                return Err(failure(
                    model,
                    format!("this node has a place for no local model{beside}"),
                ));
            }
            if others.len() >= places {
                return Err(failure(
                    model,
                    format!(
                        "this node has no more than {places} local models loaded at \
                         once{beside}, and has {} in use: one is let go when it has not \
                         been used for {:?}",
                        others.join(" and "),
                        self.idle
                    ),
                ));
            }
        }
        loaded.push(model.to_string());
        Ok(Place {
            room: self.clone(),
            model: model.to_string(),
        })
    }

    /// How many models there is room for, how long one is kept unused,
    /// and which is the node's own.
    #[cfg(test)]
    pub(crate) fn set_up(&self) -> (usize, Duration, Option<&str>) {
        (self.most, self.idle, self.own.as_deref())
    }

    /// The models that have a place, in the order they took them.
    #[cfg(test)]
    pub(crate) fn loaded(&self) -> Vec<String> {
        self.loaded.lock().expect("room lock").clone()
    }
}

impl Drop for Place {
    fn drop(&mut self) {
        let mut loaded = self.room.loaded.lock().expect("room lock");
        if let Some(at) = loaded.iter().position(|model| model == &self.model) {
            loaded.remove(at);
        }
    }
}

/// A model that is loaded, and its place among those that are: let go
/// together.
struct Loaded {
    model: Model,
    _place: Place,
}

/// A model run in this process on a thread of its own, so that inference
/// never holds up the async runtime. Waiting callers' texts are run
/// together, queries and documents by turns, each run shared out among
/// the tenants that wait as a [`Line`] is, and a caller that has gone
/// away is dropped before its next run. The model is fetched into the directory
/// models are [`kept_in`] and loaded when first needed, if there is
/// [`Room`] for it, and again after a failure or after it was let go for
/// not being used.
#[derive(Debug)]
pub struct Local {
    model: String,
    jobs: mpsc::Sender<Job>,
    /// Until when the model is given to load, while it is loading.
    loading: Arc<Mutex<Option<Instant>>>,
}

impl Local {
    /// `model` is a Hugging Face name such as `BAAI/bge-base-en-v1.5`. It
    /// is kept loaded once used, whatever else is.
    pub fn new(model: &str, dimensions: u32, cache: &Path) -> Result<Self> {
        let room = Room::new(1, Duration::MAX, None, Times::default());
        Self::among(known(model, dimensions)?, cache, room)
    }

    /// `listed`, loaded when there is a place for it in `room`.
    pub(crate) fn among(listed: &Listed, cache: &Path, room: Arc<Room>) -> Result<Self> {
        let known = listed.model.clone();
        let cache = cache.to_path_buf();
        let kept = kept_in(&cache, std::env::var("HF_HOME").ok());
        let threads = std::thread::available_parallelism().map_or(1, |cores| cores.get());
        let name = listed.model_code.clone();
        Self::spawn(
            &listed.model_code,
            Box::new(move || {
                let options = InitOptions::new(known.clone())
                    .with_cache_dir(cache.clone())
                    .with_intra_threads(threads.saturating_sub(1).max(1))
                    .with_show_download_progress(false);
                let mut session = TextEmbedding::try_new(options)
                    .map_err(|error| unloaded(&name, &kept, &error))?;
                Ok(Box::new(move |texts: Vec<&str>| {
                    let together = texts.len();
                    session
                        .embed(texts, Some(together))
                        .map_err(|error| format!("{error:#}"))
                }) as Model)
            }),
            room,
        )
    }

    pub(crate) fn spawn(model: &str, load: Load, room: Arc<Room>) -> Result<Self> {
        let (jobs, waiting) = mpsc::channel(WAITING);
        let name = model.to_string();
        let loading = Arc::new(Mutex::new(None));
        let said = loading.clone();
        std::thread::Builder::new()
            .name(format!("embed {model}"))
            .spawn(move || run(&name, load, &room, waiting, said))
            .map_err(|error| failure(model, format!("no thread to run it on: {error}")))?;
        Ok(Self {
            model: model.to_string(),
            jobs,
            loading,
        })
    }
}

/// The directory models are fetched to and loaded from: the one the node
/// is given, unless `HF_HOME` is set, which what fetches them puts first.
fn kept_in(cache: &Path, hf_home: Option<String>) -> PathBuf {
    hf_home.map_or_else(|| cache.to_path_buf(), PathBuf::from)
}

/// Why `model`, kept in `kept`, could not be loaded. A directory the node
/// may not write keeps it from ever loading until the node is set up
/// otherwise: that is a failure of the fetcher's own to make or write a
/// file, found wherever it lies under what was said of it. Anything else
/// may pass, such as a network the model cannot be fetched over just now,
/// which a node that is never to have one cannot be told from. So may a
/// fetch given up after several tries, were tries ever asked for: by then
/// the file was open, and what failed was the fetching.
fn unloaded(model: &str, kept: &Path, error: &fastembed::Error) -> EmbedError {
    let barred = error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<FetchError>())
        .any(|cause| match cause {
            FetchError::IoError(cause) => matches!(
                cause.kind(),
                ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem
            ),
            _ => false,
        });
    match barred {
        true => EmbedError::unusable(
            model,
            vellev_core::ModelFault::Setup,
            format!("the node may not write {}: {error:#}", kept.display()),
        ),
        false => failure(model, format!("{error:#}")),
    }
}

/// A failure of `model` that may pass.
fn failure(model: &str, reason: impl std::fmt::Display) -> EmbedError {
    EmbedError::failed(model, reason)
}

/// A model as this release's fastembed lists it: which it is, the name
/// it is listed by, and its size.
pub(crate) type Listed = ModelInfo<EmbeddingModel>;

/// The models that can be run, which are those this release's fastembed
/// lists.
static LISTED: LazyLock<Vec<Listed>> = LazyLock::new(TextEmbedding::list_supported_models);

/// The model of this name, whichever organisation's copy of it is named
/// and however the name is cased: one model, however many ways it is
/// asked for.
pub(crate) fn named(model: &str) -> Option<&'static Listed> {
    let name = |code: &str| code.rsplit('/').next().unwrap_or(code).to_lowercase();
    let asked = name(model);
    LISTED
        .iter()
        .find(|listed| name(&listed.model_code) == asked)
}

/// The model of this name, at the size it has: a name that is not listed,
/// or a listed model at another size, is no model of this release, built
/// however it is.
pub(crate) fn known(model: &str, dimensions: u32) -> Result<&'static Listed> {
    let found = named(model)
        .ok_or_else(|| EmbedError::NoSuchModel(format!("{model} cannot be run locally")))?;
    if found.dim != dimensions as usize {
        return Err(EmbedError::NoSuchModel(format!(
            "{model} has {} dimensions, not {dimensions}",
            found.dim
        )));
    }
    Ok(found)
}

/// No job came in the time waited.
struct Idle;

/// The next job, waited for on this thread for no longer than `within`:
/// or `None` when no one is left to send one.
fn next_within(
    waiting: &mut mpsc::Receiver<Job>,
    within: Duration,
) -> std::result::Result<Option<Job>, Idle> {
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    // A time too long to be counted from now is one that never comes.
    let until = Instant::now().checked_add(within);
    let mut next = std::pin::pin!(waiting.recv());
    loop {
        if let Poll::Ready(job) = next.as_mut().poll(&mut context) {
            return Ok(job);
        }
        match until.map(|until| until.saturating_duration_since(Instant::now())) {
            Some(left) if left.is_zero() => return Err(Idle),
            Some(left) => std::thread::park_timeout(left),
            None => std::thread::park(),
        }
    }
}

/// How much goes into a run, to take about `run_for`, when each unit of
/// [`weight`] took `each` in the runs before it: as much as a run can be
/// of if runs are to take however long they take or nothing is yet
/// known, and one text's worth at the least.
fn weight_of_a_run(run_for: Duration, each: Option<Duration>) -> usize {
    match each {
        _ if run_for == Duration::MAX => usize::MAX,
        None => usize::MAX,
        Some(each) if each.is_zero() => usize::MAX,
        Some(each) => usize::try_from(run_for.as_nanos() / each.as_nanos())
            .unwrap_or(usize::MAX)
            .max(1),
    }
}

/// How long a tenant whose texts have failed the model `strikes` times
/// running is kept from it: the first backoff, doubled for each failure
/// after the first, and never longer than the longest.
fn backoff(strikes: u32, times: &Times) -> Duration {
    let doublings = strikes.saturating_sub(1).min(16);
    times
        .first_backoff
        .saturating_mul(1 << doublings)
        .min(times.longest_backoff)
}

/// How long the model took in its last runs, by which the next is sized
/// and what waits is reckoned.
#[derive(Debug, Default)]
struct Pace {
    /// Over each unit of weight, in the last run and in the one before.
    lately: [Option<Duration>; 2],
    /// Over each text of the last run of documents, on the clock:
    /// loading and asking again included.
    each_text: Option<Duration>,
}

impl Pace {
    /// How many texts, and how much weight, the next run is of, to take
    /// about `run_for`: by the slower of the last two runs, so that one
    /// run of short texts does not make the next too long. Few texts
    /// when nothing is yet known.
    fn run(&self, run_for: Duration) -> (usize, usize) {
        let slower = self.lately.into_iter().flatten().max();
        let texts = match slower {
            None if run_for != Duration::MAX => FIRST_RUN,
            _ => RUN,
        };
        (texts, weight_of_a_run(run_for, slower))
    }

    /// A run took `each` over each unit of weight.
    fn took(&mut self, each: Duration) {
        self.lately = [Some(each), self.lately[0]];
    }

    /// Nothing is known of how long runs take any more: the model was
    /// let go, and the one loaded next may run on another machine's
    /// worth of cores.
    fn forget(&mut self) {
        *self = Self::default();
    }
}

/// A tenant whose texts the model has failed on: how many times running,
/// and until when its writes are kept from the model.
struct Trouble {
    strikes: u32,
    until: Instant,
}

/// What came of a run.
#[derive(Default)]
struct Ran {
    /// Whether the model made anything.
    made: bool,
    /// How long it took over each unit of weight and each text, if it
    /// made the run as it was first asked.
    each: Option<(Duration, Duration)>,
    /// The tenants a job of whose the model failed on, each with how.
    failed: Vec<(Arc<str>, EmbedError)>,
    /// The tenants that shared a run the model failed, which may be any
    /// one's doing: each is to be run by itself next, in its turn.
    suspect: Vec<Arc<str>>,
}

fn run(
    name: &str,
    load: Load,
    room: &Arc<Room>,
    mut waiting: mpsc::Receiver<Job>,
    loading: Arc<Mutex<Option<Instant>>>,
) {
    let times = room.times;
    let mut asking = Asking {
        name,
        load,
        among: room,
        model: None,
        loading,
        times,
        just_loaded: false,
    };
    let (mut queries, mut documents) = (Line::default(), Line::default());
    let mut pace = Pace::default();
    let mut troubles: HashMap<Arc<str>, Trouble> = HashMap::new();
    // Whether the last run was of queries: when both kinds wait, they
    // take turns.
    let mut queries_ran_last = false;
    let queue = |job: Job, queries: &mut Line, documents: &mut Line| match job.purpose {
        Purpose::Query => queries.push(job, Instant::now()),
        Purpose::Document => documents.push(job, Instant::now()),
    };
    let kept = |troubles: &HashMap<Arc<str>, Trouble>, tenant: &str, now: Instant| {
        troubles
            .get(tenant)
            .is_some_and(|trouble| trouble.until > now)
    };
    let any_to_write = |documents: &Line, troubles: &HashMap<Arc<str>, Trouble>, now| {
        let mut tenants = documents.queues.iter();
        tenants.any(|queue| !kept(troubles, &queue.tenant, now))
    };
    loop {
        let now = Instant::now();
        if queries.is_empty() && !any_to_write(&documents, &troubles, now) {
            // With nothing to run, a model that is loaded is kept for as
            // long as the node keeps one unused, and then let go, and
            // with it its place: it is loaded again when next asked for,
            // if there is room then. With only the writes of tenants
            // kept from the model to run, it waits for the first of
            // those to be let at it.
            let soonest = troubles.values().map(|trouble| trouble.until).min();
            let within = match (documents.is_empty(), &asking.model, soonest) {
                (false, _, Some(soonest)) => soonest.saturating_duration_since(now),
                (_, Some(_), _) => room.kept_idle(name),
                (_, None, _) => Duration::MAX,
            };
            match next_within(&mut waiting, within) {
                Ok(Some(job)) => queue(job, &mut queries, &mut documents),
                Ok(None) => return,
                Err(Idle) if documents.is_empty() => {
                    asking.model = None;
                    pace.forget();
                    troubles.clear();
                    tracing::info!(
                        model = name,
                        "a local embedding model not used for {:?} is let go",
                        within
                    );
                    continue;
                }
                Err(Idle) => {}
            }
        }
        // A model that is to be loaded is said to be loading before the
        // jobs that came meanwhile are taken in: so whoever sends one
        // from now on is told, and whoever sent one before is taken in
        // here and told when the load begins. None falls between.
        if asking.model.is_none() {
            let until = Instant::now() + times.loading;
            *asking.loading.lock().expect("loading lock") = Some(until);
        }
        while let Ok(job) = waiting.try_recv() {
            queue(job, &mut queries, &mut documents);
        }
        let now = Instant::now();
        queries.drop_gone(now);
        documents.drop_gone(now);
        // What the model will not come to soon is refused now, and not
        // kept to be given up on later.
        let alternating = if queries.is_empty() { 1 } else { 2 };
        if let Some(each) = pace.each_text {
            documents.refuse_held_too_long(name, each * alternating, times.held_at_most);
        }
        let to_write = any_to_write(&documents, &troubles, now);
        let run_queries = !(queries.is_empty() || (to_write && queries_ran_last));
        if !run_queries && !to_write {
            documents.tell(now, pace.each_text, |tenant| {
                troubles.get(tenant).map(|trouble| trouble.until)
            });
            continue;
        }
        // A tenant the model has failed on is let at it again by itself,
        // when its time comes: the others' texts are not in a run it may
        // fail.
        let again = documents
            .queues
            .iter()
            .find(|queue| {
                let trouble = troubles.get(&queue.tenant);
                trouble.is_some_and(|trouble| trouble.until <= now)
            })
            .map(|queue| queue.tenant.clone());
        let (texts, weighs) = pace.run(times.run_for);
        let shares = match run_queries {
            true => queries.shared_out(texts, weighs, |_| false),
            false => documents.shared_out(texts, weighs, |tenant| match &again {
                Some(again) => tenant != &**again,
                None => troubles.contains_key(tenant),
            }),
        };
        let (line, other) = match run_queries {
            true => (&mut queries, &documents),
            false => (&mut documents, &queries),
        };
        let ran = asking.advance(line, other, shares);
        if !run_queries {
            if let (Some(again), true, true) = (&again, ran.made, ran.failed.is_empty()) {
                troubles.remove(again);
            }
            // Those who shared a run the model failed are each run by
            // themselves next, in their turn, so that whoever's doing it
            // was is found out and no one else is failed for it.
            for tenant in &ran.suspect {
                troubles.entry(tenant.clone()).or_insert(Trouble {
                    strikes: 0,
                    until: now,
                });
            }
            for (tenant, cause) in &ran.failed {
                let trouble = troubles.entry(tenant.clone()).or_insert(Trouble {
                    strikes: 0,
                    until: now,
                });
                trouble.strikes += 1;
                let kept_for = backoff(trouble.strikes, &times);
                trouble.until = Instant::now() + kept_for;
                tracing::error!(
                    model = name,
                    tenant = %tenant,
                    failures = trouble.strikes,
                    %cause,
                    "the embedding model failed on a tenant's texts: its writes are made \
                     apart from others', and not again for {kept_for:?}"
                );
            }
        }
        // Any run the model made says how fast it is, queries' and
        // documents' alike: it is one model. A model loaded again after
        // a panic is the same model, and its pace is kept.
        if let Some((weight, text)) = ran.each {
            pace.took(weight);
            pace.each_text = Some(text);
        }
        queries_ran_last = run_queries;
        if std::mem::take(&mut asking.just_loaded) {
            // The time the load took was no one's getting nowhere.
            let now = Instant::now();
            queries.all_got_somewhere(now);
            documents.all_got_somewhere(now);
        }
        // Those who came while it ran wait in the lines too.
        while let Ok(job) = waiting.try_recv() {
            queue(job, &mut queries, &mut documents);
        }
        let now = Instant::now();
        let alternating = if queries.is_empty() { 1 } else { 2 };
        let each = pace.each_text.map(|each| each * alternating);
        documents.tell(now, each, |tenant| {
            troubles.get(tenant).map(|trouble| trouble.until)
        });
    }
}

/// What came of asking the model for texts.
enum Asked {
    /// There was no model to ask: it did not load, or there is no place
    /// for it.
    Unloaded(EmbedError),
    /// A vector for each text, and how long the model took over them:
    /// the model's own time, a load before it not counted.
    Made(Vec<Vec<f32>>, Duration),
    /// The model was asked, and failed.
    Failed(EmbedError),
}

/// What came of a job in a run: what was made for it; how it failed,
/// and whether that was the job's own doing; or nothing, the run having
/// failed with others in it.
enum Made {
    Vectors(Vec<Vec<f32>>),
    Failed(EmbedError, bool),
    Deferred,
}

/// The model, loaded or to be, and what it is asked through.
struct Asking<'a> {
    name: &'a str,
    load: Load,
    among: &'a Arc<Room>,
    model: Option<Loaded>,
    /// Until when the model is given to load, while it is loading: said
    /// to callers who come meanwhile.
    loading: Arc<Mutex<Option<Instant>>>,
    times: Times,
    /// Whether the model was loaded since this was last looked at: all
    /// who waited through the load have got somewhere by it.
    just_loaded: bool,
}

impl Asking<'_> {
    /// Load the model, if it is not loaded and there is a place for it
    /// among those the node has loaded, telling `waiting` first how long
    /// a load is given: a model that is loading is not one that has
    /// stopped, and no one is given up on for it. The place is taken
    /// first, and given up again should the model not load.
    fn load(&mut self, waiting: &dyn Fn(Duration)) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let (name, among, given) = (self.name, self.among, self.times.loading);
        *self.loading.lock().expect("loading lock") = Some(Instant::now() + given);
        waiting(given);
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            let place = among.place(name)?;
            Ok(Loaded {
                model: (self.load)()?,
                _place: place,
            })
        }));
        *self.loading.lock().expect("loading lock") = None;
        match loaded {
            Ok(Ok(loaded)) => {
                self.model = Some(loaded);
                self.just_loaded = true;
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(failure(name, "the model panicked")),
        }
    }

    /// Ask the model for `texts`, loading it first if it is not loaded.
    fn ask(&mut self, texts: Vec<&str>, waiting: &dyn Fn(Duration)) -> Asked {
        if let Err(error) = self.load(waiting) {
            return Asked::Unloaded(error);
        }
        let (name, asked) = (self.name, texts.len());
        let began = Instant::now();
        let ran = catch_unwind(AssertUnwindSafe(|| {
            let loaded = self.model.as_mut().expect("loaded just now");
            (loaded.model)(texts).map_err(|reason| failure(name, reason))
        }));
        match ran {
            Ok(Ok(vectors)) if vectors.len() == asked => Asked::Made(vectors, began.elapsed()),
            Ok(Ok(vectors)) => Asked::Failed(failure(
                name,
                format!("{} vectors for {asked} texts", vectors.len()),
            )),
            Ok(Err(error)) => Asked::Failed(error),
            Err(_) => {
                self.model = None;
                Asked::Failed(failure(name, "the model panicked"))
            }
        }
    }

    /// What the model makes for each of `jobs`, asked for together.
    ///
    /// A run the model fails is not the fault of every job in it, and
    /// finding whose it is must not keep everyone else waiting: a run
    /// of several jobs that fails is put off, and each tenant in it is
    /// run by itself in its turn after. One run alone that fails is
    /// that job's own doing. So a text the model cannot take fails the
    /// job it is in and no other, at the cost of one failed run for
    /// those who shared it, and the model loaded again once should it
    /// have panicked.
    ///
    /// How long the model took over the run, if it made it.
    fn made_for(
        &mut self,
        jobs: &[Vec<&str>],
        waiting: &dyn Fn(Duration),
    ) -> (Vec<Made>, Option<Duration>) {
        match self.ask(jobs.concat(), waiting) {
            Asked::Made(vectors, took) => {
                let mut vectors = vectors.into_iter();
                let each = jobs
                    .iter()
                    .map(|texts| Made::Vectors(vectors.by_ref().take(texts.len()).collect()));
                (each.collect(), Some(took))
            }
            // There being no model to ask is no job's doing.
            Asked::Unloaded(error) => {
                let unasked = jobs.iter().map(|_| Made::Failed(error.clone(), false));
                (unasked.collect(), None)
            }
            Asked::Failed(error) if jobs.len() == 1 => (vec![Made::Failed(error, true)], None),
            Asked::Failed(_) => (jobs.iter().map(|_| Made::Deferred).collect(), None),
        }
    }

    /// Run the model once, on as many texts of each job in `line` as its
    /// share of `shares`, and answer the callers it finishes or fails.
    /// Those with no share are left as they are. Should the model have
    /// to be loaded for it, all who wait in `line` and in the `other`
    /// are told so.
    fn advance(&mut self, line: &mut Line, other: &Line, shares: Shares) -> Ran {
        let (made, took, texts, weighs) = {
            let jobs = line.texts(&shares);
            if jobs.is_empty() {
                return Ran::default();
            }
            let texts: usize = jobs.iter().map(Vec::len).sum();
            let weighs: usize = jobs.iter().flatten().map(|text| weight(text)).sum();
            let waiting = |given: Duration| {
                for work in line.works().chain(other.works()) {
                    work.allowed(given);
                }
            };
            let (made, took) = self.made_for(&jobs, &waiting);
            (made, took, texts, weighs)
        };
        // Whose fault each job that was failed is, in their order; and
        // whose the jobs put off are.
        let failed_on: Vec<bool> = made
            .iter()
            .filter_map(|made| match made {
                Made::Failed(_, own_doing) => Some(*own_doing),
                _ => None,
            })
            .collect();
        let deferred = made.iter().any(|made| matches!(made, Made::Deferred));
        let suspect: Vec<Arc<str>> = match deferred {
            true => line.tenants_sharing(&shares),
            false => Vec::new(),
        };
        let made: Vec<Option<Result<Vec<Vec<f32>>>>> = made
            .into_iter()
            .map(|made| match made {
                Made::Vectors(vectors) => Some(Ok(vectors)),
                Made::Failed(error, _) => Some(Err(error)),
                Made::Deferred => None,
            })
            .collect();
        let any = made.iter().any(|made| matches!(made, Some(Ok(_))));
        let unmade = line.settle(shares, made, Instant::now());
        let per = |all: Duration, of: usize| all / u32::try_from(of.max(1)).unwrap_or(u32::MAX);
        Ran {
            made: any,
            each: took.map(|took| (per(took, weighs), per(took, texts))),
            failed: unmade
                .into_iter()
                .zip(failed_on)
                .filter(|(_, failed_on)| *failed_on)
                .map(|(unmade, _)| unmade)
                .collect(),
            suspect,
        }
    }
}

#[async_trait]
impl Provider for Local {
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        self.embedded(texts, purpose, "", None).await
    }

    fn says_how_it_gets_on(&self) -> bool {
        true
    }

    async fn embed_saying(
        &self,
        texts: Vec<String>,
        purpose: Purpose,
        tenant: &str,
        progress: Progress,
    ) -> Result<Vec<Vec<f32>>> {
        self.embedded(texts, purpose, tenant, Some(progress)).await
    }
}

impl Local {
    /// A vector for each text, made on the model's thread when `tenant`'s
    /// turn comes, with how long the model is given said through
    /// `progress`.
    async fn embedded(
        &self,
        texts: Vec<String>,
        purpose: Purpose,
        tenant: &str,
        progress: Option<Progress>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let stopped = || failure(&self.model, "the model's thread has stopped");
        let (reply, answer) = oneshot::channel();
        let says = progress.clone();
        let job = Job {
            texts: texts.into_iter().map(cut).collect(),
            purpose,
            tenant: tenant.into(),
            reply,
            progress,
        };
        self.jobs.send(job).await.map_err(|_| stopped())?;
        // A model that is loading is given as long as a load is: said
        // here, once the job is sent, since the model's thread says
        // nothing to a job it has not yet taken in.
        let loading = *self.loading.lock().expect("loading lock");
        if let (Some(until), Some(says)) = (loading, says) {
            says.allowed(until.saturating_duration_since(Instant::now()));
        }
        answer.await.map_err(|_| stopped())?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{channel, Receiver, Sender};

    use super::*;
    use crate::line::{HEAVIEST, LIGHTEST, LONGEST_TEXT, SMALL};

    /// Room for one model, kept however long it goes unused, whose runs
    /// are as long as they take.
    fn alone() -> Arc<Room> {
        Room::new(1, Duration::MAX, None, unpaced())
    }

    /// Runs as long as they take, and the other times as a node keeps
    /// them unless told otherwise.
    fn unpaced() -> Times {
        Times {
            run_for: Duration::MAX,
            ..Times::default()
        }
    }

    /// Runs of about `run_for`, and the other times as a node keeps them.
    fn paced(run_for: Duration) -> Times {
        Times {
            run_for,
            ..Times::default()
        }
    }

    /// A stand-in model that reports each run's texts and then waits to be
    /// let through. A text's vector is its length; "panic" panics and
    /// "fail" fails.
    struct Runs {
        ran: Receiver<Vec<String>>,
        go: Sender<()>,
        loads: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Runs {
        fn next(&self) -> Vec<String> {
            self.ran.recv_timeout(Duration::from_secs(10)).unwrap()
        }

        fn let_through(&self, runs: usize) {
            (0..runs).for_each(|_| self.go.send(()).unwrap());
        }

        fn none_within(&self, wait: Duration) -> bool {
            self.ran.recv_timeout(wait).is_err()
        }
    }

    fn local(failing_loads: usize) -> (Arc<Local>, Runs) {
        local_among("stand-in", failing_loads, alone())
    }

    /// As [`local`], by the name `model` and loaded when there is a place
    /// for it in `room`.
    fn local_among(model: &str, failing_loads: usize, room: Arc<Room>) -> (Arc<Local>, Runs) {
        local_loading(model, failing_loads, room, Duration::ZERO)
    }

    /// As [`local_among`], with a load that takes `load_takes`.
    fn local_loading(
        model: &str,
        failing_loads: usize,
        room: Arc<Room>,
        load_takes: Duration,
    ) -> (Arc<Local>, Runs) {
        let (report, ran) = channel();
        let (go, gate) = channel::<()>();
        let gate = Arc::new(std::sync::Mutex::new(gate));
        let mut failing = failing_loads;
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = loads.clone();
        let load: Load = Box::new(move || {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if failing > 0 {
                failing -= 1;
                return Err(failed("no model to be had"));
            }
            std::thread::sleep(load_takes);
            let (report, gate) = (report.clone(), gate.clone());
            Ok(Box::new(move |texts: Vec<&str>| {
                report
                    .send(texts.iter().map(|text| text.to_string()).collect())
                    .unwrap();
                gate.lock().unwrap().recv().unwrap();
                assert!(
                    !texts.iter().any(|text| text.starts_with("panic")),
                    "told to"
                );
                if texts.iter().any(|text| text.starts_with("fail")) {
                    return Err("told to".to_string());
                }
                if texts.contains(&"quote") {
                    return Err("401 for hf_abcdefghij0123456789ABCD".to_string());
                }
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        (
            Arc::new(Local::spawn(model, load, room).unwrap()),
            Runs { ran, go, loads },
        )
    }

    fn texts(prefix: &str, count: usize) -> Vec<String> {
        (0..count).map(|n| format!("{prefix}{n}")).collect()
    }

    /// Wait for `jobs` jobs to be waiting for a model that is kept at a
    /// run meanwhile, and so takes none of them yet.
    async fn waiting(local: &Arc<Local>, jobs: usize) {
        until(|| WAITING - local.jobs.capacity() == jobs).await;
    }

    /// Begin a job. Should the model never answer it, whoever waits on
    /// it fails rather than hangs.
    fn start(
        local: &Arc<Local>,
        texts: Vec<String>,
        purpose: Purpose,
    ) -> tokio::task::JoinHandle<Result<Vec<Vec<f32>>>> {
        let local = local.clone();
        tokio::spawn(async move {
            let answered = local.embed(texts, purpose);
            let answered = tokio::time::timeout(Duration::from_secs(30), answered).await;
            answered.expect("the model answers")
        })
    }

    fn failed(reason: &str) -> EmbedError {
        EmbedError::Failed {
            model: "stand-in".into(),
            reason: reason.into(),
        }
    }

    /// Begin a job of `tenant`'s, told how the model gets on with it.
    /// Should the model never answer it, whoever waits on it fails rather
    /// than hangs.
    fn start_for(
        local: &Arc<Local>,
        tenant: &str,
        texts: Vec<String>,
        purpose: Purpose,
    ) -> tokio::task::JoinHandle<Result<Vec<Vec<f32>>>> {
        let (local, tenant) = (local.clone(), tenant.to_string());
        tokio::spawn(async move {
            let (progress, _hears) = Progress::heard();
            let answered = local.embed_saying(texts, purpose, &tenant, progress);
            let answered = tokio::time::timeout(Duration::from_secs(30), answered).await;
            answered.expect("the model answers")
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_long_job_is_run_in_parts_and_waiting_callers_are_run_together() {
        let (local, runs) = local(0);
        let long = start(&local, texts("d", 70), Purpose::Document);
        assert_eq!(runs.next(), texts("d", 70)[..32]);
        let short: Vec<_> = (0..3)
            .map(|n| start(&local, vec![format!("s{n}")], Purpose::Document))
            .collect();
        waiting(&local, 3).await;
        runs.let_through(4);
        // The run is shared with those who came meanwhile: small writes
        // of the one tenant go ahead of its long one, and the long one
        // has the rest.
        let mut shared = runs.next();
        shared[..3].sort();
        assert_eq!(
            shared[..3],
            ["s0", "s1", "s2"],
            "in the order they came, which three tasks race for"
        );
        assert_eq!(shared[3..], texts("d", 70)[32..61]);
        assert_eq!(runs.next(), texts("d", 70)[61..]);

        let expected: Vec<Vec<f32>> = texts("d", 70)
            .iter()
            .map(|text| vec![text.len() as f32])
            .collect();
        assert_eq!(long.await.unwrap().unwrap(), expected);
        for job in short {
            assert_eq!(job.await.unwrap().unwrap(), [vec![2.0]]);
        }
        assert_eq!(
            local.embed(Vec::new(), Purpose::Query).await.unwrap(),
            [] as [Vec<f32>; 0]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_query_goes_ahead_of_documents_and_a_caller_that_left_is_not_run() {
        let (local, runs) = local(0);
        let documents = start(&local, texts("d", 64), Purpose::Document);
        assert_eq!(runs.next(), texts("d", 64)[..32]);
        let abandoned = start(&local, texts("gone", 5), Purpose::Document);
        let query = start(&local, vec!["q".into()], Purpose::Query);
        waiting(&local, 2).await;
        abandoned.abort();
        let _ = abandoned.await;

        runs.let_through(3);
        assert_eq!(runs.next(), ["q"]);
        assert_eq!(query.await.unwrap().unwrap(), [vec![1.0]]);
        assert_eq!(runs.next(), texts("d", 64)[32..]);
        assert_eq!(documents.await.unwrap().unwrap().len(), 64);
        assert!(runs.none_within(Duration::from_millis(200)));
    }

    /// A model run on no texts at all is one that may panic, and so be
    /// loaded all over again.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_is_not_run_on_nothing_when_the_only_caller_waiting_has_left() {
        let (local, runs) = local(0);
        let first = start(&local, texts("d", 32), Purpose::Document);
        assert_eq!(runs.next().len(), 32);
        let abandoned = start(&local, texts("gone", 5), Purpose::Query);
        waiting(&local, 1).await;
        abandoned.abort();
        let _ = abandoned.await;

        runs.let_through(2);
        assert_eq!(first.await.unwrap().unwrap().len(), 32);
        assert!(runs.none_within(Duration::from_millis(300)));
        assert_eq!(
            local.embed(vec!["fox".into()], Purpose::Query).await,
            Ok(vec![vec![3.0]])
        );
        assert_eq!(runs.next(), ["fox"]);
        assert_eq!(runs.loads.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// Queries and documents take turns when both wait: a query waits
    /// for the run under way and one more at the most, however many
    /// documents wait, and documents are not kept waiting by a stream of
    /// queries either.
    #[tokio::test(flavor = "multi_thread")]
    async fn queries_and_documents_take_turns_so_neither_waits_for_the_other_long() {
        let (local, runs) = local(0);
        let long = start(&local, texts("d", 320), Purpose::Document);
        assert_eq!(runs.next(), texts("d", 320)[..32]);
        let others: Vec<_> = (0..5)
            .map(|n| {
                start_for(
                    &local,
                    &format!("t{n}"),
                    texts(&format!("o{n}-"), 40),
                    Purpose::Document,
                )
            })
            .collect();
        let query = start(&local, vec!["q".into()], Purpose::Query);
        waiting(&local, 6).await;

        // The run under way cannot be stopped. The next is the query's,
        // and the query's alone.
        runs.let_through(2);
        assert_eq!(runs.next(), ["q"]);
        assert_eq!(query.await.unwrap().unwrap(), [vec![1.0]]);
        let after = runs.next();
        assert_eq!(after.len(), 32, "and then documents again");
        assert_eq!(
            of_each(&after, ["d", "o"]),
            [6, 26],
            "shared among the six tenants"
        );
        assert!(!long.is_finished());

        // So is each query that comes while documents are run. And a
        // query that comes while a query runs waits for the documents'
        // turn: it is not ahead of them for ever.
        for n in 0..3 {
            let query = start(&local, vec![format!("again{n}")], Purpose::Query);
            waiting(&local, 1).await;
            runs.let_through(2);
            assert_eq!(runs.next(), [format!("again{n}")]);
            assert_eq!(query.await.unwrap().unwrap(), [vec![6.0]]);
            assert_eq!(runs.next().len(), 32, "and then documents again");
        }
        runs.let_through(64);
        assert_eq!(long.await.unwrap().unwrap().len(), 320);
        for other in others {
            assert_eq!(other.await.unwrap().unwrap().len(), 40);
        }
    }

    /// How many texts of `run` begin with each of `prefixes`.
    fn of_each<const N: usize>(run: &[String], prefixes: [&str; N]) -> [usize; N] {
        prefixes.map(|prefix| run.iter().filter(|text| text.starts_with(prefix)).count())
    }

    /// Begin each of `jobs` as a write of its tenant's, in the order
    /// given, and say in what order they finish.
    type Finished = Arc<std::sync::Mutex<Vec<usize>>>;
    type Written = tokio::task::JoinHandle<Vec<Result<Vec<Vec<f32>>>>>;
    fn start_in_order(local: &Arc<Local>, jobs: Vec<(&str, Vec<String>)>) -> (Written, Finished) {
        let finished = Finished::default();
        let writes = jobs.into_iter().enumerate().map(|(nth, (tenant, texts))| {
            let (local, finished) = (local.clone(), finished.clone());
            let tenant = tenant.to_string();
            async move {
                let (progress, _hears) = Progress::heard();
                let made = local
                    .embed_saying(texts, Purpose::Document, &tenant, progress)
                    .await;
                finished.lock().unwrap().push(nth);
                made
            }
        });
        // One task begins them all, each in its turn.
        let writes = futures::future::join_all(writes.collect::<Vec<_>>());
        (tokio::spawn(writes), finished)
    }

    /// What `writes` come to, which they must within seconds: left
    /// waiting on a model that never comes to them, this fails rather
    /// than hangs.
    async fn all_of(writes: Written) -> Vec<Result<Vec<Vec<f32>>>> {
        let answered = tokio::time::timeout(Duration::from_secs(30), writes).await;
        answered.expect("every write is answered").unwrap()
    }

    /// A query that the model is kept at until it is let through, so
    /// that whatever is begun meanwhile is all waiting when it is.
    async fn held_at_a_query(local: &Arc<Local>, runs: &Runs) {
        let query = start(local, vec!["hold".into()], Purpose::Query);
        assert_eq!(runs.next(), ["hold"]);
        tokio::spawn(async move { query.await.unwrap().unwrap() });
    }

    /// Each tenant has as much of a run as any other, whatever each asks
    /// for: a tenant's one text is in the next run beside another's
    /// thousands, and three tenants' long writes take turns run by run.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_is_shared_among_the_tenants_that_wait_whatever_each_asks_for() {
        let (local, runs) = local(0);
        held_at_a_query(&local, &runs).await;
        let jobs = vec![
            ("a", texts("a", 96)),
            ("b", texts("b", 96)),
            ("c", texts("c", 96)),
        ];
        let (writes, _) = start_in_order(&local, jobs);
        waiting(&local, 3).await;
        // Each run is shared as evenly as it divides, and each run begins
        // with the tenant after the one the last text went to: after
        // three runs each has had as many.
        let mut had = [0; 3];
        for shares in [[11, 11, 10], [11, 10, 11], [10, 11, 11]] {
            runs.let_through(1);
            let run = runs.next();
            assert_eq!(of_each(&run, ["a", "b", "c"]), shares);
            had = [0, 1, 2].map(|n| had[n] + shares[n]);
        }
        assert_eq!(had, [32, 32, 32]);

        // A fourth tenant's one text, and a fifth's five: in the very
        // next run, beside the three.
        let small = start_for(&local, "d", vec!["small".into()], Purpose::Document);
        let five = start_for(&local, "e", texts("five", 5), Purpose::Document);
        waiting(&local, 2).await;
        runs.let_through(1);
        let run = runs.next();
        assert!(run.contains(&"small".to_string()), "{run:?}");
        // Thirty-two: the one, the five, and twenty-six among the three.
        assert_eq!(of_each(&run, ["five", "a", "b", "c"]), [5, 9, 9, 8]);
        runs.let_through(1);
        assert_eq!(small.await.unwrap().unwrap(), [vec![5.0]]);
        assert_eq!(five.await.unwrap().unwrap().len(), 5);
        runs.let_through(32);
        for write in all_of(writes).await {
            let vectors = write.unwrap();
            assert_eq!(vectors.len(), 96);
            assert_eq!(vectors[95], [3.0], "each in the order it was asked");
        }
    }

    /// One tenant's writes, among themselves: in the order they came,
    /// but that a small one goes ahead; and no tenant's share goes to
    /// another that is barred.
    #[test]
    fn a_tenants_share_of_a_run_goes_to_its_jobs_in_order_with_the_small_ones_first() {
        let shares_of = |jobs: &[(&str, usize)], most: usize, barred: &[&str]| {
            let mut line = Line::default();
            let now = Instant::now();
            for (tenant, texts) in jobs {
                let (reply, _answer) = oneshot::channel();
                line.push(
                    Job {
                        texts: vec!["x".into(); *texts],
                        purpose: Purpose::Document,
                        tenant: (*tenant).into(),
                        reply,
                        progress: None,
                    },
                    now,
                );
            }
            let barred: Vec<&str> = barred.to_vec();
            let shares = line.shared_out(most, usize::MAX, |tenant| barred.contains(&tenant));
            // By tenant, in the order the tenants stand, each job's share
            // in the order the jobs stand.
            line.queues
                .iter()
                .zip(shares)
                .map(|(queue, shares)| (queue.tenant.to_string(), shares))
                .collect::<Vec<_>>()
        };
        assert_eq!(SMALL, 4);
        // In order, and a small one ahead of larger ones that came first.
        let jobs = [("a", 100), ("a", 100), ("a", 4), ("a", 5)];
        assert_eq!(
            shares_of(&jobs, 32, &[]),
            [("a".to_string(), vec![4, 28, 0, 0])]
        );
        // Small by what was asked, not by what is left: a large write
        // with three texts to go stays behind one that came after it,
        // and a small one that came after both goes ahead of both.
        let mut line = Line::default();
        let now = Instant::now();
        let push = |line: &mut Line, texts: usize| {
            let (reply, _answer) = oneshot::channel();
            std::mem::forget(_answer);
            line.push(
                Job {
                    texts: vec!["x".into(); texts],
                    purpose: Purpose::Document,
                    tenant: "a".into(),
                    reply,
                    progress: None,
                },
                now,
            );
        };
        push(&mut line, 100);
        line.queues[0].works[0].done = vec![vec![1.0]; 97];
        push(&mut line, 50);
        push(&mut line, 2);
        let shares = line.shared_out(32, usize::MAX, |_| false);
        let lefts: Vec<usize> = line.queues[0]
            .works
            .iter()
            .map(|work| work.left())
            .collect();
        assert_eq!(lefts, [2, 3, 50]);
        assert_eq!(shares, [vec![2, 3, 27]]);
        // A text at a time to each tenant in turn, and what one tenant
        // does not want the others have.
        let jobs = [("a", 100), ("b", 1), ("c", 100), ("b", 3)];
        assert_eq!(
            shares_of(&jobs, 32, &[]),
            [
                ("a".to_string(), vec![14]),
                ("b".to_string(), vec![1, 3]),
                ("c".to_string(), vec![14]),
            ]
        );
        assert_eq!(
            shares_of(&jobs, 32, &["a"]),
            [
                ("a".to_string(), vec![0]),
                ("b".to_string(), vec![1, 3]),
                ("c".to_string(), vec![28]),
            ]
        );
        // One text at the least, however heavy.
        assert_eq!(shares_of(&[("a", 3)], 1, &[]), [("a".to_string(), vec![1])]);
        assert_eq!(shares_of(&[], 32, &[]), []);
    }

    /// Each tenant that waits is owed an equal share of each run's
    /// weight, and sends a text when it is owed the text's weight: so a
    /// tenant with short texts sends them while a tenant with long ones
    /// is not yet owed a long text's weight, and the long ones then go
    /// one to a run. Over a few runs each has had the same of the
    /// model's time, whatever its texts weigh.
    #[test]
    fn tenants_have_the_same_of_the_models_time_whatever_their_texts_weigh() {
        // The runs it takes to make every text, each as (tenant, texts):
        // jobs of so many texts of such a length, in runs of such a
        // weight, after so many runs of no stated weight and four texts,
        // as the first runs after a load are.
        let runs_of = |jobs: &[(&str, usize, usize)], unstated: usize, heaviest: usize| {
            let mut line = Line::default();
            let now = Instant::now();
            let mut answers = Vec::new();
            for (tenant, texts, length) in jobs {
                let (reply, answer) = oneshot::channel();
                answers.push((answer, *texts));
                line.push(
                    Job {
                        texts: vec!["x".repeat(*length); *texts],
                        purpose: Purpose::Document,
                        tenant: (*tenant).into(),
                        reply,
                        progress: None,
                    },
                    now,
                );
            }
            let mut served = Vec::new();
            while !line.is_empty() {
                let shares = match served.len() < unstated {
                    true => line.shared_out(FIRST_RUN, usize::MAX, |_| false),
                    false => line.shared_out(RUN, heaviest, |_| false),
                };
                let made = line
                    .texts(&shares)
                    .iter()
                    .map(|texts| Some(Ok(vec![vec![1.0]; texts.len()])))
                    .collect();
                let run: Vec<(String, usize)> = line
                    .queues
                    .iter()
                    .zip(&shares)
                    .filter(|(_, shares)| shares.iter().any(|share| *share > 0))
                    .map(|(queue, shares)| (queue.tenant.to_string(), shares.iter().sum()))
                    .collect();
                served.push(run);
                line.settle(shares, made, Instant::now());
            }
            for (mut answer, texts) in answers {
                assert_eq!(answer.try_recv().unwrap().unwrap().len(), texts);
            }
            served
        };
        let run = |of: &[(&str, usize)]| -> Vec<(String, usize)> {
            of.iter().map(|(t, n)| (t.to_string(), *n)).collect()
        };
        // Each is owed half a run's weight: the short tenant's three
        // texts go, and the long tenant's first waits until it is owed a
        // whole long text; then one long text fills each run.
        assert_eq!(
            runs_of(&[("long", 3, HEAVIEST), ("short", 3, 1)], 0, HEAVIEST),
            [
                run(&[("short", 3)]),
                run(&[("long", 1)]),
                run(&[("long", 1)]),
                run(&[("long", 1)])
            ]
        );
        // A short tenant with more than its share waiting sends its half
        // of a run, and the half the long tenant cannot yet use. Then the
        // long tenant's text fills a run; the short tenant, owed its
        // share and finding no room, is first into the next.
        assert_eq!(
            runs_of(&[("long", 3, HEAVIEST), ("short", 48, 1)], 0, HEAVIEST),
            [
                run(&[("short", 32)]),
                run(&[("long", 1)]),
                run(&[("short", 16)]),
                run(&[("long", 1)]),
                run(&[("long", 1)])
            ]
        );
        // The same after a first run of no stated weight, which is
        // shared out a text at a time and owes no one anything: the runs
        // after it are shared out by weight all the same.
        assert_eq!(
            runs_of(&[("long", 3, HEAVIEST), ("short", 48, 1)], 1, HEAVIEST),
            [
                run(&[("long", 2), ("short", 2)]),
                run(&[("short", 32)]),
                run(&[("long", 1)]),
                run(&[("short", 14)]),
            ]
        );
        // In a run that holds four long texts, each tenant has what it is
        // owed and no more: two long texts are the long tenant's half,
        // whatever room the short tenant's leave.
        assert_eq!(
            runs_of(&[("long", 4, HEAVIEST), ("short", 40, 1)], 0, 4 * HEAVIEST),
            [
                run(&[("long", 2), ("short", 30)]),
                run(&[("long", 2), ("short", 10)]),
            ]
        );
        // A tenant owed a text there is no room left for is first into
        // the next run, wherever it stands: between two tenants whose
        // short texts fill every run, a long text waits the four runs it
        // takes to be owed it, finds no room in the fourth, and has the
        // fifth. Left to its turn it would never come first.
        let between = [("a", 96, 1), ("long", 2, HEAVIEST), ("b", 96, 1)];
        assert_eq!(
            runs_of(&between, 0, HEAVIEST)[..5],
            [
                run(&[("a", 16), ("b", 16)]),
                run(&[("a", 16), ("b", 16)]),
                run(&[("a", 16), ("b", 16)]),
                run(&[("a", 16), ("b", 16)]),
                run(&[("long", 1)]),
            ]
        );
    }

    /// A run is of as much as takes about as long as a run is to, by
    /// what the texts weigh: long texts make for a run of few, and short
    /// ones for a run of many, so one tenant's long texts cannot make a
    /// run long for everyone.
    #[test]
    fn a_run_is_of_as_much_weight_as_takes_about_as_long_as_a_run_is_to() {
        let ms = Duration::from_millis;
        assert_eq!((LIGHTEST, HEAVIEST), (64, 2_048));
        assert_eq!(weight("x"), LIGHTEST);
        assert_eq!(weight(&"x".repeat(500)), 500);
        assert_eq!(weight(&"x".repeat(50_000)), HEAVIEST);
        // A text is cut to what a model reads of it, where a character
        // ends.
        assert_eq!(cut("short".into()), "short");
        let long = "é".repeat(LONGEST_TEXT + 7);
        assert_eq!(cut(long).chars().count(), LONGEST_TEXT);
        assert_eq!(LONGEST_TEXT, 51_200);

        let mut pace = Pace::default();
        assert_eq!(
            pace.run(ms(1_000)),
            (FIRST_RUN, usize::MAX),
            "nothing is known yet"
        );
        assert_eq!(
            pace.run(Duration::MAX),
            (RUN, usize::MAX),
            "as long as they take"
        );
        // A microsecond a unit of weight: a second's run is a million
        // units, which is 32 short texts and no more, or 488 long ones'
        // worth, which 32 of them exceed.
        pace.took(Duration::from_micros(1));
        assert_eq!(pace.run(ms(1_000)), (RUN, 1_000_000));
        pace.took(Duration::from_micros(10));
        assert_eq!(
            pace.run(ms(1_000)),
            (RUN, 100_000),
            "by the slower of the two"
        );
        pace.took(Duration::from_micros(2));
        assert_eq!(pace.run(ms(1_000)), (RUN, 100_000));
        pace.took(Duration::from_micros(4));
        assert_eq!(pace.run(ms(1_000)), (RUN, 250_000));
        // A second a unit: one text's worth, at the least.
        pace.took(ms(1_000));
        assert_eq!(pace.run(ms(1_000)), (RUN, 1));
        pace.forget();
        assert_eq!(pace.run(ms(1_000)), (FIRST_RUN, usize::MAX));
    }

    /// A stand-in whose cost follows the length of its texts, as a model
    /// that reads so many tokens of a text and no more does: `per_byte`
    /// over each byte of the first four thousand, and nothing over the
    /// rest. It says how many bytes and texts each run was of.
    fn by_length(per_byte: Duration, room: Arc<Room>) -> (Arc<Local>, Receiver<(usize, usize)>) {
        let (report, ran) = channel();
        let load: Load = Box::new(move || {
            let report = report.clone();
            Ok(Box::new(move |texts: Vec<&str>| {
                let bytes: usize = texts.iter().map(|text| text.len()).sum();
                let read: usize = texts.iter().map(|text| text.len().min(4_096)).sum();
                let _ = report.send((texts.len(), bytes));
                std::thread::sleep(per_byte * u32::try_from(read).unwrap());
                assert!(
                    !texts.iter().any(|text| text.starts_with("panic")),
                    "told to"
                );
                if texts.iter().any(|text| text.starts_with("fail")) {
                    return Err("told to".to_string());
                }
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        (Arc::new(Local::spawn("stand-in", load, room).unwrap()), ran)
    }

    /// One tenant's long texts, cut to what a model reads and weighed,
    /// do not make a run long for another tenant's short write or a
    /// search: on a model that takes a microsecond a byte, thirty-two
    /// texts of four hundred kilobytes would be thirteen seconds a run,
    /// and are not.
    #[tokio::test(flavor = "multi_thread")]
    async fn long_texts_are_cut_and_weighed_so_a_run_stays_short_for_everyone() {
        let ms = Duration::from_millis;
        let room = Room::new(1, Duration::MAX, None, paced(ms(200)));
        let (local, ran) = by_length(Duration::from_micros(1), room);
        // The first run teaches the pace.
        assert!(local.embed(texts("d", 4), Purpose::Document).await.is_ok());
        let long: Vec<String> = (0..32)
            .map(|n| format!("{n} {}", "x".repeat(400_000)))
            .collect();
        let bulk = start_for(&local, "bulk", long, Purpose::Document);
        tokio::time::sleep(ms(80)).await;
        let began = std::time::Instant::now();
        let small = start_for(&local, "small", vec!["one".into()], Purpose::Document);
        let query = local.embed(vec!["q".into()], Purpose::Query).await;
        let searched = began.elapsed();
        assert_eq!(query, Ok(vec![vec![1.0]]));
        assert_eq!(small.await.unwrap(), Ok(vec![vec![3.0]]));
        let written = began.elapsed();
        assert!(searched < ms(1_000), "a search took {searched:?}");
        assert!(written < ms(1_500), "a small write took {written:?}");
        let made = bulk.await.unwrap().unwrap();
        assert_eq!(made.len(), 32);
        assert!(made
            .iter()
            .all(|vector| vector[0] as usize <= LONGEST_TEXT + 3));
        // No run was of more than what takes about a run's time, and
        // none was of more bytes than thirty-two texts cut.
        let runs: Vec<(usize, usize)> = ran.try_iter().collect();
        for (texts, bytes) in &runs {
            assert!(*bytes <= 32 * (LONGEST_TEXT + 3), "a run of {bytes} bytes");
            assert!(*texts <= RUN);
        }
    }

    /// Tenants writing beside a stream of searches from others: the
    /// searches go first, and the writes are done all the same, within a
    /// few turns, and none is given up on.
    #[tokio::test(flavor = "multi_thread")]
    async fn searches_do_not_keep_writes_from_the_model() {
        let ms = Duration::from_millis;
        for searchers in [8, 40] {
            let (local, _ran) = taking(ms(1), Duration::MAX);
            let embedder = unless_stalled(&local, ms(500));
            let searching = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let searches: Vec<_> = (0..searchers)
                .map(|n| {
                    let (embedder, searching) = (embedder.clone(), searching.clone());
                    tokio::spawn(async move {
                        let mut made = 0;
                        while searching.load(std::sync::atomic::Ordering::SeqCst) {
                            let text = format!("search {n} {made}");
                            embedder.query_for("searcher", &text).await.unwrap();
                            made += 1;
                        }
                        made
                    })
                })
                .collect();
            tokio::time::sleep(ms(100)).await;
            let began = std::time::Instant::now();
            let (one_text, thirty_texts) = (vec!["one".to_string()], texts("thirty", 30));
            let (one, thirty) = tokio::join!(
                embedder.documents_for("writer", &one_text),
                embedder.documents_for("writer", &thirty_texts)
            );
            let took = began.elapsed();
            assert_eq!(one.map(|made| made.len()), Ok(1), "{searchers} searchers");
            assert_eq!(
                thirty.map(|made| made.len()),
                Ok(30),
                "{searchers} searchers"
            );
            // Two document runs and the query runs between: a run of
            // thirty-two texts at a millisecond each, so well under a
            // second even with the queries' turns.
            assert!(
                took < ms(2_000),
                "{searchers} searchers: the writes took {took:?}"
            );
            searching.store(false, std::sync::atomic::Ordering::SeqCst);
            let made: usize = futures::future::join_all(searches)
                .await
                .into_iter()
                .map(|made| made.unwrap())
                .sum();
            assert!(made > searchers, "{made} searches made beside");
        }
    }

    /// A write the model is reckoned not to come to within the time a
    /// write is held is refused at once, as what may pass, rather than
    /// kept: those deepest in the queues of the tenants with most
    /// waiting. The others are made.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_the_model_will_not_come_to_soon_is_refused_at_once() {
        let ms = Duration::from_millis;
        let times = Times {
            run_for: Duration::MAX,
            held_at_most: ms(1_000),
            ..Times::default()
        };
        let room = Room::new(1, Duration::MAX, None, times);
        let (local, _ran) = taking_among(ms(5), room);
        // The first run teaches how long a text takes.
        assert!(local.embed(texts("d", 8), Purpose::Document).await.is_ok());
        // One tenant's long write, and another's six in a row: at five
        // milliseconds a text, two tenants in turn, the fourth of the six
        // would wait 5 ms x 2 x 121 = 1.2 s for its turn, and is not kept.
        let long = start_for(&local, "a", texts("a", 400), Purpose::Document);
        let six = (0..6)
            .map(|n| ("b", texts(&format!("b{n}-"), 40)))
            .collect();
        let began = std::time::Instant::now();
        let (six, _) = start_in_order(&local, six);
        let made: Vec<Result<usize>> = all_of(six)
            .await
            .into_iter()
            .map(|made| made.map(|made| made.len()))
            .collect();
        let refused = |made: &Result<usize>| matches!(made, Err(EmbedError::Failed { reason, .. }) if reason.contains("more waiting for it than it will come to soon"));
        assert_eq!(made[0], Ok(40), "{made:?}");
        assert_eq!(made[1], Ok(40), "{made:?}");
        assert!(refused(&made[4]) && refused(&made[5]), "{made:?}");
        assert!(
            made.iter().all(|made| made.is_ok() || refused(made)),
            "{made:?}"
        );
        let refusals = made.iter().filter(|made| refused(made)).count();
        assert!(
            began.elapsed() < ms(1_000) + ms(5) * 40 * 2 * u32::try_from(6 - refusals).unwrap()
        );
        assert_eq!(long.await.unwrap().map(|made| made.len()), Ok(400));
    }

    /// A run the model fails is put off, and each tenant in it run by
    /// itself in its turn: a text the model cannot take fails the write
    /// it is in and no other, at the cost of the one failed run. The
    /// tenant whose it was is then kept from the model for a while, and
    /// its writes made apart from others' after.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_the_model_fails_on_fails_the_write_it_is_in_and_no_other() {
        let times = Times {
            run_for: Duration::MAX,
            first_backoff: Duration::from_millis(200),
            ..Times::default()
        };
        let (local, runs) = local_among("stand-in", 0, Room::new(1, Duration::MAX, None, times));
        held_at_a_query(&local, &runs).await;
        let mut jobs: Vec<(&str, Vec<String>)> = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]
            .into_iter()
            .map(|tenant| (tenant, texts(&format!("{tenant}-"), 4)))
            .collect();
        jobs[5].1[2] = "fail".into();
        let (writes, _) = start_in_order(&local, jobs);
        waiting(&local, 8).await;
        // The run of all eight, which fails; then each tenant by itself,
        // in the order they came. Ten tokens: one for the query held at,
        // and nine runs.
        runs.let_through(10);
        let sizes: Vec<usize> = (0..9).map(|_| runs.next().len()).collect();
        assert_eq!(sizes, [32, 4, 4, 4, 4, 4, 4, 4, 4]);
        let made = all_of(writes).await;
        for (nth, made) in made.iter().enumerate() {
            match nth {
                5 => assert_eq!(made, &Err(failed("told to"))),
                _ => assert_eq!(made.as_ref().map(Vec::len), Ok(4), "write {nth}"),
            }
        }

        // The tenant is kept from the model for a while: another's write
        // and its next run apart, the other's first.
        held_at_a_query(&local, &runs).await;
        let other = start_for(&local, "t0", texts("again", 2), Purpose::Document);
        let again = start_for(&local, "t5", texts("sound", 2), Purpose::Document);
        waiting(&local, 2).await;
        runs.let_through(2);
        assert_eq!(runs.next(), ["again0", "again1"]);
        assert_eq!(other.await.unwrap().unwrap().len(), 2);
        runs.let_through(1);
        assert_eq!(
            runs.next(),
            ["sound0", "sound1"],
            "alone, when its time comes"
        );
        assert_eq!(again.await.unwrap().unwrap().len(), 2);
        runs.let_through(1);
        assert_eq!(
            local.embed(vec!["fox".into()], Purpose::Query).await,
            Ok(vec![vec![3.0]])
        );
    }

    /// A tenant whose texts have failed the model is kept from it for a
    /// second, then twice as long for each failure running, and never
    /// longer than a minute.
    #[test]
    fn a_tenant_the_model_keeps_failing_on_is_kept_from_it_twice_as_long_each_time_up_to_a_minute()
    {
        let (ms, secs) = (Duration::from_millis, Duration::from_secs);
        let times = Times::default();
        assert_eq!(
            (times.first_backoff, times.longest_backoff),
            (secs(1), secs(60))
        );
        let kept: Vec<Duration> = (1..=8).map(|strikes| backoff(strikes, &times)).collect();
        let expected = [1, 2, 4, 8, 16, 32, 60, 60].map(secs);
        assert_eq!(kept, expected);
        assert_eq!(backoff(1_000, &times), secs(60), "however many");
        let quick = Times {
            first_backoff: ms(50),
            longest_backoff: ms(400),
            ..times
        };
        assert_eq!(backoff(4, &quick), ms(400));
        assert_eq!(backoff(5, &quick), ms(400));
    }

    /// A tenant the model has failed on is run by itself until a run of
    /// its own goes well, and then shares runs with the others again.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_tenant_the_model_failed_on_shares_runs_again_once_a_run_of_its_own_goes_well() {
        let ms = Duration::from_millis;
        let times = Times {
            run_for: Duration::MAX,
            first_backoff: ms(50),
            longest_backoff: ms(400),
            ..Times::default()
        };
        let room = Room::new(1, Duration::MAX, None, times);
        let (local, runs) = local_among("stand-in", 0, room);
        // A write of "bad"'s fails: it is in trouble.
        let failing = start_for(&local, "bad", vec!["fail".into()], Purpose::Document);
        assert_eq!(runs.next(), ["fail"]);
        runs.let_through(1);
        assert!(failing.await.unwrap().is_err());
        // Its next write, after its time, is run alone and goes well.
        tokio::time::sleep(ms(100)).await;
        let alone = start_for(&local, "bad", vec!["mended".into()], Purpose::Document);
        assert_eq!(runs.next(), ["mended"]);
        runs.let_through(1);
        assert_eq!(alone.await.unwrap(), Ok(vec![vec![6.0]]));
        // From then on it shares runs with the others: its four texts and
        // "good"'s four, waiting together, are one run.
        held_at_a_query(&local, &runs).await;
        let bad = start_for(&local, "bad", texts("b", 4), Purpose::Document);
        let good = start_for(&local, "good", texts("g", 4), Purpose::Document);
        waiting(&local, 2).await;
        runs.let_through(1);
        let run = runs.next();
        assert_eq!(of_each(&run, ["b", "g"]), [4, 4], "{run:?}");
        runs.let_through(1);
        assert_eq!(bad.await.unwrap().map(|made| made.len()), Ok(4));
        assert_eq!(good.await.unwrap().map(|made| made.len()), Ok(4));
    }

    /// Tenants whose texts keep failing the model, or making it panic,
    /// do not keep another tenant's writes from it: each is let at the
    /// model apart from the others, in its own turn, less and less
    /// often, and the model is loaded again only so often for it.
    #[tokio::test(flavor = "multi_thread")]
    async fn tenants_whose_texts_keep_failing_the_model_do_not_keep_others_from_it() {
        let ms = Duration::from_millis;
        let times = Times {
            run_for: Duration::MAX,
            first_backoff: ms(50),
            longest_backoff: ms(400),
            ..Times::default()
        };
        let room = Room::new(1, Duration::MAX, None, times);
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = loads.clone();
        let load: Load = Box::new(move || {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Box::new(move |texts: Vec<&str>| {
                std::thread::sleep(ms(1) * u32::try_from(texts.len()).unwrap());
                assert!(
                    !texts.iter().any(|text| text.starts_with("panic")),
                    "told to"
                );
                if texts.iter().any(|text| text.starts_with("fail")) {
                    return Err("told to".to_string());
                }
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        let local = Arc::new(Local::spawn("stand-in", load, room).unwrap());
        let embedder = unless_stalled(&local, ms(2_000));
        let going = Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Three tenants whose every write has a text the model fails on,
        // and three whose every write makes it panic: each writing again
        // and again.
        let loops: Vec<_> = (0..6)
            .map(|n| {
                let (embedder, going) = (embedder.clone(), going.clone());
                tokio::spawn(async move {
                    let (tenant, word) = (format!("bad{n}"), ["fail", "panic"][n % 2]);
                    let mut tried = 0;
                    while going.load(std::sync::atomic::Ordering::SeqCst) {
                        let mut texts = texts(&format!("{tenant}-"), 8);
                        texts[3] = format!("{word} {tried}");
                        let made = embedder.documents_for(&tenant, &texts).await;
                        assert!(made.is_err());
                        tried += 1;
                    }
                    tried
                })
            })
            .collect();
        tokio::time::sleep(ms(100)).await;
        // Another tenant's six larger writes, meanwhile: all made, and
        // not long after they would be alone.
        let began = std::time::Instant::now();
        let writes = (0..6).map(|n| {
            let embedder = embedder.clone();
            async move {
                embedder
                    .documents_for("good", &texts(&format!("good{n}-"), 200))
                    .await
            }
        });
        let made = futures::future::join_all(writes).await;
        let took = began.elapsed();
        for (nth, made) in made.iter().enumerate() {
            assert_eq!(made.as_ref().map(Vec::len), Ok(200), "write {nth}");
        }
        assert!(took < ms(8_000), "the sound tenant's writes took {took:?}");
        going.store(false, std::sync::atomic::Ordering::SeqCst);
        let tried: Vec<usize> = futures::future::join_all(loops)
            .await
            .into_iter()
            .map(|tried| tried.unwrap())
            .collect();
        // Each failing tenant was tried less and less often: a few times
        // a second at most, and the model loaded again no more often
        // than the panicking ones were let at it.
        // A panic in a run shared with others costs a load, and the
        // tenant's run by itself after it another: two a time at most.
        let loaded = loads.load(std::sync::atomic::Ordering::SeqCst);
        let panics: usize = [1, 3, 5].iter().map(|n| tried[*n]).sum();
        assert!(
            loaded <= 1 + 2 * panics,
            "{loaded} loads for {panics} panics"
        );
        assert!(
            tried.iter().all(|tried| *tried <= 40),
            "{tried:?} in {took:?}"
        );
    }

    /// A stand-in that takes `each` over every text of a run, and says
    /// how many texts each run was of. It fails a run with a text that
    /// says "fail" in it.
    fn taking(each: Duration, run_for: Duration) -> (Arc<Local>, Receiver<usize>) {
        taking_among(each, Room::new(1, Duration::MAX, None, paced(run_for)))
    }

    fn taking_among(each: Duration, room: Arc<Room>) -> (Arc<Local>, Receiver<usize>) {
        let (report, ran) = channel();
        let load: Load = Box::new(move || {
            let report = report.clone();
            Ok(Box::new(move |texts: Vec<&str>| {
                let _ = report.send(texts.len());
                std::thread::sleep(each * texts.len() as u32);
                if texts.iter().any(|text| text.starts_with("fail")) {
                    return Err("told to".to_string());
                }
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        (Arc::new(Local::spawn("stand-in", load, room).unwrap()), ran)
    }

    /// The model's runs are sized by how long its last ones took. A run
    /// that fails says nothing against that: a failure is one write's.
    /// A model let go and loaded again begins small again.
    #[tokio::test(flavor = "multi_thread")]
    async fn runs_begin_small_again_only_after_the_model_is_let_go() {
        let ms = Duration::from_millis;
        let idle = ms(300);
        let room = Room::new(1, idle, None, paced(ms(10_000)));
        let (local, ran) = taking_among(ms(1), room.clone());
        let next = || ran.recv_timeout(Duration::from_secs(10)).unwrap();
        let written = |texts: Vec<String>| local.embed(texts, Purpose::Document);
        // Quick texts: after the first run, runs are as long as can be.
        assert_eq!(written(texts("d", 40)).await.unwrap().len(), 40);
        assert_eq!([next(), next(), next()], [FIRST_RUN, RUN, 4]);
        // A run that fails: the next is as long as before.
        let mut failing = texts("e", 40);
        failing[0] = "fail".into();
        assert!(written(failing).await.is_err());
        assert_eq!(next(), RUN);
        assert_eq!(written(texts("f", 40)).await.unwrap().len(), 40);
        assert_eq!([next(), next()], [RUN, 8]);
        // Let go for want of use, and loaded again: small again.
        until(|| room.loaded().is_empty()).await;
        assert_eq!(written(texts("g", 40)).await.unwrap().len(), 40);
        assert_eq!([next(), next(), next()], [FIRST_RUN, RUN, 4]);
    }

    /// An embedder of `local` that gives up on a write the model has got
    /// nowhere with for `stalled`, and a millisecond more for each text.
    fn unless_stalled(local: &Arc<Local>, stalled: Duration) -> Arc<crate::Embedder> {
        let embedder = giving(local, Duration::from_secs(5));
        let each = Duration::from_millis(1);
        Arc::new(embedder.answering_within(Duration::from_secs(5), stalled, each))
    }

    /// Writes made all at once, as many and as long as each row says,
    /// all one tenant's and each its own tenant's. Every one is made,
    /// and no text is made twice or for nothing: the model is asked for
    /// exactly as many as were written. Each write is given up on only
    /// if the model gets nowhere for 300 ms, which is far less than any
    /// but the first few wait for their turn.
    #[tokio::test(flavor = "multi_thread")]
    async fn writes_made_at_once_are_all_made_however_many_and_however_long() {
        let ms = Duration::from_millis;
        let grid = [
            (10, 200),
            (20, 200),
            (40, 50),
            (12, 1_000),
            (16, 30),
            (48, 1_000),
            (300, 4),
        ];
        for (writes, each_of) in grid {
            for own_tenant in [false, true] {
                let (local, ran) = taking(Duration::from_micros(200), Duration::MAX);
                let embedder = unless_stalled(&local, ms(300));
                let began = std::time::Instant::now();
                let written = (0..writes).map(|nth| {
                    let embedder = embedder.clone();
                    let texts = texts(&format!("w{nth}-"), each_of);
                    let tenant = if own_tenant {
                        format!("t{nth}")
                    } else {
                        String::new()
                    };
                    async move { embedder.documents_for(&tenant, &texts).await }
                });
                let written = futures::future::join_all(written);
                let written = tokio::time::timeout(Duration::from_secs(60), written);
                let written = written.await.expect("they are made or given up on");
                for (nth, made) in written.iter().enumerate() {
                    let made = made.as_ref().map(Vec::len);
                    assert_eq!(made, Ok(each_of), "write {nth} of {writes} x {each_of}");
                }
                let asked: usize = ran.try_iter().sum();
                assert_eq!(asked, writes * each_of, "{writes} x {each_of}");
                println!(
                    "{writes} writes of {each_of}, {}: {} made, {asked} texts asked of the \
                     model for {} written, in {:.2?}",
                    if own_tenant {
                        "a tenant each"
                    } else {
                        "one tenant"
                    },
                    written.iter().filter(|made| made.is_ok()).count(),
                    writes * each_of,
                    began.elapsed()
                );
            }
        }
    }

    /// A model that has stopped is given up on by every write, those it
    /// had turned to and those still waiting, once it has got nowhere
    /// for as long as it is given: and not before.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_has_stopped_is_given_up_on_by_all_who_write() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = unless_stalled(&local, ms(1_000));
        let stalled = failed("nothing from the model in 1s");
        let written = (0..7).map(|nth| {
            let embedder = embedder.clone();
            let texts = texts(&format!("w{nth}-"), [40, 40, 40, 40, 40, 40, 1][nth]);
            async move {
                let began = std::time::Instant::now();
                let made = embedder.documents_for(&format!("t{nth}"), &texts).await;
                (made.map(|made| made.len()), began.elapsed())
            }
        });
        let written = tokio::spawn(futures::future::join_all(written.collect::<Vec<_>>()));
        // Two runs come back, 300 ms apart: no write is given up on,
        // though some have had little of either. The third does not.
        for _ in 0..2 {
            runs.next();
            tokio::time::sleep(ms(300)).await;
            assert!(!written.is_finished());
            runs.let_through(1);
        }
        runs.next();
        let written = tokio::time::timeout(Duration::from_secs(10), written);
        let written = written.await.expect("they are given up on").unwrap();
        // The small one was made in the first run it could be in.
        assert_eq!(written[6].0, Ok(1));
        for (made, took) in &written[..6] {
            assert!(
                matches!(made, Err(EmbedError::Failed { reason, .. }) if reason.starts_with("nothing from the model in") || reason.contains("did not come to it within")),
                "{made:?}, expected as {stalled}"
            );
            assert!((ms(1_600)..ms(6_000)).contains(took), "{took:?}");
        }
        runs.let_through(4);
    }

    /// Texts that take 50 ms each would take 1.6 seconds a run of
    /// thirty-two, and a query would wait that long behind one. Runs are
    /// to take 300 ms here, so they are of six texts or so, and a query
    /// waits about that long and never much longer.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_query_waits_behind_documents_about_as_long_as_a_run_is_to_take() {
        let ms = Duration::from_millis;
        let (local, ran) = taking(ms(50), ms(300));
        let documents = start(&local, texts("d", 60), Purpose::Document);
        // The first run is short, and those after are sized by it.
        let next = || ran.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(next(), FIRST_RUN);
        let mut waits = Vec::new();
        for _ in 0..6 {
            let size = next();
            assert!((1..=6).contains(&size), "a run of {size}");
            let began = std::time::Instant::now();
            let query = local.embed(vec!["q".into()], Purpose::Query).await;
            waits.push(began.elapsed());
            assert_eq!(query, Ok(vec![vec![1.0]]));
            assert_eq!(next(), 1, "the query's own run");
        }
        let longest = waits.iter().max().unwrap();
        assert!(
            *longest < ms(1_200),
            "a query waited {longest:?}: {waits:?}"
        );
        assert_eq!(documents.await.unwrap().unwrap().len(), 60);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_fails_to_load_fails_or_panics_is_tried_afresh() {
        let (local, runs) = local(1);
        runs.let_through(4);
        let embed = |text: &str| local.embed(vec![text.to_string()], Purpose::Document);
        assert_eq!(embed("fox").await, Err(failed("no model to be had")));
        assert_eq!(embed("fox").await, Ok(vec![vec![3.0]]));
        assert_eq!(embed("fail").await, Err(failed("told to")));
        assert_eq!(runs.loads.load(std::sync::atomic::Ordering::SeqCst), 2);
        runs.let_through(1);
        assert_eq!(
            embed("quote").await,
            Err(failed("401 for <redacted>")),
            "a token the model quotes is not passed on"
        );
        assert_eq!(embed("panic").await, Err(failed("the model panicked")));
        assert_eq!(embed("hen").await, Ok(vec![vec![3.0]]));
        assert_eq!(runs.loads.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_thread_ends_when_the_model_is_let_go() {
        let (local, runs) = local(0);
        runs.let_through(1);
        local
            .embed(vec!["fox".into()], Purpose::Query)
            .await
            .unwrap();
        assert_eq!(runs.next(), ["fox"]);
        drop(local);
        let ended = runs.ran.recv_timeout(Duration::from_secs(10));
        assert_eq!(ended, Err(std::sync::mpsc::RecvTimeoutError::Disconnected));
    }

    /// What a model is refused with when the `most` a node has loaded at
    /// once are `others`, each kept an hour unused.
    fn no_room(model: &str, most: usize, others: &str) -> EmbedError {
        EmbedError::Failed {
            model: model.into(),
            reason: format!(
                "this node has no more than {most} local models loaded at once, and has \
                 {others} in use: one is let go when it has not been used for 3600s"
            ),
        }
    }

    async fn fox(local: &Local) -> Result<Vec<Vec<f32>>> {
        local.embed(vec!["fox".into()], Purpose::Query).await
    }

    /// Wait for `what` to be so, which it must be within seconds.
    async fn until(what: impl Fn() -> bool) {
        for _ in 0..500 {
            if what() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("it never was so");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_more_models_are_loaded_at_once_than_a_node_has_room_for() {
        let room = Room::new(2, Duration::from_secs(60 * 60), None, unpaced());
        let models = ["one", "two", "three"].map(|model| local_among(model, 0, room.clone()));
        let [(one, ones), (two, twos), (three, threes)] = models;
        for runs in [&ones, &twos, &threes] {
            runs.let_through(16);
        }
        let loads = |runs: &Runs| runs.loads.load(std::sync::atomic::Ordering::SeqCst);
        assert!(room.loaded().is_empty(), "none is loaded until it is used");

        assert_eq!(fox(&one).await, Ok(vec![vec![3.0]]));
        assert_eq!(fox(&two).await, Ok(vec![vec![3.0]]));
        assert_eq!(room.loaded(), ["one", "two"]);
        // A third is refused, and is not loaded: each time it is asked
        // for, while the others go on. That passes, when one of them goes
        // unused: it is no fault that stands, to keep the model from
        // being asked again or have whoever asks told not to.
        let refused = no_room("three", 2, "one and two");
        assert_eq!(refused.lasting(), None);
        for _ in 0..3 {
            assert_eq!(fox(&three).await, Err(refused.clone()));
            assert_eq!(fox(&one).await, Ok(vec![vec![3.0]]));
            assert_eq!(fox(&two).await, Ok(vec![vec![3.0]]));
        }
        assert_eq!(room.loaded(), ["one", "two"]);
        assert_eq!((loads(&ones), loads(&twos), loads(&threes)), (1, 1, 0));

        // A model no collection embeds with any more gives up its place.
        drop(one);
        until(|| room.loaded() == ["two"]).await;
        assert_eq!(fox(&three).await, Ok(vec![vec![3.0]]));
        assert_eq!(room.loaded(), ["two", "three"]);
        assert_eq!(loads(&threes), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_not_used_for_long_is_let_go_and_loaded_again_when_next_asked_for() {
        let ms = Duration::from_millis;
        let idle = ms(1_000);
        let room = Room::new(1, idle, None, unpaced());
        let (one, ones) = local_among("one", 0, room.clone());
        let (two, twos) = local_among("two", 0, room.clone());
        ones.let_through(16);
        twos.let_through(16);
        let loads = |runs: &Runs| runs.loads.load(std::sync::atomic::Ordering::SeqCst);

        // Used again and again, it is kept, and there is no room for
        // another.
        for _ in 0..3 {
            assert_eq!(fox(&one).await, Ok(vec![vec![3.0]]));
            assert!(matches!(fox(&two).await, Err(EmbedError::Failed { .. })));
            tokio::time::sleep(idle / 3).await;
        }
        assert_eq!(
            (room.loaded(), loads(&ones), loads(&twos)),
            (vec!["one".to_string()], 1, 0)
        );

        // Left unused for as long as a node keeps one, it is let go: and
        // then there is room for the other.
        let last_used = std::time::Instant::now();
        assert_eq!(fox(&one).await, Ok(vec![vec![3.0]]));
        until(|| room.loaded().is_empty()).await;
        assert!(last_used.elapsed() >= idle, "{:?}", last_used.elapsed());
        assert_eq!(fox(&two).await, Ok(vec![vec![3.0]]));
        assert_eq!(room.loaded(), ["two"]);

        // It is loaded again when next asked for, once there is room.
        assert!(fox(&one).await.is_err());
        until(|| room.loaded().is_empty()).await;
        assert_eq!(fox(&one).await, Ok(vec![vec![3.0]]));
        assert_eq!(
            (room.loaded(), loads(&ones), loads(&twos)),
            (vec!["one".to_string()], 2, 1)
        );
    }

    /// Collections that embed with other models, as many as a node has
    /// room for and more, and in use all the while: they have the places
    /// that are not the node's own model's, and no more.
    #[tokio::test(flavor = "multi_thread")]
    async fn other_models_never_keep_the_nodes_own_model_from_its_place() {
        let idle = Duration::from_millis(900);
        let room = Room::new(2, idle, Some("own".into()), unpaced());
        let models = ["own", "a", "b"].map(|model| local_among(model, 0, room.clone()));
        let [(own, owns), (a, as_), (b, bs)] = models;
        for runs in [&owns, &as_, &bs] {
            runs.let_through(64);
        }
        let loads = |runs: &Runs| runs.loads.load(std::sync::atomic::Ordering::SeqCst);
        let refused = |model: &str, others: &str| EmbedError::Failed {
            model: model.into(),
            reason: format!(
                "this node has no more than 1 local models loaded at once beside its own, \
                 own, and has {others} in use: one is let go when it has not been used for \
                 900ms"
            ),
        };

        // The others are asked for first. They have one place between
        // them, though the node has two.
        assert_eq!(fox(&a).await, Ok(vec![vec![3.0]]));
        assert_eq!(fox(&b).await, Err(refused("b", "a")));
        assert_eq!(room.loaded(), ["a"]);
        // The node's own model has the other, whenever it is asked for,
        // and whatever the others are doing.
        for _ in 0..3 {
            assert_eq!(fox(&own).await, Ok(vec![vec![3.0]]));
            assert_eq!(fox(&a).await, Ok(vec![vec![3.0]]));
            assert_eq!(fox(&b).await, Err(refused("b", "a")));
        }
        assert_eq!(room.loaded(), ["a", "own"]);
        // No room is no fault that stands. Whoever embeds with the model
        // is told each time it is asked, and it is asked each time.
        let embedder = giving(&b, Duration::from_secs(5));
        for _ in 0..3 {
            assert_eq!(embedder.query("fox").await, Err(refused("b", "a")));
        }
        assert_eq!(embedder.query("hen").await, Err(refused("b", "a")));

        // Unused, another model is let go, and its place goes to another
        // of them. The node's own is not let go, however long unused.
        // It was last used a moment ago, so that is not at once.
        let lately = std::time::Instant::now();
        until(|| room.loaded() == ["own"]).await;
        assert!(lately.elapsed() >= idle / 2, "{:?}", lately.elapsed());
        assert_eq!(embedder.query("fox").await, Ok(Some(vec![3.0])));
        assert_eq!(fox(&a).await, Err(refused("a", "b")));
        until(|| room.loaded() == ["own"]).await;
        tokio::time::sleep(idle * 2).await;
        assert_eq!(room.loaded(), ["own"]);
        assert_eq!(fox(&own).await, Ok(vec![vec![3.0]]));
        assert_eq!((loads(&owns), loads(&as_), loads(&bs)), (1, 1, 1));

        // A node with room for one has room for its own alone: another
        // is refused, in a way that passes only when the node is set up
        // otherwise, and its own is not.
        let room = Room::new(1, idle, Some("own".into()), unpaced());
        let (other, others) = local_among("other", 0, room.clone());
        let (own, owns) = local_among("own", 0, room.clone());
        others.let_through(4);
        owns.let_through(4);
        let none = EmbedError::Failed {
            model: "other".into(),
            reason: "this node has a place for no local model beside its own, own".into(),
        };
        assert_eq!(fox(&other).await, Err(none.clone()));
        assert_eq!(fox(&own).await, Ok(vec![vec![3.0]]));
        assert_eq!(fox(&other).await, Err(none));
        assert_eq!(room.loaded(), ["own"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_does_not_load_or_panics_keeps_no_place() {
        let room = Room::new(1, Duration::from_secs(60 * 60), None, unpaced());
        let (failing, failings) = local_among("stand-in", 1, room.clone());
        let (other, others) = local_among("other", 0, room.clone());
        failings.let_through(16);
        others.let_through(16);
        assert_eq!(fox(&failing).await, Err(failed("no model to be had")));
        assert!(room.loaded().is_empty());
        assert_eq!(fox(&other).await, Ok(vec![vec![3.0]]));
        assert_eq!(room.loaded(), ["other"]);
        assert_eq!(fox(&failing).await, Err(no_room("stand-in", 1, "other")));
        let loads = failings.loads.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(loads, 1, "it is not loaded to find there is no room");

        assert!(other
            .embed(vec!["panic".into()], Purpose::Query)
            .await
            .is_err());
        assert!(room.loaded().is_empty());
        assert_eq!(fox(&failing).await, Ok(vec![vec![3.0]]));
        assert_eq!(room.loaded(), ["stand-in"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_can_never_load_says_so_each_time_it_is_tried() {
        let tries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tried = tries.clone();
        let barred = || EmbedError::Unusable {
            model: "stand-in".into(),
            reason: "its cache may not be written".into(),
            fault: vellev_core::ModelFault::Setup,
        };
        let load: Load = Box::new(move || {
            tried.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(barred())
        });
        let local = Local::spawn("stand-in", load, alone()).unwrap();
        for _ in 0..2 {
            assert_eq!(
                local.embed(vec!["fox".into()], Purpose::Query).await,
                Err(barred())
            );
        }
        assert_eq!(tries.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// An embedder of the stand-in, giving it so long to answer a query,
    /// and twice as long to get somewhere with a call for documents.
    fn giving(local: &Arc<Local>, query: Duration) -> crate::Embedder {
        let embedding = vellev_core::Embedding {
            provider: vellev_core::EmbeddingProvider::Local,
            model: "stand-in".into(),
            dimensions: 1,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        };
        crate::Embedder::new(local.clone(), embedding).answering_within(
            query,
            query * 2,
            query / 10,
        )
    }

    /// A run cannot be stopped, and nothing in the model gives up on one.
    /// Whoever embeds through it waits no longer than a model is given;
    /// and the model, when the run does come back, goes on to the next.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_hangs_is_given_up_on_in_the_time_a_model_is_given() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = giving(&local, ms(200));
        let late = |within: &str| failed(&format!("no answer within {within}"));

        // Left waiting on the run, this fails and does not hang.
        let seconds = Duration::from_secs(5);
        let began = std::time::Instant::now();
        let query = tokio::time::timeout(seconds, embedder.query("q")).await;
        assert_eq!(query, Ok(Err(late("200ms"))));
        let took = began.elapsed();
        assert!((ms(200)..ms(2_000)).contains(&took), "{took:?}");
        assert_eq!(runs.next(), ["q"], "it was run, and has yet to come back");
        let began = std::time::Instant::now();
        let documents = texts("d", 5);
        let documents = tokio::time::timeout(seconds, embedder.documents(&documents)).await;
        let stalled = failed("nothing from the model in 400ms");
        assert_eq!(documents, Ok(Err(stalled)));
        let took = began.elapsed();
        assert!((ms(400)..ms(2_500)).contains(&took), "{took:?}");

        // The run comes back. No one waits for it or for the documents
        // any longer, so the next run is for whoever asks now.
        runs.let_through(2);
        assert_eq!(embedder.query("hen").await, Ok(Some(vec![3.0])));
        assert_eq!(runs.next(), ["hen"]);
    }

    /// The time a model takes to load is no waiting write's getting
    /// nowhere: when the model has loaded, everyone waiting has got
    /// somewhere, and what a write is then told it may wait is counted
    /// from then. Here the model is loaded again after a panic, twice,
    /// with a tenant's writes waiting in the line through both.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_time_a_model_takes_to_load_is_no_waiting_writes_getting_nowhere() {
        let ms = Duration::from_millis;
        let times = Times {
            first_backoff: Duration::from_secs(30),
            ..unpaced()
        };
        let room = Room::new(1, Duration::MAX, None, times);
        let (local, runs) = local_loading("stand-in", 0, room, ms(1_500));
        let embedder = unless_stalled(&local, ms(1_000));
        // While the model is held at a search: a text that will make it
        // panic, then forty texts of tenant t's, then five more of t's
        // behind them, which are given up on if the model gets nowhere
        // with them for a second over what they are told to expect.
        held_at_a_query(&local, &runs).await;
        let bad = start_for(&local, "bad", vec!["panic".into()], Purpose::Document);
        waiting(&local, 1).await;
        let forty = start_for(&local, "t", texts("f", 40), Purpose::Document);
        waiting(&local, 2).await;
        let five = {
            let (embedder, five) = (embedder.clone(), texts("v", 5));
            tokio::spawn(async move { embedder.documents_for("t", &five).await })
        };
        waiting(&local, 3).await;
        runs.let_through(1);
        // The run the two tenants share panics, and is put off. The bad
        // tenant's text is run alone, after a load of a second and a
        // half, and panics again.
        assert_eq!(runs.next().len(), RUN);
        runs.let_through(1);
        assert_eq!(runs.next(), ["panic"]);
        runs.let_through(1);
        assert!(bad.await.unwrap().is_err());
        // Then t's turn, after another load: thirty-two of the forty,
        // held 800 ms, which is 25 ms a text.
        assert_eq!(runs.next().len(), RUN);
        tokio::time::sleep(ms(800)).await;
        runs.let_through(1);
        // The next run, of the eight left and the five, is held 1.4 s.
        // After the run before, the five were told to expect 900 ms:
        // four times the nine texts to theirs at 25 ms each, less
        // nothing, having got somewhere when the model loaded. With the
        // second allowed for a stall that is 1.9 s, and they are waited
        // for. Counted from when they were sent, two loads before,
        // nothing of the 900 ms would be left, and they would be given
        // up on a second into this run.
        assert_eq!(runs.next().len(), 8 + 5);
        tokio::time::sleep(ms(1_400)).await;
        runs.let_through(1);
        assert_eq!(five.await.unwrap().map(|made| made.len()), Ok(5));
        assert_eq!(forty.await.unwrap().map(|made| made.len()), Ok(40));
        assert_eq!(runs.loads.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    /// A model is fetched and loaded when first asked for, which takes as
    /// long as it takes: longer than a query is kept. It is not begun
    /// again for each query given up on, and a write waits for it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_slow_to_load_goes_on_loading_and_a_write_waits_for_it() {
        let ms = Duration::from_millis;
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = loads.clone();
        let load: Load = Box::new(move || {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(ms(500));
            Ok(Box::new(|texts: Vec<&str>| {
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        let local = Arc::new(Local::spawn("stand-in", load, alone()).unwrap());
        let embedder = Arc::new(giving(&local, ms(100)));
        // A write made while the model loads is given the time a load
        // is given, and waits: a load is not a model that has stopped.
        let written = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        // One that writes while the model loads is told, once its write
        // is sent, that it may be as long as a load is given.
        tokio::time::sleep(ms(100)).await;
        let (progress, hears) = Progress::heard();
        let hen = {
            let local = local.clone();
            tokio::spawn(async move {
                local
                    .embed_saying(vec!["hen".into()], Purpose::Document, "t", progress)
                    .await
            })
        };
        tokio::time::sleep(ms(50)).await;
        let told = hears.borrow().allowed;
        let loading = Times::default().loading;
        assert!(
            told.is_some_and(|told| told > loading - ms(1_000) && told <= loading),
            "{told:?}"
        );
        for _ in 0..3 {
            let asked = embedder.query("fox").await;
            assert_eq!(asked, Err(failed("no answer within 100ms")));
        }
        assert_eq!(written.await.unwrap(), Ok(vec![Some(vec![3.0])]));
        assert_eq!(hen.await.unwrap(), Ok(vec![vec![3.0]]));
        assert_eq!(embedder.query("fox").await, Ok(Some(vec![3.0])));
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn a_directory_the_node_may_not_write_keeps_a_model_from_loading_and_a_network_does_not() {
        // As fastembed reports a fetch that failed: what failed, under
        // what it was doing.
        let fetching = |cause: FetchError| {
            fastembed::Error::new(cause).context("Failed to retrieve model.onnx")
        };
        let kept = Path::new("/models");
        // The same failure further down, as it would lie were it wrapped
        // on its way up.
        #[derive(Debug, thiserror::Error)]
        #[error("the model could not be fetched")]
        struct Wrapped(#[source] FetchError);
        let wrapped = fastembed::Error::new(Wrapped(FetchError::IoError(
            ErrorKind::PermissionDenied.into(),
        )));
        assert_eq!(
            unloaded("m", kept, &wrapped).lasting(),
            Some(vellev_core::ModelFault::Setup),
            "{wrapped:#}"
        );
        for kind in [ErrorKind::PermissionDenied, ErrorKind::ReadOnlyFilesystem] {
            let error = fetching(FetchError::IoError(kind.into()));
            assert_eq!(
                unloaded("m", kept, &error),
                EmbedError::Unusable {
                    model: "m".into(),
                    reason: format!("the node may not write /models: {error:#}"),
                    fault: vellev_core::ModelFault::Setup,
                },
                "{kind:?}"
            );
        }
        // What is said of either kind is said without a token quoted in it.
        let token = "hf_abcdefghij0123456789ABCD";
        let quoting = fastembed::Error::msg(format!("401 Unauthorized for {token}"));
        assert_eq!(
            unloaded("m", kept, &quoting),
            failed_as("m", "401 Unauthorized for <redacted>")
        );
        let barred = fastembed::Error::new(FetchError::IoError(ErrorKind::PermissionDenied.into()))
            .context(format!("fetching as {token}"));
        let said = unloaded("m", kept, &barred).to_string();
        assert!(
            said.contains("fetching as <redacted>") && !said.contains(token),
            "{said}"
        );
        let passing = [
            fetching(FetchError::IoError(ErrorKind::StorageFull.into())),
            fetching(FetchError::IoError(ErrorKind::TimedOut.into())),
            fetching(FetchError::TooManyRetries(Box::new(FetchError::IoError(
                ErrorKind::PermissionDenied.into(),
            )))),
            fetching(FetchError::InvalidResume),
            // Not the fetcher's own: a connection refused leave to be made,
            // or anything else that happens to be an I/O error.
            fastembed::Error::new(std::io::Error::from(ErrorKind::PermissionDenied))
                .context("Failed to retrieve model.onnx"),
            fastembed::Error::msg("request error: dns failed"),
        ];
        for error in passing {
            assert_eq!(
                unloaded("m", kept, &error),
                failed_as("m", &format!("{error:#}")),
                "{error:#}"
            );
        }
    }

    fn failed_as(model: &str, reason: &str) -> EmbedError {
        EmbedError::Failed {
            model: model.into(),
            reason: reason.into(),
        }
    }

    /// How many versions of `name` a lock file holds.
    fn versions(lock: &str, name: &str) -> usize {
        let entry = format!("name = \"{name}\"");
        lock.lines().filter(|line| *line == entry).count()
    }

    /// A model directory the node may not write is known by the type of
    /// the fetcher's error. Were this crate built against one version of
    /// the fetcher and fastembed against another, the types would differ,
    /// nothing would fail to build, and that case would quietly be taken
    /// for one that may pass.
    #[test]
    fn the_fetcher_whose_errors_are_known_is_the_one_fastembed_fetches_with() {
        let two = [
            "[[package]]",
            "name = \"hf-hub\"",
            "version = \"0.4.3\"",
            "",
            "[[package]]",
            "name = \"hf-hub\"",
            "version = \"0.5.0\"",
            "",
            "[[package]]",
            "name = \"hf-hub-extras\"",
        ]
        .join("\n");
        assert_eq!(versions(&two, "hf-hub"), 2);
        let lock = include_str!("../../../Cargo.lock");
        assert_eq!(
            versions(lock, "hf-hub"),
            1,
            "this crate and fastembed must use one hf-hub: see Cargo.toml"
        );
        // And fastembed does fetch with it: were it to fetch some other
        // way, one version in the lock file would show nothing.
        assert!(!depends_on(&two, "hf-hub", "fastembed"));
        let with = "[[package]]\nname = \"fastembed\"\ndependencies = [\n \"anyhow\",\n \"hf-hub\",\n]\n\n\
                    [[package]]\nname = \"other\"\ndependencies = [\n \"hf-hub 0.5.0\",\n]\n";
        assert!(depends_on(with, "fastembed", "hf-hub"));
        assert!(depends_on(with, "other", "hf-hub"));
        assert!(!depends_on(with, "fastembed", "hf"));
        assert!(!depends_on(with, "anyhow", "hf-hub"));
        assert!(
            depends_on(lock, "fastembed", "hf-hub"),
            "fastembed no longer fetches models with hf-hub: see Cargo.toml"
        );
    }

    /// Whether the package `name` of a lock file depends on `on`.
    fn depends_on(lock: &str, name: &str, on: &str) -> bool {
        let entry = format!("name = \"{name}\"");
        lock.split("[[package]]")
            .filter(|package| package.lines().any(|line| line == entry))
            .any(|package| {
                package
                    .lines()
                    .filter_map(|line| line.trim().strip_prefix('"'))
                    .filter_map(|line| line.strip_suffix("\","))
                    .any(|dependency| dependency.split(' ').next() == Some(on))
            })
    }

    #[test]
    fn models_are_kept_where_the_node_says_unless_hf_home_says_otherwise() {
        let cache = Path::new("/var/lib/vellev/models");
        assert_eq!(kept_in(cache, None), cache);
        assert_eq!(kept_in(cache, Some("/hf".into())), Path::new("/hf"));
    }

    #[test]
    fn a_model_is_known_by_its_name_at_its_own_size() {
        for model in [
            "BAAI/bge-base-en-v1.5",
            "Xenova/bge-base-en-v1.5",
            "bge-base-en-v1.5",
        ] {
            assert!(known(model, 768).is_ok(), "{model}");
        }
        assert_eq!(
            known("BAAI/bge-base-en-v1.5", 1024).unwrap_err(),
            EmbedError::NoSuchModel("BAAI/bge-base-en-v1.5 has 768 dimensions, not 1024".into())
        );
        assert_eq!(
            known("acme/unheard-of", 768).unwrap_err(),
            EmbedError::NoSuchModel("acme/unheard-of cannot be run locally".into())
        );
    }
}
