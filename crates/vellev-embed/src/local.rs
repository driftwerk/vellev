use std::collections::VecDeque;
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

use crate::{Activity, EmbedError, Provider, Purpose, Result};

/// Texts in one run of the model. A run cannot be interrupted, so this is
/// also how long a query can be kept waiting by documents, and it bounds
/// the memory a run takes.
const RUN: usize = 32;
/// The most characters of a text a model is given. No model here reads
/// more than 512 tokens of a text, and a word of text is seldom less
/// than a token: so for most texts what comes after these is read by
/// none. Not for all. What a tokenizer makes one token of, or none,
/// however long it is, takes up characters here and few tokens there: a
/// word of more than a hundred characters, as a long run of hex, base64
/// or a URL without a break is, is one token whatever its length; and
/// white space, control characters and characters of no width are
/// dropped. A text with enough of those before the cut has less of it
/// read than a model would have read of the whole.
pub(crate) const LONGEST_TEXT: usize = 512 * 100;
/// As many tokens of a text as a model reads. A byte may be a token, as
/// in punctuation, digits and JSON, which run to three to five times the
/// tokens of prose to the byte: so a text counts for its bytes up to
/// this many, and no more.
const READS: usize = 512;
/// What a run may come to, as its texts times what the longest of them
/// counts for. A model is given the texts of a run padded to the longest
/// in tokens, so that is what a run costs: thirty-two texts of 64 bytes
/// cost what eight of 256 or four of 512 or more do, and one long text
/// among thirty-one short ones makes thirty-two long ones of them. A run
/// is of as many texts as come to this, in the order they wait, and of
/// one at the least. It is sized for the densest text, a token to the
/// byte: prose, at four bytes or so to the token, makes runs a quarter
/// of what the model could take in the same time.
const RUN_COSTS: usize = RUN * 64;
/// How many runs of queries are made one after another while documents
/// wait, before the documents have a run: so that searches, however many
/// and however fast they come, leave writes one run in every three.
const QUERY_RUNS: usize = 2;
/// How long a model is given to load before those waiting on it stop
/// taking it to be at work: fetching a model for the first time over a
/// slow link takes minutes.
const LOADING: Duration = Duration::from_secs(10 * 60);
/// Callers that can be waiting for the model before others wait to join
/// them.
const WAITING: usize = 256;

/// A loaded model: a vector for each text, or why not.
type Model = Box<dyn FnMut(Vec<&str>) -> std::result::Result<Vec<Vec<f32>>, String>>;
type Load = Box<dyn FnMut() -> Result<Model> + Send>;

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
    loaded: Mutex<Vec<String>>,
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
    /// go.
    pub(crate) fn new(most: usize, idle: Duration, own: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            most: most.max(1),
            idle,
            own,
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

struct Job {
    texts: Vec<String>,
    purpose: Purpose,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
}

/// A job and the vectors made for it so far.
struct Work {
    job: Job,
    done: Vec<Vec<f32>>,
}

/// A model run in this process on a thread of its own, so that inference
/// never holds up the async runtime. Waiting callers' texts are run
/// together, queries before documents, and a caller that has gone away is
/// dropped before its next run. The model is fetched into the directory
/// models are [`kept_in`] and loaded when first needed, if there is
/// [`Room`] for it, and again after a failure or after it was let go for
/// not being used.
#[derive(Debug)]
pub struct Local {
    model: String,
    jobs: mpsc::Sender<Job>,
    /// What the model's thread has lately done, for anyone.
    activity: Arc<Activity>,
}

impl Local {
    /// `model` is a Hugging Face name such as `BAAI/bge-base-en-v1.5`. It
    /// is kept loaded once used, whatever else is.
    pub fn new(model: &str, dimensions: u32, cache: &Path) -> Result<Self> {
        let room = Room::new(1, Duration::MAX, None);
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

    fn spawn(model: &str, load: Load, room: Arc<Room>) -> Result<Self> {
        Self::spawn_timed(model, load, room, crate::clock::tokio())
    }

    /// As [`Self::spawn`], the model's activity read by `clock`: the
    /// clock is handed to the model's thread, which reads it there.
    fn spawn_timed(
        model: &str,
        load: Load,
        room: Arc<Room>,
        clock: Arc<dyn crate::clock::Clock>,
    ) -> Result<Self> {
        let (jobs, waiting) = mpsc::channel(WAITING);
        let name = model.to_string();
        let activity = Arc::new(Activity::timed_by(clock));
        let acts = activity.clone();
        std::thread::Builder::new()
            .name(format!("embed {model}"))
            .spawn(move || run(&name, load, &room, waiting, &acts))
            .map_err(|error| failure(model, format!("no thread to run it on: {error}")))?;
        Ok(Self {
            model: model.to_string(),
            jobs,
            activity,
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

/// `text`, no longer than [`LONGEST_TEXT`] characters: the rest is read
/// by no model, and is not carried to one. It is cut where a character
/// ends.
fn cut(mut text: String) -> String {
    if text.len() > LONGEST_TEXT {
        if let Some((at, _)) = text.char_indices().nth(LONGEST_TEXT) {
            text.truncate(at);
        }
    }
    text
}

/// Which line the next run is of, when `queries` and `documents` of them
/// wait, after `query_runs` runs of queries that documents waited
/// through since they last had one. Queries go first, someone waiting on each: but
/// after [`QUERY_RUNS`] runs of them with documents waiting, the
/// documents have a run.
fn queries_next(queries: usize, documents: usize, query_runs: usize) -> Option<bool> {
    match (queries, documents) {
        (0, 0) => None,
        (0, _) => Some(false),
        (_, 0) => Some(true),
        _ => Some(query_runs < QUERY_RUNS),
    }
}

fn run(
    name: &str,
    mut load: Load,
    room: &Arc<Room>,
    mut waiting: mpsc::Receiver<Job>,
    activity: &Activity,
) {
    let mut model: Option<Loaded> = None;
    // Runs of queries made, with documents waiting, since documents last
    // had a run.
    let mut query_runs = 0;
    let mut queries: VecDeque<Work> = VecDeque::new();
    let mut documents: VecDeque<Work> = VecDeque::new();
    let queue = |job: Job, queries: &mut VecDeque<Work>, documents: &mut VecDeque<Work>| {
        let queue = match job.purpose {
            Purpose::Query => queries,
            Purpose::Document => documents,
        };
        queue.push_back(Work {
            job,
            done: Vec::new(),
        });
    };
    loop {
        if queries.is_empty() && documents.is_empty() {
            // With nothing to run, a model that is loaded is kept for as
            // long as the node keeps one unused, and then let go, and
            // with it its place: it is loaded again when next asked for,
            // if there is room then.
            let within = match model {
                Some(_) => room.kept_idle(name),
                None => Duration::MAX,
            };
            match next_within(&mut waiting, within) {
                Ok(Some(job)) => queue(job, &mut queries, &mut documents),
                Ok(None) => return,
                Err(Idle) => {
                    model = None;
                    tracing::info!(
                        model = name,
                        "a local embedding model not used for {:?} is let go",
                        within
                    );
                    continue;
                }
            }
        }
        while let Ok(job) = waiting.try_recv() {
            queue(job, &mut queries, &mut documents);
        }
        for queue in [&mut queries, &mut documents] {
            queue.retain(|work| !work.job.reply.is_closed());
        }
        let next = match queries_next(queries.len(), documents.len(), query_runs) {
            // Whoever was waiting has gone away, as one given up on
            // has: the model is not run on nothing.
            None => continue,
            Some(true) => {
                // Only a run of queries made while documents wait is one
                // the documents waited through.
                query_runs = match documents.is_empty() {
                    true => 0,
                    false => query_runs + 1,
                };
                &mut queries
            }
            Some(false) => {
                query_runs = 0;
                &mut documents
            }
        };
        advance(name, &mut load, room, &mut model, next, activity);
    }
}

/// Run the model once, on the texts at the front of `queue`, and answer
/// the callers it finishes or fails. A model not yet loaded is loaded
/// first, if there is a place for it `among` those the node has loaded.
fn advance(
    name: &str,
    load: &mut Load,
    among: &Arc<Room>,
    model: &mut Option<Loaded>,
    queue: &mut VecDeque<Work>,
    activity: &Activity,
) {
    let taken = taken(queue.iter().map(|work| &work.job.texts[work.done.len()..]));
    let texts: Vec<&str> = queue
        .iter()
        .zip(&taken)
        .flat_map(|(work, take)| &work.job.texts[work.done.len()..][..*take])
        .map(String::as_str)
        .collect();
    let asked = texts.len();
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let loaded = match model.as_mut() {
            Some(loaded) => loaded,
            None => {
                // The place is taken first, and given up again should
                // the model not load. Those who wait are not to take a
                // model that is loading for one that has stopped.
                let place = among.place(name)?;
                activity.loading(LOADING);
                let loaded = load()?;
                // Loaded: the time it took is not held against anyone.
                // It made nothing.
                activity.loaded();
                model.insert(Loaded {
                    model: loaded,
                    _place: place,
                })
            }
        };
        (loaded.model)(texts).map_err(|reason| failure(name, reason))
    }));
    // However a load ended, it is over: one that failed or panicked is
    // not a model at work.
    activity.not_loading();
    let vectors = match ran {
        Ok(Ok(vectors)) if vectors.len() == asked => Ok(vectors),
        Ok(Ok(vectors)) => Err(failure(
            name,
            format!("{} vectors for {asked} texts", vectors.len()),
        )),
        Ok(Err(error)) => Err(error),
        Err(_) => {
            *model = None;
            Err(failure(name, "the model panicked"))
        }
    };
    match vectors {
        Ok(vectors) => {
            // The model is at work, for whoever this was.
            activity.made();
            let mut vectors = vectors.into_iter();
            for (work, take) in queue.iter_mut().zip(&taken) {
                work.done.extend(vectors.by_ref().take(*take));
            }
            while queue
                .front()
                .is_some_and(|work| work.done.len() == work.job.texts.len())
            {
                if let Some(Work { job, done }) = queue.pop_front() {
                    let _ = job.reply.send(Ok(done));
                }
            }
        }
        Err(error) => queue.drain(..taken.len()).for_each(|work| {
            let _ = work.job.reply.send(Err(error.clone()));
        }),
    }
}

/// How many texts of each job at the front of a line go into the next
/// run, given what each job has left: as many, in the order they wait, as
/// are no more than [`RUN`] and cost no more than [`RUN_COSTS`], the
/// texts so far times the longest of them; and one at the least. No text
/// is passed over for one behind it.
fn taken<'a>(left: impl Iterator<Item = &'a [String]>) -> Vec<usize> {
    let (mut texts, mut longest) = (0usize, 0usize);
    let mut taken = Vec::new();
    'jobs: for left in left {
        let mut take = 0;
        for text in left {
            let reads = longest.max(text.len().min(READS));
            if texts > 0 && (texts == RUN || (texts + 1) * reads > RUN_COSTS) {
                if take > 0 {
                    taken.push(take);
                }
                break 'jobs;
            }
            (texts, longest, take) = (texts + 1, reads, take + 1);
        }
        match take {
            0 => break,
            _ => taken.push(take),
        }
    }
    taken
}

#[async_trait]
impl Provider for Local {
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let stopped = || failure(&self.model, "the model's thread has stopped");
        let (reply, answer) = oneshot::channel();
        let job = Job {
            texts: texts.into_iter().map(cut).collect(),
            purpose,
            reply,
        };
        self.jobs.send(job).await.map_err(|_| stopped())?;
        answer.await.map_err(|_| stopped())?
    }

    /// One thread runs the model for everyone, a run at a time: a call
    /// waits its turn behind others, and for the model to load.
    fn activity(&self) -> Option<&Activity> {
        Some(&self.activity)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{channel, Receiver, Sender};

    use super::*;

    /// Room for one model, kept however long it goes unused.
    fn alone() -> Arc<Room> {
        Room::new(1, Duration::MAX, None)
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
        local_timed(model, failing_loads, room, crate::clock::tokio())
    }

    /// As [`local_among`], the model's activity read by `clock`.
    fn local_timed(
        model: &str,
        failing_loads: usize,
        room: Arc<Room>,
        clock: Arc<dyn crate::clock::Clock>,
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
            let (report, gate) = (report.clone(), gate.clone());
            Ok(Box::new(move |texts: Vec<&str>| {
                report
                    .send(texts.iter().map(|text| text.to_string()).collect())
                    .unwrap();
                gate.lock().unwrap().recv().unwrap();
                assert!(!texts.contains(&"panic"), "told to");
                if texts.contains(&"fail") {
                    return Err("told to".to_string());
                }
                if texts.contains(&"quote") {
                    return Err("401 for hf_abcdefghij0123456789ABCD".to_string());
                }
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        (
            Arc::new(Local::spawn_timed(model, load, room, clock).unwrap()),
            Runs { ran, go, loads },
        )
    }

    fn texts(prefix: &str, count: usize) -> Vec<String> {
        (0..count).map(|n| format!("{prefix}{n}")).collect()
    }

    fn start(
        local: &Arc<Local>,
        texts: Vec<String>,
        purpose: Purpose,
    ) -> tokio::task::JoinHandle<Result<Vec<Vec<f32>>>> {
        let local = local.clone();
        tokio::spawn(async move { local.embed(texts, purpose).await })
    }

    /// A job sent to the model's thread, and what it will be answered:
    /// sent when this returns, and not only on its way, as a job counted
    /// from the room left in the channel may be.
    async fn sent(
        local: &Local,
        texts: Vec<String>,
        purpose: Purpose,
    ) -> oneshot::Receiver<Result<Vec<Vec<f32>>>> {
        let (reply, answer) = oneshot::channel();
        let job = Job {
            texts,
            purpose,
            reply,
        };
        local.jobs.send(job).await.expect("the model's thread");
        answer
    }

    fn failed(reason: &str) -> EmbedError {
        EmbedError::Failed {
            model: "stand-in".into(),
            reason: reason.into(),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_long_job_is_run_in_parts_and_waiting_callers_are_run_together() {
        let (local, runs) = local(0);
        let long = start(&local, texts("d", 70), Purpose::Document);
        assert_eq!(runs.next(), texts("d", 70)[..32]);
        let short: Vec<_> = (0..3)
            .map(|n| start(&local, vec![format!("s{n}")], Purpose::Document))
            .collect();
        tokio::time::sleep(Duration::from_millis(100)).await;
        runs.let_through(4);
        assert_eq!(runs.next(), texts("d", 70)[32..64]);
        let last = runs.next();
        assert_eq!(last[..6], texts("d", 70)[64..]);
        assert_eq!(last.len(), 9);

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
        tokio::time::sleep(Duration::from_millis(100)).await;
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
        tokio::time::sleep(Duration::from_millis(100)).await;
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

    #[tokio::test(flavor = "multi_thread")]
    async fn a_query_waits_for_the_run_under_way_and_no_more_however_many_documents_wait() {
        let (local, runs) = local(0);
        // Ten runs of documents from one caller, and more from others.
        let long = start(&local, texts("d", 320), Purpose::Document);
        assert_eq!(runs.next(), texts("d", 320)[..32]);
        let others: Vec<_> = (0..5)
            .map(|n| start(&local, texts(&format!("o{n}-"), 40), Purpose::Document))
            .collect();
        let query = start(&local, vec!["q".into()], Purpose::Query);
        tokio::time::sleep(Duration::from_millis(100)).await;

        // The run under way cannot be stopped. The next is the query's,
        // and the query's alone.
        runs.let_through(2);
        assert_eq!(runs.next(), ["q"]);
        assert_eq!(query.await.unwrap().unwrap(), [vec![1.0]]);
        assert_eq!(runs.next(), texts("d", 320)[32..64]);
        assert!(!long.is_finished());

        // So is each query that comes while documents are run.
        for n in 0..3 {
            let query = start(&local, vec![format!("again{n}")], Purpose::Query);
            tokio::time::sleep(Duration::from_millis(100)).await;
            runs.let_through(2);
            assert_eq!(runs.next(), [format!("again{n}")]);
            assert_eq!(query.await.unwrap().unwrap(), [vec![6.0]]);
            assert_eq!(runs.next().len(), 32, "and then documents again");
        }
        runs.let_through(32);
        assert_eq!(long.await.unwrap().unwrap().len(), 320);
        for other in others {
            assert_eq!(other.await.unwrap().unwrap().len(), 40);
        }
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
    async fn a_failed_run_fails_only_the_callers_in_it() {
        let (local, runs) = local(0);
        let first = start(&local, texts("d", 32), Purpose::Document);
        assert_eq!(runs.next().len(), 32);
        let failing = start(&local, vec!["fail".into()], Purpose::Document);
        let beside = start(&local, vec!["beside".into()], Purpose::Document);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let after = start(&local, texts("e", 40), Purpose::Document);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let behind = start(&local, texts("f", 5), Purpose::Document);
        tokio::time::sleep(Duration::from_millis(100)).await;
        runs.let_through(4);
        assert_eq!(first.await.unwrap().unwrap().len(), 32);
        assert_eq!(failing.await.unwrap(), Err(failed("told to")));
        assert_eq!(beside.await.unwrap(), Err(failed("told to")));
        assert_eq!(after.await.unwrap(), Err(failed("told to")));
        assert_eq!(behind.await.unwrap().unwrap().len(), 5);
        assert_eq!(
            local.embed(vec!["fox".into()], Purpose::Query).await,
            Ok(vec![vec![3.0]])
        );
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
    /// How long a test waits on what a right build does within a bound
    /// of at most a second or so, before failing rather than hanging: a
    /// hundred times as long and more.
    const BACKSTOP: Duration = Duration::from_secs(60);

    /// Waits until `count` calls are asleep on `clock`, each until a
    /// time still to come: whatever the clock last woke has looked and
    /// gone back to sleep, or gone. When each ends, earliest first.
    async fn asleep_on(clock: &crate::clock::Manual, count: usize) -> Vec<tokio::time::Instant> {
        use crate::clock::Clock;
        let began = std::time::Instant::now();
        loop {
            let asleep = clock.asleep();
            if asleep.len() == count && asleep.iter().all(|at| *at > clock.now()) {
                return asleep;
            }
            assert!(
                began.elapsed() < BACKSTOP,
                "{} asleep, not {count}",
                asleep.len()
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

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
        let room = Room::new(2, Duration::from_secs(60 * 60), None);
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
        let room = Room::new(1, idle, None);
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
        let room = Room::new(2, idle, Some("own".into()));
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
        // When the other was last used is no later than the moment
        // before it was last asked.
        let mut last_used = std::time::Instant::now();
        for _ in 0..3 {
            assert_eq!(fox(&own).await, Ok(vec![vec![3.0]]));
            last_used = std::time::Instant::now();
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
        until(|| room.loaded() == ["own"]).await;
        assert!(last_used.elapsed() >= idle, "{:?}", last_used.elapsed());
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
        let room = Room::new(1, idle, Some("own".into()));
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
        let room = Room::new(1, Duration::from_secs(60 * 60), None);
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
    /// and giving up on a call for documents when it has made nothing for
    /// anyone for twice as long.
    fn giving(local: &Arc<Local>, query: Duration) -> crate::Embedder {
        let embedding = vellev_core::Embedding {
            provider: vellev_core::EmbeddingProvider::Local,
            model: "stand-in".into(),
            dimensions: 1,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        };
        crate::Embedder::new(local.clone(), embedding).answering_within(query, query * 2)
    }

    /// A run cannot be stopped, and nothing in the model gives up on one.
    /// Whoever embeds through it waits no longer than a model is given;
    /// and the model, when the run does come back, goes on to the next.
    /// The run is held, so the clock is paused while it is: it moves only
    /// when nothing is left to do but wait, and so by exactly the bounds.
    #[tokio::test(start_paused = true)]
    async fn a_model_that_hangs_is_given_up_on_in_the_time_a_model_is_given() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = Arc::new(giving(&local, ms(200)));
        let late = |within: &str| failed(&format!("no answer within {within}"));

        // Left waiting on the run, this fails and does not hang. The run
        // is waited for on a thread of the blocking pool, which keeps the
        // paused clock where it is until it is seen: so the query was run
        // before any time passed.
        let runs = Arc::new(std::sync::Mutex::new(runs));
        let next = |runs: &Arc<std::sync::Mutex<Runs>>| {
            let runs = runs.clone();
            tokio::task::spawn_blocking(move || runs.lock().unwrap().next())
        };
        let seconds = Duration::from_secs(5);
        let began = tokio::time::Instant::now();
        let query = {
            let embedder = embedder.clone();
            tokio::spawn(async move { tokio::time::timeout(seconds, embedder.query("q")).await })
        };
        let ran = next(&runs).await.unwrap();
        assert_eq!(ran, ["q"], "it was run, and has yet to come back");
        assert_eq!(began.elapsed(), Duration::ZERO);
        assert_eq!(query.await.unwrap(), Ok(Err(late("200ms"))));
        // To the millisecond the timer keeps time in.
        let took = began.elapsed();
        assert!((ms(200)..=ms(200) + ms(1)).contains(&took), "{took:?}");
        // A write behind it is given up on when the model has made
        // nothing for anyone for as long as it may go without: twice as
        // long as a query is given, here.
        let began = tokio::time::Instant::now();
        let documents = texts("d", 5);
        let documents = tokio::time::timeout(seconds, embedder.documents(&documents)).await;
        let stopped = failed("the model has made nothing for anyone in 400ms");
        assert_eq!(documents, Ok(Err(stopped)));
        // To the millisecond the timer keeps time in.
        let took = began.elapsed();
        assert!((ms(400)..=ms(400) + ms(1)).contains(&took), "{took:?}");

        // The run comes back, on a clock that runs again: the model's
        // thread answers in its own time. No one waits for it or for the
        // documents any longer, so the next run is for whoever asks now.
        tokio::time::resume();
        runs.lock().unwrap().let_through(2);
        assert_eq!(embedder.query("hen").await, Ok(Some(vec![3.0])));
        assert_eq!(next(&runs).await.unwrap(), ["hen"]);
    }

    /// A model is fetched and loaded when first asked for, which takes as
    /// long as it takes: longer than a caller is kept. It is not begun
    /// again for each caller given up on.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_slow_to_load_goes_on_loading_when_its_first_caller_is_given_up_on() {
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
        let embedder = giving(&local, ms(100));
        for _ in 0..3 {
            let asked = embedder.query("fox").await;
            assert_eq!(asked, Err(failed("no answer within 100ms")));
        }
        tokio::time::sleep(ms(500)).await;
        assert_eq!(embedder.query("fox").await, Ok(Some(vec![3.0])));
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// As long as no test waits: a run that takes this never comes back.
    const NEVER: Duration = Duration::from_secs(60 * 60);

    /// A stand-in that takes `load_takes` to load, and then `run_takes`
    /// over each run.
    fn slow_to_load(load_takes: Duration, run_takes: Duration) -> Arc<Local> {
        let load: Load = Box::new(move || {
            std::thread::sleep(load_takes);
            Ok(Box::new(move |texts: Vec<&str>| {
                std::thread::sleep(run_takes);
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        Arc::new(Local::spawn("stand-in", load, alone()).unwrap())
    }

    /// A text is cut to what a model reads of it, where a character
    /// ends: a model is carried no more.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_is_cut_to_what_a_model_reads_of_it_where_a_character_ends() {
        assert_eq!(LONGEST_TEXT, 51_200);
        assert_eq!(cut("short".into()), "short");
        let long = "é".repeat(LONGEST_TEXT + 7);
        let kept = cut(long.clone());
        assert_eq!(kept.chars().count(), LONGEST_TEXT);
        assert!(long.starts_with(&kept));
        let exactly = "é".repeat(LONGEST_TEXT);
        assert_eq!(cut(exactly.clone()), exactly);
        // What the model is run on is cut, for a write and a search alike.
        let (local, runs) = local(0);
        for purpose in [Purpose::Document, Purpose::Query] {
            let asked = start(&local, vec![long.clone(), "fox".into()], purpose);
            let ran = runs.next();
            assert_eq!(ran.len(), 2);
            assert_eq!(ran[0].chars().count(), LONGEST_TEXT);
            assert_eq!(ran[1], "fox");
            runs.let_through(1);
            assert_eq!(asked.await.unwrap().unwrap().len(), 2);
        }
    }

    /// A run costs its texts times the longest of them, the model being
    /// given them padded to the longest: so a run is of as many texts, in
    /// the order they wait, as come to no more than thirty-two texts of
    /// 256 bytes do, of thirty-two at the most and one at the least.
    #[test]
    fn a_run_is_of_as_many_texts_as_cost_no_more_than_a_run_may() {
        assert_eq!((RUN, READS, RUN_COSTS), (32, 512, 2_048));
        // How many texts of each job of these lengths go into a run.
        let run_of = |jobs: &[&[usize]]| {
            let jobs: Vec<Vec<String>> = jobs
                .iter()
                .map(|lengths| lengths.iter().map(|length| "x".repeat(*length)).collect())
                .collect();
            taken(jobs.iter().map(Vec::as_slice))
        };
        // Short texts: thirty-two, across the jobs in order.
        assert_eq!(run_of(&[&[10; 40]]), [32]);
        assert_eq!(run_of(&[&[10; 5], &[10; 20], &[10; 20]]), [5, 20, 7]);
        // As many as cost what thirty-two of 64 bytes do: thirty-two of
        // 64, eight of 256, four of 512, and one fewer of a byte more.
        assert_eq!(run_of(&[&[64; 40]]), [32]);
        assert_eq!(run_of(&[&[65; 40]]), [31]);
        assert_eq!(run_of(&[&[256; 40]]), [8]);
        assert_eq!(run_of(&[&[257; 40]]), [7]);
        assert_eq!(run_of(&[&[511; 40]]), [4]);
        assert_eq!(run_of(&[&[512; 40]]), [4]);
        // A text counts for no more than the tokens a model reads of it:
        // four of the longest make a run, however long they are.
        assert_eq!(run_of(&[&[513; 40]]), [4]);
        assert_eq!(run_of(&[&[2_048; 9]]), [4]);
        assert_eq!(run_of(&[&[400_000; 9]]), [4]);
        // One long text makes the texts with it as long: thirty-one short
        // ones and a page are not one run, but the short ones before the
        // page, and then the page with as many as fit beside it.
        let mut page_last = [10; 32];
        page_last[31] = 2_000;
        assert_eq!(run_of(&[&page_last]), [31]);
        assert_eq!(run_of(&[&[2_000, 10, 10, 10, 10, 10]]), [4]);
        assert_eq!(run_of(&[&[10, 10, 2_000, 10]]), [4]);
        assert_eq!(run_of(&[&[10, 10, 10, 10, 2_000, 10]]), [4]);
        // No text is passed over for one behind it, in its job or the
        // next: what does not fit ends the run.
        assert_eq!(run_of(&[&[300; 7], &[2_000], &[10; 9]]), [6]);
        assert_eq!(run_of(&[&[300; 3], &[2_000], &[10; 9]]), [3, 1]);
        assert_eq!(run_of(&[&[300; 3], &[10; 9]]), [3, 3]);
        assert_eq!(run_of(&[&[10, 10, 10, 10, 2_000], &[10; 9]]), [4]);
        // One text at the least, whatever it costs; and none of nothing.
        assert_eq!(run_of(&[&[400_000]]), [1]);
        assert_eq!(run_of(&[]), [] as [usize; 0]);
    }

    /// Queries go first, but documents that wait have a run after two
    /// runs of queries.
    #[test]
    fn documents_that_wait_have_a_run_after_two_runs_of_queries() {
        assert_eq!(QUERY_RUNS, 2);
        assert_eq!(queries_next(0, 0, 0), None);
        assert_eq!(queries_next(5, 0, 0), Some(true));
        assert_eq!(queries_next(5, 0, 99), Some(true), "nothing else waits");
        assert_eq!(queries_next(0, 5, 0), Some(false));
        assert_eq!(queries_next(5, 5, 0), Some(true));
        assert_eq!(queries_next(5, 5, 1), Some(true));
        assert_eq!(queries_next(5, 5, 2), Some(false));
        assert_eq!(queries_next(5, 5, 3), Some(false));
    }

    /// Searches that never stop coming do not keep a write from the
    /// model: with more searches waiting than the runs can hold, every
    /// third run is the write's, until it is done.
    #[tokio::test(flavor = "multi_thread")]
    async fn searches_however_many_leave_a_write_every_third_run() {
        let (local, runs) = local(0);
        let hold = start(&local, vec!["hold".into()], Purpose::Query);
        assert_eq!(runs.next(), ["hold"]);
        // While the model is held: seventy texts to write, and three
        // hundred searches.
        let write = start(&local, texts("d", 70), Purpose::Document);
        let searches: Vec<_> = (0..300)
            .map(|n| start(&local, vec![format!("q{n}")], Purpose::Query))
            .collect();
        until(|| WAITING - local.jobs.capacity() >= 200).await;
        runs.let_through(1);
        hold.await.unwrap().unwrap();
        // Two runs of searches, then one of the write, and so on: the
        // write is done in the ninth run, with searches still waiting.
        let mut kinds = String::new();
        for _ in 0..9 {
            let run = runs.next();
            assert!(run.len() <= RUN);
            kinds.push(if run[0].starts_with('d') { 'd' } else { 'q' });
            runs.let_through(1);
        }
        assert_eq!(kinds, "qqdqqdqqd");
        assert_eq!(write.await.unwrap().unwrap().len(), 70);
        // And with no write waiting, the searches have every run.
        runs.let_through(20);
        for search in searches {
            assert_eq!(search.await.unwrap().unwrap().len(), 1);
        }
    }

    /// How long a search waits, stated as runs: the run under way, then a
    /// run of searches for each run's worth of searches up to and with
    /// its own, and a run of writes after every two runs of searches
    /// while a write waits. How many runs of writes that comes to
    /// depends on how many runs of searches the write has waited through
    /// already when the search comes: none, one, or the two after which
    /// its own run is next. A run of searches holds what any run does,
    /// by the length of each as it is sent: thirty-two of up to 64
    /// bytes, nineteen of 105, four of 512 or more.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_search_waits_a_run_of_writes_for_every_two_runs_of_searches_ahead_of_it() {
        // The runs made until every one of so many searches of such a
        // length is answered, with a write waiting throughout that has
        // waited through so many runs of searches before they come.
        async fn runs_until_searched(searches: usize, bytes: usize, waited: usize) -> String {
            runs_until_searched_after(searches, bytes, waited, 0).await
        }
        // As above, after so many runs of searches made alone, with no
        // write waiting, before the write comes.
        async fn runs_until_searched_after(
            searches: usize,
            bytes: usize,
            waited: usize,
            alone: usize,
        ) -> String {
            let (local, runs) = local(0);
            let hold = sent(&local, vec!["hold".into()], Purpose::Query).await;
            assert_eq!(runs.next(), ["hold"]);
            // Searches with no write waiting, a run each.
            for before in 0..alone {
                let search = sent(&local, vec![format!("alone {before}")], Purpose::Query).await;
                runs.let_through(1);
                assert_eq!(runs.next(), [format!("alone {before}")]);
                tokio::spawn(search);
            }
            let write = sent(&local, texts("d", 20 * RUN), Purpose::Document).await;
            // A search by itself for each run the write is to have
            // waited through, each held while the next comes.
            for before in 0..waited {
                let search = sent(&local, vec![format!("before {before}")], Purpose::Query).await;
                runs.let_through(1);
                assert_eq!(runs.next(), [format!("before {before}")]);
                tokio::spawn(search);
            }
            let mut asked = Vec::new();
            for n in 0..searches {
                let text = format!("q{n:04}{}", "x".repeat(bytes - 5));
                asked.push(sent(&local, vec![text], Purpose::Query).await);
            }
            runs.let_through(1);
            let mut kinds = String::new();
            let mut answered = 0;
            while answered < searches {
                let run = runs.next();
                let of_writes = run[0].starts_with('d');
                kinds.push(if of_writes { 'd' } else { 'q' });
                answered += if of_writes { 0 } else { run.len() };
                runs.let_through(1);
            }
            for search in asked {
                assert_eq!(search.await.unwrap().unwrap().len(), 1);
            }
            drop((hold, write));
            kinds
        }
        let count = |kinds: &str, kind: char| kinds.chars().filter(|k| *k == kind).count();
        // 255 searches of 64 bytes, thirty-two to a run: eight runs of
        // them, and three runs of writes or four.
        assert_eq!(runs_until_searched(255, 64, 0).await, "qqdqqdqqdqq");
        assert_eq!(runs_until_searched(255, 64, 1).await, "qdqqdqqdqqdq");
        assert_eq!(runs_until_searched(255, 64, 2).await, "dqqdqqdqqdqq");
        // 250 of 105 bytes, as a search of 48 bytes is with the 57 bytes
        // bge puts before it, nineteen to a run: fourteen runs of them,
        // and six runs of writes or seven.
        for (waited, writes) in [(0, 6), (1, 7), (2, 7)] {
            let kinds = runs_until_searched(250, 105, waited).await;
            assert_eq!((count(&kinds, 'q'), count(&kinds, 'd')), (14, writes));
        }
        // Sixty-four of 512 bytes, four to a run: sixteen runs of them,
        // and seven runs of writes or eight.
        for (waited, writes) in [(0, 7), (1, 8), (2, 8)] {
            let kinds = runs_until_searched(64, 512, waited).await;
            assert_eq!((count(&kinds, 'q'), count(&kinds, 'd')), (16, writes));
        }
        // One search alone has its own run next, unless the write has
        // waited through two runs of searches: then the write's is.
        assert_eq!(runs_until_searched(1, 64, 0).await, "q");
        assert_eq!(runs_until_searched(1, 64, 1).await, "q");
        assert_eq!(runs_until_searched(1, 64, 2).await, "dq");
        // Runs of searches made while no write waits are none that a
        // write waited through: a write that comes after three of them
        // waits two more before its run, as any other.
        assert_eq!(
            runs_until_searched_after(255, 64, 0, 3).await,
            "qqdqqdqqdqq"
        );
        assert_eq!(runs_until_searched_after(1, 64, 0, 3).await, "q");
    }

    /// A text counts for a run as it is sent to the model, with what its
    /// collection puts before it: a search of 48 bytes with the 57 bytes
    /// bge puts before a search is 105, and nineteen go to a run.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_counts_for_a_run_as_it_is_sent_with_what_is_put_before_it() {
        let before = "Represent this sentence for searching relevant passages: ";
        assert_eq!(before.len(), 57);
        let (local, runs) = local(0);
        let embedding = vellev_core::Embedding {
            provider: vellev_core::EmbeddingProvider::Local,
            model: "stand-in".into(),
            dimensions: 1,
            query_prefix: Some(before.into()),
            document_prefix: None,
            properties: Vec::new(),
        };
        let embedder = Arc::new(crate::Embedder::new(local.clone(), embedding));
        let hold = start(&local, vec!["hold".into()], Purpose::Query);
        assert_eq!(runs.next(), ["hold"]);
        let asked: Vec<_> = (0..40)
            .map(|n| {
                let embedder = embedder.clone();
                let text = format!("{n:02}{}", "x".repeat(46));
                tokio::spawn(async move { embedder.query(&text).await })
            })
            .collect();
        until(|| local.jobs.capacity() == WAITING - 40).await;
        runs.let_through(1);
        hold.await.unwrap().unwrap();
        let run = runs.next();
        assert!(run
            .iter()
            .all(|text| text.len() == 105 && text.starts_with(before)));
        assert_eq!(run.len(), 19);
        runs.let_through(3);
        for search in asked {
            assert_eq!(search.await.unwrap(), Ok(Some(vec![105.0])));
        }
    }

    /// A run that fails is not the model at work. A write that waits
    /// behind runs the model fails, one after another, is given up on
    /// when nothing has been made for anyone for as long as a model may
    /// go without, though the line ahead of it is moving. The runs fail
    /// for as long as it waits: were a run failed the model at work, it
    /// would wait through all of them, and no clock decides which.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_that_fails_is_not_the_model_at_work() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = Arc::new(giving(&local, ms(200)));
        // More writes the model fails than there is time to fail before
        // the backstop, each a run by itself: a text that fails and three
        // that fill the run.
        let failing: Vec<_> = (0..1_000)
            .map(|_| {
                let mut texts = vec!["fail".to_string()];
                texts.extend((0..3).map(|n| format!("{n}{}", "x".repeat(READS))));
                start(&local, texts, Purpose::Document)
            })
            .collect();
        assert_eq!(runs.next().len(), 4);
        let began = std::time::Instant::now();
        let behind = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        // A run fails every hundred milliseconds, until the write behind
        // them is given up on.
        let mut failed_meanwhile = 0;
        while !behind.is_finished() {
            assert!(
                began.elapsed() < BACKSTOP,
                "not given up on through {failed_meanwhile} runs failed"
            );
            tokio::time::sleep(ms(100)).await;
            runs.let_through(1);
            failed_meanwhile += 1;
        }
        let given_up_on = behind.await.unwrap();
        let stopped = failed("the model has made nothing for anyone in 400ms");
        assert_eq!(given_up_on, Err(stopped));
        assert!(began.elapsed() >= ms(400), "{:?}", began.elapsed());
        runs.let_through(failing.len());
        for write in failing {
            assert!(write.await.unwrap().is_err());
        }
    }

    /// Callers arriving are not the model at work. A model that has
    /// stopped is given up on by one who waits on it though others keep
    /// coming to wait on it too.
    #[tokio::test(flavor = "multi_thread")]
    async fn callers_arriving_are_not_the_model_at_work() {
        let ms = Duration::from_millis;
        let local = slow_to_load(ms(0), NEVER);
        let embedder = Arc::new(giving(&local, ms(200)));
        let began = std::time::Instant::now();
        let first = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        // Another arrives every hundred milliseconds, for as long as the
        // first waits: were a caller arriving the model at work, the
        // first would wait for ever.
        let mut arrived = 0;
        while !first.is_finished() {
            assert!(
                began.elapsed() < BACKSTOP,
                "not given up on through {arrived} callers arriving"
            );
            tokio::time::sleep(ms(100)).await;
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&[format!("hen {arrived}")]).await });
            arrived += 1;
        }
        let given_up_on = first.await.unwrap();
        let stopped = failed("the model has made nothing for anyone in 400ms");
        assert_eq!(given_up_on, Err(stopped));
        assert!(began.elapsed() >= ms(400), "{:?}", began.elapsed());
    }

    /// Loads that fail, one after another, each slower than a write is
    /// given to wait with the model not at work: the write is given up
    /// on, not kept for as many loads as there are writes ahead of it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_is_given_up_on_behind_loads_that_keep_failing() {
        let ms = Duration::from_millis;
        let load: Load = Box::new(move || {
            std::thread::sleep(ms(300));
            Err(failed("no model to be had"))
        });
        let local = Arc::new(Local::spawn("stand-in", load, alone()).unwrap());
        let embedder = Arc::new(giving(&local, ms(200)));
        // More writes ahead than their loads, at 300 ms each, could fail
        // before the backstop: were a load that fails the model at work,
        // the write behind them would be kept for every one.
        let ahead: Vec<_> = (0..300)
            .map(|_| {
                let texts = (0..4)
                    .map(|n| format!("{n}{}", "x".repeat(READS)))
                    .collect();
                start(&local, texts, Purpose::Document)
            })
            .collect();
        tokio::time::sleep(ms(50)).await;
        let given_up_on = tokio::time::timeout(BACKSTOP, embedder.documents(&["owl".to_string()]))
            .await
            .expect("given up on before the loads ahead of it are done");
        let stopped = failed("the model has made nothing for anyone in 400ms");
        assert_eq!(given_up_on, Err(stopped));
        for write in ahead {
            write.abort();
        }
    }

    /// A load longer than a write is given to wait, that then fails: the
    /// write waits for it, and is given up on within that time of its
    /// failing, whenever that is. The load is held until the test fails
    /// it, and the waits are timed by a clock the test moves itself, only
    /// once the write is asleep on it: so the write is given up on when
    /// it next looks, exactly a stall after the load failed.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_waits_for_a_long_load_and_is_given_up_on_soon_after_it_fails() {
        use crate::clock::{Clock, Manual};
        let ms = Duration::from_millis;
        for load_takes in [ms(700), ms(2_000)] {
            let clock = Manual::new();
            let (fail, failing) = channel::<()>();
            let failing = std::sync::Mutex::new(failing);
            let load: Load = Box::new(move || {
                failing.lock().unwrap().recv().unwrap();
                Err(failed("no model to be had"))
            });
            let local = Local::spawn_timed("stand-in", load, alone(), clock.clone());
            let local = Arc::new(local.unwrap());
            let embedder = Arc::new(giving(&local, ms(200)));
            let began = clock.now();
            let first = start(&local, vec!["first".into()], Purpose::Document);
            until(|| local.activity.loading_until().is_some()).await;
            let behind = {
                let embedder = embedder.clone();
                tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
            };
            asleep_on(&clock, 1).await;
            // The load takes so long by the clock: the write waits
            // through it, looking as it goes.
            clock.move_to(began + load_takes);
            asleep_on(&clock, 1).await;
            fail.send(()).unwrap();
            assert!(first.await.unwrap().is_err());
            let next = asleep_on(&clock, 1).await;
            clock.move_to(next[0]);
            let stopped = failed("the model has made nothing for anyone in 400ms");
            let given_up_on = tokio::time::timeout(BACKSTOP, behind).await;
            assert_eq!(given_up_on.expect("given up on").unwrap(), Err(stopped));
            // Not before the load failed, and a stall after it.
            assert_eq!(clock.now() - began, load_takes + ms(400));
        }
    }

    /// Runs that panic, each followed by a reload: the reloads are not
    /// held against a write waiting, the runs are, and it is given up on.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_is_given_up_on_behind_runs_that_keep_panicking() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = Arc::new(giving(&local, ms(200)));
        let panicking: Vec<_> = (0..1_000)
            .map(|_| {
                let mut texts = vec!["panic".to_string()];
                texts.extend((0..3).map(|n| format!("{n}{}", "x".repeat(READS))));
                start(&local, texts, Purpose::Document)
            })
            .collect();
        assert_eq!(runs.next().len(), 4);
        let began = std::time::Instant::now();
        let behind = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        // A run panics every hundred milliseconds, until the write
        // behind them is given up on: were the reloads after them held
        // against it as the runs are not, it would wait through all.
        let mut panicked = 0;
        while !behind.is_finished() {
            assert!(
                began.elapsed() < BACKSTOP,
                "not given up on through {panicked} runs that panicked"
            );
            tokio::time::sleep(ms(100)).await;
            runs.let_through(1);
            panicked += 1;
        }
        let given_up_on = behind.await.unwrap();
        let stopped = failed("the model has made nothing for anyone in 400ms");
        assert_eq!(given_up_on, Err(stopped));
        assert!(began.elapsed() >= ms(400), "{:?}", began.elapsed());
        for write in panicking {
            write.abort();
        }
    }

    /// A model that has loaded was at work until then: a write that
    /// waited through a long load is given, from the load's end, as long
    /// as a model may go without making anything, for the first run to
    /// come back in.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_has_loaded_was_at_work_until_then() {
        let ms = Duration::from_millis;
        // A load of 1.4 s and a run of half a second, with a write given
        // up on after 800 ms of nothing: looked at 800 ms in, the model
        // is loading; at 1.6 s, it has loaded 200 ms before and made
        // nothing yet, which is no stall; and the run is back at 1.9 s.
        let local = slow_to_load(ms(1_400), ms(500));
        let embedder = Arc::new(giving(&local, ms(400)));
        let began = std::time::Instant::now();
        let written = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        // While it loads it is given ten minutes to, and is taken to be
        // loading for no longer.
        tokio::time::sleep(ms(300)).await;
        let now = tokio::time::Instant::now();
        let until = local.activity.loading_until().expect("it is loading");
        let given = until.duration_since(now);
        let ten_minutes = Duration::from_secs(600);
        assert!(
            given <= ten_minutes && given > ten_minutes - ms(2_000),
            "{given:?}"
        );
        assert_eq!(written.await.unwrap(), Ok(vec![Some(vec![3.0])]));
        assert!(began.elapsed() >= ms(1_900), "{:?}", began.elapsed());
    }

    /// A load that fails is over: the model is not taken to be loading
    /// still, for the ten minutes a load is given.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_load_that_fails_is_over() {
        let (local, _runs) = local(1);
        let asked = local.embed(vec!["fox".into()], Purpose::Document).await;
        assert_eq!(asked, Err(failed("no model to be had")));
        assert_eq!(local.activity.loading_until(), None);
    }

    /// A write waits its turn for as long as the model is at work for
    /// anyone: behind another's long write, with nothing of its own made
    /// for far longer than a model may go without making anything, it is
    /// made. And when the model stops, it is given up on. The model's
    /// runs are let through one by one, and the waits are timed by a
    /// clock the test moves itself, only once the write is asleep on it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_write_waits_its_turn_while_the_model_works_for_others_and_no_longer() {
        use crate::clock::{Clock, Manual};
        let ms = Duration::from_millis;
        let clock = Manual::new();
        let (local, runs) = local_timed("stand-in", 0, alone(), clock.clone());
        // Given up on after 400 ms of nothing for anyone.
        let embedder = Arc::new(giving(&local, ms(200)));
        let began = clock.now();
        let ahead = start(&local, texts("a", 5 * RUN), Purpose::Document);
        assert_eq!(runs.next().len(), RUN);
        let behind = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.documents(&["owl".to_string()]).await })
        };
        asleep_on(&clock, 1).await;
        // Five runs of the write ahead, 250 ms each by the clock: a
        // second and a quarter in which nothing is made for the one
        // behind. Each run is made before the next is begun.
        for run in 0..5 {
            clock.move_to(clock.now() + ms(250));
            runs.let_through(1);
            if run < 4 {
                assert_eq!(runs.next().len(), RUN);
            }
            asleep_on(&clock, 1).await;
        }
        assert_eq!(ahead.await.unwrap().unwrap().len(), 5 * RUN);
        assert_eq!(runs.next(), ["owl"]);
        runs.let_through(1);
        let made = tokio::time::timeout(BACKSTOP, behind).await.expect("made");
        assert_eq!(made.unwrap(), Ok(vec![Some(vec![3.0])]));
        assert_eq!(clock.now() - began, ms(1_250));

        // The model stops, at a run that never comes back: the write in
        // it and the write behind it are both given up on, 400 ms after
        // the model last made anything, which was just now.
        let stopped = failed("the model has made nothing for anyone in 400ms");
        let stopping = clock.now();
        let writes: Vec<_> = ["hen", "fox"]
            .into_iter()
            .map(|text| {
                let embedder = embedder.clone();
                tokio::spawn(async move { embedder.documents(&[text.to_string()]).await })
            })
            .collect();
        assert!(!runs.next().is_empty());
        let asleep = asleep_on(&clock, 2).await;
        clock.move_to(asleep[1]);
        for write in writes {
            let given_up_on = tokio::time::timeout(BACKSTOP, write).await;
            assert_eq!(
                given_up_on.expect("given up on").unwrap(),
                Err(stopped.clone())
            );
        }
        assert_eq!(clock.now() - stopping, ms(400));
    }

    /// A model that is loading is at work: six hundred writes made at
    /// once while it loads, for three times as long as a model may go
    /// without making anything, are all made. More wait than the model
    /// takes in at once, and those are waited for like the rest.
    #[tokio::test(flavor = "multi_thread")]
    async fn writes_made_while_a_model_loads_wait_for_it_however_many() {
        let ms = Duration::from_millis;
        let local = slow_to_load(ms(1_200), ms(0));
        let embedder = Arc::new(giving(&local, ms(200)));
        let writes = (0..600).map(|n| {
            let embedder = embedder.clone();
            async move { embedder.documents(&[format!("text {n}")]).await }
        });
        let written = futures::future::join_all(writes);
        let written = tokio::time::timeout(Duration::from_secs(30), written).await;
        let written = written.expect("every write is answered");
        let made = written.iter().filter(|made| made.is_ok()).count();
        assert_eq!(made, 600, "{:?}", written.iter().find(|made| made.is_err()));
    }

    /// A model that has stopped is given up on by everyone who waits on
    /// it, in the time a model may go without making anything: those in
    /// the run, those in the line, and those not yet taken into the
    /// line, of whom there are more than it takes in at once. The run
    /// never comes back, and the clock is paused: it moves only once the
    /// run is seen begun, and then only when nothing is left but to wait,
    /// so every write is given up on exactly that long after it was made.
    #[tokio::test(start_paused = true)]
    async fn a_model_that_stops_is_given_up_on_by_all_who_wait_however_many() {
        let ms = Duration::from_millis;
        let (local, runs) = local(0);
        let embedder = Arc::new(giving(&local, ms(200)));
        let began = tokio::time::Instant::now();
        let writes: Vec<_> = (0..600)
            .map(|n| {
                let embedder = embedder.clone();
                tokio::spawn(async move { embedder.documents(&[format!("text {n}")]).await })
            })
            .collect();
        // Seen on a thread of the blocking pool, which holds the paused
        // clock still until it is.
        let ran = tokio::task::spawn_blocking(move || (runs.next(), runs));
        let (ran, runs) = ran.await.unwrap();
        assert!(!ran.is_empty(), "the first run, which never comes back");
        assert_eq!(began.elapsed(), Duration::ZERO);
        let stopped = failed("the model has made nothing for anyone in 400ms");
        for write in writes {
            assert_eq!(write.await.unwrap(), Err(stopped.clone()));
        }
        // To the millisecond the timer keeps time in.
        let took = began.elapsed();
        assert!((ms(400)..=ms(401)).contains(&took), "{took:?}");
        drop(runs);
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
