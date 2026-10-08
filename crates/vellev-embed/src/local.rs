use std::collections::VecDeque;
use std::io::ErrorKind;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hf_hub::api::sync::ApiError as FetchError;
use tokio::sync::{mpsc, oneshot};

use crate::{EmbedError, Provider, Purpose, Result};

/// Texts in one run of the model. A run cannot be interrupted, so this is
/// also how long a query can be kept waiting by documents, and it bounds
/// the memory a run takes.
const RUN: usize = 32;
/// Callers that can be waiting for the model before others wait to join
/// them.
const WAITING: usize = 256;

/// A loaded model: a vector for each text, or why not.
type Model = Box<dyn FnMut(Vec<&str>) -> std::result::Result<Vec<Vec<f32>>, String>>;
type Load = Box<dyn FnMut() -> Result<Model> + Send>;

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
/// models are [`kept_in`] and loaded when first needed, and again after a
/// failure.
#[derive(Debug)]
pub struct Local {
    model: String,
    jobs: mpsc::Sender<Job>,
}

impl Local {
    /// `model` is a Hugging Face name such as `BAAI/bge-base-en-v1.5`.
    pub fn new(model: &str, dimensions: u32, cache: &Path) -> Result<Self> {
        let known = known(model, dimensions)?;
        let cache = cache.to_path_buf();
        let kept = kept_in(&cache, std::env::var("HF_HOME").ok());
        let threads = std::thread::available_parallelism().map_or(1, |cores| cores.get());
        let name = model.to_string();
        Self::spawn(
            model,
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
        )
    }

    fn spawn(model: &str, load: Load) -> Result<Self> {
        let (jobs, waiting) = mpsc::channel(WAITING);
        let name = model.to_string();
        std::thread::Builder::new()
            .name(format!("embed {model}"))
            .spawn(move || run(&name, load, waiting))
            .map_err(|error| EmbedError::Failed {
                model: model.to_string(),
                reason: format!("no thread to run it on: {error}"),
            })?;
        Ok(Self {
            model: model.to_string(),
            jobs,
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
        true => EmbedError::Unusable {
            model: model.to_string(),
            reason: format!("the node may not write {}: {error:#}", kept.display()),
        },
        false => failure(model, format!("{error:#}")),
    }
}

/// A failure of `model` that may pass.
fn failure(model: &str, reason: impl Into<String>) -> EmbedError {
    EmbedError::Failed {
        model: model.to_string(),
        reason: reason.into(),
    }
}

/// The model of this name, whichever organisation's copy of it is run.
fn known(model: &str, dimensions: u32) -> Result<EmbeddingModel> {
    let name = |code: &str| code.rsplit('/').next().unwrap_or(code).to_lowercase();
    let found = TextEmbedding::list_supported_models()
        .into_iter()
        .find(|info| name(&info.model_code) == name(model))
        .ok_or_else(|| EmbedError::Unsupported(format!("{model} cannot be run locally")))?;
    if found.dim != dimensions as usize {
        return Err(EmbedError::Unsupported(format!(
            "{model} has {} dimensions, not {dimensions}",
            found.dim
        )));
    }
    Ok(found.model)
}

fn run(name: &str, mut load: Load, mut waiting: mpsc::Receiver<Job>) {
    let mut model: Option<Model> = None;
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
            match waiting.blocking_recv() {
                Some(job) => queue(job, &mut queries, &mut documents),
                None => return,
            }
        }
        while let Ok(job) = waiting.try_recv() {
            queue(job, &mut queries, &mut documents);
        }
        for queue in [&mut queries, &mut documents] {
            queue.retain(|work| !work.job.reply.is_closed());
        }
        let next = if queries.is_empty() {
            &mut documents
        } else {
            &mut queries
        };
        advance(name, &mut load, &mut model, next);
    }
}

/// Run the model once, on the texts at the front of `queue`, and answer
/// the callers it finishes or fails.
fn advance(name: &str, load: &mut Load, model: &mut Option<Model>, queue: &mut VecDeque<Work>) {
    let mut room = RUN;
    let taken: Vec<usize> = queue
        .iter()
        .map_while(|work| {
            let take = room.min(work.job.texts.len() - work.done.len());
            room -= take;
            (take > 0).then_some(take)
        })
        .collect();
    let texts: Vec<&str> = queue
        .iter()
        .zip(&taken)
        .flat_map(|(work, take)| &work.job.texts[work.done.len()..][..*take])
        .map(String::as_str)
        .collect();
    let asked = texts.len();
    let ran = catch_unwind(AssertUnwindSafe(|| {
        let model = match model.as_mut() {
            Some(model) => model,
            None => model.insert(load()?),
        };
        model(texts).map_err(|reason| failure(name, reason))
    }));
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

#[async_trait]
impl Provider for Local {
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let stopped = || failure(&self.model, "the model's thread has stopped");
        let (reply, answer) = oneshot::channel();
        let job = Job {
            texts,
            purpose,
            reply,
        };
        self.jobs.send(job).await.map_err(|_| stopped())?;
        answer.await.map_err(|_| stopped())?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

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
                Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
            }) as Model)
        });
        (
            Arc::new(Local::spawn("stand-in", load).unwrap()),
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

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_fails_to_load_fails_or_panics_is_tried_afresh() {
        let (local, runs) = local(1);
        runs.let_through(4);
        let embed = |text: &str| local.embed(vec![text.to_string()], Purpose::Document);
        assert_eq!(embed("fox").await, Err(failed("no model to be had")));
        assert_eq!(embed("fox").await, Ok(vec![vec![3.0]]));
        assert_eq!(embed("fail").await, Err(failed("told to")));
        assert_eq!(runs.loads.load(std::sync::atomic::Ordering::SeqCst), 2);
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

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_that_can_never_load_says_so_each_time_it_is_tried() {
        let tries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tried = tries.clone();
        let barred = || EmbedError::Unusable {
            model: "stand-in".into(),
            reason: "its cache may not be written".into(),
        };
        let load: Load = Box::new(move || {
            tried.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(barred())
        });
        let local = Local::spawn("stand-in", load).unwrap();
        for _ in 0..2 {
            assert_eq!(
                local.embed(vec!["fox".into()], Purpose::Query).await,
                Err(barred())
            );
        }
        assert_eq!(tries.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn a_directory_the_node_may_not_write_keeps_a_model_from_loading_and_a_network_does_not() {
        // As fastembed reports a fetch that failed: what failed, under
        // what it was doing.
        let fetching = |cause: FetchError| {
            fastembed::Error::new(cause).context("Failed to retrieve model.onnx")
        };
        let kept = Path::new("/models");
        for kind in [ErrorKind::PermissionDenied, ErrorKind::ReadOnlyFilesystem] {
            let error = fetching(FetchError::IoError(kind.into()));
            assert_eq!(
                unloaded("m", kept, &error),
                EmbedError::Unusable {
                    model: "m".into(),
                    reason: format!("the node may not write /models: {error:#}")
                },
                "{kind:?}"
            );
        }
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
            EmbedError::Unsupported("BAAI/bge-base-en-v1.5 has 768 dimensions, not 1024".into())
        );
        assert_eq!(
            known("acme/unheard-of", 768).unwrap_err(),
            EmbedError::Unsupported("acme/unheard-of cannot be run locally".into())
        );
    }
}
