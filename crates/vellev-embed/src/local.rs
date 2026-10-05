use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tokio::sync::oneshot;

use crate::{EmbedError, Provider, Result};

/// Texts in one inference call.
const INFERENCE_BATCH: usize = 32;
/// Texts taken from waiting callers before the model is run on them.
const GATHERED: usize = 256;

struct Job {
    texts: Vec<String>,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
}

/// A model run in this process on a thread of its own, so that inference
/// never holds up the async runtime. Callers that arrive while the model is
/// busy are embedded together in its next run. The model is fetched into
/// the cache directory and loaded when first needed.
#[derive(Debug)]
pub struct Local {
    model: String,
    jobs: Sender<Job>,
}

impl Local {
    /// `model` is a Hugging Face name such as `BAAI/bge-base-en-v1.5`.
    pub fn new(model: &str, dimensions: u32, cache: &Path) -> Result<Self> {
        let known = known(model, dimensions)?;
        let (jobs, waiting) = channel();
        let (name, cache) = (model.to_string(), cache.to_path_buf());
        std::thread::Builder::new()
            .name(format!("embed {model}"))
            .spawn(move || run(&name, known, cache, waiting))
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

fn run(name: &str, model: EmbeddingModel, cache: PathBuf, waiting: Receiver<Job>) {
    let failed = |error: fastembed::Error| EmbedError::Failed {
        model: name.to_string(),
        reason: error.to_string(),
    };
    let mut loaded: Option<TextEmbedding> = None;
    while let Ok(first) = waiting.recv() {
        let mut jobs = vec![first];
        let mut gathered = jobs[0].texts.len();
        while gathered < GATHERED {
            let Ok(job) = waiting.try_recv() else { break };
            gathered += job.texts.len();
            jobs.push(job);
        }
        let texts: Vec<&str> = jobs
            .iter()
            .flat_map(|job| job.texts.iter().map(String::as_str))
            .collect();
        let session = match &mut loaded {
            Some(session) => Ok(session),
            None => {
                let options = InitOptions::new(model.clone())
                    .with_cache_dir(cache.clone())
                    .with_show_download_progress(false);
                TextEmbedding::try_new(options).map(|session| loaded.insert(session))
            }
        };
        let embedded = session
            .and_then(|session| session.embed(texts, Some(INFERENCE_BATCH)))
            .map_err(failed);
        let mut vectors = embedded.map(Vec::into_iter);
        for job in jobs {
            let own = vectors
                .as_mut()
                .map(|vectors| vectors.take(job.texts.len()).collect())
                .map_err(|error| error.clone());
            let _ = job.reply.send(own);
        }
    }
}

#[async_trait]
impl Provider for Local {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let stopped = || EmbedError::Failed {
            model: self.model.clone(),
            reason: "the model's thread has stopped".into(),
        };
        let (reply, answer) = oneshot::channel();
        self.jobs
            .send(Job { texts, reply })
            .map_err(|_| stopped())?;
        answer.await.map_err(|_| stopped())?
    }
}
