use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::time::Instant;
use vellev_core::{Embedding, EmbeddingProvider};

use crate::{Bedrock, Embedder, Http, Provider, Result, FAULT_STANDS};

/// What embedding needs from the node rather than from a collection.
#[derive(Clone, Default)]
pub struct Settings {
    /// Where local models are kept once fetched.
    pub cache: PathBuf,
    /// The bearer token for each HTTP endpoint that wants one, by its URL
    /// as collections give it. No other endpoint is sent a token.
    pub keys: HashMap<String, String>,
}

impl std::fmt::Debug for Settings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Settings")
            .field("cache", &self.cache)
            .field("keys", &self.keys.keys())
            .finish()
    }
}

/// A model as a provider runs it: where, which, and at what size.
type Model = (EmbeddingProvider, String, u32);

/// The node's embedders, one per way of embedding. Collections that embed
/// alike share one, and with it one loaded model and one query cache.
#[derive(Debug)]
pub struct Embedders {
    settings: Settings,
    providers: Mutex<HashMap<Model, Arc<dyn Provider>>>,
    embedders: Mutex<HashMap<Embedding, Arc<Embedder>>>,
    /// The ways of embedding lately found to be ones this build cannot
    /// run, so that each is logged once while it stands.
    refused: Mutex<Told>,
}

/// What has been said lately, and when: each thing is to be said once in
/// any [`FAULT_STANDS`], however often it comes up.
#[derive(Debug, Default)]
struct Told(HashMap<Embedding, Instant>);

impl Told {
    /// Whether `embedding` was not spoken of within the time, in which
    /// case it now is: once in each such time, however often it comes up
    /// meanwhile. What was spoken of longer ago is forgotten.
    fn first(&mut self, embedding: &Embedding, now: Instant) -> bool {
        self.0
            .retain(|_, told| now.duration_since(*told) < FAULT_STANDS);
        match self.0.contains_key(embedding) {
            true => false,
            false => {
                self.0.insert(embedding.clone(), now);
                true
            }
        }
    }
}

impl Embedders {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            providers: Mutex::default(),
            embedders: Mutex::default(),
            refused: Mutex::default(),
        }
    }

    /// The embedder for a collection that embeds as `embedding` says, or
    /// `None` if its clients supply every vector.
    pub fn embedder(&self, embedding: &Embedding) -> Result<Option<Arc<Embedder>>> {
        let mut embedders = self.embedders.lock().expect("embedders lock");
        if let Some(embedder) = embedders.get(embedding) {
            return Ok(Some(embedder.clone()));
        }
        let provider = self.provider(embedding).inspect_err(|error| {
            // Only a collection there already comes to this: one made by
            // a release or a build that could run its model.
            let mut refused = self.refused.lock().expect("refused lock");
            if error.lasting().is_some() && refused.first(embedding, Instant::now()) {
                tracing::error!(
                    model = %embedding.model,
                    cause = %error,
                    "a collection's embedding model cannot be run by this build"
                );
            }
        })?;
        Ok(provider.map(|provider| {
            let embedder = Arc::new(Embedder::new(provider, embedding.clone()));
            embedders.insert(embedding.clone(), embedder.clone());
            embedder
        }))
    }

    /// Whether a collection that embeds as `embedding` says could be
    /// embedded for here, as far as can be told without using the model:
    /// that its model is one of this release's, at that size, and that
    /// this build can run it. Nothing is built, started or kept for the
    /// asking, so that asking costs nothing however often and of however
    /// many models it is done.
    pub fn admits(&self, embedding: &Embedding) -> Result<()> {
        let Embedding {
            provider,
            model,
            dimensions,
            ..
        } = embedding;
        match provider {
            EmbeddingProvider::None | EmbeddingProvider::Http { .. } => Ok(()),
            EmbeddingProvider::Local => Self::runs_locally(model, *dimensions),
            EmbeddingProvider::Bedrock { .. } => Bedrock::admits(model, *dimensions),
        }
    }

    #[cfg(feature = "local")]
    fn runs_locally(model: &str, dimensions: u32) -> Result<()> {
        crate::local::known(model, dimensions).map(drop)
    }

    #[cfg(not(feature = "local"))]
    fn runs_locally(model: &str, _dimensions: u32) -> Result<()> {
        Err(Self::without_local(model))
    }

    /// What a build without local models says of one.
    #[cfg_attr(feature = "local", cfg(test))]
    fn without_local(model: &str) -> crate::EmbedError {
        crate::EmbedError::Unsupported(format!("this build cannot run {model} locally"))
    }

    /// How many embedders are held.
    pub fn held(&self) -> usize {
        self.embedders.lock().expect("embedders lock").len()
    }

    fn provider(&self, embedding: &Embedding) -> Result<Option<Arc<dyn Provider>>> {
        let Embedding {
            provider,
            model,
            dimensions,
            ..
        } = embedding;
        let key = (provider.clone(), model.clone(), *dimensions);
        let mut providers = self.providers.lock().expect("providers lock");
        if let Some(provider) = providers.get(&key) {
            return Ok(Some(provider.clone()));
        }
        let built: Arc<dyn Provider> = match provider {
            EmbeddingProvider::None => return Ok(None),
            EmbeddingProvider::Local => self.local(model, *dimensions)?,
            EmbeddingProvider::Bedrock { region } => {
                Arc::new(Bedrock::new(region, model, *dimensions)?)
            }
            EmbeddingProvider::Http { url, api } => {
                let token = self.settings.keys.get(url.trim_end_matches('/')).cloned();
                Arc::new(Http::new(url, *api, model, token)?)
            }
        };
        providers.insert(key, built.clone());
        Ok(Some(built))
    }

    #[cfg(feature = "local")]
    fn local(&self, model: &str, dimensions: u32) -> Result<Arc<dyn Provider>> {
        Ok(Arc::new(crate::Local::new(
            model,
            dimensions,
            &self.settings.cache,
        )?))
    }

    #[cfg(not(feature = "local"))]
    fn local(&self, model: &str, _dimensions: u32) -> Result<Arc<dyn Provider>> {
        Err(Self::without_local(model))
    }
}

#[cfg(test)]
mod tests {
    use vellev_core::ModelFault;

    use super::*;

    #[tokio::test(start_paused = true)]
    async fn a_thing_is_said_once_while_it_stands_and_then_forgotten() {
        let embedding = |model: &str| Embedding {
            provider: EmbeddingProvider::Local,
            model: model.into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        };
        let mut told = Told::default();
        assert!(told.first(&embedding("one"), Instant::now()));
        assert!(!told.first(&embedding("one"), Instant::now()));
        assert!(told.first(&embedding("two"), Instant::now()));
        // Coming up all the while, it is said once in each such time: it
        // is not put off for having come up.
        let step = crate::FAULT_STANDS / 4;
        let mut said = 1;
        for _ in 0..12 {
            tokio::time::advance(step).await;
            said += usize::from(told.first(&embedding("one"), Instant::now()));
        }
        assert_eq!(said, 1 + 3, "once at first, and once in each time after");
        assert_eq!(
            told.0.len(),
            1,
            "what was last spoken of long ago is forgotten"
        );
    }

    #[test]
    fn a_build_without_local_models_is_at_fault_not_whoever_asks_for_one() {
        let said = Embedders::without_local("BAAI/bge-base-en-v1.5");
        assert_eq!(
            said,
            crate::EmbedError::Unsupported(
                "this build cannot run BAAI/bge-base-en-v1.5 locally".into()
            )
        );
        assert_eq!(said.lasting(), Some(ModelFault::Build));
    }
}
