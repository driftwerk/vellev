use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use vellev_core::{Embedding, EmbeddingProvider};

use crate::{Bedrock, Embedder, Http, Provider, Result};

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
}

impl Embedders {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            providers: Mutex::default(),
            embedders: Mutex::default(),
        }
    }

    /// The embedder for a collection that embeds as `embedding` says, or
    /// `None` if its clients supply every vector.
    pub fn embedder(&self, embedding: &Embedding) -> Result<Option<Arc<Embedder>>> {
        let mut embedders = self.embedders.lock().expect("embedders lock");
        if let Some(embedder) = embedders.get(embedding) {
            return Ok(Some(embedder.clone()));
        }
        Ok(self.provider(embedding)?.map(|provider| {
            let embedder = Arc::new(Embedder::new(provider, embedding.clone()));
            embedders.insert(embedding.clone(), embedder.clone());
            embedder
        }))
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
        Err(crate::EmbedError::Unsupported(format!(
            "this build cannot run {model} locally"
        )))
    }
}
