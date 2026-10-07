use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use tokio::sync::OnceCell;
use vellev_core::Embedding;

use crate::{EmbedError, Provider, Purpose, Result};

/// How many query vectors an embedder keeps: one request often searches
/// several times with the same text.
const QUERY_CACHE: usize = 1_024;
/// Queries longer than this, in bytes, are not kept.
const LONGEST_KEPT: usize = 1_024;

type Kept = Arc<OnceCell<Option<Vec<f32>>>>;

/// The vectors of the queries last first asked, each filled in by whoever
/// asks first while the others wait for it.
#[derive(Debug, Default)]
struct Queries {
    vectors: HashMap<Arc<str>, Kept>,
    order: VecDeque<Arc<str>>,
}

impl Queries {
    fn kept(&mut self, text: &str) -> Kept {
        if let Some(kept) = self.vectors.get(text) {
            return kept.clone();
        }
        let text: Arc<str> = text.into();
        let kept = Kept::default();
        self.vectors.insert(text.clone(), kept.clone());
        self.order.push_back(text);
        if self.order.len() > QUERY_CACHE {
            self.order
                .pop_front()
                .and_then(|oldest| self.vectors.remove(&oldest));
        }
        kept
    }
}

/// One collection's way of embedding: its model, with the prefixes the
/// model was trained with. Blank text has no vector, and is never sent to a
/// model.
#[derive(Debug)]
pub struct Embedder {
    provider: Arc<dyn Provider>,
    embedding: Embedding,
    queries: Mutex<Queries>,
}

impl Embedder {
    pub fn new(provider: Arc<dyn Provider>, embedding: Embedding) -> Self {
        Self {
            provider,
            embedding,
            queries: Mutex::default(),
        }
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// A vector for each text to be stored, in order.
    pub async fn documents(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        let prefix = self.embedding.document_prefix.as_deref();
        self.embed(texts, prefix, Purpose::Document).await
    }

    /// The vector to search with for `text`. Asked for again, or by several
    /// at once, it is worked out once.
    pub async fn query(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let embed = || async {
            let prefix = self.embedding.query_prefix.as_deref();
            let texts = [text.to_string()];
            let vectors = self.embed(&texts, prefix, Purpose::Query).await?;
            Ok(vectors.into_iter().next().flatten())
        };
        if text.len() > LONGEST_KEPT {
            return embed().await;
        }
        let kept = self.queries.lock().expect("query cache lock").kept(text);
        kept.get_or_try_init(embed).await.cloned()
    }

    async fn embed(
        &self,
        texts: &[String],
        prefix: Option<&str>,
        purpose: Purpose,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let wanted: Vec<String> = texts
            .iter()
            .filter(|text| !text.trim().is_empty())
            .map(|text| format!("{}{text}", prefix.unwrap_or_default()))
            .collect();
        if wanted.is_empty() {
            return Ok(vec![None; texts.len()]);
        }
        let asked = wanted.len();
        let vectors = self.provider.embed(wanted, purpose).await?;
        let malformed = |reason: String| EmbedError::Malformed {
            model: self.embedding.model.clone(),
            reason,
        };
        if vectors.len() != asked {
            return Err(malformed(format!(
                "{} vectors for {asked} texts",
                vectors.len()
            )));
        }
        // Vectors of another size mean the model is not the one the
        // collection was made for, however often it is asked.
        let dimensions = self.embedding.dimensions as usize;
        if let Some(wrong) = vectors.iter().find(|vector| vector.len() != dimensions) {
            return Err(EmbedError::Unusable {
                model: self.embedding.model.clone(),
                reason: format!("a vector of {} dimensions, not {dimensions}", wrong.len()),
            });
        }
        if !vectors.iter().flatten().all(|value| value.is_finite()) {
            return Err(malformed("a vector that is not finite".into()));
        }
        let mut vectors = vectors.into_iter();
        Ok(texts
            .iter()
            .map(|text| (!text.trim().is_empty()).then(|| vectors.next()).flatten())
            .collect())
    }
}
