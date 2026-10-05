use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use vellev_core::{Embedding, Object, Value};

use crate::{EmbedError, Provider, Result};

/// How many recent query vectors an embedder keeps: one request often
/// searches several times with the same text.
const QUERY_CACHE: usize = 1_024;

/// The text of `object` that its collection embeds: the listed properties'
/// values, in order, a line each.
pub fn document(embedding: &Embedding, object: &Object) -> String {
    embedding
        .properties
        .iter()
        .filter_map(|name| object.properties.get(name))
        .flat_map(|value| match value {
            Value::Text(text) => std::slice::from_ref(text),
            Value::TextArray(texts) => texts.as_slice(),
            _ => &[],
        })
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Default)]
struct Recent {
    vectors: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
}

impl Recent {
    fn insert(&mut self, text: &str, vector: &[f32]) {
        if self
            .vectors
            .insert(text.to_string(), vector.to_vec())
            .is_none()
        {
            self.order.push_back(text.to_string());
            if self.order.len() > QUERY_CACHE {
                self.order
                    .pop_front()
                    .and_then(|oldest| self.vectors.remove(&oldest));
            }
        }
    }
}

/// One collection's way of embedding: its model, with the prefixes the
/// model was trained with. Blank text has no vector, and is never sent to a
/// model.
#[derive(Debug)]
pub struct Embedder {
    provider: Arc<dyn Provider>,
    embedding: Embedding,
    queries: Mutex<Recent>,
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
        self.embed(texts, self.embedding.document_prefix.as_deref())
            .await
    }

    /// The vector to search with for `text`.
    pub async fn query(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let cached = self.recent().vectors.get(text).cloned();
        if cached.is_some() {
            return Ok(cached);
        }
        let prefix = self.embedding.query_prefix.as_deref();
        let vector = self
            .embed(&[text.to_string()], prefix)
            .await?
            .pop()
            .flatten();
        if let Some(vector) = &vector {
            self.recent().insert(text, vector);
        }
        Ok(vector)
    }

    fn recent(&self) -> std::sync::MutexGuard<'_, Recent> {
        self.queries.lock().expect("query cache lock")
    }

    async fn embed(&self, texts: &[String], prefix: Option<&str>) -> Result<Vec<Option<Vec<f32>>>> {
        let wanted: Vec<String> = texts
            .iter()
            .filter(|text| !text.trim().is_empty())
            .map(|text| format!("{}{text}", prefix.unwrap_or_default()))
            .collect();
        if wanted.is_empty() {
            return Ok(vec![None; texts.len()]);
        }
        let asked = wanted.len();
        let vectors = self.provider.embed(wanted).await?;
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
        let dimensions = self.embedding.dimensions as usize;
        if let Some(wrong) = vectors.iter().find(|vector| vector.len() != dimensions) {
            return Err(malformed(format!(
                "a vector of {} dimensions, not {dimensions}",
                wrong.len()
            )));
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
