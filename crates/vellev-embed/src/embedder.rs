use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{watch, OnceCell};
use tokio::time::Instant;
use vellev_core::{Embedding, ModelFault};

use crate::{EmbedError, Provider, Purpose, Result};

/// How long a fault that will not pass until the setup is put right is
/// taken to stand, without the model being asked again: a node set up
/// wrong would otherwise ask whatever serves the model, and write to its
/// log, on every request. A setup put right takes effect within it.
pub const FAULT_STANDS: Duration = Duration::from_secs(20);

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
    health: Mutex<Health>,
}

/// What is known of whether the model can be used.
///
/// While nothing is known, at first and again once a fault no longer
/// stands, one call at a time is let through to the model to find out, and
/// the others wait for it: were the model to refuse for good, each would
/// otherwise be refused in turn, and the fault said as many times. As soon
/// as a call comes back any other way, every call goes to the model as it
/// arrives, however many at once, as it always did.
#[derive(Debug, Default)]
struct Health {
    /// The lasting fault that stands, and since when.
    fault: Option<(Instant, EmbedError)>,
    /// Whether the model has been asked, since it last stood refused, and
    /// did not refuse for good.
    tried: bool,
    /// How many times the model has answered well.
    answered: u64,
    /// The call finding out whether the model can be used, as those that
    /// wait for it see it: closed once it is over.
    finding_out: Option<watch::Receiver<()>>,
}

/// What a call is to do, by what is known of the model.
enum Turn {
    /// Ask the model. With a sender, this call is the one finding out, and
    /// lets the others go by dropping it.
    Ask(Option<watch::Sender<()>>),
    /// Wait for the call that is finding out.
    Wait(watch::Receiver<()>),
    Refused(EmbedError),
}

impl Health {
    fn turn(&mut self) -> Turn {
        if self
            .fault
            .as_ref()
            .is_some_and(|(met, _)| met.elapsed() >= FAULT_STANDS)
        {
            *self = Health {
                answered: self.answered,
                ..Health::default()
            };
        }
        if let Some((_, fault)) = &self.fault {
            return Turn::Refused(fault.clone());
        }
        // A call that was finding out and never said, its caller gone.
        if self
            .finding_out
            .as_ref()
            .is_some_and(|finding| finding.has_changed().is_err())
        {
            self.finding_out = None;
        }
        match (self.tried, &self.finding_out) {
            (true, _) => Turn::Ask(None),
            (false, Some(finding)) => Turn::Wait(finding.clone()),
            (false, None) => {
                let (found, finding) = watch::channel(());
                self.finding_out = Some(finding);
                Turn::Ask(Some(found))
            }
        }
    }
}

impl Embedder {
    pub fn new(provider: Arc<dyn Provider>, embedding: Embedding) -> Self {
        Self {
            provider,
            embedding,
            queries: Mutex::default(),
            health: Mutex::default(),
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
        let (found, before) = loop {
            let (turn, answered) = {
                let mut health = self.health.lock().expect("health lock");
                (health.turn(), health.answered)
            };
            match turn {
                Turn::Ask(found) => break (found, answered),
                Turn::Refused(fault) => return Err(fault),
                // Over when the call finding out lets go, however it ended.
                Turn::Wait(mut finding) => drop(finding.changed().await),
            }
        };
        let made = self.made(texts, wanted, purpose).await;
        let mut health = self.health.lock().expect("health lock");
        match made.as_ref().map_err(|error| (error, error.lasting())) {
            // The model answers: whatever was said of it before, by a call
            // that had yet to come back, it can be used.
            Ok(_) => {
                health.answered += 1;
                health.tried = true;
                health.fault = None;
            }
            // A fault met by a call the model has since answered another
            // well is one put right meanwhile: it does not stand, however
            // late the news of it comes. And one that stands already is
            // not said again, nor made to stand the longer.
            Err((error, Some(lies_in))) => {
                if health.answered == before && health.fault.is_none() {
                    tracing::error!(
                        model = %self.embedding.model,
                        ?lies_in,
                        cause = %error,
                        "the embedding model cannot be used; not asking it again for {FAULT_STANDS:?}"
                    );
                    health.fault = Some((Instant::now(), error.clone()));
                    health.tried = false;
                }
            }
            Err((_, None)) => health.tried = true,
        }
        if found.is_some() {
            health.finding_out = None;
        }
        drop(health);
        drop(found);
        made
    }

    /// The vectors the model makes for `wanted`, the texts of `texts` that
    /// are not blank, checked and put back in their places.
    async fn made(
        &self,
        texts: &[String],
        wanted: Vec<String>,
        purpose: Purpose,
    ) -> Result<Vec<Option<Vec<f32>>>> {
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
                fault: ModelFault::Provider,
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
