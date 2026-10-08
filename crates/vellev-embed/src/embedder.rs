use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::OnceCell;
use tokio::time::Instant;
use vellev_core::{Embedding, ModelFault};

use crate::{EmbedError, Provider, Purpose, Result};

/// How long a fault that will not pass until the setup is put right is
/// taken to stand, unless an embedder is told otherwise, without the model
/// being asked again: a node set up wrong would otherwise ask whatever
/// serves the model, and write to its log, on every request. A setup put
/// right takes effect within it.
pub const FAULT_STANDS: Duration = Duration::from_secs(20);

/// The shortest and the longest a fault can be told to stand. With no
/// time at all it would be said on every call that met it, which is what
/// its standing is there to prevent; and one standing for days would keep
/// a node whose setup had been put right from finding that out.
const SHORTEST_STANDING: Duration = Duration::from_secs(1);
const LONGEST_STANDING: Duration = Duration::from_secs(60 * 60);

/// `asked`, as long as a fault can be told to stand: no less than a
/// second, and no more than an hour.
pub(crate) fn standing(asked: Duration) -> Duration {
    asked.clamp(SHORTEST_STANDING, LONGEST_STANDING)
}

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
    /// How long a lasting fault stands.
    fault_stands: Duration,
}

/// What is known of whether the model can be used. It is looked at and
/// changed in a moment, and no call ever waits on another for it.
#[derive(Debug, Default)]
struct Health {
    /// The lasting fault that stands, and since when.
    fault: Option<(Instant, EmbedError)>,
    /// How many times the model has answered well.
    answered: u64,
    /// When a fault was last said, for each place one can lie: a fault of
    /// one kind does not keep one of another from being said. Nothing but
    /// time passing lets one be said again: not the fault being lifted,
    /// which a model that answers some calls and refuses others has done
    /// to it over and over.
    said: HashMap<ModelFault, Instant>,
}

impl Embedder {
    pub fn new(provider: Arc<dyn Provider>, embedding: Embedding) -> Self {
        Self {
            provider,
            embedding,
            queries: Mutex::default(),
            health: Mutex::default(),
            fault_stands: FAULT_STANDS,
        }
    }

    /// This embedder, taking a lasting fault to stand for `fault_stands`
    /// rather than [`FAULT_STANDS`]: no less than a second, and no more
    /// than an hour, whatever is asked.
    #[must_use]
    pub fn standing_for(self, fault_stands: Duration) -> Self {
        Self {
            fault_stands: standing(fault_stands),
            ..self
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
        let before = self.standing()?;
        let made = self.made(texts, wanted, purpose).await;
        if let Some((fault, lies_in)) = self.heard(before, &made) {
            // Said here, with nothing held, and no more often than once
            // in the time a fault stands.
            tracing::error!(
                model = %self.embedding.model,
                ?lies_in,
                cause = %fault,
                "the embedding model cannot be used; this is said no more than once in {:?}",
                self.fault_stands
            );
        }
        made
    }

    /// How many times the model had answered well when this was asked, or
    /// the fault that stands. While one does, the model is not asked.
    ///
    /// Nothing else keeps a call from the model: no call waits on another
    /// to find out whether the model can be used. So the calls that are
    /// out when a fault first shows, or go out before the first of them
    /// hears of it, each ask the model once and are each refused. And a
    /// model that answers some calls and refuses others goes on being
    /// asked: each answer lifts the fault. What is kept to is that a
    /// model that refuses every call is not asked again while the fault
    /// stands: not that it is asked once.
    fn standing(&self) -> Result<u64> {
        let mut health = self.health.lock().expect("health lock");
        if health
            .fault
            .as_ref()
            .is_some_and(|(met, _)| met.elapsed() >= self.fault_stands)
        {
            health.fault = None;
        }
        match &health.fault {
            Some((_, fault)) => Err(fault.clone()),
            None => Ok(health.answered),
        }
    }

    /// Take note of how a call that went out when the model had answered
    /// well `before` times came back. The lasting fault it met, with
    /// where that lies, if it is to be said: which it is if none that lies
    /// there has been said in the time a fault stands, whether or not
    /// this one does. So a fault of each kind is said at most once in
    /// that time, whatever the model does meanwhile.
    ///
    /// The model answering lifts whatever fault stands: a call that had
    /// yet to come back when the fault was met shows it has passed. A
    /// fault met by a call the model has answered another well since is
    /// one put right meanwhile, and does not stand, however late the news
    /// of it comes. And one that stands already is not made to stand the
    /// longer.
    fn heard<T>(&self, before: u64, made: &Result<T>) -> Option<(EmbedError, ModelFault)> {
        let mut health = self.health.lock().expect("health lock");
        match made {
            Ok(_) => {
                health.answered += 1;
                health.fault = None;
                None
            }
            Err(error) => {
                let lies_in = error.lasting()?;
                let now = Instant::now();
                if health.answered == before && health.fault.is_none() {
                    health.fault = Some((now, error.clone()));
                }
                let unsaid = health
                    .said
                    .get(&lies_in)
                    .is_none_or(|said| now.duration_since(*said) >= self.fault_stands);
                unsaid.then(|| {
                    health.said.insert(lies_in, now);
                    (error.clone(), lies_in)
                })
            }
        }
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
        let malformed = |reason: String| EmbedError::malformed(&self.embedding.model, reason);
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
            return Err(EmbedError::unusable(
                &self.embedding.model,
                ModelFault::Provider,
                format!("a vector of {} dimensions, not {dimensions}", wrong.len()),
            ));
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
