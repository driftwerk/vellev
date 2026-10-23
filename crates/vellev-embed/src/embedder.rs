use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::OnceCell;
use tokio::time::Instant;

use crate::clock::Clock;
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

/// How long a model is given to answer a query, unless an embedder is
/// told otherwise: someone is waiting on it, and a search that cannot be
/// embedded for in this long is better told so, to try again or to rank
/// by the words alone, than kept.
pub const QUERY_WITHIN: Duration = Duration::from_secs(10);
/// The longest a model may be given to answer a query, whatever a node
/// is told: what [`QUERY_WITHIN`] was before it was ten seconds. A search
/// kept longer than this holds a place among the reads a node answers at
/// once for longer than any client waits.
pub const LONGEST_QUERY_WITHIN: Duration = Duration::from_secs(120);
/// How long a model may go without making a vector for anyone, while it
/// is not loading, before a call for documents that waits on it is given
/// up on, unless an embedder is told otherwise. A model that many wait on
/// at once says what it is doing, and a call waits its turn for as long
/// as the model is at work, for whomever. Of one that does not say, a
/// call is given this long in all.
pub const DOCUMENTS_WITHIN: Duration = Duration::from_secs(60);

/// The longest any call is given, whatever it is told and however many
/// its texts: a time that can be added to the present one.
const LONGEST_WAIT: Duration = Duration::from_secs(24 * 60 * 60);

/// How many query vectors are kept for a model: one request often
/// searches several times with the same text.
const QUERY_CACHE: usize = 1_024;
/// Queries longer than this, in bytes, are not kept.
const LONGEST_KEPT: usize = 1_024;

type Kept = Arc<OnceCell<Result<Option<Vec<f32>>>>>;

/// The vectors of the queries last first asked of a model, by the text
/// the model was sent, prefix and all, each filled in by whoever asks
/// first while the others wait for it. Those waiting are told what
/// the first was, a failure as much as a vector: but a failure is not
/// kept, and whoever asks after it asks the model.
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

    /// Have whoever next asks for `text` ask the model, if `failed` is
    /// what is kept for it: those who wait on it already are told how it
    /// failed, and no one else is.
    fn forget(&mut self, text: &str, failed: &Kept) {
        if let Some(kept) = self.vectors.get_mut(text) {
            if Arc::ptr_eq(kept, failed) {
                *kept = Kept::default();
            }
        }
    }
}

/// How long a call is given, and when that is up.
#[derive(Debug, Clone, Copy)]
struct Within {
    given: Duration,
    by: Instant,
}

impl Within {
    /// `given` from now on `clock`, or a day if that is less.
    fn from_now(given: Duration, clock: &dyn Clock) -> Self {
        let given = given.min(LONGEST_WAIT);
        Self {
            given,
            by: clock.now() + given,
        }
    }
}

/// One collection's way of embedding: its model, with the prefixes the
/// model was trained with. Blank text has no vector, and is never sent to a
/// model.
#[derive(Debug)]
pub struct Embedder {
    shared: Arc<Shared>,
    embedding: Embedding,
    /// How long a lasting fault stands.
    fault_stands: Duration,
    /// How long the model is given to answer a query; and how long it
    /// may go without making anything for anyone before a call for
    /// documents is given up on.
    query_within: Duration,
    documents_within: Duration,
    /// What calls are timed by: tokio's clock, unless a test hands in
    /// its own.
    clock: Arc<dyn Clock>,
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

/// What all who embed with one model share, whatever prefixes they put
/// before their texts and whichever properties they embed: the model,
/// the vectors it has made for queries, and what is known of whether it
/// can be used.
///
/// A fault that stands is the model's, as it is reached and with the key
/// it is sent: so one model that cannot be used is found so once, and
/// said to be so once, however many collections embed with it and
/// however each words its texts. What is shared with a model is shared
/// with no other: not with the same model at another size, or reached
/// elsewhere or with another key, which may well be one that can be
/// used.
#[derive(Debug)]
pub(crate) struct Shared {
    provider: Arc<dyn Provider>,
    queries: Mutex<Queries>,
    health: Mutex<Health>,
}

impl Shared {
    pub(crate) fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            queries: Mutex::default(),
            health: Mutex::default(),
        }
    }

    /// Whether nothing is known of the model that would have to be found
    /// out again, or said again, were this forgotten: no fault stands,
    /// and none was said within `stands`.
    pub(crate) fn quiet(&self, stands: Duration) -> bool {
        let health = self.health.lock().expect("health lock");
        let lately = |met: &Instant| met.elapsed() < stands;
        !health.fault.as_ref().is_some_and(|(met, _)| lately(met))
            && !health.said.values().any(lately)
    }
}

impl Embedder {
    /// An embedder that shares its model with no other.
    pub fn new(provider: Arc<dyn Provider>, embedding: Embedding) -> Self {
        Self::sharing(Arc::new(Shared::new(provider)), embedding)
    }

    /// An embedder as `embedding` says, of the model whose embedders
    /// share `shared`.
    pub(crate) fn sharing(shared: Arc<Shared>, embedding: Embedding) -> Self {
        Self {
            shared,
            embedding,
            fault_stands: FAULT_STANDS,
            query_within: QUERY_WITHIN,
            documents_within: DOCUMENTS_WITHIN,
            clock: crate::clock::tokio(),
        }
    }

    /// This embedder, timing its calls by `clock` rather than tokio's:
    /// for tests that move the clock themselves, which hand the same
    /// clock to what the model's activity is read by.
    #[doc(hidden)]
    #[must_use]
    pub fn timed_by(self, clock: Arc<dyn Clock>) -> Self {
        Self { clock, ..self }
    }

    /// What `future` comes to, unless the clock reaches `at` first; it
    /// is looked at first, as a timeout looks at what it times.
    async fn by<T>(&self, at: Instant, future: impl std::future::Future<Output = T>) -> Option<T> {
        tokio::select! {
            biased;
            done = future => Some(done),
            () = self.clock.sleep_until(at) => None,
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

    /// This embedder, giving the model `query` to answer a query, and
    /// giving up on a call for documents when the model has made nothing
    /// for anyone in `documents`, rather than [`QUERY_WITHIN`] and
    /// [`DOCUMENTS_WITHIN`].
    #[must_use]
    pub fn answering_within(self, query: Duration, documents: Duration) -> Self {
        Self {
            query_within: query,
            documents_within: documents,
            ..self
        }
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// A vector for each text to be stored, in order. The call waits its
    /// turn at the model for as long as the model is at work, for
    /// whomever: it is given up on when the model has made nothing for
    /// anyone, and is not loading, for as long as a model may go without,
    /// and has then failed in a way that may pass. A model that does not
    /// say what it is doing is given that long for the call in all.
    pub async fn documents(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        let prefix = self.embedding.document_prefix.as_deref();
        let within = Within::from_now(self.documents_within, &*self.clock);
        let sent = AtomicBool::new(false);
        self.embed(texts, prefix, Purpose::Document, within, &sent)
            .await
    }

    /// The vector to search with for `text`. Asked for again, or by several
    /// at once, it is worked out once: those who ask while it is being
    /// worked out wait for whoever asked first, and are told what that
    /// one is, at the same moment, be it a vector or a failure. A failure
    /// is told to those and no one after: the next to ask asks the model.
    ///
    /// No one waits on a call for another text. And no one waits longer
    /// than the model is given to answer a query, whoever it is waiting
    /// on: a model that takes longer has failed in a way that may pass.
    pub async fn query(&self, text: &str) -> Result<Option<Vec<f32>>> {
        self.query_noting(text, &AtomicBool::new(false)).await
    }

    /// As [`Self::query`], setting `sent` if this call itself sends the
    /// text to the model, at the moment it does: whatever then comes of
    /// it, and whether or not the caller waits for it. It is left unset
    /// by a call answered with a vector kept, by one that waits for a
    /// vector another call is having made, by one for a blank text, and
    /// by one refused for a fault that stands, none of which asks the
    /// model anything.
    pub async fn query_noting(&self, text: &str, sent: &AtomicBool) -> Result<Option<Vec<f32>>> {
        if text.trim().is_empty() {
            return Ok(None);
        }
        let within = Within::from_now(self.query_within, &*self.clock);
        let prefix = self.embedding.query_prefix.as_deref();
        let embed = || async {
            let texts = [text.to_string()];
            let vectors = self
                .embed(&texts, prefix, Purpose::Query, within, sent)
                .await?;
            Ok(vectors.into_iter().next().flatten())
        };
        if text.len() > LONGEST_KEPT {
            return embed().await;
        }
        // A vector is kept by what the model was sent to make it: all
        // who embed with the model share what is kept, and those who put
        // another prefix before the text have sent it something else.
        let asked_for = format!("{}{text}", prefix.unwrap_or_default());
        let kept = self.queries().kept(&asked_for);
        // Whoever is first to ask works it out, for as long as that one
        // is given: and a failure is forgotten before anyone is told of
        // it, so that only those waiting by then are. Should the first go
        // away, the next waiting takes its place, with its own time.
        let asked = kept.get_or_init(|| async {
            let made = embed().await;
            if made.is_err() {
                self.queries().forget(&asked_for, &kept);
            }
            made
        });
        match self.by(within.by, asked).await {
            Some(made) => made.clone(),
            None => Err(self.late(within)),
        }
    }

    fn queries(&self) -> std::sync::MutexGuard<'_, Queries> {
        self.shared.queries.lock().expect("query cache lock")
    }

    fn health(&self) -> std::sync::MutexGuard<'_, Health> {
        self.shared.health.lock().expect("health lock")
    }

    /// What a model that has not answered in the time it was given has
    /// failed with: it may yet, asked again.
    fn late(&self, within: Within) -> EmbedError {
        EmbedError::failed(
            &self.embedding.model,
            format!("no answer within {:?}", within.given),
        )
    }

    async fn embed(
        &self,
        texts: &[String],
        prefix: Option<&str>,
        purpose: Purpose,
        within: Within,
        sent: &AtomicBool,
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
        // Said at the moment the text is on its way to the model,
        // whatever comes of it, and whether or not the caller waits.
        sent.store(true, Ordering::Relaxed);
        let made = self.made(texts, wanted, purpose, within).await;
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
        let mut health = self.health();
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
        let mut health = self.health();
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

    /// What `call` comes to, unless the model stops meanwhile: which it
    /// has when it has not been at work, for this call or any other, for
    /// `stalled`, as [`crate::Activity`] counts it. A load under way is
    /// waited out, for as long as a load is given: where it then fails,
    /// a call is given up on within `stalled` of its failing. What the
    /// model is doing is looked at when the time is up, and not when the
    /// call is made: so it does not matter when the call came, before a
    /// load, during one, or while its texts still wait to be taken in.
    async fn unless_stopped<T>(
        &self,
        call: impl std::future::Future<Output = Result<T>>,
        activity: &crate::Activity,
        stalled: Duration,
    ) -> Result<T> {
        let since = self.clock.now();
        let mut look = since + stalled;
        tokio::pin!(call);
        loop {
            tokio::select! {
                made = &mut call => return made,
                () = self.clock.sleep_until(look) => {}
            }
            // What has waited with the model not at work grows no faster
            // than time does: it cannot reach the stall before this is
            // looked at again. A load that fails while it is waited for
            // is held against the waiter when it is next looked at, which
            // is within a stall of it.
            let quiet = activity.quiet(since);
            if quiet >= stalled {
                return Err(EmbedError::failed(
                    &self.embedding.model,
                    format!("the model has made nothing for anyone in {stalled:?}"),
                ));
            }
            look = self.clock.now() + (stalled - quiet);
        }
    }

    /// The vectors the model makes for `wanted`, the texts of `texts` that
    /// are not blank, checked and put back in their places.
    async fn made(
        &self,
        texts: &[String],
        wanted: Vec<String>,
        purpose: Purpose,
        within: Within,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let asked = wanted.len();
        let provider = &self.shared.provider;
        let call = provider.embed(wanted, purpose);
        // Whatever the model is and however it is reached, it is not
        // waited on for ever. Giving up is not hearing that it cannot be
        // used: it may answer the next call.
        let vectors = match (purpose, provider.activity()) {
            // A call for documents waits its turn at a model that says
            // what it is doing, for as long as the model is at work.
            (Purpose::Document, Some(activity)) => {
                self.unless_stopped(call, activity, within.given).await?
            }
            _ => self
                .by(within.by, call)
                .await
                .ok_or_else(|| self.late(within))??,
        };
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

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use vellev_core::EmbeddingProvider;

    use super::*;

    /// A model that never answers.
    #[derive(Debug)]
    struct Silent;

    #[async_trait]
    impl Provider for Silent {
        async fn embed(&self, _: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
            std::future::pending().await
        }
    }

    /// Those who embed with one model wait on one another for the same
    /// query, and may each give the model another time to answer: none
    /// waits longer than its own, whoever it is waiting on.
    #[tokio::test(start_paused = true)]
    async fn one_who_waits_on_another_asking_waits_no_longer_than_its_own_time() {
        let seconds = Duration::from_secs;
        let shared = Arc::new(Shared::new(Arc::new(Silent)));
        let giving = |query: Duration| {
            let embedding = Embedding {
                provider: EmbeddingProvider::Local,
                model: "silent".into(),
                dimensions: 2,
                query_prefix: None,
                document_prefix: None,
                properties: Vec::new(),
            };
            Embedder::sharing(shared.clone(), embedding).answering_within(query, DOCUMENTS_WITHIN)
        };
        let (patient, hurried) = (giving(seconds(60)), giving(seconds(2)));
        let late = |within: &str| EmbedError::Failed {
            model: "silent".into(),
            reason: format!("no answer within {within}"),
        };
        let started = Instant::now();
        let (first, second) = tokio::join!(
            async { (patient.query("fox").await, started.elapsed()) },
            async {
                tokio::time::sleep(seconds(1)).await;
                (hurried.query("fox").await, started.elapsed())
            }
        );
        assert_eq!(second, (Err(late("2s")), seconds(3)));
        assert_eq!(first, (Err(late("60s")), seconds(60)));
    }
}
