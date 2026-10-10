use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::OnceCell;
use tokio::time::Instant;
use vellev_core::{Embedding, ModelFault};

use crate::{EmbedError, Progress, Provider, Purpose, Result};

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
/// How long a model may go without getting anywhere with a call for
/// documents, unless an embedder is told otherwise, before the call is
/// given up on: without making any of its texts, and without what stands
/// ahead of them in their own tenant's line being done with. It is not
/// how long the call may take. A model that others are asking much of is
/// slow to finish a call and still gets on with it all the while; one
/// that has stopped does not. A model that keeps a line says besides how
/// long it is reckoned to be before it comes to each call, and the call
/// is given that too.
///
/// A model that does not say how it is getting on is given this for the
/// whole of a call, and [`EACH_DOCUMENT_WITHIN`] more for each text.
pub const DOCUMENTS_WITHIN: Duration = Duration::from_secs(60);
pub const EACH_DOCUMENT_WITHIN: Duration = Duration::from_secs(1);
/// The longest a call for documents is kept waiting for its turn at a
/// model that keeps a line, unless an embedder is told otherwise. One the
/// model is reckoned not to come to within this is refused at once, to
/// be sent again when there is less to do, rather than kept: a node with
/// more to do than it can do soon sheds it, and holds nothing without
/// bound. Two minutes is as long as a call to an endpoint is given, and
/// longer than any client waits.
pub const DOCUMENTS_HELD_AT_MOST: Duration = Duration::from_secs(120);
/// How many times over a call for documents may take what it is given
/// alone, [`DOCUMENTS_WITHIN`] and [`EACH_DOCUMENT_WITHIN`] for each text,
/// and [`DOCUMENTS_HELD_AT_MOST`] besides, from when it is made, however
/// steadily the model gets on with it. Nothing a model does by sharing
/// itself among those who ask of it slows one of them by as much: this
/// is there for a model that goes on saying it gets somewhere and never
/// finishes.
const SHARED: u32 = 8;

/// The longest any call is given, whatever it is told and however many
/// its texts: a time that can be added to the present one.
const LONGEST_WAIT: Duration = Duration::from_secs(24 * 60 * 60);

/// How many query vectors are kept for a model: one request often
/// searches several times with the same text.
const QUERY_CACHE: usize = 1_024;
/// Queries longer than this, in bytes, are not kept.
const LONGEST_KEPT: usize = 1_024;

type Kept = Arc<OnceCell<Result<Option<Vec<f32>>>>>;

/// The vectors of the queries last asked of a model, by the text the
/// model was sent, prefix and all, each filled in by whoever asks first
/// while the others wait for it. Those waiting are told what the first
/// was, a failure as much as a vector: but a failure is not kept, and
/// whoever asks after it asks the model.
///
/// When there are more than are kept, the one asked for longest ago is
/// forgotten: so a text searched with again and again stays, whatever
/// else is searched with between. It is still the model's, shared by
/// every tenant whose collections embed with it. Anyone who searches
/// with enough texts of their own has every other forgotten, which
/// costs the others an embedding each. And a search answered from what
/// is kept is answered sooner than one that is not: whoever can time a
/// search can so tell that someone, on that node and with that model,
/// searched with exactly that text not long ago, though not who. Keeping
/// each tenant's apart needs the caller to say whose a search is, which
/// it does not.
#[derive(Debug, Default)]
struct Queries {
    /// What is kept for each text, and when it was last asked for.
    vectors: HashMap<Arc<str>, (Kept, u64)>,
    /// The texts by when each was last asked for.
    last_asked: BTreeMap<u64, Arc<str>>,
    /// How many times a text has been asked for.
    asked: u64,
}

impl Queries {
    /// What is kept for `text`, which is thereby the last asked for.
    fn kept(&mut self, text: &str) -> Kept {
        self.asked += 1;
        if let Some((kept, last)) = self.vectors.get_mut(text) {
            if let Some(text) = self.last_asked.remove(last) {
                self.last_asked.insert(self.asked, text);
            }
            *last = self.asked;
            return kept.clone();
        }
        let text: Arc<str> = text.into();
        let kept = Kept::default();
        self.vectors
            .insert(text.clone(), (kept.clone(), self.asked));
        self.last_asked.insert(self.asked, text);
        if self.vectors.len() > QUERY_CACHE {
            if let Some((_, longest_ago)) = self.last_asked.pop_first() {
                self.vectors.remove(&longest_ago);
            }
        }
        kept
    }

    /// Have whoever next asks for `text` ask the model, if `failed` is
    /// what is kept for it: those who wait on it already are told how it
    /// failed, and no one else is. It may no longer be what is kept:
    /// forgotten for being asked for longest ago while it was still being
    /// worked out, and the text asked for and answered afresh since. What
    /// is kept then is a vector, and stays.
    fn forget(&mut self, text: &str, failed: &Kept) {
        if let Some((kept, _)) = self.vectors.get_mut(text) {
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
    /// How long the model may go without getting anywhere with the
    /// call, if it says how it gets on: the time the call is given is
    /// then counted from when the model turns to it, several times over.
    stalled: Option<Duration>,
}

impl Within {
    /// `given` from now, or a day if that is less.
    fn from_now(given: Duration) -> Self {
        let given = given.min(LONGEST_WAIT);
        Self {
            given,
            by: Instant::now() + given,
            stalled: None,
        }
    }

    /// `given` from now by a model that does not say how it gets on;
    /// and by one that does, for as long as it is never `stalled` so
    /// long.
    fn unless_stalled(given: Duration, stalled: Duration) -> Self {
        Self {
            stalled: Some(stalled.min(LONGEST_WAIT)),
            ..Self::from_now(given)
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
    /// How long the model is given to answer a query; a call for
    /// documents; and each text of one besides.
    query_within: Duration,
    documents_within: Duration,
    each_document_within: Duration,
    /// The longest a call for documents is kept waiting for its turn.
    documents_held_at_most: Duration,
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
            each_document_within: EACH_DOCUMENT_WITHIN,
            documents_held_at_most: DOCUMENTS_HELD_AT_MOST,
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
    /// `documents` to get somewhere with a call for documents, with
    /// `each_document` for each text, rather than [`QUERY_WITHIN`],
    /// [`DOCUMENTS_WITHIN`] and [`EACH_DOCUMENT_WITHIN`]. No call is given
    /// more than a day, whatever is asked.
    #[must_use]
    pub fn answering_within(
        self,
        query: Duration,
        documents: Duration,
        each_document: Duration,
    ) -> Self {
        Self {
            query_within: query,
            documents_within: documents,
            each_document_within: each_document,
            ..self
        }
    }

    /// This embedder, having a call for documents kept waiting for its
    /// turn no longer than `held`, rather than [`DOCUMENTS_HELD_AT_MOST`].
    #[must_use]
    pub fn holding_at_most(self, held: Duration) -> Self {
        Self {
            documents_held_at_most: held,
            ..self
        }
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// A vector for each text to be stored, in order. A model that says
    /// how it gets on is given up on when it has got nowhere with them,
    /// nor with the line they wait in, for so long: however long the call
    /// takes for what others ask of the model meanwhile. One that does
    /// not say is given so long for the call, and longer the more texts
    /// there are. Either way the model has failed in a way that may pass.
    pub async fn documents(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        self.documents_for("", texts).await
    }

    /// As [`Self::documents`], for `tenant`: an opaque name, by which a
    /// model that keeps a line shares itself out among those who ask of
    /// it. All who give none share one turn.
    pub async fn documents_for(
        &self,
        tenant: &str,
        texts: &[String],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let prefix = self.embedding.document_prefix.as_deref();
        let each = u32::try_from(texts.len()).unwrap_or(u32::MAX);
        let within = self
            .documents_within
            .saturating_add(self.each_document_within.saturating_mul(each));
        let within = Within::unless_stalled(within, self.documents_within);
        let sent = AtomicBool::new(false);
        self.embed(texts, prefix, Purpose::Document, tenant, within, &sent)
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
        self.query_noting_for("", text, &AtomicBool::new(false))
            .await
    }

    /// As [`Self::query`], for `tenant`, as [`Self::documents_for`] has
    /// it. What is kept for a query is kept for the model, and shared by
    /// every tenant that embeds with it.
    pub async fn query_for(&self, tenant: &str, text: &str) -> Result<Option<Vec<f32>>> {
        self.query_noting_for(tenant, text, &AtomicBool::new(false))
            .await
    }

    /// As [`Self::query`], setting `sent` if this call itself sends the
    /// text to the model, at the moment it does: whatever then comes of
    /// it, and whether or not the caller waits for it. It is left unset
    /// by a call answered with a vector kept, by one that waits for a
    /// vector another call is having made, by one for a blank text, and
    /// by one refused for a fault that stands, none of which asks the
    /// model anything.
    pub async fn query_noting(&self, text: &str, sent: &AtomicBool) -> Result<Option<Vec<f32>>> {
        self.query_noting_for("", text, sent).await
    }

    /// As [`Self::query_noting`], for `tenant`.
    pub async fn query_noting_for(
        &self,
        tenant: &str,
        text: &str,
        sent: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        if text.trim().is_empty() {
            return Ok(None);
        }
        let within = Within::from_now(self.query_within);
        let prefix = self.embedding.query_prefix.as_deref();
        let embed = || async {
            let texts = [text.to_string()];
            let vectors = self
                .embed(&texts, prefix, Purpose::Query, tenant, within, sent)
                .await?;
            Ok(vectors.into_iter().next().flatten())
        };
        if text.len() > LONGEST_KEPT {
            return embed().await;
        }
        // A vector is kept by what the model was sent to make it: all
        // who embed with the model share what is kept, and those who put
        // another prefix before the text have sent it something else.
        let sent = format!("{}{text}", prefix.unwrap_or_default());
        let kept = self.queries().kept(&sent);
        // Whoever is first to ask works it out, for as long as that one
        // is given. A failure is forgotten before it is told: whoever
        // took up this text before then is told it, having waited on it
        // or come only that moment, and whoever asks after asks the
        // model. Should the one working it out go away, the next waiting
        // works it out, for as long as that one is given. And each waits
        // no longer than its own time, whoever it waits on.
        let asked = kept.get_or_init(|| async {
            let made = embed().await;
            if made.is_err() {
                self.queries().forget(&sent, &kept);
            }
            made
        });
        match tokio::time::timeout_at(within.by, asked).await {
            Ok(made) => made.clone(),
            Err(_) => Err(self.late(within)),
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
        tenant: &str,
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
        // Said at the moment the text is on its way to the model, whatever
        // comes of it, and whether or not the caller waits for it.
        sent.store(true, Ordering::Relaxed);
        let made = self.made(texts, wanted, purpose, tenant, within).await;
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

    /// What a model that says how it gets on makes of `wanted` for
    /// `tenant`: waited on for as long as it is never `stalled` so long,
    /// and as long again as the model last said it will be before it
    /// comes to them; and for no more than several times what the call
    /// is given alone, from now.
    async fn while_it_gets_on(
        &self,
        wanted: Vec<String>,
        purpose: Purpose,
        tenant: &str,
        within: Within,
        stalled: Duration,
    ) -> Result<Vec<Vec<f32>>> {
        let (progress, mut hears) = Progress::heard();
        let provider = &self.shared.provider;
        let mut made = std::pin::pin!(provider.embed_saying(wanted, purpose, tenant, progress));
        let failed = |reason: String| EmbedError::failed(&self.embedding.model, reason);
        let at_most = within
            .given
            .saturating_mul(SHARED)
            .saturating_add(self.documents_held_at_most)
            .min(LONGEST_WAIT);
        let by = Instant::now() + at_most;
        // How long the model last said it may be, if it said.
        let mut allowed: Option<Duration> = None;
        let given_up = tokio::time::sleep(stalled);
        tokio::pin!(given_up);
        // Whether the model can still be heard from. One that lets go of
        // what it would say it with has no more to say, and is given no
        // longer for it than it was last given.
        let mut hearing = true;
        loop {
            tokio::select! {
                biased;
                made = &mut made => return made,
                heard = hears.changed(), if hearing => {
                    hearing = heard.is_ok();
                    if hearing {
                        allowed = hears.borrow_and_update().allowed;
                        let within = stalled.saturating_add(allowed.unwrap_or_default());
                        given_up.as_mut().reset(Instant::now() + within.min(LONGEST_WAIT));
                    }
                }
                () = tokio::time::sleep_until(by) => {
                    return Err(failed(format!("no answer within {at_most:?}")));
                }
                () = &mut given_up => {
                    return Err(failed(match allowed {
                        Some(allowed) => format!(
                            "the model did not come to it within {allowed:?} and {stalled:?} \
                             more, which is as long as it was given for what stood ahead of it"
                        ),
                        None => format!("nothing from the model in {stalled:?}"),
                    }));
                }
            }
        }
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

    /// The vectors the model makes for `wanted`, the texts of `texts` that
    /// are not blank, checked and put back in their places.
    async fn made(
        &self,
        texts: &[String],
        wanted: Vec<String>,
        purpose: Purpose,
        tenant: &str,
        within: Within,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let asked = wanted.len();
        // Whatever the model is and however it is reached, it is waited
        // on no longer than this. Giving up is not hearing that it cannot
        // be used: it may answer the next call.
        let provider = &self.shared.provider;
        let vectors = match within.stalled {
            Some(stalled) if provider.says_how_it_gets_on() => {
                self.while_it_gets_on(wanted, purpose, tenant, within, stalled)
                    .await?
            }
            _ => tokio::time::timeout_at(within.by, provider.embed(wanted, purpose))
                .await
                .map_err(|_| self.late(within))??,
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
            Embedder::sharing(shared.clone(), embedding).answering_within(
                query,
                DOCUMENTS_WITHIN,
                EACH_DOCUMENT_WITHIN,
            )
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

#[cfg(test)]
mod kept {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use vellev_core::EmbeddingProvider;

    use super::*;

    /// A model that answers every call but its first for "fox", which it
    /// keeps until it is let go, and then fails.
    #[derive(Debug, Default)]
    struct FailingFirst {
        calls: AtomicUsize,
        foxes: AtomicUsize,
        let_go: Notify,
    }

    #[async_trait]
    impl Provider for FailingFirst {
        async fn embed(&self, texts: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if texts == ["fox"] && self.foxes.fetch_add(1, Ordering::SeqCst) == 0 {
                self.let_go.notified().await;
                return Err(EmbedError::failed("fake", "not now"));
            }
            Ok(vec![vec![1.0, 2.0]; texts.len()])
        }
    }

    fn embedder(model: Arc<FailingFirst>) -> Arc<Embedder> {
        let embedding = Embedding {
            provider: EmbeddingProvider::Local,
            model: "fake".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        };
        Arc::new(Embedder::new(model, embedding))
    }

    #[tokio::test]
    async fn a_text_searched_with_again_stays_whatever_else_is_searched_with_between() {
        let model = Arc::new(FailingFirst::default());
        let embedder = embedder(model.clone());
        let calls = || model.calls.load(Ordering::SeqCst);
        embedder.query("hen").await.unwrap();
        // More texts than are kept, with the first searched with again
        // partway: it is the others asked for longest ago that go.
        for n in 0..QUERY_CACHE * 3 {
            embedder.query(&format!("other {n}")).await.unwrap();
            if n % (QUERY_CACHE / 2) == 0 {
                embedder.query("hen").await.unwrap();
            }
        }
        assert_eq!(calls(), 1 + QUERY_CACHE * 3, "it was never made again");
        // No more are kept than are kept, and those are the last asked
        // for: the first of the others is made again, and the last not.
        embedder
            .query(&format!("other {}", QUERY_CACHE * 3 - 1))
            .await
            .unwrap();
        assert_eq!(calls(), 1 + QUERY_CACHE * 3);
        embedder.query("other 0").await.unwrap();
        assert_eq!(calls(), 2 + QUERY_CACHE * 3);
        let queries = embedder.queries();
        assert_eq!(queries.vectors.len(), QUERY_CACHE);
        assert_eq!(queries.last_asked.len(), QUERY_CACHE);
    }

    /// A failure is forgotten so that the next to ask asks the model. But
    /// what is kept for the text by then may be a vector made since, for
    /// someone who asked after this asking was itself forgotten.
    #[tokio::test]
    async fn a_failure_that_comes_late_does_not_forget_a_vector_made_since() {
        let model = Arc::new(FailingFirst::default());
        let embedder = embedder(model.clone());
        let calls = || model.calls.load(Ordering::SeqCst);
        let first = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.query("fox").await })
        };
        while calls() == 0 {
            tokio::task::yield_now().await;
        }
        // While it is being worked out, enough else is searched with for
        // it to be forgotten, and it is asked for afresh and answered.
        for n in 0..QUERY_CACHE {
            embedder.query(&format!("other {n}")).await.unwrap();
        }
        assert_eq!(embedder.query("fox").await, Ok(Some(vec![1.0, 2.0])));
        let so_far = calls();
        assert_eq!(so_far, 1 + QUERY_CACHE + 1);

        model.let_go.notify_one();
        assert!(first.await.unwrap().is_err());
        assert_eq!(embedder.query("fox").await, Ok(Some(vec![1.0, 2.0])));
        assert_eq!(calls(), so_far, "the vector made since is still kept");
    }
}
