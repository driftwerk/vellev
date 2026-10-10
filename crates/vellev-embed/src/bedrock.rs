use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::{ProvideErrorMetadata, SdkError};
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{oneshot, OnceCell, Semaphore};
use tokio::time::Instant;

use crate::{EmbedError, Progress, Provider, Purpose, Result};

const MODEL: &str = "amazon.titan-embed-text-v2";
/// The names Titan goes by: with its version, as Bedrock lists it, or
/// without.
const MODELS: [&str; 2] = [MODEL, "amazon.titan-embed-text-v2:0"];
const DIMENSIONS: [u32; 3] = [256, 512, 1024];
/// Calls in flight at once, from the whole node: Titan takes one text per
/// call.
const CONCURRENCY: usize = 10;
/// How many of those are kept for queries: documents never have more than
/// the rest in flight, however many are waiting, so a query finds a call
/// free unless more queries than this are being embedded at once.
const KEPT_FOR_QUERIES: usize = 2;
/// Tries per call, throttling and server errors being retried.
const ATTEMPTS: u32 = 5;
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a region that was not reached, or did not answer in the time
/// a call is given, is not tried again: every text meant for it meanwhile
/// fails at once. A call that cannot reach Bedrock is tried several times
/// over, for seconds, and each text of each write would hold one of the
/// region's calls for as long, one after another, to find out the same.
pub const UNREACHED_FOR: Duration = Duration::from_secs(5);
/// How long one text may take, every try included.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Titan v2 takes up to 8,192 tokens; characters beyond this are dropped.
const MAX_CHARS: usize = 30_000;

/// Amazon Titan Text Embeddings v2 on Bedrock, normalised, with the node's
/// own AWS credentials.
#[derive(Debug)]
pub struct Bedrock {
    region: String,
    model: String,
    dimensions: u32,
    client: OnceCell<Client>,
    calls: Arc<Calls>,
    /// How long a call may go unanswered before a caller that gives up
    /// on it marks the region as not reached.
    unreached_after: Duration,
}

/// How long a call may go unanswered before a caller that gives up on
/// it marks the region as not reached: one try's time. A caller gives up
/// long before the tries a call is given are spent, so a region that
/// never answers would never be found out by the calls themselves.
const UNREACHED_AFTER: Duration = ATTEMPT_TIMEOUT;

/// The calls a node has to Titan in one region at once, which are that
/// region's share of the calls the node has to Titan in all.
///
/// The node's are [`CONCURRENCY`], whatever regions, models and sizes its
/// collections embed with. Each region the node embeds through has an
/// equal share of them for itself, two at the least, and no more: so a
/// region that is slow or has stopped answering holds its own share of
/// the node's calls and leaves every other region its own. Of a region's
/// share, some are kept for queries, as of a node's, and one at the
/// least. A node embeds through no more than [`MOST_REGIONS`], so that
/// the shares are never more than the node's between them.
#[derive(Debug)]
pub(crate) struct Calls {
    /// A permit for each call in flight to this region, whatever it is
    /// for; and how many there are.
    any: Semaphore,
    at_once: usize,
    /// The calls for documents to this region, and how many there are,
    /// which is also how many texts of one write are in flight or
    /// waiting at once. They are shared out among the tenants that
    /// write, each in turn: so one tenant's thousand texts are not ahead
    /// of another's one. A document waits for one of these first, and
    /// for a call only once it has one. So documents never hold more
    /// calls than are theirs, and of all the documents waiting, no more
    /// than that few ever stand in the line for a call, which is served
    /// in the order it was joined: a query joining it has those few
    /// ahead of it at most, and none while fewer queries are in flight
    /// than calls are kept for them.
    documents: Arc<Turns>,
    documents_at_once: usize,
    /// Since when the region has not been reached, and what was said of
    /// it, if a call to it lately was not answered at all.
    unreached: Mutex<Option<(Instant, String)>>,
}

/// The most regions a node embeds through: as many as can each have two
/// of its calls, one of them kept for queries.
pub(crate) const MOST_REGIONS: usize = CONCURRENCY / 2;

/// So many turns at something, shared out among the tenants that wait
/// for one, each tenant in turn. A tenant that waits is given the next
/// turn to come free before any tenant that has had one since, however
/// many of its own are waiting.
#[derive(Debug)]
struct Turns(Mutex<Waiting>);

#[derive(Debug)]
struct Waiting {
    /// Turns that no one has.
    free: usize,
    /// Those waiting for one, each tenant's in a queue of its own, and
    /// the queues in the order their turns come.
    tenants: VecDeque<(Arc<str>, VecDeque<oneshot::Sender<Turn>>)>,
}

/// A turn, given back when let go of.
#[derive(Debug)]
struct Turn(Arc<Turns>);

impl Turns {
    fn of(turns: usize) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Waiting {
            free: turns,
            tenants: VecDeque::new(),
        })))
    }

    /// A turn for `tenant`, when its turn comes.
    async fn turn(self: &Arc<Self>, tenant: &str) -> Turn {
        let hears = {
            let mut waiting = self.0.lock().expect("turns lock");
            if waiting.free > 0 && waiting.tenants.is_empty() {
                waiting.free -= 1;
                return Turn(self.clone());
            }
            let (says, hears) = oneshot::channel();
            match waiting
                .tenants
                .iter_mut()
                .find(|(waits, _)| **waits == *tenant)
            {
                Some((_, queue)) => queue.push_back(says),
                None => waiting.tenants.push_back((tenant.into(), [says].into())),
            }
            hears
        };
        match hears.await {
            Ok(turn) => turn,
            // What would have given the turn is gone with the turns
            // themselves, which is never while anyone waits for one.
            Err(_) => Turn(self.clone()),
        }
    }
}

impl Drop for Turn {
    /// The turn goes to the tenant whose turn is next, if any waits:
    /// its first waiter, with that tenant then last in turn. One whose
    /// waiter has gone away meanwhile is passed over for it.
    fn drop(&mut self) {
        let next = {
            let mut waiting = self.0 .0.lock().expect("turns lock");
            let mut next = None;
            'tenants: while let Some((tenant, mut queue)) = waiting.tenants.pop_front() {
                while let Some(says) = queue.pop_front() {
                    if !says.is_closed() {
                        if !queue.is_empty() {
                            waiting.tenants.push_back((tenant, queue));
                        }
                        next = Some(says);
                        break 'tenants;
                    }
                }
            }
            if next.is_none() {
                waiting.free += 1;
            }
            next
        };
        // Handed over with nothing held. A waiter gone away between
        // being found and being given the turn gives it up again, which
        // hands it on to the next in turn.
        if let Some(says) = next {
            if let Err(turn) = says.send(Turn(self.0.clone())) {
                drop(turn);
            }
        }
    }
}

impl Calls {
    /// The calls to one region of a node that embeds through `regions`
    /// of them: an equal share of the node's, and two at the least, one
    /// of them kept for queries.
    fn of(regions: usize) -> Arc<Self> {
        let at_once = (CONCURRENCY / regions.max(1)).max(2);
        let kept = (KEPT_FOR_QUERIES * at_once / CONCURRENCY).max(1);
        let documents_at_once = at_once - kept;
        Arc::new(Self {
            any: Semaphore::new(at_once),
            at_once,
            documents: Turns::of(documents_at_once),
            documents_at_once,
            unreached: Mutex::default(),
        })
    }

    /// The calls of a node that embeds through one region alone.
    fn alone() -> Arc<Self> {
        Self::of(1)
    }

    /// The calls to each of `regions`, for a node that embeds through
    /// those and no others: which are [`MOST_REGIONS`] at the most.
    pub(crate) fn shared_among(regions: &[String]) -> std::collections::HashMap<String, Arc<Self>> {
        let calls = |region: &String| (region.clone(), Self::of(regions.len()));
        regions.iter().map(calls).collect()
    }

    /// The region was not reached, as `said`: it is not tried again for
    /// a while.
    fn mark_unreached(&self, said: String) {
        let mut unreached = self.unreached.lock().expect("unreached lock");
        *unreached = Some((Instant::now(), said));
    }

    /// What was said of the region when it was not reached, if that was
    /// so lately that it is not to be tried again yet.
    fn unreached(&self) -> Option<String> {
        let mut unreached = self.unreached.lock().expect("unreached lock");
        match &*unreached {
            Some((since, said)) if since.elapsed() < UNREACHED_FOR => Some(said.clone()),
            _ => unreached.take().and(None),
        }
    }
}

/// A call in flight, which marks the region as not reached if whoever
/// made it gives up on it unanswered after long enough.
struct GivenUpOn<'a> {
    calls: &'a Calls,
    began: Instant,
    after: Duration,
    answered: bool,
}

impl Drop for GivenUpOn<'_> {
    fn drop(&mut self) {
        let waited = self.began.elapsed();
        if !self.answered && waited >= self.after {
            self.calls.mark_unreached(format!(
                "a call to it was given up on after {waited:.0?} without an answer"
            ));
        }
    }
}

/// Whether `error` says the region was not reached at all: nothing came
/// back in the time a call is given, or the connection failed, timed out
/// or broke. What goes wrong before the call is sent, such as credentials
/// that cannot be had or a request that cannot be made, and what the
/// request itself caused, is not the region's doing, and says nothing of
/// it.
fn out_of_reach<R>(error: &SdkError<InvokeModelError, R>) -> bool {
    match error {
        SdkError::TimeoutError(_) => true,
        SdkError::DispatchFailure(failure) => failure
            .as_connector_error()
            .is_some_and(|connector| !connector.is_user()),
        _ => false,
    }
}

#[derive(Deserialize)]
struct Answer {
    embedding: Vec<f32>,
}

impl Bedrock {
    /// Whether `model` at `dimensions` is one that can be embedded with
    /// through Bedrock, as far as can be told without asking it: Titan
    /// by one of its names, at a size it has.
    pub fn admits(model: &str, dimensions: u32) -> Result<()> {
        if !MODELS.contains(&model) {
            return Err(EmbedError::NoSuchModel(format!(
                "{model} is not {MODEL}, the one Bedrock model that can be embedded with"
            )));
        }
        if !DIMENSIONS.contains(&dimensions) {
            return Err(EmbedError::NoSuchModel(format!(
                "{model} has 256, 512 or 1024 dimensions, not {dimensions}"
            )));
        }
        Ok(())
    }

    /// Titan in `region`, standing in for a node of its own: its calls
    /// are counted by themselves, as many at once as a node has.
    pub fn new(region: &str, model: &str, dimensions: u32) -> Result<Self> {
        Self::among(region, model, dimensions, Calls::alone())
    }

    /// Titan in `region`, its calls counted among `calls`, the node's to
    /// that region.
    pub(crate) fn among(
        region: &str,
        model: &str,
        dimensions: u32,
        calls: Arc<Calls>,
    ) -> Result<Self> {
        Self::admits(model, dimensions)?;
        Ok(Self {
            region: region.to_string(),
            model: model.to_string(),
            dimensions,
            client: OnceCell::new(),
            calls,
            unreached_after: UNREACHED_AFTER,
        })
    }

    /// This, with a call a caller gives up on after `after` unanswered
    /// marking the region as not reached, rather than after one try's
    /// time.
    #[must_use]
    pub fn unreached_after(self, after: Duration) -> Self {
        Self {
            unreached_after: after,
            ..self
        }
    }

    /// Titan through a client already configured, which should have these
    /// [`Self::retries`] and [`Self::timeouts`]. It stands in for a node
    /// of its own: its calls are counted by themselves, as many at once
    /// as a node has.
    pub fn with_client(client: Client, model: &str, dimensions: u32) -> Result<Self> {
        Self::admits(model, dimensions)?;
        Ok(Self {
            region: String::new(),
            model: model.to_string(),
            dimensions,
            client: OnceCell::new_with(Some(client)),
            calls: Calls::alone(),
            unreached_after: UNREACHED_AFTER,
        })
    }

    pub fn retries() -> RetryConfig {
        RetryConfig::standard().with_max_attempts(ATTEMPTS)
    }

    pub fn timeouts() -> TimeoutConfig {
        TimeoutConfig::builder()
            .operation_attempt_timeout(ATTEMPT_TIMEOUT)
            .operation_timeout(CALL_TIMEOUT)
            .build()
    }

    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let config = aws_config::defaults(BehaviorVersion::latest())
                    .region(Region::new(self.region.clone()))
                    .retry_config(Self::retries())
                    .timeout_config(Self::timeouts())
                    .load()
                    .await;
                Client::new(&config)
            })
            .await
    }

    fn failed(&self, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::failed(&self.model, reason)
    }

    /// The vector of one text of `tenant`'s, made when there is a call
    /// for it: its turn coming is said through `progress`, as is its
    /// being made.
    async fn one(
        &self,
        text: &str,
        purpose: Purpose,
        tenant: &str,
        progress: Option<&Progress>,
    ) -> Result<Vec<f32>> {
        let _document = match purpose {
            Purpose::Document => Some(self.calls.documents.turn(tenant).await),
            Purpose::Query => None,
        };
        let _call = self.calls.any.acquire().await.map_err(|e| self.failed(e))?;
        if let Some(progress) = progress {
            progress.made();
        }
        // A region found out of reach while this text waited for a call
        // is not asked again for it: the texts ahead of it found that
        // out, and this one would only take as long to find the same.
        let made = match self.calls.unreached() {
            Some(said) => Err(self.failed(format!(
                "Bedrock was not reached a moment ago, and is not tried again for \
                 {UNREACHED_FOR:?}: {said}"
            ))),
            None => {
                // A caller that gives up on the call, long before its
                // tries are spent, finds the region out of reach for
                // everyone: the call itself never would.
                let mut given_up = GivenUpOn {
                    calls: &self.calls,
                    began: Instant::now(),
                    after: self.unreached_after,
                    answered: false,
                };
                let made = self.called(text).await;
                given_up.answered = true;
                made
            }
        };
        if let Some(progress) = progress {
            progress.made();
        }
        made
    }

    /// Ask Titan for the vector of one text.
    async fn called(&self, text: &str) -> Result<Vec<f32>> {
        let text: String = text.chars().take(MAX_CHARS).collect();
        let body = json!({
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": true,
        });
        let answer = self
            .client()
            .await
            .invoke_model()
            .model_id(&self.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body.to_string()))
            .send()
            .await
            .map_err(|error| {
                let said = match (error.code(), error.message()) {
                    (Some(code), Some(message)) => format!("{code}: {message}"),
                    (Some(said), None) | (None, Some(said)) => said.to_string(),
                    (None, None) => error.to_string(),
                };
                // Nothing came back at all, after every try: the region
                // is taken to be out of reach for a while.
                if out_of_reach(&error) {
                    self.calls.mark_unreached(crate::without_keys(&said));
                }
                // Bedrock says the region has no such model: nothing passes
                // until the collection or the region is another. Access
                // denied is not taken so, being also what is said while a
                // permission just granted spreads; nor a refused input,
                // which may be this text's alone.
                let lasting = error
                    .as_service_error()
                    .is_some_and(|refused| refused.is_resource_not_found_exception());
                match lasting {
                    true => {
                        EmbedError::unusable(&self.model, vellev_core::ModelFault::Provider, said)
                    }
                    false => self.failed(said),
                }
            })?;
        serde_json::from_slice::<Answer>(answer.body().as_ref())
            .map(|answer| answer.embedding)
            .map_err(|error| {
                EmbedError::malformed(&self.model, format!("an unreadable answer: {error}"))
            })
    }
}

impl Bedrock {
    /// A vector for each text, as many in flight at once as this region
    /// has calls for their purpose.
    async fn all(
        &self,
        texts: &[String],
        purpose: Purpose,
        tenant: &str,
        progress: Option<&Progress>,
    ) -> Result<Vec<Vec<f32>>> {
        let together = match purpose {
            Purpose::Document => self.calls.documents_at_once,
            Purpose::Query => self.calls.at_once,
        };
        // Each is begun, and so stands in line, only when its turn comes.
        let calls: Vec<_> = texts
            .iter()
            .map(|text| self.one(text, purpose, tenant, progress))
            .collect();
        stream::iter(calls).buffered(together).try_collect().await
    }
}

#[async_trait]
impl Provider for Bedrock {
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        self.all(&texts, purpose, "", None).await
    }

    fn says_how_it_gets_on(&self) -> bool {
        true
    }

    /// The texts wait for calls in their tenant's turn. Each turn that
    /// comes, and each call that comes back, is somewhere got: a write
    /// is given up on when the region has stopped answering, and not for
    /// how many write to it.
    async fn embed_saying(
        &self,
        texts: Vec<String>,
        purpose: Purpose,
        tenant: &str,
        progress: Progress,
    ) -> Result<Vec<Vec<f32>>> {
        self.all(&texts, purpose, tenant, Some(&progress)).await
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::error::ConnectorError;

    use super::*;

    /// Each region has an equal share of the node's calls, two at the
    /// least, and one of a region's share is kept for queries at the
    /// least: so a search is never kept waiting for a write's call,
    /// however many regions a node embeds through.
    #[test]
    fn each_region_has_a_share_of_the_calls_with_one_kept_for_queries_at_the_least() {
        let shares: Vec<(usize, usize)> = (1..=MOST_REGIONS)
            .map(|regions| {
                let calls = Calls::of(regions);
                (calls.at_once, calls.documents_at_once)
            })
            .collect();
        assert_eq!(shares, [(10, 8), (5, 4), (3, 2), (2, 1), (2, 1)]);
        let all: usize = shares.iter().map(|(at_once, _)| at_once).sum::<usize>();
        assert!(shares
            .iter()
            .all(|(at_once, documents)| documents < at_once));
        assert!(all >= CONCURRENCY, "the shares cover the node's calls");
        assert_eq!(Calls::of(MOST_REGIONS).at_once * MOST_REGIONS, CONCURRENCY);
    }

    /// Only a call that reached nothing says the region is out of reach:
    /// not one that could not be made, as for want of credentials.
    #[test]
    fn only_a_call_that_reached_nothing_marks_a_region_out_of_reach() {
        let io = || std::io::Error::other("connection refused");
        let out: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::io(io().into()));
        assert!(out_of_reach(&out));
        let timed_out: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::timeout(io().into()));
        assert!(out_of_reach(&timed_out));
        let no_answer: SdkError<InvokeModelError> = SdkError::timeout_error(io());
        assert!(out_of_reach(&no_answer));
        // A connection that broke in some way the connector does not
        // name is still the region's doing: it is what a region that
        // closes every connection unanswered looks like.
        let broke: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::other("connection closed".into(), None));
        assert!(out_of_reach(&broke));
        let request: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::user("no credentials".into()));
        assert!(!out_of_reach(&request));
        let unmade: SdkError<InvokeModelError> = SdkError::construction_failure(io());
        assert!(!out_of_reach(&unmade));
    }

    /// Two turns at a time, and more tenants than that waiting: each
    /// tenant in turn, its first waiter first; a waiter that has gone
    /// away is passed over; and a turn no one waits for is free for the
    /// next to ask.
    #[tokio::test]
    async fn turns_go_to_each_tenant_in_turn_and_a_waiter_that_left_is_passed_over() {
        let turns = Turns::of(2);
        let (first, second) = (turns.turn("a").await, turns.turn("a").await);
        let given: Arc<Mutex<Vec<&'static str>>> = Arc::default();
        let wait = |tenant: &'static str, name: &'static str| {
            let (turns, given) = (turns.clone(), given.clone());
            tokio::spawn(async move {
                let turn = turns.turn(tenant).await;
                given.lock().unwrap().push(name);
                turn
            })
        };
        let settle = || tokio::time::sleep(Duration::from_millis(20));
        // Waiting, in this order: three of a, one of b, one of c, and one
        // of d that goes away.
        let (a3, a4, a5) = (wait("a", "a3"), wait("a", "a4"), wait("a", "a5"));
        settle().await;
        let (b1, c1) = (wait("b", "b1"), wait("c", "c1"));
        settle().await;
        let gone = wait("d", "d1");
        settle().await;
        gone.abort();
        let _ = gone.await;
        assert!(given.lock().unwrap().is_empty());

        // a's turn, then b's, then c's, then a's again, and a's again:
        // d's waiter is gone, and its turn comes free.
        let given_last = || given.lock().unwrap().last().copied();
        drop(first);
        settle().await;
        assert_eq!(given_last(), Some("a3"));
        let a3 = a3.await.unwrap();
        drop(second);
        settle().await;
        assert_eq!(given_last(), Some("b1"));
        let b1 = b1.await.unwrap();
        drop(a3);
        settle().await;
        assert_eq!(given_last(), Some("c1"));
        let c1 = c1.await.unwrap();
        drop(b1);
        settle().await;
        assert_eq!(given_last(), Some("a4"));
        drop(c1);
        settle().await;
        assert_eq!(given_last(), Some("a5"));
        let (a4, a5) = (a4.await.unwrap(), a5.await.unwrap());
        assert_eq!(turns.0.lock().unwrap().free, 0);
        drop(a4);
        assert_eq!(turns.0.lock().unwrap().free, 1, "no one waits for it");
        assert!(turns.0.lock().unwrap().tenants.is_empty());
        let at_once = tokio::time::timeout(Duration::from_millis(50), turns.turn("e")).await;
        assert!(at_once.is_ok());
        drop(a5);
        assert_eq!(given.lock().unwrap().len(), 5);
    }
}
