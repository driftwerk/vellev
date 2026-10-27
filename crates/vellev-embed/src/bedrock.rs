use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, Region};
use aws_credential_types::provider::error::CredentialsError;
use aws_sdk_bedrockruntime::error::{ProvideErrorMetadata, SdkError};
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OnceCell, Semaphore};
use tokio::time::Instant;

use crate::{Activity, EmbedError, Provider, Purpose, Result};

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
    /// A permit for each call for a document to this region, in flight
    /// or next to be, and how many there are, which is also how many
    /// texts of one write are in flight or waiting at once: so writes
    /// made side by side take turns, and none stands in line with all
    /// its texts. A document waits for one of these first, and for a
    /// call only once it has one. So documents never hold more
    /// calls than are theirs, and of all the documents waiting, no more
    /// than that few ever stand in the line for a call, which is served
    /// in the order it was joined: a query joining it has those few
    /// ahead of it at most, and none while fewer queries are in flight
    /// than calls are kept for them.
    documents: Semaphore,
    documents_at_once: usize,
    /// Since when the region has not been reached, and what was said of
    /// it, if a call to it lately was not answered at all.
    unreached: Mutex<Option<(Instant, String)>>,
    /// When Titan in this region last answered a text with its vector,
    /// at whatever size and for whomever: all who write through the
    /// region wait on the same calls.
    activity: Activity,
}

/// The most regions a node embeds through: as many as can each have two
/// of its calls, one of them kept for queries.
pub(crate) const MOST_REGIONS: usize = CONCURRENCY / 2;

impl Calls {
    /// The calls to one region of a node that embeds through `regions`
    /// of them: an equal share of the node's, and two at the least, one
    /// of them kept for queries.
    fn of(regions: usize) -> Arc<Self> {
        Self::timed(regions, crate::clock::tokio())
    }

    /// As [`Self::of`], what the region does read by `clock`.
    fn timed(regions: usize, clock: Arc<dyn crate::clock::Clock>) -> Arc<Self> {
        debug_assert!(regions <= MOST_REGIONS, "{regions} regions");
        let at_once = CONCURRENCY / regions.max(1);
        let kept = (KEPT_FOR_QUERIES * at_once / CONCURRENCY).max(1);
        let documents_at_once = at_once - kept;
        Arc::new(Self {
            any: Semaphore::new(at_once),
            at_once,
            documents: Semaphore::new(documents_at_once),
            documents_at_once,
            unreached: Mutex::default(),
            activity: Activity::timed_by(clock),
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
        *unreached = Some((self.activity.clock().now(), said));
    }

    /// What was said of the region when it was not reached, if that was
    /// so lately that it is not to be tried again yet.
    fn unreached(&self) -> Option<String> {
        let mut unreached = self.unreached.lock().expect("unreached lock");
        match &*unreached {
            Some((since, said)) if self.activity.clock().now() - *since < UNREACHED_FOR => {
                Some(said.clone())
            }
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
        let waited = self.calls.activity.clock().now() - self.began;
        if !self.answered && waited >= self.after {
            self.calls.mark_unreached(format!(
                "a call to it was given up on after {waited:.0?} without an answer"
            ));
        }
    }
}

/// What an error that is not Bedrock's answer says of where the fault
/// lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unanswered {
    /// The node has no credentials to call with, or they are set up
    /// wrong: nothing was sent, and nothing will be until the node is
    /// put right. It says nothing of the region.
    NoCredentials,
    /// Credentials could not be had this time, in a way that may pass:
    /// whatever gives them out did not answer. Nothing was sent, and it
    /// says nothing of the region.
    CredentialsLate,
    /// The region was not reached at all: nothing came back in the time
    /// a call is given, or the connection failed, timed out or broke.
    OutOfReach,
    /// Anything else: an answer, though it be a refusal or one that
    /// could not be read, or a request that could not be made.
    Other,
}

/// What `error` says of where the fault lies. The SDK reports credentials
/// it could not load as a failure to dispatch the call, as it does a
/// region it could not connect to: they are told apart by what lies under
/// the failure, a credentials error or not.
fn unanswered<R>(error: &SdkError<InvokeModelError, R>) -> Unanswered {
    let mut under: Option<&(dyn std::error::Error + 'static)> = match error {
        SdkError::DispatchFailure(failure) => failure
            .as_connector_error()
            .map(|connector| connector as &(dyn std::error::Error + 'static)),
        _ => None,
    };
    while let Some(cause) = under {
        match cause.downcast_ref::<CredentialsError>() {
            Some(CredentialsError::CredentialsNotLoaded(_))
            | Some(CredentialsError::InvalidConfiguration(_)) => {
                return Unanswered::NoCredentials;
            }
            Some(_) => return Unanswered::CredentialsLate,
            None => under = cause.source(),
        }
    }
    match error {
        SdkError::TimeoutError(_) => Unanswered::OutOfReach,
        SdkError::DispatchFailure(failure) => {
            let connector = failure.as_connector_error();
            match connector.is_some_and(|connector| !connector.is_user()) {
                true => Unanswered::OutOfReach,
                false => Unanswered::Other,
            }
        }
        _ => Unanswered::Other,
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

    /// This, standing in for a node of its own as [`Self::with_client`]
    /// makes it, with what Titan does read by `clock`: for tests that
    /// move the clock themselves. Whatever waits on it times its waits
    /// by the same clock.
    #[cfg(any(test, feature = "test-clock"))]
    #[must_use]
    pub fn timed_by(self, clock: Arc<dyn crate::clock::Clock>) -> Self {
        Self {
            calls: Calls::timed(1, clock),
            ..self
        }
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

    /// The vector of one text, made when there is a call for it.
    async fn one(&self, text: &str, purpose: Purpose) -> Result<Vec<f32>> {
        let _document = match purpose {
            Purpose::Document => Some(self.calls.documents.acquire().await),
            Purpose::Query => None,
        }
        .transpose()
        .map_err(|e| self.failed(e))?;
        let _call = self.calls.any.acquire().await.map_err(|e| self.failed(e))?;
        // A region found out of reach while this text waited for a call
        // is not asked again for it: the texts ahead of it found that
        // out, and this one would only take as long to find the same.
        if let Some(said) = self.calls.unreached() {
            return Err(self.failed(format!(
                "Bedrock was not reached a moment ago, and is not tried again for \
                 {UNREACHED_FOR:?}: {said}"
            )));
        }
        // A caller that gives up on the call, long before its tries are
        // spent, finds the region out of reach for everyone: the call
        // itself never would.
        let mut given_up = GivenUpOn {
            calls: &self.calls,
            began: self.calls.activity.clock().now(),
            after: self.unreached_after,
            answered: false,
        };
        let made = self.called(text).await;
        given_up.answered = true;
        // Only a vector is Titan at work: not a refusal, nor being told
        // to slow down.
        if made.is_ok() {
            self.calls.activity.made();
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
                match unanswered(&error) {
                    // The node has nothing to call with: its own setup,
                    // which stands until it is put right.
                    Unanswered::NoCredentials => {
                        let said =
                            format!("the node has no AWS credentials to call Bedrock with: {said}");
                        return EmbedError::unusable(
                            &self.model,
                            vellev_core::ModelFault::Setup,
                            said,
                        );
                    }
                    Unanswered::CredentialsLate => {
                        let said = format!("AWS credentials could not be had: {said}");
                        return self.failed(said);
                    }
                    // Nothing came back at all, after every try: the
                    // region is taken to be out of reach for a while.
                    Unanswered::OutOfReach => {
                        self.calls.mark_unreached(crate::without_keys(&said));
                    }
                    Unanswered::Other => {}
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

#[async_trait]
impl Provider for Bedrock {
    /// A vector for each text, as many in flight at once as this region
    /// has calls for their purpose.
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        let together = match purpose {
            Purpose::Document => self.calls.documents_at_once,
            Purpose::Query => self.calls.at_once,
        };
        // Each is begun, and so stands in line, only when its turn comes.
        let calls: Vec<_> = texts.iter().map(|text| self.one(text, purpose)).collect();
        stream::iter(calls).buffered(together).try_collect().await
    }

    fn activity(&self) -> Option<&Activity> {
        Some(&self.calls.activity)
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
        assert!(shares
            .iter()
            .all(|(at_once, documents)| documents < at_once));
        // The regions' shares are never more than the node's between
        // them, and are all of the node's where they divide evenly: of
        // ten, three regions have nine between them and four have eight.
        let between: Vec<usize> = (1..=MOST_REGIONS)
            .map(|regions| Calls::of(regions).at_once * regions)
            .collect();
        assert_eq!(between, [10, 10, 9, 8, 10]);
        assert_eq!(CONCURRENCY, 10);
    }

    /// Only a call that reached nothing says the region is out of reach.
    /// These are the failures the connector itself names; what the SDK
    /// makes of missing credentials, and of Bedrock's refusals, is tested
    /// against the SDK, in `tests/bedrock_no_credentials.rs` and
    /// `tests/providers.rs`.
    #[test]
    fn only_a_call_that_reached_nothing_marks_a_region_out_of_reach() {
        let io = || std::io::Error::other("connection refused");
        let out: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::io(io().into()));
        assert_eq!(unanswered(&out), Unanswered::OutOfReach);
        let timed_out: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::timeout(io().into()));
        assert_eq!(unanswered(&timed_out), Unanswered::OutOfReach);
        let no_answer: SdkError<InvokeModelError> = SdkError::timeout_error(io());
        assert_eq!(unanswered(&no_answer), Unanswered::OutOfReach);
        // A connection that broke in some way the connector does not
        // name is still the region's doing: it is what a region that
        // closes every connection unanswered looks like.
        let broke: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::other("connection closed".into(), None));
        assert_eq!(unanswered(&broke), Unanswered::OutOfReach);
        let request: SdkError<InvokeModelError> =
            SdkError::dispatch_failure(ConnectorError::user("a request that cannot be".into()));
        assert_eq!(unanswered(&request), Unanswered::Other);
        let unmade: SdkError<InvokeModelError> = SdkError::construction_failure(io());
        assert_eq!(unanswered(&unmade), Unanswered::Other);
        // Credentials that could not be loaded lie under a failure of
        // the same kind as a broken connection, as the SDK reports them:
        // which is the node's setup, or whatever gives credentials out
        // being slow, and never the region.
        let under = |cause: CredentialsError| -> SdkError<InvokeModelError> {
            SdkError::dispatch_failure(ConnectorError::other(cause.into(), None))
        };
        let none = under(CredentialsError::not_loaded("no credentials in the chain"));
        assert_eq!(unanswered(&none), Unanswered::NoCredentials);
        let wrong = under(CredentialsError::invalid_configuration("no such profile"));
        assert_eq!(unanswered(&wrong), Unanswered::NoCredentials);
        let late = under(CredentialsError::provider_timed_out(Duration::from_secs(5)));
        assert_eq!(unanswered(&late), Unanswered::CredentialsLate);
        let failed = under(CredentialsError::provider_error(
            "the token service is down",
        ));
        assert_eq!(unanswered(&failed), Unanswered::CredentialsLate);
    }
}
