//! Client for the vellev HTTP API.
//!
//! A [`Client`] knows several nodes and asks whichever serves a tenant. A
//! request that fails in a way that may pass is sent again, on the next
//! node, as long as that cannot make a change twice. A read a node cannot
//! do as it is set up is asked of every other address first, and then
//! tried again like any other: behind one address there may be another
//! node. A write it cannot do fails at once: one node makes a tenant's
//! writes, whichever address is asked. Each write is named,
//! the same on every try, and a node answers a try of a write it has
//! already made as it did then, so a write whose fate is unknown is sent
//! again like a read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::{Method, StatusCode};
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::Serialize;
use vellev_api::{
    After, Backup, Backups, Batch, Bounded, Change, Changes, Code, Collections, Content, Counted,
    Failure, Hybrid, IssuedKey, Keyword, Limit, Limits, Missing, Nearest, Objects, Page, Reason,
    RestorePoints, RestoreTo, RestoreWindow, Restored, Results, Stored, TenantKey, TenantKeys,
    Tenants, Written, AFTER, REQUEST, ROOT,
};
use vellev_core::{Collection, CollectionName, ObjectId, Property, RequestId, TenantId, Version};
use vellev_index::filter::Filter;

pub use vellev_api as api;

mod schedule;

use schedule::{Next, Schedule, Tried};

/// How much of an answer that is not the API's is kept for the error.
const KEPT_BODY: usize = 300;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("request to vellev failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The node answered with a failure of its own.
    #[error("vellev answered {status}: {message}")]
    #[non_exhaustive]
    Api {
        status: StatusCode,
        code: Code,
        message: String,
        /// What the request asked too much of, and the most it may ask.
        limit: Option<Limit>,
        /// What the request named that does not exist, if the node said.
        missing: Option<Missing>,
        /// Where the fault lies, for a model a node cannot use.
        reason: Option<Reason>,
        /// How long the node said to leave it before asking again, if it
        /// said: as one does that is answering as many reads as it does
        /// at once.
        wait: Option<Duration>,
        /// The version an object is at, when a write conditional on it
        /// was refused.
        version: Option<Version>,
    },
    /// The tenant asked more of the node than it may; the request had no
    /// effect, and may be sent again after `wait`.
    #[error("vellev turned the request away: {message}")]
    #[non_exhaustive]
    Limited { message: String, wait: Duration },
    /// Something answered that is not the API, such as a load balancer.
    #[error("{status} from something that is not vellev: {body}")]
    Status { status: StatusCode, body: String },
    /// The node answered a success the client could not read.
    #[error("vellev's answer could not be read: {0}")]
    Decode(#[from] serde_json::Error),
    /// [`Records::store_all`] failed part of the way: `stored` objects are
    /// stored, the last of them readable from `position`.
    #[error("{stored} objects stored before a failure: {source}")]
    #[non_exhaustive]
    Partial {
        stored: usize,
        position: Option<u64>,
        source: Box<Error>,
    },
    #[error("{0}")]
    Unusable(&'static str),
}

/// How a request ended, as far as resending it is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// It had no effect, and the same request may succeed later or on
    /// another node.
    NoEffect,
    /// It may or may not have been carried out.
    Unknown,
    /// It had no effect, and the node that answered cannot do it as it is
    /// set up, however often it is asked; another node may, at another
    /// address or behind the same one.
    NotByThisNode,
    /// It was refused for good.
    Final,
}

impl Error {
    /// The API's code for the failure, if the node gave one.
    pub fn code(&self) -> Option<Code> {
        match self {
            Self::Api { code, .. } => Some(*code),
            Self::Limited { .. } => Some(Code::Limited),
            Self::Partial { source, .. } => source.code(),
            _ => None,
        }
    }

    pub fn is_not_found(&self) -> bool {
        self.code() == Some(Code::NotFound)
    }

    /// Whether the node that answered last cannot use the collection's
    /// embedding model as it is set up: for a read, after every address
    /// was asked and the tries were spent; for a write, the node that
    /// holds the tenant, asked once. Nothing was done. Keep the request,
    /// and send it again once a node has been put right; do not drop it.
    /// [`Error::reason`] says where the fault lies.
    pub fn is_model_unavailable(&self) -> bool {
        self.code() == Some(Code::ModelUnavailable)
    }

    /// What the request asked too much of, and the most it may ask, if the
    /// node refused it for that. [`Client::limits`] tells them all.
    pub fn limit(&self) -> Option<Limit> {
        match self {
            Self::Api { limit, .. } => *limit,
            Self::Partial { source, .. } => source.limit(),
            _ => None,
        }
    }

    /// The most of `name` a request may ask for, if the node refused this
    /// one for asking more.
    pub fn limit_max(&self, name: Bounded) -> Option<u64> {
        self.limit()
            .filter(|limit| limit.name == name)
            .map(|limit| limit.max)
    }

    /// What the request named that does not exist, if the node said. A
    /// tenant's key is told every other tenant is missing.
    pub fn missing(&self) -> Option<Missing> {
        match self {
            Self::Api { missing, .. } => *missing,
            Self::Partial { source, .. } => source.missing(),
            _ => None,
        }
    }

    /// Where the fault lies, for a model the node that answered cannot
    /// use as it is set up: in its build, its setup, or what serves the
    /// model.
    pub fn reason(&self) -> Option<Reason> {
        match self {
            Self::Api { reason, .. } => *reason,
            Self::Partial { source, .. } => source.reason(),
            _ => None,
        }
    }

    /// The version an object is at, if a write conditional on it, by
    /// `only_if` or `only_if_version`, was refused as [`Code::Conflict`]:
    /// what to send to write it as it is now.
    pub fn version(&self) -> Option<Version> {
        match self {
            Self::Api { version, .. } => *version,
            Self::Partial { source, .. } => source.version(),
            _ => None,
        }
    }

    /// Whether this is how a node of a release before `GET /v1/limits`
    /// answers it: an admin key that there is no such route, with nothing
    /// said to be missing or with the route said to be; a tenant's key, in
    /// those releases' very words, that the route is not for it. The words
    /// are matched on purpose: nothing else tells that answer from one a
    /// later release may give for another reason, and those releases'
    /// words will not change.
    fn is_from_before_the_limits(&self) -> bool {
        match self {
            Self::Api {
                code: Code::NotFound,
                missing: None | Some(Missing::Route),
                ..
            } => true,
            Self::Api {
                code: Code::Forbidden,
                message,
                limit: None,
                missing: None,
                reason: None,
                ..
            } => message == "only an admin key may do this",
            _ => false,
        }
    }

    /// Whether a write that failed this way, every try of it, may
    /// nevertheless have been made. Read before making it again: a new
    /// call is a new write.
    pub fn is_unknown_outcome(&self) -> bool {
        self.fate() == Fate::Unknown
    }

    fn fate(&self) -> Fate {
        match self {
            Self::Http(error) if error.is_connect() => Fate::NoEffect,
            Self::Http(_) | Self::Decode(_) => Fate::Unknown,
            Self::Limited { .. } => Fate::NoEffect,
            Self::Api { code, .. } => match code {
                Code::Unavailable => Fate::NoEffect,
                Code::Uncertain => Fate::Unknown,
                Code::ModelUnavailable | Code::NotServedHere => Fate::NotByThisNode,
                Code::Invalid
                | Code::Unauthorized
                | Code::Forbidden
                | Code::NotFound
                | Code::Conflict
                | Code::Limited
                | Code::Internal
                | Code::Unknown => Fate::Final,
                // A code added after this client was built. A failure that
                // may pass keeps a code every client knows, so one this
                // client does not know is final.
                _ => Fate::Final,
            },
            Self::Status { status, .. } if status.is_server_error() => Fate::Unknown,
            Self::Status { .. } | Self::Unusable(_) => Fate::Final,
            Self::Partial { source, .. } => source.fate(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// How a client asks. Start from [`Options::default`] and change what is
/// to be otherwise:
///
/// ```
/// use std::time::Duration;
/// let options = vellev_client::Options::default()
///     .timeout(Duration::from_secs(5))
///     .attempts(3);
/// assert_eq!(options.attempts, 3);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    /// How long one try of a request may take; see [`Client::within`].
    pub timeout: Duration,
    /// How many tries a request has: how many times it may fail in a way
    /// that may pass before it is given up. It is sent at most this many
    /// times and one less than there are addresses: an address that says
    /// it cannot do the request as it is set up costs no try while
    /// another has yet to say so, and an address asked whether it can
    /// better a search ranked by the words alone costs none. A write is
    /// sent at most this many times. A request may so take as long as
    /// that many tries of [`Self::timeout`] each, and the waits between,
    /// none longer than a minute.
    pub attempts: u32,
    /// How long to wait before an address is asked again; each wait after
    /// is twice the one before, up to five seconds. An address that turns
    /// a request away for asking too much says how long itself, and so
    /// may one that cannot answer just now.
    pub backoff: Duration,
    /// The most objects [`Records::store_all`] sends in one request.
    pub batch_objects: usize,
    /// The most bytes, near enough, it sends in one request. Keep it under
    /// the node's own limit on a request's body, which [`Client::limits`]
    /// tells.
    pub batch_bytes: usize,
    /// How long each other address is given to better a hybrid search a
    /// node could only rank by the words alone, for want of a model it
    /// can use: somebody is waiting for the answer in hand. A node that
    /// takes longer than this to embed a text does not get to better it,
    /// and with none no address has time to. No longer than
    /// [`Self::timeout`] in any case.
    pub better_within: Duration,
}

impl Options {
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    #[must_use]
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }

    #[must_use]
    pub fn backoff(mut self, backoff: Duration) -> Self {
        self.backoff = backoff;
        self
    }

    #[must_use]
    pub fn batch_objects(mut self, batch_objects: usize) -> Self {
        self.batch_objects = batch_objects;
        self
    }

    #[must_use]
    pub fn batch_bytes(mut self, batch_bytes: usize) -> Self {
        self.batch_bytes = batch_bytes;
        self
    }

    #[must_use]
    pub fn better_within(mut self, better_within: Duration) -> Self {
        self.better_within = better_within;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            attempts: 4,
            backoff: Duration::from_millis(100),
            batch_objects: 500,
            batch_bytes: 8 * 1024 * 1024,
            better_within: Duration::from_secs(2),
        }
    }
}

/// What the client has learnt about a tenant.
#[derive(Debug, Default, Clone, Copy)]
struct Known {
    /// The node that last answered for it.
    node: usize,
    /// Where this client's last write to it left it.
    after: Option<After>,
}

struct Inner {
    nodes: Vec<String>,
    key: Option<String>,
    http: reqwest::Client,
    options: Options,
    /// By tenant; requests that concern no tenant are filed under `None`.
    known: Mutex<HashMap<Option<TenantId>, Known>>,
}

#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
    timeout: Duration,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("nodes", &self.inner.nodes)
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

/// What a request is, as far as sending it again is concerned. Every kind
/// is sent again when its outcome is unknown.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Read,
    /// A change to the tenants whose effect is the same however many times
    /// it is made.
    Repeatable,
    /// A change to a tenant, to its data or its state, named so that a node
    /// answers a try of it again as it answered the first.
    Write(RequestId),
}

impl Client {
    /// A client of one node or load balancer, e.g. `http://vellev:8080`,
    /// for the health checks, which need no key.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Self> {
        let options = Options {
            timeout,
            attempts: 1,
            ..Options::default()
        };
        Self::build(&[base_url], None, options)
    }

    /// A client of `nodes`, any of which may be asked, presenting `key`.
    pub fn connect(nodes: &[&str], key: &str, options: Options) -> Result<Self> {
        Self::build(nodes, Some(key.to_string()), options)
    }

    fn build(nodes: &[&str], key: Option<String>, options: Options) -> Result<Self> {
        if nodes.is_empty() {
            return Err(Error::Unusable("a client needs a node to ask"));
        }
        if options.attempts == 0 || options.batch_objects == 0 {
            return Err(Error::Unusable(
                "a client needs at least one try, and one object to a request",
            ));
        }
        let nodes = nodes
            .iter()
            .map(|node| {
                let node = node.trim_end_matches('/');
                reqwest::Url::parse(node)
                    .map(|_| node.to_string())
                    .map_err(|_| Error::Unusable("a node must be given as a URL"))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            timeout: options.timeout,
            inner: Arc::new(Inner {
                nodes,
                key,
                http: reqwest::Client::builder().build()?,
                options,
                known: Mutex::default(),
            }),
        })
    }

    /// This client with `timeout` for each try of its requests, in place
    /// of [`Options::timeout`]: longer for a large write, shorter where
    /// someone is waiting.
    pub fn within(&self, timeout: Duration) -> Self {
        Self {
            inner: self.inner.clone(),
            timeout,
        }
    }

    /// Whether the first node's process is up.
    pub async fn live(&self) -> Result<()> {
        let path = "/health/live";
        let _: IgnoredAny = self
            .once(0, &Method::GET, path, None::<&()>, None, None)
            .await?;
        Ok(())
    }

    /// Whether the first node can serve requests.
    pub async fn ready(&self) -> Result<()> {
        let path = "/health/ready";
        let _: IgnoredAny = self
            .once(0, &Method::GET, path, None::<&()>, None, None)
            .await?;
        Ok(())
    }

    /// The most a caller may ask of the node; any key may ask. `None`
    /// from a node of a release before it told them, which answers an
    /// admin key that there is no such route, saying nothing of what is
    /// missing, and a tenant's key that the route is not for it. Then go
    /// by what a refusal says: see [`Error::limit`]. Any other failure is
    /// one: a key the node does not know, or anything else forbidden.
    pub async fn limits(&self) -> Result<Option<Limits>> {
        let path = format!("{ROOT}/limits");
        let told = self
            .ask(Kind::Read, Method::GET, &path, None::<&()>, None)
            .await;
        match told {
            Ok(limits) => Ok(Some(limits)),
            Err(error) if error.is_from_before_the_limits() => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Up to `limit` tenants after `after`, in id order.
    pub async fn tenants(&self, after: Option<&TenantId>, limit: u32) -> Result<Tenants> {
        let after = after.map_or(String::new(), |after| format!("&after={after}"));
        let path = format!("{ROOT}/tenants?limit={limit}{after}");
        self.ask(Kind::Read, Method::GET, &path, None::<&()>, None)
            .await
    }

    pub fn tenant(&self, tenant: &TenantId) -> Tenant {
        Tenant {
            client: self.clone(),
            id: tenant.clone(),
        }
    }

    /// Where this client's last write to `tenant` left it: what a read of
    /// the tenant through this client is answered at or past.
    pub fn after(&self, tenant: &TenantId) -> Option<After> {
        self.known(Some(tenant)).after
    }

    fn learnt(&self) -> std::sync::MutexGuard<'_, HashMap<Option<TenantId>, Known>> {
        self.inner.known.lock().expect("client lock")
    }

    fn known(&self, tenant: Option<&TenantId>) -> Known {
        self.learnt()
            .get(&tenant.cloned())
            .copied()
            .unwrap_or_default()
    }

    /// Send a request, and again while it fails in a way that may pass
    /// and sending it again is safe, to the address and after the wait
    /// the [`schedule`] gives.
    async fn ask<T: DeserializeOwned>(
        &self,
        kind: Kind,
        method: Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        tenant: Option<&TenantId>,
    ) -> Result<T> {
        self.ask_for(kind, method, path, body, tenant, |_| true)
            .await
    }

    /// As [`Self::ask`], for a request a node may answer with less than
    /// another would. An answer that does not pass `whole` is kept, and
    /// each address not yet asked is asked once, at once, for no longer
    /// than [`Options::better_within`], whether it can do better: the first whole
    /// answer is given, and failing one the answer kept, never a failure.
    /// Only a whole answer teaches the client which node to ask first.
    ///
    /// A write a node cannot do as it is set up fails there and then: it
    /// is made by the one node that holds its tenant whichever address is
    /// asked, so every address would say the same.
    async fn ask_for<T: DeserializeOwned>(
        &self,
        kind: Kind,
        method: Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        tenant: Option<&TenantId>,
        whole: impl Fn(&T) -> bool,
    ) -> Result<T> {
        let Inner { nodes, options, .. } = self.inner.as_ref();
        let mut node = self.known(tenant).node;
        let mut schedule = Schedule::new(nodes.len(), node, options.attempts, options.backoff);
        let mut kept: Result<T> = Err(Error::Unusable("a client needs at least one try"));
        let mut asking = self.clone();
        loop {
            let (after, request) = match kind {
                Kind::Read => (tenant.and_then(|tenant| self.after(tenant)), None),
                Kind::Repeatable => (None, None),
                Kind::Write(request) => (None, Some(request)),
            };
            let answer = asking.once(node, &method, path, body, after, request).await;
            let tried = match answer {
                Ok(answer) if whole(&answer) => {
                    self.learnt().entry(tenant.cloned()).or_default().node = node;
                    return Ok(answer);
                }
                Ok(less) => {
                    kept = Ok(less);
                    Tried::Less
                }
                // With an answer in hand, how another address failed to
                // better it is of no account.
                Err(_) if kept.is_ok() => Tried::MayPass,
                Err(failure) => {
                    let tried = match (&failure, failure.fate(), kind) {
                        (_, Fate::Final, _) | (_, Fate::NotByThisNode, Kind::Write(_)) => None,
                        (Error::Limited { wait, .. }, ..) => Some(Tried::TurnedAway(*wait)),
                        (_, Fate::NotByThisNode, _) => Some(Tried::NotByThisAddress),
                        (
                            Error::Api {
                                wait: Some(wait), ..
                            },
                            Fate::NoEffect | Fate::Unknown,
                            _,
                        ) => Some(Tried::Busy(*wait)),
                        (_, Fate::NoEffect | Fate::Unknown, _) => Some(Tried::MayPass),
                    };
                    kept = Err(failure);
                    match tried {
                        Some(tried) => tried,
                        None => return kept,
                    }
                }
            };
            match schedule.after(tried) {
                Next::GiveUp => return kept,
                Next::Ask { address, after } => {
                    if !after.is_zero() {
                        tokio::time::sleep(after).await;
                    }
                    node = address;
                }
                Next::AskForBetter { address } => {
                    asking = self.within(self.timeout.min(options.better_within));
                    node = address;
                }
            }
        }
    }

    async fn once<T: DeserializeOwned>(
        &self,
        node: usize,
        method: &Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        after: Option<After>,
        named: Option<RequestId>,
    ) -> Result<T> {
        let Inner {
            nodes, key, http, ..
        } = self.inner.as_ref();
        let request = http
            .request(method.clone(), format!("{}{path}", nodes[node]))
            .timeout(self.timeout);
        let request = match key {
            Some(key) => request.bearer_auth(key),
            None => request,
        };
        let request = match after {
            Some(after) => request.header(AFTER, after.to_string()),
            None => request,
        };
        let request = match named {
            Some(named) => request.header(REQUEST, named.to_string()),
            None => request,
        };
        let request = match body {
            Some(body) => request.json(body),
            None => request,
        };
        let response = request.send().await?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok()?.parse().ok())
            .map(Duration::from_secs);
        let bytes = response.bytes().await?;
        if status.is_success() {
            let body: &[u8] = if bytes.is_empty() { b"null" } else { &bytes };
            return Ok(serde_json::from_slice(body)?);
        }
        Err(match serde_json::from_slice::<Failure>(&bytes) {
            Ok(Failure { error, .. }) if error.code == Code::Limited => Error::Limited {
                message: error.message,
                wait: retry_after.unwrap_or(Duration::from_secs(1)),
            },
            Ok(Failure { error, .. }) => Error::Api {
                status,
                code: error.code,
                message: error.message,
                limit: error.limit,
                missing: error.missing,
                reason: error.reason,
                version: error.version,
                wait: retry_after,
            },
            Err(_) => Error::Status {
                status,
                body: String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(KEPT_BODY)
                    .collect(),
            },
        })
    }
}

/// One tenant, as a client sees it.
#[derive(Debug, Clone)]
pub struct Tenant {
    client: Client,
    id: TenantId,
}

impl Tenant {
    pub fn id(&self) -> &TenantId {
        &self.id
    }

    fn path(&self, rest: &str) -> String {
        format!("{ROOT}/tenants/{}{rest}", self.id)
    }

    async fn read<T: DeserializeOwned>(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<T> {
        let path = self.path(rest);
        self.client
            .ask(Kind::Read, method, &path, body, Some(&self.id))
            .await
    }

    /// Send a write, named afresh, and remember where it can be read from.
    async fn write(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<Written> {
        let written: Written = self.named(method, rest, body).await?;
        self.reached(written.after());
        Ok(written)
    }

    /// Send a change named afresh, the same name on every try, so that a
    /// node answers a try of it again as it answered the first.
    async fn named<T: DeserializeOwned>(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<T> {
        let path = self.path(rest);
        let kind = Kind::Write(RequestId::generate());
        self.client
            .ask(kind, method, &path, body, Some(&self.id))
            .await
    }

    /// Remember that this client's change left the tenant at `after`, so
    /// that its reads are answered at or past it.
    fn reached(&self, after: After) {
        let mut learnt = self.client.learnt();
        let known = learnt.entry(Some(self.id.clone())).or_default();
        known.after = Some(match known.after {
            Some(known) => known.later(after),
            None => after,
        });
    }

    /// Send a request that may be sent again whatever came of it.
    async fn repeated<T: DeserializeOwned>(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<T> {
        let path = self.path(rest);
        self.client
            .ask(Kind::Repeatable, method, &path, body, Some(&self.id))
            .await
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create(&self) -> Result<()> {
        self.repeated(Method::PUT, "", None::<&()>).await
    }

    /// Forget the tenant and everything in it. Returns whether it existed.
    pub async fn drop(&self) -> Result<bool> {
        let path = self.path("");
        let dropped: Result<()> = self
            .client
            .ask(
                Kind::Repeatable,
                Method::DELETE,
                &path,
                None::<&()>,
                Some(&self.id),
            )
            .await;
        self.client.learnt().remove(&Some(self.id.clone()));
        found(dropped).map(|dropped| dropped.is_some())
    }

    /// Issue a key that reaches this tenant alone; needs an admin key. A
    /// try sent again after its answer was lost issues another key, whose
    /// secret nobody learns; revoke it from [`Self::keys`].
    pub async fn issue_key(&self) -> Result<IssuedKey> {
        self.repeated(Method::POST, "/keys", None::<&()>).await
    }

    /// Back this tenant up now; needs an admin key, and a node with a
    /// backup store. A try sent again after its answer was lost makes
    /// another backup, or is refused as a conflict while the first runs.
    pub async fn back_up(&self) -> Result<Backup> {
        self.repeated(Method::POST, "/backups", None::<&()>).await
    }

    /// This tenant's backups, oldest first; needs an admin key.
    pub async fn backups(&self) -> Result<Vec<Backup>> {
        let listed: Backups = self.read(Method::GET, "/backups", None::<&()>).await?;
        Ok(listed.backups)
    }

    /// How far back this tenant may be restored, and the earliest and
    /// latest states it may go to; needs an admin key.
    pub async fn restore_points(&self) -> Result<RestorePoints> {
        self.read(Method::GET, "/restore-points", None::<&()>).await
    }

    /// Put this tenant back as it was; needs an admin key. Restoring a
    /// large tenant takes a while: give the client time with
    /// [`Client::within`]. The restore is named, the same on every try, so
    /// a try sent again after an answer was lost is answered as the first
    /// was, rather than going back again past what was written since.
    /// Another restore of the tenant is refused as a conflict while one
    /// runs. This client's reads are answered at or past the restore.
    pub async fn restore(&self, to: RestoreTo) -> Result<Restored> {
        let restored: Restored = self.named(Method::POST, "/restore", Some(&to)).await?;
        self.reached(restored.after());
        Ok(restored)
    }

    /// Set how far back this tenant may be restored, in seconds, or with
    /// none go back to the default; needs an admin key.
    pub async fn set_restore_window(&self, secs: Option<u64>) -> Result<()> {
        let window = RestoreWindow::new(secs);
        self.repeated(Method::PUT, "/restore-window", Some(&window))
            .await
    }

    /// The keys that reach this tenant, without their secrets; needs an
    /// admin key.
    pub async fn keys(&self) -> Result<Vec<TenantKey>> {
        let listed: TenantKeys = self.read(Method::GET, "/keys", None::<&()>).await?;
        Ok(listed.keys)
    }

    /// Revoke one of this tenant's keys; needs an admin key. Returns
    /// whether it was one.
    pub async fn revoke_key(&self, id: uuid::Uuid) -> Result<bool> {
        let path = self.path(&format!("/keys/{id}"));
        let revoked: Result<()> = self
            .client
            .ask(
                Kind::Repeatable,
                Method::DELETE,
                &path,
                None::<&()>,
                Some(&self.id),
            )
            .await;
        found(revoked).map(|revoked| revoked.is_some())
    }

    pub async fn collections(&self) -> Result<Vec<Collection>> {
        let listed: Collections = self.read(Method::GET, "/collections", None::<&()>).await?;
        Ok(listed.collections)
    }

    /// Apply `changes` as one atomic step, each seeing those before it.
    pub async fn apply(&self, changes: Vec<Change>) -> Result<Written> {
        self.batch(changes).await
    }

    async fn batch(&self, changes: Vec<Change>) -> Result<Written> {
        let batch = Batch::new(changes);
        self.write(Method::POST, "/changes", Some(&batch)).await
    }

    pub fn collection(&self, collection: &CollectionName) -> Records {
        Records {
            tenant: self.clone(),
            name: collection.clone(),
        }
    }
}

/// One collection of a tenant, as a client sees it.
#[derive(Debug, Clone)]
pub struct Records {
    tenant: Tenant,
    name: CollectionName,
}

#[derive(Serialize)]
struct Counting<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<&'a Filter>,
}

impl Records {
    fn path(&self, rest: &str) -> String {
        format!("/collections/{}{rest}", self.name)
    }

    fn object(&self, id: ObjectId) -> String {
        self.path(&format!("/objects/{id}"))
    }

    async fn search<T: DeserializeOwned>(&self, kind: &str, asked: &impl Serialize) -> Result<T> {
        let path = self.path(kind);
        self.tenant.read(Method::POST, &path, Some(asked)).await
    }

    /// The collection's schema, or `None` if there is no such collection.
    pub async fn schema(&self) -> Result<Option<Collection>> {
        let path = self.path("");
        found(self.tenant.read(Method::GET, &path, None::<&()>).await)
    }

    /// Creating a collection that exists exactly so changes nothing.
    pub async fn create(&self, schema: &Collection) -> Result<Written> {
        let path = self.path("");
        self.tenant.write(Method::PUT, &path, Some(schema)).await
    }

    pub async fn drop(&self) -> Result<Written> {
        let path = self.path("");
        self.tenant.write(Method::DELETE, &path, None::<&()>).await
    }

    /// Adding a property that exists unchanged changes nothing.
    pub async fn add_property(&self, property: &Property) -> Result<Written> {
        let path = self.path("/properties");
        self.tenant.write(Method::POST, &path, Some(property)).await
    }

    pub async fn get(&self, id: ObjectId) -> Result<Option<Stored>> {
        let path = self.object(id);
        found(self.tenant.read(Method::GET, &path, None::<&()>).await)
    }

    /// Store the object whole, replacing any stored under its id.
    pub async fn put(&self, id: ObjectId, content: &Content) -> Result<Written> {
        let path = self.object(id);
        self.tenant.write(Method::PUT, &path, Some(content)).await
    }

    /// Merge into a stored object: a property sent as null is removed.
    pub async fn patch(&self, id: ObjectId, changes: &Changes) -> Result<Written> {
        let path = self.object(id);
        self.tenant.write(Method::PATCH, &path, Some(changes)).await
    }

    /// Deleting an object that is not stored changes nothing.
    pub async fn delete(&self, id: ObjectId) -> Result<Written> {
        let path = self.object(id);
        self.tenant.write(Method::DELETE, &path, None::<&()>).await
    }

    /// Delete the object only if it is stored and still at `version`:
    /// [`Code::Conflict`] otherwise, with [`Error::version`] the one it
    /// is at, and [`Code::NotFound`] if it is not stored.
    ///
    /// Sent as a batch of one change rather than as a `DELETE` with the
    /// condition in its query: a node of 0.1 ignores a query it does not
    /// know and would delete unconditionally, where it refuses a change
    /// it does not know.
    pub async fn delete_if_version(&self, id: ObjectId, version: Version) -> Result<Written> {
        let delete = Change::delete_if_version(self.name.clone(), id, version);
        self.tenant.batch(vec![delete]).await
    }

    /// Store every object whole, several to a request, each request one
    /// atomic step. Returns the position of the last, or `None` if there
    /// was nothing to store. If a request fails, those before it stay
    /// stored, and [`Error::Partial`] says how many objects that is.
    pub async fn store_all(
        &self,
        objects: impl IntoIterator<Item = (ObjectId, Content)>,
    ) -> Result<Option<u64>> {
        let options = &self.tenant.client.inner.options;
        let mut stored = 0;
        let mut position = None;
        let mut batch: Vec<Change> = Vec::new();
        let mut bytes = 0;
        let upserts = objects
            .into_iter()
            .map(|(id, content)| Change::upsert(self.name.clone(), id, content));
        for upsert in upserts.map(Some).chain([None]) {
            let size = match &upsert {
                Some(upsert) => serde_json::to_vec(upsert)?.len(),
                None => 0,
            };
            let full = batch.len() >= options.batch_objects
                || bytes + size > options.batch_bytes
                || upsert.is_none();
            if full && !batch.is_empty() {
                let sent = std::mem::take(&mut batch);
                let count = sent.len();
                let written = self
                    .tenant
                    .batch(sent)
                    .await
                    .map_err(|source| Error::Partial {
                        stored,
                        position,
                        source: Box::new(source),
                    })?;
                stored += count;
                position = Some(written.position);
                bytes = 0;
            }
            bytes += size;
            batch.extend(upsert);
        }
        Ok(position)
    }

    pub async fn nearest(&self, asked: &Nearest) -> Result<Results> {
        self.search("/nearest", asked).await
    }

    pub async fn keyword(&self, asked: &Keyword) -> Result<Results> {
        self.search("/keyword", asked).await
    }

    /// A node that cannot embed the text ranks by the words alone and
    /// says so in [`Results::degraded_by`]. If it cannot as it is set up,
    /// each other address not yet asked is asked once, at once and for no
    /// longer than [`Options::better_within`], and its answer is given
    /// only if none does better; the address that does is the one asked
    /// first from then on. A client of one address is given it straight
    /// away.
    pub async fn hybrid(&self, asked: &Hybrid) -> Result<Results> {
        let path = self.tenant.path(&self.path("/hybrid"));
        let whole = |results: &Results| results.degraded_by != Some(Code::ModelUnavailable);
        let tenant = Some(&self.tenant.id);
        self.tenant
            .client
            .ask_for(Kind::Read, Method::POST, &path, Some(asked), tenant, whole)
            .await
    }

    pub async fn list(&self, page: &Page) -> Result<Objects> {
        self.search("/list", page).await
    }

    pub async fn count(&self, filter: Option<&Filter>) -> Result<u64> {
        let counted: Counted = self.search("/count", &Counting { filter }).await?;
        Ok(counted.count)
    }
}

/// What was asked for, or `None` if the node says there is no such thing.
fn found<T>(answer: Result<T>) -> Result<Option<T>> {
    match answer {
        Ok(found) => Ok(Some(found)),
        Err(error) if error.is_not_found() => Ok(None),
        Err(error) => Err(error),
    }
}
