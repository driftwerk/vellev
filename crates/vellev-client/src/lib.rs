//! Client for the vellev HTTP API.
//!
//! A [`Client`] knows several nodes and asks whichever serves a tenant. A
//! request that fails in a way that may pass is sent again, on the next
//! node, as long as that cannot make a change twice. Each write is named,
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
    After, Batch, Change, Changes, Code, Collections, Content, Counted, Failure, Hybrid, IssuedKey,
    Keyword, Nearest, Objects, Page, Results, Stored, TenantKey, TenantKeys, Tenants, Written,
    AFTER, REQUEST, ROOT,
};
use vellev_core::{Collection, CollectionName, ObjectId, Property, RequestId, TenantId};
use vellev_index::filter::Filter;

pub use vellev_api as api;

/// The longest wait between two tries.
const LONGEST_BACKOFF: Duration = Duration::from_secs(5);
/// How much of an answer that is not the API's is kept for the error.
const KEPT_BODY: usize = 300;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("request to vellev failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The node answered with a failure of its own.
    #[error("vellev answered {status}: {message}")]
    Api {
        status: StatusCode,
        code: Code,
        message: String,
    },
    /// Something answered that is not the API, such as a load balancer.
    #[error("{status} from something that is not vellev: {body}")]
    Status { status: StatusCode, body: String },
    /// The node answered a success the client could not read.
    #[error("vellev's answer could not be read: {0}")]
    Decode(#[from] serde_json::Error),
    /// [`Records::store_all`] failed part of the way: `stored` objects are
    /// stored, the last of them readable from `position`.
    #[error("{stored} objects stored before a failure: {source}")]
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
    /// It was refused for good.
    Final,
}

impl Error {
    /// The API's code for the failure, if the node gave one.
    pub fn code(&self) -> Option<Code> {
        match self {
            Self::Api { code, .. } => Some(*code),
            Self::Partial { source, .. } => source.code(),
            _ => None,
        }
    }

    pub fn is_not_found(&self) -> bool {
        self.code() == Some(Code::NotFound)
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
            Self::Api { code, .. } => match code {
                Code::Unavailable | Code::Limited => Fate::NoEffect,
                Code::Uncertain => Fate::Unknown,
                _ => Fate::Final,
            },
            Self::Status { status, .. } if status.is_server_error() => Fate::Unknown,
            Self::Status { .. } | Self::Unusable(_) => Fate::Final,
            Self::Partial { source, .. } => source.fate(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Options {
    /// How long one try of a request may take; see [`Client::within`].
    pub timeout: Duration,
    /// How many times a request is tried in all.
    pub attempts: u32,
    /// How long to wait once every node has been tried; each wait after
    /// is twice the one before.
    pub backoff: Duration,
    /// The most objects [`Records::store_all`] sends in one request.
    pub batch_objects: usize,
    /// The most bytes, near enough, it sends in one request. Keep it under
    /// the node's own limit on a request's body.
    pub batch_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            attempts: 4,
            backoff: Duration::from_millis(100),
            batch_objects: 500,
            batch_bytes: 8 * 1024 * 1024,
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
    /// A change to a tenant's data, named so that a node answers a try of
    /// it again as it answered the first.
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
    /// and sending it again is safe, starting from the node that last
    /// answered for the tenant. Every node is tried before any wait.
    async fn ask<T: DeserializeOwned>(
        &self,
        kind: Kind,
        method: Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        tenant: Option<&TenantId>,
    ) -> Result<T> {
        let Inner { nodes, options, .. } = self.inner.as_ref();
        let first = self.known(tenant).node;
        let mut backoff = options.backoff;
        for attempt in 0..options.attempts as usize {
            let node = (first + attempt) % nodes.len();
            let (after, request) = match kind {
                Kind::Read => (tenant.and_then(|tenant| self.after(tenant)), None),
                Kind::Repeatable => (None, None),
                Kind::Write(request) => (None, Some(request)),
            };
            let failure = match self.once(node, &method, path, body, after, request).await {
                Ok(answer) => {
                    self.learnt().entry(tenant.cloned()).or_default().node = node;
                    return Ok(answer);
                }
                Err(failure) => failure,
            };
            if failure.fate() == Fate::Final || attempt + 1 == options.attempts as usize {
                return Err(failure);
            }
            if (attempt + 1) % nodes.len() == 0 {
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2).min(LONGEST_BACKOFF);
            }
        }
        Err(Error::Unusable("a client needs at least one try"))
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
        let bytes = response.bytes().await?;
        if status.is_success() {
            let body: &[u8] = if bytes.is_empty() { b"null" } else { &bytes };
            return Ok(serde_json::from_slice(body)?);
        }
        Err(match serde_json::from_slice::<Failure>(&bytes) {
            Ok(Failure { error }) => Error::Api {
                status,
                code: error.code,
                message: error.message,
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
        let path = self.path(rest);
        let written: Written = self
            .client
            .ask(
                Kind::Write(RequestId::generate()),
                method,
                &path,
                body,
                Some(&self.id),
            )
            .await?;
        let mut learnt = self.client.learnt();
        let known = learnt.entry(Some(self.id.clone())).or_default();
        known.after = Some(match known.after {
            Some(known) => known.later(written.after()),
            None => written.after(),
        });
        Ok(written)
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create(&self) -> Result<()> {
        let path = self.path("");
        self.client
            .ask(
                Kind::Repeatable,
                Method::PUT,
                &path,
                None::<&()>,
                Some(&self.id),
            )
            .await
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
        let path = self.path("/keys");
        self.client
            .ask(
                Kind::Repeatable,
                Method::POST,
                &path,
                None::<&()>,
                Some(&self.id),
            )
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
        let batch = Batch { changes };
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
        let upserts = objects.into_iter().map(|(id, content)| Change::Upsert {
            collection: self.name.clone(),
            id,
            properties: content.properties,
            vector: content.vector,
        });
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

    pub async fn hybrid(&self, asked: &Hybrid) -> Result<Results> {
        self.search("/hybrid", asked).await
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
