//! Client for the vellev HTTP API.
//!
//! A [`Client`] knows several nodes and asks whichever answers. What a
//! node cannot serve just now, or a node that cannot be reached, is tried
//! again on the next node, unless that could make a change twice. The
//! client remembers the position of its own last write to each tenant and
//! asks every read to be at least that fresh, so that it always reads its
//! own writes whichever node answers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::{Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::Serialize;
use vellev_api::{
    Batch, Change, Changes, Code, Collections, Content, Counted, Counting, Failure, Hybrid,
    Keyword, Nearest, Objects, Page, Results, Stored, Tenants, Written, MIN_POSITION, ROOT,
};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
use vellev_index::filter::Filter;

pub use vellev_api as api;

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
    /// The node answered with something that is not the API's.
    #[error("vellev answered {status} with a body that is not its own")]
    Status { status: StatusCode },
    #[error("{0}")]
    Invalid(String),
}

impl Error {
    /// The API's code for the failure, if the node gave one.
    pub fn code(&self) -> Option<Code> {
        match self {
            Self::Api { code, .. } => Some(*code),
            _ => None,
        }
    }

    pub fn is_not_found(&self) -> bool {
        self.code() == Some(Code::NotFound)
    }

    /// Whether the request never reached a node, so that it cannot have
    /// had any effect.
    fn never_arrived(&self) -> bool {
        matches!(self, Self::Http(error) if error.is_connect())
    }

    /// Whether another node, or the same one later, might answer.
    fn may_pass(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout(),
            Self::Api { code, .. } => *code == Code::Unavailable,
            Self::Status { status } => status.is_server_error(),
            Self::Invalid(_) => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Options {
    /// How long one try of a request may take.
    pub timeout: Duration,
    /// How many times a request is tried in all.
    pub attempts: u32,
    /// How long to wait before the second try; each wait after is twice
    /// the one before.
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

struct Inner {
    nodes: Vec<String>,
    key: Option<String>,
    http: reqwest::Client,
    options: Options,
    /// The node that last answered, asked first next time.
    preferred: AtomicUsize,
    positions: Mutex<HashMap<TenantId, u64>>,
}

#[derive(Clone)]
pub struct Client(Arc<Inner>);

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("nodes", &self.0.nodes)
            .field("options", &self.0.options)
            .finish_non_exhaustive()
    }
}

/// Whether a request may be sent again when nobody knows if it arrived.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Repeat {
    Freely,
    OnlyIfItNeverArrived,
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
        if nodes.is_empty() || options.attempts == 0 {
            return Err(Error::Invalid(
                "a client needs a node to ask and a try to ask it with".into(),
            ));
        }
        Ok(Self(Arc::new(Inner {
            nodes: nodes
                .iter()
                .map(|node| node.trim_end_matches('/').to_string())
                .collect(),
            key,
            http: reqwest::Client::builder()
                .timeout(options.timeout)
                .build()?,
            options,
            preferred: AtomicUsize::new(0),
            positions: Mutex::default(),
        })))
    }

    /// Whether the node's process is up.
    pub async fn live(&self) -> Result<()> {
        self.check("/health/live").await
    }

    /// Whether the node can serve requests.
    pub async fn ready(&self) -> Result<()> {
        self.check("/health/ready").await
    }

    async fn check(&self, path: &str) -> Result<()> {
        let response = self
            .0
            .http
            .get(format!("{}{path}", self.0.nodes[0]))
            .send()
            .await?;
        match response.status() {
            status if status.is_success() => Ok(()),
            status => Err(Error::Status { status }),
        }
    }

    /// Up to `limit` tenants after `after`, in id order.
    pub async fn tenants(&self, after: Option<&TenantId>, limit: u32) -> Result<Tenants> {
        let after = after.map_or(String::new(), |after| format!("&after={after}"));
        let path = format!("/tenants?limit={limit}{after}");
        self.ask(Method::GET, &path, None::<&()>, None, Repeat::Freely)
            .await
    }

    pub fn tenant(&self, tenant: &TenantId) -> Tenant {
        Tenant {
            client: self.clone(),
            id: tenant.clone(),
        }
    }

    /// The position of this client's last write to `tenant`, which its
    /// reads of the tenant are asked to have reached.
    pub fn position(&self, tenant: &TenantId) -> Option<u64> {
        self.positions().get(tenant).copied()
    }

    fn positions(&self) -> std::sync::MutexGuard<'_, HashMap<TenantId, u64>> {
        self.0.positions.lock().expect("positions lock")
    }

    /// Send a request, and again while it fails in a way that may pass and
    /// sending it again is safe, starting each time from the node that
    /// last answered.
    async fn ask<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        tenant: Option<&TenantId>,
        repeat: Repeat,
    ) -> Result<T> {
        let inner = &self.0;
        let mut backoff = inner.options.backoff;
        let mut attempt = 1;
        loop {
            let node = inner.preferred.load(Ordering::Relaxed) % inner.nodes.len();
            let failure = match self.once(node, &method, path, body, tenant).await {
                Ok(answer) => return Ok(answer),
                Err(failure) => failure,
            };
            let again = match repeat {
                Repeat::Freely => failure.may_pass(),
                Repeat::OnlyIfItNeverArrived => failure.never_arrived(),
            };
            if !again || attempt == inner.options.attempts {
                return Err(failure);
            }
            inner.preferred.store(node + 1, Ordering::Relaxed);
            tokio::time::sleep(backoff).await;
            backoff *= 2;
            attempt += 1;
        }
    }

    async fn once<T: DeserializeOwned>(
        &self,
        node: usize,
        method: &Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        tenant: Option<&TenantId>,
    ) -> Result<T> {
        let inner = &self.0;
        let url = format!("{}{ROOT}{path}", inner.nodes[node]);
        let request = inner.http.request(method.clone(), url);
        let request = match &inner.key {
            Some(key) => request.bearer_auth(key),
            None => request,
        };
        let request = match tenant.and_then(|tenant| self.position(tenant)) {
            Some(position) => request.header(MIN_POSITION, position),
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
            return serde_json::from_slice(body).map_err(|_| Error::Status { status });
        }
        Err(match serde_json::from_slice::<Failure>(&bytes) {
            Ok(Failure { error }) => Error::Api {
                status,
                code: error.code,
                message: error.message,
            },
            Err(_) => Error::Status { status },
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
        format!("/tenants/{}{rest}", self.id)
    }

    async fn read<T: DeserializeOwned>(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<T> {
        let path = self.path(rest);
        self.client
            .ask(method, &path, body, Some(&self.id), Repeat::Freely)
            .await
    }

    /// Send a write, and remember where it can be read from.
    async fn write(
        &self,
        method: Method,
        rest: &str,
        body: Option<&(impl Serialize + ?Sized)>,
        repeat: Repeat,
    ) -> Result<Written> {
        let path = self.path(rest);
        let written: Written = self
            .client
            .ask(method, &path, body, Some(&self.id), repeat)
            .await?;
        let mut positions = self.client.positions();
        let reached = positions.entry(self.id.clone()).or_default();
        *reached = written.position.max(*reached);
        Ok(written)
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create(&self) -> Result<()> {
        let path = self.path("");
        self.client
            .ask(Method::PUT, &path, None::<&()>, None, Repeat::Freely)
            .await
    }

    /// Forget the tenant and everything in it. Returns whether it existed.
    pub async fn drop(&self) -> Result<bool> {
        let path = self.path("");
        let dropped: Result<()> = self
            .client
            .ask(Method::DELETE, &path, None::<&()>, None, Repeat::Freely)
            .await;
        self.client.positions().remove(&self.id);
        match dropped {
            Ok(()) => Ok(true),
            Err(error) if error.is_not_found() => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub async fn collections(&self) -> Result<Vec<Collection>> {
        let listed: Collections = self.read(Method::GET, "/collections", None::<&()>).await?;
        Ok(listed.collections)
    }

    /// Apply `changes` as one atomic step, each seeing those before it.
    /// Sent again after an unknown outcome only if every change can be
    /// made twice without harm.
    pub async fn change(&self, changes: Vec<Change>) -> Result<Written> {
        let repeat = match changes.iter().all(Change::is_repeatable) {
            true => Repeat::Freely,
            false => Repeat::OnlyIfItNeverArrived,
        };
        let batch = Batch { changes };
        self.write(Method::POST, "/changes", Some(&batch), repeat)
            .await
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
        found(
            self.tenant
                .read(Method::GET, &self.path(""), None::<&()>)
                .await,
        )
    }

    /// Creating a collection that exists exactly so changes nothing.
    pub async fn create(&self, schema: &Collection) -> Result<Written> {
        let path = self.path("");
        self.tenant
            .write(Method::PUT, &path, Some(schema), Repeat::Freely)
            .await
    }

    pub async fn drop(&self) -> Result<Written> {
        let path = self.path("");
        self.tenant
            .write(Method::DELETE, &path, None::<&()>, Repeat::Freely)
            .await
    }

    /// Adding a property that exists unchanged changes nothing.
    pub async fn add_property(&self, property: &Property) -> Result<Written> {
        let path = self.path("/properties");
        self.tenant
            .write(Method::POST, &path, Some(property), Repeat::Freely)
            .await
    }

    pub async fn get(&self, id: ObjectId) -> Result<Option<Stored>> {
        found(
            self.tenant
                .read(Method::GET, &self.object(id), None::<&()>)
                .await,
        )
    }

    /// Store the object whole, replacing any stored under its id.
    pub async fn put(&self, id: ObjectId, content: &Content) -> Result<Written> {
        let path = self.object(id);
        self.tenant
            .write(Method::PUT, &path, Some(content), Repeat::Freely)
            .await
    }

    /// Merge into a stored object. With a condition it is sent again only
    /// if it never arrived: a second arrival would find its own change.
    pub async fn patch(&self, id: ObjectId, changes: &Changes) -> Result<Written> {
        let repeat = match changes.only_if {
            None => Repeat::Freely,
            Some(_) => Repeat::OnlyIfItNeverArrived,
        };
        self.tenant
            .write(Method::PATCH, &self.object(id), Some(changes), repeat)
            .await
    }

    pub async fn delete(&self, id: ObjectId) -> Result<Written> {
        let path = self.object(id);
        self.tenant
            .write(Method::DELETE, &path, None::<&()>, Repeat::Freely)
            .await
    }

    /// Store every object whole, several to a request, each request one
    /// atomic step. Returns the position of the last, or `None` if there
    /// was nothing to store.
    pub async fn store_all(
        &self,
        objects: impl IntoIterator<Item = (ObjectId, Content)>,
    ) -> Result<Option<u64>> {
        let options = &self.tenant.client.0.options;
        let mut position = None;
        let mut batch: Vec<Change> = Vec::new();
        let mut bytes = 0;
        for (id, content) in objects {
            let Content { properties, vector } = content;
            let upsert = Change::Upsert {
                collection: self.name.clone(),
                id,
                properties,
                vector,
            };
            let size = serde_json::to_vec(&upsert).map_or(0, |encoded| encoded.len());
            let full =
                batch.len() >= options.batch_objects.max(1) || bytes + size > options.batch_bytes;
            if full && !batch.is_empty() {
                let sent = std::mem::take(&mut batch);
                position = Some(self.tenant.change(sent).await?.position);
                bytes = 0;
            }
            bytes += size;
            batch.push(upsert);
        }
        if !batch.is_empty() {
            position = Some(self.tenant.change(batch).await?.position);
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

    pub async fn count(&self, filter: Option<Filter>) -> Result<u64> {
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
