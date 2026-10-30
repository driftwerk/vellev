use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::extract::{
    DefaultBodyLimit, FromRequestParts, MatchedPath, RawPathParams, Request, State as Shared,
};
use axum::http::header::{self, HeaderMap, HeaderName, AUTHORIZATION};
use axum::http::{Method, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use utoipa::openapi::security::{Http, HttpAuthScheme, SecurityScheme};
use utoipa::openapi::OpenApi;
use utoipa::{Modify, OpenApi as _};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use vellev_api::{Bounded, Code, Missing};
use vellev_core::{Collection, EmbeddingProvider, TenantId};
use vellev_node::{Node, NodeError};
use vellev_storage::coordinator::MAX_RESTORE_WINDOW;
use vellev_storage::{DEFAULT_CANDIDATES, DEFAULT_OVERSAMPLE, DELETED_AT_MOST};

use crate::allocator::Allocator;
use crate::api;
use crate::identity::{Caller, IdentityKeys};
use crate::memory::MemoryFiles;
use crate::metrics::{self, Metrics, Sources, Stage};
use crate::problem::ApiError;
use crate::turns::{Turn, Turns};

/// The most candidates per result a search may ask to have scored exactly.
const MAX_OVERSAMPLE: u32 = 1_024;

/// What a caller may ask of the node.
#[derive(Debug, Clone)]
pub struct Limits {
    /// The most results a search or a page may ask for.
    pub results: usize,
    /// The most bytes a request's body may have.
    pub body: usize,
    /// The embedding endpoints a collection may name.
    pub embedding_urls: Arc<[String]>,
    /// The most reads answered at once, half of them at most for any one
    /// tenant while another's waits.
    pub reads: usize,
    /// How long a read waits for its turn before it is turned away.
    pub reads_wait: Duration,
}

/// How long a read turned away is told to wait before it asks again.
const READ_AGAIN_AFTER: Duration = Duration::from_secs(1);

/// A read's turn, one of the node's. See [`App::reading`].
#[doc(hidden)]
#[derive(Debug)]
pub struct Reading {
    _turn: Turn,
}

impl Reading {
    /// How many parts this read may split its search into: see
    /// [`crate::turns::split_for`].
    #[doc(hidden)]
    pub fn split(&self) -> usize {
        self._turn.split()
    }

    /// Run `work`, the read itself, off the async threads. The turn goes
    /// with it, to be kept until its thread is done with the read,
    /// whoever stops waiting for the answer first.
    #[doc(hidden)]
    pub async fn run<T: Send + 'static>(
        self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, ApiError> {
        let queued = std::time::Instant::now();
        let read = tokio::task::spawn_blocking(move || {
            let _turn = self;
            let began = std::time::Instant::now();
            let done = work();
            (
                done,
                began.saturating_duration_since(queued),
                began.elapsed(),
            )
        });
        let (done, waited, ran) = read.await.map_err(|error| {
            tracing::error!(%error, "a read did not finish");
            ApiError::new(Code::Internal, "the node failed to serve the request")
        })?;
        metrics::record(Stage::Queue, waited);
        metrics::record(Stage::Read, ran);
        Ok(done)
    }
}

/// What every request is served with. The node may not be there yet: a
/// process that is up answers that it is live, and that it is not ready,
/// while what it depends on is still being reached.
#[derive(Clone)]
pub struct App {
    node: Arc<OnceLock<Arc<Node>>>,
    /// Set once the node has been told to stop: it reports not ready, and
    /// asks callers to close each connection after its answer, for as long
    /// as it serves before it closes its port.
    stopping: Arc<AtomicBool>,
    /// The keys a caller may present; with none, nobody may call.
    keys: Arc<[String]>,
    /// The keys whose identity tokens are trusted.
    pub(crate) identity_keys: Arc<IdentityKeys>,
    limits: Limits,
    /// A turn for each read being answered: see [`Self::reading`].
    reading: Arc<Turns>,
    /// What the node measures of the requests it answers.
    metrics: Arc<Metrics>,
    /// Where it reads what memory it really holds.
    memory: Sources,
    /// For sending a request on to the node that holds its tenant.
    http: reqwest::Client,
}

/// Marks a request one node sent on to another, which goes no further, and
/// carries the protocol the sender speaks.
const FORWARDED: &str = "x-vellev-forwarded";
/// How long a node waits for the node it sent a request on to, and for a
/// connection to it. Shorter than a client's usual patience, so that the
/// client hears `unavailable` and asks again rather than giving up.
const FORWARD_TIMEOUT: Duration = Duration::from_secs(25);
/// How long a node waits for the node it sent a restore on to, which
/// loads and writes a whole tenant.
const RESTORE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const FORWARD_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

impl App {
    /// A turn to answer a read of `tenant`, kept until the read is done.
    ///
    /// A read runs on one of the runtime's threads for work off the async
    /// ones, where commits do part of theirs: were reads to take every
    /// one of them, as they would with each stuck on a disk that stopped
    /// answering, no commit could be made. So a node answers so many reads
    /// at once. A tenant may have them all while no other asks; while
    /// another tenant's read waits for a turn, none is given more than
    /// half (see [`crate::turns`] for the rule exactly). A
    /// read that finds no turn waits a moment for one, and is then turned
    /// away to be asked again shortly.
    ///
    /// The turn is for the thread and nothing else. It is taken last,
    /// when the read has all it needs and only the reading is left: what
    /// a read waits for before that, a text to be embedded, its tenant to
    /// be opened, the coordinator to say what is committed, uses no
    /// thread, has its own bound, and would otherwise keep turns from
    /// reads that need nothing of the kind.
    #[doc(hidden)]
    pub async fn reading(&self, tenant: &TenantId) -> Result<Reading, ApiError> {
        let turn = self.reading.take(tenant);
        let waited = tokio::time::timeout(self.limits.reads_wait, turn);
        match metrics::timed(Stage::Turn, waited).await {
            Ok(turn) => Ok(Reading { _turn: turn }),
            Err(_) => {
                let busy = "the node is answering as many reads as it does at once; ask again";
                Err(ApiError::unavailable(busy).after(READ_AGAIN_AFTER))
            }
        }
    }

    pub fn new(keys: Arc<[String]>, limits: Limits) -> Self {
        let node = Arc::<OnceLock<Arc<Node>>>::default();
        let reading = Turns::new(limits.reads);
        let memory = Sources::default();
        Self {
            metrics: Arc::new(Metrics::new(node.clone(), reading.clone(), memory.clone())),
            memory,
            node,
            stopping: Arc::default(),
            keys,
            identity_keys: Arc::default(),
            reading,
            limits,
            http: reqwest::Client::builder()
                .connect_timeout(FORWARD_CONNECT_TIMEOUT)
                .timeout(FORWARD_TIMEOUT)
                .build()
                .expect("an HTTP client with default settings"),
        }
    }

    /// Trust identity tokens signed by `keys`.
    pub fn trusting(self, keys: IdentityKeys) -> Self {
        Self {
            identity_keys: Arc::new(keys),
            ..self
        }
    }

    /// Report what `allocator` holds, which must be the process's global
    /// allocator, as it is the `vellev` binary's: in a process on another,
    /// jemalloc's counts say nothing of it. So it is taken only if it is
    /// seen to be ([`Allocator::is_global`]); otherwise none of its counts
    /// are reported, and that is said in the log.
    pub fn running_on(self, allocator: &'static Allocator) -> Self {
        match allocator.is_global() {
            Some(true) => {}
            Some(false) => {
                tracing::warn!(
                    "jemalloc is not this process's global allocator: its counts are not reported"
                );
                return self;
            }
            None => {
                tracing::warn!("jemalloc's statistics cannot be read: its counts are not reported");
                return self;
            }
        }
        let memory = Sources {
            allocator: Some(allocator),
            ..self.memory.clone()
        };
        self.measuring(memory)
    }

    /// Read the process's and its cgroup's memory from `files`, in place
    /// of this process's own.
    pub fn reading_memory_from(self, files: MemoryFiles) -> Self {
        let memory = Sources {
            files,
            ..self.memory.clone()
        };
        self.measuring(memory)
    }

    /// As this, its measures reading memory from `memory`: made afresh, as
    /// a node is set up, before anything is measured.
    fn measuring(self, memory: Sources) -> Self {
        Self {
            metrics: Arc::new(Metrics::new(
                self.node.clone(),
                self.reading.clone(),
                memory.clone(),
            )),
            memory,
            ..self
        }
    }

    /// Serve from `node` from now on, and have it do what a node does of
    /// its own accord: say that it runs, let idle tenants go, back tenants
    /// up and purge those removed. Only the first node given is kept.
    pub fn start(&self, node: Arc<Node>) {
        if self.serve_from(node.clone()) {
            // Before it opens any tenant, as warming would.
            self.metrics.started();
            node.announcing();
            node.warming();
            node.sweeping();
            node.backing_up();
            node.purging();
        }
    }

    /// Serve from `node` from now on, leaving what a node does of its own
    /// accord to whoever asks it to. Returns whether it is the node served
    /// from: only the first given is kept.
    pub fn serve_from(&self, node: Arc<Node>) -> bool {
        self.node.set(node).is_ok()
    }

    /// Report not ready from now on: the node is about to stop.
    pub fn stopping(&self) {
        self.stopping.store(true, Ordering::Relaxed);
    }

    /// What the node measures of the requests it answers.
    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    /// The node being served from, once there is one.
    pub fn started(&self) -> Option<&Arc<Node>> {
        self.node.get()
    }

    pub(crate) fn node(&self) -> Result<&Arc<Node>, ApiError> {
        self.started()
            .ok_or_else(|| ApiError::unavailable("the node is still starting"))
    }

    /// The most a caller may ask of the node, as it is published and as
    /// its refusals say.
    pub(crate) fn allowed(&self) -> vellev_api::Limits {
        let mut allowed = vellev_api::Limits::default();
        allowed.results = self.limits.results as u64;
        allowed.candidates = self.limits.results.max(DEFAULT_CANDIDATES) as u64;
        allowed.oversample = MAX_OVERSAMPLE.into();
        allowed.body_bytes = self.limits.body as u64;
        allowed.restore_window_secs = MAX_RESTORE_WINDOW.as_secs();
        allowed.deleted = DELETED_AT_MOST as u64;
        allowed.reads_at_once = self.reading.most() as u64;
        allowed
    }

    /// `asked` of `name`, sent as `field`, if it is from 1 to `most`. Only
    /// a request for more is told the most: one for none is not helped by
    /// it.
    fn within(name: Bounded, field: &str, asked: u32, most: u64) -> Result<usize, ApiError> {
        let message = || format!("{field} must be from 1 to {most}, not {asked}");
        match u64::from(asked) {
            0 => Err(ApiError::invalid(message())),
            more if more > most => Err(ApiError::beyond(name, most, message())),
            _ => Ok(asked as usize),
        }
    }

    /// `asked` if it is a number of results the node will return.
    pub(crate) fn limit(&self, asked: u32) -> Result<usize, ApiError> {
        Self::within(Bounded::Results, "limit", asked, self.allowed().results)
    }

    /// How many results each side of a hybrid search offers.
    pub(crate) fn candidates(&self, asked: Option<u32>) -> Result<usize, ApiError> {
        asked.map_or(Ok(DEFAULT_CANDIDATES), |asked| {
            let most = self.allowed().candidates;
            Self::within(Bounded::Candidates, "candidates", asked, most)
        })
    }

    /// The oversample of a hybrid search for `limit` results: as asked,
    /// within the most allowed; or, unasked,
    /// [`vellev_storage::default_oversample`], which keeps `limit` times it
    /// to [`vellev_storage::HYBRID_DEFAULT_RESCORE`] per shortlisting
    /// segment. A caller who asks scores as many as it asks for.
    pub(crate) fn hybrid_oversample(
        &self,
        asked: Option<u32>,
        limit: usize,
    ) -> Result<usize, ApiError> {
        match asked {
            Some(_) => self.oversample(asked),
            None => Ok(vellev_storage::default_oversample(limit)),
        }
    }

    pub(crate) fn oversample(&self, asked: Option<u32>) -> Result<usize, ApiError> {
        asked.map_or(Ok(DEFAULT_OVERSAMPLE), |asked| {
            let most = self.allowed().oversample;
            Self::within(Bounded::Oversample, "oversample", asked, most)
        })
    }

    /// Refuse a collection that would have the node call an embedding
    /// endpoint its operator has not named.
    pub(crate) fn embeds_through(&self, schema: &Collection) -> Result<(), ApiError> {
        match &schema.embedding.provider {
            EmbeddingProvider::Http { url, .. }
                if !self
                    .limits
                    .embedding_urls
                    .iter()
                    .any(|allowed| allowed == url.trim_end_matches('/')) =>
            {
                Err(ApiError::invalid(format!(
                    "{url} is not an embedding endpoint this node may call"
                )))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    version: &'static str,
}

const OK: Health = Health {
    status: "ok",
    version: env!("CARGO_PKG_VERSION"),
};

/// Liveness must never depend on Postgres, object storage or other nodes: a
/// platform that restarts on failed liveness would otherwise take every node
/// down with an outage it cannot fix.
async fn live() -> Json<Health> {
    Json(OK)
}

async fn ready(Shared(app): Shared<App>) -> Result<Json<Health>, ApiError> {
    if app.stopping.load(Ordering::Relaxed) {
        return Err(ApiError::unavailable("the node is stopping"));
    }
    app.node()?.ready().await?;
    Ok(Json(OK))
}

/// Once the node is stopping, ask callers not to send anything more on the
/// connection an answer goes back on: one closed under a request sent on
/// it as it closes leaves the caller not knowing whether it arrived.
async fn closing(Shared(app): Shared<App>, request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    if app.stopping.load(Ordering::Relaxed) {
        response.headers_mut().insert(
            header::CONNECTION,
            header::HeaderValue::from_static("close"),
        );
    }
    response
}

/// Whether two keys are the same, in time that depends on neither where
/// they differ nor how long they are.
fn same(a: &str, b: &str) -> bool {
    let (a, b) = (Sha256::digest(a.as_bytes()), Sha256::digest(b.as_bytes()));
    a.iter()
        .zip(b.iter())
        .fold(0, |differ, (x, y)| differ | (x ^ y))
        == 0
}

/// The routes only an admin key reaches: the tenants themselves, their
/// keys, backing them up and putting them back as they were. Every other
/// route that needs a key is one tenant's, or one of [`EVERY_KEY`].
pub const ADMIN_ONLY: [&str; 8] = [
    "/v1/tenants",
    "/v1/tenants/{tenant}",
    "/v1/tenants/{tenant}/keys",
    "/v1/tenants/{tenant}/keys/{id}",
    "/v1/tenants/{tenant}/restore-points",
    "/v1/tenants/{tenant}/restore-window",
    "/v1/tenants/{tenant}/restore",
    "/v1/tenants/{tenant}/backups",
];

/// The routes every key reaches, which say nothing of any tenant: what the
/// node allows every caller.
pub const EVERY_KEY: [&str; 1] = ["/v1/limits"];

/// The admin routes done where the request arrives, which need no writer:
/// a backup is read from the store. A restore is sent on to the node that
/// writes the tenant, as its writes are.
const COORDINATED: [&str; 7] = [
    "/v1/tenants",
    "/v1/tenants/{tenant}",
    "/v1/tenants/{tenant}/keys",
    "/v1/tenants/{tenant}/keys/{id}",
    "/v1/tenants/{tenant}/restore-points",
    "/v1/tenants/{tenant}/restore-window",
    "/v1/tenants/{tenant}/backups",
];

/// Let a request through if its key reaches what it asks for. An admin key,
/// from the node's configuration, reaches everything; a tenant's key, from
/// the coordinator, reaches that tenant's data alone, and another tenant is
/// to it as one that does not exist.
async fn authorized(
    Shared(app): Shared<App>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let unauthorized = || ApiError::new(Code::Unauthorized, "a valid API key is required");
    let forbidden = || ApiError::new(Code::Forbidden, "only an admin key may do this");
    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, key)| key)
        .ok_or_else(unauthorized)?;
    let admin = app
        .keys
        .iter()
        .fold(false, |known, key| known | same(key, presented));
    if admin {
        request.extensions_mut().insert(Caller::Admin);
        return Ok(next.run(request).await);
    }
    let reaches = metrics::timed(Stage::Auth, app.node()?.key_tenant(presented))
        .await?
        .ok_or_else(unauthorized)?;
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);
    let for_every_key = route.is_some_and(|route| EVERY_KEY.contains(&route));
    if route.is_none_or(|route| ADMIN_ONLY.contains(&route)) {
        return Err(forbidden());
    }
    if for_every_key {
        request.extensions_mut().insert(Caller::Tenant(reaches));
        return Ok(next.run(request).await);
    }
    // The tenant as the router read it, which is what the handler is given.
    let (mut parts, body) = request.into_parts();
    let named = RawPathParams::from_request_parts(&mut parts, &app)
        .await
        .ok()
        .and_then(|params| {
            params
                .iter()
                .find(|(name, _)| *name == "tenant")
                .map(|(_, tenant)| tenant.to_string())
        });
    match named {
        Some(tenant) if tenant == reaches.as_str() => {
            parts.extensions.insert(Caller::Tenant(reaches));
            Ok(next.run(Request::from_parts(parts, body)).await)
        }
        Some(tenant) => Err(ApiError::not_found(
            Missing::Tenant,
            format!("tenant {tenant} does not exist"),
        )),
        None => Err(forbidden()),
    }
}

/// The tenant whose data a request's path names: one with something after
/// the tenant. Creating and dropping a tenant are the coordinator's and
/// are done where the request arrives.
fn tenant_of(path: &str) -> Option<TenantId> {
    let (tenant, rest) = path.strip_prefix("/v1/tenants/")?.split_once('/')?;
    (!rest.is_empty())
        .then(|| TenantId::new(tenant).ok())
        .flatten()
}

/// The routes that POST a query rather than a change.
const QUERIES: [&str; 5] = [
    "/v1/tenants/{tenant}/collections/{collection}/nearest",
    "/v1/tenants/{tenant}/collections/{collection}/keyword",
    "/v1/tenants/{tenant}/collections/{collection}/hybrid",
    "/v1/tenants/{tenant}/collections/{collection}/list",
    "/v1/tenants/{tenant}/collections/{collection}/count",
];

/// Whether a request to the route `matched` only reads, and so is answered
/// by whichever node it reaches: every GET and HEAD, and the queries.
fn reads(method: &Method, matched: Option<&str>) -> bool {
    matches!(*method, Method::GET | Method::HEAD)
        || (*method == Method::POST && matched.is_some_and(|route| QUERIES.contains(&route)))
}

/// Whether a header is about one connection rather than the request, and
/// so is not passed on.
fn hop_by_hop(name: &HeaderName) -> bool {
    [
        header::CONNECTION,
        header::CONTENT_LENGTH,
        header::HOST,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ]
    .contains(name)
        || name.as_str() == "keep-alive"
        || name.as_str() == "proxy-connection"
}

/// Refuse a request for a tenant this node does not serve, whatever the
/// route: a read or a write of its data, its keys, backups and restores,
/// making it and dropping it. The node checks again wherever a tenant is
/// opened, read or backed up; this answers for the routes that only reach
/// the coordinator or the backup store, and comes before any request is
/// sent on, so that none for the tenant is. A caller's key is checked
/// first: one without a valid key learns nothing of where tenants are
/// served.
async fn served(
    Shared(app): Shared<App>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Ok(node) = app.node() else {
        return Ok(next.run(request).await);
    };
    // The tenant as the router read it, which is what the handler is given.
    let (mut parts, body) = request.into_parts();
    let named = RawPathParams::from_request_parts(&mut parts, &app)
        .await
        .ok()
        .and_then(|params| {
            params
                .iter()
                .find(|(name, _)| *name == "tenant")
                .and_then(|(_, tenant)| TenantId::new(tenant).ok())
        });
    match named {
        Some(tenant) if !node.serves(&tenant) => Err(NodeError::NotServedHere { tenant }.into()),
        _ => Ok(next.run(Request::from_parts(parts, body)).await),
    }
}

/// Send a request for a tenant that another node holds on to that node,
/// and answer with its answer. A request that has been sent on once goes
/// no further: the node it reached answers for itself.
async fn forwarded(
    Shared(app): Shared<App>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let matched = request.extensions().get::<MatchedPath>();
    // A tenant's keys are the coordinator's, and are changed where the
    // request arrives, so that node forgets the old ones at once.
    if reads(request.method(), matched.map(MatchedPath::as_str))
        || matched.is_some_and(|route| COORDINATED.contains(&route.as_str()))
    {
        return Ok(next.run(request).await);
    }
    let path = request.uri().path();
    let dotted = path.to_ascii_lowercase().contains("%2e")
        || path
            .split('/')
            .any(|segment| segment == "." || segment == "..");
    if dotted {
        return Err(ApiError::invalid("a path may not hold dot segments"));
    }
    let tenant = tenant_of(request.uri().path());
    let sent_on = request.headers().contains_key(FORWARDED);
    let (Some(tenant), false, Ok(node)) = (tenant, sent_on, app.node()) else {
        return Ok(next.run(request).await);
    };
    let holder = match node.tenant(&tenant).await {
        Err(NodeError::Elsewhere { holder, .. }) => holder,
        _ => return Ok(next.run(request).await),
    };
    let Some(address) = holder.address() else {
        return Ok(next.run(request).await);
    };
    // Only a request that never reached the holder certainly had no effect.
    let unreachable = |error: &reqwest::Error| {
        tracing::warn!(%tenant, %holder, %error, "the node holding the tenant could not be reached");
        let message = format!("the node holding tenant {tenant} could not be reached");
        match error.is_connect() {
            true => ApiError::unavailable(message),
            false => ApiError::new(Code::Uncertain, message),
        }
    };
    let timeout = match matched {
        Some(route) if route.as_str() == "/v1/tenants/{tenant}/restore" => RESTORE_TIMEOUT,
        _ => FORWARD_TIMEOUT,
    };
    let (parts, body) = request.into_parts();
    let body = axum::body::to_bytes(body, app.limits.body)
        .await
        .map_err(|error| {
            let message = error.to_string();
            match error.into_inner().is::<http_body_util::LengthLimitError>() {
                true => ApiError::too_large(app.allowed().body_bytes, message),
                false => {
                    ApiError::invalid(format!("the request's body could not be read: {message}"))
                }
            }
        })?;
    let url = format!(
        "{address}{}",
        parts.uri.path_and_query().map_or("", |pq| pq.as_str())
    );
    let passed_on = parts
        .headers
        .iter()
        .filter(|(name, _)| !hop_by_hop(name))
        .fold(HeaderMap::new(), |mut headers, (name, value)| {
            headers.append(name, value.clone());
            headers
        });
    let answer = app
        .http
        .request(parts.method, url)
        .timeout(timeout)
        .headers(passed_on)
        .header(FORWARDED, vellev_node::PROTOCOL.to_string())
        .body(body)
        .send()
        .await
        .map_err(|error| unreachable(&error))?;
    let status = answer.status();
    let answered = answer.headers().clone();
    let bytes = answer.bytes().await.map_err(|error| unreachable(&error))?;
    let mut response = (status, bytes).into_response();
    for (name, value) in answered.iter().filter(|(name, _)| !hop_by_hop(name)) {
        response.headers_mut().append(name, value.clone());
    }
    Ok(response)
}

#[derive(utoipa::OpenApi)]
#[openapi(
    info(title = "Vellev", description = "The vellev vector database's HTTP API."),
    components(schemas(vellev_api::Consistency)),
    modifiers(&Keyed),
    security(("key" = []))
)]
struct Described;

struct Keyed;

impl Modify for Keyed {
    fn modify(&self, openapi: &mut OpenApi) {
        let bearer = SecurityScheme::Http(Http::new(HttpAuthScheme::Bearer));
        openapi
            .components
            .get_or_insert_with(Default::default)
            .add_security_scheme("key", bearer);
    }
}

/// The routes that need a key, with the description generated from them.
fn described() -> (Router<App>, OpenApi) {
    OpenApiRouter::with_openapi(Described::openapi())
        .routes(routes!(api::limits))
        .routes(routes!(api::tenants))
        .routes(routes!(api::create_tenant, api::drop_tenant))
        .routes(routes!(api::issue_key, api::tenant_keys))
        .routes(routes!(api::restore_points))
        .routes(routes!(api::restore))
        .routes(routes!(api::set_restore_window))
        .routes(routes!(api::back_up, api::backups))
        .routes(routes!(api::revoke_key))
        .routes(routes!(api::changes))
        .routes(routes!(api::collections))
        .routes(routes!(
            api::collection,
            api::create_collection,
            api::drop_collection
        ))
        .routes(routes!(api::add_property))
        .routes(routes!(
            api::object,
            api::put_object,
            api::patch_object,
            api::delete_object
        ))
        .routes(routes!(api::nearest))
        .routes(routes!(api::keyword))
        .routes(routes!(api::hybrid))
        .routes(routes!(api::list))
        .routes(routes!(api::count))
        .split_for_parts()
}

/// The API as an OpenAPI document.
pub fn openapi() -> OpenApi {
    described().1
}

pub fn router(app: App) -> Router {
    let (keyed, description) = described();
    let description = Arc::new(description);
    let body = DefaultBodyLimit::max(app.limits.body);
    Router::new()
        .merge(
            keyed
                .route_layer(from_fn_with_state(app.clone(), forwarded))
                .route_layer(from_fn_with_state(app.clone(), served))
                .route_layer(from_fn_with_state(app.clone(), authorized))
                .route_layer(from_fn_with_state(app.clone(), metrics::timing)),
        )
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route(
            "/openapi.json",
            get(move || async move { Json(description.as_ref().clone()) }),
        )
        .fallback(|| async { ApiError::not_found(Missing::Route, "no such route") })
        .method_not_allowed_fallback(|| async {
            ApiError::refused(
                StatusCode::METHOD_NOT_ALLOWED,
                "the route has no such method",
            )
        })
        .layer(body)
        .layer(from_fn_with_state(app.clone(), closing))
        .with_state(app)
}

/// Serve until `shutdown` resolves, then finish in-flight requests.
pub async fn serve(
    listener: TcpListener,
    app: App,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(app))
        .with_graceful_shutdown(shutdown)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_queries_are_routes_that_post_and_nothing_else_that_posts_reads() {
        let described = openapi();
        let posting: Vec<&str> = described
            .paths
            .paths
            .iter()
            .filter(|(_, item)| item.post.is_some())
            .map(|(path, _)| path.as_str())
            .collect();
        assert!(QUERIES.iter().all(|query| posting.contains(query)));
        for path in posting {
            let query = QUERIES.contains(&path);
            assert_eq!(reads(&Method::POST, Some(path)), query, "{path}");
        }
        let named_count = "/v1/tenants/{tenant}/collections/{collection}";
        assert!(!reads(&Method::PUT, Some(named_count)));
        assert!(!reads(&Method::DELETE, Some(named_count)));
        assert!(reads(&Method::HEAD, Some(named_count)));
        assert!(!reads(&Method::POST, None));
    }
}
