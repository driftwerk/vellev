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
use vellev_api::Code;
use vellev_core::{Collection, EmbeddingProvider, TenantId};
use vellev_node::{Node, NodeError};
use vellev_storage::{DEFAULT_CANDIDATES, DEFAULT_OVERSAMPLE};

use crate::api;
use crate::identity::{Caller, IdentityKeys};
use crate::problem::ApiError;

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
const FORWARD_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

impl App {
    pub fn new(keys: Arc<[String]>, limits: Limits) -> Self {
        Self {
            node: Arc::default(),
            stopping: Arc::default(),
            keys,
            identity_keys: Arc::default(),
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

    /// Serve from `node` from now on. Only the first node given is kept.
    pub fn start(&self, node: Arc<Node>) {
        if self.node.set(node.clone()).is_ok() {
            node.announcing();
            node.sweeping();
        }
    }

    /// Report not ready from now on: the node is about to stop.
    pub fn stopping(&self) {
        self.stopping.store(true, Ordering::Relaxed);
    }

    /// The node being served from, once there is one.
    pub fn started(&self) -> Option<&Arc<Node>> {
        self.node.get()
    }

    pub(crate) fn node(&self) -> Result<&Arc<Node>, ApiError> {
        self.started()
            .ok_or_else(|| ApiError::unavailable("the node is still starting"))
    }

    fn within(name: &str, asked: u32, most: usize) -> Result<usize, ApiError> {
        let asked = asked as usize;
        if (1..=most).contains(&asked) {
            Ok(asked)
        } else {
            Err(ApiError::invalid(format!(
                "{name} must be from 1 to {most}, not {asked}"
            )))
        }
    }

    /// `asked` if it is a number of results the node will return.
    pub(crate) fn limit(&self, asked: u32) -> Result<usize, ApiError> {
        Self::within("limit", asked, self.limits.results)
    }

    /// How many results each side of a hybrid search offers.
    pub(crate) fn candidates(&self, asked: Option<u32>) -> Result<usize, ApiError> {
        let most = self.limits.results.max(DEFAULT_CANDIDATES);
        asked.map_or(Ok(DEFAULT_CANDIDATES), |asked| {
            Self::within("candidates", asked, most)
        })
    }

    pub(crate) fn oversample(&self, asked: Option<u32>) -> Result<usize, ApiError> {
        asked.map_or(Ok(DEFAULT_OVERSAMPLE), |asked| {
            Self::within("oversample", asked, MAX_OVERSAMPLE as usize)
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

/// The routes only an admin key reaches: the tenants themselves and their
/// keys. Every other route that needs a key is one tenant's.
pub const ADMIN_ONLY: [&str; 4] = [
    "/v1/tenants",
    "/v1/tenants/{tenant}",
    "/v1/tenants/{tenant}/keys",
    "/v1/tenants/{tenant}/keys/{id}",
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
    let reaches = app
        .node()?
        .key_tenant(presented)
        .await?
        .ok_or_else(unauthorized)?;
    let route = request.extensions().get::<MatchedPath>();
    if route.is_none_or(|route| ADMIN_ONLY.contains(&route.as_str())) {
        return Err(forbidden());
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
        Some(tenant) => Err(ApiError::not_found(format!(
            "tenant {tenant} does not exist"
        ))),
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
        || matched.is_some_and(|route| ADMIN_ONLY.contains(&route.as_str()))
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
    let (parts, body) = request.into_parts();
    let body = axum::body::to_bytes(body, app.limits.body)
        .await
        .map_err(|error| ApiError::refused(StatusCode::PAYLOAD_TOO_LARGE, error.to_string()))?;
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
        .routes(routes!(api::tenants))
        .routes(routes!(api::create_tenant, api::drop_tenant))
        .routes(routes!(api::issue_key, api::tenant_keys))
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
                .route_layer(from_fn_with_state(app.clone(), authorized)),
        )
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route(
            "/openapi.json",
            get(move || async move { Json(description.as_ref().clone()) }),
        )
        .fallback(|| async { ApiError::not_found("no such route") })
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
