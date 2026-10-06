use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State as Shared};
use axum::http::header::{self, HeaderMap, HeaderName, AUTHORIZATION};
use axum::http::{Method, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
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
    /// The keys a caller may present; with none, nobody may call.
    keys: Arc<[String]>,
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
            keys,
            limits,
            http: reqwest::Client::builder()
                .connect_timeout(FORWARD_CONNECT_TIMEOUT)
                .timeout(FORWARD_TIMEOUT)
                .build()
                .expect("an HTTP client with default settings"),
        }
    }

    /// Serve from `node` from now on. Only the first node given is kept.
    pub fn start(&self, node: Arc<Node>) {
        if self.node.set(node.clone()).is_ok() {
            node.announcing();
        }
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
    app.node()?.ready().await?;
    Ok(Json(OK))
}

/// Whether two keys are the same, in time that does not depend on where
/// they first differ.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0, |differ, (x, y)| differ | (x ^ y))
            == 0
}

async fn authorized(
    Shared(app): Shared<App>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, key)| key);
    let known = presented.is_some_and(|presented| {
        app.keys
            .iter()
            .fold(false, |known, key| known | same(key, presented))
    });
    if known {
        Ok(next.run(request).await)
    } else {
        Err(ApiError::new(
            Code::Unauthorized,
            "a valid API key is required",
        ))
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
    if reads(request.method(), matched.map(MatchedPath::as_str)) {
        return Ok(next.run(request).await);
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
    let unreachable = |error: &dyn std::fmt::Display| {
        tracing::warn!(%tenant, %holder, %error, "the node holding the tenant could not be reached");
        ApiError::unavailable(format!(
            "the node holding tenant {tenant} could not be reached"
        ))
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
