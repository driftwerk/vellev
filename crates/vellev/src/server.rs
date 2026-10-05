use std::future::Future;
use std::sync::Arc;

use axum::extract::{Request, State as Shared};
use axum::http::header::AUTHORIZATION;
use axum::http::StatusCode;
use axum::middleware::{from_fn_with_state, Next};
use axum::response::Response;
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
use vellev_node::Node;

use crate::api;
use crate::problem::ApiError;

/// What every request is served with.
#[derive(Debug, Clone)]
pub struct App {
    pub node: Arc<Node>,
    /// The keys a caller may present; with none, nobody may call.
    pub keys: Arc<[String]>,
    /// The most results a search or a page may ask for.
    pub max_limit: usize,
}

impl App {
    /// `asked` if it is a number of results the node will return.
    pub(crate) fn limit(&self, asked: usize) -> Result<usize, ApiError> {
        if (1..=self.max_limit).contains(&asked) {
            Ok(asked)
        } else {
            Err(ApiError::invalid(format!(
                "limit must be from 1 to {}, not {asked}",
                self.max_limit
            )))
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
    app.node.ready().await?;
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
        .and_then(|value| value.strip_prefix("Bearer "));
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

#[derive(utoipa::OpenApi)]
#[openapi(
    info(title = "Vellev", description = "The vellev vector database's HTTP API."),
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
    Router::new()
        .merge(keyed.route_layer(from_fn_with_state(app.clone(), authorized)))
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route(
            "/openapi.json",
            get(move || async move { Json(description.as_ref().clone()) }),
        )
        .fallback(|| async { (StatusCode::NOT_FOUND, ApiError::not_found("no such route")) })
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
