//! Failures as the API reports them.

use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use vellev_api::{Code, Failure};
use vellev_node::NodeError;
use vellev_storage::coordinator::CoordinatorError;
use vellev_storage::StorageError;

#[derive(Debug)]
pub struct ApiError {
    code: Code,
    message: String,
}

impl ApiError {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Code::Invalid, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(Code::NotFound, message)
    }

    fn status(&self) -> StatusCode {
        match self.code {
            Code::Invalid => StatusCode::BAD_REQUEST,
            Code::Unauthorized => StatusCode::UNAUTHORIZED,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::Conflict => StatusCode::CONFLICT,
            Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let Self { code, message } = self;
        (status, Json(Failure::new(code, message))).into_response()
    }
}

/// What kind of failure a storage error is to a caller. Whatever is not
/// the caller's doing and may pass is unavailable; only what should never
/// happen is internal, and that is said without the detail.
fn code(error: &StorageError) -> Code {
    use StorageError as E;
    match error {
        E::UnknownCollection(_) | E::UnknownObject { .. } => Code::NotFound,
        E::CollectionExists(_) | E::ObjectExists { .. } | E::ConditionFailed { .. } => {
            Code::Conflict
        }
        E::InvalidMutation(_) | E::InvalidQuery(_) | E::Schema(_) | E::Filter(_) | E::Object(_) => {
            Code::Invalid
        }
        E::Coordinator(CoordinatorError::InvalidArgument(_)) => Code::Internal,
        E::Coordinator(_)
        | E::Blob { .. }
        | E::Embedding { .. }
        | E::Closed
        | E::Diverged { .. } => Code::Unavailable,
        E::Shared(shared) => code(shared),
        E::Missing { .. }
        | E::Corrupt { .. }
        | E::UnsupportedVersion { .. }
        | E::Exhausted(_)
        | E::Encode(_) => Code::Internal,
    }
}

impl From<StorageError> for ApiError {
    fn from(error: StorageError) -> Self {
        match code(&error) {
            Code::Internal => {
                tracing::error!(%error, "request failed");
                Self::new(Code::Internal, "the node failed to serve the request")
            }
            code => Self::new(code, error.root().to_string()),
        }
    }
}

impl From<NodeError> for ApiError {
    fn from(error: NodeError) -> Self {
        match error {
            NodeError::UnknownTenant(_) => Self::not_found(error.to_string()),
            NodeError::Elsewhere { .. } | NodeError::Stopped | NodeError::Embed(_) => {
                Self::new(Code::Unavailable, error.to_string())
            }
            NodeError::Storage(error) => error.into(),
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::invalid(rejection.body_text())
    }
}

impl From<PathRejection> for ApiError {
    fn from(rejection: PathRejection) -> Self {
        Self::invalid(rejection.body_text())
    }
}

impl From<QueryRejection> for ApiError {
    fn from(rejection: QueryRejection) -> Self {
        Self::invalid(rejection.body_text())
    }
}
