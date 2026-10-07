//! Failures as the API reports them.

use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::http::header::{RETRY_AFTER, WWW_AUTHENTICATE};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use vellev_api::{Code, Failure};
use vellev_node::NodeError;
use vellev_storage::coordinator::CoordinatorError;
use vellev_storage::{Refused, StorageError};

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: Code,
    message: String,
    /// How long to wait before asking again, for a request turned away.
    retry_after: Option<std::time::Duration>,
}

impl ApiError {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        let status = match code {
            Code::Invalid => StatusCode::BAD_REQUEST,
            Code::Unauthorized => StatusCode::UNAUTHORIZED,
            Code::Forbidden => StatusCode::FORBIDDEN,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::Conflict => StatusCode::CONFLICT,
            Code::Limited => StatusCode::TOO_MANY_REQUESTS,
            Code::Unavailable | Code::Uncertain => StatusCode::SERVICE_UNAVAILABLE,
            Code::Internal | Code::Unknown => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            code,
            message: message.into(),
            retry_after: None,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Code::Invalid, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(Code::NotFound, message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(Code::Unavailable, message)
    }

    /// A request that is invalid in a way HTTP has its own status for.
    pub fn refused(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            ..Self::invalid(message)
        }
    }

    /// What is said when a collection's embedding model cannot be used.
    /// The cause goes to the log, not to the caller: it may quote whatever
    /// answered at the address the collection names.
    fn unembedded(cause: impl std::fmt::Display) -> Self {
        tracing::warn!(%cause, "embedding failed");
        Self::unavailable("the collection's embedding model could not be used just now")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let Self {
            status,
            code,
            message,
            retry_after,
        } = self;
        let body = Json(Failure::new(code, message));
        match (code, retry_after) {
            (Code::Unauthorized, _) => {
                (status, [(WWW_AUTHENTICATE, "Bearer")], body).into_response()
            }
            (_, Some(wait)) => {
                let seconds = wait.as_secs_f64().ceil().max(1.0).to_string();
                (status, [(RETRY_AFTER, seconds)], body).into_response()
            }
            (_, None) => (status, body).into_response(),
        }
    }
}

/// What kind of failure a storage error is to a caller. Whatever is not
/// the caller's doing and may pass is unavailable; only what should never
/// happen is internal, and that is said without the detail. A commit that
/// failed as a whole, rather than one caller's part of it, may or may not
/// have been recorded.
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
        E::Coordinator(CoordinatorError::Purging(_)) => Code::Conflict,
        E::Coordinator(_)
        | E::Blob { .. }
        | E::Embedding { .. }
        | E::Closed
        | E::Restoring
        | E::Diverged { .. } => Code::Unavailable,
        E::Shared(shared) if matches!(**shared, E::Restoring) => Code::Unavailable,
        E::Shared(_) => Code::Uncertain,
        E::Refused(refusal) => match refusal.kind {
            Refused::NotFound => Code::NotFound,
            Refused::Conflict => Code::Conflict,
            Refused::Invalid => Code::Invalid,
        },
        E::Missing { .. }
        | E::Corrupt { .. }
        | E::UnsupportedVersion { .. }
        | E::Exhausted(_)
        | E::Encode(_)
        | E::Background(_) => Code::Internal,
    }
}

impl From<StorageError> for ApiError {
    fn from(error: StorageError) -> Self {
        match (code(&error), error.root()) {
            (_, StorageError::Embedding { .. }) => Self::unembedded(&error),
            (Code::Internal, _) => {
                tracing::error!(%error, "request failed");
                Self::new(Code::Internal, "the node failed to serve the request")
            }
            (code, root) => Self::new(code, root.to_string()),
        }
    }
}

impl From<NodeError> for ApiError {
    fn from(error: NodeError) -> Self {
        match error {
            NodeError::UnknownTenant(_)
            | NodeError::NoRestorePoint(_)
            | NodeError::UnknownBackup(..) => Self::not_found(error.to_string()),
            NodeError::Ahead(_) | NodeError::NoBackups => Self::invalid(error.to_string()),
            NodeError::BackingUp(_) => Self::new(Code::Conflict, error.to_string()),
            NodeError::Elsewhere { .. }
            | NodeError::Outdated { .. }
            | NodeError::Stopped
            | NodeError::Busy
            | NodeError::Lost(_)
            | NodeError::Behind(_) => Self::unavailable(error.to_string()),
            NodeError::Embed(error) => Self::unembedded(error),
            NodeError::Limited { wait, .. } => Self {
                retry_after: Some(wait),
                ..Self::new(Code::Limited, error.to_string())
            },
            NodeError::Storage(error) => error.into(),
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => {
                Self::refused(StatusCode::PAYLOAD_TOO_LARGE, rejection.body_text())
            }
            _ => Self::invalid(rejection.body_text()),
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn a_commit_that_failed_as_a_whole_is_uncertain_and_one_callers_refusal_is_not() {
        let blob = || StorageError::Blob {
            key: "k".into(),
            source: "connection reset".into(),
        };
        let told = |error: StorageError| {
            let ApiError { status, code, .. } = error.into();
            (status, code)
        };
        let unavailable = (StatusCode::SERVICE_UNAVAILABLE, Code::Unavailable);
        let uncertain = (StatusCode::SERVICE_UNAVAILABLE, Code::Uncertain);
        assert_eq!(told(StorageError::Shared(Arc::new(blob()))), uncertain);
        assert_eq!(
            told(StorageError::Shared(Arc::new(StorageError::Closed))),
            uncertain
        );
        assert_eq!(told(StorageError::Closed), unavailable);
        assert_eq!(told(blob()), unavailable);
        let unembedded = || StorageError::Embedding {
            model: "m".into(),
            reason: "away".into(),
        };
        assert_eq!(told(unembedded()), unavailable);
        assert_eq!(
            told(StorageError::Shared(Arc::new(unembedded()))),
            unavailable
        );
    }
}
