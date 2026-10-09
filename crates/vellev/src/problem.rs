//! Failures as the API reports them.

use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::http::header::{RETRY_AFTER, WWW_AUTHENTICATE};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use vellev_api::{Bounded, Code, Failure, Limit, Missing, Problem};
use vellev_core::ModelFault;
use vellev_node::NodeError;
use vellev_storage::coordinator::CoordinatorError;
use vellev_storage::{Capped, Lacking, Over, Refused, StorageError};

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    problem: Problem,
    /// How long to wait before asking again, for a request turned away.
    retry_after: Option<std::time::Duration>,
}

impl ApiError {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        let status = StatusCode::from_u16(code.status()).expect("every code's status is one");
        Self {
            status,
            problem: Problem::new(code, message),
            retry_after: None,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Code::Invalid, message)
    }

    /// A request naming something that does not exist, saying what.
    pub fn not_found(missing: Missing, message: impl Into<String>) -> Self {
        let mut error = Self::new(Code::NotFound, message);
        error.problem.missing = Some(missing);
        error
    }

    /// A request that asks for more of `name` than the most the node
    /// allows, `max`.
    pub fn beyond(name: Bounded, max: u64, message: impl Into<String>) -> Self {
        let mut error = Self::invalid(message);
        error.problem.limit = Some(Limit::new(name, max));
        error
    }

    /// A request whose body is longer than the most the node takes, `max`
    /// bytes.
    pub fn too_large(max: u64, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            ..Self::beyond(Bounded::BodyBytes, max, message)
        }
    }

    /// A body that could not be read as JSON; one longer than `max` bytes
    /// is refused as too large.
    pub fn unreadable(rejection: JsonRejection, max: u64) -> Self {
        match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => Self::too_large(max, rejection.body_text()),
            _ => Self::invalid(rejection.body_text()),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(Code::Unavailable, message)
    }

    /// This failure, with word of how long to wait before asking again.
    pub fn after(self, wait: std::time::Duration) -> Self {
        Self {
            retry_after: Some(wait),
            ..self
        }
    }

    /// A request that is invalid in a way HTTP has its own status for.
    pub fn refused(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            ..Self::invalid(message)
        }
    }

    /// What is said when a collection's embedding model cannot be used:
    /// just now, or with `lasting`, by this node until that fault is put
    /// right. The cause goes to the log, not to the caller: it may quote
    /// whatever answered at the address the collection names. The caller
    /// is told only where the fault lies.
    fn unembedded(cause: impl std::fmt::Display, lasting: Option<ModelFault>) -> Self {
        match lasting {
            Some(fault) => {
                // Said at length where it is met, once while it stands;
                // here once a request, for whoever is following one.
                tracing::debug!(%cause, ?fault, "the embedding model cannot be used");
                let mut error = Self::new(
                    Code::ModelUnavailable,
                    "this node cannot use the collection's embedding model as it is set up",
                );
                error.problem.reason = Some(fault.into());
                error
            }
            None => {
                tracing::warn!(%cause, "embedding failed");
                Self::unavailable("the collection's embedding model could not be used just now")
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let Self {
            status,
            problem,
            retry_after,
        } = self;
        let code = problem.code;
        let body = Json(Failure::from(problem));
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
        E::CollectionExists(_)
        | E::ObjectExists { .. }
        | E::ConditionFailed { .. }
        | E::StaleVersion { .. } => Code::Conflict,
        E::InvalidMutation(_)
        | E::TooMany { .. }
        | E::InvalidQuery(_)
        | E::Schema(_)
        | E::Filter(_)
        | E::Object(_) => Code::Invalid,
        E::Coordinator(CoordinatorError::InvalidArgument(_)) => Code::Internal,
        E::Coordinator(CoordinatorError::Purging { .. }) => Code::Purging,
        E::Coordinator(CoordinatorError::Adopting(_) | CoordinatorError::Unadopted(_)) => {
            Code::Conflict
        }
        E::Coordinator(_)
        | E::Blob { .. }
        | E::Embedding { .. }
        | E::Closed
        | E::Restoring
        | E::Unheld(_)
        | E::Diverged { .. } => Code::Unavailable,
        E::Shared(shared) => match shared.root() {
            root if root.refusal().is_some() => code(root),
            E::Restoring | E::Embedding { .. } => Code::Unavailable,
            _ => Code::Uncertain,
        },
        E::Refused(refusal) => match refusal.kind {
            Refused::NotFound => Code::NotFound,
            Refused::Conflict => Code::Conflict,
            Refused::Invalid => Code::Invalid,
        },
        E::Missing { .. }
        | E::Corrupt { .. }
        | E::UnusableRecords { .. }
        | E::UnsupportedVersion { .. }
        | E::Exhausted(_)
        | E::Encode(_)
        | E::Background(_) => Code::Internal,
    }
}

/// What a storage error says does not exist. A refusal answered again as
/// it was says what it kept: nothing, if it was kept by a release from
/// before that was.
fn missing(error: &StorageError) -> Option<Missing> {
    let lacking = match error {
        StorageError::UnknownCollection(_) => Some(Lacking::Collection),
        StorageError::UnknownObject { .. } => Some(Lacking::Object),
        StorageError::Refused(refusal) => refusal.lacking,
        _ => None,
    };
    lacking.map(|lacking| match lacking {
        Lacking::Collection => Missing::Collection,
        Lacking::Object => Missing::Object,
    })
}

/// What a storage error says was asked too much of, and the most there
/// may be of it. A refusal answered again as it was says what it kept.
fn limit(error: &StorageError) -> Option<Limit> {
    let Over { capped, most } = match error {
        StorageError::TooMany { over, .. } => *over,
        StorageError::Refused(refusal) => refusal.over?,
        _ => return None,
    };
    let name = match capped {
        Capped::Deleted => Bounded::Deleted,
    };
    Some(Limit::new(name, most))
}

impl From<StorageError> for ApiError {
    fn from(error: StorageError) -> Self {
        match (code(&error), error.root()) {
            (_, StorageError::Embedding { lasting, .. }) => Self::unembedded(&error, *lasting),
            (Code::Internal, _) => {
                tracing::error!(%error, "request failed");
                Self::new(Code::Internal, "the node failed to serve the request")
            }
            // How long the purge has left is not known here: asked again as
            // often as a node looks for tenants to purge.
            (Code::Purging, root) => Self::new(Code::Purging, root.to_string())
                .after(vellev_node::PURGES_LOOKED_FOR_EVERY),
            (Code::NotFound, root) => {
                let mut told = Self::new(Code::NotFound, root.to_string());
                told.problem.missing = missing(root);
                told
            }
            (code, root) => {
                let mut told = Self::new(code, root.to_string());
                told.problem.limit = limit(root);
                told.problem.version = root.stored_version();
                told
            }
        }
    }
}

impl From<NodeError> for ApiError {
    fn from(error: NodeError) -> Self {
        match error {
            NodeError::UnknownTenant(_) => Self::not_found(Missing::Tenant, error.to_string()),
            NodeError::NoRestorePoint(_) => {
                Self::not_found(Missing::RestorePoint, error.to_string())
            }
            NodeError::UnknownBackup(..) => Self::not_found(Missing::Backup, error.to_string()),
            NodeError::Ahead(_) | NodeError::NoBackups | NodeError::NoSuchModel(_) => {
                Self::invalid(error.to_string())
            }
            NodeError::Empty(_) => Self::new(Code::Empty, error.to_string()),
            NodeError::BackingUp(_)
            | NodeError::Restoring(_)
            | NodeError::NotLost(_)
            | NodeError::Keyed(_) => Self::new(Code::Conflict, error.to_string()),
            NodeError::Purging { wait, .. } => {
                Self::new(Code::Purging, error.to_string()).after(wait)
            }
            NodeError::Reused { .. } => {
                Self::refused(StatusCode::UNPROCESSABLE_ENTITY, error.to_string())
            }
            NodeError::Elsewhere { .. }
            | NodeError::Outdated { .. }
            | NodeError::Stopped
            | NodeError::Busy
            | NodeError::Lost(_)
            | NodeError::ClaimLost(_)
            | NodeError::Behind(_) => Self::unavailable(error.to_string()),
            NodeError::Settings(_) => Self::new(Code::Internal, error.to_string()),
            NodeError::Embed(error) => {
                let lasting = error.lasting();
                Self::unembedded(error, lasting)
            }
            NodeError::Limited { wait, .. } => Self {
                retry_after: Some(wait),
                ..Self::new(Code::Limited, error.to_string())
            },
            NodeError::Storage(error) => error.into(),
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

    use vellev_api::Reason;
    use vellev_core::{CollectionName, ObjectId, TenantId};
    use vellev_embed::EmbedError;
    use vellev_storage::Refusal;

    use super::*;

    fn told(error: impl Into<ApiError>) -> (StatusCode, Code) {
        let ApiError {
            status, problem, ..
        } = error.into();
        (status, problem.code)
    }

    #[test]
    fn a_commit_that_failed_as_a_whole_is_uncertain_and_one_callers_refusal_is_not() {
        let blob = || StorageError::Blob {
            key: "k".into(),
            source: "connection reset".into(),
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
            lasting: None,
        };
        assert_eq!(told(unembedded()), unavailable);
        assert_eq!(
            told(StorageError::Shared(Arc::new(unembedded()))),
            unavailable
        );
        assert_eq!(
            told(StorageError::Shared(Arc::new(StorageError::Restoring))),
            unavailable
        );
    }

    #[test]
    fn a_request_reused_for_another_restore_is_unprocessable_and_final() {
        let reused = NodeError::Reused {
            tenant: vellev_core::TenantId::new("acme").unwrap(),
            request: vellev_core::RequestId::generate(),
        };
        let ApiError {
            status, problem, ..
        } = reused.into();
        assert_eq!(
            (status, problem.code),
            (StatusCode::UNPROCESSABLE_ENTITY, Code::Invalid)
        );
    }

    #[test]
    fn a_tenant_being_restored_or_backed_up_is_a_conflict_and_not_a_node_to_ask_again() {
        let acme = || vellev_core::TenantId::new("acme").unwrap();
        for busy in [NodeError::Restoring(acme()), NodeError::BackingUp(acme())] {
            let said = busy.to_string();
            let ApiError {
                status,
                problem,
                retry_after,
            } = busy.into();
            assert_eq!(
                (status, problem.code, retry_after),
                (StatusCode::CONFLICT, Code::Conflict, None),
                "{said}"
            );
            assert_eq!(problem.message, said);
        }
    }

    #[test]
    fn a_request_to_delete_too_many_says_the_most_it_may_however_it_is_told() {
        let most = Over {
            capped: Capped::Deleted,
            most: 8_192,
        };
        let said = |error: StorageError| {
            let told = ApiError::from(error);
            (told.status, told.problem.code, told.problem.limit)
        };
        let too_many = || StorageError::TooMany {
            over: most,
            message: "a delete takes a limit of 1 to 8192".into(),
        };
        let refused = (
            StatusCode::BAD_REQUEST,
            Code::Invalid,
            Some(Limit::new(Bounded::Deleted, 8_192)),
        );
        assert_eq!(said(too_many()), refused);
        // To every caller of the commit it was refused in, and to one
        // asking again by the same name, who is answered from what was
        // kept of it.
        assert_eq!(said(StorageError::Shared(Arc::new(too_many()))), refused);
        let kept = too_many().refusal().expect("the request's own doing");
        assert_eq!(kept.over, Some(most));
        assert_eq!(said(StorageError::Refused(kept.clone())), refused);
        let from_before = vellev_storage::Refusal { over: None, ..kept };
        assert_eq!(
            said(StorageError::Refused(from_before)),
            (StatusCode::BAD_REQUEST, Code::Invalid, None),
            "kept by a release from before the most was"
        );
        let other = StorageError::InvalidMutation("a delete needs a condition".into());
        assert_eq!(said(other).2, None);
    }

    #[test]
    fn a_shared_failure_that_is_a_refusal_is_told_as_that_refusal() {
        let told = |error: StorageError| ApiError::from(error).problem.code;
        let exists =
            StorageError::CollectionExists(vellev_core::CollectionName::new("notes").unwrap());
        assert_eq!(told(StorageError::Shared(Arc::new(exists))), Code::Conflict);
        let refused = StorageError::Refused(vellev_storage::Refusal {
            kind: Refused::NotFound,
            message: "object 7 does not exist in notes".into(),
            lacking: None,
            over: None,
            version: None,
        });
        let nested = StorageError::Shared(Arc::new(StorageError::Shared(Arc::new(refused))));
        assert_eq!(told(nested), Code::NotFound);
        let invalid = StorageError::InvalidMutation("no such property".into());
        assert_eq!(told(StorageError::Shared(Arc::new(invalid))), Code::Invalid);
    }

    #[test]
    fn a_model_the_node_cannot_use_is_final_and_one_failing_for_now_is_not() {
        let model_unavailable = (StatusCode::NOT_IMPLEMENTED, Code::ModelUnavailable);
        let unavailable = (StatusCode::SERVICE_UNAVAILABLE, Code::Unavailable);
        let lasting = || StorageError::Embedding {
            model: "m".into(),
            reason: "401 Unauthorized".into(),
            lasting: Some(ModelFault::Setup),
        };
        assert_eq!(told(lasting()), model_unavailable);
        assert_eq!(
            told(StorageError::Shared(Arc::new(lasting()))),
            model_unavailable,
            "not uncertain: the commit failed before it was made"
        );
        assert_eq!(told(NodeError::Storage(lasting())), model_unavailable);
        let unsupported = EmbedError::Unsupported("acme/unheard-of cannot be run locally".into());
        assert_eq!(told(NodeError::Embed(unsupported)), model_unavailable);
        let failed = EmbedError::Failed {
            model: "m".into(),
            reason: "503 Service Unavailable".into(),
        };
        assert_eq!(told(NodeError::Embed(failed)), unavailable);

        let reason = |error: ApiError| error.problem.reason;
        assert_eq!(reason(lasting().into()), Some(Reason::Setup));
        let unsupported = || EmbedError::Unsupported("acme/unheard-of cannot be run".into());
        assert_eq!(
            reason(NodeError::Embed(unsupported()).into()),
            Some(Reason::Build)
        );
        let elsewhere = EmbedError::Unusable {
            model: "m".into(),
            reason: "model_not_found".into(),
            fault: ModelFault::Provider,
        };
        assert_eq!(
            reason(NodeError::Embed(elsewhere).into()),
            Some(Reason::Provider)
        );
        let passing = EmbedError::Failed {
            model: "m".into(),
            reason: "503".into(),
        };
        assert_eq!(reason(NodeError::Embed(passing).into()), None);
        let ApiError { problem, .. } = lasting().into();
        assert_eq!(
            problem.message,
            "this node cannot use the collection's embedding model as it is set up",
            "the cause stays in the log"
        );
    }

    #[test]
    fn a_failure_to_find_says_what_is_missing_when_it_is_known() {
        let missing = |error: ApiError| (error.status, error.problem.missing);
        let tenant = TenantId::new("acme").unwrap();
        let collection = CollectionName::new("notes").unwrap();
        let not_found = StatusCode::NOT_FOUND;
        assert_eq!(
            missing(NodeError::UnknownTenant(tenant.clone()).into()),
            (not_found, Some(Missing::Tenant))
        );
        assert_eq!(
            missing(NodeError::NoRestorePoint(tenant.clone()).into()),
            (not_found, Some(Missing::RestorePoint))
        );
        assert_eq!(
            missing(NodeError::UnknownBackup(tenant, uuid::Uuid::nil()).into()),
            (not_found, Some(Missing::Backup))
        );
        let unknown = || StorageError::UnknownCollection(collection.clone());
        assert_eq!(
            missing(unknown().into()),
            (not_found, Some(Missing::Collection))
        );
        // A refusal shared with whoever sent the same request again is the
        // request's own doing, and the same however often it is told: it
        // says what is missing to each of them.
        assert_eq!(
            missing(StorageError::Shared(Arc::new(unknown())).into()),
            (not_found, Some(Missing::Collection))
        );
        let lost = CoordinatorError::backend(std::io::Error::other("connection reset"));
        assert_eq!(
            missing(StorageError::Shared(Arc::new(lost.into())).into()),
            (StatusCode::SERVICE_UNAVAILABLE, None),
            "a commit failed as a whole is uncertain, and says nothing missing"
        );
        let object = StorageError::UnknownObject {
            collection: collection.clone(),
            id: ObjectId(uuid::Uuid::nil()),
        };
        assert_eq!(missing(object.into()), (not_found, Some(Missing::Object)));
        let answered_again = |lacking| {
            StorageError::Refused(Refusal {
                kind: Refused::NotFound,
                message: format!("collection {collection} does not exist"),
                lacking,
                over: None,
                version: None,
            })
        };
        assert_eq!(
            missing(answered_again(Some(Lacking::Collection)).into()),
            (not_found, Some(Missing::Collection))
        );
        assert_eq!(
            missing(answered_again(Some(Lacking::Object)).into()),
            (not_found, Some(Missing::Object))
        );
        assert_eq!(
            missing(answered_again(None).into()),
            (not_found, None),
            "kept by a release from before what was missing was"
        );
        assert_eq!(
            missing(StorageError::Closed.into()),
            (StatusCode::SERVICE_UNAVAILABLE, None)
        );
    }
}
