//! The API's routes: what each takes, does and answers.

use std::time::Duration;

use axum::extract::{FromRequest, FromRequestParts, Path, Query, State as Shared};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::Json;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value as Document;
use vellev_api::{
    After, Backup, Backups, Batch, Bounded, Change, Changes, Code, Collections, Consistency,
    Content, Counted, Counting, Failure, Found, Hybrid, IssuedKey, Keyword, Limits, Missing,
    Nearest, Objects, OnlyIf, Outcome, Page, RestorePoint, RestorePoints, RestoreTo, RestoreWindow,
    Restored, Results, Stored, TenantKey, TenantKeys, TenantPage, Tenants, Weighted, Written,
    AFTER, CONSISTENCY, REQUEST,
};
use vellev_core::{
    Collection, CollectionName, Object, ObjectId, Property, RequestId, TenantId, Version,
};
use vellev_node::{Freshness, Reader};
use vellev_storage::coordinator::{Incarnation, LogPosition, Point, MAX_RESTORE_WINDOW};
use vellev_storage::{Batch as Writes, Hybrid as HybridQuery, Keywords, Work};
use vellev_storage::{
    Condition, Fused, Hit, Mutation, Query as VectorQuery, Receipt, Scored, State, DEFAULT_ALPHA,
};

use crate::identity::Viewer;
use crate::metrics::{self, Stage};
use crate::problem::ApiError;
use crate::server::App;

type Answer<T> = Result<Json<T>, ApiError>;

/// A JSON body, refused in the API's own words when it does not fit.
pub struct Body<T>(T);

impl<T: DeserializeOwned> FromRequest<App> for Body<T> {
    type Rejection = ApiError;

    async fn from_request(request: axum::extract::Request, app: &App) -> Result<Self, ApiError> {
        let Json(read) = json(request, app).await?;
        Ok(Body(read))
    }
}

/// A JSON body, refused as too large past the most the node takes.
async fn json<T: DeserializeOwned>(
    request: axum::extract::Request,
    app: &App,
) -> Result<Json<T>, ApiError> {
    metrics::timed(Stage::Body, Json::<T>::from_request(request, app))
        .await
        .map_err(|rejection| ApiError::unreadable(rejection, app.allowed().body_bytes))
}

/// A JSON body that must hold nothing the API does not understand, at any
/// depth. Schemas tolerate fields they do not know, so that what a newer
/// release stores or answers still reads; a caller's misspelt field is
/// refused here instead. What is understood is what reading the body and
/// writing it back keeps, so a type read here may skip writing a field
/// only when it is `None`, or a field sent with its default would be
/// taken for one not understood.
pub struct Strict<T>(T);

impl<T: DeserializeOwned + Serialize> FromRequest<App> for Strict<T> {
    type Rejection = ApiError;

    async fn from_request(request: axum::extract::Request, app: &App) -> Result<Self, ApiError> {
        let Json(sent) = json::<Document>(request, app).await?;
        let read = T::deserialize(&sent).map_err(|error| ApiError::invalid(error.to_string()))?;
        let understood =
            serde_json::to_value(&read).map_err(|error| ApiError::invalid(error.to_string()))?;
        match not_understood(&sent, &understood) {
            Some(steps) => Err(ApiError::invalid(format!(
                "unknown field `{}`",
                path(&steps)
            ))),
            None => Ok(Strict(read)),
        }
    }
}

/// The first field of `sent` that reading it into `understood` dropped,
/// as the steps to it from the outside in. A null is as good as absent.
fn not_understood<'a>(sent: &'a Document, understood: &Document) -> Option<Vec<Step<'a>>> {
    let deeper = |step: Step<'a>, mut rest: Vec<Step<'a>>| {
        rest.insert(0, step);
        rest
    };
    match (sent, understood) {
        (Document::Object(sent), Document::Object(understood)) => {
            sent.iter()
                .find_map(|(name, value)| match understood.get(name) {
                    Some(kept) => {
                        not_understood(value, kept).map(|rest| deeper(Step::Field(name), rest))
                    }
                    None if value.is_null() => None,
                    None => Some(vec![Step::Field(name)]),
                })
        }
        (Document::Array(sent), Document::Array(understood)) => sent
            .iter()
            .zip(understood)
            .enumerate()
            .find_map(|(at, (value, kept))| {
                not_understood(value, kept).map(|rest| deeper(Step::Item(at), rest))
            }),
        _ => None,
    }
}

/// One step into a JSON document.
enum Step<'a> {
    Field(&'a str),
    Item(usize),
}

/// Steps written as a path, `properties[0].searchable`.
fn path(steps: &[Step<'_>]) -> String {
    steps
        .iter()
        .enumerate()
        .fold(String::new(), |path, (at, step)| match step {
            Step::Field(name) if at == 0 => format!("{path}{name}"),
            Step::Field(name) => format!("{path}.{name}"),
            Step::Item(index) => format!("{path}[{index}]"),
        })
}

/// What a route's path names, refused likewise.
#[derive(FromRequestParts)]
#[from_request(via(Path), rejection(ApiError))]
pub struct Named<T>(T);

#[derive(FromRequestParts)]
#[from_request(via(Query), rejection(ApiError))]
pub struct Asked<T>(T);

/// The headers that say how fresh a read must be.
#[derive(utoipa::IntoParams)]
#[into_params(parameter_in = Header)]
pub struct ReadHeaders {
    /// `incarnation.position`, as a write of the caller's answered: the
    /// read is answered at or past it, or refused.
    #[param(rename = "x-vellev-after")]
    after: Option<String>,
    /// `session` (the default), `strong` or `eventual`.
    #[param(rename = "x-vellev-consistency", value_type = Option<Consistency>)]
    consistency: Option<String>,
}

/// The header that names a write, so that a try of it again is answered
/// as the first was rather than made twice.
#[derive(utoipa::IntoParams)]
#[into_params(parameter_in = Header)]
pub struct WriteHeaders {
    /// A UUID naming the write, the same on every try of it. A write
    /// naming one answered in the last fifteen minutes is answered as it
    /// was then, and nothing of it is made again.
    #[param(rename = "x-vellev-request", value_type = Option<String>, format = Uuid)]
    request: Option<String>,
}

/// The request a write names in its [`WriteHeaders`], if any.
pub struct Requested(Option<RequestId>);

impl<S: Send + Sync> FromRequestParts<S> for Requested {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, ApiError> {
        let invalid = || ApiError::invalid(format!("{REQUEST} is not a UUID"));
        let request = parts
            .headers
            .get(REQUEST)
            .map(|value| value.to_str().map(str::to_string).map_err(|_| invalid()))
            .transpose()?;
        let asked = WriteHeaders { request };
        asked
            .request
            .map(|text| text.parse().map_err(|_| invalid()))
            .transpose()
            .map(Requested)
    }
}

/// How fresh a read must be, from its [`ReadHeaders`]. A session read
/// without a point to reach is answered with what the node has.
pub struct Fresh(Freshness);

impl<S: Send + Sync> FromRequestParts<S> for Fresh {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, ApiError> {
        let header = |name: &str| -> Result<Option<String>, ApiError> {
            parts
                .headers
                .get(name)
                .map(|value| {
                    value
                        .to_str()
                        .map(str::to_string)
                        .map_err(|_| ApiError::invalid(format!("{name} is not text")))
                })
                .transpose()
        };
        let asked = ReadHeaders {
            after: header(AFTER)?,
            consistency: header(CONSISTENCY)?,
        };
        let consistency = match asked.consistency {
            Some(text) => text.parse().map_err(ApiError::invalid)?,
            None => Consistency::default(),
        };
        let after = asked
            .after
            .map(|text| text.parse::<After>().map_err(ApiError::invalid))
            .transpose()?;
        Ok(Fresh(match (consistency, after) {
            (Consistency::Strong, _) => Freshness::Strong,
            (Consistency::Session, Some(after)) => {
                Freshness::After(Incarnation(after.incarnation), LogPosition(after.position))
            }
            (Consistency::Session, None) | (Consistency::Eventual, _) => Freshness::Eventual,
        }))
    }
}

/// `object` as it is answered, with its vector if `vectors` and its
/// `version` either way: what found the object says it is, of the vector
/// stored, which a row in a segment keeps apart from the object.
fn stored(object: &Object, vector: Option<&[f32]>, vectors: bool, version: Version) -> Stored {
    Stored::new(
        object.id,
        object.properties_json(),
        vector.filter(|_| vectors).map(<[f32]>::to_vec),
        version,
    )
}

fn written(receipt: Receipt) -> Written {
    use vellev_storage::Outcome as Did;
    let outcomes = receipt
        .outcomes
        .into_iter()
        .map(|outcome| match outcome {
            Did::Applied => Outcome::Applied,
            Did::Unchanged => Outcome::Unchanged,
            Did::Value(value) => Outcome::Value {
                value: value.to_json(),
            },
            Did::Deleted(count) => Outcome::Deleted { count },
        })
        .collect();
    Written::new(receipt.incarnation.0, receipt.position.0, outcomes)
}

fn condition(only_if: Option<OnlyIf>) -> Option<Condition> {
    only_if.map(|OnlyIf { property, equals }| Condition { property, equals })
}

fn mutation(change: Change) -> Mutation {
    match change {
        Change::CreateCollection { schema } => Mutation::CreateCollection(schema),
        Change::AddProperty {
            collection,
            property,
        } => Mutation::AddProperty {
            collection,
            property,
        },
        Change::DropCollection { collection } => Mutation::DropCollection(collection),
        Change::Upsert {
            collection,
            id,
            properties,
            vector,
            only_if_version,
        } => Mutation::Upsert {
            collection,
            id,
            properties,
            vector,
            only_if_version,
        },
        Change::Insert {
            collection,
            id,
            properties,
            vector,
        } => Mutation::Insert {
            collection,
            id,
            properties,
            vector,
        },
        Change::Patch {
            collection,
            id,
            properties,
            vector,
            only_if,
            only_if_version,
        } => Mutation::Patch {
            collection,
            id,
            properties,
            vector,
            only_if: condition(only_if),
            only_if_version,
        },
        Change::Increment {
            collection,
            id,
            property,
            by,
        } => Mutation::Increment {
            collection,
            id,
            property,
            by,
        },
        Change::Append {
            collection,
            id,
            property,
            values,
            unique,
        } => Mutation::Append {
            collection,
            id,
            property,
            values,
            unique,
        },
        Change::Delete {
            collection,
            id,
            only_if_version,
        } => Mutation::Delete {
            collection,
            id,
            only_if_version,
        },
        Change::DeleteWhere {
            collection,
            filter,
            limit,
        } => Mutation::DeleteWhere {
            collection,
            filter,
            limit: limit.map(|limit| limit as usize),
        },
    }
}

async fn write(
    app: &App,
    tenant: &TenantId,
    request: Option<RequestId>,
    mutations: Vec<Mutation>,
) -> Answer<Written> {
    mutations.iter().try_for_each(|mutation| match mutation {
        Mutation::CreateCollection(schema) => app.embeds_through(schema),
        _ => Ok(()),
    })?;
    let tenant = app.node()?.tenant(tenant).await?;
    let batch = Writes { request, mutations };
    Ok(Json(written(tenant.write(batch).await?)))
}

/// Read `tenant` through `reader` in a turn of the node's, off the async
/// runtime's threads: a search can take long enough to hold up other
/// requests.
///
/// The turn is taken here, with nothing left to wait for but a thread,
/// so that it is kept for the reading alone. A read turned away for want
/// of one was let through by its tenant's rate before, and the node is
/// told, which gives that back unless something costly was done for the
/// read on the way here. A read that had its turn is a query, however it
/// then ends.
async fn read<T: Send + 'static>(
    app: &App,
    tenant: &TenantId,
    reader: Reader,
    f: impl FnOnce(&State, &mut Work) -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    let turn = match app.reading(tenant).await {
        Ok(turn) => turn,
        Err(turned_away) => {
            app.node()?.unread(tenant, &reader);
            return Err(turned_away);
        }
    };
    let (read, work) = turn
        .run(move || {
            let mut work = Work::default();
            let read = reader.read(|state| f(state, &mut work));
            (read, work)
        })
        .await?;
    metrics::read_ended(&work);
    read
}

fn weights(properties: &[Weighted]) -> Vec<(&str, f32)> {
    properties
        .iter()
        .map(|weighted| (weighted.property.as_str(), weighted.weight))
        .collect()
}

/// The most a caller may ask of the node: the same for every caller, and
/// reached by every key.
#[utoipa::path(
    get,
    path = "/v1/limits",
    responses((status = OK, body = Limits), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn limits(Shared(app): Shared<App>) -> Json<Limits> {
    Json(app.allowed())
}

#[utoipa::path(
    get,
    path = "/v1/tenants",
    params(TenantPage),
    responses((status = OK, body = Tenants), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn tenants(Shared(app): Shared<App>, Asked(page): Asked<TenantPage>) -> Answer<Tenants> {
    let limit = app.limit(page.limit)?;
    let tenants = app.node()?.tenants(page.after.as_ref(), limit).await?;
    let next = tenants.last().filter(|_| tenants.len() == limit).cloned();
    Ok(Json(Tenants::new(tenants, next)))
}

/// Creating a tenant that exists changes nothing.
#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}",
    params(("tenant" = TenantId, Path,)),
    responses((status = NO_CONTENT), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn create_tenant(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Result<StatusCode, ApiError> {
    app.node()?.create_tenant(&tenant).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// How far back the tenant may be restored, and the earliest and latest
/// states it may be put back to.
#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/restore-points",
    params(("tenant" = TenantId, Path,)),
    responses((status = OK, body = RestorePoints), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn restore_points(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Answer<RestorePoints> {
    let (window, points) = app.node()?.restore_points(&tenant).await?.ok_or_else(|| {
        ApiError::not_found(Missing::Tenant, format!("tenant {tenant} does not exist"))
    })?;
    let (earliest, latest) = points.unzip();
    Ok(Json(RestorePoints::new(
        window.as_secs(),
        earliest.map(described),
        latest.map(described),
    )))
}

/// Put the tenant back as it stood at a moment or a commit within its
/// restore window, or as one of its own backups took it; a point out of
/// reach, a moment to come, or a backup that is not its own is not found.
/// To adopt a backup instead takes up again a tenant this cluster has
/// lost, from any backup kept under its name, and with the keys the backup
/// kept only if asked: as any restore, only an admin key may ask for it.
/// Nothing is lost: the restore is a commit of its own, and the states
/// before it can be gone back to as before. Writes are refused as
/// unavailable while it runs, and requests answered before it stay
/// answered. A restore goes on even if the caller stops waiting, until it
/// ends or the node lets the tenant go. One that fails or is stopped once
/// its commit was begun is looked for: it is answered as made if it was,
/// and otherwise as of unknown outcome, to be asked for again by its name.
/// One named by `x-vellev-request` is made once: asked for again, it is
/// answered as it was, waiting for it if it still runs. Another restore of
/// the tenant is refused as a conflict while one runs.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/restore",
    params(WriteHeaders, ("tenant" = TenantId, Path,)),
    request_body = RestoreTo,
    responses((status = OK, body = Restored), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn restore(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
    Requested(request): Requested,
    Body(to): Body<RestoreTo>,
) -> Answer<Restored> {
    let node = app.node()?;
    let request = request.unwrap_or_else(RequestId::generate);
    let restored = match to {
        RestoreTo::At(at) => node.restore(&tenant, Point::At(at), request).await?,
        RestoreTo::Position(position) => {
            let point = Point::Position(LogPosition(position));
            node.restore(&tenant, point, request).await?
        }
        RestoreTo::Adopt(adopt) => {
            let (backup, with_keys) = (adopt.backup, adopt.with_keys);
            node.adopt_backup(&tenant, backup, with_keys, request)
                .await?
        }
        RestoreTo::Backup(id) => node.restore_backup(&tenant, id, request).await?,
    };
    let vellev_storage::coordinator::Restored {
        restore,
        incarnation,
        position,
    } = restored;
    let from = RestorePoint::new(restore.from.0, restore.from_at);
    Ok(Json(Restored::new(from, incarnation.0, position.0)))
}

/// Back the tenant up now, whenever it was last backed up.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/backups",
    params(("tenant" = TenantId, Path,)),
    responses((status = CREATED, body = Backup), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn back_up(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Result<(StatusCode, Json<Backup>), ApiError> {
    let backup = app.node()?.back_up(&tenant).await?;
    Ok((StatusCode::CREATED, Json(kept(backup))))
}

/// The tenant's backups, oldest first.
#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/backups",
    params(("tenant" = TenantId, Path,)),
    responses((status = OK, body = Backups), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn backups(Shared(app): Shared<App>, Named(tenant): Named<TenantId>) -> Answer<Backups> {
    let backups = app.node()?.backups(&tenant).await?;
    Ok(Json(Backups::new(backups.into_iter().map(kept).collect())))
}

/// A backup as the API describes it.
fn kept(backup: vellev_storage::backups::Backup) -> Backup {
    let mut kept = Backup::new(
        backup.id,
        backup.taken_at,
        backup.position().0,
        backup.files().count(),
    );
    kept.lineage = Some(backup.lineage.0);
    kept.cluster = Some(backup.cluster.0);
    kept
}

/// Set how far back the tenant may be restored, or with none go back to
/// the default.
#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}/restore-window",
    params(("tenant" = TenantId, Path,)),
    request_body = RestoreWindow,
    responses((status = NO_CONTENT), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn set_restore_window(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
    Body(window): Body<RestoreWindow>,
) -> Result<StatusCode, ApiError> {
    let most = MAX_RESTORE_WINDOW.as_secs();
    let message = || format!("a restore window is from 1 to {most} seconds");
    match window.secs {
        Some(0) => return Err(ApiError::invalid(message())),
        Some(secs) if secs > most => {
            return Err(ApiError::beyond(
                Bounded::RestoreWindowSecs,
                most,
                message(),
            ))
        }
        _ => {}
    }
    let set = app
        .node()?
        .set_restore_window(&tenant, window.secs.map(Duration::from_secs))
        .await?;
    match set {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(ApiError::not_found(
            Missing::Tenant,
            format!("tenant {tenant} does not exist"),
        )),
    }
}

/// A restore point as the API describes it.
fn described(point: vellev_storage::coordinator::RestorePoint) -> RestorePoint {
    RestorePoint::new(point.position.0, point.at)
}

/// Issue a key that reaches this tenant alone. Only an admin key may.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/keys",
    params(("tenant" = TenantId, Path,)),
    responses((status = CREATED, body = IssuedKey), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn issue_key(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Result<(StatusCode, Json<IssuedKey>), ApiError> {
    let issued = app.node()?.issue_key(&tenant).await?.ok_or_else(|| {
        ApiError::not_found(Missing::Tenant, format!("tenant {tenant} does not exist"))
    })?;
    let issued = IssuedKey::new(issued.key.id, issued.secret, issued.key.created);
    Ok((StatusCode::CREATED, Json(issued)))
}

/// The keys that reach this tenant, without their secrets. Only an admin
/// key may list them.
#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/keys",
    params(("tenant" = TenantId, Path,)),
    responses((status = OK, body = TenantKeys), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn tenant_keys(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Answer<TenantKeys> {
    let keys = app.node()?.keys(&tenant).await?;
    let keys = keys
        .into_iter()
        .map(|key| TenantKey::new(key.id, key.created))
        .collect();
    Ok(Json(TenantKeys::new(keys)))
}

/// Revoke one of this tenant's keys. Only an admin key may.
#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}/keys/{id}",
    params(("tenant" = TenantId, Path,), ("id" = uuid::Uuid, Path,)),
    responses((status = NO_CONTENT), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn revoke_key(
    Shared(app): Shared<App>,
    Named((tenant, id)): Named<(TenantId, uuid::Uuid)>,
) -> Result<StatusCode, ApiError> {
    match app.node()?.revoke_key(&tenant, id).await? {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(ApiError::not_found(
            Missing::Key,
            format!("tenant {tenant} has no key {id}"),
        )),
    }
}

/// Forget the tenant and everything in it.
#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}",
    params(("tenant" = TenantId, Path,)),
    responses((status = NO_CONTENT), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn drop_tenant(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Result<StatusCode, ApiError> {
    match app.node()?.drop_tenant(&tenant).await? {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(ApiError::not_found(
            Missing::Tenant,
            format!("tenant {tenant} does not exist"),
        )),
    }
}

/// Apply several changes as one atomic step, each seeing those before it.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/changes",
    params(WriteHeaders, ("tenant" = TenantId, Path,)),
    request_body = Batch,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn changes(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
    Requested(request): Requested,
    Strict(batch): Strict<Batch>,
) -> Answer<Written> {
    write(
        &app,
        &tenant,
        request,
        batch.changes.into_iter().map(mutation).collect(),
    )
    .await
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections",
    params(ReadHeaders, ("tenant" = TenantId, Path,)),
    responses((status = OK, body = Collections), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn collections(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
    Fresh(fresh): Fresh,
) -> Answer<Collections> {
    let tenant = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    let collections = tenant.read(|state| state.collections().cloned().collect());
    Ok(Json(Collections::new(collections)))
}

fn schema(state: &State, collection: &CollectionName) -> Result<Collection, ApiError> {
    state.collection(collection).cloned().ok_or_else(|| {
        ApiError::not_found(
            Missing::Collection,
            format!("collection {collection} does not exist"),
        )
    })
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    responses((status = OK, body = Collection), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
) -> Answer<Collection> {
    let tenant = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    Ok(Json(tenant.read(|state| schema(state, &collection))?))
}

#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(WriteHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Collection,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn create_collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Requested(request): Requested,
    Strict(schema): Strict<Collection>,
) -> Answer<Written> {
    if schema.name != collection {
        return Err(ApiError::invalid(format!(
            "the schema is named {}, and the path {collection}",
            schema.name
        )));
    }
    write(
        &app,
        &tenant,
        request,
        vec![Mutation::CreateCollection(Box::new(schema))],
    )
    .await
}

#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(WriteHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn drop_collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Requested(request): Requested,
) -> Answer<Written> {
    write(
        &app,
        &tenant,
        request,
        vec![Mutation::DropCollection(collection)],
    )
    .await
}

/// Adding a property that exists unchanged changes nothing.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/properties",
    params(WriteHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Property,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn add_property(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Requested(request): Requested,
    Strict(property): Strict<Property>,
) -> Answer<Written> {
    let added = Mutation::AddProperty {
        collection,
        property,
    };
    write(&app, &tenant, request, vec![added]).await
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    responses((status = OK, body = Stored), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
) -> Answer<Stored> {
    let tenant = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    tenant
        .read(|state| {
            let schema = schema(state, &collection)?;
            let found = state
                .get(&collection, id)
                .filter(|found| viewer.sees(&schema, found.object))
                .map(|found| stored(found.object, found.vector, true, found.version()));
            found.ok_or_else(|| {
                ApiError::not_found(
                    Missing::Object,
                    format!("object {id} does not exist in {collection}"),
                )
            })
        })
        .map(Json)
}

/// Store the object whole, replacing any stored under its id.
#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(WriteHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    request_body = Content,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn put_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Requested(request): Requested,
    Body(content): Body<Content>,
) -> Answer<Written> {
    let Content {
        properties,
        vector,
        only_if_version,
        ..
    } = content;
    let upsert = Mutation::Upsert {
        collection,
        id,
        properties,
        vector,
        only_if_version,
    };
    write(&app, &tenant, request, vec![upsert]).await
}

/// Merge into a stored object: a property sent as null is removed.
#[utoipa::path(
    patch,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(WriteHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    request_body = Changes,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn patch_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Requested(request): Requested,
    Body(changes): Body<Changes>,
) -> Answer<Written> {
    let Changes {
        properties,
        vector,
        only_if,
        only_if_version,
        ..
    } = changes;
    let patch = Mutation::Patch {
        collection,
        id,
        properties,
        vector,
        only_if: condition(only_if),
        only_if_version,
    };
    write(&app, &tenant, request, vec![patch]).await
}

/// What a delete of one object may ask: a delete has no body.
#[derive(Debug, Default, serde::Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
#[serde(deny_unknown_fields)]
pub struct Deleting {
    /// Delete the object only if it is stored and at this version:
    /// `conflict` (409) otherwise, with the version it is at, and
    /// `not_found` (404) if it is not stored.
    #[param(value_type = Option<String>)]
    only_if_version: Option<Version>,
}

/// Deleting an object that is not stored changes nothing, unless the
/// delete is conditional on a version.
#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(WriteHeaders, Deleting, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn delete_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Requested(request): Requested,
    Asked(deleting): Asked<Deleting>,
) -> Answer<Written> {
    let delete = Mutation::Delete {
        collection,
        id,
        only_if_version: deleting.only_if_version,
    };
    write(&app, &tenant, request, vec![delete]).await
}

fn hit(hit: Hit<'_>, vectors: bool) -> Found {
    let version = hit.version();
    let mut found = Found::new(stored(hit.object, Some(hit.vector), vectors, version));
    found.distance = Some(hit.distance);
    found.certainty = hit.certainty;
    found
}

fn scored(scored: Scored<'_>, vectors: bool) -> Found {
    let version = scored.version();
    let mut found = Found::new(stored(scored.object, scored.vector, vectors, version));
    found.keyword = Some(scored.score);
    found
}

fn fused(fused: Fused<'_>, vectors: bool) -> Found {
    let version = fused.version();
    let mut found = Found::new(stored(fused.object, fused.vector, vectors, version));
    found.score = Some(fused.score);
    found.distance = fused.distance;
    found.certainty = fused.certainty;
    found.keyword = fused.keyword;
    found
}

/// The objects nearest a vector, or nearest what a text embeds to.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/nearest",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Nearest,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn nearest(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
    Body(asked): Body<Nearest>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let oversample = app.oversample(asked.oversample)?;
    let reader = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    let vector = match (asked.vector, &asked.text) {
        (Some(vector), None) => vector,
        (None, Some(text)) => metrics::timed(Stage::Embed, reader.query_vector(&collection, text))
            .await?
            .ok_or_else(|| {
                ApiError::invalid(format!(
                    "{collection} has no vector for that text; send a vector"
                ))
            })?,
        _ => return Err(ApiError::invalid("send either a vector or a text")),
    };
    let results = read(&app, &tenant, reader, move |state, work| {
        let filter = viewer.narrowed(&schema(state, &collection)?, asked.filter);
        let query = VectorQuery {
            vector: &vector,
            limit,
            filter: filter.as_ref(),
            oversample,
        };
        let hits = state.search_counted(&collection, &query, work)?;
        Ok(hits
            .into_iter()
            .map(|found| hit(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(Results::found(results)))
}

/// The objects whose searchable text best matches the words of a text.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/keyword",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Keyword,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn keyword(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
    Body(asked): Body<Keyword>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let reader = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    let results = read(&app, &tenant, reader, move |state, work| {
        let filter = viewer.narrowed(&schema(state, &collection)?, asked.filter);
        let query = Keywords {
            text: &asked.text,
            properties: &weights(&asked.properties),
            limit,
            filter: filter.as_ref(),
        };
        let found = state.keyword_counted(&collection, &query, work)?;
        Ok(found
            .into_iter()
            .map(|found| scored(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(Results::found(results)))
}

/// The objects that best match a text by meaning and by wording together.
/// If the node cannot embed the text, the words alone decide, and the
/// answer says why.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/hybrid",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Hybrid,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn hybrid(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
    Body(asked): Body<Hybrid>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let candidates = app.candidates(asked.candidates)?;
    let oversample = app.hybrid_oversample(asked.oversample, limit)?;
    let alpha = asked.alpha.unwrap_or(DEFAULT_ALPHA);
    let reader = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    let weighs_meaning = alpha > 0.0 && alpha <= 1.0;
    // Why the words alone decide, if the text could not be embedded: as
    // the search would have been refused had it not fallen back.
    let (vector, degraded_by) = match asked.vector {
        Some(vector) => (Some(vector), None),
        None if !weighs_meaning => (None, None),
        None => match metrics::timed(Stage::Embed, reader.query_vector(&collection, &asked.text))
            .await
        {
            Ok(vector) => (vector, None),
            Err(vellev_node::NodeError::Embed(error)) => {
                // A fault that stands was said where it was met, once for
                // as long as it stands; one that may pass is said here.
                let refused = match error.lasting() {
                    Some(_) => {
                        tracing::debug!(%error, "searching by words alone");
                        Code::ModelUnavailable
                    }
                    None => {
                        tracing::warn!(%error, "searching by words alone");
                        Code::Unavailable
                    }
                };
                (None, Some(refused))
            }
            Err(error) => return Err(error.into()),
        },
    };
    let results = read(&app, &tenant, reader, move |state, work| {
        let filter = viewer.narrowed(&schema(state, &collection)?, asked.filter);
        let query = HybridQuery {
            text: &asked.text,
            vector: vector.as_deref(),
            alpha,
            fusion: asked.fusion,
            properties: &weights(&asked.properties),
            limit,
            candidates,
            filter: filter.as_ref(),
            oversample,
        };
        let found = state.hybrid_counted(&collection, &query, work)?;
        Ok(found
            .into_iter()
            .map(|found| fused(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(match degraded_by {
        Some(refused) => Results::by_words(results, refused),
        None => Results::found(results),
    }))
}

/// One page of a collection in id order.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/list",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Page,
    responses((status = OK, body = Objects), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn list(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
    Body(page): Body<Page>,
) -> Answer<Objects> {
    let limit = app.limit(page.limit)?;
    let reader = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    read(&app, &tenant, reader, move |state, _work| {
        let filter = viewer.narrowed(&schema(state, &collection)?, page.filter);
        let found = state.list(&collection, filter.as_ref(), page.after, limit + 1)?;
        let next = (found.len() > limit).then(|| found[limit - 1].id());
        let objects = found
            .into_iter()
            .take(limit)
            .map(|found| stored(found.object, found.vector, page.vectors, found.version()))
            .collect();
        Ok(Json(Objects::new(objects, next)))
    })
    .await
}

/// How many objects a collection holds, or how many pass a filter.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/count",
    params(ReadHeaders, ("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Counting,
    responses((status = OK, body = Counted), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn count(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Fresh(fresh): Fresh,
    viewer: Viewer,
    Body(counting): Body<Counting>,
) -> Answer<Counted> {
    let reader = metrics::timed(Stage::Reader, app.node()?.reader(&tenant, fresh)).await?;
    read(&app, &tenant, reader, move |state, _work| {
        let filter = viewer.narrowed(&schema(state, &collection)?, counting.filter);
        let count = match &filter {
            Some(filter) => state.select(&collection, filter)?.len(),
            None => state.count(&collection),
        };
        Ok(Json(Counted::new(count as u64)))
    })
    .await
}
