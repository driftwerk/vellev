//! The API's routes: what each takes, does and answers.

use std::sync::Arc;

use axum::extract::{FromRequest, FromRequestParts, Path, Query, State as Shared};
use axum::http::StatusCode;
use axum::Json;
use vellev_api::{
    Batch, Change, Changes, Collections, Content, Counted, Counting, Failure, Found, Hybrid,
    Keyword, Nearest, Objects, OnlyIf, Outcome, Page, Results, Stored, TenantPage, Tenants,
    Weighted, Written,
};
use vellev_core::{Collection, CollectionName, Object, ObjectId, Property, TenantId};
use vellev_node::Tenant;
use vellev_storage::{
    Condition, Fused, Hit, Mutation, Query as VectorQuery, Receipt, Scored, State, DEFAULT_ALPHA,
};
use vellev_storage::{Hybrid as HybridQuery, Keywords};

use crate::problem::ApiError;
use crate::server::App;

type Answer<T> = Result<Json<T>, ApiError>;

/// A JSON body, refused in the API's own words when it does not fit.
#[derive(FromRequest)]
#[from_request(via(Json), rejection(ApiError))]
pub struct Body<T>(T);

/// What a route's path names, refused likewise.
#[derive(FromRequestParts)]
#[from_request(via(Path), rejection(ApiError))]
pub struct Named<T>(T);

#[derive(FromRequestParts)]
#[from_request(via(Query), rejection(ApiError))]
pub struct Asked<T>(T);

fn stored(object: &Object, vector: Option<&[f32]>, vectors: bool) -> Stored {
    Stored {
        id: object.id,
        properties: object.properties_json(),
        vector: vector.filter(|_| vectors).map(<[f32]>::to_vec),
    }
}

fn written(receipt: Receipt) -> Written {
    use vellev_storage::Outcome as Did;
    Written {
        incarnation: receipt.incarnation.0,
        position: receipt.position.0,
        outcomes: receipt
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
            .collect(),
    }
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
        } => Mutation::Upsert {
            collection,
            id,
            properties,
            vector,
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
        } => Mutation::Patch {
            collection,
            id,
            properties,
            vector,
            only_if: condition(only_if),
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
        Change::Delete { collection, id } => Mutation::Delete { collection, id },
        Change::DeleteWhere { collection, filter } => Mutation::DeleteWhere { collection, filter },
    }
}

async fn write(app: &App, tenant: &TenantId, mutations: Vec<Mutation>) -> Answer<Written> {
    mutations.iter().try_for_each(|mutation| match mutation {
        Mutation::CreateCollection(schema) => app.embeds_through(schema),
        _ => Ok(()),
    })?;
    let tenant = app.node()?.tenant(tenant).await?;
    Ok(Json(written(tenant.write(mutations).await?)))
}

/// Read the tenant off the async runtime's threads: a search can take
/// long enough to hold up other requests.
async fn read<T: Send + 'static>(
    tenant: Arc<Tenant>,
    f: impl FnOnce(&State) -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(move || tenant.read(f))
        .await
        .map_err(|error| {
            tracing::error!(%error, "a read did not finish");
            ApiError::new(
                vellev_api::Code::Internal,
                "the node failed to serve the request",
            )
        })?
}

fn weights(properties: &[Weighted]) -> Vec<(&str, f32)> {
    properties
        .iter()
        .map(|weighted| (weighted.property.as_str(), weighted.weight))
        .collect()
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
    Ok(Json(Tenants { tenants, next }))
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
        false => Err(ApiError::not_found(format!(
            "tenant {tenant} does not exist"
        ))),
    }
}

/// Apply several changes as one atomic step, each seeing those before it.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/changes",
    params(("tenant" = TenantId, Path,)),
    request_body = Batch,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn changes(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
    Body(batch): Body<Batch>,
) -> Answer<Written> {
    write(
        &app,
        &tenant,
        batch.changes.into_iter().map(mutation).collect(),
    )
    .await
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections",
    params(("tenant" = TenantId, Path,)),
    responses((status = OK, body = Collections), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn collections(
    Shared(app): Shared<App>,
    Named(tenant): Named<TenantId>,
) -> Answer<Collections> {
    let tenant = app.node()?.tenant(&tenant).await?;
    let collections = tenant.read(|state| state.collections().cloned().collect());
    Ok(Json(Collections { collections }))
}

fn schema(state: &State, collection: &CollectionName) -> Result<Collection, ApiError> {
    state
        .collection(collection)
        .cloned()
        .ok_or_else(|| ApiError::not_found(format!("collection {collection} does not exist")))
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    responses((status = OK, body = Collection), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
) -> Answer<Collection> {
    let tenant = app.node()?.tenant(&tenant).await?;
    Ok(Json(tenant.read(|state| schema(state, &collection))?))
}

#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Collection,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn create_collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(schema): Body<Collection>,
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
        vec![Mutation::CreateCollection(Box::new(schema))],
    )
    .await
}

#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}/collections/{collection}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn drop_collection(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
) -> Answer<Written> {
    write(&app, &tenant, vec![Mutation::DropCollection(collection)]).await
}

/// Adding a property that exists unchanged changes nothing.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/properties",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Property,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn add_property(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(property): Body<Property>,
) -> Answer<Written> {
    let added = Mutation::AddProperty {
        collection,
        property,
    };
    write(&app, &tenant, vec![added]).await
}

#[utoipa::path(
    get,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    responses((status = OK, body = Stored), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
) -> Answer<Stored> {
    let tenant = app.node()?.tenant(&tenant).await?;
    tenant
        .read(|state| {
            schema(state, &collection)?;
            let found = state
                .get(&collection, id)
                .map(|found| stored(found.object, found.vector, true));
            found.ok_or_else(|| {
                ApiError::not_found(format!("object {id} does not exist in {collection}"))
            })
        })
        .map(Json)
}

/// Store the object whole, replacing any stored under its id.
#[utoipa::path(
    put,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    request_body = Content,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn put_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Body(content): Body<Content>,
) -> Answer<Written> {
    let Content { properties, vector } = content;
    let upsert = Mutation::Upsert {
        collection,
        id,
        properties,
        vector,
    };
    write(&app, &tenant, vec![upsert]).await
}

/// Merge into a stored object: a property sent as null is removed.
#[utoipa::path(
    patch,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    request_body = Changes,
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn patch_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
    Body(changes): Body<Changes>,
) -> Answer<Written> {
    let Changes {
        properties,
        vector,
        only_if,
    } = changes;
    let patch = Mutation::Patch {
        collection,
        id,
        properties,
        vector,
        only_if: condition(only_if),
    };
    write(&app, &tenant, vec![patch]).await
}

/// Deleting an object that is not stored changes nothing.
#[utoipa::path(
    delete,
    path = "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,), ("id" = ObjectId, Path,)),
    responses((status = OK, body = Written), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn delete_object(
    Shared(app): Shared<App>,
    Named((tenant, collection, id)): Named<(TenantId, CollectionName, ObjectId)>,
) -> Answer<Written> {
    write(&app, &tenant, vec![Mutation::Delete { collection, id }]).await
}

fn hit(hit: Hit<'_>, vectors: bool) -> Found {
    Found {
        object: stored(hit.object, Some(hit.vector), vectors),
        score: None,
        distance: Some(hit.distance),
        certainty: hit.certainty,
        keyword: None,
    }
}

fn scored(scored: Scored<'_>, vectors: bool) -> Found {
    Found {
        object: stored(scored.object, scored.vector, vectors),
        score: None,
        distance: None,
        certainty: None,
        keyword: Some(scored.score),
    }
}

fn fused(fused: Fused<'_>, vectors: bool) -> Found {
    Found {
        object: stored(fused.object, fused.vector, vectors),
        score: Some(fused.score),
        distance: fused.distance,
        certainty: fused.certainty,
        keyword: fused.keyword,
    }
}

/// The objects nearest a vector, or nearest what a text embeds to.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/nearest",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Nearest,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn nearest(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(asked): Body<Nearest>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let oversample = app.oversample(asked.oversample)?;
    let tenant = app.node()?.tenant(&tenant).await?;
    let vector = match (asked.vector, &asked.text) {
        (Some(vector), None) => vector,
        (None, Some(text)) => tenant
            .query_vector(&collection, text)
            .await?
            .ok_or_else(|| {
                ApiError::invalid(format!(
                    "{collection} has no vector for that text; send a vector"
                ))
            })?,
        _ => return Err(ApiError::invalid("send either a vector or a text")),
    };
    let results = read(tenant, move |state| {
        let query = VectorQuery {
            vector: &vector,
            limit,
            filter: asked.filter.as_ref(),
            oversample,
        };
        let hits = state.search(&collection, &query)?;
        Ok(hits
            .into_iter()
            .map(|found| hit(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(Results {
        results,
        degraded: false,
    }))
}

/// The objects whose searchable text best matches the words of a text.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/keyword",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Keyword,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn keyword(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(asked): Body<Keyword>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let tenant = app.node()?.tenant(&tenant).await?;
    let results = read(tenant, move |state| {
        let query = Keywords {
            text: &asked.text,
            properties: &weights(&asked.properties),
            limit,
            filter: asked.filter.as_ref(),
        };
        let found = state.keyword(&collection, &query)?;
        Ok(found
            .into_iter()
            .map(|found| scored(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(Results {
        results,
        degraded: false,
    }))
}

/// The objects that best match a text by meaning and by wording together.
/// If the text cannot be embedded just now, the words alone decide.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/hybrid",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Hybrid,
    responses((status = OK, body = Results), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn hybrid(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(asked): Body<Hybrid>,
) -> Answer<Results> {
    let limit = app.limit(asked.limit)?;
    let candidates = app.candidates(asked.candidates)?;
    let oversample = app.oversample(asked.oversample)?;
    let alpha = asked.alpha.unwrap_or(DEFAULT_ALPHA);
    let tenant = app.node()?.tenant(&tenant).await?;
    let weighs_meaning = alpha > 0.0 && alpha <= 1.0;
    let (vector, degraded) = match asked.vector {
        Some(vector) => (Some(vector), false),
        None if !weighs_meaning => (None, false),
        None => match tenant.query_vector(&collection, &asked.text).await {
            Ok(vector) => (vector, false),
            Err(vellev_node::NodeError::Embed(error)) => {
                tracing::warn!(%error, "searching by words alone");
                (None, true)
            }
            Err(error) => return Err(error.into()),
        },
    };
    let results = read(tenant, move |state| {
        let query = HybridQuery {
            text: &asked.text,
            vector: vector.as_deref(),
            alpha,
            fusion: asked.fusion,
            properties: &weights(&asked.properties),
            limit,
            candidates,
            filter: asked.filter.as_ref(),
            oversample,
        };
        let found = state.hybrid(&collection, &query)?;
        Ok(found
            .into_iter()
            .map(|found| fused(found, asked.vectors))
            .collect())
    })
    .await?;
    Ok(Json(Results { results, degraded }))
}

/// One page of a collection in id order.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/list",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Page,
    responses((status = OK, body = Objects), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn list(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(page): Body<Page>,
) -> Answer<Objects> {
    let limit = app.limit(page.limit)?;
    let tenant = app.node()?.tenant(&tenant).await?;
    read(tenant, move |state| {
        let found = state.list(&collection, page.filter.as_ref(), page.after, limit + 1)?;
        let next = (found.len() > limit).then(|| found[limit - 1].id());
        let objects = found
            .into_iter()
            .take(limit)
            .map(|found| stored(found.object, found.vector, page.vectors))
            .collect();
        Ok(Json(Objects { objects, next }))
    })
    .await
}

/// How many objects a collection holds, or how many pass a filter.
#[utoipa::path(
    post,
    path = "/v1/tenants/{tenant}/collections/{collection}/count",
    params(("tenant" = TenantId, Path,), ("collection" = CollectionName, Path,)),
    request_body = Counting,
    responses((status = OK, body = Counted), (status = "4XX", body = Failure),
        (status = "5XX", body = Failure)
    )
)]
pub async fn count(
    Shared(app): Shared<App>,
    Named((tenant, collection)): Named<(TenantId, CollectionName)>,
    Body(counting): Body<Counting>,
) -> Answer<Counted> {
    let tenant = app.node()?.tenant(&tenant).await?;
    read(tenant, move |state| {
        let count = match &counting.filter {
            Some(filter) => state.select(&collection, filter)?.len(),
            None => {
                schema(state, &collection)?;
                state.count(&collection)
            }
        };
        Ok(Json(Counted {
            count: count as u64,
        }))
    })
    .await
}
