mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

#[tokio::test]
async fn every_route_but_health_and_the_description_needs_a_key() {
    let served = Served::new().await;
    for key in [
        None,
        Some("wrong"),
        Some("test-ke"),
        Some("test-key2"),
        Some(""),
    ] {
        let (status, body) = served
            .call_as(key, Method::GET, "/v1/tenants?limit=5", None)
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{key:?}");
        assert_eq!(code(&body), "unauthorized");
        let (status, _) = served
            .call_as(key, Method::PUT, "/v1/tenants/acme", None)
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
    assert_eq!(served.get("/v1/tenants?limit=5").await.0, StatusCode::OK);
    let second = served
        .call_as(Some("second-key"), Method::GET, "/v1/tenants?limit=5", None)
        .await;
    assert_eq!(second, (StatusCode::OK, json!({ "tenants": [] })));
    let basic = served
        .http
        .get(format!("{}/v1/tenants?limit=5", served.url))
        .header("authorization", format!("Basic {KEY}"))
        .send()
        .await
        .unwrap();
    assert_eq!(basic.status(), StatusCode::UNAUTHORIZED);

    for open in ["/health/live", "/health/ready", "/openapi.json"] {
        let (status, _) = served.call_as(None, Method::GET, open, None).await;
        assert_eq!(status, StatusCode::OK, "{open}");
    }
    let (status, body) = served.get("/v1/nowhere").await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
}

#[tokio::test]
async fn tenants_are_created_listed_a_page_at_a_time_and_dropped() {
    let served = Served::new().await;
    for tenant in ["mid", "acme", "zeta", "acme"] {
        let (status, body) = served
            .call(Method::PUT, &format!("/v1/tenants/{tenant}"), None)
            .await;
        assert_eq!((status, body), (StatusCode::NO_CONTENT, Value::Null));
    }
    assert_eq!(
        served.get("/v1/tenants?limit=2").await.1,
        json!({ "tenants": ["acme", "mid"], "next": "mid" })
    );
    assert_eq!(
        served.get("/v1/tenants?limit=2&after=mid").await.1,
        json!({ "tenants": ["zeta"] })
    );
    for refused in [
        "/v1/tenants",
        "/v1/tenants?limit=0",
        "/v1/tenants?limit=51",
        "/v1/tenants?limit=many",
        "/v1/tenants?limit=5&after=Not%20A%20Tenant",
        "/v1/tenants?limit=5&before=acme",
    ] {
        let (status, body) = served.get(refused).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{refused}"
        );
    }
    let (status, body) = served
        .call(Method::PUT, "/v1/tenants/Not%20A%20Tenant", None)
        .await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));

    assert_eq!(
        served.delete("/v1/tenants/mid").await.0,
        StatusCode::NO_CONTENT
    );
    let (status, body) = served.delete("/v1/tenants/mid").await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
    let (status, body) = served.get("/v1/tenants/mid/collections").await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
    assert_eq!(
        served.get("/v1/tenants?limit=50").await.1,
        json!({ "tenants": ["acme", "zeta"] })
    );
}

#[tokio::test]
async fn collections_are_created_read_extended_and_dropped() {
    let served = Served::new().await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    assert_eq!(
        served.get("/v1/tenants/acme/collections").await.1,
        json!({ "collections": [] })
    );
    let created = served.ok(Method::PUT, NOTES, Some(notes(supplied()))).await;
    assert_eq!(
        created,
        json!({ "position": 1, "outcomes": [{ "outcome": "applied" }] })
    );

    let again = served.ok(Method::PUT, NOTES, Some(notes(supplied()))).await;
    assert_eq!(
        again,
        json!({ "position": 1, "outcomes": [{ "outcome": "unchanged" }] })
    );
    let mut resized = notes(supplied());
    resized["embedding"]["dimensions"] = json!(3);
    let (status, body) = served.put(NOTES, resized).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    let mut misspelt = notes(supplied());
    misspelt["properties"][0]["searchble"] = json!(true);
    let (status, body) = served.put(NOTES, misspelt).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    let (status, body) = served
        .put("/v1/tenants/acme/collections/other", notes(supplied()))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    let mut unembeddable = notes(supplied());
    unembeddable["embedding"]["dimensions"] = json!(0);
    let (status, body) = served.put(NOTES, unembeddable).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));

    let (status, schema) = served.get(NOTES).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        (&schema["name"], &schema["embedding"]["dimensions"]),
        (&json!("notes"), &json!(2))
    );
    let listed = served.get("/v1/tenants/acme/collections").await.1;
    assert_eq!(listed["collections"], json!([schema]));

    let reviewed = json!({ "name": "reviewed", "data_type": "bool" });
    let added = served
        .ok(
            Method::POST,
            &format!("{NOTES}/properties"),
            Some(reviewed.clone()),
        )
        .await;
    assert_eq!(added["outcomes"], json!([{ "outcome": "applied" }]));
    let again = served
        .ok(Method::POST, &format!("{NOTES}/properties"), Some(reviewed))
        .await;
    assert_eq!(again["outcomes"], json!([{ "outcome": "unchanged" }]));
    let retyped = json!({ "name": "reviewed", "data_type": "int" });
    let (status, body) = served.post(&format!("{NOTES}/properties"), retyped).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    assert_eq!(
        served.get(NOTES).await.1["properties"]
            .as_array()
            .unwrap()
            .len(),
        4
    );

    let dropped = served.ok(Method::DELETE, NOTES, None).await;
    assert_eq!(dropped["outcomes"], json!([{ "outcome": "applied" }]));
    for (method, path) in [
        (Method::GET, NOTES.to_string()),
        (Method::DELETE, NOTES.to_string()),
        (Method::GET, object(1)),
    ] {
        let (status, body) = served.call(method.clone(), &path, None).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::NOT_FOUND, "not_found"),
            "{method} {path}"
        );
    }
    let (status, body) = served.get("/v1/tenants/nobody/collections/notes").await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
}

#[tokio::test]
async fn an_object_is_stored_read_patched_and_deleted() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let fox = json!({ "properties": { "title": "A red fox", "n": 1 }, "vector": [1.0, 0.0] });
    let stored = served.ok(Method::PUT, &object(1), Some(fox)).await;
    assert_eq!(
        stored,
        json!({ "position": 2, "outcomes": [{ "outcome": "applied" }] })
    );
    assert_eq!(
        served.get(&object(1)).await.1,
        json!({ "id": id(1), "properties": { "title": "A red fox", "n": 1 }, "vector": [1.0, 0.0] })
    );

    let patched = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "n": 2, "title": null, "tags": ["den"] } })),
        )
        .await;
    assert_eq!(patched["position"], 3);
    assert_eq!(
        served.get(&object(1)).await.1["properties"],
        json!({ "n": 2, "tags": ["den"] })
    );
    let guarded =
        |n: i64| json!({ "properties": { "n": 9 }, "only_if": { "property": "n", "equals": n } });
    let (status, body) = served.patch(&object(1), guarded(1)).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(served.patch(&object(1), guarded(2)).await.0, StatusCode::OK);
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 9);
    let (status, body) = served
        .patch(&object(7), json!({ "properties": { "n": 1 } }))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));

    for refused in [
        json!({ "properties": { "n": "one" } }),
        json!({ "properties": { "undeclared": 1 } }),
        json!({ "properties": {}, "vector": [1.0, 0.0, 0.0] }),
        json!({ "properties": {}, "vectr": [1.0, 0.0] }),
        json!({ "properties": [] }),
        json!("not an object"),
    ] {
        let (status, body) = served.put(&object(2), refused.clone()).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{refused}"
        );
    }
    let unparsed = served
        .http
        .put(format!("{}{}", served.url, object(2)))
        .bearer_auth(KEY)
        .header("content-type", "application/json")
        .body("{ not json")
        .send()
        .await
        .unwrap();
    assert_eq!(unparsed.status(), StatusCode::BAD_REQUEST);
    assert_eq!(code(&unparsed.json::<Value>().await.unwrap()), "invalid");
    let (status, body) = served
        .put(&format!("{NOTES}/objects/not-an-id"), json!({}))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    assert_eq!(served.get(&object(2)).await.0, StatusCode::NOT_FOUND);

    let deleted = served.ok(Method::DELETE, &object(1), None).await;
    assert_eq!(deleted["outcomes"], json!([{ "outcome": "applied" }]));
    let again = served.ok(Method::DELETE, &object(1), None).await;
    assert_eq!(again["outcomes"], json!([{ "outcome": "unchanged" }]));
    assert_eq!(again["position"], deleted["position"]);
    let (status, body) = served.get(&object(1)).await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
}

#[tokio::test]
async fn changes_sent_together_are_applied_together_or_not_at_all() {
    let served = Served::new().await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let change = |op: &str, n: u64, rest: Value| {
        let mut change = json!({ "op": op, "collection": "notes", "id": id(n) });
        change
            .as_object_mut()
            .unwrap()
            .extend(rest.as_object().unwrap().clone());
        change
    };
    let batch = json!({ "changes": [
        { "op": "create_collection", "schema": notes(supplied()) },
        { "op": "add_property", "collection": "notes", "property": { "name": "score", "data_type": "number" } },
        change("upsert", 1, json!({ "properties": { "title": "fox", "n": 1 } })),
        change("insert", 2, json!({ "properties": { "title": "hen", "n": 5 }, "vector": [0.0, 1.0] })),
        change("patch", 1, json!({ "properties": { "score": 0.5 }, "only_if": { "property": "n", "equals": 1 } })),
        change("increment", 1, json!({ "property": "n", "by": 2 })),
        change("append", 1, json!({ "property": "tags", "values": ["a", "b", "a"], "unique": true })),
        change("delete", 9, json!({})),
        { "op": "delete_where", "collection": "notes", "filter": { "gte": { "property": "n", "value": 5 } } },
    ] });
    let written = served
        .ok(Method::POST, "/v1/tenants/acme/changes", Some(batch))
        .await;
    assert_eq!(written["position"], 1);
    assert_eq!(
        written["outcomes"],
        json!([
            { "outcome": "applied" },
            { "outcome": "applied" },
            { "outcome": "applied" },
            { "outcome": "applied" },
            { "outcome": "applied" },
            { "outcome": "value", "value": 3 },
            { "outcome": "applied" },
            { "outcome": "unchanged" },
            { "outcome": "deleted", "count": 1 },
        ])
    );
    assert_eq!(
        served.get(&object(1)).await.1["properties"],
        json!({ "title": "fox", "n": 3, "score": 0.5, "tags": ["a", "b"] })
    );
    assert_eq!(served.get(&object(2)).await.0, StatusCode::NOT_FOUND);

    let undone = json!({ "changes": [
        change("upsert", 3, json!({ "properties": { "title": "owl" } })),
        change("insert", 1, json!({ "properties": { "title": "taken" } })),
    ] });
    let (status, body) = served.post("/v1/tenants/acme/changes", undone).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(served.get(&object(3)).await.0, StatusCode::NOT_FOUND);

    for (refused, expected) in [
        (
            json!({ "changes": [{ "op": "explode" }] }),
            StatusCode::BAD_REQUEST,
        ),
        (json!({ "mutations": [] }), StatusCode::BAD_REQUEST),
        (
            json!({ "changes": [{ "op": "delete_where", "collection": "notes", "filter": { "and": [] } }] }),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "changes": [{ "op": "drop_collection", "collection": "nope" }] }),
            StatusCode::NOT_FOUND,
        ),
    ] {
        let (status, _) = served
            .post("/v1/tenants/acme/changes", refused.clone())
            .await;
        assert_eq!(status, expected, "{refused}");
    }
    let nothing = served
        .ok(
            Method::POST,
            "/v1/tenants/acme/changes",
            Some(json!({ "changes": [] })),
        )
        .await;
    assert_eq!(nothing, json!({ "position": 1, "outcomes": [] }));
}

async fn four_notes(served: &Served) {
    served.with_notes(supplied()).await;
    for (n, title, vector) in [
        (1, "a red fox", json!([1.0, 0.0])),
        (2, "fox fox den", json!([0.0, 1.0])),
        (3, "a hen", json!([-1.0, 0.0])),
        (4, "the fox and the hen", Value::Null),
    ] {
        let mut content = json!({ "properties": { "title": title, "n": n } });
        if !vector.is_null() {
            content["vector"] = vector;
        }
        served.ok(Method::PUT, &object(n), Some(content)).await;
    }
}

fn ids(results: &Value) -> Vec<String> {
    results["results"]
        .as_array()
        .unwrap_or_else(|| panic!("not results: {results}"))
        .iter()
        .map(|found| found["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn the_three_searches_rank_and_score_and_obey_filters_and_limits() {
    let served = Served::new().await;
    four_notes(&served).await;
    let search = |kind: &'static str, body: Value| {
        let served = &served;
        async move {
            served
                .ok(Method::POST, &format!("{NOTES}/{kind}"), Some(body))
                .await
        }
    };

    let nearest = search("nearest", json!({ "vector": [1.0, 0.0], "limit": 10 })).await;
    assert_eq!(ids(&nearest), [id(1), id(2), id(3)]);
    assert_eq!(
        nearest["results"][0],
        json!({ "id": id(1), "properties": { "title": "a red fox", "n": 1 }, "distance": 0.0, "certainty": 1.0 })
    );
    let with_vectors = search(
        "nearest",
        json!({ "vector": [1.0, 0.0], "limit": 1, "vectors": true }),
    )
    .await;
    assert_eq!(with_vectors["results"][0]["vector"], json!([1.0, 0.0]));
    let low = json!({ "lte": { "property": "n", "value": 2 } });
    let filtered = search(
        "nearest",
        json!({ "vector": [-1.0, 0.0], "limit": 10, "filter": low }),
    )
    .await;
    assert_eq!(ids(&filtered), [id(2), id(1)]);

    let keyword = search("keyword", json!({ "text": "the FOX", "limit": 2 })).await;
    assert_eq!(ids(&keyword).len(), 2);
    assert!(ids(&keyword).contains(&id(2)));
    let top = &keyword["results"][0];
    assert!(top["keyword"].as_f64().unwrap() > 0.0);
    assert!(
        top.get("distance").is_none() && top.get("score").is_none() && top.get("vector").is_none()
    );
    let weighted = json!({ "text": "hen", "limit": 10, "properties": [{ "property": "title", "weight": 2.0 }] });
    assert_eq!(ids(&search("keyword", weighted).await).len(), 2);

    let hybrid = search(
        "hybrid",
        json!({ "text": "fox", "vector": [1.0, 0.0], "limit": 10 }),
    )
    .await;
    assert_eq!(ids(&hybrid)[0], id(1));
    assert_eq!(ids(&hybrid).len(), 4);
    let best = &hybrid["results"][0];
    assert!(best["score"].as_f64().unwrap() > 0.75);
    assert!(best["keyword"].is_number() && best["distance"] == 0.0);
    let words = search("hybrid", json!({ "text": "fox", "limit": 10 })).await;
    assert_eq!(
        ids(&words),
        ids(&search("keyword", json!({ "text": "fox", "limit": 10 })).await)
    );
    let ranked = json!({ "text": "fox", "vector": [1.0, 0.0], "limit": 1, "fusion": "ranked", "alpha": 0.9, "candidates": 2 });
    assert_eq!(ids(&search("hybrid", ranked).await), [id(1)]);

    for (kind, refused) in [
        ("nearest", json!({ "vector": [1.0, 0.0] })),
        ("nearest", json!({ "vector": [1.0, 0.0], "limit": 0 })),
        ("nearest", json!({ "vector": [1.0, 0.0], "limit": 51 })),
        ("nearest", json!({ "vector": [1.0], "limit": 5 })),
        ("nearest", json!({ "limit": 5 })),
        (
            "nearest",
            json!({ "vector": [1.0, 0.0], "text": "fox", "limit": 5 }),
        ),
        ("nearest", json!({ "text": "fox", "limit": 5 })),
        (
            "nearest",
            json!({ "vector": [1.0, 0.0], "limit": 5, "filter": { "eq": { "property": "nope", "value": 1 } } }),
        ),
        ("keyword", json!({ "text": "fox" })),
        (
            "keyword",
            json!({ "text": "fox", "limit": 5, "properties": [{ "property": "n", "weight": 1.0 }] }),
        ),
        ("hybrid", json!({ "text": "fox", "limit": 99 })),
        ("hybrid", json!({ "text": "fox", "limit": 5, "alpha": 1.5 })),
        (
            "hybrid",
            json!({ "text": "fox", "limit": 5, "fusion": "magic" }),
        ),
    ] {
        let (status, body) = served
            .post(&format!("{NOTES}/{kind}"), refused.clone())
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{kind} {refused}"
        );
    }
    for (kind, body) in [
        ("nearest", json!({ "vector": [1.0, 0.0], "limit": 5 })),
        ("keyword", json!({ "text": "fox", "limit": 5 })),
        ("hybrid", json!({ "text": "fox", "limit": 5 })),
    ] {
        let unknown = format!("/v1/tenants/acme/collections/nope/{kind}");
        let (status, body) = served.post(&unknown, body).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::NOT_FOUND, "not_found"),
            "{kind}"
        );
    }
}

#[tokio::test]
async fn a_collection_is_listed_a_page_at_a_time_and_counted() {
    let served = Served::new().await;
    four_notes(&served).await;
    let list = |body: Value| {
        let served = &served;
        async move {
            served
                .ok(Method::POST, &format!("{NOTES}/list"), Some(body))
                .await
        }
    };
    let first = list(json!({ "limit": 3 })).await;
    let listed: Vec<&Value> = first["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| &o["id"])
        .collect();
    assert_eq!(listed, [&json!(id(1)), &json!(id(2)), &json!(id(3))]);
    assert_eq!(first["next"], json!(id(3)));
    assert!(first["objects"][0].get("vector").is_none());
    let rest = list(json!({ "limit": 3, "after": first["next"], "vectors": true })).await;
    assert_eq!(
        rest,
        json!({ "objects": [{ "id": id(4), "properties": { "title": "the fox and the hen", "n": 4 } }] })
    );
    let exact = list(json!({ "limit": 4 })).await;
    assert_eq!(exact["objects"].as_array().unwrap().len(), 4);
    assert!(exact.get("next").is_none());
    let even = json!({ "or": [
        { "eq": { "property": "n", "value": 2 } },
        { "eq": { "property": "n", "value": 4 } },
    ] });
    let filtered = list(json!({ "limit": 1, "filter": even.clone(), "vectors": true })).await;
    assert_eq!(filtered["objects"][0]["vector"], json!([0.0, 1.0]));
    assert_eq!(filtered["next"], json!(id(2)));

    let count = |body: Value| {
        let served = &served;
        async move {
            served
                .ok(Method::POST, &format!("{NOTES}/count"), Some(body))
                .await
        }
    };
    assert_eq!(count(json!({})).await, json!({ "count": 4 }));
    assert_eq!(
        count(json!({ "filter": even })).await,
        json!({ "count": 2 })
    );

    for (kind, refused) in [
        ("list", json!({})),
        ("list", json!({ "limit": 51 })),
        ("list", json!({ "limit": 5, "after": "not-an-id" })),
        (
            "count",
            json!({ "filter": { "eq": { "property": "nope", "value": 1 } } }),
        ),
        ("count", json!({ "limit": 5 })),
    ] {
        let (status, body) = served
            .post(&format!("{NOTES}/{kind}"), refused.clone())
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{kind} {refused}"
        );
    }
    for (kind, body) in [("list", json!({ "limit": 5 })), ("count", json!({}))] {
        let unknown = format!("/v1/tenants/acme/collections/nope/{kind}");
        let (status, body) = served.post(&unknown, body).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::NOT_FOUND, "not_found"),
            "{kind}"
        );
    }
}

#[tokio::test]
async fn a_tenant_written_by_a_node_that_cannot_be_reached_is_unavailable_here() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.serve_advertising("").await, cluster.serve().await);
    first.with_notes(supplied()).await;
    for path in ["/v1/tenants/acme/collections", NOTES] {
        let (status, body) = second.get(path).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::SERVICE_UNAVAILABLE, "unavailable"),
            "{path}"
        );
    }
    assert_eq!(
        second.get("/v1/tenants?limit=5").await.1,
        json!({ "tenants": ["acme"] })
    );
    first.node.shutdown().await;
    assert_eq!(second.get(NOTES).await.0, StatusCode::OK);
}

#[tokio::test]
async fn the_description_covers_every_route_that_needs_a_key() {
    let served = Served::new().await;
    let (status, described) = served
        .call_as(None, Method::GET, "/openapi.json", None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(described["info"]["title"], "Vellev");
    assert_eq!(
        described["components"]["securitySchemes"]["key"],
        json!({ "type": "http", "scheme": "bearer" })
    );
    let operations: Vec<(String, String)> = described["paths"]
        .as_object()
        .unwrap()
        .iter()
        .flat_map(|(path, methods)| {
            methods
                .as_object()
                .unwrap()
                .keys()
                .map(move |method| (method.to_uppercase(), path.clone()))
        })
        .collect();
    assert_eq!(operations.len(), 18, "{operations:?}");

    served.with_notes(supplied()).await;
    for (method, path) in &operations {
        let concrete = path
            .replace("{tenant}", "acme")
            .replace("{collection}", "notes")
            .replace("{id}", &id(1));
        let method = Method::from_bytes(method.as_bytes()).unwrap();
        let (status, body) = served.call(method.clone(), &concrete, None).await;
        assert_ne!(
            body["error"]["message"], "no such route",
            "{method} {concrete}"
        );
        assert_ne!(
            status,
            StatusCode::METHOD_NOT_ALLOWED,
            "{method} {concrete}"
        );
        let (unkeyed, _) = served.call_as(None, method.clone(), &concrete, None).await;
        assert_eq!(unkeyed, StatusCode::UNAUTHORIZED, "{method} {concrete}");
        let operation = &described["paths"][path][method.as_str().to_lowercase()];
        assert!(
            operation["responses"]
                .as_object()
                .is_some_and(|r| !r.is_empty()),
            "{method} {path}"
        );
    }
    let listing = &described["paths"]["/v1/tenants"]["get"];
    let located: Vec<(&Value, &Value)> = listing["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|parameter| (&parameter["name"], &parameter["in"]))
        .collect();
    assert_eq!(
        located,
        [
            (&json!("after"), &json!("query")),
            (&json!("limit"), &json!("query"))
        ]
    );
    assert_eq!(listing["parameters"][1]["schema"]["minimum"], 1);
    assert_eq!(listing["parameters"][0]["required"], false);
    for status in ["200", "4XX", "5XX"] {
        assert!(listing["responses"].get(status).is_some(), "{status}");
    }
    assert_eq!(
        described["components"]["schemas"]["Nearest"]["properties"]["limit"]["minimum"],
        1
    );
    let schemas = described["components"]["schemas"].as_object().unwrap();
    for named in [
        "Collection",
        "Filter",
        "Change",
        "Hybrid",
        "Found",
        "Failure",
        "Written",
    ] {
        assert!(schemas.contains_key(named), "{named} is not described");
    }
}

/// A stand-in embedding server, which can be made to fail: a text's
/// vector is its length and its number of words.
fn calls() -> &'static std::sync::atomic::AtomicUsize {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    &CALLS
}

async fn embedding_server(down: std::sync::Arc<std::sync::atomic::AtomicBool>) -> String {
    use axum::extract::State;
    use std::sync::atomic::Ordering;
    let embed = |State(down): State<std::sync::Arc<std::sync::atomic::AtomicBool>>,
                 axum::Json(body): axum::Json<Value>| async move {
        calls().fetch_add(1, Ordering::SeqCst);
        if down.load(Ordering::SeqCst) {
            return Err((axum::http::StatusCode::BAD_REQUEST, "INTERNAL-ONLY detail"));
        }
        let vectors: Vec<[usize; 2]> = body["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| text.as_str().unwrap())
            .map(|text| [text.len(), text.split_whitespace().count()])
            .collect();
        Ok(axum::Json(json!(vectors)))
    };
    let router = axum::Router::new()
        .route("/embed", axum::routing::post(embed))
        .with_state(down);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

#[tokio::test]
async fn a_collection_that_embeds_takes_text_and_falls_back_to_words_when_it_cannot() {
    let down = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let url = embedding_server(down.clone()).await;
    let served = Cluster::new()
        .serve_embedding_through(std::slice::from_ref(&url))
        .await;
    let elsewhere = json!({ "provider": "http", "url": "http://169.254.169.254/latest", "model": "m", "dimensions": 2 });
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let (status, body) = served.put(NOTES, notes(elsewhere.clone())).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    let batched = json!({ "changes": [{ "op": "create_collection", "schema": notes(elsewhere) }] });
    let (status, body) = served.post("/v1/tenants/acme/changes", batched).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    let embedding = json!({
        "provider": "http",
        "url": format!("{url}/"),
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    served.with_notes(embedding).await;
    for (n, title) in [(1, "fox"), (2, "a red fox in a den")] {
        let content = json!({ "properties": { "title": title, "n": n } });
        served.ok(Method::PUT, &object(n), Some(content)).await;
    }
    assert_eq!(served.get(&object(1)).await.1["vector"], json!([3.0, 1.0]));
    assert_eq!(served.get(&object(2)).await.1["vector"], json!([18.0, 6.0]));

    let nearest = served
        .ok(
            Method::POST,
            &format!("{NOTES}/nearest"),
            Some(json!({ "text": "fox", "limit": 1 })),
        )
        .await;
    assert_eq!(ids(&nearest), [id(1)]);
    let hybrid = json!({ "text": "den", "limit": 5 });
    let fused = served
        .ok(
            Method::POST,
            &format!("{NOTES}/hybrid"),
            Some(hybrid.clone()),
        )
        .await;
    assert_eq!(fused["results"].as_array().unwrap().len(), 2);
    assert!(fused["results"][0]["distance"].is_number());
    assert!(fused.get("degraded").is_none());
    let before = calls().load(std::sync::atomic::Ordering::SeqCst);
    let words_only = json!({ "text": "never embedded", "limit": 5, "alpha": 0.0 });
    served
        .ok(Method::POST, &format!("{NOTES}/hybrid"), Some(words_only))
        .await;
    assert_eq!(
        calls().load(std::sync::atomic::Ordering::SeqCst),
        before,
        "no weight, no embedding"
    );

    down.store(true, std::sync::atomic::Ordering::SeqCst);
    let by_words = served
        .ok(
            Method::POST,
            &format!("{NOTES}/hybrid"),
            Some(json!({ "text": "red", "limit": 5 })),
        )
        .await;
    assert_eq!(ids(&by_words), [id(2)]);
    assert!(by_words["results"][0].get("distance").is_none());
    assert_eq!(by_words["degraded"], true);
    let (status, body) = served
        .post(
            &format!("{NOTES}/nearest"),
            json!({ "text": "hen", "limit": 1 }),
        )
        .await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    let unembedded = json!({ "properties": { "title": "an owl" } });
    let (status, body) = served.put(&object(3), unembedded.clone()).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    assert!(!body.to_string().contains("INTERNAL-ONLY"), "{body}");
    let supplied = json!({ "properties": { "title": "an owl" }, "vector": [1.0, 1.0] });
    assert_eq!(served.put(&object(3), supplied).await.0, StatusCode::OK);
    let cached = served
        .ok(Method::POST, &format!("{NOTES}/hybrid"), Some(hybrid))
        .await;
    assert!(
        cached["results"][0]["distance"].is_number(),
        "a query embedded before is remembered"
    );
}

#[tokio::test]
async fn what_a_caller_may_ask_is_bounded_and_refusals_keep_their_shape() {
    let served = Served::new().await;
    four_notes(&served).await;
    for (kind, refused) in [
        (
            "hybrid",
            json!({ "text": "fox", "limit": 1, "candidates": 0 }),
        ),
        (
            "hybrid",
            json!({ "text": "fox", "limit": 1, "candidates": 101 }),
        ),
        (
            "hybrid",
            json!({ "text": "fox", "limit": 1, "candidates": 4294967295u32 }),
        ),
        (
            "hybrid",
            json!({ "text": "fox", "limit": 1, "candidates": 18446744073709551615u64 }),
        ),
        (
            "hybrid",
            json!({ "text": "fox", "limit": 1, "oversample": 1025 }),
        ),
        (
            "nearest",
            json!({ "vector": [1.0, 0.0], "limit": 1, "oversample": 0 }),
        ),
        (
            "nearest",
            json!({ "vector": [1.0, 0.0], "limit": 4294967296u64 }),
        ),
    ] {
        let (status, body) = served
            .post(&format!("{NOTES}/{kind}"), refused.clone())
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{kind} {refused}"
        );
    }
    let allowed = json!({ "text": "fox", "limit": 1, "candidates": 100, "oversample": 1024 });
    assert_eq!(
        served.post(&format!("{NOTES}/hybrid"), allowed).await.0,
        StatusCode::OK
    );

    let large = json!({ "properties": { "title": "x".repeat(MAX_BODY) } });
    let (status, body) = served.put(&object(9), large).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::PAYLOAD_TOO_LARGE, "invalid")
    );

    let (status, body) = served.call(Method::POST, "/v1/tenants/acme", None).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::METHOD_NOT_ALLOWED, "invalid")
    );

    let unkeyed = served
        .http
        .get(format!("{}/v1/tenants?limit=5", served.url))
        .send()
        .await
        .unwrap();
    assert_eq!(unkeyed.headers()["www-authenticate"], "Bearer");
    let lowercase = served
        .http
        .get(format!("{}/v1/tenants?limit=5", served.url))
        .header("authorization", format!("bearer {KEY}"))
        .send()
        .await
        .unwrap();
    assert_eq!(lowercase.status(), StatusCode::OK);
}

#[tokio::test]
async fn a_node_still_starting_is_live_but_not_ready_and_serves_nothing_yet() {
    let served = Cluster::new().unstarted(&[], None).await;
    let open = |path: &'static str| served.call_as(None, Method::GET, path, None);
    assert_eq!(open("/health/live").await.0, StatusCode::OK);
    let (status, body) = open("/health/ready").await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    let (status, body) = served.get("/v1/tenants?limit=5").await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    assert_eq!(
        open("/v1/tenants?limit=5").await.0,
        StatusCode::UNAUTHORIZED
    );

    served.app.start(served.node.clone());
    assert_eq!(open("/health/ready").await.0, StatusCode::OK);
    assert_eq!(served.get("/v1/tenants?limit=5").await.0, StatusCode::OK);
}

#[tokio::test]
async fn a_request_for_a_tenant_another_node_holds_is_sent_on_to_it() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.serve().await, cluster.serve().await);
    first.with_notes(supplied()).await;
    let fox = json!({ "properties": { "title": "a red fox", "n": 1 }, "vector": [1.0, 0.0] });
    let written = second.ok(Method::PUT, &object(1), Some(fox)).await;
    assert_eq!(written["position"], 2);
    assert_eq!(first.get(&object(1)).await.1["properties"]["n"], 1);
    assert_eq!(second.get(&object(1)).await.1["properties"]["n"], 1);
    let searched = second
        .ok(
            Method::POST,
            &format!("{NOTES}/nearest"),
            Some(json!({ "vector": [1.0, 0.0], "limit": 1 })),
        )
        .await;
    assert_eq!(ids(&searched), [id(1)]);
    let (status, body) = second
        .put(&object(2), json!({ "properties": { "n": "no" } }))
        .await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::BAD_REQUEST, "invalid"),
        "the holder's answer comes back as is"
    );
    let (status, _) = second.call_as(None, Method::GET, &object(1), None).await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a key is checked before anything is sent on"
    );

    let once_only = second
        .http
        .get(format!("{}{}", second.url, object(1)))
        .bearer_auth(KEY)
        .header("x-vellev-forwarded", "1")
        .send()
        .await
        .unwrap();
    assert_eq!(
        once_only.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "a request sent on goes no further"
    );

    let unreachable = cluster.serve_advertising("http://127.0.0.1:1").await;
    unreachable.ok(Method::PUT, "/v1/tenants/other", None).await;
    unreachable
        .ok(
            Method::PUT,
            "/v1/tenants/other/collections/notes",
            Some(notes(supplied())),
        )
        .await;
    let (status, body) = second.get("/v1/tenants/other/collections").await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("could not be reached"));
}
