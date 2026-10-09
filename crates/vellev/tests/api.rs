mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use vellev_api::REQUEST;

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
        json!({ "incarnation": 1, "position": 1, "outcomes": [{ "outcome": "applied" }] })
    );

    let again = served.ok(Method::PUT, NOTES, Some(notes(supplied()))).await;
    assert_eq!(
        again,
        json!({ "incarnation": 1, "position": 1, "outcomes": [{ "outcome": "unchanged" }] })
    );
    let mut resized = notes(supplied());
    resized["embedding"]["dimensions"] = json!(3);
    let (status, body) = served.put(NOTES, resized).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    let mut misspelt = notes(supplied());
    misspelt["properties"][0]["searchble"] = json!(true);
    let (status, body) = served.put(NOTES, misspelt).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("properties[0].searchble"));
    let mut nulled = notes(supplied());
    nulled["access_rule"] = json!(null);
    nulled["embedding"]["query_prefix"] = json!(null);
    assert_eq!(
        served.put(NOTES, nulled).await.0,
        StatusCode::OK,
        "null is absent"
    );
    let in_a_batch = json!({ "changes": [{
        "op": "add_property",
        "collection": "notes",
        "property": { "name": "draft", "data_type": "bool", "filtrable": false },
    }] });
    let (status, body) = served.post("/v1/tenants/acme/changes", in_a_batch).await;
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
        json!({ "incarnation": 1, "position": 2, "outcomes": [{ "outcome": "applied" }] })
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
    assert_eq!(
        nothing,
        json!({ "incarnation": 1, "position": 1, "outcomes": [] })
    );
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
async fn a_tenant_written_by_a_node_that_cannot_be_reached_is_read_here_but_not_written() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.serve_advertising("").await, cluster.serve().await);
    first.with_notes(supplied()).await;
    for path in ["/v1/tenants/acme/collections", NOTES] {
        assert_eq!(second.get(path).await.0, StatusCode::OK, "{path}");
    }
    let fox = json!({ "properties": { "title": "a red fox", "n": 1 }, "vector": [1.0, 0.0] });
    let (status, body) = second.put(&object(1), fox.clone()).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    assert_eq!(
        second.get("/v1/tenants?limit=5").await.1,
        json!({ "tenants": ["acme"] })
    );
    first.node.shutdown().await;
    assert_eq!(second.put(&object(1), fox).await.0, StatusCode::OK);
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
    assert_eq!(operations.len(), 27, "{operations:?}");

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
    let mut refs = Vec::new();
    collect_refs(&described, &mut refs);
    for reference in refs {
        let name = reference.trim_start_matches("#/components/schemas/");
        assert!(
            described["components"]["schemas"][name].is_object(),
            "{reference} is described"
        );
    }
    let searching = &described["paths"]["/v1/tenants/{tenant}/collections/{collection}/count"]
        ["post"]["parameters"];
    let headers: Vec<&Value> = searching
        .as_array()
        .unwrap()
        .iter()
        .filter(|parameter| parameter["in"] == "header")
        .map(|parameter| &parameter["name"])
        .collect();
    assert_eq!(
        headers,
        [&json!("x-vellev-after"), &json!("x-vellev-consistency")]
    );
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
    assert_eq!(by_words["degraded_by"], "unavailable", "it may pass");
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
    let (status, body) = served.put(&object(9), large.clone()).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::PAYLOAD_TOO_LARGE, "invalid")
    );
    let body_bytes = json!({ "name": "body_bytes", "max": MAX_BODY });
    assert_eq!(body["error"]["limit"], body_bytes);
    let batched = json!({ "changes": [{ "op": "upsert", "collection": "notes", "id": id(9), "properties": large["properties"] }] });
    let (status, body) = served.post("/v1/tenants/acme/changes", batched).await;
    assert_eq!(
        (status, &body["error"]["limit"]),
        (StatusCode::PAYLOAD_TOO_LARGE, &body_bytes),
        "a body read strictly is bounded alike"
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
    let large = json!({ "properties": { "title": "x".repeat(MAX_BODY) } });
    let (status, body) = second.put(&object(3), large).await;
    assert_eq!(
        (status, body["error"].clone()),
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            json!({
                "code": "invalid",
                "message": "length limit exceeded",
                "limit": { "name": "body_bytes", "max": MAX_BODY },
            })
        ),
        "a body too large to send on is refused where it arrives"
    );
    let garbled = {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let address = second.url.trim_start_matches("http://");
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request = format!(
            "PUT {} HTTP/1.1\r\nhost: {address}\r\nauthorization: Bearer {KEY}\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\nnot a chunk\r\n\r\n",
            object(3)
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).await.unwrap();
        answer
    };
    assert!(
        garbled.starts_with("HTTP/1.1 400"),
        "a body that cannot be read is invalid, not too large: {garbled}"
    );
    assert!(!garbled.contains("\"limit\""), "{garbled}");
    let (status, _) = second.call_as(None, Method::GET, &object(1), None).await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a key is checked before anything is sent on"
    );

    let once_only = second
        .http
        .delete(format!("{}{}", second.url, object(1)))
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
    let (status, body) = second
        .put(
            "/v1/tenants/other/collections/notes/objects/00000000-0000-0000-0000-000000000001",
            json!({ "properties": { "n": 1 }, "vector": [1.0, 0.0] }),
        )
        .await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("could not be reached"));
    assert_eq!(
        second.get("/v1/tenants/other/collections").await.0,
        StatusCode::OK,
        "a read is answered here whoever holds the tenant"
    );
    assert_eq!(
        second.call(Method::PUT, "/v1/tenants/other", None).await.0,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        second.delete("/v1/tenants/other").await.0,
        StatusCode::NO_CONTENT
    );

    let blackhole = cluster.serve_advertising("http://10.255.255.1:9").await;
    blackhole.ok(Method::PUT, "/v1/tenants/dark", None).await;
    blackhole
        .ok(
            Method::PUT,
            "/v1/tenants/dark/collections/notes",
            Some(notes(supplied())),
        )
        .await;
    let started = std::time::Instant::now();
    let (status, _) = second
        .put(
            "/v1/tenants/dark/collections/notes/objects/00000000-0000-0000-0000-000000000001",
            json!({ "properties": { "n": 1 }, "vector": [1.0, 0.0] }),
        )
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(8),
        "gave up connecting in {:?}",
        started.elapsed()
    );

    // A holder that takes the request and then cuts the connection may
    // have made the write.
    let cutting = {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let _ = socket.read(&mut [0; 4096]).await;
            }
        });
        url
    };
    let cut = cluster.serve_advertising(&cutting).await;
    cut.ok(Method::PUT, "/v1/tenants/torn", None).await;
    cut.ok(
        Method::PUT,
        "/v1/tenants/torn/collections/notes",
        Some(notes(supplied())),
    )
    .await;
    let (status, body) = second
        .put(
            "/v1/tenants/torn/collections/notes/objects/00000000-0000-0000-0000-000000000001",
            json!({ "properties": { "n": 1 }, "vector": [1.0, 0.0] }),
        )
        .await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "uncertain")
    );
}

#[tokio::test]
async fn creating_a_tenant_through_another_node_does_not_take_it() {
    let cluster = Cluster::new();
    let (first, second, third) = (
        cluster.serve().await,
        cluster.serve().await,
        cluster.serve().await,
    );
    first.with_notes(supplied()).await;
    first.node.shutdown().await;
    assert_eq!(
        second.call(Method::PUT, "/v1/tenants/acme", None).await.0,
        StatusCode::NO_CONTENT
    );
    assert!(
        third
            .node
            .tenant(&vellev_core::TenantId::new("acme").unwrap())
            .await
            .is_ok(),
        "nobody took the tenant by creating it again"
    );
    let fox = json!({ "properties": { "n": 1 }, "vector": [1.0, 0.0] });
    assert_eq!(
        third.ok(Method::PUT, &object(1), Some(fox)).await["position"],
        2
    );
}

#[tokio::test]
async fn a_read_on_another_node_sees_the_callers_own_writes_and_deletions() {
    let cluster = Cluster::new();
    let (writing, reading) = (cluster.serve().await, cluster.serve().await);
    writing.with_notes(supplied()).await;
    let read = |method: Method, path: String, body: Option<Value>, after: String| {
        let reading = &reading;
        async move {
            reading
                .call_headed(
                    Some(KEY),
                    method,
                    &path,
                    body,
                    &[("x-vellev-after", &after)],
                )
                .await
        }
    };
    let opened = reading.get(NOTES).await;
    assert_eq!(opened.0, StatusCode::OK);

    for n in 1..=20u64 {
        let fox = json!({ "properties": { "title": "fox", "n": n }, "vector": [1.0, 0.0] });
        let written = writing.ok(Method::PUT, &object(n), Some(fox)).await;
        let after = format!("{}.{}", written["incarnation"], written["position"]);
        let (status, found) = read(Method::GET, object(n), None, after.clone()).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "object {n} as soon as it is written"
        );
        assert_eq!(found["properties"]["n"], n);

        let deleted = writing.ok(Method::DELETE, &object(n), None).await;
        let after = format!("{}.{}", deleted["incarnation"], deleted["position"]);
        let (_, listed) = read(
            Method::POST,
            format!("{NOTES}/list"),
            Some(json!({ "limit": 50 })),
            after.clone(),
        )
        .await;
        assert_eq!(
            listed["objects"],
            json!([]),
            "gone as soon as it is deleted"
        );
        let (status, _) = read(Method::GET, object(n), None, after).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    let (status, body) = read(Method::GET, NOTES.to_string(), None, "1.999".into()).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::BAD_REQUEST, "invalid"),
        "a point past anything committed is refused"
    );
    for (name, value) in [
        ("x-vellev-after", "12"),
        ("x-vellev-after", "a.b"),
        ("x-vellev-consistency", "linear"),
    ] {
        let (status, body) = reading
            .call_headed(Some(KEY), Method::GET, NOTES, None, &[(name, value)])
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{name}: {value}"
        );
    }
    let (status, _) = reading
        .call_headed(
            Some(KEY),
            Method::GET,
            NOTES,
            None,
            &[("x-vellev-consistency", "strong")],
        )
        .await;
    assert_eq!(status, StatusCode::OK);
}

/// Every `$ref` anywhere in `value`.
fn collect_refs<'a>(value: &'a Value, refs: &mut Vec<&'a str>) {
    match value {
        Value::Object(fields) => fields.iter().for_each(|(name, field)| match field {
            Value::String(reference) if name == "$ref" => refs.push(reference),
            other => collect_refs(other, refs),
        }),
        Value::Array(items) => items.iter().for_each(|item| collect_refs(item, refs)),
        _ => {}
    }
}

#[tokio::test]
async fn a_write_named_again_is_answered_as_it_was_and_made_once() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let object = object(1);
    served
        .ok(
            Method::PUT,
            &object,
            Some(json!({ "properties": { "n": 1 }, "vector": [1, 0] })),
        )
        .await;
    let named = "6f1c2b9e-1d0a-4c55-9a3e-2f6c4d8b7a10";
    let increment = json!({ "changes": [{
        "op": "increment", "collection": "notes", "id": id(1), "property": "n", "by": 5
    }] });
    let changes = "/v1/tenants/acme/changes";
    let (served, increment) = (&served, &increment);
    let send = |name: &'static str| async move {
        let headers = [(REQUEST, name)];
        served
            .call_headed(
                Some(KEY),
                Method::POST,
                changes,
                Some(increment.clone()),
                &headers,
            )
            .await
    };
    let (status, first) = send(named).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    served
        .ok(
            Method::PUT,
            &object,
            Some(json!({ "properties": { "n": 100 }, "vector": [1, 0] })),
        )
        .await;
    let (status, again) = send(named).await;
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(again, first);
    let stored = served.ok(Method::GET, &object, None).await;
    assert_eq!(stored["properties"]["n"], json!(100));

    let (status, refused) = send("not-a-uuid").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code(&refused), "invalid");

    // A refusal answered again says what it said, with what was missing,
    // even once what was missing is there.
    for (name, change, message, missing) in [
        (
            "6f1c2b9e-1d0a-4c55-9a3e-2f6c4d8b7a11",
            json!({ "op": "patch", "collection": "notes", "id": id(7), "properties": { "n": 1 } }),
            format!("object {} does not exist in notes", id(7)),
            "object",
        ),
        (
            "6f1c2b9e-1d0a-4c55-9a3e-2f6c4d8b7a12",
            json!({ "op": "delete", "collection": "later", "id": id(7) }),
            "collection later does not exist".to_string(),
            "collection",
        ),
    ] {
        let named = [(REQUEST, name)];
        let send = || {
            let batch = json!({ "changes": [change] });
            served.call_headed(Some(KEY), Method::POST, changes, Some(batch), &named)
        };
        let expected = (
            StatusCode::NOT_FOUND,
            failed("not_found", &message, json!({ "missing": missing })),
        );
        assert_eq!(send().await, expected, "{missing}");
        let mut later = notes(supplied());
        later["name"] = json!("later");
        served
            .ok(
                Method::PUT,
                "/v1/tenants/acme/collections/later",
                Some(later),
            )
            .await;
        let seven = json!({ "properties": { "n": 7 }, "vector": [1, 0] });
        served
            .ok(Method::PUT, &common::object(7), Some(seven))
            .await;
        assert_eq!(send().await, expected, "{missing}, answered again");
        served.ok(Method::DELETE, &common::object(7), None).await;
        served
            .ok(Method::DELETE, "/v1/tenants/acme/collections/later", None)
            .await;
    }
}

async fn as_key(
    node: &Served,
    key: &str,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    node.call_as(Some(key), method, path, body).await
}

#[tokio::test]
async fn a_tenant_key_reaches_its_tenant_alone_and_learns_nothing_of_others() {
    let cluster = Cluster::new();
    let (first, second) = (cluster.serve().await, cluster.serve().await);
    first.with_notes(supplied()).await;
    first.ok(Method::PUT, "/v1/tenants/beta", None).await;
    let beta_notes = "/v1/tenants/beta/collections/notes";
    first
        .ok(Method::PUT, beta_notes, Some(notes(supplied())))
        .await;
    let issued = first.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let key = issued["key"].as_str().unwrap().to_string();
    assert!(key.starts_with("vlv_"), "{key}");
    let listed = first.ok(Method::GET, "/v1/tenants/acme/keys", None).await;
    assert_eq!(listed["keys"][0]["id"], issued["id"]);
    assert!(
        listed["keys"][0].get("key").is_none(),
        "a listed key has no secret"
    );

    let fox = json!({ "properties": { "title": "a red fox", "n": 1 }, "vector": [1.0, 0.0] });
    for node in [&first, &second] {
        assert_eq!(
            as_key(node, &key, Method::PUT, &object(1), Some(fox.clone()))
                .await
                .0,
            StatusCode::OK,
            "a write to its own tenant, sent on or not"
        );
        assert_eq!(
            as_key(node, &key, Method::GET, &object(1), None).await.0,
            StatusCode::OK
        );

        let missing = as_key(
            node,
            &key,
            Method::GET,
            "/v1/tenants/gamma/collections",
            None,
        )
        .await;
        assert_eq!(missing.0, StatusCode::NOT_FOUND);
        let beta = as_key(
            node,
            &key,
            Method::GET,
            "/v1/tenants/beta/collections",
            None,
        )
        .await;
        let beta_object = beta_notes.to_string() + "/objects/" + &id(1);
        let written = as_key(node, &key, Method::PUT, &beta_object, Some(fox.clone())).await;
        let as_if_absent = |tenant: &str| {
            json!({ "error": {
                "code": "not_found",
                "message": format!("tenant {tenant} does not exist"),
                "missing": "tenant",
            } })
        };
        assert_eq!(missing.1, as_if_absent("gamma"));
        assert_eq!(
            node.get("/v1/tenants/gamma/collections").await,
            (StatusCode::NOT_FOUND, as_if_absent("gamma")),
            "a tenant that is not there, as an admin key is told"
        );
        for (status, body) in [beta, written] {
            assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
            assert_eq!(body, as_if_absent("beta"), "as if beta were not there");
        }

        for (method, path) in [
            (Method::GET, "/v1/tenants"),
            (Method::PUT, "/v1/tenants/acme"),
            (Method::DELETE, "/v1/tenants/beta"),
            (Method::POST, "/v1/tenants/acme/keys"),
            (Method::GET, "/v1/tenants/acme/keys"),
        ] {
            let (status, body) = as_key(node, &key, method.clone(), path, None).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{method} {path}: {body}");
        }
    }
    let beta_count = first
        .get(&(beta_notes.to_string() + "/objects/" + &id(1)))
        .await;
    assert_eq!(
        beta_count.0,
        StatusCode::NOT_FOUND,
        "nothing was written to beta"
    );

    let guessed = format!("vlv_{}", "0".repeat(64));
    for wrong in [guessed.as_str(), "not-a-key", ""] {
        let (status, _) = first
            .call_as(
                Some(wrong),
                Method::GET,
                "/v1/tenants/acme/collections",
                None,
            )
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{wrong:?}");
    }
    let (status, _) = first
        .call_as(None, Method::GET, "/v1/tenants/acme/collections", None)
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // However the path is written, a key reaches no further than its tenant.
    for path in [
        "/v1/tenants/%62eta/collections",
        "/v1/tenants/ACME/collections",
        "/v1/tenants//acme/collections",
        "/v1/tenants/beta%2Facme/collections",
        "/v1/tenants/acme/../beta/collections",
        "/v1/tenants/acme/collections/notes/../../../beta/collections",
        "/v1/tenants/beta/keys",
    ] {
        let (status, body) = as_key(&first, &key, Method::GET, path, None).await;
        assert!(
            [
                StatusCode::NOT_FOUND,
                StatusCode::FORBIDDEN,
                StatusCode::BAD_REQUEST
            ]
            .contains(&status),
            "{path}: {status} {body}"
        );
        assert!(body.get("collections").is_none(), "{path}: {body}");
    }
    // Sent as written: an HTTP client would take the dots out first.
    let raw = {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let address = second.url.trim_start_matches("http://");
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request = format!(
            "DELETE /v1/tenants/acme/collections/%2e%2e HTTP/1.1\r\nhost: {address}\r\nauthorization: Bearer {key}\r\nconnection: close\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).await.unwrap();
        answer
    };
    assert!(
        raw.starts_with("HTTP/1.1 400"),
        "no dot segment is sent on: {raw}"
    );
    let (status, _) = second
        .call_headed(
            Some(&key),
            Method::GET,
            "/v1/tenants/beta/collections",
            None,
            &[("x-vellev-forwarded", "1")],
        )
        .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "nor by claiming to be sent on"
    );

    // Revoked through the node that does not hold the tenant, the key stops
    // working on that node at once.
    let id = issued["id"].as_str().unwrap();
    second
        .ok(Method::DELETE, &format!("/v1/tenants/acme/keys/{id}"), None)
        .await;
    let (status, _) = second
        .call_as(
            Some(&key),
            Method::GET,
            "/v1/tenants/acme/collections",
            None,
        )
        .await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a revoked key, at once where revoked"
    );
    assert_eq!(
        first.delete(&format!("/v1/tenants/acme/keys/{id}")).await.0,
        StatusCode::NOT_FOUND
    );

    let other = first.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let other = other["key"].as_str().unwrap().to_string();
    first.ok(Method::DELETE, "/v1/tenants/acme", None).await;
    first.node.purge_due().await.unwrap();
    first.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let (status, _) = first
        .call_as(
            Some(&other),
            Method::GET,
            "/v1/tenants/acme/collections",
            None,
        )
        .await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "a tenant made again is not reached by the keys of the one dropped"
    );
    assert_eq!(
        first.post("/v1/tenants/gone/keys", json!(null)).await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn every_route_is_either_for_admins_or_one_tenants() {
    let described: Value = serde_json::to_value(vellev::server::openapi()).unwrap();
    let mut routes: Vec<&str> = described["paths"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    routes.sort_unstable();
    // A route added must be put in one list or the other, knowingly.
    let tenants_own = [
        "/v1/tenants/{tenant}/changes",
        "/v1/tenants/{tenant}/collections",
        "/v1/tenants/{tenant}/collections/{collection}",
        "/v1/tenants/{tenant}/collections/{collection}/count",
        "/v1/tenants/{tenant}/collections/{collection}/hybrid",
        "/v1/tenants/{tenant}/collections/{collection}/keyword",
        "/v1/tenants/{tenant}/collections/{collection}/list",
        "/v1/tenants/{tenant}/collections/{collection}/nearest",
        "/v1/tenants/{tenant}/collections/{collection}/objects/{id}",
        "/v1/tenants/{tenant}/collections/{collection}/properties",
    ];
    for route in vellev::server::EVERY_KEY {
        assert!(
            !route.contains("{tenant}"),
            "{route} names a tenant, which a route for every key must not: it is one tenant's"
        );
    }
    let mut classified: Vec<&str> = vellev::server::ADMIN_ONLY
        .iter()
        .chain(&vellev::server::EVERY_KEY)
        .chain(&tenants_own)
        .copied()
        .collect();
    classified.sort_unstable();
    assert_eq!(routes, classified);
}

#[tokio::test]
async fn a_tenant_asking_too_much_is_answered_429_with_when_to_ask_again() {
    let served = Cluster::new().serve_limited(3.0).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, "/v1/tenants/beta", None).await;
    let mut statuses = Vec::new();
    for _ in 0..5 {
        let answer = served
            .http
            .get(format!("{}/v1/tenants/acme/collections", served.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap();
        let retry_after = answer.headers().get("retry-after").cloned();
        let status = answer.status();
        let body: Value = answer.json().await.unwrap();
        statuses.push(status);
        if status == StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(code(&body), "limited");
            assert_eq!(retry_after.unwrap().to_str().unwrap(), "1");
        }
    }
    assert_eq!(
        statuses.iter().filter(|s| **s == StatusCode::OK).count(),
        3,
        "{statuses:?}"
    );
    assert_eq!(
        served.get("/v1/tenants/beta/collections").await.0,
        StatusCode::OK,
        "another tenant is not held up"
    );
}

#[tokio::test]
async fn reads_past_as_many_as_a_node_answers_at_once_are_turned_away_and_writes_go_on() {
    let served = Served::new().await;
    four_notes(&served).await;
    let nearest = json!({ "vector": [1.0, 0.0], "limit": 1 });
    let path = format!("{NOTES}/nearest");
    let search = || async {
        let asked = served.http.post(format!("{}{path}", served.url));
        let answer = asked.bearer_auth(KEY).json(&nearest).send().await.unwrap();
        let again = answer.headers().get("retry-after").cloned();
        (
            answer.status(),
            again,
            answer.json::<Value>().await.unwrap(),
        )
    };

    // Every turn there is for a read is taken, as by reads that never
    // end, of tenants other than the one asking.
    let others = ["beta", "gamma"].map(|other| vellev_core::TenantId::new(other).unwrap());
    let mut taken = Vec::new();
    for other in others.iter().cycle().take(READS) {
        taken.push(served.app.reading(other).await.unwrap());
    }
    let asked = std::time::Instant::now();
    let (status, again, refused) = search().await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{refused}");
    assert_eq!(code(&refused), "unavailable", "to be asked again");
    assert_eq!(again.unwrap(), "1", "and told when");
    assert!(
        asked.elapsed() >= std::time::Duration::from_millis(50),
        "it waited a moment first"
    );
    // A write takes no such turn.
    let note = json!({ "properties": { "title": "an owl", "n": 9 }, "vector": [0.0, 1.0] });
    let (status, written) = served
        .put(&format!("{NOTES}/objects/{}", id(9)), note)
        .await;
    assert_eq!(status, StatusCode::OK, "{written}");

    // A turn that comes free while a read waits for one is the read's.
    let freeing = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        drop(taken);
    });
    let (status, _, found) = search().await;
    assert_eq!(status, StatusCode::OK, "{found}");
    freeing.await.unwrap();
}

#[tokio::test]
async fn a_tenant_alone_may_have_every_read_a_node_answers_at_once() {
    let served = Served::new().await;
    four_notes(&served).await;
    let acme = vellev_core::TenantId::new("acme").unwrap();
    let nearest = json!({ "vector": [1.0, 0.0], "limit": 1 });
    let search = || async {
        served
            .post(&format!("{NOTES}/nearest"), nearest.clone())
            .await
            .0
    };

    // With no other tenant asking, it is held to no share of the turns.
    let mut all = Vec::new();
    for _ in 0..READS {
        all.push(served.app.reading(&acme).await.expect("a turn"));
    }
    assert_eq!(search().await, StatusCode::SERVICE_UNAVAILABLE);
    all.pop();
    assert_eq!(search().await, StatusCode::OK);
}

/// What each kind of read is asked with, of a collection whose vectors
/// its clients supply.
fn reads() -> [(&'static str, Value); 5] {
    [
        ("nearest", json!({ "vector": [1.0, 0.0], "limit": 1 })),
        ("keyword", json!({ "text": "fox", "limit": 1 })),
        (
            "hybrid",
            json!({ "text": "fox", "vector": [1.0, 0.0], "limit": 1 }),
        ),
        ("list", json!({ "limit": 1 })),
        ("count", json!({})),
    ]
}

/// Tenant `tenant` with a `notes` collection and one note in it.
async fn one_note(served: &Served, tenant: &str, embedding: Value) {
    let at = format!("/v1/tenants/{tenant}");
    served.ok(Method::PUT, &at, None).await;
    let notes_at = format!("{at}/collections/notes");
    let supplies = embedding["provider"] == "none";
    served
        .ok(Method::PUT, &notes_at, Some(notes(embedding)))
        .await;
    let mut note = json!({ "properties": { "title": "a red fox", "n": 1 } });
    if supplies {
        note["vector"] = json!([1.0, 0.0]);
    }
    let path = format!("{notes_at}/objects/{}", id(1));
    served.ok(Method::PUT, &path, Some(note)).await;
}

/// Every kind of read of `tenant`, asked over and over for a while: the
/// first that is not answered, if any.
async fn unanswered(served: &Served, tenant: &str) -> Option<(&'static str, StatusCode, Value)> {
    let until = std::time::Instant::now() + std::time::Duration::from_millis(300);
    while std::time::Instant::now() < until {
        for (kind, asked) in reads() {
            let path = format!("/v1/tenants/{tenant}/collections/notes/{kind}");
            let (status, body) = served.post(&path, asked).await;
            if status != StatusCode::OK {
                return Some((kind, status, body));
            }
        }
    }
    None
}

#[tokio::test]
async fn a_read_turned_away_for_want_of_a_turn_is_not_counted_against_its_tenant() {
    // Two queries a second: a third within the second is turned away as
    // one too many.
    let served = Cluster::new().serve_limited(2.0).await;
    four_notes(&served).await;
    let other = vellev_core::TenantId::new("beta").unwrap();
    let mut taken = Vec::new();
    for _ in 0..READS {
        taken.push(served.app.reading(&other).await.unwrap());
    }
    // However many are turned away for want of a turn, of whichever kind,
    // none is one of the two.
    for (kind, asked) in reads() {
        for nth in 0..4 {
            let (status, refused) = served.post(&format!("{NOTES}/{kind}"), asked.clone()).await;
            assert_eq!(
                (status, code(&refused)),
                (StatusCode::SERVICE_UNAVAILABLE, "unavailable"),
                "{kind} {nth}: {refused}"
            );
        }
    }
    drop(taken);
    for (kind, asked) in reads().into_iter().take(2) {
        let (status, answered) = served.post(&format!("{NOTES}/{kind}"), asked).await;
        assert_eq!(status, StatusCode::OK, "{kind}: {answered}");
    }
}

/// The statuses of `asked` reads of one kind sent one after another, as
/// fast as they are answered: far faster than two a second.
async fn burst(
    served: &Served,
    asked: usize,
    read: impl Fn(usize) -> (String, Value, &'static [(&'static str, &'static str)]),
) -> Vec<StatusCode> {
    let mut statuses = Vec::new();
    for nth in 0..asked {
        let (path, body, headers) = read(nth);
        let answer = served.call_headed(Some(KEY), Method::POST, &path, Some(body), headers);
        statuses.push(answer.await.0);
    }
    statuses
}

/// How many of `statuses` are each of the two, if they are all one of
/// the two.
fn either(statuses: &[StatusCode], one: StatusCode, other: StatusCode) -> (usize, usize) {
    let count = |status| statuses.iter().filter(|told| **told == status).count();
    assert_eq!(count(one) + count(other), statuses.len(), "{statuses:?}");
    (count(one), count(other))
}

#[tokio::test]
async fn a_read_that_had_its_turn_is_a_query_however_it_ends() {
    // Two queries a second, and twenty reads in far less than ten: most
    // are one too many, were each of them counted.
    let nearest = || json!({ "vector": [1.0, 0.0], "limit": 1 });
    let answered = Cluster::new().serve_limited(2.0).await;
    four_notes(&answered).await;
    let statuses = burst(&answered, 20, |_| {
        (format!("{NOTES}/nearest"), nearest(), &[])
    })
    .await;
    assert_eq!(statuses[..2], [StatusCode::OK; 2], "{statuses:?}");
    let (_, too_many) = either(&statuses, StatusCode::OK, StatusCode::TOO_MANY_REQUESTS);
    assert!(too_many > 0, "a search answered is a query: {statuses:?}");

    // One that fails once it has its turn, as a search of a collection
    // that is not there does, is one as well.
    let failing = Cluster::new().serve_limited(2.0).await;
    four_notes(&failing).await;
    let absent = "/v1/tenants/acme/collections/absent/nearest";
    let statuses = burst(&failing, 20, |_| (absent.to_string(), nearest(), &[])).await;
    assert_eq!(statuses[..2], [StatusCode::NOT_FOUND; 2], "{statuses:?}");
    let (_, too_many) = either(
        &statuses,
        StatusCode::NOT_FOUND,
        StatusCode::TOO_MANY_REQUESTS,
    );
    assert!(too_many > 0, "a search that failed is one: {statuses:?}");
}

#[tokio::test]
async fn a_read_turned_away_after_something_costly_was_done_for_it_stays_counted() {
    let down = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let url = embedding_server(down).await;
    let served = Cluster::new()
        .serve_limited_embedding_through(2.0, std::slice::from_ref(&url))
        .await;
    let embedding = json!({
        "provider": "http", "url": format!("{url}/"), "model": "stand-in",
        "dimensions": 2, "properties": ["title"],
    });
    one_note(&served, "acme", embedding).await;
    let by_text = |text: String| {
        (
            format!("{NOTES}/nearest"),
            json!({ "text": text, "limit": 1 }),
            &[] as &[_],
        )
    };
    // One search by a text, answered: its vector is kept.
    assert_eq!(
        burst(&served, 1, |_| by_text("fox".into())).await,
        [StatusCode::OK]
    );

    // Every turn is taken. The same text again is sent to no model, and
    // however often it is turned away it is not counted.
    let other = vellev_core::TenantId::new("beta").unwrap();
    let mut taken = Vec::new();
    for _ in 0..READS {
        taken.push(served.app.reading(&other).await.unwrap());
    }
    let turned_away = StatusCode::SERVICE_UNAVAILABLE;
    let again = burst(&served, 12, |_| by_text("fox".into())).await;
    assert_eq!(again, [turned_away; 12]);
    // A text not asked before is sent to the model each time, and each
    // is counted though it is turned away: the tenant is soon told it
    // asks too much, and the model is asked no more for it.
    let fresh = burst(&served, 20, |nth| {
        by_text(format!("a text never asked {nth}"))
    })
    .await;
    assert_eq!(fresh[0], turned_away, "{fresh:?}");
    let (_, too_many) = either(&fresh, turned_away, StatusCode::TOO_MANY_REQUESTS);
    assert!(too_many > 0, "a search embedded is a query: {fresh:?}");
    drop(taken);

    // So is a read the coordinator was asked for.
    let served = Cluster::new().serve_limited(2.0).await;
    four_notes(&served).await;
    let mut taken = Vec::new();
    for _ in 0..READS {
        taken.push(served.app.reading(&other).await.unwrap());
    }
    type Headers = &'static [(&'static str, &'static str)];
    let strong: Headers = &[("x-vellev-consistency", "strong")];
    let count = |headers: Headers| (format!("{NOTES}/count"), json!({}), headers);
    let eventual = burst(&served, 12, |_| count(&[])).await;
    assert_eq!(eventual, [turned_away; 12]);
    let asked = burst(&served, 20, |_| count(strong)).await;
    assert_eq!(asked[0], turned_away, "{asked:?}");
    let (_, too_many) = either(&asked, turned_away, StatusCode::TOO_MANY_REQUESTS);
    assert!(too_many > 0, "a strong read is a query: {asked:?}");
}

/// A stand-in embedding server that keeps every text beginning `wait`
/// unanswered while `held` is set, and counts those it is keeping.
async fn holding_embedding_server(
    held: tokio::sync::watch::Receiver<bool>,
    keeping: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> String {
    let embed = move |axum::Json(body): axum::Json<Value>| {
        let (mut held, keeping) = (held.clone(), keeping.clone());
        async move {
            let texts: Vec<&str> = body["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|text| text.as_str().unwrap())
                .collect();
            if texts.iter().any(|text| text.starts_with("wait")) {
                keeping.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                held.wait_for(|held| !held).await.unwrap();
            }
            let vectors: Vec<[usize; 2]> = texts
                .iter()
                .map(|text| [text.len(), text.split_whitespace().count()])
                .collect();
            axum::Json(json!(vectors))
        }
    };
    let router = axum::Router::new().route("/embed", axum::routing::post(embed));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

/// Wait until `count` is no longer nothing.
async fn some(count: &std::sync::atomic::AtomicUsize) {
    let counted = async {
        while count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(30), counted)
        .await
        .expect("something to count");
}

#[tokio::test]
async fn a_search_waiting_for_its_text_to_be_embedded_keeps_no_turn_from_other_reads() {
    let (hold, held) = tokio::sync::watch::channel(false);
    let keeping = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let url = holding_embedding_server(held, keeping.clone()).await;
    let cluster = Cluster::new();
    let served = cluster
        .serve_embedding_through(std::slice::from_ref(&url))
        .await;
    let served = std::sync::Arc::new(served);
    let embedding = json!({
        "provider": "http", "url": format!("{url}/"), "model": "stand-in",
        "dimensions": 2, "properties": ["title"],
    });
    one_note(&served, "acme", embedding.clone()).await;
    one_note(&served, "beta", embedding).await;
    one_note(&served, "gamma", supplied()).await;

    for kind in ["nearest", "hybrid"] {
        // The embedder stops answering, and more searches by text than
        // the node has turns wait for it.
        hold.send(true).unwrap();
        keeping.store(0, std::sync::atomic::Ordering::SeqCst);
        let waiting: Vec<_> = (0..2 * READS)
            .map(|nth| {
                let served = served.clone();
                let tenant = ["acme", "beta"][nth % 2];
                tokio::spawn(async move {
                    let path = format!("/v1/tenants/{tenant}/collections/notes/{kind}");
                    let asked = json!({ "text": format!("wait {kind} {nth}"), "limit": 1 });
                    served.post(&path, asked).await
                })
            })
            .collect();
        some(&keeping).await;
        // None of them is on a thread, and none has a turn: every read
        // that needs no embedding is answered, of a tenant that searches
        // by text and of one that does not.
        for tenant in ["gamma", "acme"] {
            let refused = unanswered(&served, tenant).await;
            assert!(refused.is_none(), "{kind}, {tenant}: {refused:?}");
        }
        hold.send(false).unwrap();
        for waited in waiting {
            let (status, answer) = waited.await.unwrap();
            assert_eq!(status, StatusCode::OK, "{kind}: {answer}");
        }
    }
}

/// A store that keeps what it is asked to get of one tenant unanswered
/// while `held` is set, and counts what it is keeping.
#[derive(Debug)]
struct HoldingStore {
    inner: std::sync::Arc<dyn vellev_storage::blobs::Blobs>,
    of: &'static str,
    held: tokio::sync::watch::Receiver<bool>,
    keeping: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl vellev_storage::blobs::Blobs for HoldingStore {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }

    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        let mut held = self.held.clone();
        if key.contains(self.of) && *held.borrow() {
            self.keeping
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            held.wait_for(|held| !held).await.unwrap();
        }
        self.inner.get(key).await
    }

    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        self.inner.list(prefix).await
    }

    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_read_waiting_for_its_tenant_to_be_opened_keeps_no_turn_from_another_tenants() {
    let cluster = Cluster::new();
    let holder = cluster.serve().await;
    one_note(&holder, "acme", supplied()).await;
    one_note(&holder, "gamma", supplied()).await;

    for (kind, asked) in reads() {
        // Another node, as one just started: it has opened one tenant,
        // and its store is slow to give it the other's files.
        let (hold, held) = tokio::sync::watch::channel(false);
        let keeping = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let store = HoldingStore {
            inner: cluster.blobs(),
            of: "acme",
            held,
            keeping: keeping.clone(),
        };
        let other = cluster.through(std::sync::Arc::new(store)).serve().await;
        let other = std::sync::Arc::new(other);
        assert_eq!(unanswered(&other, "gamma").await, None);
        hold.send(true).unwrap();
        // More reads than the node has turns wait for the tenant to open.
        let waiting: Vec<_> = (0..2 * READS)
            .map(|_| {
                let (other, asked) = (other.clone(), asked.clone());
                let path = format!("{NOTES}/{kind}");
                tokio::spawn(async move { other.post(&path, asked).await })
            })
            .collect();
        some(&keeping).await;
        let refused = unanswered(&other, "gamma").await;
        assert!(refused.is_none(), "{kind}: {refused:?}");
        hold.send(false).unwrap();
        for waited in waiting {
            let (status, answer) = waited.await.unwrap();
            assert_eq!(status, StatusCode::OK, "{kind}: {answer}");
        }
    }
}

#[tokio::test]
async fn a_read_keeps_its_turn_until_it_is_done_whoever_stops_waiting_for_it() {
    let served = Served::new().await;
    let acme = vellev_core::TenantId::new("acme").unwrap();
    // A read that takes its time, and whoever asked for it gone.
    let (done, wait) = std::sync::mpsc::channel::<()>();
    let turn = served.app.reading(&acme).await.unwrap();
    let reading = tokio::spawn(turn.run(move || wait.recv()));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    reading.abort();
    assert!(reading.await.unwrap_err().is_cancelled());

    // It is being read still, and counts for one of the node's turns.
    let mut rest = Vec::new();
    for _ in 1..READS {
        rest.push(served.app.reading(&acme).await.unwrap());
    }
    assert!(served.app.reading(&acme).await.is_err(), "all are taken");
    drop(done);
    let last = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            if let Ok(turn) = served.app.reading(&acme).await {
                break turn;
            }
        }
    });
    last.await.expect("free once the read is done");
    drop(rest);
}

/// A failure's whole body, as the API writes it.
fn failed(code: &str, message: &str, more: Value) -> Value {
    let mut problem = json!({ "code": code, "message": message });
    problem
        .as_object_mut()
        .unwrap()
        .extend(more.as_object().unwrap().clone());
    json!({ "error": problem })
}

#[tokio::test]
async fn a_failure_to_find_says_what_is_missing() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let missing = |what: &str| json!({ "missing": what });
    let nil = "00000000-0000-0000-0000-000000000000";
    let patched = json!({ "changes": [{ "op": "patch", "collection": "notes", "id": id(9), "properties": { "n": 1 } }] });
    let elsewhere = json!({ "changes": [{ "op": "delete", "collection": "absent", "id": id(9) }] });
    let ghost = || {
        failed(
            "not_found",
            "tenant ghost does not exist",
            missing("tenant"),
        )
    };
    let no_object = || {
        failed(
            "not_found",
            &format!("object {} does not exist in notes", id(9)),
            missing("object"),
        )
    };
    let no_collection = || {
        failed(
            "not_found",
            "collection absent does not exist",
            missing("collection"),
        )
    };
    for (method, path, body, expected) in [
        (
            Method::GET,
            "/v1/nowhere".to_string(),
            None,
            failed("not_found", "no such route", missing("route")),
        ),
        (
            Method::GET,
            "/v1/tenants/ghost/collections".to_string(),
            None,
            ghost(),
        ),
        (
            Method::PUT,
            "/v1/tenants/ghost/collections/notes".to_string(),
            Some(notes(supplied())),
            ghost(),
        ),
        (
            Method::GET,
            "/v1/tenants/ghost/restore-points".to_string(),
            None,
            ghost(),
        ),
        (
            Method::GET,
            "/v1/tenants/acme/collections/absent".to_string(),
            None,
            no_collection(),
        ),
        (
            Method::POST,
            "/v1/tenants/acme/changes".to_string(),
            Some(elsewhere),
            no_collection(),
        ),
        (Method::GET, object(9), None, no_object()),
        (
            Method::POST,
            "/v1/tenants/acme/changes".to_string(),
            Some(patched),
            no_object(),
        ),
        (
            Method::DELETE,
            format!("/v1/tenants/acme/keys/{nil}"),
            None,
            failed(
                "not_found",
                &format!("tenant acme has no key {nil}"),
                missing("key"),
            ),
        ),
        (
            Method::POST,
            "/v1/tenants/acme/restore".to_string(),
            Some(json!({ "at": "2001-01-01T00:00:00Z" })),
            failed(
                "not_found",
                "tenant acme has no state to go back to at that point",
                missing("restore_point"),
            ),
        ),
    ] {
        let answer = served.call(method.clone(), &path, body).await;
        assert_eq!(answer, (StatusCode::NOT_FOUND, expected), "{method} {path}");
    }
}

#[tokio::test]
async fn a_request_for_more_than_the_node_allows_says_the_most_it_may_ask() {
    let served = Served::new().await;
    four_notes(&served).await;
    let limit = |name: &str, max: u64| json!({ "limit": { "name": name, "max": max } });
    let window = 10 * 366 * 24 * 60 * 60;
    let published = served.get("/v1/limits").await.1;
    for (method, path, body, expected) in [
        (
            Method::POST,
            format!("{NOTES}/nearest"),
            Some(json!({ "vector": [1.0, 0.0], "limit": 51 })),
            failed(
                "invalid",
                "limit must be from 1 to 50, not 51",
                limit("results", 50),
            ),
        ),
        (
            Method::GET,
            "/v1/tenants?limit=51".to_string(),
            None,
            failed(
                "invalid",
                "limit must be from 1 to 50, not 51",
                limit("results", 50),
            ),
        ),
        (
            Method::POST,
            format!("{NOTES}/hybrid"),
            Some(json!({ "text": "fox", "limit": 1, "candidates": 101 })),
            failed(
                "invalid",
                "candidates must be from 1 to 100, not 101",
                limit("candidates", 100),
            ),
        ),
        (
            Method::POST,
            format!("{NOTES}/nearest"),
            Some(json!({ "vector": [1.0, 0.0], "limit": 1, "oversample": 1025 })),
            failed(
                "invalid",
                "oversample must be from 1 to 1024, not 1025",
                limit("oversample", 1024),
            ),
        ),
        (
            Method::PUT,
            "/v1/tenants/acme/restore-window".to_string(),
            Some(json!({ "secs": window + 1 })),
            failed(
                "invalid",
                &format!("a restore window is from 1 to {window} seconds"),
                limit("restore_window_secs", window),
            ),
        ),
    ] {
        let answer = served.call(method.clone(), &path, body).await;
        assert_eq!(
            answer,
            (StatusCode::BAD_REQUEST, expected.clone()),
            "{method} {path}"
        );
        let refused = &expected["error"]["limit"];
        assert_eq!(
            published[refused["name"].as_str().unwrap()],
            refused["max"],
            "what is refused is what is published"
        );
    }
    // A request for none at all is not helped by the most it may ask.
    for (method, path, body, expected) in [
        (
            Method::POST,
            format!("{NOTES}/list"),
            json!({ "limit": 0 }),
            failed("invalid", "limit must be from 1 to 50, not 0", json!({})),
        ),
        (
            Method::POST,
            format!("{NOTES}/hybrid"),
            json!({ "text": "fox", "limit": 1, "candidates": 0 }),
            failed(
                "invalid",
                "candidates must be from 1 to 100, not 0",
                json!({}),
            ),
        ),
        (
            Method::PUT,
            "/v1/tenants/acme/restore-window".to_string(),
            json!({ "secs": 0 }),
            failed(
                "invalid",
                &format!("a restore window is from 1 to {window} seconds"),
                json!({}),
            ),
        ),
    ] {
        let answer = served.call(method.clone(), &path, Some(body)).await;
        assert_eq!(
            answer,
            (StatusCode::BAD_REQUEST, expected),
            "{method} {path}"
        );
    }
    let (status, body) = served
        .post(
            &format!("{NOTES}/nearest"),
            json!({ "vector": [1.0], "limit": 1 }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"].get("limit").is_none(),
        "a request invalid in another way names no limit: {body}"
    );
}

#[tokio::test]
async fn the_limits_are_told_to_every_key_and_to_no_caller_without_one() {
    let served = Served::new().await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let tenants_key = issued["key"].as_str().unwrap();
    let limits = json!({
        "results": MAX_LIMIT,
        "candidates": 100,
        "oversample": 1024,
        "body_bytes": MAX_BODY,
        "restore_window_secs": 10 * 366 * 24 * 60 * 60,
        "deleted": 8_192,
        "reads_at_once": READS,
    });
    for key in [KEY, tenants_key] {
        let told = served
            .call_as(Some(key), Method::GET, "/v1/limits", None)
            .await;
        assert_eq!(told, (StatusCode::OK, limits.clone()));
    }
    for key in [None, Some("not-a-key")] {
        let (status, body) = served.call_as(key, Method::GET, "/v1/limits", None).await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::UNAUTHORIZED, "unauthorized")
        );
    }
    let (status, body) = served
        .call_as(Some(tenants_key), Method::POST, "/v1/limits", None)
        .await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::METHOD_NOT_ALLOWED, "invalid")
    );
}

#[tokio::test]
async fn a_request_to_delete_more_than_one_may_says_the_most_one_may() {
    let served = Served::new().await;
    four_notes(&served).await;
    let (_, limits) = served.get("/v1/limits").await;
    let most = limits["deleted"].as_u64().expect("the node says the most");
    let delete = |limit: Option<u64>| {
        let mut change = json!({ "op": "delete_where", "collection": "notes",
            "filter": { "gte": { "property": "n", "value": 0 } } });
        if let Some(limit) = limit {
            change["limit"] = json!(limit);
        }
        change
    };
    let send = |changes: Vec<Value>| {
        let served = &served;
        async move {
            let body = json!({ "changes": changes });
            served.post("/v1/tenants/acme/changes", body).await
        }
    };
    let too_many = |message: &str| {
        (
            StatusCode::BAD_REQUEST,
            failed(
                "invalid",
                message,
                json!({ "limit": { "name": "deleted", "max": most } }),
            ),
        )
    };
    let between_them = |may: u64| {
        too_many(&format!(
            "the deletes by filter of one request may delete {may} objects between them, each \
             without a limit counting for {most}, which is the most one request may delete; \
             send them in requests of their own, or with limits that come to no more"
        ))
    };

    // A limit past the most, and limits that come to more between them,
    // with or without one that has none.
    assert_eq!(
        send(vec![delete(Some(most + 1))]).await,
        too_many(&format!("a delete takes a limit of 1 to {most}"))
    );
    assert_eq!(
        send(vec![delete(Some(most)), delete(Some(1))]).await,
        between_them(most + 1)
    );
    assert_eq!(
        send(vec![delete(None), delete(None)]).await,
        between_them(2 * most)
    );
    // A limit of none is no limit to give: it asks for nothing too much,
    // and is told no most.
    let (status, none) = send(vec![delete(Some(0))]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{none}");
    assert_eq!(
        none,
        failed(
            "invalid",
            &format!("a delete takes a limit of 1 to {most}"),
            json!({})
        )
    );
    // Nothing was deleted by any of them, and the most itself is taken.
    let (_, counted) = served.post(&format!("{NOTES}/count"), json!({})).await;
    assert_eq!(counted["count"], 4);
    let (status, deleted) = send(vec![delete(Some(most))]).await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    let (_, counted) = served.post(&format!("{NOTES}/count"), json!({})).await;
    assert_eq!(counted["count"], 0);
}

/// What a node that cannot use a collection's model answers.
fn model_unavailable(reason: &str) -> (StatusCode, Value) {
    (
        StatusCode::NOT_IMPLEMENTED,
        failed(
            "model_unavailable",
            "this node cannot use the collection's embedding model as it is set up",
            json!({ "reason": reason }),
        ),
    )
}

#[tokio::test]
async fn a_collection_with_a_model_no_build_runs_is_refused_as_the_requests_fault() {
    let served = Served::new().await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let local = |model: &str, dimensions: u32| json!({ "provider": "local", "model": model, "dimensions": dimensions, "properties": ["title"] });
    let unheard_of = local("acme/unheard-of", 2);
    let invalid = |message: &str| {
        (
            StatusCode::BAD_REQUEST,
            failed("invalid", message, json!({})),
        )
    };
    let titan = json!({ "provider": "bedrock", "region": "eu-west-2", "model": "amazon.titan-embed-text-v2:0", "dimensions": 2 });
    for (embedding, message) in [
        (unheard_of.clone(), "acme/unheard-of cannot be run locally"),
        (
            local("BAAI/bge-base-en-v1.5", 1024),
            "BAAI/bge-base-en-v1.5 has 768 dimensions, not 1024",
        ),
        (
            titan,
            "amazon.titan-embed-text-v2:0 has 256, 512 or 1024 dimensions, not 2",
        ),
    ] {
        assert_eq!(
            served.put(NOTES, notes(embedding.clone())).await,
            invalid(message),
            "{embedding}"
        );
    }
    let batched = json!({ "changes": [
        { "op": "create_collection", "schema": notes(supplied()) },
        { "op": "drop_collection", "collection": "notes" },
        { "op": "create_collection", "schema": notes(unheard_of.clone()) },
    ] });
    assert_eq!(
        served.post("/v1/tenants/acme/changes", batched).await,
        invalid("acme/unheard-of cannot be run locally"),
        "nor in a batch, none of which is made"
    );
    assert_eq!(
        served.get("/v1/tenants/acme/collections").await.1,
        json!({ "collections": [] })
    );

    // What is wrong with the request otherwise is what is said of it: the
    // model is looked at only for a collection that would be made.
    let mut undeclared = notes(unheard_of.clone());
    undeclared["embedding"]["properties"] = json!(["absent"]);
    let (status, body) = served.put(NOTES, undeclared).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    assert!(
        !body.to_string().contains("cannot be run locally"),
        "the schema's own fault comes first: {body}"
    );
    served.ok(Method::PUT, NOTES, Some(notes(supplied()))).await;
    let (status, body) = served.put(NOTES, notes(unheard_of.clone())).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::CONFLICT, "conflict"),
        "a collection there already, otherwise, is a clash: {body}"
    );

    // A request answered already is answered as it was, whatever it is
    // sent with the second time: even a collection that is not there,
    // with a model that could not be run, which is not looked at.
    let named = [(REQUEST, "6f1c2b9e-1d0a-4c55-9a3e-2f6c4d8b7a14")];
    let making = |name: &str, embedding: Value| {
        let mut schema = notes(embedding);
        schema["name"] = json!(name);
        Some(json!({ "changes": [{ "op": "create_collection", "schema": schema }] }))
    };
    let changes = "/v1/tenants/acme/changes";
    let made = served
        .call_headed(
            Some(KEY),
            Method::POST,
            changes,
            making("others", supplied()),
            &named,
        )
        .await;
    assert_eq!(made.0, StatusCode::OK, "{}", made.1);
    let again = served
        .call_headed(
            Some(KEY),
            Method::POST,
            changes,
            making("third", unheard_of.clone()),
            &named,
        )
        .await;
    assert_eq!(again, made);
    let unnamed = served
        .post(changes, making("third", unheard_of).unwrap())
        .await;
    assert_eq!(
        unnamed,
        invalid("acme/unheard-of cannot be run locally"),
        "sent as a request of its own, it is refused"
    );
}

#[tokio::test]
async fn a_node_set_up_wrong_for_a_model_says_so_and_one_set_up_right_answers() {
    let url = keyed_embedding_server("sk-right").await;
    let cluster = Cluster::new();
    let right = cluster.serve_embedding_keyed(&url, "sk-right").await;
    let wrong = cluster.serve_embedding_keyed(&url, "sk-wrong").await;
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": url,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    // The node set up right holds the tenant, and so makes its writes
    // whichever node they reach.
    right.with_notes(embedding).await;
    let fox = json!({ "properties": { "title": "a red fox" } });
    wrong.ok(Method::PUT, &object(1), Some(fox)).await;
    assert_eq!(right.get(&object(1)).await.1["vector"], json!([9.0, 3.0]));

    // A search is embedded for by the node it reaches.
    let nearest = json!({ "text": "a red fox", "limit": 1 });
    let found = right
        .ok(
            Method::POST,
            &format!("{NOTES}/nearest"),
            Some(nearest.clone()),
        )
        .await;
    assert_eq!(ids(&found), [id(1)]);
    assert_eq!(
        wrong.post(&format!("{NOTES}/nearest"), nearest).await,
        model_unavailable("setup")
    );
    let by_words = wrong
        .ok(
            Method::POST,
            &format!("{NOTES}/hybrid"),
            Some(json!({ "text": "fox", "limit": 1 })),
        )
        .await;
    assert_eq!(ids(&by_words), [id(1)]);
    assert_eq!(by_words["degraded"], true);
    assert_eq!(
        by_words["degraded_by"], "model_unavailable",
        "another node may answer in full"
    );
    let supplied = json!({ "vector": [9.0, 3.0], "limit": 1 });
    let found = wrong
        .ok(Method::POST, &format!("{NOTES}/nearest"), Some(supplied))
        .await;
    assert_eq!(
        ids(&found),
        [id(1)],
        "a search that brings its vector needs no model"
    );
}

#[tokio::test]
async fn a_write_the_holding_node_cannot_embed_is_refused_by_every_node_and_makes_nothing() {
    let url = keyed_embedding_server("sk-right").await;
    let cluster = Cluster::new();
    let wrong = cluster.serve_embedding_keyed(&url, "sk-wrong").await;
    let right = cluster.serve_embedding_keyed(&url, "sk-right").await;
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": url,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    wrong.with_notes(embedding).await;
    let fox = json!({ "properties": { "title": "a red fox" } });
    let named = [(REQUEST, "6f1c2b9e-1d0a-4c55-9a3e-2f6c4d8b7a13")];
    for node in [&wrong, &right, &wrong] {
        let refused = node
            .call_headed(
                Some(KEY),
                Method::PUT,
                &object(1),
                Some(fox.clone()),
                &named,
            )
            .await;
        assert_eq!(refused, model_unavailable("setup"));
    }
    assert_eq!(wrong.get(&object(1)).await.0, StatusCode::NOT_FOUND);
    let batched = json!({ "changes": [{ "op": "upsert", "collection": "notes", "id": id(1), "properties": { "title": "a red fox" } }] });
    assert_eq!(
        right.post("/v1/tenants/acme/changes", batched).await,
        model_unavailable("setup")
    );
    // Nothing of the refusal is kept: the same write, named the same,
    // is made once it brings what the node could not make.
    let supplied = json!({ "properties": { "title": "a red fox" }, "vector": [9.0, 3.0] });
    let made = right
        .call_headed(Some(KEY), Method::PUT, &object(1), Some(supplied), &named)
        .await;
    assert_eq!(made.0, StatusCode::OK, "{}", made.1);
    assert_eq!(wrong.get(&object(1)).await.0, StatusCode::OK);
}
