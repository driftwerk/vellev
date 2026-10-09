//! Every object read carries a version of its content, and a write may be
//! made only if an object is still at one.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use vellev_api::REQUEST;

fn version(body: &Value) -> String {
    let version = body["version"]
        .as_str()
        .unwrap_or_else(|| panic!("no version in {body}"));
    assert_eq!(version.len(), 32, "{version}");
    assert!(
        version
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{version}"
    );
    version.to_string()
}

/// A version no object has: that of content never written.
const STALE: &str = "00000000000000000000000000000000";

fn fox(n: i64) -> Value {
    json!({ "properties": { "title": "A red fox", "n": n }, "vector": [1.0, 0.0] })
}

#[tokio::test]
async fn every_object_read_carries_the_version_of_its_content() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(fox(1))).await;
    let hen = json!({ "properties": { "title": "a hen", "n": 2 }, "vector": [0.0, 1.0] });
    served.ok(Method::PUT, &object(2), Some(hen)).await;
    let got = served.get(&object(1)).await.1;
    let v1 = version(&got);
    assert_eq!(got["properties"], json!({ "title": "A red fox", "n": 1 }));

    // The same version wherever the object is read, with or without its
    // vector sent along.
    let list = |vectors: bool| json!({ "limit": 10, "vectors": vectors });
    for page in [list(false), list(true)] {
        let listed = served
            .ok(Method::POST, &format!("{NOTES}/list"), Some(page))
            .await;
        assert_eq!(version(&listed["objects"][0]), v1);
        assert_eq!(listed["objects"][0]["id"], json!(id(1)));
        assert_ne!(version(&listed["objects"][1]), v1);
    }
    let searches = [
        ("nearest", json!({ "vector": [1.0, 0.0], "limit": 1 })),
        ("keyword", json!({ "text": "fox", "limit": 1 })),
        (
            "hybrid",
            json!({ "text": "fox", "vector": [1.0, 0.0], "limit": 1 }),
        ),
    ];
    for (search, asked) in searches {
        let found = served
            .ok(Method::POST, &format!("{NOTES}/{search}"), Some(asked))
            .await;
        assert_eq!(found["results"][0]["id"], json!(id(1)), "{search}");
        assert_eq!(version(&found["results"][0]), v1, "{search}");
    }

    // The version is of the content alone: the same content under
    // another id has it, a vector-only change gives another, and the
    // content put back has the old one again.
    served.ok(Method::PUT, &object(3), Some(fox(1))).await;
    assert_eq!(version(&served.get(&object(3)).await.1), v1);
    served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": {}, "vector": [0.5, 0.5] })),
        )
        .await;
    let revectored = version(&served.get(&object(1)).await.1);
    assert_ne!(revectored, v1);
    served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": {}, "vector": [1.0, 0.0] })),
        )
        .await;
    assert_eq!(version(&served.get(&object(1)).await.1), v1);
    // A patch that changes nothing changes no version.
    let unchanged = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "n": 1 } })),
        )
        .await;
    assert_eq!(unchanged["outcomes"], json!([{ "outcome": "unchanged" }]));
    assert_eq!(version(&served.get(&object(1)).await.1), v1);
}

#[tokio::test]
async fn a_write_conditional_on_a_version_is_made_only_while_the_object_is_at_it() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(fox(1))).await;
    let v1 = version(&served.get(&object(1)).await.1);

    let patched = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "n": 2 }, "only_if_version": v1 })),
        )
        .await;
    assert_eq!(patched["outcomes"], json!([{ "outcome": "applied" }]));
    let v2 = version(&served.get(&object(1)).await.1);
    assert_ne!(v2, v1);

    // Refused at any other version, and told the one the object is at,
    // whatever the change.
    let stale_patch = json!({ "properties": { "n": 3 }, "only_if_version": v1 });
    let (status, body) = served.patch(&object(1), stale_patch).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v2));
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("version"));
    let mut stale_put = fox(3);
    stale_put["only_if_version"] = json!(v1);
    let (status, body) = served.put(&object(1), stale_put).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v2));
    let (status, body) = served
        .delete(&format!("{}?only_if_version={v1}", object(1)))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v2));
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 2);

    // Made at the version it is at, which the put then moves on.
    let mut put = fox(4);
    put["only_if_version"] = json!(v2);
    let replaced = served.ok(Method::PUT, &object(1), Some(put)).await;
    assert_eq!(replaced["outcomes"], json!([{ "outcome": "applied" }]));
    let v4 = version(&served.get(&object(1)).await.1);
    assert_ne!(v4, v2);
    let deleted = served
        .ok(
            Method::DELETE,
            &format!("{}?only_if_version={v4}", object(1)),
            None,
        )
        .await;
    assert_eq!(deleted["outcomes"], json!([{ "outcome": "applied" }]));
    assert_eq!(served.get(&object(1)).await.0, StatusCode::NOT_FOUND);

    // On an object that is not there, a conditional write is not found,
    // as a patch is; a plain delete still changes nothing.
    let mut put = fox(5);
    put["only_if_version"] = json!(v4);
    let (status, body) = served.put(&object(1), put).await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
    assert_eq!(body["error"]["missing"], json!("object"));
    let (status, body) = served
        .patch(
            &object(1),
            json!({ "properties": { "n": 5 }, "only_if_version": v4 }),
        )
        .await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
    assert_eq!(body["error"]["missing"], json!("object"));
    let (status, body) = served
        .delete(&format!("{}?only_if_version={v4}", object(1)))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::NOT_FOUND, "not_found"));
    assert_eq!(body["error"]["missing"], json!("object"));
    let again = served.ok(Method::DELETE, &object(1), None).await;
    assert_eq!(again["outcomes"], json!([{ "outcome": "unchanged" }]));
    assert_eq!(served.get(&object(1)).await.0, StatusCode::NOT_FOUND);

    // A version is 32 hexadecimal characters, wherever it is sent, and
    // only a replace, a patch or a delete takes one.
    served.ok(Method::PUT, &object(2), Some(fox(2))).await;
    for malformed in ["", "abc", "zz000000000000000000000000000000"] {
        let (status, body) = served
            .patch(
                &object(2),
                json!({ "properties": { "n": 3 }, "only_if_version": malformed }),
            )
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{malformed:?}"
        );
        let (status, body) = served
            .delete(&format!("{}?only_if_version={malformed}", object(2)))
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::BAD_REQUEST, "invalid"),
            "{malformed:?}"
        );
    }
    let (status, body) = served
        .delete(&format!("{}?only_if_versoin={STALE}", object(2)))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    let insert = json!({ "changes": [{
        "op": "insert", "collection": "notes", "id": id(9),
        "properties": { "n": 9 }, "only_if_version": STALE,
    }] });
    let (status, body) = served.post("/v1/tenants/acme/changes", insert).await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
    assert_eq!(served.get(&object(2)).await.1["properties"]["n"], 2);
}

#[tokio::test]
async fn a_property_condition_and_a_version_condition_must_both_hold() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(fox(1))).await;
    let v1 = version(&served.get(&object(1)).await.1);
    let patch = |n: i64, equals: i64, only_if_version: &str| {
        json!({
            "properties": { "n": n },
            "only_if": { "property": "n", "equals": equals },
            "only_if_version": only_if_version,
        })
    };
    // The property fails while the version holds, and the other way
    // about: refused either way, and told the version.
    let (status, body) = served.patch(&object(1), patch(5, 2, &v1)).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v1));
    assert!(body["error"]["message"].as_str().unwrap().contains("\"n\""));
    let (status, body) = served.patch(&object(1), patch(5, 1, STALE)).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v1));
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 1);
    // Both hold: a patch that changes nothing changes nothing, and one
    // that does is made.
    let unchanged = served
        .ok(Method::PATCH, &object(1), Some(patch(1, 1, &v1)))
        .await;
    assert_eq!(unchanged["outcomes"], json!([{ "outcome": "unchanged" }]));
    assert_eq!(version(&served.get(&object(1)).await.1), v1);
    let applied = served
        .ok(Method::PATCH, &object(1), Some(patch(2, 1, &v1)))
        .await;
    assert_eq!(applied["outcomes"], json!([{ "outcome": "applied" }]));
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 2);
    // The property condition alone is still told the version.
    let (status, body) = served
        .patch(
            &object(1),
            json!({ "properties": { "n": 3 }, "only_if": { "property": "n", "equals": 1 } }),
        )
        .await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_ne!(body["error"]["version"], json!(v1));
    assert_eq!(
        body["error"]["version"],
        json!(version(&served.get(&object(1)).await.1))
    );
}

#[tokio::test]
async fn a_batch_with_one_stale_version_is_refused_whole_and_alike_when_named_again() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(fox(1))).await;
    served.ok(Method::PUT, &object(2), Some(fox(2))).await;
    let v1 = version(&served.get(&object(1)).await.1);
    let v2 = version(&served.get(&object(2)).await.1);
    let batch = |patch_at: &str| {
        json!({ "changes": [
            { "op": "upsert", "collection": "notes", "id": id(3), "properties": { "n": 3 } },
            { "op": "patch", "collection": "notes", "id": id(1), "properties": { "n": 10 }, "only_if_version": patch_at },
            { "op": "delete", "collection": "notes", "id": id(2), "only_if_version": v2 },
        ] })
    };
    let request = uuid::Uuid::new_v4().to_string();
    let refused = || async {
        served
            .call_headed(
                Some(KEY),
                Method::POST,
                "/v1/tenants/acme/changes",
                Some(batch(STALE)),
                &[(REQUEST, request.as_str())],
            )
            .await
    };
    let (status, body) = refused().await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v1));
    assert_eq!(served.get(&object(3)).await.0, StatusCode::NOT_FOUND);
    assert_eq!(served.get(&object(2)).await.0, StatusCode::OK);
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 1);
    // Named again, it is answered from what was kept of it: refused, with
    // the version it was refused at.
    served.ok(Method::PUT, &object(1), Some(fox(7))).await;
    let (status, again) = refused().await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(again, body);

    // A stale version on the upsert alone, and on the delete alone,
    // refuses the batch just the same.
    let v1 = version(&served.get(&object(1)).await.1);
    served.ok(Method::PUT, &object(3), Some(fox(3))).await;
    let v3 = version(&served.get(&object(3)).await.1);
    let one_stale = |stale_on: &str| {
        let at = |op: &str, fresh: &str| if op == stale_on { STALE } else { fresh }.to_string();
        json!({ "changes": [
            { "op": "upsert", "collection": "notes", "id": id(3), "properties": { "n": 30 }, "only_if_version": at("upsert", &v3) },
            { "op": "patch", "collection": "notes", "id": id(1), "properties": { "n": 10 }, "only_if_version": at("patch", &v1) },
            { "op": "delete", "collection": "notes", "id": id(2), "only_if_version": at("delete", &v2) },
        ] })
    };
    for (stale_on, stored) in [("upsert", &v3), ("delete", &v2)] {
        let (status, body) = served
            .post("/v1/tenants/acme/changes", one_stale(stale_on))
            .await;
        assert_eq!(
            (status, code(&body)),
            (StatusCode::CONFLICT, "conflict"),
            "{stale_on}"
        );
        assert_eq!(body["error"]["version"], json!(stored), "{stale_on}");
        assert_eq!(served.get(&object(3)).await.1["properties"]["n"], 3);
        assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 7);
        assert_eq!(served.get(&object(2)).await.0, StatusCode::OK);
    }
    served.ok(Method::DELETE, &object(3), None).await;
    let made = served
        .ok(Method::POST, "/v1/tenants/acme/changes", Some(batch(&v1)))
        .await;
    assert_eq!(
        made["outcomes"],
        json!([{ "outcome": "applied" }, { "outcome": "applied" }, { "outcome": "applied" }])
    );
    assert_eq!(served.get(&object(1)).await.1["properties"]["n"], 10);
    assert_eq!(served.get(&object(2)).await.0, StatusCode::NOT_FOUND);
    assert_eq!(served.get(&object(3)).await.0, StatusCode::OK);
}

#[tokio::test]
async fn a_version_is_the_same_once_the_object_is_flushed_to_a_segment() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(fox(1))).await;
    let hen = json!({ "properties": { "title": "a hen", "n": 2 } });
    served.ok(Method::PUT, &object(2), Some(hen)).await;
    async fn read(served: &Served) -> Vec<String> {
        let mut found = vec![
            version(&served.get(&object(1)).await.1),
            version(&served.get(&object(2)).await.1),
        ];
        let listed = served
            .ok(
                Method::POST,
                &format!("{NOTES}/list"),
                Some(json!({ "limit": 10 })),
            )
            .await;
        found.extend(listed["objects"].as_array().unwrap().iter().map(version));
        let near = served
            .ok(
                Method::POST,
                &format!("{NOTES}/nearest"),
                Some(json!({ "vector": [1.0, 0.0], "limit": 1 })),
            )
            .await;
        found.push(version(&near["results"][0]));
        found
    }
    let before = read(&served).await;
    assert_eq!(before[0], before[2]);
    assert_eq!(before[0], before[4]);
    assert_eq!(before[1], before[3]);
    let acme = vellev_core::TenantId::new("acme").unwrap();
    served
        .node
        .tenant(&acme)
        .await
        .unwrap()
        .flush()
        .await
        .unwrap();
    // A row in a segment keeps its vector apart from the object: its
    // version is of both still, for the one with a vector and the one
    // without, and a write at it is made.
    assert_eq!(read(&served).await, before);
    let moved = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "n": 5 }, "only_if_version": before[0] })),
        )
        .await;
    assert_eq!(moved["outcomes"], json!([{ "outcome": "applied" }]));
    let (status, body) = served
        .delete(&format!("{}?only_if_version={}", object(2), before[0]))
        .await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(before[1]));
}

#[tokio::test]
async fn a_version_covers_the_vector_a_node_embeds_and_a_conditional_write_passes_at_it() {
    let url = keyed_embedding_server("sk-right").await;
    let served = Cluster::new().serve_embedding_keyed(&url, "sk-right").await;
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": url,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    served.with_notes(embedding).await;
    let fox = json!({ "properties": { "title": "a red fox", "n": 1 } });
    served.ok(Method::PUT, &object(1), Some(fox)).await;
    let got = served.get(&object(1)).await.1;
    assert_eq!(got["vector"], json!([9.0, 3.0]), "embedded by the node");
    let v1 = version(&got);
    // The same text keeps its vector, so a patch of another property at
    // the version is made and moves it; a patch that changes the text
    // changes the vector, and so the version, though nothing else of
    // the object the client sent did.
    let moved = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "n": 2 }, "only_if_version": v1 })),
        )
        .await;
    assert_eq!(moved["outcomes"], json!([{ "outcome": "applied" }]));
    let v2 = version(&served.get(&object(1)).await.1);
    assert_ne!(v2, v1);
    let (status, body) = served
        .patch(
            &object(1),
            json!({ "properties": { "n": 3 }, "only_if_version": v1 }),
        )
        .await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(body["error"]["version"], json!(v2));
    let retitled = served
        .ok(
            Method::PATCH,
            &object(1),
            Some(json!({ "properties": { "title": "a hen" }, "only_if_version": v2 })),
        )
        .await;
    assert_eq!(retitled["outcomes"], json!([{ "outcome": "applied" }]));
    let got = served.get(&object(1)).await.1;
    assert_eq!(got["vector"], json!([5.0, 2.0]));
    let v3 = version(&got);
    assert_ne!(v3, v2);
    let deleted = served
        .ok(
            Method::POST,
            "/v1/tenants/acme/changes",
            Some(json!({ "changes": [
                { "op": "delete", "collection": "notes", "id": id(1), "only_if_version": v3 },
            ] })),
        )
        .await;
    assert_eq!(deleted["outcomes"], json!([{ "outcome": "applied" }]));
}
