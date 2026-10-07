//! The walls between tenants, tried route by route: every route the API
//! describes is asked with no key, a wrong key, a key of another tenant and
//! a tenant's key where only an admin's will do, and must refuse each
//! without saying anything of what lies behind it. Inside a tenant, names
//! and ids that exist only in another are answered as if they existed
//! nowhere.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

const OTHER_OBJECT: u64 = 7;

/// Every operation the API describes, as (method, path template).
fn operations() -> Vec<(Method, String)> {
    let described: Value = serde_json::to_value(vellev::server::openapi()).unwrap();
    described["paths"]
        .as_object()
        .unwrap()
        .iter()
        .flat_map(|(path, methods)| {
            methods.as_object().unwrap().keys().map(move |method| {
                (
                    Method::from_bytes(method.to_uppercase().as_bytes()).unwrap(),
                    path.clone(),
                )
            })
        })
        .collect()
}

fn concrete(path: &str, tenant: &str, collection: &str, id: &str) -> String {
    path.replace("{tenant}", tenant)
        .replace("{collection}", collection)
        .replace("{id}", id)
}

/// Two tenants: acme with notes, beta with notes and a collection and an
/// object acme has nothing of; and a key for acme alone.
async fn walled() -> (Served, String) {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, "/v1/tenants/beta", None).await;
    for collection in ["notes", "secret"] {
        let mut schema = notes(supplied());
        schema["name"] = json!(collection);
        served
            .ok(
                Method::PUT,
                &format!("/v1/tenants/beta/collections/{collection}"),
                Some(schema),
            )
            .await;
    }
    let object = json!({ "properties": { "title": "beta's own", "n": 1 }, "vector": [1.0, 0.0] });
    served
        .ok(
            Method::PUT,
            &format!(
                "/v1/tenants/beta/collections/notes/objects/{}",
                id(OTHER_OBJECT)
            ),
            Some(object),
        )
        .await;
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let key = issued["key"].as_str().unwrap().to_string();
    (served, key)
}

#[tokio::test]
async fn no_route_is_reached_without_a_key_that_reaches_it() {
    let (served, _) = walled().await;
    let guessed = format!("vlv_{}", "a".repeat(64));
    for (method, path) in operations() {
        let path = concrete(&path, "acme", "notes", &id(1));
        for key in [None, Some("wrong"), Some(guessed.as_str()), Some("")] {
            let (status, body) = served.call_as(key, method.clone(), &path, None).await;
            assert_eq!(
                status,
                StatusCode::UNAUTHORIZED,
                "{method} {path} with {key:?}: {body}"
            );
            assert_eq!(code(&body), "unauthorized");
        }
    }
}

#[tokio::test]
async fn a_tenant_key_reaches_no_other_tenant_and_learns_nothing_of_it() {
    let (served, key) = walled().await;
    let admin_only = vellev::server::ADMIN_ONLY;
    let mut compared = 0;
    for (method, template) in operations() {
        if admin_only.contains(&template.as_str()) {
            for tenant in ["acme", "beta", "gamma"] {
                let path = concrete(&template, tenant, "notes", &id(1));
                let (status, body) = served
                    .call_as(Some(&key), method.clone(), &path, None)
                    .await;
                assert_eq!(status, StatusCode::FORBIDDEN, "{method} {path}: {body}");
                assert_eq!(
                    body["error"]["message"],
                    json!("only an admin key may do this"),
                    "says the same whoever the tenant is: {method} {path}"
                );
            }
            continue;
        }
        // Beta exists and gamma does not: a key for acme cannot tell.
        let path = |tenant: &str| concrete(&template, tenant, "notes", &id(OTHER_OBJECT));
        let (beta_path, gamma_path) = (path("beta"), path("gamma"));
        let (beta_status, beta) = served
            .call_as(Some(&key), method.clone(), &beta_path, None)
            .await;
        let (gamma_status, gamma) = served
            .call_as(Some(&key), method.clone(), &gamma_path, None)
            .await;
        assert_eq!(
            beta_status,
            StatusCode::NOT_FOUND,
            "{method} {template}: {beta}"
        );
        assert_eq!(
            gamma_status,
            StatusCode::NOT_FOUND,
            "{method} {template}: {gamma}"
        );
        assert_eq!(
            beta["error"]["message"]
                .as_str()
                .unwrap()
                .replace("beta", "gamma"),
            gamma["error"]["message"].as_str().unwrap(),
            "{method} {template}"
        );
        // And as a tenant that does not exist is to anyone, where the
        // request, sent without a body, gets as far as asking.
        let (status, truly) = served.call(method.clone(), &gamma_path, None).await;
        if status == StatusCode::NOT_FOUND {
            assert_eq!(
                gamma["error"]["message"], truly["error"]["message"],
                "{method} {template}"
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 4,
        "compared with a missing tenant on {compared} routes"
    );
    let stored = served
        .get(&format!(
            "/v1/tenants/beta/collections/notes/objects/{}",
            id(OTHER_OBJECT)
        ))
        .await;
    assert_eq!(
        stored.1["properties"]["title"],
        json!("beta's own"),
        "nothing of beta changed"
    );
}

#[tokio::test]
async fn names_and_ids_of_another_tenant_are_nothing_in_ones_own() {
    let (served, key) = walled().await;
    let as_acme = |path: String| {
        let (served, key) = (&served, &key);
        async move { served.call_as(Some(key), Method::GET, &path, None).await }
    };
    let (status, secret) = as_acme("/v1/tenants/acme/collections/secret".into()).await;
    let (_, made_up) = as_acme("/v1/tenants/acme/collections/madeup".into()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        secret["error"]["message"]
            .as_str()
            .unwrap()
            .replace("secret", "madeup"),
        made_up["error"]["message"].as_str().unwrap()
    );
    let (status, theirs) = as_acme(format!(
        "/v1/tenants/acme/collections/notes/objects/{}",
        id(OTHER_OBJECT)
    ))
    .await;
    let (_, nobodys) = as_acme(format!(
        "/v1/tenants/acme/collections/notes/objects/{}",
        id(999)
    ))
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        theirs["error"]["message"]
            .as_str()
            .unwrap()
            .replace(&id(OTHER_OBJECT), &id(999)),
        nobodys["error"]["message"].as_str().unwrap()
    );
    let listed = served
        .call_as(
            Some(&key),
            Method::POST,
            "/v1/tenants/acme/collections/notes/list",
            Some(json!({ "limit": MAX_LIMIT })),
        )
        .await;
    assert_eq!(listed.0, StatusCode::OK);
    assert!(
        listed.1["objects"].as_array().unwrap().is_empty(),
        "{}",
        listed.1
    );
    let found = served
        .call_as(
            Some(&key),
            Method::POST,
            "/v1/tenants/acme/collections/notes/nearest",
            Some(json!({ "vector": [1.0, 0.0], "limit": 10 })),
        )
        .await;
    assert_eq!(found.0, StatusCode::OK, "{}", found.1);
    assert!(
        found.1["results"].as_array().unwrap().is_empty(),
        "{}",
        found.1
    );
}
