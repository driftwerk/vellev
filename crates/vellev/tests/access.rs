//! A collection's access rule, applied to every read made with a tenant's
//! key on behalf of the person an identity token names.

mod common;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use common::*;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use reqwest::{Method, StatusCode};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use serde_json::{json, Value};
use vellev_api::IDENTITY;

const MEMORIES: &str = "/v1/tenants/acme/collections/memories";

/// A key that signs identity tokens, and how a node is told to trust it.
struct Signer {
    kid: String,
    key: EncodingKey,
    trusted: String,
}

impl Signer {
    fn new(kid: &str) -> Self {
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).unwrap();
        let pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let x = URL_SAFE_NO_PAD.encode(pair.public_key().as_ref());
        Self {
            kid: kid.into(),
            key: EncodingKey::from_ed_der(pkcs8.as_ref()),
            trusted: format!("{kid}={x}"),
        }
    }

    /// A token for `identity` in `teams`, of `tenant`, expiring `in_secs`
    /// from now (or before, if negative).
    fn token(&self, tenant: &str, identity: &str, teams: &[&str], in_secs: i64) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let claims = json!({
            "tenant": tenant,
            "identity_id": identity,
            "teams": teams,
            "iat": now,
            "exp": now + in_secs,
        });
        let header = Header {
            kid: Some(self.kid.clone()),
            ..Header::new(Algorithm::EdDSA)
        };
        jsonwebtoken::encode(&header, &claims, &self.key).unwrap()
    }
}

fn memories() -> Value {
    json!({
        "name": "memories",
        "properties": [
            { "name": "title", "data_type": "text", "searchable": true },
            { "name": "owner", "data_type": "text" },
            { "name": "visibility", "data_type": "text" },
        ],
        "embedding": supplied(),
        "distance": "cosine",
        "access_rule": { "owner": "owner", "team": "visibility", "team_prefix": "team:" },
    })
}

/// Memories: one of ada's own, one of bob's own, one shared with team
/// core and one with team ops; and a key for acme.
async fn ruled(trusted: &[String]) -> (Served, String) {
    let served = Cluster::new().serve_trusting(trusted).await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    served.ok(Method::PUT, MEMORIES, Some(memories())).await;
    let rows = [
        (1, "ada's own note", "ada", "private"),
        (2, "bob's own note", "bob", "private"),
        (3, "core team note", "carl", "team:core"),
        (4, "ops team note", "dee", "team:ops"),
    ];
    for (n, title, owner, visibility) in rows {
        let object = json!({
            "properties": { "title": title, "owner": owner, "visibility": visibility },
            "vector": [1.0, n as f32],
        });
        served
            .ok(
                Method::PUT,
                &format!("{MEMORIES}/objects/{}", id(n)),
                Some(object),
            )
            .await;
    }
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    (served, issued["key"].as_str().unwrap().to_string())
}

/// The titles each kind of read finds, as one caller sees them.
async fn seen(served: &Served, key: &str, token: Option<&str>) -> Vec<(String, Value)> {
    let headers: Vec<(&str, &str)> = token.map(|token| (IDENTITY, token)).into_iter().collect();
    let ask = |method: Method, path: String, body: Option<Value>| {
        let headers = headers.clone();
        async move {
            served
                .call_headed(Some(key), method, &path, body, &headers)
                .await
        }
    };
    let titles = |answer: &Value, field: &str| {
        let mut titles: Vec<String> = answer[field]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|found| found["properties"]["title"].as_str().unwrap().to_string())
            .collect();
        titles.sort();
        json!(titles)
    };
    let searched = json!({ "vector": [1.0, 2.0], "limit": 10 });
    let mut seen = Vec::new();
    let (_, listed) = ask(
        Method::POST,
        format!("{MEMORIES}/list"),
        Some(json!({ "limit": 10 })),
    )
    .await;
    seen.push(("list".into(), titles(&listed, "objects")));
    let (_, near) = ask(Method::POST, format!("{MEMORIES}/nearest"), Some(searched)).await;
    seen.push(("nearest".into(), titles(&near, "results")));
    let words = json!({ "text": "note", "limit": 10 });
    let (_, found) = ask(Method::POST, format!("{MEMORIES}/keyword"), Some(words)).await;
    seen.push(("keyword".into(), titles(&found, "results")));
    let hybrid = json!({ "text": "note", "vector": [1.0, 2.0], "limit": 10 });
    let (_, fused) = ask(Method::POST, format!("{MEMORIES}/hybrid"), Some(hybrid)).await;
    seen.push(("hybrid".into(), titles(&fused, "results")));
    let (_, counted) = ask(Method::POST, format!("{MEMORIES}/count"), Some(json!({}))).await;
    seen.push(("count".into(), counted["count"].clone()));
    let mut fetched = Vec::new();
    for n in 1..=4 {
        let (status, _) = ask(Method::GET, format!("{MEMORIES}/objects/{}", id(n)), None).await;
        if status == StatusCode::OK {
            fetched.push(n);
        }
    }
    seen.push(("get".into(), json!(fetched)));
    seen
}

fn every_read(titles: &[&str], ids: &[u64]) -> Vec<(String, Value)> {
    let mut titles: Vec<&str> = titles.to_vec();
    titles.sort();
    ["list", "nearest", "keyword", "hybrid"]
        .iter()
        .map(|read| (read.to_string(), json!(titles)))
        .chain([
            ("count".to_string(), json!(titles.len())),
            ("get".to_string(), json!(ids)),
        ])
        .collect()
}

#[tokio::test]
async fn a_person_reads_what_is_theirs_or_their_teams_and_nothing_else() {
    let signer = Signer::new("hub-1");
    let (served, key) = ruled(std::slice::from_ref(&signer.trusted)).await;

    let ada = signer.token("acme", "ada", &["core"], 300);
    assert_eq!(
        seen(&served, &key, Some(&ada)).await,
        every_read(&["ada's own note", "core team note"], &[1, 3])
    );
    let bob = signer.token("acme", "bob", &["ops", "sales"], 300);
    assert_eq!(
        seen(&served, &key, Some(&bob)).await,
        every_read(&["bob's own note", "ops team note"], &[2, 4])
    );
    assert_eq!(
        seen(&served, &key, None).await,
        every_read(&[], &[]),
        "a tenant's key naming nobody reads nothing of a ruled collection"
    );
    let (status, everything) = served
        .call(Method::POST, &format!("{MEMORIES}/count"), Some(json!({})))
        .await;
    assert_eq!(
        (status, everything["count"].clone()),
        (StatusCode::OK, json!(4)),
        "an admin key reads all"
    );
}

#[tokio::test]
async fn a_filter_cannot_widen_what_the_rule_allows() {
    let signer = Signer::new("hub-1");
    let (served, key) = ruled(std::slice::from_ref(&signer.trusted)).await;
    let ada = signer.token("acme", "ada", &[], 300);
    let widening = json!({ "limit": 10, "filter": { "or": [
        { "eq": { "property": "owner", "value": "bob" } },
        { "not": { "eq": { "property": "owner", "value": "nobody" } } }
    ] } });
    let (status, listed) = served
        .call_headed(
            Some(&key),
            Method::POST,
            &format!("{MEMORIES}/list"),
            Some(widening),
            &[(IDENTITY, &ada)],
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let titles: Vec<&str> = listed["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|found| found["properties"]["title"].as_str().unwrap())
        .collect();
    assert_eq!(titles, ["ada's own note"]);
}

#[tokio::test]
async fn a_token_not_trusted_not_current_or_not_for_this_tenant_is_refused() {
    let (signer, stranger) = (Signer::new("hub-1"), Signer::new("hub-1"));
    let (served, key) = ruled(std::slice::from_ref(&signer.trusted)).await;
    for (why, token) in [
        ("expired", signer.token("acme", "ada", &[], -120)),
        ("for another tenant", signer.token("beta", "ada", &[], 300)),
        (
            "signed by another key of the same id",
            stranger.token("acme", "ada", &[], 300),
        ),
        (
            "signed by a key id not trusted",
            Signer::new("other").token("acme", "ada", &[], 300),
        ),
        ("not a token", "not.a.token".to_string()),
    ] {
        let (status, body) = served
            .call_headed(
                Some(&key),
                Method::POST,
                &format!("{MEMORIES}/count"),
                Some(json!({})),
                &[(IDENTITY, &token)],
            )
            .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{why}: {body}");
        assert_eq!(code(&body), "unauthorized", "{why}");
    }
}

#[tokio::test]
async fn a_key_is_rotated_by_trusting_the_new_one_beside_the_old() {
    let (old, new) = (Signer::new("hub-1"), Signer::new("hub-2"));
    let (served, key) = ruled(&[old.trusted.clone(), new.trusted.clone()]).await;
    for signer in [&old, &new] {
        let token = signer.token("acme", "ada", &[], 300);
        let (status, counted) = served
            .call_headed(
                Some(&key),
                Method::POST,
                &format!("{MEMORIES}/count"),
                Some(json!({})),
                &[(IDENTITY, &token)],
            )
            .await;
        assert_eq!(
            (status, counted["count"].clone()),
            (StatusCode::OK, json!(1))
        );
    }
}
