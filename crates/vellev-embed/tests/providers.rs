use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_sdk_bedrockruntime::config::retry::RetryConfig;
use aws_sdk_bedrockruntime::config::{BehaviorVersion, Credentials, Region};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use vellev_core::HttpApi;
use vellev_embed::{Bedrock, EmbedError, Http, Provider};

/// What a stand-in server was sent, and how many of its first requests it
/// is to refuse.
#[derive(Default)]
struct Seen {
    bodies: Mutex<Vec<Value>>,
    authorizations: Mutex<Vec<Option<String>>>,
    refusals: AtomicUsize,
    refuse_with: AtomicUsize,
    in_flight: AtomicUsize,
    most_in_flight: AtomicUsize,
}

impl Seen {
    fn refusing(times: usize, status: u16) -> Arc<Self> {
        Arc::new(Self {
            refusals: AtomicUsize::new(times),
            refuse_with: AtomicUsize::new(status as usize),
            ..Self::default()
        })
    }

    /// Record a request; the refusal to answer it with, if it is one of
    /// those to be refused.
    fn record(&self, headers: &HeaderMap, body: &Value) -> Option<(StatusCode, String)> {
        let authorization = headers
            .get("authorization")
            .map(|value| value.to_str().unwrap().to_string());
        self.bodies.lock().unwrap().push(body.clone());
        self.authorizations
            .lock()
            .unwrap()
            .push(authorization.clone());
        let refuse = self
            .refusals
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok();
        refuse.then(|| {
            let status = StatusCode::from_u16(self.refuse_with.load(Ordering::SeqCst) as u16);
            (
                status.unwrap(),
                format!("refused {}", authorization.unwrap_or_default()),
            )
        })
    }

    fn bodies(&self) -> Vec<Value> {
        self.bodies.lock().unwrap().clone()
    }
}

async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

fn numbered(count: usize) -> Vec<String> {
    (0..count).map(|n| format!("text {n}")).collect()
}

/// The number at the end of a text, as its one-dimensional vector.
fn vector_of(text: &str) -> Vec<f32> {
    vec![text.rsplit(' ').next().unwrap().parse().unwrap()]
}

async fn tei(seen: Arc<Seen>) -> String {
    let embed = |State(seen): State<Arc<Seen>>, headers: HeaderMap, Json(body): Json<Value>| async move {
        if let Some(refusal) = seen.record(&headers, &body) {
            return Err(refusal);
        }
        let inputs = body["inputs"].as_array().unwrap();
        Ok(Json(json!(inputs
            .iter()
            .map(|text| vector_of(text.as_str().unwrap()))
            .collect::<Vec<_>>())))
    };
    serve(Router::new().route("/embed", post(embed)).with_state(seen)).await
}

async fn openai(seen: Arc<Seen>) -> String {
    let embed = |State(seen): State<Arc<Seen>>, headers: HeaderMap, Json(body): Json<Value>| async move {
        if let Some(refusal) = seen.record(&headers, &body) {
            return Err(refusal);
        }
        let inputs = body["input"].as_array().unwrap();
        let data: Vec<Value> = inputs
            .iter()
            .enumerate()
            .rev()
            .map(|(index, text)| json!({ "index": index, "embedding": vector_of(text.as_str().unwrap()) }))
            .collect();
        Ok(Json(json!({ "data": data })))
    };
    serve(
        Router::new()
            .route("/v1/embeddings", post(embed))
            .with_state(seen),
    )
    .await
}

#[tokio::test]
async fn a_tei_server_is_sent_batches_and_answers_in_order() {
    let seen = Arc::new(Seen::default());
    let url = tei(seen.clone()).await;
    let provider = Http::new(&format!("{url}/"), HttpApi::Tei, "bge", None);
    let vectors = provider.embed(numbered(70)).await.unwrap();
    assert_eq!(vectors, (0..70).map(|n| vec![n as f32]).collect::<Vec<_>>());

    let bodies = seen.bodies();
    let sizes: Vec<usize> = bodies
        .iter()
        .map(|body| body["inputs"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, [32, 32, 6]);
    assert_eq!(
        bodies[2],
        json!({ "inputs": numbered(70)[64..], "truncate": true })
    );
    assert_eq!(*seen.authorizations.lock().unwrap(), [None, None, None]);
}

#[tokio::test]
async fn an_openai_endpoint_is_sent_the_model_and_key_and_answers_in_any_order() {
    let seen = Arc::new(Seen::default());
    let url = openai(seen.clone()).await;
    let provider = Http::new(
        &url,
        HttpApi::Openai,
        "text-embedding-3-small",
        Some("sk-secret".into()),
    );
    let vectors = provider.embed(numbered(3)).await.unwrap();
    assert_eq!(vectors, [vec![0.0], vec![1.0], vec![2.0]]);
    assert_eq!(
        seen.bodies(),
        [json!({ "model": "text-embedding-3-small", "input": numbered(3) })]
    );
    assert_eq!(
        *seen.authorizations.lock().unwrap(),
        [Some("Bearer sk-secret".to_string())]
    );
}

#[tokio::test]
async fn throttling_and_server_errors_are_tried_again_but_a_refusal_is_final() {
    for status in [429, 500, 503] {
        let seen = Seen::refusing(2, status);
        let provider = Http::new(&tei(seen.clone()).await, HttpApi::Tei, "bge", None);
        assert_eq!(
            provider.embed(numbered(1)).await.unwrap(),
            [vec![0.0]],
            "{status}"
        );
        assert_eq!(seen.bodies().len(), 3);
    }

    let seen = Seen::refusing(usize::MAX, 400);
    let provider = Http::new(
        &openai(seen.clone()).await,
        HttpApi::Openai,
        "m",
        Some("sk-secret".into()),
    );
    let refused = provider.embed(numbered(1)).await.unwrap_err();
    assert_eq!(
        refused,
        EmbedError::Failed {
            model: "m".into(),
            reason: "400 Bad Request: refused Bearer <redacted>".into()
        }
    );
    assert_eq!(seen.bodies().len(), 1);
}

#[tokio::test]
async fn a_server_that_never_recovers_or_cannot_be_reached_fails_after_four_tries() {
    let seen = Seen::refusing(usize::MAX, 500);
    let provider = Http::new(&tei(seen.clone()).await, HttpApi::Tei, "bge", None);
    let failed = provider.embed(numbered(1)).await.unwrap_err();
    assert!(
        matches!(&failed, EmbedError::Failed { reason, .. } if reason.starts_with("500")),
        "{failed}"
    );
    assert_eq!(seen.bodies().len(), 4);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unreachable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let provider = Http::new(&unreachable, HttpApi::Tei, "bge", None);
    assert!(matches!(
        provider.embed(numbered(1)).await,
        Err(EmbedError::Failed { .. })
    ));
}

#[tokio::test]
async fn an_answer_that_is_not_embeddings_is_malformed() {
    let router = Router::new()
        .route(
            "/embed",
            post(|| async { Json(json!({ "not": "vectors" })) }),
        )
        .route(
            "/v1/embeddings",
            post(|| async { Json(json!({ "data": [] })) }),
        );
    let url = serve(router).await;
    for api in [HttpApi::Tei, HttpApi::Openai] {
        let failed = Http::new(&url, api, "m", None)
            .embed(numbered(1))
            .await
            .unwrap_err();
        assert!(
            matches!(failed, EmbedError::Malformed { .. }),
            "{api:?}: {failed}"
        );
    }
}

const TITAN: &str = "amazon.titan-embed-text-v2:0";

async fn bedrock(seen: Arc<Seen>) -> Bedrock {
    let invoke = |State(seen): State<Arc<Seen>>,
                  Path(model): Path<String>,
                  headers: HeaderMap,
                  Json(body): Json<Value>| async move {
        assert_eq!(model, TITAN);
        let now = seen.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        seen.most_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(20)).await;
        seen.in_flight.fetch_sub(1, Ordering::SeqCst);
        if let Some((status, _)) = seen.record(&headers, &body) {
            let kind = match status.as_u16() {
                429 => "ThrottlingException",
                400 => "ValidationException",
                _ => "InternalServerException",
            };
            let refusal = Json(json!({ "message": "not now" }));
            return Err((status, [("x-amzn-errortype", kind)], refusal));
        }
        let text = body["inputText"].as_str().unwrap();
        Ok(Json(
            json!({ "embedding": [text.chars().count(), 1], "inputTextTokenCount": 3 }),
        ))
    };
    let router = Router::new()
        .route("/model/{model}/invoke", post(invoke))
        .with_state(seen);
    let retries = RetryConfig::standard()
        .with_max_attempts(vellev_embed::BEDROCK_ATTEMPTS)
        .with_initial_backoff(Duration::from_millis(5));
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("eu-west-2"))
        .credentials_provider(Credentials::new(
            "AKIDEXAMPLE",
            "secret",
            None,
            None,
            "test",
        ))
        .endpoint_url(serve(router).await)
        .retry_config(retries)
        .build();
    Bedrock::with_client(aws_sdk_bedrockruntime::Client::from_conf(config), TITAN, 2)
}

#[tokio::test]
async fn titan_is_sent_one_signed_normalised_text_a_call_with_few_in_flight() {
    let seen = Arc::new(Seen::default());
    let provider = bedrock(seen.clone()).await;
    let texts: Vec<String> = (1..=40).map(|n| "x".repeat(n)).collect();
    let vectors = provider.embed(texts).await.unwrap();
    assert_eq!(
        vectors,
        (1..=40).map(|n| vec![n as f32, 1.0]).collect::<Vec<_>>()
    );

    let bodies = seen.bodies();
    assert_eq!(bodies.len(), 40);
    assert!(bodies.contains(&json!({ "inputText": "xxx", "dimensions": 2, "normalize": true })));
    let most = seen.most_in_flight.load(Ordering::SeqCst);
    assert!((2..=10).contains(&most), "{most} calls in flight");
    assert!(seen
        .authorizations
        .lock()
        .unwrap()
        .iter()
        .all(|authorization| {
            authorization
                .as_deref()
                .is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"))
        }));
}

#[tokio::test]
async fn titan_is_tried_again_when_throttled_and_long_text_is_cut() {
    for status in [429, 500, 503] {
        let seen = Seen::refusing(2, status);
        let provider = bedrock(seen.clone()).await;
        let vectors = provider.embed(vec!["é".repeat(40_000)]).await.unwrap();
        assert_eq!(vectors, [vec![30_000.0, 1.0]], "{status}");
        assert_eq!(seen.bodies().len(), 3);
    }

    let seen = Seen::refusing(usize::MAX, 400);
    let provider = bedrock(seen.clone()).await;
    let failed = provider.embed(vec!["fox".into()]).await.unwrap_err();
    assert!(
        matches!(&failed, EmbedError::Failed { model, .. } if model == TITAN),
        "{failed}"
    );
    assert_eq!(seen.bodies().len(), 1);
}
