use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_sdk_bedrockruntime::config::{BehaviorVersion, Credentials, Region};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use vellev_core::HttpApi;
use vellev_core::{Embedding, EmbeddingProvider};
use vellev_embed::{Bedrock, EmbedError, Embedders, Http, Provider, Purpose, Settings};

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
    let provider = Http::new(&format!("{url}/"), HttpApi::Tei, "bge", None).unwrap();
    let vectors = provider
        .embed(numbered(70), Purpose::Document)
        .await
        .unwrap();
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
    )
    .unwrap();
    let vectors = provider
        .embed(numbered(3), Purpose::Document)
        .await
        .unwrap();
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
        let provider = Http::new(&tei(seen.clone()).await, HttpApi::Tei, "bge", None).unwrap();
        assert_eq!(
            provider
                .embed(numbered(1), Purpose::Document)
                .await
                .unwrap(),
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
    )
    .unwrap();
    let refused = provider
        .embed(numbered(1), Purpose::Document)
        .await
        .unwrap_err();
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
async fn a_server_that_never_recovers_or_cannot_be_reached_fails_after_five_tries() {
    let seen = Seen::refusing(usize::MAX, 500);
    let provider = Http::new(&tei(seen.clone()).await, HttpApi::Tei, "bge", None).unwrap();
    let failed = provider
        .embed(numbered(1), Purpose::Document)
        .await
        .unwrap_err();
    assert!(
        matches!(&failed, EmbedError::Failed { reason, .. } if reason.starts_with("500")),
        "{failed}"
    );
    assert_eq!(seen.bodies().len(), 5);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unreachable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let provider = Http::new(&unreachable, HttpApi::Tei, "bge", None).unwrap();
    assert!(matches!(
        provider.embed(numbered(1), Purpose::Document).await,
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
            .unwrap()
            .embed(numbered(1), Purpose::Document)
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
    let retries = Bedrock::retries().with_initial_backoff(Duration::from_millis(5));
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
        .timeout_config(Bedrock::timeouts())
        .build();
    Bedrock::with_client(
        aws_sdk_bedrockruntime::Client::from_conf(config),
        TITAN,
        256,
    )
    .unwrap()
}

#[tokio::test]
async fn titan_is_sent_one_signed_normalised_text_a_call_with_few_in_flight() {
    let seen = Arc::new(Seen::default());
    let provider = bedrock(seen.clone()).await;
    let texts: Vec<String> = (1..=40).map(|n| "x".repeat(n)).collect();
    let vectors = provider.embed(texts, Purpose::Document).await.unwrap();
    assert_eq!(
        vectors,
        (1..=40).map(|n| vec![n as f32, 1.0]).collect::<Vec<_>>()
    );

    let bodies = seen.bodies();
    assert_eq!(bodies.len(), 40);
    assert!(bodies.contains(&json!({ "inputText": "xxx", "dimensions": 256, "normalize": true })));
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
        let vectors = provider
            .embed(vec!["é".repeat(40_000)], Purpose::Document)
            .await
            .unwrap();
        assert_eq!(vectors, [vec![30_000.0, 1.0]], "{status}");
        assert_eq!(seen.bodies().len(), 3);
    }

    let seen = Seen::refusing(usize::MAX, 400);
    let provider = bedrock(seen.clone()).await;
    let failed = provider
        .embed(vec!["fox".into()], Purpose::Document)
        .await
        .unwrap_err();
    assert!(
        matches!(&failed, EmbedError::Failed { model, .. } if model == TITAN),
        "{failed}"
    );
    assert_eq!(seen.bodies().len(), 1);
}

#[tokio::test]
async fn a_key_goes_only_to_the_endpoint_it_is_for_and_never_into_an_error() {
    let seen = Seen::refusing(usize::MAX, 400);
    let url = openai(seen.clone()).await;
    let embedding = |url: &str| Embedding {
        provider: EmbeddingProvider::Http {
            url: url.into(),
            api: HttpApi::Openai,
        },
        model: "m".into(),
        dimensions: 1,
        query_prefix: None,
        document_prefix: None,
        properties: vec![],
    };
    let embedders = Embedders::new(Settings {
        keys: [(url.clone(), "sk-secret".to_string())].into(),
        ..Settings::default()
    });
    for (asked, sent) in [
        (format!("{url}/"), Some("Bearer sk-secret".to_string())),
        (format!("{url}/v1"), None),
    ] {
        let embedder = embedders.embedder(&embedding(&asked)).unwrap().unwrap();
        let refused = embedder.query("fox").await.unwrap_err().to_string();
        assert!(!refused.contains("sk-secret"), "{refused}");
        assert_eq!(seen.authorizations.lock().unwrap().last().unwrap(), &sent);
    }

    let padded = |State(key): State<String>| async move {
        (StatusCode::BAD_REQUEST, format!("{}{key}", "x".repeat(290)))
    };
    let router = Router::new()
        .route("/embed", post(padded))
        .with_state("sk-0123456789abcdef".to_string());
    let provider = Http::new(
        &serve(router).await,
        HttpApi::Tei,
        "m",
        Some("sk-0123456789abcdef".into()),
    )
    .unwrap();
    let refused = provider
        .embed(numbered(1), Purpose::Document)
        .await
        .unwrap_err()
        .to_string();
    assert!(!refused.contains("sk-0"), "{refused}");

    let seen = Seen::refusing(usize::MAX, 400);
    let provider = Http::new(
        &tei(seen.clone()).await,
        HttpApi::Tei,
        "m",
        Some("  ".into()),
    )
    .unwrap();
    let refused = provider
        .embed(numbered(1), Purpose::Document)
        .await
        .unwrap_err();
    assert_eq!(
        refused,
        EmbedError::Failed {
            model: "m".into(),
            reason: "400 Bad Request: refused ".into()
        }
    );
    assert_eq!(*seen.authorizations.lock().unwrap(), [None]);
}

#[tokio::test]
async fn an_openai_endpoint_may_be_named_with_its_version_and_must_number_its_vectors_in_turn() {
    let seen = Arc::new(Seen::default());
    let url = openai(seen.clone()).await;
    let provider = Http::new(&format!("{url}/v1/"), HttpApi::Openai, "m", None).unwrap();
    assert_eq!(
        provider.embed(numbered(2), Purpose::Query).await.unwrap(),
        [vec![0.0], vec![1.0]]
    );

    let twice = || async {
        Json(json!({ "data": [
            { "index": 0, "embedding": [1.0] },
            { "index": 0, "embedding": [2.0] },
        ] }))
    };
    let url = serve(Router::new().route("/v1/embeddings", post(twice))).await;
    let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
    assert_eq!(
        provider.embed(numbered(2), Purpose::Document).await,
        Err(EmbedError::Malformed {
            model: "m".into(),
            reason: "vectors numbered other than in turn".into()
        })
    );
}

#[tokio::test]
async fn an_endpoint_is_given_the_wait_it_asks_for_and_a_request_that_cannot_be_sent_is_not_retried(
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let throttled = |State(calls): State<Arc<AtomicUsize>>| async move {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err((StatusCode::TOO_MANY_REQUESTS, [("retry-after", "1")]));
        }
        Ok(Json(json!([[7.0]])))
    };
    let router = Router::new()
        .route("/embed", post(throttled))
        .with_state(calls);
    let provider = Http::new(&serve(router).await, HttpApi::Tei, "m", None).unwrap();
    let started = std::time::Instant::now();
    assert_eq!(
        provider.embed(numbered(1), Purpose::Query).await.unwrap(),
        [vec![7.0]]
    );
    assert!(
        started.elapsed() >= Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );

    let seen = Arc::new(Seen::default());
    let provider = Http::new(
        &tei(seen.clone()).await,
        HttpApi::Tei,
        "m",
        Some("two\nlines".into()),
    )
    .unwrap();
    let started = std::time::Instant::now();
    assert!(matches!(
        provider.embed(numbered(1), Purpose::Query).await,
        Err(EmbedError::Failed { .. })
    ));
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "{:?}",
        started.elapsed()
    );
    assert!(seen.bodies().is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_bedrock_that_never_answers_fails_rather_than_hangs() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
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
        .endpoint_url(url)
        .retry_config(Bedrock::retries())
        .timeout_config(Bedrock::timeouts())
        .build();
    let provider = Bedrock::with_client(
        aws_sdk_bedrockruntime::Client::from_conf(config),
        TITAN,
        256,
    )
    .unwrap();
    let waited = tokio::time::timeout(
        Duration::from_secs(600),
        provider.embed(vec!["fox".into()], Purpose::Document),
    );
    assert!(matches!(waited.await, Ok(Err(EmbedError::Failed { .. }))));
}
