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
use vellev_core::{Embedding, EmbeddingProvider, ModelFault};
use vellev_embed::{Bedrock, EmbedError, Embedder, Embedders, Http, Provider, Purpose, Settings};

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
    /// Whether texts that say they are held are answered now.
    released: std::sync::atomic::AtomicBool,
    release: tokio::sync::Notify,
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

    /// Waits until held texts are released: so a test knows, and not
    /// guesses from a clock, that what it asks meanwhile was answered
    /// while they were still waited on.
    async fn held(&self) {
        loop {
            let released = self.release.notified();
            if self.released.load(Ordering::SeqCst) {
                return;
            }
            released.await;
        }
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_waiters();
    }
}

/// How long a test waits on what a right build does at once, or within a
/// bound of at most a few seconds, before failing rather than hanging.
const BACKSTOP: Duration = Duration::from_secs(60);

/// Waits until `now` holds, failing after [`BACKSTOP`].
async fn until(what: &str, now: impl Fn() -> bool) {
    let began = std::time::Instant::now();
    while !now() {
        assert!(began.elapsed() < BACKSTOP, "{what} did not come to hold");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `future`, failing after [`BACKSTOP`] rather than hanging: for what a
/// right build answers while something else is held.
async fn soon<T>(what: &str, future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(BACKSTOP, future)
        .await
        .unwrap_or_else(|_| panic!("{what} waited on what was held"))
}

fn held(count: usize) -> Vec<String> {
    (0..count).map(|n| format!("held {n}")).collect()
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
        Some("sk-secret-0123456789abcdef".into()),
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
        [Some("Bearer sk-secret-0123456789abcdef".to_string())]
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
        Some("sk-secret-0123456789abcdef".into()),
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
    assert_eq!(refused.lasting(), None, "a refusal may be about the texts");
    assert_eq!(seen.bodies().len(), 1);
}

#[tokio::test]
async fn a_bare_refusal_of_the_key_or_the_route_may_pass_whatever_stands_in_front_says_it_too() {
    for (status, said) in [
        (401, "401 Unauthorized"),
        (403, "403 Forbidden"),
        (404, "404 Not Found"),
    ] {
        let seen = Seen::refusing(usize::MAX, status);
        let provider = Http::new(
            &openai(seen.clone()).await,
            HttpApi::Openai,
            "m",
            Some("sk-secret-0123456789abcdef".into()),
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
                reason: format!("{said}: refused Bearer <redacted>")
            }
        );
        assert_eq!(refused.lasting(), None, "{status}");
        assert_eq!(seen.bodies().len(), 1, "{status} is not tried again");
    }
}

/// An endpoint that refuses every request as OpenAI words a refusal.
async fn refusing_as_openai(status: u16, error: Value, seen: Arc<Seen>) -> String {
    let embed =
        move |State(seen): State<Arc<Seen>>, headers: HeaderMap, Json(body): Json<Value>| {
            let error = error.clone();
            async move {
                seen.record(&headers, &body);
                (StatusCode::from_u16(status).unwrap(), Json(error))
            }
        };
    let embeddings = Router::new().route("/v1/embeddings", post(embed.clone()));
    serve(embeddings.route("/embed", post(embed)).with_state(seen)).await
}

#[tokio::test]
async fn an_endpoint_that_says_in_its_own_code_that_the_setup_is_wrong_cannot_be_used() {
    let coded = |code: &str| json!({ "error": { "message": "no", "type": "t", "code": code } });
    for (status, code, said, fault) in [
        (
            404,
            "model_not_found",
            "404 Not Found",
            ModelFault::Provider,
        ),
        (
            401,
            "invalid_api_key",
            "401 Unauthorized",
            ModelFault::Setup,
        ),
        // Said with a status that is otherwise worth another try, it is
        // still said: the endpoint is not asked again.
        (
            503,
            "model_not_found",
            "503 Service Unavailable",
            ModelFault::Provider,
        ),
        (
            429,
            "invalid_api_key",
            "429 Too Many Requests",
            ModelFault::Setup,
        ),
    ] {
        let seen = Arc::new(Seen::default());
        let url = refusing_as_openai(status, coded(code), seen.clone()).await;
        let refused = Http::new(&url, HttpApi::Openai, "m", None)
            .unwrap()
            .embed(numbered(1), Purpose::Document)
            .await
            .unwrap_err();
        assert_eq!(
            refused,
            EmbedError::Unusable {
                model: "m".into(),
                reason: format!("{said}: {}", coded(code)),
                fault,
            }
        );
        assert_eq!(refused.lasting(), Some(fault), "{code}");
        assert_eq!(seen.bodies().len(), 1, "{code} is not tried again");

        let tei = Http::new(&url, HttpApi::Tei, "m", None).unwrap();
        let refused = tei.embed(numbered(1), Purpose::Document).await;
        assert!(
            matches!(refused, Err(EmbedError::Failed { .. })),
            "only an OpenAI endpoint is taken at OpenAI's word: {code}"
        );
    }
    // A key the endpoint quotes back, with its middle masked as some do,
    // is not passed on to be logged.
    let quoting = json!({ "error": {
        "message": "Incorrect API key provided: sk-ab…wxyz. Find yours at example.test.",
        "code": "invalid_api_key",
    } });
    let seen = Arc::new(Seen::default());
    let url = refusing_as_openai(401, quoting, seen.clone()).await;
    let refused = Http::new(&url, HttpApi::Openai, "m", None)
        .unwrap()
        .embed(numbered(1), Purpose::Document)
        .await
        .unwrap_err();
    let said = refused.to_string();
    assert!(
        said.contains("Incorrect API key provided: <redacted>. Find yours")
            && !said.contains("wxyz"),
        "{said}"
    );
    assert_eq!(refused.lasting(), Some(ModelFault::Setup));

    // What is said is cut short, and a key that lies across the cut is
    // taken out whole: no part of it is left at the end.
    let key = "sk-proj-AbCd1234EfGh5678IjKl9012MnOp";
    for kept in [1, 3, 7, 8, 10, 19, 20, 30, key.len()] {
        let said = format!("{} {key} is not a key of ours", "é".repeat(298 - kept));
        let seen = Arc::new(Seen::default());
        let url = refusing_as_openai(400, json!(said), seen.clone()).await;
        let refused = Http::new(&url, HttpApi::Openai, "m", None)
            .unwrap()
            .embed(numbered(1), Purpose::Document)
            .await
            .unwrap_err();
        let whole = json!(said).to_string().replace(key, "<redacted>");
        let cut: String = whole.chars().take(300).collect();
        assert_eq!(
            refused,
            EmbedError::Failed {
                model: "m".into(),
                reason: format!("400 Bad Request: {cut}"),
            },
            "with {kept} of the key before the cut"
        );
        assert!(!refused.to_string().contains("sk-"), "{refused}");
    }

    // Whatever else is said, however plainly, may pass: an account with
    // nothing left to spend is paid up without a change to the node.
    for (status, said) in [
        (429, coded("rate_limit_exceeded")),
        (429, coded("insufficient_quota")),
        (403, coded("insufficient_quota")),
        (
            404,
            json!({ "error": { "message": "model \"m\" not found", "code": null } }),
        ),
        (
            404,
            json!({ "object": "error", "message": "The model `m` does not exist.", "code": 404 }),
        ),
        (400, json!({ "error": "model_not_found" })),
        (403, json!("insufficient_quota")),
    ] {
        let seen = Arc::new(Seen::default());
        let url = refusing_as_openai(status, said.clone(), seen.clone()).await;
        let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
        let refused = tokio::time::timeout(
            Duration::from_secs(1),
            provider.embed(numbered(1), Purpose::Document),
        )
        .await;
        match status {
            429 => assert!(refused.is_err(), "throttling is waited out: {said}"),
            _ => assert!(
                matches!(refused, Ok(Err(EmbedError::Failed { .. }))),
                "{said}: {refused:?}"
            ),
        }
    }
}

/// A refusal is read as far as 64 KiB and no further, and one that goes
/// on past that is not read for the fault it names: it may pass.
#[tokio::test]
async fn an_endpoint_that_says_a_great_deal_in_refusing_is_read_only_so_far() {
    let saying = |words: usize| async move {
        let refusal = json!({ "error": {
            "message": "no ".repeat(words),
            "code": "invalid_api_key",
        } });
        let said = refusal.to_string();
        let seen = Arc::new(Seen::default());
        let url = refusing_as_openai(401, refusal, seen.clone()).await;
        let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
        let refused = provider.embed(numbered(1), Purpose::Document).await;
        assert_eq!(seen.bodies().len(), 1);
        (said, refused.unwrap_err())
    };
    let passed_on = |said: &str| {
        let start: String = said.chars().take(300).collect();
        format!("401 Unauthorized: {start}")
    };
    let (said, refused) = saying(21_800).await;
    assert!(said.len() < 64 * 1024, "{}", said.len());
    assert_eq!(
        refused,
        EmbedError::Unusable {
            model: "m".into(),
            reason: passed_on(&said),
            fault: ModelFault::Setup,
        }
    );
    for words in [21_900, 6_000_000] {
        let (said, refused) = saying(words).await;
        assert!(said.len() > 64 * 1024, "{}", said.len());
        assert_eq!(
            refused,
            EmbedError::Failed {
                model: "m".into(),
                reason: passed_on(&said),
            },
            "{words} words"
        );
    }
}

/// An endpoint that answers every request as `respond` writes to it,
/// byte for byte, once it has been sent a request's head.
async fn answering<F, R>(respond: F) -> String
where
    F: Fn(tokio::net::TcpStream) -> R + Send + Sync + 'static,
    R: std::future::Future<Output = ()> + Send,
{
    use tokio::io::AsyncReadExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let respond = Arc::new(respond);
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let respond = respond.clone();
            tokio::spawn(async move {
                let mut asked = Vec::new();
                while !asked.windows(4).any(|end| end == b"\r\n\r\n") {
                    let mut more = [0; 1024];
                    match socket.read(&mut more).await {
                        Ok(read) if read > 0 => asked.extend_from_slice(&more[..read]),
                        _ => return,
                    }
                }
                respond(socket).await;
            });
        }
    });
    url
}

/// What is read of a refusal for the fault it names is what the endpoint
/// said, all of it: not the start of more than is read, and not as much
/// as came of what was cut off on its way. Either could be made to read
/// as a refusal that names a fault, and the model would be taken for one
/// that cannot be used.
#[tokio::test]
async fn a_refusal_that_is_not_all_there_is_not_read_for_the_fault_it_names() {
    use tokio::io::AsyncWriteExt;
    const CODED: &str = r#"{"error":{"message":"no","type":"t","code":"invalid_api_key"}}"#;
    let refused = |url: String| async move {
        let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
        let asked = provider.embed(numbered(1), Purpose::Document);
        let refused = tokio::time::timeout(Duration::from_secs(5), asked).await;
        refused.expect("no more is read than is read").unwrap_err()
    };
    let head = |length: usize| {
        format!(
            "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\n\
             content-length: {length}\r\nconnection: close\r\n\r\n"
        )
    };
    let unusable = EmbedError::Unusable {
        model: "m".into(),
        reason: format!("401 Unauthorized: {CODED}"),
        fault: ModelFault::Setup,
    };
    let may_pass = EmbedError::Failed {
        model: "m".into(),
        reason: format!("401 Unauthorized: {CODED}"),
    };

    // Said whole, it names a fault.
    let whole = answering(move |mut socket| async move {
        let said = format!("{}{CODED}", head(CODED.len()));
        let _ = socket.write_all(said.as_bytes()).await;
    });
    assert_eq!(refused(whole.await).await, unusable);

    // Said with more after it than is read: what is read of it would
    // read as the same refusal, and is not taken for one.
    let padded = answering(move |mut socket| async move {
        let padding = " ".repeat(70 * 1024);
        let said = format!("{}{CODED}{padding}", head(CODED.len() + padding.len()));
        let _ = socket.write_all(said.as_bytes()).await;
    });
    let EmbedError::Failed { reason, .. } = refused(padded.await).await else {
        panic!("it was read for a fault");
    };
    assert_eq!(reason.trim_end(), format!("401 Unauthorized: {CODED}"));

    // To the byte: a refusal of exactly as much as is read is all
    // there, and one a byte longer is not.
    for (over, read_for_a_fault) in [(0, true), (1, false)] {
        let exactly = answering(move |mut socket| async move {
            let padding = " ".repeat(64 * 1024 + over - CODED.len());
            let said = format!("{}{CODED}{padding}", head(64 * 1024 + over));
            let _ = socket.write_all(said.as_bytes()).await;
        });
        let lasting = refused(exactly.await).await.lasting();
        assert_eq!(lasting.is_some(), read_for_a_fault, "{over} over");
    }

    // Cut off before all that was promised came.
    let cut_off = answering(move |mut socket| async move {
        let said = format!("{}{CODED}", head(CODED.len() + 1_000));
        let _ = socket.write_all(said.as_bytes()).await;
    });
    assert_eq!(refused(cut_off.await).await, may_pass);

    // And one that never ends is read as far as any is, and no further:
    // it does not keep whoever asked until the endpoint tires.
    let endless = answering(move |mut socket| async move {
        let head = "HTTP/1.1 401 Unauthorized\r\ntransfer-encoding: chunked\r\n\r\n";
        let _ = socket.write_all(head.as_bytes()).await;
        let chunk = format!("{:x}\r\n{}\r\n", 4096, "no! ".repeat(1024));
        while socket.write_all(chunk.as_bytes()).await.is_ok() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    });
    let said = refused(endless.await).await;
    let start = "no! ".repeat(75);
    assert_eq!(
        said,
        EmbedError::Failed {
            model: "m".into(),
            reason: format!("401 Unauthorized: {start}"),
        }
    );
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
        // A text that says it is held is answered once released; one
        // that says it is slow takes a second, not 20 ms.
        if body["inputText"].as_str().unwrap().starts_with("held") {
            seen.held().await;
        }
        let slow = body["inputText"].as_str().unwrap().starts_with("slow");
        tokio::time::sleep(Duration::from_millis(if slow { 1_000 } else { 20 })).await;
        seen.in_flight.fetch_sub(1, Ordering::SeqCst);
        if let Some((status, _)) = seen.record(&headers, &body) {
            let kind = match status.as_u16() {
                429 => "ThrottlingException",
                400 => "ValidationException",
                403 => "AccessDeniedException",
                404 => "ResourceNotFoundException",
                _ => "InternalServerException",
            };
            // A refusal that quotes back how the request was signed.
            let message = match status.as_u16() {
                400 => "not now: Credential=AKIAIOSFODNN7EXAMPLE/20261008/eu-west-2",
                _ => "not now",
            };
            let refusal = Json(json!({ "message": message }));
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
    assert_eq!(
        failed,
        EmbedError::Failed {
            model: TITAN.into(),
            reason: "ValidationException: not now: Credential=<redacted>/20261008/eu-west-2".into()
        },
        "with the key it was signed with taken out"
    );
    assert_eq!(
        failed.lasting(),
        None,
        "a refused input may be this text's alone"
    );
    assert_eq!(seen.bodies().len(), 1);
}

#[tokio::test]
async fn a_region_without_titan_cannot_be_used_and_access_denied_may_pass() {
    let refused = |status: u16| async move {
        let seen = Seen::refusing(usize::MAX, status);
        let provider = bedrock(seen.clone()).await;
        let failed = provider
            .embed(vec!["fox".into()], Purpose::Document)
            .await
            .unwrap_err();
        assert_eq!(seen.bodies().len(), 1, "{status} is not tried again");
        failed
    };
    assert_eq!(
        refused(404).await,
        EmbedError::Unusable {
            model: TITAN.into(),
            reason: "ResourceNotFoundException: not now".into(),
            fault: ModelFault::Provider,
        }
    );
    assert_eq!(
        refused(403).await,
        EmbedError::Failed {
            model: TITAN.into(),
            reason: "AccessDeniedException: not now".into()
        },
        "as is said while a permission just granted spreads"
    );
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
        keys: [(url.clone(), "sk-secret-0123456789abcdef".to_string())].into(),
        ..Settings::default()
    });
    for (asked, sent) in [
        (
            format!("{url}/"),
            Some("Bearer sk-secret-0123456789abcdef".to_string()),
        ),
        (format!("{url}/v1"), None),
    ] {
        let embedder = embedders.embedder(&embedding(&asked)).unwrap().unwrap();
        let refused = embedder.query("fox").await.unwrap_err().to_string();
        assert!(!refused.contains("sk-secret-0123456789abcdef"), "{refused}");
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
async fn an_endpoint_is_given_the_wait_it_asks_for() {
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
}

/// A request that cannot be sent, its key being no header, is not tried
/// again: on a paused clock, which a wait to try again would move, no
/// time passes at all.
#[tokio::test(start_paused = true)]
async fn a_request_that_cannot_be_sent_is_not_retried() {
    let seen = Arc::new(Seen::default());
    let provider = Http::new(
        &tei(seen.clone()).await,
        HttpApi::Tei,
        "m",
        Some("sk-0123456789abcdef\u{7f}x".into()),
    )
    .unwrap();
    let started = tokio::time::Instant::now();
    assert!(matches!(
        provider.embed(numbered(1), Purpose::Query).await,
        Err(EmbedError::Failed { .. })
    ));
    assert_eq!(started.elapsed(), Duration::ZERO);
    assert!(seen.bodies().is_empty());
}

/// Where something listens, takes every connection, and never answers.
async fn silent() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
    url
}

/// Titan as it would be reached at `url`, with the tries and the times
/// the node gives it.
fn bedrock_at(url: String) -> Bedrock {
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
    Bedrock::with_client(
        aws_sdk_bedrockruntime::Client::from_conf(config),
        TITAN,
        256,
    )
    .unwrap()
}

#[tokio::test(start_paused = true)]
async fn a_bedrock_that_never_answers_fails_rather_than_hangs() {
    let provider = bedrock_at(silent().await);
    let waited = tokio::time::timeout(
        Duration::from_secs(600),
        provider.embed(vec!["fox".into()], Purpose::Document),
    );
    assert!(matches!(waited.await, Ok(Err(EmbedError::Failed { .. }))));
}

/// Left to themselves, an endpoint that never answers is waited on for
/// two minutes and Titan for as long for each text, every try included.
/// Whoever embeds through them waits no longer than a model is given: on
/// a paused clock, which moves only when nothing is left to do but wait,
/// exactly that long.
#[tokio::test(start_paused = true)]
async fn an_endpoint_or_a_bedrock_that_never_answers_is_given_up_on_in_the_time_a_model_is_given() {
    let ms = Duration::from_millis;
    let late = |model: &str, within: &str| EmbedError::Failed {
        model: model.into(),
        reason: format!("no answer within {within}"),
    };
    // What each of a query and three documents comes to, and that it
    // comes to it when the time is up, give or take.
    let given_up_on = |embedder: Arc<Embedder>, model: &'static str| async move {
        // Left waiting on what never answers, this fails and does not
        // hang.
        let seconds = Duration::from_secs(5);
        let began = tokio::time::Instant::now();
        let query = tokio::time::timeout(seconds, embedder.query("fox")).await;
        assert_eq!(query, Ok(Err(late(model, "1s"))));
        // To the millisecond the timer keeps time in.
        let took = began.elapsed();
        assert!((ms(1_000)..=ms(1_000) + ms(1)).contains(&took), "{took:?}");
        let began = tokio::time::Instant::now();
        let texts = numbered(3);
        let documents = tokio::time::timeout(seconds, embedder.documents(&texts)).await;
        let stalled = EmbedError::Failed {
            model: model.into(),
            reason: "the model has made nothing for anyone in 1s".into(),
        };
        assert_eq!(documents, Ok(Err(stalled.clone())));
        // To the millisecond the timer keeps time in.
        let took = began.elapsed();
        assert!((ms(1_000)..=ms(1_000) + ms(1)).contains(&took), "{took:?}");
        assert_eq!(stalled.lasting(), None, "it may pass");
    };

    // An endpoint, as a node set up to give so long embeds through it.
    let url = silent().await;
    let embedders = Embedders::new(Settings {
        query_within: ms(1_000),
        documents_within: ms(1_000),
        ..Settings::default()
    });
    let through = Embedding {
        provider: EmbeddingProvider::Http {
            url,
            api: HttpApi::Tei,
        },
        model: "m".into(),
        dimensions: 1,
        query_prefix: None,
        document_prefix: None,
        properties: vec![],
    };
    given_up_on(embedders.embedder(&through).unwrap().unwrap(), "m").await;

    let titan = Embedding {
        provider: EmbeddingProvider::Bedrock {
            region: "eu-west-2".into(),
        },
        model: TITAN.into(),
        dimensions: 256,
        ..through
    };
    let embedder = Embedder::new(Arc::new(bedrock_at(silent().await)), titan);
    let embedder = embedder.answering_within(ms(1_000), ms(1_000));
    given_up_on(Arc::new(embedder), TITAN).await;
}

/// Titan stands in at 20 ms a text. Before queries had calls kept for
/// them, one asked beside many documents waited for every one of them:
/// 4.8 seconds beside two thousand. Here the documents are held, and not
/// answered until the queries are: a query that waited for them would
/// never be answered, and no clock says whether it was quick enough.
#[tokio::test(flavor = "multi_thread")]
async fn a_query_beside_many_documents_on_bedrock_is_answered_in_about_its_own_time() {
    let seen = Arc::new(Seen::default());
    let provider = Arc::new(bedrock(seen.clone()).await);
    let query = |n: usize| provider.embed(vec![format!("query {n}")], Purpose::Query);
    let in_flight = || seen.in_flight.load(Ordering::SeqCst);
    let answered = |task: &tokio::task::JoinHandle<_>| task.is_finished();
    query(0).await.unwrap();

    // Documents held hold no more calls than are theirs, however many
    // wait: the rest are free for queries, which so do not wait even for
    // one text of them.
    let documents = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.embed(held(16), Purpose::Document).await })
    };
    until("eight calls held", || in_flight() == 8).await;
    let (one, another) = soon("two queries", async { tokio::join!(query(1), query(2)) }).await;
    assert!(one.is_ok() && another.is_ok());
    assert!(!answered(&documents), "the documents were held");
    seen.release();
    assert_eq!(documents.await.unwrap().unwrap().len(), 16);
    assert_eq!(seen.most_in_flight.load(Ordering::SeqCst), 10);

    // Two calls of six hundred held documents each, side by side.
    let seen = Arc::new(Seen::default());
    let provider = Arc::new(bedrock(seen.clone()).await);
    let query = |n: usize| provider.embed(vec![format!("query {n}")], Purpose::Query);
    let in_flight = || seen.in_flight.load(Ordering::SeqCst);
    query(0).await.unwrap();
    let bulk = || {
        let provider = provider.clone();
        tokio::spawn(async move { provider.embed(held(600), Purpose::Document).await })
    };
    let (one, another) = (bulk(), bulk());
    until("eight calls held", || in_flight() == 8).await;
    assert_eq!(soon("a query", query(1)).await.unwrap(), [vec![7.0, 1.0]]);
    // Many at once take every call as it comes free: the documents wait
    // for them, not they for the documents.
    let answers = soon(
        "twenty queries",
        futures::future::join_all((0..20).map(query)),
    )
    .await;
    assert!(answers.iter().all(Result::is_ok));
    assert!(!answered(&one) && !answered(&another));

    // Calls for documents made side by side take turns: a short one made
    // now, behind them, is answered after a turn or so of theirs, and not
    // kept until they are done. Turns, not a clock: it needs eight texts,
    // and they twelve hundred.
    let short = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.embed(numbered(8), Purpose::Document).await })
    };
    seen.release();
    assert_eq!(short.await.unwrap().unwrap().len(), 8);
    assert!(!answered(&one) && !answered(&another));
    for bulk in [one, another] {
        let vectors = bulk.await.unwrap().unwrap();
        assert_eq!(vectors.len(), 600);
        assert_eq!(vectors[599], [8.0, 1.0], "in the order asked");
    }
    let most = seen.most_in_flight.load(Ordering::SeqCst);
    assert!(most <= 10, "{most} calls in flight");
}

/// An endpoint has no line of this node's making to stand in: a call
/// sends its batches one after another, and no call waits on another's.
/// The documents' batch is held until the queries are answered: a query
/// that waited on it would never be.
#[tokio::test(flavor = "multi_thread")]
async fn a_query_beside_many_documents_on_an_endpoint_is_sent_at_once() {
    let seen = Arc::new(Seen::default());
    let embed = |State(seen): State<Arc<Seen>>, Json(body): Json<Value>| async move {
        let inputs = body["inputs"].as_array().unwrap();
        if inputs[0].as_str().unwrap().starts_with("held") {
            seen.in_flight.fetch_add(1, Ordering::SeqCst);
            seen.held().await;
        }
        Json(json!(inputs
            .iter()
            .map(|text| vector_of(text.as_str().unwrap()))
            .collect::<Vec<_>>()))
    };
    let url = serve(
        Router::new()
            .route("/embed", post(embed))
            .with_state(seen.clone()),
    )
    .await;
    let provider = Arc::new(Http::new(&url, HttpApi::Tei, "bge", None).unwrap());
    let query = |n: usize| provider.embed(vec![format!("query {n}")], Purpose::Query);
    query(0).await.unwrap();
    let bulk = {
        let provider = provider.clone();
        tokio::spawn(async move { provider.embed(held(1_280), Purpose::Document).await })
    };
    until("a batch held", || {
        seen.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;

    assert_eq!(soon("a query", query(1)).await.unwrap(), [vec![1.0]]);
    let answers = soon(
        "twenty queries",
        futures::future::join_all((0..20).map(query)),
    )
    .await;
    assert!(answers.iter().all(Result::is_ok));
    assert!(!bulk.is_finished());
    seen.release();
    assert_eq!(bulk.await.unwrap().unwrap().len(), 1_280);
}

/// The way of embedding of a collection whose model is `model`.
/// Two collections that embed with one model but put different prefixes
/// before a search's text send the model different texts, and share
/// nothing kept for a search: what is kept is kept by what was sent. A
/// collection worded like another shares what it kept.
#[tokio::test]
async fn collections_that_word_a_search_differently_share_no_kept_vector() {
    let seen = Arc::new(Seen::default());
    let url = tei(seen.clone()).await;
    let through = EmbeddingProvider::Http {
        url: url.clone(),
        api: HttpApi::Tei,
    };
    let embedders = Embedders::new(Settings::default());
    let plain = embedding_with(through, "m", 1);
    let prefixed = Embedding {
        query_prefix: Some("query: ".into()),
        ..plain.clone()
    };
    let alike = Embedding {
        properties: vec!["title".into()],
        ..plain.clone()
    };
    let plain = embedders.embedder(&plain).unwrap().unwrap();
    let prefixed = embedders.embedder(&prefixed).unwrap().unwrap();
    let alike = embedders.embedder(&alike).unwrap().unwrap();
    assert_eq!(embedders.held(), 1, "one model");
    let sent = |seen: &Seen| -> Vec<Value> {
        seen.bodies()
            .iter()
            .map(|body| body["inputs"].clone())
            .collect()
    };
    assert_eq!(plain.query("3").await.unwrap(), Some(vec![3.0]));
    assert_eq!(prefixed.query("3").await.unwrap(), Some(vec![3.0]));
    assert_eq!(sent(&seen), [json!(["3"]), json!(["query: 3"])]);
    // Asked again, each is answered from what was kept for what it sent,
    // and the one worded like the first from what the first kept.
    assert_eq!(plain.query("3").await.unwrap(), Some(vec![3.0]));
    assert_eq!(prefixed.query("3").await.unwrap(), Some(vec![3.0]));
    assert_eq!(alike.query("3").await.unwrap(), Some(vec![3.0]));
    assert_eq!(sent(&seen).len(), 2, "nothing more was sent");
}

/// A region that answers has been reached, whatever it answers: one
/// that refuses a text, denies the call, says slow down through every
/// try, or fails within, is asked again by the next call, and is not
/// taken to be out of reach.
#[tokio::test]
async fn a_region_that_refuses_or_throttles_has_been_reached_and_is_asked_again() {
    // Refused once for what will not be tried again; and through all
    // five tries for what is.
    for (status, tries) in [(400, 1), (403, 1), (429, 5), (500, 5)] {
        let seen = Seen::refusing(tries, status);
        let provider = bedrock(seen.clone()).await;
        let refused = provider.embed(vec!["one".into()], Purpose::Document).await;
        let said = refused.unwrap_err().to_string();
        assert!(!said.contains("was not reached"), "{status}: {said}");
        assert_eq!(seen.bodies().len(), tries, "{status}");
        // The next call is sent, and answered.
        let asked = provider.embed(vec!["two".into()], Purpose::Query).await;
        assert_eq!(asked, Ok(vec![vec![3.0, 1.0]]), "{status}");
        assert_eq!(seen.bodies().len(), tries + 1, "{status}");
    }
}

/// Titan refusing texts is not Titan at work: a write whose text Titan
/// has yet to answer is given up on when Titan has made no vector for
/// anyone for as long as a model may go without, however many texts of
/// other writes it refuses meanwhile.
///
/// Titan never answers the write's text, and goes on refusing others for
/// as long as the write is waited on. Were a refusal taken for Titan at
/// work, the write would never be given up on while they go on, and the
/// refusals would run out first: no clock decides which.
#[tokio::test(flavor = "multi_thread")]
async fn a_text_bedrock_refuses_is_not_bedrock_at_work() {
    let ms = Duration::from_millis;
    let seen = Seen::refusing(usize::MAX, 400);
    let titan = EmbeddingProvider::Bedrock {
        region: "eu-west-2".into(),
    };
    let embedder = Embedder::new(
        Arc::new(bedrock(seen.clone()).await),
        embedding_with(titan, TITAN, 2),
    );
    let embedder = Arc::new(embedder.answering_within(ms(2_000), ms(500)));
    let began = std::time::Instant::now();
    let held = {
        let embedder = embedder.clone();
        tokio::spawn(async move { embedder.documents(&["held one".to_string()]).await })
    };
    // Meanwhile it refuses a text every eighty milliseconds or so, until
    // the write is given up on. The minute is only so that a wrong build
    // cannot wait forever: a right one gives up after half a second, and
    // would reach it only were its timer late by a hundred times as long.
    let mut refused = 0;
    while !held.is_finished() {
        assert!(
            began.elapsed() < BACKSTOP,
            "the write was not given up on through {refused} refusals"
        );
        let said = embedder
            .documents(&[format!("quick {refused}")])
            .await
            .unwrap_err()
            .to_string();
        assert!(said.contains("ValidationException"), "{said}");
        refused += 1;
        tokio::time::sleep(ms(80)).await;
    }
    let given_up_on = held.await.unwrap().unwrap_err().to_string();
    assert!(
        given_up_on.ends_with("the model has made nothing for anyone in 500ms"),
        "{given_up_on}"
    );
    // Not before then, and with refusals made meanwhile.
    assert!(began.elapsed() >= ms(500), "{:?}", began.elapsed());
    assert!(refused >= 1);
}

/// One batch is given so long, every try included: an endpoint that goes
/// on saying it is busy is given up on then, and not after every try it
/// asks for, which here would take seven and a half seconds. What it is
/// given up on with names the batch's bound, and the endpoint was asked
/// no more than the twice there is time for: five tries would be five.
#[tokio::test]
async fn a_batch_is_given_so_long_however_many_tries_the_endpoint_asks_for() {
    let ms = Duration::from_millis;
    let busy = Seen::refusing(usize::MAX, 503);
    let url = tei(busy.clone()).await;
    let provider = Http::new(&url, HttpApi::Tei, "bge", None)
        .unwrap()
        .answering_each_batch_within(ms(1_200));
    let began = std::time::Instant::now();
    let given_up_on = provider.embed(vec!["fox".into()], Purpose::Document).await;
    let took = began.elapsed();
    assert_eq!(
        given_up_on,
        Err(EmbedError::Failed {
            model: "bge".into(),
            reason: "no answer within 1.2s".into(),
        })
    );
    assert!(took >= ms(1_200), "{took:?}");
    // It had been asked twice, and was to be asked a third time at a
    // second and a half: once, if the second try was kept past the bound
    // by a loaded machine.
    let asked = busy.bodies().len();
    assert!((1..=2).contains(&asked), "asked {asked} times");
}

/// An endpoint whose refusal is cut off on its way, partway through a
/// key it was quoting back: nothing of the key is passed on, though what
/// came of it is far short of where looking for keys ends, and too short
/// to be shaped like a key.
#[tokio::test]
async fn a_refusal_cut_off_on_its_way_passes_on_nothing_of_a_key_it_was_quoting() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let (mut caller, _) = listener.accept().await.unwrap();
            let mut asked = [0; 4096];
            let _ = caller.read(&mut asked).await;
            // It says it will say five hundred bytes, says these, and
            // hangs up.
            let said = "HTTP/1.1 401 Unauthorized\r\ncontent-type: text/plain\r\n\
                        content-length: 500\r\n\r\nno such key: sk-proj-AbCd1234Ef";
            let _ = caller.write_all(said.as_bytes()).await;
            let _ = caller.shutdown().await;
        }
    });
    let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
    let refused = provider.embed(vec!["fox".into()], Purpose::Query).await;
    let said = refused.unwrap_err().to_string();
    assert!(
        said.contains("401") && said.contains("no such key: "),
        "{said}"
    );
    assert!(
        !said.contains("sk-proj") && !said.contains("AbCd"),
        "{said}"
    );
}

fn embedding_with(provider: EmbeddingProvider, model: &str, dimensions: u32) -> Embedding {
    Embedding {
        provider,
        model: model.into(),
        dimensions,
        query_prefix: None,
        document_prefix: None,
        properties: vec![],
    }
}

/// A write through an endpoint or through Bedrock is given up on when
/// what serves the model stops answering, and not for how long the write
/// takes while it is answered: however many batches it is of, and however
/// many other writes it waits for a call behind.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_is_not_given_up_on_for_how_many_write_or_how_much_while_it_is_answered() {
    let ms = Duration::from_millis;
    // An endpoint that takes 150 ms a batch, and a write of sixteen
    // batches: 2.4 seconds, given up on only after 500 ms of nothing.
    let slow = |Json(body): Json<Value>| async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let inputs = body["inputs"].as_array().unwrap();
        Json(json!(inputs
            .iter()
            .map(|text| vector_of(text.as_str().unwrap()))
            .collect::<Vec<_>>()))
    };
    let url = serve(Router::new().route("/embed", post(slow))).await;
    let endpoint = EmbeddingProvider::Http {
        url: url.clone(),
        api: HttpApi::Tei,
    };
    let provider = Arc::new(Http::new(&url, HttpApi::Tei, "bge", None).unwrap());
    let embedder = Embedder::new(provider, embedding_with(endpoint, "bge", 1));
    let embedder = embedder.answering_within(ms(500), ms(500));
    let began = std::time::Instant::now();
    let written = embedder.documents(&numbered(16 * 32)).await;
    assert_eq!(written.map(|vectors| vectors.len()), Ok(16 * 32));
    assert!(began.elapsed() > ms(2_000), "{:?}", began.elapsed());

    // An endpoint that is busy says so, and is asked again after half a
    // second and then a second more. Saying so is not the model at work:
    // only a vector is. A write given three seconds of nothing waits out
    // the tries and is made; one given 1.2 seconds is given up on while
    // the endpoint is still only saying it is busy.
    let busy = |stalled: Duration| async move {
        let busy = Seen::refusing(2, 503);
        let url = tei(busy.clone()).await;
        let endpoint = EmbeddingProvider::Http {
            url: url.clone(),
            api: HttpApi::Tei,
        };
        let provider = Arc::new(Http::new(&url, HttpApi::Tei, "bge", None).unwrap());
        let embedder = Embedder::new(provider, embedding_with(endpoint, "bge", 1));
        let embedder = embedder.answering_within(ms(1_200), stalled);
        let began = std::time::Instant::now();
        let written = embedder.documents(&numbered(3)).await;
        (written.map(|vectors| vectors.len()), began.elapsed())
    };
    let (written, took) = busy(ms(3_000)).await;
    assert_eq!(written, Ok(3));
    assert!(took > ms(1_400), "{took:?}");
    let (written, took) = busy(ms(1_200)).await;
    let stopped = "the model has made nothing for anyone in 1.2s";
    assert!(
        matches!(&written, Err(EmbedError::Failed { reason, .. }) if reason == stopped),
        "{written:?}"
    );
    // Given up on, and not made: the third try, at a second and a half,
    // would have made it.
    assert!(took >= ms(1_200), "{took:?}");

    // Titan at 20 ms a text, and twenty-four writes of forty texts at
    // once, for the eight calls there are for documents: the last is
    // done after some two and a half seconds. None is given up on,
    // though each would be after two seconds of Titan answering no one.
    let seen = Arc::new(Seen::default());
    let titan = EmbeddingProvider::Bedrock {
        region: "eu-west-2".into(),
    };
    let embedder = Embedder::new(
        Arc::new(bedrock(seen.clone()).await),
        embedding_with(titan, TITAN, 2),
    );
    let embedder = Arc::new(embedder.answering_within(ms(2_000), ms(2_000)));
    let writes = (0..24).map(|_| {
        let embedder = embedder.clone();
        async move { embedder.documents(&numbered(40)).await }
    });
    let began = std::time::Instant::now();
    let written = futures::future::join_all(writes).await;
    for (nth, written) in written.into_iter().enumerate() {
        assert_eq!(written.map(|vectors| vectors.len()), Ok(40), "write {nth}");
    }
    assert!(began.elapsed() > ms(2_000), "{:?}", began.elapsed());
    assert_eq!(seen.bodies().len(), 24 * 40, "and none was asked for twice");
}

/// A region that cannot be reached is found so by the first texts sent
/// to it, each after every try a call is given. The texts behind them
/// are not each sent to find out the same, holding the region's calls
/// while they do: they fail at once, until it is time to try again.
#[tokio::test(flavor = "multi_thread")]
async fn a_region_that_is_not_reached_is_not_tried_again_for_every_text_that_waits() {
    // Something that takes every connection and closes it unanswered,
    // counting them.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let reached = Arc::new(AtomicUsize::new(0));
    let counted = reached.clone();
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            counted.fetch_add(1, Ordering::SeqCst);
            drop(socket);
        }
    });
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
        .endpoint_url(url)
        .retry_config(retries)
        .timeout_config(Bedrock::timeouts())
        .build();
    let client = aws_sdk_bedrockruntime::Client::from_conf(config);
    let provider = Arc::new(Bedrock::with_client(client, TITAN, 256).unwrap());
    let tries = || reached.load(Ordering::SeqCst);

    // Forty writes of one text each, at once: eight calls are for
    // documents, so eight texts are sent, five tries each at the most,
    // and the other thirty-two are not.
    let writes = (0..40).map(|nth| {
        let provider = provider.clone();
        async move {
            provider
                .embed(vec![format!("text {nth}")], Purpose::Document)
                .await
        }
    });
    let written = futures::future::join_all(writes).await;
    assert!(written
        .iter()
        .all(|written| matches!(written, Err(EmbedError::Failed { .. }))));
    let unsent = written
        .iter()
        .filter(|written| format!("{written:?}").contains("was not reached a moment ago"))
        .count();
    assert!(unsent >= 32, "{written:?}");
    assert!((8..=8 * 5).contains(&tries()), "{} tries", tries());
    // Nor is a search, meanwhile.
    let tried = tries();
    assert!(provider.embed(numbered(1), Purpose::Query).await.is_err());
    assert_eq!(tries(), tried);

    // When that has stood as long as it stands, the region is tried
    // again.
    tokio::time::sleep(vellev_embed::UNREACHED_FOR + Duration::from_millis(200)).await;
    assert!(provider.embed(numbered(1), Purpose::Query).await.is_err());
    assert!(tries() > tried, "{} tries", tries());
}

/// A region that answers no call is never found out by the calls, which
/// a caller gives up on long before their tries are spent: it is found
/// out by the caller's giving up, once a call has gone unanswered for
/// one try's time. What is meant for it meanwhile fails at once. A
/// caller that gives up sooner than that says nothing of the region.
#[tokio::test(flavor = "multi_thread")]
async fn a_region_a_caller_gives_up_on_unanswered_is_marked_as_not_reached() {
    let ms = Duration::from_millis;
    let titan = || EmbeddingProvider::Bedrock {
        region: "eu-west-2".into(),
    };
    let giving_up = |provider: Bedrock| {
        let embedder = Embedder::new(Arc::new(provider), embedding_with(titan(), TITAN, 2));
        Arc::new(embedder.answering_within(ms(300), ms(300)))
    };
    // Titan takes a second over a text that says it is slow, and the
    // search is given 300 ms: given up on after 200 ms unanswered marks
    // the region, and the next search is refused, unsent: Titan had only
    // ever the slow one in hand.
    let seen = Arc::new(Seen::default());
    let provider = bedrock(seen.clone()).await.unreached_after(ms(200));
    let embedder = giving_up(provider);
    let began = std::time::Instant::now();
    assert!(embedder.query("slow one").await.is_err());
    let refused = embedder.query("quick").await.unwrap_err().to_string();
    assert!(
        refused.contains("was not reached a moment ago") && refused.contains("given up on after"),
        "{refused}"
    );
    assert!(began.elapsed() >= ms(300), "{:?}", began.elapsed());
    // Titan was called once: the quick one was never sent.
    assert_eq!(seen.most_in_flight.load(Ordering::SeqCst), 1);

    // Given up on sooner than a call may go unanswered, nothing is said
    // of the region, and the next search goes out.
    let seen = Arc::new(Seen::default());
    let provider = bedrock(seen.clone()).await.unreached_after(ms(2_000));
    let embedder = giving_up(provider);
    assert!(embedder.query("slow two").await.is_err());
    assert_eq!(embedder.query("quick").await, Ok(Some(vec![5.0, 1.0])));
}
