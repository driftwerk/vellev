//! The calls a node has to Titan at once are the node's, shared out
//! among the regions it embeds through: however many collections embed
//! through Bedrock, at whichever sizes, and however what is held for
//! each is let go and held again. And a region that has stopped answering
//! holds its own share of them and no other region's. The test is alone
//! in its file, and so in its process: it tells the process where
//! Bedrock is, which every test in a process would be told.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use vellev_core::{Embedding, EmbeddingProvider, HttpApi};
use vellev_embed::{EmbedError, Embedders, Settings};

const TITAN: &str = "amazon.titan-embed-text-v2:0";
/// A region that never answers, as none in America does here.
const STOPPED: &str = "us-west-2";

/// How many calls a stand-in for Bedrock has in flight, in all, for
/// documents and to each region, and the most it has had of each.
#[derive(Default)]
struct Seen {
    in_flight: AtomicUsize,
    most_in_flight: AtomicUsize,
    documents: AtomicUsize,
    most_documents: AtomicUsize,
    calls: AtomicUsize,
    /// To each region: in flight now, and the most.
    regions: Mutex<HashMap<String, (usize, usize)>>,
}

impl Seen {
    fn most(&self) -> (usize, usize) {
        let most = |count: &AtomicUsize| count.load(Ordering::SeqCst);
        (most(&self.most_in_flight), most(&self.most_documents))
    }

    /// The most calls there have been in flight to `region` at once.
    fn most_to(&self, region: &str) -> usize {
        let regions = self.regions.lock().unwrap();
        regions.get(region).map_or(0, |(_, most)| *most)
    }

    fn to(&self, region: &str, by: isize) {
        let mut regions = self.regions.lock().unwrap();
        let (now, most) = regions.entry(region.to_string()).or_default();
        *now = now.checked_add_signed(by).unwrap();
        *most = (*most).max(*now);
    }
}

/// Titan, taking 50 ms a text, and a second over one that says it is
/// slow: in every region but those that never answer. The region a
/// call is to is the one it was signed for.
fn titan(seen: Arc<Seen>) -> Router {
    let invoke = |State(seen): State<Arc<Seen>>, headers: HeaderMap, Json(body): Json<Value>| async move {
        let signed = headers["authorization"].to_str().unwrap();
        let region = signed.split('/').nth(2).unwrap().to_string();
        let text = body["inputText"].as_str().unwrap().to_string();
        let document = !text.starts_with("query");
        seen.calls.fetch_add(1, Ordering::SeqCst);
        seen.to(&region, 1);
        let now = seen.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        seen.most_in_flight.fetch_max(now, Ordering::SeqCst);
        if document {
            let now = seen.documents.fetch_add(1, Ordering::SeqCst) + 1;
            seen.most_documents.fetch_max(now, Ordering::SeqCst);
        }
        if region.starts_with("us-") {
            std::future::pending::<()>().await;
        }
        let slow = text.starts_with("slow");
        tokio::time::sleep(Duration::from_millis(if slow { 1_000 } else { 50 })).await;
        seen.in_flight.fetch_sub(1, Ordering::SeqCst);
        seen.to(&region, -1);
        if document {
            seen.documents.fetch_sub(1, Ordering::SeqCst);
        }
        let size = body["dimensions"].as_u64().unwrap() as usize;
        Json(json!({ "embedding": vec![1.0; size], "inputTextTokenCount": 3 }))
    };
    Router::new()
        .route("/model/{model}/invoke", post(invoke))
        .with_state(seen)
}

fn through_bedrock(region: &str, dimensions: u32) -> Embedding {
    Embedding {
        provider: EmbeddingProvider::Bedrock {
            region: region.into(),
        },
        model: TITAN.into(),
        dimensions,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    }
}

fn through_an_endpoint(model: usize) -> Embedding {
    Embedding {
        provider: EmbeddingProvider::Http {
            url: "http://one.test".into(),
            api: HttpApi::Tei,
        },
        model: format!("model {model}"),
        ..through_bedrock("eu-west-2", 2)
    }
}

/// A node that embeds through Bedrock in `regions`.
fn node(regions: &[&str]) -> Arc<Embedders> {
    Arc::new(Embedders::new(Settings {
        bedrock_regions: regions.iter().map(|region| region.to_string()).collect(),
        ..Settings::default()
    }))
}

fn texts(word: &str, count: usize) -> Vec<String> {
    (0..count).map(|n| format!("{word} {n}")).collect()
}

/// Begin a write of `texts` to a collection that embeds as `embedding`
/// says: how many vectors were made.
fn writing(
    embedders: &Arc<Embedders>,
    embedding: &Embedding,
    texts: Vec<String>,
) -> tokio::task::JoinHandle<Result<usize, EmbedError>> {
    let (embedders, embedding) = (embedders.clone(), embedding.clone());
    tokio::spawn(async move {
        let embedder = embedders.embedder(&embedding).unwrap().unwrap();
        embedder.documents(&texts).await.map(|made| made.len())
    })
}

#[test]
fn a_node_has_no_more_calls_to_titan_at_once_however_many_embed_through_it() {
    // Where Bedrock is, is told to the process before it has any thread
    // but this one to read it meanwhile: so the address is had first,
    // and whatever runs on other threads is begun after.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::env::set_var("AWS_ENDPOINT_URL", url);
    std::env::set_var("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret");
    std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let serve = |seen: Arc<Seen>, listener: std::net::TcpListener| {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            tokio::spawn(async move { axum::serve(listener, titan(seen)).await.unwrap() })
        };
        // One address, and what answers at it counting afresh for each
        // part of this.
        let again = listener.try_clone().unwrap();
        let seen = Arc::new(Seen::default());
        let served = serve(seen.clone(), listener);
        let one = tokio::time::timeout(Duration::from_secs(60), one_region(seen));
        one.await.expect("nothing waits for ever");
        served.abort();
        let _ = served.await;
        let seen = Arc::new(Seen::default());
        serve(seen.clone(), again);
        let two = tokio::time::timeout(Duration::from_secs(60), one_stopped(seen.clone()));
        two.await.expect("nothing waits for ever");
        let many = tokio::time::timeout(Duration::from_secs(60), many_regions(seen.clone()));
        many.await.expect("nothing waits for ever");
        let most = tokio::time::timeout(Duration::from_secs(60), all_but_one_stopped(seen));
        most.await.expect("nothing waits for ever");
    });
}

/// A node that embeds through one region has all its calls to it.
async fn one_region(seen: Arc<Seen>) {
    let embedders = node(&["eu-west-2"]);
    // Collections that embed at three sizes, all written to at once:
    // three models, as the node holds them, and one node's calls
    // between them.
    let models = [256, 512, 1024].map(|size| through_bedrock("eu-west-2", size));
    let writes: Vec<_> = models
        .iter()
        .map(|model| writing(&embedders, model, texts("text", 80)))
        .collect();
    for write in writes {
        assert_eq!(write.await.unwrap(), Ok(80));
    }
    assert_eq!(embedders.held(), 3);
    assert_eq!(seen.most(), (8, 8));

    // What is held for Titan is not forgotten while it is written
    // through, however many other models are asked for meanwhile: and
    // forgotten once it is not, it is held afresh with the same calls.
    let slow = writing(&embedders, &models[0], texts("slow", 16));
    tokio::time::sleep(Duration::from_millis(200)).await;
    for model in 0..200 {
        embedders.embedder(&through_an_endpoint(model)).unwrap();
    }
    let again = writing(&embedders, &models[0], texts("slow", 16));
    let another = writing(&embedders, &models[2], texts("slow", 16));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(seen.documents.load(Ordering::SeqCst), 8);
    // The two calls kept for queries are still kept, whichever of the
    // models a query is for.
    let began = std::time::Instant::now();
    let query = |model: &Embedding, text: &str| {
        let embedder = embedders.embedder(model).unwrap().unwrap();
        let text = text.to_string();
        async move { embedder.query(&text).await.unwrap().unwrap().len() }
    };
    let sizes = tokio::join!(
        query(&models[1], "query one"),
        query(&models[2], "query two")
    );
    assert_eq!(sizes, (512, 1024));
    let took = began.elapsed();
    assert!(took < Duration::from_millis(800), "two took {took:?}");
    for write in [slow, again, another] {
        assert_eq!(write.await.unwrap(), Ok(16));
    }
    assert_eq!(seen.most(), (10, 8));
    assert_eq!(seen.calls.load(Ordering::SeqCst), 3 * 80 + 3 * 16 + 2);
}

/// A node that embeds through as many regions as it can gives each two
/// of its calls, and still has no more in flight than a node has.
async fn many_regions(seen: Arc<Seen>) {
    let regions = [
        "eu-west-1",
        "eu-west-3",
        "eu-north-1",
        "eu-south-1",
        "ap-south-1",
    ];
    let embedders = node(&regions);
    let already = seen.in_flight.load(Ordering::SeqCst);
    assert_eq!(already, 5, "those to the region that has stopped");
    let writes: Vec<_> = regions
        .iter()
        .map(|region| writing(&embedders, &through_bedrock(region, 256), texts("slow", 4)))
        .collect();
    let queries = regions.iter().map(|region| {
        let embedder = embedders
            .embedder(&through_bedrock(region, 256))
            .unwrap()
            .unwrap();
        async move { embedder.query("query").await.map(|made| made.is_some()) }
    });
    for searched in futures::future::join_all(queries).await {
        assert_eq!(searched, Ok(true));
    }
    for write in writes {
        assert_eq!(write.await.unwrap(), Ok(4));
    }
    let most = seen.most().0 - already;
    assert!((6..=10).contains(&most), "{most} at once");
    for region in regions {
        assert!(seen.most_to(region) <= 2, "{region}");
    }
}

/// Five regions, and four of them stopped, with as much written and
/// searched through each as it will take. Each holds its two calls, and
/// the fifth has its own two still: one for a write's texts, and one
/// kept for a search.
async fn all_but_one_stopped(seen: Arc<Seen>) {
    let stopped = ["us-west-1", "us-west-2", "us-east-1", "us-east-2"];
    let embedders = node(&[stopped.as_slice(), &["eu-west-2"]].concat());
    let well = through_bedrock("eu-west-2", 256);
    let before = seen.most_to("eu-west-2");
    let mut hung = Vec::new();
    for region in stopped {
        let through = through_bedrock(region, 256);
        for _ in 0..3 {
            hung.push(writing(&embedders, &through, texts("text", 20)));
        }
        for nth in 0..4 {
            let embedder = embedders.embedder(&through).unwrap().unwrap();
            let text = format!("query {nth}");
            hung.push(tokio::spawn(async move {
                let made = embedder.query(&text).await;
                made.map(|made| made.map_or(0, |made| made.len()))
            }));
        }
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    let began = std::time::Instant::now();
    let write = writing(&embedders, &well, texts("text", 4));
    let embedder = embedders.embedder(&well).unwrap().unwrap();
    assert_eq!(
        embedder.query("query one").await.map(|made| made.is_some()),
        Ok(true)
    );
    let searched = began.elapsed();
    assert!(
        searched < Duration::from_millis(800),
        "it took {searched:?}"
    );
    assert_eq!(write.await.unwrap(), Ok(4));
    let written = began.elapsed();
    assert!(
        written < Duration::from_millis(1_500),
        "it took {written:?}"
    );
    for region in ["us-west-1", "us-east-1", "us-east-2"] {
        assert_eq!(seen.most_to(region), 2, "{region}");
    }
    assert!(seen.most_to("eu-west-2").max(before) <= 5);
    assert!(hung.iter().all(|hung| !hung.is_finished()));
    for hung in hung {
        hung.abort();
    }
}

/// A node that embeds through two regions gives each half its calls: so
/// one that has stopped answering holds its own half, however much is
/// written and searched through it, and the other is embedded through as
/// if nothing were wrong.
async fn one_stopped(seen: Arc<Seen>) {
    let embedders = node(&["eu-west-2", STOPPED]);
    let (well, stopped) = (
        through_bedrock("eu-west-2", 256),
        through_bedrock(STOPPED, 256),
    );
    // Much is written and searched through the region that has stopped,
    // and none of it comes back.
    let mut hung = Vec::new();
    for _ in 0..3 {
        hung.push(writing(&embedders, &stopped, texts("text", 40)));
    }
    for nth in 0..6 {
        let embedder = embedders.embedder(&stopped).unwrap().unwrap();
        let text = format!("query {nth}");
        hung.push(tokio::spawn(async move {
            embedder
                .query(&text)
                .await
                .map(|made| made.map_or(0, |made| made.len()))
        }));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(seen.most_to(STOPPED), 5, "its half of the node's ten");

    // Through the other, a write and searches are answered in about
    // their own time: 16 texts, four at a time at 50 ms, and two queries.
    let began = std::time::Instant::now();
    let write = writing(&embedders, &well, texts("text", 16));
    let query = |text: &str| {
        let embedder = embedders.embedder(&well).unwrap().unwrap();
        let text = text.to_string();
        async move {
            embedder
                .query(&text)
                .await
                .map(|made| made.map(|made| made.len()))
        }
    };
    assert_eq!(query("query one").await, Ok(Some(256)));
    let searched = began.elapsed();
    assert!(
        searched < Duration::from_millis(800),
        "it took {searched:?}"
    );
    assert_eq!(write.await.unwrap(), Ok(16));
    let written = began.elapsed();
    assert!(
        written < Duration::from_millis(1_500),
        "it took {written:?}"
    );
    assert_eq!(query("query two").await, Ok(Some(256)));

    // Each region has had its half and no more, and the node its ten.
    assert_eq!((seen.most_to(STOPPED), seen.most_to("eu-west-2")), (5, 5));
    assert!(seen.most().0 <= 10, "{:?}", seen.most());
    assert!(hung.iter().all(|hung| !hung.is_finished()));
    for hung in hung {
        hung.abort();
    }
}
