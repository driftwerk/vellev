//! A node with no AWS credentials, as the SDK itself finds it: in a test
//! binary of its own, since what credentials there are is the process's
//! environment, read when a client is first made.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::json;
use vellev_core::{Embedding, EmbeddingProvider, ModelFault};
use vellev_embed::{EmbedError, Embedders, Settings};

const TITAN: &str = "amazon.titan-embed-text-v2:0";

fn titan(dimensions: u32) -> Embedding {
    Embedding {
        provider: EmbeddingProvider::Bedrock {
            region: "eu-west-2".into(),
        },
        model: TITAN.into(),
        dimensions,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    }
}

/// The SDK reports credentials it could not load as a failure to send
/// the call, the same kind of error as a region it could not connect to.
/// It is the node's setup all the same: said as that, standing until the
/// node is put right, and saying nothing of the region, which was never
/// asked and is not taken to be out of reach.
#[test]
fn a_node_with_no_credentials_is_set_up_wrong_and_its_region_is_not_out_of_reach() {
    // Bedrock stands in on this machine, and counts what reaches it.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    // Before anything else runs: no credentials anywhere the SDK looks.
    // What is set for Bedrock alone comes before what is set for all of
    // Amazon's services: a developer's own must not send this anywhere
    // real.
    std::env::remove_var("AWS_ENDPOINT_URL_BEDROCK_RUNTIME");
    std::env::set_var("AWS_ENDPOINT_URL", url);
    for name in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ] {
        std::env::remove_var(name);
    }
    std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");
    std::env::set_var("AWS_CONFIG_FILE", "/nonexistent/vellev/config");
    std::env::set_var(
        "AWS_SHARED_CREDENTIALS_FILE",
        "/nonexistent/vellev/credentials",
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let reached = Arc::new(AtomicUsize::new(0));
        let counted = reached.clone();
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let app = Router::new().route(
            "/model/{model}/invoke",
            post(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                async { Json(json!({ "embedding": vec![1.0; 256] })) }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let node = Embedders::new(Settings {
            bedrock_regions: vec!["eu-west-2".into()],
            ..Settings::default()
        });
        let set_up_wrong = |made: EmbedError| {
            assert_eq!(made.lasting(), Some(ModelFault::Setup), "{made}");
            let said = made.to_string();
            assert!(
                said.contains("the node has no AWS credentials to call Bedrock with"),
                "{said}"
            );
            assert!(!said.contains("was not reached"), "{said}");
        };
        // A write, and then a search of the same model, which is not
        // asked again while the fault stands and says the same.
        let small = node.embedder(&titan(256)).unwrap().unwrap();
        set_up_wrong(small.documents(&["fox".to_string()]).await.unwrap_err());
        set_up_wrong(small.query("fox").await.unwrap_err());
        // Titan at other sizes, in the same region: each is another
        // model, asked for itself, with the region's calls. Were the
        // region taken to be out of reach, these would be told so
        // without being tried: each is tried, and finds the same fault.
        for dimensions in [512, 1024] {
            let other = node.embedder(&titan(dimensions)).unwrap().unwrap();
            set_up_wrong(other.query("hen").await.unwrap_err());
            set_up_wrong(other.documents(&["hen".to_string()]).await.unwrap_err());
        }
        assert_eq!(reached.load(Ordering::SeqCst), 0, "nothing was sent");
    });
}
