//! Runs the real model, which is fetched on first use (about 400 MB):
//!
//!   cargo test --release -p vellev-embed --test local -- --ignored --nocapture
#![cfg(feature = "local")]

use std::sync::Arc;
use std::time::Instant;

use vellev_core::{Embedding, EmbeddingProvider};
use vellev_embed::{Embedders, Settings};

const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn bge_base_embeds_text_so_that_like_is_near_like() {
    let cache = std::env::var("VELLEV_MODEL_CACHE")
        .map(Into::into)
        .unwrap_or_else(|_| std::env::temp_dir().join("vellev-models"));
    let embedders = Embedders::new(Settings {
        cache,
        ..Settings::default()
    });
    let embedder = embedders
        .embedder(&Embedding {
            provider: EmbeddingProvider::Local,
            model: "BAAI/bge-base-en-v1.5".into(),
            dimensions: 768,
            query_prefix: Some(QUERY_PREFIX.into()),
            document_prefix: None,
            properties: vec![],
        })
        .unwrap()
        .unwrap();

    let documents: Vec<String> = [
        "Deploys go out on Tuesdays after the morning stand-up.",
        "The database is backed up to object storage every night.",
        "",
        "Hedgehogs are nocturnal and eat insects.",
    ]
    .map(String::from)
    .to_vec();
    let vectors = embedder.documents(&documents).await.unwrap();
    assert!(vectors[2].is_none());
    let vectors: Vec<Vec<f32>> = vectors.into_iter().flatten().collect();
    assert_eq!(vectors.len(), 3);
    for vector in &vectors {
        assert_eq!(vector.len(), 768);
        assert!(
            (cosine(vector, vector) - 1.0).abs() < 1e-3,
            "not unit length"
        );
    }

    let query = embedder
        .query("when do we release?")
        .await
        .unwrap()
        .unwrap();
    let nearness: Vec<f32> = vectors
        .iter()
        .map(|vector| cosine(&query, vector))
        .collect();
    assert!(
        nearness[0] > nearness[1] && nearness[0] > nearness[2],
        "{nearness:?}"
    );
    let as_document = embedder
        .documents(&["when do we release?".to_string()])
        .await
        .unwrap()
        .remove(0)
        .unwrap();
    assert!(
        cosine(&query, &as_document) < 0.999,
        "the query prefix changed nothing"
    );

    let embedder = Arc::new(embedder);
    let callers = (0..64).map(|n| {
        let embedder = embedder.clone();
        tokio::spawn(async move {
            let text = format!("Memory {n}: the deploy on day {n} went out late.");
            embedder.documents(&[text]).await.unwrap()
        })
    });
    let together = futures::future::join_all(callers).await;
    assert!(together
        .iter()
        .all(|vectors| vectors.as_ref().unwrap()[0].is_some()));

    let passage = "The quarterly review covered hiring, the migration off the old cluster, \
        an incident in the billing service, and plans for the next release. "
        .repeat(3);
    let batch: Vec<String> = (0..256).map(|n| format!("{n}. {passage}")).collect();
    let started = Instant::now();
    embedder.documents(&batch).await.unwrap();
    let took = started.elapsed();
    println!(
        "256 passages of about 75 words in {took:.2?}: {:.0} a second",
        256.0 / took.as_secs_f64()
    );
    let started = Instant::now();
    for n in 0..20 {
        embedder
            .query(&format!("what happened on day {n}?"))
            .await
            .unwrap();
    }
    println!("a short query in {:.1?}", started.elapsed() / 20);
}
