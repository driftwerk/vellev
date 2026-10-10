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

/// Sixteen writes of thirty passages at once, a write of one text beside
/// them, and a search: how long each takes on the real model.
///
///   cargo test --release -p vellev-embed --test local sixteen -- --ignored --nocapture
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn sixteen_writes_at_once_are_done_a_few_at_a_time_and_a_small_one_at_once() {
    let cache = std::env::var("VELLEV_MODEL_CACHE")
        .map(Into::into)
        .unwrap_or_else(|_| std::env::temp_dir().join("vellev-models"));
    let embedders = Embedders::new(Settings {
        cache,
        ..Settings::default()
    });
    let embedding = Embedding {
        provider: EmbeddingProvider::Local,
        model: "BAAI/bge-base-en-v1.5".into(),
        dimensions: 768,
        query_prefix: Some(QUERY_PREFIX.into()),
        document_prefix: None,
        properties: vec![],
    };
    let embedder = embedders.embedder(&embedding).unwrap().unwrap();
    // The model is loaded, and its runs sized, before anything is timed.
    let passage = "The quarterly review covered hiring, the migration off the old cluster, \
        an incident in the billing service, and plans for the next release. ";
    let write =
        |nth: usize| -> Vec<String> { (0..30).map(|n| format!("{nth}.{n}. {passage}")).collect() };
    embedder.documents(&write(99)).await.unwrap();
    let started = Instant::now();
    embedder.documents(&write(98)).await.unwrap();
    let alone = started.elapsed();
    println!("one write of thirty passages alone: {alone:.2?}");

    let started = Instant::now();
    let writes = (0..16).map(|nth| {
        let (embedder, texts) = (embedder.clone(), write(nth));
        async move {
            let made = embedder.documents(&texts).await;
            (made.map(|made| made.len()), started.elapsed())
        }
    });
    let beside = async {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let asked = Instant::now();
        let small = embedder.documents(&["one small write".to_string()]).await;
        let small = (small.map(|made| made.len()), asked.elapsed());
        let asked = Instant::now();
        let search = embedder.query("what did the review cover?").await;
        (small, search.map(|_| asked.elapsed()))
    };
    let (written, (small, search)) = tokio::join!(futures::future::join_all(writes), beside);
    for (nth, (made, after)) in written.iter().enumerate() {
        assert_eq!(made, &Ok(30), "write {nth}");
        println!("write {nth} done after {after:.2?}");
    }
    assert_eq!(small.0, Ok(1));
    println!("a write of one text beside them: {:.2?}", small.1);
    println!("a search beside them: {:.2?}", search.unwrap());
    let first = written.iter().map(|(_, after)| *after).min().unwrap();
    let last = written.iter().map(|(_, after)| *after).max().unwrap();
    println!("the first of sixteen after {first:.2?}, the last after {last:.2?}");
    assert!(
        first < last / 2,
        "they are not all done together at the end"
    );
}

/// The four cases that failed on the real model before writes were given
/// up on for getting nowhere rather than for how long they took, and a
/// local model was shared among tenants: forty closed-loop searchers and
/// eight beside a one-text and a thirty-text write; one tenant's 24 MB
/// text and 32 texts of 400 kB beside another's search and small writes.
///
///   cargo test --release -p vellev-embed --test local probes -- --ignored --nocapture
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probes_searches_and_long_texts_of_one_tenant_do_not_fail_anothers_writes() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let cache = std::env::var("VELLEV_MODEL_CACHE")
        .map(Into::into)
        .unwrap_or_else(|_| std::env::temp_dir().join("vellev-models"));
    let embedders = Embedders::new(Settings {
        cache,
        ..Settings::default()
    });
    let embedding = Embedding {
        provider: EmbeddingProvider::Local,
        model: "BAAI/bge-base-en-v1.5".into(),
        dimensions: 768,
        query_prefix: Some(QUERY_PREFIX.into()),
        document_prefix: None,
        properties: vec![],
    };
    let embedder = embedders.embedder(&embedding).unwrap().unwrap();
    let passage = "The quarterly review covered hiring, the migration off the old cluster, \
        an incident in the billing service, and plans for the next release. ";
    embedder
        .documents_for("warm", &[passage.to_string()])
        .await
        .unwrap();

    // Searchers, in a closed loop, for tenant "searchers".
    let search_beside = |searchers: usize| {
        let embedder = embedder.clone();
        async move {
            let going = Arc::new(AtomicBool::new(true));
            let made = Arc::new(AtomicUsize::new(0));
            let tasks: Vec<_> = (0..searchers)
                .map(|n| {
                    let (embedder, going, made) = (embedder.clone(), going.clone(), made.clone());
                    tokio::spawn(async move {
                        let mut nth = 0;
                        while going.load(Ordering::SeqCst) {
                            nth += 1;
                            let text = format!("what happened on day {n} of week {nth}?");
                            if embedder.query_for("searchers", &text).await.is_ok() {
                                made.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    })
                })
                .collect();
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let began = Instant::now();
            let one_text = vec!["one small write".to_string()];
            let thirty_texts: Vec<String> = (0..30).map(|n| format!("{n}. {passage}")).collect();
            let one = embedder.documents_for("writer", &one_text);
            let thirty = embedder.documents_for("writer", &thirty_texts);
            let (one, thirty) = tokio::join!(one, thirty);
            let took = began.elapsed();
            going.store(false, Ordering::SeqCst);
            futures::future::join_all(tasks).await;
            println!(
                "{searchers} searchers, {} searches made: a 1-text write and a 30-text write beside them in {took:.2?}",
                made.load(Ordering::SeqCst)
            );
            assert_eq!(one.map(|made| made.len()), Ok(1), "{searchers} searchers");
            assert_eq!(
                thirty.map(|made| made.len()),
                Ok(30),
                "{searchers} searchers"
            );
            took
        }
    };
    let eight = search_beside(8).await;
    let forty = search_beside(40).await;
    assert!(eight < std::time::Duration::from_secs(10), "{eight:?}");
    assert!(forty < std::time::Duration::from_secs(20), "{forty:?}");

    // One tenant's long texts, another's search and writes beside them.
    let long_beside = |name: &'static str, long: Vec<String>| {
        let embedder = embedder.clone();
        async move {
            let bulk = {
                let embedder = embedder.clone();
                tokio::spawn(async move { embedder.documents_for("long", &long).await })
            };
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let began = Instant::now();
            let search = embedder
                .query_for("other", "what did the review cover?")
                .await;
            let searched = began.elapsed();
            let began = Instant::now();
            let one = embedder.documents_for("other", &["one".to_string()]).await;
            let ten: Vec<String> = (0..10).map(|n| format!("{n}. {passage}")).collect();
            let ten = embedder.documents_for("other", &ten).await;
            let written = began.elapsed();
            let bulk = bulk.await.unwrap();
            println!("{name}: a search beside took {searched:.2?}, writes of 1 and 10 took {written:.2?}; the long texts: {:?}", bulk.as_ref().map(|made| made.len()));
            assert!(search.is_ok(), "{name}: {search:?}");
            assert_eq!(one.map(|made| made.len()), Ok(1), "{name}");
            assert_eq!(ten.map(|made| made.len()), Ok(10), "{name}");
            assert!(bulk.is_ok(), "{name}: {bulk:?}");
            assert!(
                searched < std::time::Duration::from_secs(10),
                "{name}: {searched:?}"
            );
        }
    };
    long_beside(
        "one text of 24 MB",
        vec![passage.repeat(24 * 1024 * 1024 / passage.len())],
    )
    .await;
    let kb400: Vec<String> = (0..32)
        .map(|n| format!("{n} {}", passage.repeat(400 * 1024 / passage.len())))
        .collect();
    long_beside("32 texts of 400 kB", kb400).await;
}
