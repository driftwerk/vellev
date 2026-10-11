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

/// The node's own model, fresh: nothing of it is loaded.
fn fresh() -> Arc<vellev_embed::Embedder> {
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
    embedders.embedder(&embedding).unwrap().unwrap()
}

const PASSAGE: &str = "The quarterly review covered hiring, the migration off the old cluster, \
    an incident in the billing service, and plans for the next release. ";

/// Text of about so many bytes.
fn of_length(n: usize, bytes: usize) -> String {
    let mut text = format!("{n}. ");
    while text.len() < bytes {
        text.push_str(PASSAGE);
    }
    text.truncate(bytes.max(4));
    text
}

/// What runs cost on this machine, by what is in them: each a write made
/// alone, of texts that make one run. The model is given a run's texts
/// padded to the longest, so a run costs about its texts times the
/// longest; a run is of as many texts as cost what thirty-two of 256
/// bytes do.
///
///   cargo test --release -p vellev-embed --test local probe_ -- --ignored --nocapture --test-threads 1
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_what_a_run_costs() {
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    for (name, texts, bytes) in [
        ("32 texts of 40 bytes", 32, 40),
        ("32 texts of 256 bytes", 32, 256),
        ("8 texts of 1,000 bytes", 8, 1_000),
        ("4 texts of 2,048 bytes", 4, 2_048),
        ("1 text of 2,048 bytes", 1, 2_048),
        ("4 texts of 400 kB", 4, 400 * 1024),
    ] {
        let mut took = Vec::new();
        for round in 0..3 {
            let texts: Vec<String> = (0..texts)
                .map(|n| of_length(round * 100 + n, bytes))
                .collect();
            let began = Instant::now();
            embedder.documents(&texts).await.unwrap();
            took.push(began.elapsed());
        }
        println!("PROBE a run of {name}: {took:.0?}");
    }
}

/// What a write costs by what its texts are made of: prose, rows of
/// JSON, and punctuation, which carry more tokens to the byte. Each is a
/// write made alone, so it is as many runs as the node makes of it: the
/// time is the write's, and the longest any search would wait behind it
/// is one of its runs.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_what_dense_text_costs() {
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    let json = |n: usize, bytes: usize| {
        let mut text = String::new();
        let mut field = n;
        while text.len() < bytes {
            text.push_str(&format!(
                "{{\"id\":{field},\"ok\":true,\"v\":[1.5,-2,3e4]}},"
            ));
            field += 1;
        }
        text.truncate(bytes);
        text
    };
    let bangs = |_: usize, bytes: usize| "!".repeat(bytes);
    let prose = |n: usize, bytes: usize| of_length(n, bytes);
    type Makes<'a> = &'a dyn Fn(usize, usize) -> String;
    let kinds: [(&str, Makes); 3] = [("prose", &prose), ("JSON", &json), ("'!'", &bangs)];
    for (texts, bytes) in [(32, 64), (32, 256), (8, 256), (4, 2_048), (1, 2_048)] {
        for (kind, make) in kinds {
            let mut took = Vec::new();
            for round in 0..2 {
                let texts: Vec<String> = (0..texts).map(|n| make(round * 100 + n, bytes)).collect();
                let began = Instant::now();
                embedder.documents(&texts).await.unwrap();
                took.push(began.elapsed());
            }
            println!("PROBE a write of {texts} texts of {bytes} bytes of {kind}: {took:.0?}");
        }
    }
}

/// Searchers in a closed loop, and a write of one text and one of thirty
/// beside them.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_searchers_beside_writes() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    for searchers in [8, 40, 64] {
        let going = Arc::new(AtomicBool::new(true));
        let tasks: Vec<_> = (0..searchers)
            .map(|n| {
                let (embedder, going) = (embedder.clone(), going.clone());
                tokio::spawn(async move {
                    let (mut made, mut failed, mut slowest) = (0, 0, std::time::Duration::ZERO);
                    let mut nth = 0;
                    while going.load(Ordering::SeqCst) {
                        nth += 1;
                        let text =
                            format!("what happened on day {n} of week {nth} of {searchers}?");
                        let began = Instant::now();
                        match embedder.query(&text).await {
                            Ok(_) => made += 1,
                            Err(_) => failed += 1,
                        }
                        slowest = slowest.max(began.elapsed());
                    }
                    (made, failed, slowest)
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let began = Instant::now();
        let one = vec![of_length(1, 40)];
        let thirty: Vec<String> = (0..30).map(|n| of_length(n, 160)).collect();
        let (one, thirty) = tokio::join!(embedder.documents(&one), embedder.documents(&thirty));
        let wrote = began.elapsed();
        going.store(false, Ordering::SeqCst);
        let searched = futures::future::join_all(tasks).await;
        let (mut made, mut failed, mut slowest) = (0, 0, std::time::Duration::ZERO);
        for (m, f, s) in searched.into_iter().map(|searched| searched.unwrap()) {
            (made, failed, slowest) = (made + m, failed + f, slowest.max(s));
        }
        println!(
            "PROBE {searchers} searchers: writes of 1 and 30 texts beside them in {wrote:.2?}; \
             {made} searches made, {failed} failed, the slowest {slowest:.2?}"
        );
        assert_eq!(one.map(|made| made.len()), Ok(1), "{searchers} searchers");
        assert_eq!(
            thirty.map(|made| made.len()),
            Ok(30),
            "{searchers} searchers"
        );
        assert_eq!(failed, 0, "{searchers} searchers");
    }
}

/// What searches wait behind: other searches, long ones above all, and
/// a bulk write of dense text. Nothing is asserted of how long they
/// take or whether they are answered in time, which is not promised:
/// the numbers are for the README.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_what_searches_wait_behind() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    // Forty searchers of 2,048 bytes in a closed loop, and six short
    // searches made beside them, one after another.
    let going = Arc::new(AtomicBool::new(true));
    let long: Vec<_> = (0..40)
        .map(|n| {
            let (embedder, going) = (embedder.clone(), going.clone());
            tokio::spawn(async move {
                let (mut made, mut failed, mut nth) = (0, 0, 0);
                while going.load(Ordering::SeqCst) {
                    nth += 1;
                    match embedder.query(&of_length(n * 1_000 + nth, 2_048)).await {
                        Ok(_) => made += 1,
                        Err(_) => failed += 1,
                    }
                }
                (made, failed)
            })
        })
        .collect();
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    let mut short = Vec::new();
    for n in 0..6 {
        let began = Instant::now();
        let asked = embedder.query(&format!("a short search, number {n}")).await;
        short.push((asked.is_ok(), began.elapsed()));
    }
    going.store(false, Ordering::SeqCst);
    let (mut made, mut failed) = (0, 0);
    for (m, f) in futures::future::join_all(long)
        .await
        .into_iter()
        .map(|l| l.unwrap())
    {
        (made, failed) = (made + m, failed + f);
    }
    println!(
        "PROBE beside 40 searchers of 2,048 bytes ({made} made, {failed} failed): six short \
         searches (answered, in): {short:.2?}"
    );

    // A bulk write of 320 rows of JSON of 256 bytes, and beside it eight
    // searchers of 2,048 bytes and eight short ones, in closed loops.
    let rows: Vec<String> = (0..320)
        .map(|n| {
            let mut row = String::new();
            let mut field = n * 100;
            while row.len() < 256 {
                row.push_str(&format!(
                    "{{\"id\":{field},\"ok\":true,\"v\":[1.5,-2,3e4]}},"
                ));
                field += 1;
            }
            row.truncate(256);
            row
        })
        .collect();
    let going = Arc::new(AtomicBool::new(true));
    let searchers: Vec<_> = (0..16)
        .map(|n| {
            let (embedder, going) = (embedder.clone(), going.clone());
            tokio::spawn(async move {
                let (mut made, mut failed, mut slowest) = (0, 0, std::time::Duration::ZERO);
                let mut nth = 0;
                while going.load(Ordering::SeqCst) {
                    nth += 1;
                    let text = match n % 2 {
                        0 => of_length(n * 1_000 + nth + 500_000, 2_048),
                        _ => format!("short search {n} number {nth} beside a write"),
                    };
                    let began = Instant::now();
                    match embedder.query(&text).await {
                        Ok(_) => made += 1,
                        Err(_) => failed += 1,
                    }
                    slowest = slowest.max(began.elapsed());
                }
                (n % 2 == 0, made, failed, slowest)
            })
        })
        .collect();
    let began = Instant::now();
    let written = embedder.documents(&rows).await.map(|made| made.len());
    let wrote = began.elapsed();
    going.store(false, Ordering::SeqCst);
    let mut of_long = (0, 0, std::time::Duration::ZERO);
    let mut of_short = of_long;
    for searched in futures::future::join_all(searchers).await {
        let (long, made, failed, slowest) = searched.unwrap();
        let of = if long { &mut of_long } else { &mut of_short };
        *of = (of.0 + made, of.1 + failed, of.2.max(slowest));
    }
    println!(
        "PROBE beside a write of 320 rows of JSON of 256 bytes ({written:?} in {wrote:.2?}): long \
         searches (made, failed, slowest) {of_long:.2?}, short ones {of_short:.2?}"
    );
}

/// One caller's long texts, and another's search and small writes beside
/// them.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_long_texts_beside_a_search_and_small_writes() {
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    let cases: [(&str, Vec<String>); 2] = [
        ("one text of 24 MB", vec![of_length(0, 24 * 1024 * 1024)]),
        (
            "32 texts of 400 kB",
            (0..32).map(|n| of_length(n, 400 * 1024)).collect(),
        ),
    ];
    for (name, long) in cases {
        let bulk = {
            let embedder = embedder.clone();
            tokio::spawn(async move {
                let began = Instant::now();
                (
                    embedder.documents(&long).await.map(|made| made.len()),
                    began.elapsed(),
                )
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let began = Instant::now();
        let search = embedder
            .query(&format!("what did the review cover in {name}?"))
            .await;
        let searched = began.elapsed();
        let began = Instant::now();
        let one = embedder.documents(&[of_length(1, 40)]).await;
        let ten: Vec<String> = (0..10).map(|n| of_length(n, 160)).collect();
        let ten = embedder.documents(&ten).await;
        let wrote = began.elapsed();
        let (bulk, bulk_took) = bulk.await.unwrap();
        println!(
            "PROBE {name} ({bulk:?} in {bulk_took:.2?}): a search beside took {searched:.2?}, \
             writes of 1 and 10 took {wrote:.2?}"
        );
        assert!(search.is_ok(), "{name}: {search:?}");
        assert_eq!(one.map(|made| made.len()), Ok(1), "{name}");
        assert_eq!(ten.map(|made| made.len()), Ok(10), "{name}");
        assert!(bulk.is_ok(), "{name}: {bulk:?}");
    }
}

/// Writes made all at once, as many and as long as each row says: every
/// one is made.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_writes_made_at_once() {
    let embedder = fresh();
    embedder.documents(&[of_length(0, 40)]).await.unwrap();
    // (writes, texts in each, bytes a text)
    for (writes, texts, bytes) in [
        (10, 200, 40),
        (20, 200, 40),
        (100, 10, 40),
        (50, 40, 160),
        (5, 100, 1_000),
        (16, 16, 2_048),
        (4, 32, 400 * 1024),
    ] {
        let began = Instant::now();
        let made = (0..writes).map(|write| {
            let embedder = embedder.clone();
            async move {
                let texts: Vec<String> = (0..texts)
                    .map(|n| of_length(write * 1_000 + n, bytes))
                    .collect();
                let began = Instant::now();
                (
                    embedder.documents(&texts).await.map(|made| made.len()),
                    began.elapsed(),
                )
            }
        });
        let made = futures::future::join_all(made).await;
        let failed: Vec<_> = made.iter().filter(|(made, _)| made.is_err()).collect();
        let first = made.iter().map(|(_, took)| *took).min().unwrap();
        println!(
            "PROBE {writes} writes of {texts} texts of {bytes} bytes at once: all in {:.2?}, the \
             first in {first:.2?}, {} failed {:?}",
            began.elapsed(),
            failed.len(),
            failed.first()
        );
        assert!(failed.is_empty());
    }
}

/// Six hundred callers at a model that has yet to load: every one is
/// answered.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "fetches and runs the model"]
async fn probe_many_callers_at_a_cold_load() {
    let embedder = fresh();
    let began = Instant::now();
    let made = (0..600).map(|n| {
        let embedder = embedder.clone();
        async move {
            let began = Instant::now();
            (
                embedder
                    .documents(&[of_length(n, 40)])
                    .await
                    .map(|made| made.len()),
                began.elapsed(),
            )
        }
    });
    let made = futures::future::join_all(made).await;
    let failed: Vec<_> = made.iter().filter(|(made, _)| made.is_err()).collect();
    let first = made.iter().map(|(_, took)| *took).min().unwrap();
    println!(
        "PROBE 600 callers of 1 text at a cold load: all in {:.2?}, the first in {first:.2?}, {} \
         failed {:?}",
        began.elapsed(),
        failed.len(),
        failed.first()
    );
    assert!(failed.is_empty());
}
