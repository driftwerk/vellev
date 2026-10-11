use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use vellev_core::{Embedding, EmbeddingProvider, HttpApi};
use vellev_embed::{EmbedError, Embedder, Embedders, Provider, Purpose, Result, Settings};

/// Embeds each text as its length and its count of spaces, or as told.
#[derive(Debug, Default)]
struct Fake {
    asked: Mutex<Vec<Vec<String>>>,
    purposes: Mutex<Vec<Purpose>>,
    answer: Option<Vec<Vec<f32>>>,
    failures: Mutex<usize>,
    /// What every call is refused with, while there is something.
    refusing: Mutex<Option<EmbedError>>,
}

#[async_trait]
impl Provider for Fake {
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        tokio::task::yield_now().await;
        self.purposes.lock().unwrap().push(purpose);
        let failing = {
            let mut failures = self.failures.lock().unwrap();
            let failing = *failures > 0;
            *failures = failures.saturating_sub(1);
            failing
        };
        if let Some(refusal) = self.refusing.lock().unwrap().clone() {
            return Err(refusal);
        }
        if failing {
            return Err(EmbedError::Failed {
                model: "fake".into(),
                reason: "not now".into(),
            });
        }
        let vectors = texts
            .iter()
            .map(|text| vec![text.len() as f32, text.matches(' ').count() as f32])
            .collect();
        self.asked.lock().unwrap().push(texts);
        Ok(self.answer.clone().unwrap_or(vectors))
    }
}

/// A model that never answers.
#[derive(Debug)]
struct Silent;

#[async_trait]
impl Provider for Silent {
    async fn embed(&self, _texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        std::future::pending().await
    }
}

#[tokio::test(start_paused = true)]
async fn a_search_waits_only_so_long_for_its_vector_whoever_asked_for_it_first() {
    let silent = Embedder::new(Arc::new(Silent), embedding(EmbeddingProvider::Local));
    let started = tokio::time::Instant::now();
    // Two ask for the same text, the second a minute after the first and
    // waiting on it; and one for a text too long to be kept.
    let long = "x".repeat(2_000);
    let later = async {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        let asked = tokio::time::Instant::now();
        (silent.query("fox").await, asked.elapsed())
    };
    let (first, (second, waited), unkept) =
        tokio::join!(silent.query("fox"), later, silent.query(&long));
    for unanswered in [first, second, unkept] {
        assert!(
            matches!(&unanswered, Err(EmbedError::Failed { reason, .. }) if reason.contains("within")),
            "{unanswered:?}"
        );
    }
    assert_eq!(waited, vellev_embed::QUERY_WITHIN, "each its own while");
    assert_eq!(
        started.elapsed(),
        std::time::Duration::from_secs(60) + vellev_embed::QUERY_WITHIN
    );
}

/// Whether a query for `text` sent it to the model itself, and how the
/// query ended.
async fn sent(embedder: &Embedder, text: &str) -> (bool, Result<Option<Vec<f32>>>) {
    let sent = AtomicBool::new(false);
    let answer = embedder.query_noting(text, &sent).await;
    (sent.load(Ordering::SeqCst), answer)
}

#[tokio::test]
async fn a_query_says_whether_it_sent_its_text_to_the_model_itself() {
    let (fake, lengths) = embedder(Fake::default());
    let asked = || fake.asked.lock().unwrap().len();
    assert!(sent(&lengths, "fox").await.0, "never asked before");
    assert!(!sent(&lengths, "fox").await.0, "its vector is kept");
    assert!(sent(&lengths, "hen").await.0, "another text");
    assert_eq!(asked(), 2);
    // A blank text is sent to no model, and one too long to keep to the
    // model every time.
    let (blank, none) = sent(&lengths, "  ").await;
    assert!(!blank && none.unwrap().is_none());
    let long = "x".repeat(2_000);
    assert!(sent(&lengths, &long).await.0 && sent(&lengths, &long).await.0);
    assert_eq!(asked(), 4);

    // A model that refuses for good is asked once, and not again while
    // the fault stands: the query refused for the fault sent nothing.
    let refusing = Fake {
        refusing: Mutex::new(Some(EmbedError::Unusable {
            model: "fake".into(),
            reason: "no such key".into(),
            fault: vellev_core::ModelFault::Setup,
        })),
        ..Fake::default()
    };
    let (fake, unusable) = embedder(refusing);
    let (first, refused) = sent(&unusable, "fox").await;
    assert!(first && refused.is_err());
    let (again, refused) = sent(&unusable, "fox").await;
    assert!(!again && refused.is_err(), "the fault stands");
    assert_eq!(fake.purposes.lock().unwrap().len(), 1, "asked once");
}

/// A model that answers each call only when told to, and counts them.
#[derive(Debug)]
struct Held {
    calls: AtomicUsize,
    answer: tokio::sync::Semaphore,
}

#[async_trait]
impl Provider for Held {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.answer.acquire().await.unwrap().forget();
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 0.0])
            .collect())
    }
}

#[tokio::test(start_paused = true)]
async fn a_query_that_waits_for_a_vector_another_is_having_made_sent_nothing_itself() {
    let held = Arc::new(Held {
        calls: AtomicUsize::new(0),
        answer: tokio::sync::Semaphore::new(0),
    });
    let embedder = Arc::new(Embedder::new(
        held.clone(),
        embedding(EmbeddingProvider::Local),
    ));
    let asking = |embedder: &Arc<Embedder>| {
        let embedder = embedder.clone();
        tokio::spawn(async move { sent(&embedder, "fox").await })
    };
    // The first is with the model when the second and third ask.
    let first = asking(&embedder);
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let (second, third) = (asking(&embedder), asking(&embedder));
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    assert_eq!(held.calls.load(Ordering::SeqCst), 1);
    held.answer.add_permits(1);
    for (asked, itself) in [(first, true), (second, false), (third, false)] {
        let (sent, vector) = asked.await.unwrap();
        assert_eq!((sent, vector.unwrap()), (itself, Some(vec![9.0, 0.0])));
    }
    assert_eq!(held.calls.load(Ordering::SeqCst), 1);

    // One that stops waiting for the model sent the text all the same;
    // whoever waited on it asks the model in its place, and says so.
    let (gone, waiting) = (
        tokio::spawn({
            let embedder = embedder.clone();
            async move {
                let sent = Arc::new(AtomicBool::new(false));
                let asked = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    embedder.query_noting("owl", &sent),
                );
                (asked.await.is_err(), sent.load(Ordering::SeqCst))
            }
        }),
        tokio::spawn({
            let embedder = embedder.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                sent(&embedder, "owl").await
            }
        }),
    );
    assert_eq!(gone.await.unwrap(), (true, true), "given up, and sent");
    held.answer.add_permits(8);
    let (sent, vector) = waiting.await.unwrap();
    assert_eq!((sent, vector.unwrap()), (true, Some(vec![9.0, 0.0])));
    assert_eq!(held.calls.load(Ordering::SeqCst), 3);
}

/// Counts the calls made for texts that hold `TARGET`.
#[derive(Debug, Default)]
struct Tally(AtomicUsize);

#[async_trait]
impl Provider for Tally {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        if texts.iter().any(|text| text.contains("TARGET")) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 0.0])
            .collect())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_model_is_never_asked_by_a_query_that_does_not_say_so() {
    let tally = Arc::new(Tally::default());
    let embedder = Arc::new(Embedder::new(
        tally.clone(),
        embedding(EmbeddingProvider::Local),
    ));
    let mut put_out = 0;
    for round in 0..200 {
        // The target is asked, then as many others as leave it the
        // oldest kept: the next text asked puts it out.
        let target = format!("TARGET {round}");
        assert!(sent(&embedder, &target).await.0);
        for filler in 0..1_023 {
            let filler = format!("filler {round} {filler}");
            embedder.query(&filler).await.unwrap();
        }
        // Another text is asked while the target is asked over and over:
        // somewhere among the askings it is put out, and asked of the
        // model again. Each asking says so exactly when the model was
        // asked for it.
        let putting_out = tokio::spawn({
            let embedder = embedder.clone();
            async move {
                for _ in 0..(round % 7) * 50 {
                    std::hint::spin_loop();
                }
                embedder.query(&format!("puts out {round}")).await.unwrap();
            }
        });
        for _ in 0..64 {
            let before = tally.0.load(Ordering::SeqCst);
            let (said, _) = sent(&embedder, &target).await;
            let asked = tally.0.load(Ordering::SeqCst) != before;
            assert_eq!(said, asked, "round {round}");
            put_out += usize::from(asked);
        }
        putting_out.await.unwrap();
    }
    assert!(put_out > 0, "the target was never put out while asked");
}

fn embedding(provider: EmbeddingProvider) -> Embedding {
    Embedding {
        provider,
        model: "fake".into(),
        dimensions: 2,
        query_prefix: Some("find: ".into()),
        document_prefix: None,
        properties: vec!["title".into(), "labels".into(), "body".into()],
    }
}

fn embedder(fake: Fake) -> (Arc<Fake>, Embedder) {
    let fake = Arc::new(fake);
    (
        fake.clone(),
        Embedder::new(fake, embedding(EmbeddingProvider::Local)),
    )
}

fn texts(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|text| text.to_string()).collect()
}

#[tokio::test]
async fn blank_text_has_no_vector_and_is_never_sent() {
    let (fake, embedder) = embedder(Fake::default());
    let vectors = embedder
        .documents(&texts(&["fox", "", "a hen", " \n\t"]))
        .await
        .unwrap();
    assert_eq!(
        vectors,
        [Some(vec![3.0, 0.0]), None, Some(vec![5.0, 1.0]), None]
    );
    assert_eq!(*fake.asked.lock().unwrap(), [texts(&["fox", "a hen"])]);

    assert_eq!(
        embedder.documents(&texts(&["", "  "])).await.unwrap(),
        [None, None]
    );
    assert_eq!(embedder.documents(&[]).await.unwrap(), []);
    assert_eq!(embedder.query(" ").await.unwrap(), None);
    assert_eq!(fake.asked.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn each_kind_of_text_carries_its_prefix() {
    let (fake, embedder) = embedder(Fake::default());
    embedder.documents(&texts(&["fox"])).await.unwrap();
    assert_eq!(embedder.query("fox").await.unwrap(), Some(vec![9.0, 1.0]));
    assert_eq!(
        *fake.asked.lock().unwrap(),
        [texts(&["fox"]), texts(&["find: fox"])]
    );

    let prefixed = Embedding {
        query_prefix: None,
        document_prefix: Some("store: ".into()),
        ..embedding(EmbeddingProvider::Local)
    };
    let fake = Arc::new(Fake::default());
    let embedder = Embedder::new(fake.clone(), prefixed);
    embedder.documents(&texts(&["fox"])).await.unwrap();
    embedder.query("fox").await.unwrap();
    assert_eq!(
        *fake.asked.lock().unwrap(),
        [texts(&["store: fox"]), texts(&["fox"])]
    );
}

#[tokio::test]
async fn a_repeated_query_is_embedded_once_and_the_oldest_are_forgotten() {
    let (fake, embedder) = embedder(Fake::default());
    let first = embedder.query("fox").await.unwrap();
    assert_eq!(embedder.query("fox").await.unwrap(), first);
    embedder.query("hen").await.unwrap();
    embedder.documents(&texts(&["fox"])).await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), 3);

    for n in 0..1_024 {
        embedder.query(&format!("query {n}")).await.unwrap();
    }
    let before = fake.asked.lock().unwrap().len();
    embedder.query("query 1023").await.unwrap();
    embedder.query("query 0").await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), before);
    embedder.query("fox").await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), before + 1);
}

/// The vector of a search of up to 1,024 bytes is kept, by its own
/// length and not its prefix's: one of a byte more is asked of the model
/// each time.
#[tokio::test]
async fn a_search_of_up_to_1024_bytes_is_kept_and_a_longer_one_is_not() {
    let (fake, lengths) = embedder(Fake::default());
    let asked = || fake.asked.lock().unwrap().len();
    for (bytes, kept) in [(512, true), (1_023, true), (1_024, true), (1_025, false)] {
        let before = asked();
        let text = "x".repeat(bytes);
        let first = lengths.query(&text).await.unwrap();
        assert_eq!(lengths.query(&text).await.unwrap(), first);
        let asked_again = asked() - before == 2;
        assert_eq!(asked_again, !kept, "{bytes} bytes");
    }
}

#[tokio::test]
async fn an_answer_that_does_not_fit_is_refused_and_one_of_another_size_for_good() {
    let malformed = |reason: &str| EmbedError::Malformed {
        model: "fake".into(),
        reason: reason.into(),
    };
    let unusable = |reason: &str| EmbedError::Unusable {
        model: "fake".into(),
        reason: reason.into(),
        fault: vellev_core::ModelFault::Provider,
    };
    for (answer, expected) in [
        (vec![], malformed("0 vectors for 1 texts")),
        (
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            malformed("2 vectors for 1 texts"),
        ),
        (
            vec![vec![1.0, 2.0, 3.0]],
            unusable("a vector of 3 dimensions, not 2"),
        ),
        (vec![vec![]], unusable("a vector of 0 dimensions, not 2")),
        (
            vec![vec![1.0, f32::NAN]],
            malformed("a vector that is not finite"),
        ),
        (
            vec![vec![f32::INFINITY, 0.0]],
            malformed("a vector that is not finite"),
        ),
    ] {
        let fake = Fake {
            answer: Some(answer),
            ..Fake::default()
        };
        let (_, embedder) = embedder(fake);
        assert_eq!(
            embedder.documents(&texts(&["fox"])).await,
            Err(expected.clone())
        );
        assert_eq!(embedder.query("fox").await, Err(expected));
    }
}

#[tokio::test(start_paused = true)]
async fn a_model_that_cannot_be_used_is_not_asked_again_while_the_fault_stands() {
    let (fake, embedder) = embedder(Fake::default());
    let asked = || fake.purposes.lock().unwrap().len();
    let unusable = EmbedError::Unusable {
        model: "fake".into(),
        reason: "401 Unauthorized: no such key".into(),
        fault: vellev_core::ModelFault::Setup,
    };
    *fake.refusing.lock().unwrap() = Some(unusable.clone());
    assert_eq!(embedder.query("fox").await, Err(unusable.clone()));
    assert_eq!(embedder.query("hen").await, Err(unusable.clone()));
    assert_eq!(
        embedder.documents(&texts(&["owl"])).await,
        Err(unusable.clone())
    );
    assert_eq!(asked(), 1, "asked once, and taken at its word after");
    assert_eq!(
        embedder.documents(&texts(&[" "])).await,
        Ok(vec![None]),
        "blank text needs no model"
    );

    // The setup is put right: it is found so once the fault no longer
    // stands, and not before.
    *fake.refusing.lock().unwrap() = None;
    tokio::time::advance(vellev_embed::FAULT_STANDS - std::time::Duration::from_millis(1)).await;
    assert_eq!(embedder.query("fox").await, Err(unusable.clone()));
    assert_eq!(asked(), 1);
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert_eq!(embedder.query("fox").await, Ok(Some(vec![9.0, 1.0])));
    assert_eq!(asked(), 2);

    // However long it is told to stand, it stands no less than a second
    // and no more than an hour.
    let second = std::time::Duration::from_secs(1);
    let ms = std::time::Duration::from_millis(1);
    for (told, stands) in [
        (std::time::Duration::ZERO, second),
        (ms, second),
        (std::time::Duration::MAX, second * 3_600),
    ] {
        let fake = Arc::new(Fake::default());
        *fake.refusing.lock().unwrap() = Some(unusable.clone());
        let embedder =
            Embedder::new(fake.clone(), embedding(EmbeddingProvider::Local)).standing_for(told);
        let asked = || fake.purposes.lock().unwrap().len();
        assert_eq!(embedder.query("fox").await, Err(unusable.clone()));
        tokio::time::advance(stands - ms).await;
        assert_eq!(embedder.query("fox").await, Err(unusable.clone()));
        assert_eq!(asked(), 1, "told {told:?}, it still stands");
        tokio::time::advance(ms).await;
        assert_eq!(embedder.query("fox").await, Err(unusable.clone()));
        assert_eq!(asked(), 2, "told {told:?}, it stands no longer");
    }

    // A fault of the model's own making stands likewise: vectors of
    // another size than the collection's.
    let wrong_size = Fake {
        answer: Some(vec![vec![1.0, 2.0, 3.0]]),
        ..Fake::default()
    };
    let (fake, embedder) = self::embedder(wrong_size);
    assert!(embedder.query("fox").await.is_err());
    assert!(embedder.query("hen").await.is_err());
    assert_eq!(fake.purposes.lock().unwrap().len(), 1);

    // A failure that may pass is asked about every time.
    let (fake, embedder) = self::embedder(Fake {
        failures: Mutex::new(2),
        ..Fake::default()
    });
    assert!(embedder.query("fox").await.is_err());
    assert!(embedder.query("fox").await.is_err());
    assert_eq!(embedder.query("fox").await, Ok(Some(vec![9.0, 1.0])));
    assert_eq!(fake.purposes.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn collections_that_embed_with_one_model_share_what_is_held_for_it() {
    let embedders = Embedders::new(Settings::default());
    assert!(embedders
        .embedder(&embedding(EmbeddingProvider::None))
        .unwrap()
        .is_none());
    assert_eq!(embedders.held(), 0);

    let http = |url: &str| {
        embedding(EmbeddingProvider::Http {
            url: url.into(),
            api: HttpApi::Tei,
        })
    };
    let first = embedders
        .embedder(&http("http://one.test"))
        .unwrap()
        .unwrap();
    // One model, however the collections that embed with it word their
    // texts and whichever properties they embed: and each embedder is
    // still its collection's own way of embedding.
    let reworded = Embedding {
        query_prefix: None,
        document_prefix: Some("store: ".into()),
        properties: vec!["title".into()],
        ..http("http://one.test")
    };
    let other = embedders.embedder(&reworded).unwrap().unwrap();
    assert_eq!(embedders.held(), 1);
    assert_eq!(first.embedding(), &http("http://one.test"));
    assert_eq!(other.embedding(), &reworded);
    // Another endpoint is another model.
    embedders
        .embedder(&http("http://two.test"))
        .unwrap()
        .unwrap();
    assert_eq!(embedders.held(), 2);
}

#[cfg(feature = "local")]
#[test]
fn a_model_no_build_runs_is_refused_when_asked_for_and_asking_builds_nothing() {
    let embedders = Embedders::new(Settings {
        bedrock_regions: vec!["eu-west-2".into()],
        ..Settings::default()
    });
    let with = |provider: EmbeddingProvider, model: &str, dimensions| Embedding {
        model: model.into(),
        dimensions,
        ..embedding(provider)
    };
    let bedrock = || EmbeddingProvider::Bedrock {
        region: "eu-west-2".into(),
    };
    let http = || EmbeddingProvider::Http {
        url: "http://one.test".into(),
        api: HttpApi::Openai,
    };
    let no_such = |reason: &str| Err(EmbedError::NoSuchModel(reason.into()));
    let cases = [
        (
            with(EmbeddingProvider::Local, "BAAI/bge-base-en-v1.5", 768),
            Ok(()),
        ),
        (
            with(EmbeddingProvider::Local, "BAAI/bge-base-en-v1.5", 1024),
            no_such("BAAI/bge-base-en-v1.5 has 768 dimensions, not 1024"),
        ),
        (
            with(EmbeddingProvider::Local, "acme/unheard-of", 768),
            no_such("acme/unheard-of cannot be run locally"),
        ),
        (with(bedrock(), "amazon.titan-embed-text-v2:0", 256), Ok(())),
        (with(bedrock(), "amazon.titan-embed-text-v2:0", 512), Ok(())),
        (
            with(bedrock(), "amazon.titan-embed-text-v2:0", 1024),
            Ok(()),
        ),
        (
            with(bedrock(), "amazon.titan-embed-text-v2:0", 768),
            no_such("amazon.titan-embed-text-v2:0 has 256, 512 or 1024 dimensions, not 768"),
        ),
        (
            with(bedrock(), "cohere.embed-english-v3", 1024),
            no_such(
                "cohere.embed-english-v3 is not amazon.titan-embed-text-v2, \
                 the one Bedrock model that can be embedded with",
            ),
        ),
        (with(http(), "whatever the endpoint serves", 3), Ok(())),
        (with(EmbeddingProvider::None, "client", 3), Ok(())),
    ];
    // Asking builds and keeps nothing, whatever is asked.
    for (embedding, admitted) in &cases {
        assert_eq!(&embedders.admits(embedding), admitted, "{embedding:?}");
    }
    assert_eq!(embedders.held(), 0);
    // Building one says the same.
    for (embedding, admitted) in cases {
        let built = embedders.embedder(&embedding).map(drop);
        assert_eq!(built, admitted, "{embedding:?}");
        if let Err(refused) = built {
            assert_eq!(
                refused.lasting(),
                Some(vellev_core::ModelFault::Build),
                "met in a collection there already, it is this build that cannot"
            );
        }
    }
    assert_eq!(embedders.held(), 5, "one for each that embeds");
}

#[tokio::test]
async fn a_query_asked_by_many_at_once_is_embedded_once_and_a_failure_is_not_kept() {
    let (fake, embedder) = embedder(Fake::default());
    let embedder = Arc::new(embedder);
    let asking = (0..8).map(|_| {
        let embedder = embedder.clone();
        tokio::spawn(async move { embedder.query("fox").await.unwrap() })
    });
    for answer in futures::future::join_all(asking).await {
        assert_eq!(answer.unwrap(), Some(vec![9.0, 1.0]));
    }
    assert_eq!(fake.asked.lock().unwrap().len(), 1);
    assert_eq!(*fake.purposes.lock().unwrap(), [Purpose::Query]);
    embedder.documents(&texts(&["fox"])).await.unwrap();
    assert_eq!(fake.purposes.lock().unwrap()[1], Purpose::Document);

    let failing = Fake {
        failures: Mutex::new(1),
        ..Fake::default()
    };
    let (fake, embedder) = self::embedder(failing);
    assert!(embedder.query("hen").await.is_err());
    assert_eq!(embedder.query("hen").await.unwrap(), Some(vec![9.0, 1.0]));
    assert_eq!(embedder.query("hen").await.unwrap(), Some(vec![9.0, 1.0]));
    assert_eq!(fake.purposes.lock().unwrap().len(), 2);

    let long = "long ".repeat(300);
    embedder.query(&long).await.unwrap();
    embedder.query(&long).await.unwrap();
    assert_eq!(fake.purposes.lock().unwrap().len(), 4);
}

/// A collection names the model it embeds through Bedrock with, and
/// may name any. Only Titan is one, by either of its names: a name of
/// the collection's own making is refused when the collection is made,
/// and nothing is held for it.
#[test]
fn a_bedrock_model_that_is_not_titan_is_refused_when_asked_for() {
    let embedders = Embedders::new(Settings {
        bedrock_regions: vec!["eu-west-2".into()],
        ..Settings::default()
    });
    let named = |model: &str| Embedding {
        model: model.into(),
        dimensions: 256,
        ..embedding(EmbeddingProvider::Bedrock {
            region: "eu-west-2".into(),
        })
    };
    for titan in ["amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v2"] {
        assert_eq!(embedders.admits(&named(titan)), Ok(()), "{titan}");
    }
    for model in [
        "amazon.titan-embed-text-v2:0-made-up-17",
        "amazon.titan-embed-text-v2:1",
        "amazon.titan-embed-text-v20",
        "Amazon.Titan-Embed-Text-v2:0",
    ] {
        let refused = embedders.admits(&named(model));
        assert!(
            matches!(refused, Err(EmbedError::NoSuchModel(_))),
            "{model}"
        );
        assert!(embedders.embedder(&named(model)).is_err());
    }
    assert_eq!(embedders.held(), 0);
}

#[test]
fn settings_never_print_a_key() {
    let settings = Settings {
        cache: "/models".into(),
        keys: [(
            "https://api.example.test".to_string(),
            "sk-secret-0123456789abcdef".to_string(),
        )]
        .into(),
        ..Settings::default()
    };
    let printed = format!("{settings:?} {:?}", Embedders::new(settings.clone()));
    assert!(printed.contains("https://api.example.test"), "{printed}");
    assert!(!printed.contains("sk-secret-0123456789abcdef"), "{printed}");
}

/// A model that answers each text as the text says: "refused" is refused
/// for good, "failing" fails in a way that may pass, "held" waits to be
/// let go first, and "never" never answers.
/// While `refusing`, every text is refused for good, after `slow`.
#[derive(Debug, Default)]
struct Told {
    calls: std::sync::atomic::AtomicUsize,
    in_flight: std::sync::atomic::AtomicUsize,
    most_in_flight: std::sync::atomic::AtomicUsize,
    refusing: std::sync::atomic::AtomicBool,
    slow: std::time::Duration,
    let_go: tokio::sync::Notify,
}

impl Told {
    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn refused() -> EmbedError {
        EmbedError::Unusable {
            model: "fake".into(),
            reason: "401 Unauthorized: no such key".into(),
            fault: vellev_core::ModelFault::Setup,
        }
    }
}

#[async_trait]
impl Provider for Told {
    async fn embed(&self, texts: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
        use std::sync::atomic::Ordering::SeqCst;
        self.calls.fetch_add(1, SeqCst);
        let now = self.in_flight.fetch_add(1, SeqCst) + 1;
        self.most_in_flight.fetch_max(now, SeqCst);
        let says = |word: &str| texts.iter().any(|text| text.contains(word));
        tokio::time::sleep(self.slow).await;
        if says("never") {
            std::future::pending::<()>().await;
        }
        if says("held") {
            self.let_go.notified().await;
        }
        self.in_flight.fetch_sub(1, SeqCst);
        if says("failing") {
            return Err(EmbedError::Failed {
                model: "fake".into(),
                reason: "503 Service Unavailable".into(),
            });
        }
        match self.refusing.load(SeqCst) || says("refused") {
            true => Err(Self::refused()),
            false => Ok(vec![vec![1.0, 2.0]; texts.len()]),
        }
    }
}

/// Wait, with the clock where it is, for `model` to have been called
/// `calls` times. Should it never be, this fails rather than waits for
/// ever.
async fn called(model: &Told, calls: usize) {
    for _ in 0..10_000 {
        if model.calls() >= calls {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("the model was called {} times, not {calls}", model.calls());
}

/// What `what` comes to, which it must within seconds: should a call in
/// it come to wait on another that is held, it fails rather than waits
/// for ever.
async fn soon<T>(what: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(std::time::Duration::from_secs(5), what)
        .await
        .expect("no call waits on another")
}

fn told(model: Told) -> (Arc<Told>, Arc<Embedder>) {
    let model = Arc::new(model);
    let embedder = Embedder::new(model.clone(), embedding(EmbeddingProvider::Local));
    (model, Arc::new(embedder))
}

/// Every text of `texts` asked of `embedder` at once.
async fn at_once(embedder: &Arc<Embedder>, texts: &[String]) -> Vec<Result<Option<Vec<f32>>>> {
    futures::future::join_all(texts.iter().map(|text| embedder.query(text))).await
}

fn numbered(word: &str, count: usize) -> Vec<String> {
    (0..count).map(|n| format!("{word} {n}")).collect()
}

#[tokio::test(start_paused = true)]
async fn a_fault_met_by_many_at_once_is_met_by_each_once_and_by_none_while_it_stands() {
    let (model, embedder) = told(Told {
        slow: std::time::Duration::from_millis(50),
        refusing: true.into(),
        ..Told::default()
    });
    let refused = |answers: Vec<Result<Option<Vec<f32>>>>| {
        assert_eq!(answers.len(), 8);
        assert!(answers.iter().all(|answer| answer == &Err(Told::refused())));
    };
    // Nothing is known of the model, and no call waits on another to find
    // out: each that is out before the first hears of the fault asks.
    refused(at_once(&embedder, &numbered("first", 8)).await);
    assert_eq!(model.calls(), 8);
    // While it stands, none does.
    tokio::time::advance(vellev_embed::FAULT_STANDS / 2).await;
    refused(at_once(&embedder, &numbered("meanwhile", 8)).await);
    refused(vec![embedder.query("one more").await; 8]);
    assert_eq!(model.calls(), 8);

    // Put right, the model is asked by every call as it comes, however
    // many at once, once the fault no longer stands.
    model
        .refusing
        .store(false, std::sync::atomic::Ordering::SeqCst);
    tokio::time::advance(vellev_embed::FAULT_STANDS / 2).await;
    let answers = at_once(&embedder, &numbered("put right", 8)).await;
    assert!(answers.iter().all(Result::is_ok), "{answers:?}");
    assert_eq!(model.calls(), 8 + 8);
    let most = model
        .most_in_flight
        .load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(most, 8, "all together");
}

#[tokio::test(start_paused = true)]
async fn a_call_never_waits_on_another_whatever_the_other_is_about() {
    // Documents are being embedded, and slowly: for as long as they are
    // held. Nothing is known of the model yet.
    let (model, embedder) = told(Told::default());
    let documents = {
        let embedder = embedder.clone();
        tokio::spawn(async move { embedder.documents(&["held".to_string()]).await })
    };
    called(&model, 1).await;
    // A query meanwhile is answered in its own time: at once.
    let started = tokio::time::Instant::now();
    let answered = tokio::time::timeout(std::time::Duration::from_secs(1), embedder.query("fox"));
    let answered = answered.await.expect("it does not wait for the documents");
    assert_eq!(answered, Ok(Some(vec![1.0, 2.0])));
    assert_eq!(started.elapsed(), std::time::Duration::ZERO);
    assert_eq!(model.calls(), 2);
    assert!(!documents.is_finished());
    // So are calls that fail in a way that may pass: all go, together.
    let answers = at_once(&embedder, &numbered("failing", 8)).await;
    assert!(answers.iter().all(Result::is_err));
    assert_eq!(model.calls(), 2 + 8);
    model.let_go.notify_one();
    assert!(documents.await.unwrap().is_ok());
}

#[tokio::test]
async fn a_fault_that_comes_to_light_after_the_model_has_answered_well_since_does_not_stand() {
    soon(async {
        let (model, embedder) = told(Told::default());
        assert!(embedder.query("first").await.is_ok());
        let late = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.query("refused, once held").await })
        };
        called(&model, 2).await;
        // The setup is put right, and the model answers, before the earlier
        // call hears that it was refused.
        assert!(embedder.query("asked since").await.is_ok());
        model.let_go.notify_one();
        assert_eq!(late.await.unwrap(), Err(Told::refused()));
        assert!(
            embedder.query("asked after").await.is_ok(),
            "and the model is still asked"
        );
        assert_eq!(model.calls(), 4);
    })
    .await;
}

#[tokio::test]
async fn a_fault_that_stands_is_lifted_by_the_model_answering_well_however_late() {
    soon(async {
        let (model, embedder) = told(Told::default());
        assert!(embedder.query("first").await.is_ok());
        let slow = {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.query("held").await })
        };
        called(&model, 2).await;
        assert_eq!(embedder.query("refused").await, Err(Told::refused()));
        assert_eq!(
            embedder.query("asked while it stands").await,
            Err(Told::refused())
        );
        assert_eq!(model.calls(), 3, "it stands: the model is not asked");
        // The call that was out all along comes back answered.
        model.let_go.notify_one();
        assert!(slow.await.unwrap().is_ok());
        assert!(embedder.query("asked after").await.is_ok());
        assert_eq!(model.calls(), 4);
    })
    .await;
}

fn seconds(seconds: u64) -> std::time::Duration {
    std::time::Duration::from_secs(seconds)
}

/// What a model that did not answer `within` such a time fails with.
fn late(within: &str) -> EmbedError {
    EmbedError::Failed {
        model: "fake".into(),
        reason: format!("no answer within {within}"),
    }
}

/// The same query asked by `askers` at once, each a second after the one
/// before: what each was told, and how long after the first asked.
async fn one_after_another(
    embedder: &Arc<Embedder>,
    text: &str,
    askers: u64,
) -> Vec<(Result<Option<Vec<f32>>>, std::time::Duration)> {
    let started = tokio::time::Instant::now();
    let asking = (0..askers).map(|nth| async move {
        tokio::time::sleep(seconds(nth)).await;
        (embedder.query(text).await, started.elapsed())
    });
    futures::future::join_all(asking).await
}

#[tokio::test(start_paused = true)]
async fn a_query_asked_by_many_at_once_fails_for_them_all_at_once_and_is_asked_again_after() {
    // A model that fails, in a way that may pass, half a minute after it
    // is asked: and is given two minutes to answer.
    let model = Arc::new(Told {
        slow: seconds(30),
        ..Told::default()
    });
    let embedder = Embedder::new(model.clone(), embedding(EmbeddingProvider::Local))
        .answering_within(seconds(120), seconds(120));
    let embedder = Arc::new(embedder);
    let failed = EmbedError::Failed {
        model: "fake".into(),
        reason: "503 Service Unavailable".into(),
    };

    // Ten ask the same, over ten seconds. The model is asked once, and
    // all are told how it failed when it does: none asks in turn after,
    // to be told half a minute later each.
    let told = one_after_another(&embedder, "failing fox", 10).await;
    for (answer, after) in told {
        assert_eq!((answer, after), (Err(failed.clone()), seconds(30)));
    }
    assert_eq!(model.calls(), 1);

    // The failure is not kept: whoever asks next asks the model.
    let started = tokio::time::Instant::now();
    assert_eq!(embedder.query("failing fox").await, Err(failed.clone()));
    assert_eq!((model.calls(), started.elapsed()), (2, seconds(30)));

    // Those who ask for different texts each ask for themselves, all at
    // once: none waits on another.
    let started = tokio::time::Instant::now();
    let answers = at_once(&embedder, &numbered("failing", 10)).await;
    assert!(answers.iter().all(|answer| answer == &Err(failed.clone())));
    assert_eq!((model.calls(), started.elapsed()), (12, seconds(30)));
    let most = model
        .most_in_flight
        .load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(most, 10);

    // A vector is told to all who wait likewise, and kept.
    let told = one_after_another(&embedder, "fox", 10).await;
    for (answer, after) in told {
        assert_eq!((answer, after), (Ok(Some(vec![1.0, 2.0])), seconds(30)));
    }
    let started = tokio::time::Instant::now();
    assert_eq!(embedder.query("fox").await, Ok(Some(vec![1.0, 2.0])));
    assert_eq!((model.calls(), started.elapsed()), (13, seconds(0)));
}

#[tokio::test]
async fn a_query_whose_first_asker_goes_away_is_worked_out_for_the_next() {
    soon(async {
        let (model, embedder) = told(Told::default());
        let ask = || {
            let embedder = embedder.clone();
            tokio::spawn(async move { embedder.query("held").await })
        };
        let first = ask();
        called(&model, 1).await;
        let second = ask();
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(model.calls(), 1, "the second waits on the first");
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        // No one is left asking the model: the second does.
        called(&model, 2).await;
        model.let_go.notify_one();
        assert_eq!(second.await.unwrap(), Ok(Some(vec![1.0, 2.0])));
        assert_eq!(embedder.query("held").await, Ok(Some(vec![1.0, 2.0])));
        assert_eq!(model.calls(), 2, "and it is kept");
    })
    .await;
}

#[tokio::test(start_paused = true)]
async fn a_model_that_does_not_answer_in_time_is_given_up_on_and_asked_again_by_the_next_call() {
    // Should the model be waited on for ever, this fails rather than
    // hangs.
    tokio::time::timeout(seconds(7 * 24 * 60 * 60), given_up_on())
        .await
        .expect("no model is waited on for ever");
}

async fn given_up_on() {
    let (model, embedder) = told(Told::default());
    assert_eq!(
        (vellev_embed::QUERY_WITHIN, vellev_embed::DOCUMENTS_WITHIN),
        (seconds(10), seconds(60))
    );

    // A query is given ten seconds. Giving up may pass: it is not kept,
    // and it is not a fault that stands, so the next call asks again.
    let started = tokio::time::Instant::now();
    assert_eq!(embedder.query("never").await, Err(late("10s")));
    assert_eq!(started.elapsed(), seconds(10));
    assert_eq!(late("10s").lasting(), None);
    assert_eq!(embedder.query("never").await, Err(late("10s")));
    assert_eq!(model.calls(), 2);
    assert_eq!(embedder.query("fox").await, Ok(Some(vec![1.0, 2.0])));
    assert_eq!(model.calls(), 3);

    // Those who wait on another's asking are told when that one's time
    // is up: none waits out a time of its own after it.
    let told = one_after_another(&embedder, "never again", 5).await;
    for (answer, after) in told {
        assert_eq!((answer, after), (Err(late("10s")), seconds(10)));
    }
    assert_eq!(model.calls(), 4);

    // A call for documents to a model that does not say what it is
    // doing is given a minute in all, however many its texts.
    let started = tokio::time::Instant::now();
    let documents = texts(&["never", "fox", "hen"]);
    assert_eq!(embedder.documents(&documents).await, Err(late("60s")));
    assert_eq!(started.elapsed(), seconds(60));
    assert_eq!(embedder.documents(&documents[1..]).await.unwrap().len(), 2);

    // Or as long as the embedder is told, which is never so long that
    // it cannot be counted from now.
    let told_how_long = |query, documents| {
        Embedder::new(model.clone(), embedding(EmbeddingProvider::Local))
            .answering_within(query, documents)
    };
    let embedder = told_how_long(seconds(2), seconds(5));
    let started = tokio::time::Instant::now();
    assert_eq!(embedder.query("never").await, Err(late("2s")));
    assert_eq!(started.elapsed(), seconds(2));
    let documents = texts(&["never", "", "fox", "hen"]);
    assert_eq!(embedder.documents(&documents).await, Err(late("5s")));
    assert_eq!(started.elapsed(), seconds(2 + 5));
    let for_ever = std::time::Duration::MAX;
    let embedder = told_how_long(for_ever, for_ever);
    assert_eq!(embedder.query("never").await, Err(late("86400s")));
    assert_eq!(embedder.documents(&documents).await, Err(late("86400s")));
}

/// A model that many wait on at once, which says what it is doing. A
/// call's texts are answered `answers` seconds after they are asked for.
/// Meanwhile the model loads for `loads` seconds, given `load_given` to,
/// and from then on makes a vector for someone, not this caller, every
/// `makes` seconds until `stops`: or never, if that is none.
#[derive(Debug, Default)]
struct Shared {
    answers: u64,
    loads: u64,
    load_given: u64,
    makes: u64,
    stops: u64,
    activity: vellev_embed::Activity,
}

#[async_trait]
impl Provider for Shared {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        let work = async {
            if self.loads > 0 {
                self.activity.loading(seconds(self.load_given));
                tokio::time::sleep(seconds(self.loads)).await;
                self.activity.loaded();
            }
            let mut worked = self.loads;
            while self.makes > 0 && worked + self.makes <= self.stops {
                tokio::time::sleep(seconds(self.makes)).await;
                self.activity.made();
                worked += self.makes;
            }
            std::future::pending::<()>().await;
        };
        tokio::select! {
            () = work => unreachable!("it never ends"),
            () = tokio::time::sleep(seconds(self.answers)) => {}
        }
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 0.0])
            .collect())
    }

    fn activity(&self) -> Option<&vellev_embed::Activity> {
        Some(&self.activity)
    }
}

/// A write waits its turn for as long as the model is at work, for
/// whomever, or loading: and is given up on a minute after the model
/// last made anything for anyone, and not for how long the write takes.
#[tokio::test(start_paused = true)]
async fn a_write_is_given_up_on_when_the_model_stops_and_not_for_how_long_it_waits() {
    let stopped = || EmbedError::Failed {
        model: "fake".into(),
        reason: "the model has made nothing for anyone in 60s".into(),
    };
    // What a write of three texts to `model` comes to, and how long
    // after it was asked.
    let written = |model: Shared| async move {
        let embedder = Embedder::new(Arc::new(model), embedding(EmbeddingProvider::Local));
        let started = tokio::time::Instant::now();
        let made = embedder.documents(&texts(&["fox", "hen", "owl"])).await;
        (made.map(|vectors| vectors.len()), started.elapsed())
    };
    let every_half_minute = |answers, stops| Shared {
        answers,
        makes: 30,
        stops,
        ..Shared::default()
    };

    // Kept waiting an hour behind others, with the model making vectors
    // for them all the while: it is made. Nothing of its own was made
    // in that hour, and nothing was given up on for others' load.
    assert_eq!(
        written(every_half_minute(3_600, 3_600)).await,
        (Ok(3), seconds(3_600))
    );
    // A model that makes nothing for anyone is given up on in a minute.
    let silent = Shared {
        answers: 3_600,
        ..Shared::default()
    };
    assert_eq!(written(silent).await, (Err(stopped()), seconds(60)));
    // One that stops after five minutes, a minute after that.
    assert_eq!(
        written(every_half_minute(3_600, 300)).await,
        (Err(stopped()), seconds(360))
    );
    // One that makes a vector too seldom, a minute after the last.
    let too_seldom = Shared {
        makes: 61,
        ..every_half_minute(3_600, 3_600)
    };
    assert_eq!(written(too_seldom).await, (Err(stopped()), seconds(60)));
    let seldom_enough = Shared {
        makes: 59,
        ..every_half_minute(3_600, 3_600)
    };
    assert_eq!(written(seldom_enough).await, (Ok(3), seconds(3_600)));

    // A model that loads for five minutes, given ten to, is at work all
    // that while, though it makes nothing: the write waits for it, and
    // is made.
    let loading = |loads, load_given, answers| Shared {
        loads,
        load_given,
        answers,
        ..Shared::default()
    };
    assert_eq!(written(loading(300, 600, 310)).await, (Ok(3), seconds(310)));
    // Loaded, it must make something within the minute like any other.
    assert_eq!(
        written(loading(300, 600, 3_600)).await,
        (Err(stopped()), seconds(360))
    );
    // And a load that goes on past what a load is given is excused for
    // that long, and no longer: a minute after, it is given up on.
    assert_eq!(
        written(loading(3_600, 600, 3_600)).await,
        (Err(stopped()), seconds(660))
    );
}

/// What a model does, a step at a time, while a call waits on it.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// A load that takes so long and ends with the model loaded.
    Loads(u64),
    /// A load that takes so long and fails.
    FailsToLoad(u64),
    /// A run that takes so long and fails, or panics: no vector.
    Fails(u64),
    /// A run that takes so long and makes a vector, for someone else.
    Makes(u64),
}

/// A model that does as its `steps` say, in turn, each load given
/// `load_given`, and then nothing; and answers the call `answers`
/// seconds after it is asked.
#[derive(Debug)]
struct Scripted {
    steps: Vec<Step>,
    load_given: u64,
    answers: u64,
    activity: vellev_embed::Activity,
}

#[async_trait]
impl Provider for Scripted {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        let work = async {
            for step in &self.steps {
                match *step {
                    Step::Loads(took) => {
                        self.activity.loading(seconds(self.load_given));
                        tokio::time::sleep(seconds(took)).await;
                        self.activity.loaded();
                    }
                    Step::FailsToLoad(took) => {
                        self.activity.loading(seconds(self.load_given));
                        tokio::time::sleep(seconds(took)).await;
                        self.activity.not_loading();
                    }
                    Step::Fails(took) => tokio::time::sleep(seconds(took)).await,
                    Step::Makes(took) => {
                        tokio::time::sleep(seconds(took)).await;
                        self.activity.made();
                    }
                }
            }
            std::future::pending::<()>().await;
        };
        tokio::select! {
            () = work => unreachable!("it never ends"),
            () = tokio::time::sleep(seconds(self.answers)) => {}
        }
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 0.0])
            .collect())
    }

    fn activity(&self) -> Option<&vellev_embed::Activity> {
        Some(&self.activity)
    }
}

/// Only a vector made is the model at work. A load is not, but the time
/// a load takes that ends with the model loaded is not held against a
/// call waiting: that of one that fails, or of a run that fails, is. A
/// call is given up on at the instant each row says, with a stall of a
/// minute and loads given ten.
#[tokio::test(start_paused = true)]
async fn only_a_vector_is_the_model_at_work_and_a_load_only_if_it_loads() {
    use Step::*;
    let stopped = || EmbedError::Failed {
        model: "fake".into(),
        reason: "the model has made nothing for anyone in 60s".into(),
    };
    // What a write comes to, and when, of a model that does as `steps`
    // say and would answer it after `answers`.
    let written = |steps: Vec<Step>, answers: u64| async move {
        let model = Scripted {
            steps,
            load_given: 600,
            answers,
            activity: vellev_embed::Activity::default(),
        };
        let embedder = Embedder::new(Arc::new(model), embedding(EmbeddingProvider::Local));
        let started = tokio::time::Instant::now();
        let made = embedder.documents(&texts(&["fox"])).await;
        (
            made.map(|vectors| vectors.len()),
            started.elapsed().as_secs(),
        )
    };
    // What each row is called, what the model does, when it would
    // answer, and what the call comes to and when.
    type Row = (&'static str, Vec<Step>, u64, Result<usize>, u64);
    let rows: [Row; 8] = [
        // A load that fails long after the stall: given up on when next
        // looked at after it fails, which is within a stall.
        (
            "a load failing after 290s",
            vec![FailsToLoad(290)],
            3_600,
            Err(stopped()),
            300,
        ),
        // A slow load that succeeds, and then a vector: the call is
        // answered, never given up on.
        (
            "a load of 295s, then a vector",
            vec![Loads(295), Makes(15)],
            310,
            Ok(1),
            310,
        ),
        // A load that goes on past the ten minutes it is given: excused
        // for those, and then a stall.
        (
            "a load that never ends",
            vec![Loads(100_000)],
            100_000,
            Err(stopped()),
            660,
        ),
        // Two failed loads, then one that loads, then a vector: fifty
        // seconds held against it, and no more.
        (
            "two failed loads, then a load and a vector",
            vec![FailsToLoad(25), FailsToLoad(25), Loads(50), Makes(5)],
            105,
            Ok(1),
            105,
        ),
        // Three failed loads: seventy-five seconds held against it, found
        // when next looked at.
        (
            "three failed loads",
            vec![FailsToLoad(25), FailsToLoad(25), FailsToLoad(25), Loads(50)],
            3_600,
            Err(stopped()),
            80,
        ),
        // A run that panics, a reload, and so on: the runs are held
        // against it, the reloads are not.
        (
            "panic, reload, panic, reload, panic",
            vec![
                Fails(20),
                Loads(30),
                Fails(20),
                Loads(30),
                Fails(20),
                Loads(30),
            ],
            3_600,
            Err(stopped()),
            120,
        ),
        // Failing loads one after another, each shorter than the stall:
        // given up on when next looked at after what failed comes to a
        // stall. (Of 7 seconds, so that no load ends when it is looked.)
        (
            "failing loads of 7s, one after another",
            vec![FailsToLoad(7); 20],
            3_600,
            Err(stopped()),
            64,
        ),
        // Vectors for others every half minute: answered after an hour.
        (
            "vectors every 30s",
            vec![Makes(30); 200],
            3_600,
            Ok(1),
            3_600,
        ),
    ];
    for (name, steps, answers, outcome, at) in rows {
        assert_eq!(written(steps, answers).await, (outcome, at), "{name}");
    }
}
