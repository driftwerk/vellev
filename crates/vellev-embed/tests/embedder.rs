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
async fn collections_that_embed_alike_share_an_embedder() {
    let embedders = Embedders::new(Settings::default());
    assert!(embedders
        .embedder(&embedding(EmbeddingProvider::None))
        .unwrap()
        .is_none());

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
    let again = embedders
        .embedder(&http("http://one.test"))
        .unwrap()
        .unwrap();
    let elsewhere = embedders
        .embedder(&http("http://two.test"))
        .unwrap()
        .unwrap();
    let reworded = Embedding {
        query_prefix: None,
        ..http("http://one.test")
    };
    let reworded = embedders.embedder(&reworded).unwrap().unwrap();
    assert!(Arc::ptr_eq(&first, &again));
    assert!(!Arc::ptr_eq(&first, &elsewhere));
    assert!(!Arc::ptr_eq(&first, &reworded));
    assert_eq!(first.embedding(), &http("http://one.test"));
}

#[cfg(feature = "local")]
#[test]
fn a_model_no_build_runs_is_refused_when_asked_for_and_asking_builds_nothing() {
    let embedders = Embedders::new(Settings::default());
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

#[test]
fn settings_never_print_a_key() {
    let settings = Settings {
        cache: "/models".into(),
        keys: [(
            "https://api.example.test".to_string(),
            "sk-secret".to_string(),
        )]
        .into(),
    };
    let printed = format!("{settings:?} {:?}", Embedders::new(settings.clone()));
    assert!(printed.contains("https://api.example.test"), "{printed}");
    assert!(!printed.contains("sk-secret"), "{printed}");
}

/// A model whose first call is refused for good, but not until it is let
/// go, and whose every other call is answered at once.
#[derive(Debug, Default)]
struct RefusedLate {
    calls: std::sync::atomic::AtomicUsize,
    let_go: tokio::sync::Notify,
}

#[async_trait]
impl Provider for RefusedLate {
    async fn embed(&self, texts: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            self.let_go.notified().await;
            return Err(EmbedError::Unusable {
                model: "fake".into(),
                reason: "401 Unauthorized: no such key".into(),
                fault: vellev_core::ModelFault::Setup,
            });
        }
        Ok(vec![vec![1.0, 2.0]; texts.len()])
    }
}

#[tokio::test]
async fn a_fault_that_comes_to_light_after_the_model_has_answered_well_since_does_not_stand() {
    let model = Arc::new(RefusedLate::default());
    let embedder = Arc::new(Embedder::new(
        model.clone(),
        embedding(EmbeddingProvider::Local),
    ));
    let calls = || model.calls.load(std::sync::atomic::Ordering::SeqCst);
    let first = {
        let embedder = embedder.clone();
        tokio::spawn(async move { embedder.query("asked while the key was wrong").await })
    };
    while calls() == 0 {
        tokio::task::yield_now().await;
    }
    // The setup is put right, and the model answers, before the first
    // call hears that it was refused.
    assert_eq!(
        embedder.query("asked since").await,
        Ok(Some(vec![1.0, 2.0]))
    );
    model.let_go.notify_one();
    assert!(
        first.await.unwrap().is_err(),
        "the call itself still failed"
    );
    assert_eq!(
        embedder.query("asked after").await,
        Ok(Some(vec![1.0, 2.0])),
        "and the model is still asked"
    );
    assert_eq!(calls(), 3);
}
