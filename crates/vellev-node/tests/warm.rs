//! A node keeps the tenants it lately served warm across a restart: it
//! remembers them, and opens copies of them as it starts, before anyone
//! reads them, so that the first read after a restart opens nothing
//! (burst plan F7).

use std::sync::Arc;
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Freshness, Node, Settings};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator};
use vellev_storage::Mutation;

fn tenant(name: &str) -> TenantId {
    TenantId::new(name).unwrap()
}

fn notes() -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: CollectionName::new("notes").unwrap(),
        properties: vec![Property::new("title", DataType::Text).searchable()],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "stand-in".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

fn note(n: u128) -> Mutation {
    Mutation::Upsert {
        collection: CollectionName::new("notes").unwrap(),
        id: ObjectId(uuid::Uuid::from_u128(n)),
        properties: serde_json::json!({ "title": "fox" })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![1.0, n as f32]),
        only_if_version: None,
    }
}

struct Shared {
    coordinator: Arc<dyn Coordinator>,
    blobs: Arc<dyn Blobs>,
    dir: tempfile::TempDir,
}

impl Shared {
    fn new() -> Self {
        Self {
            coordinator: Arc::new(MemoryCoordinator::default()),
            blobs: Arc::new(ObjectBlobs::new(Arc::new(
                object_store::memory::InMemory::new(),
            ))),
            dir: tempfile::tempdir().unwrap(),
        }
    }

    fn node(&self, warm_most: usize) -> Node {
        Node::new(
            self.coordinator.clone(),
            self.blobs.clone(),
            Arc::new(Embedders::new(EmbedSettings::default())),
            Settings {
                warm: Some(self.dir.path().join("warm-tenants.json")),
                warm_most,
                purge_after: Duration::ZERO,
                ..Settings::new(NodeId::generate("node"))
            },
        )
    }
}

async fn soon(what: impl Fn() -> bool) -> bool {
    for _ in 0..200 {
        if what() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

/// Tenants `names`, each with a note, written through `node`.
async fn written(node: &Node, names: &[&str]) {
    for name in names {
        node.create_tenant(&tenant(name)).await.unwrap();
        let held = node.tenant(&tenant(name)).await.unwrap();
        held.write(vec![notes(), note(1)]).await.unwrap();
    }
}

#[tokio::test]
async fn a_node_started_again_opens_what_it_lately_read_before_anyone_reads_it() {
    let shared = Shared::new();
    let first = shared.node(16);
    written(&first, &["acme", "beta", "gamma"]).await;
    for name in ["acme", "gamma"] {
        first
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    first.remember_warm();
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(
        soon(|| again.copies() == 3).await,
        "copies of the tenants it served opened as it starts: {}",
        again.copies()
    );
    let opens = again.opens();
    for name in ["acme", "beta", "gamma"] {
        let reader = again
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
        assert!(!reader.worked(), "{name}: the first read opened nothing");
    }
    assert_eq!(again.opens(), opens, "no tenant opened by a read");
}

#[tokio::test]
async fn a_node_keeps_no_more_tenants_warm_than_it_is_set_to() {
    let shared = Shared::new();
    let first = shared.node(2);
    written(&first, &["acme", "beta", "gamma", "delta"]).await;
    // Read last, so most lately: these two are kept.
    for name in ["acme", "beta", "gamma", "delta", "gamma", "delta"] {
        first
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    first.remember_warm();
    let listed: Vec<String> = serde_json::from_slice(
        &std::fs::read(shared.dir.path().join("warm-tenants.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        listed,
        ["delta", "gamma"],
        "the two read last, the last first"
    );
    first.shutdown().await;
    let again = shared.node(2);
    again.warming();
    assert!(soon(|| again.copies() == 2).await, "{}", again.copies());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(again.copies(), 2, "no more than two");
    let opens = again.opens();
    for name in ["gamma", "delta"] {
        again
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    assert_eq!(again.opens(), opens, "the two read last are the two kept");
}

#[tokio::test]
async fn a_tenant_gone_since_or_a_list_unreadable_is_passed_over() {
    let shared = Shared::new();
    let first = shared.node(16);
    written(&first, &["acme", "beta"]).await;
    for name in ["acme", "beta"] {
        first
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    first.remember_warm();
    first.drop_tenant(&tenant("beta")).await.unwrap();
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
    // A list that cannot be read is no list, and no failure.
    std::fs::write(shared.dir.path().join("warm-tenants.json"), b"not a list").unwrap();
    let third = shared.node(16);
    third.warming();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(third.copies(), 0);
    third
        .reader(&tenant("acme"), Freshness::Eventual)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_node_stopping_remembers_what_it_served() {
    let shared = Shared::new();
    let first = shared.node(16);
    written(&first, &["acme"]).await;
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
}

#[tokio::test]
async fn a_node_warms_no_tenant_it_does_not_serve() {
    let shared = Shared::new();
    let first = shared.node(16);
    written(&first, &["acme", "beta"]).await;
    for name in ["acme", "beta"] {
        first
            .reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
    first.shutdown().await;
    let again = Node::new(
        shared.coordinator.clone(),
        shared.blobs.clone(),
        Arc::new(Embedders::new(EmbedSettings::default())),
        Settings {
            warm: Some(shared.dir.path().join("warm-tenants.json")),
            warm_most: 16,
            served: vellev_node::Served::Only([tenant("acme")].into_iter().collect()),
            ..Settings::new(NodeId::generate("node"))
        },
    );
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(again.copies(), 1, "only the tenant it serves");
}
