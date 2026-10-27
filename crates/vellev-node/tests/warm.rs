//! A node keeps the tenants it lately served warm across a restart: it
//! remembers them, and opens copies of them as it starts, before anyone
//! reads them, so that the first read after a restart opens nothing
//! (burst plan F7). The tenants it writes for it opens as their writer
//! when they are asked for, never as copies beside them; and warming keeps
//! within the node's memory budget.

use std::sync::Arc;
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Freshness, Node, Served, Settings};
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

    fn list(&self) -> std::path::PathBuf {
        self.dir.path().join("warm-tenants.json")
    }

    /// A node that writes for the tenants, keeping no warm list.
    fn writer(&self) -> Node {
        self.node_on(self.blobs.clone(), |settings| settings.warm = None)
    }

    /// A node keeping a warm list of `warm_most` tenants.
    fn node(&self, warm_most: usize) -> Node {
        self.node_with(|settings| settings.warm_most = warm_most)
    }

    fn node_with(&self, set: impl FnOnce(&mut Settings)) -> Node {
        self.node_on(self.blobs.clone(), set)
    }

    fn node_on(&self, blobs: Arc<dyn Blobs>, set: impl FnOnce(&mut Settings)) -> Node {
        let mut settings = Settings {
            warm: Some(self.list()),
            warm_most: 16,
            purge_after: Duration::ZERO,
            ..Settings::new(NodeId::generate("node"))
        };
        set(&mut settings);
        Node::new(
            self.coordinator.clone(),
            blobs,
            Arc::new(Embedders::new(EmbedSettings::default())),
            settings,
        )
    }

    fn listed(&self) -> Vec<String> {
        let entries: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(self.list()).unwrap()).unwrap();
        entries
            .into_iter()
            .map(|entry| entry["tenant"].as_str().unwrap().to_string())
            .collect()
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

/// Tenants `names`, each with `notes` notes, written through `node`.
async fn written_with(node: &Node, names: &[&str], notes_each: u128) {
    for name in names {
        node.create_tenant(&tenant(name)).await.unwrap();
        let held = node.tenant(&tenant(name)).await.unwrap();
        let mut mutations = vec![notes()];
        mutations.extend((1..=notes_each).map(note));
        held.write(mutations).await.unwrap();
    }
}

/// Tenants `names`, each with a note, written through `node`.
async fn written(node: &Node, names: &[&str]) {
    written_with(node, names, 1).await
}

async fn read(node: &Node, names: &[&str]) {
    for name in names {
        node.reader(&tenant(name), Freshness::Eventual)
            .await
            .unwrap();
    }
}

/// Wait for `node` to have opened as many copies as it is going to.
async fn settled(node: &Node) {
    let mut seen = node.opens();
    loop {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let now = node.opens();
        if now == seen && !node.warming_now() {
            return;
        }
        seen = now;
    }
}

#[tokio::test]
async fn a_node_started_again_opens_what_it_lately_read_before_anyone_reads_it() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta", "gamma"]).await;
    let first = shared.node(16);
    read(&first, &["acme", "beta", "gamma"]).await;
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
    let writer = shared.writer();
    written(&writer, &["acme", "beta", "gamma", "delta"]).await;
    let first = shared.node(2);
    // Read last, so most lately: these two are kept.
    read(
        &first,
        &["acme", "beta", "gamma", "delta", "gamma", "delta"],
    )
    .await;
    first.remember_warm().await;
    assert_eq!(
        shared.listed(),
        ["delta", "gamma"],
        "the two read last, the last first"
    );
    first.shutdown().await;
    let again = shared.node(2);
    again.warming();
    assert!(soon(|| again.copies() == 2).await, "{}", again.copies());
    settled(&again).await;
    assert_eq!(again.copies(), 2, "no more than two");
    let opens = again.opens();
    read(&again, &["gamma", "delta"]).await;
    assert_eq!(again.opens(), opens, "the two read last are the two kept");
}

#[tokio::test]
async fn a_tenant_gone_since_or_a_list_unreadable_is_passed_over() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta"]).await;
    let first = shared.node(16);
    read(&first, &["acme", "beta"]).await;
    first.remember_warm().await;
    writer.drop_tenant(&tenant("beta")).await.unwrap();
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
    again.shutdown().await;
    // A list that cannot be read is no list, and no failure.
    std::fs::write(shared.list(), b"not a list").unwrap();
    let third = shared.node(16);
    third.warming();
    settled(&third).await;
    assert_eq!(third.copies(), 0);
    read(&third, &["acme"]).await;
}

#[tokio::test]
async fn a_node_stopping_remembers_what_it_served() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme"]).await;
    let first = shared.node(16);
    read(&first, &["acme"]).await;
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
}

#[tokio::test]
async fn a_node_warms_no_tenant_it_does_not_serve() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta"]).await;
    let first = shared.node(16);
    read(&first, &["acme", "beta"]).await;
    first.shutdown().await;
    let again = shared.node_with(|settings| {
        settings.served = Served::Only([tenant("acme")].into_iter().collect())
    });
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
    settled(&again).await;
    assert_eq!(again.copies(), 1, "only the tenant it serves");
}

#[tokio::test]
async fn a_tenant_this_node_wrote_for_is_not_warmed_as_a_copy() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme"]).await;
    let first = shared.node(16);
    // Written for here, and read here too: it is the writer that comes
    // back when it is asked for, not a copy beside it.
    written(&first, &["beta"]).await;
    read(&first, &["acme", "beta"]).await;
    first.shutdown().await;
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 1).await, "{}", again.copies());
    settled(&again).await;
    assert_eq!(again.copies(), 1, "acme alone, as a copy");
    assert_eq!(again.opens().writers, 0, "no writer opened by warming");
}

#[tokio::test]
async fn a_copy_warmed_is_not_counted_as_used() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta", "gamma"]).await;
    let first = shared.node(16);
    read(&first, &["beta", "acme"]).await;
    first.shutdown().await;
    assert_eq!(shared.listed(), ["acme", "beta"]);
    let again = shared.node(16);
    again.warming();
    assert!(soon(|| again.copies() == 2).await, "{}", again.copies());
    settled(&again).await;
    read(&again, &["gamma"]).await;
    again.remember_warm().await;
    // Read since the start, gamma is newest; the two warmed and not read
    // keep the order they were listed in, after it.
    assert_eq!(shared.listed(), ["gamma", "acme", "beta"]);
}

/// About how many bytes the copy of one of `names`, each written alike,
/// takes, as a node holding all of them counts.
async fn each_copy_takes(shared: &Shared, names: &[&str]) -> usize {
    let first = shared.node(16);
    read(&first, names).await;
    let each = first.resident() / names.len();
    first.shutdown().await;
    assert!(each > 0);
    each
}

#[tokio::test]
async fn warming_stops_short_of_the_memory_budget() {
    let shared = Shared::new();
    let writer = shared.writer();
    let names = ["acme", "beta", "gamma", "delta"];
    written_with(&writer, &names, 300).await;
    let each = each_copy_takes(&shared, &names).await;
    let budget = each * 2;
    let again = shared.node_with(|settings| settings.memory = Some(budget));
    again.warming();
    assert!(soon(|| again.copies() >= 1).await, "{}", again.copies());
    settled(&again).await;
    assert!(again.copies() < names.len(), "{}", again.copies());
    assert!(
        again.resident() <= budget * 3 / 4,
        "warmed {} bytes of a budget of {budget}",
        again.resident()
    );
}

#[tokio::test]
async fn warming_never_keeps_a_copy_past_the_memory_budget() {
    let shared = Shared::new();
    let writer = shared.writer();
    let names = ["acme", "beta"];
    written_with(&writer, &names, 300).await;
    let each = each_copy_takes(&shared, &names).await;
    // Smaller than any one copy: nothing is kept.
    let budget = each / 2;
    let again = shared.node_with(|settings| settings.memory = Some(budget));
    again.warming();
    settled(&again).await;
    assert_eq!(again.copies(), 0);
    assert!(again.resident() <= budget, "{}", again.resident());
}

#[tokio::test]
async fn a_list_too_large_to_be_one_is_not_read() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme"]).await;
    let mut entries = vec![serde_json::json!({ "tenant": "acme" })];
    entries.extend(std::iter::repeat_n(
        serde_json::json!({ "tenant": "x".repeat(63) }),
        100_000,
    ));
    std::fs::write(shared.list(), serde_json::to_vec(&entries).unwrap()).unwrap();
    let again = shared.node(16);
    again.warming();
    settled(&again).await;
    assert_eq!(again.copies(), 0, "a list that large was read");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_of_the_list_at_once_leave_it_whole() {
    let shared = Shared::new();
    let writer = shared.writer();
    let names: Vec<String> = (0..16).map(|n| format!("tenant_{n}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    written(&writer, &names).await;
    let first = shared.node(16);
    read(&first, &names).await;
    for _ in 0..20 {
        let writes: Vec<_> = (0..16)
            .map(|_| {
                let node = first.clone();
                tokio::spawn(async move { node.remember_warm().await })
            })
            .collect();
        for write in writes {
            write.await.unwrap();
        }
        assert_eq!(shared.listed().len(), 16);
    }
    let left: Vec<_> = std::fs::read_dir(shared.dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(left, ["warm-tenants.json"], "nothing half written left");
}

/// A store whose fetches wait while it is held.
#[derive(Debug)]
struct Held {
    inner: Arc<dyn Blobs>,
    open: tokio::sync::watch::Sender<bool>,
    waiting: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl Blobs for Held {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        let mut open = self.open.subscribe();
        self.waiting
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = open.wait_for(|open| *open).await;
        self.inner.get(key).await
    }
    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_node_stopping_stops_warming_and_remembers_what_it_had_yet_to_warm() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta"]).await;
    let first = shared.node(16);
    read(&first, &["beta", "acme"]).await;
    first.shutdown().await;
    let held = Arc::new(Held {
        inner: shared.blobs.clone(),
        open: tokio::sync::watch::channel(false).0,
        waiting: Default::default(),
    });
    let again = shared.node_on(held.clone(), |_| {});
    again.warming();
    assert!(
        soon(|| held.waiting.load(std::sync::atomic::Ordering::SeqCst) > 0).await,
        "warming fetched nothing"
    );
    tokio::time::timeout(Duration::from_secs(5), again.shutdown())
        .await
        .expect("shutdown waited on a warming that does not end");
    assert!(!again.warming_now(), "warming goes on past shutdown");
    assert_eq!(shared.listed(), ["acme", "beta"], "what it had yet to warm");
    held.open.send_replace(true);
}

#[tokio::test]
async fn a_node_stopped_before_it_has_read_its_list_keeps_the_list() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta"]).await;
    let first = shared.node(16);
    read(&first, &["beta", "acme"]).await;
    first.shutdown().await;
    assert_eq!(shared.listed(), ["acme", "beta"]);
    // Stopped at once: warming has not read the list yet, and the list it
    // has yet to read is not written over with what the node holds, none.
    let again = shared.node(16);
    again.warming();
    again.shutdown().await;
    assert_eq!(shared.listed(), ["acme", "beta"]);
}

#[tokio::test]
async fn the_list_a_node_writes_as_it_stops_is_the_last_written() {
    let shared = Shared::new();
    let writer = shared.writer();
    written(&writer, &["acme", "beta"]).await;
    let first = shared.node(16);
    first.warming();
    read(&first, &["beta", "acme"]).await;
    first.shutdown().await;
    assert_eq!(shared.listed(), ["acme", "beta"]);
    // A write of the list begun before the node stopped, or after it, is
    // not written over what it wrote as it stopped.
    first.remember_warm().await;
    assert_eq!(shared.listed(), ["acme", "beta"]);
}
