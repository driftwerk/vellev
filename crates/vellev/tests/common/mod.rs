#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use object_store::memory::InMemory;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use vellev::identity::IdentityKeys;
use vellev::server::{App, Limits};
use vellev_core::NodeId;
use vellev_embed::Embedders;
use vellev_node::{Node, Settings};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::MemoryCoordinator;

pub const KEY: &str = "test-key";
pub const MAX_LIMIT: usize = 50;
pub const MAX_BODY: usize = 64 * 1024;

pub fn memory() -> Arc<dyn Blobs> {
    Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))
}

/// What the nodes of one cluster share.
#[derive(Clone)]
pub struct Cluster {
    coordinator: Arc<MemoryCoordinator>,
    blobs: Arc<dyn Blobs>,
}

impl Cluster {
    pub fn new() -> Self {
        Self {
            coordinator: Arc::new(MemoryCoordinator::default()),
            blobs: memory(),
        }
    }

    /// Start a node and answer with a caller that holds its key.
    pub async fn serve(&self) -> Served {
        self.serve_embedding_through(&[]).await
    }

    /// As [`Self::serve`], with collections allowed to name these
    /// embedding endpoints.
    pub async fn serve_embedding_through(&self, urls: &[String]) -> Served {
        let served = self.unstarted(urls, None).await;
        served.app.start(served.node.clone());
        served
    }

    /// A node that tells other nodes it can be reached at `address`
    /// rather than where it listens.
    pub async fn serve_advertising(&self, address: &str) -> Served {
        let served = self.unstarted(&[], Some(address)).await;
        served.app.start(served.node.clone());
        served
    }

    /// A node that backs tenants up to `store`.
    pub async fn serve_backing_up_to(&self, store: Arc<dyn Blobs>) -> Served {
        let served = self
            .unstarted_with(&[], None, &[], |settings| Settings {
                backups: Some(store),
                ..settings
            })
            .await;
        served.app.start(served.node.clone());
        served
    }

    /// A node limited to `queries` a second for each tenant.
    pub async fn serve_limited(&self, queries: f64) -> Served {
        let served = self
            .unstarted_with(&[], None, &[], |settings| Settings {
                queries_per_second: Some(queries),
                ..settings
            })
            .await;
        served.app.start(served.node.clone());
        served
    }

    /// A node that trusts identity tokens signed by these keys, given as
    /// `KID=X`.
    pub async fn serve_trusting(&self, keys: &[String]) -> Served {
        let served = self
            .unstarted_with(&[], None, keys, |settings| settings)
            .await;
        served.app.start(served.node.clone());
        served
    }

    /// A node that is listening but has not yet reached what it serves
    /// from.
    pub async fn unstarted(&self, urls: &[String], advertised: Option<&str>) -> Served {
        self.unstarted_with(urls, advertised, &[], |settings| settings)
            .await
    }

    async fn unstarted_with(
        &self,
        urls: &[String],
        advertised: Option<&str>,
        trusted: &[String],
        adjusted: impl FnOnce(Settings) -> Settings,
    ) -> Served {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let node = Arc::new(Node::new(
            self.coordinator.clone(),
            self.blobs.clone(),
            Arc::new(Embedders::new(Default::default())),
            adjusted(Settings::new(NodeId::reachable_at(
                advertised.unwrap_or(&url),
                "node",
            ))),
        ));
        let limits = Limits {
            results: MAX_LIMIT,
            body: MAX_BODY,
            embedding_urls: urls.into(),
        };
        let app = App::new([KEY.to_string(), "second-key".to_string()].into(), limits)
            .trusting(IdentityKeys::parse(trusted).unwrap());
        let (stop, stopped) = oneshot::channel::<()>();
        let served = tokio::spawn(vellev::server::serve(listener, app.clone(), async {
            stopped.await.ok();
        }));
        Served {
            url,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            node,
            app,
            stop: Some(stop),
            served: Some(served),
        }
    }
}

pub struct Served {
    pub url: String,
    pub http: reqwest::Client,
    pub node: Arc<Node>,
    pub app: App,
    stop: Option<oneshot::Sender<()>>,
    served: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl Served {
    pub async fn new() -> Self {
        Cluster::new().serve().await
    }

    /// Send a request with the key, and answer with its status and body.
    pub async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        self.call_as(Some(KEY), method, path, body).await
    }

    pub async fn call_as(
        &self,
        key: Option<&str>,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        self.call_headed(key, method, path, body, &[]).await
    }

    /// As [`Self::call_as`], with these headers too.
    pub async fn call_headed(
        &self,
        key: Option<&str>,
        method: Method,
        path: &str,
        body: Option<Value>,
        headers: &[(&str, &str)],
    ) -> (StatusCode, Value) {
        let request = self.http.request(method, format!("{}{path}", self.url));
        let request = headers.iter().fold(request, |request, (name, value)| {
            request.header(*name, *value)
        });
        let request = key
            .into_iter()
            .fold(request, |request, key| request.bearer_auth(key));
        let request = body
            .into_iter()
            .fold(request, |request, body| request.json(&body));
        let response = request.send().await.unwrap();
        let status = response.status();
        let text = response.text().await.unwrap();
        let body = match text.is_empty() {
            true => Value::Null,
            false => serde_json::from_str(&text).unwrap_or_else(|_| panic!("not JSON: {text}")),
        };
        (status, body)
    }

    pub async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.call(Method::GET, path, None).await
    }

    pub async fn put(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::PUT, path, Some(body)).await
    }

    pub async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::POST, path, Some(body)).await
    }

    pub async fn patch(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.call(Method::PATCH, path, Some(body)).await
    }

    pub async fn delete(&self, path: &str) -> (StatusCode, Value) {
        self.call(Method::DELETE, path, None).await
    }

    /// A request that must succeed, answering with its body.
    pub async fn ok(&self, method: Method, path: &str, body: Option<Value>) -> Value {
        let (status, answer) = self.call(method.clone(), path, body).await;
        assert!(status.is_success(), "{method} {path}: {status} {answer}");
        answer
    }

    /// Create tenant `acme` with a `notes` collection of two dimensions.
    pub async fn with_notes(&self, embedding: Value) {
        self.ok(Method::PUT, "/v1/tenants/acme", None).await;
        self.ok(Method::PUT, NOTES, Some(notes(embedding))).await;
    }

    pub async fn stop(mut self) {
        self.stop.take().unwrap().send(()).unwrap();
        self.served.take().unwrap().await.unwrap().unwrap();
    }
}

pub const NOTES: &str = "/v1/tenants/acme/collections/notes";

/// The collection's schema, embedding as given.
pub fn notes(embedding: Value) -> Value {
    json!({
        "name": "notes",
        "properties": [
            { "name": "title", "data_type": "text", "searchable": true },
            { "name": "n", "data_type": "int" },
            { "name": "tags", "data_type": "text[]" },
        ],
        "embedding": embedding,
        "distance": "cosine",
        "bm25": { "k1": 1.2, "b": 0.75 },
        "access_rule": null,
    })
}

/// Vectors supplied by the client.
pub fn supplied() -> Value {
    json!({ "provider": "none", "model": "client", "dimensions": 2 })
}

pub fn id(n: u64) -> String {
    format!("00000000-0000-0000-0000-{n:012}")
}

pub fn object(n: u64) -> String {
    format!("{NOTES}/objects/{}", id(n))
}

/// What a failure's body must be for `code`.
pub fn code(body: &Value) -> &str {
    body["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("not a failure: {body}"))
}
