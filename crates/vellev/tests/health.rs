mod common;

use std::time::Duration;

use common::*;
use reqwest::StatusCode;
use vellev_client::Client;

#[tokio::test]
async fn a_running_node_is_live_and_ready_until_shut_down() {
    let served = Served::new().await;
    let client = Client::new(&served.url, Duration::from_secs(2)).unwrap();
    client.live().await.unwrap();
    client.ready().await.unwrap();
    let (status, body) = served
        .call_as(None, reqwest::Method::GET, "/health/ready", None)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");

    served.stop().await;
    assert!(client.live().await.is_err());
}
