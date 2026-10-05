use std::time::Duration;

use tokio::net::TcpListener;
use tokio::sync::oneshot;
use vellev_client::Client;

#[tokio::test]
async fn a_running_node_is_live_and_ready_until_shut_down() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(vellev::server::serve(listener, async {
        stopped.await.ok();
    }));

    let client = Client::new(&url, Duration::from_secs(2)).unwrap();
    client.live().await.unwrap();
    client.ready().await.unwrap();

    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
    assert!(client.live().await.is_err());
}
