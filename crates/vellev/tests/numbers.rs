//! Numbers through the API are the numbers that were sent.

mod common;

use common::*;
use reqwest::Method;
use serde_json::json;

/// Doubles of every size, most of them with as many digits as a double
/// has: those a parser that rounds twice gets wrong about three times in
/// ten.
fn doubles() -> Vec<f64> {
    let mut state = 7_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let any = std::iter::repeat_with(move || f64::from_bits(next())).filter(|n| n.is_finite());
    let plain = [0.1 + 0.2, 1.0 / 3.0, 5e-324, f64::MAX, -0.0, 4.35, 1e23];
    plain.into_iter().chain(any.take(2_000)).collect()
}

#[tokio::test]
async fn a_number_sent_is_stored_and_returned_bit_for_bit() {
    let served = Served::new().await;
    let mut schema = notes(supplied());
    let score = json!({ "name": "score", "data_type": "number" });
    schema["properties"].as_array_mut().unwrap().push(score);
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    served.ok(Method::PUT, NOTES, Some(schema)).await;
    for (n, sent) in doubles().into_iter().enumerate() {
        let stored = json!({ "properties": { "score": sent }, "vector": [1.0, 0.0] });
        let (status, body) = served.put(&object(n as u64), stored).await;
        assert!(status.is_success(), "{sent:e}: {body}");
        let (_, body) = served.get(&object(n as u64)).await;
        let returned = body["properties"]["score"].as_f64().unwrap();
        assert_eq!(
            returned.to_bits(),
            sent.to_bits(),
            "{sent:e} came back {returned:e}"
        );
    }
}

#[tokio::test]
async fn a_vector_sent_is_stored_and_returned_bit_for_bit() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let mut state = 11_u64;
    let mut part = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64 - 0.5) as f32
    };
    for n in 0..500 {
        let sent = [part(), part()];
        let stored = json!({ "properties": {}, "vector": sent });
        let (status, body) = served.put(&object(n), stored).await;
        assert!(status.is_success(), "{body}");
        let (_, body) = served.get(&object(n)).await;
        let returned: Vec<f32> = serde_json::from_value(body["vector"].clone()).unwrap();
        let bits = |parts: &[f32]| parts.iter().map(|part| part.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&returned), bits(&sent), "{n}");
    }
}
