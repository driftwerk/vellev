//! What an endpoint says in refusing is passed on without a key in it,
//! and without the start of one: however the endpoint frames what it
//! says, and wherever what it says is cut off. Over real sockets, since
//! whether a body is known to be whole is a matter of how it was framed.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use vellev_core::HttpApi;
use vellev_embed::{Http, Provider, Purpose};

/// Where looking for keys ends, in bytes of what an endpoint says.
const LOOKED_THROUGH: usize = 4 * 1024;

/// How an endpoint frames what it says, and whether it says all of it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Framing {
    /// As long as it said it would be.
    Length,
    /// Shorter than it said it would be: it hung up.
    LengthCutShort,
    /// In chunks, with the last.
    Chunked,
    /// Hung up partway through a chunk.
    ChunkCutShort,
    /// Hung up between chunks, with no last one.
    ChunkedWithNoLast,
    /// Until it hangs up, HTTP/1.1.
    UntilClosed,
    /// Until it hangs up, HTTP/1.0.
    UntilClosedOld,
    /// A length beside a coding other than chunked, which HTTP/1.1 says
    /// is read until the endpoint hangs up, the length being of no
    /// account.
    LengthBesideACoding,
    /// Chunked, and then another coding: which is read until the
    /// endpoint hangs up, chunked not being the last.
    ChunkedThenACoding,
    /// A coding whose name has `chunked` in it and is not it.
    NotQuiteChunked,
    /// In chunks, with the last, the coding named in capitals.
    ChunkedInCapitals,
}

impl Framing {
    const ALL: [Self; 11] = [
        Self::Length,
        Self::LengthCutShort,
        Self::Chunked,
        Self::ChunkCutShort,
        Self::ChunkedWithNoLast,
        Self::UntilClosed,
        Self::UntilClosedOld,
        Self::LengthBesideACoding,
        Self::ChunkedThenACoding,
        Self::NotQuiteChunked,
        Self::ChunkedInCapitals,
    ];

    /// Whether the endpoint said how much it would send and sent it: so
    /// that what it said is known to be all of it.
    fn known_whole(self) -> bool {
        matches!(self, Self::Length | Self::Chunked | Self::ChunkedInCapitals)
    }

    /// A 401 with `body`, on the wire.
    fn framed(self, body: &[u8]) -> Vec<u8> {
        let status = match self {
            Self::UntilClosedOld => "HTTP/1.0 401 Unauthorized\r\n",
            _ => "HTTP/1.1 401 Unauthorized\r\n",
        };
        let size = body.len();
        let head = match self {
            Self::Length => format!("content-length: {size}\r\n\r\n"),
            Self::LengthCutShort => format!("content-length: {}\r\n\r\n", size + 50),
            Self::Chunked | Self::ChunkedWithNoLast => {
                format!("transfer-encoding: chunked\r\n\r\n{size:x}\r\n")
            }
            Self::ChunkedInCapitals => {
                format!("transfer-encoding: CHUNKED\r\n\r\n{size:x}\r\n")
            }
            Self::ChunkCutShort => {
                format!("transfer-encoding: chunked\r\n\r\n{:x}\r\n", size + 50)
            }
            Self::UntilClosed => "connection: close\r\n\r\n".to_string(),
            Self::UntilClosedOld => "\r\n".to_string(),
            Self::LengthBesideACoding => {
                "transfer-encoding: gzip\r\ncontent-length: 500\r\n\r\n".to_string()
            }
            Self::ChunkedThenACoding => "transfer-encoding: chunked, gzip\r\n\r\n".to_string(),
            Self::NotQuiteChunked => "transfer-encoding: xchunked\r\n\r\n".to_string(),
        };
        let tail: &[u8] = match self {
            Self::Chunked | Self::ChunkedInCapitals => b"\r\n0\r\n\r\n",
            Self::ChunkedWithNoLast => b"\r\n",
            _ => b"",
        };
        [status.as_bytes(), head.as_bytes(), body, tail].concat()
    }
}

/// An endpoint that answers each caller with the next of what it is
/// given to say, byte for byte, and hangs up.
async fn endpoint() -> (String, Arc<Mutex<VecDeque<Vec<u8>>>>) {
    let says: Arc<Mutex<VecDeque<Vec<u8>>>> = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let next = says.clone();
    tokio::spawn(async move {
        loop {
            let (mut caller, _) = listener.accept().await.unwrap();
            let says = next.lock().unwrap().pop_front();
            tokio::spawn(async move {
                // The request, to the end of its body.
                let mut asked = Vec::new();
                let mut more = [0; 4096];
                while !asked.ends_with(b"}") {
                    match caller.read(&mut more).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => asked.extend_from_slice(&more[..read]),
                    }
                }
                let _ = caller.write_all(&says.expect("something to say")).await;
                let _ = caller.shutdown().await;
            });
        }
    });
    (url, says)
}

/// A key as an endpoint might quote it back: what comes before it that
/// it is known by, the key, and whether it is the endpoint's own.
struct Key {
    before: &'static str,
    key: &'static str,
    own: bool,
}

const KEYS: [Key; 12] = [
    Key::own("its-own-key-of-no-shape-0123456789"),
    Key::own("cl\u{e9}-secr\u{e8}te-\u{e9}t\u{e9}-0123456789-abcdef"),
    Key::own("its_own_key,of;no:shape@at!all1"),
    Key::shaped("", "sk-AbCd1234EfGh5678IjKl9012MnOp"),
    Key::shaped("", "sk-proj-AbCd1234EfGh5678IjKl9012MnOp"),
    Key::shaped("", "sk-AbCdEf9\u{2026}XyZ4"),
    Key::shaped("", "hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"),
    Key::shaped("", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl"),
    Key::shaped("", "AKIAIOSFODNN7EXAMPLE"),
    Key::shaped("", "ASIAIOSFODNN7EXAMPLE"),
    Key::shaped("", "AIzaSyA-0123456789abcdefghijklmnopqrstu"),
    Key::shaped("Bearer ", "abcdefghij.klmnop_qrst7"),
];

impl Key {
    const fn own(key: &'static str) -> Self {
        Self {
            before: "",
            key,
            own: true,
        }
    }

    const fn shaped(before: &'static str, key: &'static str) -> Self {
        Self {
            before,
            key,
            own: false,
        }
    }

    /// The first four characters of the key: no more of it than less
    /// than these is ever to be passed on.
    fn start(&self) -> String {
        self.key.chars().take(4).collect()
    }
}

/// What is passed on of `body`, said in refusing with `framing` to a
/// node whose own key is `key`'s, if it is its own.
async fn passed_on(
    url: &str,
    says: &Mutex<VecDeque<Vec<u8>>>,
    key: &Key,
    framing: Framing,
    body: &[u8],
) -> String {
    says.lock().unwrap().push_back(framing.framed(body));
    let own = key.own.then(|| key.key.to_string());
    let provider = Http::new(url, HttpApi::Openai, "m", own).unwrap();
    let refused = provider.embed(vec!["fox".into()], Purpose::Query).await;
    refused.unwrap_err().to_string()
}

/// An error passed on never carries the start of a key: whatever the
/// key's shape, however the endpoint frames what it says, and at
/// whichever byte of the key what it says is cut off, or looking for
/// keys ends. Less than four characters of a key may be left, which are
/// no more a key than any word; the status is always there.
#[tokio::test(flavor = "multi_thread")]
async fn no_error_passed_on_carries_the_start_of_a_key_however_it_was_framed_or_cut() {
    let (url, says) = endpoint().await;
    let mut cells = 0;
    let mut carried = Vec::new();
    for key in &KEYS {
        let said = format!("bad key: {}", key.before);
        let check = |passed: &str, framing: Framing, what: String| {
            assert!(passed.contains("401"), "{framing:?} {what}: {passed}");
            passed
                .contains(&key.start())
                .then(|| format!("{framing:?}, {}, {what}: {passed}", key.key))
        };
        for framing in Framing::ALL {
            // Cut off at every byte of the key: which only an endpoint
            // that did not say all it said it would can be.
            for cut in 1..key.key.len() {
                if framing.known_whole() {
                    break;
                }
                let body = [said.as_bytes(), &key.key.as_bytes()[..cut]].concat();
                let passed = passed_on(&url, &says, key, framing, &body).await;
                carried.extend(check(&passed, framing, format!("cut off at byte {cut}")));
                cells += 1;
            }
            // The whole key, with more after it, and looking for keys
            // ending at every byte of it and either side.
            for seen in 0..=key.key.len() + 1 {
                let before = LOOKED_THROUGH + 1 - seen - said.len();
                let body = format!("{}{said}{} and more", "x ".repeat(before / 2), key.key);
                let passed = passed_on(&url, &says, key, framing, body.as_bytes()).await;
                carried.extend(check(&passed, framing, format!("{seen} bytes of it seen")));
                cells += 1;
            }
            // The whole key and no more.
            let body = format!("{said}{}", key.key);
            let passed = passed_on(&url, &says, key, framing, body.as_bytes()).await;
            carried.extend(check(&passed, framing, "all of it".into()));
            cells += 1;
        }
    }
    println!(
        "{cells} cells, {} carried the start of a key",
        carried.len()
    );
    assert!(
        carried.is_empty(),
        "{} of {cells}:\n{}",
        carried.len(),
        carried.join("\n")
    );
}

/// The node's own key is taken out of the middle of what an endpoint
/// says, eight characters of it or more, however the endpoint framed
/// it, and whatever escaped it; and the status before it is never
/// touched, though the key have the status's words in it.
#[tokio::test(flavor = "multi_thread")]
async fn the_own_key_is_taken_out_of_the_middle_whatever_the_framing() {
    let (url, says) = endpoint().await;
    let own = Key::own("xxUnauthorizedxx-0123456789-abcdef");
    let bodies = [
        r#"{"error":"Incorrect API key provided: xxUnauthor"}"#,
        r#"{"error":"key ending 456789-abcdef is invalid"}"#,
        r#"{"error":"bad key xxunauthorizedXX-0123"}"#,
        r#"{"error":"bad key xxUnauthorized"}"#,
        "bad key xxUnauth%6Frized and more",
    ];
    for framing in Framing::ALL {
        for body in bodies {
            let passed = passed_on(&url, &says, &own, framing, body.as_bytes()).await;
            assert!(
                passed.contains("401 Unauthorized: "),
                "{framing:?}, {body}: {passed}"
            );
            let after = passed.split_once("401 Unauthorized: ").unwrap().1;
            let key: Vec<char> = own.key.chars().collect();
            for piece in key.windows(8) {
                let piece: String = piece.iter().collect::<String>().to_ascii_lowercase();
                assert!(
                    !after.to_ascii_lowercase().contains(&piece),
                    "{framing:?}, {body}: {passed}"
                );
            }
        }
    }
}

/// What is no key is still passed on, cut off or not: an endpoint's
/// own words stay readable.
#[tokio::test(flavor = "multi_thread")]
async fn what_an_endpoint_says_that_is_no_key_stays_readable_when_cut_off() {
    let (url, says) = endpoint().await;
    let none = Key::shaped("", "no key at all");
    let limit = r#"{"error":{"message":"Rate limit reached for requests. Please try again in "#;
    for framing in Framing::ALL {
        let passed = passed_on(&url, &says, &none, framing, limit.as_bytes()).await;
        assert!(
            passed.contains("Rate limit reached for requests. Please try again in"),
            "{framing:?}: {passed}"
        );
        // What ends in a word is passed on with it when the endpoint
        // said how much it would send and sent it, by its length or in
        // chunks with a last one. Otherwise the word may be a key begun,
        // and is left out: that much is lost for not knowing.
        let ends_in_a_word = "Too many requests, try again later";
        let passed = passed_on(&url, &says, &none, framing, ends_in_a_word.as_bytes()).await;
        match framing.known_whole() {
            true => assert!(passed.ends_with("try again later"), "{framing:?}: {passed}"),
            false => assert!(passed.ends_with("try again "), "{framing:?}: {passed}"),
        }
    }
}

/// A refusal that names a fault is read for it when it came to an end
/// and reads as all of something, though the endpoint never said how
/// much it would send: and not when it was cut off.
#[tokio::test(flavor = "multi_thread")]
async fn a_refusal_that_ended_is_read_for_its_fault_and_one_cut_off_is_not() {
    let (url, says) = endpoint().await;
    let refusal = br#"{"error":{"message":"no such key","code":"invalid_api_key"}}"#;
    let provider = Http::new(&url, HttpApi::Openai, "m", None).unwrap();
    let setup = Some(vellev_core::ModelFault::Setup);
    for framing in Framing::ALL {
        let ended = !matches!(
            framing,
            Framing::LengthCutShort | Framing::ChunkCutShort | Framing::ChunkedWithNoLast
        );
        // All of it, and then short of its end, which reads as nothing
        // however it was framed.
        for (body, names_a_fault) in [
            (&refusal[..], ended),
            (&refusal[..refusal.len() - 2], false),
        ] {
            says.lock().unwrap().push_back(framing.framed(body));
            let refused = provider.embed(vec!["fox".into()], Purpose::Query).await;
            let read = refused.unwrap_err().lasting();
            let expected = if names_a_fault { setup } else { None };
            assert_eq!(read, expected, "{framing:?}, {} bytes", body.len());
        }
    }
}
