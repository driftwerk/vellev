//! A property can be embedded without its text being stored (track 2, V3).
//!
//! The rules, each a test below:
//! - The text of an unstored property goes into the vector its object is
//!   given, and nowhere else: not the row read back, not the log, not a
//!   segment, so not a backup either. What reads the partition later, a
//!   follower or another writer, replays the vector the log carries and
//!   embeds nothing.
//! - A vector is taken over only where the text it was made from is known
//!   to be the same. A write of a whole object cannot know that of a row
//!   whose text was not kept, so it is always embedded afresh.
//! - A change to a part of an object, a patch or an append, that changes
//!   what is embedded is refused unless it brings its vector, since the
//!   text it would be embedded from is not all there. One that changes
//!   nothing embedded keeps the vector, embedding nothing. Which of the two
//!   it is depends on the change alone, not on what else shares its commit.
//! - The text not kept is memory not held.

mod common;

use common::*;
use vellev_storage::{Merging, Unembedded, Vectorize};

const PAGES: &str = "pages";
/// Text of the unstored property: the store must never hold these bytes.
const SECRET: &str = "never-stored body words";

/// Embeds a text as its length and its number of words, and remembers what
/// it was asked for.
#[derive(Debug, Default)]
struct Lengths {
    asked: Mutex<Vec<String>>,
}

fn vector_of(text: &str) -> Vec<f32> {
    vec![text.len() as f32, text.split_whitespace().count() as f32]
}

#[async_trait]
impl Vectorize for Lengths {
    async fn vectors(
        &self,
        _: &Embedding,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Unembedded> {
        self.asked.lock().unwrap().extend(texts.iter().cloned());
        Ok(texts.iter().map(|text| vector_of(text)).collect())
    }
}

impl Lengths {
    fn asked(&self) -> Vec<String> {
        std::mem::take(&mut self.asked.lock().unwrap())
    }
}

/// Pages embed their title, body and tags; the body, if `kept` is false,
/// is not stored.
fn pages(kept: bool) -> Mutation {
    let body = Property::new("body", DataType::Text);
    let body = match kept {
        true => body,
        false => body.unstored(),
    };
    Mutation::CreateCollection(Box::new(Collection {
        name: name(PAGES),
        properties: vec![
            Property::new("title", DataType::Text),
            body,
            Property::new("tags", DataType::TextArray),
            Property::new("n", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::Local,
            model: "lengths".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec!["title".into(), "body".into(), "tags".into()],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

fn page(object: u8, properties: serde_json::Value) -> Mutation {
    Mutation::Upsert {
        collection: name(PAGES),
        id: id(object),
        properties: properties.as_object().unwrap().clone(),
        vector: None,
        only_if_version: None,
    }
}

fn patch(object: u8, properties: serde_json::Value, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Patch {
        collection: name(PAGES),
        id: id(object),
        properties: properties.as_object().unwrap().clone(),
        vector,
        only_if: None,
        only_if_version: None,
    }
}

fn append(object: u8, tag: &str) -> Mutation {
    Mutation::Append {
        collection: name(PAGES),
        id: id(object),
        property: "tags".into(),
        values: vec![tag.into()],
        unique: true,
    }
}

async fn writer(cluster: &Cluster, kept: bool) -> (Writer, Arc<Lengths>) {
    let lengths = Arc::new(Lengths::default());
    let writer = cluster
        .writer()
        .await
        .embedding_with(lengths.clone())
        .merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
    writer.commit(vec![pages(kept)]).await.unwrap();
    (writer, lengths)
}

/// The object's properties and vector as `state` reads them.
fn read(
    state: &State,
    object: u8,
) -> (serde_json::Map<String, serde_json::Value>, Option<Vec<f32>>) {
    let found = state.get(&name(PAGES), id(object)).unwrap();
    let vector = found.vector.map(<[f32]>::to_vec);
    (found.object.properties_json(), vector)
}

/// Every file in the store that holds `text`.
async fn holding(cluster: &Cluster, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for listed in cluster.blobs.list("").await.unwrap() {
        let bytes = cluster.blobs.get(&listed.key).await.unwrap();
        if bytes
            .windows(text.len())
            .any(|window| window == text.as_bytes())
        {
            found.push(listed.key);
        }
    }
    found
}

#[tokio::test]
async fn the_text_of_an_unstored_property_is_embedded_and_never_stored() {
    let cluster = Cluster::new().await;
    let (writer, lengths) = writer(&cluster, false).await;
    let whole = format!("a kept title\n{SECRET}\nred");
    writer
        .commit(vec![page(
            1,
            json!({ "title": "a kept title", "body": SECRET, "tags": ["red"], "n": 1 }),
        )])
        .await
        .unwrap();
    assert_eq!(lengths.asked(), vec![whole.clone()]);
    let expected = json!({ "title": "a kept title", "tags": ["red"], "n": 1 });
    let expected = expected.as_object().unwrap().clone();
    let read_back = writer.partition().read(|state| read(state, 1));
    assert_eq!(read_back, (expected.clone(), Some(vector_of(&whole))));

    // Only in the log so far: what replays it reads the vector, and no body.
    let follower = cluster.open().await;
    assert_eq!(follower.partition().read(|state| read(state, 1)), read_back);
    assert_eq!(holding(&cluster, SECRET).await, Vec::<String>::new());

    // Nor in a segment, a flushed one or a merged one.
    writer.flush().await.unwrap();
    writer
        .commit(vec![page(2, json!({ "title": "two", "body": SECRET }))])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    assert!(
        writer.merge().await.unwrap().is_some(),
        "two segments merge"
    );
    let reopened = cluster.open().await;
    assert_eq!(reopened.partition().read(|state| read(state, 1)), read_back);
    assert_eq!(holding(&cluster, SECRET).await, Vec::<String>::new());
    // What is stored is found where it is looked for: the search is sound.
    assert!(!holding(&cluster, "a kept title").await.is_empty());
}

#[tokio::test]
async fn an_upsert_takes_no_vector_from_a_row_whose_text_was_not_kept() {
    let cluster = Cluster::new().await;
    let (writer, lengths) = writer(&cluster, false).await;
    writer
        .commit(vec![page(1, json!({ "title": "same", "body": SECRET }))])
        .await
        .unwrap();
    lengths.asked();
    // The row kept reads "same", as this write's text does, but its vector
    // was made from more than that.
    writer
        .commit(vec![page(1, json!({ "title": "same" }))])
        .await
        .unwrap();
    assert_eq!(lengths.asked(), vec!["same".to_string()]);
    let vector = writer.partition().read(|state| read(state, 1).1);
    assert_eq!(vector, Some(vector_of("same")));

    // Written again as it was, it is embedded again: whether its text is
    // the same cannot be told.
    let again = || page(1, json!({ "title": "same", "body": SECRET }));
    writer.commit(vec![again()]).await.unwrap();
    writer.commit(vec![again()]).await.unwrap();
    let whole = format!("same\n{SECRET}");
    assert_eq!(lengths.asked(), vec![whole.clone(), whole.clone()]);
    let vector = writer.partition().read(|state| read(state, 1).1);
    assert_eq!(vector, Some(vector_of(&whole)));
}

#[tokio::test]
async fn an_upsert_into_a_collection_storing_everything_still_takes_over_its_vector() {
    let cluster = Cluster::new().await;
    let (writer, lengths) = writer(&cluster, true).await;
    let whole = || page(1, json!({ "title": "same", "body": "kept body" }));
    writer.commit(vec![whole()]).await.unwrap();
    lengths.asked();
    writer.commit(vec![whole()]).await.unwrap();
    assert_eq!(lengths.asked(), Vec::<String>::new());
}

#[tokio::test]
async fn a_change_to_embedded_text_not_all_kept_is_refused_unless_it_brings_its_vector() {
    let cluster = Cluster::new().await;
    let (writer, lengths) = writer(&cluster, false).await;
    let whole = format!("title\n{SECRET}\nred");
    writer
        .commit(vec![page(
            1,
            json!({ "title": "title", "body": SECRET, "tags": ["red"], "n": 1 }),
        )])
        .await
        .unwrap();
    lengths.asked();
    let refused = |result: vellev_storage::Result<_>| matches!(result, Err(StorageError::InvalidMutation(ref why)) if why.contains("body"));
    for change in [
        patch(1, json!({ "title": "other" }), None),
        patch(1, json!({ "body": "other" }), None),
        patch(1, json!({ "tags": null }), None),
        append(1, "blue"),
    ] {
        let result = writer.commit(vec![change.clone()]).await;
        assert!(refused(result), "{change:?}");
        // So too in the commit that writes the object: what a change is
        // depends on it alone, not on what shares its commit.
        let alongside = vec![page(2, json!({ "title": "t", "body": SECRET })), change];
        let result = writer.commit(alongside).await;
        assert!(refused(result));
    }
    assert_eq!(lengths.asked(), Vec::<String>::new());
    let unchanged = writer.partition().read(|state| read(state, 1).1);
    assert_eq!(unchanged, Some(vector_of(&whole)));

    // A change to nothing embedded keeps the vector, and embeds nothing.
    writer
        .commit(vec![patch(1, json!({ "n": 2 }), None)])
        .await
        .unwrap();
    assert_eq!(lengths.asked(), Vec::<String>::new());
    let (properties, vector) = writer.partition().read(|state| read(state, 1));
    assert_eq!(properties["n"], json!(2));
    assert_eq!(vector, Some(vector_of(&whole)));

    // One that brings its vector is taken, and its text is not kept.
    writer
        .commit(vec![patch(
            1,
            json!({ "title": "new", "body": SECRET }),
            Some(vec![9.0, 9.0]),
        )])
        .await
        .unwrap();
    let (properties, vector) = writer.partition().read(|state| read(state, 1));
    assert_eq!(properties["title"], json!("new"));
    assert!(!properties.contains_key("body"));
    assert_eq!(vector, Some(vec![9.0, 9.0]));
    assert_eq!(holding(&cluster, SECRET).await, Vec::<String>::new());
}

#[tokio::test]
async fn the_text_not_kept_is_memory_not_held() {
    let body = "word ".repeat(200);
    let mut resident = Vec::new();
    for kept in [true, false] {
        let cluster = Cluster::new().await;
        let (writer, _) = writer(&cluster, kept).await;
        let rows = (0..20).map(|n| page(n, json!({ "title": "t", "body": body })));
        writer.commit(rows.collect()).await.unwrap();
        let in_memtable = writer.partition().read(State::resident);
        writer.flush().await.unwrap();
        let in_segment = writer.partition().read(State::resident);
        resident.push((in_memtable, in_segment));
    }
    let (kept, unkept) = (resident[0], resident[1]);
    let bodies = 20 * body.len();
    assert!(kept.0 >= unkept.0 + bodies, "{kept:?} {unkept:?}");
    assert!(kept.1 >= unkept.1 + bodies, "{kept:?} {unkept:?}");
}

/// A change to nothing embedded, after a whole write of the same object in
/// its commit, leaves the object with the vector that write's text makes:
/// however the two are grouped, and whatever the object was before.
#[tokio::test]
async fn a_change_after_a_whole_write_in_its_commit_keeps_the_text_that_write_brings() {
    let whole = format!("T\n{SECRET}");
    for before in [true, false] {
        for grouped in [false, true] {
            let cluster = Cluster::new().await;
            let (writer, lengths) = writer(&cluster, false).await;
            if before {
                let old = page(1, json!({ "title": "T", "body": "old words" }));
                writer.commit(vec![old]).await.unwrap();
            }
            let upsert = page(1, json!({ "title": "T", "body": SECRET }));
            let changes = [patch(1, json!({ "n": 2 }), None), {
                Mutation::Increment {
                    collection: name(PAGES),
                    id: id(1),
                    property: "n".into(),
                    by: 1.0,
                }
            }];
            match grouped {
                false => {
                    let batch = [vec![upsert], changes.to_vec()].concat();
                    writer.commit(batch).await.unwrap();
                }
                true => {
                    let batches = vec![vec![upsert], changes.to_vec()];
                    for result in writer.commit_group(batches).await.unwrap() {
                        result.unwrap();
                    }
                }
            }
            let (properties, vector) = writer.partition().read(|state| read(state, 1));
            assert_eq!(
                properties["n"],
                json!(3),
                "before: {before}, grouped: {grouped}"
            );
            assert_eq!(
                vector,
                Some(vector_of(&whole)),
                "before: {before}, grouped: {grouped}"
            );
            assert!(lengths.asked().contains(&whole));
        }
    }
}

/// Of two whole writes of an object in one commit, which read the same
/// once their text is gone, the last one's text is what is embedded.
#[tokio::test]
async fn the_last_whole_write_in_a_commit_is_the_text_embedded() {
    for grouped in [false, true] {
        let cluster = Cluster::new().await;
        let (writer, _) = writer(&cluster, false).await;
        let first = page(1, json!({ "title": "T", "body": "first words" }));
        let last = page(1, json!({ "title": "T", "body": SECRET }));
        match grouped {
            false => drop(writer.commit(vec![first, last]).await.unwrap()),
            true => {
                for result in writer
                    .commit_group(vec![vec![first], vec![last]])
                    .await
                    .unwrap()
                {
                    result.unwrap();
                }
            }
        }
        let vector = writer.partition().read(|state| read(state, 1).1);
        assert_eq!(
            vector,
            Some(vector_of(&format!("T\n{SECRET}"))),
            "grouped: {grouped}"
        );
    }
}
