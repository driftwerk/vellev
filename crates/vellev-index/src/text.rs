//! Keyword search: words, postings, and the BM25 formula, each as Weaviate
//! has it so that the two rank alike.
//!
//! Text is split into [`words`] on anything that is not a letter or a
//! digit, and lowercased. English stopwords count toward a row's length but
//! are never looked for, so a query's [`terms`] leave them out.

use std::collections::HashMap;

use vellev_core::{Bm25, Collection, Object, Value};

/// The words too common to be worth matching on, in the list Lucene and the
/// engines built on it use for English. Sorted, to be searched.
const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// The words of `text`, in order, with repeats. Each letter is lowercased
/// on its own, and whatever that adds besides letters is dropped, so that
/// `İ` becomes `i`.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            word.chars()
                .flat_map(char::to_lowercase)
                .filter(|c| c.is_alphanumeric())
                .collect()
        })
}

fn is_stopword(word: &str) -> bool {
    STOPWORDS.binary_search(&word).is_ok()
}

/// The words of `text` worth looking for.
pub fn terms(text: &str) -> impl Iterator<Item = String> + '_ {
    words(text).filter(|word| !is_stopword(word))
}

/// One row holding a token in a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub row: u32,
    /// How many times the row holds the token.
    pub frequency: u32,
    /// How many tokens the row has in the property in all.
    pub length: u32,
}

/// How many postings share one upper bound. A scan skips a whole block
/// when nothing in it can reach the score it needs.
pub const BLOCK: usize = 128;

/// The most any posting in a block of them can contribute: the highest
/// frequency and the shortest row in the block, which together bound the
/// score of every posting in it, whatever the average length and weight.
#[derive(Debug, Clone, Copy)]
struct Block {
    max_frequency: u32,
    min_length: u32,
}

/// The rows holding one token, as positions in the field's `rows`, with a
/// bound per [`BLOCK`] of them.
#[derive(Debug, Clone, Default)]
struct Postings {
    holders: Vec<u32>,
    frequencies: Vec<u32>,
    blocks: Vec<Block>,
}

impl Postings {
    fn push(&mut self, holder: u32, frequency: u32, length: u32) {
        if self.holders.len().is_multiple_of(BLOCK) {
            self.blocks.push(Block {
                max_frequency: frequency,
                min_length: length,
            });
        } else if let Some(block) = self.blocks.last_mut() {
            block.max_frequency = block.max_frequency.max(frequency);
            block.min_length = block.min_length.min(length);
        }
        self.holders.push(holder);
        self.frequencies.push(frequency);
    }
}

/// A walk forward along one token's postings in one field, by posting or
/// by skipping to a row, with the bound of the block it is in. As an
/// iterator it yields the postings left.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    field: &'a Field,
    postings: &'a Postings,
    at: usize,
}

impl Cursor<'_> {
    /// How many postings the token has in all.
    pub fn len(&self) -> usize {
        self.postings.holders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.postings.holders.is_empty()
    }

    /// The posting the cursor is at, if any is left.
    pub fn posting(&self) -> Option<Posting> {
        let holder = *self.postings.holders.get(self.at)?;
        Some(Posting {
            row: self.field.rows[holder as usize],
            frequency: self.postings.frequencies[self.at],
            length: self.field.lengths[holder as usize],
        })
    }

    pub fn row(&self) -> Option<u32> {
        let holder = *self.postings.holders.get(self.at)?;
        Some(self.field.rows[holder as usize])
    }

    pub fn advance(&mut self) {
        self.at += 1;
    }

    /// Move to the first posting at or past `row`.
    pub fn seek(&mut self, row: u32) {
        let rest = &self.postings.holders[self.at.min(self.len())..];
        self.at += rest.partition_point(|holder| self.field.rows[*holder as usize] < row);
    }

    /// The block holding the first posting at or past `row`, without moving
    /// there: the last row in it, and the most any posting in it can score
    /// with `weight`. `None` once no posting is left that far on.
    pub fn block_at(
        &self,
        row: u32,
        parameters: Bm25,
        weight: f32,
        average: f32,
    ) -> Option<(u32, f32)> {
        let rest = &self.postings.holders[self.at.min(self.len())..];
        let at = self.at + rest.partition_point(|holder| self.field.rows[*holder as usize] < row);
        if at >= self.len() {
            return None;
        }
        let block = &self.postings.blocks[at / BLOCK];
        let last = ((at / BLOCK + 1) * BLOCK).min(self.len()) - 1;
        let end = self.field.rows[self.postings.holders[last] as usize];
        Some((end, bound(parameters, weight, average, block)))
    }

    /// The most any posting of the token can score with `weight`.
    pub fn bound(&self, parameters: Bm25, weight: f32, average: f32) -> f32 {
        self.postings
            .blocks
            .iter()
            .map(|block| bound(parameters, weight, average, block))
            .fold(0.0, f32::max)
    }
}

impl Iterator for Cursor<'_> {
    type Item = Posting;

    fn next(&mut self) -> Option<Posting> {
        let posting = self.posting()?;
        self.advance();
        Some(posting)
    }
}

fn bound(parameters: Bm25, weight: f32, average: f32, block: &Block) -> f32 {
    score(
        parameters,
        weight,
        block.max_frequency as f32,
        block.min_length as f32,
        average,
    )
}

/// One property's text: the rows that have it, ascending, with their
/// lengths, and for every token the rows holding it.
#[derive(Debug, Clone, Default)]
pub struct Field {
    rows: Vec<u32>,
    lengths: Vec<u32>,
    postings: HashMap<String, Postings>,
}

impl Field {
    /// The rows holding `token`, ascending.
    pub fn holding<'a>(&'a self, token: &str) -> impl Iterator<Item = Posting> + 'a {
        self.cursor(token).into_iter().flatten()
    }

    /// A walk along the rows holding `token`, if any does.
    pub fn cursor(&self, token: &str) -> Option<Cursor<'_>> {
        let postings = self.postings.get(token)?;
        Some(Cursor {
            field: self,
            postings,
            at: 0,
        })
    }

    fn length(&self, row: u32) -> Option<u32> {
        let at = self.rows.binary_search(&row).ok()?;
        Some(self.lengths[at])
    }
}

/// The searchable text of one collection's rows, by property. Rows are
/// numbered by the caller and must be inserted in ascending order.
#[derive(Debug, Clone, Default)]
pub struct TextIndex {
    fields: HashMap<String, Field>,
}

/// One row's searchable text, split into tokens and counted, ready for
/// [`TextIndex::insert_tokenized`]: worked out apart from any index, so
/// that many rows can be tokenized at once and then indexed in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tokenized {
    fields: Vec<TokenizedField>,
}

/// One searchable property's text in a row: its name, how many words it
/// has, and how often each token not a stopword occurs.
#[derive(Debug, Clone, PartialEq)]
struct TokenizedField {
    name: String,
    length: u32,
    frequencies: Vec<(String, u32)>,
}

impl TextIndex {
    /// `object`'s searchable text as [`Self::insert`] would index it.
    pub fn tokenize(collection: &Collection, object: &Object) -> Tokenized {
        let searchable = object.properties.iter().filter(|(name, _)| {
            collection
                .property(name)
                .is_some_and(|property| property.searchable)
        });
        let fields = searchable
            .filter_map(|(name, value)| {
                let texts: &[String] = match value {
                    Value::Text(text) => std::slice::from_ref(text),
                    Value::TextArray(texts) => texts,
                    _ => return None,
                };
                let mut frequencies: HashMap<String, u32> = HashMap::new();
                let mut length = 0;
                for word in texts.iter().flat_map(|text| words(text)) {
                    length += 1;
                    if !is_stopword(&word) {
                        *frequencies.entry(word).or_default() += 1;
                    }
                }
                Some(TokenizedField {
                    name: name.clone(),
                    length,
                    frequencies: frequencies.into_iter().collect(),
                })
            })
            .collect();
        Tokenized { fields }
    }

    /// Record a row tokenized as [`Self::tokenize`] does, as row `row`:
    /// rows must come in ascending order, as for [`Self::insert`].
    pub fn insert_tokenized(&mut self, row: u32, tokenized: Tokenized) {
        for TokenizedField {
            name,
            length,
            frequencies,
        } in tokenized.fields
        {
            let field = match self.fields.get_mut(&name) {
                Some(field) => field,
                None => self.fields.entry(name).or_default(),
            };
            let holder = field.rows.len() as u32;
            field.rows.push(row);
            field.lengths.push(length);
            for (token, frequency) in frequencies {
                field
                    .postings
                    .entry(token)
                    .or_default()
                    .push(holder, frequency, length);
            }
        }
    }

    /// Record `object` as row `row`, indexing its searchable properties.
    pub fn insert(&mut self, collection: &Collection, row: u32, object: &Object) {
        self.insert_tokenized(row, Self::tokenize(collection, object));
    }

    pub fn field(&self, property: &str) -> Option<&Field> {
        self.fields.get(property)
    }

    /// How many tokens `row` has in each property it has text in.
    pub fn lengths(&self, row: u32) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.fields
            .iter()
            .filter_map(move |(name, field)| Some((name.as_str(), field.length(row)?)))
    }
}

/// How much rarer tokens count: `objects` in all, `holding` of them with
/// the token. Never negative, however common the token.
pub fn idf(objects: u64, holding: u64) -> f32 {
    let (objects, holding) = (objects as f32, holding as f32);
    (1.0 + (objects - holding + 0.5) / (holding + 0.5)).ln()
}

/// One token's contribution to a row's score: its weight `idf`, how often
/// the row holds it, and the row's length against the average. More
/// occurrences count for less each, never reaching `idf` between them, and
/// a long row needs more of them.
pub fn score(parameters: Bm25, idf: f32, frequency: f32, length: f32, average: f32) -> f32 {
    let Bm25 { k1, b } = parameters;
    let relative = if average > 0.0 { length / average } else { 0.0 };
    idf * frequency / (frequency + k1 * (1.0 - b + b * relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        super::words(text).collect()
    }

    fn terms(text: &str) -> Vec<String> {
        super::terms(text).collect()
    }

    #[test]
    fn text_is_split_on_anything_but_letters_and_digits_and_lowercased() {
        assert_eq!(
            words("Deploys go-out on Tuesdays!"),
            ["deploys", "go", "out", "on", "tuesdays"]
        );
        assert_eq!(
            words("date:2026-07-05 v1.2_final"),
            ["date", "2026", "07", "05", "v1", "2", "final"]
        );
        assert_eq!(words("  \t\n--"), [] as [&str; 0]);
        assert_eq!(
            words("Ærøskøbing ÉCOLE naïve"),
            ["ærøskøbing", "école", "naïve"]
        );
        assert_eq!(words("東京 tower"), ["東京", "tower"]);
        assert_eq!(words("İstanbul ΟΔΟΣ"), ["istanbul", "οδοσ"]);
        assert_eq!(
            words("The cat AND the hat"),
            ["the", "cat", "and", "the", "hat"]
        );
    }

    #[test]
    fn stopwords_are_not_looked_for_whatever_their_case() {
        assert!(STOPWORDS.is_sorted());
        assert_eq!(terms("The cat AND the hat"), ["cat", "hat"]);
        assert_eq!(terms("to be or not to be"), [] as [&str; 0]);
        assert_eq!(terms("there their theremin"), ["theremin"]);
    }

    fn indexed(texts: &[&str]) -> TextIndex {
        use vellev_core::{CollectionName, DataType, Embedding, EmbeddingProvider, Property};
        let collection = Collection {
            name: CollectionName::new("notes").unwrap(),
            properties: vec![Property::new("body", DataType::Text).searchable()],
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "none".into(),
                dimensions: 2,
                query_prefix: None,
                document_prefix: None,
                properties: vec![],
            },
            distance: Default::default(),
            bm25: Bm25::default(),
            access_rule: None,
        };
        let mut index = TextIndex::default();
        for (row, text) in texts.iter().enumerate() {
            let object = Object {
                id: vellev_core::ObjectId::generate(),
                properties: [("body".to_string(), Value::Text(text.to_string()))].into(),
                vector: None,
            };
            index.insert(&collection, row as u32 * 3, &object);
        }
        index
    }

    #[test]
    fn a_cursor_walks_seeks_and_bounds_its_blocks() {
        let texts: Vec<String> = (0..300)
            .map(|n| match n % 3 {
                0 => "fox fox fox".to_string(),
                1 => "fox and a hen in a den".to_string(),
                _ => "hen".to_string(),
            })
            .collect();
        let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
        let index = indexed(&borrowed);
        let field = index.field("body").unwrap();
        let mut cursor = field.cursor("fox").unwrap();
        assert_eq!(cursor.len(), 200);
        assert_eq!(cursor.row(), Some(0));
        cursor.advance();
        assert_eq!(cursor.row(), Some(3));
        cursor.seek(10);
        assert_eq!(cursor.row(), Some(12), "row 10 holds no fox");
        assert_eq!(
            cursor.posting(),
            Some(Posting {
                row: 12,
                frequency: 1,
                length: 7
            })
        );
        let bm25 = Bm25::default();
        let best = score(bm25, 2.0, 3.0, 3.0, 5.0);
        let end = field.rows[cursor.postings.holders[BLOCK - 1] as usize];
        assert_eq!(cursor.block_at(12, bm25, 2.0, 5.0), Some((end, best)));
        assert_eq!(cursor.block_at(end + 1, bm25, 2.0, 5.0).unwrap().1, best);
        assert!(cursor.block_at(end + 1, bm25, 2.0, 5.0).unwrap().0 > end);
        assert_eq!(cursor.bound(bm25, 2.0, 5.0), best);
        cursor.seek(u32::MAX);
        assert_eq!(cursor.row(), None);
        assert_eq!(cursor.block_at(0, bm25, 2.0, 5.0), None);
        assert!(field.cursor("owl").is_none());
        assert!(field.cursor("hen").is_some_and(|hen| hen.len() == 200));
    }

    #[test]
    fn rarer_tokens_weigh_more_and_no_weight_is_negative() {
        assert!(idf(1000, 1) > idf(1000, 10));
        assert!(idf(1000, 10) > idf(1000, 999));
        assert!(idf(1000, 1000) > 0.0);
        assert!(idf(1, 1) > 0.0);
    }

    #[test]
    fn frequency_saturates_and_longer_rows_score_lower() {
        let bm25 = Bm25::default();
        let at = |frequency: f32, length: f32| score(bm25, 1.0, frequency, length, 10.0);
        assert!(at(2.0, 10.0) > at(1.0, 10.0));
        assert!(at(2.0, 10.0) - at(1.0, 10.0) > at(20.0, 10.0) - at(19.0, 10.0));
        assert!(at(1000.0, 10.0) < 1.0);
        assert!(at(1.0, 5.0) > at(1.0, 10.0));
        assert!(at(1.0, 10.0) > at(1.0, 50.0));
        assert_eq!(at(0.0, 10.0), 0.0);
        assert!(score(bm25, 1.0, 1.0, 0.0, 0.0).is_finite());

        let ignoring_length = Bm25 { k1: 1.2, b: 0.0 };
        assert_eq!(
            score(ignoring_length, 1.0, 1.0, 5.0, 10.0),
            score(ignoring_length, 1.0, 1.0, 50.0, 10.0)
        );
        assert_eq!(score(bm25, 2.0, 1.0, 10.0, 10.0), 2.0 / 2.2);
    }

    /// Rows tokenized apart, on several threads, and then indexed in order
    /// make the index rows inserted one by one make: every token's walk the
    /// same, posting for posting, block for block.
    #[test]
    fn rows_tokenized_apart_and_indexed_in_order_index_as_rows_inserted_one_by_one() {
        use rayon::prelude::*;
        use vellev_core::{
            CollectionName, DataType, Distance, Embedding, EmbeddingProvider, ObjectId, Property,
        };
        let collection = Collection {
            name: CollectionName::new("notes").unwrap(),
            properties: vec![
                Property::new("title", DataType::Text).searchable(),
                Property::new("tags", DataType::TextArray).searchable(),
                Property::new("owner", DataType::Text),
            ],
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "m".into(),
                dimensions: 2,
                query_prefix: None,
                document_prefix: None,
                properties: vec![],
            },
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        };
        let words = [
            "Fox", "hen", "den", "the", "owl", "ÉCOLE", "naïve", "go-out", "a", "東京",
        ];
        let objects: Vec<Object> = (0..3_000u32)
            .map(|n| {
                let pick = |k: u32| words[((n * 7 + k * 13) % 10) as usize];
                let title: Vec<&str> = (0..(n % 40)).map(pick).collect();
                let mut properties = std::collections::BTreeMap::new();
                if n % 9 != 0 {
                    properties.insert("title".to_string(), Value::Text(title.join(" ")));
                }
                if n % 4 == 0 {
                    properties.insert(
                        "tags".to_string(),
                        Value::TextArray(vec![pick(1).into(), pick(2).into()]),
                    );
                }
                properties.insert("owner".to_string(), Value::Text("ada".into()));
                Object {
                    id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
                    properties,
                    vector: None,
                }
            })
            .collect();
        let mut one_by_one = TextIndex::default();
        for (row, object) in objects.iter().enumerate() {
            one_by_one.insert(&collection, row as u32 * 2, object);
        }
        let tokenized: Vec<Tokenized> = objects
            .par_iter()
            .map(|object| TextIndex::tokenize(&collection, object))
            .collect();
        let mut apart = TextIndex::default();
        for (row, tokenized) in tokenized.into_iter().enumerate() {
            apart.insert_tokenized(row as u32 * 2, tokenized);
        }
        let parameters = Bm25::default();
        for property in ["title", "tags"] {
            let (a, b) = (
                one_by_one.field(property).unwrap(),
                apart.field(property).unwrap(),
            );
            assert_eq!((&a.rows, &a.lengths), (&b.rows, &b.lengths), "{property}");
            let mut tokens: Vec<&String> = a.postings.keys().collect();
            tokens.sort();
            let mut theirs: Vec<&String> = b.postings.keys().collect();
            theirs.sort();
            assert_eq!(tokens, theirs, "{property}");
            for token in tokens {
                let walk = |field: &Field| -> Vec<(Posting, Option<(u32, u32)>)> {
                    let cursor = field.cursor(token).unwrap();
                    let mut walked = Vec::new();
                    let mut at = cursor.clone();
                    while let Some(row) = at.row() {
                        let block = at
                            .block_at(row, parameters, 1.0, 10.0)
                            .map(|(end, bound)| (end, bound.to_bits()));
                        walked.push((at.posting().unwrap(), block));
                        at.advance();
                    }
                    walked
                };
                assert_eq!(walk(a), walk(b), "{property}: {token}");
            }
        }
    }
}
