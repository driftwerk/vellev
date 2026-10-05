//! Keyword search: tokens, postings, and the BM25 formula.
//!
//! Text is split on anything that is not a letter or a digit, lowercased,
//! and stripped of English stopwords, both when it is stored and when it is
//! asked for, by the one [`tokens`] function.

use std::collections::HashMap;

use vellev_core::{Bm25, Collection, Object, Value};

/// The words too common to be worth matching on, in the list Lucene and the
/// engines built on it use for English. Sorted, to be searched.
const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// The tokens of `text`, in order, with repeats.
pub fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .filter(|word| STOPWORDS.binary_search(&word.as_str()).is_err())
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

/// The rows holding one token, as positions in the field's `rows`.
#[derive(Debug, Clone, Default)]
struct Postings {
    holders: Vec<u32>,
    frequencies: Vec<u32>,
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
        self.postings.get(token).into_iter().flat_map(|postings| {
            postings
                .holders
                .iter()
                .zip(&postings.frequencies)
                .map(|(holder, frequency)| Posting {
                    row: self.rows[*holder as usize],
                    frequency: *frequency,
                    length: self.lengths[*holder as usize],
                })
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

impl TextIndex {
    /// Record `object` as row `row`, indexing its searchable properties.
    pub fn insert(&mut self, collection: &Collection, row: u32, object: &Object) {
        let searchable = object.properties.iter().filter(|(name, _)| {
            collection
                .property(name)
                .is_some_and(|property| property.searchable)
        });
        for (name, value) in searchable {
            let texts: &[String] = match value {
                Value::Text(text) => std::slice::from_ref(text),
                Value::TextArray(texts) => texts,
                _ => continue,
            };
            let mut frequencies: HashMap<String, u32> = HashMap::new();
            let mut length = 0;
            for token in texts.iter().flat_map(|text| tokens(text)) {
                *frequencies.entry(token).or_default() += 1;
                length += 1;
            }
            let field = match self.fields.get_mut(name) {
                Some(field) => field,
                None => self.fields.entry(name.clone()).or_default(),
            };
            let holder = field.rows.len() as u32;
            field.rows.push(row);
            field.lengths.push(length);
            for (token, frequency) in frequencies {
                let postings = field.postings.entry(token).or_default();
                postings.holders.push(holder);
                postings.frequencies.push(frequency);
            }
        }
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
/// occurrences count for less each, and a long row needs more of them.
pub fn score(parameters: Bm25, idf: f32, frequency: f32, length: f32, average: f32) -> f32 {
    let Bm25 { k1, b } = parameters;
    let relative = if average > 0.0 { length / average } else { 0.0 };
    idf * frequency * (k1 + 1.0) / (frequency + k1 * (1.0 - b + b * relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        tokens(text).collect()
    }

    #[test]
    fn text_is_split_on_anything_but_letters_and_digits_and_lowercased() {
        assert_eq!(
            words("Deploys go-out on Tuesdays!"),
            ["deploys", "go", "out", "tuesdays"]
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
    }

    #[test]
    fn stopwords_are_dropped_whatever_their_case() {
        assert!(STOPWORDS.is_sorted());
        assert_eq!(words("The cat AND the hat"), ["cat", "hat"]);
        assert_eq!(words("to be or not to be"), [] as [&str; 0]);
        assert_eq!(words("there their theremin"), ["theremin"]);
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
        assert!(at(1000.0, 10.0) < bm25.k1 + 1.0);
        assert!(at(1.0, 5.0) > at(1.0, 10.0));
        assert!(at(1.0, 10.0) > at(1.0, 50.0));
        assert_eq!(at(0.0, 10.0), 0.0);
        assert!(score(bm25, 1.0, 1.0, 0.0, 0.0).is_finite());

        let ignoring_length = Bm25 { k1: 1.2, b: 0.0 };
        assert_eq!(
            score(ignoring_length, 1.0, 1.0, 5.0, 10.0),
            score(ignoring_length, 1.0, 1.0, 50.0, 10.0)
        );
        assert_eq!(score(bm25, 2.0, 1.0, 10.0, 10.0), 2.0);
    }
}
