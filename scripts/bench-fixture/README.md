# Benchmark fixture: 1M Wikipedia passages

The data for vellev's nightly benchmark against Weaviate. **It is not a vellev release.** `prepare.py` builds the text; `crates/vellev-embed/examples/embed_shards.rs` embeds it in resumable shards.

## Sources

- **Wikipedia:** the Hugging Face dataset `wikimedia/wikipedia`, `20231101.en`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- **SQuAD 2.0:** train and dev, answerable questions only.
- **Natural Questions:** the test questions, as BEIR ships them.

## Known skew: some large articles are missing

The Wikipedia source leaves out some large articles, among them *New York City*, *Hydrogen* and *Middle Ages*. These are probably pages its own extraction could not parse.

So:
- the corpus leans away from the longest, most-linked articles;
- **42,377 SQuAD questions and 885 NQ questions were dropped** because their article is not in the source.

Every measured question's article is in the corpus, and the truth is exact over the corpus as it is. The skew changes what the corpus is like, not whether the recall figures are right.
