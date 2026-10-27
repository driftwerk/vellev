# vellev benchmark fixture: 1M Wikipedia passages, bge-base-en-v1.5 (v1)

This fixture is the data vellev's nightly benchmark against Weaviate runs on. It adapts three works.

**Wikipedia.** The passages are from English Wikipedia articles, written by Wikipedia's contributors. They were taken from the Hugging Face dataset `wikimedia/wikipedia`, configuration `20231101.en` (the dump of 1 November 2023), at revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Wikipedia's text is licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), most of it also under the GFDL; see https://en.wikipedia.org/wiki/Wikipedia:Copyrights. `articles.tsv.zst` lists the title and URL of every one of the 268,306 articles a passage is from. Every passage in `passages.jsonl.zst` also carries its article's title and URL.

**SQuAD 2.0** (the Stanford Question Answering Dataset), by Pranav Rajpurkar, Robin Jia and Percy Liang (https://rajpurkar.github.io/SQuAD-explorer/), licensed under CC BY-SA 4.0. The questions with an answer were taken from its train and dev sets. Unanswerable questions were left out.

**Natural Questions**, by Tom Kwiatkowski et al., Google (https://ai.google.com/research/NaturalQuestions), licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). It is adapted here under CC BY-SA 4.0, as section 4(b) of CC BY-SA 3.0 allows. Only the question strings and the titles of the articles they were asked of were used. They come from the test questions as BEIR distributes them (`nq.zip`, https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip, MD5 `d4d3d2e48787a744b6f6e691ff534307`, SHA-256 `2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752`). BEIR adds no licence of its own to the data. The questions taken are those in `queries.jsonl` and `warmup.jsonl` whose `source` is `nq`.

**The embedding model.** The vectors were made with BAAI/bge-base-en-v1.5 (MIT licence, https://huggingface.co/BAAI/bge-base-en-v1.5), as the ONNX export `Xenova/bge-base-en-v1.5` at revision `4d6cd88e18e51a5e020c2c305726d76ada9c03cf`.

## Changes made

- **Passages.** Each article's text was split into paragraphs: runs of text between blank lines with no line break in them. Headings, lists and tables, which are lines of their own, were dropped, along with the references. Paragraphs of 60 to 300 words were kept as passages.
- **Articles.** First every article a chosen question was asked of; then the rest in the order of the SHA-256 of their titles, until there were exactly 1,000,000 passages.
- **Questions.** 1,000 measured questions (`queries.jsonl`) and a disjoint 1,000 to warm up with (`warmup.jsonl`), half SQuAD 2.0 and half Natural Questions. Each question's article is in the corpus. At most five questions were taken of any one article.
- **Row attributes.** Each passage was given `bucket` (its id mod 100), `owner` (the first byte of the SHA-256 of its article's title, mod 8) and `deleted` (true for about 5%, from a fixed seed).
- **Vectors.** Every passage was embedded with no prefix. Every question was embedded with the prefix "Represent this sentence for searching relevant passages: ". All vectors are 768 little-endian f32s a row, L2-normalised by the model.
- **Truth.** `truth.jsonl` holds the exact top 100 passages by cosine of every measured question under each filter case, computed in f64.

The scripts are `scripts/bench-fixture/prepare.py` and `scripts/bench-fixture/truth.py`, and the `vellev-embed` example `embed_shards`, in the vellev repository. `manifest.json` records how the vectors were made.

## Licence of this fixture

This fixture is licensed under **CC BY-SA 4.0**, as the material it adapts requires: the passages, the questions, the vectors, the truth and the lists. It is not covered by vellev's own licence. It is a test input, downloaded by CI; it is not part of vellev's source or of any vellev release.

## Inputs

- SQuAD 2.0 `train-v2.0.json`, SHA-256 `68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002`
- SQuAD 2.0 `dev-v2.0.json`, SHA-256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`
- Natural Questions through BEIR: `nq.zip` as above
- Wikipedia: `wikimedia/wikipedia` `20231101.en` at the revision above, its 41 parquet files

## Known skew

The Wikipedia source lacks some large articles, among them *New York City*, *Hydrogen* and *Middle Ages*. 42,377 SQuAD questions and 885 Natural Questions were left out because their article is not in it. See the README.
