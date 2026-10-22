# vellev recall fixture: SQuAD 2.0, bge-base-en-v1.5 (v1)

This fixture is an adaptation of **SQuAD 2.0** (the Stanford Question
Answering Dataset), by Pranav Rajpurkar, Robin Jia and Percy Liang
(https://rajpurkar.github.io/SQuAD-explorer/), whose passages are drawn from
English Wikipedia articles written by Wikipedia's contributors.

SQuAD 2.0 is licensed under the Creative Commons Attribution-ShareAlike 4.0
International licence (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/

**Changes made.** The train and dev sets' paragraphs were taken whole and
split into sentences. Those of at least 12 characters were kept, and each
distinct text became one row with an id derived from it (UUID v5). The
first 500 answerable questions, after a fixed shuffle, became the queries.
Every row was embedded with BAAI/bge-base-en-v1.5 (768 dimensions, run
locally through fastembed), with no prefix. Each query was embedded with
the prefix "Represent this sentence for searching relevant passages: ".
The scripts are `scripts/recall-fixture/prepare.py` and the
`vellev-embed` example `recall_fixture` in the vellev repository.

**Licence of this fixture.** This fixture (`memories.jsonl.zst`, the rows'
text with their vectors, and `queries.vectors.jsonl.zst`, the questions
with theirs) is licensed under **CC BY-SA 4.0**, as the material it adapts
requires. It is not covered by vellev's own licence. It is a test input,
downloaded by CI; it is not part of vellev's source or of any vellev
release.

Inputs: train-v2.0.json, SHA-256 68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002;
dev-v2.0.json, SHA-256 80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8.
Uncompressed outputs: memories.jsonl, 118,609 rows, SHA-256
40fd720d1e40c5f66619914b8639b09b6cbd9317b9ff108153800e494115c3c5;
queries.vectors.jsonl, 500 queries, SHA-256
1717fc3c6162383d2e0b8550cc02f603aa4173a3993dde5a5827c2135f560a8e.
