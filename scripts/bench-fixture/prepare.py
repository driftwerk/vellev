"""The benchmark fixture's text: 1,000,000 Wikipedia passages, the questions
measured against them, and a disjoint set to warm up with. Deterministic:
the same inputs give the same files, byte for byte.

Inputs (SRC):
  wiki/train-*-of-00041.parquet  wikimedia/wikipedia 20231101.en, revision
                                 b04c8d1ceb2f5cd4588862100d08de323dccfbaa
  squad/dev-v2.0.json            SQuAD 2.0 dev
  nq/queries.jsonl, nq/corpus.jsonl, nq/qrels/test.tsv
                                 Natural Questions test, as BEIR ships it

Outputs (OUT):
  passages.jsonl  {id, text, title, url, bucket, owner, deleted}, id 0.. in order
  queries.jsonl   {id, text, source, title}, measured
  warmup.jsonl    {id, text, source, title}, never measured
  articles.tsv    title and URL of every article a passage is from
  prepare.json    counts, and what was left out and why

Passages are a Wikipedia article's paragraphs of 60 to 300 words, as they
are: a paragraph is a run of text between blank lines with no line break
in it, so headings, lists and tables, which are lines of their own, are
not passages. Longer and shorter paragraphs are left out.

Articles: first every article a chosen question was asked of, so its
answer is in the corpus; then the rest in the order of the SHA-256 of
their titles, until there are exactly 1,000,000 passages. The articles
asked of come first, so that the first 100,000 passages, the slice a pull
request is checked on, hold the answers too.

Questions: SQuAD 2.0 dev questions that have an answer (the unanswerable
ones are left out), and Natural Questions test questions, each only if
its article is in the dump under that title. Each source's questions are
shuffled with a fixed seed; the first 500 of each are measured and the
next 500 warm up.
"""

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

PASSAGES = 1_000_000
MEASURED, WARMUP = 500, 500
WORDS = (60, 300)
# Articles whose title hashes below this share of the space are kept as
# candidates to fill with, so the whole dump need not be held at once.
CANDIDATES = 0.12


def h(title: str) -> bytes:
    return hashlib.sha256(title.encode()).digest()


def unit(title: str) -> float:
    return int.from_bytes(h(title)[:8], "big") / 2**64


def paragraphs(text: str):
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or "\n" in block:
            continue
        if WORDS[0] <= len(block.split()) <= WORDS[1]:
            yield block


def main(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    squad = json.load(open(src / "squad/dev-v2.0.json"))["data"]
    asked = defaultdict(list)  # source -> [(question, title)]
    for article in squad:
        title = article["title"].replace("_", " ")
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa.get("is_impossible") and qa["question"].strip():
                    asked["squad"].append((qa["question"].strip(), title))
    nq_titles = {}
    with open(src / "nq/corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            nq_titles[row["_id"]] = row["title"]
    nq_text = {}
    with open(src / "nq/queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            nq_text[row["_id"]] = row["text"].strip()
    nq_of = {}
    with open(src / "nq/qrels/test.tsv") as f:
        next(f)
        for line in f:
            query, doc, _ = line.rstrip("\n").split("\t")
            nq_of.setdefault(query, set()).add(nq_titles[doc])
    left_out = defaultdict(int)
    for query in sorted(nq_of, key=lambda q: int(q.removeprefix("test"))):
        titles = nq_of[query]
        if len(titles) != 1:
            left_out["nq: answers in more than one article"] += 1
            continue
        asked["nq"].append((nq_text[query], next(iter(titles))))

    files = sorted((src / "wiki").glob("train-*-of-00041.parquet"))
    assert len(files) == 41, f"{len(files)} parquet files"
    wanted = {title for pairs in asked.values() for _, title in pairs}
    found = {}  # title -> (url, [passages]) of articles asked of
    candidates = {}  # title -> (url, [passages])
    for file in files:
        table = pq.read_table(file, columns=["url", "title", "text"])
        for url, title, text in zip(*(table.column(c).to_pylist() for c in ("url", "title", "text"))):
            keep_found = title in wanted
            keep_candidate = unit(title) < CANDIDATES
            if not (keep_found or keep_candidate):
                continue
            kept = list(paragraphs(text))
            if not kept:
                continue
            if keep_found:
                found[title] = (url, kept)
            elif keep_candidate:
                candidates[title] = (url, kept)
        print(f"{file.name}: {len(found)} asked of, {len(candidates)} candidates", file=sys.stderr)

    chosen = {}
    for source in ("squad", "nq"):
        pairs = [p for p in asked[source] if p[1] in found]
        left_out[f"{source}: article not in the dump or with no passage"] += len(asked[source]) - len(pairs)
        random.Random(f"bench-{source}").shuffle(pairs)
        assert len(pairs) >= MEASURED + WARMUP, f"{source}: only {len(pairs)} questions"
        chosen[source] = pairs[: MEASURED + WARMUP]

    first = sorted({t for pairs in chosen.values() for _, t in pairs}, key=h)
    order = first + sorted((t for t in candidates if t not in found), key=h)
    articles = {**candidates, **found}
    passages, used = [], []
    for title in order:
        url, kept = articles[title]
        used.append((title, url))
        for text in kept:
            passages.append((text, title, url))
            if len(passages) == PASSAGES:
                break
        if len(passages) == PASSAGES:
            break
    assert len(passages) == PASSAGES, f"only {len(passages)} passages: raise CANDIDATES"
    first_set = set(first)
    assert all(t in {u for u, _ in used} for t in first_set)

    deleted = random.Random("bench-deleted")
    with open(out / "passages.jsonl", "w") as f:
        for n, (text, title, url) in enumerate(passages):
            row = {
                "id": n,
                "text": text,
                "title": title,
                "url": url,
                "bucket": n % 100,
                "owner": h(title)[0] % 8,
                "deleted": deleted.random() < 0.05,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "articles.tsv", "w") as f:
        for title, url in used:
            f.write(f"{title}\t{url}\n")
    n = 0
    for name, part in (("queries.jsonl", slice(0, MEASURED)), ("warmup.jsonl", slice(MEASURED, MEASURED + WARMUP))):
        with open(out / name, "w") as f:
            for source in ("squad", "nq"):
                for text, title in chosen[source][part]:
                    f.write(json.dumps({"id": n, "text": text, "source": source, "title": title}, ensure_ascii=False) + "\n")
                    n += 1
    first_count = sum(1 for _, t, _ in passages if t in first_set)
    report = {
        "passages": len(passages),
        "articles": len(used),
        "articles_asked_of": len(first),
        "passages_of_articles_asked_of": first_count,
        "questions_measured": 2 * MEASURED,
        "questions_warmup": 2 * WARMUP,
        "left_out": dict(sorted(left_out.items())),
        "deleted": sum(1 for line in open(out / "passages.jsonl") if '"deleted": true' in line),
    }
    json.dump(report, open(out / "prepare.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
