"""SQuAD 2.0 -> rows.jsonl ({id, content, article}) of paragraphs and their sentences, and
queries.jsonl ({text}) of answerable questions. Deterministic."""
import json, re, sys, uuid, random
src, out = sys.argv[1], sys.argv[2]
NS = uuid.UUID("6f0b5c1e-4d1a-4a4e-9b1e-5f1d0c6a7e21")
rows, questions, seen = [], [], set()
for name in ["train-v2.0.json", "dev-v2.0.json"]:
    data = json.load(open(f"{src}/{name}"))["data"]
    for article in data:
        for paragraph in article["paragraphs"]:
            whole = paragraph["context"].strip()
            if whole not in seen:
                seen.add(whole)
                rows.append({"id": str(uuid.uuid5(NS, whole)), "content": whole, "article": article["title"]})
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", paragraph["context"].strip()):
                sentence = sentence.strip()
                if len(sentence) < 12 or sentence in seen:
                    continue
                seen.add(sentence)
                rows.append({"id": str(uuid.uuid5(NS, sentence)), "content": sentence, "article": article["title"]})
            for qa in paragraph["qas"]:
                if not qa.get("is_impossible") and qa["question"].strip():
                    questions.append(qa["question"].strip())
random.Random(668).shuffle(questions)
with open(f"{out}/rows.jsonl", "w") as f:
    for row in rows: f.write(json.dumps(row) + "\n")
with open(f"{out}/queries.jsonl", "w") as f:
    for q in questions[:500]: f.write(json.dumps({"text": q}) + "\n")
print(len(rows), "rows,", min(500, len(questions)), "queries")
