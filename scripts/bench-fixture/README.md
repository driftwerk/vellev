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

## Embedding on a GPU

The 1M vectors were embedded on a RunPod A40 with the `cuda` feature. It took 56 minutes and cost about $0.67; the Mac would have taken about a day. ONNX Runtime's own download for the `cuda` feature is built only for CUDA 13, so the run linked Microsoft's `onnxruntime-linux-x64-gpu_cuda12-1.28.0` build, the same ONNX Runtime version, into the image `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`:

```sh
ORT_LIB_LOCATION=/opt/ort/lib ORT_PREFER_DYNAMIC_LINK=1 \
  cargo build --release --locked -p vellev-embed --features cuda --example embed_shards
LD_LIBRARY_PATH=/opt/ort/lib target/release/examples/embed_shards passages.jsonl out/passages
```

Before the full run, 10,000 passages and 200 prefixed queries were re-embedded on the GPU and compared with the Mac's CPU vectors. The smallest cosine was 1.000000000 and the largest element difference 3.6e-7. The CUDA build can segfault at exit after its last shard is written, so judge a run by its shard sizes, not its exit code.
