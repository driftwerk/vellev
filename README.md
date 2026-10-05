# vellev

A vector database written in Rust. Stateless nodes serve hybrid vector and
keyword search; object storage holds the data and a Postgres row per tenant
holds the commit pointer and writer lease.

Design: https://claude.ai/code/artifact/4a8d0a03-c431-4595-8810-c2b8e64787b6

## Layout

| Crate | Purpose |
| --- | --- |
| `vellev` | The node binary: `vellev serve` and `vellev health` |
| `vellev-client` | Rust client for the HTTP API |
| `vellev-core` | Shared types |
| `vellev-index` | Vector and keyword indexes, filter engine |
| `vellev-storage` | Segments and manifests on object storage, commit coordinator |

## Develop

```sh
docker compose -f docker-compose.dev.yml up -d   # Postgres on 54329, S3 on 59090 (bucket vellev-dev)
cargo test --workspace
cargo run -p vellev -- serve --listen 127.0.0.1:8080
cargo run -p vellev -- health --url http://127.0.0.1:8080
```

CI runs `cargo fmt --check`, `cargo clippy -D warnings`, the tests, a licence
check (`cargo deny check`, config in `deny.toml`) and an image build.
