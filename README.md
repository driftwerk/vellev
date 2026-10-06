# vellev

A vector database written in Rust. Stateless nodes serve hybrid vector and
keyword search; object storage holds the data and a Postgres row per tenant
holds the commit pointer and writer lease.

Design: https://claude.ai/code/artifact/4a8d0a03-c431-4595-8810-c2b8e64787b6

## Layout

| Crate | Purpose |
| --- | --- |
| `vellev` | The node binary and its HTTP API: `vellev serve`, `vellev health`, `vellev openapi` |
| `vellev-api` | The bodies of the HTTP API's requests and responses |
| `vellev-client` | Rust client for the HTTP API |
| `vellev-core` | Shared types |
| `vellev-embed` | Turning text into vectors: a local model, Bedrock, or an HTTP endpoint |
| `vellev-index` | Vector and keyword indexes, filter engine |
| `vellev-node` | The tenants one node writes for |
| `vellev-storage` | Segments and manifests on object storage, commit coordinator |

## Develop

```sh
docker compose -f docker-compose.dev.yml up -d   # Postgres on 54329, S3 on 59090 (bucket vellev-dev)
export VELLEV_TEST_DATABASE_URL=postgres://vellev:vellev@127.0.0.1:54329/vellev
cargo test --workspace                           # Postgres tests skip when the URL is unset
cargo run -p vellev -- health --url http://127.0.0.1:8080
```

## Run

```sh
export VELLEV_DATABASE_URL=postgres://vellev:vellev@127.0.0.1:54329/vellev
export VELLEV_STORE=file:///tmp/vellev        # or s3://bucket, with the usual AWS variables
export VELLEV_API_KEYS=dev-key
cargo run -p vellev -- serve --listen 127.0.0.1:8080
```

`vellev serve --help` lists every setting. A node answers `/health/live` as
soon as it is up and `/health/ready` once it has reached Postgres; until then
the API answers 503.

```sh
call() { curl -sS -H 'authorization: Bearer dev-key' -H 'content-type: application/json' "$@"; }
notes=http://127.0.0.1:8080/v1/tenants/acme/collections/notes
call -X PUT http://127.0.0.1:8080/v1/tenants/acme
call -X PUT "$notes" -d '{"name":"notes","properties":[{"name":"title","data_type":"text","searchable":true}],"embedding":{"provider":"none","model":"client","dimensions":2}}'
call -X PUT "$notes/objects/00000000-0000-0000-0000-000000000001" -d '{"properties":{"title":"a red fox"},"vector":[1,0]}'
call -X POST "$notes/hybrid" -d '{"text":"fox","vector":[1,0],"limit":5}'
```

A tenant's segments are merged in the background once enough small ones
have piled up or one is mostly deleted rows, and files nothing reaches any
more are deleted after `VELLEV_GRACE_SECS` (a day by default), within which
the tenant can still be put back as it was at any moment.

Several nodes may run against the same Postgres and store. Each tenant is
written by one node at a time; a request that reaches another node is sent
on to the holder, so set `VELLEV_ADVERTISE_URL` to each node's own address
(never a load balancer's) wherever the default of this host's name and port
would not reach it.

The API is described at `/openapi.json`, and by `vellev openapi` without a
running node. Every route under `/v1` needs one of the configured keys. A
collection may only name an embedding endpoint listed in
`VELLEV_EMBEDDING_URLS` or given a key in `VELLEV_EMBEDDING_KEYS`.

## Simulation

`crates/vellev-storage/tests/simulation.rs` drives random histories of writes,
checkpoints, takeovers and catch-ups with faults injected into the store and
the coordinator, and checks after every step that nothing acknowledged is
lost and nothing unrecorded is visible. A failure prints the shortest history
that reproduces it.

```sh
cargo test -p vellev-storage --test simulation
PROPTEST_CASES=20000 cargo test --release -p vellev-storage --test simulation   # as the nightly run does
```

CI runs `cargo fmt --check`, `cargo clippy -D warnings`, the tests, a licence
check (`cargo deny check`, config in `deny.toml`) and an image build.
