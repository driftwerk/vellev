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
docker compose -f docker-compose.dev.yml up -d   # Postgres on 54329, S3 on 59090 (bucket vellev-dev), Weaviate on 58080
export VELLEV_TEST_DATABASE_URL=postgres://vellev:vellev@127.0.0.1:54329/vellev
export VELLEV_WEAVIATE_URL=http://127.0.0.1:58080   # the comparison with Weaviate
cargo test --workspace                           # Postgres tests skip when the URL is unset
cargo run -p vellev -- health --url http://127.0.0.1:8080
```

## Import from Weaviate

```sh
vellev import weaviate --from http://weaviate:8080 --to http://vellev:8080 --tenant acme
```

Copies every class (or each `--class`) into a collection of the same name, with ids and vectors, keeping progress in `vellev-import-TENANT.json` so a run that stops resumes. A class vellev should embed itself instead takes `--embedding CLASS=embedding.json`. Writes made in Weaviate during the import are caught up at the end, and counts are checked against Weaviate's. `--created-as` and `--updated-as` keep Weaviate's object times as date properties; `--per-second` limits the load on the source.

## Run

```sh
export VELLEV_DATABASE_URL=postgres://vellev:vellev@127.0.0.1:54329/vellev
export VELLEV_STORE=file:///tmp/vellev        # or s3://bucket, with the usual AWS variables
export VELLEV_API_KEYS=dev-key
cargo run -p vellev -- serve --listen 127.0.0.1:8080
```

`VELLEV_API_KEYS` are admin keys: they reach every tenant, and only they
create, drop and list tenants. Give each tenant's caller a key of its own,
which reaches that tenant alone; to it every other tenant is as one that
does not exist:

```sh
curl -sS -X POST -H 'authorization: Bearer dev-key' http://127.0.0.1:8080/v1/tenants/acme/keys
# {"id":"…","key":"vlv_…","created":"…"}: the key is shown this once
```

Keys are listed with `GET` and revoked with `DELETE …/keys/{id}`; a revoked
key stops working at once on the node that revoked it and within ten seconds
on the others. Dropping a tenant takes its keys with it, on the same terms: a
tenant made again under the same id within those ten seconds may still be
reached by an old key through another node. A key on an admin route is
answered 403.

A node lets go of a tenant nobody has used for `VELLEV_TENANT_IDLE_SECS`
(15 minutes), flushing it first, so any node can take it. `VELLEV_MEMORY_MB`
bounds what the tenants a node holds may take in memory together; past it
the least recently used are let go. `VELLEV_TENANT_QUERIES_PER_SECOND` and
`VELLEV_TENANT_WRITES_PER_SECOND` (objects) bound what one tenant may ask of
each node, not of the cluster; a tenant asking more is answered 429 with
`Retry-After`, and the client waits as long and asks the same node again.

A collection with an `access_rule` is read with a tenant's key only on
behalf of a person: the caller sends an identity token in
`x-vellev-identity`, a JWT signed with Ed25519 under a key id, for the
audience `vellev`, naming the tenant, the person's `identity_id` and
`teams`, and `exp`. Every read is then narrowed to the rows the person owns
or that name one of their teams, whatever filter the caller adds; with no
token nothing of the collection is read. Admin keys read everything. Writes
are not narrowed: a tenant's key is trusted with every write, as an indexer
must be, and a write's answer says something of the rows it touches. A node trusts the public keys in
`VELLEV_IDENTITY_KEYS` (`KID=X`, X the key's 32 bytes in unpadded base64url),
so a key is rotated by trusting the new one beside the old.

`vellev serve --help` lists every setting. On SIGTERM a node reports not
ready, keeps serving for `VELLEV_STOP_DELAY_SECS` (a few seconds on Docker
Swarm; none is needed on ECS), finishes what is in flight within
`VELLEV_DRAIN_SECS`, and lets go of its tenants; keep both under the
platform's grace period. A node answers `/health/live` as
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
have piled up or one is half deleted rows. Files nothing needs any more are
deleted once no restore within the tenant's window can need them, and
`VELLEV_GRACE_SECS` (an hour) later.

A tenant can be put back as it stood at any moment of its restore window:
14 days unless `VELLEV_RESTORE_WINDOW_SECS` or the tenant says otherwise.
Every commit is a restore point, kept in Postgres. A restore is a commit of
its own, so nothing is lost by it and it can be undone by restoring to just
before it; requests answered before it stay answered. With an admin key:

```sh
call "http://127.0.0.1:8080/v1/tenants/acme/restore-points"            # the window, earliest and latest
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"at":"2026-10-07T08:00:00Z"}'
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"position":42}'
call -X PUT "http://127.0.0.1:8080/v1/tenants/acme/restore-window" -d '{"secs":2592000}'   # null for the default
```

Several nodes may run against the same Postgres and store. Each tenant is
written by one node at a time; a write that reaches another node is sent
on to the holder, so set `VELLEV_ADVERTISE_URL` to each node's own address
(never a load balancer's) wherever the default of this host's name and port
would not reach it.

Reads are answered by whichever node they reach, from a copy of the tenant
it follows. A write answers with `incarnation` and `position`; send them
back on a read as `x-vellev-after: incarnation.position` and the node
catches up that far before it answers (the Rust client does this for you).
`x-vellev-consistency: strong` waits for everything committed, and
`eventual` answers with whatever the node has, at most about a second old.

The API is described at `/openapi.json`, and by `vellev openapi` without a
running node. Every route under `/v1` needs one of the configured keys. A
collection may only name an embedding endpoint listed in
`VELLEV_EMBEDDING_URLS` or given a key in `VELLEV_EMBEDDING_KEYS`.

## Vector search

A layer with up to a few thousand vectors is scored exactly. Above that a
query is shortlisted by one-bit codes and the shortlist rescored with the
full vectors. A segment holding more than 200k vectors of a collection,
which only a merge produces, gets a clustered index instead: k-means
clusters of RaBitQ-style codes, scanning as few clusters as hold 97% of the
true nearest ten of sample queries, built when the segment is written and
kept in a file beside it. A filter is applied during the scan, so results
are exact with respect to it.

A segment's vectors live in a column file of their own, read in place
from the node's local copy of the store (`VELLEV_CACHE_DIR`, set in the
image) or from a `file://` store's own files, so memory goes on codes and
properties rather than on the vectors themselves; a node with neither holds
every vector in memory.

```sh
cargo run --release -p vellev-index --example clustered_bench -- 200000 768   # the index alone
OBJECTS=1000000 cargo run --release -p vellev-storage --example vector_scale  # commit, flush, merge, search
```

## Upgrades

Two releases run side by side during a rolling deploy and after a
rollback. What a release may change to keep that working, for files,
Postgres, the API and between nodes, is in
[docs/compatibility.md](docs/compatibility.md).

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
