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

With `VELLEV_BACKUP_STORE` set, each tenant is also backed up once a day
(`VELLEV_BACKUP_EVERY_SECS`), whether or not any node holds it, and backups
are kept 35 days (`VELLEV_BACKUPS_KEPT_SECS`), the newest always. A backup
copies only the files no earlier one has, so keeping many costs little more
than keeping one. Everything of a tenant lies under its own prefix of the
one bucket, `{tenant}/`. A tenant is put back from a backup in its own
cluster, keeping the keys it has now, or in a new one that lost it, which
gets back its keys and restore window too:

```sh
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/backups"              # back up now
call "http://127.0.0.1:8080/v1/tenants/acme/backups"                      # list them
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"backup":"<id>"}'
```

The same from the command line, with an admin key in `VELLEV_API_KEY`:

```sh
vellev backup --tenant acme take
vellev backup --tenant acme list
vellev backup --tenant acme points
vellev backup --tenant acme restore --at 2026-10-07T08:00:00Z   # or --position N, or --backup ID
vellev backup --tenant acme window 2592000                      # no number for the default
```

The backup store is trusted: a backup lists key hashes and files, and
whoever can write to the store can shape what a restore brings back. A
restore into a cluster that lost a tenant brings back every key the backup
held, including any revoked since.

Removing a tenant removes everything kept of it. Its restore points go at
once; `VELLEV_PURGE_AFTER_SECS` (ten minutes, at least two leases) later,
once any node that held it has stopped committing, one node deletes its
files and its backups, and every node drops its cached copies. A file a
stopped writer was still uploading may outlive the purge, but none can
belong to a tenant made later under the same id, which is refused until the
purge is done. A backup finishing after the removal discards itself. Removal
is final: a removed tenant cannot be restored.

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

`GET /v1/limits` tells any key the most a request may ask for: results of a
search or a page (`VELLEV_MAX_LIMIT`), candidates each side of a hybrid
search offers, `oversample`, bytes of a body (`VELLEV_MAX_BODY_BYTES`) and
seconds of a restore window. A node of a release before this answers that
it has no such route (404, or 403 to a tenant's key), for which the Rust
client's `limits()` gives `None`.

```sh
call http://127.0.0.1:8080/v1/limits
# {"results":1000,"candidates":1000,"oversample":1024,"body_bytes":33554432,"restore_window_secs":316224000}
```

Every failure is `{"error":{"code":…,"message":…}}`, and the code, not the
message, says what to do. A request for more than the node allows says
which limit and its most, as those names; one that asks for none at all,
`limit: 0`, is refused without it. A failure to find something says what
is missing: `tenant`, `collection`, `object`, `key`, `backup`,
`restore_point` or `route`. A tenant's key is told `tenant` of every other
tenant. A write refused and then named again is refused as it was, with
what was missing then, unless a node of a release before this kept the
refusal meanwhile, which keeps only that something was.

```json
{"error":{"code":"invalid","message":"limit must be from 1 to 1000, not 5000","limit":{"name":"results","max":1000}}}
{"error":{"code":"not_found","message":"collection notes does not exist","missing":"collection"}}
```

When a collection's text cannot be embedded, the answer is one of two, and
in both nothing was done.

`unavailable` (503) means it may work later, on this node or another: send
the request again. Everything that might pass is answered so, including
much that will not pass by itself, because a node cannot tell it from what
will: an endpoint or Bedrock refusing the node's key or credentials with
no more than a status (a firewall or a proxy answers 401, 403 and 404 too,
and a permission just granted takes a while to hold), a refusal worded
only in prose or in codes other than OpenAI's, which are not read, an
account with nothing left to spend (`insufficient_quota`, which paying
puts right), a model's input refused, which may be that text's alone, a
node with no AWS credentials, and a node with no network to fetch a local
model over. The cause is in the node's log each time; if it goes on, look
there.

`model_unavailable` (501) means the node that answered cannot use the
model as it is set up, and will not until someone puts that right. It is
said only of what the node's own build and setup settle, or what serves
the model says outright in its own code for it, and `reason` says which:

- `build`: the node's release does not run the model: the collection was
  made by a release that knew a model this one does not;
- `setup`: the node may not write the directory local models are kept in,
  which is `HF_HOME` if that is set and `VELLEV_MODEL_DIR` otherwise; or
  an OpenAI endpoint answers `invalid_api_key`;
- `provider`: an OpenAI endpoint answers `model_not_found`, Bedrock that
  the region has no such model, or the model's vectors are not the
  collection's size.

```json
{"error":{"code":"model_unavailable","message":"this node cannot use the collection's embedding model as it is set up","reason":"setup"}}
```

It never means the request is at fault or may be dropped. Keep it, and
send it again once the node is put right; one that brings its own vector
needs no model. Three things are promised of a node that meets such a
fault, and no more:

- It logs the fault at most once in 20 seconds for each way of embedding
  and each `reason`, however many requests meet it, at once or one after
  another, hybrid searches that fall back among them, and whatever the
  model does meanwhile.
- A model that refuses every request is not asked again within 20 seconds
  of the first refusal; so a setup put right takes effect within that.
- A model that answers some requests and refuses others goes on being
  asked: an answer lifts the fault at once.

No request waits on another to find out whether a model can be used, so
the requests already on their way to the model when a fault first shows,
or that set out before the first of them hears of it, each reach the
model once and are each refused.

What a model's server says is logged, and passed on, without what is
shaped like a key in it. In what an HTTP endpoint answers, keys are
looked for in the first four kilobytes, before it is cut to the 300
characters that are kept. Nothing is decoded to find one.

A key known by how it begins is taken out anywhere but straight after a
letter, unless that letter ends one of the escapes `\n`, `\r`, `\t`,
`\uXXXX` and `%XX`:

- a JSON web token: `eyJ`, a full stop and `eyJ` again, with more after
  each, signed or not;
- a key beginning `sk-`, with 40 characters or more after that; or 20 or
  more with a letter and a digit among them; or 20 or more after
  `sk-proj-` or `sk-ant-`; or with some of it masked by `…`, `...` or
  asterisks and two characters or more left;
- a Hugging Face token, `hf_` and 30 letters and digits or more, or 20 or
  more with some of each, or masked likewise (three full stops at its end
  do not count);
- a Google API key, `AIza` and 35 characters, unless a letter or a digit
  comes straight after;
- an AWS access key id, `AKIA` or `ASIA` and 16 capitals and digits,
  unless a capital or a digit comes straight after.

Two more are known by what comes before them:

- what follows `Bearer` or `Basic`, if it is 16 characters or more and
  not only lower-case words joined by hyphens, underscores or full stops;
- the whole value of `key`, `token`, `access_token`, `api_key`, `api-key`
  or `apikey` in a query string, a key or not, up to the next `&`, space
  or quote.

Deliberately left: a key glued to the end of a word, since `task-…` and
`shf_…` are words and no provider writes a key so; a credential after
`Bearer` or `Basic` that is all lower-case letters; a key with no shape
of its own, such as an AWS secret or an Azure key; and anything written
with escapes inside it.

The 20 seconds cut the other way too: something in front of an OpenAI
endpoint that answers `model_not_found` or `invalid_api_key` for a moment,
in OpenAI's own form, costs 20 seconds of `model_unavailable`. Nothing
reloads a node's settings while it runs, so most faults of setup are put
right by starting the node again. Meanwhile
another node may be able to answer, and which node embeds depends on the
request:

- A search is embedded for by the node it reaches. Another node, set up
  right, answers it.
- A write is embedded for by the one node that holds its tenant, whichever
  node it reaches. If that node cannot use the model, every node answers
  every such write `model_unavailable` until it is put right, or is
  stopped and so lets the tenant go to another node. No node sends a
  request on to one that can embed it.

So the Rust client does not treat the two alike. A write answered
`model_unavailable` fails at once: no other address would say otherwise.
A search is asked of every other address the client knows first, at once,
and then tried again as it is after `unavailable`, with the same tries and
waits, failing with `model_unavailable` only when they are spent. A client
given one address, a load balancer's or a service's, is so sent to another
node by whatever is behind that address, and needs no more than that; the
nodes' own addresses only find a node that can sooner.

A hybrid search that cannot embed its text ranks by the words alone
instead, and says `degraded`, with why in `degraded_by`: `unavailable` or
`model_unavailable`, as it would have been refused. A Rust client of one
address gives that answer as it comes. One of several, told
`model_unavailable`, first asks each address it has not yet asked, once and
at once, and gives the first answer ranked in full, or failing one the
answer by the words. Each is given `Options::better_within` to answer, two
seconds unless set otherwise: a node that takes longer to embed a text
does not get to better the answer, and with none no address has time to.
The
address that answered in full is the one asked first from then on.

A collection can only be made with a model this release runs: for a local
model, one of those fastembed lists, at its own size; on Bedrock, Titan
text embeddings v2 at 256, 512 or 1024 dimensions. Anything else is the
request's fault, `invalid`, as is whatever else is wrong with the schema,
which is said first; a release that knows more models may be running on
another node during a rolling deploy, and answers for itself. Making again
a collection that is there exactly so changes nothing, as ever, and one
that is there otherwise is a `conflict`, whatever the model. What only
using a model shows, such as an endpoint that refuses the node, is found at
the first write.

A request is sent at most as many times as the client has tries
(`Options::attempts`) and one less than it has addresses, each for no
longer than its timeout, with a wait before any address is asked again
that doubles up to five seconds; a node that turns a request away with
`Retry-After` is waited on that long, up to a minute.

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

## Licence

The client crates, `vellev-core`, `vellev-index`, `vellev-api` and
`vellev-client`, are MIT OR Apache-2.0: use them anywhere. The server,
`vellev`, `vellev-node`, `vellev-storage` and `vellev-embed`, is under the
[Functional Source License 1.1](LICENSE.md) with Apache-2.0 as its future
licence: anyone may use, change and run it, for themselves or for their
customers, but not offer it as a service that competes with vellev's own;
two years after each release, that release is Apache-2.0.

## Releases

The client crates, `vellev-core`, `vellev-index`, `vellev-api` and
`vellev-client`, are published to crates.io under MIT OR Apache-2.0 by
CI: bump the workspace version, then push a tag `v<version>`. The release
runs every CI gate first, refuses to start if any of the crates' versions is
on crates.io from another commit, and publishes them in dependency order. It
needs a `CARGO_REGISTRY_TOKEN` secret allowed to publish new crates. The
server's crates are not published. What each release changes for whoever
uses the crates is in [docs/releases.md](docs/releases.md).

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

Once every check has passed on main, CI pushes the image it built and
started to Docker Hub as `kingsleyh/vellev:sha-<commit>` (linux/amd64). A
release tag names that same image by its version as well, for example
`kingsleyh/vellev:0.2.0`. Pin a deployment to the `sha-` name or a version;
there is no `latest`.
