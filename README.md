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
before it; requests answered before it stay answered. A restore named by
`x-vellev-request`, as the client crate names every one, is made once: asked
for again within a day it is answered as it was, so a retry never goes back
past what was written since; asked for again later, it restores again, and
asking for another state under the same name is refused (422). A restore
that fails, or is cut short by its node letting the tenant go, once its
commit was begun is answered as made if it was, and otherwise as of unknown
outcome: ask for it again by the same name. Another restore of
the tenant is refused while one runs. A node of an older release
does not know the name, so upgrade every node before restoring (see
[docs/compatibility.md](docs/compatibility.md)). With an admin key:

```sh
call "http://127.0.0.1:8080/v1/tenants/acme/restore-points"            # the window, earliest and latest
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"at":"2026-10-07T08:00:00Z"}'
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"position":42}'
call -X PUT "http://127.0.0.1:8080/v1/tenants/acme/restore-window" -d '{"secs":2592000}'   # null for the default
```

With `VELLEV_BACKUP_STORE` set, each tenant is also backed up once a day
(`VELLEV_BACKUP_EVERY_SECS`), whether or not any node holds it, unless the
node is told not to look for backups to take of its own accord
(`VELLEV_BACKUPS_SCHEDULED=false`, for a cluster whose backups some nodes
take and others do not; such a node still backs up when asked and purges
removed tenants), and backups
are kept 35 days (`VELLEV_BACKUPS_KEPT_SECS`), the newest always. One node
at a time backs a tenant up or restores it, by a claim it renews while it
works; `VELLEV_BACKUP_CLAIM_SECS` (fifteen minutes, at least three times
what one request to the store may take with its tries and half a minute
more for the store to apply one given up: 10 minutes 51 seconds as a node
is set up by default) is how long a node that died
holding one keeps others from the tenant's backups. A backup
copies only the files no earlier one has, so keeping many costs little more
than keeping one. Everything of a tenant lies under its own prefix of the
one bucket, `{tenant}/`.

A tenant's backups are its own. Each tenant is given a lineage when it is
first made, an id it keeps for as long as it lives, and each cluster has an
id of its own. A backup says both, and lies under them in the backup store:
its record under `_backups/{tenant}/records/{lineage}/{cluster}/`, its files
under `_backups/{tenant}/data/{lineage}/{cluster}/`. A tenant made later
under the same id is another tenant, of another lineage: it neither lists
nor restores from the backups of the one before, and is given none of its
keys. A tenant is put back from one of its own backups, keeping the keys it
has now, so that a key revoked since stays revoked:

```sh
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/backups"              # back up now
call "http://127.0.0.1:8080/v1/tenants/acme/backups"                      # list them
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"backup":"<id>"}'
```

A cluster that has lost a tenant takes it up again from a backup only when
told to adopt it. Listing the backups of a tenant the cluster does not
have, or has and has never written to, lists every backup kept under that
id, each with the lineage it is of and the cluster that took it. Adopting
one makes the tenant, if there is none, and restores it; with the commit
that restores it, and not before, the tenant becomes the tenant the backup
is of, with its lineage and its restore window. A restore that fails leaves
the tenant as it was, with nothing of the backup: ask again, or adopt
another. The tenant is given no key unless `with_keys` asks for those the
backup kept, which are the keys it had when the backup was taken: a key
revoked since then reaches the tenant again.

```sh
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"adopt":{"backup":"<id>"}}'
call -X POST "http://127.0.0.1:8080/v1/tenants/acme/restore" -d '{"adopt":{"backup":"<id>","with_keys":true}}'
```

Only a tenant that has never been written to and has no key can adopt a
backup. One that has been written to is some other tenant of that id; a key
issued to an empty tenant was issued for whatever was to be put in it,
which may not be this, and is to be revoked first. Both are refused as a
conflict. Once it has adopted a backup the tenant is restored from its own
backups as any other.

A backup is listed for as long as its record is there, and nothing looks
whether its files still are until it is restored from. A backup that has
lost a file from the backup store stays listed as one to restore from, and
fails only then: the restore is refused as a `conflict` (409) whose message
names the backup and the file it lacks, and the command line prints it. A
cluster that still has the file in the tenant's own store restores from it
all the same. The next backup the same cluster takes copies back any file it
lists that went, so a backup that lists the lost file is mended by it. An
older backup whose lost file the tenant no longer lists is not mended by
anything: it fails at restore for as long as it is kept.

The same from the command line, with an admin key in `VELLEV_API_KEY`:

```sh
vellev backup --tenant acme take
vellev backup --tenant acme list
vellev backup --tenant acme points
vellev backup --tenant acme restore --at 2026-10-07T08:00:00Z   # or --position N, or --backup ID
vellev backup --tenant acme restore --adopt ID                  # with --with-keys, the keys the backup kept
vellev backup --tenant acme window 2592000                      # no number for the default
```

The backup store is trusted: a backup lists key hashes and files, and
whoever can write to the store can shape what a restore brings back.

A cluster writes, prunes and purges only the backups it took itself, under
its own id. So one backup store may serve several clusters, and a cluster
may be given another's to adopt tenants from, as a new cluster is after the
old one is lost, or one set up to rehearse that: whatever it then does with
a tenant it adopted, removing it included, deletes nothing the other
cluster took. From there on it backs the tenant up under the tenant's
lineage and its own id, copying every file once, and the tenant is restored
from what either cluster took of it. To bring tenants back in a new
cluster, start it with `VELLEV_BACKUP_STORE` set to the old one's backup
store, or to a copy of it, and for each tenant list its backups and adopt
the one wanted.

Removing a tenant removes everything the cluster keeps of it. Its keys and
restore points go at once; `VELLEV_PURGE_AFTER_SECS` (ten minutes, at least
two leases) later, once any node that held it has stopped committing, one
node deletes its files and the backups this cluster took of it, forgets its
history in Postgres a batch at a time, and every node drops its cached
copies. With a backup store the purge waits at least
`VELLEV_BACKUP_CLAIM_SECS` (fifteen minutes), so that no backup of the
tenant still holds its claim. The purge is done, and a tenant may be made
again under the same id, only once a pass over its files and backups finds
nothing left; until then making one is refused with the code `purging` and
a `Retry-After` of the least that may be, a lower bound and not when the
purge will be done. One node at a time purges a
tenant, by a claim it renews before every listing and every batch of
deletes, and which is renewed only while the purge is still that node's to
do and no tenant of the id has been made: a node that stalls in the middle
of a purge deletes nothing of a tenant made since. A purge that fails once
it has sent a delete is taken up again only when its claim has run out,
`VELLEV_BACKUP_CLAIM_SECS` later at most, and not at once: a delete it gave
up on may yet land, and must not on a tenant made since, least of all one
taken up again from a backup, whose files have the names they had.

A node that stalled may still land a file after that: a writer's upload,
or a backup's copy. So for a day after a purge the tenant is looked over
again every fifteen minutes, by one node at a time. Whatever is found among
the backups this cluster took of its lineage is deleted, whether or not a
tenant of the id has been made since. Whatever is found among its files in
the store is deleted only while there is no tenant of the id, by purging it
as before, and no tenant of the id can be made meanwhile. What is never
deleted is a file that lands more than a day after the purge, or one that
lands among the tenant's files in the store once a tenant of the id has
been made again, which lies there until that tenant's cleaning up takes it;
neither can be reached, a backup left so being of a lineage no tenant has.
A backup finishing after the removal forgets itself.

Removal is final. Nothing is kept in the cluster from which a removed
tenant could be restored: the backups it took go with it, and a tenant made
under its id afterwards is not given them even if a copy turns up. A tenant
removed here can be brought back only from backups this cluster did not
take, or from a copy made elsewhere beforehand, by adopting one. For a day
after the purge the cluster deletes whatever appears under the removed
tenant's lineage and its own id in the backup store, so a copy of its own
backups is adopted in another cluster, or put back only once that day has
passed.
A node gives up on whatever stops answering, so that a tenant held up by a
dead connection passes to another node rather than waiting with it:

- Each request to Postgres, and each wait for a connection to it, is given
  up after `VELLEV_DATABASE_TIMEOUT_SECS` (10). A request given up leaves its
  connection to be checked before it is used again, which on a connection
  that went silent lasts until TCP gives up on it. Bound that on the server
  and in the kernel too: set `statement_timeout`, `tcp_keepalives_idle`,
  `tcp_keepalives_interval` and `tcp_keepalives_count` for the role
  (`ALTER ROLE vellev SET statement_timeout = '10s'`, and so on) or in
  `postgresql.conf`, and keep the OS's TCP keepalive and retransmission
  limits short on the nodes. sqlx sets no keepalive of its own.
- A file larger than `VELLEV_UPLOAD_PART_MIB` (16, from 5 to 5120) goes up
  in parts of that size, each a request of its own, and comes down a range
  of that size at a time. So a large segment is never one request and
  never past S3's 5 GiB limit on one. A file that needs more than 10,000
  parts (156 GiB at 16 MiB) is refused before any of it goes up.
- Each try of a request to the store is given up after
  `VELLEV_STORE_TIMEOUT_SECS` (30) and as long again as a part takes at
  `VELLEV_STORE_MIN_BYTES_PER_SEC` (a mebibyte a second, about 8 Mbit/s):
  46 seconds by default. A link that carries a part slower than that is
  not supported: lower the one or raise the other for it, or its uploads
  and reads are given up. A request that fails in a way worth trying again
  is tried up to three times more, under a second apart, and after a
  timeout only if it is safe to repeat; the request is given up once all
  four tries and the waits could have run out (three minutes and seven
  seconds by default). A range of a file coming down that is given up is
  asked for once more, from the bytes that had come.
- An upload that fails, or whose caller stops waiting, is aborted. One cut
  off too abruptly for that, as by a crash, leaves parts in the bucket:
  give it a lifecycle rule that aborts incomplete multipart uploads after a
  day (`AbortIncompleteMultipartUpload`).
- Each step on the cache directory is waited for ten seconds and as long
  again as its bytes take at 20 MiB/s, after which the node carries on
  without the copy; the steps run on at most 64 threads of their own. With
  all of them busy a step waits as long again for one, so a disk that is
  busy and answering serves everything asked of it; with all of them kept
  by steps a disk never answered, the cache is passed over at once.
  Processes and containers on one host may share a cache directory. Files
  left part written there an hour ago or more are deleted as a node
  starts.
- Vectors are the exception to carrying on without the copy, since they
  may be more than memory holds: a vectors file not yet in the cache is
  fetched into it a part at a time and read there, never held whole in
  memory. Reading one through to check it is done `VELLEV_CACHE_LOADS`
  files at a time on a disk (two), the rest waiting their turn for as long
  as those that have one get on, however long that is: a tenant of many
  large files, or many tenants opened at once, all open. A file is given
  up when the disk has said nothing of it for ten seconds, or is slower
  all told than its share of 5 MiB a second, which with two read at once
  is 2.5 MiB a second each, for as long as they are read together and
  no longer; and those waiting are given up when nothing has moved for
  longer than one part of a file may take to come from the store and be
  written, which follows `VELLEV_STORE_TIMEOUT_SECS` and is about ten
  minutes as a node is set up by default. The tenant then fails to
  open, as
  unavailable, and opens when asked again once the disk answers. A file
  checked once is not checked again by that node while it lies as it did,
  so opening again goes on from where it failed, and opening a tenant a
  second time reads nothing through.
  A node whose cache directory cannot be written does not open tenants
  that have vectors.
- A flush's, a merge's and a restore's segments are built and go up before
  writes wait on them; a failure there commits nothing, leaves the writer
  as it was and deletes what went up. Writes wait only while a commit is
  made: its head file goes up, the commit is recorded, each try given up
  after a quarter of the lease, and it is adopted in memory.
- Writes that arrive together share a commit: up to 64 requests
  of them, and no more than 8,192 mutations or 12 MiB, so that the log
  file a commit writes goes up in one request and is applied in memory in
  a fraction of a second, which is as long as reads wait on it. A request
  larger than that on its own is committed alone.
- Cleaning up deletes at most a thousand files at a time, sixteen at once,
  and for no longer than a minute, after a minute at most of finding them,
  between which flushes and merges go on.
- One request deletes at most 8,192 objects by filter. A delete with no
  `limit` deletes every object that passes, and is refused as invalid,
  deleting nothing, if more than 8,192 do. A delete with a `limit` of up
  to 8,192 deletes that many of them and says how many it did; fewer than
  the limit means none were left. The deletes by filter of one request
  count together, one with no limit for 8,192: a request whose deletes may
  come to more is refused. To delete however many pass, send the delete
  with a limit until it deletes fewer than the limit.
  **Each time must be a new request, with a new `x-vellev-request`**: one
  sent again under the name it had is answered as it was and deletes
  nothing more. Every one is a commit of its own, so stopping part way
  loses nothing and going on is only sending it again; what is deleted in
  the end is whatever passes until none does, objects written meanwhile
  included, not what passed when the first was sent. A delete counts
  against the size of the commit it shares with other writes as its limit
  does, so a small one shares a commit and one of the full 8,192 has a
  commit to itself; each object deleted counts as a write against the
  tenant's rate.
- A node answers `VELLEV_READS_AT_ONCE` reads at once (256), so that
  reads stuck on a disk cannot take every thread a commit needs. One
  tenant may have every turn while no other asks. While a read waits
  whose tenant has less than half of them, no tenant that has half or
  more is given another, so turns that come free go to tenants with
  fewer; with every waiting tenant at half or more, a free turn goes to
  whichever asks first. A read that finds no turn waits
  `VELLEV_READS_WAIT_MS` for one (500) and is then turned away as
  unavailable (503) with `Retry-After: 1`, which the Rust client waits
  out before it asks again. The turn is for the reading alone, and is
  taken last: a search whose text is being embedded, or whose tenant is
  being opened, has none yet and keeps none from any other read. The
  reads that take a turn are the searches (`nearest`, `keyword`,
  `hybrid`), listing objects and counting them, which are answered off
  the request's own thread; getting one object, a collection or the
  collections takes none.
- Every read counts against its tenant's rate, those that take no turn
  as well, with one exception: a read turned away for want of a turn is
  not counted if it had nothing costly done on the way. It stays counted
  if it had the coordinator asked (a `strong` read, or one after a
  position the node had not reached), a copy of its tenant opened, or
  its text sent to a model. What it only waited for another read to have
  done is that read's: of several that come for a tenant not yet open,
  or with the same text, the one whose opening or embedding it is is
  counted, and the rest are not. A text of up to 1,024 bytes embedded
  lately is not sent again, its vector being kept, and a model with a
  fault that stands is not asked at all. So a search by a new text that
  is turned away is counted, since it was embedded; turned away again
  with the same text, it is not counted again, so long as the text is of
  no more than 1,024 bytes, was embedded, and its vector is still kept; and answered at last, it
  is counted for the answer, twice in all. A search waits no longer than
  two minutes for its text's vector, whatever makes it.
- A commit does up to three steps on a thread of its own. One that finds
  no thread free for a step is given a quarter of its lease to get one,
  each step, before the wait counts against its writer; a writer held for
  longer than everything it waits on may take counts as stuck, and gives
  the tenant up.
- A commit names the position and head it follows, and Postgres records
  it only on those, so of two writers of one tenant only the one that has
  seen everything committed can commit, whatever their leases say.
- A writer whose commit failed, or was given up, stops renewing its lease,
  as does one that holds writes up for longer than every bound on what it
  waits for; whatever was working on the tenant is stopped, and a write
  caught under way is answered as of unknown outcome.

Reads are answered by whichever node they reach, from a copy of the tenant
it follows. A write answers with `incarnation` and `position`; send them
back on a read as `x-vellev-after: incarnation.position` and the node
catches up that far before it answers (the Rust client does this for you).
`x-vellev-consistency: strong` waits for everything committed, and
`eventual` answers with whatever the node has, at most about a second old.

Every object read carries a `version`, and a `PUT`, `PATCH` or `DELETE` of
an object can be made `only_if_version` it is still at: in the body of a
`PUT` or `PATCH`, on an `upsert`, `patch` or `delete` in a batch, and as
`?only_if_version=` on a `DELETE`. A node of 0.1 ignores that query and
deletes, so send a conditional delete in a batch until every node is 0.2.
Two writers that read the same object cannot each overwrite the other's
change unknowing, as the one that comes second is refused with `conflict`
and the version the object has now, and reads again. The
version is of the object's content, properties and vector, so an object
changed and changed back is at its old version again, and a write
conditional on it then finds exactly the content it read. It is opaque:
compare versions for equality and hand them back as given; never order,
parse or make them. A conditional write of an object that is not there is
`not_found`. Pages of a collection (`/list`) come in id order at a cost of
their own size, so a walk of a whole collection costs its size once.

The API is described at `/openapi.json`, and by `vellev openapi` without a
running node. Every route under `/v1` needs one of the configured keys. A
collection may only name an embedding endpoint listed in
`VELLEV_EMBEDDING_URLS` or given a key in `VELLEV_EMBEDDING_KEYS`.

`GET /v1/limits` tells any key the most a request may ask for: results of a
search or a page (`VELLEV_MAX_LIMIT`), candidates each side of a hybrid
search offers, `oversample`, bytes of a body (`VELLEV_MAX_BODY_BYTES`),
seconds of a restore window, objects one request deletes by filter, and
reads the node answers at once (`VELLEV_READS_AT_ONCE`). A request that
asks for more of one of them than it may is refused with
`limit: {name, max}`, the name being the field here; the reads are not
any one request's to ask too many of, and a node answering as many says
`unavailable`. A node of a release before this answers that
it has no such route (404, or 403 to a tenant's key), for which the Rust
client's `limits()` gives `None`.

```sh
call http://127.0.0.1:8080/v1/limits
# {"results":1000,"candidates":1000,"oversample":1024,"body_bytes":33554432,"restore_window_secs":316224000,"deleted":8192,"reads_at_once":256}
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
`Retry-After` is waited on that long, up to a minute: the same node for a
tenant asking too much (429), and for a node too busy to answer a read
(503) after any other address has been asked first.

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
properties rather than on the vectors themselves; only a node with neither,
which is no way to run one, holds every vector in memory.

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
