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

Copies every class (or each `--class`) into a collection of the same name, with ids and vectors, keeping progress in `vellev-import-TENANT.json` so a run that stops resumes. A class vellev should embed itself instead takes `--embedding CLASS=embedding.json`. `--created-as` and `--updated-as` keep Weaviate's object times as date properties, and must be the same on every run; `--per-second` limits the load on the source.

Writes made in Weaviate during the import are caught up at the end of
the run, and by every run after it, so an import can follow a Weaviate
that is still in use: run it again until writes move over, and once more
after. A class that has gained properties since is given them in vellev,
mapped as on a first import. One that no longer fits is refused, saying
why: a property of another type, vectors of another size or model, or
another distance. A property the class no longer has stays in the
collection, since vellev drops none and objects stored before still have
their values.

### What the import changes, and what it leaves

Beside each collection it keeps a ledger, a collection named
`vellev_imported_CLASS`: for each of Weaviate's ids it has accounted for,
when Weaviate says the object last changed, a hash of the object as the
import stored it, and a mark if the collection holds something else. The
rule is:

- An object is the import's while the collection holds, under an id the
  ledger has, exactly what the ledger's hash says. The import replaces
  and deletes only its own objects. One exception errs on the side of
  leaving: an object Weaviate deleted while the collection held
  something else under its id was left, and is not asked about again,
  so it stays left even if a writer then puts back exactly what the
  import had stored, for as long as Weaviate lacks the id.
- It stores a new object only where the collection holds nothing under
  the id and the ledger has no entry. What somebody deleted in vellev is
  not brought back.
- Everything else is left as it is, and the ledger marks it: an object
  changed or replaced in vellev since the import stored it, one deleted
  there, one written under Weaviate's id by something else. The ledger
  keeps the entry of an object it leaves for as long as the collection
  holds it, so the same is said of it on every run, and forgets it once
  neither Weaviate nor the collection holds the object. An object under an
  id Weaviate never had is never looked at, with one exception, below,
  for a collection taken up without a ledger. So a writer that has
  moved to vellev, or one that mirrors Weaviate into vellev under
  Weaviate's ids, is not undone by a later run. An object a mirror
  stores exactly as the import would is the import's too, from the run
  that reads the two and finds them the same: it then goes when
  Weaviate's does. One the ledger still marks as another writer's when
  Weaviate deletes it is left, whatever it holds.

In short: the import deletes or writes over nothing in vellev it cannot
show it put there, but for one narrow act an operator asks for in so
many words, once, of a collection that was there before its ledger.

The object and its ledger entry are written in one atomic step. The step
that replaces or deletes an object is conditional on the version the
object was at when the import read it, and a version is of everything an
object holds: every property, those the collection gained since the run
began included, and its vector. So the step fails if anything at all of
the object changed, or the object went, between that read and the step,
and nothing is asked of Weaviate in between. A step that fails so is
made again from a new reading, which finds the object another writer's
and leaves it; where several objects went in one request, the whole
request fails, nothing of it is made, and each of its objects is settled
again by itself from a new reading. No write that lands between the
import's read and its write is lost: not one to a property, not one to
the vector alone.

A writer that changes an object and changes it back around the import's
write is met as one that changed it: the step is refused, the object
reads as it did before, nothing explains the refusal, and the run stops
with the node's 409 rather than guess. Nothing was written; the next
run, with the writer quiet, settles the object.

This takes nodes that give objects versions, which those of 0.2 do. Read
from a node that gives none, the run stops before it replaces or deletes
anything.

### Deleting

Nothing is deleted while Weaviate is walked. What the ledger has and the
walk did not list is first only counted.

- **A run deletes at most `--max-deletes` objects of a class (1000) and
  at most `--max-delete-fraction` (0.05) of what was imported of it or
  of what the collection holds, whichever is fewer, and never the last
  object the collection holds.** A collection of up to twenty objects
  may so lose one in a run, and one of a single object none. A run that finds more gone deletes none of them, reports
  them as `pending` beside the `budget`, and fails: a Weaviate that was
  emptied, restored from a backup, or is not the one imported looks
  exactly like one that deleted a great deal. If they are truly gone,
  raise the two for one run; only `--max-delete-fraction 1` lets a run
  empty a class.
- `--no-delete` deletes nothing, and counts what is gone as `pending`.
  The run fails while anything is pending, as one over its bound does:
  what is gone from Weaviate and still in the collection is not settled,
  whoever chose to leave it.
- Within the bound, each object is asked for by itself twice,
  `--confirm-secs` apart (5), of every copy of the class
  (`consistency_level=ALL`). Before each asking, and again before the
  collection is read, Weaviate must answer that it is ready, every shard
  of the class must be loaded, and the class must be the one that was
  imported; and an answer that it holds no such object is believed only
  if it says it is ready straight after. A Weaviate that is starting
  answers that it holds no such object, or no such class, for those it
  holds. One it turns out to hold is settled like any other.
- A class is known by the names Weaviate gave its shards when it made
  it. One made again under the same name has others: the run stops
  before storing or deleting anything of it, and goes on only with
  `--accept-remade`, still within the bound. This does not tell a class
  restored from a backup, which keeps its shards' names, nor one emptied
  without being dropped: for those, as for a class with tenants, whose
  shards are named after its tenants, the bound is the guard.

On a Weaviate of several nodes, tried with two of 1.36.8: the shards
listed are those of every node, whichever node is asked; with a copy
down, an object asked for of every copy is answered with a failure,
never as not held, and Weaviate says it is not ready. A shard's status
does not say that one of its copies is down.

### Collections that had no ledger

A collection that was there before its ledger is taken up on the next
run: every object of Weaviate's is given its entry, by what the
collection holds under its id, a page of both at a time, the walk
resuming where it stopped. **That is every collection an earlier
release imported**, and any that something else made.

Without a ledger nothing tells an object Weaviate gained since from one
somebody deleted in vellev, nor one Weaviate changed from one somebody
changed in vellev, nor one Weaviate deleted from one somebody wrote to
vellev. Weaviate's times do not tell either: a batch that writes an
object again gives it a new time of creation. So, by default:

- What the collection holds exactly as Weaviate does becomes the
  import's.
- **What it lacks, or holds otherwise, is not copied.** It is marked in
  the ledger as `withheld`, with what the collection held, counted under
  that name on every run, and the run fails, saying so: a run never
  settles over objects it did not copy. For a collection an earlier
  release imported, that is every object Weaviate gained or changed
  since that import last caught up, and every object holding a number
  the earlier release read a digit off (see the release notes).
- **What it holds that Weaviate does not is not deleted.** On the first
  taking up of a collection nothing has ever kept a ledger of, each such
  object is noted in the ledger as `unknown`, with a hash of it as it
  was then, and counted under that name on every run; the run fails
  until somebody says whose they are. This noting is the one time the
  import looks at an object under an id Weaviate never had. It lasts as
  long as that taking up does, across runs if the taking up is stopped
  and begun again, so the answer `--sole-writer --delete-unknown` asks
  for covers all that time. No later run notes any, not even a taking
  up begun again because the ledger was lost: what the collection holds
  by then may have been written since.

Two imports of a tenant must not run at once, from one machine or two.
A state file keeps out a second run that shares it, and nothing keeps
out one with another. So every write over a ledger entry, and every
forgetting of one, is made only if the entry is still at the version it
was read at, and every first note of one only if there is still none: a
run that finds the ledger written under it stops, saying so, having
made nothing of the step that found it.

The import keeps one more collection in the tenant, `vellev_imports`,
which says of each class where it was imported from and how. If that
alone is lost and the ledger is still there, the next run takes the
collection up again beside the ledger: every entry stands as it is,
nothing is noted as unknown, and an object of Weaviate's without an
entry is withheld as above. The run records the Weaviate it is given as
the one imported, having nothing to hold it against, so give it the
right one. With the ledger and the note both lost, nothing
tells the collection from one never imported with a ledger: it is taken
up as one, what it holds that Weaviate does not is noted as unknown
again, objects written since the import among them, and none of that is
deleted unless a run is told both `--sole-writer` and
`--delete-unknown`.

**The ledger and the note are the import's memory, and are not to be
written by hand.** A ledger or a note lost whole is seen for what it
is, as above. One emptied or edited is not: the import believes what a
ledger that is there says, and a ledger with no entry for an id says
the id was never imported. So if entries are deleted from a ledger, an
object somebody deliberately deleted from vellev is stored again from
Weaviate by the next run, as one new to Weaviate, and an object another
writer changed keeps no mark of whose it is beyond what it holds.
Nothing remembers otherwise. To be rid of a ledger, end the import, or
drop the ledger whole.

The taking up over, an object Weaviate lists without an entry is new to
Weaviate, and is stored. Two things settle what was withheld or is
unknown, and the run fails until one of them is said:

- **`--sole-writer`** says nothing but an import has written to the
  collection, at any time since before it was taken up. Then what was
  withheld is stored or replaced, on that run or any later one. It
  deletes nothing by itself. Say it only if it is true: anything else
  written there is written over.
- **`--sole-writer --delete-unknown`** says that too, and asks for the
  unknown objects to be deleted: each is taken for one Weaviate deleted
  before there was a ledger, and is deleted as anything is, within the
  bound, after Weaviate has twice said it does not hold it, and only if
  it is still exactly as it was noted; one changed since is left,
  counted as `unknown`, and stays so until it is said to be another
  writer's. Objects stored in the collection after the taking up are
  never among them, whatever a run is told. An object another writer
  stored during the taking up is among them, which is why the flag is
  honest only if `--sole-writer` is. Once the last of them is deleted
  or settled, nothing is owed, and the two flags do nothing more.
- **`--other-writers`** says something else has written to the
  collection. It marks as another writer's exactly this: every object
  of Weaviate's the collection lacks or holds otherwise than Weaviate
  does, and every object it held at the taking up that Weaviate did
  not; whether a run found them before and withheld or noted them, or
  finds them now, as when the flag is given on the first run over the
  collection or to go on with a taking up that was stopped. What the
  collection holds exactly as Weaviate does is not marked, whatever is
  said: it is the import's, and goes on being replaced and deleted as
  Weaviate changes and deletes it. Like the other two it is said of
  every class of the run; name the classes it is true of with
  `--class`. What it marks is left alone for good, all of it in that
  one run, whatever Weaviate has deleted of it since and however much. The
  ledger then keeps no hash of any of it, so nothing a later run reads
  can make it the import's again: no run, whatever it is told, copies
  over it, deletes it, or stores it again if the other writer deletes
  it, not even when Weaviate comes to hold another object under the id.
  These entries stay for as long as the import does. It cannot be said
  with `--sole-writer` or `--delete-unknown`, and there is no taking it
  back but ending the import and importing into a collection made
  anew.

### Ending an import

```sh
vellev import finalise --to http://vellev:8080 --tenant acme
```

Once writes have moved to vellev and a last run has settled, end the
import (of each `--class`, or of every class). It drops the ledgers, and
with them what they take on the node, and notes in the tenant that the
collections are no longer imported, so that a later `import weaviate`
refuses them rather than change what is now vellev's own.

### What a run reports

Per class:

- `stored`, `changed`, `deleted`: the objects it added, replaced and
  removed.
- `kept`: the objects of Weaviate's the ledger marks as another
  writer's. The same on every run while they stay so.
- `withheld`: the objects of Weaviate's that were not copied into a
  collection taken up without a ledger. The same on every run until one
  with `--sole-writer` or `--other-writers`.
- `unknown`: the objects such a collection held, when it was taken up,
  that Weaviate did not. The same on every run until one with
  `--sole-writer --delete-unknown` or `--other-writers`.
- `left`: objects gone from Weaviate that stay in the collection because
  they were not as the import had stored them: another writer's, and no
  fault of the import's. Their entries are kept, so the same on every
  run, and Weaviate is not asked about them again. Once the other
  writer removes one from the collection, its entry is forgotten and it
  is counted no more.
- `barred`: ids said, with `--other-writers`, to be another writer's,
  which neither Weaviate nor the collection holds now. Their entries
  are kept so that nothing is stored under them again.
- `pending` and `budget`: see above.
- `refused`: objects vellev cannot hold, by id, with why. One stops the
  run unless `--skip-refused` is given. That passes over what the
  import finds unfit before sending it, and a new object a node
  refuses; it does not pass over a replacing or a deleting a node
  refuses as invalid, since those are conditional on a version and a
  node does not say whether the object or the condition is at fault:
  the run stops.
- `weaviate`, `imported`, `vellev`: how many objects Weaviate holds, the
  ledger has accounted for, those withheld aside, and the collection
  holds.
- `ended`, beside the classes: those passed over, in a run of every
  class, because their import was ended. Named with `--class`, such a
  class is refused.

The command exits with

- **0** when everything that is the import's is as Weaviate has it.
  Objects of another writer's do not count against it.
- **3** when Weaviate was written to behind the catch-up: run again.
- **1** when something is left that another run will not put right by
  itself: objects withheld or unknown, objects gone and not deleted,
  objects refused, a comparison that found differences or
  looked at nothing, or any failure.

`--strict` makes it 1 as well whenever a collection holds anything that
is not as the import left it.

The counts do not say that the collection holds what Weaviate does: a
run does not read again an object Weaviate has not changed. `--verify`
does. It changes nothing, and compares the two object by object, by the
same hash: over 20 pages of the ledger picked at random, or as many as
it is given, or over everything Weaviate holds with `--verify-all`. The
pages are the ledger's, so that ids that lie close together, as counted
or time-ordered ids do, are looked at as any others; picking them takes
listing the ledger once. It reports, under `verified`, how many it
`looked` at, how many are the `same`, how many of the import's are
`different` or `missing`, with some of their ids, and how many are
`kept` by another writer, `withheld`, `unaccounted` for, or `gone` from
Weaviate while the collection still holds them. It counts `withheld`,
`unknown`, `left`, `barred` and `kept` over the whole ledger as a run
that imports does, and fails on the same terms: on anything withheld,
unknown, different, missing or gone, and on looking at nothing of a
class that holds objects. Of every entry that says Weaviate no longer
holds its object, or that is of what the collection held when it was
taken up, it asks Weaviate whether it has come to hold the id; one
Weaviate holds is counted as the next run would settle it, by the run's
own rule: the import's if the collection holds exactly Weaviate's,
another writer's if it holds something else, withheld if it was there
at the taking up, and `different` only if it is still what the import
stored. Told the same and with nothing changed in between, a run, a
comparison straight after it and the next run say the same, which a
test holds for every state the ledger can be in; that is all that is
promised of them. With something changed, they may differ, and the
comparison is the more wary: where Weaviate has deleted an id since
the last run, it counts the object `gone`, and fails, though the next
run may find it another writer's, leave it, and settle. `--verify-all` also counts, as `extra`, the objects the
collection holds that the ledger has no entry for: another writer's,
under ids Weaviate never had, and no fault of the import's unless it is
told to be `--strict`. `--verify` of some pages cannot see them.

### What it costs

- The ledger holds one small object for each imported object, with no
  vector, nothing indexed and nothing searched. Measured over 60,000
  entries, that is 135 bytes each in the tenant's log and 106 once in a
  segment and its manifest, and about 447 bytes each in the node's
  memory while the tenant is held: for a class of 4.2 million objects,
  some 0.45 GB of files and 1.9 GB of memory. `import finalise` frees
  both.
- It is ordinary tenant data: backed up and restored with the tenant,
  dropped with it, and listed among its collections, as is
  `vellev_imports`, which holds one object per class saying where it
  came from.
- A catch-up lists the whole ledger, a page at a time. Until the node
  lists a collection without sorting all of it for each page, that
  listing grows with the square of the ledger; `--page` is therefore as
  large as the node allows unless given.
- A catch-up asks Weaviate for every id, without vectors, and fetches
  what changed: a page at a time where much of a page changed, one
  object at a time where little did. With the default `--skew-secs` of
  300 it also fetches again whatever changed in the five minutes before
  the last catch-up began, and stores none of it that is unchanged.
- A comparison asks Weaviate for each entry that says Weaviate no
  longer holds its object, and each of what the collection held when it
  was taken up that Weaviate did not: one request each, one after
  another, whether it compares everything or some pages, since it counts
  those entries over the whole ledger. For each such id Weaviate holds,
  it reads the collection's object too: one more request. The rest of
  the entries cost it nothing but the listing of the ledger.
- Each run asks the collection whether it still holds every object the
  ledger says was left or is unknown: one at a time where a page of the
  ledger has few, and by listing the ids between them where it has
  many. A collection with a great many of another writer's objects that
  Weaviate deleted pays for that on every run.
- A request to the node carries at most `--batch-bytes` (8 MiB) and at
  most 8,192 changes, what a node commits together with other writers'
  requests, an object and its entry always in the same one.
- A request Weaviate fails or turns away is asked again for
  `--patience-secs` (600) before the run stops.

A collection is caught up only from the Weaviate it was imported from,
known by its address. Where the address changes between runs, as through
a tunnel, name the Weaviate with `--from-name` on every run. A
collection that is dropped is imported again from the start, its ledger
with it; drop one only if nothing but the import has written to it,
since whatever else did is gone with it. Two runs cannot share a state
file.

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
`VELLEV_GRACE_SECS` (an hour) later. A node does not start with a grace
below fifteen minutes, or below three times what one request to the store
may take with its tries where that is longer (it is 9 minutes 21 seconds as
a node is set up by default, so fifteen minutes is the least): the grace is
all that keeps a file a restore within the window names from being deleted,
and it must cover a file being written a few minutes before it is
committed (its upload, its copy to local disk and three tries to commit: 3
minutes 51 seconds at most by default), a restore waiting up to a minute
for a clean-up to settle what it deletes, and what the clocks of the node,
the store and Postgres differ by. That least is a stop-gap for a node set
up near its defaults, and follows the store's bound only. It does not cover
a long lease with a long database timeout (a file waits for its commit
through three tries, each a quarter of `VELLEV_LEASE_SECS` or
`VELLEV_DATABASE_TIMEOUT_SECS` if that is shorter: an hour's lease and a
timeout of fifteen minutes make that forty-five minutes of tries, after the
upload), a head file of more than one part (16 MiB by default), a clock
wrong by more than the grace, or a flush or merge that takes longer than
the grace. Nor does it cover a reader still fetching a segment of several
parts under a short restore window of the tenant's own: each file is given
two requests a part, 6 minutes 14 seconds a part by default. Until cleaning
up goes by the commits Postgres still holds instead of by time (DW-681,
with DW-668), raise the grace with the lease and the database timeout, and
with both, as raising one alone may change nothing where the other is the
lesser: it should exceed what one request to the store may take plus three
times the lesser of a quarter of the lease and the database timeout, plus
some twenty seconds for the file's copy to local disk, with room to spare.

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
