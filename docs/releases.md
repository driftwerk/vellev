# Release notes

What each release of the published crates, `vellev-core`, `vellev-index`,
`vellev-api` and `vellev-client`, changes for whoever uses them, and what a
node of that release answers differently. The breaking changes are those
`cargo semver-checks` finds against the release before, and whatever it
cannot see.

## 0.2.0

Bring clients up to this release before the nodes: see
`model_unavailable` below.

The workspace's version is 0.2.0 from the change that added what follows
on restores, deletes and reads. Upgrade every node before asking for a
restore or sending a delete with a `limit`: see `docs/compatibility.md`.
Migration `0009_restores.sql` adds the table restores are recorded in, and
`0010_lineage.sql` gives the cluster an id and each tenant a lineage; both
are applied as a node of this release starts, and a node of the release
before runs on the migrated database. Postgres 13 or newer is needed for
`0010`. While both releases run, remove no tenant and ask for no restore
from a backup; backups go on, each release keeping its own where the other
never looks, so that a tenant backed up by both has two sets until the
window ends, and nothing either release keeps is pruned by the other: see
`docs/compatibility.md`.

Before you upgrade, look at `VELLEV_GRACE_SECS`: **a node set to a grace
below the least it may be no longer starts**, where any number, 0 among
them, used to be taken. The least is fifteen minutes (900), or three times
what one request to the store may take with its tries where that is longer:
9 minutes 21 seconds as a node is set up by default, so fifteen minutes,
and more only where a request is given longer than by default
(`VELLEV_STORE_TIMEOUT_SECS`, `VELLEV_STORE_MIN_BYTES_PER_SEC`,
`VELLEV_UPLOAD_PART_MIB`) and `VELLEV_BACKUP_CLAIM_SECS` is raised too:
at its default of fifteen minutes the backup claim already refuses a
request of more than four and a half. The grace is how much longer than
a tenant's restore window a file nothing reaches any more is kept, and it
is all that keeps cleaning up from deleting a file a restore within the
window names: it has to cover a file being written a few minutes before
its commit (its upload, its copy to local disk and three tries to commit:
3 minutes 51 seconds at most by default), a restore waiting up to a minute
for a clean-up to settle what it deletes, and what the clocks of the node,
the store and Postgres differ by. A node left at the default, an hour,
starts as before.

The least is a stop-gap, and covers a node set up near its defaults and no
more. It follows the store's bound only, and does not cover a long lease
with a long database timeout (a file waits for its commit through three
tries, each a quarter of `VELLEV_LEASE_SECS` or
`VELLEV_DATABASE_TIMEOUT_SECS` if that is shorter), a head file of more
than one part (16 MiB by default), a clock wrong by more than the grace,
or a flush or merge that takes longer than the grace. Nor does it cover a
reader still fetching a segment of several parts under a short restore
window of the tenant's own: each file is given two requests a part, 6
minutes 14 seconds a part by default. Until cleaning up goes by the
commits Postgres still holds instead of by time (DW-681, with DW-668), a
node whose lease and database timeout are both longer should have a longer
grace: more than what one request to the store may take plus three times
the lesser of a quarter of the lease and the database timeout, plus some
twenty seconds for the file's copy to local disk, with room to spare.
Raising one of the two alone may change nothing: it is the lesser that
counts, so a longer database timeout does nothing while a quarter of the
lease is shorter, nor a longer lease once a quarter of it is past the
database timeout.

Whoever builds on `vellev-node` gets the same refusal from
`Settings::check`, and only by calling it: `Node::new` takes its settings
as they are. `Writer::clean` and `Tenant::compact` take any grace still,
and say what the caller then answers for.

### What a node answers

- **A new code, `model_unavailable` (501)**, where a node used to answer
  `unavailable` (503): it cannot use the collection's embedding model as
  it is set up, and will not until someone puts that right. Nothing was
  done; keep the request and send it again later, never drop it. `reason`
  says where the fault lies: `build`, `setup` or `provider`. A client of
  0.1 reads the code as `unknown` and gives up at the first node, where it
  used to try again.
- **A failure may say more**, in fields that are absent when there is
  nothing to say: `limit: {name, max}` when a request asks for more than
  the node allows, `missing` when something it names does not exist
  (`tenant`, `collection`, `object`, `key`, `backup`, `restore_point`,
  `route`), and `reason` as above.
- **`GET /v1/limits`** tells any key the most a request may ask for:
  `results`, `candidates`, `oversample`, `body_bytes`,
  `restore_window_secs`, `deleted` (objects one request deletes by filter)
  and `reads_at_once` (reads the node answers at once).
- **A hybrid search that ranked by the words alone says why**, in
  `degraded_by`: `unavailable` or `model_unavailable`.
- **A request for none**, `limit: 0`, is refused as before, without a
  `limit`.
- **A collection can only be made with a model the release runs**: any
  other is `invalid` (400), where it used to be made and every write to it
  refused. Making again a collection that is there exactly so still
  changes nothing.
- **A lasting embedding fault is logged at most once in 20 seconds** for
  each way of embedding and each `reason`, whatever the model does
  meanwhile. A model that
  refuses every request is not asked again within 20 seconds of the first
  refusal, though requests on their way to it when the fault first shows
  each still reach it once. A model that answers some requests and
  refuses others goes on being asked.
- What a model's server says is logged and passed on without what is
  shaped like a key in it: see the README for the shapes.
- An OpenAI endpoint's `model_not_found` and `invalid_api_key`, Bedrock's
  resource not found, a model directory the node may not write, and
  vectors of another size than the collection's are `model_unavailable`.
  Everything else that stops a text being embedded is `unavailable`, as
  before.

- **A restore is made once.** `POST /v1/tenants/{tenant}/restore` takes
  `x-vellev-request`, as writes do: asked for again under the same name
  within a day, on any node, it is answered as it was and restores
  nothing more. The client crate names every restore. One that fails, or
  is cut short by its node, once its commit was begun is answered as made
  if it was, and otherwise `uncertain` (503): ask again by the same name.
- **New answers to a restore**: `conflict` (409) while another restore of
  the tenant runs, or while it is being backed up or restored from a
  backup; and 422 with `invalid` when a request name is used again for a
  restore to another state.
- **`Restored` carries `incarnation`**, the tenant's incarnation the
  restore was made in, with `position`: what a read must have caught up
  to. A node of the release before sends none, which reads as 0.
- **Writes are refused as `unavailable` (503) while a restore runs**, and
  nothing of them is made.
- **A tenant's backups are its own, and a cluster's backups are its own.**
  Each tenant has a lineage, given when it is first made; each cluster an
  id. A backup says both and lies under both in the backup store
  (`_backups/{tenant}/records/{lineage}/{cluster}/`, its files under
  `_backups/{tenant}/data/{lineage}/{cluster}/`, beside where the release
  before kept its own). A tenant made again
  under a removed tenant's id is another tenant: it neither lists nor
  restores from the backups of the one before, and is given none of its
  keys. A cluster prunes and purges only the backups it took, so one
  backup store may serve several clusters. Backups taken before this
  release say neither and are not carried over.
- **Restoring `{"backup": id}` restores only the tenant's own backup**,
  and no longer makes a tenant the cluster does not have: both are
  `not_found`. A cluster that lost a tenant asks to
  `{"adopt": {"backup": id, "with_keys": false}}` instead (`--adopt ID`,
  `--with-keys` on the command line): the tenant is made if there is none
  and, with the commit that restores it, becomes the tenant the backup is
  of. Nothing of the backup is the tenant's if the restore fails. The
  tenant is given the backup's keys only with `with_keys`, which brings
  back a key revoked after the backup was taken. Only a tenant never
  written to and without keys adopts; any other is a `conflict` (409).
  So is a restore from a backup that lists a file no longer in the backup
  store, which says which backup and which file, and was a 500 that said
  neither. So is a commit refused while a tenant is taking a backup up, and the
  restore of an adoption that finds it no longer noted for the lease that
  commits it, as when it was given up meanwhile: nothing was restored, and
  the adoption may be asked for again. A node that lost the tenant to
  another while it adopted answers `unavailable` (503), to be tried again.
- **Two new codes, both 409**: `empty`, for backing up a tenant that has
  committed nothing, which answered 201 with a backup of nothing; and
  `purging`, with a `Retry-After`, for making a tenant whose id is still
  being purged, which was `conflict`. The `Retry-After` is the least the
  purge may still take, not when it will be done: until the purge is due
  and a node next looks, and no account of a purge that failed and whose
  claim is running out, which may add up to `VELLEV_BACKUP_CLAIM_SECS`.
  Asked again after it, the node says `purging` again, with another. A
  client of the release before reads either code as `unknown`. Neither is
  one to send again at once: the Rust client takes both as final, and
  whoever makes a tenant again waits at least the `Retry-After` out.
- **A removed tenant is purged under a renewed claim**, by one node, which
  deletes nothing of a tenant made again under the id meanwhile; a purge
  that fails once it has sent a delete is taken up again when that claim
  has run out, not before, so that no delete of unknown fate is on its
  way when the id is taken again; and the
  tenant is looked over again for a day after, for files that landed
  late. A backup claim is held for `VELLEV_BACKUP_CLAIM_SECS` (15 minutes;
  at least three times what one request to the store may take with its
  tries and half a minute more, 10 minutes 51 seconds as a node is set up
  by default) and renewed,
  where it was held for six hours.
- **`Backup` says its `lineage` and its `cluster`.**
- **New settings**: `VELLEV_BACKUP_CLAIM_SECS` and
  `VELLEV_BACKUPS_SCHEDULED` (true; false has a node back up only when
  asked); see the README.
- **A delete by filter deletes at most 8,192 objects.** With no `limit`
  it is refused as `invalid` (400), deleting nothing, if more pass; it
  used to delete however many did. With `limit` (1 to 8,192) it deletes up
  to that many and says how many; fewer than the limit means none were
  left. Send it again, **each time under a new `x-vellev-request`**, until
  it deletes fewer than the limit. The deletes by filter of one request
  count together, one with no limit for 8,192, and a request whose
  deletes may come to more is refused. A refusal for asking to delete
  too many, by a limit past the most, by limits that come to more, or by
  a delete with no limit that more pass, says so in
  `limit: {name: "deleted", max: 8192}`.
- **A read may be turned away** as `unavailable` (503) with
  `Retry-After: 1`, when the node is answering as many searches, listings
  and counts as it does at once. One tenant may have all of them while no
  other asks; while a read waits whose tenant has less than half, no
  tenant with half or more is given another. A read turned away does not
  count against its tenant's rate, unless it had something costly done
  first: the coordinator asked (a `strong` read, or one after a position
  the node had not reached), a copy of the tenant opened, or its text
  sent to a model. What it only waited for another read to have done is
  counted for that read alone. A text of up to 1,024 bytes embedded
  lately is not sent again, so a search by a new text that is turned
  away is counted once for that; further turn-aways of the same text are
  free, while its vector is kept (a text of up to 1,024 bytes whose
  embedding succeeded, among the 1,024 asked last); and the search answered at last is counted like any other. Ask
  again after the wait.
- **A search waits at most two minutes for its text to be embedded**,
  by a model the node runs itself as by one it calls. `nearest` is then
  refused as `unavailable` (503); `hybrid` answers by the words alone,
  with `degraded_by: "unavailable"`.
- **New settings**: `VELLEV_READS_AT_ONCE` (256), `VELLEV_READS_WAIT_MS`
  (500), `VELLEV_CACHE_LOADS` (2), `VELLEV_STORE_TIMEOUT_SECS`,
  `VELLEV_STORE_MIN_BYTES_PER_SEC`, `VELLEV_UPLOAD_PART_MIB` and
  `VELLEV_DATABASE_TIMEOUT_SECS`; see the README.
- **Vectors are never held in memory by a node with a cache directory.**
  A tenant whose vectors the cache cannot take, or whose disk does not
  answer, does not open, and is `unavailable` until it does.
- **Every object read carries `version`**: 32 hexadecimal characters, a
  hash of its properties and vector (docs/compatibility.md says exactly
  how), from `GET …/objects/{id}`, `/list`, `/nearest`, `/keyword` and
  `/hybrid`, whether or not the vector is sent along. Opaque: compare
  for equality only. The same content has the same version under any
  id, on any node, after any restore; a vector-only change gives a new
  one; an object changed and changed back has its old one. It is
  computed as an object is answered, never stored: in the same runs as
  below, about 1.6 µs an object with a 768-dimension vector (`get` 0.4
  µs → 1.6 µs; a page of 1,000 with versions 1.7 ms, where writing that
  page as JSON with its vectors takes 15 ms and without them about 0.3
  ms; a top-100 search gains 0.2 ms on 38 ms), and under 0.2 µs with a
  2-dimension one. On x86-64 the version costs about 1.6 µs per
  768-dimension object with the SHA extensions (Zen and later, Ice Lake
  and later); without them about 9 µs, which is about 9 ms of CPU per
  1,000-row page: check for `sha_ni` in `/proc/cpuinfo` on your hosts.
  On aarch64 the fast path takes `sha2`'s assembly backend, which
  `vellev-core` now builds with there.
- **A write can be made only if an object is still at a version**:
  `only_if_version` in the body of `PUT` and `PATCH …/objects/{id}`, as
  `?only_if_version=` on `DELETE …/objects/{id}`, and on `upsert`,
  `patch` and `delete` in `POST …/changes`. At any other version the
  write is refused as `conflict` (409) and `version` in the failure says
  the one the object is at; a `patch` with `only_if` too must pass both,
  and a `patch` refused by `only_if` now says `version` as well. On an
  object that is not stored, a conditional write is `not_found` (404,
  `missing: object`), as a `patch` is; a plain `delete` of one still
  changes nothing. A change that leaves the content as it is is still
  `unchanged`, and the version stays. In a batch, one change at a stale
  version refuses the whole batch, as `only_if` does; a request named
  again (`x-vellev-request`) is answered as it was, version and all. A
  node of 0.1 refuses the field in a body as one it does not know, so a
  conditional `PUT`, `PATCH` or batch is never made unconditionally by
  mistake; **but it ignores `?only_if_version=` on a `DELETE`, and
  deletes**. Until every node is 0.2, send a conditional delete as a
  `delete` in `POST …/changes`, which is what the client crate's
  `delete_if_version` does. `DELETE …/objects/{id}` now refuses a query
  parameter it does not know as `invalid` (400), where 0.1 ignored it.
- **A page of a collection costs its own size.** `/list` used to find
  and sort every row of the collection for each page, so walking a
  collection of N rows cost about N²/page. Measured in process, release
  build, random ids, several segments and a memtable, before → after: a
  page of 50 from the start of 4M rows 460 ms → 0.8 µs, of 1,000 483 ms
  → 9 µs, of 50 from the middle 282 ms → 0.6 µs; a page of 50 under a
  filter half the rows pass 246 ms → about 0.5 ms; a whole walk of
  300k rows at page 50 77.9 s → 15 ms, of 4M rows at page 100 (about
  3.7 hours before, by its per-page cost) 0.3–0.5 s, and under that
  filter about 10 s. A filter few rows pass was and stays microseconds
  a page. The live rows are now kept in key order for this, which costs
  a lookup a little: on an idle machine, `get` 250 ns → 208 ns (noise),
  writing 4M rows in commits of 8,192 34.3 s → 33.5 s (bulk 13,828 →
  13,340 objects/s, group commits unchanged), opening 4M rows from
  their segments about 12% longer, resident memory about 10% more.
  Searches are unchanged.

### Numbers are read exactly

- **A node now stores the number it was sent.** `serde_json` was built
  without `float_roundtrip`, and read about three in ten doubles with a
  full set of digits as a neighbouring double: `-2.3624865652539077e-47`
  came back `-2.3624865652539072e-47`. Every number property written
  through the API was open to it; vectors were not, an `f32` surviving
  the older reader. Numbers stored before stay as they were stored, so
  a filter for a number equal to the one a client sent may miss an
  object stored before this release, which holds its neighbour, and
  find it once the object is stored again.
- It costs time reading JSON: a request of 200 vectors of 768 dimensions
  parses in 7.7 ms where it took 4.1 ms, one of 1536 dimensions in
  15.3 ms where it took 7.9 ms. Against the rest of a write it does not
  show: storing vectors through the API measured 100 to 120 microseconds
  a vector of 768 dimensions and 220 to 290 a vector of 1536 both before
  and after, the runs differing more among themselves than across the
  change.
- **This must be on every node before an import keeps a ledger there.**
  The ledger's hash is of the object as Weaviate holds it; a node that
  stores another number makes every object with such a number look like
  another writer's.

### vellev import weaviate

- **A catch-up deletes only what the import stored.** It used to delete
  every object of the collection that Weaviate did not hold, whoever
  wrote it. The import now keeps a ledger in the tenant,
  `vellev_imported_CLASS`, and replaces or deletes an object only while
  it is exactly as the import left it; anything else is left, and marked
  in the ledger. The README states the rule. Each replacing and each
  deleting is conditional on the version the object was read at, so that
  nothing another writer does to its properties or its vector between
  the import's read and its write is lost.
- **Nothing is deleted until Weaviate has been walked, and then only
  within a bound**: `--max-deletes` (1000) and `--max-delete-fraction`
  (0.05) of a class, the fewer. A run that finds more gone deletes none
  and fails, as does one told to delete nothing, `--no-delete`, while
  anything is gone. The last object of a class is never deleted unless
  a run may delete all of one. Each object is asked for twice,
  `--confirm-secs` apart, with Weaviate ready and the class's shards
  loaded, before it is deleted.
- **A class made again in Weaviate is not taken for the one imported**:
  the run stops, unless given `--accept-remade`.
- **A class that gained a property is caught up**, the property added to
  the collection, where a run used to fail. A class that no longer fits
  is refused with every reason.
- **A collection an earlier release imported is not caught up by a
  plain run of this one.** It has no ledger. The run takes it up: what
  it holds exactly as Weaviate does becomes the import's, and
  everything else of Weaviate's, which is whatever Weaviate gained or
  changed since that import last caught up and whatever holds a number
  read the old way, is **not copied**. It is counted as `withheld` and
  the run fails, every time, saying so. What the collection held that
  Weaviate did not is noted, once, as `unknown`, and not deleted. The
  remedy, if nothing but an import has ever written to the collection,
  is one run with `--sole-writer`, which stores and replaces what was
  withheld, and `--delete-unknown` beside it, which deletes what was
  noted, within the usual bound, if it is still as it was noted. If
  something else has written to the collection, `--other-writers`
  leaves all of it alone for good, in one run, and no later run copies
  over it, deletes it or stores it again, whatever that run is told. It
  marks so only what the collection lacks or holds otherwise than
  Weaviate does, on the first run as on a later one; what both hold
  alike stays the import's. It is said of every class of the run.
  Otherwise import into a collection made anew.
- **`vellev import finalise`** ends an import: it drops the ledgers and
  notes that the collections are no longer imported.
- **The exit status says what is left**: 0 when everything that is the
  import's is settled, whatever another writer has done beside it; 3
  when Weaviate was written to behind the catch-up and another run is
  all it takes; 1 otherwise. `--strict` fails as before on anything of
  another writer's.
- **The report is exact, and the same every run.** An object deleted
  from Weaviate between being listed and being fetched is deleted in the
  same run. `changed` no longer counts objects stored again unchanged.
  New fields: `kept`, `withheld`, `unknown`, `barred`, `left`,
  `pending`, `budget`, `refused`, `imported`, `verified`,
  and `ended` beside the classes. An object gone from Weaviate that
  stays in the collection as another writer's keeps its ledger entry,
  and is counted as `left` on every run, until the collection no longer
  holds it.
- **`--verify` and `--verify-all`** compare what a collection holds with
  what Weaviate does, object by object, and change nothing. They fail
  on what a run that imports fails on, and `--verify-all` counts what
  the collection holds that the ledger has no entry for.
- A class whose import was ended is passed over by a run of every
  class, and refused only when named.
- **A collection is caught up only from the Weaviate it was imported
  from.** `--from-name` names one whose address changes between runs.
  `--created-as` and `--updated-as` must stay as they were.
- `--skip-refused` passes over an object vellev cannot hold, and reports
  it, where a run otherwise stops at it.
- A catch-up no longer holds every id of a class in memory, and fetches
  and stores by the page where much of a page changed.
- A request Weaviate turns away with 429 or 408, or whose answer is cut
  short, is asked again, as a failure of the server was, and for
  `--patience-secs` (600) rather than five times.
- `--per-second` holds after a stall too, where the time lost used to be
  made up in a burst, and covers the objects a catch-up fetches one at a
  time. A rate that is not a number is refused.
- `--page` is the node's limit on results unless given, where it was
  100. `--batch-bytes` is the most bytes sent to the node in one
  request, 8 MiB unless given; a request carries at most 8,192 changes
  besides, what a node commits together with other writers' requests.
- Two runs with the same state file: the second is refused. Two with
  different ones are not kept apart, but the ledger is written only over
  what was read, at the version it was read at: a run that finds it
  written under it stops.
- A node of 0.1 among those asked refuses a write conditional on a
  version; the run stops at it, saying so, with or without
  `--skip-refused`.

### vellev-client

Breaking:

- `Options` is `#[non_exhaustive]`: it can no longer be written as a struct
  literal, with or without `..Options::default()`. Write
  `Options::default().timeout(..).attempts(..)`; there is a setter for each
  field, and the fields can still be read and assigned.
- `Error` is `#[non_exhaustive]`, as are its variants `Api`, `Limited` and
  `Partial`: a `match` on an `Error` needs a wildcard arm, and a pattern
  naming one of those variants needs `..`.
- **What `Options::attempts` bounds has changed.** A request is sent at
  most `attempts` times and one less than there are addresses, not at most
  `attempts` times: an address that answers `model_unavailable` costs no
  try while another has yet to say so. A write is still sent at most
  `attempts` times.
- **How `model_unavailable` is met.** A read is asked of every other
  address first, at once, and then tried again with the usual waits, so
  that a client of one address is sent to another node by whatever is
  behind it. A write fails at the first node to say it: the one node that
  holds the tenant makes it, whichever address is asked.
- **A hybrid search answered by the words alone for `model_unavailable`**
  is, by a client of several addresses, first asked of each address not
  yet asked, once and at once, each for `Options::better_within` at most
  (two seconds unless set; with none, no address has time to). A client of one address gets
  the answer as before.
- An address that turns a request away is waited on for the `Retry-After`
  it gives, up to a minute. So is one that answers `unavailable` with a
  `Retry-After`, as a node does that is answering as many reads as it
  does at once: every other address is asked first, and the wait before
  any is asked a second time is no shorter than the one given.

New:

- `Client::limits()`: `Ok(None)` from a node of 0.1, which has no such
  route.
- `Error::limit()`, `Error::limit_max(Bounded)`, `Error::missing()`,
  `Error::reason()`, `Error::is_model_unavailable()`.
- `Error::Api` has `wait`: how long the node said to leave it before
  asking again, if it said.
- `Options::better_within`.
- `Records::delete_if_version(id, version)`, and `Error::version()`: the
  version an object is at, when a write conditional on it was refused.
  `Stored::version` on everything read.

### vellev-api

Breaking:

- `Code` and `Outcome` are `#[non_exhaustive]`: a `match` needs a wildcard
  arm. `Code` gains `ModelUnavailable`, which 0.1 read as `Unknown`; cast
  to a number, the codes there were are what they were, and
  `ModelUnavailable` is 10.
- These are `#[non_exhaustive]` and can no longer be written as struct
  literals. What a caller sends has a constructor taking what it must have,
  and a setter for each thing more:
  - `Nearest::to_vector(vector, limit)` and `Nearest::to_text(text, limit)`,
    then `.filter()`, `.oversample()`, `.vectors()`;
  - `Keyword::new(text, limit)`, then `.properties()`, `.filter()`,
    `.vectors()`;
  - `Hybrid::new(text, limit)`, then `.vector()`, `.alpha()`, `.fusion()`,
    `.properties()`, `.candidates()`, `.filter()`, `.oversample()`,
    `.vectors()`;
  - `Page::new(limit)`, then `.filter()`, `.after()`, `.vectors()`;
  - `Content::new(properties)`, then `.vector()`;
  - `Changes::new(properties)`, then `.vector()`, `.only_if()`;
  - `Batch::new(changes)`, `RestoreWindow::new(secs)`,
    `Counting::new(filter)`, `TenantPage::new(after, limit)`.

  What a node answers has a constructor taking all of it, which only a
  node, or a test standing in for one, needs: `Problem::new`,
  `Failure::new` and `Failure::from(Problem)`, `Results::found` and
  `Results::by_words`, `IssuedKey`, `TenantKey`, `TenantKeys`,
  `RestorePoint`, `RestorePoints`, `Backup`, `Backups`, `Restored`,
  `Tenants`, `Collections`, `Stored`, `Written`, `Objects`, `Counted`, each
  with `new`, and `Found::new(object)`, whose scores are then assigned. A
  pattern that takes one of these apart needs `..`.

  Their fields are public still, and can be read and assigned.
- `Problem` gains `limit`, `missing` and `reason`; `Results` gains
  `degraded_by`.
- `Restored` gains `incarnation`, and `Restored::new` takes it:
  `new(from, incarnation, position)`. `Restored::after()` is what to send
  as `x-vellev-after` to read past the restore.
- `RestoreTo` gains `Adopt(Adopt)`, with `Adopt { backup, with_keys }`:
  a `match` on `RestoreTo` needs the arm. `Code` gains `Empty` (11) and
  `Purging` (12), which 0.1 read as `Unknown`. `Backup` gains `lineage`
  and `cluster`, each `Option<Uuid>`, `None` from a node of 0.1.
- `Change::DeleteWhere` gains `limit: Option<u32>`, so writing the variant
  out needs the field. `Change::delete_where(collection, filter)` and
  `Change::delete_up_to(collection, filter, limit)` build it.
- `Change::Upsert`, `Change::Patch` and `Change::Delete` gain
  `only_if_version: Option<Version>`, so writing them out needs the
  field. `Change::upsert(collection, id, content)`,
  `Change::patch(collection, id, changes)`, `Change::delete(collection,
  id)` and `Change::delete_if_version(collection, id, version)` build
  them.
- `Stored::new` takes the object's `version` as a fourth argument, and
  `Stored` gains `version: Option<Version>`: `None` from a node of 0.1.
- `Problem` gains `version`.

New: `Bounded`, `Limit`, `Limits`, `Missing`, `Reason`, `Code::status()`,
`Content::only_if_version`, `Changes::only_if_version`.

Not changed, and why: `Change`, `RestoreTo` and `Consistency` stay
exhaustive, and `OnlyIf` and `Weighted` stay plain structs. A caller only
ever builds the three enums, which `#[non_exhaustive]` does not bear on,
and the only `match` on them is the node's, where an op not handled should
not build. A new op, or a new field of an op, will be a breaking change,
as `RestoreTo::Adopt` is in this release, which lands before 0.2.0 is
published.
`OnlyIf::new` and `Weighted::new` are there for whoever would rather not
write the two out.

### vellev-core

Breaking:

- `ObjectError` is `#[non_exhaustive]`.

New: `ModelFault`, itself `#[non_exhaustive]`; `Version` and
`VersionError`, with `Version::of(properties, vector)` and
`Object::version()`.

Not changed, and why: `EmbeddingProvider`, `HttpApi`, `DataType` and
`Distance` stay exhaustive. They are stored, so a new variant of any is a
new file format and a release that says so; and they are matched in the
index and the embedders, where a variant not handled should not build.

### vellev-index

Breaking:

- `VectorIndex` has two more required methods, `estimates` and
  `unmeasured`, its `shortlist` takes five parameters where it took four,
  and it can no longer be implemented outside the crate.
- `BinaryIndex::build`, which was `build(vectors: &[(u32, &[f32])])`, is
  `build(metric: Distance, vectors: &[(u32, &[f32])])`.
- `ClusteredIndex::estimates` is no longer a method of its own: it is the
  `VectorIndex` trait's, with a new last parameter, `ties: Ties<'_>`, and
  whoever calls it must have `VectorIndex` in scope.

New: `ClusteredIndex::fit`, which says whether an index is fit for the
vectors it is given, and `Unfit`, why it is not.
