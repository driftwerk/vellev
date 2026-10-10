# Release notes

What each release of the published crates, `vellev-core`, `vellev-index`,
`vellev-api` and `vellev-client`, changes for whoever uses them, and what a
node of that release answers differently. The breaking changes are those
`cargo semver-checks` finds against the release before, and whatever it
cannot see.

## 0.2.0

### Before you upgrade

- **Bring clients up to this release before the nodes**: see
  `model_unavailable` below.
- **Name the local models your collections embed with.** A node used to
  run any model fastembed lists. It now runs its own,
  `VELLEV_LOCAL_MODEL`, which is `BAAI/bge-base-en-v1.5` unless set, and
  the others named in `VELLEV_LOCAL_MODELS_ALLOWED`, and no other. A
  collection that embeds with a model not named is answered
  `model_unavailable` (501, `setup`) from the upgrade on. A node does not
  start if a name is not a model the release runs, or if it is told to
  run more models than `VELLEV_LOCAL_MODELS` could ever have loaded.
- **Name the regions your collections embed through Bedrock in**, in
  `VELLEV_BEDROCK_REGIONS`, five at the most, unless the only one is the
  node's own: `AWS_REGION`, or `AWS_DEFAULT_REGION` if that is unset or
  blank, which is the default, read from those two variables only and
  not from a profile or the machine. A collection in any other region is
  answered `model_unavailable` (501, `setup`) from the upgrade on. So is one whose Bedrock model is not
  `amazon.titan-embed-text-v2:0` or the same without its version, with
  `build`: any name that began as Titan's used to be taken.
- **Give a node memory for the local models it may have loaded**, about
  a gigabyte each, `VELLEV_LOCAL_MODELS` of them: two unless set.
- **Texts longer than 51,200 characters are cut** before a local model
  reads them. A model reads 512 tokens and no more, so a vector changes
  only for a text whose first 512 tokens lay beyond that, which takes
  runs of white space longer than a hundred characters: such a text's
  vector may differ from the one the same text gave before.
- **Pass a tenant key when embedding through `vellev-embed` directly**:
  `Embedder::documents_for` and `query_for`. A local model shares itself
  among tenants, and callers that give none share one line.

The workspace's version is 0.2.0 from the change that added what follows
on restores, deletes and reads. Upgrade every node before asking for a
restore or sending a delete with a `limit`: see `docs/compatibility.md`.
Migration `0009_restores.sql` adds the table restores are recorded in; it
is applied as a node of this release starts, and a node of the release
before runs on the migrated database.

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
  each model and each `reason`, however many collections embed with the
  model and however each words its texts, and whatever the model does
  meanwhile. A model that
  refuses every request is not asked again within 20 seconds of the first
  refusal, though requests on their way to it when the fault first shows
  each still reach it once. A model that answers some requests and
  refuses others goes on being asked.
- What a model's server says is logged and passed on without what is
  shaped like a key in it: see the README for the shapes. Nor is the
  start of a key that lies cut short four kilobytes into what an endpoint
  says, and no more than 64 kilobytes of a refusal are read.
- An OpenAI endpoint's `model_not_found` and `invalid_api_key`, Bedrock's
  resource not found, a model directory the node may not write, and
  vectors of another size than the collection's are `model_unavailable`.
  Everything else that stops a text being embedded is `unavailable`, as
  before.
- **A write is given up on when the model has stopped, not for others'
  load.** A search whose text is not embedded within 10 seconds
  (`VELLEV_EMBED_QUERY_SECS`) is `unavailable` (503): send it again. A
  write is `unavailable` when the model has got nowhere with it for a
  minute (`VELLEV_EMBED_DOCUMENTS_SECS`) beyond the wait it was told to
  expect: nothing of its own made, and nothing ahead of it in its own
  tenant's line done with. What other tenants do is not its progress and
  not its stalling. A write a local model is reckoned not to come to
  within two minutes (`VELLEV_EMBED_DOCUMENTS_HELD_SECS`) is `unavailable`
  at once rather than held, the tenants with most queued first. A local
  model had no limit, and Titan none for a write of many texts. A commit
  still waits five minutes at most for its vectors. A local model still
  being fetched when a request is given up on goes on being fetched.
  `VELLEV_EMBED_EACH_DOCUMENT_SECS` is a second at the least.
- **A node runs its own local model alone unless told otherwise**: see
  above. A collection made with a model it does not run is `invalid`
  (400).
- **A node holds each local model once, and two loaded at most**
  (`VELLEV_LOCAL_MODELS`). A model named by another organisation's copy,
  or in another case, was loaded again for each name, about a gigabyte a
  time while it runs, with no limit. A place is kept for the node's own
  model, which stays loaded. Any other not used for 15 minutes
  (`VELLEV_LOCAL_MODEL_IDLE_SECS`) is let go, and loaded again when asked
  for. A collection whose model there is no room for just now is
  `unavailable` (503). A model let go may not give its memory back to the
  operating system.
- **A local model is shared among tenants, not among requests.** Each
  tenant has a line of its own, and each run owes each waiting tenant an
  equal share of its weight, which the tenant spends a text at a time in
  turn, and a share not used passes to the next in turn: so tenants have
  the same of the model's time whatever their texts weigh, and a run is
  no longer for it; within a tenant, writes are made in
  the order they came, with writes of four texts or fewer first. Searches
  likewise, and runs of searches and of writes take turns. One tenant's
  bulk import no longer puts another tenant's one-text write at the back
  of a queue, and forty searchers no longer starve every write. A run is
  as much as takes about a second, by the length of the texts, not 32
  whatever they are. A run the model fails is put off and each tenant in
  it run by itself in its turn, so a text the model fails on fails the
  write it is in and no other; a tenant whose texts keep failing is run
  apart from the others, less and less often, and logged each time. The
  README states what is promised as bounds.
- **A search no longer waits behind a write's texts on Bedrock.** Two of
  the ten calls a node has to Titan at once are kept for searches, and
  writes' texts have the other eight, shared out among the tenants that
  write in turn, so a large write takes about three tenths longer to
  embed than before, and tenants' writes take turns. The ten are the
  node's, whatever sizes its collections name, and are shared equally
  among the regions it embeds through, so one region that stops answering
  does not stop the others. A region a caller gives up on after thirty
  seconds unanswered is not tried again for five. A search beside a
  write of two thousand texts waited for them all.
- **A collection can only be made to embed through Bedrock in a region
  the node calls, with Titan by name**: any other is `invalid` (400).
- **Those who search with the same text at once are all told how it went
  together.** When the embedding failed, each used to ask again in its
  turn, the last of ten waiting ten times as long as the first.
- **A node keeps what it knows of 64 models served elsewhere**, and
  forgets the least lately asked for, never one in use. With all 64 in
  use, a request for another is `unavailable` (503).

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
- **A search waits at most ten seconds for its text to be embedded**
  (`VELLEV_EMBED_QUERY_SECS`), by a model the node runs itself as by one
  it calls. `nearest` is then refused as `unavailable` (503); `hybrid`
  answers by the words alone, with `degraded_by: "unavailable"`.
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
not build. A new op, or a new field of an op, will be a breaking change.
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
