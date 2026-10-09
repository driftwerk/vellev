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

New: `Bounded`, `Limit`, `Limits`, `Missing`, `Reason`, `Code::status()`.

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

New: `ModelFault`, itself `#[non_exhaustive]`.

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
