# Release notes

What each release of the published crates, `vellev-core`, `vellev-index`,
`vellev-api` and `vellev-client`, changes for whoever uses them, and what a
node of that release answers differently. The breaking changes are those
`cargo semver-checks` finds against the release before, and whatever it
cannot see.

## 0.2.0

Bring clients up to this release before the nodes: see
`model_unavailable` below.

The workspace's version is raised to 0.2.0 by a later change, and nothing
is published from main before that.

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
- **`GET /v1/limits`** tells any key the most a request may ask for.
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
  it gives, up to a minute.

New:

- `Client::limits()`: `Ok(None)` from a node of 0.1, which has no such
  route.
- `Error::limit()`, `Error::limit_max(Bounded)`, `Error::missing()`,
  `Error::reason()`, `Error::is_model_unavailable()`.
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
