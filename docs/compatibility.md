# Compatibility

Two releases of vellev, one and the next, must run side by side against
the same Postgres and object storage: a rolling deploy runs both for a
while, and a rollback runs the older one on everything the newer one
wrote. This page says what a release may change so that both hold.

A release here is a version of the `vellev` node. The client crate is
published separately and follows the same rules for the API.

## Files

Every file in object storage is a frame: a magic number, the kind of
file, a format version, the body's length and a checksum. A node refuses
a frame of a format newer than it reads (`StorageError::UnsupportedVersion`)
rather than guess at it.

The formats so far:

| Format | Release | What it added |
| --- | --- | --- |
| 1 | first | logs (with the requests each answered), manifests (naming the checkpoint before, the logs committed while a flush or a merge built to replay over its segments from the position those segments hold, and the requests answered in the last fifteen minutes), segments, vectors files, clustered index files |

A release may, without a new format:

- add an optional field that old code may both ignore and lose: during a
  rolling deploy a node of the release before can take a tenant and write
  a checkpoint, which writes the schema and rows back as it knows them,
  without the field. Only a field whose loss changes nothing that matters
  qualifies.
- read anything old formats wrote in a new way, as long as the result is
  the same data.

A refusal's `lacking`, what a refused request named that was not there,
is such a field: a node of the release before reads past it, and drops it
when it writes the manifest again, after which a try of that request is
still refused alike and says only that something was not found. It is
read so that a kind of thing a later release names, which this one does
not know, is nothing kept rather than a file that cannot be read: a later
release may add one without a new format.

Anything else is a new format:

- a new kind of file;
- a new variant of any enum in what is stored (a data type, embedding
  provider, distance, op, value), which the release before cannot read at
  all;
- a stored field the release before would ignore while misreading the
  data, or would lose when it rewrites a file. Moving vectors out of the
  segment file into a vectors file is the example: a node that ignored
  `SegmentMeta.vectors` would load every segment without its vectors and
  say nothing;
- a different encoding of anything existing.

Then:

1. The release reads the new format and every format before it back to
   the previous release's.
2. It writes the new format only once the cluster does. The format the
   cluster writes is kept in Postgres and only rises: each node registers
   what it reads while it runs and raises the cluster's format to the
   newest it may write, which happens only once every node registered
   reads it. A node that goes quiet stays registered (for a day, unless it
   stops cleanly and takes itself off), so its silence never lets the
   format rise past it. Every writer (`format::encode`, and the vectors
   file's own frame) must then take its version from the cluster's format
   rather than from `FORMAT_VERSION`; today, with one format, they stamp
   `FORMAT_VERSION`.
3. A node may write no newer than `VELLEV_MAX_FORMAT`, whose default is
   `DEFAULT_MAX_FORMAT`. A release that adds format N keeps that default
   at N - 1, so the cluster's format does not rise and a rollback finds
   nothing it cannot read. Operators raise it once the release has run
   long enough that they no longer want to roll back. A node refuses to
   become ready while the cluster writes a format newer than it reads, so
   rolling back past a raise fails at once and plainly.
4. The API refuses whatever would be stored in the new format (a new
   field or variant in a schema, say) until the cluster writes it.
5. Its tests carry fixtures of every format it reads, under
   `crates/vellev-storage/tests/fixtures/format-N`, and a test reads them.
   The fixtures of a released format are never written again: they stand
   for what is already in someone's bucket.

Storage types never refuse fields they do not know, so that what a newer
release wrote reads wherever the field is optional.

The gate works only when the release before already registers itself,
which releases from this one on do: the first new format can come no
earlier than the release after the one that first registers.

## Postgres

Each release's migrations leave the release before working:

- add columns with a default, or with a trigger that fills them when the
  older release inserts without them (as `incarnation` does);
- never rename or drop a column or table, or change what one means, in
  the same release that stops using it. Drop it a release later;
- run inside the migration's transaction: nothing that cannot, such as
  `CREATE INDEX CONCURRENTLY`, so that a migration that fails leaves the
  database as it was.

A node starts on a database a newer release has migrated: it checks the
migrations it knows were applied as it knows them and leaves newer ones
alone. It refuses one where a migration it knows was skipped.

## The API

Clients and nodes of different releases talk to each other.

- A response may gain fields; clients ignore what they do not know, and an
  outcome of a kind they do not know reads as `unknown`. A response never
  loses a field or changes what one means.
- A failure may gain codes, and its `limit` and `missing` may gain names. A
  client reads a code, a limit's name or a kind of missing thing it does
  not know as `unknown`, and takes an unknown code as final: it does not
  send the request again. A failure that may pass keeps `unavailable` or
  `uncertain`, which every client retries; a new code is only for one that
  will not. `model_unavailable` is such a code with one difference a
  client must know it to use: another node may answer where one cannot. A
  client from before it stops at the first node that says so, where it
  used to be told `unavailable` and try again, so bring clients up to this
  release before the nodes.
- A hybrid search's `degraded_by` and a failure's `reason` hold a code or a
  name a client may not know, read as `unknown`.
- The limits a node tells (`GET /v1/limits`) may gain fields. One added
  later is optional in the client's types, so that a node from before it
  still reads; the first seven are always told.
- A request may gain optional fields. A node refuses a field it does not
  understand, at any depth, so a client using a new field against an older
  node is told so rather than ignored.
- A route is never removed or changed in meaning; a new one may be added.

## Object versions

Every object a node answers with carries a `version`, 32 hexadecimal
characters, written in lower case and read in either, and a write may ask
to be made only if the object is still at one (`only_if_version`). The
version is a hash of the object's content,
computed afresh wherever it is read or checked and stored nowhere, so a
reader on one node and a writer on another must compute it alike, or a
client that read from the one and writes through the other is refused for
nothing. How it is computed is therefore a contract between releases, like
a file format: changing any of it is a breaking change, made only in a
release that says so, after which every version a client holds is stale
and its conditional writes are refused until it reads again.

The version is the first 16 bytes of SHA-256 over, in this order:

1. the bytes `vellev object version 1` and a newline;
2. the number of properties the object has, as a little-endian u64. Only
   properties with a value count: a property never sent and one sent as
   JSON `null` are alike absent, and the object has no entry for either;
3. each property in the order of its name's bytes (the order the stored
   map keeps): the name's length in bytes as a little-endian u64, the
   name's UTF-8 bytes, one tag byte for the stored type, then the value:
   - text (tag 1): the length as a u64, then the UTF-8 bytes, byte for
     byte, with no normalisation of any kind;
   - text array (tag 2): the count as a u64, then each text as above;
   - int (tag 3): the i64, little-endian;
   - number (tag 4): the f64's bits, little-endian, with `-0.0` taken as
     `0.0`. What is hashed is the stored value, not the JSON it came as:
     `1` and `1.0` sent for a number property parse to one f64 and are
     one version, and a decimal that does not round-trip is hashed as the
     f64 it parsed to;
   - bool (tag 5): one byte, 0 or 1;
   - date (tag 6): the instant as seconds since the Unix epoch, an i64
     little-endian, then its nanoseconds within the second, a u32
     little-endian, so that an instant before the epoch has the second
     floored and nanoseconds from 0 upward (`-0.25 s` is `-1` and
     `750000000`), and a leap second (`23:59:60`, which the node keeps
     as the 59th second with nanoseconds from 1,000,000,000 upward) is
     that second and those nanoseconds; the zone it was sent in does not
     count;
4. the vector: one byte 0 if the object has none, else one byte 1, the
   number of dimensions as a u64, then each f32's bits, little-endian, in
   order, with `-0.0` taken as `0.0`.

The one normalisation, of negative zero, is what the writer already does
when it asks whether a write changes anything: a number or a vector
element compared equal is the same content, and so the same version.

Not covered: the object's id, which is the key the version belongs to;
the collection and its schema; the position the object was written at;
who wrote it, or when. Two objects of the same content have the same
version, under any id, in any collection, on any node, after any restore.

`vellev-core` pins the contract with golden versions of fixed objects
(`version::tests::a_version_is_what_the_contract_says_it_is`), computed
again outside the crate from this description; a change that gives those
objects other versions fails the test, which is the point.

A node of a release before versions refuses `only_if_version` in a body,
as it does any field it does not know, so a conditional `PUT`, `PATCH` or
change in `POST …/changes` sent to one is refused rather than made
unconditionally. **Not so the `DELETE …/objects/{id}` route**: a node of
0.1 ignores a query it does not know, so `?only_if_version=` sent to one
deletes the object whatever the version. The client crate therefore sends
`delete_if_version` as `POST …/changes` with one `delete`; raw HTTP must
do the same until every node is 0.2, and may use the query after. A node
of 0.1 answers objects without `version`, which the client crate reads as
`None`.

## Between nodes

A node sends writes for a tenant on to the node that holds it, marked
with `x-vellev-forwarded`, which carries the protocol the sender speaks
(`vellev_node::PROTOCOL`); a node receiving one takes any value. Each node
registers the protocol it speaks.
A release whose requests may hold something the previous one does not
understand raises the protocol and sends such a request on only once
every node registered speaks it.

A restore is made once only by a node that records it under the request
that named it (`x-vellev-request`). A node of a release before that ignores
the header, so while such a node holds a tenant, a restore tried again
after its answer was lost, sent to it or on to it, is made again, going
back past whatever was written in between. Upgrade every node before
asking for a restore, and ask for none while a deploy that adds this runs.

A commit names the position and the head it was built on, and is recorded
only if those are what is committed still, in the statement that checks
its lease: two writers under one lease cannot both commit, nor one that
has not seen what the other did. That is all in the statement a node
sends, so it needs no migration, and a node of a release before, which
names nothing, has its commits recorded as they always were: on whatever
is committed, if they advance the position. While both releases run, only
this release's writers are held to what they follow. A node opens one
writer for each lease it takes, so two under one lease is not something a
node does; the check is what makes that not matter.

A delete by filter may carry a `limit`. A node of a release before refuses
the field, as it does any it does not know, so page deletes only once
every node is upgraded; without the field a delete of more than 8,192
objects is refused by this release and made by the one before. So is a
request with two deletes by filter and no limits, or with limits that come
to more than 8,192 between them: the release before made it, and this one
refuses it as invalid.

## What rollback covers

Rolling back to the previous release reads everything written during the
upgrade as long as `VELLEV_MAX_FORMAT` was not raised: the files, because
nothing newer was written; the database, because migrations left it
working; the API, because only optional fields were added. Fields the
release before does not know are lost when it rewrites a file, which is
why only fields that may be lost are added without a new format. After
the setting is raised and the cluster's format has risen, the release
before refuses to start, and rolling back means restoring from before.
