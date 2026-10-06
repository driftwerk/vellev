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
| 1 | first | logs (with the requests each answered), manifests (naming the checkpoint before, the logs committed while a flush built to replay over its segments from the position those segments hold, and the requests answered in the last fifteen minutes), segments, vectors files, clustered index files |

A release may, without a new format:

- add an optional field that old code may both ignore and lose: during a
  rolling deploy a node of the release before can take a tenant and write
  a checkpoint, which writes the schema and rows back as it knows them,
  without the field. Only a field whose loss changes nothing that matters
  qualifies.
- read anything old formats wrote in a new way, as long as the result is
  the same data.

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
- A request may gain optional fields. A node refuses a field it does not
  understand, at any depth, so a client using a new field against an older
  node is told so rather than ignored.
- A route is never removed or changed in meaning; a new one may be added.

## Between nodes

A node sends writes for a tenant on to the node that holds it, marked
with `x-vellev-forwarded`, which carries the protocol the sender speaks
(`vellev_node::PROTOCOL`); a node receiving one takes any value. Each node
registers the protocol it speaks.
A release whose requests may hold something the previous one does not
understand raises the protocol and sends such a request on only once
every node registered speaks it.

## What rollback covers

Rolling back to the previous release reads everything written during the
upgrade as long as `VELLEV_MAX_FORMAT` was not raised: the files, because
nothing newer was written; the database, because migrations left it
working; the API, because only optional fields were added. Fields the
release before does not know are lost when it rewrites a file, which is
why only fields that may be lost are added without a new format. After
the setting is raised and the cluster's format has risen, the release
before refuses to start, and rolling back means restoring from before.
