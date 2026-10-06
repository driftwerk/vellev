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
| 1 | first | logs, manifests (naming the checkpoint before), segments, vectors files, clustered index files |

A release may, without a new format:

- add an optional field whose absence old code reads correctly. A field
  that old code would ignore while misreading the data is not optional in
  this sense: if a node that does not know it would load the wrong rows,
  vectors or schema, it needs a new format. Moving vectors out of the
  segment file into a vectors file is the example: a node that ignored
  `SegmentMeta.vectors` would load every segment without its vectors and
  say nothing.
- read anything old formats wrote in a new way, as long as the result is
  the same data.

Anything else is a new format: a new kind of file, a field whose absence
changes what the data is, or a different encoding of an existing one.
Then:

1. The release reads the new format and every format before it back to
   the previous release's.
2. It writes the new format only once every node announced within a
   lease reads it, and no newer than `VELLEV_MAX_FORMAT`. A release that
   adds format N keeps that setting's default at N - 1, so a rollback
   finds nothing it cannot read. Operators raise it once the release has
   run long enough that they no longer want to roll back; from then on,
   rolling back means restoring from before the raise.
3. Its tests carry fixtures of every format it reads, under
   `crates/vellev-storage/tests/fixtures/format-N`, and a test reads them.
   The fixtures of a released format are never written again: they stand
   for what is already in someone's bucket.

Storage types never refuse fields they do not know, so that what a newer
release wrote reads wherever the field is optional.

## Postgres

Each release's migrations leave the release before working:

- add columns with a default, or with a trigger that fills them when the
  older release inserts without them (as `incarnation` does);
- never rename or drop a column or table, or change what one means, in
  the same release that stops using it. Drop it a release later.

A node starts on a database a newer release has migrated: it checks the
migrations it knows were applied as it knows them and leaves newer ones
alone.

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
(`vellev_node::PROTOCOL`). Each node announces the protocol it speaks.
A release whose requests may hold something the previous one does not
understand raises the protocol and sends such a request on only once
every node announced within a lease speaks it.

## What rollback covers

Rolling back to the previous release reads everything written during the
upgrade as long as `VELLEV_MAX_FORMAT` was not raised: the files, because
nothing newer was written; the database, because migrations left it
working; the API, because only optional fields were added. After the
setting is raised, a rollback can no longer read the newer files.
