# Compatibility

Two releases of vellev, one and the next, must run side by side against
the same Postgres and object storage: a rolling deploy runs both for a
while, and a rollback runs the older one on everything the newer one
wrote. This page says what a release may change so that both hold.

A release here is a version of the `vellev` node. The client crate is
published separately and follows the same rules for the API.

## Notes for the release that gave tenants a lineage

While nodes of this release and of the one before run side by side, remove
no tenant and ask for no restore from a backup. A node of the release
before purges everything kept under a removed tenant's id, whichever
tenant's or cluster's it is, this release's backups among it, and restores
a backup into a tenant of the id whoever the backup is of; a restore is
done by whichever node writes the tenant, which the caller does not
choose.

Backups go on through the window, each release's beside the other's. A
node of the release before keeps what it takes under
`_backups/{tenant}/backups/` and `_backups/{tenant}/files/`, lists those
two prefixes to their ends as its own, prunes under them whatever no
record it reads lists, and restores from the records it finds there. A
node of this release keeps what it takes under `_backups/{tenant}/records/`
and `_backups/{tenant}/data/`, and lists, prunes, purges and restores from
those alone. Neither lists the other's place, so neither deletes, nor
restores from, what the other took: a tenant backed up by a node of each
has two sets of backups, each seen by the nodes of the release that took
it. Were this release's layout nested under the old prefixes instead, a
node of the release before would list this release's records as its own,
look for their files under its own prefix, find them unlisted there, and
delete them in its daily prune. After a rollback, this release's backups
stay where they are, unseen, and are seen again once this release runs.

- Backups taken before this release are not carried over. They lie where
  this release never looks, and say neither the lineage nor the cluster
  this release goes by: it does not list them, restore from them, adopt
  them, prune them or purge them. Take a backup of every tenant once the
  upgrade is done, and delete `_backups/{tenant}/backups/` and
  `_backups/{tenant}/files/` by hand when what the release before took is
  no longer wanted. No cluster had been deployed when this was decided.
- Restoring `{"backup": id}` no longer makes a tenant the cluster does not
  have, nor restores a backup that is not the tenant's own: both are not
  found. A cluster that lost a tenant asks to `{"adopt": {"backup": id}}`
  instead, `--adopt` on the command line, which a node of the release
  before refuses as a field it does not know. Adopting gives the tenant no
  key unless `with_keys` asks for those the backup kept.
- Two refusals have codes of their own, both 409: `empty`, for backing up
  a tenant that has committed nothing, which answered 201 with a backup of
  nothing; and `purging`, with a `Retry-After`, for making a tenant whose
  id is still being purged, which was `conflict`. A client of the release
  before reads either as `unknown`, and this release's Rust client reads
  both as final, as it does any code it is not told to retry: the request
  as sent will not pass by being sent again at once, and `purging` says
  in `Retry-After` the least that is, minutes rather than seconds, for
  whoever makes the tenant again to wait out before asking again: a
  lower bound, not when the purge will be done.
- Migration `0010_lineage.sql` follows `0009_restores.sql`. A database
  migrated to `0009` by the release before takes `0010` as a node of this
  release starts.
- A backup as the API lists it says its `lineage` and its `cluster`.
- Postgres 13 or newer is needed: the migration calls `gen_random_uuid()`.
- A node of the release before does not see the backups this release
  takes; after a rollback they are kept, and seen again once this release
  runs.

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
  older release inserts without them (as `incarnation` and `lineage` do),
  or notes what the older release would not when it deletes (as the
  trigger that records which tenant a purge is of does);
- never rename or drop a column or table, or change what one means, in
  the same release that stops using it. Drop it a release later;
- run inside the migration's transaction: nothing that cannot, such as
  `CREATE INDEX CONCURRENTLY`, so that a migration that fails leaves the
  database as it was.

The migration that gives tenants a lineage calls `gen_random_uuid()`,
which Postgres has had built in since version 13.

A release's migrations are numbered from one with none left out, and a
database is only ever migrated forwards. A node that finds one of its
migrations applied otherwise than it has it, or left out below one that
was applied, was started on a database another build of vellev migrated:
it changes nothing and says so. Run the release that migrated the
database, or a later one; never drop or recreate the database to get past
it.

A node starts on a database a newer release has migrated: it checks the
migrations it knows were applied as it knows them and leaves newer ones
alone. It refuses one where a migration it knows was skipped.

## Backup records

A backup is a JSON record in the backup store listing the files it
needs, which backups share. Each record carries a `version`
(`vellev_storage::backups::RECORD_VERSION`); a record without one is of
version 1. Pruning deletes only the files no record lists, so a node must
never delete around a record it cannot read.

A record says which tenant it is of, by the lineage the tenant was given
when it was first made, and which cluster took it, by the id the cluster
was given when its database was made (`vellev.cluster.id`, which anything
else that must tell clusters apart reads too). It lies under
`_backups/{tenant}/records/{lineage}/{cluster}/`, and the files it lists
under `_backups/{tenant}/data/{lineage}/{cluster}/`. A tenant lists and
restores from the records of its own lineage, whichever cluster took them.
A cluster writes, prunes and purges only under its own id, and shares no
file with another: one that adopts a tenant another cluster backed up
copies every file it needs once more, under its own id. A record that
names another tenant, lineage or cluster than where it lies, or none, is
one that cannot be used. A cluster that has lost a tenant takes up its
lineage with the commit that restores it from a backup it was asked to
adopt, and only then.

A backup is taken only of a tenant one of whose partitions has
committed; it records every partition it takes, one that never committed
as empty, so that restoring it empties that partition too. Pruning keeps,
whatever their age, the newest backup and the newest that holds data, and
holding data means having committed: a tenant whose every object was
deleted since has still committed, so its backups hold data, and its last
backup with objects in it ages out as any other.

A release may, without a new record version:

- add a field with a default, so that records written before it still
  read, as long as the release before may ignore the field and still
  restore and prune correctly.

Anything else raises the version: a field the release before would
misread, a change to where files are kept or what a list of them means,
or making an existing field optional.

A node that finds, among the records its cluster took of a tenant, a
complete one it cannot use (unreadable, of a newer version, or naming
another tenant, lineage or cluster than where it lies), or a pending one
written less
than a day ago, which may be of a backup under way, lists the backups it
can use but deletes no file of the tenant's backups. It logs every such
record at error level and keeps refusing until each is mended or
removed. A pending record older than a day is forgotten whether it can
be read or not, as no backup is still writing it.

Releases before the one that added `version` pass over a record they
cannot read and delete the files only it lists. So a release that writes
a record of a new version must not run beside any of those: it may be
deployed only once every node runs a release that refuses instead.
Within a rolling deploy of a new version, the older nodes refuse to prune
the tenants the newer ones backed up until they are replaced.

### Backups taken before tenants had a lineage

Releases before the one that gave tenants a lineage kept every backup of
a tenant id under `_backups/{tenant}/backups/` and `_backups/{tenant}/files/`
themselves, whichever tenant of that id it was of, and their records say
neither lineage nor cluster. They are nobody's to this release: not
listed, restored from, adopted, pruned or purged. Nothing tells which
tenant of an id such a record is of but an incarnation, which is one
cluster's own count and the same in any two clusters, so going by it would
hand one customer's backup to another's tenant. The record version is not
raised for the new layout, as no release reads the other's place: this
release's lies beside those prefixes, not under them, since a release
before lists each to its end and deletes under it whatever the records it
reads do not list.

### Upgrading from a release before claims were renewed

Releases before the one that renews backup claims (DW-665) differ in ways
that matter while both run:

- they take a claim for six hours and never renew it, and do so without
  the tenant's lock, so two of them, or one of them and a newer node, can
  both believe they hold a tenant;
- when a tenant is removed while they back it up, they discard the backup
  with every file it lists, files its older backups share among them
  (a newer node only forgets such a backup, and leaves its files to the
  tenant's purge);
- they back up a tenant that holds nothing, and prune by that backup;
- they pass over a record they cannot read and delete what only it lists.

A newer node honours an older node's claim, which simply runs out, but
nothing makes the older node honour the newer node's rules. So, during a
rolling deploy from such a release:

1. Prefer to keep the older release from backing up at all while the newer
   one runs: first roll the older release out again without a backup store
   (`VELLEV_BACKUP_STORE` unset), then roll out the newer release with it.
   No node of the older release then backs up, prunes or discards beside a
   newer one.
2. If that is not possible, keep the mixed window short, and while it lasts
   ask for no backups or restores and remove no tenants: the older nodes'
   daily backups are then the only backup work they do, and their discard
   runs only when a tenant is removed mid-backup.
3. Once every node runs the newer release, a claim an older node took
   before it stopped may hold its tenant for up to six hours; it runs out
   by itself. A backup of nothing an older node took stays listed, and
   pruning keeps the newest backup that holds data beside it.

## Links

Continuous backup (DW-668) ships one link per committed head to the
backup store, under
`_backups/{tenant}/ship/{lineage}/{cluster}/{incarnation}/{partition}/`:
`links/{position}.link`, and, for a frame too large to go inline,
`frames/{original key}` (`vellev_storage::link`). Nothing ships links
yet; this says what holds once something does.

A link is not a file of the cluster's format. It has its own magic
(`VLNK`), its own version (`vellev_storage::link::LINK_VERSION`) and a
CRC32 over its header, and it is read only by backup code. So a link
version is not gated by `FORMAT_VERSION`, and a new link version is not
a new file format. The frame it carries is a log or a manifest of the
cluster's format, byte for byte as the primary holds it.

A node refuses a link of a version newer than it reads
(`StorageError::UnsupportedVersion`). It reads the version before it
checks the CRC, as a later version may frame its header otherwise, so a
link whose version bytes were damaged into a later version is refused as
one it does not read, not as corrupt.

**Every field added to the header is a new link version**, and so is any
other change to what a header holds or how it is encoded. A field added
without one breaks both ways:
- A reader of the release before cannot tell an added field it may
  ignore from one it must not.
- Two links at one position are of one commit exactly when every header
  field but `body` is equal, and anything else is a fork. During a
  rolling upgrade, a commit shipped by a node of the release before
  (without the field) and by one of the new release (with it) would
  compare as a fork.

A new link version follows the rollout of a file format: the release
reads it, together with every link version back to the previous
release's, and writes it only once every node of the cluster reads it.
Until then, links of one commit are of one version, and compare as they
should. Once it writes the new version, a commit already shipped at the
version before can be shipped again at the new one, by a node that takes
the tenant over. So the release that adds a version also says how a link
of each version before compares with one of the new: by the fields both
carry, unless what the new field says can be told for the older link
too. A test pins the encoding of link version 1 byte for byte, so a
field added without a new version fails it.

Its tests carry fixtures of every link version under
`crates/vellev-storage/tests/fixtures/link-N`, which a test reads and
writes back byte for byte. The fixtures of a released version are never
written again.

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

What changed in the API with the release that gave tenants a lineage is
listed in its notes above.

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

## The Weaviate import

`vellev import weaviate` of one release goes on from where that of the
release before stopped.

- The state file keeps its fields and their meaning; a release may add
  fields that the one before ignores.
- A tenant imported into holds the import's own collections beside the
  imported ones: `vellev_imported_CLASS` for each class and
  `vellev_imports`. No Weaviate class can be named so, since theirs begin
  with a capital. A release may add properties to them, which it does to
  those already there, and never changes what one means. Each object in
  them has a version, the first when it says none; an import that reads
  a later one than it knows stops, and changes nothing.
- The ledger's hash is of an object as that release's nodes return it.
  If a release changes how a stored value is written out, the import of
  that release takes the objects for another writer's and leaves them,
  rather than replacing or deleting them on a hash it can no longer make.
  A test holds the hash of fixed objects to what it was.
- **Numbers.** A node of this release reads the numbers in a request
  exactly. One of an earlier release read about three in ten doubles
  with a full set of digits as a neighbouring double, and stored that.
  Whatever such a node stored stays as it stored it, so a filter for a
  number equal to one a client sent then may not find it, the stored
  number being its neighbour; one stored since is found. Storing the
  object again puts it right. Run no import that keeps a ledger against
  nodes from before this release: the hashes it would keep are of
  numbers the node did not store.
- A collection imported before there was a ledger is taken up, not
  caught up as it was: what it lacks or holds otherwise is withheld, not
  copied, what it holds that Weaviate does not is noted and not
  deleted, and the run fails until one is told whose they are. What a
  run is told is another writer's stays so in the ledger for as long as
  the import does. See the README.
- The report may gain fields. Its exit status is 0 when everything that
  is the import's is settled, 3 when another run is all it takes, and 1
  otherwise; before, it was 1 whenever the counts differed.

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

A restore from a backup copies the backup's files into the tenant's store
under names of its own and commits a head naming only those (DW-680). No
file format changes: a node of the release before opens, cleans and backs
up such a tenant as any other, its file names keeping the position first,
and passes over an index file that says another segment, as it always
has. A node of the release before that restores from a backup does it the
old way, by the names the files had, and a delete given up before the
restore and landing after it can take a file its head names. That is one
more reason to upgrade every node before asking for a restore; see
[runbook.md](runbook.md).

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

Rolling back past DW-680 reads every tenant this release restored, and
brings back the hazard above for restores the older release makes, for as
long as it runs.
