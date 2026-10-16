# Running vellev: restores, clocks and cutover

What an operator needs to know about restoring tenants, the clocks a
cluster goes by, and what must hold before a production cutover. The
reasons are in the DW-680 design (`dw-680-design.md`, sections 4, 8 and
12); this page says what to do.

## Before a production cutover

None of these is optional:

1. **Every node that may restore a tenant has the head-room for it
   (DW-680, gate M).** Its cost was measured on the largest tenant shape on
   hand (below). Size each node by the rule under "Head-room for a restore".

2. **Cleaning up goes by the commits Postgres holds (DW-681).** Until then
   whether a restore to a point in the window finds its files rests on a
   margin of time alone (see "Clocks", below). No production cutover before
   DW-681 has merged.
3. **The grace stays at its default of one hour** (`VELLEV_GRACE_SECS`)
   until DW-681. Do not lower it. Raise it with the lease and the database
   timeout as the README says.
4. **Every node runs a release with DW-680's restore** before any restore
   from a backup is asked for (see "Rolling deploys and rollback").

## Head-room for a restore

Gate M measured a restore from a backup on the whole driftwerk Weaviate
import in one tenant: 4.3 million objects with 384-dimension code vectors,
8.6 million rows with the import's ledgers, and 9.9 GB of files (2.8 GB of
segments, 6.7 GB of vectors, 0.3 GB of indexes). Loaded, that tenant takes
17.8 GiB of memory on its node.

**These figures are for that full shape, code vectors included: a stress
bound, and the size that matters for Patchwork.** Driftwerk's cutover moves
only its memories and artifacts to vellev; its code vectors stay on
Weaviate, so its tenants are far smaller than this.

A restore costs what it can reuse. The tenant's own backup, taken recently
enough that the segments it lists are still the ones the node holds, reuses
almost everything. An old backup (one whose segments merges have since
replaced), another cluster's, or an adoption reuses nothing.

| On the shape above | Own recent backup | Old backup, another cluster's, or an adoption |
|---|---|---|
| Peak memory, against the tenant loaded | 20.5 GiB: **about 1.15×** | 37 GiB: **about 2×** |
| Memory needed free beyond the tenant loaded | about 0.15× (2.7 GiB) | about 1.1× (19 GiB) |
| Local disk high-water | +0.4 GB (the re-keyed index files) | +7.2 GB (the vectors, and the index files) |
| Time writes are refused | **about 3 minutes** (172 s) | 6 to 7 minutes (350 to 420 s) |
| Read from the backup store | 9.9 GB, about 620 requests | the same |
| Put to the tenant's store, then read back to check | 9.9 GB, then 9.5 GB | the same |
| The next backup | copies the whole tenant again: 9.9 GB, about 720 requests, about 3 minutes | the same |

A backup partly overtaken by merges falls in between. One taken a day of
steady writes before, with 14 of its 16 segments still the node's, peaked
at 22 GiB, refused writes for about 3 minutes, and took +0.9 GB of local
disk.

The times were measured against a local S3 on a USB disk on a shared
machine. They scale with the tenant's size divided by the slower of the
backup store's reads and the tenant store's writes.

**The rule (the owner's choice (i), 2026-10-12):** a restore from an old
backup, from another cluster's, or an adoption is allowed to take about 2×
the tenant's memory. Restore only on a node with **at least 1.15× the
tenant's loaded footprint free, beyond what it already uses, and the
tenant's vectors once more free on local disk**. A restore from the
tenant's own recent backup needs far less, but whether a backup is still
recent in that sense depends on the merges since, which an operator cannot
easily see: plan for the larger figure. A node without that room should
not be asked. Move other tenants off it first, or use a larger node.

To find a tenant's loaded footprint, read the node's process footprint
before and after the tenant loads. `State::resident` reads low.

## Restoring from a backup

A restore from a backup copies every file it needs out of the backup into
the tenant's store under names of its own, and commits a head that names
only those copies. Nothing of the backup comes in under the name it had.
That is what keeps a delete sent before the restore (by a clean, a purge or
a node since deposed), and landing after it, from taking a file the
restored tenant needs.

What follows for operators:

- **A backup that lacks a file is refused, by every cluster.** The restore
  fails as a `conflict` (409) naming the backup and the file. A cluster
  whose tenant still has the file in its own store is refused too: a
  restore takes nothing from the tenant's store. The next backup the same
  cluster takes copies back any file it lists that went, and mends the
  backup; restore from it after that. An older backup whose lost file the
  tenant no longer lists stays broken for as long as it is kept.
- **Every restore copies the whole tenant.** Every flushed byte comes down
  from the backup store, goes up to the tenant's store under new names, and
  is read back once to check that each copy is whole. The tenant's store
  holds the tenant twice until the old files are cleaned, after the restore
  window and the grace. The next backup copies the whole tenant again, and
  the backup store holds it twice for as long as backups are kept.
- **What it loads depends on the backup.** A segment the node already holds,
  under the same name in the backup, whose copy is read back and found to
  be exactly the file the backup sent, is taken as it is: not decoded
  again, and its vectors given to the copy's new name on local disk as a
  link. Any other segment is decoded from its copy, two at a time, beside
  the tenant as the node holds it, and its vectors come down to local disk
  again. See "Head-room for a restore" for what each costs.
- **Writes are refused for the whole restore**; reads are served.
- **A restore that fails leaves the tenant as it was.** Copies it made are
  deleted, or, if its commit was tried and its outcome is unknown, left to
  be cleaned up after the restore window and the grace. Asking again is
  safe. A client that keeps retrying a restore that keeps failing leaves a
  tenant's worth of such copies each time.

## Clocks

Three clocks matter, and nothing checks that they agree:

- **Postgres's `now()`** dates every commit, and so decides which states a
  restore to a point in the window may go back to and which history is
  forgotten. Every comparison it makes is with times it stamped itself, so
  a constant error in it cancels out.
- **The object store** dates every file as it is written.
- **Each node's clock** decides, at clean-up, how old a file is: a file is
  old once the store's date for it is more than the window and the grace
  behind the node's clock.

What can go wrong, each in one direction only:

- **A**: a node's clock runs ahead of the store's.
- **B**: Postgres's clock steps back after it dated a commit, as on a
  failover to a replica that is behind.

A state a restore may still reach keeps its files while

> **S ≤ grace − (P + W)**, with **S = A + B**

where P is how long a file may sit in the store before its commit is
recorded, and W how long a restore may wait behind a clean-up before it
takes the building lock. P + W is about five minutes at the defaults. So:

- at the default one-hour grace, the clocks may disagree by about 55
  minutes;
- at the least grace a node accepts, fifteen minutes, by about 10 minutes;
- raising the lease or the database timeout raises P, and the floor on the
  grace follows only the store's bound: a long lease with a short grace can
  leave no margin at all.

When S is exceeded, nothing fails at once and nothing signals it. The
tenant's current state is never touched. A restore point quietly loses its
files, and a restore to it fails as missing and commits nothing. A tenant
that cannot be opened at all needs that and a delete given up on top.

S is not checked. That is acceptable only because there is no production
cutover before DW-681, which caps what a clean deletes by the commits
Postgres still holds and removes S. Until then keep the one-hour grace,
keep node clocks on NTP, and watch for clock steps on database failover.

## Rolling deploys and rollback

- **During a rolling deploy** an older node that restores a tenant from a
  backup restores it the old way, by the names the files had, with the old
  hazard: a delete landing late can take a file the restored head names.
  Upgrade every node before asking for a restore, and ask for none while
  the deploy runs.
- **Rollback** to a release before DW-680 reads everything this release
  wrote: a restored tenant's files differ only in their names. It brings
  the old hazard back for as long as the older release runs.

## A tenant already broken by a late delete

A tenant restored before DW-680, whose restored head named a file a delete
given up then took, cannot be opened: every node fails to load it, naming
the missing file. It cannot be restored in place either, as a restore
opens the tenant first. By hand: once no delete can still be in flight
(the tenant's backup claim and every node's lease have long run out), copy
the missing file from the backup the tenant was restored from,
`_backups/{tenant}/data/{lineage}/{cluster}/{key}`, back into the tenant's
store under `{key}`. Then open the tenant and take a backup. DW-668's
`open_lost` is to make this a restore.
