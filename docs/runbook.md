# Running vellev: restores, clocks and cutover

What an operator needs to know about restoring tenants, the clocks a
cluster goes by, and what must hold before a production cutover. The
reasons are in the DW-680 design (`dw-680-design.md`, sections 4, 8 and
12); this page says what to do.

## Before a production cutover

None of these is optional:

1. **The restore head-room is measured (DW-680, gate M).** A restore from a
   backup is a full reload of the tenant (below). Its cost has been measured
   on the largest real tenant shape, and a node restoring that tenant has
   at least this much room:

   | What | Head-room |
   |---|---|
   | Wall time of the restore | _to come from M_ |
   | Time writes are refused | _to come from M_ |
   | Peak resident memory | _to come from M_ |
   | Local disk high-water mark | _to come from M_ |
   | Bytes read from the backup store | _to come from M_ |
   | The next backup: time and bytes copied | _to come from M_ |

2. **Cleaning up goes by the commits Postgres holds (DW-681).** Until then
   whether a restore to a point in the window finds its files rests on a
   margin of time alone (see "Clocks", below). No production cutover before
   DW-681 has merged.
3. **The grace stays at its default of one hour** (`VELLEV_GRACE_SECS`)
   until DW-681. Do not lower it. Raise it with the lease and the database
   timeout as the README says.
4. **Every node runs a release with DW-680's restore** before any restore
   from a backup is asked for (see "Rolling deploys and rollback").

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
- **Every restore is a full reload.** Every flushed byte of the tenant comes
  down from the backup store and goes up to the tenant's store. Then every
  vectors file comes down again to the node's local disk. The tenant's rows
  are held in memory twice while the restore loads, beside what the node
  already holds. Local disk and the tenant's store hold the tenant twice
  until the old files are cleaned, after the restore window and the grace.
  The next backup copies the whole tenant again, and the backup store holds
  it twice for as long as backups are kept.
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
