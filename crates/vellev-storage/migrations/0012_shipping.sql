-- How a partition's history is shipped to the backup store (DW-668).
--
-- On vellev.partitions, written only by single-row conditional updates
-- of the partition's own row, so that the statements that delete what
-- shipping needs, which lock that row, are ordered against them:
--
-- - ship_hold_from and ship_cap: shipping's hold, taken before a base is
--   chosen. Commit rows at and above ship_hold_from are kept from then on
--   (by forget_history, from a later release on), and a clean records
--   nothing above ship_cap, the newest checkpoint when the hold was
--   taken, or 0 if there was none (by clean_below, from a later release
--   on).
-- - ship_base and the base checkpoint's position and key: the base of
--   the run of shipping under way, raised only.
--
-- No index, so that the update of the row every commit makes stays
-- heap-only. Nullable with no default: a node of the release before
-- neither reads nor writes them, and inserts its rows without them. They
-- go with the partition's row, so a partition made again starts with
-- none.
ALTER TABLE vellev.partitions
    ADD COLUMN ship_hold_from bigint,
    ADD COLUMN ship_cap bigint,
    ADD COLUMN ship_base bigint,
    ADD COLUMN ship_base_checkpoint text,
    ADD COLUMN ship_base_checkpoint_position bigint,
    ADD CONSTRAINT ship_held CHECK (
        (ship_hold_from IS NULL) = (ship_cap IS NULL)
        AND ship_hold_from > 0 AND ship_cap >= 0
    ),
    ADD CONSTRAINT ship_base_whole CHECK (
        (ship_base IS NULL) = (ship_base_checkpoint IS NULL)
        AND (ship_base IS NULL) = (ship_base_checkpoint_position IS NULL)
        AND (ship_base IS NULL OR ship_hold_from IS NOT NULL)
    ),
    ADD CONSTRAINT ship_base_held CHECK (
        ship_hold_from <= ship_base
        AND ship_base <= ship_base_checkpoint_position
        AND ship_base_checkpoint_position <= ship_cap
    );

-- The run of shipping under way: its base, copied from vellev.partitions
-- by the transaction that records the base, the partition's row first, so
-- that a shipper's writes are fenced on this row of their own; the newest
-- position up to which every commit from the base on has been shipped,
-- only ever raised, none before any is; and why the run stopped, if it
-- has. The watermark is a table of its own as it moves with what is
-- shipped. It goes with the partition's row: the foreign key deletes it
-- when the row is.
CREATE TABLE vellev.shipping (
    tenant      text   NOT NULL,
    partition   bigint NOT NULL,
    incarnation bigint NOT NULL,
    base        bigint NOT NULL CHECK (base > 0),
    shipped     bigint CHECK (shipped >= base),
    stalled     text,
    PRIMARY KEY (tenant, partition),
    FOREIGN KEY (tenant, partition)
        REFERENCES vellev.partitions (tenant, partition) ON DELETE CASCADE
);
