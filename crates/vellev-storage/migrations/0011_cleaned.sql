-- The checkpoint a clean of the partition last deleted below: the key of
-- its base and that checkpoint's position. A clean records it, under its
-- lease, before it deletes anything, and it is only ever raised, so every
-- checkpoint a clean has deleted lies below it. A clean that finds a
-- checkpoint's predecessor missing takes it for a clean's work only below
-- the one recorded here; anywhere else it is a loss (DW-681).
--
-- Nullable with no default: a node of the release before neither reads nor
-- writes it, and inserts its rows without it. It goes with the partition's
-- row when the tenant is removed, so a tenant made again starts with none.
ALTER TABLE vellev.partitions
    ADD COLUMN cleaned_below text,
    ADD COLUMN cleaned_position bigint,
    ADD CONSTRAINT cleaned_whole
        CHECK ((cleaned_below IS NULL) = (cleaned_position IS NULL));
