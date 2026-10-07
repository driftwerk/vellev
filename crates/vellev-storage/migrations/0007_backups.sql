-- Who is backing a tenant up, or restoring it from a backup, and until
-- when; and when it was last backed up.
ALTER TABLE vellev.partitions
    ADD COLUMN backed_up_at timestamptz,
    ADD COLUMN backup_claim uuid,
    ADD COLUMN backup_claimed_until timestamptz;
