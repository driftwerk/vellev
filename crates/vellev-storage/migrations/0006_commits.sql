-- Every commit, by the database's clock, so that a partition can be put
-- back as it stood at any moment its restore window reaches. A commit
-- repeated after a lost answer is kept once. The rows go with the
-- partition, and those older than the window are forgotten, save the
-- newest, which is how the partition stood when the window began.
CREATE TABLE vellev.commits (
    tenant       text        NOT NULL,
    partition    bigint      NOT NULL,
    incarnation  bigint      NOT NULL,
    position     bigint      NOT NULL,
    head         text        NOT NULL,
    committed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant, partition, incarnation, position)
);

-- How far back a tenant may be restored, where it is not the default.
ALTER TABLE vellev.partitions ADD COLUMN restore_window_secs bigint
    CHECK (restore_window_secs > 0);
