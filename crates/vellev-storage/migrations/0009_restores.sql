-- The restores made lately, each by the request that asked for it, so that
-- the request made again, on any node, is answered as it was rather than
-- made twice, and refused if it asks for another state (`asked` is a digest
-- of the one it asked for). Each is noted by the statement that commits it.
-- The rows are forgotten once a day old, whatever became of their tenant,
-- and a partition's go once it is removed. Only those of the partition's
-- incarnation now are its own: a release that does not know this table
-- leaves them behind when it removes the partition.
CREATE TABLE vellev.restores (
    tenant        text        NOT NULL,
    partition     bigint      NOT NULL,
    request       uuid        NOT NULL,
    asked         bytea       NOT NULL,
    incarnation   bigint      NOT NULL,
    position      bigint      NOT NULL,
    from_position bigint      NOT NULL,
    from_at       timestamptz NOT NULL,
    made_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant, partition, incarnation, request)
);
CREATE INDEX restores_by_age ON vellev.restores (made_at);
