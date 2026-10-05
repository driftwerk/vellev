CREATE TABLE vellev.partitions (
    tenant           text        NOT NULL,
    partition        integer     NOT NULL,
    generation       bigint      NOT NULL DEFAULT 0,
    lease_holder     text,
    lease_expires_at timestamptz,
    log_position     bigint      NOT NULL DEFAULT 0,
    manifest         text,
    committed_at     timestamptz,
    PRIMARY KEY (tenant, partition),
    CHECK ((lease_holder IS NULL) = (lease_expires_at IS NULL))
);
