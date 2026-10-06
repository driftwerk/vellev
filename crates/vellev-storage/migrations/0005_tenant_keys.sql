-- Keys that reach one tenant each, kept by the SHA-256 of their secret.
-- A tenant's keys go with its last partition, so that none reaches a
-- tenant made again under the same id.
CREATE TABLE vellev.tenant_keys (
    id         uuid        PRIMARY KEY,
    tenant     text        NOT NULL,
    hash       bytea       NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX tenant_keys_by_tenant ON vellev.tenant_keys (tenant, created_at);
