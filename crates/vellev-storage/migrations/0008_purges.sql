-- Tenants removed whose files are still to be deleted, from the store, the
-- backup store and every node's cache. A tenant is not made again under
-- the same id until its purge is done, so that nothing new is deleted with
-- the old; a row is kept for a while after, so that every node drops what
-- it cached of the tenant.
CREATE TABLE vellev.purges (
    tenant       text        PRIMARY KEY,
    requested_at timestamptz NOT NULL DEFAULT now(),
    purged_at    timestamptz
);
