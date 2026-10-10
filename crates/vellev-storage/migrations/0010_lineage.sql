-- Which cluster this is: an id given once, here, and kept. It is written
-- into the backups the cluster takes and into where they lie, so that no
-- cluster deletes what another took; anything else that must tell one
-- cluster's from another's reads the same id.
ALTER TABLE vellev.cluster ADD COLUMN id uuid NOT NULL DEFAULT gen_random_uuid();

-- Which tenant a tenant is, wherever it is: an id it is given when it is
-- first made, keeps for as long as it lives, and takes with it into its
-- backups, so that a tenant made later under the same name is never taken
-- for it. Unlike the incarnation, which is this cluster's own count, it
-- can be taken up by a cluster that lost the tenant and adopts a backup of
-- it. Every partition of a tenant has its tenant's.
ALTER TABLE vellev.partitions ADD COLUMN lineage uuid;
-- Materialized, so that each tenant is given one, whatever its partitions.
WITH given AS MATERIALIZED (
    SELECT tenant, gen_random_uuid() AS lineage
    FROM (SELECT DISTINCT tenant FROM vellev.partitions) named)
UPDATE vellev.partitions p SET lineage = given.lineage
FROM given WHERE p.tenant = given.tenant;
ALTER TABLE vellev.partitions ALTER COLUMN lineage SET NOT NULL;

-- A trigger fills it in, as the incarnation's does, so that nodes still
-- inserting rows without it keep working through a rolling deploy.
CREATE FUNCTION vellev.registered_lineage() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.lineage := COALESCE(
        NEW.lineage,
        (SELECT lineage FROM vellev.partitions WHERE tenant = NEW.tenant LIMIT 1),
        gen_random_uuid());
    RETURN NEW;
END
$$;

CREATE TRIGGER registered_lineage BEFORE INSERT ON vellev.partitions
    FOR EACH ROW EXECUTE FUNCTION vellev.registered_lineage();

-- What the partition's tenant is to become with the commit that restores
-- it from a backup it adopts, and the generation of the lease that is to
-- make that commit: the lineage, the position of the commit, the keys and
-- the restore window, as JSON. Nothing of the backup is the tenant's until
-- that commit, which takes all of it at once; while this is set, a
-- partition that never committed commits nothing else.
ALTER TABLE vellev.partitions ADD COLUMN adopting text;

-- Which tenant a purge is of: the lineage the tenant had when it was
-- removed, so that only its backups are deleted. A purge asked for before
-- this is of a lineage nobody has, as backups taken before tenants had one
-- are nobody's: its tenant's files are purged as before. And when a node
-- last looked for anything written for the tenant since its purge, so that
-- one node at a time looks, now and then. The default is for the release
-- before, which notes a purge without one, after the trigger below has
-- noted it with the tenant's.
ALTER TABLE vellev.purges
    ADD COLUMN lineage uuid NOT NULL DEFAULT gen_random_uuid(),
    ADD COLUMN looked_at timestamptz;

-- A tenant is to be purged once its last partition is removed. A trigger
-- notes it, and which tenant it was, so that it is noted the same whichever
-- release's node removes it through a rolling deploy: the release before
-- notes the purge itself, after this has, and knows nothing of lineages.
-- It is the lineage the tenant has that is noted, never one it was to
-- adopt: a tenant removed before it took a backup up takes nothing of that
-- backup's with it.
CREATE FUNCTION vellev.removed_tenant() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO vellev.purges (tenant, lineage)
    SELECT OLD.tenant, OLD.lineage
    WHERE NOT EXISTS (SELECT 1 FROM vellev.partitions WHERE tenant = OLD.tenant)
    ON CONFLICT (tenant) DO UPDATE
    SET requested_at = now(), purged_at = NULL, claim = NULL, claimed_until = NULL,
        lineage = EXCLUDED.lineage, looked_at = NULL;
    RETURN NULL;
END
$$;

CREATE TRIGGER removed_tenant AFTER DELETE ON vellev.partitions
    FOR EACH ROW EXECUTE FUNCTION vellev.removed_tenant();
