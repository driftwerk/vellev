-- What a tenant is from its registration to its removal. Positions start
-- again when a tenant is recreated, so a reader tells a position of the
-- tenant as it was from one of the tenant as it is by this. It is the
-- generation the row was registered with, from the same sequence, so a
-- later registration has a greater one. A trigger fills it in, so that
-- nodes still inserting rows without it keep working through a rolling
-- deploy.
ALTER TABLE vellev.partitions ADD COLUMN incarnation bigint;
UPDATE vellev.partitions SET incarnation = generation;
ALTER TABLE vellev.partitions ALTER COLUMN incarnation SET NOT NULL;

CREATE FUNCTION vellev.registered_incarnation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.incarnation := COALESCE(NEW.incarnation, NEW.generation);
    RETURN NEW;
END
$$;

CREATE TRIGGER registered_incarnation BEFORE INSERT ON vellev.partitions
    FOR EACH ROW EXECUTE FUNCTION vellev.registered_incarnation();
