-- What a tenant is from its registration to its removal. Positions start
-- again when a tenant is recreated, so a reader tells a position of the
-- tenant as it was from one of the tenant as it is by this. Taken from the
-- same sequence as generations, so a later registration has a greater one.
ALTER TABLE vellev.partitions ADD COLUMN incarnation bigint;
UPDATE vellev.partitions SET incarnation = generation;
ALTER TABLE vellev.partitions ALTER COLUMN incarnation SET NOT NULL;
