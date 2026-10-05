-- Tenants are listed in byte order, a page at a time. Under the database's
-- own collation the primary key could not serve that order.
ALTER TABLE vellev.partitions ALTER COLUMN tenant TYPE text COLLATE "C";
