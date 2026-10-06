-- The nodes running against this database and the newest file format and
-- protocol each can read, so that nothing newer is written while a node
-- that cannot read it is still about.
CREATE TABLE vellev.nodes (
    node     text        PRIMARY KEY,
    format   integer     NOT NULL,
    protocol integer     NOT NULL,
    seen_at  timestamptz NOT NULL
);
