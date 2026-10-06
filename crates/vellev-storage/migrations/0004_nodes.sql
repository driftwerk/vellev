-- The nodes running against this database and the newest file format and
-- protocol each can read. A node removes itself when it stops; one that
-- did not is forgotten a day after it was last seen.
CREATE TABLE vellev.nodes (
    node     text        PRIMARY KEY,
    format   integer     NOT NULL,
    protocol integer     NOT NULL,
    seen_at  timestamptz NOT NULL
);

-- The file format the cluster writes. It only rises, and only once every
-- node registered reads the new one, so a node that goes quiet never lets
-- it rise past what that node reads.
CREATE TABLE vellev.cluster (
    single boolean PRIMARY KEY DEFAULT true CHECK (single),
    format integer NOT NULL
);
INSERT INTO vellev.cluster (format) VALUES (1);
