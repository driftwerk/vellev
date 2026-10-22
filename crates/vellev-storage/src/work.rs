//! What a search did, counted as it does it, so that its cost can be told
//! apart from the time it took: the time depends on what else the node is
//! doing, and the work does not.

/// The work of one or more searches, summed over their layers and legs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Work {
    /// Rows of the memtable scored exactly against the query's vector.
    /// The memtable has no vector index, so this is every allowed live row
    /// of it.
    pub exact_memtable: u64,
    /// Rows of segments scored exactly against the query's vector: those
    /// a small layer offers whole, those its index shortlisted, and those
    /// it has no estimate for.
    pub exact_segments: u64,
    /// Rows a segment's clustered index estimated a distance for and
    /// offered, before the nearest of them were picked to score exactly.
    pub estimated: u64,
    /// Postings a keyword search read where it scored, or passed over, a
    /// row, and those it counted to weigh each term.
    pub postings: u64,
    /// Rows a keyword search scored.
    pub keyword_rows: u64,
    /// The rows of the layers a filtered leg searched, and how many of
    /// them its filter allowed. Their ratio is the filter's selectivity;
    /// a hybrid search counts both legs, which leaves it unchanged.
    pub filtered: u64,
    pub allowed: u64,
}

impl Work {
    /// The share of the rows a filter allowed, or `None` where no leg was
    /// filtered.
    pub fn selectivity(&self) -> Option<f64> {
        (self.filtered > 0).then(|| self.allowed as f64 / self.filtered as f64)
    }
}
