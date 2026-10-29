//! One-bit codes laid out in blocks of rows, so that a query is weighed
//! against a whole block at once: the rows' codes byte by byte, each byte
//! of a block holding the same byte of [`BLOCK`] rows, and the query as a
//! table of what each four bits of a code add.

/// How many rows a block holds: one byte of each in a 32-byte register.
pub(crate) const BLOCK: usize = 32;

/// How many byte columns are summed in 16-bit lanes before the sums are
/// carried into 32 bits: each adds at most 2 × 60, and 256 × 120 is under
/// 2¹⁶. A code of 8,192 bits, the widest, has 1,024.
const COLUMNS_A_RUN: usize = 256;

/// The one-bit codes of a cluster's rows, laid out a block of [`BLOCK`]
/// rows at a time: for each byte of a code in turn, that byte of every row
/// of the block. The last block is filled out with zeros. Each row's count
/// of set bits is kept beside, as the estimate needs it and no query
/// changes it.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Blocks {
    bytes: Vec<u8>,
    ones: Vec<u16>,
    rows: usize,
    words: usize,
}

impl Blocks {
    /// `codes`, `words` 64-bit words a row, laid out in blocks. A code is
    /// at most 8,192 bits, so its count of set bits fits 16.
    pub(crate) fn new(codes: &[u64], words: usize) -> Self {
        let rows = codes.len().checked_div(words).unwrap_or(0);
        debug_assert_eq!(rows * words, codes.len(), "codes are whole rows");
        debug_assert!(
            words * 64 <= usize::from(u16::MAX),
            "a code's ones fit 16 bits"
        );
        let stride = words * 8 * BLOCK;
        let mut bytes = vec![0u8; rows.div_ceil(BLOCK) * stride];
        let mut ones = Vec::with_capacity(rows);
        for (row, code) in codes.chunks_exact(words.max(1)).take(rows).enumerate() {
            ones.push(code.iter().map(|word| word.count_ones()).sum::<u32>() as u16);
            let base = (row / BLOCK) * stride + row % BLOCK;
            for (at, byte) in code.iter().flat_map(|word| word.to_le_bytes()).enumerate() {
                bytes[base + at * BLOCK] = byte;
            }
        }
        Self {
            bytes,
            ones,
            rows,
            words,
        }
    }

    #[cfg(test)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// 64-bit words a row.
    pub(crate) fn words(&self) -> usize {
        self.words
    }

    /// How many blocks there are.
    pub(crate) fn count(&self) -> usize {
        self.rows.div_ceil(BLOCK)
    }

    /// Block `block`'s bytes: for each byte of a code, that byte of each of
    /// its rows.
    pub(crate) fn block(&self, block: usize) -> &[u8] {
        let stride = self.words * 8 * BLOCK;
        &self.bytes[block * stride..][..stride]
    }

    /// Row `row`'s code, into `code`, `words` long.
    pub(crate) fn code(&self, row: usize, code: &mut [u64]) {
        let stride = self.words * 8 * BLOCK;
        let base = (row / BLOCK) * stride + row % BLOCK;
        for (word, out) in code.iter_mut().enumerate().take(self.words) {
            let mut bytes = [0u8; 8];
            for (at, byte) in bytes.iter_mut().enumerate() {
                *byte = self.bytes[base + (word * 8 + at) * BLOCK];
            }
            *out = u64::from_le_bytes(bytes);
        }
    }

    /// Every row's code, one after another, as they were written.
    pub(crate) fn codes(&self) -> Vec<u64> {
        let mut codes = vec![0u64; self.rows * self.words];
        for (row, code) in codes
            .chunks_exact_mut(self.words.max(1))
            .enumerate()
            .take(self.rows)
        {
            self.code(row, code);
        }
        codes
    }

    /// About how many bytes the blocks take in memory.
    pub(crate) fn resident(&self) -> usize {
        self.bytes.capacity() + self.ones.capacity() * std::mem::size_of::<u16>()
    }

    /// Row `row`'s count of set bits.
    pub(crate) fn ones(&self, row: usize) -> u32 {
        u32::from(self.ones[row])
    }

    /// The set-bit counts of block `block`'s rows.
    pub(crate) fn ones_of_block(&self, block: usize) -> &[u16] {
        &self.ones[block * BLOCK..self.rows.min((block + 1) * BLOCK)]
    }
}

/// A query's levels, one a dimension, as tables of what each four bits of
/// a code add: for each byte of a code, a table of 16 for its low four
/// bits and one for its high four, each written twice over, as a 32-byte
/// lookup reads it. An entry is the sum of at most four levels of at most
/// 15, so a byte holds it.
#[derive(Debug, Clone, Default)]
pub(crate) struct Tables {
    bytes: Vec<u8>,
    columns: usize,
}

impl Tables {
    /// The tables for `levels`, as many as a code has bits, each at most
    /// 15.
    #[cfg(test)]
    pub(crate) fn new(levels: &[u8]) -> Self {
        let mut tables = Self::default();
        tables.fill(levels);
        tables
    }

    /// These tables made over for `levels`, reusing their space.
    pub(crate) fn fill(&mut self, levels: &[u8]) {
        let columns = levels.len() / 8;
        self.columns = columns;
        self.bytes.resize(columns * 64, 0);
        let bytes = &mut self.bytes;
        for (column, levels) in levels.chunks_exact(8).enumerate() {
            for (half, levels) in levels.chunks_exact(4).enumerate() {
                // Entry n adds the levels of n's set bits: all 16 entries
                // at once as the bytes of one 128-bit integer, each level
                // copied into every byte and kept in the entries with its
                // bit set. No byte reaches 256, so none carries into the
                // next.
                let table = levels
                    .iter()
                    .zip(WITH_BIT)
                    .fold(0u128, |table, (level, with)| {
                        table + ((u128::from(*level) * EVERY_BYTE) & with)
                    })
                    .to_le_bytes();
                let at = column * 64 + half * 32;
                bytes[at..at + 16].copy_from_slice(&table);
                bytes[at + 16..at + 32].copy_from_slice(&table);
            }
        }
    }
}

/// A one in every byte of a 128-bit integer.
const EVERY_BYTE: u128 = u128::MAX / 255;

/// For each of a nibble's four bits, the bytes of a 16-entry table whose
/// entry's nibble has that bit set, all ones, the rest zero.
const WITH_BIT: [u128; 4] = {
    let mut masks = [0u128; 4];
    let mut bit = 0;
    while bit < 4 {
        let mut nibble = 0;
        while nibble < 16 {
            if nibble >> bit & 1 == 1 {
                masks[bit] |= 0xff << (8 * nibble);
            }
            nibble += 1;
        }
        bit += 1;
    }
    masks
};

/// Weighs a block against a query's tables: for each row of the block, the
/// sum of the levels of its set bits.
pub(crate) type Weigh = fn(&[u8], &Tables, &mut [u32; BLOCK]);

/// One byte column at a time, row by row: any machine.
fn weighed_portably(block: &[u8], tables: &Tables, weighed: &mut [u32; BLOCK]) {
    assert!(
        block.len() >= tables.columns * BLOCK,
        "a block of the tables' width"
    );
    *weighed = [0; BLOCK];
    for column in 0..tables.columns {
        let bytes = &block[column * BLOCK..][..BLOCK];
        let low = &tables.bytes[column * 64..][..16];
        let high = &tables.bytes[column * 64 + 32..][..16];
        for (sum, byte) in weighed.iter_mut().zip(bytes) {
            *sum +=
                u32::from(low[usize::from(byte & 15)]) + u32::from(high[usize::from(byte >> 4)]);
        }
    }
}

/// A byte column of all 32 rows at a time, each row's four-bit halves
/// looked up in its tables by one shuffle each, summed in 16-bit lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn weighed_with_avx2(block: &[u8], tables: &Tables, weighed: &mut [u32; BLOCK]) {
    use std::arch::x86_64::*;
    assert!(
        block.len() >= tables.columns * BLOCK,
        "a block of the tables' width"
    );
    assert!(
        tables.bytes.len() >= tables.columns * 64,
        "a table for each column"
    );
    let nibbles = _mm256_set1_epi8(0x0f);
    let low_bytes = _mm256_set1_epi16(0x00ff);
    *weighed = [0; BLOCK];
    let mut column = 0;
    while column < tables.columns {
        let end = (column + COLUMNS_A_RUN).min(tables.columns);
        let (mut even, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        while column < end {
            // SAFETY: the asserts above keep every 32-byte read within the
            // block and the tables; reads need no alignment.
            let (codes, low, high) = unsafe {
                (
                    _mm256_loadu_si256(block.as_ptr().add(column * BLOCK).cast()),
                    _mm256_loadu_si256(tables.bytes.as_ptr().add(column * 64).cast()),
                    _mm256_loadu_si256(tables.bytes.as_ptr().add(column * 64 + 32).cast()),
                )
            };
            let lows = _mm256_and_si256(codes, nibbles);
            let highs = _mm256_and_si256(_mm256_srli_epi16(codes, 4), nibbles);
            let sum = _mm256_add_epi8(
                _mm256_shuffle_epi8(low, lows),
                _mm256_shuffle_epi8(high, highs),
            );
            // Row 2k is the low byte of 16-bit lane k, row 2k + 1 its high.
            even = _mm256_add_epi16(even, _mm256_and_si256(sum, low_bytes));
            odd = _mm256_add_epi16(odd, _mm256_srli_epi16(sum, 8));
            column += 1;
        }
        let (mut evens, mut odds) = ([0u16; 16], [0u16; 16]);
        // SAFETY: each array is 32 bytes; stores need no alignment.
        unsafe {
            _mm256_storeu_si256(evens.as_mut_ptr().cast(), even);
            _mm256_storeu_si256(odds.as_mut_ptr().cast(), odd);
        }
        for lane in 0..16 {
            weighed[2 * lane] += u32::from(evens[lane]);
            weighed[2 * lane + 1] += u32::from(odds[lane]);
        }
    }
}

/// A byte column of 16 rows at a time, each half looked up by one table
/// instruction, summed widening into 16-bit lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn weighed_with_neon(block: &[u8], tables: &Tables, weighed: &mut [u32; BLOCK]) {
    use std::arch::aarch64::*;
    assert!(
        block.len() >= tables.columns * BLOCK,
        "a block of the tables' width"
    );
    assert!(
        tables.bytes.len() >= tables.columns * 64,
        "a table for each column"
    );
    let nibbles = vdupq_n_u8(0x0f);
    *weighed = [0; BLOCK];
    let mut column = 0;
    while column < tables.columns {
        let end = (column + COLUMNS_A_RUN).min(tables.columns);
        let mut sums = [vdupq_n_u16(0); 4];
        while column < end {
            // SAFETY: the asserts above keep every 16-byte read within the
            // block and the tables.
            let (first, second, low, high) = unsafe {
                (
                    vld1q_u8(block.as_ptr().add(column * BLOCK)),
                    vld1q_u8(block.as_ptr().add(column * BLOCK + 16)),
                    vld1q_u8(tables.bytes.as_ptr().add(column * 64)),
                    vld1q_u8(tables.bytes.as_ptr().add(column * 64 + 32)),
                )
            };
            let first = vaddq_u8(
                vqtbl1q_u8(low, vandq_u8(first, nibbles)),
                vqtbl1q_u8(high, vshrq_n_u8::<4>(first)),
            );
            let second = vaddq_u8(
                vqtbl1q_u8(low, vandq_u8(second, nibbles)),
                vqtbl1q_u8(high, vshrq_n_u8::<4>(second)),
            );
            sums[0] = vaddw_u8(sums[0], vget_low_u8(first));
            sums[1] = vaddw_high_u8(sums[1], first);
            sums[2] = vaddw_u8(sums[2], vget_low_u8(second));
            sums[3] = vaddw_high_u8(sums[3], second);
            column += 1;
        }
        for (part, sum) in sums.into_iter().enumerate() {
            let mut lanes = [0u16; 8];
            // SAFETY: the array is 16 bytes.
            unsafe { vst1q_u16(lanes.as_mut_ptr(), sum) };
            for (lane, value) in lanes.into_iter().enumerate() {
                weighed[part * 8 + lane] += u32::from(value);
            }
        }
    }
}

/// Every way of weighing a block this machine can run, by name.
pub(crate) fn ways_here() -> Vec<(&'static str, Weigh)> {
    #[allow(unused_mut)]
    let mut ways: Vec<(&'static str, Weigh)> = vec![("portable", weighed_portably)];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: handed out only where the CPU has just said it has AVX2,
        // which is all the function asks.
        ways.push(("avx2", |block, tables, weighed| unsafe {
            weighed_with_avx2(block, tables, weighed)
        }));
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: handed out only where the CPU has just said it has NEON.
        ways.push(("neon", |block, tables, weighed| unsafe {
            weighed_with_neon(block, tables, weighed)
        }));
    }
    ways
}

/// The way of weighing chosen for this machine, by name: the last, so the
/// widest, of [`ways_here`]. Decided once per process.
pub(crate) fn chosen() -> (&'static str, Weigh) {
    static CHOSEN: std::sync::OnceLock<(&'static str, Weigh)> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| *ways_here().last().expect("the portable way is always here"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_words(seed: u64, count: usize) -> Vec<u64> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect()
    }

    /// Each row's levels summed over its set bits, one bit at a time: what
    /// the query's planes count, and what any way of weighing must equal.
    fn weighed_bit_by_bit(code: &[u64], levels: &[u8]) -> u32 {
        (0..levels.len())
            .filter(|d| code[d / 64] >> (d % 64) & 1 == 1)
            .map(|d| u32::from(levels[d]))
            .sum()
    }

    proptest::proptest! {
        /// Codes laid out in blocks read back as they were written, whole
        /// and row by row, with each row's set bits.
        #[test]
        fn codes_laid_out_in_blocks_read_back_as_written(
            seed in 0u64..u64::MAX,
            rows in 0usize..100,
            words in proptest::prop_oneof![1usize..20, 32usize..=48, 120usize..=128],
        ) {
            let codes = random_words(seed, rows * words);
            let blocks = Blocks::new(&codes, words);
            proptest::prop_assert_eq!(blocks.rows(), rows);
            proptest::prop_assert_eq!(blocks.count(), rows.div_ceil(BLOCK));
            proptest::prop_assert_eq!(blocks.codes(), codes.clone());
            let mut code = vec![0u64; words];
            for row in 0..rows {
                blocks.code(row, &mut code);
                let written = &codes[row * words..][..words];
                proptest::prop_assert_eq!(&code[..], written);
                let ones: u32 = written.iter().map(|word| word.count_ones()).sum();
                proptest::prop_assert_eq!(blocks.ones(row), ones);
            }
        }

        /// Every way of weighing a block this machine has gives each row
        /// the sum of the levels of its set bits, to the integer: the same
        /// count the query's planes give, so the same estimate.
        #[test]
        fn every_way_of_weighing_a_block_sums_each_rows_levels(
            seed in 0u64..u64::MAX,
            rows in 1usize..100,
            words in proptest::prop_oneof![1usize..20, 32usize..=48, 120usize..=128],
            top in 0u8..=15,
        ) {
            let codes = random_words(seed, rows * words);
            let levels: Vec<u8> = random_words(seed ^ 0x9e37_79b9, words * 64)
                .into_iter()
                .map(|word| (word % (u64::from(top) + 1)) as u8)
                .collect();
            let blocks = Blocks::new(&codes, words);
            let tables = Tables::new(&levels);
            let ways = ways_here();
            proptest::prop_assert!(ways.iter().any(|(name, _)| *name == "portable"));
            for (name, weigh) in ways {
                for block in 0..blocks.count() {
                    let mut weighed = [0u32; BLOCK];
                    weigh(blocks.block(block), &tables, &mut weighed);
                    for (lane, got) in weighed.iter().enumerate() {
                        let row = block * BLOCK + lane;
                        if row >= rows {
                            break;
                        }
                        let want = weighed_bit_by_bit(&codes[row * words..][..words], &levels);
                        proptest::prop_assert_eq!(*got, want, "{} row {}", name, row);
                    }
                }
            }
        }
    }

    /// The widest code with every bit set, against a query at the top level
    /// in every dimension, sums past what 16 bits hold: 8,192 × 15 =
    /// 122,880 a row. Every way gives it whole, carrying its 16-bit sums
    /// into 32 bits before they overflow. Random codes and levels, as
    /// above, stay far below 2¹⁶ and never show a missing carry.
    #[test]
    fn every_way_carries_the_widest_codes_full_sums() {
        let (words, rows) = (128, 40);
        let codes = vec![u64::MAX; rows * words];
        let levels = vec![15u8; words * 64];
        let blocks = Blocks::new(&codes, words);
        let tables = Tables::new(&levels);
        for (name, weigh) in ways_here() {
            for block in 0..blocks.count() {
                let mut weighed = [0u32; BLOCK];
                weigh(blocks.block(block), &tables, &mut weighed);
                let lanes = (rows - block * BLOCK).min(BLOCK);
                assert_eq!(&weighed[..lanes], &vec![122_880u32; lanes][..], "{name}");
            }
        }
    }

    /// The way chosen for this machine is the widest it has.
    #[test]
    fn the_way_chosen_is_the_widest_this_machine_has() {
        let ways = ways_here();
        assert_eq!(chosen().0, ways.last().unwrap().0);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(chosen().0, "avx2");
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(chosen().0, "neon");
    }
}
