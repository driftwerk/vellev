//! One-bit codes laid out in blocks of rows, so that a query is weighed
//! against a whole block at once: the rows' codes byte by byte, each byte
//! of a block holding the same byte of [`BLOCK`] rows, and the query as a
//! table of what each four bits of a code add.

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

    /// The way chosen for this machine is one of those it has.
    #[test]
    fn the_way_chosen_is_one_this_machine_has() {
        let chosen = weigher() as usize;
        assert!(ways_here().iter().any(|(_, way)| *way as usize == chosen));
    }
}
