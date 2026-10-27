//! What is measured of a set of answers, and how a knob point is matched
//! to a recall target (design r2, §4 and §5). Pure functions, so that the
//! rules are tested on tiny hand-made truths.

/// Of the first `k` true ids, how many are among the first `k` ids
/// returned, and how many there were to find: a truth shorter than `k` is
/// held to its own length. Counted, so that a recall summed over queries
/// is compared with a target exactly.
pub fn found_at(k: usize, got: &[u64], truth: &[u64]) -> Recall {
    let truth = &truth[..k.min(truth.len())];
    let mut got: Vec<u64> = got[..k.min(got.len())].to_vec();
    got.sort_unstable();
    got.dedup();
    let found = got.iter().filter(|id| truth.contains(id)).count();
    Recall {
        found: found as u64,
        of: truth.len() as u64,
    }
}

/// True ids found, of those there were to find.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Recall {
    pub found: u64,
    pub of: u64,
}

impl Recall {
    pub fn share(&self) -> f64 {
        match self.of {
            0 => 1.0,
            of => self.found as f64 / of as f64,
        }
    }

    /// Whether this is at or above a target of `hundredths`.
    pub fn at_or_above(&self, hundredths: u64) -> bool {
        u128::from(self.found) * 100 >= u128::from(hundredths) * u128::from(self.of)
    }
}

impl std::ops::Add for Recall {
    type Output = Recall;
    fn add(self, other: Recall) -> Recall {
        Recall {
            found: self.found + other.found,
            of: self.of + other.of,
        }
    }
}

impl std::iter::Sum for Recall {
    fn sum<I: Iterator<Item = Recall>>(iter: I) -> Recall {
        iter.fold(Recall::default(), |a, b| a + b)
    }
}

/// Whether the first answer is the true first.
pub fn same_top_1(got: &[u64], truth: &[u64]) -> bool {
    matches!((got.first(), truth.first()), (Some(a), Some(b)) if a == b)
}

/// The nearest-rank percentile `p` (in (0, 100]) of `samples`: the
/// smallest sample at or above `p`% of them. Never interpolated.
pub fn percentile(samples: &[f64], p: f64) -> f64 {
    assert!(!samples.is_empty(), "a percentile of nothing");
    assert!(p > 0.0 && p <= 100.0, "percentile {p}");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The median of `values`, the lower middle one for an even count, so that
/// it is always one of them.
pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

/// A measured knob point: its recall, and its p99 latency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub recall: Recall,
    pub p99: f64,
}

/// The matched point for a target of `hundredths` (§5): of the points
/// measured, the one with the lowest p99 whose recall is at or above the
/// target, the first such on a tie. `None` is "not reached". Nothing is
/// interpolated: the answer is always a point that was measured.
pub fn matched(points: &[Point], hundredths: u64) -> Option<usize> {
    points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.recall.at_or_above(hundredths))
        .min_by(|(a, x), (b, y)| x.p99.total_cmp(&y.p99).then(a.cmp(b)))
        .map(|(at, _)| at)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recall_at(k: usize, got: &[u64], truth: &[u64]) -> f64 {
        found_at(k, got, truth).share()
    }

    #[test]
    fn recall_counts_true_ids_found_within_k_in_any_order() {
        assert_eq!(recall_at(4, &[1, 2, 3, 4], &[1, 2, 3, 4]), 1.0);
        assert_eq!(recall_at(4, &[4, 3, 2, 1], &[1, 2, 3, 4]), 1.0);
        assert_eq!(
            found_at(4, &[1, 9, 3, 8], &[1, 2, 3, 4]),
            Recall { found: 2, of: 4 }
        );
    }

    #[test]
    fn recall_ignores_whatever_lies_past_k_on_either_side() {
        // 5 is returned past k, and is true only past k.
        assert_eq!(recall_at(2, &[1, 9, 5], &[1, 5, 7]), 0.5);
        assert_eq!(recall_at(2, &[1, 5], &[1, 2, 5]), 0.5);
    }

    #[test]
    fn fewer_answers_than_k_count_as_missed() {
        assert_eq!(recall_at(4, &[1], &[1, 2, 3, 4]), 0.25);
        assert_eq!(recall_at(4, &[], &[1, 2, 3, 4]), 0.0);
    }

    #[test]
    fn a_truth_shorter_than_k_is_held_to_its_own_length() {
        assert_eq!(recall_at(10, &[1, 2, 3], &[1, 2]), 1.0);
        assert_eq!(recall_at(10, &[3, 1], &[1, 2]), 0.5);
    }

    #[test]
    fn a_duplicate_answer_is_found_once() {
        assert_eq!(recall_at(2, &[1, 1], &[1, 2]), 0.5);
    }

    #[test]
    fn recall_summed_over_queries_is_compared_with_a_target_exactly() {
        // 95 of 100 is 0.95 exactly, however a float sum of 0.95s rounds.
        let summed: Recall = (0..100)
            .map(|n| Recall {
                found: u64::from(n >= 5),
                of: 1,
            })
            .sum();
        assert!(summed.at_or_above(95));
        assert!(!summed.at_or_above(96));
        assert!(!Recall {
            found: 9_499,
            of: 10_000
        }
        .at_or_above(95));
    }

    #[test]
    fn same_top_1_is_the_first_answer_only() {
        assert!(same_top_1(&[1, 2], &[1, 3]));
        assert!(!same_top_1(&[2, 1], &[1, 2]));
        assert!(!same_top_1(&[], &[1]));
    }

    #[test]
    fn percentiles_are_nearest_rank_and_never_interpolated() {
        let hundred: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        assert_eq!(percentile(&hundred, 50.0), 50.0);
        assert_eq!(percentile(&hundred, 99.0), 99.0);
        assert_eq!(percentile(&hundred, 100.0), 100.0);
        let thousand: Vec<f64> = (1..=1000).map(f64::from).collect();
        assert_eq!(percentile(&thousand, 99.0), 990.0);
        assert_eq!(percentile(&[1.0, 10.0], 50.0), 1.0);
        assert_eq!(percentile(&[1.0, 10.0], 99.0), 10.0);
    }

    #[test]
    fn the_median_is_always_one_of_the_values() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.0);
    }

    /// A point of recall `found` in ten thousand.
    fn point(found: u64, p99: f64) -> Point {
        Point {
            recall: Recall { found, of: 10_000 },
            p99,
        }
    }

    #[test]
    fn the_match_is_the_lowest_p99_at_or_above_the_target() {
        let points = [
            point(9_000, 1.0),
            point(9_600, 3.0),
            point(9_700, 2.0),
            point(9_950, 5.0),
        ];
        assert_eq!(matched(&points, 95), Some(2), "not the first to reach it");
        assert_eq!(matched(&points, 99), Some(3));
    }

    #[test]
    fn a_recall_exactly_at_the_target_reaches_it_and_one_just_below_does_not() {
        assert_eq!(matched(&[point(9_500, 1.0)], 95), Some(0));
        assert_eq!(matched(&[point(9_499, 1.0)], 95), None);
    }

    #[test]
    fn a_target_no_point_reaches_is_not_reached() {
        assert_eq!(matched(&[point(5_000, 1.0), point(9_800, 2.0)], 99), None);
        assert_eq!(matched(&[], 95), None);
    }

    #[test]
    fn equal_p99s_match_the_first_measured() {
        assert_eq!(
            matched(&[point(9_900, 2.0), point(10_000, 2.0)], 99),
            Some(0)
        );
    }
}
