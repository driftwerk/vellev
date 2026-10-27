//! Combining the results of several searches into one ranking, as Weaviate
//! does.

use serde::{Deserialize, Serialize};

/// How far down a list a result must be for its share to halve under
/// [`Fusion::Ranked`], near enough.
const RANK_OFFSET: f32 = 60.0;

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fusion {
    /// Each search's values are spread between 0 for its worst result and 1
    /// for its best.
    #[default]
    RelativeScore,
    /// Each search's results count by their place in it alone.
    Ranked,
}

impl Fusion {
    /// What each result of one search adds to its object's fused score.
    /// `values` are the search's results best first, larger being better,
    /// and `weight` is the most any of them adds.
    pub fn shares(self, weight: f32, values: &[f32]) -> Vec<f32> {
        match self {
            Self::RelativeScore => {
                let values: Vec<f64> = values.iter().map(|value| bounded(*value)).collect();
                let (worst, best) = values.iter().fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(worst, best), value| (worst.min(*value), best.max(*value)),
                );
                let spread = best - worst;
                let share = |value: &f64| {
                    if spread > 0.0 {
                        weight * ((value - worst) / spread) as f32
                    } else {
                        weight
                    }
                };
                values.iter().map(share).collect()
            }
            Self::Ranked => (0..values.len())
                .map(|rank| weight / (rank as f32 + RANK_OFFSET))
                .collect(),
        }
    }
}

/// One object's place in a fused ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fused<K> {
    pub key: K,
    /// Larger is better.
    pub score: f32,
    /// What the vector search gave it, when it offered it: larger is nearer.
    pub nearness: Option<f32>,
    /// What the keyword search gave it, when it offered it.
    pub keyword: Option<f32>,
}

impl<K> Fused<K> {
    /// What orders objects with the same score, as in Weaviate: the
    /// wording's value where there is one, otherwise the vector's.
    fn second(&self) -> f32 {
        self.keyword.or(self.nearness).unwrap_or(f32::MIN)
    }
}

/// The best `limit` of the objects two searches offered, fused: those
/// `by_meaning` offered (with their nearness, best first) share `alpha`,
/// those `by_wording` offered (with their keyword scores, best first)
/// share the rest, each as `fusion` gives it, and an object both offered
/// adds both. Equal scores are ordered by the keyword score, or without
/// one the nearness, and then by key. The whole of a hybrid search but
/// the searching, so that whoever needs a hybrid answer from legs found
/// otherwise fuses them exactly as the engine does.
pub fn fuse<K: Copy + Ord + std::hash::Hash>(
    fusion: Fusion,
    alpha: f32,
    by_meaning: &[(K, f32)],
    by_wording: &[(K, f32)],
    limit: usize,
) -> Vec<Fused<K>> {
    let nearness: Vec<f32> = by_meaning.iter().map(|(_, value)| *value).collect();
    let wording: Vec<f32> = by_wording.iter().map(|(_, value)| *value).collect();
    let meant = by_meaning
        .iter()
        .zip(fusion.shares(alpha, &nearness))
        .map(|(&(key, value), share)| (key, share, Some(value), None));
    let worded = by_wording
        .iter()
        .zip(fusion.shares(1.0 - alpha, &wording))
        .map(|(&(key, value), share)| (key, share, None, Some(value)));
    let mut fused: Vec<Fused<K>> = meant
        .chain(worded)
        .fold(
            std::collections::HashMap::<K, Fused<K>>::new(),
            |mut fused, (key, share, nearness, keyword)| {
                let entry = fused.entry(key).or_insert(Fused {
                    key,
                    score: 0.0,
                    nearness: None,
                    keyword: None,
                });
                entry.score += share;
                entry.nearness = nearness.or(entry.nearness);
                entry.keyword = keyword.or(entry.keyword);
                fused
            },
        )
        .into_values()
        .collect();
    fused.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(b.second().total_cmp(&a.second()))
            .then(a.key.cmp(&b.key))
    });
    fused.truncate(limit);
    fused
}

/// `value` as a number that can be subtracted from any other: within what
/// an `f32` holds, with one that is not a number counted as the least.
fn bounded(value: f32) -> f64 {
    let value = if value.is_nan() { f32::MIN } else { value };
    f64::from(value.clamp(f32::MIN, f32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_scores_run_from_nothing_for_the_worst_to_the_weight_for_the_best() {
        let shares = Fusion::RelativeScore.shares(0.5, &[9.0, 6.0, 3.0, 1.0]);
        assert_eq!(shares, [0.5, 0.3125, 0.125, 0.0]);
        assert_eq!(
            Fusion::RelativeScore.shares(0.75, &[-0.2, -0.4, -1.0]),
            [0.75, 0.5625, 0.0]
        );
    }

    #[test]
    fn results_that_cannot_be_told_apart_each_get_the_whole_weight() {
        assert_eq!(Fusion::RelativeScore.shares(0.25, &[4.0]), [0.25]);
        assert_eq!(
            Fusion::RelativeScore.shares(0.25, &[2.0, 2.0, 2.0]),
            [0.25; 3]
        );
        assert_eq!(Fusion::RelativeScore.shares(0.25, &[]), [] as [f32; 0]);
    }

    #[test]
    fn the_best_gets_exactly_the_weight_and_no_share_is_ever_not_a_number() {
        let shares = Fusion::RelativeScore.shares(0.10957855, &[17.71248, 5.3077054]);
        assert_eq!(shares, [0.10957855, 0.0]);
        let relative = |values: &[f32]| Fusion::RelativeScore.shares(0.75, values);
        let above_all = relative(&[f32::INFINITY, 1.0, 0.0]);
        assert_eq!((above_all[0], above_all[2]), (0.75, 0.0));
        assert!((0.0..1e-30).contains(&above_all[1]));
        assert_eq!(relative(&[1.0, 0.0, f32::NEG_INFINITY]), [0.75, 0.75, 0.0]);
        assert_eq!(relative(&[3e38, 0.0, -3e38]), [0.75, 0.375, 0.0]);
        assert_eq!(relative(&[1.0, 0.0, f32::NAN])[..2], [0.75, 0.75]);
        assert_eq!(relative(&[f32::NAN, f32::NAN]), [0.75, 0.75]);
        assert_eq!(relative(&[f32::INFINITY, f32::INFINITY]), [0.75, 0.75]);
    }

    #[test]
    fn ranked_shares_depend_on_place_alone() {
        let shares = Fusion::Ranked.shares(0.6, &[100.0, 1.0, 1.0]);
        assert_eq!(shares, [0.6 / 60.0, 0.6 / 61.0, 0.6 / 62.0]);
        assert_eq!(Fusion::Ranked.shares(0.6, &[]), [] as [f32; 0]);
    }

    fn keys<K: Copy>(fused: &[Fused<K>]) -> Vec<K> {
        fused.iter().map(|f| f.key).collect()
    }

    #[test]
    fn each_search_gives_its_weight_and_an_object_both_offered_adds_both() {
        let fused = fuse(
            Fusion::RelativeScore,
            0.75,
            &[(1, 0.9), (2, 0.5), (3, 0.1)],
            &[(3, 8.0), (4, 4.0), (1, 0.0)],
            10,
        );
        // 1: 0.75 + 0; 3: 0 + 0.25; 2: 0.75 * 0.5; 4: 0.25 * 0.5.
        assert_eq!(keys(&fused), [1, 2, 3, 4]);
        assert_eq!(fused[0].score, 0.75);
        assert_eq!(fused[1].score, 0.375);
        assert_eq!(fused[2].score, 0.25);
        assert_eq!(fused[3].score, 0.125);
        assert_eq!(fused[0].nearness, Some(0.9));
        assert_eq!(fused[0].keyword, Some(0.0));
        assert_eq!(fused[3].nearness, None);
    }

    #[test]
    fn equal_scores_are_ordered_by_keyword_then_nearness_then_key() {
        // All three alone in their search, so each gets the whole weight.
        let fused = fuse(Fusion::RelativeScore, 0.5, &[(9, 0.3)], &[(7, 2.0)], 10);
        assert_eq!(keys(&fused), [7, 9], "keyword 2.0 before nearness 0.3");
        let fused = fuse(Fusion::RelativeScore, 0.5, &[(9, 0.3), (8, 0.3)], &[], 10);
        assert_eq!(keys(&fused), [8, 9], "then by key");
    }

    #[test]
    fn one_search_alone_ranks_as_it_did() {
        let fused = fuse(
            Fusion::RelativeScore,
            0.0,
            &[],
            &[(5, 3.0), (6, 2.0), (4, 1.0)],
            10,
        );
        assert_eq!(keys(&fused), [5, 6, 4]);
        let fused = fuse(Fusion::Ranked, 1.0, &[(5, 0.1), (6, 0.9)], &[], 10);
        assert_eq!(keys(&fused), [5, 6], "ranked counts place alone");
    }

    #[test]
    fn only_the_best_limit_are_kept() {
        let meaning: Vec<(u32, f32)> = (0..50).map(|n| (n, 1.0 - n as f32 / 100.0)).collect();
        let fused = fuse(Fusion::RelativeScore, 0.75, &meaning, &[], 10);
        assert_eq!(keys(&fused), (0..10).collect::<Vec<_>>());
        assert!(fuse(Fusion::RelativeScore, 0.75, &meaning, &[], 0).is_empty());
    }

    #[test]
    fn the_default_is_relative_score_and_names_are_snake_case() {
        assert_eq!(Fusion::default(), Fusion::RelativeScore);
        assert_eq!(
            serde_json::to_string(&Fusion::RelativeScore).unwrap(),
            "\"relative_score\""
        );
        assert_eq!(
            serde_json::from_str::<Fusion>("\"ranked\"").unwrap(),
            Fusion::Ranked
        );
    }
}
