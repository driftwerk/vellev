//! Combining the results of several searches into one ranking, as Weaviate
//! does.

use serde::{Deserialize, Serialize};

/// How far down a list a result must be for its share to halve under
/// [`Fusion::Ranked`], near enough.
const RANK_OFFSET: f32 = 60.0;

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
                let (worst, best) = values.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(worst, best), value| (worst.min(*value), best.max(*value)),
                );
                let spread = best - worst;
                let share = |value: &f32| {
                    if spread > 0.0 {
                        weight * (value - worst) / spread
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
    fn ranked_shares_depend_on_place_alone() {
        let shares = Fusion::Ranked.shares(0.6, &[100.0, 1.0, 1.0]);
        assert_eq!(shares, [0.6 / 60.0, 0.6 / 61.0, 0.6 / 62.0]);
        assert_eq!(Fusion::Ranked.shares(0.6, &[]), [] as [f32; 0]);
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
