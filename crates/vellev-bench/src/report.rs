//! What a run publishes (design r2, §5 and §8): the matched point of each
//! engine at each target, the hard gate on vellev's defaults, and a
//! Markdown summary. The whole grid goes into `results.json` beside these.

use serde::Serialize;

use crate::engine::Knob;
use crate::fixture::Case;
use crate::measure::{matched, Point};
use crate::run::Measured;

/// The recall@10 targets, in hundredths, stated before the first run.
pub const TARGETS: [u64; 2] = [95, 99];
/// The hard gate's threshold, in hundredths (§8), as the #96 gate's.
pub const GATE: u64 = 99;

#[derive(Debug, Clone, Serialize)]
pub struct Match {
    pub engine: String,
    pub case: Case,
    pub target: f64,
    /// The matched point, or `None` for "not reached".
    pub point: Option<Measured>,
}

/// The matched point of each engine, case and target, at limit 10.
pub fn matches(engine: &str, measured: &[Measured]) -> Vec<Match> {
    let mut out = Vec::new();
    for case in Case::ALL {
        let points: Vec<&Measured> = measured
            .iter()
            .filter(|m| m.limit == 10 && m.case == case)
            .collect();
        if points.is_empty() {
            continue;
        }
        let measured_points: Vec<Point> = points
            .iter()
            .map(|m| Point {
                recall: m.recall,
                p99: m.p99_ms.median,
            })
            .collect();
        for target in TARGETS {
            out.push(Match {
                engine: engine.into(),
                case,
                target: target as f64 / 100.0,
                point: matched(&measured_points, target).map(|at| points[at].clone()),
            });
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub check: String,
    pub case: Case,
    pub value: Option<f64>,
    pub required: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    pub checks: Vec<Check>,
    /// Whether every check of the gate was measured and passed.
    pub passed: bool,
    pub missing: Vec<String>,
}

/// The hard gate (§8) on vellev at its defaults, at limit 10: vector-only
/// recall@10 and same-top-1 under every case. The fused hybrid check is
/// added where it is measured.
pub fn gate(vellev: &[Measured], cases: &[Case], extra: Vec<Check>) -> Gate {
    let mut checks = Vec::new();
    let mut missing = Vec::new();
    for &case in &Case::ALL {
        let point = vellev
            .iter()
            .find(|m| m.limit == 10 && m.case == case && m.knob == Knob::VellevDefault);
        let Some(point) = point else {
            if !cases.contains(&case) {
                missing.push(format!("{} not measured", case.name()));
            } else {
                missing.push(format!("{}: no default point at limit 10", case.name()));
            }
            continue;
        };
        checks.push(Check {
            check: "recall@10".into(),
            case,
            value: Some(point.recall.share()),
            required: GATE as f64 / 100.0,
            passed: point.recall.at_or_above(GATE),
        });
        checks.push(Check {
            check: "same-top-1".into(),
            case,
            value: Some(point.same_top_1.share()),
            required: GATE as f64 / 100.0,
            passed: point.same_top_1.at_or_above(GATE),
        });
    }
    let has_hybrid = extra.iter().any(|check| check.check.starts_with("hybrid"));
    checks.extend(extra);
    if !has_hybrid {
        missing.push("fused hybrid against vellev's exact-leg answer: not measured".into());
    }
    let passed = missing.is_empty() && checks.iter().all(|check| check.passed);
    Gate {
        checks,
        passed,
        missing,
    }
}

/// The Markdown summary: the matched points first, then the gate.
pub fn summary(matches: &[Match], gate: &Gate, rows: u64, header: &str) -> String {
    let mut out = format!("# vellev against Weaviate, {rows} rows\n\n{header}\n\n");
    out.push_str("## Matched points (recall@10, vector-only, limit 10)\n\n");
    out.push_str("| Case | Target | Engine | Point | recall@10 | p50 ms | p99 ms |\n|---|---|---|---|---|---|---|\n");
    for m in matches {
        match &m.point {
            Some(p) => out.push_str(&format!(
                "| {} | {:.2} | {} | {} | {:.4} | {:.2} | {:.2} ({:.2}–{:.2}) |\n",
                m.case.name(),
                m.target,
                m.engine,
                p.label,
                p.recall_share,
                p.p50_ms.median,
                p.p99_ms.median,
                p.p99_ms.min,
                p.p99_ms.max
            )),
            None => out.push_str(&format!(
                "| {} | {:.2} | {} | not reached | | | |\n",
                m.case.name(),
                m.target,
                m.engine
            )),
        }
    }
    out.push_str(&format!(
        "\n## Hard gate: {}\n\n| Check | Case | Value | Required | |\n|---|---|---|---|---|\n",
        if gate.passed { "passed" } else { "FAILED" }
    ));
    for check in &gate.checks {
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {} |\n",
            check.check,
            check.case.name(),
            check.value.map_or("—".into(), |v| format!("{v:.4}")),
            check.required,
            if check.passed { "pass" } else { "FAIL" }
        ));
    }
    for missing in &gate.missing {
        out.push_str(&format!("\n- Missing: {missing}"));
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::Recall;
    use crate::run::Repeated;

    fn at(case: Case, knob: Knob, found: u64, top: u64, p99: f64) -> Measured {
        let r = |v| Repeated {
            median: v,
            min: v,
            max: v,
        };
        Measured {
            limit: 10,
            case,
            knob,
            label: knob.label(),
            recall: Recall { found, of: 10_000 },
            recall_share: found as f64 / 10_000.0,
            same_top_1: Recall {
                found: top,
                of: 1_000,
            },
            p50_ms: r(p99 / 2.0),
            p99_ms: r(p99),
        }
    }

    #[test]
    fn the_gate_holds_vellevs_default_point_not_its_best() {
        let measured: Vec<Measured> = Case::ALL
            .iter()
            .flat_map(|&case| {
                [
                    at(case, Knob::VellevDefault, 9_890, 995, 1.0),
                    at(case, Knob::Oversample(128), 10_000, 1_000, 2.0),
                ]
            })
            .collect();
        let hybrid = vec![Check {
            check: "hybrid recall@10".into(),
            case: Case::Broad,
            value: Some(1.0),
            required: 0.99,
            passed: true,
        }];
        let gate = gate(&measured, &Case::ALL, hybrid);
        assert!(!gate.passed);
        assert!(gate
            .checks
            .iter()
            .filter(|c| !c.passed)
            .all(|c| c.check == "recall@10"));
    }

    #[test]
    fn a_gate_missing_a_case_or_the_hybrid_check_does_not_pass() {
        let measured: Vec<Measured> = [Case::Broad, Case::Mid, Case::Narrow]
            .iter()
            .map(|&case| at(case, Knob::VellevDefault, 10_000, 1_000, 1.0))
            .collect();
        let gate = gate(&measured, &[Case::Broad, Case::Mid, Case::Narrow], vec![]);
        assert!(!gate.passed);
        assert_eq!(gate.missing.len(), 2, "{:?}", gate.missing);
    }

    #[test]
    fn same_top_1_at_99_in_100_passes_and_below_fails() {
        let pass = at(Case::Broad, Knob::VellevDefault, 10_000, 990, 1.0);
        let fail = at(Case::Broad, Knob::VellevDefault, 10_000, 989, 1.0);
        let top = |m: Measured| gate(&[m], &[Case::Broad], vec![]).checks[1].passed;
        assert!(top(pass));
        assert!(!top(fail));
    }

    #[test]
    fn matches_are_made_at_limit_10_per_case_and_target() {
        let measured = vec![
            at(Case::Broad, Knob::Ef(16), 9_400, 900, 1.0),
            at(Case::Broad, Knob::Ef(64), 9_600, 950, 2.0),
            at(Case::Broad, Knob::Ef(512), 9_950, 999, 9.0),
        ];
        let found = matches("weaviate", &measured);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].point.as_ref().unwrap().knob, Knob::Ef(64));
        assert_eq!(found[1].point.as_ref().unwrap().knob, Knob::Ef(512));
    }
}
