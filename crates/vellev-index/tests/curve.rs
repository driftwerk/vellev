//! The probes a curve says a query scans.

use vellev_index::curve::{Curve, Source, POINTS};

/// A curve of rows scans every cluster that held a sample's nearest row:
/// the deepest rank, plus one.
#[test]
fn a_curve_of_rows_scans_every_cluster_its_samples_needed() {
    let ranks: Vec<u32> = (0..4000).map(|n| (n * 7919 % 38) as u32).collect();
    let curve = Curve::of(Source::Rows, 400, 900, ranks).unwrap();
    assert_eq!(curve.probes(), 38);
}

/// A curve of questions scans the clusters that held the target share of
/// their nearest rows: of 1,000 ranks 0 to 999, 0.998 of them are held by
/// the first 998.
#[test]
fn a_curve_of_questions_scans_what_holds_the_target() {
    let ranks: Vec<u32> = (0..1000).rev().collect();
    let curve = Curve::of(Source::Queries, 100, 1000, ranks).unwrap();
    assert_eq!(curve.probes(), 998);
    assert_eq!(curve.ranks.len(), POINTS);
    assert_eq!(
        curve.ranks[0], 949,
        "0.95 of them are held by the first 950"
    );
    assert_eq!(*curve.ranks.last().unwrap(), 999);
}

/// A curve scans one cluster at least, and none past the last.
#[test]
fn a_curve_scans_at_least_one_cluster_and_no_more_than_there_are() {
    let curve = Curve::of(Source::Queries, 1, 5, vec![0; 10]).unwrap();
    assert_eq!(curve.probes(), 1);
    let mut wide = Curve::of(Source::Rows, 1, 5, vec![4; 10]).unwrap();
    assert_eq!(wide.probes(), 5);
    wide.clusters = 3;
    assert_eq!(wide.probes(), 3);
}

/// No curve is made of nothing, of a rank past the last cluster, or for
/// more clusters than a rank can name; and a curve fits only an index of
/// its clusters, whole and in order.
#[test]
fn a_curve_is_made_and_used_only_where_it_fits() {
    assert_eq!(Curve::of(Source::Rows, 0, 10, vec![]), None);
    assert_eq!(Curve::of(Source::Rows, 1, 10, vec![10]), None);
    assert_eq!(Curve::of(Source::Rows, 1, 70_000, vec![1]), None);
    let curve = Curve::of(Source::Rows, 1, 10, vec![3, 9]).unwrap();
    assert!(curve.fits(10));
    assert!(!curve.fits(11));
    let mut cut = curve.clone();
    cut.ranks.pop();
    assert!(!cut.fits(10));
    let mut disordered = curve.clone();
    disordered.ranks[0] = 9;
    disordered.ranks[1] = 0;
    assert!(!disordered.fits(10));
}

/// A curve is written with its source by name, and read back whole.
#[test]
fn a_curve_reads_back_as_written() {
    let curve = Curve::of(
        Source::Queries,
        400,
        931,
        (0..4000).map(|n| n % 931).collect(),
    )
    .unwrap();
    let bytes = rmp_serde::to_vec_named(&curve).unwrap();
    assert!(bytes.windows(7).any(|w| w == b"queries"));
    let back: Curve = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(back, curve);
}

/// A curve of a source a later version names is read, so that the segment
/// it belongs to is, but never fits, so that the segment scans as its
/// index was built to.
#[test]
fn a_curve_of_a_source_this_version_does_not_know_is_read_and_never_used() {
    #[derive(serde::Serialize)]
    struct Later {
        source: &'static str,
        samples: u32,
        clusters: u32,
        ranks: Vec<u16>,
    }
    let bytes = rmp_serde::to_vec_named(&Later {
        source: "calibration",
        samples: 400,
        clusters: 10,
        ranks: vec![3; POINTS],
    })
    .unwrap();
    let read: Curve = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(read.source, Source::Unknown);
    assert!(!read.fits(10));
}

/// A curve's source is written as a plain string, and read from whatever a
/// later version writes there: a name this version does not know, or a
/// value that is not a string at all, such as a source that carries data,
/// is read as unknown, never as a failure that would keep the segment's
/// manifest from being read.
#[test]
fn a_curves_source_is_a_string_and_anything_else_there_reads_as_unknown() {
    let curve = Curve::of(Source::Queries, 400, 10, vec![3; 40]).unwrap();
    let written = rmp_serde::to_vec_named(&curve).unwrap();
    // The key "source", then the string "queries", as MessagePack writes
    // short strings: a byte of 0xa0 plus the length, then the bytes.
    let mut plain = vec![0xa0 + 6];
    plain.extend_from_slice(b"source");
    plain.push(0xa0 + 7);
    plain.extend_from_slice(b"queries");
    assert!(
        written.windows(plain.len()).any(|w| w == plain.as_slice()),
        "a plain string"
    );

    #[derive(serde::Serialize)]
    enum LaterSource {
        Mixed(u32),
        Weighted { rows: f32, questions: f32 },
    }
    #[derive(serde::Serialize)]
    struct Later<S> {
        source: S,
        samples: u32,
        clusters: u32,
        ranks: Vec<u16>,
    }
    let later = |source| Later {
        source,
        samples: 400,
        clusters: 10,
        ranks: vec![3; POINTS],
    };
    for bytes in [
        rmp_serde::to_vec_named(&later(LaterSource::Mixed(7))).unwrap(),
        rmp_serde::to_vec_named(&later(LaterSource::Weighted {
            rows: 0.5,
            questions: 0.5,
        }))
        .unwrap(),
        rmp_serde::to_vec_named(&Later {
            source: 12u32,
            samples: 400,
            clusters: 10,
            ranks: vec![3; POINTS],
        })
        .unwrap(),
    ] {
        let read: Curve = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(read.source, Source::Unknown);
        assert!(!read.fits(10));
    }
}
