use super::super::correct::bh_adjust;
use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    matrix: Vec<Vec<f64>>,
    n_a: usize,
    bootstrap: Vec<Vec<usize>>,
    permutations: Vec<Vec<usize>>,
    statistic: Vec<f64>,
    pvalue: Vec<f64>,
    bh: Vec<f64>,
    native_fdr: Vec<f64>,
    a1: f64,
    a2: f64,
    k: usize,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn check(case: &Case) {
    let b = case.bootstrap.len() / 2;
    let stats = |indices: &[Vec<usize>]| {
        indices
            .iter()
            .map(|v| group_stats(&permute_columns(&case.matrix, v), case.n_a))
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };
    let (db, sb) = stats(&case.bootstrap);
    let (dp, sp) = stats(&case.permutations);
    let grid = build_n_grid(case.matrix.len() / 4);
    let (a1, a2, k) = optimize_statistics(b, &grid, &db, &sb, &dp, &sp);
    assert_eq!((a1, a2, k), (case.a1, case.a2, case.k));
    let (d, s) = group_stats(&case.matrix, case.n_a);
    let observed = compute_d_stat(&d, &s, a1, a2);
    let perm = dp
        .iter()
        .zip(&sp)
        .map(|(d, s)| compute_d_stat(d, s, a1, a2))
        .collect::<Vec<_>>();
    let p = calculate_p(&observed, &perm);
    let bh = bh_adjust(&p);
    let fdr = permutation_fdr(&observed, &perm);
    for (label, actual, expected) in [
        ("statistic", &observed, &case.statistic),
        ("p", &p, &case.pvalue),
        ("BH", &bh, &case.bh),
        ("native FDR", &fdr, &case.native_fdr),
    ] {
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-12, "{label}: max error {max}");
    }
}

#[test]
fn shared_samples_match_official_rots_2_2_0() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../tests/data/rots_official.json"))
            .unwrap_or_else(|e| panic!("invalid official JSON fixture: {e}"));
    for case in fixture.cases {
        check(&case);
    }
}

#[test]
fn null_reproducibility_pairs_the_two_null_halves() {
    let d = vec![
        vec![4.0, 1.0],
        vec![1.0, 4.0],
        vec![4.0, 1.0],
        vec![4.0, 1.0],
    ];
    let dp = vec![vec![1.0, 4.0]; 4];
    let s = vec![vec![1.0; 2]; 4];
    let row = fill_repro_row(None, 2, &[1], &d, &s, &dp, &s);
    assert_eq!(row.reprotable, [0.5]);
    assert_eq!(row.reprotable_p, [1.0]);
}

#[test]
fn cutoff_ties_use_the_official_pair_order_and_threshold() {
    assert_eq!(
        official_overlaps(&[5.0, 5.0, 1.0], &[1.0, 4.0, 4.0], &[1, 2]),
        [1.0, 0.5]
    );
    assert_eq!(
        official_overlaps(&[4.0, 3.0, 2.0], &[1.0, 1.0, 1.0], &[1, 2]),
        [1.0, 1.0]
    );
}

#[test]
fn missing_permutation_statistics_follow_official_sort_na_removal() {
    // ROTS 2.2.0 calculateP(c(2,1), c(3,NaN,0,1)); R sort removes NaN.
    let actual = calculate_p(&[2.0, 1.0], &[vec![3.0, f64::NAN], vec![0.0, 1.0]]);
    assert_eq!(actual, [1.0 / 3.0, 2.0 / 3.0]);
    assert_eq!(calculate_p(&[1.0], &[vec![f64::INFINITY, 0.0]]), [0.5]);
    // Preserve the matrix contract for non-estimable outputs instead of p=0.
    assert!(calculate_p(&[f64::NAN], &[vec![1.0]])[0].is_nan());
    assert!(calculate_p(&[1.0], &[vec![f64::NAN]])[0].is_nan());
}
