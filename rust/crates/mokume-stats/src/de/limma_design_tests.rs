use super::*;
use crate::de::deqms_design;
use serde::Deserialize;

#[derive(Deserialize)]
struct Contrast {
    weights: Vec<f64>,
    group_a: Vec<usize>,
    group_b: Vec<usize>,
    coefficient: Vec<Option<f64>>,
    stdev_unscaled: Vec<Option<f64>>,
    t: Vec<Option<f64>>,
    p_value: Vec<Option<f64>>,
    sca_t: Option<Vec<f64>>,
    sca_p: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    matrix: Vec<Vec<Option<f64>>>,
    design: Vec<Vec<f64>>,
    /// Random-block cases: block labels and R's consensus correlation.
    block: Option<Vec<usize>>,
    consensus: Option<f64>,
    counts: Option<Vec<f64>>,
    sigma: Vec<Option<f64>>,
    df_residual: Vec<f64>,
    df_prior: f64,
    contrasts: Vec<Contrast>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../tests/data/limma_design_official.json"))
        .unwrap_or_else(|e| panic!("invalid official JSON fixture: {e}"))
}

/// Fixed-effects cases followed by the random-block cases.
fn all_cases() -> Vec<Case> {
    let block: Fixture =
        serde_json::from_str(include_str!("../../tests/data/limma_block_official.json"))
            .unwrap_or_else(|e| panic!("invalid official JSON fixture: {e}"));
    fixture().cases.into_iter().chain(block.cases).collect()
}

fn matrix(case: &Case) -> Vec<Vec<f64>> {
    case.matrix
        .iter()
        .map(|row| row.iter().map(|value| value.unwrap_or(f64::NAN)).collect())
        .collect()
}

fn spec<'a>(case: &'a Case, contrast: &'a Contrast) -> DesignContrast<'a> {
    DesignContrast {
        design: &case.design,
        weights: &contrast.weights,
        group_a: &contrast.group_a,
        group_b: &contrast.group_b,
        block: case.block.as_deref().map(|blocks| RandomBlock {
            blocks,
            correlation: case.consensus.unwrap_or(f64::NAN),
        }),
    }
}

// Fit quantities match lm.series and gls.series to rounding. With unequal
// residual df the prior df comes from the shared squeezeVar, whose bounded
// Brent minimiser (SciPy's, xatol 1e-5) stops at a slightly different point
// than R's optimize() (tol 1.2e-4): the prior df differs by up to ~5e-5 and
// small moderated p-values by up to ~2e-4 (relative).
const FIT_TOL: f64 = 1e-9;
const MODERATED_TOL: f64 = 3e-4;

fn assert_close(actual: f64, expected: Option<f64>, tol: f64, label: &str) {
    match expected {
        Some(expected) => assert!(
            (actual - expected).abs() <= 1e-12 + tol * expected.abs(),
            "{label}: {actual} != {expected}"
        ),
        None => assert!(actual.is_nan(), "{label}: {actual} != NA"),
    }
}

#[test]
fn fits_match_limma_3_66_0() {
    for case in all_cases() {
        let rows = matrix(&case);
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        for (j, contrast) in case.contrasts.iter().enumerate() {
            let (moderations, df_prior) = run_design_moderation(&refs, &spec(&case, contrast))
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            let label = format!("{} df_prior", case.name);
            assert_close(df_prior, Some(case.df_prior), MODERATED_TOL, &label);
            assert_eq!(moderations.len(), rows.len(), "{}", case.name);
            for (i, fit) in moderations {
                let label = format!("{} contrast {j} protein {i}", case.name);
                assert_eq!(fit.df_residual, case.df_residual[i], "{label} df");
                for (actual, expected, tol, name) in [
                    (fit.sigma, case.sigma[i], FIT_TOL, "sigma"),
                    (
                        fit.log2_fold_change,
                        contrast.coefficient[i],
                        FIT_TOL,
                        "coefficient",
                    ),
                    (
                        fit.stdev_unscaled,
                        contrast.stdev_unscaled[i],
                        FIT_TOL,
                        "stdev",
                    ),
                    (fit.t_statistic, contrast.t[i], MODERATED_TOL, "t"),
                    (fit.p_value, contrast.p_value[i], MODERATED_TOL, "p"),
                ] {
                    assert_close(actual, expected, tol, &format!("{label} {name}"));
                }
            }
        }
    }
}

#[test]
fn deqms_matches_spectra_count_ebayes_1_28_0() {
    for case in all_cases() {
        let Some(counts) = &case.counts else {
            continue;
        };
        let rows = matrix(&case);
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let proteins = (0..rows.len()).map(|i| i.to_string()).collect::<Vec<_>>();
        for contrast in &case.contrasts {
            let (Some(sca_t), Some(sca_p)) = (&contrast.sca_t, &contrast.sca_p) else {
                panic!("{}: DEqMS output missing", case.name);
            };
            let results = deqms_design(&proteins, &refs, &spec(&case, contrast), counts, 0.05, 0.5)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            // Reported: a residual df and two finite values in each group.
            let reported = (0..rows.len())
                .filter(|&i| {
                    case.df_residual[i] > 0.0
                        && [&contrast.group_a, &contrast.group_b].iter().all(|group| {
                            group.iter().filter(|&&s| rows[i][s].is_finite()).count() >= 2
                        })
                })
                .count();
            assert_eq!(results.len(), reported, "{}", case.name);
            for row in results {
                let i = row
                    .protein
                    .parse::<usize>()
                    .unwrap_or_else(|e| panic!("invalid protein index: {e}"));
                let label = format!("{} protein {i}", case.name);
                assert_close(row.t_statistic, Some(sca_t[i]), FIT_TOL, &label);
                assert_close(row.p_value, Some(sca_p[i]), FIT_TOL, &label);
            }
        }
    }
}

#[test]
fn reports_proteins_with_two_values_per_group() {
    let fixture = fixture();
    let Some(case) = fixture
        .cases
        .iter()
        .find(|case| case.name == "paired_subjects")
    else {
        panic!("paired_subjects case missing");
    };
    let rows = matrix(case);
    let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let proteins = (0..rows.len()).map(|i| i.to_string()).collect::<Vec<_>>();
    let results = limma_design(&proteins, &refs, &spec(case, &case.contrasts[0]), 0.05, 0.5)
        .unwrap_or_else(|e| panic!("{e}"));
    // Protein 4 keeps one treated value; protein 2 misses one subject but stays.
    assert_eq!(results.len(), rows.len() - 1);
    assert!(results.iter().all(|row| row.protein != "4"));
    assert!(results.iter().any(|row| row.protein == "2"));
}

#[test]
fn rejects_rank_deficient_design() {
    // The covariate equals the first condition's indicator.
    let design = vec![
        vec![1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    assert!(validate_design(&design).is_err());
    assert!(validate_design(&design[..2]).is_err());
}
