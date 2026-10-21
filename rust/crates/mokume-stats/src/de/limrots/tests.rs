use super::super::correct::bh_adjust;
use super::{
    analyze, analyze_fits, fit, limrots_two_group, limrots_two_group_with_options, LimRotsOptions,
    SamplingPlan,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(serde::Deserialize)]
struct ReferenceFit {
    contrast: Vec<f64>,
    standard_error: Vec<f64>,
}

impl ReferenceFit {
    fn to_fit(&self) -> fit::Fit {
        fit::Fit {
            contrast: self.contrast.clone(),
            standard_error: self.standard_error.clone(),
        }
    }
}

#[derive(serde::Deserialize)]
struct Expected {
    a1: f64,
    a2: f64,
    k: usize,
    #[serde(rename = "R")]
    reproducibility: f64,
    #[serde(rename = "Z")]
    z: f64,
    statistics: Vec<f64>,
    pvalues: Vec<f64>,
    adjusted: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    data: Vec<Vec<f64>>,
    n_a: usize,
    n_b: usize,
    bootstraps: Vec<Vec<usize>>,
    permutations: Vec<Vec<usize>>,
    k_values: Vec<usize>,
    full: ReferenceFit,
    boot_fits: Vec<ReferenceFit>,
    null_fits: Vec<ReferenceFit>,
    expected: Expected,
}

fn cases() -> Result<Vec<Case>, serde_json::Error> {
    serde_json::from_str(include_str!("../../../tests/data/limrots_1_2_8.json"))
}

fn close(a: f64, e: f64, atol: f64, rtol: f64, label: &str) {
    assert!(
        (a - e).abs() <= atol + rtol * e.abs(),
        "{label}: {a:.17e} != {e:.17e}"
    );
}

fn check_analysis(actual: &super::Analysis, case: &Case) {
    let expected = &case.expected;
    assert_eq!(actual.statistics.len(), expected.statistics.len());
    assert_eq!(actual.pvalues.len(), expected.pvalues.len());
    assert_eq!(actual.pvalues.len(), expected.adjusted.len());
    close(actual.optimum.a1, expected.a1, 1e-12, 0.0, &case.name);
    assert_eq!(actual.optimum.a2, expected.a2, "{} a2", case.name);
    assert_eq!(actual.optimum.k, expected.k, "{} k", case.name);
    close(
        actual.optimum.z,
        expected.z,
        1e-10,
        1e-10,
        &format!("{} Z", case.name),
    );
    close(
        actual.optimum.reproducibility,
        expected.reproducibility,
        1e-12,
        0.0,
        &case.name,
    );
    let adjusted = bh_adjust(&actual.pvalues);
    for (i, &adj) in adjusted.iter().enumerate() {
        close(
            actual.statistics[i],
            expected.statistics[i],
            1e-8,
            1e-6,
            &format!("{} statistic {i}", case.name),
        );
        close(
            actual.pvalues[i],
            expected.pvalues[i],
            1e-12,
            0.0,
            &format!("{} pvalue {i}", case.name),
        );
        close(
            adj,
            expected.adjusted[i],
            1e-12,
            0.0,
            &format!("{} BH {i}", case.name),
        );
    }
}

#[test]
fn official_fits_reproduce_optimization_and_pvalues() -> TestResult {
    for case in cases()? {
        let boots = case
            .boot_fits
            .iter()
            .map(ReferenceFit::to_fit)
            .collect::<Vec<_>>();
        let null = case
            .null_fits
            .iter()
            .map(ReferenceFit::to_fit)
            .collect::<Vec<_>>();
        let actual = analyze_fits(case.full.to_fit(), &boots, &null, &case.k_values)?;
        check_analysis(&actual, &case);
    }
    Ok(())
}

#[test]
fn shared_indices_reproduce_full_official_output() -> TestResult {
    for case in cases()? {
        let plan = SamplingPlan {
            bootstraps: case.bootstraps.clone(),
            permutations: case.permutations.clone(),
        };
        let actual = analyze(&case.data, case.n_a, &plan, &case.k_values)?;
        check_analysis(&actual, &case);
    }
    Ok(())
}

#[test]
fn degenerate_bootstrap_uses_qr_residuals() -> TestResult {
    #[derive(serde::Deserialize)]
    struct Degenerate {
        data: Vec<Vec<f64>>,
        contrast: Vec<f64>,
        standard_error: Vec<f64>,
    }
    let case: Degenerate = serde_json::from_str(include_str!(
        "../../../tests/data/limrots_degenerate_qr.json"
    ))?;
    let actual = fit::fit(&case.data, &[true, true, true, false, false, false]);
    for i in 0..case.data.len() {
        close(
            actual.contrast[i],
            case.contrast[i],
            1e-12,
            1e-12,
            "QR contrast",
        );
        close(
            actual.standard_error[i],
            case.standard_error[i],
            1e-25,
            1e-8,
            "degenerate QR standard error",
        );
    }
    Ok(())
}

#[test]
fn invalid_search_and_missing_input_return_errors() {
    let proteins = vec!["P".to_string(); 6];
    let row = [1.0; 6];
    let rows = vec![row.as_slice(); 6];
    assert!(limrots_two_group(&proteins, &rows, 3, 3, 0.05, 0.5).is_err());
    let missing = [1., 2., f64::NAN, 4., 5., 6.];
    let missing_rows = vec![missing.as_slice(); 30];
    assert!(limrots_two_group(&vec!["P".to_string(); 30], &missing_rows, 3, 3, 0.05, 0.5).is_err());
}

#[test]
fn same_seed_is_independent_of_rayon_thread_count() -> TestResult {
    let case = cases()?.remove(0);
    let proteins = (0..case.data.len())
        .map(|i| format!("P{i}"))
        .collect::<Vec<_>>();
    let rows = case.data.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let run = || {
        limrots_two_group_with_options(
            &proteins,
            &rows,
            case.n_a,
            case.n_b,
            0.05,
            0.5,
            LimRotsOptions {
                n_iterations: 12,
                ..LimRotsOptions::default()
            },
        )
    };
    let a = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()?
        .install(run)?;
    let b = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()?
        .install(run)?;
    let actual_ids = a
        .results
        .iter()
        .map(|r| &r.protein)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        actual_ids,
        proteins.iter().collect::<std::collections::BTreeSet<_>>()
    );
    assert_eq!((a.a1, a.a2, a.k, a.z_score), (b.a1, b.a2, b.k, b.z_score));
    for (x, y) in a.results.iter().zip(&b.results) {
        assert_eq!(
            (x.p_value, x.adj_p_value, x.t_statistic),
            (y.p_value, y.adj_p_value, y.t_statistic)
        );
        assert!(x.t_statistic >= 0.0);
    }
    Ok(())
}
