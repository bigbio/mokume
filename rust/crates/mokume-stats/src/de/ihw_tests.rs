use super::*;

#[test]
fn auto_small_input_is_official_uniform_bh() -> Result<()> {
    let p = [0.001, 0.02, f64::NAN, 0.8];
    let cov = [1.0, 2.0, f64::NAN, 4.0];
    let result = ihw_correction(&p, &cov, 0.05, 0)?;
    let expected = bh_adjust(&p);
    for (a, b) in result.iter().zip(expected) {
        assert!((a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-15);
    }
    Ok(())
}

#[test]
fn invalid_covariate_is_not_silently_excluded() {
    assert!(ihw_correction(&[0.01, 0.2], &[1.0, f64::NAN], 0.05, 0).is_err());
    assert!(ihw_correction(&[1.1], &[1.0], 0.05, 0).is_err());
    assert!(ihw_correction(&[0.1], &[], 0.05, 0).is_err());
}

#[derive(serde::Deserialize)]
struct Fixtures {
    cases: Vec<OfficialCase>,
}
#[derive(serde::Deserialize)]
struct OfficialCase {
    name: String,
    pvalues: Vec<f64>,
    n_bins: usize,
    lambda_value: Option<f64>,
    cross_validation: bool,
    expected_weights: Vec<f64>,
    expected_adjusted: Vec<f64>,
    fold_lambdas: Option<Vec<Option<f64>>>,
}

fn fixtures() -> std::result::Result<Fixtures, serde_json::Error> {
    serde_json::from_str(include_str!("../../tests/data/ihw_official.json"))
}

fn options(case: &OfficialCase) -> IhwOptions {
    IhwOptions {
        n_bins: case.n_bins,
        folds: Some((0..case.pvalues.len()).map(|i| i % 5).collect()),
        lambdas: (!case.cross_validation).then(|| vec![case.lambda_value.unwrap_or(f64::INFINITY)]),
        ..IhwOptions::default()
    }
}

#[test]
fn cross_weighting_and_nested_cv_match_official_ihw(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    for case in fixtures()?.cases {
        let cov: Vec<f64> = (0..case.pvalues.len()).map(|i| i as f64).collect();
        let output = ihw_with_options(&case.pvalues, &cov, 0.1, &options(&case))?;
        for (actual, expected) in [
            (&output.weights, &case.expected_weights),
            (&output.adjusted_pvalues, &case.expected_adjusted),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-8 + 1e-6 * e.abs(),
                    "{}[{i}]: {a} vs {e}",
                    case.name
                );
            }
        }
        if let Some(expected) = case.fold_lambdas {
            let expected: Vec<f64> = expected
                .into_iter()
                .map(|v| v.unwrap_or(f64::INFINITY))
                .collect();
            assert_eq!(output.fold_lambdas, expected, "{}", case.name);
        }
        assert!(output.weights.iter().any(|w| (w - 1.0).abs() > 0.01));
    }
    Ok(())
}

#[test]
fn held_out_pvalues_do_not_train_their_own_weights(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(case) = fixtures()?.cases.into_iter().next() else {
        panic!("missing official fixture");
    };
    let opt = options(&case);
    let cov: Vec<f64> = (0..case.pvalues.len()).map(|i| i as f64).collect();
    let original = ihw_with_options(&case.pvalues, &cov, 0.1, &opt)?;
    let mut changed = case.pvalues;
    for i in (0..changed.len()).step_by(5) {
        changed[i] = 0.99;
    }
    let after = ihw_with_options(&changed, &cov, 0.1, &opt)?;
    for i in (0..changed.len()).step_by(5) {
        assert!((original.weights[i] - after.weights[i]).abs() < 1e-12);
    }
    for fold in 0..5 {
        let weights: Vec<f64> = original
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (i % 5 == fold).then_some(w))
            .collect();
        assert!((weights.iter().sum::<f64>() / weights.len() as f64 - 1.0).abs() < 1e-12);
    }
    Ok(())
}
