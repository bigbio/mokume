use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    matrix: Vec<Vec<f64>>,
    counts: Vec<f64>,
    n_a: usize,
    log2fc: Vec<f64>,
    statistic: Vec<f64>,
    pvalue: Vec<f64>,
    bh: Vec<f64>,
    loess_fitted: Vec<f64>,
    prior_df: f64,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[test]
fn gaussian_quadratic_loess_and_de_match_deqms_1_28_0() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../tests/data/deqms_official.json"))
            .unwrap_or_else(|e| panic!("invalid official JSON fixture: {e}"));
    for case in fixture.cases {
        let refs = case.matrix.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let proteins = (0..refs.len()).map(|i| i.to_string()).collect::<Vec<_>>();
        let (moderation, _) = run_limma_moderation(&refs, case.n_a, case.n_a);
        let inputs = spectra_inputs(&moderation, &case.counts);
        let fitted = loess::predict(&inputs.log_var, &inputs.x, &inputs.valid);
        let (_, df) = count_prior(&inputs, &fitted);
        assert_eq!(df, case.prior_df);
        for (a, b) in fitted.iter().zip(&case.loess_fitted) {
            assert!((a - b).abs() < 1e-10, "loess {a} != {b}");
        }
        let results = deqms_two_group(
            &proteins,
            &refs,
            case.n_a,
            case.n_a,
            &case.counts,
            0.05,
            0.5,
        );
        assert_eq!(results.len(), refs.len());
        for row in results {
            let i = row
                .protein
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid fixture protein index: {e}"));
            for (a, b) in [
                (row.log2_fold_change, case.log2fc[i]),
                (row.t_statistic, case.statistic[i]),
                (row.p_value, case.pvalue[i]),
                (row.adj_p_value, case.bh[i]),
            ] {
                assert!(
                    (a - b).abs() < 1e-10 + 1e-8 * b.abs(),
                    "protein {i}: {a} != {b}"
                );
            }
        }
    }
}
