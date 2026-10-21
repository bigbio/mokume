use super::*;
use serde_json::Value;

fn vector(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("official fixture array missing"))
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}
fn matrix(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
        .iter()
        .map(vector)
        .collect()
}
fn number(value: &Value) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("official fixture must contain a number"))
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../tests/data/proda_official.json"))
        .unwrap_or_else(|e| panic!("invalid official fixture: {e}"))
}
fn context<'a>(case: &Value, x: &'a Design, index: usize) -> FitContext<'a> {
    let y = vector(&case["matrix"][index]);
    let obs = y.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
    let missing = (0..y.len()).filter(|&i| !obs[i]).collect::<Vec<_>>();
    let rho = vector(&case["rho"]);
    let zeta = vector(&case["zeta_inv"])
        .iter()
        .map(|v| 1.0 / v)
        .collect::<Vec<_>>();
    let pv = &case["priors"];
    let priors = Priors {
        mu0: number(&pv["mu0"]),
        sigma20: number(&pv["sigma20"]),
        tau20: number(&pv["tau20"]),
        df0: 1.0 / number(&pv["df0_inv"]),
    };
    FitContext {
        p: 2,
        n: 8,
        xo: x.select_rows(|i| obs[i]),
        yo: y.into_iter().filter(|v| v.is_finite()).collect(),
        x_full: x,
        has_dropout: !missing.is_empty(),
        xm_d: missing.iter().map(|&i| x.row(i).to_vec()).collect(),
        rm: missing.iter().map(|&i| rho[i]).collect(),
        zm: missing.iter().map(|&i| zeta[i]).collect(),
        loc: Some(priors),
        var: Some(priors),
    }
}
fn close(a: f64, b: f64, label: &str) {
    if a.is_nan() || b.is_nan() {
        assert!(a.is_nan() && b.is_nan(), "{label}: finite status differs");
        return;
    }
    if a.is_infinite() || b.is_infinite() {
        assert_eq!(a, b, "{label}: infinite status differs");
        return;
    }
    assert!(
        (a - b).abs() <= 1e-9 + 1e-10 * b.abs(),
        "{label}: {a} != {b}"
    );
}

#[test]
fn original_coordinate_objective_gradient_hessian_match_official_trace() {
    let f = fixture();
    for case in f["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
    {
        let x = Design::two_group(4, 4);
        let ctx = context(case, &x, 95);
        for event in case["solver_trace"]["events"]
            .as_array()
            .unwrap_or_else(|| panic!("official fixture must contain an array"))
        {
            let par = vector(&event["x"]);
            match event["type"]
                .as_str()
                .unwrap_or_else(|| panic!("official fixture must contain a string"))
            {
                "objective" => close(neg_ll(&ctx, &par), number(&event["value"]), "objective"),
                "gradient" => {
                    for (a, b) in gradient(&ctx, &par).iter().zip(vector(&event["value"])) {
                        close(*a, b, "gradient");
                    }
                }
                "hessian" => {
                    for (a, b) in full_analytic_hessian(&ctx, &par)
                        .iter()
                        .flatten()
                        .zip(matrix(&event["value"]).iter().flatten())
                    {
                        close(*a, *b, "Hessian");
                    }
                }
                _ => panic!("unknown trace event"),
            }
        }
    }
}

#[test]
fn posterior_variance_at_shared_official_mle_matches_proda() {
    let f = fixture();
    for case in f["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
    {
        let x = Design::two_group(4, 4);
        for (i, mle) in case["first_mle"]
            .as_array()
            .unwrap_or_else(|| panic!("official fixture must contain an array"))
            .iter()
            .enumerate()
        {
            let ctx = context(case, &x, i);
            let mut par = vector(&mle["beta"]);
            par.push(number(&mle["sigma2"]));
            let fit = finalize_fit(&ctx, &par);
            let expected = &case["first_reg"][i];
            close(
                fit.s2,
                expected["s2"].as_f64().unwrap_or(f64::NAN),
                "posterior s2",
            );
            close(
                fit.df,
                expected["df"].as_f64().unwrap_or(f64::NAN),
                "moderated df",
            );
            for (a, b) in fit
                .coef_var
                .iter()
                .flatten()
                .zip(matrix(&expected["vcov"]).iter().flatten())
            {
                close(*a, *b, "posterior covariance");
            }
        }
    }
}

#[test]
fn shared_initialization_gives_the_official_hyperparameters() {
    let f = fixture();
    for case in f["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
    {
        let x = Design::two_group(4, 4);
        let y = matrix(&case["matrix"]);
        let ols = ols_per_protein(&matrix(&case["y_init"]), &x);
        let (mu, var) = location_prior(&ols.pred, &ols.pred_var, None, None);
        let (scale, df) = variance_prior(&ols.s2, &ols.df);
        let (rho, zeta) = fit_dropout_curves(&y, &x, &ols.pred, &ols.pred_var);
        let p = &case["priors"];
        close(mu, number(&p["mu0"]), "location mean");
        close(var, number(&p["sigma20"]), "location variance");
        close(scale, number(&p["tau20"]), "variance scale");
        close(1.0 / df, number(&p["df0_inv"]), "variance prior df inverse");
        for (a, b) in rho.iter().zip(vector(&case["rho"])) {
            close(*a, b, "dropout position");
        }
        for (a, b) in zeta.iter().zip(vector(&case["zeta_inv"])) {
            close(1.0 / a, b, "dropout inverse scale");
        }
    }
}

#[test]
fn port_matches_the_official_local_modes_and_convergence_counts() {
    let f = fixture();
    for case in f["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
    {
        let x = Design::two_group(4, 4);
        let ctx = context(case, &x, 95);
        let start = vector(&case["solver_trace"]["events"][0]["x"]);
        let result = super::super::port::minimize(
            &|p| neg_ll(&ctx, p),
            &|p| gradient(&ctx, p),
            &|p| full_analytic_hessian(&ctx, p),
            &start,
            &[f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
        );
        let expected = &case["solver_trace"]["result"];
        assert!(result.converged);
        assert_eq!(
            result.iterations,
            expected["iterations"]
                .as_u64()
                .unwrap_or_else(|| panic!("official fixture must contain an integer"))
                as usize
        );
        assert_eq!(
            result.evaluations,
            expected["evaluations"][0]
                .as_u64()
                .unwrap_or_else(|| panic!("official fixture must contain an integer"))
                as usize
        );
        assert_eq!(result.status, 4);
        for (a, b) in result.x.iter().zip(vector(&expected["par"])) {
            close(*a, b, "PORT solution");
        }
    }
}

#[test]
fn complete_em_with_shared_initial_draws_matches_official_results() {
    let f = fixture();
    for (index, case) in f["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
        .iter()
        .enumerate()
    {
        let x = Design::two_group(4, 4);
        let y = matrix(&case["matrix"]);
        let (fits, iterations, error) = fit_model(&y, &x, &matrix(&case["y_init"]));
        assert_eq!(iterations, [2, 4][index]);
        assert!(error < EM_TOL);
        let stats = wald_test(&y, 8, 2, &fits);
        let p = stats.iter().map(|s| s.p_value).collect::<Vec<_>>();
        let bh = super::super::correct::bh_adjust(&p);
        // Original audit tolerance, fixed before repair: rtol=1e-6, atol=1e-10.
        for (i, (stat, q)) in stats.iter().zip(bh).enumerate() {
            for (key, a) in [
                ("log2FC", stat.log2_fold_change),
                ("statistic", stat.t_stat),
                ("pvalue", stat.p_value),
                ("adjusted_pvalue", q),
            ] {
                let b = number(&case["expected"][key][i]);
                assert!(
                    (a - b).abs() <= 1e-10 + 1e-6 * b.abs(),
                    "case {index}, protein {i}, {key}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn sparse_and_nearly_constant_fits_match_official_proda() {
    let f = fixture();
    let case = &f["cases"][1];
    let x = Design::two_group(4, 4);
    let ctx = context(case, &x, 0);
    let rho = vector(&case["rho"]);
    let zeta = vector(&case["zeta_inv"])
        .iter()
        .map(|v| 1.0 / v)
        .collect::<Vec<_>>();
    for edge in f["edge_cases"]
        .as_array()
        .unwrap_or_else(|| panic!("official fixture must contain an array"))
    {
        let fit = pd_lm_fit(&vector(&edge["row"]), &x, &rho, &zeta, ctx.loc);
        let pairs = fit
            .coef
            .iter()
            .copied()
            .zip(vector(&edge["beta"]))
            .chain(
                fit.coef_var
                    .iter()
                    .flatten()
                    .copied()
                    .zip(matrix(&edge["vcov"]).into_iter().flatten()),
            )
            .chain([(fit.s2, number(&edge["s2"])), (fit.df, number(&edge["df"]))]);
        for (a, b) in pairs {
            assert!(
                (a - b).abs() <= 1e-10 + 1e-6 * b.abs(),
                "{}: {a} != {b}",
                edge["label"]
            );
        }
    }
}
