// Deterministic golden tests for the proDA port. Included from `proda.rs`.
//
// Each test isolates a deterministic kernel (no RNG / no optimizer) and pins it
// cell-exact to the Python `proda.py` value captured verbatim from
// `conda run -n Bigbio python`. The optimizer-driven fixed point is covered by
// the end-to-end parity tests in the golden-tests crate.
mod proda_unit_tests {
    use super::*;

    fn close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs().max(1.0),
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    // proda.py:36 _invprobit, all four flag combos across both zeta signs.
    #[test]
    fn invprobit_matches_python() {
        // (x, rho, zeta, log, oneminus, expected)
        let cases: [(f64, f64, f64, bool, bool, f64); 16] = [
            (1.0, 0.5, 2.0, false, false, 0.5987063256829237),
            (1.0, 0.5, 2.0, false, true, 0.4012936743170763),
            (1.0, 0.5, 2.0, true, false, -0.5129840754094305),
            (1.0, 0.5, 2.0, true, true, -0.913061764811135),
            (1.0, 0.5, -2.0, false, false, 0.4012936743170763),
            (1.0, 0.5, -2.0, false, true, 0.5987063256829237),
            (1.0, 0.5, -2.0, true, false, -0.913061764811135),
            (1.0, 0.5, -2.0, true, true, -0.5129840754094305),
            (-3.0, 0.0, 1.5, false, false, 0.022750131948179195),
            (-3.0, 0.0, 1.5, false, true, 0.9772498680518208),
            (-3.0, 0.0, 1.5, true, false, -3.7831843336820317),
            (-3.0, 0.0, 1.5, true, true, -0.023012909328963476),
            (5.0, 2.0, -0.3, false, false, 7.61985302416047e-24),
            (5.0, 2.0, -0.3, false, true, 1.0),
            (5.0, 2.0, -0.3, true, false, -53.23128515051248),
            (5.0, 2.0, -0.3, true, true, -7.61985302416047e-24),
        ];
        for (x, rho, zeta, log, oneminus, expected) in cases {
            close(invprobit(x, rho, zeta, log, oneminus), expected, 1e-11);
        }
    }

    // proda.py:54 _trimmed_mean, including NaN filtering and trim=0.
    #[test]
    fn trimmed_mean_matches_python() {
        let nan = f64::NAN;
        let cases: [(&[f64], f64, f64); 8] = [
            (&[1.0, 2.0, 3.0, 4.0, 5.0], 0.2, 3.0),
            (&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 3.0),
            (
                &[10.0, -1.0, 3.0, 2.0, 8.0, 7.0, 0.0, 5.0, 4.0, 6.0],
                0.2,
                4.5,
            ),
            (
                &[10.0, -1.0, 3.0, 2.0, 8.0, 7.0, 0.0, 5.0, 4.0, 6.0],
                0.0,
                4.4,
            ),
            (&[nan, 2.0, 3.0, nan, 1.0, 5.0, 4.0], 0.2, 3.0),
            (&[nan, 2.0, 3.0, nan, 1.0, 5.0, 4.0], 0.0, 3.0),
            (&[42.0], 0.2, 42.0),
            (&[42.0], 0.0, 42.0),
        ];
        for (a, trim, expected) in cases {
            close(trimmed_mean(a, trim), expected, 1e-12);
        }
    }

    fn two_group_design() -> Design {
        Design::two_group(3, 3)
    }

    // proda.py:489 _ols_per_protein: Pred / Pred_var / s2 / df cell-exact.
    #[test]
    fn ols_per_protein_matches_python() {
        let x = two_group_design();
        let mat = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9],
            vec![15.0, 15.1, 14.9, 13.0, 13.2, 12.8],
            vec![8.0, 8.1, 7.9, 8.05, 7.95, 8.0],
        ];
        let fit = ols_per_protein(&mat, &x);
        // Pred row 0.
        let exp_pred0 = [
            10.000000000000002,
            10.000000000000002,
            10.000000000000002,
            12.000000000000004,
            12.000000000000004,
            12.000000000000004,
        ];
        for (got, want) in fit.pred[0].iter().zip(exp_pred0) {
            close(*got, want, 1e-12);
        }
        // Pred_var rows are constant per protein (leverage * s2).
        close(fit.pred_var[0][0], 0.008333333333333273, 1e-10);
        close(fit.pred_var[2][0], 0.002083333333333326, 1e-10);
        // s2 / df.
        close(fit.s2[0], 0.02499999999999982, 1e-10);
        close(fit.s2[2], 0.006249999999999978, 1e-10);
        close(fit.df[0], 4.0, 1e-12);
    }

    // proda.py:114 _location_prior on the OLS output of the matrix above.
    #[test]
    fn location_prior_matches_python() {
        let x = two_group_design();
        let mat = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9],
            vec![15.0, 15.1, 14.9, 13.0, 13.2, 12.8],
            vec![8.0, 8.1, 7.9, 8.05, 7.95, 8.0],
        ];
        let fit = ols_per_protein(&mat, &x);
        let (mu0, sigma20) = location_prior(&fit.pred, &fit.pred_var, None, None);
        close(mu0, 10.750000000000002, 1e-10);
        close(sigma20, 8.220833333333342, 1e-6);
    }

    // proda.py:148 _variance_prior (Nelder-Mead on the F-logpdf objective).
    // Optimizer-driven, so a 1e-3 relative match to Python is the target.
    #[test]
    fn variance_prior_matches_python() {
        let s2 = [0.04, 0.1, 0.02, 0.5, 0.3, 0.07, 0.15, 0.25];
        let df = [4.0; 8];
        let (tau20, df0) = variance_prior(&s2, &df);
        close(tau20, 0.13463220616071536, 1e-3);
        close(df0, 7.346049806758587, 1e-3);
    }

    /// Build the regularized [`FitContext`] used by the fixed-parameter
    /// likelihood tests (one dropout entry at sample index 2).
    fn fixed_context(x: &Design) -> FitContext<'_> {
        let y: [f64; 6] = [10.0, 10.2, f64::NAN, 12.0, 12.1, 11.9];
        let rho: [f64; 6] = [9.0, 9.1, 9.2, 9.3, 9.4, 9.5];
        let zeta: [f64; 6] = [-1.2, -1.3, -1.1, -1.0, -0.9, -0.8];
        let priors = Priors {
            mu0: 11.0,
            sigma20: 0.7,
            tau20: 0.1,
            df0: 4.0,
        };
        let obs: Vec<bool> = y.iter().map(|v| !v.is_nan()).collect();
        let xo = x.select_rows(|i| obs[i]);
        let yo: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();
        let mut xm_d = Vec::new();
        let mut rm = Vec::new();
        let mut zm = Vec::new();
        for i in 0..y.len() {
            if !obs[i] && !rho[i].is_nan() {
                xm_d.push(x.row(i).to_vec());
                rm.push(rho[i]);
                zm.push(zeta[i]);
            }
        }
        FitContext {
            p: 2,
            n: 6,
            xo,
            yo,
            x_full: x,
            has_dropout: true,
            xm_d,
            rm,
            zm,
            loc: Some(priors),
            var: Some(priors),
        }
    }

    // proDA 1.24.0 negative_log_likelihood, including the prior constant.
    #[test]
    fn neg_ll_matches_official_proda() {
        let x = two_group_design();
        let ctx = fixed_context(&x);
        let par = [11.5, 11.7, 0.3];
        close(neg_ll(&ctx, &par), 20.466856612064582, 1e-9);
    }

    // Analytic gradient at fixed [beta0, beta1, sigma2].
    #[test]
    fn gradient_at_fixed_parameters() {
        let x = two_group_design();
        let ctx = fixed_context(&x);
        let par = [11.5, 11.7, 0.3];
        let g = gradient(&ctx, &par);
        let expected = [
            13.725761679476664,
            0.24324324324323499,
            -3.6368896571234006 / 0.3,
        ];
        for (got, want) in g.iter().zip(expected) {
            close(*got, want, 1e-9);
        }
    }

    // proda.py:410 _analytic_hess_bb at fixed beta / sigma2.
    #[test]
    fn analytic_hess_bb_matches_python() {
        let x = two_group_design();
        let ctx = fixed_context(&x);
        let beta = [11.5, 11.7];
        let zetastar2: Vec<f64> = ctx.zm.iter().map(|z| z * z + 0.3).collect();
        let h = analytic_hess_bb(&ctx, &beta, 0.3, &zetastar2);
        close(h[0][0], -11.267890794130999, 1e-9);
        close(h[0][1], 0.0, 1e-12);
        close(h[1][0], 0.0, 1e-12);
        close(h[1][1], -12.88010017739748, 1e-9);
    }

    // 2x2 inverse sanity (the linalg backing the fit).
    #[test]
    fn invert_round_trips() {
        let a = vec![vec![4.0, 1.0], vec![2.0, 3.0]];
        let Some(inv) = invert(&a) else {
            panic!("non-singular");
        };
        // A * A^-1 == I.
        let prod = [
            [
                a[0][0] * inv[0][0] + a[0][1] * inv[1][0],
                a[0][0] * inv[0][1] + a[0][1] * inv[1][1],
            ],
            [
                a[1][0] * inv[0][0] + a[1][1] * inv[1][0],
                a[1][0] * inv[0][1] + a[1][1] * inv[1][1],
            ],
        ];
        close(prod[0][0], 1.0, 1e-12);
        close(prod[0][1], 0.0, 1e-12);
        close(prod[1][0], 0.0, 1e-12);
        close(prod[1][1], 1.0, 1e-12);
        let singular = [vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert(&singular).is_none());
    }
}
