// proDA EM driver, independent random initialization and two-sided Wald test.
// Default model controls match Bioconductor proDA: 20 EM rounds, epsilon 1e-3.

/// Per-protein proDA statistics returned by [`run_proda`], one entry per input
/// row (in input order; the public wrapper applies the testability filter and
/// BH adjustment). `t_stat` is the Wald statistic (the proDA extra column).
pub(crate) struct ProdaStat {
    pub log2_fold_change: f64,
    pub t_stat: f64,
    pub p_value: f64,
}

/// proda.py:520 seed for the initial imputation (only seeds the EM trajectory).
const DEFAULT_SEED: u64 = 42;
/// proda.py:569 EM iteration cap.
const EM_ITERS: usize = 20;
/// proda.py:657 EM convergence threshold on the aggregate parameter change.
const EM_TOL: f64 = 1e-3;

/// The current EM parameter state (priors + per-sample dropout curves).
struct EmState {
    mu0: f64,
    sigma20: f64,
    tau20: f64,
    df0: f64,
    rho: Vec<f64>,
    zeta: Vec<f64>,
}

/// Predictions / variances / s2 / df extracted from one round of fits
/// (proda.py:603 `_extract`).
struct Extracted {
    pred: Vec<Vec<f64>>,
    pred_var: Vec<Vec<f64>>,
    s2: Vec<f64>,
    df: Vec<f64>,
}

/// proda.py:514 `run_proda`. `mat` is the protein x sample log2 matrix (group A
/// columns first, then B) with NaN for missing; returns one [`ProdaStat`] per
/// protein in input order.
fn run_proda(mat: &[Vec<f64>], n_a: usize, n_b: usize) -> Vec<ProdaStat> {
    let n_samples = n_a + n_b;
    let x = Design::two_group(n_a, n_b);

    // Independent random initialization; all later likelihoods retain missingness.
    let y_init = impute_initial(mat, n_samples, DEFAULT_SEED);

    let (fits, _, _) = fit_model(mat, &x, &y_init);
    wald_test(mat, n_samples, x.n_cols, &fits)
}

/// Returns final fits plus the EM iteration count and final convergence error.
/// Accepting the prepared initialization also permits official shared-draw tests.
fn fit_model(mat: &[Vec<f64>], x: &Design, y_init: &[Vec<f64>]) -> (Vec<PdLmFit>, usize, f64) {
    // proda.py:560 OLS init.
    let ols = ols_per_protein(y_init, x);
    // proda.py:562-564 prior init.
    let (mu0, sigma20) = location_prior(&ols.pred, &ols.pred_var, None, None);
    let (rho, zeta) = fit_dropout_curves(mat, x, &ols.pred, &ols.pred_var);
    let (tau20, df0) = variance_prior(&ols.s2, &ols.df);

    let mut state = EmState {
        mu0,
        sigma20,
        tau20,
        df0,
        rho,
        zeta,
    };

    let mut fits_reg: Vec<PdLmFit> = Vec::new();
    let (mut iterations, mut error) = (0, f64::INFINITY);
    for iteration in 0..EM_ITERS {
        let step = em_step(mat, x, &state);
        let converged = step.err < EM_TOL;
        iterations = iteration + 1;
        error = step.err;
        fits_reg = step.fits_reg;
        state = step.next_state;
        if converged {
            break;
        }
    }
    (fits_reg, iterations, error)
}

/// Official initialization uses q10 and sample SD over the entire observed
/// matrix. Draws visit missing cells in R's column-major order. Only the random
/// generator differs; the input's missing mask is retained for all likelihoods.
fn impute_initial(mat: &[Vec<f64>], n_samples: usize, seed: u64) -> Vec<Vec<f64>> {
    let observed = mat
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if observed.is_empty() {
        return mat.to_vec();
    }
    let q10 = percentile10(&observed);
    let sd5 = sample_std(&observed) / 5.0;
    let mut rng = SplitMix64::new(seed);
    let mut y = mat.to_vec();
    for col in 0..n_samples {
        for row in &mut y {
            if row[col].is_nan() {
                row[col] = q10 + sd5 * rng.next_gaussian();
            }
        }
    }
    y
}

/// One EM iteration outcome (proda.py:569-658).
struct EmStep {
    err: f64,
    fits_reg: Vec<PdLmFit>,
    next_state: EmState,
}

/// proda.py:569-658 — a single EM iteration: unregularized + regularized fits,
/// re-estimate the priors / dropout curves, and compute the change `err`.
fn em_step(mat: &[Vec<f64>], x: &Design, state: &EmState) -> EmStep {
    let n_prot = mat.len();
    let priors = Priors {
        mu0: state.mu0,
        sigma20: state.sigma20,
        tau20: state.tau20,
        df0: state.df0,
    };

    let fits_unreg: Vec<PdLmFit> = (0..n_prot)
        .map(|i| pd_lm_fit(&mat[i], x, &state.rho, &state.zeta, None))
        .collect();
    let fits_reg: Vec<PdLmFit> = (0..n_prot)
        .map(|i| pd_lm_fit(&mat[i], x, &state.rho, &state.zeta, Some(priors)))
        .collect();

    let unreg = extract(&fits_unreg, x);
    let reg = extract(&fits_reg, x);

    let (mu0_new, sigma20_new) = location_prior(
        &reg.pred,
        &reg.pred_var,
        Some(&unreg.pred),
        Some(&unreg.pred_var),
    );
    let (rho_new, zeta_new) = fit_dropout_curves(mat, x, &reg.pred, &reg.pred_var);

    let (tau20_new, df0_new) = variance_prior(&unreg.s2, &unreg.df);

    let err = em_error(
        state,
        mu0_new,
        sigma20_new,
        &rho_new,
        &zeta_new,
        tau20_new,
        df0_new,
    );

    EmStep {
        err,
        fits_reg,
        next_state: EmState {
            mu0: mu0_new,
            sigma20: sigma20_new,
            tau20: tau20_new,
            df0: df0_new,
            rho: rho_new,
            zeta: zeta_new,
        },
    }
}

/// Build predictions from current fits. Failed coefficients remain missing;
/// an earlier iteration's values must not replace a failed upstream fit.
fn extract(fits: &[PdLmFit], x: &Design) -> Extracted {
    let n_samples = x.n_rows;
    let mut pred = Vec::with_capacity(fits.len());
    let mut pred_var = Vec::with_capacity(fits.len());
    let mut s2 = vec![0.0; fits.len()];
    let mut df = vec![0.0; fits.len()];
    for (idx, fit) in fits.iter().enumerate() {
        pred.push(design_times(x, &fit.coef));
        let pv: Vec<f64> = (0..n_samples)
            .map(|j| {
                let xj = x.row(j);
                let v = mat_vec(&fit.coef_var, xj);
                dot(xj, &v)
            })
            .collect();
        pred_var.push(pv);
        s2[idx] = fit.s2;
        df[idx] = fit.df;
    }
    Extracted {
        pred,
        pred_var,
        s2,
        df,
    }
}

/// proda.py:632-651 — the aggregate parameter-change error driving convergence.
/// Each term is the squared difference of nan-mean-reduced parameter vectors;
/// scalar parameters reduce to a plain squared difference.
fn em_error(
    state: &EmState,
    mu0_new: f64,
    sigma20_new: f64,
    rho_new: &[f64],
    zeta_new: &[f64],
    tau20_new: f64,
    df0_new: f64,
) -> f64 {
    let zi_new = zeta_to_inv(zeta_new);
    let zi_old = zeta_to_inv(&state.zeta);
    let df0_term_new = if df0_new > 0.0 { 1.0 / df0_new } else { 0.0 };
    let df0_term_old = if state.df0 > 0.0 {
        1.0 / state.df0
    } else {
        0.0
    };

    sq_diff(mu0_new, state.mu0)
        + sq_diff(sigma20_new, state.sigma20)
        + mean_change_squared(rho_new, &state.rho)
        + mean_change_squared(&zi_new, &zi_old)
        + sq_diff(tau20_new, state.tau20)
        + sq_diff(df0_term_new, df0_term_old)
}

/// proda.py:632-633 `where(abs(zeta) > 1e-100, 1/zeta, nan)`.
fn zeta_to_inv(zeta: &[f64]) -> Vec<f64> {
    zeta.iter()
        .map(|z| if z.abs() > 1e-100 { 1.0 / z } else { f64::NAN })
        .collect()
}

/// `sum(new-old, na.rm=TRUE)/length(new)`, squared: missing differences
/// are removed after subtraction, so a changed missing mask is handled as in R.
fn mean_change_squared(new: &[f64], old: &[f64]) -> f64 {
    let sum: f64 = new
        .iter()
        .zip(old)
        .map(|(a, b)| a - b)
        .filter(|v| !v.is_nan())
        .sum();
    (sum / new.len() as f64).powi(2)
}

fn sq_diff(a: f64, b: f64) -> f64 {
    (a - b) * (a - b)
}

/// Official test_diff uses the full contrast covariance and design n-p df.
fn wald_test(_mat: &[Vec<f64>], n_samples: usize, p: usize, fits: &[PdLmFit]) -> Vec<ProdaStat> {
    let df = (n_samples - p) as f64;
    let contrast = [1.0, -1.0];
    fits.iter()
        .map(|fit| {
            let log2fc = fit.coef[0] - fit.coef[1];
            let variance = dot(&contrast, &mat_vec(&fit.coef_var, &contrast));
            let t = log2fc / variance.sqrt();
            ProdaStat {
                log2_fold_change: log2fc,
                t_stat: t,
                p_value: 2.0 * student_t_sf(t.abs(), df),
            }
        })
        .collect()
}

// --- statistics helpers for imputation ---

/// `np.percentile(x, 10)` (linear interpolation, the numpy default).
fn percentile10(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    // numpy: rank = (n - 1) * q, with q = 0.10.
    let rank = (n as f64 - 1.0) * 0.10;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let frac = rank - lo as f64;
        v[lo] * (1.0 - frac) + v[hi] * frac
    }
}

/// `np.std(x, ddof=1)` (sample standard deviation).
fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

/// Two-group proDA differential expression on a log2 protein x sample matrix.
///
/// `rows[i]` labels `proteins[i]`; the first `n_a` columns are condition A, the
/// next `n_b` condition B. Returns per-protein [`ProdaStat`] in INPUT order
/// (the public DE wrapper applies the testability filter and BH adjustment).
pub(crate) fn proda_run(rows: &[&[f64]], n_a: usize, n_b: usize) -> Vec<ProdaStat> {
    let mat: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    run_proda(&mat, n_a, n_b)
}
