// Per-protein proDA likelihood, analytic derivatives and posterior uncertainty.
// The bounded original-coordinate optimizer follows R's default nlminb path.

/// Result of [`pd_lm_fit`] — the proDA `pd_lm.fit` dict (proda.py:186), minus
/// the `n_obs` entry, which the Rust DE output derives separately from the
/// per-group finite counts (see `de::proda_two_group`) and so is not carried
/// here.
#[derive(Clone)]
struct PdLmFit {
    coef: Vec<f64>,
    s2: f64,
    coef_var: Vec<Vec<f64>>,
    df: f64,
}

impl PdLmFit {
    /// proda.py:186 all-NaN result.
    fn nan(p: usize) -> Self {
        Self {
            coef: vec![f64::NAN; p],
            s2: f64::NAN,
            coef_var: vec![vec![f64::NAN; p]; p],
            df: f64::NAN,
        }
    }
}

/// Optional empirical-Bayes priors for the regularized fit (proda.py:178).
#[derive(Clone, Copy)]
struct Priors {
    mu0: f64,
    sigma20: f64,
    tau20: f64,
    df0: f64,
}

/// Per-protein observed/missing partition and dropout parameters, bundled so the
/// likelihood closures take one context rather than a long argument list.
struct FitContext<'a> {
    p: usize,
    n: usize,
    xo: Vec<Vec<f64>>,
    yo: Vec<f64>,
    x_full: &'a Design,
    has_dropout: bool,
    xm_d: Vec<Vec<f64>>,
    rm: Vec<f64>,
    zm: Vec<f64>,
    loc: Option<Priors>,
    var: Option<Priors>,
}

/// proda.py:178 `_pd_lm_fit`. `priors` present => regularized fit (`mod_loc` and
/// `mod_var` both active, as proDA always passes all four prior args together).
fn pd_lm_fit(y: &[f64], x: &Design, rho: &[f64], zeta: &[f64], priors: Option<Priors>) -> PdLmFit {
    let p = x.n_cols;
    let n = y.len();
    let obs: Vec<bool> = y.iter().map(|v| !v.is_nan()).collect();
    let n_obs = obs.iter().filter(|b| **b).count();
    if n_obs == 0 && priors.is_none() {
        return PdLmFit::nan(p);
    }

    let mod_loc = priors.is_some();
    let mod_var = priors.is_some();

    // Observed / missing partitions (proda.py:196-199).
    let xo = x.select_rows(|i| obs[i]);
    let yo: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();

    // Missing-row dropout params; drop entries with NaN rho (proda.py:201-226).
    let mut xm_d = Vec::new();
    let mut rm = Vec::new();
    let mut zm = Vec::new();
    let all_obs = n_obs == n;
    for i in 0..n {
        if !obs[i] && !rho[i].is_nan() {
            xm_d.push(x.row(i).to_vec());
            rm.push(rho[i]);
            zm.push(zeta[i]);
        }
    }
    let has_dropout = !all_obs && !xm_d.is_empty();

    // proda.py:204 plain-OLS fast path: no dropout, no priors.
    if all_obs && !mod_var && !mod_loc {
        return ols_only_fit(x, &xo, &yo, n, p);
    }

    let ctx = FitContext {
        p,
        n,
        xo,
        yo,
        x_full: x,
        has_dropout,
        xm_d,
        rm,
        zm,
        loc: if mod_loc { priors } else { None },
        var: if mod_var { priors } else { None },
    };

    // Initialization (proda.py:228-233).
    let mut beta_init = vec![0.0; p];
    if let Some(pr) = ctx.loc {
        beta_init[0] = pr.mu0;
    } else {
        let mean = ctx.yo.iter().sum::<f64>() / ctx.yo.len() as f64;
        beta_init.iter_mut().for_each(|b| *b = mean);
    }
    let s2_init = match ctx.var {
        Some(pr) => pr.df0 * pr.tau20 / (pr.df0 + 2.0),
        None => 1.0,
    };

    let Some(xopt) = optimize_original(&ctx, &beta_init, s2_init) else {
        return PdLmFit::nan(p);
    };

    finalize_fit(&ctx, &xopt)
}

/// proda.py:204-221 plain-OLS branch.
fn ols_only_fit(x: &Design, xo: &[Vec<f64>], yo: &[f64], n: usize, p: usize) -> PdLmFit {
    let beta = lstsq_rows(xo, yo, p);
    let resid: Vec<f64> = yo
        .iter()
        .zip(rows_times(xo, &beta))
        .map(|(y, pr)| y - pr)
        .collect();
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    let s2 = rss / ((n - p).max(1)) as f64;
    let xtxi = gram_inverse_rows(xo, p).unwrap_or_else(|| identity(p));
    let vcov: Vec<Vec<f64>> = xtxi
        .iter()
        .map(|row| row.iter().map(|v| v * s2).collect())
        .collect();
    let _ = x;
    PdLmFit {
        coef: beta,
        s2,
        coef_var: vcov,
        df: (n - p).max(1) as f64,
    }
}

/// proda.py:235 `_neg_ll`: negative log-likelihood at `par = [beta..., sigma2]`.
fn neg_ll(ctx: &FitContext, par: &[f64]) -> f64 {
    let p = ctx.p;
    let beta = &par[..p];
    let sigma2 = par[p].max(1e-100);
    let mut ll = 0.0;

    if let Some(pr) = ctx.loc {
        let scale = pr.sigma20.sqrt();
        for v in design_times(ctx.x_full, beta) {
            ll += t_logpdf(v, LOC_DF, pr.mu0, scale);
        }
    }
    if let Some(pr) = ctx.var {
        let shape = pr.df0 / 2.0;
        ll += shape * (pr.df0 * pr.tau20 / 2.0).ln() - libm::lgamma(shape);
        ll +=
            -(pr.df0 / 2.0 + 1.0) * sigma2.ln() - pr.df0 * pr.tau20 / (2.0 * sigma2) + sigma2.ln();
    }
    let sd = sigma2.sqrt();
    for (yi, mu) in ctx.yo.iter().zip(rows_times(&ctx.xo, beta)) {
        ll += norm_logpdf_scaled(*yi, mu, sd);
    }
    if ctx.has_dropout {
        let mm = rows_times(&ctx.xm_d, beta);
        for ((m, r), z) in mm.iter().zip(&ctx.rm).zip(&ctx.zm) {
            let zs = z * (1.0 + sigma2 / (z * z)).sqrt();
            ll += invprobit(*m, *r, zs, true, false);
        }
    }
    -ll
}

/// Analytic gradient in the original `[beta..., sigma2]` coordinates.
fn gradient(ctx: &FitContext, par: &[f64]) -> Vec<f64> {
    let p = ctx.p;
    let beta = &par[..p];
    let sigma2 = par[p].max(1e-100);
    let mut g_b = vec![0.0; p];
    let mut g_s = 0.0;

    let resid: Vec<f64> = ctx
        .yo
        .iter()
        .zip(rows_times(&ctx.xo, beta))
        .map(|(y, mu)| y - mu)
        .collect();
    // g_b -= (1/sigma2) X_o^T resid.
    for (row, r) in ctx.xo.iter().zip(&resid) {
        for k in 0..p {
            g_b[k] -= row[k] * r / sigma2;
        }
    }
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    g_s += ctx.yo.len() as f64 / (2.0 * sigma2) - rss / (2.0 * sigma2 * sigma2);

    if let Some(pr) = ctx.loc {
        let xb = design_times(ctx.x_full, beta);
        for (i, v) in xb.iter().enumerate() {
            let diff = v - pr.mu0;
            let term = (LOC_DF + 1.0) * diff / (LOC_DF * pr.sigma20 + diff * diff);
            let row = ctx.x_full.row(i);
            for (gk, xk) in g_b.iter_mut().zip(row) {
                *gk += xk * term;
            }
        }
    }
    if let Some(pr) = ctx.var {
        g_s += pr.df0 / (2.0 * sigma2) - pr.df0 * pr.tau20 / (2.0 * sigma2 * sigma2);
    }
    if ctx.has_dropout {
        let mm = rows_times(&ctx.xm_d, beta);
        for (idx, (m, (r, z))) in mm.iter().zip(ctx.rm.iter().zip(&ctx.zm)).enumerate() {
            let zs2 = z * z + sigma2;
            let imr = inverse_mills(m - r, zs2);
            for (gk, xk) in g_b.iter_mut().zip(&ctx.xm_d[idx]) {
                *gk -= xk * imr;
            }
            g_s += (m - r) / (2.0 * zs2) * imr;
        }
    }
    let mut g = g_b;
    g.push(g_s);
    g
}

/// The public upstream analytic-Hessian path uses `nlminb` on sigma2 itself.
/// Failed convergence returns missing estimates, as `proDA::pd_lm.fit` does.
fn optimize_original(ctx: &FitContext, beta_init: &[f64], s2_init: f64) -> Option<Vec<f64>> {
    let mut start = beta_init.to_vec();
    start.push(s2_init);
    let mut lower = vec![f64::NEG_INFINITY; ctx.p];
    lower.push(0.0);
    let result = super::port::minimize(
        &|par| neg_ll(ctx, par),
        &|par| gradient(ctx, par),
        &|par| full_analytic_hessian(ctx, par),
        &start,
        &lower,
    );
    if result.converged {
        Some(result.x)
    } else {
        None
    }
}

/// proda.py:352-486 — variance from the analytic Hessian + CF calibration, and
/// the `df_approx` / `s2_approx` derivation. `xopt` here is `[beta..., sigma2]`.
fn finalize_fit(ctx: &FitContext, xopt: &[f64]) -> PdLmFit {
    let p = ctx.p;
    let mut beta: Vec<f64> = xopt[..p].to_vec();
    let fit_sigma2 = xopt[p].max(1e-100);

    let fit_sigma2_var = sigma2_variance(ctx, &beta, fit_sigma2);
    if fit_sigma2_var < 0.0 {
        return PdLmFit::nan(p);
    }
    let mut approx = df_and_s2_approx(ctx, fit_sigma2, fit_sigma2_var);

    let (coef_var, unestimable) = coef_variance(ctx, &beta, fit_sigma2, approx.s2_approx);
    for (value, missing) in beta.iter_mut().zip(&unestimable) {
        if *missing {
            *value = f64::NAN;
        }
    }
    if unestimable.iter().all(|v| *v) {
        approx.df_approx = f64::NAN;
        approx.s2_approx = f64::NAN;
    }

    PdLmFit {
        coef: beta,
        s2: approx.s2_approx,
        coef_var,
        df: approx.df_approx,
    }
}

/// proda.py:355-387 — variance of sigma2 from the negative second derivative of
/// the log-likelihood wrt sigma2. Negative variance marks a failed fit.
fn sigma2_hessian(ctx: &FitContext, beta: &[f64], fit_sigma2: f64) -> f64 {
    let resid: Vec<f64> = ctx
        .yo
        .iter()
        .zip(rows_times(&ctx.xo, beta))
        .map(|(y, mu)| y - mu)
        .collect();
    let dss_o: f64 = resid
        .iter()
        .map(|r| (fit_sigma2 - 2.0 * r * r) / (2.0 * fit_sigma2.powi(3)))
        .sum();

    let mut dss_m = 0.0;
    if ctx.has_dropout {
        let mm = rows_times(&ctx.xm_d, beta);
        let zs2: Vec<f64> = ctx.zm.iter().map(|z| z * z + fit_sigma2).collect();
        for ((m, r), zs2_h) in mm.iter().zip(&ctx.rm).zip(&zs2) {
            let imr = inverse_mills(m - r, *zs2_h);
            let diff = m - r;
            dss_m += diff / (4.0 * zs2_h * zs2_h) * imr * (3.0 - diff * imr - diff * diff / zs2_h);
        }
    }

    let mut dss_p = 0.0;
    if let Some(pr) = ctx.var {
        dss_p = (1.0 + pr.df0 / 2.0) / fit_sigma2.powi(2)
            - pr.df0 * pr.tau20 / fit_sigma2.powi(3)
            - 1.0 / fit_sigma2.powi(2);
    }
    -(dss_p + dss_o + dss_m)
}

fn sigma2_variance(ctx: &FitContext, beta: &[f64], sigma2: f64) -> f64 {
    1.0 / sigma2_hessian(ctx, beta, sigma2)
}

/// Complete negative-log-likelihood Hessian in original [beta..., sigma2]
/// coordinates, matching proDA:::hess_fnc including beta/variance cross terms.
fn full_analytic_hessian(ctx: &FitContext, par: &[f64]) -> Vec<Vec<f64>> {
    let p = ctx.p;
    let (beta, sigma2) = (&par[..p], par[p].max(1e-100));
    let zstar = ctx.zm.iter().map(|z| z * z + sigma2).collect::<Vec<_>>();
    let bb = analytic_hess_bb(ctx, beta, sigma2, &zstar);
    let mut h = vec![vec![0.0; p + 1]; p + 1];
    for i in 0..p {
        for j in 0..p {
            h[i][j] = -bb[i][j];
        }
    }
    h[p][p] = sigma2_hessian(ctx, beta, sigma2);
    for (row, &y) in ctx.xo.iter().zip(&ctx.yo) {
        let factor = (dot(row, beta) - y) / sigma2.powi(2);
        for i in 0..p {
            h[i][p] -= row[i] * factor;
        }
    }
    for (j, row) in ctx.xm_d.iter().enumerate() {
        let diff = dot(row, beta) - ctx.rm[j];
        let z2 = zstar[j];
        let imr = inverse_mills(diff, z2);
        let factor =
            diff / (2.0 * z2) * imr.powi(2) - (z2 - diff.powi(2)) / (2.0 * z2.powi(2)) * imr;
        for i in 0..p {
            h[i][p] -= row[i] * factor;
        }
    }
    let (upper, last) = h.split_at_mut(p);
    for (i, row) in upper.iter().enumerate() {
        last[0][i] = row[p];
    }
    h
}

fn inverse_mills(diff: f64, variance: f64) -> f64 {
    let sd = variance.sqrt();
    let z = diff / sd;
    -(norm_logpdf(z) - norm_logsf(z)).exp() / sd
}

/// proda.py:389-403 outputs.
struct DfApprox {
    s2_approx: f64,
    df_approx: f64,
}

/// proda.py:389-403 — derive `(s2_approx, df_approx)` from the fitted sigma2 and
/// its variance, with the edge cases for negative s2 / tiny df / huge df.
fn df_and_s2_approx(ctx: &FitContext, fit_sigma2: f64, fit_sigma2_var: f64) -> DfApprox {
    let p = ctx.p as f64;
    let mut n_approx_raw = 2.0 * fit_sigma2.powi(2) / fit_sigma2_var;
    let rss_approx = 2.0 * fit_sigma2.powi(3) / fit_sigma2_var;
    let mut s2_approx = rss_approx / (n_approx_raw - p);
    let df_approx;

    if s2_approx < 0.0 || n_approx_raw <= p {
        let df = 0.001;
        n_approx_raw = df + p;
        s2_approx = (fit_sigma2_var * n_approx_raw.powi(3) / (2.0 * df * df)).sqrt();
        df_approx = df;
    } else if let Some(pr) = ctx.var {
        n_approx_raw -= pr.df0;
        let mut df = n_approx_raw - p + pr.df0;
        if df > 30.0 * ctx.n as f64 && df > 100.0 {
            df = f64::INFINITY;
            s2_approx = pr.tau20;
        }
        df_approx = df;
    } else {
        df_approx = n_approx_raw - p;
    }
    DfApprox {
        s2_approx,
        df_approx,
    }
}

/// proda.py:405-477 — the analytic-Hessian coefficient variance with the CF
/// calibration loop. Returns covariance and non-estimable coefficient flags.
fn coef_variance(
    ctx: &FitContext,
    beta: &[f64],
    fit_sigma2: f64,
    s2_approx: f64,
) -> (Vec<Vec<f64>>, Vec<bool>) {
    let p = ctx.p;
    let zetastar2: Vec<f64> = ctx.zm.iter().map(|z| z * z + fit_sigma2).collect();

    let hess_fit = analytic_hess_bb(ctx, beta, fit_sigma2, &zetastar2);
    let neg_hess_fit: Vec<Vec<f64>> = hess_fit
        .iter()
        .map(|row| row.iter().map(|v| -v).collect())
        .collect();
    let var_coef_fit = invert_hessian(&neg_hess_fit);

    let cf = cf_calibration(ctx, beta, fit_sigma2, &var_coef_fit);

    // Var_coef_s2a from the analytic Hessian at s2_approx (proda.py:459-467).
    let hess_s2a = analytic_hess_bb(ctx, beta, s2_approx, &zetastar2);
    let mut neg_hess_s2a: Vec<Vec<f64>> = hess_s2a
        .iter()
        .map(|row| row.iter().map(|v| -v).collect())
        .collect();
    if (0..p).any(|i| neg_hess_s2a[i][i] < 0.0) {
        neg_hess_s2a = vec![vec![0.0; p]; p];
    }
    let var_coef_s2a = invert_hessian(&neg_hess_s2a);

    // Var_coef_unbiased = CF @ Var_coef_s2a @ CF (CF diagonal) => scale row/col.
    let mut coef_var = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            coef_var[i][j] = cf[i] * var_coef_s2a[i][j] * cf[j];
        }
    }
    // proda.py:470-472 large-variance override.
    for i in 0..p {
        if var_coef_fit[i][i] > 1e6 {
            coef_var[i][i] = f64::INFINITY;
        }
    }
    (coef_var, (0..p).map(|i| var_coef_fit[i][i] > 1e6).collect())
}

/// proDA protects small diagonal entries and caps positive Hessian elements.
fn invert_hessian(hessian: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = hessian.len();
    let mut fallback = vec![vec![0.0; p]; p];
    for (i, row) in fallback.iter_mut().enumerate() {
        row[i] = f64::INFINITY;
    }
    if (0..p).all(|i| hessian[i][i] < 1e-10) {
        return fallback;
    }
    let mut h = hessian.to_vec();
    for (i, row) in h.iter_mut().enumerate() {
        row[i] = row[i].max(1e-10);
        for value in row {
            if *value > 1e10 {
                *value = 1e10;
            }
        }
    }
    invert(&h).unwrap_or(fallback)
}

/// proda.py:410-428 `_analytic_hess_bb(sigma2_val)`.
fn analytic_hess_bb(
    ctx: &FitContext,
    beta: &[f64],
    sigma2_val: f64,
    zetastar2: &[f64],
) -> Vec<Vec<f64>> {
    let p = ctx.p;
    // h_o = -(X_o^T X_o) / sigma2.
    let mut h = vec![vec![0.0; p]; p];
    for row in &ctx.xo {
        for i in 0..p {
            for j in 0..p {
                h[i][j] -= row[i] * row[j] / sigma2_val;
            }
        }
    }
    // h_p (location prior).
    if let Some(pr) = ctx.loc {
        let xb = design_times(ctx.x_full, beta);
        for (idx, v) in xb.iter().enumerate() {
            let d = v - pr.mu0;
            let denom = (LOC_DF * pr.sigma20 + d * d).powi(2);
            let t_pf = (LOC_DF * pr.sigma20 - d * d) / denom;
            let row = ctx.x_full.row(idx);
            for i in 0..p {
                for j in 0..p {
                    h[i][j] -= (LOC_DF + 1.0) * row[i] * t_pf * row[j];
                }
            }
        }
    }
    // h_m (dropout).
    if ctx.has_dropout {
        let mm = rows_times(&ctx.xm_d, beta);
        for (idx, (m, r)) in mm.iter().zip(&ctx.rm).enumerate() {
            let zs2 = zetastar2[idx];
            let imr = inverse_mills(m - r, zs2);
            let w_m = imr * imr + (m - r) / zs2 * imr;
            let xm_row = &ctx.xm_d[idx];
            for (hi, &xi) in h.iter_mut().zip(xm_row) {
                for (hij, &xj) in hi.iter_mut().zip(xm_row) {
                    *hij -= xi * w_m * xj;
                }
            }
        }
    }
    h
}

/// proda.py:433-457 — the calibration-factor (CF) loop. For each coefficient,
/// shift along its conditional standard deviation by `sqrt(out_factor * vc_corr)`
/// and rescale by the observed change in the negative log-likelihood.
fn cf_calibration(
    ctx: &FitContext,
    beta: &[f64],
    fit_sigma2: f64,
    var_coef_fit: &[Vec<f64>],
) -> Vec<f64> {
    let p = ctx.p;
    let mut cf = vec![1.0; p];
    if beta.iter().any(|b| b.is_nan()) {
        return cf;
    }
    let out_factor = 8.0;
    let mut par_fit = beta.to_vec();
    par_fit.push(fit_sigma2);
    let offset = neg_ll(ctx, &par_fit);

    for idx in 0..p {
        let vc_corr = conditional_variance(var_coef_fit, idx, p);
        if vc_corr < 0.0 || vc_corr.is_nan() {
            cf[idx] = f64::NAN;
            continue;
        }
        let mut par_shift = par_fit.clone();
        par_shift[idx] += (out_factor * vc_corr).sqrt();
        let diff_ll = (neg_ll(ctx, &par_shift) - offset).abs();
        cf[idx] = (diff_ll / (out_factor / 2.0)).powf(-0.5);
    }
    cf
}

/// proda.py:441-447 — conditional variance of coefficient `idx` given the others:
/// `V[idx,idx] - V[idx,others] V[others,others]^{-1} V[others,idx]`.
fn conditional_variance(v: &[Vec<f64>], idx: usize, p: usize) -> f64 {
    let others: Vec<usize> = (0..p).filter(|&j| j != idx).collect();
    if others.is_empty() {
        return v[idx][idx];
    }
    let v_sub: Vec<Vec<f64>> = others
        .iter()
        .map(|&i| others.iter().map(|&j| v[i][j]).collect())
        .collect();
    let v_cross: Vec<f64> = others.iter().map(|&j| v[idx][j]).collect();
    match invert(&v_sub) {
        Some(inv) => {
            let tmp = mat_vec(&inv, &v_cross);
            v[idx][idx] - dot(&v_cross, &tmp)
        }
        None => f64::NAN,
    }
}

// --- small helpers used only by the fit ---

/// `lstsq` over a plain row-major matrix (normal equations).
fn lstsq_rows(rows: &[Vec<f64>], y: &[f64], p: usize) -> Vec<f64> {
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (r, &yi) in rows.iter().zip(y) {
        for i in 0..p {
            xty[i] += r[i] * yi;
            for j in 0..p {
                xtx[i][j] += r[i] * r[j];
            }
        }
    }
    match invert(&xtx) {
        Some(inv) => mat_vec(&inv, &xty),
        None => vec![0.0; p],
    }
}

/// `(X_o^T X_o)^{-1}` over a plain row-major matrix.
fn gram_inverse_rows(rows: &[Vec<f64>], p: usize) -> Option<Vec<Vec<f64>>> {
    let mut gram = vec![vec![0.0; p]; p];
    for r in rows {
        for i in 0..p {
            for j in 0..p {
                gram[i][j] += r[i] * r[j];
            }
        }
    }
    invert(&gram)
}
