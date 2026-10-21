//! Two-group probabilistic-dropout differential expression following proDA 1.24.0.
//!
//! The common scope is an unpaired two-group log2 matrix, moderated location
//! and variance, analytic-Hessian `nlminb`, and a two-sided Wald test with BH.
//! Missing values remain in the likelihood. Initial normal draws use SplitMix64
//! and Box-Muller rather than R's generator, so identical seeds do not imply
//! identical missing-data results. Shared-initialization fixtures independently
//! verify the deterministic EM path against the Bioconductor package.

include!("proda_optim.rs");
use super::rots::SplitMix64;
use super::special::{f_logpdf, norm_cdf, norm_logcdf, norm_logpdf, norm_logsf, norm_sf, t_logpdf};
use super::student_t::student_t_sf;

/// A floor mirroring `np.maximum(np.abs(zeta), 1e-100)` (proda.py:40).
const ZETA_FLOOR: f64 = 1e-100;
/// `loc_df` default for the location-prior Student-t (proda.py:178).
const LOC_DF: f64 = 3.0;
/// Number of design columns for the two-group contrast (intercept-per-group).
const N_COEF: usize = 2;

/// proda.py:36 `_invprobit` — dropout / observation probability matching R
/// `proDA:::invprobit`. The branch on `sign(zeta)` selects which tail of the
/// standard normal applies; `log` toggles log-space and `oneminus` flips
/// observation vs. dropout probability.
fn invprobit(x: f64, rho: f64, zeta: f64, log: bool, oneminus: bool) -> f64 {
    let denom = zeta.abs().max(ZETA_FLOOR);
    let z = (x - rho) / denom;
    // proda.py:39 `np.nansum(np.sign(zeta)) < 0`: for a scalar zeta this is just
    // `sign(zeta) < 0`, i.e. zeta < 0.
    if zeta < 0.0 {
        // proda.py:41-45 (zeta < 0 branch).
        if oneminus {
            if log {
                norm_logcdf(z)
            } else {
                norm_cdf(z)
            }
        } else if log {
            norm_logsf(z)
        } else {
            norm_sf(z)
        }
    } else {
        // proda.py:46-51 (zeta >= 0 branch).
        if oneminus {
            if log {
                norm_logsf(z)
            } else {
                norm_sf(z)
            }
        } else if log {
            norm_logcdf(z)
        } else {
            norm_cdf(z)
        }
    }
}

/// proda.py:54 `_trimmed_mean` — `mean(sort(finite)[lo:hi])` with
/// `lo = floor(n * trim)`, `hi = n - lo`. Empty input yields 0.0.
fn trimmed_mean(values: &[f64], trim: f64) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return 0.0;
    }
    finite.sort_by(f64::total_cmp);
    let lo = (n as f64 * trim).floor() as usize;
    let hi = n - lo;
    if hi <= lo {
        return 0.0;
    }
    let slice = &finite[lo..hi];
    slice.iter().sum::<f64>() / slice.len() as f64
}

/// Output of [`ols_per_protein`]: per-protein predictions, prediction variance,
/// residual variance, and residual degrees of freedom.
struct OlsFit {
    pred: Vec<Vec<f64>>,
    pred_var: Vec<Vec<f64>>,
    s2: Vec<f64>,
    df: Vec<f64>,
}

/// proda.py:489 `_ols_per_protein` — OLS fit per protein on the (NaN-free)
/// imputed matrix. `pred = X beta`, `pred_var[i,j] = (X_j XtXi X_j^T) * s2_i`,
/// `s2 = RSS / max(n - p, 1)`, `df = max(n - p, 1)`.
fn ols_per_protein(mat: &[Vec<f64>], x: &Design) -> OlsFit {
    let n_prot = mat.len();
    let n_samples = x.n_rows;
    let p = x.n_cols;
    let xtxi = x.gram_inverse();
    // leverage[j] = X_j XtXi X_j^T (same for every protein).
    let leverage: Vec<f64> = (0..n_samples)
        .map(|j| {
            let xj = x.row(j);
            let v = mat_vec(&xtxi, xj);
            dot(xj, &v)
        })
        .collect();

    let mut pred = Vec::with_capacity(n_prot);
    let mut pred_var = Vec::with_capacity(n_prot);
    let mut s2 = Vec::with_capacity(n_prot);
    let mut df = Vec::with_capacity(n_prot);
    for row in mat {
        let beta = x.lstsq(row);
        let pred_i: Vec<f64> = (0..n_samples).map(|j| dot(x.row(j), &beta)).collect();
        let rss: f64 = row
            .iter()
            .zip(&pred_i)
            .map(|(y, p)| (y - p) * (y - p))
            .sum();
        let s2_i = rss / ((n_samples - p).max(1)) as f64;
        pred_var.push(leverage.iter().map(|l| l * s2_i).collect());
        pred.push(pred_i);
        s2.push(s2_i);
        df.push((n_samples - p).max(1) as f64);
    }
    OlsFit {
        pred,
        pred_var,
        s2,
        df,
    }
}

/// proda.py:114 `_location_prior`. Estimates `(mu0, sigma20)` for the location
/// prior via the fixed-point objective solved with R-compatible uniroot when it
/// changes sign on `[0, 1000]`, else the closed-form mean of `p^2`.
fn location_prior(
    pred_reg: &[Vec<f64>],
    pred_var_reg: &[Vec<f64>],
    pred_unreg: Option<&[Vec<f64>]>,
    pred_var_unreg: Option<&[Vec<f64>]>,
) -> (f64, f64) {
    let pred_unreg = pred_unreg.unwrap_or(pred_reg);
    let pred_var_unreg = pred_var_unreg.unwrap_or(pred_var_reg);
    let mu0 = trimmed_mean(&flatten(pred_reg), 0.2);

    // pred_flat = where(isnan(pu), pr, pu); pv_flat likewise (proda.py:125-126).
    let mut p = Vec::new();
    let mut pv = Vec::new();
    for i in 0..pred_reg.len() {
        for j in 0..pred_reg[i].len() {
            let pr = pred_reg[i][j];
            let pu = pred_unreg[i][j];
            let pvr = pred_var_reg[i][j];
            let pvu = pred_var_unreg[i][j];
            let pred_flat = if pu.is_nan() { pr } else { pu };
            let pv_flat = if pu.is_nan() { pvr } else { pvu };
            // above = pr > mu0 (proda.py:127); `p` is built from pred_flat - mu0.
            if pr > mu0 {
                p.push(pred_flat - mu0);
                pv.push(pv_flat);
            }
        }
    }

    let obj = |a: f64| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (pi, pvi) in p.iter().zip(&pv) {
            let w = 1.0 / (2.0 * (a + pvi).powi(2));
            num += (pi * pi - pvi) * w;
            den += w;
        }
        num / den - a
    };

    let sigma20 = if p.is_empty() {
        0.0
    } else {
        let (lo, hi) = (0.0, 1000.0);
        let f_lo = obj(lo);
        let f_hi = obj(hi);
        if f_lo.is_finite() && f_hi.is_finite() && f_lo.signum() != f_hi.signum() {
            let root = r_uniroot(&obj, lo, hi);
            if root.is_finite() {
                root
            } else {
                fallback_sigma20(&p)
            }
        } else {
            fallback_sigma20(&p)
        }
    };
    (mu0, sigma20.max(1e-10))
}

/// proda.py:142 fallback `sum(p^2) / max(len(p), 1)`.
fn fallback_sigma20(p: &[f64]) -> f64 {
    let sum: f64 = p.iter().map(|v| v * v).sum();
    sum / (p.len().max(1)) as f64
}

/// proda.py:148 `_variance_prior`. Empirical-Bayes `(tau20, df0)` by minimizing
/// the F-distribution negative log-likelihood with [`r_nelder_mead`]. Filters to
/// finite, positive `(s2, df)` pairs; returns `(1.0, 1.0)` if fewer than two.
fn variance_prior(s2_arr: &[f64], df_arr: &[f64]) -> (f64, f64) {
    let mut s2 = Vec::new();
    let mut df = Vec::new();
    for (a, b) in s2_arr.iter().zip(df_arr) {
        if a.is_finite() && b.is_finite() && *a > 0.0 && *b > 0.0 {
            s2.push(*a);
            df.push(*b);
        }
    }
    if s2.len() < 2 {
        return (1.0, 1.0);
    }

    let neg_ll = |par: &[f64]| -> f64 {
        let (tau, dfinv) = (par[0], par[1]);
        if tau <= 0.0 || dfinv <= 0.0 {
            return f64::INFINITY;
        }
        let df0 = 1.0 / dfinv;
        let mut ll = 0.0;
        for (a, b) in s2.iter().zip(&df) {
            ll += f_logpdf(a / tau, *b, df0) - tau.ln();
        }
        -ll
    };

    let (par, _converged) = r_nelder_mead(&neg_ll, &[1.0, 1.0]);
    let tau20 = par[0];
    let df0 = 1.0 / par[1];
    (tau20, df0)
}

/// proda.py:66 `_fit_dropout_curves`. Per-sample sigmoidal dropout curve
/// `(rho, zeta)` fit by [`r_nelder_mead`] on the `_neg_ll` of proda.py:87. Samples
/// with no missing values keep `NaN` rho/zeta (no dropout to fit).
fn fit_dropout_curves(
    y: &[Vec<f64>],
    x: &Design,
    pred: &[Vec<f64>],
    pred_var: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = x.n_rows;
    let mu0 = trimmed_mean(&flatten(pred), 0.2);
    let diff2: Vec<f64> = flatten(pred).iter().map(|v| (mu0 - v).powi(2)).collect();
    let mut sigma20 = trimmed_mean(&diff2, 0.2);
    if sigma20 == 0.0 {
        sigma20 = 5.0;
    }

    let mut rho = vec![f64::NAN; n_samples];
    let mut zeta = vec![f64::NAN; n_samples];

    for col in 0..n_samples {
        // Collect observed responses and the predictions for the missing rows.
        let mut yo = Vec::new();
        let mut predm = Vec::new();
        let mut pvm = Vec::new();
        for i in 0..y.len() {
            if y[i][col].is_nan() {
                predm.push(pred[i][col]);
                pvm.push(pred_var[i][col]);
            } else {
                yo.push(y[i][col]);
            }
        }
        if predm.is_empty() {
            continue; // obs_mask.all(): nothing missing in this sample.
        }

        let neg_ll = |par: &[f64]| dropout_neg_ll(par, &yo, &predm, &pvm, mu0, sigma20);
        let x0 = [mu0, -1.0 / sigma20.sqrt()];
        let (par, _converged) = r_nelder_mead(&neg_ll, &x0);
        rho[col] = par[0];
        let zinv = if par[1].abs() <= 10000.0 {
            par[1]
        } else {
            -10000.0
        };
        zeta[col] = 1.0 / zinv;
    }
    let _ = x; // design width is implied by `n_samples`; kept for parity.
    (rho, zeta)
}

/// proda.py:87 `_neg_ll` inside `_fit_dropout_curves`.
fn dropout_neg_ll(
    par: &[f64],
    yo: &[f64],
    predm: &[f64],
    pvm: &[f64],
    mu0: f64,
    sigma20: f64,
) -> f64 {
    let (r, zinv) = (par[0], par[1]);
    if zinv >= 0.0 {
        return f64::INFINITY;
    }
    let z = 1.0 / zinv;
    let mut ll = norm_logpdf_scaled(r, mu0, sigma20.sqrt());
    ll += zinv.abs().ln().min(10000.0_f64.ln());
    for &yi in yo {
        ll += invprobit(yi, r, z, true, true);
    }
    let sign = zinv.signum();
    for (&pm, &pv) in predm.iter().zip(pvm) {
        let zs = sign * (1.0 / (zinv * zinv) + pv).sqrt();
        ll += invprobit(pm, r, zs, true, false);
    }
    -ll
}

/// `scipy.stats.norm.logpdf(x, loc, scale)` for a location-scale normal.
fn norm_logpdf_scaled(x: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    norm_logpdf(z) - scale.ln()
}

/// Flatten a protein x sample matrix into a single row-major vector
/// (numpy `.ravel()` order).
fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

// ----------------------------------------------------------------------------
// Small dense linear algebra (the proDA design is 2 columns; the Newton Hessian
// is 3x3). Implemented in-crate to avoid an external dependency.
// ----------------------------------------------------------------------------

/// The two-group design matrix `X` (`n_samples` rows, `N_COEF` columns), stored
/// row-major. Column 0 is the group-A indicator, column 1 group B.
struct Design {
    rows: Vec<[f64; N_COEF]>,
    n_rows: usize,
    n_cols: usize,
}

impl Design {
    /// Two-group indicator design: first `n_a` rows are group A, the rest B.
    fn two_group(n_a: usize, n_b: usize) -> Self {
        let mut rows = Vec::with_capacity(n_a + n_b);
        for _ in 0..n_a {
            rows.push([1.0, 0.0]);
        }
        for _ in 0..n_b {
            rows.push([0.0, 1.0]);
        }
        Self {
            n_rows: n_a + n_b,
            n_cols: N_COEF,
            rows,
        }
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.rows[i]
    }

    /// `(X^T X)^{-1}` (the design is full rank for a two-group contrast).
    fn gram_inverse(&self) -> Vec<Vec<f64>> {
        let mut gram = vec![vec![0.0; self.n_cols]; self.n_cols];
        for r in &self.rows {
            for i in 0..self.n_cols {
                for j in 0..self.n_cols {
                    gram[i][j] += r[i] * r[j];
                }
            }
        }
        invert(&gram).unwrap_or_else(|| identity(self.n_cols))
    }

    /// `lstsq(X, y)` over the FINITE entries of `y` only would change the design;
    /// proDA calls this on the imputed (NaN-free) matrix, so a plain
    /// normal-equations solve over all rows reproduces `np.linalg.lstsq`.
    fn lstsq(&self, y: &[f64]) -> Vec<f64> {
        let mut xtx = vec![vec![0.0; self.n_cols]; self.n_cols];
        let mut xty = vec![0.0; self.n_cols];
        for (r, &yi) in self.rows.iter().zip(y) {
            for i in 0..self.n_cols {
                xty[i] += r[i] * yi;
                for j in 0..self.n_cols {
                    xtx[i][j] += r[i] * r[j];
                }
            }
        }
        match invert(&xtx) {
            Some(inv) => mat_vec(&inv, &xty),
            None => vec![0.0; self.n_cols],
        }
    }

    /// Subset of rows (for the observed / missing partitions), as a plain matrix.
    fn select_rows(&self, mask: impl Fn(usize) -> bool) -> Vec<Vec<f64>> {
        (0..self.n_rows)
            .filter(|&i| mask(i))
            .map(|i| self.rows[i].to_vec())
            .collect()
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// Gauss-Jordan inverse of a small square matrix; `None` if singular.
fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        let diag = m[col][col];
        for v in m[col].iter_mut() {
            *v /= diag;
        }
        // Snapshot the (now normalized) pivot row; it stays fixed while every
        // other row is eliminated against it.
        let pivot_row = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row[col];
            if factor == 0.0 {
                continue;
            }
            for (slot, pv) in row.iter_mut().zip(&pivot_row) {
                *slot -= factor * pv;
            }
        }
    }
    Some(m.iter().map(|row| row[n..].to_vec()).collect())
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| if *x == 0.0 || *y == 0.0 { 0.0 } else { x * y })
        .sum()
}

/// `X beta` for the full design.
fn design_times(x: &Design, beta: &[f64]) -> Vec<f64> {
    (0..x.n_rows).map(|i| dot(x.row(i), beta)).collect()
}

/// `M @ beta` for a plain row-major matrix.
fn rows_times(rows: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    rows.iter().map(|r| dot(r, beta)).collect()
}

include!("proda_fit.rs");
include!("proda_run.rs");

#[cfg(test)]
include!("proda_tests.rs");

#[cfg(test)]
#[path = "proda_official_tests.rs"]
mod official_tests;
