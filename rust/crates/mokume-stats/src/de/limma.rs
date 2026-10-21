//! limma moderated t-test for two-group differential expression.
//!
//! This is a from-scratch port of mokume's `analysis/limma.py` (the
//! empirical-Bayes moderated t-test, itself a Python reimplementation of R's
//! limma) for the common case: a single two-condition contrast on a
//! fully-observed (imputed) log2 protein x sample matrix. The pipeline arranges
//! the matrix columns as group A followed by group B; [`run_limma`] returns one
//! [`LimmaStats`] per testable protein.
//!
//! Steps mirror the reference operation-by-operation: cell-means least squares
//! (`lmFit`), the `[1, -1]` contrast (`contrasts.fit`), empirical-Bayes
//! variance moderation (`squeezeVar` via the legacy equal-df `fitFDist`), and
//! the moderated t / two-sided p-value (`eBayes`). `AveExpr` is the per-protein
//! row mean of the pre-contrast expression over finite cells (limma's
//! `Amean = rowMeans(y)`), matching the Python reference; the `B == 0` quirk of
//! the Python implementation is reproduced.
//!
//! Missing values are supported per protein: [`GroupFit`] sums and counts only
//! the finite cells of each group, so the per-protein residual df and unscaled
//! standard error track the finite-sample OLS that mokume's `_lm_fit_with_na`
//! performs (the cell-means design makes the OLS coefficients the group means
//! and `sqrt(diag((X'X)^-1))` the `[1/sqrt(n_a), 1/sqrt(n_b)]` errors). When the
//! residual df is the same for every tested protein (no missing values, or
//! balanced missingness), `squeezeVar` uses the legacy equal-df `fitFDist`,
//! which is oracle-verified and bit-identical to before. When missing values
//! make the residual df vary across proteins, `squeezeVar` switches to the
//! unequal-df1 maximum-likelihood `fitFDist` (`fitFDistUnequalDF1`), a faithful
//! port of mokume's `_fit_f_dist_unequal_df1`: it maximises the marginal F
//! likelihood over a reparameterised prior df using a bounded Brent minimizer
//! that reproduces SciPy's `minimize_scalar(method="bounded")` step for step.

use super::special::{tetragamma, trigamma};
use super::student_t::student_t_sf;

/// Per-protein moderated statistics from [`run_limma`].
#[derive(Debug, Clone)]
pub(crate) struct LimmaStats {
    pub log2_fold_change: f64,
    pub ave_expr: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub n_a: usize,
    pub n_b: usize,
}

/// Minimum finite observations required per group to test a protein.
const MIN_PER_GROUP: usize = 2;

/// Per-protein cell-means fit and moderated statistics used by DEqMS.
/// LimROTS uses its own QR fit and shares only the variance-moderation helper.
#[derive(Debug, Clone)]
pub(crate) struct LimmaModeration {
    pub log2_fold_change: f64,
    pub ave_expr: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub count_a: usize,
    pub count_b: usize,
    pub sigma: f64,
    pub df_residual: f64,
    pub stdev_unscaled: f64,
    pub t_statistic: f64,
    pub p_value: f64,
}

/// Run the cell-means fit, `squeezeVar`, and standard eBayes moderation, keeping
/// every per-protein intermediate. Returns `(per-protein moderations, df_prior)`
/// where each entry pairs the original `rows` index with its [`LimmaModeration`].
/// This is the shared engine behind both [`run_limma`] and the DEqMS port so the
/// limma math is never duplicated.
pub(crate) fn run_limma_moderation(
    rows: &[&[f64]],
    n_a: usize,
    n_b: usize,
) -> (Vec<(usize, LimmaModeration)>, f64) {
    let fits = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| GroupFit::new(row, n_a, n_b).map(|fit| (index, fit)))
        .collect::<Vec<_>>();
    if fits.is_empty() {
        return (Vec::new(), 0.0);
    }

    let variances = fits
        .iter()
        .map(|(_, fit)| fit.sigma * fit.sigma)
        .collect::<Vec<_>>();
    let df_residual = fits
        .iter()
        .map(|(_, fit)| fit.df_residual)
        .collect::<Vec<_>>();
    let (var_post, _s2_prior, df_prior) = squeeze_var(&variances, &df_residual);

    let df_pooled: f64 = df_residual.iter().sum();
    let moderations = fits
        .iter()
        .zip(var_post)
        .map(|((index, fit), var_post)| {
            let s_post = var_post.sqrt();
            let t_statistic = fit.log2_fold_change / (fit.stdev_unscaled * s_post);
            let df_total = (fit.df_residual + df_prior).min(df_pooled);
            let p_value = 2.0 * student_t_sf(t_statistic.abs(), df_total);
            (
                *index,
                LimmaModeration {
                    log2_fold_change: fit.log2_fold_change,
                    ave_expr: fit.ave_expr,
                    mean_a: fit.mean_a,
                    mean_b: fit.mean_b,
                    count_a: fit.count_a,
                    count_b: fit.count_b,
                    sigma: fit.sigma,
                    df_residual: fit.df_residual,
                    stdev_unscaled: fit.stdev_unscaled,
                    t_statistic,
                    p_value,
                },
            )
        })
        .collect();
    (moderations, df_prior)
}

/// Run the limma moderated t-test. `rows` holds one log2 intensity row per
/// protein, with the first `n_a` columns being group A and the next `n_b`
/// columns group B. Proteins with fewer than two finite values in either group
/// are dropped; each returned entry pairs the original row index with its
/// statistics so the caller can recover protein identity.
pub(crate) fn run_limma(rows: &[&[f64]], n_a: usize, n_b: usize) -> Vec<(usize, LimmaStats)> {
    let (moderations, _df_prior) = run_limma_moderation(rows, n_a, n_b);
    moderations
        .into_iter()
        .map(|(index, moderation)| {
            (
                index,
                LimmaStats {
                    log2_fold_change: moderation.log2_fold_change,
                    ave_expr: moderation.ave_expr,
                    t_statistic: moderation.t_statistic,
                    p_value: moderation.p_value,
                    mean_a: moderation.mean_a,
                    mean_b: moderation.mean_b,
                    n_a: moderation.count_a,
                    n_b: moderation.count_b,
                },
            )
        })
        .collect()
}

/// Cell-means least-squares fit of one protein's two groups (`lmFit` +
/// `contrasts.fit` for the `[1, -1]` contrast).
struct GroupFit {
    log2_fold_change: f64,
    ave_expr: f64,
    mean_a: f64,
    mean_b: f64,
    count_a: usize,
    count_b: usize,
    sigma: f64,
    df_residual: f64,
    stdev_unscaled: f64,
}

impl GroupFit {
    fn new(row: &[f64], n_a: usize, n_b: usize) -> Option<Self> {
        let group_a = &row[..n_a];
        let group_b = &row[n_a..n_a + n_b];
        let (sum_a, count_a) = finite_sum_count(group_a);
        let (sum_b, count_b) = finite_sum_count(group_b);
        if count_a < MIN_PER_GROUP || count_b < MIN_PER_GROUP {
            return None;
        }
        let mean_a = sum_a / count_a as f64;
        let mean_b = sum_b / count_b as f64;

        let mut rss = 0.0;
        for value in group_a.iter().filter(|value| value.is_finite()) {
            let residual = value - mean_a;
            rss += residual * residual;
        }
        for value in group_b.iter().filter(|value| value.is_finite()) {
            let residual = value - mean_b;
            rss += residual * residual;
        }
        let df_residual = (count_a + count_b) as f64 - 2.0;
        let sigma = if df_residual > 0.0 {
            (rss / df_residual).sqrt()
        } else {
            f64::NAN
        };
        // contrasts.fit for [1, -1]: stdev_unscaled = sqrt(1/n_a + 1/n_b).
        let stdev_unscaled = (1.0 / count_a as f64 + 1.0 / count_b as f64).sqrt();
        // AveExpr = row mean of the pre-contrast expression over finite cells
        // (limma's `Amean = rowMeans(y)`), matching `np.nanmean(mat, axis=1)`.
        let ave_expr = (sum_a + sum_b) / (count_a + count_b) as f64;
        Some(Self {
            log2_fold_change: mean_a - mean_b,
            ave_expr,
            mean_a,
            mean_b,
            count_a,
            count_b,
            sigma,
            df_residual,
            stdev_unscaled,
        })
    }
}

fn finite_sum_count(values: &[f64]) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for value in values.iter().filter(|value| value.is_finite()) {
        sum += value;
        count += 1;
    }
    (sum, count)
}

/// `squeezeVar`: empirical-Bayes posterior variances. Returns
/// `(var_post, s2_prior, df_prior)`. Mirrors mokume's `_squeeze_var`: when more
/// than one residual df is present, variances at `df == 0` are zeroed, then the
/// non-zero residual df decide the `fitFDist` branch. If every positive residual
/// df is equal (the balanced two-group, no-missing case), the legacy equal-df
/// `fitFDist` is used (oracle-verified, unchanged). Otherwise the unequal-df1
/// maximum-likelihood `fitFDist` is used. Falls back to the `n < 3` shortcut of
/// the reference.
pub(super) fn squeeze_var(variances: &[f64], df_residual: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = variances.len();
    if n < 3 {
        let mut sorted = variances.to_vec();
        let prior = median(&mut sorted).unwrap_or(f64::NAN);
        return (variances.to_vec(), prior, 0.0);
    }

    // Match `if len(df) > 1: var = np.where(df == 0, 0.0, var)`.
    let variances = variances
        .iter()
        .zip(df_residual)
        .map(|(var, df)| if *df == 0.0 { 0.0 } else { *var })
        .collect::<Vec<_>>();

    // `dfp = df[df > 0]; legacy = dfp.min() == dfp.max()`.
    let positive_df = df_residual
        .iter()
        .copied()
        .filter(|df| *df > 0.0)
        .collect::<Vec<_>>();
    let legacy = match (
        positive_df.iter().copied().reduce(f64::min),
        positive_df.iter().copied().reduce(f64::max),
    ) {
        (Some(min_df), Some(max_df)) => min_df == max_df,
        _ => true,
    };

    let (s2_prior, df_prior) = if legacy {
        let df1 = df_residual.first().copied().unwrap_or(f64::NAN);
        fit_f_dist(&variances, df1)
    } else {
        fit_f_dist_unequal_df1(&variances, df_residual)
    };

    let var_post = if df_prior.is_finite() {
        variances
            .iter()
            .zip(df_residual)
            .map(|(var, df)| (df * var + df_prior * s2_prior) / (df + df_prior))
            .collect()
    } else {
        vec![s2_prior; n]
    };
    (var_post, s2_prior, df_prior)
}

/// Legacy equal-`df1` `fitFDist` (Smyth 2004): estimate the prior variance and
/// prior degrees of freedom of the scaled-inverse-chi-square prior.
fn fit_f_dist(variances: &[f64], df1: f64) -> (f64, f64) {
    let mut x = variances
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > -1e-15)
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let nok = x.len();
    if nok < 2 {
        return (x.first().copied().unwrap_or(f64::NAN), f64::INFINITY);
    }

    let mut sorted = x.clone();
    let median_x = median(&mut sorted).unwrap_or(0.0);
    let median_x = if median_x == 0.0 { 1.0 } else { median_x };
    let floor = 1e-5 * median_x;
    for value in &mut x {
        *value = value.max(floor);
    }

    let d1_half = df1 / 2.0;
    let shift = log_m_digamma(d1_half);
    let shifted = x.iter().map(|value| value.ln() + shift).collect::<Vec<_>>();
    let mean_e = accurate_mean(&shifted);
    let mut var_e = accurate_sum(shifted.iter().map(|value| {
        let deviation = value - mean_e;
        deviation * deviation
    })) / (nok as f64 - 1.0);
    var_e -= trigamma(d1_half);

    if var_e > 0.0 {
        let df2 = 2.0 * trigamma_inverse(var_e);
        let d2_half = df2 / 2.0;
        let s2_prior = (mean_e - log_m_digamma(d2_half)).exp();
        (s2_prior, df2)
    } else {
        let mean_x = accurate_mean(&x);
        (mean_x, f64::INFINITY)
    }
}

/// `log(x) - digamma(x)`, matching R's `statmod::logmdigamma` (mokume's
/// `_logmdigamma`).
fn log_m_digamma(x: f64) -> f64 {
    if x < 5.0 {
        return (x / (x + 5.0)).ln()
            + log_m_digamma(x + 5.0)
            + 1.0 / x
            + 1.0 / (x + 1.0)
            + 1.0 / (x + 2.0)
            + 1.0 / (x + 3.0)
            + 1.0 / (x + 4.0);
    }
    let t = 1.0 / (x * x);
    let tail = t
        * (-1.0 / 12.0
            + t * (1.0 / 120.0
                + t * (-1.0 / 252.0
                    + t * (1.0 / 240.0
                        + t * (-1.0 / 132.0
                            + t * (691.0 / 32760.0 + t * (-1.0 / 12.0 + 3617.0 * t / 8160.0)))))));
    1.0 / (2.0 * x) - tail
}

/// Compensated reductions reproduce R's extended-precision sum/mean closely
/// enough to retain small between-permutation differences in prior estimates.
fn accurate_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, correction) = sum_with_correction(values);
    sum + correction
}

fn sum_with_correction(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut sum: f64 = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    (sum, correction)
}

fn accurate_mean(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let (sum, correction) = sum_with_correction(values.iter().copied());
    let mean = sum / n;
    let mean = mean + ((-mean).mul_add(n, sum) + correction) / n;
    mean + accurate_sum(values.iter().map(|v| v - mean)) / n
}

/// Unequal-`df1` maximum-likelihood `fitFDist` (R `limma::fitFDistUnequalDF1`,
/// mokume's `_fit_f_dist_unequal_df1`): estimate the scaled-F prior variance and
/// prior df when the residual df differ across proteins. Returns
/// `(s2_prior, df_prior)`.
///
/// The marginal log likelihood of the scaled-F prior is maximised over the
/// reparameterised prior df `par = d2 / (1 + d2)` (so `d2 = par / (1 - par)`,
/// `df_prior = 2 * d2`), exactly as mokume does. `emean` is the
/// `polygamma(1, d1)`-weighted mean of `log(var) + logmdigamma(d1)` and is held
/// fixed during the optimisation, so the only free parameter is `par`. The
/// optimiser is a bounded Brent minimiser over `(0.5, 0.9998)` that reproduces
/// SciPy's `minimize_scalar(method="bounded")` step for step.
fn fit_f_dist_unequal_df1(variances: &[f64], df_residual: &[f64]) -> (f64, f64) {
    // `ok = isfinite(var) & (var > 0) & isfinite(df1) & (df1 > 0.01)`.
    let mut x = Vec::new();
    let mut d1_full = Vec::new();
    for (var, df) in variances.iter().zip(df_residual) {
        if var.is_finite() && *var > 0.0 && df.is_finite() && *df > 0.01 {
            x.push(*var);
            d1_full.push(*df);
        }
    }
    if x.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut sorted = x.clone();
    let median_x = median(&mut sorted).unwrap_or(0.0);
    let floor = 1e-12 * median_x;
    let xpos = x.iter().map(|value| value.max(floor)).collect::<Vec<_>>();
    let d1 = d1_full.iter().map(|df| df / 2.0).collect::<Vec<_>>();

    // emean = sum(w * (log(xpos) + logmdigamma(d1))) / sum(w), w = 1/trigamma(d1).
    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for (xp, d) in xpos.iter().zip(&d1) {
        let weight = 1.0 / trigamma(*d);
        weighted_sum += weight * (xp.ln() + log_m_digamma(*d));
        weight_total += weight;
    }
    let emean = weighted_sum / weight_total;

    let d1x = d1
        .iter()
        .zip(&xpos)
        .map(|(d, xp)| d * xp)
        .collect::<Vec<_>>();

    // neg2loglik(par) = -2 * sum_i [ -(d1+d2) log1p(d1x/d2s20) - d1 log(d2s20)
    //                                + lgamma(d1+d2) - lgamma(d2) ].
    let objective = |par: f64| -> f64 {
        let d2 = par / (1.0 - par);
        let d2s20 = d2 * (emean - log_m_digamma(d2)).exp();
        let log_d2s20 = d2s20.ln();
        let lgamma_d2 = libm::lgamma(d2);
        let mut log_likelihood = 0.0;
        for (d, d1x_value) in d1.iter().zip(&d1x) {
            log_likelihood += -(d + d2) * libm::log1p(d1x_value / d2s20) - d * log_d2s20
                + libm::lgamma(d + d2)
                - lgamma_d2;
        }
        -2.0 * log_likelihood
    };

    let par_opt = minimize_scalar_bounded(objective, 0.5, 0.9998);
    let d2 = par_opt / (1.0 - par_opt);
    let s2_prior = (emean - log_m_digamma(d2)).exp();
    (s2_prior, 2.0 * d2)
}

/// Bounded scalar Brent minimiser, a faithful port of SciPy's
/// `_minimize_scalar_bounded` (the engine behind
/// `scipy.optimize.minimize_scalar(method="bounded")`). It combines parabolic
/// interpolation with golden-section fallback and the same `xatol = 1e-5`,
/// `maxiter = 500`, and `sqrt_eps = sqrt(2.2e-16)` constants, so the returned
/// minimiser is bit-for-bit identical to SciPy for the same objective. All steps
/// are deterministic f64 arithmetic, so no oracle drift is introduced here.
fn minimize_scalar_bounded(func: impl Fn(f64) -> f64, lower: f64, upper: f64) -> f64 {
    const XATOL: f64 = 1e-5;
    const MAXITER: usize = 500;
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());

    let (mut a, mut b) = (lower, upper);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = 0.0_f64;
    let mut e = 0.0_f64;
    let mut x = xf;
    let mut fx = func(x);
    let mut num = 1;

    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + XATOL / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r_prev = e;
            e = rat;

            if (p.abs() < (0.5 * q * r_prev).abs()) && (p > q * (a - xf)) && (p < q * (b - xf)) {
                rat = p / q;
                x = xf + rat;

                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign_with_zero_positive(xm - xf);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }

        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }

        let si = sign_with_zero_positive(rat);
        x = xf + si * rat.abs().max(tol1);
        let fu = func(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + XATOL / 3.0;
        tol2 = 2.0 * tol1;

        if num >= MAXITER {
            break;
        }
    }

    xf
}

/// `np.sign(value) + (value == 0)`: the sign of `value`, but `+1` when `value`
/// is exactly zero (SciPy's `si` convention inside the bounded minimiser).
fn sign_with_zero_positive(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Newton solve of `trigamma(y) = x` (Smyth's `trigammaInverse`).
fn trigamma_inverse(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::INFINITY;
    }
    if x > 1e7 {
        return 1.0 / x.sqrt();
    }
    if x < 1e-6 {
        return 1.0 / x;
    }
    let mut y = 0.5 + 1.0 / x;
    for _ in 0..50 {
        let tri = trigamma(y);
        let tetra = tetragamma(y);
        let delta = tri * (1.0 - tri / x) / tetra;
        y += delta;
        if -delta / y < 1e-8 {
            break;
        }
    }
    y
}

/// `np.nanmedian` over a mutable slice (sorts in place); `None` if empty after
/// dropping non-finite values.
fn median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    } else {
        Some(values[midpoint])
    }
}

#[cfg(test)]
mod tests {
    use super::run_limma;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected}"
        );
    }

    // Oracle from `conda run -n Bigbio python` on mokume's limma (6 proteins,
    // 3 vs 3, log2 input). Expected statistics from the porting spec.
    #[test]
    fn matches_mokume_limma_oracle() {
        let rows: Vec<Vec<f64>> = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9],
            vec![15.0, 15.1, 14.9, 13.0, 13.2, 12.8],
            vec![8.0, 8.1, 7.9, 8.05, 7.95, 8.0],
            vec![20.0, 22.0, 18.0, 21.0, 19.0, 23.0],
            vec![5.0, 5.01, 4.99, 6.0, 6.01, 5.99],
            vec![9.0, 9.5, 8.5, 9.2, 10.5, 7.5],
        ];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let stats = run_limma(&refs, 3, 3);
        assert_eq!(stats.len(), 6);

        let expected_logfc = [-2.0, 2.0, 0.0, -1.0, -1.0, -0.06666666666666643];
        // AveExpr = row mean over all finite cells (limma `Amean`), != log2FC.
        let expected_ave_expr = [11.0, 14.0, 8.0, 20.5, 5.5, 9.033_333_333_333_333];
        let expected_t = [
            -16.365735545639872,
            16.365735545639872,
            0.0,
            -0.653_792_985_966_897_9,
            -51.768_990_011_434_59,
            -0.077_750_859_569_460_5,
        ];
        let expected_p = [
            3.1734308960086415e-05,
            3.1734308960086415e-05,
            1.0,
            0.544_763_715_713_143_1,
            1.712_569_861_574_927e-7,
            0.941_320_895_121_956_2,
        ];
        for (index, (row_index, stat)) in stats.iter().enumerate() {
            assert_eq!(*row_index, index);
            assert_close(stat.log2_fold_change, expected_logfc[index], 1e-9);
            assert_close(stat.ave_expr, expected_ave_expr[index], 1e-9);
            assert_close(stat.t_statistic, expected_t[index], 1e-6);
            assert_close(stat.p_value, expected_p[index], 1e-9);
        }
    }

    // Oracle from `conda run -n Bigbio python` on mokume's limma with NaN cells
    // (8 proteins, 3 vs 3, log2 input). NaNs are scattered so the per-protein
    // residual df varies (df in {2, 3, 4}); this drives `squeezeVar` through the
    // unequal-df1 maximum likelihood `fitFDist` (bounded Brent optimiser) instead
    // of the legacy equal-df path. Per-protein log2FC / t / pvalue are checked
    // here; the BH-adjusted table and significance calls are asserted in
    // `de::tests::limma_two_group_with_missing_values_matches_oracle_table`.
    //
    // Cell-exact 1e-9 is achieved: the Brent port reproduces SciPy's
    // `_minimize_scalar_bounded` step for step, and the ML objective's
    // `libm::lgamma` / `libm::log1p` agree bit-for-bit with SciPy's `gammaln` /
    // `np.log1p` on these inputs (the optimiser `par` differs by exactly 0.0).
    #[test]
    fn matches_mokume_limma_oracle_with_missing_values() {
        const NAN: f64 = f64::NAN;
        let rows: Vec<Vec<f64>> = vec![
            vec![10.0, 10.2, 9.8, 12.0, 12.1, 11.9],
            vec![15.0, 15.1, NAN, 13.0, 13.2, 12.8],
            vec![8.0, 8.1, 7.9, 8.05, NAN, 8.0],
            vec![20.0, 22.0, NAN, 21.0, NAN, 23.0],
            vec![5.0, 5.01, 4.99, 6.0, 6.01, 5.99],
            vec![9.0, 9.5, 8.5, 9.2, 10.5, 7.5],
            vec![11.0, NAN, 10.5, 12.0, 12.3, NAN],
            vec![3.0, 3.2, 2.8, 4.0, NAN, 4.2],
        ];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let stats = run_limma(&refs, 3, 3);
        assert_eq!(stats.len(), 8);

        let expected_logfc = [
            -2.0,
            2.050_000_000_000_000_7,
            -0.025_000_000_000_000_355,
            -1.0,
            -1.0,
            -0.066_666_666_666_666_43,
            -1.400_000_000_000_000_4,
            -1.1,
        ];
        // AveExpr = row mean over finite cells (limma `Amean`), NA-aware, != log2FC.
        let expected_ave_expr = [
            11.0,
            13.819_999_999_999_999,
            8.01,
            21.5,
            5.5,
            9.033_333_333_333_333,
            11.45,
            3.44,
        ];
        let expected_t = [
            -16.050_021_365_458_303,
            14.217_054_789_305_822,
            -0.247_941_752_467_916_8,
            -0.995_073_154_494_544_4,
            -14.971_855_305_299_965,
            -0.088_854_830_774_118_57,
            -6.113_761_717_242_744_5,
            -7.207_638_258_102_837,
        ];
        let expected_p = [
            3.716_779_595_593_482_4e-6,
            3.100_836_190_996_106_4e-5,
            0.814_039_992_891_240_4,
            0.376_020_843_169_737_66,
            5.591_139_450_715_676e-6,
            0.932_088_898_901_693_8,
            0.003_623_737_568_954_826_5,
            0.000_801_088_550_466_308_3,
        ];
        // Finite counts per group (n_a, n_b) confirm the per-protein NA-aware
        // OLS: residual df = n_a + n_b - 2 varies across proteins (2, 3, 4).
        let expected_counts = [
            (3, 3),
            (2, 3),
            (3, 2),
            (2, 2),
            (3, 3),
            (3, 3),
            (2, 2),
            (3, 2),
        ];

        for (index, (row_index, stat)) in stats.iter().enumerate() {
            assert_eq!(*row_index, index);
            assert_close(stat.log2_fold_change, expected_logfc[index], 1e-9);
            assert_close(stat.ave_expr, expected_ave_expr[index], 1e-9);
            assert_close(stat.t_statistic, expected_t[index], 1e-9);
            assert_close(stat.p_value, expected_p[index], 1e-9);
            assert_eq!((stat.n_a, stat.n_b), expected_counts[index]);
        }
    }
}
