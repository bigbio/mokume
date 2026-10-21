use mokume_core::stats::mean_finite;
use mokume_core::{ProteinId, SampleId};

use crate::support::{fill_missing_by_sample, finite_sample_values};

/// Number of Gaussian components fitted per sample, matching the Python
/// `impute_gms` default (`n_components = 2`). The method models each sample's
/// observed log2 abundances as a two-component mixture and treats the
/// lower-mean component as the missing-not-at-random (MNAR) tail.
const N_COMPONENTS: usize = 2;

/// Regularization added to every component variance, mirroring scikit-learn's
/// `GaussianMixture(reg_covar=1e-6)` default so a collapsed (single-point)
/// component cannot reach zero variance.
const REG_COVAR: f64 = 1e-6;

/// EM stopping tolerance on the average log-likelihood change, matching
/// scikit-learn's `tol=1e-3` default.
const EM_TOL: f64 = 1e-3;

/// Maximum EM iterations, matching scikit-learn's `max_iter=100` default.
const EM_MAX_ITER: usize = 100;

/// GMS (Gaussian Mixture Sampling) imputation.
///
/// Faithful port of `mokume.imputation.gms.impute_gms`. For each sample
/// (column) the observed log2 values are modeled with a two-component Gaussian
/// mixture; the lower-mean component approximates the MNAR distribution and its
/// center `mu` is used to fill that sample's missing cells.
///
/// Determinism note: the fill is the center `mu` of the lower component (the
/// expected value of a draw from it) rather than a random draw; the pure-Python
/// `impute_gms` fills the same center. The mixture is fitted with a
/// deterministic median-split initialization plus EM, so the same matrix always
/// yields the same fills.
///
/// Fit fidelity (verified against scikit-learn `GaussianMixture(n_components=2,
/// random_state=42)`): on real-scale columns (tens to hundreds of observations,
/// uni- or bi-modal) this EM converges to the same lower-component `mu` as
/// scikit-learn -- e.g. 40- and 60-point columns matched to 1e-10 and a
/// 200-point column to 2.9e-5. On very small columns (3 observations, 2 present /
/// 1 missing) scikit-learn's random k-means++ seeding and this median split can
/// settle on different two-point splits, so `mu` collapses onto the minimum
/// observation here while scikit-learn's lands higher; that small-sample band is
/// what the golden `..._matches_stochastic_oracle_property` test pins.
///
/// Fallback: when a sample has fewer than `N_COMPONENTS + 1` observed values
/// (too few to fit the mixture) the cell is filled with the mean of that
/// sample's observed values. This branch is deterministic and matches Python
/// exactly (`numpy.nanmean(observed)`). Values are in log2 space throughout, so
/// this mean is the geometric mean once the pipeline applies `2**`.
///
/// Returns fills only for cells that are missing under `value_at`.
pub(crate) fn gms_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    value_at: &mut F,
) -> Vec<(ProteinId, SampleId, f64)>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let fills = samples
        .iter()
        .filter_map(|sample| {
            let observed = finite_sample_values(proteins, *sample, value_at);
            sample_fill(&observed).map(|fill| (*sample, fill))
        })
        .collect::<std::collections::HashMap<_, _>>();
    fill_missing_by_sample(proteins, samples, &fills, value_at)
}

/// Compute the fill value for a single sample from its observed log2 values.
///
/// Mirrors the per-column branch in `impute_gms`: too few observations fall back
/// to the mean; otherwise the lower-mean component of a two-component Gaussian
/// mixture provides the center used as the deterministic fill.
fn sample_fill(observed: &[f64]) -> Option<f64> {
    if observed.is_empty() {
        return None;
    }
    if observed.len() < N_COMPONENTS + 1 {
        // Fallback path: identical to Python's `np.nanmean(observed)`.
        return mean_finite(observed).filter(|value| value.is_finite());
    }
    let (mu, _sigma) = lower_component(observed)?;
    mu.is_finite().then_some(mu)
}

/// Fit a two-component, one-dimensional Gaussian mixture to `observed` and
/// return the lower-mean component's `(mu, sigma)`.
///
/// Implements the EM algorithm directly (no external crates). The Gaussian
/// density is `f(x | mu, var) = exp(-(x - mu)^2 / (2 var)) / sqrt(2 pi var)`.
/// Each iteration:
///   - E-step: responsibility `gamma_ik = w_k f(x_i | mu_k, var_k) /
///     sum_j w_j f(x_i | mu_j, var_j)`.
///   - M-step: `N_k = sum_i gamma_ik`, `w_k = N_k / N`,
///     `mu_k = sum_i gamma_ik x_i / N_k`,
///     `var_k = sum_i gamma_ik (x_i - mu_k)^2 / N_k + REG_COVAR`.
///
/// Convergence is declared when the average log-likelihood changes by less than
/// `EM_TOL`, matching scikit-learn's default convergence test.
///
/// Initialization is deterministic: the sorted observations are split at their
/// median index into a low and a high cluster, and each component is seeded with
/// that cluster's mean, variance, and weight. This replaces scikit-learn's
/// random k-means++ seeding with a reproducible scheme.
fn lower_component(observed: &[f64]) -> Option<(f64, f64)> {
    let mut sorted = observed.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    if count == 0 {
        return None;
    }

    let split = (count / 2).max(1).min(count - 1);
    let (low, high) = sorted.split_at(split);

    let mut means = [slice_mean(low)?, slice_mean(high)?];
    let mut variances = [
        slice_variance(low, means[0]) + REG_COVAR,
        slice_variance(high, means[1]) + REG_COVAR,
    ];
    let mut weights = [
        low.len() as f64 / count as f64,
        high.len() as f64 / count as f64,
    ];

    let mut previous_log_likelihood = f64::NEG_INFINITY;
    for _ in 0..EM_MAX_ITER {
        let mut responsibilities = vec![[0.0_f64; N_COMPONENTS]; count];
        let mut log_likelihood = 0.0;

        for (point, gamma) in sorted.iter().zip(responsibilities.iter_mut()) {
            let densities = [
                weights[0] * gaussian_density(*point, means[0], variances[0]),
                weights[1] * gaussian_density(*point, means[1], variances[1]),
            ];
            let total = densities[0] + densities[1];
            let total = if total > 0.0 {
                total
            } else {
                f64::MIN_POSITIVE
            };
            gamma[0] = densities[0] / total;
            gamma[1] = densities[1] / total;
            log_likelihood += total.ln();
        }

        let mut effective = [0.0_f64; N_COMPONENTS];
        let mut weighted_sum = [0.0_f64; N_COMPONENTS];
        for (point, gamma) in sorted.iter().zip(responsibilities.iter()) {
            for component in 0..N_COMPONENTS {
                effective[component] += gamma[component];
                weighted_sum[component] += gamma[component] * point;
            }
        }

        for component in 0..N_COMPONENTS {
            let denominator = if effective[component] > 0.0 {
                effective[component]
            } else {
                f64::MIN_POSITIVE
            };
            weights[component] = effective[component] / count as f64;
            means[component] = weighted_sum[component] / denominator;
        }

        let mut weighted_variance = [0.0_f64; N_COMPONENTS];
        for (point, gamma) in sorted.iter().zip(responsibilities.iter()) {
            for component in 0..N_COMPONENTS {
                let delta = point - means[component];
                weighted_variance[component] += gamma[component] * delta * delta;
            }
        }
        for component in 0..N_COMPONENTS {
            let denominator = if effective[component] > 0.0 {
                effective[component]
            } else {
                f64::MIN_POSITIVE
            };
            variances[component] = weighted_variance[component] / denominator + REG_COVAR;
        }

        let average_log_likelihood = log_likelihood / count as f64;
        if (average_log_likelihood - previous_log_likelihood).abs() < EM_TOL {
            break;
        }
        previous_log_likelihood = average_log_likelihood;
    }

    let lower = if means[0] <= means[1] { 0 } else { 1 };
    let mut sigma = variances[lower].max(0.0).sqrt();
    // Mirror Python's sigma floor: a collapsed component is widened to 0.1 so a
    // downstream draw would not be degenerate. The deterministic fill uses only
    // `mu`, but the value is reported for parity and potential property checks.
    if sigma < 1e-10 {
        sigma = 0.1;
    }
    Some((means[lower], sigma))
}

/// Gaussian probability density `exp(-(x - mu)^2 / (2 var)) / sqrt(2 pi var)`.
/// A non-positive variance (guarded elsewhere by `REG_COVAR`) yields `0.0`.
fn gaussian_density(point: f64, mean: f64, variance: f64) -> f64 {
    if variance <= 0.0 {
        return 0.0;
    }
    let delta = point - mean;
    let exponent = -0.5 * delta * delta / variance;
    exponent.exp() / (std::f64::consts::TAU * variance).sqrt()
}

/// Arithmetic mean of a non-empty slice; `None` when empty.
fn slice_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Population variance of `values` about `mean` (0.0 for fewer than two points).
fn slice_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64
}
