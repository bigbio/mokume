//! IHW for a numeric ordinal covariate: independent folds, Grenander CDFs,
//! total-variation regularization selected by nested cross-validation, and BH.
//!
//! This implements the IHW 1.38.0 `ihw.default` common contract, using microlp
//! instead of lpsymphony and SplitMix64 instead of R's random stream. Equal
//! seeds therefore do not imply equal folds. Explicit folds and a fixed lambda
//! permit deterministic comparison of weight learning and weighted BH.

#[path = "ihw_folds.rs"]
mod folds;
#[path = "ihw_lp.rs"]
mod lp;
#[cfg(test)]
#[path = "ihw_tests.rs"]
mod tests;

use super::{correct::bh_adjust, rots::SplitMix64};
use mokume_core::{MokumeError, Result};

/// Options for ordinal, Grenander, BH IHW without null-proportion estimation.
#[derive(Debug, Clone)]
pub struct IhwOptions {
    /// Zero selects the official rule: clamp(floor(number tested / 1500), 1, 40).
    pub n_bins: usize,
    pub n_folds: usize,
    pub inner_folds: usize,
    pub inner_splits: usize,
    /// None uses the official ordered grid [0, 1, bins/8, bins/4, bins/2, bins, Inf].
    pub lambdas: Option<Vec<f64>>,
    pub seed: u64,
    /// Optional zero-based fold labels, aligned to the original input.
    pub folds: Option<Vec<usize>>,
}

impl Default for IhwOptions {
    fn default() -> Self {
        Self {
            n_bins: 0,
            n_folds: 5,
            inner_folds: 5,
            inner_splits: 1,
            lambdas: None,
            seed: 1,
            folds: None,
        }
    }
}

/// Complete output in original hypothesis order; excluded p-values remain NaN.
#[derive(Debug, Clone)]
pub struct IhwResult {
    pub adjusted_pvalues: Vec<f64>,
    pub weights: Vec<f64>,
    pub weighted_pvalues: Vec<f64>,
    pub groups: Vec<Option<usize>>,
    pub folds: Vec<Option<usize>>,
    pub fold_lambdas: Vec<f64>,
}

/// Official IHW defaults, with an optional explicit bin count (zero = auto).
/// Invalid inputs or LP failures are errors, never a silent BH substitution.
pub fn ihw_correction(
    pvalues: &[f64],
    covariate: &[f64],
    alpha: f64,
    n_bins: usize,
) -> Result<Vec<f64>> {
    let options = IhwOptions {
        n_bins,
        ..IhwOptions::default()
    };
    Ok(ihw_with_options(pvalues, covariate, alpha, &options)?.adjusted_pvalues)
}

pub fn ihw_with_options(
    pvalues: &[f64],
    covariate: &[f64],
    alpha: f64,
    options: &IhwOptions,
) -> Result<IhwResult> {
    validate(pvalues, covariate, alpha, options)?;
    let mut indices: Vec<usize> = (0..pvalues.len())
        .filter(|&i| !pvalues[i].is_nan())
        .collect();
    let n_bins = match options.n_bins {
        0 => (indices.len() / 1500).clamp(1, 40),
        n => n,
    };
    let mut result = empty_result(pvalues.len());
    if indices.is_empty() {
        return Ok(result);
    }
    let cov: Vec<f64> = indices.iter().map(|&i| covariate[i]).collect();
    let groups = rank_groups(&cov, n_bins, options.seed);
    for (&index, group) in indices.iter().zip(groups) {
        result.groups[index] = Some(group);
    }
    indices.sort_by(|&a, &b| pvalues[a].total_cmp(&pvalues[b]));
    let points: Vec<folds::Point> = indices
        .iter()
        .map(|&i| folds::Point {
            p: pvalues[i],
            group: result.groups[i].unwrap_or(0),
        })
        .collect();
    let labels = options
        .folds
        .as_ref()
        .map(|labels| indices.iter().map(|&i| labels[i]).collect::<Vec<_>>());
    let fit = fit_weights(&points, labels.as_deref(), alpha, n_bins, options)?;
    restore_fit(&mut result, &indices, &points, fit);
    Ok(result)
}

fn restore_fit(
    result: &mut IhwResult,
    indices: &[usize],
    points: &[folds::Point],
    fit: folds::FoldFit,
) {
    let weighted: Vec<f64> = points
        .iter()
        .zip(&fit.weights)
        .map(|(p, &w)| weighted_p(p.p, w))
        .collect();
    let adjusted = bh_adjust(&weighted);
    for (j, &i) in indices.iter().enumerate() {
        result.weights[i] = fit.weights[j];
        result.weighted_pvalues[i] = weighted[j];
        result.adjusted_pvalues[i] = adjusted[j];
        result.folds[i] = Some(fit.labels[j]);
    }
    result.fold_lambdas = fit.lambdas;
}

fn fit_weights(
    points: &[folds::Point],
    labels: Option<&[usize]>,
    alpha: f64,
    n_bins: usize,
    options: &IhwOptions,
) -> Result<folds::FoldFit> {
    let mut rng = SplitMix64::new(options.seed);
    if n_bins == 1 {
        Ok(folds::FoldFit::uniform(points.len()))
    } else {
        let lambdas = lambda_grid(options, n_bins);
        let config = folds::FoldConfig {
            alpha,
            n_bins,
            n_folds: options.n_folds,
            inner_folds: options.inner_folds,
            inner_splits: options.inner_splits,
        };
        folds::fit(points, labels, &lambdas, &config, &mut rng)
    }
}

fn invalid(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

fn validate(p: &[f64], cov: &[f64], alpha: f64, opt: &IhwOptions) -> Result<()> {
    if p.len() != cov.len() || !(0.0 < alpha && alpha < 1.0) {
        return Err(invalid(
            "IHW requires equally sized inputs and 0 < alpha < 1",
        ));
    }
    if p.iter()
        .zip(cov)
        .any(|(&v, &c)| !v.is_nan() && (!(0.0..=1.0).contains(&v) || !c.is_finite()))
    {
        return Err(invalid(
            "IHW requires p-values in [0,1] and finite covariates for every tested hypothesis",
        ));
    }
    validate_options(opt, p.len())
}

fn validate_options(opt: &IhwOptions, n: usize) -> Result<()> {
    if opt.n_folds < 2 || opt.inner_folds < 2 || opt.inner_splits == 0 {
        return Err(invalid(
            "IHW testing requires at least two outer/inner folds and one inner split",
        ));
    }
    if let Some(labels) = &opt.folds {
        if labels.len() != n || labels.iter().any(|&f| f >= opt.n_folds) {
            return Err(invalid(
                "IHW fold labels must match input length and lie in 0..n_folds",
            ));
        }
    }
    if let Some(lambdas) = &opt.lambdas {
        if lambdas.is_empty() || lambdas.iter().any(|&v| v.is_nan() || v < 0.0) {
            return Err(invalid(
                "IHW lambdas must be a nonempty sequence of nonnegative values",
            ));
        }
    }
    Ok(())
}

fn empty_result(n: usize) -> IhwResult {
    IhwResult {
        adjusted_pvalues: vec![f64::NAN; n],
        weights: vec![f64::NAN; n],
        weighted_pvalues: vec![f64::NAN; n],
        groups: vec![None; n],
        folds: vec![None; n],
        fold_lambdas: Vec::new(),
    }
}

fn lambda_grid(options: &IhwOptions, n_bins: usize) -> Vec<f64> {
    options.lambdas.clone().unwrap_or_else(|| {
        let n = n_bins as f64;
        vec![0.0, 1.0, n / 8.0, n / 4.0, n / 2.0, n, f64::INFINITY]
    })
}

fn rank_groups(values: &[f64], bins: usize, seed: u64) -> Vec<usize> {
    // Random tie ranks use an independent seed scope, as groups_by_filter does.
    let mut rng = SplitMix64::new(seed);
    let mut order = rng.permutation(values.len());
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut groups = vec![0; values.len()];
    for (rank, &index) in order.iter().enumerate() {
        groups[index] = ((rank + 1) * bins).div_ceil(values.len()) - 1;
    }
    groups
}

fn weighted_p(p: f64, weight: f64) -> f64 {
    if p == 0.0 {
        0.0
    } else if weight == 0.0 {
        1.0
    } else {
        (p / weight).min(1.0)
    }
}
