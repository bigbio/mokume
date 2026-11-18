use super::log2_canonical_cells;
use crate::peptide_normalization::QuantilePeptideKey;
use crate::peptide_normalization::tmm::finite_median;
use crate::state::CellKey;
use crate::{HashMap, HashSet, PeptideId, SampleId};

/// Robust Linear Regression (RLR) sample normalization, mirroring NormalyzerDE
/// (Chawade et al., 2014) as implemented by Python `RlrNormalizer`.
///
/// `stages.apply_rlr` constructs `RlrNormalizer(log_transform=False)` and calls
/// it through `_apply_dataset_normalizer(..., log_space=False)`. With
/// `log_transform=False` the normalizer log2s the input itself (dropping
/// non-positive values), fits per sample, divides out the fit in log2 space,
/// and returns `2 ** result` — so the pipeline's outer `log_space=False` means
/// the wide matrix handed in is *linear* and the output is *linear*. We
/// therefore reproduce the whole transform in log2 space internally and
/// exponentiate at the end, exactly like the Python class.
///
/// Per sample, an M-estimator (Huber) robust regression is fitted between the
/// sample's log2 intensities `y` and the per-row log2 median `reference`
/// (pseudo-reference, `median(axis=1)`). The normalized value is
/// `(log2(x) - intercept) / slope`. This matches `statsmodels`
/// `sm.RLM(y, sm.add_constant(reference), M=sm.robust.norms.HuberT()).fit()`
/// with the default IRLS configuration (OLS init, MAD scale with center 0 and
/// `c = 0.6744897501960817`, Huber tuning constant `t = 1.345`, `conv="dev"`,
/// `tol = 1e-8`, `maxiter = 50`, `update_scale=True`).
///
/// Determinism: the only ordering-sensitive steps are medians; these collect
/// finite values into a `Vec` that is sorted with `f64::total_cmp` before
/// taking the median, so the result is independent of `HashMap` iteration
/// order. The IRLS recurrence is a deterministic fixed-point iteration over the
/// collected `(reference, y)` pairs, which are sorted by row id before fitting.
pub(crate) fn apply_rlr_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    // (1) Collapse each allowed cell's peptidoforms to canonical peptides by
    // SUM, then log2 the strictly-positive sums (Python drops non-positive
    // values before `np.log2`). Rows of the wide matrix are (protein,
    // canonical); columns are samples.
    let log2_cells = log2_canonical_cells(cells, allowed_cells, peptide_to_canonical);
    let reference = rlr_reference(&log2_cells);
    let fits = rlr_fit_columns(&log2_cells, &reference);

    // (4) Write back: clear each allowed cell and reinsert canonical ->
    // 2 ** ((log2(value) - intercept) / slope) (exponentiate because we log2'd
    // and the rollup consumes linear values).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let (intercept, slope) = fits.get(&cell.sample).copied().unwrap_or((0.0, 1.0));
        peptides.clear();
        for (peptide, value) in log_row {
            let normalized = (value - intercept) / slope;
            peptides.insert(*peptide, normalized.exp2());
        }
    }
}

/// Fit a single RLR column: robust regression of `y` on `[1, reference]` via
/// Huber IRLS, returning `(intercept, slope)`.
///
/// Reproduces `sm.RLM(y, sm.add_constant(reference), M=HuberT()).fit()` with
/// statsmodels defaults. Returns the identity fit `(0.0, 1.0)` when fewer than
/// five valid points are available, when the design is singular, or when the
/// fitted slope is non-finite or `|slope| < 1e-6` — mirroring the Python
/// `_fit_sample` guards. With identity, `(y - 0) / 1 == y`, so the column is
/// left unchanged.
fn fit_rlr_sample(reference: &[f64], y: &[f64]) -> (f64, f64) {
    let n = reference.len();
    if n < 5 || y.len() != n {
        return (0.0, 1.0);
    }

    // Ordinary least squares initial estimate of (intercept, slope) via the
    // 2x2 normal equations X^T X b = X^T y, X = [1, reference].
    let Some((mut intercept, mut slope)) =
        solve_weighted_normal_equations(reference, y, &vec![1.0; n])
    else {
        return (0.0, 1.0);
    };

    let mut resid = rlr_residual(reference, y, intercept, slope);
    let mut scale = rlr_mad_scale(&resid);
    // Iteration 1 deviance: OLS fit (unit weights) at its own WLS scale.
    let mut previous_deviance = rlr_deviance(&resid, &vec![1.0; n], n);

    let mut iteration = 1;
    loop {
        if scale == 0.0 {
            break;
        }
        let weights = rlr_huber_weights(&resid, scale);
        let Some((next_intercept, next_slope)) =
            solve_weighted_normal_equations(reference, y, &weights)
        else {
            return (0.0, 1.0);
        };
        intercept = next_intercept;
        slope = next_slope;
        resid = rlr_residual(reference, y, intercept, slope);
        scale = rlr_mad_scale(&resid); // update_scale=True
        let deviance = rlr_deviance(&resid, &weights, n);
        iteration += 1;
        let converged = (deviance - previous_deviance).abs() <= TOL || iteration >= MAX_ITER;
        previous_deviance = deviance;
        if converged {
            break;
        }
    }

    if !slope.is_finite() || slope.abs() < 1e-6 {
        return (0.0, 1.0);
    }
    (intercept, slope)
}

const HUBER_T: f64 = 1.345;
// 0.75 quantile of the standard normal; statsmodels `scale.mad` default c.
const MAD_C: f64 = 0.674_489_750_196_081_7;
const TOL: f64 = 1e-8;
const MAX_ITER: usize = 50;

fn rlr_residual(reference: &[f64], y: &[f64], intercept: f64, slope: f64) -> Vec<f64> {
    reference
        .iter()
        .zip(y)
        .map(|(x, y)| y - (intercept + slope * x))
        .collect::<Vec<_>>()
}

// MAD scale with center 0: median(|resid|) / c.
fn rlr_mad_scale(resid: &[f64]) -> f64 {
    let mut abs = resid.iter().map(|value| value.abs()).collect::<Vec<_>>();
    finite_median(&mut abs).unwrap_or(0.0) / MAD_C
}

// Huber weight: 1 if |z| <= t, else t / |z|, with z = resid / scale.
fn rlr_huber_weights(resid: &[f64], scale: f64) -> Vec<f64> {
    resid
        .iter()
        .map(|value| {
            let abs_z = (value / scale).abs();
            if abs_z <= HUBER_T {
                1.0
            } else {
                HUBER_T / abs_z
            }
        })
        .collect::<Vec<_>>()
}

// Un-normalized Huber objective evaluated at z = resid / wls_scale, used by
// the `conv="dev"` stopping rule. `wls_scale` is the weighted-residual
// SSR / (n - 2), the scale `_MinimalWLS` reports for the current fit.
fn rlr_wls_scale(resid: &[f64], weights: &[f64], n: usize) -> f64 {
    let ssr = resid
        .iter()
        .zip(weights)
        .map(|(value, weight)| weight * value * value)
        .sum::<f64>();
    if n > 2 { ssr / (n - 2) as f64 } else { 0.0 }
}

fn rlr_huber_objective(resid: &[f64], scale: f64) -> f64 {
    resid
        .iter()
        .map(|value| {
            let abs_z = (value / scale).abs();
            if abs_z <= HUBER_T {
                0.5 * abs_z * abs_z
            } else {
                HUBER_T * abs_z - 0.5 * HUBER_T * HUBER_T
            }
        })
        .sum::<f64>()
}

fn rlr_deviance(resid: &[f64], weights: &[f64], n: usize) -> f64 {
    let scale = rlr_wls_scale(resid, weights, n);
    if scale > 0.0 {
        rlr_huber_objective(resid, scale)
    } else {
        0.0
    }
}

fn rlr_reference(
    log2_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
) -> HashMap<QuantilePeptideKey, f64> {
    // (2) Per-row pseudo-reference: log2 median across samples (skipna). A row
    // is identified by (protein, canonical); we gather every sample's log2
    // value for that row.
    let mut row_values = HashMap::<QuantilePeptideKey, Vec<f64>>::new();
    for (cell, peptides) in log2_cells {
        for (peptide, value) in peptides {
            row_values
                .entry(QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                })
                .or_default()
                .push(*value);
        }
    }
    row_values
        .into_iter()
        .filter_map(|(key, mut values)| finite_median(&mut values).map(|median| (key, median)))
        .collect::<HashMap<_, _>>()
}

fn rlr_fit_columns(
    log2_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
    reference: &HashMap<QuantilePeptideKey, f64>,
) -> HashMap<SampleId, (f64, f64)> {
    // (3) Fit (intercept, slope) per sample via Huber IRLS against the
    // reference, then transform each value as (y - intercept) / slope. The
    // identity fit (0, 1) leaves the column untouched, matching the Python
    // guards (< 5 valid points, non-finite / near-zero slope, fit failure).
    let mut samples = log2_cells
        .keys()
        .map(|cell| cell.sample)
        .collect::<Vec<_>>();
    samples.sort();
    samples.dedup();

    let mut fits = HashMap::<SampleId, (f64, f64)>::new();
    for sample in samples {
        // Collect (reference, y) pairs where both are finite, in a stable
        // row-id order so the regression is deterministic.
        let mut pairs = Vec::<(QuantilePeptideKey, f64, f64)>::new();
        for (cell, peptides) in log2_cells {
            if cell.sample != sample {
                continue;
            }
            for (peptide, value) in peptides {
                let key = QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                };
                if let Some(reference_value) = reference.get(&key).copied()
                    && reference_value.is_finite()
                    && value.is_finite()
                {
                    pairs.push((key, reference_value, *value));
                }
            }
        }
        pairs.sort_by_key(|pair| pair.0);
        let reference_column = pairs.iter().map(|(_, x, _)| *x).collect::<Vec<_>>();
        let response_column = pairs.iter().map(|(_, _, y)| *y).collect::<Vec<_>>();
        fits.insert(sample, fit_rlr_sample(&reference_column, &response_column));
    }
    fits
}

/// Solve the weighted least squares of `y` on `[1, x]`, i.e. the 2x2 system
/// `(Xᵀ W X) [intercept, slope]ᵀ = Xᵀ W y` with `X = [1, x]`. Returns `None`
/// when the normal matrix is singular (zero determinant), letting the caller
/// fall back to the identity fit. The closed-form 2x2 solve reproduces the
/// `numpy.linalg.pinv`-based WLS that statsmodels uses for a full-rank,
/// well-conditioned design.
fn solve_weighted_normal_equations(x: &[f64], y: &[f64], weights: &[f64]) -> Option<(f64, f64)> {
    let mut s_w = 0.0;
    let mut s_wx = 0.0;
    let mut s_wxx = 0.0;
    let mut s_wy = 0.0;
    let mut s_wxy = 0.0;
    for ((xi, yi), wi) in x.iter().zip(y).zip(weights) {
        s_w += wi;
        s_wx += wi * xi;
        s_wxx += wi * xi * xi;
        s_wy += wi * yi;
        s_wxy += wi * xi * yi;
    }
    // Normal matrix [[s_w, s_wx], [s_wx, s_wxx]]; right-hand side [s_wy, s_wxy].
    let determinant = s_w * s_wxx - s_wx * s_wx;
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let intercept = (s_wxx * s_wy - s_wx * s_wxy) / determinant;
    let slope = (s_w * s_wxy - s_wx * s_wy) / determinant;
    (intercept.is_finite() && slope.is_finite()).then_some((intercept, slope))
}
