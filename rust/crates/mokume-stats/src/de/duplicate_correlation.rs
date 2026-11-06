//! limma's `duplicateCorrelation` for a random block.
//!
//! Ports limma 3.66.0 `duplicateCorrelation(block =)` with statmod 1.5.2
//! `mixedModel2Fit(only.varcomp = TRUE, maxit = 20)` and `glmgam.fit`. For each
//! protein with enough finite cells in several blocks, REML on the residual
//! space of the fixed design splits the variance into a residual and a block
//! component; the protein's intra-block correlation is block / (residual +
//! block). The consensus is the tanh of the 15% trimmed mean of the clamped
//! atanh correlations.

use faer::Mat;
use mokume_core::{MokumeError, Result};
use rayon::prelude::*;

use super::limma_design::{column_major, Qr};

const TRIM: f64 = 0.15;
const RHO_MAX: f64 = 0.99;
const GLM_TOL: f64 = 1e-6;
const GLM_MAXIT: usize = 20;
/// `glmgam.fit` stops damping once lambda exceeds this multiple of the information.
const MAX_DAMPING: f64 = 1e15;

/// The consensus intra-block correlation and the per-protein atanh
/// correlations (NaN where a protein gave no estimate).
pub struct BlockCorrelationEstimate {
    pub consensus: f64,
    pub atanh_correlations: Vec<f64>,
    /// limma's warning when it sets the correlation to zero without fitting.
    pub warning: Option<&'static str>,
}

/// `duplicateCorrelation(rows, design, block = blocks)`. `blocks[s]` labels the
/// block of design row `s`.
pub fn duplicate_correlation(
    rows: &[&[f64]],
    design: &[Vec<f64>],
    blocks: &[usize],
) -> Result<BlockCorrelationEstimate> {
    let max_block_size = max_block_size(blocks);
    let zero = |warning| BlockCorrelationEstimate {
        consensus: 0.0,
        atanh_correlations: vec![0.0; rows.len()],
        warning: Some(warning),
    };
    if max_block_size == 1 {
        return Ok(zero(
            "Blocks all of size 1: setting intrablock correlation to zero.",
        ));
    }
    if block_in_design(design, blocks) {
        return Ok(zero("Block factor already encoded in the design matrix: setting intrablock correlation to zero."));
    }
    let n_beta = design[0].len();
    let rho_min = 1.0 / (1.0 - max_block_size as f64) + 0.01;
    let atanh_correlations = rows
        .par_iter()
        .map(|row| protein_correlation(row, design, blocks, n_beta))
        .map(|rho| rho.clamp(rho_min, RHO_MAX).atanh())
        .collect::<Vec<_>>();
    let consensus = trimmed_mean(&atanh_correlations).tanh();
    if !consensus.is_finite() {
        return Err(MokumeError::InvalidInput {
            message: "the intra-block correlation could not be estimated: no protein has enough values in several blocks".to_string(),
        });
    }
    Ok(BlockCorrelationEstimate {
        consensus,
        atanh_correlations,
        warning: None,
    })
}

fn max_block_size(blocks: &[usize]) -> usize {
    let mut sizes = vec![0_usize; blocks.iter().max().map_or(0, |&label| label + 1)];
    for &label in blocks {
        sizes[label] += 1;
    }
    sizes.into_iter().max().unwrap_or(0)
}

/// Whether the block indicators (all but the first level) lie in the design's
/// column space: limma's `QtBlock[-(1:rank), ]` below 1e-8.
fn block_in_design(design: &[Vec<f64>], blocks: &[usize]) -> bool {
    let n = design.len();
    let qr = Qr::new(column_major(design, None), n, design[0].len());
    let mut levels = blocks.to_vec();
    levels.sort_unstable();
    levels.dedup();
    levels.iter().skip(1).all(|&level| {
        let mut column = blocks
            .iter()
            .map(|&label| f64::from(label == level))
            .collect::<Vec<_>>();
        qr.apply_qt(&mut column);
        column[qr.rank..].iter().all(|value| value.abs() < 1e-8)
    })
}

/// One protein's REML intra-block correlation; NaN when it has too few cells
/// or blocks, or the fit fails.
fn protein_correlation(row: &[f64], design: &[Vec<f64>], blocks: &[usize], n_beta: usize) -> f64 {
    let observed = (0..row.len())
        .filter(|&sample| row[sample].is_finite())
        .collect::<Vec<_>>();
    let mut levels = observed
        .iter()
        .map(|&sample| blocks[sample])
        .collect::<Vec<_>>();
    levels.sort_unstable();
    levels.dedup();
    let (n_obs, n_blocks) = (observed.len(), levels.len());
    if !(n_obs > n_beta + 2 && n_blocks > 1 && n_blocks < n_obs - 1) {
        return f64::NAN;
    }
    let qr = Qr::new(column_major(design, Some(&observed)), n_obs, n_beta);
    let residual = |mut values: Vec<f64>| {
        qr.apply_qt(&mut values);
        values.split_off(qr.rank)
    };
    let qtz = levels
        .iter()
        .flat_map(|&level| {
            residual(
                observed
                    .iter()
                    .map(|&sample| f64::from(blocks[sample] == level))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let qty = residual(observed.iter().map(|&sample| row[sample]).collect());
    let Some((d, dy)) = block_regression(&qtz, n_blocks, &qty) else {
        return f64::NAN;
    };
    let [residual_var, block_var] = variance_components(&d, &dy);
    if residual_var.is_nan() {
        return f64::NAN;
    }
    block_var / (residual_var + block_var)
}

/// `mixedModel2Fit`'s regression data: the squared singular values `d` of the
/// `m_q x n_blocks` residual-space block matrix (zero beyond them) and the
/// squared residual coordinates `dy` on its full set of left singular vectors.
/// `None` where the SVD fails.
fn block_regression(qtz: &[f64], n_blocks: usize, qty: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let m_q = qty.len();
    let svd = Mat::from_fn(m_q, n_blocks, |i, j| qtz[j * m_q + i])
        .svd()
        .ok()?;
    let singular = svd.S().column_vector();
    let u = svd.U();
    let d = (0..m_q)
        .map(|i| {
            if i < singular.nrows() {
                singular[i] * singular[i]
            } else {
                0.0
            }
        })
        .collect();
    let dy = (0..m_q)
        .map(|j| {
            let coordinate: f64 = (0..m_q).map(|i| u[(i, j)] * qty[i]).sum();
            coordinate * coordinate
        })
        .collect();
    Some((d, dy))
}

/// `[residual, block]` variance components: least squares of `dy` on
/// `(1, d)`, refined by the gamma GLM when `d` varies.
fn variance_components(d: &[f64], dy: &[f64]) -> [f64; 2] {
    let m_q = d.len();
    if m_q == 0 {
        return [f64::NAN; 2];
    }
    let dx = [vec![1.0; m_q], d.to_vec()].concat();
    let qr = Qr::new(dx.clone(), m_q, 2);
    let mut effects = dy.to_vec();
    qr.apply_qt(&mut effects);
    let solution = qr.solve(&effects);
    let mut start = [f64::NAN; 2];
    for (position, &column) in qr.pivot[..qr.rank].iter().enumerate() {
        start[column] = solution[position];
    }
    let n_nonzero = d.iter().filter(|value| value.abs() > 1e-15).count();
    if !(m_q > 2 && n_nonzero > 1 && sample_variance(d) > 1e-15) {
        return start;
    }
    // mixedModel2Fit checks lm.fit's fitted values (`dy - residuals`), while
    // glmgam.fit errors when `dx %*% start` is negative; the two can disagree
    // in sign where the residual component is zero up to rounding.
    let residuals = qr.residuals(&effects);
    if !dy
        .iter()
        .zip(&residuals)
        .all(|(observed, residual)| observed - residual >= 0.0)
    {
        start = [mean(dy), 0.0];
    }
    glmgam_fit(d, dy, start).unwrap_or([f64::NAN; 2])
}

/// statmod `glmgam.fit`: gamma GLM with identity link on the columns `(1, d)`,
/// Levenberg-damped Fisher scoring from `start`. `None` where R errors.
fn glmgam_fit(d: &[f64], y: &[f64], start: [f64; 2]) -> Option<[f64; 2]> {
    if max_value(y) == 0.0 {
        return Some([0.0; 2]);
    }
    let mut state = GamState::new(d, y, start);
    if state.mu.iter().any(|&value| value < 0.0) {
        return None;
    }
    let mut lambda = f64::NAN;
    for iter in 1.. {
        let (xvx, dl) = scoring_terms(d, y, &state.mu);
        let max_info = xvx[0][0].max(xvx[1][1]);
        if iter == 1 {
            lambda = (xvx[0][0] + xvx[1][1]).abs() / 4.0;
        }
        let (next, step, tries) = damped_update(d, y, &state, (xvx, dl), &mut lambda, max_info)?;
        state = next;
        if lambda / max_info > MAX_DAMPING {
            break;
        }
        if tries == 1 {
            lambda /= 10.0;
        }
        let decrease = dl[0] * step[0] + dl[1] * step[1];
        if !decrease.is_finite() {
            return None;
        }
        if decrease < GLM_TOL || state.dev / max_value(&state.mu) < 1e-15 || iter > GLM_MAXIT {
            break;
        }
    }
    Some(state.beta)
}

/// Coefficients with their fitted means and gamma deviance.
struct GamState {
    beta: [f64; 2],
    mu: Vec<f64>,
    dev: f64,
}

impl GamState {
    fn new(d: &[f64], y: &[f64], beta: [f64; 2]) -> Self {
        let mu = fitted(d, beta);
        let dev = deviance_gamma(y, &mu);
        Self { beta, mu, dev }
    }
}

/// `glmgam.fit`'s inner loop: double `lambda` until the deviance does not
/// increase; past the damping limit the old coefficients are kept. Returns the
/// new state, the last step and the number of tries.
fn damped_update(
    d: &[f64],
    y: &[f64],
    old: &GamState,
    (xvx, dl): ([[f64; 2]; 2], [f64; 2]),
    lambda: &mut f64,
    max_info: f64,
) -> Option<(GamState, [f64; 2], usize)> {
    let mut tries = 0;
    loop {
        tries += 1;
        let step = damped_step(xvx, dl, *lambda)?;
        let mut state = GamState::new(d, y, [old.beta[0] + step[0], old.beta[1] + step[1]]);
        if state.dev.is_finite()
            && (state.dev <= old.dev || state.dev / max_value(&state.mu) < 1e-15)
        {
            return Some((state, step, tries));
        }
        if *lambda / max_info > MAX_DAMPING {
            state.beta = old.beta;
            return Some((state, step, tries));
        }
        *lambda *= 2.0;
    }
}

fn fitted(d: &[f64], beta: [f64; 2]) -> Vec<f64> {
    d.iter().map(|value| beta[0] + beta[1] * value).collect()
}

/// `X' V^-1 X` and the score `X' (y - mu) / V` with `V = pmax(mu^2, max(mu^2) / 1e3)`.
fn scoring_terms(d: &[f64], y: &[f64], mu: &[f64]) -> ([[f64; 2]; 2], [f64; 2]) {
    let v_max = mu
        .iter()
        .map(|value| value * value)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut xvx = [[0.0; 2]; 2];
    let mut dl = [0.0; 2];
    for ((&x, &observed), &mean) in d.iter().zip(y).zip(mu) {
        let v = (mean * mean).max(v_max / 1e3);
        xvx[0][0] += 1.0 / v;
        xvx[0][1] += x / v;
        xvx[1][1] += x * x / v;
        dl[0] += (observed - mean) / v;
        dl[1] += x * (observed - mean) / v;
    }
    xvx[1][0] = xvx[0][1];
    (xvx, dl)
}

/// Solve `(X'V^-1X + lambda I) step = dl` through its Cholesky factor.
fn damped_step(xvx: [[f64; 2]; 2], dl: [f64; 2], lambda: f64) -> Option<[f64; 2]> {
    let r11 = (xvx[0][0] + lambda).sqrt();
    let r12 = xvx[0][1] / r11;
    let r22_squared = xvx[1][1] + lambda - r12 * r12;
    if !(r11 > 0.0 && r22_squared > 0.0) {
        return None;
    }
    let r22 = r22_squared.sqrt();
    let z0 = dl[0] / r11;
    let z1 = (dl[1] - r12 * z0) / r22;
    let step1 = z1 / r22;
    Some([(z0 - r12 * step1) / r11, step1])
}

/// statmod's gamma deviance; infinite for a negative mean.
fn deviance_gamma(y: &[f64], mu: &[f64]) -> f64 {
    if mu.iter().any(|&value| value < 0.0) {
        return f64::INFINITY;
    }
    let tiny = |(&observed, &mean): (&f64, &f64)| observed < 1e-15 && mean < 1e-15;
    if y.iter().zip(mu).all(tiny) {
        return 0.0;
    }
    2.0 * y
        .iter()
        .zip(mu)
        .filter(|pair| !tiny(*pair))
        .map(|(&observed, &mean)| (observed - mean) / mean - (observed / mean).ln())
        .sum::<f64>()
}

fn max_value(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// R's `mean`: a second pass corrects the first.
fn mean(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let first = values.iter().sum::<f64>() / n;
    first + values.iter().map(|value| value - first).sum::<f64>() / n
}

fn sample_variance(values: &[f64]) -> f64 {
    let center = mean(values);
    values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0)
}

/// R's `mean(x, trim = 0.15, na.rm = TRUE)`.
fn trimmed_mean(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    finite.sort_by(f64::total_cmp);
    let low = (finite.len() as f64 * TRIM).floor() as usize;
    mean(&finite[low..finite.len() - low])
}

#[cfg(test)]
#[path = "duplicate_correlation_tests.rs"]
mod tests;
