//! limma with a fixed-effects design matrix.
//!
//! Ports R limma's `lmFit` (`lm.series`), `contrasts.fit` and `eBayes` for a
//! design whose columns are condition cell means followed by covariates
//! (`model.matrix(~0 + condition + covariates)`).
//!
//! - `lm.series`: each protein is fitted by least squares on its finite cells
//!   with LINPACK `dqrdc2` (R's `lm.fit`: limited column pivoting at
//!   `tol = 1e-7`), so a coefficient without support among the protein's cells
//!   is not estimable, as in R.
//! - `gls.series`: with a random block, each protein's cells and design rows
//!   are first whitened by the Cholesky factor of their block correlation
//!   matrix (the intra-block correlation from `duplicateCorrelation`).
//! - `contrasts.fit`: the contrast standard error combines the protein's own
//!   unscaled standard errors with the coefficient correlations of the full
//!   design (limma's approximation when cells are missing).
//! - `eBayes`: `squeezeVar` over the proteins with positive residual df; a
//!   protein with zero residual df takes the prior variance.

use mokume_core::{MokumeError, Result};

use super::correct::bh_adjust;
use super::limma::{squeeze_var, LimmaModeration, MIN_PER_GROUP};
use super::student_t::student_t_sf;
use super::{classify, DeResult};

/// `lm.fit`'s rank tolerance.
const LM_TOL: f64 = 1e-7;

/// A design (one row per sample column of the protein rows, one column per
/// coefficient) and one contrast of its coefficients. `group_a`/`group_b` are
/// the sample columns of the two compared conditions; they only feed the
/// per-group summaries and the reporting filter.
pub struct DesignContrast<'a> {
    pub design: &'a [Vec<f64>],
    pub weights: &'a [f64],
    pub group_a: &'a [usize],
    pub group_b: &'a [usize],
    pub block: Option<RandomBlock<'a>>,
}

/// `lmFit(block =, correlation =)`: the block of each design row and the
/// intra-block correlation.
#[derive(Clone, Copy)]
pub struct RandomBlock<'a> {
    pub blocks: &'a [usize],
    pub correlation: f64,
}

/// Reject a design that is not of full column rank: a covariate confounded
/// with the conditions or with another covariate has no estimable effect.
pub fn validate_design(design: &[Vec<f64>]) -> Result<()> {
    let p = design.first().map_or(0, Vec::len);
    if p == 0 || design.iter().any(|row| row.len() != p) {
        return Err(invalid(
            "the DE design matrix must be non-empty and rectangular",
        ));
    }
    if Qr::new(column_major(design, None), design.len(), p).rank < p {
        return Err(invalid(
            "the DE design is not of full rank: a covariate is confounded with the conditions or with another covariate",
        ));
    }
    Ok(())
}

/// limma differential expression for one contrast of a fixed-effects design.
///
/// `proteins[i]` labels `rows[i]`; each row holds log2 intensities for the
/// design's samples. Every protein enters the model fit and the variance prior;
/// proteins with at least two finite values in each compared group and an
/// estimable contrast are reported, sorted by BH-adjusted p-value.
pub fn limma_design(
    proteins: &[String],
    rows: &[&[f64]],
    spec: &DesignContrast<'_>,
    fdr_threshold: f64,
    log2fc_threshold: f64,
) -> Result<Vec<DeResult>> {
    let (moderations, _df_prior) = run_design_moderation(rows, spec)?;
    let reported = moderations
        .into_iter()
        .filter(|(_, fit)| reportable(fit) && fit.p_value.is_finite())
        .collect::<Vec<_>>();
    let p_values = reported
        .iter()
        .map(|(_, fit)| fit.p_value)
        .collect::<Vec<_>>();
    let mut results = reported
        .iter()
        .zip(bh_adjust(&p_values))
        .filter_map(|((index, fit), adj_p_value)| {
            Some(DeResult {
                protein: proteins.get(*index)?.clone(),
                log2_fold_change: fit.log2_fold_change,
                p_value: fit.p_value,
                log_p_value: fit.p_value.ln(),
                adj_p_value,
                t_statistic: fit.t_statistic,
                ave_expr: fit.ave_expr,
                b: 0.0,
                mean_a: fit.mean_a,
                mean_b: fit.mean_b,
                n_a: fit.count_a,
                n_b: fit.count_b,
                significance: classify(
                    adj_p_value,
                    fit.log2_fold_change,
                    fdr_threshold,
                    log2fc_threshold,
                ),
            })
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| left.adj_p_value.total_cmp(&right.adj_p_value));
    Ok(results)
}

/// At least two finite values in each compared group, as in the two-group path.
pub(super) fn reportable(fit: &LimmaModeration) -> bool {
    fit.count_a >= MIN_PER_GROUP && fit.count_b >= MIN_PER_GROUP
}

/// Fit every row, take the contrast and apply limma's eBayes moderation.
/// Returns `(row index, moderation)` for every row with a finite cell, plus the
/// prior df. `t_statistic`/`p_value` are NaN when the contrast is not estimable.
pub(super) fn run_design_moderation(
    rows: &[&[f64]],
    spec: &DesignContrast<'_>,
) -> Result<(Vec<(usize, LimmaModeration)>, f64)> {
    validate_contrast(spec)?;
    let correlation = design_correlation(spec.design, spec.block)?;
    let fits = rows
        .iter()
        .enumerate()
        .map(
            |(index, row)| Ok(fit_row(spec.design, row, spec.block)?.map(|fit| (index, *row, fit))),
        )
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // squeezeVar leaves zero-df rows out of the prior fit.
    let (variances, dfs): (Vec<f64>, Vec<f64>) = fits
        .iter()
        .filter(|(_, _, fit)| fit.df_residual > 0.0)
        .map(|(_, _, fit)| (fit.sigma * fit.sigma, fit.df_residual))
        .unzip();
    let (var_post, s2_prior, df_prior) = squeeze_var(&variances, &dfs);
    let mut posterior = var_post.into_iter();
    let df_pooled: f64 = dfs.iter().sum();

    let moderations = fits
        .into_iter()
        .map(|(index, row, fit)| {
            let var_post = if fit.df_residual > 0.0 {
                posterior.next().unwrap_or(f64::NAN)
            } else {
                s2_prior
            };
            let df_total = (fit.df_residual + df_prior).min(df_pooled);
            let moderation = moderate(row, &fit, var_post, df_total, spec, &correlation);
            (index, moderation)
        })
        .collect();
    Ok((moderations, df_prior))
}

fn validate_contrast(spec: &DesignContrast<'_>) -> Result<()> {
    validate_design(spec.design)?;
    if spec.weights.len() != spec.design[0].len() {
        return Err(invalid(
            "the DE contrast needs one weight per design coefficient",
        ));
    }
    if let Some(block) = spec.block {
        if block.blocks.len() != spec.design.len() {
            return Err(invalid("the DE block needs one label per design row"));
        }
        if block.correlation.is_nan() || block.correlation.abs() >= 1.0 {
            return Err(invalid(
                "the intra-block correlation is 1 or -1, so the model is degenerate",
            ));
        }
    }
    Ok(())
}

/// `contrasts.fit` and the moderated t for one fitted row.
fn moderate(
    row: &[f64],
    fit: &RowFit,
    var_post: f64,
    df_total: f64,
    spec: &DesignContrast<'_>,
    correlation: &[Vec<f64>],
) -> LimmaModeration {
    let (coefficient, stdev_unscaled) = contrast(fit, spec.weights, correlation);
    let t_statistic = coefficient / stdev_unscaled / var_post.sqrt();
    let (sum_a, count_a) = sum_count(row, spec.group_a.iter().copied());
    let (sum_b, count_b) = sum_count(row, spec.group_b.iter().copied());
    let (sum_all, count_all) = sum_count(row, 0..row.len());
    LimmaModeration {
        log2_fold_change: coefficient,
        ave_expr: sum_all / count_all as f64,
        mean_a: sum_a / count_a as f64,
        mean_b: sum_b / count_b as f64,
        count_a,
        count_b,
        sigma: fit.sigma,
        df_residual: fit.df_residual,
        stdev_unscaled,
        t_statistic,
        p_value: 2.0 * student_t_sf(t_statistic.abs(), df_total),
    }
}

fn sum_count(row: &[f64], columns: impl Iterator<Item = usize>) -> (f64, usize) {
    columns
        .map(|column| row[column])
        .filter(|value| value.is_finite())
        .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
}

/// One protein's least-squares fit (`lm.series` on its finite cells).
struct RowFit {
    /// Per design coefficient; NaN where not estimable.
    coefficients: Vec<f64>,
    stdev_unscaled: Vec<f64>,
    /// NaN when the residual df is zero.
    sigma: f64,
    df_residual: f64,
}

fn fit_row(
    design: &[Vec<f64>],
    row: &[f64],
    block: Option<RandomBlock<'_>>,
) -> Result<Option<RowFit>> {
    let observed = (0..row.len())
        .filter(|&sample| row[sample].is_finite())
        .collect::<Vec<_>>();
    if observed.is_empty() {
        return Ok(None);
    }
    let p = design[0].len();
    let n = observed.len();
    let mut columns = column_major(design, Some(&observed));
    columns.extend(observed.iter().map(|&sample| row[sample]));
    if let Some(block) = block {
        whiten(&mut columns, &observed, block)?;
    }
    let mut effects = columns.split_off(n * p);
    let qr = Qr::new(columns, n, p);
    qr.apply_qt(&mut effects);
    let solution = qr.solve(&effects);
    let unscaled = qr.inverse_cross_product();

    let mut coefficients = vec![f64::NAN; p];
    let mut stdev_unscaled = vec![f64::NAN; p];
    for (position, &column) in qr.pivot[..qr.rank].iter().enumerate() {
        coefficients[column] = solution[position];
        stdev_unscaled[column] = unscaled[position][position].sqrt();
    }
    let df_residual = (n - qr.rank) as f64;
    let sigma = if n > qr.rank {
        let rss: f64 = effects[qr.rank..].iter().map(|value| value * value).sum();
        (rss / df_residual).sqrt()
    } else {
        f64::NAN
    };
    Ok(Some(RowFit {
        coefficients,
        stdev_unscaled,
        sigma,
        df_residual,
    }))
}

/// `backsolve(chol(V), x, transpose = TRUE)` for every column of the
/// column-major `columns` (one row per sample of `samples`), where `V` is the
/// samples' block correlation matrix. `V` is block diagonal, so each block's
/// rows are whitened by the Cholesky factor of its own compound-symmetric
/// matrix, in sample order.
fn whiten(columns: &mut [f64], samples: &[usize], block: RandomBlock<'_>) -> Result<()> {
    let n = samples.len();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut seen = Vec::new();
    for (position, &sample) in samples.iter().enumerate() {
        let label = block.blocks[sample];
        match seen.iter().position(|&known| known == label) {
            Some(index) => members[index].push(position),
            None => {
                seen.push(label);
                members.push(vec![position]);
            }
        }
    }
    for positions in members {
        let factor = compound_symmetric_cholesky(positions.len(), block.correlation)?;
        for column in columns.chunks_exact_mut(n) {
            for (i, &target) in positions.iter().enumerate() {
                let sum: f64 = (0..i).map(|k| factor[i][k] * column[positions[k]]).sum();
                column[target] = (column[target] - sum) / factor[i][i];
            }
        }
    }
    Ok(())
}

/// Lower Cholesky factor of the `m x m` matrix with unit diagonal and
/// `correlation` elsewhere.
fn compound_symmetric_cholesky(m: usize, correlation: f64) -> Result<Vec<Vec<f64>>> {
    let mut factor = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let target = if i == j { 1.0 } else { correlation };
            let sum: f64 = (0..j).map(|k| factor[i][k] * factor[j][k]).sum();
            if i == j {
                let pivot = target - sum;
                if pivot <= 0.0 {
                    return Err(invalid(
                        "the block correlation matrix is not positive definite",
                    ));
                }
                factor[i][i] = pivot.sqrt();
            } else {
                factor[i][j] = (target - sum) / factor[j][j];
            }
        }
    }
    Ok(factor)
}

/// `cov2cor(chol2inv(qr(design)))` for a full-rank design, after whitening by
/// the block correlation when there is one.
fn design_correlation(
    design: &[Vec<f64>],
    block: Option<RandomBlock<'_>>,
) -> Result<Vec<Vec<f64>>> {
    let p = design[0].len();
    let mut columns = column_major(design, None);
    if let Some(block) = block {
        let samples = (0..design.len()).collect::<Vec<_>>();
        whiten(&mut columns, &samples, block)?;
    }
    let covariance = Qr::new(columns, design.len(), p).inverse_cross_product();
    Ok((0..p)
        .map(|left| {
            (0..p)
                .map(|right| {
                    covariance[left][right]
                        / (covariance[left][left] * covariance[right][right]).sqrt()
                })
                .collect()
        })
        .collect())
}

/// `contrasts.fit` for one contrast: coefficients with a zero weight drop out;
/// a non-estimable coefficient with a non-zero weight makes the contrast NA.
fn contrast(fit: &RowFit, weights: &[f64], correlation: &[Vec<f64>]) -> (f64, f64) {
    let used = (0..weights.len())
        .filter(|&column| weights[column] != 0.0)
        .collect::<Vec<_>>();
    if used.iter().any(|&column| fit.coefficients[column].is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    let coefficient = used
        .iter()
        .map(|&column| fit.coefficients[column] * weights[column])
        .sum();
    let mut variance = 0.0;
    for &left in &used {
        for &right in &used {
            variance += fit.stdev_unscaled[left]
                * weights[left]
                * correlation[left][right]
                * fit.stdev_unscaled[right]
                * weights[right];
        }
    }
    (coefficient, variance.sqrt())
}

/// Column-major copy of the design, optionally restricted to some samples.
pub(super) fn column_major(design: &[Vec<f64>], samples: Option<&[usize]>) -> Vec<f64> {
    let rows = match samples {
        Some(samples) => samples.iter().map(|&sample| &design[sample]).collect(),
        None => design.iter().collect::<Vec<_>>(),
    };
    (0..design[0].len())
        .flat_map(|column| rows.iter().map(move |row| row[column]))
        .collect()
}

/// LINPACK `dqrdc2` Householder QR with R's limited column pivoting: a column
/// whose remaining norm falls below `tol` times its original norm moves to the
/// end, and the leading `rank` columns keep their original order.
pub(super) struct Qr {
    /// Column-major `n x p`: R in the upper triangle, Householder vectors below.
    qr: Vec<f64>,
    n: usize,
    pub(super) rank: usize,
    pub(super) pivot: Vec<usize>,
    qraux: Vec<f64>,
}

impl Qr {
    pub(super) fn new(mut x: Vec<f64>, n: usize, p: usize) -> Self {
        let mut pivot = (0..p).collect::<Vec<_>>();
        let mut qraux = (0..p)
            .map(|column| norm2(&x[column * n..(column + 1) * n]))
            .collect::<Vec<_>>();
        let mut original = qraux
            .iter()
            .map(|&norm| if norm == 0.0 { 1.0 } else { norm })
            .collect::<Vec<_>>();
        // LINPACK's one-based `k`: retained columns plus one.
        let mut k = p + 1;
        for l in 0..n.min(p) {
            while l + 1 < k && qraux[l] < original[l] * LM_TOL {
                x[l * n..p * n].rotate_left(n);
                pivot[l..].rotate_left(1);
                qraux[l..].rotate_left(1);
                original[l..].rotate_left(1);
                k -= 1;
            }
            if l + 1 == n {
                break;
            }
            householder_step(&mut x, &mut qraux, n, l);
        }
        Self {
            qr: x,
            n,
            rank: (k - 1).min(n),
            pivot,
            qraux,
        }
    }

    fn at(&self, row: usize, column: usize) -> f64 {
        self.qr[column * self.n + row]
    }

    /// `dqrsl`'s `qty`: overwrite `y` with `Q' y` (the `lm.fit` effects).
    pub(super) fn apply_qt(&self, y: &mut [f64]) {
        for j in 0..self.rank.min(self.n - 1) {
            self.reflect(j, y);
        }
    }

    /// `dqrsl`'s `rsd` from the effects: `Q (0, effects[rank..])`, so `lm.fit`'s
    /// fitted values are `y - residuals`.
    pub(super) fn residuals(&self, effects: &[f64]) -> Vec<f64> {
        let mut residuals = effects.to_vec();
        residuals[..self.rank].fill(0.0);
        for j in (0..self.rank.min(self.n - 1)).rev() {
            self.reflect(j, &mut residuals);
        }
        residuals
    }

    /// Apply the `j`th Householder reflection to `y`.
    fn reflect(&self, j: usize, y: &mut [f64]) {
        let head = self.qraux[j];
        if head == 0.0 {
            return;
        }
        let start = j * self.n + j;
        let vector = |i: usize| if i == 0 { head } else { self.qr[start + i] };
        let dot: f64 = (0..self.n - j).map(|i| vector(i) * y[j + i]).sum();
        let t = -dot / head;
        for i in 0..self.n - j {
            y[j + i] += t * vector(i);
        }
    }

    /// Back-substitute `R b = (Q' y)[..rank]`; `b` is in pivot order.
    pub(super) fn solve(&self, effects: &[f64]) -> Vec<f64> {
        let mut solution = effects[..self.rank].to_vec();
        for j in (0..self.rank).rev() {
            solution[j] /= self.at(j, j);
            let value = solution[j];
            for (i, entry) in solution[..j].iter_mut().enumerate() {
                *entry -= value * self.at(i, j);
            }
        }
        solution
    }

    /// `chol2inv(qr, size = rank)`: `(R'R)^-1` in pivot order.
    fn inverse_cross_product(&self) -> Vec<Vec<f64>> {
        let rank = self.rank;
        // Columns of the upper-triangular R^-1.
        let mut inverse = Vec::with_capacity(rank);
        for column in 0..rank {
            let mut values = vec![0.0; rank];
            values[column] = 1.0 / self.at(column, column);
            for row in (0..column).rev() {
                let sum: f64 = (row + 1..=column)
                    .map(|k| self.at(row, k) * values[k])
                    .sum();
                values[row] = -sum / self.at(row, row);
            }
            inverse.push(values);
        }
        let mut product = vec![vec![0.0; rank]; rank];
        for row in 0..rank {
            for column in row..rank {
                let sum: f64 = inverse[column..]
                    .iter()
                    .map(|values| values[row] * values[column])
                    .sum();
                product[row][column] = sum;
                product[column][row] = sum;
            }
        }
        product
    }
}

/// One `dqrdc2` Householder step on column `l`: reflect the remaining columns
/// and downdate their norms.
fn householder_step(x: &mut [f64], qraux: &mut [f64], n: usize, l: usize) {
    let p = qraux.len();
    let start = l * n + l;
    let end = (l + 1) * n;
    let mut norm = norm2(&x[start..end]);
    if norm == 0.0 {
        return;
    }
    if x[start] != 0.0 {
        norm = norm.copysign(x[start]);
    }
    let scale = 1.0 / norm;
    for value in &mut x[start..end] {
        *value *= scale;
    }
    x[start] += 1.0;
    for j in l + 1..p {
        let target = j * n + l;
        let dot: f64 = (0..n - l).map(|i| x[start + i] * x[target + i]).sum();
        let t = -dot / x[start];
        for i in 0..n - l {
            x[target + i] += t * x[start + i];
        }
        if qraux[j] != 0.0 {
            let ratio = x[target].abs() / qraux[j];
            let tt = (1.0 - ratio * ratio).max(0.0);
            qraux[j] = if tt < 1e-6 {
                norm2(&x[target + 1..(j + 1) * n])
            } else {
                qraux[j] * tt.sqrt()
            };
        }
    }
    qraux[l] = x[start];
    x[start] = -norm;
}

fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn invalid(message: &str) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.to_string(),
    }
}

#[cfg(test)]
#[path = "limma_design_tests.rs"]
mod tests;
