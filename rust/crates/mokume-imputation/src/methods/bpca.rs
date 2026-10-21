//! Bayesian PCA (BPCA) imputation ported from the pure-Python
//! `mokume.imputation.bpca.impute_bpca`, itself a reimplementation of R
//! `pcaMethods::bpca` (Oba et al., *Bioinformatics* 2003).
//!
//! The matrix is laid out with proteins as rows and samples as columns, the
//! same orientation the Python pipeline feeds to `impute_bpca`: an `n x p`
//! matrix with `n = proteins` observations of a `p = samples`-dimensional
//! vector. Values arrive already in log2 space, so no extra log/exp is applied
//! here; only the originally-missing cells are returned.
//!
//! Determinism: the Python implementation uses only `numpy` linear algebra
//! (`svd`, `inv`, `cov`, `nanmean`) and contains no `numpy.random` or
//! scikit-learn randomness, so it is fully deterministic. This port reproduces
//! it in pure Rust and is deterministic as well.
//!
//! Algorithm (matching the Python reference exactly):
//! 1. `k = min(2, min(n, p) - 1)`. When `k < 1` (degenerate shape) fall back to
//!    per-sample (column) observed means over the log2 values.
//! 2. Center each column by its observed mean (`numpy.nanmean`), keeping the
//!    missing pattern.
//! 3. Initialise the model from the column covariance of the zero-filled
//!    centered matrix (`numpy.cov`, `ddof = 1`): take its eigendecomposition
//!    (the SVD of a symmetric PSD matrix), seed the loading matrix `PA` from the
//!    top `k` eigenpairs, and derive the noise precision `tau` and ARD hyper-
//!    parameter `alpha`.
//! 4. Run up to 100 EM steps. Each step computes latent scores for the fully
//!    observed rows in one batch and, for each incomplete row, the conditional
//!    latent posterior given its observed entries; it then updates `PA`, `tau`,
//!    `SigW`, and `alpha`. Convergence is tested every 10 steps on the change in
//!    `log10(tau)`.
//! 5. Reconstruct `scores @ PA^T + center` and write the reconstruction into the
//!    originally-missing cells.
//!
//! The dense linear algebra (small symmetric eigendecomposition via cyclic
//! Jacobi rotations, and matrix inverses built column-by-column from
//! [`solve_linear_system`]) is implemented here with no external crates.

use mokume_core::{MokumeError, ProteinId, Result, SampleId};

use crate::linalg::solve_linear_system;

/// Number of principal components requested, matching the Python default
/// (`n_pcs = 2`). The effective rank is clamped to `min(n, p) - 1`.
const N_PCS: usize = 2;

/// Maximum EM iterations, matching the Python `max_steps = 100`.
const MAX_STEPS: usize = 100;

/// Convergence threshold on the `log10(tau)` change, matching the Python
/// `threshold = 1e-4`. The test fires only every tenth step, as in the
/// reference.
const TAU_THRESHOLD: f64 = 1e-4;

/// Sweeps used by the cyclic Jacobi eigensolver. The covariance matrix is
/// `p x p` (here `p = samples`, a handful of columns), so a fixed budget of
/// sweeps drives the off-diagonal mass to machine precision.
const JACOBI_SWEEPS: usize = 100;

/// Floor used wherever the Python reference clamps a pivot or precision away
/// from zero (`max(abs(.), 1e-10)` / pivot guards).
const TINY: f64 = 1e-10;

pub(crate) fn bpca_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    value_at: &mut F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let n = proteins.len();
    let p = samples.len();
    if n == 0 || p == 0 {
        return Ok(Vec::new());
    }

    // Read the matrix once. `None` marks a missing cell; non-finite reads are
    // treated as missing as well.
    let data = proteins
        .iter()
        .map(|protein| {
            samples
                .iter()
                .map(|sample| value_at(*protein, *sample).filter(|value| value.is_finite()))
                .collect::<Vec<Option<f64>>>()
        })
        .collect::<Vec<_>>();

    let missing = data
        .iter()
        .map(|row| row.iter().map(Option::is_none).collect::<Vec<bool>>())
        .collect::<Vec<_>>();
    if missing.iter().all(|row| row.iter().all(|cell| !*cell)) {
        return Ok(Vec::new());
    }

    let filled = bpca_em(&data, n, p)?;

    let mut imputed = Vec::new();
    for (protein_index, protein) in proteins.iter().enumerate() {
        for (sample_index, sample) in samples.iter().enumerate() {
            if !missing[protein_index][sample_index] {
                continue;
            }
            let value = filled[protein_index][sample_index];
            if value.is_finite() {
                imputed.push((*protein, *sample, value));
            }
        }
    }
    Ok(imputed)
}

/// BPCA EM imputation over the `n x p` matrix (`None` = missing). Returns a
/// fully observed copy whose missing cells hold the BPCA reconstruction.
/// Mirrors `mokume.imputation.bpca._bpca_em`.
fn bpca_em(data: &[Vec<Option<f64>>], n: usize, p: usize) -> Result<Vec<Vec<f64>>> {
    let smaller = n.min(p);
    // k = min(N_PCS, min(n, p) - 1), guarding the unsigned subtraction.
    let k = if smaller >= 1 {
        N_PCS.min(smaller - 1)
    } else {
        0
    };

    // Column-wise observed means, used both as the center and (in the degenerate
    // branch) as the fill value.
    let center = column_observed_means(data, n, p);

    if k < 1 {
        // Degenerate shape: fill every missing cell with its column mean.
        let mut result = vec![vec![0.0; p]; n];
        for (row_index, row) in data.iter().enumerate().take(n) {
            for (column, cell) in row.iter().enumerate() {
                result[row_index][column] = match cell {
                    Some(value) => *value,
                    None => center[column],
                };
            }
        }
        return Ok(result);
    }

    // y = data - center (per column), preserving the missing pattern.
    let centered = data
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, cell)| cell.map(|value| value - center[column]))
                .collect::<Vec<Option<f64>>>()
        })
        .collect::<Vec<_>>();

    let mut model = BpcaModel::init(&centered, n, p, k)?;
    let mut tau_old = 1000.0_f64;
    for step in 1..=MAX_STEPS {
        model.do_step(&centered, n, p, k);
        if step % 10 == 0 {
            let new_log = model.tau.max(1e-300).log10();
            let old_log = tau_old.max(1e-300).log10();
            if (new_log - old_log).abs() < TAU_THRESHOLD {
                break;
            }
            tau_old = model.tau;
        }
    }

    // recon = scores @ PA^T + center, written into the missing cells only.
    let mut result = vec![vec![0.0; p]; n];
    for (row_index, row) in data.iter().enumerate().take(n) {
        for (column, cell) in row.iter().enumerate() {
            match cell {
                Some(value) => result[row_index][column] = *value,
                None => {
                    let mut acc = center[column];
                    for component in 0..k {
                        acc += model.scores[row_index][component] * model.pa[column][component];
                    }
                    result[row_index][column] = acc;
                }
            }
        }
    }
    Ok(result)
}

/// Mutable BPCA model state, mirroring the dict returned by the Python
/// `_bpca_initmodel`.
struct BpcaModel {
    /// Loading matrix `PA`, shape `p x k`.
    pa: Vec<Vec<f64>>,
    /// Noise precision `tau`.
    tau: f64,
    /// ARD hyperparameter `alpha`, length `k`.
    alpha: Vec<f64>,
    /// `SigW`, shape `k x k`.
    sig_w: Vec<Vec<f64>>,
    /// Per-row latent scores, shape `n x k`.
    scores: Vec<Vec<f64>>,
    /// Observed-column means of the centered matrix (`mean` in Python).
    mean: Vec<f64>,
}

/// Fixed hyperparameters shared by `init` and `do_step` (Python constants).
const GALPHA0: f64 = 1e-10;
const BALPHA0: f64 = 1.0;
const GMU0: f64 = 0.001;
const BTAU0: f64 = 1.0;
const GTAU0: f64 = 1e-10;

impl BpcaModel {
    /// Initialise the model from the centered matrix `y` (`None` = missing),
    /// matching `_bpca_initmodel`.
    fn init(y: &[Vec<Option<f64>>], n: usize, p: usize, k: usize) -> Result<Self> {
        // yest = y with missing entries set to 0.
        let yest = y
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| cell.unwrap_or(0.0))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<_>>();

        // covy = numpy.cov(yest, rowvar=False, ddof=1): column covariance.
        let covy = column_covariance(&yest, n, p);

        // SVD of a symmetric PSD matrix == eigendecomposition; eigenpairs sorted
        // by descending eigenvalue match numpy's singular-value ordering.
        let (eigenvalues, eigenvectors) = symmetric_eigen_descending(&covy, p)?;

        // PA = U[:, :k] @ sqrt(diag(s[:k])).
        let mut pa = vec![vec![0.0; k]; p];
        for (row, pa_row) in pa.iter_mut().enumerate() {
            for (component, pa_cell) in pa_row.iter_mut().enumerate() {
                *pa_cell = eigenvectors[row][component] * eigenvalues[component].max(0.0).sqrt();
            }
        }

        // mean = per-column observed mean of y (missing entries excluded).
        let mean = column_observed_means_f64(y, n, p);

        // tau from the residual spectrum.
        let trace_cov: f64 = (0..p).map(|i| covy[i][i]).sum();
        let top_sum: f64 = eigenvalues.iter().take(k).sum();
        let denom = trace_cov - top_sum;
        let tau = if denom < 0.0 {
            TINY
        } else {
            1.0 / denom.abs().clamp(TINY, 1e10)
        };

        // alpha = (2 galpha0 + p) / (tau * diag(PA^T PA) + 2 galpha0 / balpha0).
        let pat_pa = gram_columns(&pa, p, k);
        let alpha = (0..k)
            .map(|component| {
                (2.0 * GALPHA0 + p as f64)
                    / (tau * pat_pa[component][component] + 2.0 * GALPHA0 / BALPHA0)
            })
            .collect::<Vec<f64>>();

        Ok(BpcaModel {
            pa,
            tau,
            alpha,
            sig_w: identity(k),
            scores: vec![vec![0.0; k]; n],
            mean,
        })
    }

    /// One EM step, mirroring `_bpca_dostep`. `y` is the centered matrix
    /// (`None` = missing).
    fn do_step(&mut self, y: &[Vec<Option<f64>>], n: usize, p: usize, k: usize) {
        let tau = self.tau;

        // Rx = I + tau * (PA^T PA) + SigW.
        let pat_pa = gram_columns(&self.pa, p, k);
        let mut rx = vec![vec![0.0; k]; k];
        for (a, rx_row) in rx.iter_mut().enumerate() {
            for (b, rx_cell) in rx_row.iter_mut().enumerate() {
                let identity_term = if a == b { 1.0 } else { 0.0 };
                *rx_cell = identity_term + tau * pat_pa[a][b] + self.sig_w[a][b];
            }
        }

        let mut scores = vec![vec![f64::NAN; k]; n];
        let mut t_acc = vec![vec![0.0; k]; p];
        let mut tr_s = 0.0;

        // Partition rows into fully observed and incomplete.
        let row_nomiss = (0..n)
            .filter(|&row| y[row].iter().all(Option::is_some))
            .collect::<Vec<usize>>();

        if !row_nomiss.is_empty() {
            // dy = y[idx] - mean (idx fully observed): one (count x p) block.
            let dy = row_nomiss
                .iter()
                .map(|&row| {
                    (0..p)
                        .map(|column| y[row][column].unwrap_or(0.0) - self.mean[column])
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>();

            // x = tau * Rxinv @ PA^T @ dy^T, shape (k x count). We accumulate
            // PA^T @ dy^T per observation, then apply tau * Rxinv.
            let Some(rx_inv) = invert(&rx, k) else {
                // Singular Rx: leave this step's contribution at zero (the
                // Python reference would raise; the pipeline guards against it).
                self.scores = scores;
                return;
            };
            for (obs_index, &row) in row_nomiss.iter().enumerate() {
                // pat_dy[component] = sum_column PA[column][component] * dy[column].
                let mut pat_dy = vec![0.0; k];
                for (component, pat_dy_cell) in pat_dy.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (column, &dy_val) in dy[obs_index].iter().enumerate().take(p) {
                        sum += self.pa[column][component] * dy_val;
                    }
                    *pat_dy_cell = sum;
                }
                // x_obs = tau * Rxinv @ pat_dy.
                let mut x_obs = vec![0.0; k];
                for (a, x_cell) in x_obs.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (b, &rhs) in pat_dy.iter().enumerate() {
                        sum += rx_inv[a][b] * rhs;
                    }
                    *x_cell = tau * sum;
                }
                // T += dy^T (outer) x ; trS += dy . dy.
                for (column, t_row) in t_acc.iter_mut().enumerate() {
                    let dy_val = dy[obs_index][column];
                    for (component, t_cell) in t_row.iter_mut().enumerate() {
                        *t_cell += dy_val * x_obs[component];
                    }
                    tr_s += dy_val * dy_val;
                }
                scores[row] = x_obs;
            }
        }

        // Incomplete rows: conditional latent posterior given observed entries.
        for (row, y_row) in y.iter().enumerate().take(n) {
            if y_row.iter().all(Option::is_some) {
                continue;
            }
            let observed = (0..p)
                .filter(|&c| y_row[c].is_some())
                .collect::<Vec<usize>>();
            let missing = (0..p)
                .filter(|&c| y_row[c].is_none())
                .collect::<Vec<usize>>();

            // dyo = y[row, obs] - mean[obs].
            let dyo = observed
                .iter()
                .map(|&column| y_row[column].unwrap_or(0.0) - self.mean[column])
                .collect::<Vec<f64>>();

            // Rx_local = Rx - tau * (Wm^T Wm), Wm = PA[missing].
            let mut wmt_wm = vec![vec![0.0; k]; k];
            for &m in &missing {
                for (a, wm_row) in wmt_wm.iter_mut().enumerate() {
                    for (b, wm_cell) in wm_row.iter_mut().enumerate() {
                        *wm_cell += self.pa[m][a] * self.pa[m][b];
                    }
                }
            }
            let mut rx_local = vec![vec![0.0; k]; k];
            for (a, local_row) in rx_local.iter_mut().enumerate() {
                for (b, local_cell) in local_row.iter_mut().enumerate() {
                    *local_cell = rx[a][b] - tau * wmt_wm[a][b];
                }
            }
            let Some(rx_local_inv) = invert(&rx_local, k) else {
                // Singular: keep the unconditional latent mean (zeros) and skip.
                scores[row] = vec![0.0; k];
                continue;
            };

            // ex = tau * Wo^T dyo, Wo = PA[observed].
            let mut ex = vec![0.0; k];
            for (a, ex_cell) in ex.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (obs_index, &o) in observed.iter().enumerate() {
                    sum += self.pa[o][a] * dyo[obs_index];
                }
                *ex_cell = tau * sum;
            }
            // x = Rx_local_inv @ ex.
            let mut x_latent = vec![0.0; k];
            for (a, x_cell) in x_latent.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (b, &rhs) in ex.iter().enumerate() {
                    sum += rx_local_inv[a][b] * rhs;
                }
                *x_cell = sum;
            }

            // dy_full: observed entries = dyo, missing entries = Wm @ x.
            let mut dy_full = vec![0.0; p];
            for (obs_index, &o) in observed.iter().enumerate() {
                dy_full[o] = dyo[obs_index];
            }
            for &m in &missing {
                let mut sum = 0.0;
                for (component, &x_val) in x_latent.iter().enumerate() {
                    sum += self.pa[m][component] * x_val;
                }
                dy_full[m] = sum;
            }

            // T += outer(dy_full, x).
            for (column, t_row) in t_acc.iter_mut().enumerate() {
                for (component, t_cell) in t_row.iter_mut().enumerate() {
                    *t_cell += dy_full[column] * x_latent[component];
                }
            }
            // T[missing] += Wm @ Rx_local_inv.
            for &m in &missing {
                for (b, t_cell) in t_acc[m].iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (component, inv_row) in rx_local_inv.iter().enumerate() {
                        sum += self.pa[m][component] * inv_row[b];
                    }
                    *t_cell += sum;
                }
            }

            // trS += dy_full . dy_full + n_miss / tau + trace(Wm Rx_local_inv Wm^T).
            let dy_dot: f64 = dy_full.iter().map(|value| value * value).sum();
            let n_miss = missing.len() as f64;
            let mut trace_term = 0.0;
            for &m in &missing {
                // (Wm Rx_local_inv)[m] . Wm[m] for the diagonal of the triple product.
                for (a, &pa_a) in self.pa[m].iter().enumerate().take(k) {
                    let mut wr = 0.0;
                    for (b, inv_row) in rx_local_inv.iter().enumerate() {
                        wr += self.pa[m][b] * inv_row[a];
                    }
                    trace_term += wr * pa_a;
                }
            }
            tr_s += dy_dot + n_miss / tau + trace_term;
            scores[row] = x_latent;
        }

        // T /= n ; trS /= n.
        for t_row in t_acc.iter_mut() {
            for t_cell in t_row.iter_mut() {
                *t_cell /= n as f64;
            }
        }
        tr_s /= n as f64;

        let Some(rx_inv) = invert(&rx, k) else {
            self.scores = scores;
            return;
        };

        // Dw = Rxinv + tau * T^T PA Rxinv + diag(alpha) / n.
        let tt_pa = transpose_mul(&t_acc, &self.pa, p, k); // (k x k)
        let tt_pa_rxinv = mat_mul(&tt_pa, &rx_inv, k, k, k);
        let mut dw = vec![vec![0.0; k]; k];
        for (a, dw_row) in dw.iter_mut().enumerate() {
            for (b, dw_cell) in dw_row.iter_mut().enumerate() {
                let diag = if a == b {
                    self.alpha[a] / n as f64
                } else {
                    0.0
                };
                *dw_cell = rx_inv[a][b] + tau * tt_pa_rxinv[a][b] + diag;
            }
        }
        let Some(dw_inv) = invert(&dw, k) else {
            self.scores = scores;
            return;
        };

        // PA = T @ Dwinv (p x k).
        let new_pa = mat_mul(&t_acc, &dw_inv, p, k, k);

        // tau update.
        let tt_pa_new = transpose_mul(&t_acc, &new_pa, p, k);
        let tau_num = p as f64 + 2.0 * GTAU0 / n as f64;
        let trace_tt_pa: f64 = (0..k).map(|i| tt_pa_new[i][i]).sum();
        let mean_dot: f64 = self.mean.iter().map(|value| value * value).sum();
        let tau_denom = tr_s - trace_tt_pa + (mean_dot * GMU0 + 2.0 * GTAU0 / BTAU0) / n as f64;
        let mut new_tau = tau_num / tau_denom.abs().max(TINY);
        new_tau = new_tau.clamp(TINY, 1e10);

        // SigW = Dwinv * (p / n).
        let mut new_sig_w = vec![vec![0.0; k]; k];
        for (a, sig_row) in new_sig_w.iter_mut().enumerate() {
            for (b, sig_cell) in sig_row.iter_mut().enumerate() {
                *sig_cell = dw_inv[a][b] * (p as f64 / n as f64);
            }
        }

        // alpha update.
        let pat_pa_new = gram_columns(&new_pa, p, k);
        let new_alpha = (0..k)
            .map(|component| {
                (2.0 * GALPHA0 + p as f64)
                    / (new_tau * pat_pa_new[component][component]
                        + new_sig_w[component][component]
                        + 2.0 * GALPHA0 / BALPHA0)
            })
            .collect::<Vec<f64>>();

        self.pa = new_pa;
        self.tau = new_tau;
        self.sig_w = new_sig_w;
        self.alpha = new_alpha;
        self.scores = scores;
    }
}

/// `numpy.nanmean(values, axis=0)` over the `Option` matrix: per-column mean of
/// the observed entries. A column with no observed entry defaults to `0.0`.
fn column_observed_means(data: &[Vec<Option<f64>>], n: usize, p: usize) -> Vec<f64> {
    column_observed_means_f64(data, n, p)
}

/// Shared per-column observed-mean helper (`numpy.nanmean(axis=0)`).
fn column_observed_means_f64(data: &[Vec<Option<f64>>], n: usize, p: usize) -> Vec<f64> {
    let mut means = vec![0.0; p];
    for (column, mean_slot) in means.iter_mut().enumerate() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in data.iter().take(n) {
            if let Some(value) = row[column] {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            *mean_slot = sum / count as f64;
        }
    }
    means
}

/// `numpy.cov(matrix, rowvar=False, ddof=1)`: column covariance with the
/// columns centered by their own means and a `1 / (n - 1)` normalization.
fn column_covariance(matrix: &[Vec<f64>], n: usize, p: usize) -> Vec<Vec<f64>> {
    let mut col_mean = vec![0.0; p];
    for (column, mean_slot) in col_mean.iter_mut().enumerate() {
        let sum: f64 = matrix.iter().take(n).map(|row| row[column]).sum();
        *mean_slot = sum / n as f64;
    }
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut cov = vec![vec![0.0; p]; p];
    for (a, cov_row) in cov.iter_mut().enumerate() {
        for (b, cov_cell) in cov_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for row in matrix.iter().take(n) {
                sum += (row[a] - col_mean[a]) * (row[b] - col_mean[b]);
            }
            *cov_cell = sum / denom;
        }
    }
    cov
}

/// Gram matrix `M^T M` of a `rows x cols` matrix, returning a `cols x cols`
/// matrix (used for `PA^T PA`).
fn gram_columns(matrix: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut gram = vec![vec![0.0; cols]; cols];
    for (a, gram_row) in gram.iter_mut().enumerate() {
        for (b, gram_cell) in gram_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for row in matrix.iter().take(rows) {
                sum += row[a] * row[b];
            }
            *gram_cell = sum;
        }
    }
    gram
}

/// `A^T B` where `A` and `B` are `rows x cols` matrices, returning a
/// `cols x cols` matrix (used for `T^T PA`).
fn transpose_mul(a: &[Vec<f64>], b: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; cols]; cols];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, out_cell) in out_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += a[row][i] * b[row][j];
            }
            *out_cell = sum;
        }
    }
    out
}

/// Plain matrix product `A (rows x inner) @ B (inner x cols)`.
fn mat_mul(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; cols]; rows];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, out_cell) in out_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for t in 0..inner {
                sum += a[i][t] * b[t][j];
            }
            *out_cell = sum;
        }
    }
    out
}

/// `m x m` identity matrix.
fn identity(m: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; m]; m];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

/// Invert a square matrix by solving against each identity column with the
/// shared Gauss-Jordan solver. Returns `None` when singular, mirroring
/// `numpy.linalg.inv` raising on a singular matrix.
fn invert(matrix: &[Vec<f64>], m: usize) -> Option<Vec<Vec<f64>>> {
    let mut inverse = vec![vec![0.0; m]; m];
    for column in 0..m {
        let mut rhs = vec![0.0; m];
        rhs[column] = 1.0;
        let solution = solve_linear_system(matrix, &rhs)?;
        for (row, &value) in solution.iter().enumerate() {
            inverse[row][column] = value;
        }
    }
    Some(inverse)
}

/// Eigendecomposition of a symmetric matrix by cyclic Jacobi rotations,
/// returning `(eigenvalues, eigenvectors)` sorted by descending eigenvalue. For
/// a symmetric PSD matrix this equals `numpy.linalg.svd` (singular values =
/// eigenvalues in descending order, left singular vectors = eigenvectors). The
/// fill values depend only on the reconstruction `scores @ PA^T`, which is
/// invariant to the sign of each eigenvector, so a sign convention is not
/// required.
fn symmetric_eigen_descending(matrix: &[Vec<f64>], m: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut a = matrix.to_vec();
    let mut vectors = identity(m);
    let norm = matrix.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let tolerance = norm * 1e-14;
    for _ in 0..JACOBI_SWEEPS {
        if off_diagonal_norm(&a) <= tolerance {
            break;
        }
        // A sweep rotates every pair; the previous implementation rotated only
        // one pair and stopped before larger covariance matrices converged.
        for p in 0..m {
            for q in p + 1..m {
                rotate_eigen_pair(&mut a, &mut vectors, p, q);
            }
        }
    }
    if off_diagonal_norm(&a) > tolerance {
        return Err(MokumeError::InvalidInput {
            message: "BPCA covariance eigendecomposition did not converge in 100 sweeps".to_owned(),
        });
    }
    let mut order = (0..m).collect::<Vec<_>>();
    order.sort_by(|&left, &right| a[right][right].total_cmp(&a[left][left]));
    let values = order.iter().map(|&i| a[i][i]).collect();
    let vectors = vectors
        .iter()
        .map(|row| order.iter().map(|&j| row[j]).collect())
        .collect();
    Ok((values, vectors))
}

fn off_diagonal_norm(a: &[Vec<f64>]) -> f64 {
    (2.0 * a
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().skip(i + 1).map(|v| v * v).sum::<f64>())
        .sum::<f64>())
    .sqrt()
}

fn rotate_eigen_pair(a: &mut [Vec<f64>], vectors: &mut [Vec<f64>], p: usize, q: usize) {
    let apq = a[p][q];
    if apq == 0.0 {
        return;
    }
    let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
    let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
    let t = if theta == 0.0 { 1.0 } else { t };
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    let app = a[p][p];
    let aqq = a[q][q];
    let rotated = a
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != p && *k != q)
        .map(|(k, row)| (k, c * row[p] - s * row[q], s * row[p] + c * row[q]))
        .collect::<Vec<_>>();
    for (k, left, right) in rotated {
        a[k][p] = left;
        a[p][k] = left;
        a[k][q] = right;
        a[q][k] = right;
    }
    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;
    for row in vectors {
        let vip = row[p];
        let viq = row[q];
        row[p] = c * vip - s * viq;
        row[q] = s * vip + c * viq;
    }
}
