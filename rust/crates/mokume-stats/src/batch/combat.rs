//! ComBat batch correction.
//!
//! Parametric and non-parametric empirical-Bayes ComBat follow Bioconductor sva 3.58.0
//! (`ComBat.R` and `helper.R`, Johnson, Li & Rabinovic 2007).
//! It covers the `ref_batch` (reference
//! batch left unmodified) and `mean_only` (additive effect only) options, plus:
//!   - optional covariates (`covar_mod`), whose biological signal is preserved;
//!   - both the parametric (`par_prior=true`) and non-parametric
//!     (`par_prior=false`) prior estimations.
//!
//! With no covariates and a one-hot batch design the two least-squares solves
//! in the reference reduce to per-batch means, so [`combat_parametric`] keeps a
//! solver-free fast path. When covariates are supplied the design becomes dense
//! ([batch one-hot | covariate columns]) and standardization regresses out
//! batch + covariates via a Gaussian-elimination solve of the normal equations
//! `B_hat = solve(design design^T, design data^T)` (`calculate_mean_var` in the
//! reference). Numeric covariates enter the design unchanged; nominal SDRF
//! covariates are expanded to k-1 indicator columns before this module is
//! called, avoiding an artificial ordering between categories.
//!
//! Parametric batch and prior variance estimates use sample variance; pooled
//! residual variance uses the sample count as its divisor, as in sva's complete
//! data path. `it_sol` uses sva's relative-change stop at `1e-4`.
//! Non-parametric integration uses the same sample variance and likelihood
//! evaluation as sva, including an undefined (`NaN`) posterior when every
//! likelihood underflows. It does not replace zero likelihoods with a floor.

/// `it_sol` relative-change convergence threshold (`conv` in the reference).
const CONVERGENCE: f64 = 1e-4;
/// Hard iteration cap (`exit_iteration` in the reference).
const MAX_ITERATIONS: usize = 1_000_000;

/// Options for [`combat`], mirroring the `pycombat_norm` kwargs mokume exposes.
/// `ref_batch` is a batch *label* (not an index); `mean_only` skips the
/// multiplicative (variance) batch effect; `par_prior` selects the parametric
/// empirical-Bayes prior (`true`) versus sva's non-parametric integration
/// (`false`).
#[derive(Debug, Clone, Copy)]
pub struct ComBatParams {
    pub ref_batch: Option<usize>,
    pub mean_only: bool,
    pub par_prior: bool,
}

impl Default for ComBatParams {
    fn default() -> Self {
        // `par_prior` defaults to `true`, matching `pycombat_norm`'s signature.
        Self {
            ref_batch: None,
            mean_only: false,
            par_prior: true,
        }
    }
}

/// Parametric ComBat on a feature x sample matrix, no covariates.
///
/// Thin wrapper over [`combat`] preserving the original signature for callers
/// (the `correct-batches` CLI and the no-covariate golden tests). Forces
/// `par_prior = true` regardless of `params`, matching the historical
/// parametric-only contract.
pub fn combat_parametric(
    data: &[Vec<f64>],
    batch: &[usize],
    params: ComBatParams,
) -> Vec<Vec<f64>> {
    combat(
        data,
        batch,
        None,
        ComBatParams {
            par_prior: true,
            ..params
        },
    )
}

/// ComBat on a feature x sample matrix, with optional covariates.
///
/// `data[g]` is feature `g` across all samples; `batch[n]` is the batch label of
/// sample `n`. `covariates`, when present, is sample-major
/// (`covariates[n]` is the covariate row for sample `n`, all rows equal length),
/// matching the `samples x covariates` layout mokume builds from the SDRF before
/// handing it to `pycombat_norm(covar_mod=...)`. Returns the batch-corrected
/// matrix in the feature x sample orientation. Batches are processed in
/// ascending label order. The caller must supply a finite rectangular matrix
/// and an identifiable design. Both modes leave features constant within
/// any batch unchanged, drops redundant all-one covariates, and uses mean-only
/// correction if a batch has one sample, following sva. Missing rows are handled
/// by the pipeline before this call. An empty matrix or fewer than two batches
/// is returned unchanged.
pub fn combat(
    data: &[Vec<f64>],
    batch: &[usize],
    covariates: Option<&[Vec<f64>]>,
    params: ComBatParams,
) -> Vec<Vec<f64>> {
    if data.is_empty() || batch.is_empty() {
        return combat_inner(data, batch, covariates, params);
    }
    let mut batches = std::collections::BTreeMap::<_, Vec<_>>::new();
    for (sample, &label) in batch.iter().enumerate() {
        batches.entry(label).or_default().push(sample);
    }
    let params = ComBatParams {
        mean_only: params.mean_only || batches.values().any(|samples| samples.len() == 1),
        ..params
    };
    let kept = data
        .iter()
        .enumerate()
        .filter_map(|(i, row)| {
            let constant = batches.values().any(|samples| {
                samples.len() > 1 && samples.iter().all(|&s| row[s] == row[samples[0]])
            });
            (!constant).then_some(i)
        })
        .collect::<Vec<_>>();
    let covariates = covariates.map(without_intercept);
    if kept.len() == data.len() {
        return combat_inner(data, batch, covariates.as_deref(), params);
    }
    let active = kept.iter().map(|&i| data[i].clone()).collect::<Vec<_>>();
    let corrected = combat_inner(&active, batch, covariates.as_deref(), params);
    let mut result = data.to_vec();
    for (i, row) in kept.into_iter().zip(corrected) {
        result[i] = row;
    }
    result
}

fn without_intercept(covariates: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let columns = covariates.first().map_or(0, Vec::len);
    let kept = (0..columns)
        .filter(|&c| !covariates.iter().all(|r| r.get(c) == Some(&1.0)))
        .collect::<Vec<_>>();
    covariates
        .iter()
        .map(|r| kept.iter().map(|&c| r[c]).collect())
        .collect()
}

fn combat_inner(
    data: &[Vec<f64>],
    batch: &[usize],
    covariates: Option<&[Vec<f64>]>,
    params: ComBatParams,
) -> Vec<Vec<f64>> {
    let n_features = data.len();
    let n_samples = batch.len();
    if n_features == 0 || n_samples == 0 {
        return data.to_vec();
    }

    // Distinct batch labels in ascending order, and each batch's sample indices.
    let mut labels = batch.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let n_batch = labels.len();
    if n_batch < 2 {
        return data.to_vec();
    }
    let batch_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
        .collect::<std::collections::HashMap<_, _>>();
    let sample_batch = batch
        .iter()
        .map(|label| batch_index.get(label).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let reference = params
        .ref_batch
        .and_then(|label| batch_index.get(&label).copied());
    let mut batches_ind = vec![Vec::new(); n_batch];
    for (sample, &k) in sample_batch.iter().enumerate() {
        batches_ind[k].push(sample);
    }
    let batch_sizes = batches_ind.iter().map(Vec::len).collect::<Vec<_>>();

    // A covariate matrix with at least one column makes the design dense and
    // routes through the general normal-equations solve. With no usable
    // covariates the one-hot batch design reduces every solve to per-batch means
    // (the original solver-free path), which we keep for fidelity and speed.
    let covar_columns = covariates
        .and_then(|rows| rows.first().map(Vec::len))
        .filter(|&columns| columns > 0);

    let Standardized {
        s_data,
        grand_mean,
        std_dev,
        cov_mean,
    } = match covar_columns {
        Some(columns) => {
            match standardize_with_covariates(
                data,
                covariates.unwrap_or(&[]),
                columns,
                &sample_batch,
                &batches_ind,
                &batch_sizes,
                n_batch,
                reference,
            ) {
                Some(standardized) => standardized,
                // Singular normal equations (e.g. a covariate confounded with
                // the batches): fall back to returning the input unchanged
                // rather than emitting NaNs.
                None => return data.to_vec(),
            }
        }
        None => standardize_one_hot(
            data,
            &sample_batch,
            &batches_ind,
            &batch_sizes,
            n_batch,
            reference,
        ),
    };

    // Per-batch L/S estimates on standardized data. gamma_hat is the per-batch
    // mean of the standardized data (the batch-design solve with a one-hot
    // batch_design), identical with and without covariates.
    let gamma_hat = (0..n_batch)
        .map(|k| {
            (0..n_features)
                .map(|g| mean(batches_ind[k].iter().map(|&n| s_data[g][n])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let delta_hat = (0..n_batch)
        .map(|k| {
            (0..n_features)
                .map(|g| {
                    if params.mean_only {
                        1.0
                    } else {
                        variance(
                            batches_ind[k].iter().map(|&n| s_data[g][n]),
                            gamma_hat[k][g],
                            1,
                        )
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Empirical-Bayes shrinkage per batch.
    let mut gamma_star = vec![vec![0.0; n_features]; n_batch];
    let mut delta_star = vec![vec![1.0; n_features]; n_batch];
    for k in 0..n_batch {
        let gamma_bar = mean(gamma_hat[k].iter().copied());
        let t2 = variance(gamma_hat[k].iter().copied(), gamma_bar, 1);
        if params.mean_only && params.par_prior {
            // Closed-form additive effect (n = 1); multiplicative effect = 1.
            gamma_star[k] = (0..n_features)
                .map(|g| (t2 * gamma_hat[k][g] + gamma_bar) / (t2 + 1.0))
                .collect();
        } else if params.par_prior {
            let (a_prior, b_prior) = inverse_gamma_prior(&delta_hat[k]);
            let priors = BatchPriors {
                gamma_bar,
                t2,
                a_prior,
                b_prior,
            };
            let (g_star, d_star) = it_sol(
                &s_data,
                &batches_ind[k],
                &gamma_hat[k],
                &delta_hat[k],
                priors,
            );
            gamma_star[k] = g_star;
            delta_star[k] = d_star;
        } else {
            let (g_star, d_star) =
                int_eprior(&s_data, &batches_ind[k], &gamma_hat[k], &delta_hat[k]);
            gamma_star[k] = g_star;
            delta_star[k] = d_star;
        }
    }

    // The reference batch is left unmodified.
    if let Some(r) = reference {
        gamma_star[r] = vec![0.0; n_features];
        delta_star[r] = vec![1.0; n_features];
    }

    // Adjust and map back to the original scale; reference-batch samples are
    // restored to the raw input exactly. `stand_mean[g][n] = grand_mean[g] +
    // cov_mean[g][n]` (the covariate contribution, zero in the one-hot path).
    (0..n_features)
        .map(|g| {
            (0..n_samples)
                .map(|n| {
                    let k = sample_batch[n];
                    if reference == Some(k) {
                        data[g][n]
                    } else {
                        let adjusted = (s_data[g][n] - gamma_star[k][g]) / delta_star[k][g].sqrt();
                        let stand_mean = grand_mean[g] + cov_mean[g][n];
                        adjusted * std_dev[g] + stand_mean
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Standardization outputs shared by both design paths. `s_data` is the
/// standardized matrix; `grand_mean[g]` and `std_dev[g]` are per-feature;
/// `cov_mean[g][n]` is the per-sample covariate contribution to the standardized
/// mean (all zero when there are no covariates).
struct Standardized {
    s_data: Vec<Vec<f64>>,
    grand_mean: Vec<f64>,
    std_dev: Vec<f64>,
    cov_mean: Vec<Vec<f64>>,
}

/// One-hot (no-covariate) standardization: the two reference solves collapse to
/// per-batch means, so `B_hat[k][g]` is the batch-`k` mean of feature `g`.
/// Mirrors `calculate_mean_var` / `standardise_data` for a pure batch design.
fn standardize_one_hot(
    data: &[Vec<f64>],
    sample_batch: &[usize],
    batches_ind: &[Vec<usize>],
    batch_sizes: &[usize],
    n_batch: usize,
    reference: Option<usize>,
) -> Standardized {
    let n_features = data.len();
    let n_samples = sample_batch.len();

    let b_hat = (0..n_batch)
        .map(|k| {
            (0..n_features)
                .map(|g| mean(batches_ind[k].iter().map(|&n| data[g][n])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // grand_mean: the reference batch intercept when a reference is set,
    // otherwise the size-weighted average of the batch intercepts.
    let grand_mean = (0..n_features)
        .map(|g| match reference {
            Some(r) => b_hat[r][g],
            None => (0..n_batch)
                .map(|k| batch_sizes[k] as f64 / n_samples as f64 * b_hat[k][g])
                .sum::<f64>(),
        })
        .collect::<Vec<_>>();

    // var_pooled[g]: residual variance (ddof=0) about the batch means, taken
    // over the reference batch only when a reference is set, otherwise all
    // samples.
    let var_pooled = (0..n_features)
        .map(|g| match reference {
            Some(r) => {
                let sum_sq = batches_ind[r]
                    .iter()
                    .map(|&n| {
                        let residual = data[g][n] - b_hat[r][g];
                        residual * residual
                    })
                    .sum::<f64>();
                sum_sq / batch_sizes[r] as f64
            }
            None => {
                let sum_sq = (0..n_samples)
                    .map(|n| {
                        let residual = data[g][n] - b_hat[sample_batch[n]][g];
                        residual * residual
                    })
                    .sum::<f64>();
                sum_sq / n_samples as f64
            }
        })
        .collect::<Vec<_>>();

    let std_dev = var_pooled.iter().map(|v| v.sqrt()).collect::<Vec<_>>();
    let s_data = (0..n_features)
        .map(|g| {
            (0..n_samples)
                .map(|n| (data[g][n] - grand_mean[g]) / std_dev[g])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Standardized {
        s_data,
        grand_mean,
        std_dev,
        cov_mean: vec![vec![0.0; n_samples]; n_features],
    }
}

/// Covariate-aware standardization via the dense normal-equations solve, porting
/// `calculate_mean_var` / `calculate_stand_mean` / `standardise_data`.
///
/// The design (rows = predictors, columns = samples) is
/// `[batch one-hot (n_batch rows) | covariate columns]`. When a reference batch
/// is set its one-hot row is forced to all-ones (matching inmoose's
/// `batch_mod[:, ref] = 1`). `B_hat = solve(design design^T, design data^T)` is
/// the predictor x feature coefficient matrix. Returns `None` if the normal
/// equations are singular.
#[allow(clippy::too_many_arguments)]
fn standardize_with_covariates(
    data: &[Vec<f64>],
    covariates: &[Vec<f64>],
    covar_columns: usize,
    sample_batch: &[usize],
    batches_ind: &[Vec<usize>],
    batch_sizes: &[usize],
    n_batch: usize,
    reference: Option<usize>,
) -> Option<Standardized> {
    let n_features = data.len();
    let n_samples = sample_batch.len();
    if covariates.len() != n_samples {
        return None;
    }

    // design[row][sample]: batch one-hot rows then covariate columns.
    let n_cols = n_batch + covar_columns;
    let mut design = vec![vec![0.0; n_samples]; n_cols];
    for (sample, &k) in sample_batch.iter().enumerate() {
        design[k][sample] = 1.0;
    }
    // Reference batch row is forced all-ones (inmoose `batch_mod[:, ref] = 1`).
    if let Some(r) = reference {
        for value in &mut design[r] {
            *value = 1.0;
        }
    }
    for c in 0..covar_columns {
        for (sample, row) in covariates.iter().enumerate() {
            design[n_batch + c][sample] = *row.get(c)?;
        }
    }

    // Normal equations: gram = design design^T (n_cols x n_cols);
    // rhs[g] = design data[g]^T (length n_cols). B_hat[col][g] solves gram.
    let gram = (0..n_cols)
        .map(|i| {
            (0..n_cols)
                .map(|j| {
                    (0..n_samples)
                        .map(|n| design[i][n] * design[j][n])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut b_hat = vec![vec![0.0; n_features]; n_cols];
    for g in 0..n_features {
        let rhs = (0..n_cols)
            .map(|i| {
                (0..n_samples)
                    .map(|n| design[i][n] * data[g][n])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let solution = solve_linear_system(&gram, &rhs)?;
        for (col, value) in solution.into_iter().enumerate() {
            b_hat[col][g] = value;
        }
    }

    // grand_mean[g]: reference batch intercept, else size-weighted batch
    // intercepts. With a reference, B_hat[ref] is that batch's intercept.
    let grand_mean = (0..n_features)
        .map(|g| match reference {
            Some(r) => b_hat[r][g],
            None => (0..n_batch)
                .map(|k| batch_sizes[k] as f64 / n_samples as f64 * b_hat[k][g])
                .sum::<f64>(),
        })
        .collect::<Vec<_>>();

    // fitted[g][n] = (design^T B_hat)[n][g]: the full regression prediction.
    let fitted = |g: usize, n: usize| -> f64 {
        (0..n_cols)
            .map(|col| design[col][n] * b_hat[col][g])
            .sum::<f64>()
    };

    // var_pooled[g]: residual variance about the full fit, over the reference
    // batch only when set, otherwise all samples (`calculate_mean_var`).
    let var_pooled = (0..n_features)
        .map(|g| match reference {
            Some(r) => {
                let sum_sq = batches_ind[r]
                    .iter()
                    .map(|&n| {
                        let residual = data[g][n] - fitted(g, n);
                        residual * residual
                    })
                    .sum::<f64>();
                sum_sq / batch_sizes[r] as f64
            }
            None => {
                let sum_sq = (0..n_samples)
                    .map(|n| {
                        let residual = data[g][n] - fitted(g, n);
                        residual * residual
                    })
                    .sum::<f64>();
                sum_sq / n_samples as f64
            }
        })
        .collect::<Vec<_>>();
    let std_dev = var_pooled.iter().map(|v| v.sqrt()).collect::<Vec<_>>();

    // cov_mean[g][n]: the covariate-only contribution to the standardized mean
    // (`calculate_stand_mean` zeroes the batch rows before the dot product).
    let cov_mean = (0..n_features)
        .map(|g| {
            (0..n_samples)
                .map(|n| {
                    (n_batch..n_cols)
                        .map(|col| design[col][n] * b_hat[col][g])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // s_data[g][n] = (data - grand_mean - cov_mean) / std_dev.
    let s_data = (0..n_features)
        .map(|g| {
            (0..n_samples)
                .map(|n| (data[g][n] - grand_mean[g] - cov_mean[g][n]) / std_dev[g])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Some(Standardized {
        s_data,
        grand_mean,
        std_dev,
        cov_mean,
    })
}

/// Solve the square system `a x = b` by Gauss-Jordan elimination with partial
/// pivoting (the same scheme as `mokume-imputation`'s solver, inlined here to
/// avoid a cross-crate dependency). Returns `None` when `a` is singular,
/// mirroring `np.linalg.solve` raising `LinAlgError`.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut augmented = a
        .iter()
        .zip(b)
        .map(|(row, &rhs)| {
            let mut extended = row.clone();
            extended.push(rhs);
            extended
        })
        .collect::<Vec<Vec<f64>>>();

    for column in 0..n {
        let pivot = (column..n).max_by(|&left, &right| {
            augmented[left][column]
                .abs()
                .total_cmp(&augmented[right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-300 {
            return None;
        }
        augmented.swap(column, pivot);

        let pivot_value = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= pivot_value;
        }

        let (before, rest) = augmented.split_at_mut(column);
        let (pivot_row, after) = rest.split_first_mut()?;
        for row in before.iter_mut().chain(after.iter_mut()) {
            let factor = row[column];
            if factor != 0.0 {
                for (target, &source) in row.iter_mut().zip(pivot_row.iter()) {
                    *target -= factor * source;
                }
            }
        }
    }

    Some(augmented.iter().map(|row| row[n]).collect())
}

/// Method-of-moments inverse-gamma prior `(a_prior, b_prior)` from a batch's
/// per-feature variances (`compute_prior` in the reference).
fn inverse_gamma_prior(delta_hat: &[f64]) -> (f64, f64) {
    let m = mean(delta_hat.iter().copied());
    let s2 = variance(delta_hat.iter().copied(), m, 1);
    let a_prior = (2.0 * s2 + m * m) / s2;
    let b_prior = (m * s2 + m * m * m) / s2;
    (a_prior, b_prior)
}

/// Per-batch empirical-Bayes prior hyperparameters for [`it_sol`].
struct BatchPriors {
    gamma_bar: f64,
    t2: f64,
    a_prior: f64,
    b_prior: f64,
}

/// `it_sol`: parametric empirical-Bayes fixed-point for one batch's
/// `(gamma_star, delta_star)`.
fn it_sol(
    s_data: &[Vec<f64>],
    samples: &[usize],
    gamma_hat: &[f64],
    delta_hat: &[f64],
    priors: BatchPriors,
) -> (Vec<f64>, Vec<f64>) {
    let BatchPriors {
        gamma_bar,
        t2,
        a_prior,
        b_prior,
    } = priors;
    let n = samples.len() as f64;
    let t2_n = t2 * n;
    let mut g_old = gamma_hat.to_vec();
    let mut d_old = delta_hat.to_vec();
    let n_features = gamma_hat.len();

    for _ in 0..MAX_ITERATIONS {
        let g_new = (0..n_features)
            .map(|g| (t2_n * gamma_hat[g] + d_old[g] * gamma_bar) / (t2_n + d_old[g]))
            .collect::<Vec<_>>();
        let d_new = (0..n_features)
            .map(|g| {
                let sum_sq = samples
                    .iter()
                    .map(|&sample| {
                        let residual = s_data[g][sample] - g_new[g];
                        residual * residual
                    })
                    .sum::<f64>();
                (0.5 * sum_sq + b_prior) / (0.5 * n + a_prior - 1.0)
            })
            .collect::<Vec<_>>();

        let change = (0..n_features)
            .map(|g| {
                // sva uses the signed old value in the denominator. Taking
                // its absolute value changes the stopping iteration.
                let gamma_change = (g_new[g] - g_old[g]).abs() / g_old[g];
                let delta_change = (d_new[g] - d_old[g]).abs() / d_old[g];
                gamma_change.max(delta_change)
            })
            .fold(0.0_f64, f64::max);

        g_old = g_new;
        d_old = d_new;
        if change <= CONVERGENCE {
            break;
        }
    }
    (g_old, d_old)
}

/// `int_eprior`: non-parametric empirical-Bayes estimation for one batch's
/// `(gamma_star, delta_star)`, ported from the reference's deterministic
/// numerical integration (no Monte Carlo / RNG despite the docstring's wording).
///
/// For each feature `i` the prior is integrated over all *other* features'
/// `(gamma_hat, delta_hat)` in this batch: with `x` the standardized batch row
/// of feature `i` (length `n` = batch size), `g`/`d` the other features'
/// estimates, and `sum2[k] = sum_j (x_j - g_k)^2`, the likelihood weight is
/// `LH[k] = (1 / (pi * 2 d_k))^(n/2) * exp(-sum2[k] / (2 d_k))`. The posterior
/// means are `sum(g LH) / sum(LH)` and `sum(d LH) / sum(LH)`. NaN weights are
/// zeroed as in sva; all-zero likelihoods yield undefined posterior estimates.
fn int_eprior(
    s_data: &[Vec<f64>],
    samples: &[usize],
    gamma_hat: &[f64],
    delta_hat: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n_features = gamma_hat.len();
    let n = samples.len() as f64;
    let half_n = n / 2.0;
    let mut g_star = vec![0.0; n_features];
    let mut d_star = vec![0.0; n_features];

    for i in 0..n_features {
        // Likelihood weight of feature i's batch row under every OTHER feature's
        // (gamma_hat, delta_hat) prior.
        let mut weights = Vec::with_capacity(n_features.saturating_sub(1));
        for k in 0..n_features {
            if k == i {
                continue;
            }
            let g_k = gamma_hat[k];
            let two_d_k = 2.0 * delta_hat[k];
            let sum_sq = samples
                .iter()
                .map(|&sample| {
                    let residual = s_data[i][sample] - g_k;
                    residual * residual
                })
                .sum::<f64>();
            let mut weight =
                (1.0 / (std::f64::consts::PI * two_d_k).powf(half_n)) * (-sum_sq / two_d_k).exp();
            if weight.is_nan() {
                // sva's LH[LH == "NaN"] = 0.
                weight = 0.0;
            }
            weights.push((gamma_hat[k], delta_hat[k], weight));
        }

        let denom: f64 = weights.iter().map(|&(_, _, w)| w).sum();
        let g_num: f64 = weights.iter().map(|&(g, _, w)| g * w).sum();
        let d_num: f64 = weights.iter().map(|&(_, d, w)| d * w).sum();
        g_star[i] = g_num / denom;
        d_star[i] = d_num / denom;
    }

    (g_star, d_star)
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Variance about a precomputed mean; sva priors use `ddof=1`.
fn variance(values: impl Iterator<Item = f64>, mean: f64, ddof: usize) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for value in values {
        let deviation = value - mean;
        sum_sq += deviation * deviation;
        count += 1;
    }
    if count <= ddof {
        f64::NAN
    } else {
        sum_sq / (count - ddof) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{combat, combat_parametric, ComBatParams};

    #[derive(serde::Deserialize)]
    struct OfficialCase {
        name: String,
        data: Vec<Vec<f64>>,
        batch: Vec<usize>,
        covariates: Option<Vec<Vec<f64>>>,
        mean_only: bool,
        reference: Option<usize>,
        expected: Vec<Vec<f64>>,
    }

    #[test]
    fn matches_sva_3_58_0() -> Result<(), Box<dyn std::error::Error>> {
        let cases: Vec<OfficialCase> =
            serde_json::from_str(include_str!("../../tests/data/combat_sva_3_58_0.json"))?;
        for case in cases {
            let params = ComBatParams {
                mean_only: case.mean_only,
                ref_batch: case.reference,
                par_prior: true,
            };
            let actual = match &case.covariates {
                None => combat_parametric(&case.data, &case.batch, params),
                Some(cov) => combat(&case.data, &case.batch, Some(cov), params),
            };
            assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
            for (i, (row, expected)) in actual.iter().zip(&case.expected).enumerate() {
                assert_eq!(row.len(), expected.len());
                for (j, (&a, &e)) in row.iter().zip(expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-9 + 1e-9 * e.abs(),
                        "{}[{i},{j}]: {a} != {e}",
                        case.name
                    );
                    if case.reference == Some(case.batch[j]) {
                        assert_eq!(a, case.data[i][j], "reference batch changed");
                    }
                }
            }
        }
        Ok(())
    }
    #[derive(serde::Deserialize)]
    struct NonparametricFixtures {
        cases: Vec<OfficialCase>,
    }

    #[test]
    fn nonparametric_matches_sva_3_58_0() -> Result<(), Box<dyn std::error::Error>> {
        let fixtures: NonparametricFixtures = serde_json::from_str(include_str!(
            "../../tests/data/combat_nonparametric_sva_3_58_0.json"
        ))?;
        for case in fixtures.cases {
            let actual = combat(
                &case.data,
                &case.batch,
                case.covariates.as_deref(),
                ComBatParams {
                    par_prior: false,
                    mean_only: case.mean_only,
                    ref_batch: case.reference,
                },
            );
            assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
            for (i, (row, expected)) in actual.iter().zip(&case.expected).enumerate() {
                assert_eq!(row.len(), expected.len(), "{}[{i}]", case.name);
                for (j, (&a, &e)) in row.iter().zip(expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-9 + 1e-9 * e.abs(),
                        "{}[{i},{j}]: {a} != {e}",
                        case.name
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn nonparametric_likelihood_underflow_stays_undefined() {
        // sva:::int.eprior(matrix(0, 2, 1000), c(0,0), c(2,2)) returns NaN.
        let (gamma, delta) = super::int_eprior(
            &vec![vec![0.0; 1000]; 2],
            &(0..1000).collect::<Vec<_>>(),
            &[0.0, 0.0],
            &[2.0, 2.0],
        );
        assert!(gamma.iter().chain(&delta).all(|v| v.is_nan()));
    }
}
