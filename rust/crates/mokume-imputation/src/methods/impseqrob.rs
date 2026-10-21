//! Robust sequential imputation matching rrcovNA::impSeqRob (alpha=0.9).
//! The initial robust support uses at most 500 official deterministic directions.
//! If its covariance lacks rank, norm's EM model supplies stochastic initial rows.
//! Protein rows and sample columns remain on the supplied (normally log2) scale.

mod normal;

use mokume_core::{MokumeError, ProteinId, Result, SampleId};

use crate::linalg::solve_linear_system;

/// MCD support fraction, matching rrcovNA's `alpha` default.
const ALPHA: f64 = 0.9;
/// Cap on the number of difference directions; below this every row pair is
/// used; above the cap rrcovNA's fixed generator selects 500 pairs.
const MAX_EXTRADIR_DIRECTIONS: usize = 500;
/// Numerical floor reused throughout for "effectively zero" magnitudes, matching
/// the `1e-12` guards in the Python reference.
const TINY: f64 = 1e-12;
/// Consistency constant turning a MAD into a robust standard-deviation estimate
/// (NumPy `* 1.4826`).
const MAD_TO_SD: f64 = 1.4826;

pub(crate) fn impseqrob_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    value_at: &mut F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let n_rows = proteins.len();
    let n_cols = samples.len();
    if n_rows == 0 || n_cols == 0 {
        return Ok(Vec::new());
    }

    // Read the matrix as `Option<f64>` cells (None = missing / non-finite).
    let matrix = proteins
        .iter()
        .map(|protein| {
            samples
                .iter()
                .map(|sample| value_at(*protein, *sample).filter(|value| value.is_finite()))
                .collect::<Vec<Option<f64>>>()
        })
        .collect::<Vec<_>>();

    let original_missing = matrix
        .iter()
        .map(|row| row.iter().map(Option::is_none).collect::<Vec<bool>>())
        .collect::<Vec<_>>();

    // Nothing to do when there are no missing cells.
    if original_missing.iter().all(|row| row.iter().all(|m| !*m)) {
        return Ok(Vec::new());
    }

    let filled = impseqrob_core(&matrix, n_cols)?;

    // Emit only the originally-missing cells whose fill is finite.
    let mut imputed = Vec::new();
    for (row_index, protein) in proteins.iter().enumerate() {
        for (col_index, sample) in samples.iter().enumerate() {
            if !original_missing[row_index][col_index] {
                continue;
            }
            let value = filled[row_index][col_index];
            if value.is_finite() {
                imputed.push((*protein, *sample, value));
            }
        }
    }
    Ok(imputed)
}

/// Robust sequential imputation core, mirroring Python `_impseqrob_core`.
///
/// Returns a fully filled `n x p` matrix in the original row order. `matrix`
/// cells are `Some(value)` when observed and `None` when missing.
fn impseqrob_core(matrix: &[Vec<Option<f64>>], p: usize) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    let miss_per_row = matrix
        .iter()
        .map(|row| row.iter().filter(|cell| cell.is_none()).count())
        .collect::<Vec<usize>>();

    // Stable argsort by ascending missing-count, then the inverse permutation
    // so the result can be restored to the original row order.
    let mut sort_order = (0..n).collect::<Vec<usize>>();
    sort_order.sort_by(|&left, &right| {
        miss_per_row[left]
            .cmp(&miss_per_row[right])
            .then(left.cmp(&right))
    });
    let mut unsort = vec![0usize; n];
    for (sorted_position, &original_index) in sort_order.iter().enumerate() {
        unsort[original_index] = sorted_position;
    }

    // `x` holds the working matrix in sorted-row order; `None` marks a hole.
    let mut x = sort_order
        .iter()
        .map(|&original| matrix[original].clone())
        .collect::<Vec<Vec<Option<f64>>>>();

    let mut incomplete_rows = sorted_incomplete_rows(&x);

    if incomplete_rows.is_empty() {
        return Ok(restore_order(&materialize(&x), &unsort));
    }

    initialize_support(&mut x, p)?;
    let complete_rows = sorted_complete_rows(&x);
    incomplete_rows = sorted_incomplete_rows(&x);

    robust_sequential_core(&mut x, p, &complete_rows, &incomplete_rows)?;
    Ok(restore_order(&materialize(&x), &unsort))
}

fn initialize_support(x: &mut [Vec<Option<f64>>], p: usize) -> Result<()> {
    let n_complete = sorted_complete_rows(x).len();
    let min_complete = (p as f64 / ALPHA).ceil() as usize;
    let needs_initialization =
        n_complete < min_complete || (n_complete < 5 * p && !initial_support_has_rank(x, p));
    if needs_initialization {
        let draws = normal::impute(x)?;
        let mut target = min_complete.max(n_complete + 1);
        loop {
            if target > x.len() {
                return Err(MokumeError::InvalidInput {
                    message: "impSeqRob cannot form a full-rank initial robust covariance"
                        .to_owned(),
                });
            }
            for i in n_complete..target {
                x[i] = draws[i].iter().copied().map(Some).collect();
            }
            if initial_support_has_rank(x, p) {
                break;
            }
            target += 1;
        }
    }

    Ok(())
}

/// Indices (in sorted order) of rows that have no missing cell.
fn sorted_complete_rows(x: &[Vec<Option<f64>>]) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter(|(_, row)| row.iter().all(Option::is_some))
        .map(|(index, _)| index)
        .collect()
}

/// Indices (in sorted order) of rows that still have a missing cell.
fn sorted_incomplete_rows(x: &[Vec<Option<f64>>]) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter(|(_, row)| row.iter().any(Option::is_none))
        .map(|(index, _)| index)
        .collect()
}

/// Replace every cell with a concrete `f64` (`NaN` for any residual hole).
fn materialize(x: &[Vec<Option<f64>>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .map(|cell| cell.unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Undo the missing-count sort, returning rows to their original positions.
fn restore_order(sorted: &[Vec<f64>], unsort: &[usize]) -> Vec<Vec<f64>> {
    unsort
        .iter()
        .map(|&sorted_position| sorted[sorted_position].clone())
        .collect()
}

/// Match rankMM's scale-dependent cutoff for a symmetric covariance.
fn initial_support_has_rank(x: &[Vec<Option<f64>>], p: usize) -> bool {
    let data = x
        .iter()
        .filter(|row| row.iter().all(Option::is_some))
        .map(|row| {
            row.iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let Some(robust) = cov_stahel_donoho(&data, p, h_alpha_n(data.len(), p)) else {
        return false;
    };
    let mut a = robust.cov;
    let norm = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    for _ in 0..100 {
        let off = a
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().skip(i + 1))
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if off <= norm * f64::EPSILON {
            break;
        }
        for q in 1..p {
            for r in 0..q {
                rotate_rank_pair(&mut a, r, q);
            }
        }
    }
    let maximum = (0..p).map(|i| a[i][i].abs()).fold(0.0, f64::max);
    let tolerance = p as f64 * f64::EPSILON * maximum;
    (0..p).all(|i| a[i][i].abs() >= tolerance) && maximum > 0.0
}

fn rotate_rank_pair(a: &mut [Vec<f64>], r: usize, q: usize) {
    let off = a[r][q];
    if off == 0.0 {
        return;
    }
    let theta = (a[q][q] - a[r][r]) / (2.0 * off);
    let t = if theta == 0.0 {
        1.0
    } else {
        theta.signum() / (theta.abs() + theta.hypot(1.0))
    };
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    let rotated = a
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != r && *k != q)
        .map(|(k, row)| (k, c * row[r] - s * row[q], s * row[r] + c * row[q]))
        .collect::<Vec<_>>();
    for (k, left, right) in rotated {
        a[k][r] = left;
        a[r][k] = left;
        a[k][q] = right;
        a[q][k] = right;
    }
    a[r][r] -= t * off;
    a[q][q] += t * off;
    a[r][q] = 0.0;
    a[q][r] = 0.0;
}

/// Robust sequential imputation over the incomplete rows, mirroring the
/// main loop of Python `_impseqrob_core` once a robust covariance is available.
fn robust_sequential_core(
    x: &mut [Vec<Option<f64>>],
    p: usize,
    complete_rows: &[usize],
    incomplete_rows: &[usize],
) -> Result<()> {
    let n_complete = complete_rows.len();
    let complete_data = complete_rows
        .iter()
        .map(|&index| {
            x[index]
                .iter()
                .map(|cell| cell.unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let h0 = h_alpha_n(n_complete, p);
    let robust =
        cov_stahel_donoho(&complete_data, p, h0).ok_or_else(|| MokumeError::InvalidInput {
            message: "impSeqRob cannot estimate a nondegenerate robust covariance".to_owned(),
        })?;

    let mut covx = robust.cov;
    let mut mx = robust.mean;
    let directions = robust.directions;
    let med_y = robust.med_y;
    let mad_y = robust.mad_y;

    // Sorted outlyingness list (`ds_vals`); grows as imputed rows are added.
    let mut ds_vals = robust.outlyingness.clone();
    ds_vals.sort_by(f64::total_cmp);

    let mut h = h0;
    let mut n_update = h0;

    // `flag == false` marks an accepted (non-outlying) row used to grow the
    // estimates; the MCD support of the complete rows starts accepted.
    let mut accepted_prev = false;
    let mut icovx: Option<Vec<Vec<f64>>> = None;
    let mut mxo = mx.clone();

    for (loop_index, &row_index) in incomplete_rows.iter().enumerate() {
        if loop_index == 0 {
            icovx = invert_matrix(&covx, p);
        } else if accepted_prev {
            if let Some(current) = icovx.as_ref() {
                let denom = (n_update as f64 - 1.0).max(1.0).sqrt();
                let prev_row = &x_row_values(x, incomplete_rows[loop_index - 1]);
                let f = prev_row
                    .iter()
                    .zip(mxo.iter())
                    .map(|(value, mean)| (value - mean) / denom)
                    .collect::<Vec<f64>>();
                let scale = (n_update as f64 - 1.0) / (n_update as f64 - 2.0).max(1.0);
                let scaled = scale_matrix(current, scale);
                icovx = Some(update_invcov(&scaled, &f));
            }
        }

        let inverse = icovx.as_ref().ok_or_else(|| MokumeError::InvalidInput {
            message: "impSeqRob robust covariance is singular".to_owned(),
        })?;

        impute_robust_row(&mut x[row_index], &mx, inverse)?;

        // Outlyingness of the now-complete row against the robust directions.
        let row_values = x_row_values(x, row_index);
        let dx = outlyingness(&row_values, &directions, &med_y, &mad_y);

        // Accept the row when its outlyingness is within the current threshold.
        let threshold_index = h.saturating_sub(1).min(ds_vals.len().saturating_sub(1));
        let threshold = ds_vals
            .get(threshold_index)
            .copied()
            .unwrap_or(f64::INFINITY);
        if dx <= threshold {
            accepted_prev = true;
            n_update += 1;
            h += 1;
            mxo = mx.clone();
            // mx = ((n_update - 1) mxo + row) / n_update.
            mx = mxo
                .iter()
                .zip(row_values.iter())
                .map(|(old, value)| ((n_update as f64 - 1.0) * old + value) / n_update as f64)
                .collect();
            // covx = (n_update - 2)/(n_update - 1) covx + outer(row - mxo) / (n_update - 1).
            let cov_scale = (n_update as f64 - 2.0) / (n_update as f64 - 1.0);
            let outer_scale = 1.0 / (n_update as f64 - 1.0);
            let delta = row_values
                .iter()
                .zip(mxo.iter())
                .map(|(value, old)| value - old)
                .collect::<Vec<f64>>();
            for a in 0..p {
                for b in 0..p {
                    covx[a][b] = cov_scale * covx[a][b] + outer_scale * delta[a] * delta[b];
                }
            }
        } else {
            accepted_prev = false;
            mxo = mx.clone();
        }

        // Append dx to the sorted outlyingness list.
        let insert_at = ds_vals.partition_point(|value| *value < dx);
        ds_vals.insert(insert_at, dx);
    }
    Ok(())
}

fn impute_robust_row(row: &mut [Option<f64>], mx: &[f64], inverse: &[Vec<f64>]) -> Result<()> {
    let missing = missing_columns(row);
    let observed = observed_columns(row);
    // Conditional mean via the inverse covariance blocks:
    // x_m = mu_m - icov_mm^-1 icov_mo (x_o - mu_o).
    if observed.is_empty() {
        for &col in &missing {
            row[col] = Some(mx[col]);
        }
    } else {
        let icov_mm = submatrix(inverse, &missing, &missing);
        let residual = observed
            .iter()
            .map(|&col| row[col].unwrap_or(mx[col]) - mx[col])
            .collect::<Vec<f64>>();
        // rhs = icov_mo (x_o - mu_o).
        let rhs = missing
            .iter()
            .map(|&m| {
                observed
                    .iter()
                    .enumerate()
                    .map(|(k, &o)| inverse[m][o] * residual[k])
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        let adjustment =
            solve_linear_system(&icov_mm, &rhs).ok_or_else(|| MokumeError::InvalidInput {
                message: "impSeqRob conditional covariance is singular".to_owned(),
            })?;
        for (k, &m) in missing.iter().enumerate() {
            row[m] = Some(mx[m] - adjustment[k]);
        }
    }

    Ok(())
}

/// Snapshot a row's current values (holes already filled by this point).
fn x_row_values(x: &[Vec<Option<f64>>], row_index: usize) -> Vec<f64> {
    x[row_index]
        .iter()
        .map(|cell| cell.unwrap_or(f64::NAN))
        .collect()
}

/// `h` for the MCD support, matching `robustbase:::h.alpha.n`:
/// `floor(2 * h2 - n + 2 (n - h2) alpha)` with `h2 = (n + p + 1) / 2`.
fn h_alpha_n(n: usize, p: usize) -> usize {
    let h2 = (n + p).div_ceil(2);
    let value = 2.0 * h2 as f64 - n as f64 + 2.0 * (n as f64 - h2 as f64) * ALPHA;
    value.floor().max(1.0) as usize
}

/// Robust covariance summary returned by the Stahel-Donoho estimator.
struct RobustCov {
    cov: Vec<Vec<f64>>,
    mean: Vec<f64>,
    directions: Vec<Vec<f64>>,
    med_y: Vec<f64>,
    mad_y: Vec<f64>,
    outlyingness: Vec<f64>,
}

/// Stahel-Donoho robust covariance over the complete rows, mirroring Python
/// `_covSD`. Returns `None` when no usable direction survives normalisation.
fn cov_stahel_donoho(data: &[Vec<f64>], p: usize, h: usize) -> Option<RobustCov> {
    let raw_directions = extradir(data, p);
    let mut directions = Vec::new();
    for direction in &raw_directions {
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm > TINY {
            directions.push(
                direction
                    .iter()
                    .map(|value| value / norm)
                    .collect::<Vec<f64>>(),
            );
        }
    }
    if directions.is_empty() {
        return None;
    }

    // Projections Y = data @ A^T, robust centre/scale per direction.
    let projections = data
        .iter()
        .map(|row| {
            directions
                .iter()
                .map(|direction| {
                    row.iter()
                        .zip(direction.iter())
                        .map(|(value, weight)| value * weight)
                        .sum::<f64>()
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let direction_count = directions.len();
    let mut med_y = vec![0.0; direction_count];
    let mut mad_y = vec![0.0; direction_count];
    for column in 0..direction_count {
        let mut values = projections
            .iter()
            .map(|row| row[column])
            .collect::<Vec<f64>>();
        let median = median_of(&mut values);
        med_y[column] = median;
        let mut deviations = values
            .iter()
            .map(|value| (value - median).abs())
            .collect::<Vec<f64>>();
        let mad = median_of(&mut deviations) * MAD_TO_SD;
        if mad == 0.0 || !mad.is_finite() {
            return None;
        }
        mad_y[column] = mad;
    }

    // Per-row outlyingness d = max over directions of |Y - medY| / madY.
    let outlyingness = projections
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, value)| (value - med_y[column]).abs() / mad_y[column])
                .fold(0.0_f64, f64::max)
        })
        .collect::<Vec<f64>>();

    // MCD support: the `h` rows with smallest outlyingness (stable argsort).
    let mut order = (0..data.len()).collect::<Vec<usize>>();
    order.sort_by(|&left, &right| {
        outlyingness[left]
            .total_cmp(&outlyingness[right])
            .then(left.cmp(&right))
    });
    let support = &order[..h.min(order.len())];
    let support_rows = support
        .iter()
        .map(|&index| data[index].clone())
        .collect::<Vec<_>>();

    let mean = column_means(&support_rows, p);
    let cov = sample_covariance(&support_rows, &mean, p);

    Some(RobustCov {
        cov,
        mean,
        directions,
        med_y,
        mad_y,
        outlyingness,
    })
}

/// Original rrcovNA direction selection: all pairs up to 500, then 500
/// draws from the package's fixed 16-bit congruential generator.
fn extradir(data: &[Vec<f64>], _p: usize) -> Vec<Vec<f64>> {
    let n = data.len();
    let difference =
        |i: usize, j: usize| data[i].iter().zip(&data[j]).map(|(a, b)| a - b).collect();
    if n.saturating_mul(n.saturating_sub(1)) / 2 <= MAX_EXTRADIR_DIRECTIONS {
        return (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| difference(i, j))
            .collect();
    }
    let mut seed = 0_u32;
    let mut draw = || {
        seed = (seed * 5761 + 999) % 65536;
        (seed as usize * n) / 65536
    };
    (0..MAX_EXTRADIR_DIRECTIONS)
        .map(|_| {
            let i = draw();
            let mut j = draw();
            while i == j {
                j = draw();
            }
            difference(i, j)
        })
        .collect()
}

/// Sherman-Morrison rank-one inverse update, mirroring `_update_invcov`:
/// `A - (A f)(A f)^T / (1 + f^T A f)`. Returns `A` unchanged when the
/// denominator is effectively zero.
fn update_invcov(a: &[Vec<f64>], f: &[f64]) -> Vec<Vec<f64>> {
    let p = f.len();
    let af = (0..p)
        .map(|row| {
            a[row]
                .iter()
                .zip(f.iter())
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
        })
        .collect::<Vec<f64>>();
    let denom = 1.0
        + f.iter()
            .zip(af.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f64>();
    if denom.abs() <= TINY {
        return a.iter().map(Clone::clone).collect();
    }
    let mut result = a.iter().map(Clone::clone).collect::<Vec<Vec<f64>>>();
    for row in 0..p {
        for column in 0..p {
            result[row][column] -= af[row] * af[column] / denom;
        }
    }
    result
}

/// Multiply every entry of a square matrix by a scalar.
fn scale_matrix(matrix: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect())
        .collect()
}

/// Outlyingness of a single row against the robust directions:
/// `max_k |row . A_k - medY_k| / madY_k`.
fn outlyingness(row: &[f64], directions: &[Vec<f64>], med_y: &[f64], mad_y: &[f64]) -> f64 {
    directions
        .iter()
        .enumerate()
        .map(|(column, direction)| {
            let projection = row
                .iter()
                .zip(direction.iter())
                .map(|(value, weight)| value * weight)
                .sum::<f64>();
            (projection - med_y[column]).abs() / mad_y[column]
        })
        .fold(0.0_f64, f64::max)
}

/// Invert a square matrix by solving against the identity columns; `None` when
/// singular. Reuses the shared Gauss-Jordan solver one column at a time.
fn invert_matrix(matrix: &[Vec<f64>], p: usize) -> Option<Vec<Vec<f64>>> {
    let mut columns = Vec::with_capacity(p);
    for column in 0..p {
        let mut unit = vec![0.0; p];
        unit[column] = 1.0;
        columns.push(solve_linear_system(matrix, &unit)?);
    }
    // `columns[c]` is the c-th column of the inverse; transpose into rows.
    let mut inverse = vec![vec![0.0; p]; p];
    for (column, solution) in columns.iter().enumerate() {
        for (row, &value) in solution.iter().enumerate() {
            inverse[row][column] = value;
        }
    }
    Some(inverse)
}

/// Column indices of the observed cells of a row.
fn observed_columns(row: &[Option<f64>]) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, cell)| cell.is_some())
        .map(|(index, _)| index)
        .collect()
}

/// Column indices of the missing cells of a row.
fn missing_columns(row: &[Option<f64>]) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, cell)| cell.is_none())
        .map(|(index, _)| index)
        .collect()
}

/// Extract the submatrix selected by `rows` and `cols`.
fn submatrix(matrix: &[Vec<f64>], rows: &[usize], cols: &[usize]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|&row| cols.iter().map(|&col| matrix[row][col]).collect())
        .collect()
}

/// Plain per-column mean over the supplied rows.
fn column_means(rows: &[Vec<f64>], p: usize) -> Vec<f64> {
    let count = rows.len();
    (0..p)
        .map(|col| {
            if count == 0 {
                0.0
            } else {
                rows.iter().map(|row| row[col]).sum::<f64>() / count as f64
            }
        })
        .collect()
}

/// Sample covariance (`ddof = 1`) over the supplied rows, matching NumPy
/// `np.cov(rows, rowvar=False, ddof=1)`. A single row yields all zeros.
fn sample_covariance(rows: &[Vec<f64>], mean: &[f64], p: usize) -> Vec<Vec<f64>> {
    let count = rows.len();
    let mut cov = vec![vec![0.0; p]; p];
    if count < 2 {
        return cov;
    }
    let denom = (count - 1) as f64;
    for a in 0..p {
        for b in 0..p {
            let sum = rows
                .iter()
                .map(|row| (row[a] - mean[a]) * (row[b] - mean[b]))
                .sum::<f64>();
            cov[a][b] = sum / denom;
        }
    }
    cov
}

/// Median of a slice (NumPy `np.median`: lower-of-middle-two averaged). Sorts in
/// place with `total_cmp`; returns `0.0` for an empty slice.
fn median_of(values: &mut [f64]) -> f64 {
    let count = values.len();
    if count == 0 {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    if count % 2 == 1 {
        values[count / 2]
    } else {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    }
}
