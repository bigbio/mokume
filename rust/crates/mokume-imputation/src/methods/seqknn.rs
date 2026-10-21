use mokume_core::{MokumeError, ProteinId, Result, SampleId};

/// Incomplete rows whose distances to the shared complete rows are computed
/// in one pass, so each complete row is read once per batch.
const BATCH: usize = 32;
/// Complete rows compared with one incomplete row at a time. Each distance is
/// summed in column order, so its additions wait on each other; independent
/// sums for several rows keep the processor busy without changing any sum.
const LANES: usize = 16;

pub(crate) fn seqknn_imputed_values<F>(
    proteins: &[ProteinId],
    samples: &[SampleId],
    n_neighbors: usize,
    value_at: &mut F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let n_neighbors = n_neighbors.max(1);
    let matrix = proteins
        .iter()
        .map(|protein| {
            samples
                .iter()
                .map(|sample| value_at(*protein, *sample).filter(|value| value.is_finite()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if matrix.is_empty() || samples.is_empty() {
        return Ok(Vec::new());
    }

    let missing_counts = matrix
        .iter()
        .map(|row| row.iter().filter(|value| value.is_none()).count())
        .collect::<Vec<_>>();
    if missing_counts.iter().all(|count| *count == 0) {
        return Ok(Vec::new());
    }

    // Rows by missing count; the stable sort keeps ties in row order.
    let mut order = (0..proteins.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| missing_counts[*index]);
    let n_complete = missing_counts.iter().filter(|count| **count == 0).count();
    if n_complete == 0 {
        return Err(MokumeError::InvalidInput {
            message: "SeqKNN requires an initial set of complete protein rows".to_owned(),
        });
    }

    let pool = imputed_pool(&matrix, &order, n_complete, n_neighbors);
    Ok(imputed_cells(
        proteins,
        samples,
        &matrix,
        &order[n_complete..],
        &pool[n_complete * samples.len()..],
    ))
}

/// The originally missing cells of the incomplete rows, from their imputed rows.
fn imputed_cells(
    proteins: &[ProteinId],
    samples: &[SampleId],
    matrix: &[Vec<Option<f64>>],
    incomplete: &[usize],
    imputed_rows: &[f64],
) -> Vec<(ProteinId, SampleId, f64)> {
    let mut imputed = Vec::new();
    for (index, row) in incomplete
        .iter()
        .zip(imputed_rows.chunks_exact(samples.len()))
    {
        for ((sample, cell), value) in samples.iter().zip(&matrix[*index]).zip(row) {
            if cell.is_none() && value.is_finite() {
                imputed.push((proteins[*index], *sample, *value));
            }
        }
    }
    imputed
}

/// Rows in imputation order, row-major: the complete rows, then each
/// incomplete row once imputed from the rows before it. Every row in the pool
/// is complete, so it can stand as a neighbour for any sample.
fn imputed_pool(
    matrix: &[Vec<Option<f64>>],
    order: &[usize],
    n_complete: usize,
    n_neighbors: usize,
) -> Vec<f64> {
    let n_cols = matrix[order[0]].len();
    let mut pool = order[..n_complete]
        .iter()
        .flat_map(|index| matrix[*index].iter().flatten().copied())
        .collect::<Vec<_>>();
    for batch in order[n_complete..].chunks(BATCH) {
        let targets = batch
            .iter()
            .map(|index| Target::new(&matrix[*index]))
            .collect::<Vec<_>>();
        let shared = pool.len();
        let distances = shared_distances(&pool, n_cols, &targets);
        for (target, mut candidates) in targets.iter().zip(distances) {
            // Rows imputed earlier in this batch.
            candidates.extend(
                pool[shared..]
                    .chunks_exact(n_cols)
                    .map(|row| target.distance(row)),
            );
            let row = target.imputed_row(&pool, n_cols, candidates, n_neighbors);
            pool.extend_from_slice(&row);
        }
    }
    pool
}

/// Distance of each target to each pool row, in pool order.
fn shared_distances(pool: &[f64], n_cols: usize, targets: &[Target]) -> Vec<Vec<f64>> {
    let rows = pool.len() / n_cols;
    let mut distances = targets
        .iter()
        .map(|_| Vec::with_capacity(rows + BATCH))
        .collect::<Vec<_>>();
    let mut blocks = pool.chunks_exact(n_cols * LANES);
    for block in &mut blocks {
        let lanes: [&[f64]; LANES] =
            std::array::from_fn(|lane| &block[lane * n_cols..(lane + 1) * n_cols]);
        for (target, out) in targets.iter().zip(distances.iter_mut()) {
            out.extend(target.lane_distances(lanes));
        }
    }
    for row in blocks.remainder().chunks_exact(n_cols) {
        for (target, out) in targets.iter().zip(distances.iter_mut()) {
            out.push(target.distance(row));
        }
    }
    distances
}

/// An incomplete row: its observed cells, in column order, and its missing columns.
struct Target {
    observed: Vec<(usize, f64)>,
    missing: Vec<usize>,
}

impl Target {
    fn new(row: &[Option<f64>]) -> Self {
        Self {
            observed: row
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.map(|value| (index, value)))
                .collect(),
            missing: row
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.is_none().then_some(index))
                .collect(),
        }
    }

    /// Squared Euclidean distance over the observed columns, as the original
    /// SeqKnn 1.0.1 nnmiss.
    fn distance(&self, row: &[f64]) -> f64 {
        let mut squared = 0.0;
        for (index, value) in &self.observed {
            let delta = row[*index] - value;
            squared += delta * delta;
        }
        squared
    }

    /// `distance` to several rows at once; each sum keeps the column order.
    fn lane_distances(&self, rows: [&[f64]; LANES]) -> [f64; LANES] {
        let mut squared = [0.0; LANES];
        for (index, value) in &self.observed {
            for (sum, row) in squared.iter_mut().zip(rows) {
                let delta = row[*index] - value;
                *sum += delta * delta;
            }
        }
        squared
    }

    /// The row with each missing cell set to the inverse-distance weighted
    /// mean of its nearest pool rows (ties by pool order).
    fn imputed_row(
        &self,
        pool: &[f64],
        n_cols: usize,
        distances: Vec<f64>,
        n_neighbors: usize,
    ) -> Vec<f64> {
        let mut neighbors = distances
            .into_iter()
            .enumerate()
            .map(|(index, distance)| (distance, index))
            .collect::<Vec<_>>();
        let nearest = |left: &(f64, usize), right: &(f64, usize)| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        if neighbors.len() > n_neighbors {
            neighbors.select_nth_unstable_by(n_neighbors, nearest);
            neighbors.truncate(n_neighbors);
        }
        neighbors.sort_unstable_by(nearest);

        let mut row = vec![f64::NAN; n_cols];
        for (index, value) in &self.observed {
            row[*index] = *value;
        }
        for &sample_index in &self.missing {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            for (distance, index) in &neighbors {
                let weight = 1.0 / (distance + 1e-15);
                weighted_sum += pool[index * n_cols + sample_index] * weight;
                weight_sum += weight;
            }
            row[sample_index] = weighted_sum / weight_sum;
        }
        row
    }
}
