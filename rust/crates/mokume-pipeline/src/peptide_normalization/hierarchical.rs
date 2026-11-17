use crate::peptide_normalization::QuantilePeptideKey;
use crate::state::CellKey;
use crate::{
    HashMap, HashSet, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator, PeptideId,
    SampleId, StringIdRegistry,
};
use rayon::prelude::ParallelSliceMut;
/// Samples in DirectLFQ's quadratic alignment subset for hierarchical sample
/// normalization.
const HIERARCHICAL_NUM_SAMPLES_QUADRATIC: usize = 50;

/// DirectLFQ sample normalization on canonical-peptide log2 profiles. Samples
/// follow the pipeline pivot's name order; the shared DirectLFQ primitive picks
/// pairs by variance, shifts by their median difference, and merges profiles.
pub(crate) fn apply_hierarchical_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    sample_registry: &StringIdRegistry<SampleId>,
) {
    // (No trimming in hierarchical normalization.)

    // (1) Collapse to canonical and log2 the strictly-positive sums.
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let mut summed = HashMap::<PeptideId, f64>::new();
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *summed.entry(canonical).or_insert(0.0) += *intensity;
        }
        let log_row = summed
            .into_iter()
            .filter(|(_, value)| value.is_finite() && *value > 0.0)
            .map(|(canonical, value)| (canonical, value.log2()))
            .collect::<HashMap<_, _>>();
        if !log_row.is_empty() {
            log2_cells.insert(*cell, log_row);
        }
    }
    if log2_cells.is_empty() {
        return;
    }

    // (2) One column per sample maps each (protein, canonical) row to its log2
    // value; compute per-column log2 shifts with the official DirectLFQ algorithm.
    let mut columns = HashMap::<SampleId, Vec<(QuantilePeptideKey, f64)>>::new();
    for (cell, peptides) in &log2_cells {
        let column = columns.entry(cell.sample).or_default();
        for (peptide, value) in peptides {
            column.push((
                QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                },
                *value,
            ));
        }
    }
    let shifts = hierarchical_shifts_by_sample(columns, sample_registry);

    // (6) Write back: 2 ** (log2(value) + shift[sample]).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let Some(&shift) = shifts.get(&cell.sample) else {
            continue;
        };
        peptides.clear();
        for (peptide, value) in log_row {
            peptides.insert(*peptide, (value + shift).exp2());
        }
    }
}

/// DirectLFQ log2 shift of each sample in `columns` (sample -> distinct rows
/// with their log2 values). Samples are ordered ascending by NAME (then SampleId as a tie-break),
/// matching the Python `pivot_table(columns=SAMPLE_ID)` column order. The
/// ordering is load-bearing: it drives the pairwise distance indices, merge
/// tie-breaking, and the anchor sample.
pub(crate) fn hierarchical_shifts_by_sample(
    mut columns: HashMap<SampleId, Vec<(QuantilePeptideKey, f64)>>,
    sample_registry: &StringIdRegistry<SampleId>,
) -> HashMap<SampleId, f64> {
    let mut samples = columns.keys().copied().collect::<Vec<_>>();
    samples.sort_by(|left, right| {
        sample_registry
            .resolve(*left)
            .cmp(&sample_registry.resolve(*right))
            .then_with(|| left.cmp(right))
    });
    let ordered = samples
        .iter()
        .map(|sample| columns.remove(sample).unwrap_or_default())
        .collect::<Vec<_>>();
    let shifts = hierarchical_compute_shifts(ordered, HIERARCHICAL_NUM_SAMPLES_QUADRATIC);
    samples.into_iter().zip(shifts).collect()
}

/// Preserve the sparse pivot's feature order when using the shared DirectLFQ
/// sample-alignment kernel. Missing canonical peptides remain NaN.
fn hierarchical_compute_shifts(
    columns: Vec<Vec<(QuantilePeptideKey, f64)>>,
    num_samples_quadratic: usize,
) -> Vec<f64> {
    // Distinct keys without first collecting every value's key.
    let mut keys = columns
        .par_iter()
        .fold(HashSet::new, |mut keys, column| {
            keys.extend(column.iter().map(|&(key, _)| key));
            keys
        })
        .reduce(HashSet::new, |mut left, right| {
            left.extend(right);
            left
        })
        .into_iter()
        .collect::<Vec<_>>();
    keys.par_sort_unstable();
    // Columns keep only their observed rows; a dense keys x samples matrix
    // dominated peak memory on large studies.
    let columns = columns
        .into_par_iter()
        .map(|column| {
            column
                .into_iter()
                .filter_map(|(key, value)| keys.binary_search(&key).ok().map(|row| (row, value)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let counts = columns.iter().map(Vec::len).collect();
    let shifts =
        mokume_quant::direct_lfq_sparse_sample_shifts(&columns, keys.len(), num_samples_quadratic);
    mask_unaligned_samples(shifts, counts, num_samples_quadratic)
}

pub(crate) fn hierarchical_sample_shifts(
    rows: &[Vec<f64>],
    n_samples: usize,
    quadratic_limit: usize,
) -> Vec<f64> {
    let shifts = mokume_quant::direct_lfq_sample_shifts(rows, n_samples, quadratic_limit);
    let counts = (0..n_samples)
        .map(|sample| rows.iter().filter(|row| row[sample].is_finite()).count())
        .collect();
    mask_unaligned_samples(shifts, counts, quadratic_limit)
}

/// Upstream quadratic alignment masks samples with fewer than two observed
/// features. Linear-to-reference samples are not subject to that masking.
fn mask_unaligned_samples(
    mut shifts: Vec<f64>,
    counts: Vec<usize>,
    quadratic_limit: usize,
) -> Vec<f64> {
    let mut counts = counts.into_iter().enumerate().collect::<Vec<_>>();
    counts.sort_by(|(i, a), (j, b)| b.cmp(a).then(i.cmp(j)));
    for (sample, count) in counts.into_iter().take(quadratic_limit) {
        if count < 2 {
            shifts[sample] = f64::NAN;
        }
    }
    shifts
}
