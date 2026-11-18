use crate::peptide_normalization::QuantilePeptideKey;
use crate::state::CellKey;
use crate::{
    HashMap, HashSet, IntoParallelRefIterator, ParallelIterator, PeptideId, SampleId, median_finite,
};

/// LOESS (LOcally Estimated Scatterplot Smoothing) sample normalization.
/// Mirrors the Python `LOESSNormalizer.fit_transform` (frac=0.75,
/// reference="median") applied by `NormalizationStage.apply_loess` in log2
/// space (`_apply_dataset_normalizer(..., log_space=True)`).
///
/// Algorithm (Cleveland WS, J Am Stat Assoc 1979; Yang et al, NAR 2002):
/// build the (protein, canonical) x sample matrix in log2 space. Let `ref[row]`
/// be the across-sample row reference (pandas `median(axis=1)`, NaN-skipping; for
/// an even count this is the mean of the two central order statistics). For each
/// sample column with at least 10 finite (sample, ref) pairs, form the MA points
/// `M = sample - ref` (log-ratio) and `A = (sample + ref) / 2` (average), fit
/// `M ~ A` by LOWESS (statsmodels `lowess`, frac=0.75, it=3, delta=0.0,
/// return_sorted=False), and replace each fitted value with `sample - fitted(A)`.
/// Columns with fewer than 10 valid pairs are left UNCHANGED (the Python
/// `if valid.sum() < 10: continue` guard). The pipeline log2s before
/// quantification and the rollup consumes linear values, so results are
/// exponentiated back here.
pub(crate) fn apply_loess_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    let log2_cells = canonical_log2_cells(cells, allowed_cells, peptide_to_canonical);
    let reference = row_median_reference(&log2_cells);

    // Fit and apply per sample column. A sample is corrected only if it has at
    // least 10 finite (sample, reference) pairs; otherwise it passes through.
    let mut by_sample = HashMap::<SampleId, Vec<(QuantilePeptideKey, f64)>>::new();
    for (cell, peptides) in &log2_cells {
        for (peptide, value) in peptides {
            by_sample.entry(cell.sample).or_default().push((
                QuantilePeptideKey {
                    protein: cell.protein,
                    peptide: *peptide,
                },
                *value,
            ));
        }
    }

    // Each sample is fitted against the fixed reference and writes only its own
    // keys, so samples run in parallel without changing the result.
    let corrected = by_sample
        .par_iter()
        .flat_map_iter(|(sample, rows)| {
            loess_correct_sample(rows, &reference)
                .into_iter()
                .map(move |(key, value)| ((key, *sample), value))
        })
        .collect::<HashMap<(QuantilePeptideKey, SampleId), f64>>();

    // Write canonical results back into the cells, exponentiating out of log2.
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        peptides.clear();
        for peptide in log_row.keys() {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            };
            if let Some(value) = corrected.get(&(key, cell.sample)).copied() {
                peptides.insert(*peptide, value.exp2());
            }
        }
    }
}

/// Collapse each allowed cell to canonical peptides by SUM, then log2 the
/// strictly positive finite values (matching `np.log2(wide.replace(0, nan))`).
/// Cells without such a value are left out.
fn canonical_log2_cells(
    cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) -> HashMap<CellKey, HashMap<PeptideId, f64>> {
    let mut log2_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells {
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
    log2_cells
}

/// Per-row (protein, canonical) reference: the NaN-skipping median across the
/// samples that observe the row.
fn row_median_reference(
    log2_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
) -> HashMap<QuantilePeptideKey, f64> {
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
        .filter_map(|(key, mut values)| median_finite(&mut values).map(|median| (key, median)))
        .collect()
}

/// LOESS-correct one sample's log2 values against the per-row reference.
/// A sample is corrected only if it has at least 10 finite (sample, reference)
/// pairs; otherwise its paired values pass through.
fn loess_correct_sample(
    rows: &[(QuantilePeptideKey, f64)],
    reference: &HashMap<QuantilePeptideKey, f64>,
) -> Vec<(QuantilePeptideKey, f64)> {
    // Pair each row's log2 value with its reference; drop rows lacking a
    // finite reference (matches the `sample.notna() & ref.notna()` mask).
    let mut paired = rows
        .iter()
        .filter_map(|(key, value)| {
            reference
                .get(key)
                .filter(|r| r.is_finite() && value.is_finite())
                .map(|r| (*key, *value, *r))
        })
        .collect::<Vec<_>>();
    if paired.len() < 10 {
        // < 10 valid points: leave the column unchanged.
        return paired
            .into_iter()
            .map(|(key, value, _)| (key, value))
            .collect();
    }
    // Deterministic tie-break: sort by (A ascending, key) before fitting so
    // the window selection and unsort match a stable reference order.
    paired.sort_by(|left, right| {
        let a_left = (left.1 + left.2) / 2.0;
        let a_right = (right.1 + right.2) / 2.0;
        a_left
            .total_cmp(&a_right)
            .then_with(|| left.0.cmp(&right.0))
    });
    let a_values = paired
        .iter()
        .map(|(_, value, r)| (value + r) / 2.0)
        .collect::<Vec<_>>();
    let m_values = paired
        .iter()
        .map(|(_, value, r)| value - r)
        .collect::<Vec<_>>();
    let fitted = mokume_core::stats::lowess_fit(&a_values, &m_values, LOESS_FRAC, LOESS_ITERATIONS);
    paired
        .iter()
        .zip(fitted)
        .map(|((key, value, _), bias)| (*key, value - bias))
        .collect()
}

/// LOWESS smoothing fraction used by `LOESSNormalizer` (frac=0.75).
const LOESS_FRAC: f64 = 0.75;
/// Robustifying reweight iterations matching the statsmodels `lowess` default
/// (`it=3`).
const LOESS_ITERATIONS: usize = 3;
