use crate::state::CellKey;
use crate::{HashMap, HashSet, PeptideId, SampleId, median_finite};

/// Whether the centering statistic is the per-sample median or mean of the
/// log2 peptide values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CenterStat {
    Median,
    Mean,
}

/// Sample-level median/mean centering. Mirrors the Python pipeline's
/// `NormalizationStage.apply_median_center` / `apply_mean_center`, which are
/// `is_dataset_level` methods: `_apply_dataset_normalizer` pivots the
/// peptide-level long table to the `(protein, canonical) x sample` wide matrix
/// (`aggfunc="sum"`), `np.log2`s after replacing 0 with NaN, subtracts each
/// column's median/mean (`median_center(axis=0)` / `x - x.mean(axis=0)`, both
/// NaN-skipping), then `2 ** result`. So the center is computed over the SAME
/// summed-canonical-peptide matrix the other dataset methods use, NOT over the
/// raw per-feature intensities (which back GlobalMedian/ConditionMedian via the
/// Python `get_median_map` SQL `MEDIAN(intensity)`). Because the center is a
/// single per-sample log2 offset applied uniformly to every peptide, it commutes
/// with the downstream sum-to-protein rollup; applying it here on the canonical
/// cells reproduces Python cell-for-cell.
///
/// Algorithm:
///  1. Collapse each allowed cell's peptidoforms to canonical peptides by SUM via
///     `peptide_to_canonical`; keep only strictly-positive finite sums and store
///     their `log2`.
///  2. Per sample, compute the median (or mean) of all log2 values in that
///     sample's column.
///  3. Write back `2 ** (log2(value) - center[sample])`. Samples with no valid
///     log2 value (empty column) are left unshifted.
pub(crate) fn apply_center_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    stat: CenterStat,
) {
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

    // (2) Per-sample center over the column's log2 values.
    let mut sample_log_values = HashMap::<SampleId, Vec<f64>>::new();
    for (cell, peptides) in &log2_cells {
        let column = sample_log_values.entry(cell.sample).or_default();
        column.extend(peptides.values().copied());
    }
    let centers = sample_log_values
        .into_iter()
        .filter_map(|(sample, mut values)| {
            let center = match stat {
                CenterStat::Median => median_finite(&mut values),
                CenterStat::Mean => {
                    // The values arrive in hash-map order, which changes between
                    // runs; sum them in a fixed order so the mean is reproducible.
                    values.sort_by(f64::total_cmp);
                    mokume_core::stats::mean_finite(&values)
                }
            }?;
            center.is_finite().then_some((sample, center))
        })
        .collect::<HashMap<SampleId, f64>>();

    // (3) Write back 2 ** (log2(value) - center[sample]).
    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(log_row) = log2_cells.get(cell) else {
            continue;
        };
        let center = centers.get(&cell.sample).copied().unwrap_or(0.0);
        peptides.clear();
        for (peptide, value) in log_row {
            peptides.insert(*peptide, (value - center).exp2());
        }
    }
}
