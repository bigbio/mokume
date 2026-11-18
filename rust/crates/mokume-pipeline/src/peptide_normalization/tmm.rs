use super::sum_canonical_cell;
use crate::peptide_normalization::QuantilePeptideKey;
use crate::state::CellKey;
use crate::{HashMap, HashSet, PeptideId, SampleId, StringIdRegistry, tmm_norm_factors};

/// Fit edgeR TMM factors on the canonical-peptide x sample linear matrix.
/// Divide original peptidoforms by their sample's centered composition factor;
/// the later canonical sum therefore commutes with this scaling. This is not
/// edgeR CPM: the library-size normalization is used during factor fitting only.
/// Sample names are sorted to preserve the pipeline pivot's reference tie-break.
pub(crate) fn apply_tmm_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) {
    // (1) Collapse each allowed cell's peptidoforms to canonical peptides by
    // SUM. Rows of the wide matrix are (protein, canonical); columns are
    // samples. Keep every finite sum (including <= 0) so the matrix has the
    // exact shape Python's pivot produces; `tmm_norm_factors` treats zero /
    // non-finite as missing internally.
    let mut summed_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let summed = sum_canonical_cell(peptides, peptide_to_canonical);
        if !summed.is_empty() {
            summed_cells.insert(*cell, summed);
        }
    }
    if summed_cells.is_empty() {
        return;
    }

    let row_keys = tmm_row_keys(&summed_cells);
    let (sample_ids, column_names) = tmm_sample_columns(&summed_cells, samples);
    let matrix = tmm_matrix(&summed_cells, &row_keys, &sample_ids);

    // Compute centered edgeR factors and divide the original peptidoforms.
    //
    // TMM's factor is a single per-sample scalar, so it commutes with the
    // downstream sum/median canonical collapse that `finalize` performs. We
    // therefore scale the ORIGINAL peptidoform cells in place rather than
    // replacing them with the canonical-summed values: `finalize` remains the
    // sole place that collapses peptidoforms to canonical, which avoids a
    // double collapse (writing canonical ids back into the cell would let
    // `finalize`'s own `collapse_to_canonical` re-map them and merge distinct
    // canonicals, corrupting median/`Median` outputs). The wide matrix built
    // above (canonical-summed, matching Python's pivot) is only used to compute
    // the factors, exactly reproducing Python's `TMMNormalizer` fit on the
    // pivoted matrix; dividing every peptidoform by the same per-sample factor
    // yields the same canonical-collapsed result Python's melt-back produces.
    let factors_by_name = tmm_norm_factors(&column_names, &matrix);
    let factors_by_sample: HashMap<SampleId, f64> = sample_ids
        .iter()
        .zip(column_names.iter())
        .filter_map(|(id, name)| factors_by_name.get(name).map(|factor| (*id, *factor)))
        .collect();

    for (cell, peptides) in cells.iter_mut() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let factor = factors_by_sample.get(&cell.sample).copied().unwrap_or(1.0);
        if !tmm_factor_applies(factor) {
            continue;
        }
        for value in peptides.values_mut() {
            *value /= factor;
        }
    }
}

fn tmm_row_keys(
    summed_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
) -> Vec<QuantilePeptideKey> {
    // (2) Build the deterministic row order: every (protein, canonical) key that
    // appears in any cell, sorted by id so all columns share the same row order.
    let mut row_keys: Vec<QuantilePeptideKey> = summed_cells
        .iter()
        .flat_map(|(cell, peptides)| {
            peptides.keys().map(move |peptide| QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            })
        })
        .collect();
    row_keys.sort();
    row_keys.dedup();
    row_keys
}

fn tmm_sample_columns(
    summed_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
    samples: &StringIdRegistry<SampleId>,
) -> (Vec<SampleId>, Vec<String>) {
    // (3) Column order: samples sorted by their string label, matching pandas'
    // pivoted column ordering (drives the reference tie-break).
    let mut sample_ids: Vec<SampleId> = summed_cells
        .keys()
        .map(|cell| cell.sample)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    sample_ids.sort_by(|a, b| {
        let name_a = samples.resolve(*a).unwrap_or("");
        let name_b = samples.resolve(*b).unwrap_or("");
        name_a.cmp(name_b).then_with(|| {
            let raw_a: u32 = (*a).into();
            let raw_b: u32 = (*b).into();
            raw_a.cmp(&raw_b)
        })
    });
    let column_names: Vec<String> = sample_ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            samples
                .resolve(*id)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("__sample_{index}"))
        })
        .collect();
    (sample_ids, column_names)
}

fn tmm_matrix(
    summed_cells: &HashMap<CellKey, HashMap<PeptideId, f64>>,
    row_keys: &[QuantilePeptideKey],
    sample_ids: &[SampleId],
) -> Vec<Vec<f64>> {
    let row_index: HashMap<QuantilePeptideKey, usize> = row_keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect();

    let sample_column: HashMap<SampleId, usize> = sample_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();

    // (4) Populate the column-major wide matrix. Missing (protein, canonical) x
    // sample entries stay 0.0, which `tmm_norm_factors` treats as missing by
    // filtering them out — the same outcome as Python, whose pivot fills gaps
    // with NaN and `replace(0, NaN)` maps zeros to NaN before `skipna` drops them.
    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; row_keys.len()]; sample_ids.len()];
    for (cell, peptides) in summed_cells {
        let Some(&column) = sample_column.get(&cell.sample) else {
            continue;
        };
        for (peptide, value) in peptides {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide: *peptide,
            };
            if let Some(&row) = row_index.get(&key) {
                matrix[column][row] = *value;
            }
        }
    }
    matrix
}

/// A per-sample TMM factor is applied only when it is a genuine rescale: not the
/// identity (1.0) and a positive, finite number. `finalize` divides every
/// peptidoform by the factor, so the identity and any non-finite factor are
/// skipped to leave the sample untouched (tmm.py:334-343).
fn tmm_factor_applies(factor: f64) -> bool {
    factor != 1.0 && factor > 0.0 && factor.is_finite()
}

/// Median over the finite values of `values` (non-finite dropped, sign
/// preserved). Unlike `mokume_core::stats::median` this keeps negative values,
/// which log2 references and residuals can take. Sorts in place with
/// `f64::total_cmp` for determinism.
pub(crate) fn finite_median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    } else {
        Some(values[midpoint])
    }
}
