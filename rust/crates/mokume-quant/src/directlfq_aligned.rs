//! DirectLFQ protein-intensity estimation aligned with the Mann-Labs
//! `directlfq` package (v0.3.3) as invoked by Python `mokume`'s
//! `features2proteins --quant-method directlfq` path.
//!
//! The algorithm has two stages, both built on the same agglomerative
//! shift-clustering primitive ([`get_normfacts`]):
//!
//! 1. Global sample normalization across the whole ion x sample matrix
//!    (`NormalizationManagerSamplesOnSelectedProteins`, `num_samples_quadratic = 50`).
//! 2. Per-protein intensity estimation: align the protein's ion profiles
//!    (`NormalizationManagerProtein`, `num_samples_quadratic = 10`), take the
//!    per-sample median of the aligned ions, then rescale so the protein's
//!    total equals the summed linear ion intensity.
//!
//! Everything is deterministic and operates in log2 space. This module mirrors
//! the reference source operation-by-operation (matrix orientation, tie-breaks,
//! NaN handling, float reduction order) so the protein matrix matches Python
//! cell-for-cell within floating-point round-off.

use std::collections::HashMap;

use mokume_core::{PeptideId, ProteinId, SampleId};
use rayon::prelude::*;

/// `num_samples_quadratic` for the per-protein ion-normalization stage
/// (hard-coded to 10 by mokume's streaming caller).
const PROTEIN_QUADRATIC_LIMIT: usize = 10;
/// Maximum ions kept per protein before estimation (`ProtvalCutter`).
const MAX_IONS_PER_PROTEIN: usize = 100;

/// A single ion observation: which protein and ion (bare sequence) it belongs
/// to, in which sample, and its linear intensity. Duplicates of the same
/// `(protein, ion, sample)` are summed, mirroring mokume's pre-aggregation.
///
/// `ion_seq_rank` is the global alphabetical rank of the ion's bare sequence
/// string. DirectLFQ orders its ion rows by `(protein, sequence)` lexically
/// (`stages.py:887` `.sort(["protein", "sequence"])`, mirroring directlfq's
/// `sort_input_df_by_protein_and_quant_id`), and `get_normfacts`' agglomerative
/// merge breaks variance ties by row position. Ordering rows by the raw
/// `PeptideId` (registration order) instead would feed `get_normfacts` a
/// different row order, rerouting the merge tree and perturbing the protein
/// profile. We therefore sort rows by this sequence rank to reproduce Python's
/// row order exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectLfqIon {
    pub protein: ProteinId,
    pub ion: PeptideId,
    pub ion_seq_rank: u32,
    pub sample: SampleId,
    pub intensity: f64,
}

/// One DirectLFQ-normalized ion row in linear intensity space.
///
/// Values include global sample normalization and within-protein ion alignment,
/// but not the final protein-profile rescaling. This matches DirectLFQ's
/// `shifted_peptides` table and the Python `--export-ions` output. Missing values
/// are represented as `0.0`, as in Python's `np.nan_to_num` export.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectLfqNormalizedIon {
    pub protein: ProteinId,
    pub ion: PeptideId,
    pub ion_seq_rank: u32,
    pub quantities: Vec<(SampleId, f64)>,
}

/// Protein quantities and, when requested, the aligned ion rows produced by the
/// same DirectLFQ calculation.
#[derive(Debug, Default, PartialEq)]
pub struct DirectLfqResult {
    pub protein_quantities: Vec<(ProteinId, Vec<(SampleId, f64)>)>,
    pub normalized_ions: Vec<DirectLfqNormalizedIon>,
}

/// Estimate DirectLFQ protein intensities for every protein.
///
/// Returns, per protein that has at least one quantified sample, the linear
/// protein intensity for each sample in ascending [`SampleId`] order (missing
/// samples reported as `0.0`, matching the Python output which writes zeros).
///
/// `sample_quadratic_limit` is directlfq's `num_samples_quadratic` for the
/// global sample stage (mokume's default is 50); the per-protein ion stage is
/// fixed at 10 to match mokume's streaming caller. `normalize_samples = false`
/// skips the global sample stage, like directlfq's `deactivate_normalization`.
pub fn direct_lfq_aligned(
    ions: &[DirectLfqIon],
    min_nonan: usize,
    sample_quadratic_limit: usize,
    normalize_samples: bool,
) -> Vec<(ProteinId, Vec<(SampleId, f64)>)> {
    direct_lfq_aligned_impl(
        ions,
        min_nonan,
        sample_quadratic_limit,
        normalize_samples,
        false,
    )
    .protein_quantities
}

/// Estimate proteins and retain DirectLFQ's normalized, within-protein-aligned
/// ion rows in linear intensity space.
///
/// Ion rows survive even when `min_nonan` prevents a protein profile from being
/// reported, matching Python's streamed `shifted_peptides` export.
pub fn direct_lfq_aligned_with_ions(
    ions: &[DirectLfqIon],
    min_nonan: usize,
    sample_quadratic_limit: usize,
    normalize_samples: bool,
) -> DirectLfqResult {
    direct_lfq_aligned_impl(
        ions,
        min_nonan,
        sample_quadratic_limit,
        normalize_samples,
        true,
    )
}

fn direct_lfq_aligned_impl(
    ions: &[DirectLfqIon],
    min_nonan: usize,
    sample_quadratic_limit: usize,
    normalize_samples: bool,
    retain_ions: bool,
) -> DirectLfqResult {
    let matrix = IonMatrix::build(ions);
    if matrix.samples.is_empty() || matrix.rows.is_empty() {
        return DirectLfqResult::default();
    }
    let mut matrix = matrix;
    if normalize_samples {
        matrix.normalize_samples(sample_quadratic_limit);
    }
    matrix.estimate(min_nonan, retain_ions)
}

/// The log2 ion x sample matrix: `rows[r][c]` is the log2 intensity of ion `r`
/// in sample `c`, or `NaN` if missing. Rows are sorted by `(protein, ion)` and
/// columns by `SampleId`, matching the layout Python hands to directlfq.
struct IonMatrix {
    proteins: Vec<ProteinId>,
    ions: Vec<PeptideId>,
    ion_seq_ranks: Vec<u32>,
    samples: Vec<SampleId>,
    rows: Vec<Vec<f64>>,
}

struct IonMatrixRows {
    proteins: Vec<ProteinId>,
    ions: Vec<PeptideId>,
    ion_seq_ranks: Vec<u32>,
    values: Vec<Vec<f64>>,
}

impl IonMatrix {
    fn build(ions: &[DirectLfqIon]) -> Self {
        let mut samples = ions.iter().map(|ion| ion.sample).collect::<Vec<_>>();
        samples.sort_by_key(|sample| sample.get());
        samples.dedup();
        let sample_index = samples
            .iter()
            .enumerate()
            .map(|(index, sample)| (*sample, index))
            .collect::<HashMap<_, _>>();

        let mut linear: HashMap<(ProteinId, PeptideId), Vec<f64>> = HashMap::new();
        // Per `(protein, ion)`, the global alphabetical rank of the ion's bare
        // sequence; used as the primary within-protein row-sort key so the row
        // order handed to `get_normfacts` matches Python's `sort([protein,
        // sequence])`.
        let mut seq_rank: HashMap<(ProteinId, PeptideId), u32> = HashMap::new();
        for ion in ions {
            if !ion.intensity.is_finite() || ion.intensity <= 0.0 {
                continue;
            }
            let Some(&column) = sample_index.get(&ion.sample) else {
                continue;
            };
            let cells = linear
                .entry((ion.protein, ion.ion))
                .or_insert_with(|| vec![0.0; samples.len()]);
            cells[column] += ion.intensity;
            seq_rank.insert((ion.protein, ion.ion), ion.ion_seq_rank);
        }

        let mut keyed = linear.into_iter().collect::<Vec<_>>();
        keyed.sort_by(|((protein_a, ion_a), _), ((protein_b, ion_b), _)| {
            // `(protein, sequence-rank, ion-id)`: the sequence rank reproduces
            // DirectLFQ's lexical `sort([protein, sequence])` row order, and the
            // trailing `ion-id` only breaks the (impossible after dedup) rank
            // ties deterministically.
            let rank_a = seq_rank
                .get(&(*protein_a, *ion_a))
                .copied()
                .unwrap_or(u32::MAX);
            let rank_b = seq_rank
                .get(&(*protein_b, *ion_b))
                .copied()
                .unwrap_or(u32::MAX);
            protein_a
                .get()
                .cmp(&protein_b.get())
                .then(rank_a.cmp(&rank_b))
                .then(ion_a.get().cmp(&ion_b.get()))
        });

        let rows = collect_matrix_rows(keyed, &seq_rank);

        Self {
            proteins: rows.proteins,
            ions: rows.ions,
            ion_seq_ranks: rows.ion_seq_ranks,
            samples,
            rows: rows.values,
        }
    }

    /// Stage 1: shift each sample column by an additive log2 factor so the
    /// sample distributions overlay (`NormalizationManagerSamples...`).
    fn normalize_samples(&mut self, sample_quadratic_limit: usize) {
        let shifts = sample_shifts(&self.rows, self.samples.len(), sample_quadratic_limit);
        for row in &mut self.rows {
            for (column, value) in row.iter_mut().enumerate() {
                *value += shifts[column];
            }
        }
    }

    /// Stage 2: per-protein intensity estimation over contiguous protein blocks.
    /// Each protein is independent after the global sample normalization, so the
    /// blocks are estimated in parallel.
    fn estimate(&self, min_nonan: usize, retain_ions: bool) -> DirectLfqResult {
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < self.proteins.len() {
            let mut end = start + 1;
            while end < self.proteins.len() && self.proteins[end] == self.proteins[start] {
                end += 1;
            }
            ranges.push((self.proteins[start], start, end));
            start = end;
        }

        let blocks = ranges
            .into_par_iter()
            .map(|(protein, start, end)| {
                self.estimate_block(protein, start, end, min_nonan, retain_ions)
            })
            .collect::<Vec<_>>();

        let mut result = DirectLfqResult::default();
        for block in blocks {
            if let Some(quantities) = block.protein_quantities {
                result.protein_quantities.push(quantities);
            }
            result.normalized_ions.extend(block.normalized_ions);
        }
        result
    }

    fn estimate_block(
        &self,
        protein: ProteinId,
        start: usize,
        end: usize,
        min_nonan: usize,
        retain_ions: bool,
    ) -> ProteinBlockResult {
        let alignment = align_protein(&self.rows[start..end], self.samples.len(), min_nonan);
        let protein_quantities = alignment.quantities.as_ref().map(|profile| {
            let quantities = self
                .samples
                .iter()
                .copied()
                .zip(profile.iter().copied())
                .collect();
            (protein, quantities)
        });
        let normalized_ions = if retain_ions {
            self.normalized_ion_rows(protein, start, &alignment)
        } else {
            Vec::new()
        };
        ProteinBlockResult {
            protein_quantities,
            normalized_ions,
        }
    }

    fn normalized_ion_rows(
        &self,
        protein: ProteinId,
        start: usize,
        alignment: &ProteinAlignment,
    ) -> Vec<DirectLfqNormalizedIon> {
        alignment
            .selected_rows
            .iter()
            .copied()
            .zip(&alignment.shifted)
            .map(|(block_row, shifted)| {
                let matrix_row = start + block_row;
                DirectLfqNormalizedIon {
                    protein,
                    ion: self.ions[matrix_row],
                    ion_seq_rank: self.ion_seq_ranks[matrix_row],
                    quantities: self
                        .samples
                        .iter()
                        .copied()
                        .zip(shifted.iter().map(
                            |value| {
                                if value.is_finite() {
                                    value.exp2()
                                } else {
                                    0.0
                                }
                            },
                        ))
                        .collect(),
                }
            })
            .collect()
    }
}

fn collect_matrix_rows(
    keyed: Vec<((ProteinId, PeptideId), Vec<f64>)>,
    seq_rank: &HashMap<(ProteinId, PeptideId), u32>,
) -> IonMatrixRows {
    let mut rows = IonMatrixRows {
        proteins: Vec::with_capacity(keyed.len()),
        ions: Vec::with_capacity(keyed.len()),
        ion_seq_ranks: Vec::with_capacity(keyed.len()),
        values: Vec::with_capacity(keyed.len()),
    };
    for ((protein, ion), cells) in keyed {
        rows.proteins.push(protein);
        rows.ions.push(ion);
        rows.ion_seq_ranks
            .push(seq_rank.get(&(protein, ion)).copied().unwrap_or(u32::MAX));
        rows.values
            .push(cells.into_iter().map(linear_to_log2).collect());
    }
    rows
}

struct ProteinBlockResult {
    protein_quantities: Option<(ProteinId, Vec<(SampleId, f64)>)>,
    normalized_ions: Vec<DirectLfqNormalizedIon>,
}

struct ProteinAlignment {
    selected_rows: Vec<usize>,
    shifted: Vec<Vec<f64>>,
    quantities: Option<Vec<f64>>,
}

fn linear_to_log2(value: f64) -> f64 {
    if value > 0.0 {
        value.log2()
    } else {
        f64::NAN
    }
}

/// Compute per-sample additive log2 shifts for the global normalization stage.
/// `rows` is the ion x sample matrix; the returned vector has one shift per
/// sample column. Mirrors `NormalizationManagerSamplesOnSelectedProteins`:
/// quadratic clustering up to `quadratic_limit`, otherwise that many samples
/// plus a linear shift of the remaining samples onto the subset median.
///
/// Each row must have `n_samples` log2 values, with missing values encoded as
/// NaN. `quadratic_limit` must be positive when `n_samples` is positive.
pub fn sample_shifts(rows: &[Vec<f64>], n_samples: usize, quadratic_limit: usize) -> Vec<f64> {
    let mut sample_major = transpose(rows, n_samples);
    if n_samples <= quadratic_limit {
        let mut work = drop_na_columns(&sample_major);
        return get_normfacts(&mut work);
    }

    let order = rows_sorted_by_nan_count(&sample_major);
    let quadratic = &order[..quadratic_limit];
    let linear = &order[quadratic_limit..];

    let subset = quadratic
        .iter()
        .map(|&index| sample_major[index].clone())
        .collect::<Vec<_>>();
    let mut work = drop_na_columns(&subset);
    let subset_shifts = get_normfacts(&mut work);

    let mut shifts = vec![0.0; n_samples];
    for (subset_position, &sample_index) in quadratic.iter().enumerate() {
        let shift = subset_shifts[subset_position];
        shifts[sample_index] = shift;
        for value in &mut sample_major[sample_index] {
            *value += shift;
        }
    }

    let reference = per_column_median(&sample_major, quadratic);
    for &sample_index in linear {
        let shift = median_distance(&reference, &sample_major[sample_index]);
        if shift.is_finite() {
            shifts[sample_index] = shift;
        }
    }
    shifts
}

/// Align one protein's ions and derive its optional linear protein profile.
fn align_protein(block: &[Vec<f64>], n_samples: usize, min_nonan: usize) -> ProteinAlignment {
    let (selected_rows, trimmed) = trim_ions(block);
    let block = trimmed.as_slice();

    let summed_pepint: f64 = block
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| value.is_finite())
        .map(|value| value.exp2())
        .sum();

    let shifted = if n_samples < 2 {
        block.to_vec()
    } else {
        normalize_ion_profiles(block)
    };

    let quantities = protein_quantities(&shifted, summed_pepint, min_nonan, n_samples);
    ProteinAlignment {
        selected_rows,
        shifted,
        quantities,
    }
}

fn protein_quantities(
    shifted: &[Vec<f64>],
    summed_pepint: f64,
    min_nonan: usize,
    n_samples: usize,
) -> Option<Vec<f64>> {
    let mut intens_vec = vec![f64::NAN; n_samples];
    for (column, slot) in intens_vec.iter_mut().enumerate() {
        let mut finite = shifted
            .iter()
            .map(|row| row[column])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite.len() >= min_nonan {
            if let Some(value) = median_in_place(&mut finite) {
                *slot = value;
            }
        }
    }

    let summed_intensity: f64 = intens_vec
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.exp2())
        .sum();
    (summed_intensity != 0.0).then(|| {
        let conversion = (summed_pepint / summed_intensity).log2();
        intens_vec
            .into_iter()
            .map(|value| {
                let scaled = value + conversion;
                if scaled.is_finite() {
                    scaled.exp2()
                } else {
                    0.0
                }
            })
            .collect()
    })
}

/// `ProtvalCutter`: keep at most 100 ions, ranked by NaN count ascending then
/// summed log2 intensity descending (stable). A no-op for <= 100 ions.
///
/// The secondary key matches directlfq's `_determine_nansorted_df_index`
/// (`protein_intensity_estimation.py:191-194`), which sorts by `-np.nansum(row)`
/// over the *log2* dataframe — i.e. the sum of the raw log2 values, NOT the sum
/// of linear intensities. Python's `sorted()` is stable, so ties on both keys
/// keep their original order; we reproduce that with the trailing index compare.
fn trim_ions(block: &[Vec<f64>]) -> (Vec<usize>, Vec<Vec<f64>>) {
    if block.len() <= MAX_IONS_PER_PROTEIN {
        return ((0..block.len()).collect(), block.to_vec());
    }
    let mut indexed = block.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(index_a, row_a), (index_b, row_b)| {
        let nan_a = row_a.iter().filter(|value| value.is_nan()).count();
        let nan_b = row_b.iter().filter(|value| value.is_nan()).count();
        let sum_a = nansum_log2_skip(row_a);
        let sum_b = nansum_log2_skip(row_b);
        nan_a
            .cmp(&nan_b)
            .then(sum_b.total_cmp(&sum_a))
            .then(index_a.cmp(index_b))
    });
    let selected = indexed
        .into_iter()
        .take(MAX_IONS_PER_PROTEIN)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let rows = selected.iter().map(|&index| block[index].clone()).collect();
    (selected, rows)
}

/// `np.nansum(row)` over the log2 row: sum of the finite (non-NaN) log2 values,
/// with no exp2 conversion to linear intensity.
fn nansum_log2_skip(row: &[f64]) -> f64 {
    row.iter().filter(|value| value.is_finite()).copied().sum()
}

/// Per-protein ion alignment (`NormalizationManagerProtein`). Unlike the sample
/// stage, the single-intensity row mutation propagates into the shifted output,
/// so ions present in fewer than two samples become all-NaN.
fn normalize_ion_profiles(block: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_ions = block.len();
    if n_ions <= PROTEIN_QUADRATIC_LIMIT {
        let mut work = block.to_vec();
        let shifts = get_normfacts(&mut work);
        for (row, shift) in work.iter_mut().zip(&shifts) {
            for value in row.iter_mut() {
                *value += *shift;
            }
        }
        return work;
    }

    let order = rows_sorted_by_nan_count(block);
    let quadratic = &order[..PROTEIN_QUADRATIC_LIMIT];
    let linear = &order[PROTEIN_QUADRATIC_LIMIT..];

    let mut result = block.to_vec();
    let mut subset = quadratic
        .iter()
        .map(|&index| block[index].clone())
        .collect::<Vec<_>>();
    let subset_shifts = get_normfacts(&mut subset);
    for (subset_position, &ion_index) in quadratic.iter().enumerate() {
        result[ion_index] = subset[subset_position].clone();
        let shift = subset_shifts[subset_position];
        for value in &mut result[ion_index] {
            *value += shift;
        }
    }

    let reference = per_column_median(&result, quadratic);
    for &ion_index in linear {
        let shift = median_distance(&reference, &result[ion_index]);
        if shift.is_finite() {
            for value in &mut result[ion_index] {
                *value += shift;
            }
        } else {
            // `SampleShifterLinear._shift_to_reference_sample` adds the
            // distance-to-reference to the *whole* row unconditionally
            // (`normalization.py:412`). When the reference shares no finite
            // sample with this ion the distance is `NaN`, so `row += NaN` masks
            // every cell — including the lone finite value that triggered the
            // empty overlap. Skipping the shift (the previous behaviour) instead
            // left that value finite, which let a single-intensity linear ion
            // resurrect a sample Python drops, flipping that protein x sample
            // cell from 0 to a value and rescaling the whole protein profile.
            for value in &mut result[ion_index] {
                *value = f64::NAN;
            }
        }
    }
    result
}

/// The agglomerative shift-clustering primitive (`directlfq.normalization.get_normfacts`).
///
/// `rows` is mutated in place: rows with fewer than two finite values are set to
/// all-NaN (matching `set_samples_with_only_single_intensity_to_nan`). Returns
/// the total additive shift for every row (`0.0` for rows that were never
/// shifted, including the cluster root).
fn get_normfacts(rows: &mut [Vec<f64>]) -> Vec<f64> {
    let n = rows.len();
    set_single_intensity_rows_to_nan(rows);

    let mut merged = rows.to_vec();
    let mut shift = vec![0.0; n];
    let mut counts = vec![1.0_f64; n];
    let mut anchor_of: HashMap<usize, usize> = HashMap::new();
    let mut excluded = Vec::new();

    let mut distance = distance_matrix(rows, Metric::Median);
    let mut variance = distance_matrix(rows, Metric::Variance);
    exclude_unconnected(&mut distance);
    exclude_unconnected(&mut variance);

    for _ in 0..n.saturating_sub(1) {
        let Some((i, j)) = argmin_pair(&variance) else {
            break;
        };
        let raw_distance = distance[i][j];
        if !raw_distance.is_finite() {
            break;
        }
        let (anchor, shift_idx, signed) = if counts[i] >= counts[j] {
            (i, j, raw_distance)
        } else {
            (j, i, -raw_distance)
        };

        anchor_of.insert(shift_idx, anchor);
        shift[shift_idx] = signed;
        excluded.push(shift_idx);

        // Python shifts the *original* (single-intensity-masked) sample row, not
        // the accumulated merged row, before merging it into the anchor.
        let shifted = rows[shift_idx]
            .iter()
            .map(|value| value + signed)
            .collect::<Vec<_>>();
        merged[anchor] =
            merge_distribs(&merged[anchor], &shifted, counts[anchor], counts[shift_idx]);

        update_distance_matrix(&mut variance, &merged, anchor, shift_idx, Metric::Variance);
        update_distance_matrix(&mut distance, &merged, anchor, shift_idx, Metric::Median);
        counts[anchor] += 1.0;
    }

    let mut total = vec![0.0; n];
    for &idx in &excluded {
        total[idx] = total_shift(&anchor_of, &shift, idx);
    }
    total
}

fn set_single_intensity_rows_to_nan(rows: &mut [Vec<f64>]) {
    for row in rows.iter_mut() {
        if row.iter().filter(|value| value.is_finite()).count() < 2 {
            for value in row.iter_mut() {
                *value = f64::NAN;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Metric {
    Median,
    Variance,
}

fn distance_matrix(rows: &[Vec<f64>], metric: Metric) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut matrix = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            matrix[i][j] = calc_distance(metric, &rows[i], &rows[j]);
        }
    }
    matrix
}

fn calc_distance(metric: Metric, a: &[f64], b: &[f64]) -> f64 {
    let diffs = a
        .iter()
        .zip(b)
        .map(|(left, right)| left - right)
        .collect::<Vec<_>>();
    let value = match metric {
        Metric::Median => nanmedian(&diffs),
        Metric::Variance => nanvar(&diffs),
    };
    if value.is_nan() {
        f64::INFINITY
    } else {
        value
    }
}

fn update_distance_matrix(
    matrix: &mut [Vec<f64>],
    merged: &[Vec<f64>],
    anchor: usize,
    shift_idx: usize,
    metric: Metric,
) {
    for i in 0..anchor {
        if matrix[i][anchor].is_finite() {
            matrix[i][anchor] = calc_distance(metric, &merged[i], &merged[anchor]);
        }
    }
    for j in (anchor + 1)..merged.len() {
        if matrix[anchor][j].is_finite() {
            matrix[anchor][j] = calc_distance(metric, &merged[anchor], &merged[j]);
        }
    }
    for value in matrix[shift_idx].iter_mut() {
        *value = f64::INFINITY;
    }
    for row in matrix.iter_mut() {
        row[shift_idx] = f64::INFINITY;
    }
}

fn merge_distribs(
    anchor: &[f64],
    shifted: &[f64],
    count_anchor: f64,
    count_shifted: f64,
) -> Vec<f64> {
    anchor
        .iter()
        .zip(shifted)
        .map(|(&a, &s)| match (a.is_nan(), s.is_nan()) {
            (true, true) => f64::NAN,
            (true, false) => s,
            (false, true) => a,
            (false, false) => {
                (a * count_anchor + s * count_shifted) / (count_anchor + count_shifted)
            }
        })
        .collect()
}

fn total_shift(anchor_of: &HashMap<usize, usize>, shift: &[f64], start: usize) -> f64 {
    let mut total = 0.0;
    let mut idx = start;
    loop {
        total += shift[idx];
        match anchor_of.get(&idx) {
            Some(&next) => idx = next,
            None => break,
        }
    }
    total
}

/// `np.argmin` over the variance matrix: the first row-major index of the
/// minimum finite entry. Returns `None` when every entry is infinite.
fn argmin_pair(matrix: &[Vec<f64>]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut best_value = f64::INFINITY;
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value < best_value {
                best_value = value;
                best = Some((i, j));
            }
        }
    }
    best
}

/// `tracefilter.exclude_unconnected_samples`: keep only the connected component
/// (over finite edges) reachable from the most-connected node; isolate the rest
/// by setting their row and column to infinity.
fn exclude_unconnected(matrix: &mut [Vec<f64>]) {
    let n = matrix.len();
    if n < 2 {
        return;
    }
    let mut full = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j].is_finite() {
                full[i][j] = matrix[i][j];
                full[j][i] = matrix[i][j];
            }
        }
    }
    let mut best_start = 0;
    let mut best_degree = usize::MIN;
    for (index, row) in full.iter().enumerate() {
        let degree = row.iter().filter(|value| value.is_finite()).count();
        if degree > best_degree {
            best_degree = degree;
            best_start = index;
        }
    }

    let mut connected = vec![false; n];
    let mut stack = vec![best_start];
    while let Some(node) = stack.pop() {
        for (neighbor, &value) in full[node].iter().enumerate() {
            if value.is_finite() && !connected[neighbor] {
                connected[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }

    for (index, is_connected) in connected.iter().enumerate() {
        if !is_connected {
            for value in matrix[index].iter_mut() {
                *value = f64::INFINITY;
            }
            for row in matrix.iter_mut() {
                row[index] = f64::INFINITY;
            }
        }
    }
}

/// `drop_nas_if_possible`: for the sample-normalization stage only, restrict
/// the shift computation to ion columns that are finite in every sample, but
/// only when there are at least 1000 such complete ions and they make up at
/// least 0.1% of all ions. Otherwise the full matrix (with NaNs) is used.
/// `rows` is sample-major (rows = samples, columns = ions).
fn drop_na_columns(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_columns = rows.first().map_or(0, Vec::len);
    if n_columns == 0 {
        return rows.to_vec();
    }
    let complete = (0..n_columns)
        .filter(|&column| rows.iter().all(|row| row[column].is_finite()))
        .collect::<Vec<_>>();
    let keep_subset = complete.len() >= 1000 && complete.len() as f64 >= 0.001 * n_columns as f64;
    if !keep_subset {
        return rows.to_vec();
    }
    rows.iter()
        .map(|row| complete.iter().map(|&column| row[column]).collect())
        .collect()
}

fn transpose(rows: &[Vec<f64>], n_columns: usize) -> Vec<Vec<f64>> {
    let mut columns = vec![Vec::with_capacity(rows.len()); n_columns];
    for row in rows {
        for (column, value) in row.iter().enumerate() {
            columns[column].push(*value);
        }
    }
    columns
}

/// Stable sort of row indices by NaN count ascending, matching Python's
/// `sorted(key=num_nas)` (ties keep original order).
fn rows_sorted_by_nan_count(rows: &[Vec<f64>]) -> Vec<usize> {
    let mut order = (0..rows.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| rows[index].iter().filter(|value| value.is_nan()).count());
    order
}

/// Per-column median over a subset of rows (`DataFrame.median(axis=0)`),
/// skipping NaN. Used to build the linear-stage reference distribution.
fn per_column_median(rows: &[Vec<f64>], subset: &[usize]) -> Vec<f64> {
    let n_columns = rows.first().map_or(0, Vec::len);
    (0..n_columns)
        .map(|column| {
            let values = subset
                .iter()
                .map(|&index| rows[index][column])
                .collect::<Vec<_>>();
            nanmedian(&values)
        })
        .collect()
}

/// `SampleShifterLinear._calc_distance`: `nanmedian(reference - sample)`, or
/// `NaN` if no overlapping finite value exists.
fn median_distance(reference: &[f64], sample: &[f64]) -> f64 {
    let diffs = reference
        .iter()
        .zip(sample)
        .map(|(&r, &s)| r - s)
        .collect::<Vec<_>>();
    nanmedian(&diffs)
}

fn nanmedian(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    median_in_place(&mut finite).unwrap_or(f64::NAN)
}

fn median_in_place(finite: &mut [f64]) -> Option<f64> {
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f64::total_cmp);
    let midpoint = finite.len() / 2;
    if finite.len().is_multiple_of(2) {
        Some((finite[midpoint - 1] + finite[midpoint]) / 2.0)
    } else {
        Some(finite[midpoint])
    }
}

/// Population variance over finite values (`np.nanvar`, ddof = 0); `NaN` if
/// there are no finite values.
fn nanvar(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    finite
        .iter()
        .map(|value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>()
        / finite.len() as f64
}

#[cfg(test)]
mod tests {
    use super::{
        direct_lfq_aligned, direct_lfq_aligned_with_ions, get_normfacts, trim_ions, DirectLfqIon,
    };
    use mokume_core::{PeptideId, ProteinId, SampleId};

    // Cross-check the agglomerative shift clustering against directlfq's
    // `get_normfacts` on a 5-row matrix with a clear merge tree and one missing
    // value. Reference shifts from `directlfq.normalization.get_normfacts`.
    #[test]
    fn get_normfacts_matches_directlfq() {
        let nan = f64::NAN;
        let mut rows = vec![
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            vec![10.5, 11.4, 12.6, 13.5, 14.4, 15.6],
            vec![12.0, 13.0, 14.0, nan, 16.0, 17.0],
            vec![9.0, 10.1, 11.0, 12.1, 13.0, 14.1],
            vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let shifts = get_normfacts(&mut rows);
        let expected = [0.0, -0.5, -2.0, 0.95, -10.0];
        for (index, (actual, want)) in shifts.iter().zip(expected).enumerate() {
            assert!(
                (actual - want).abs() <= 1e-9,
                "row {index}: actual={actual} expected={want}"
            );
        }
    }

    fn ion(protein: u32, ion: u32, sample: u32, intensity: f64) -> DirectLfqIon {
        DirectLfqIon {
            protein: ProteinId::new(protein),
            ion: PeptideId::new(ion),
            // These oracle tests assign ion ids in the same order DirectLFQ
            // would sort the sequences, so the rank equals the id.
            ion_seq_rank: ion,
            sample: SampleId::new(sample),
            intensity,
        }
    }

    fn value_for(quantities: &[(SampleId, f64)], sample: u32) -> f64 {
        quantities
            .iter()
            .find_map(|(candidate, value)| (*candidate == SampleId::new(sample)).then_some(*value))
            .unwrap_or(f64::NAN)
    }

    fn assert_rel(actual: f64, expected: f64) {
        let tol = 1e-9 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected}"
        );
    }

    // Oracle 3 from the Mann-Labs directlfq reference run (see the porting spec):
    // ProtA has ions A1/A2/A3, ProtB has B1/B2, across 4 samples; ProtA A2 is
    // missing in sample S3. Expected linear protein intensities are taken from
    // `conda run -n Bigbio python /tmp/dlfq_oracle3.py`.
    #[test]
    fn matches_directlfq_oracle3() {
        let ions = vec![
            ion(0, 0, 0, 1000.0),
            ion(0, 0, 1, 2200.0),
            ion(0, 0, 2, 4000.0),
            ion(0, 0, 3, 9000.0),
            ion(0, 1, 0, 3000.0),
            ion(0, 1, 1, 4000.0),
            ion(0, 1, 3, 16000.0),
            ion(0, 2, 0, 900.0),
            ion(0, 2, 1, 1000.0),
            ion(0, 2, 2, 2500.0),
            ion(0, 2, 3, 3000.0),
            ion(1, 3, 0, 8000.0),
            ion(1, 3, 1, 15000.0),
            ion(1, 3, 2, 30000.0),
            ion(1, 3, 3, 70000.0),
            ion(1, 4, 0, 5000.0),
            ion(1, 4, 1, 8000.0),
            ion(1, 4, 2, 17000.0),
            ion(1, 4, 3, 30000.0),
        ];

        let result = direct_lfq_aligned(&ions, 1, 50, true);
        let Some((_, prot_a)) = result
            .iter()
            .find(|(protein, _)| *protein == ProteinId::new(0))
        else {
            panic!("ProtA missing from result");
        };
        let Some((_, prot_b)) = result
            .iter()
            .find(|(protein, _)| *protein == ProteinId::new(1))
        else {
            panic!("ProtB missing from result");
        };

        assert_rel(value_for(prot_a, 0), 6_916.228_325_456_703);
        assert_rel(value_for(prot_a, 1), 5_829.654_646_593_098);
        assert_rel(value_for(prot_a, 2), 6_528.489_300_300_889);
        assert_rel(value_for(prot_a, 3), 5_829.654_646_593_098);

        assert_rel(value_for(prot_b, 0), 21_075.355_412_017_83);
        assert_rel(value_for(prot_b, 1), 23076.519292805977);
        assert_rel(value_for(prot_b, 2), 23076.519292805977);
        assert_rel(value_for(prot_b, 3), 24_134.001_554_791_45);
    }

    #[test]
    fn exports_aligned_ions_when_min_nonan_drops_the_protein() {
        let ions = vec![
            ion(0, 0, 0, 400.0),
            ion(0, 0, 1, 800.0),
            ion(0, 1, 0, 150.0),
            ion(0, 1, 1, 200.0),
        ];

        let result = direct_lfq_aligned_with_ions(&ions, 3, 50, true);
        assert!(result.protein_quantities.is_empty());
        assert_eq!(result.normalized_ions.len(), 2);

        let first = &result.normalized_ions[0];
        assert_eq!(first.protein, ProteinId::new(0));
        assert_eq!(first.ion, PeptideId::new(0));
        assert_rel(value_for(&first.quantities, 0), 400.0);
        assert_rel(value_for(&first.quantities, 1), 489.897_948_556_635_7);

        let second = &result.normalized_ions[1];
        assert_eq!(second.ion, PeptideId::new(1));
        assert_rel(value_for(&second.quantities, 0), 489.897_948_556_635_7);
        assert_rel(value_for(&second.quantities, 1), 400.0);
    }

    #[test]
    fn skipping_sample_normalization_keeps_loading_differences() {
        // Sample 1 carries twice the material of sample 0 for every ion.
        let ions = vec![
            ion(0, 0, 0, 1000.0),
            ion(0, 0, 1, 2000.0),
            ion(0, 1, 0, 3000.0),
            ion(0, 1, 1, 6000.0),
            ion(1, 2, 0, 500.0),
            ion(1, 2, 1, 1000.0),
            ion(1, 3, 0, 700.0),
            ion(1, 3, 1, 1400.0),
        ];
        let ratio = |normalize_samples| {
            let result = direct_lfq_aligned(&ions, 1, 50, normalize_samples);
            value_for(&result[0].1, 1) / value_for(&result[0].1, 0)
        };

        assert_rel(ratio(true), 1.0);
        assert_rel(ratio(false), 2.0);
    }

    #[test]
    fn single_observation_ion_is_masked_like_directlfq() {
        let result = direct_lfq_aligned_with_ions(
            &[
                ion(0, 0, 0, 100.0),
                ion(1, 1, 0, 200.0),
                ion(1, 1, 1, 300.0),
                ion(1, 2, 0, 400.0),
                ion(1, 2, 1, 600.0),
            ],
            1,
            50,
            true,
        );

        assert!(result
            .protein_quantities
            .iter()
            .all(|(protein, _)| *protein != ProteinId::new(0)));
        let exported = match result
            .normalized_ions
            .iter()
            .find(|row| row.protein == ProteinId::new(0))
        {
            Some(row) => row,
            None => panic!("single-observation ion must still be exported"),
        };
        assert_eq!(value_for(&exported.quantities, 0), 0.0);
        assert_eq!(value_for(&exported.quantities, 1), 0.0);
    }

    // A 104-ion log2 block (4 samples) where every ion ties on NaN count (zero
    // NaNs), so the `ProtvalCutter` secondary key alone decides which 100
    // survive. 100 "big" ions are flat at log2 15 (log2-sum 60); 4 "spike" ions
    // are rotations of [20, 1, 1, 1] (log2-sum 23 but the largest *linear* sum).
    // directlfq sorts by `-np.nansum(log2_row)` so the big ions win and every
    // spike is dropped; the old `-sum(2**value)` key would have kept the spikes
    // and dropped 4 big ions instead. Reference: directlfq `ProtvalCutter`
    // (`protein_intensity_estimation.py:183-194`) run on this exact block.
    fn tie_break_block() -> Vec<Vec<f64>> {
        let mut block: Vec<Vec<f64>> = (0..100).map(|_| vec![15.0, 15.0, 15.0, 15.0]).collect();
        let spike = [20.0, 1.0, 1.0, 1.0];
        for rotation in 0..4 {
            let mut row = spike.to_vec();
            row.rotate_right(rotation);
            block.push(row);
        }
        block
    }

    #[test]
    fn trim_ions_breaks_ties_on_log2_sum_not_linear() {
        let (_, kept) = trim_ions(&tie_break_block());
        assert_eq!(kept.len(), 100);
        // Only the flat big ions survive; no spike (which carries a 20.0 and
        // would only rank in under the buggy linear-sum key) gets through.
        for row in &kept {
            assert_eq!(row, &vec![15.0, 15.0, 15.0, 15.0]);
        }
    }

    // End-to-end through `direct_lfq_aligned`: the same protein fed as linear
    // intensities. The four spike rotations keep every sample column's value
    // multiset identical, so the global sample-normalization shifts are all 0
    // and the result is exact. Dropping the spikes yields a flat per-sample
    // profile of 3_276_800 (= 2**15 * 100, the summed-pepint rescaling of the
    // 100 flat ions); the old linear-sum key would keep the spikes and produce
    // 4_194_310. Oracle: mokume's directlfq path (`conda run -n Bigbio python
    // dlfq_protvalcutter_oracle.py`).
    #[test]
    fn matches_directlfq_protvalcutter_over_100_ions() {
        let mut ions = Vec::new();
        for (ion_id, row) in tie_break_block().iter().enumerate() {
            for (sample, &log2_value) in row.iter().enumerate() {
                ions.push(ion(0, ion_id as u32, sample as u32, log2_value.exp2()));
            }
        }

        let result = direct_lfq_aligned_with_ions(&ions, 1, 50, true);
        let Some((_, profile)) = result
            .protein_quantities
            .iter()
            .find(|(protein, _)| *protein == ProteinId::new(0))
        else {
            panic!("protein missing from result");
        };

        for sample in 0..4 {
            assert_rel(value_for(profile, sample), 3_276_800.0);
        }
        assert_eq!(
            result
                .normalized_ions
                .iter()
                .map(|row| row.ion.get())
                .collect::<Vec<_>>(),
            (0..100).collect::<Vec<_>>(),
            "ProtvalCutter export must retain the same ordered ions used for estimation"
        );
    }
}
