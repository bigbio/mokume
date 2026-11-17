mod center;
pub(crate) mod hierarchical;
pub(crate) mod lfq_traces;
mod loess;
mod rlr;
mod tmm;
use crate::peptide_normalization::center::{CenterStat, apply_center_to_peptide_cells};
use crate::peptide_normalization::hierarchical::apply_hierarchical_to_peptide_cells;
use crate::peptide_normalization::lfq_traces::quantile_normalized_assignments;
use crate::peptide_normalization::loess::apply_loess_to_peptide_cells;
use crate::peptide_normalization::rlr::apply_rlr_to_peptide_cells;
use crate::peptide_normalization::tmm::apply_tmm_to_peptide_cells;
use crate::state::CellKey;
use crate::{
    HashMap, HashSet, PeptideId, ProteinId, SampleId, SampleNormalizationMethod, StringIdRegistry,
};

pub(crate) fn push_peptide_max(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    cell: CellKey,
    peptide: PeptideId,
    intensity: f64,
) {
    let peptides = cells.entry(cell).or_default();
    let current = peptides.entry(peptide).or_insert(0.0);
    if intensity > *current {
        *current = intensity;
    }
}

pub(crate) fn sum_peptide_values(peptides: &HashMap<PeptideId, f64>) -> f64 {
    let mut values = peptides.iter().collect::<Vec<_>>();
    values.sort_by_key(|(peptide, _)| **peptide);
    values.into_iter().map(|(_, value)| *value).sum()
}

/// Dataset normalizers other than TMM replace contextual ions with canonical
/// peptide cells; TMM scales the ions in place.
pub(crate) fn writes_canonical_cells(method: SampleNormalizationMethod) -> bool {
    method != SampleNormalizationMethod::Tmm
}

/// The `topn` most intense of a cell's canonical peptide values. With
/// weights, higher-weight (identified) peptides are taken before lower-weight
/// (transferred) ones, the most intense first within a weight.
pub(crate) fn top_n_intensities(
    canonical: HashMap<PeptideId, f64>,
    weights: Option<&HashMap<PeptideId, f64>>,
    topn: usize,
) -> Vec<f64> {
    let mut ranked = canonical
        .into_iter()
        .map(|(peptide, intensity)| {
            let weight = weights
                .and_then(|weights| weights.get(&peptide))
                .copied()
                .unwrap_or(1.0);
            (weight, intensity)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.total_cmp(&left.1))
    });
    ranked
        .into_iter()
        .take(topn)
        .map(|(_, intensity)| intensity)
        .collect()
}

/// Collapse per-(peptidoform, charge) values into one value per canonical
/// peptide by summing, mirroring the Python loader which keeps the max
/// intensity per ion across runs and then sums ions into the canonical peptide
/// (`get_peptidoform_normalize_intensities` then `sum_peptidoform_intensities`).
/// Median/Abd/TopN/PeptideCount operate on these canonical values.
pub(crate) fn collapse_to_canonical(
    peptides: HashMap<PeptideId, f64>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) -> HashMap<PeptideId, f64> {
    let mut canonical = HashMap::with_capacity(peptides.len());
    let mut peptides = peptides.into_iter().collect::<Vec<_>>();
    peptides.sort_by_key(|(peptide, _)| *peptide);
    for (peptide, value) in peptides {
        let key = peptide_to_canonical
            .get(&peptide)
            .copied()
            .unwrap_or(peptide);
        *canonical.entry(key).or_insert(0.0) += value;
    }
    canonical
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct QuantilePeptideKey {
    protein: ProteinId,
    peptide: PeptideId,
}

/// Apply a dataset-level sample normalization to the canonical-peptide x sample
/// cells in place. Shared by the protein-matrix aggregation (`Self::Sum` /
/// `Median` / `Abd` / `PeptideCount` / `TopN`) and the `--export-peptides`
/// path, so the exported peptides carry the exact same normalization Python
/// applies in `load_for_mokume` before writing the peptide CSV.
pub(crate) fn apply_dataset_norm_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    method: SampleNormalizationMethod,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) {
    match method {
        SampleNormalizationMethod::Quantile => {
            apply_quantile_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Rlr => {
            apply_rlr_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Loess => {
            apply_loess_to_peptide_cells(cells, allowed_cells, peptide_to_canonical);
        }
        SampleNormalizationMethod::Hierarchical => {
            apply_hierarchical_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                samples,
            );
        }
        SampleNormalizationMethod::MedianCenter => {
            apply_center_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                CenterStat::Median,
            );
        }
        SampleNormalizationMethod::MeanCenter => {
            apply_center_to_peptide_cells(
                cells,
                allowed_cells,
                peptide_to_canonical,
                CenterStat::Mean,
            );
        }
        SampleNormalizationMethod::Tmm => {
            apply_tmm_to_peptide_cells(cells, allowed_cells, peptide_to_canonical, samples);
        }
        _ => {}
    }
}

fn apply_quantile_to_peptide_cells(
    cells: &mut HashMap<CellKey, HashMap<PeptideId, f64>>,
    allowed_cells: &HashSet<CellKey>,
    peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
) {
    let mut canonical_cells = HashMap::<CellKey, HashMap<PeptideId, f64>>::new();
    for (cell, peptides) in cells.iter() {
        if !allowed_cells.contains(cell) {
            continue;
        }
        for (peptide, intensity) in peptides {
            let canonical = peptide_to_canonical
                .get(peptide)
                .copied()
                .unwrap_or(*peptide);
            *canonical_cells
                .entry(*cell)
                .or_default()
                .entry(canonical)
                .or_insert(0.0) += *intensity;
        }
    }

    let assignments =
        quantile_normalized_assignments(canonical_cells.iter().flat_map(|(cell, peptides)| {
            peptides.iter().map(|(peptide, intensity)| {
                (
                    QuantilePeptideKey {
                        protein: cell.protein,
                        peptide: *peptide,
                    },
                    cell.sample,
                    *intensity,
                )
            })
        }));

    for (cell, peptides) in cells {
        if !allowed_cells.contains(cell) {
            continue;
        }
        let Some(canonical_peptides) = canonical_cells.remove(cell) else {
            continue;
        };
        peptides.clear();
        for (peptide, _) in canonical_peptides {
            let key = QuantilePeptideKey {
                protein: cell.protein,
                peptide,
            };
            if let Some(normalized) = assignments.get(&(key, cell.sample)).copied() {
                peptides.insert(peptide, normalized);
            }
        }
    }
}
