use crate::aggregation::maxlfq::lexical_id_remap;
use crate::protein_matrix::ProteinValues;
use crate::state::{CellKey, DirectLfqCellKey};
use crate::{
    DirectLfqIon, DirectLfqNormalizedIon, DirectLfqResult, DirectLfqRetain, File, HashMap, HashSet,
    IntoParallelIterator, IntoParallelRefIterator, MokumeError, ParallelIterator, Path, PeptideId,
    PeptideMeasurement, ProfileResiduals, ProteinId, Result, SampleId, StringIdRegistry,
    WriterBuilder, cell_precisions, create_parent_dir, csv_error, direct_lfq_aligned_with,
    format_float, invalid_input, profile_residuals,
};
use rayon::prelude::ParallelSliceMut;

/// Convert one aggregated `(protein, canonical, sample)` cell into a
/// [`DirectLfqIon`]; DirectLFQ's "ion" is the bare canonical sequence.
/// `ion_seq_rank` is the canonical's global alphabetical-by-sequence rank, used
/// to reproduce DirectLFQ's `sort([protein, sequence])` row order.
fn directlfq_ion((key, intensity): (DirectLfqCellKey, f64), ion_seq_rank: u32) -> DirectLfqIon {
    DirectLfqIon {
        protein: key.protein,
        ion: key.canonical,
        ion_seq_rank,
        sample: key.sample,
        intensity,
    }
}

pub(crate) struct PreparedDirectLfq {
    pub(crate) ions: Vec<DirectLfqIon>,
    lexical_to_protein: HashMap<ProteinId, ProteinId>,
    lexical_to_sample: HashMap<SampleId, SampleId>,
    solver_samples: Vec<SampleId>,
}

fn remap_directlfq_ion(
    entry: (DirectLfqCellKey, f64),
    ion_seq_rank: u32,
    protein_to_lexical: &HashMap<ProteinId, ProteinId>,
    sample_to_lexical: &HashMap<SampleId, SampleId>,
) -> DirectLfqIon {
    let mut ion = directlfq_ion(entry, ion_seq_rank);
    ion.protein = protein_to_lexical
        .get(&ion.protein)
        .copied()
        .unwrap_or(ion.protein);
    ion.sample = sample_to_lexical
        .get(&ion.sample)
        .copied()
        .unwrap_or(ion.sample);
    ion
}

/// Build DirectLFQ input with name-sorted ids. Matrix row and column positions
/// participate in DirectLFQ tie-breaking, so encounter-order registry ids must
/// not leak into the solver.
pub(crate) fn prepare_directlfq_ions(
    directlfq_sums: Vec<(DirectLfqCellKey, f64)>,
    allowed_cells: &HashSet<CellKey>,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) -> PreparedDirectLfq {
    let (protein_to_lexical, lexical_to_protein) = lexical_id_remap(proteins);
    let (sample_to_lexical, lexical_to_sample) = lexical_id_remap(samples);
    let seq_rank = canonical_sequence_ranks(&directlfq_sums, canonical_peptides);
    let mut ions = directlfq_sums
        .into_par_iter()
        .filter(|(key, _)| {
            allowed_cells.contains(&CellKey {
                protein: key.protein,
                sample: key.sample,
            })
        })
        .map(|entry| {
            let rank = seq_rank
                .get(&entry.0.canonical)
                .copied()
                .unwrap_or(u32::MAX);
            remap_directlfq_ion(entry, rank, &protein_to_lexical, &sample_to_lexical)
        })
        .collect::<Vec<_>>();
    // Each (protein, ion, sample) occurs once, so the keys are distinct.
    ions.par_sort_unstable_by_key(|ion| {
        (
            ion.protein.get(),
            ion.ion_seq_rank,
            ion.ion.get(),
            ion.sample.get(),
        )
    });
    let mut solver_samples = ions.iter().map(|ion| ion.sample).collect::<Vec<_>>();
    solver_samples.sort_by_key(|sample| sample.get());
    solver_samples.dedup();
    PreparedDirectLfq {
        ions,
        lexical_to_protein,
        lexical_to_sample,
        solver_samples,
    }
}

/// One solved MaxLFQ protein: its positive values, the stabilized ratio pairs,
/// and its ion residuals when cell precision is requested.
type MaxLfqRow = (
    ProteinId,
    Vec<(SampleId, f64)>,
    usize,
    Option<ProfileResiduals>,
);

/// The proteins with values, and the cell precisions of every protein with
/// residuals.
pub(crate) fn maxlfq_values_and_precision(
    rows: Vec<MaxLfqRow>,
) -> (ProteinValues, HashMap<CellKey, f64>) {
    let mut residuals = Vec::new();
    let values = rows
        .into_iter()
        .filter_map(|(protein, values, _, fit)| {
            residuals.extend(fit.map(|fit| (protein, fit)));
            (!values.is_empty()).then_some((protein, values))
        })
        .collect();
    let precision = cell_precisions(&residuals)
        .into_iter()
        .map(|((protein, sample), value)| (CellKey { protein, sample }, value))
        .collect();
    (ProteinValues::Rows(values), precision)
}

/// One MaxLFQ protein's species traces around its solved profile, ions in
/// species order and samples in id order.
pub(crate) fn maxlfq_residuals(
    mut measurements: Vec<PeptideMeasurement>,
    profile: &[(SampleId, f64)],
) -> ProfileResiduals {
    measurements.sort_by_key(|measurement| (measurement.peptide.get(), measurement.sample.get()));
    let ions = measurements
        .chunk_by(|left, right| left.peptide == right.peptide)
        .map(|ion| {
            log2_present(
                ion.iter()
                    .map(|measurement| (measurement.sample, measurement.intensity)),
            )
        })
        .collect::<Vec<_>>();
    profile_residuals(&ions, &log2_present(profile.iter().copied()))
}

/// DirectLFQ protein values keyed by the matrix ids, with their cell
/// precisions when requested.
pub(crate) type DirectLfqCells = (
    HashMap<ProteinId, Vec<(SampleId, f64)>>,
    HashMap<CellKey, f64>,
);

/// Solve DirectLFQ on the prepared ions, with each protein's ion residuals
/// when the cell precisions are requested.
pub(crate) fn solve_directlfq(
    mut prepared: PreparedDirectLfq,
    min_nonan: usize,
    num_samples_quadratic: usize,
    normalize_samples: bool,
    cell_precision: bool,
) -> DirectLfqCells {
    let result = direct_lfq_aligned_with(
        std::mem::take(&mut prepared.ions),
        min_nonan,
        num_samples_quadratic,
        normalize_samples,
        DirectLfqRetain {
            ions: false,
            residuals: cell_precision,
        },
    );
    directlfq_cells(result, &prepared)
}

pub(crate) fn directlfq_cells(
    result: DirectLfqResult,
    prepared: &PreparedDirectLfq,
) -> DirectLfqCells {
    let precision = directlfq_precision(&result.residuals, prepared);
    (
        remap_directlfq_values(result.protein_quantities, prepared),
        precision,
    )
}

/// Cell precisions of a DirectLFQ run, keyed by the matrix ids.
fn directlfq_precision(
    residuals: &[(ProteinId, ProfileResiduals)],
    prepared: &PreparedDirectLfq,
) -> HashMap<CellKey, f64> {
    cell_precisions(residuals)
        .into_iter()
        .map(|((protein, sample), value)| {
            let protein = prepared
                .lexical_to_protein
                .get(&protein)
                .copied()
                .unwrap_or(protein);
            let sample = prepared
                .lexical_to_sample
                .get(&sample)
                .copied()
                .unwrap_or(sample);
            (CellKey { protein, sample }, value)
        })
        .collect()
}

/// Log2 of the positive finite intensities.
fn log2_present(values: impl Iterator<Item = (SampleId, f64)>) -> Vec<(SampleId, f64)> {
    values
        .filter(|(_, value)| value.is_finite() && *value > 0.0)
        .map(|(sample, value)| (sample, value.log2()))
        .collect()
}

fn remap_directlfq_values(
    values: Vec<(ProteinId, Vec<(SampleId, f64)>)>,
    prepared: &PreparedDirectLfq,
) -> HashMap<ProteinId, Vec<(SampleId, f64)>> {
    values
        .into_iter()
        .map(|(protein, per_sample)| {
            let protein = prepared
                .lexical_to_protein
                .get(&protein)
                .copied()
                .unwrap_or(protein);
            let per_sample = per_sample
                .into_iter()
                .map(|(sample, intensity)| {
                    let sample = prepared
                        .lexical_to_sample
                        .get(&sample)
                        .copied()
                        .unwrap_or(sample);
                    (sample, intensity)
                })
                .collect();
            (protein, per_sample)
        })
        .collect()
}

fn directlfq_sample_columns<'a>(
    prepared: &PreparedDirectLfq,
    samples: &'a StringIdRegistry<SampleId>,
) -> Result<Vec<(SampleId, &'a str)>> {
    prepared
        .solver_samples
        .iter()
        .map(|lexical| {
            let original = prepared
                .lexical_to_sample
                .get(lexical)
                .copied()
                .unwrap_or(*lexical);
            let name = samples
                .resolve(original)
                .ok_or_else(|| invalid_input("DirectLFQ ion export sample id is not registered"))?;
            Ok((*lexical, name))
        })
        .collect()
}

fn directlfq_ion_row(
    normalized: &DirectLfqNormalizedIon,
    prepared: &PreparedDirectLfq,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    sample_columns: &[(SampleId, &str)],
) -> Result<Vec<String>> {
    let original_protein = prepared
        .lexical_to_protein
        .get(&normalized.protein)
        .copied()
        .unwrap_or(normalized.protein);
    let protein = proteins
        .resolve(original_protein)
        .ok_or_else(|| invalid_input("DirectLFQ ion export protein id is not registered"))?;
    let ion = canonical_peptides
        .resolve(normalized.ion)
        .ok_or_else(|| invalid_input("DirectLFQ ion export peptide id is not registered"))?;
    let quantities = normalized
        .quantities
        .iter()
        .copied()
        .collect::<HashMap<_, _>>();
    let mut row = vec![protein.to_owned(), ion.to_owned()];
    row.extend(
        sample_columns
            .iter()
            .map(|(sample, _)| format_float(quantities.get(sample).copied().unwrap_or(0.0))),
    );
    Ok(row)
}

pub(crate) fn write_directlfq_ions(
    path: &Path,
    normalized_ions: &[DirectLfqNormalizedIon],
    prepared: &PreparedDirectLfq,
    proteins: &StringIdRegistry<ProteinId>,
    canonical_peptides: &StringIdRegistry<PeptideId>,
    samples: &StringIdRegistry<SampleId>,
) -> Result<()> {
    create_parent_dir(path)?;
    let file = File::create(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = WriterBuilder::new().from_writer(file);

    let sample_columns = directlfq_sample_columns(prepared, samples)?;
    let mut header = vec!["protein".to_owned(), "ion".to_owned()];
    header.extend(sample_columns.iter().map(|(_, name)| (*name).to_owned()));
    writer
        .write_record(header)
        .map_err(|source| csv_error(path, source))?;

    let mut rows = normalized_ions.iter().collect::<Vec<_>>();
    rows.sort_by_key(|normalized| {
        (
            normalized.protein.get(),
            normalized.ion_seq_rank,
            normalized.ion.get(),
        )
    });
    for normalized in rows {
        let row = directlfq_ion_row(
            normalized,
            prepared,
            proteins,
            canonical_peptides,
            &sample_columns,
        )?;
        writer
            .write_record(row)
            .map_err(|source| csv_error(path, source))?;
    }

    writer.flush().map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Assign every distinct canonical id appearing in `directlfq_sums` a rank by
/// sorting their bare sequence strings lexically, mirroring DirectLFQ's global
/// `sort([protein, sequence])` (`stages.py:887`). Ties on the (unique) sequence
/// strings cannot occur; canonical ids missing from the registry sort last by
/// their id so the order stays deterministic.
fn canonical_sequence_ranks(
    directlfq_sums: &[(DirectLfqCellKey, f64)],
    canonical_peptides: &StringIdRegistry<PeptideId>,
) -> HashMap<PeptideId, u32> {
    let mut canonicals = directlfq_sums
        .par_iter()
        .map(|(key, _)| key.canonical)
        .collect::<Vec<_>>();
    canonicals.par_sort_unstable();
    canonicals.dedup();
    canonicals.sort_by(|a, b| {
        let seq_a = canonical_peptides.resolve(*a);
        let seq_b = canonical_peptides.resolve(*b);
        seq_a.cmp(&seq_b).then_with(|| a.get().cmp(&b.get()))
    });
    canonicals
        .into_iter()
        .enumerate()
        .map(|(rank, canonical)| (canonical, rank as u32))
        .collect()
}
