//! `peptides2protein` command: compute per-protein quantities from a
//! peptide-level intensity matrix.
//!
//! This mirrors `mokume.commands.peptides2protein.peptides2protein` in the
//! Python package. Two code paths exist:
//!
//!   * piBAQ (`--quant-method pibaq`, the default) reuses the existing piBAQ core via
//!     `mokume_pipeline::run_pibaq_from_peptides`. A FASTA is required to derive
//!     theoretical peptide counts. The output is the Python long-format table
//!     `ProteinName, SampleID, Condition, NormIntensity, PiBAQ, FamilyId,
//!     EvidenceLevel, FamilySize`.
//!
//!   * The deterministic generic methods `sum`, `top3`, and `topn` are computed
//!     directly from the long-format rows, faithfully reproducing the Python
//!     `AllPeptidesQuantification` / `TopNQuantification.quantify` group-bys
//!     (raw per-row aggregation; no canonical/charge collapse). The output is
//!     `ProteinName, SampleID, Intensity` plus `Condition` (when the input has
//!     it) and `IntensityNorm` (when `--normalize` is set).
//!
//!   * `maxlfq` uses pairwise peptide-species ratios and a least-squares protein
//!     profile (Cox et al., Eq. 3). `directlfq` uses hierarchical alignment.
//!     Both return the same positive-intensity long-format schema.
//!
//! piBAQ extras (P3) are computed as deterministic post-processing on the piBAQ
//! result, wired into `--tpa`, `--ruler`, and `--normalize` (see the
//! [`pibaq_extras`] module):
//!   * `--tpa`: Total Protein Approach -- `MolecularWeight` + `TPA` columns.
//!   * `--normalize`: the Python `normalize_pibaq` PRIDE/ProteomicsDB transforms
//!     (`PiBAQNorm`, `PiBAQLog`, `PiBAQPpb`).
//!   * `--ruler`: the ProteomicRuler copy number / moles / weight /
//!     concentration columns (requires `--tpa`).
//!
//! piBAQ FASTA digestion uses the complete protease catalog registered by the
//! installed pyOpenMS runtime, with zero missed cleavages. Python passes the full
//! canonical protein -> theoretical peptide map into this Rust kernel; shared
//! allocation, denominators, TPA, normalization, and output remain Rust-native.
//!
//! Input format: a comma-separated CSV (matching Python's `pd.read_csv`), a
//! tab-separated `.tsv`, or a parquet file (matching Python's `is_parquet`
//! magic-byte check + `pd.read_parquet`), with at minimum `ProteinName`,
//! `SampleID`, and `NormIntensity`. The peptide column is `PeptideCanonical`
//! (preferred) or `PeptideSequence`; it is required for piBAQ. `Condition` is
//! optional. Parquet `SampleID` / `Condition` columns may be dictionary-encoded
//! (pandas `Categorical`), as written by `quantify features2peptides --save-parquet`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, ErrorKind, Read, Write};
use std::path::Path;

use mokume_core::quant::parse_topn_from_method_name;
use mokume_core::{DirectLfqConfig, MaxLfqConfig, MokumeError, PibaqConfig, Result};
use mokume_io::{read_peptide_parquet, read_peptide_parquet_with_species};
use mokume_pipeline::{
    run_lfq_from_peptides_with_threads, run_maxlfq_from_peptides_with_threads,
    run_pibaq_from_peptides, LfqPeptideObservation, PeptideObservation, PibaqDigest,
    PibaqFromPeptidesParams, PibaqProteinRow,
};

use crate::Peptides2ProteinArgs;

/// Column header for the protein identifier (Python `PROTEIN_NAME`).
const PROTEIN_NAME: &str = "ProteinName";
/// Column header for the sample identifier (Python `SAMPLE_ID`).
const SAMPLE_ID: &str = "SampleID";
/// Column header for the normalized intensity (Python `NORM_INTENSITY`).
const NORM_INTENSITY: &str = "NormIntensity";
/// Optional condition column (Python `CONDITION`).
const CONDITION: &str = "Condition";
/// Preferred peptide-sequence column (Python `PEPTIDE_CANONICAL`).
const PEPTIDE_CANONICAL: &str = "PeptideCanonical";
/// Fallback peptide-sequence column (Python `PEPTIDE_SEQUENCE`).
const PEPTIDE_SEQUENCE: &str = "PeptideSequence";

/// A parsed peptide-level input table, retaining the per-row values needed by
/// every supported method.
struct PeptideTable {
    /// `(protein, sample, peptide, condition, intensity)` per accepted row.
    rows: Vec<PeptideRow>,
    /// Whether the input carried a `Condition` column.
    has_condition: bool,
    /// Whether the input carried a peptide-sequence column.
    has_peptide: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct PeptideRow {
    protein: String,
    sample: String,
    peptide: String,
    condition: String,
    intensity: f64,
}

pub fn run_peptides_to_protein_with_digest(
    args: &Peptides2ProteinArgs,
    pibaq_digest: Option<PibaqDigest>,
) -> Result<()> {
    let method = args.quant_method.to_ascii_lowercase();
    let output = peptides2protein_output(args)?;
    validate_lfq_options(args, &method)?;
    validate_pibaq_options(args, &method)?;

    match method.as_str() {
        "pibaq" => run_pibaq(args, output, pibaq_digest),
        "sum" => run_generic(args, &method, output),
        // `--quant-method` is validated by
        // `parse_peptides2protein_method`, so any `top`-prefixed name reaching
        // here is a well-formed `top<N>`.
        name if parse_topn_from_method_name(name).is_some() => run_generic(args, &method, output),
        "maxlfq" | "directlfq" => run_lfq(args, &method, output),
        other => Err(MokumeError::InvalidInput {
            message: format!("unknown peptides2protein method '{other}'"),
        }),
    }
}

fn peptides2protein_output(args: &Peptides2ProteinArgs) -> Result<&Path> {
    if !args.peptides.exists() {
        return Err(MokumeError::MissingInput {
            path: args.peptides.clone(),
        });
    }

    args.output
        .as_deref()
        .ok_or_else(|| MokumeError::InvalidInput {
            message: "peptides2protein requires --output".to_owned(),
        })
}

fn validate_lfq_options(args: &Peptides2ProteinArgs, method: &str) -> Result<()> {
    let is_lfq = matches!(method, "maxlfq" | "directlfq");
    if !is_lfq && args.threads.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--threads only applies to peptides2protein DirectLFQ/MaxLFQ".to_owned(),
        });
    }
    if method != "directlfq" && args.directlfq_min_nonan.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-min-nonan only applies to peptides2protein DirectLFQ".to_owned(),
        });
    }
    if method != "maxlfq" && args.maxlfq_min_ratio_count.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--maxlfq-min-ratio-count only applies to peptides2protein MaxLFQ".to_owned(),
        });
    }
    if method != "maxlfq" && args.stabilize {
        return Err(MokumeError::InvalidInput {
            message: "--stabilize requires --quant-method maxlfq".to_owned(),
        });
    }
    Ok(())
}

fn validate_pibaq_options(args: &Peptides2ProteinArgs, method: &str) -> Result<()> {
    let has_pibaq_options = [
        args.fasta.is_some(),
        args.enzyme.is_some(),
        args.min_aa.is_some(),
        args.max_aa.is_some(),
        args.tpa,
        args.ruler,
        args.ploidy.is_some(),
        args.organism.is_some(),
        args.cpc.is_some(),
        args.families_yaml.is_some(),
        args.min_shared.is_some(),
        args.min_anchors.is_some(),
        args.high_anchor_threshold.is_some(),
        args.qc_report.is_some(),
    ]
    .into_iter()
    .any(|configured| configured);

    if method != "pibaq" && has_pibaq_options {
        return Err(MokumeError::InvalidInput {
            message: "piBAQ digestion/TPA/ruler/QC options require --quant-method pibaq".to_owned(),
        });
    }
    if method == "pibaq"
        && !args.ruler
        && (args.ploidy.is_some() || args.organism.is_some() || args.cpc.is_some())
    {
        return Err(MokumeError::InvalidInput {
            message: "--ploidy/--organism/--cpc only apply with --ruler".to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn validate_options(args: &Peptides2ProteinArgs) -> Result<()> {
    let method = args.quant_method.to_ascii_lowercase();
    validate_lfq_options(args, &method)?;
    validate_pibaq_options(args, &method)
}

/// Dispatch to the requested LFQ algorithm and emit positive long-format rows.
fn run_lfq(args: &Peptides2ProteinArgs, method: &str, output: &Path) -> Result<()> {
    let table = load_peptide_table_with_species(&args.peptides, method == "maxlfq")?;
    if !table.has_peptide {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "{method} requires a peptide column ('{PEPTIDE_CANONICAL}' or '{PEPTIDE_SEQUENCE}')"
            ),
        });
    }

    let mut condition_by_sample: HashMap<String, String> = HashMap::new();
    let mut observations = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        condition_by_sample
            .entry(row.sample.clone())
            .or_insert_with(|| row.condition.clone());
        observations.push(LfqPeptideObservation {
            protein: row.protein.clone(),
            peptide: row.peptide.clone(),
            sample: row.sample.clone(),
            intensity: row.intensity,
        });
    }

    let quantified = if method == "maxlfq" {
        run_maxlfq_from_peptides_with_threads(
            &observations,
            args.maxlfq_min_ratio_count
                .unwrap_or(MaxLfqConfig::default().min_ratio_count),
            args.stabilize,
            args.threads,
        )?
    } else {
        // peptides2protein does not expose num_samples_quadratic, so the
        // DirectLFQ default (directlfq's own default of 50) is fixed here.
        let defaults = DirectLfqConfig::default();
        run_lfq_from_peptides_with_threads(
            &observations,
            args.directlfq_min_nonan.unwrap_or(defaults.min_nonan),
            defaults.num_samples_quadratic,
            args.threads,
        )?
    };
    let mut results: Vec<GenericRow> = quantified
        .into_iter()
        .map(|row| GenericRow {
            protein: row.protein,
            sample: row.sample,
            intensity: row.intensity,
        })
        .collect();
    results.sort_by(|left, right| {
        left.protein
            .cmp(&right.protein)
            .then_with(|| left.sample.cmp(&right.sample))
    });

    // `--normalize` divides each protein's intensity by its sample total, exactly
    // as the Python generic path does after `quantify`.
    let norm = if args.normalize {
        let mut totals: HashMap<&str, f64> = HashMap::new();
        for row in &results {
            *totals.entry(row.sample.as_str()).or_insert(0.0) += row.intensity;
        }
        Some(totals)
    } else {
        None
    };

    write_generic_output(
        output,
        &results,
        &condition_by_sample,
        table.has_condition,
        norm.as_ref(),
    )
}

/// piBAQ path: reuse the existing piBAQ core; emit the Python long-format table
/// plus any requested extras (`--tpa`, `--normalize`, `--ruler`).
///
/// The Python wheel supplies the complete theoretical-peptide map from its
/// installed pyOpenMS catalog before this Rust aggregation path starts.
fn resolve_pibaq_organism(args: &Peptides2ProteinArgs) -> Result<Option<pibaq_extras::Organism>> {
    if !args.ruler {
        return Ok(None);
    }
    let organism = Some(pibaq_extras::resolve_organism(
        args.organism.as_deref().unwrap_or("human"),
    )?);
    if !args.tpa {
        return Err(MokumeError::InvalidInput {
            message:
                "`ploidy`, `cpc`, `organism` and `tpa` are required to calculate protein weight and concentration"
                    .to_owned(),
        });
    }
    Ok(organism)
}

fn pibaq_fasta(args: &Peptides2ProteinArgs) -> Result<&Path> {
    let Some(fasta) = args.fasta.as_deref() else {
        return Err(MokumeError::InvalidInput {
            message: "the --fasta option is required for the piBAQ method".to_owned(),
        });
    };
    if !fasta.exists() {
        return Err(MokumeError::MissingInput {
            path: fasta.to_path_buf(),
        });
    }
    Ok(fasta)
}

fn pibaq_observations(table: &PeptideTable) -> (HashMap<String, String>, Vec<PeptideObservation>) {
    let mut conditions = HashMap::new();
    let observations = table
        .rows
        .iter()
        .map(|row| {
            conditions
                .entry(row.sample.clone())
                .or_insert_with(|| row.condition.clone());
            PeptideObservation {
                peptide: row.peptide.clone(),
                sample: row.sample.clone(),
                intensity: row.intensity,
            }
        })
        .collect();
    (conditions, observations)
}

fn pibaq_params(args: &Peptides2ProteinArgs, fasta: &Path) -> PibaqFromPeptidesParams {
    let defaults = PibaqConfig::default();
    PibaqFromPeptidesParams {
        fasta: fasta.to_path_buf(),
        min_aa: args.pibaq_min_aa(),
        max_aa: args.pibaq_max_aa(),
        min_shared: args.min_shared.unwrap_or(defaults.min_shared),
        min_anchors: args.min_anchors.unwrap_or(defaults.min_anchors),
        high_anchor_threshold: args
            .high_anchor_threshold
            .unwrap_or(defaults.high_anchor_threshold),
        families_yaml: args.families_yaml.clone(),
        enzyme: args.pibaq_enzyme(),
        tpa: args.tpa,
    }
}

fn prepare_pibaq_records(
    rows: Vec<PibaqProteinRow>,
    conditions: &HashMap<String, String>,
    args: &Peptides2ProteinArgs,
    organism: Option<&pibaq_extras::Organism>,
) -> Result<Vec<pibaq_extras::PibaqExtraRow>> {
    let mut records = pibaq_extras::PibaqExtraRow::lift(rows, conditions);
    if args.normalize {
        pibaq_extras::normalize_pibaq(&mut records);
    }
    if args.ruler {
        let Some(organism) = organism else {
            return Err(MokumeError::InvalidInput {
                message: "the --organism option is required for the proteomic ruler".to_owned(),
            });
        };
        pibaq_extras::apply_ruler(
            &mut records,
            organism,
            args.ploidy.unwrap_or(2),
            args.cpc.unwrap_or(200.0),
        );
    }
    records.sort_by(|left, right| {
        left.protein
            .cmp(&right.protein)
            .then_with(|| left.sample.cmp(&right.sample))
    });
    Ok(records)
}

fn run_pibaq(
    args: &Peptides2ProteinArgs,
    output: &Path,
    pibaq_digest: Option<PibaqDigest>,
) -> Result<()> {
    let organism = resolve_pibaq_organism(args)?;
    let fasta = pibaq_fasta(args)?;
    let table = load_peptide_table(&args.peptides)?;
    if !table.has_peptide {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "piBAQ requires a peptide column ('{PEPTIDE_CANONICAL}' or '{PEPTIDE_SEQUENCE}')"
            ),
        });
    }

    let (condition_by_sample, observations) = pibaq_observations(&table);
    let params = pibaq_params(args, fasta);
    let digest = pibaq_digest.ok_or_else(|| MokumeError::InvalidInput {
        message: "piBAQ requires the Python wheel's runtime pyOpenMS FASTA digest".to_owned(),
    })?;
    let rows = run_pibaq_from_peptides(&observations, &params, digest)?;

    let records = prepare_pibaq_records(rows, &condition_by_sample, args, organism.as_ref())?;
    write_pibaq_output(
        output,
        &records,
        table.has_condition,
        args.tpa,
        args.normalize,
        args.ruler,
    )?;

    Ok(())
}

/// Generic deterministic path (`sum` / `top<N>`): faithful re-creation of the
/// Python `quantify` group-bys.
fn run_generic(args: &Peptides2ProteinArgs, method: &str, output: &Path) -> Result<()> {
    // N is spelled in the method name (`top5`), so there is no separate option
    // to read it from. The caller only routes `sum` and validated `top<N>` here,
    // but report rather than panic if that ever stops holding.
    let topn = match method {
        "sum" => None,
        name => match parse_topn_from_method_name(name) {
            Some(n) => Some(n),
            None => {
                return Err(MokumeError::InvalidInput {
                    message: format!("unknown peptides2protein method '{name}'"),
                })
            }
        },
    };

    let table = load_peptide_table(&args.peptides)?;

    // Group raw rows by (protein, sample); Python does not collapse charges or
    // peptidoforms here, so neither do we.
    let mut groups: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut condition_by_sample: HashMap<String, String> = HashMap::new();
    for row in &table.rows {
        condition_by_sample
            .entry(row.sample.clone())
            .or_insert_with(|| row.condition.clone());
        groups
            .entry((row.protein.clone(), row.sample.clone()))
            .or_default()
            .push(row.intensity);
    }

    let mut results: Vec<GenericRow> = groups
        .into_iter()
        .map(|((protein, sample), intensities)| {
            let intensity = aggregate(intensities, topn);
            GenericRow {
                protein,
                sample,
                intensity,
            }
        })
        .collect();
    results.sort_by(|left, right| {
        left.protein
            .cmp(&right.protein)
            .then_with(|| left.sample.cmp(&right.sample))
    });

    // `--normalize` divides each protein's intensity by its sample total, the
    // generic-path transform in the Python command (the piBAQ path is deferred).
    let norm = if args.normalize {
        let mut totals: HashMap<&str, f64> = HashMap::new();
        for row in &results {
            *totals.entry(row.sample.as_str()).or_insert(0.0) += row.intensity;
        }
        Some(totals)
    } else {
        None
    };

    write_generic_output(
        output,
        &results,
        &condition_by_sample,
        table.has_condition,
        norm.as_ref(),
    )
}

/// Aggregate one (protein, sample) group: sum (when `topn` is `None`) or the
/// mean of the `topn` most intense rows. Mirrors Python's `groupby.sum` and
/// `sort_values(desc).head(n).mean`.
fn aggregate(mut intensities: Vec<f64>, topn: Option<usize>) -> f64 {
    match topn {
        None => intensities.iter().sum(),
        Some(n) => {
            intensities.sort_by(|left, right| right.total_cmp(left));
            let selected = &intensities[..intensities.len().min(n.max(1))];
            if selected.is_empty() {
                0.0
            } else {
                selected.iter().sum::<f64>() / selected.len() as f64
            }
        }
    }
}

struct GenericRow {
    protein: String,
    sample: String,
    intensity: f64,
}

/// Read a peptide-level table from CSV / `.tsv` / parquet, dispatching on the
/// parquet magic bytes (`PAR1`) exactly like Python's `is_parquet`, so the same
/// `--peptides` path resolves to the same loader regardless of file extension.
fn load_peptide_table(path: &Path) -> Result<PeptideTable> {
    load_peptide_table_with_species(path, false)
}

fn load_peptide_table_with_species(path: &Path, preserve_species: bool) -> Result<PeptideTable> {
    if looks_like_parquet(path)? {
        load_peptide_table_parquet(path, preserve_species)
    } else {
        load_peptide_table_csv(path, preserve_species)
    }
}

/// Detect a parquet file by its leading `PAR1` magic bytes, mirroring Python's
/// `mokume.core.constants.is_parquet` (which reads the first four bytes). A file
/// shorter than four bytes is treated as non-parquet, leaving the CSV/TSV path
/// to surface any parse error.
fn looks_like_parquet(path: &Path) -> Result<bool> {
    let mut file = File::open(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut header = [0_u8; 4];
    match file.read_exact(&mut header) {
        Ok(()) => Ok(&header == b"PAR1"),
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Ok(false),
        Err(source) => Err(MokumeError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Load a peptide table from a parquet file, applying the same intensity filter
/// (`dropna` + `> 0`) and `"Empty"` condition default as the CSV path so both
/// inputs yield identical [`PeptideTable`]s. Mirrors Python's `pd.read_parquet`
/// followed by the generic / piBAQ numeric coercion.
fn load_peptide_table_parquet(path: &Path, preserve_species: bool) -> Result<PeptideTable> {
    let raw = if preserve_species {
        read_peptide_parquet_with_species(path)?
    } else {
        read_peptide_parquet(path)?
    };
    let mut rows = Vec::with_capacity(raw.rows.len());
    for row in raw.rows {
        if preserve_species && row.peptide.as_deref().is_none_or(str::is_empty) {
            continue;
        }
        let Some(intensity) = row.intensity else {
            continue;
        };
        if !intensity.is_finite() || intensity <= 0.0 {
            continue;
        }
        let condition = if raw.has_condition {
            row.condition.unwrap_or_default()
        } else {
            "Empty".to_owned()
        };
        rows.push(PeptideRow {
            protein: row.protein,
            sample: row.sample,
            peptide: row.peptide.unwrap_or_default(),
            condition,
            intensity,
        });
    }

    Ok(PeptideTable {
        rows,
        has_condition: raw.has_condition,
        has_peptide: raw.has_peptide,
    })
}

/// Read a peptide-level table from CSV (comma) or `.tsv` (tab). Rows with a
/// missing or non-positive intensity are dropped, matching the Python
/// `dropna` + `> 0` filter applied on the piBAQ path and the implicit numeric
/// coercion on the generic path.
fn load_peptide_table_csv(path: &Path, preserve_species: bool) -> Result<PeptideTable> {
    let delimiter = if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("tsv"))
    {
        b'\t'
    } else {
        b','
    };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|source| csv_error(path, source))?;

    let headers = reader
        .headers()
        .map_err(|source| csv_error(path, source))?
        .iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let protein_index = column_index(&headers, PROTEIN_NAME)?;
    let sample_index = column_index(&headers, SAMPLE_ID)?;
    let intensity_index = column_index(&headers, NORM_INTENSITY)?;
    let condition_index = optional_column_index(&headers, CONDITION);
    let peptide_index = optional_column_index(&headers, PEPTIDE_CANONICAL)
        .or_else(|| optional_column_index(&headers, PEPTIDE_SEQUENCE));
    let species_indices = preserve_species
        .then(|| {
            optional_column_index(&headers, PEPTIDE_SEQUENCE)
                .zip(optional_column_index(&headers, "PrecursorCharge"))
        })
        .flatten();

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|source| csv_error(path, source))?;
        let Some(intensity) = positive_csv_intensity(&record, intensity_index, path)? else {
            continue;
        };
        let condition = condition_index
            .map(|index| field(&record, index, path))
            .transpose()?
            .map_or_else(|| "Empty".to_owned(), ToOwned::to_owned);
        let peptide = csv_peptide_id(&record, path, peptide_index, species_indices)?;
        if preserve_species && peptide.is_empty() {
            continue;
        }
        rows.push(PeptideRow {
            protein: field(&record, protein_index, path)?.to_owned(),
            sample: field(&record, sample_index, path)?.to_owned(),
            peptide,
            condition,
            intensity,
        });
    }

    Ok(PeptideTable {
        rows,
        has_condition: condition_index.is_some(),
        has_peptide: peptide_index.is_some(),
    })
}

fn positive_csv_intensity(
    record: &csv::StringRecord,
    index: usize,
    path: &Path,
) -> Result<Option<f64>> {
    let raw = field(record, index, path)?.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let intensity = raw.parse::<f64>().map_err(|_| MokumeError::InvalidInput {
        message: format!("'{raw}' in column '{NORM_INTENSITY}' is not numeric"),
    })?;
    Ok((intensity.is_finite() && intensity > 0.0).then_some(intensity))
}

fn csv_peptide_id(
    record: &csv::StringRecord,
    path: &Path,
    peptide_index: Option<usize>,
    species_indices: Option<(usize, usize)>,
) -> Result<String> {
    if let Some((sequence, charge)) = species_indices {
        let sequence = field(record, sequence, path)?;
        let charge = field(record, charge, path)?;
        if sequence.is_empty() || charge.is_empty() {
            return Ok(String::new());
        }
        let charge = charge
            .parse::<f64>()
            .map_err(|_| MokumeError::InvalidInput {
                message: "PrecursorCharge must be numeric".to_owned(),
            })?;
        if !charge.is_finite() || charge <= 0.0 || charge.fract() != 0.0 {
            return Ok(String::new());
        }
        Ok(format!("{sequence}|z{charge}"))
    } else {
        Ok(peptide_index
            .map(|index| field(record, index, path))
            .transpose()?
            .unwrap_or("")
            .to_owned())
    }
}

/// Write the piBAQ long-format output (tab-separated, matching Python). Columns
/// are appended in the Python `peptides_to_protein` order: the base eight, then
/// the TPA pair, then the normalize triple, then the ruler quartet.
fn write_pibaq_output(
    output: &Path,
    rows: &[pibaq_extras::PibaqExtraRow],
    has_condition: bool,
    tpa: bool,
    normalize: bool,
    ruler: bool,
) -> Result<()> {
    let mut writer = create_writer(output)?;
    write_line(
        &mut writer,
        output,
        &pibaq_output_header(has_condition, tpa, normalize, ruler).join("\t"),
    )?;
    for row in rows {
        let fields = pibaq_output_fields(row, has_condition, tpa, normalize, ruler);
        write_line(&mut writer, output, &fields.join("\t"))?;
    }
    flush(&mut writer, output)
}

fn pibaq_output_header(
    has_condition: bool,
    tpa: bool,
    normalize: bool,
    ruler: bool,
) -> Vec<&'static str> {
    let mut header = vec![PROTEIN_NAME, SAMPLE_ID];
    if has_condition {
        header.push(CONDITION);
    }
    header.extend([
        NORM_INTENSITY,
        "PiBAQ",
        "FamilyId",
        "EvidenceLevel",
        "FamilySize",
    ]);
    if tpa {
        header.extend([pibaq_extras::MOLECULAR_WEIGHT, pibaq_extras::TPA]);
    }
    if normalize {
        header.extend([
            pibaq_extras::PIBAQ_NORM,
            pibaq_extras::PIBAQ_LOG,
            pibaq_extras::PIBAQ_PPB,
        ]);
    }
    if ruler {
        header.extend([
            pibaq_extras::COPY_NUMBER,
            pibaq_extras::MOLES_NMOL,
            pibaq_extras::WEIGHT_NG,
            pibaq_extras::CONCENTRATION_NM,
        ]);
    }
    header
}

fn pibaq_output_fields(
    row: &pibaq_extras::PibaqExtraRow,
    has_condition: bool,
    tpa: bool,
    normalize: bool,
    ruler: bool,
) -> Vec<String> {
    let mut fields = vec![row.protein.clone(), row.sample.clone()];
    if has_condition {
        fields.push(row.condition.clone());
    }
    fields.extend([
        format_float(row.norm_intensity),
        format_float(row.pibaq),
        row.family_id.clone(),
        row.evidence_level.to_owned(),
        row.family_size.to_string(),
    ]);
    if tpa {
        fields.extend([
            format_optional(row.molecular_weight),
            format_optional(row.tpa),
        ]);
    }
    if normalize {
        fields.extend([
            format_optional(row.pibaq_norm),
            format_optional(row.pibaq_log),
            format_optional(row.pibaq_ppb),
        ]);
    }
    if ruler {
        fields.extend([
            format_optional(row.copy_number),
            format_optional(row.moles_nmol),
            format_optional(row.weight_ng),
            format_optional(row.concentration_nm),
        ]);
    }
    fields
}

/// Write the generic-method long-format output (tab-separated, matching Python).
fn write_generic_output(
    output: &Path,
    rows: &[GenericRow],
    condition_by_sample: &HashMap<String, String>,
    has_condition: bool,
    norm: Option<&HashMap<&str, f64>>,
) -> Result<()> {
    let mut writer = create_writer(output)?;

    let mut header = vec![PROTEIN_NAME, SAMPLE_ID, "Intensity"];
    if has_condition {
        header.push(CONDITION);
    }
    if norm.is_some() {
        header.push("IntensityNorm");
    }
    write_line(&mut writer, output, &header.join("\t"))?;

    for row in rows {
        let mut fields = vec![
            row.protein.clone(),
            row.sample.clone(),
            format_float(row.intensity),
        ];
        if has_condition {
            fields.push(
                condition_by_sample
                    .get(&row.sample)
                    .cloned()
                    .unwrap_or_else(|| "Empty".to_owned()),
            );
        }
        if let Some(totals) = norm {
            let total = totals.get(row.sample.as_str()).copied().unwrap_or(0.0);
            let normalized = if total > 0.0 {
                row.intensity / total
            } else {
                0.0
            };
            fields.push(format_float(normalized));
        }
        write_line(&mut writer, output, &fields.join("\t"))?;
    }

    flush(&mut writer, output)
}

fn create_writer(output: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|source| MokumeError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    let file = File::create(output).map_err(|source| MokumeError::Io {
        path: output.to_path_buf(),
        source,
    })?;
    Ok(BufWriter::new(file))
}

fn write_line(writer: &mut BufWriter<File>, output: &Path, line: &str) -> Result<()> {
    writeln!(writer, "{line}").map_err(|source| MokumeError::Io {
        path: output.to_path_buf(),
        source,
    })
}

fn flush(writer: &mut BufWriter<File>, output: &Path) -> Result<()> {
    writer.flush().map_err(|source| MokumeError::Io {
        path: output.to_path_buf(),
        source,
    })
}

/// `{}` on `f64` gives Rust's shortest round-trip representation, matching the
/// precision the golden test compares (relative 1e-9).
fn format_float(value: f64) -> String {
    format!("{value}")
}

/// Render an optional extra-column value; absent values become an empty cell
/// (the extras are always populated when their flag is set, so this only
/// guards against a missing post-processing pass).
fn format_optional(value: Option<f64>) -> String {
    value.map(format_float).unwrap_or_default()
}

fn column_index(headers: &[String], column: &str) -> Result<usize> {
    optional_column_index(headers, column).ok_or_else(|| MokumeError::InvalidInput {
        message: format!("could not find required column '{column}' in the peptide file"),
    })
}

fn optional_column_index(headers: &[String], column: &str) -> Option<usize> {
    headers.iter().position(|header| header == column)
}

fn field<'a>(record: &'a csv::StringRecord, index: usize, path: &Path) -> Result<&'a str> {
    record.get(index).ok_or_else(|| MokumeError::InvalidInput {
        message: format!("row in '{}' is missing column {index}", path.display()),
    })
}

fn csv_error(path: &Path, source: csv::Error) -> MokumeError {
    MokumeError::InvalidInput {
        message: format!("error reading file '{}': {source}", path.display()),
    }
}

/// piBAQ post-processing extras: TPA (carried over from the piBAQ core),
/// `normalize_pibaq` (`PiBAQNorm` / `PiBAQLog` / `PiBAQPpb`), and the ProteomicRuler
/// copy-number / moles / weight / concentration columns.
///
/// Every transform is a deterministic function of the piBAQ `NormIntensity` /
/// `PiBAQ` (and, for TPA/ruler, the `MolecularWeight`) the verified core already
/// produced -- the allocation math is never re-touched. Each formula mirrors
/// `mokume.quantification.pibaq` exactly so the output cells match the Python
/// oracle to 1e-9.
mod pibaq_extras {
    use std::collections::HashMap;

    use mokume_core::{MokumeError, Result};
    use serde::Deserialize;

    /// `MolecularWeight` column header (Python `MOLECULARWEIGHT`).
    pub(super) const MOLECULAR_WEIGHT: &str = "MolecularWeight";
    /// `TPA` column header (Python `TPA`).
    pub(super) const TPA: &str = "TPA";
    /// Relative piBAQ column header (Python `PIBAQ_NORMALIZED`).
    pub(super) const PIBAQ_NORM: &str = "PiBAQNorm";
    /// Log-shifted piBAQ column header (Python `PIBAQ_LOG`).
    pub(super) const PIBAQ_LOG: &str = "PiBAQLog";
    /// Parts-per-billion piBAQ column header (Python `PIBAQ_PPB`).
    pub(super) const PIBAQ_PPB: &str = "PiBAQPpb";
    /// Copy-number column header (Python `COPYNUMBER`).
    pub(super) const COPY_NUMBER: &str = "CopyNumber";
    /// Moles column header (Python `MOLES_NMOL`).
    pub(super) const MOLES_NMOL: &str = "Moles[nmol]";
    /// Weight column header (Python `WEIGHT_NG`).
    pub(super) const WEIGHT_NG: &str = "Weight[ng]";
    /// Concentration column header (Python `CONCENTRATION_NM`).
    pub(super) const CONCENTRATION_NM: &str = "Concentration[nM]";

    /// Avogadro's number, the Python `AVAGADRO` constant (kept spelled as the
    /// physical constant; the Python source has the historical misspelling).
    const AVOGADRO: f64 = 6.02214129e23;
    /// Average base-pair mass, the Python `AVERAGE_BASE_PAIR_MASS` constant.
    const AVERAGE_BASE_PAIR_MASS: f64 = 617.96;
    /// PRIDE parts-per-billion multiplier (`* 100_000_000`).
    const PIBAQ_PPB_FACTOR: f64 = 100_000_000.0;
    /// ProteomicsDB log shift (`log10(PiBAQNorm) + 10`).
    const PIBAQ_LOG_SHIFT: f64 = 10.0;
    /// Nanomole-per-mole scale (`1e9 / AVOGADRO`) used for `Moles[nmol]`.
    const NMOL_PER_MOLE: f64 = 1e9;
    /// Nanogram divisor (`/ 1e-9`) used when deriving the ruler volume.
    const WEIGHT_NG_TO_GRAMS: f64 = 1e-9;

    /// Embedded copy of `mokume/data/organisms.json`, parsed at runtime to look
    /// up genome size and histone entries for the proteomic ruler.
    const ORGANISMS_JSON: &str = include_str!("data/organisms.json");

    /// One piBAQ output row, extended with the optional extra columns. The base
    /// columns mirror [`mokume_pipeline::PibaqProteinRow`]; the `Option` fields
    /// are populated by the corresponding post-processing pass.
    #[derive(Debug, Clone)]
    pub(super) struct PibaqExtraRow {
        pub(super) protein: String,
        pub(super) sample: String,
        pub(super) condition: String,
        pub(super) norm_intensity: f64,
        pub(super) pibaq: f64,
        pub(super) family_id: String,
        pub(super) evidence_level: &'static str,
        pub(super) family_size: usize,
        pub(super) molecular_weight: Option<f64>,
        pub(super) tpa: Option<f64>,
        pub(super) pibaq_norm: Option<f64>,
        pub(super) pibaq_log: Option<f64>,
        pub(super) pibaq_ppb: Option<f64>,
        pub(super) copy_number: Option<f64>,
        pub(super) moles_nmol: Option<f64>,
        pub(super) weight_ng: Option<f64>,
        pub(super) concentration_nm: Option<f64>,
    }

    impl PibaqExtraRow {
        /// Lift the piBAQ core rows into extra-aware records, resolving each
        /// sample's condition (defaulting to `Empty`, as the Python loader fills
        /// a missing `Condition`).
        pub(super) fn lift(
            rows: Vec<mokume_pipeline::PibaqProteinRow>,
            condition_by_sample: &HashMap<String, String>,
        ) -> Vec<Self> {
            rows.into_iter()
                .map(|row| {
                    let condition = condition_by_sample
                        .get(&row.sample)
                        .cloned()
                        .unwrap_or_else(|| "Empty".to_owned());
                    Self {
                        protein: row.protein,
                        sample: row.sample,
                        condition,
                        norm_intensity: row.norm_intensity,
                        pibaq: row.pibaq,
                        family_id: row.family_id,
                        evidence_level: row.evidence_level,
                        family_size: row.family_size,
                        molecular_weight: row.molecular_weight,
                        tpa: row.tpa,
                        pibaq_norm: None,
                        pibaq_log: None,
                        pibaq_ppb: None,
                        copy_number: None,
                        moles_nmol: None,
                        weight_ng: None,
                        concentration_nm: None,
                    }
                })
                .collect()
        }
    }

    /// `normalize_pibaq`: per (SampleID, Condition) group, divide each protein's
    /// piBAQ by the group's total piBAQ, then derive the ProteomicsDB log
    /// (`10 + log10(PiBAQNorm)` where positive, else 0) and the PRIDE parts-per-
    /// billion (`PiBAQNorm * 1e8`). Mirrors `mokume.quantification.pibaq.normalize_pibaq`.
    pub(super) fn normalize_pibaq(rows: &mut [PibaqExtraRow]) {
        let mut totals: HashMap<(&str, &str), f64> = HashMap::new();
        for row in rows.iter() {
            *totals
                .entry((row.sample.as_str(), row.condition.as_str()))
                .or_insert(0.0) += row.pibaq;
        }
        // Snapshot the totals so the borrow ends before the mutable pass.
        let totals: HashMap<(String, String), f64> = totals
            .into_iter()
            .map(|((sample, condition), total)| ((sample.to_owned(), condition.to_owned()), total))
            .collect();
        for row in rows.iter_mut() {
            let total = totals
                .get(&(row.sample.clone(), row.condition.clone()))
                .copied()
                .unwrap_or(0.0);
            let norm = row.pibaq / total;
            row.pibaq_norm = Some(norm);
            row.pibaq_log = Some(if norm > 0.0 {
                norm.log10() + PIBAQ_LOG_SHIFT
            } else {
                0.0
            });
            row.pibaq_ppb = Some(norm * PIBAQ_PPB_FACTOR);
        }
    }

    /// ProteomicRuler description for a single organism, deserialized from the
    /// embedded `organisms.json`. Only the fields the ruler reads are kept.
    #[derive(Debug, Clone, Deserialize)]
    pub(super) struct Organism {
        name: String,
        genome_size: f64,
        #[serde(default)]
        histone_entries: Vec<String>,
    }

    /// Resolve an organism name case-insensitively, matching the Python
    /// `OrganismDescription.get(name.upper())`. Returns an error for an unknown
    /// organism, mirroring the Python `KeyError`.
    pub(super) fn resolve_organism(name: &str) -> Result<Organism> {
        let registry: HashMap<String, Organism> =
            serde_json::from_str(ORGANISMS_JSON).map_err(|source| MokumeError::InvalidInput {
                message: format!("could not parse the embedded organism table: {source}"),
            })?;
        let key = name.to_ascii_uppercase();
        if let Some(organism) = registry.get(&key) {
            return Ok(organism.clone());
        }
        // The registry is keyed by the uppercased `name` field in Python; the
        // JSON keys already match, but fall back to a name scan for robustness.
        registry
            .into_values()
            .find(|organism| organism.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| MokumeError::InvalidInput {
                message: format!("could not resolve organism description for {name}"),
            })
    }

    /// Apply the proteomic ruler per Condition group, mirroring
    /// `ConcentrationWeightByProteomicRuler.apply_by_condition`. For each
    /// condition independently: histone intensity = max(sum of `NormIntensity`
    /// over histone proteins, 1.0); then per protein
    /// `CopyNumber = NormIntensity / histone * dna_mass * AVOGADRO / MW`,
    /// `Moles[nmol] = CopyNumber * 1e9 / AVOGADRO`,
    /// `Weight[ng] = Moles[nmol] * MW`; finally the group volume
    /// `= sum(Weight[ng]) / 1e-9 / cpc` and
    /// `Concentration[nM] = volume * Moles[nmol]`.
    pub(super) fn apply_ruler(
        rows: &mut [PibaqExtraRow],
        organism: &Organism,
        ploidy: i32,
        cpc: f64,
    ) {
        let dna_mass = f64::from(ploidy) * organism.genome_size * AVERAGE_BASE_PAIR_MASS / AVOGADRO;
        let histones: std::collections::HashSet<&str> = organism
            .histone_entries
            .iter()
            .map(String::as_str)
            .collect();

        // Group row indices by condition so each group's histone intensity and
        // volume are computed over exactly that condition's proteins.
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, row) in rows.iter().enumerate() {
            groups.entry(row.condition.clone()).or_default().push(index);
        }

        for indices in groups.values() {
            let histone_intensity = indices
                .iter()
                .filter(|&&index| histones.contains(rows[index].protein.as_str()))
                .map(|&index| rows[index].norm_intensity)
                .sum::<f64>()
                .max(1.0);

            // First pass: copy number, moles, weight; accumulate the group's
            // total weight for the volume term.
            let mut total_weight = 0.0;
            for &index in indices {
                let row = &mut rows[index];
                let mw = row.molecular_weight.unwrap_or(1.0);
                let copy_number = row.norm_intensity / histone_intensity * dna_mass * AVOGADRO / mw;
                let moles = copy_number * (NMOL_PER_MOLE / AVOGADRO);
                let weight = moles * mw;
                row.copy_number = Some(copy_number);
                row.moles_nmol = Some(moles);
                row.weight_ng = Some(weight);
                total_weight += weight;
            }

            // Second pass: the group volume depends on the summed weight, so the
            // concentration is derived only once every weight is known.
            let volume = total_weight / WEIGHT_NG_TO_GRAMS / cpc;
            for &index in indices {
                let row = &mut rows[index];
                let moles = row.moles_nmol.unwrap_or(0.0);
                row.concentration_nm = Some(volume * moles);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::time::{SystemTime, UNIX_EPOCH};

    // `super::*` brings the crate's single-parameter `Result` alias into scope,
    // so the test helpers spell out the two-parameter standard `Result`.
    type TestResult<T> = std::result::Result<T, Box<dyn Error>>;

    fn temp_dir(tag: &str) -> TestResult<(tempfile::TempDir, std::path::PathBuf)> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = tempfile::Builder::new()
            .prefix(&format!("mokume-peptides2protein-{tag}-{nanos}-"))
            .tempdir()?;
        let path = directory.path().to_path_buf();
        Ok((directory, path))
    }

    fn write_file(path: &Path, contents: &str) -> TestResult<()> {
        let mut file = File::create(path)?;
        file.write_all(contents.as_bytes())?;
        Ok(())
    }

    // Synthetic peptide matrix shared by the golden tests. Each
    // (protein, sample, peptide) is unique so the Python generic group-bys and
    // the Rust path coincide exactly. P3's `THIDPECK` is intentionally absent
    // from the FASTA digest (the digest yields `ATHIDPECK`), exercising the
    // FASTA-driven peptide filtering on the piBAQ path.
    const PEPTIDES_CSV: &str = "ProteinName,PeptideCanonical,SampleID,Condition,NormIntensity\n\
P1,PEPTIDEAK,S1,A,100.0\n\
P1,APEPTIDECK,S1,A,300.0\n\
P1,ASHAEDPEPK,S1,A,50.0\n\
P1,PEPTIDEAK,S2,B,40.0\n\
P1,APEPTIDECK,S2,B,60.0\n\
P2,ALYAAEK,S1,A,500.0\n\
P3,THIDPEAK,S1,A,700.0\n\
P3,THIDPECK,S1,A,900.0\n";

    const PROTEOME_FASTA: &str =
        ">P1\nPEPTIDEAKAPEPTIDECKASHAEDPEPK\n>P2\nALYAAEK\n>P3\nTHIDPEAKATHIDPECK\n";

    fn test_pibaq_digest() -> PibaqDigest {
        PibaqDigest {
            accession_peptides: HashMap::from([
                (
                    "P1".to_owned(),
                    ["PEPTIDEAK", "APEPTIDECK", "ASHAEDPEPK"]
                        .map(str::to_owned)
                        .into_iter()
                        .collect(),
                ),
                (
                    "P2".to_owned(),
                    ["ALYAAEK"].map(str::to_owned).into_iter().collect(),
                ),
                (
                    "P3".to_owned(),
                    ["THIDPEAK", "ATHIDPECK"]
                        .map(str::to_owned)
                        .into_iter()
                        .collect(),
                ),
            ]),
            provenance: mokume_pipeline::PibaqDigestProvenance {
                pyopenms_version: "test".to_owned(),
                enzyme: "Trypsin".to_owned(),
                catalog_hash: "test".to_owned(),
                min_aa: 7,
                max_aa: 30,
                missed_cleavages: 0,
            },
        }
    }

    fn base_args(peptides: &Path, output: &Path) -> Peptides2ProteinArgs {
        Peptides2ProteinArgs {
            fasta: None,
            peptides: peptides.to_path_buf(),
            quant_method: "pibaq".to_owned(),
            enzyme: None,
            normalize: false,
            min_aa: None,
            max_aa: None,
            tpa: false,
            ruler: false,
            ploidy: None,
            organism: None,
            cpc: None,
            output: Some(output.to_path_buf()),
            qc_report: None,
            threads: None,
            directlfq_min_nonan: None,
            maxlfq_min_ratio_count: None,
            stabilize: false,
            families_yaml: None,
            min_shared: None,
            min_anchors: None,
            high_anchor_threshold: None,
        }
    }

    fn read_table(path: &Path) -> TestResult<(Vec<String>, Vec<Vec<String>>)> {
        let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
        let headers = reader.headers()?.iter().map(ToOwned::to_owned).collect();
        let rows = reader
            .records()
            .map(|record| record.map(|record| record.iter().map(ToOwned::to_owned).collect()))
            .collect::<std::result::Result<Vec<Vec<String>>, _>>()?;
        Ok((headers, rows))
    }

    /// Look up the cell value at (protein, sample, column), returning an error
    /// (never panicking via unwrap/expect) when the row or column is absent.
    fn cell(
        headers: &[String],
        rows: &[Vec<String>],
        protein: &str,
        sample: &str,
        column: &str,
    ) -> TestResult<String> {
        let position = |name: &str| {
            headers
                .iter()
                .position(|header| header == name)
                .ok_or_else(|| -> Box<dyn Error> { format!("column '{name}' is missing").into() })
        };
        let protein_idx = position(PROTEIN_NAME)?;
        let sample_idx = position(SAMPLE_ID)?;
        let column_idx = position(column)?;
        let row = rows
            .iter()
            .find(|row| {
                row.get(protein_idx).is_some_and(|p| p == protein)
                    && row.get(sample_idx).is_some_and(|s| s == sample)
            })
            .ok_or_else(|| -> Box<dyn Error> {
                format!("row ({protein}, {sample}) is missing").into()
            })?;
        row.get(column_idx)
            .cloned()
            .ok_or_else(|| format!("cell ({protein}, {sample}, {column}) is missing").into())
    }

    /// Assert the numeric value at (protein, sample, column) is within 1e-9 of
    /// `expected`, without unwrap/expect (forbidden by the lint policy).
    fn assert_cell_close(
        headers: &[String],
        rows: &[Vec<String>],
        protein: &str,
        sample: &str,
        column: &str,
        expected: f64,
    ) -> TestResult<()> {
        let value = cell(headers, rows, protein, sample, column)?;
        let got = value
            .parse::<f64>()
            .map_err(|_| -> Box<dyn Error> { format!("'{value}' is not numeric").into() })?;
        assert!(
            (got - expected).abs() <= 1e-9,
            "({protein}, {sample}, {column}): got {got}, expected {expected}"
        );
        Ok(())
    }

    fn assert_threshold_evidence(
        low_headers: &[String],
        low_rows: &[Vec<String>],
        high_headers: &[String],
        high_rows: &[Vec<String>],
    ) -> TestResult<()> {
        for (headers, rows, protein, expected) in [
            (low_headers, low_rows, "P1", "high"),
            (low_headers, low_rows, "P2", "high"),
            (high_headers, high_rows, "P1", "medium"),
            (high_headers, high_rows, "P2", "medium"),
        ] {
            assert_eq!(
                cell(headers, rows, protein, "S1", "EvidenceLevel")?,
                expected
            );
        }
        Ok(())
    }

    fn assert_threshold_pibaq(
        low_headers: &[String],
        low_rows: &[Vec<String>],
        high_headers: &[String],
        high_rows: &[Vec<String>],
    ) -> TestResult<()> {
        for (protein, sample, expected) in [
            ("P1", "S1", 150.0),
            ("P2", "S1", 500.0),
            ("P3", "S1", 350.0),
        ] {
            for (headers, rows) in [(low_headers, low_rows), (high_headers, high_rows)] {
                assert_cell_close(headers, rows, protein, sample, "PiBAQ", expected)?;
            }
        }
        Ok(())
    }

    fn assert_tpa_oracle(headers: &[String], rows: &[Vec<String>]) -> TestResult<()> {
        assert!(headers.contains(&"MolecularWeight".to_owned()));
        assert!(headers.contains(&"TPA".to_owned()));
        for (protein, sample, column, expected) in [
            ("P1", "S1", "MolecularWeight", 3143.460508429001),
            ("P2", "S1", "MolecularWeight", 764.4068587863999),
            ("P1", "S1", "TPA", 450.0 / 3143.460508429001),
            ("P1", "S2", "TPA", 100.0 / 3143.460508429001),
            ("P2", "S1", "TPA", 500.0 / 764.4068587863999),
            ("P3", "S1", "TPA", 700.0 / 1903.9098218451002),
        ] {
            assert_cell_close(headers, rows, protein, sample, column, expected)?;
        }
        Ok(())
    }

    fn assert_ruler_oracle(headers: &[String], rows: &[Vec<String>]) -> TestResult<()> {
        for column in [
            "CopyNumber",
            "Moles[nmol]",
            "Weight[ng]",
            "Concentration[nM]",
        ] {
            assert!(headers.contains(&column.to_owned()), "missing {column}");
        }
        for (protein, sample, column, expected) in [
            ("P1", "S1", "CopyNumber", 569705926063.9502),
            ("P2", "S1", "CopyNumber", 2603104848063.672),
            ("P3", "S1", "CopyNumber", 1463180476321.2395),
            ("P1", "S2", "CopyNumber", 126601316903.10007),
            ("P1", "S1", "Moles[nmol]", 0.0009460188637718765),
            ("P1", "S1", "Weight[ng]", 2.9737729384957685),
            ("P1", "S1", "Concentration[nM]", 51576.16376717422),
            ("P2", "S1", "Concentration[nM]", 235662.2176540015),
            ("P3", "S1", "Concentration[nM]", 132463.49110155678),
            ("P1", "S2", "Concentration[nM]", 694.6284682447709),
        ] {
            assert_cell_close(headers, rows, protein, sample, column, expected)?;
        }
        Ok(())
    }

    // Golden oracle (Python), captured with:
    //   conda run -n Bigbio python -m mokume.mokume_cli peptides2protein \
    //     --method sum -p peptides.csv -o out.tsv
    // Columns: ProteinName SampleID Intensity Condition.
    #[test]
    fn peptides2protein_sum_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("sum")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "sum".to_owned();
        run_peptides_to_protein_with_digest(&args, None)?;

        let (headers, rows) = read_table(&output)?;
        assert_eq!(
            headers,
            vec!["ProteinName", "SampleID", "Intensity", "Condition"]
        );
        assert_cell_close(&headers, &rows, "P1", "S1", "Intensity", 450.0)?;
        assert_cell_close(&headers, &rows, "P1", "S2", "Intensity", 100.0)?;
        assert_cell_close(&headers, &rows, "P2", "S1", "Intensity", 500.0)?;
        assert_cell_close(&headers, &rows, "P3", "S1", "Intensity", 1600.0)?;
        assert_eq!(cell(&headers, &rows, "P1", "S2", "Condition")?, "B");
        Ok(())
    }

    /// Build the parquet form of [`PEPTIDES_CSV`] via the shared
    /// `write_peptide_parquet` schema (dictionary-encoded `SampleID`/`Condition`,
    /// float32 `NormIntensity`), so the parity test exercises the real reader.
    fn peptide_parquet_rows_from_csv(csv: &str) -> TestResult<Vec<mokume_io::PeptideParquetRow>> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .from_reader(csv.as_bytes());
        let mut rows = Vec::new();
        for record in reader.records() {
            let record = record?;
            let get = |index: usize| -> TestResult<String> {
                record
                    .get(index)
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| -> Box<dyn Error> { format!("missing field {index}").into() })
            };
            let intensity = get(4)?
                .parse::<f64>()
                .map_err(|_| -> Box<dyn Error> { "non-numeric intensity".into() })?;
            rows.push(mokume_io::PeptideParquetRow {
                protein_name: get(0)?,
                peptide_canonical: get(1)?,
                sample_id: get(2)?,
                bio_replicate: 1,
                condition: get(3)?,
                run: None,
                tech_replicate: None,
                norm_intensity: intensity,
            });
        }
        Ok(rows)
    }

    // The parquet and CSV loaders must produce byte-identical protein output for
    // the same data (Python reads either via `pd.read_parquet`/`pd.read_csv`).
    #[test]
    fn peptides2protein_parquet_input_matches_csv_input() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("parquet-parity")?;
        let csv_path = dir.join("peptides.csv");
        let parquet_path = dir.join("peptides.parquet");
        write_file(&csv_path, PEPTIDES_CSV)?;
        let parquet_rows = peptide_parquet_rows_from_csv(PEPTIDES_CSV)?;
        mokume_io::write_peptide_parquet(&parquet_path, &parquet_rows, false)?;

        let csv_out = dir.join("from_csv.tsv");
        let parquet_out = dir.join("from_parquet.tsv");

        let mut csv_args = base_args(&csv_path, &csv_out);
        csv_args.quant_method = "sum".to_owned();
        run_peptides_to_protein_with_digest(&csv_args, None)?;

        let mut parquet_args = base_args(&parquet_path, &parquet_out);
        parquet_args.quant_method = "sum".to_owned();
        run_peptides_to_protein_with_digest(&parquet_args, None)?;

        let (csv_headers, csv_rows) = read_table(&csv_out)?;
        let (parquet_headers, parquet_rows_out) = read_table(&parquet_out)?;
        assert_eq!(csv_headers, parquet_headers);
        assert_eq!(csv_rows, parquet_rows_out);
        Ok(())
    }

    // Oracle:
    //   ... quantify peptides2protein --quant-method top2 -p peptides.csv -o out.tsv
    // (the oracle predates the rename and was captured as `--method topn
    // --topn_n 2`; N moved into the method name, the arithmetic did not change)
    #[test]
    fn peptides2protein_topn_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("topn")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "top2".to_owned();
        run_peptides_to_protein_with_digest(&args, None)?;

        let (headers, rows) = read_table(&output)?;
        assert_cell_close(&headers, &rows, "P1", "S1", "Intensity", 200.0)?;
        assert_cell_close(&headers, &rows, "P1", "S2", "Intensity", 50.0)?;
        assert_cell_close(&headers, &rows, "P2", "S1", "Intensity", 500.0)?;
        assert_cell_close(&headers, &rows, "P3", "S1", "Intensity", 800.0)?;
        Ok(())
    }

    // Oracle:
    //   ... quantify peptides2protein --quant-method top3 -p peptides.csv -o out.tsv
    #[test]
    fn peptides2protein_top3_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("top3")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "top3".to_owned();
        run_peptides_to_protein_with_digest(&args, None)?;

        let (headers, rows) = read_table(&output)?;
        assert_cell_close(&headers, &rows, "P1", "S1", "Intensity", 150.0)?;
        assert_cell_close(&headers, &rows, "P3", "S1", "Intensity", 800.0)?;
        Ok(())
    }

    // Oracle:
    //   ... quantify peptides2protein --quant-method sum --normalize -p peptides.csv -o out.tsv
    // Adds IntensityNorm = Intensity / sum(Intensity per SampleID).
    #[test]
    fn peptides2protein_sum_normalize_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("sumnorm")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "sum".to_owned();
        args.normalize = true;
        run_peptides_to_protein_with_digest(&args, None)?;

        let (headers, rows) = read_table(&output)?;
        assert!(headers.contains(&"IntensityNorm".to_owned()));
        // S1 total = 450 + 500 + 1600 = 2550; S2 total = 100.
        assert_cell_close(&headers, &rows, "P1", "S1", "IntensityNorm", 450.0 / 2550.0)?;
        assert_cell_close(
            &headers,
            &rows,
            "P3",
            "S1",
            "IntensityNorm",
            1600.0 / 2550.0,
        )?;
        assert_cell_close(&headers, &rows, "P1", "S2", "IntensityNorm", 1.0)?;
        Ok(())
    }

    // Oracle:
    //   ... quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv -o out.tsv
    // Columns: ProteinName SampleID Condition NormIntensity PiBAQ FamilyId EvidenceLevel FamilySize.
    #[test]
    fn peptides2protein_pibaq_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("pibaq")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "pibaq".to_owned();
        args.fasta = Some(fasta);
        run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;

        let (headers, rows) = read_table(&output)?;
        assert_eq!(
            headers,
            vec![
                "ProteinName",
                "SampleID",
                "Condition",
                "NormIntensity",
                "PiBAQ",
                "FamilyId",
                "EvidenceLevel",
                "FamilySize",
            ]
        );
        // P1/S1: NormIntensity 450, PiBAQ 450/3 = 150 (3 proteotypic peptides).
        assert_cell_close(&headers, &rows, "P1", "S1", "NormIntensity", 450.0)?;
        assert_cell_close(&headers, &rows, "P1", "S1", "PiBAQ", 150.0)?;
        assert_cell_close(&headers, &rows, "P1", "S2", "PiBAQ", 100.0 / 3.0)?;
        assert_cell_close(&headers, &rows, "P2", "S1", "PiBAQ", 500.0)?;
        // P3/S1: only THIDPEAK is in the digest, so NormIntensity 700, PiBAQ 350.
        assert_cell_close(&headers, &rows, "P3", "S1", "NormIntensity", 700.0)?;
        assert_cell_close(&headers, &rows, "P3", "S1", "PiBAQ", 350.0)?;
        assert_eq!(cell(&headers, &rows, "P1", "S1", "EvidenceLevel")?, "high");
        assert_eq!(
            cell(&headers, &rows, "P2", "S1", "EvidenceLevel")?,
            "medium"
        );
        Ok(())
    }

    // `--high-anchor-threshold` only re-buckets the `EvidenceLevel` annotation
    // (Python `_classify_evidence`); it must never change the `PiBAQ` values.
    // P1 has 3 anchors, P2 has 1 (min_anchors stays 1). Lowering the threshold to
    // 1 promotes P2 medium -> high; raising it to 4 demotes P1 high -> medium.
    #[test]
    fn peptides2protein_pibaq_high_anchor_threshold_only_changes_evidence() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("pibaq-threshold")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let run = |threshold: usize, out: &Path| -> TestResult<()> {
            let mut args = base_args(&peptides, out);
            args.quant_method = "pibaq".to_owned();
            args.fasta = Some(fasta.clone());
            args.high_anchor_threshold = Some(threshold);
            run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;
            Ok(())
        };

        let low_out = dir.join("low.tsv");
        let high_out = dir.join("high.tsv");
        run(1, &low_out)?;
        run(4, &high_out)?;

        let (low_h, low_rows) = read_table(&low_out)?;
        let (high_h, high_rows) = read_table(&high_out)?;

        assert_threshold_evidence(&low_h, &low_rows, &high_h, &high_rows)?;
        assert_threshold_pibaq(&low_h, &low_rows, &high_h, &high_rows)?;
        Ok(())
    }

    #[test]
    fn peptides2protein_pibaq_requires_fasta() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("nofasta")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;

        let args = base_args(&peptides, &output);
        let Err(error) = run_peptides_to_protein_with_digest(&args, None) else {
            panic!("piBAQ without --fasta must fail");
        };
        assert!(matches!(error, MokumeError::InvalidInput { .. }));
        Ok(())
    }

    // Oracle (TPA columns MolecularWeight + TPA appended):
    //   ... quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv \
    //       --tpa -o out.tsv
    // MolecularWeight is the pyOpenMS getMonoWeight of the canonical protein;
    // TPA = NormIntensity / MolecularWeight.
    #[test]
    fn peptides2protein_pibaq_tpa_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("pibaqtpa")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.fasta = Some(fasta);
        args.tpa = true;
        run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;

        let (headers, rows) = read_table(&output)?;
        assert_tpa_oracle(&headers, &rows)?;
        Ok(())
    }

    // Oracle (normalize_pibaq columns PiBAQNorm + PiBAQLog + PiBAQPpb appended):
    //   ... quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv \
    //       -n -o out.tsv
    // PiBAQNorm = PiBAQ / sum(PiBAQ per SampleID,Condition);
    // PiBAQLog = 10 + log10(PiBAQNorm); PiBAQPpb = PiBAQNorm * 1e8.
    #[test]
    fn peptides2protein_pibaq_normalize_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("pibaqnorm")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.fasta = Some(fasta);
        args.normalize = true;
        run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;

        let (headers, rows) = read_table(&output)?;
        assert!(headers.contains(&"PiBAQNorm".to_owned()));
        assert!(headers.contains(&"PiBAQLog".to_owned()));
        assert!(headers.contains(&"PiBAQPpb".to_owned()));
        // Condition A total PiBAQ = 150 + 500 + 350 = 1000; Condition B = 33.333...
        assert_cell_close(&headers, &rows, "P1", "S1", "PiBAQNorm", 0.15)?;
        assert_cell_close(&headers, &rows, "P2", "S1", "PiBAQNorm", 0.5)?;
        assert_cell_close(&headers, &rows, "P3", "S1", "PiBAQNorm", 0.35)?;
        assert_cell_close(&headers, &rows, "P1", "S2", "PiBAQNorm", 1.0)?;
        assert_cell_close(
            &headers,
            &rows,
            "P1",
            "S1",
            "PiBAQLog",
            0.15_f64.log10() + 10.0,
        )?;
        assert_cell_close(&headers, &rows, "P1", "S2", "PiBAQLog", 10.0)?;
        assert_cell_close(&headers, &rows, "P2", "S1", "PiBAQPpb", 50_000_000.0)?;
        assert_cell_close(&headers, &rows, "P1", "S2", "PiBAQPpb", 100_000_000.0)?;
        Ok(())
    }

    // Oracle (ProteomicRuler columns; ruler requires --tpa):
    //   ... quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv \
    //       --tpa --ruler --organism human --ploidy 2 --cpc 200 -o out.tsv
    // P1/P2/P3 are not human histones, so histone_intensity = max(0, 1) = 1.
    // dna_mass = 2 * 3.22e9 * 617.96 / 6.02214129e23.
    #[test]
    fn peptides2protein_pibaq_ruler_matches_python_oracle() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("pibaqruler")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.fasta = Some(fasta);
        args.tpa = true;
        args.ruler = true;
        args.organism = Some("human".to_owned());
        args.ploidy = Some(2);
        args.cpc = Some(200.0);
        run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;

        let (headers, rows) = read_table(&output)?;
        assert_ruler_oracle(&headers, &rows)?;
        Ok(())
    }

    // The proteomic ruler requires --tpa (mirrors the Python guard); without it
    // the command must fail with an InvalidInput, not silently skip the ruler.
    #[test]
    fn peptides2protein_ruler_requires_tpa() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("rulernotpa")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.fasta = Some(fasta);
        args.ruler = true;
        assert!(matches!(
            run_peptides_to_protein_with_digest(&args, None),
            Err(MokumeError::InvalidInput { .. })
        ));
        Ok(())
    }

    // An unknown organism must fail (mirrors the Python KeyError), even when the
    // ruler is not requested -- the organism is always resolved on the piBAQ path.
    #[test]
    fn peptides2protein_rejects_unknown_organism() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("badorg")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.fasta = Some(fasta);
        args.organism = Some("martian".to_owned());
        assert!(matches!(
            run_peptides_to_protein_with_digest(&args, None),
            Err(MokumeError::InvalidInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn peptides2protein_maxlfq_preserves_species_and_ratio_count() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("maxlfq-species")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("proteins.tsv");
        write_file(
            &peptides,
            concat!(
            "ProteinName,PeptideCanonical,PeptideSequence,PrecursorCharge,SampleID,NormIntensity\n",
            "P,PEPTIDEAK,PEPTIDEAK,2,S1,1\nP,PEPTIDEAK,PEPTIDEAK,3,S1,1\n",
            "P,ANOTHERAK,ANOTHERAK,2,S1,1\nP,PEPTIDEAK,PEPTIDEAK,2,S2,1\n",
            "P,PEPTIDEAK,PEPTIDEAK,3,S2,9\nP,ANOTHERAK,ANOTHERAK,2,S2,3\n",
            "P,INVALID,INVALID,NaN,S1,1000\nP,INVALID,INVALID,NaN,S2,1000\n",
            "P,INVALID,INVALID,0,S1,1000\nP,INVALID,INVALID,0,S2,1000\n",
        ),
        )?;
        let mut args = base_args(&peptides, &output);
        args.quant_method = "maxlfq".to_owned();
        args.threads = Some(1);
        run_peptides_to_protein_with_digest(&args, None)?;
        let (_, rows) = read_table(&output)?;
        assert_eq!(rows.len(), 2);
        for (row, expected) in rows.iter().zip([4.0_f64, 12.0]) {
            assert!((row[2].parse::<f64>()? - expected).abs() < 1e-12);
        }
        args.maxlfq_min_ratio_count = Some(4);
        run_peptides_to_protein_with_digest(&args, None)?;
        assert!(read_table(&output)?.1.is_empty());
        Ok(())
    }

    #[test]
    fn peptides2protein_stabilization_changes_only_maxlfq_low_overlap() -> TestResult<()> {
        let (_tempdir, dir) = temp_dir("maxlfq-stabilize")?;
        let peptides = dir.join("peptides.csv");
        let output = dir.join("proteins.tsv");
        let mut csv = String::from("ProteinName,PeptideCanonical,SampleID,NormIntensity\n");
        for i in 0..20 {
            csv.push_str(&format!("P,PEPTIDE{i},S1,10\n"));
            if i < 2 {
                csv.push_str(&format!("P,PEPTIDE{i},S2,1\n"));
            }
        }
        write_file(&peptides, &csv)?;
        let mut args = base_args(&peptides, &output);
        args.quant_method = "maxlfq".to_owned();
        args.threads = Some(1);
        for (enabled, ratio) in [(false, 10.0), (true, 100.0)] {
            args.stabilize = enabled;
            run_peptides_to_protein_with_digest(&args, None)?;
            let (_, rows) = read_table(&output)?;
            let actual = rows[0][2].parse::<f64>()? / rows[1][2].parse::<f64>()?;
            assert!((actual - ratio).abs() < 1e-10);
        }
        for method in ["directlfq", "sum", "pibaq", "top3"] {
            args.quant_method = method.to_owned();
            args.threads = None;
            let Err(error) = run_peptides_to_protein_with_digest(&args, None) else {
                panic!("non-MaxLFQ method accepted --stabilize");
            };
            assert!(
                error
                    .to_string()
                    .contains("--stabilize requires --quant-method maxlfq"),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn peptides2protein_runs_lfq_methods() -> TestResult<()> {
        // Both solvers emit long-format intensities deterministically across
        // worker counts, while retaining their distinct algorithms.
        let (_tempdir, dir) = temp_dir("lfq")?;
        let peptides = dir.join("peptides.csv");
        write_file(&peptides, PEPTIDES_CSV)?;

        for method in ["maxlfq", "directlfq"] {
            let mut one_thread_output = None;
            for threads in [1, 4] {
                let output = dir.join(format!("{method}-{threads}.tsv"));
                let mut args = base_args(&peptides, &output);
                args.quant_method = method.to_owned();
                args.threads = Some(threads);
                run_peptides_to_protein_with_digest(&args, None)?;

                let table = read_table(&output)?;
                assert_eq!(table.0.first().map(String::as_str), Some(PROTEIN_NAME));
                assert_eq!(table.0.get(1).map(String::as_str), Some(SAMPLE_ID));
                assert_eq!(table.0.get(2).map(String::as_str), Some("Intensity"));
                assert!(!table.1.is_empty(), "{method} produced rows");
                if let Some(expected) = &one_thread_output {
                    assert_eq!(&table, expected, "{method} changed with thread count");
                } else {
                    one_thread_output = Some(table);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn peptides2protein_kernel_leaves_verbose_qc_to_python_wrapper() -> TestResult<()> {
        // The native kernel writes the data table; the Python console wrapper renders
        // the optional PDF from those exact values after a successful native run.
        let (_tempdir, dir) = temp_dir("verbose")?;
        let peptides = dir.join("peptides.csv");
        let fasta = dir.join("proteome.fasta");
        let output = dir.join("out.tsv");
        let qc = dir.join("qc.pdf");
        write_file(&peptides, PEPTIDES_CSV)?;
        write_file(&fasta, PROTEOME_FASTA)?;

        let mut args = base_args(&peptides, &output);
        args.quant_method = "pibaq".to_owned();
        args.fasta = Some(fasta);
        args.qc_report = Some(qc.clone());
        run_peptides_to_protein_with_digest(&args, Some(test_pibaq_digest()))?;

        // The protein table is written and carries the piBAQ column.
        let (headers, rows) = read_table(&output)?;
        assert!(headers.iter().any(|header| header == "PiBAQ"));
        assert!(!rows.is_empty(), "the protein table must contain rows");

        // No QC PDF is produced; that now lives in the Python wheel.
        assert!(
            !qc.exists(),
            "no QC PDF must be written by the Rust compute path"
        );
        Ok(())
    }
}
