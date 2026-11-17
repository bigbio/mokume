use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{File, create_dir_all, read_to_string};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, sync_channel};

use csv::WriterBuilder;
use mokume_core::{
    BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, FeatureToPeptidesConfig,
    FeatureToProteinsConfig, FilterConfig, ImputationConfig, InputConfig, IntensityFilterConfig,
    IrsChannelConfig, IrsConfig, IrsMissingReference, IrsScope, IrsStat, MaxLfqConfig, MokumeError,
    NamedScoreFilterConfig, NormalizationConfig, OutputConfig, OutputFormat, PeptideId,
    PibaqConfig, PibaqFamilyRows, PibaqSharedMode, PreprocessingFilterConfig, ProteinId,
    QuantMethod, RatioConfig, Result, RunQcFilterConfig, RuntimeConfig, SampleId, StringIdRegistry,
    TmtConfig, TransferredFeatures,
};
use mokume_imputation::imputed_values;
use mokume_io::{
    MsstatsReader, PeptideParquetRow, ProteinGroupOrders, QpxFeatureRecord, QpxParquetReader,
    QpxScoreValue, SdrfLookupCache, SdrfRawTable, SdrfRecord, SdrfTable, write_peptide_parquet,
};
use mokume_normalization::{
    RunCellKey, RunNormalizationMethod, RunNormalizationTransform, SampleNormalizationMethod,
    condition_median_sample_factors, coverage_filtered_proteins, global_median_sample_factors,
    irs_scaling_factors, mean_positive, median, median_finite, parse_run_normalization_method,
    parse_sample_normalization_method, quantile_linear, run_normalization_transforms,
    tmm_norm_factors,
};
use mokume_quant::{
    DirectLfqIon, DirectLfqNormalizedIon, DirectLfqResult, DirectLfqRetain, PeptideMeasurement,
    ProfileResiduals, cell_precisions, direct_lfq_aligned, direct_lfq_aligned_with,
    profile_residuals, solve_max_lfq_with_stabilization,
};
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use regex::{Regex, RegexBuilder};
use tracing::{info, warn};

mod de;
pub mod filters;
mod matrix;
mod maxlfq;
mod memory;
mod spectral_count;
mod threading;
mod tmt;

mod accessions;
mod aggregation;
mod exports;
mod features_to_peptides;
mod features_to_proteins;
mod from_peptides;
mod intensity_filters;
mod irs;
mod normalization_factors;
mod peptide_normalization;
mod pibaq;
mod plex;
mod protein_matrix;
mod qpx_stream;
mod ratio;
mod run_qc;
mod solvers;
mod state;
mod validation;

use crate::accessions::{is_contaminant, matches_sql_contaminant, parse_protein_accession};
use crate::peptide_normalization::apply_dataset_norm_to_peptide_cells;
use crate::peptide_normalization::hierarchical::hierarchical_sample_shifts;
use crate::peptide_normalization::lfq_traces::quantile_normalized_assignments;
use crate::protein_matrix::ProteinMatrix;
use crate::protein_matrix::covariates::{SDRF_SAMPLE_COLUMNS, encode_covariate, match_sdrf_column};
use crate::state::CellKey;
use crate::validation::validate_imputation_config;
pub use features_to_peptides::run_features_to_peptides;
pub use features_to_proteins::{
    run_features_to_proteins, run_features_to_proteins_with_pibaq_digest,
};
pub use from_peptides::{
    LfqPeptideObservation, LfqProteinIntensity, PeptideObservation, PibaqDigest,
    PibaqDigestProvenance, PibaqFromPeptidesParams, PibaqProteinRow, run_lfq_from_peptides,
    run_lfq_from_peptides_with_threads, run_pibaq_from_mapping, run_pibaq_from_peptides,
};
pub use irs::resolve_irs_autodetect_channel;

use memory::MemoryPlan;

pub use de::{MatrixDeInputs, MatrixDifferentialExpressionResults, differential_expression_matrix};
pub use matrix::{impute_matrix, normalize_matrix};
pub use maxlfq::run_maxlfq_from_peptides_with_threads;

const DEFAULT_REFERENCE_REGEX: &str = "pool|powder|ref|reference|bridge";
const MIN_SAMPLE_CORRELATION_OVERLAP: usize = 3;

fn format_float(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        String::new()
    }
}

fn create_parent_dir(path: &Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    create_dir_all(parent).map_err(|source| MokumeError::Io {
        path: parent.to_path_buf(),
        source,
    })
}

fn csv_error(path: &Path, source: csv::Error) -> MokumeError {
    invalid_input(format!("failed to write `{}`: {source}", path.display()))
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod contaminant_and_plex_tests;
