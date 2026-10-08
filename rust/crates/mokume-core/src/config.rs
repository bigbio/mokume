use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::QuantMethod;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureToProteinsConfig {
    pub input: InputConfig,
    pub output: OutputConfig,
    pub filtering: FilterConfig,
    pub normalization: NormalizationConfig,
    pub quantification: QuantMethod,
    pub topn_peptides: usize,
    pub maxlfq: MaxLfqConfig,
    pub pibaq: PibaqConfig,
    pub directlfq: DirectLfqConfig,
    pub batch: BatchCorrectionConfig,
    pub irs: IrsConfig,
    pub coverage_threshold: Option<f64>,
    /// Minimum mean pairwise Pearson correlation to same-condition peers,
    /// computed on pairwise-complete log2 protein intensities.
    #[serde(default)]
    pub sample_correlation_threshold: Option<f64>,
    pub ratio: RatioConfig,
    pub imputation: ImputationConfig,
    pub differential_expression: DifferentialExpressionConfig,
    pub runtime: RuntimeConfig,
}

/// Configuration for the standalone `features2peptides` command.
///
/// This mirrors the deterministic core of Python's
/// `mokume.normalization.peptide.peptide_normalization`: load features, filter,
/// aggregate to the canonical-peptide x sample level, apply run/sample
/// normalization, and export the peptide intensity matrix. The CLI builds this
/// from `Features2PeptidesArgs`; the pipeline routes it through the same
/// feature-loading and peptide-export machinery that backs
/// `features2proteins --export-peptides`.
///
/// Options that depend on not-yet-ported pieces (parquet output, channel-based
/// IRS, run-level aggregation, dataset-level sample normalization, the filter
/// pipeline) are deferred by the pipeline with `MokumeError::NotImplemented`
/// rather than silently skipped, so they are not represented here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureToPeptidesConfig {
    pub input: InputConfig,
    /// Destination path for the peptide intensity CSV.
    pub output: PathBuf,
    pub filtering: FilterConfig,
    /// Optional file of exact protein accessions (one per line) to drop.
    pub remove_ids: Option<PathBuf>,
    /// Remove peptides present in fewer than 20% of samples (Python
    /// `--remove_low_frequency_peptides`).
    pub remove_low_frequency_peptides: bool,
    /// Keep shared/non-unique peptide rows and skip the per-protein
    /// `min_unique` gate (Python `--keep-shared-peptides`).
    pub keep_shared_peptides: bool,
    /// Skip both run- and sample-level normalization (Python
    /// `--skip_normalization`).
    pub skip_normalization: bool,
    /// Run/technical-replicate normalization method (Python
    /// `--run-normalization`): one of none/mean/median/max/global/max_min/iqr.
    pub run_normalization: String,
    /// Sample normalization method (Python `--sample-normalization`): one of
    /// none/globalmedian/conditionmedian/mediancenter/meancenter for the
    /// deterministic factor path.
    pub sample_normalization: String,
    /// Apply a base-2 logarithm to the exported peptide intensities (Python
    /// `--log2`).
    pub log2: bool,
    /// Also write the peptide intensity matrix to a parquet sibling of the CSV
    /// output (Python `--save_parquet` / `WriteParquetTask`). The parquet path
    /// is the output path with its extension replaced by `.parquet`.
    pub save_parquet: bool,
    /// Aggregation granularity (Python `--aggregation_level`): `sample`
    /// (default) sums runs into samples; `run` keeps run-level rows, appending
    /// `Run` / `TechReplicate` columns.
    pub aggregation_level: AggregationLevel,
    /// Opt-in preprocessing filter pipeline (Python `--filter-config` /
    /// `--filter-*`). `None` or a configuration with `enabled: false` keeps the
    /// default load-time filtering; an enabled configuration applies its filters
    /// on top, mirroring Python's `filter_pipeline.apply` per sample.
    pub filter_pipeline: Option<PreprocessingFilterConfig>,
    /// Channel IRS (Internal Reference Scaling) for TMT/iTRAQ
    /// (Python `--irs_channel` / `--irs_stat` / `--irs_scope`). `None`
    /// disables IRS. All scopes (`global` / `by_mixture` / `two_stage`) are
    /// ported; the channel is resolved (including SDRF autodetect) before
    /// reaching this config.
    pub irs: Option<IrsChannelConfig>,
}

/// Statistic used to aggregate reference-channel intensities and to compute the
/// global center, mirroring Python's `irs_stat` (`feature.py:736`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IrsStat {
    /// DuckDB `median(intensity)` / pandas `Series.median()`.
    #[default]
    Median,
    /// DuckDB `avg(intensity)` / pandas `Series.mean()`.
    Mean,
}

/// Scope of the IRS center, mirroring Python's `irs_scope`
/// (`feature.py:779-811`). Selects how the reference-channel `irs_value` per
/// techreplicate is rescaled to a common center.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IrsScope {
    /// One center across every run (`scale = global_center / irs_value`).
    #[default]
    Global,
    /// Per-mixture center (`scale = mixture_center / irs_value`).
    ByMixture,
    /// Per-mixture center re-anchored to a global center over the distinct
    /// mixture centers (`scale = (mixture_center / irs_value) *
    /// (global_center / mixture_center)`).
    TwoStage,
}

/// Channel IRS configuration for `features2peptides` (Python
/// `get_irs_scaling_factors`, `feature.py:711`). All three scopes
/// (`global` / `by_mixture` / `two_stage`) are implemented; the SDRF-driven
/// autodetect path resolves the channel before this config is built.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrsChannelConfig {
    /// Reference channel label matched against the raw feature `label`
    /// (`channel`) before any reformat (Python `channel = ?`).
    pub channel: String,
    /// Aggregation/center statistic (Python `irs_stat`).
    pub stat: IrsStat,
    /// Normalization scope (Python `irs_scope`).
    pub scope: IrsScope,
}

/// Level at which `features2peptides` aggregates intensities, mirroring Python's
/// `AGGREGATION_LEVEL_SAMPLE` / `AGGREGATION_LEVEL_RUN` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AggregationLevel {
    #[default]
    Sample,
    Run,
}

impl AggregationLevel {
    #[must_use]
    pub const fn is_run(self) -> bool {
        matches!(self, Self::Run)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputConfig {
    pub parquet: Option<PathBuf>,
    pub msstats: Option<PathBuf>,
    pub psm: Option<PathBuf>,
    pub sdrf: Option<PathBuf>,
    pub fasta: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    #[default]
    PythonCompatible,
    RustNative,
}

impl OutputFormat {
    pub const fn protein_column(self) -> &'static str {
        match self {
            Self::PythonCompatible => "ProteinName",
            Self::RustNative => "protein",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputConfig {
    pub protein_matrix: PathBuf,
    pub export_peptides: Option<PathBuf>,
    pub export_ions: Option<PathBuf>,
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilterConfig {
    pub min_aa: usize,
    pub min_unique_peptides: usize,
    pub remove_contaminants: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: true,
        }
    }
}

/// The opt-in `features2peptides` preprocessing filter pipeline
/// (`--filter-config` / `--filter-*`), mirroring Python's
/// `PreprocessingFilterConfig` (`mokume/model/filters.py`). Field defaults match
/// the Python dataclasses so a partial YAML/JSON file (or no file at all, just
/// CLI overrides) fills the rest exactly as `from_dict` does. `#[serde(default)]`
/// makes every missing key fall back to the corresponding `Default`, matching
/// Python's `data.get(block, {})` + dataclass defaults. Unknown keys are rejected
/// so a misspelled filter never degrades to a silent no-op.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PreprocessingFilterConfig {
    pub name: String,
    pub enabled: bool,
    pub strict_mode: bool,
    pub log_filtered_counts: bool,
    pub intensity: IntensityFilterConfig,
    pub peptide: PeptideFilterConfig,
    pub protein: ProteinFilterConfig,
    pub run_qc: RunQcFilterConfig,
}

impl Default for PreprocessingFilterConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            enabled: true,
            strict_mode: false,
            log_filtered_counts: true,
            intensity: IntensityFilterConfig::default(),
            peptide: PeptideFilterConfig::default(),
            protein: ProteinFilterConfig::default(),
            run_qc: RunQcFilterConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntensityFilterConfig {
    pub min_intensity: f64,
    pub cv_threshold: Option<f64>,
    pub min_replicate_agreement: i64,
    pub quantile_lower: f64,
    pub quantile_upper: f64,
    pub remove_zero_intensity: bool,
}

impl Default for IntensityFilterConfig {
    fn default() -> Self {
        Self {
            min_intensity: 0.0,
            cv_threshold: None,
            min_replicate_agreement: 1,
            quantile_lower: 0.0,
            quantile_upper: 1.0,
            remove_zero_intensity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PeptideFilterConfig {
    pub score: Option<NamedScoreFilterConfig>,
    pub allowed_charge_states: Option<Vec<i32>>,
    pub exclude_modifications: Vec<String>,
    pub max_missed_cleavages: Option<usize>,
    pub fdr_threshold: Option<f64>,
    pub min_peptide_length: usize,
    pub max_peptide_length: usize,
    pub exclude_sequence_patterns: Vec<String>,
    pub require_unique_peptides: bool,
}

impl Default for PeptideFilterConfig {
    fn default() -> Self {
        Self {
            score: None,
            allowed_charge_states: None,
            exclude_modifications: Vec::new(),
            max_missed_cleavages: None,
            fdr_threshold: None,
            min_peptide_length: 7,
            max_peptide_length: 50,
            exclude_sequence_patterns: Vec::new(),
            require_unique_peptides: false,
        }
    }
}

/// Threshold for one explicitly named QPX `additional_scores` entry.
///
/// The comparison direction is read from that entry's `higher_better` flag:
/// higher-better scores must be greater than or equal to `threshold`, while
/// lower-better scores must be less than or equal to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedScoreFilterConfig {
    pub name: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProteinFilterConfig {
    pub fdr_threshold: Option<f64>,
    pub min_coverage: f64,
    pub min_peptides: usize,
    pub min_unique_peptides: usize,
    pub razor_peptide_handling: String,
    pub protein_grouping: String,
    pub remove_contaminants: bool,
    pub remove_decoys: bool,
    pub contaminant_patterns: Vec<String>,
}

impl Default for ProteinFilterConfig {
    fn default() -> Self {
        Self {
            fdr_threshold: None,
            min_coverage: 0.0,
            min_peptides: 1,
            min_unique_peptides: 2,
            razor_peptide_handling: "keep".to_string(),
            protein_grouping: "none".to_string(),
            remove_contaminants: true,
            remove_decoys: true,
            contaminant_patterns: vec![
                "CONTAMINANT".to_string(),
                "CONTAM_".to_string(),
                "ENTRAP".to_string(),
                "DECOY".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunQcFilterConfig {
    pub min_total_intensity: f64,
    pub min_identified_features: usize,
    pub min_identified_proteins: usize,
    pub max_missing_rate: f64,
}

impl Default for RunQcFilterConfig {
    fn default() -> Self {
        Self {
            min_total_intensity: 0.0,
            min_identified_features: 0,
            min_identified_proteins: 0,
            max_missing_rate: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub run_method: String,
    pub sample_method: String,
    pub normalization_proteins: Option<PathBuf>,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            run_method: "median".to_string(),
            sample_method: "globalmedian".to_string(),
            normalization_proteins: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MaxLfqConfig {
    pub ion_alignment: Option<String>,
    /// Force the built-in MaxLFQ implementation instead of delegating to the
    /// DirectLFQ-aligned path. Mirrors Python's `MaxLFQQuantification.force_builtin`:
    /// Python delegates `--quant-method maxlfq` to DirectLFQ whenever the directlfq
    /// package is installed (the default in the reference environment), and only
    /// uses the built-in fallback when it is absent. The Rust DirectLFQ core is
    /// always available, so the faithful default is to delegate; this flag exposes
    /// the built-in fallback for testing and comparison.
    pub force_builtin: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PibaqConfig {
    pub enzyme: String,
    pub max_aa: usize,
    pub min_shared: usize,
    pub families_yaml: Option<PathBuf>,
    pub min_anchors: usize,
    pub high_anchor_threshold: usize,
    #[serde(default)]
    pub missed_cleavages: usize,
}

impl Default for PibaqConfig {
    fn default() -> Self {
        Self {
            enzyme: "Trypsin".to_string(),
            max_aa: 30,
            min_shared: 2,
            families_yaml: None,
            min_anchors: 1,
            high_anchor_threshold: 3,
            missed_cleavages: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectLfqConfig {
    pub cores: Option<usize>,
    pub min_nonan: usize,
    pub num_samples_quadratic: usize,
}

impl Default for DirectLfqConfig {
    fn default() -> Self {
        Self {
            cores: None,
            min_nonan: 1,
            num_samples_quadratic: 50,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchCorrectionConfig {
    pub enabled: bool,
    pub method: String,
    pub column: Option<String>,
    pub covariates: Option<Vec<String>>,
    pub parametric: bool,
    pub mean_only: bool,
    /// Original SDRF/sample-prefix batch label. Numeric strings remain accepted
    /// as a compatibility spelling for the historical encoded batch ID.
    pub ref_batch: Option<String>,
}

impl Default for BatchCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: "sample_prefix".to_string(),
            column: None,
            covariates: None,
            parametric: true,
            mean_only: false,
            ref_batch: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrsConfig {
    pub enabled: bool,
    pub reference_samples: Option<Vec<String>>,
    pub sdrf_column: Option<String>,
    pub sdrf_values: Option<Vec<String>>,
    pub reference_regex: String,
    pub stat: String,
    pub remove_reference: bool,
    /// SDRF column naming each channel's plex; `None` derives plexes from shared data files.
    #[serde(default)]
    pub plex_column: Option<String>,
}

impl Default for IrsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reference_samples: None,
            sdrf_column: None,
            sdrf_values: None,
            reference_regex: "pool|powder|ref|reference|bridge".to_string(),
            stat: "median".to_string(),
            remove_reference: false,
            plex_column: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RatioConfig {
    pub fraction_merge: String,
}

impl Default for RatioConfig {
    fn default() -> Self {
        Self {
            fraction_merge: "mean".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImputationConfig {
    pub enabled: bool,
    pub method: String,
    pub quantile: f64,
    pub shift: f64,
    pub scale: f64,
    pub n_neighbors: usize,
}

impl Default for ImputationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: "none".to_string(),
            quantile: 0.01,
            shift: 1.6,
            scale: 0.3,
            n_neighbors: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifferentialExpressionConfig {
    pub enabled: bool,
    pub contrasts: Option<Vec<String>>,
    pub contrasts_file: Option<PathBuf>,
    pub method: String,
    pub ensemble_methods: Option<Vec<String>>,
    pub ensemble_min_k: usize,
    pub log2fc_threshold: f64,
    /// Optional data-driven gate method. `None` applies `log2fc_threshold`
    /// directly; a method uses that value only as its fallback.
    #[serde(default)]
    pub effect_size_gate: Option<String>,
    pub fdr_threshold: f64,
    pub fdr_method: String,
    pub output: Option<PathBuf>,
}

impl Default for DifferentialExpressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            contrasts: None,
            contrasts_file: None,
            method: "auto".to_string(),
            ensemble_methods: None,
            ensemble_min_k: 2,
            log2fc_threshold: 0.5,
            effect_size_gate: None,
            fdr_threshold: 0.05,
            fdr_method: "bh".to_string(),
            output: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Optional cross-platform soft resident-memory budget (for example `512MB` or `1GB`).
    pub memory: Option<String>,
    pub threads: Option<usize>,
}

impl RuntimeConfig {
    pub fn explicit_threads(&self) -> Option<usize> {
        self.threads
    }
}
