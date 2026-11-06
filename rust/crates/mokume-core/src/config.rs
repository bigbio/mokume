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
    #[serde(default)]
    pub tmt: TmtConfig,
    pub coverage_threshold: Option<f64>,
    /// Minimum mean pairwise Pearson correlation to same-condition peers,
    /// computed on pairwise-complete log2 protein intensities.
    #[serde(default)]
    pub sample_correlation_threshold: Option<f64>,
    /// Maximum experiment-wide protein-group q-value (`--filter-protein-fdr`).
    /// A protein group is kept when the best (minimum) QPX `pg_global_qvalue`
    /// across all of its rows is `<= threshold`; every row of a failing group
    /// is dropped before the unique-peptide gate. `None` disables the filter.
    #[serde(default)]
    pub protein_fdr_threshold: Option<f64>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InputConfig {
    pub parquet: Option<PathBuf>,
    pub msstats: Option<PathBuf>,
    pub psm: Option<PathBuf>,
    pub sdrf: Option<PathBuf>,
    pub fasta: Option<PathBuf>,
    /// QPX features quantified in a run without an identification in that run
    /// (match-between-runs transfers).
    #[serde(default)]
    pub transferred_features: TransferredFeatures,
}

/// Whether QPX feature input keeps transferred features. A feature is
/// transferred when its `id_run_file_name` is null or names another run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TransferredFeatures {
    #[default]
    Keep,
    Drop,
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
    pub export_unnormalized_peptides: Option<PathBuf>,
    pub export_unimputed: Option<PathBuf>,
    pub export_ions: Option<PathBuf>,
    /// MaxLFQ/DirectLFQ only: each cell's quantification precision, in the
    /// layout of the protein matrix.
    pub export_precision: Option<PathBuf>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxLfqConfig {
    /// Minimum shared peptide species for each sample-pair ratio (Cox et al., Fig. 2).
    #[serde(default = "default_maxlfq_min_ratio_count")]
    pub min_ratio_count: usize,
    /// Enable Cox et al. Eq. 5 large-ratio stabilization for low-overlap sample pairs.
    #[serde(default)]
    pub stabilize: bool,
}

fn default_maxlfq_min_ratio_count() -> usize {
    2
}

impl Default for MaxLfqConfig {
    fn default() -> Self {
        Self {
            min_ratio_count: default_maxlfq_min_ratio_count(),
            stabilize: false,
        }
    }
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
    #[serde(default)]
    pub shared_mode: PibaqSharedMode,
    /// `None` resolves to [`PibaqSharedMode::default_family_rows`].
    #[serde(default)]
    pub family_rows: Option<PibaqFamilyRows>,
    /// Optional per-row evidence side-car (TSV).
    #[serde(default)]
    pub evidence_output: Option<PathBuf>,
}

/// How piBAQ assigns peptides shared by members of one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PibaqSharedMode {
    /// Legacy: per-sample proportional split, equal split without anchors.
    Proportional,
    /// Member values from unique peptides only.
    #[default]
    Unique,
    /// Unique peptides plus shared signal split by cross-sample member ratios.
    StableRatio,
}

impl PibaqSharedMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Proportional => "proportional",
            Self::Unique => "unique",
            Self::StableRatio => "stable-ratio",
        }
    }

    pub const fn default_family_rows(self) -> PibaqFamilyRows {
        match self {
            Self::Proportional => PibaqFamilyRows::None,
            Self::Unique | Self::StableRatio => PibaqFamilyRows::FamilyOnly,
        }
    }
}

/// Which multi-member families also get one family-level output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PibaqFamilyRows {
    None,
    /// Only families where no member has a unique peptide.
    FamilyOnly,
    All,
}

impl PibaqFamilyRows {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FamilyOnly => "family-only",
            Self::All => "all",
        }
    }
}

impl PibaqConfig {
    pub fn resolved_family_rows(&self) -> PibaqFamilyRows {
        self.family_rows
            .unwrap_or_else(|| self.shared_mode.default_family_rows())
    }
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
            shared_mode: PibaqSharedMode::default(),
            family_rows: None,
            evidence_output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectLfqConfig {
    pub min_nonan: usize,
    pub num_samples_quadratic: usize,
    /// DirectLFQ's global sample normalization (the inverse of directlfq's
    /// `deactivate_normalization`). Without it, sample loading differences stay
    /// in the protein intensities.
    #[serde(default = "default_directlfq_normalize_samples")]
    pub normalize_samples: bool,
}

fn default_directlfq_normalize_samples() -> bool {
    true
}

impl Default for DirectLfqConfig {
    fn default() -> Self {
        Self {
            min_nonan: 1,
            num_samples_quadratic: 50,
            normalize_samples: default_directlfq_normalize_samples(),
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
    #[serde(default)]
    pub missing_reference: IrsMissingReference,
}

/// What IRS does with a protein that has no reference value in a plex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum IrsMissingReference {
    /// Leave that plex unscaled (legacy).
    #[default]
    Keep,
    /// Remove the protein's values in that plex.
    Drop,
    /// Scale by the plex's median factor over proteins with a reference.
    PlexMedian,
}

impl IrsMissingReference {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Drop => "drop",
            Self::PlexMedian => "plex-median",
        }
    }
}

/// Reporter-ion (TMT) feature corrections applied per QPX row before quantification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TmtConfig {
    #[serde(default)]
    pub row_merge: TmtRowMerge,
    /// Isotope-impurity table (channel, -2, -1, +1, +2 in percent).
    #[serde(default)]
    pub impurity_table: Option<PathBuf>,
    #[serde(default)]
    pub interference_floor: Option<TmtInterferenceFloor>,
    /// Quantile of loading-adjusted channel ratios used by the `auto` floor.
    #[serde(default = "default_floor_quantile")]
    pub floor_quantile: f64,
}

fn default_floor_quantile() -> f64 {
    0.01
}

impl Default for TmtConfig {
    fn default() -> Self {
        Self {
            row_merge: TmtRowMerge::default(),
            impurity_table: None,
            interference_floor: None,
            floor_quantile: default_floor_quantile(),
        }
    }
}

impl TmtConfig {
    pub fn is_active(&self) -> bool {
        self.row_merge != TmtRowMerge::MaxPerChannel
            || self.impurity_table.is_some()
            || self.interference_floor.is_some()
    }
}

/// How repeated rows of one ion in one plex (PSMs, fractions) are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TmtRowMerge {
    /// Max per channel independently (legacy).
    #[default]
    MaxPerChannel,
    /// Keep the whole row with the highest summed reporter intensity.
    BestRow,
}

impl TmtRowMerge {
    pub const fn label(self) -> &'static str {
        match self {
            Self::MaxPerChannel => "max",
            Self::BestRow => "best-row",
        }
    }
}

/// Additive co-isolation floor removed from every reporter channel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TmtInterferenceFloor {
    /// Fraction of the loading-adjusted row mean.
    Fixed(f64),
    /// Estimated per plex.
    Auto,
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
            missing_reference: IrsMissingReference::Keep,
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
    pub n_neighbors: usize,
    /// Seed for stochastic imputation (cross-language random streams may differ).
    #[serde(default = "default_imputation_seed")]
    pub seed: u64,
    /// Official imputeLCMD tune.sigma for QRILC and MinProb.
    #[serde(default = "default_imputation_tune_sigma")]
    pub tune_sigma: f64,
}

fn default_imputation_seed() -> u64 {
    42
}
fn default_imputation_tune_sigma() -> f64 {
    1.0
}

impl Default for ImputationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: "none".to_string(),
            quantile: 0.01,
            n_neighbors: 5,
            seed: default_imputation_seed(),
            tune_sigma: default_imputation_tune_sigma(),
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
    /// SDRF columns entering a fixed-effects design with the conditions; empty
    /// keeps the two-group tests.
    #[serde(default)]
    pub covariates: Vec<String>,
    /// SDRF column whose levels are random blocks of that design (limma's
    /// `duplicateCorrelation` + `lmFit(block =)`); `None` fits no block.
    #[serde(default)]
    pub block: Option<String>,
    /// Weight DEqMS by each cell's MaxLFQ/DirectLFQ quantification precision.
    #[serde(default)]
    pub precision_weights: bool,
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
            covariates: Vec::new(),
            block: None,
            precision_weights: false,
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
