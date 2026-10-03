use std::path::PathBuf;

use clap::{Args, ValueEnum};

use crate::parsers::{
    parse_peptides2protein_method, parse_positive_f64, parse_positive_i32, parse_positive_usize,
};

#[derive(Debug, Args)]
pub(crate) struct Peptides2ProteinArgs {
    #[arg(short = 'f', long = "fasta", value_name = "FILE")]
    pub(crate) fasta: Option<PathBuf>,

    #[arg(short = 'p', long = "peptides", value_name = "FILE")]
    pub(crate) peptides: PathBuf,

    #[arg(
        long = "quant-method",
        default_value = "pibaq",
        value_name = "METHOD",
        value_parser = parse_peptides2protein_method,
        help = "[possible values: pibaq, maxlfq, sum, directlfq, top<N> (e.g. top3)]"
    )]
    pub(crate) quant_method: String,

    #[arg(long = "enzyme", value_name = "NAME", default_value = "Trypsin")]
    pub(crate) enzyme: String,

    #[arg(long = "normalize")]
    pub(crate) normalize: bool,

    #[arg(long = "min-aa", value_name = "N", default_value_t = 7)]
    pub(crate) min_aa: usize,

    #[arg(long = "max-aa", value_name = "N", default_value_t = 30)]
    pub(crate) max_aa: usize,

    #[arg(long = "tpa")]
    pub(crate) tpa: bool,

    #[arg(long = "ruler")]
    pub(crate) ruler: bool,

    #[arg(long = "ploidy", value_name = "N", value_parser = parse_positive_i32)]
    pub(crate) ploidy: Option<i32>,

    #[arg(long = "organism", value_name = "NAME")]
    pub(crate) organism: Option<String>,

    #[arg(long = "cpc", value_name = "VALUE", value_parser = parse_positive_f64)]
    pub(crate) cpc: Option<f64>,

    #[arg(short = 'o', long = "output", value_name = "FILE", required = true)]
    pub(crate) output: Option<PathBuf>,

    #[arg(long = "qc-report", value_name = "FILE")]
    pub(crate) qc_report: Option<PathBuf>,

    #[arg(
        short = 't',
        long = "threads",
        value_name = "N",
        value_parser = parse_positive_usize,
        help = "DirectLFQ/MaxLFQ only"
    )]
    pub(crate) threads: Option<usize>,

    #[arg(
        long = "directlfq-min-nonan",
        value_name = "N",
        value_parser = parse_positive_usize
    )]
    pub(crate) directlfq_min_nonan: Option<usize>,

    #[arg(long = "families", value_name = "FILE")]
    pub(crate) families_yaml: Option<PathBuf>,

    #[arg(long = "min-shared", value_name = "N", default_value_t = 2)]
    pub(crate) min_shared: usize,

    #[arg(long = "min-anchors", value_name = "N", default_value_t = 1)]
    pub(crate) min_anchors: usize,

    #[arg(long = "high-anchor-threshold", value_name = "N", default_value_t = 3)]
    pub(crate) high_anchor_threshold: usize,
}

/// Batch-correction method of `correct-batches`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum CorrectBatchesMethod {
    /// Parametric ComBat on a complete protein x sample piBAQ matrix (default).
    #[default]
    Combat,
    /// BRIDLE (Batch Removal via Intrinsic Detectability and Latent
    /// Estimation) for multi-dataset collections (long input with dataset /
    /// anchor / gene / value columns; missing values kept).
    Bridle,
    /// Deprecated alias of `bridle` (hidden; logs a deprecation warning).
    #[value(hide = true)]
    Lim,
}

#[derive(Debug, Args)]
pub(crate) struct CorrectBatchesArgs {
    #[arg(
        long = "method",
        value_enum,
        default_value_t = CorrectBatchesMethod::Combat,
        value_name = "METHOD"
    )]
    pub(crate) method: CorrectBatchesMethod,

    /// ComBat: folder of long TSV files. BRIDLE: one long-format file
    /// (.parquet, .tsv or .csv).
    #[arg(short = 'i', long = "input", value_name = "PATH")]
    pub(crate) input: PathBuf,

    #[arg(
        short = 'p',
        long = "pattern",
        value_name = "GLOB",
        default_value = "*pibaq.tsv"
    )]
    pub(crate) pattern: String,

    #[arg(long = "comment", value_name = "PREFIX", default_value = "#")]
    pub(crate) comment: String,

    #[arg(long = "sep", value_name = "TEXT", default_value = "\t")]
    pub(crate) sep: String,

    #[arg(short = 'o', long = "output", value_name = "FILE")]
    pub(crate) output: PathBuf,

    #[arg(
        long = "sample-id-column",
        value_name = "COLUMN",
        default_value = "SampleID"
    )]
    pub(crate) sample_id_column: String,

    #[arg(
        long = "protein-id-column",
        value_name = "COLUMN",
        default_value = "ProteinName"
    )]
    pub(crate) protein_id_column: String,

    #[arg(
        long = "pibaq-raw-column",
        value_name = "COLUMN",
        default_value = "PiBAQ"
    )]
    pub(crate) pibaq_raw_column: String,

    #[arg(
        long = "pibaq-corrected-column",
        value_name = "COLUMN",
        default_value = "PiBAQBec"
    )]
    pub(crate) pibaq_corrected_column: String,

    #[arg(long = "export-anndata")]
    pub(crate) export_anndata: bool,

    #[command(flatten)]
    pub(crate) bridle: BridleArgs,
}

/// Early stop rule of the BRIDLE fit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum BridleStopRule {
    /// Mean |output change| between sweeps below --stop-tol (default).
    #[default]
    Output,
    /// Hold-out monitor MSE change over 3 sweeps (previous default).
    Monitor,
    /// Run all --sweeps.
    None,
}

/// Options of `correct-batches --method bridle`.
#[derive(Debug, Clone, Args)]
#[command(next_help_heading = "BRIDLE options (--method bridle)")]
pub(crate) struct BridleArgs {
    /// Dataset column of the long input (and of --plex-table).
    #[arg(long = "dataset-column", value_name = "COLUMN", default_value = "ds")]
    pub(crate) dataset_column: String,

    /// Anchor-sample column: the biological unit that links datasets (a cell
    /// line, reference material, pooled QC or the same patient across cohorts,
    /// e.g. a Cellosaurus id).
    // `--line-column` is kept as a hidden, deprecated alias.
    #[arg(
        long = "anchor-column",
        alias = "line-column",
        value_name = "COLUMN",
        default_value = "cvcl"
    )]
    pub(crate) anchor_column: String,

    /// Gene / protein column.
    #[arg(long = "gene-column", value_name = "COLUMN", default_value = "gene")]
    pub(crate) gene_column: String,

    /// log2 value column.
    #[arg(long = "value-column", value_name = "COLUMN", default_value = "v")]
    pub(crate) value_column: String,

    /// Reference dataset whose offsets are fixed to 0 [default: the dataset
    /// with the most profiles].
    #[arg(long = "reference", value_name = "DATASET")]
    pub(crate) reference: Option<String>,

    /// Optional lineage table (.csv/.tsv), e.g. DepMap Model.csv.
    #[arg(long = "lineage-table", value_name = "FILE")]
    pub(crate) lineage_table: Option<PathBuf>,

    #[arg(
        long = "lineage-key-column",
        value_name = "COLUMN",
        default_value = "RRID"
    )]
    pub(crate) lineage_key_column: String,

    #[arg(
        long = "lineage-column",
        value_name = "COLUMN",
        default_value = "OncotreeLineage"
    )]
    pub(crate) lineage_column: String,

    /// TMT plex / mixture id column (per profile, scoped to its dataset). Read
    /// from --plex-table when given, else from the input. Without it, plexes
    /// are inferred from shared missingness (datasets with >= 20 profiles).
    #[arg(long = "plex-column", value_name = "COLUMN")]
    pub(crate) plex_column: Option<String>,

    /// Table (.csv/.tsv) with dataset, line and plex columns.
    #[arg(long = "plex-table", value_name = "FILE")]
    pub(crate) plex_table: Option<PathBuf>,

    /// Disable the plex block entirely.
    #[arg(long = "no-plex")]
    pub(crate) no_plex: bool,

    /// Protein FASTA for technical sequence features (length, tryptic
    /// peptides, GRAVY, pI, amino-acid composition). Without it only the
    /// abundance spline is used.
    #[arg(long = "fasta", value_name = "FILE")]
    pub(crate) fasta: Option<PathBuf>,

    /// Only map gene names from FASTA headers containing this text (e.g. HUMAN).
    #[arg(long = "fasta-organism", value_name = "TEXT")]
    pub(crate) fasta_organism: Option<String>,

    /// Rank of the shared biological low-rank term.
    #[arg(long = "rank", value_name = "N", default_value_t = 16)]
    pub(crate) rank: usize,

    /// Maximum number of fitting sweeps.
    #[arg(long = "sweeps", value_name = "N", default_value_t = 400)]
    pub(crate) sweeps: usize,

    /// Early stop rule: `output` stops when the mean |change| of the output
    /// between two sweeps is below --stop-tol (after --min-sweeps); `monitor`
    /// is the previous rule (hold-out MSE change over 3 sweeps; use with
    /// --sweeps 60 for the old behaviour); `none` runs all --sweeps.
    #[arg(long = "stop-rule", value_name = "RULE", value_enum, default_value_t = BridleStopRule::Output)]
    pub(crate) stop_rule: BridleStopRule,

    /// Tolerance of --stop-rule [default: 1e-5 for `output`, 2e-4 for `monitor`].
    #[arg(long = "stop-tol", value_name = "TOL")]
    pub(crate) stop_tol: Option<f64>,

    /// Sweeps before --stop-rule may stop the fit [default: 200 for `output`,
    /// 8 for `monitor`].
    #[arg(long = "min-sweeps", value_name = "N")]
    pub(crate) min_sweeps: Option<usize>,

    #[arg(long = "seed", value_name = "N", default_value_t = 0)]
    pub(crate) seed: u32,

    /// Disable the graph prior: by default plex-aware anchor offsets chained
    /// to the reference (batches linked by >= 3 shared anchors; centring for
    /// unlinked batches with >= 5 profiles) are the initial value and prior
    /// mean of each dataset's offsets, and the feature model only where no
    /// such offset exists.
    #[arg(long = "no-graph-prior")]
    pub(crate) no_graph_prior: bool,

    /// Rescale each dataset sharing >= 20 anchor samples with the reference
    /// onto the reference's spread (per-dataset slope b_s from the anchors,
    /// e.g. TMT ratio compression); b_s is recorded in the --report.
    #[arg(long = "anchor-scale")]
    pub(crate) anchor_scale: bool,

    /// Optional pooled per-line biology (theta) for observed line/gene cells.
    #[arg(long = "theta-output", value_name = "FILE")]
    pub(crate) theta_output: Option<PathBuf>,

    /// Optional JSON fit report.
    #[arg(long = "report", value_name = "FILE")]
    pub(crate) report: Option<PathBuf>,
}
