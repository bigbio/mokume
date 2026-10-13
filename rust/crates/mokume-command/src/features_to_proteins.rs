use std::path::PathBuf;

use clap::Args;
use mokume_core::QuantMethod;

use crate::parsers::{
    parse_correlation, parse_de_log2fc, parse_finite_f64, parse_fraction, parse_memory,
    parse_nonnegative_f64, parse_positive_usize, parse_quant_method, DeLog2FcArg, QuantMethodArg,
};
use crate::PibaqDigestRequest;

mod config;
#[derive(Debug, Args)]
pub(crate) struct Features2ProteinsArgs {
    #[arg(
        short = 'p',
        long = "parquet",
        value_name = "FILE",
        required_unless_present_any = ["msstats", "psm"],
        conflicts_with = "msstats",
        help = "Feature-level QPX input; also supplies protein groups for spectral-count"
    )]
    parquet: Option<PathBuf>,

    #[arg(
        long = "msstats",
        value_name = "FILE",
        required_unless_present_any = ["parquet", "psm"],
        conflicts_with_all = ["parquet", "psm"],
        requires = "sdrf"
    )]
    msstats: Option<PathBuf>,

    #[arg(
        long = "psm",
        value_name = "FILE",
        required_unless_present_any = ["parquet", "msstats"],
        conflicts_with = "msstats",
        requires_all = ["sdrf", "parquet"],
        help = "PSM-level QPX input for spectral-count; requires matching feature QPX via --parquet"
    )]
    psm: Option<PathBuf>,

    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: PathBuf,

    #[arg(short = 's', long = "sdrf", value_name = "FILE")]
    sdrf: Option<PathBuf>,

    #[arg(
        long = "quant-method",
        default_value = "maxlfq",
        value_name = "METHOD",
        value_parser = parse_quant_method,
        help = "[possible values: directlfq, pibaq, maxlfq, sum, median, ratio, abd, \
intensity, peptide-count, spectral-count, top<N> (e.g. top3)]"
    )]
    quant_method: QuantMethodArg,

    #[arg(long = "min-aa", value_name = "N", default_value_t = 7)]
    min_aa: usize,

    #[arg(
        long = "min-unique",
        value_name = "N",
        help = "Minimum distinct peptides per protein/sample (default: 2; piBAQ: 0)"
    )]
    min_unique: Option<usize>,

    #[arg(
        long = "filter-protein-fdr",
        value_name = "FRACTION",
        value_parser = parse_fraction,
        help = "Keep protein groups whose best QPX pg_global_qvalue is <= FRACTION (e.g. 0.01)"
    )]
    filter_protein_fdr: Option<f64>,

    #[arg(long = "keep-contaminants", help = "Decoys are always removed")]
    keep_contaminants: bool,

    #[arg(long = "run-normalization", value_name = "METHOD", value_parser = [
        "none", "mean", "median", "max", "global", "max-min", "iqr",
    ], ignore_case = true,
    hide_possible_values = true,
    help = "[default: median; directlfq, ratio, peptide-count, spectral-count: none]\n\
[possible values: none, mean, median, max, global, max-min, iqr]"
    )]
    run_normalization: Option<String>,

    #[arg(long = "sample-normalization", value_name = "METHOD", value_parser = [
        "none",
        "global-median",
        "condition-median",
        "hierarchical",
        "quantile",
        "median-center",
        "mean-center",
        "rlr",
        "loess",
        "tmm",
    ], ignore_case = true,
    hide_possible_values = true,
    help = "[default: global-median; directlfq, ratio, peptide-count, spectral-count: none]\n\
[possible values: none, global-median, condition-median, hierarchical, quantile, median-center, \
mean-center, rlr, loess, tmm]"
    )]
    sample_normalization: Option<String>,

    #[arg(
        long = "normalization-proteins",
        value_name = "FILE",
        help = "Unavailable for directlfq, ratio, peptide-count, spectral-count"
    )]
    normalization_proteins: Option<PathBuf>,

    #[arg(short = 'f', long = "fasta", value_name = "FILE")]
    fasta: Option<PathBuf>,

    #[arg(long = "pibaq-enzyme", value_name = "NAME", default_value = "Trypsin")]
    pibaq_enzyme: String,

    #[arg(long = "pibaq-max-aa", value_name = "N", default_value_t = 30)]
    pibaq_max_aa: usize,

    #[arg(long = "pibaq-min-shared", value_name = "N", default_value_t = 2)]
    pibaq_min_shared: usize,

    #[arg(long = "pibaq-families", value_name = "FILE")]
    pibaq_families_yaml: Option<PathBuf>,

    #[arg(long = "pibaq-min-anchors", value_name = "N", default_value_t = 1)]
    pibaq_min_anchors: usize,

    #[arg(
        long = "directlfq-min-nonan",
        value_name = "N",
        value_parser = parse_positive_usize
    )]
    directlfq_min_nonan: Option<usize>,

    #[arg(
        long = "directlfq-num-samples-quadratic",
        value_name = "N",
        value_parser = parse_positive_usize
    )]
    directlfq_num_samples_quadratic: Option<usize>,

    #[arg(long = "export-peptides", value_name = "FILE")]
    export_peptides: Option<PathBuf>,

    #[arg(long = "export-ions", value_name = "FILE")]
    export_ions: Option<PathBuf>,

    #[arg(long = "batch-correction")]
    batch_correction: bool,

    #[arg(
        long = "batch-method",
        value_name = "METHOD",
        value_parser = ["sample-prefix", "column"],
        ignore_case = true
    )]
    batch_method: Option<String>,

    #[arg(long = "batch-column", value_name = "COLUMN")]
    batch_column: Option<String>,

    #[arg(long = "batch-covariate", value_name = "COLUMN")]
    batch_covariate: Vec<String>,

    #[arg(long = "batch-nonparametric")]
    batch_nonparametric: bool,

    #[arg(long = "batch-mean-only")]
    batch_mean_only: bool,

    #[arg(long = "batch-ref", value_name = "LABEL")]
    batch_ref: Option<String>,

    #[arg(long = "irs", help = "Unavailable for count methods")]
    irs: bool,

    #[arg(long = "irs-reference-sample", value_name = "SAMPLE")]
    irs_reference_sample: Vec<String>,

    #[arg(long = "irs-sdrf-column", value_name = "COLUMN")]
    irs_sdrf_column: Option<String>,

    #[arg(long = "irs-sdrf-value", value_name = "VALUE")]
    irs_sdrf_value: Vec<String>,

    #[arg(long = "irs-reference-regex", value_name = "REGEX")]
    irs_reference_regex: Option<String>,

    #[arg(
        long = "irs-stat",
        value_name = "STAT",
        value_parser = ["median", "mean"],
        ignore_case = true
    )]
    irs_stat: Option<String>,

    #[arg(long = "irs-remove-reference")]
    irs_remove_reference: bool,

    #[arg(
        long = "coverage-threshold",
        value_name = "FRACTION",
        value_parser = parse_fraction
    )]
    coverage_threshold: Option<f64>,

    #[arg(
        long = "min-sample-correlation",
        value_name = "CORRELATION",
        value_parser = parse_correlation,
        requires = "sdrf",
        help = "Minimum mean within-condition Pearson correlation"
    )]
    min_sample_correlation: Option<f64>,

    #[arg(
        long = "ratio-fraction-merge",
        value_name = "METHOD",
        value_parser = ["mean", "max"],
        ignore_case = true
    )]
    ratio_fraction_merge: Option<String>,

    #[arg(long = "impute-method", value_name = "METHOD", value_parser = [
        "mean",
        "median",
        "zero",
        "most-frequent",
        "knn",
        "minprob",
        "mindet",
        "qrilc",
        "seqknn",
        "impseq",
        "gms",
        "bpca",
        "impseqrob",
    ], ignore_case = true)]
    impute_method: Option<String>,

    #[arg(
        long = "impute-quantile",
        value_name = "FRACTION",
        value_parser = parse_fraction
    )]
    impute_quantile: Option<f64>,

    #[arg(
        long = "impute-shift",
        value_name = "VALUE",
        value_parser = parse_finite_f64
    )]
    impute_shift: Option<f64>,

    #[arg(
        long = "impute-scale",
        value_name = "VALUE",
        value_parser = parse_nonnegative_f64
    )]
    impute_scale: Option<f64>,

    #[arg(
        long = "impute-n-neighbors",
        value_name = "N",
        value_parser = parse_positive_usize
    )]
    impute_n_neighbors: Option<usize>,

    #[arg(
        long = "de-contrast",
        num_args = 2,
        value_names = ["GROUP_A", "GROUP_B"]
    )]
    de_contrast: Vec<String>,

    #[arg(long = "de-contrast-file", value_name = "FILE")]
    de_contrast_file: Option<PathBuf>,

    #[arg(long = "de-method", value_name = "METHOD", value_parser = [
        "auto",
        "limrots",
        "limma",
        "deqms",
        "proda",
        "rots",
        "ensemble",
    ], ignore_case = true)]
    de_method: Option<String>,

    #[arg(long = "de-ensemble-method", value_name = "METHOD")]
    de_ensemble_method: Vec<String>,

    #[arg(
        long = "de-ensemble-min-k",
        value_name = "N",
        value_parser = parse_positive_usize,
        help = "Minimum agreeing methods for ensemble DE (default: 2)"
    )]
    de_ensemble_min_k: Option<usize>,

    #[arg(
        long = "de-log2fc",
        value_name = "AUTO|VALUE",
        value_parser = parse_de_log2fc
    )]
    de_log2fc_threshold: Option<DeLog2FcArg>,

    #[arg(
        long = "de-effect-size-gate",
        value_name = "METHOD",
        value_parser = ["mixture", "null-quantile"],
        ignore_case = true
    )]
    de_effect_size_gate: Option<String>,

    #[arg(long = "de-fdr", value_name = "FRACTION", value_parser = parse_fraction)]
    de_fdr_threshold: Option<f64>,

    #[arg(
        long = "de-fdr-method",
        value_name = "METHOD",
        value_parser = ["bh", "ihw", "bky", "storey"],
        ignore_case = true
    )]
    de_fdr_method: Option<String>,

    #[arg(long = "de-output", value_name = "FILE")]
    de_output: Option<PathBuf>,

    #[arg(
        long = "memory",
        value_name = "SIZE",
        value_parser = parse_memory,
        help = "Soft resident-memory budget (e.g. 1GB); also limits QPX buffering"
    )]
    memory: Option<String>,

    #[arg(
        short = 't',
        long = "threads",
        value_name = "N",
        value_parser = parse_positive_usize
    )]
    threads: Option<usize>,
}

impl Features2ProteinsArgs {
    pub(crate) fn into_config(self) -> mokume_core::Result<mokume_core::FeatureToProteinsConfig> {
        config::into_config(self)
    }

    pub(crate) fn into_pibaq_digest_request(self) -> Option<PibaqDigestRequest> {
        if self.quant_method.method != QuantMethod::Pibaq {
            return None;
        }
        let fasta = self.fasta?;
        fasta.is_file().then_some(PibaqDigestRequest {
            fasta,
            enzyme: self.pibaq_enzyme,
            min_aa: self.min_aa,
            max_aa: self.pibaq_max_aa,
            missed_cleavages: 0,
        })
    }
}
