use std::path::PathBuf;

use clap::Args;

use crate::parsers::{parse_de_log2fc, parse_fraction, parse_positive_usize, DeLog2FcArg};

/// The differential-expression options of `features2proteins`, flattened into
/// its arguments.
#[derive(Debug, Args)]
pub(super) struct DeArgs {
    #[arg(
        long = "de-contrast",
        num_args = 2,
        value_names = ["GROUP_A", "GROUP_B"]
    )]
    pub(super) de_contrast: Vec<String>,

    #[arg(long = "de-contrast-file", value_name = "FILE")]
    pub(super) de_contrast_file: Option<PathBuf>,

    #[arg(long = "de-method", value_name = "METHOD", value_parser = [
        "auto",
        "limrots",
        "limma",
        "deqms",
        "proda",
        "rots",
        "ensemble",
    ], ignore_case = true)]
    pub(super) de_method: Option<String>,

    #[arg(
        long = "de-covariate",
        value_name = "COLUMN",
        help = "SDRF column added to a limma/DEqMS design with the conditions (repeatable)"
    )]
    pub(super) de_covariate: Vec<String>,

    #[arg(
        long = "de-block",
        value_name = "COLUMN",
        help = "SDRF column whose levels are random blocks in a limma/DEqMS design (e.g. the subject)"
    )]
    pub(super) de_block: Option<String>,

    #[arg(
        long = "de-precision-weights",
        help = "Weight DEqMS by each cell's MaxLFQ/DirectLFQ quantification precision"
    )]
    pub(super) de_precision_weights: bool,

    #[arg(long = "de-ensemble-method", value_name = "METHOD")]
    pub(super) de_ensemble_method: Vec<String>,

    #[arg(
        long = "de-ensemble-min-k",
        value_name = "N",
        value_parser = parse_positive_usize,
        help = "Minimum agreeing methods for ensemble DE (default: 2)"
    )]
    pub(super) de_ensemble_min_k: Option<usize>,

    #[arg(
        long = "de-log2fc",
        value_name = "AUTO|VALUE",
        value_parser = parse_de_log2fc
    )]
    pub(super) de_log2fc_threshold: Option<DeLog2FcArg>,

    #[arg(
        long = "de-effect-size-gate",
        value_name = "METHOD",
        value_parser = ["mixture", "null-quantile"],
        ignore_case = true
    )]
    pub(super) de_effect_size_gate: Option<String>,

    #[arg(long = "de-fdr", value_name = "FRACTION", value_parser = parse_fraction)]
    pub(super) de_fdr_threshold: Option<f64>,

    #[arg(
        long = "de-fdr-method",
        value_name = "METHOD",
        value_parser = ["bh", "ihw", "bky", "storey"],
        ignore_case = true
    )]
    pub(super) de_fdr_method: Option<String>,

    #[arg(long = "de-output", value_name = "FILE")]
    pub(super) de_output: Option<PathBuf>,
}
