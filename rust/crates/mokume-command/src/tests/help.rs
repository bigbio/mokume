use clap::CommandFactory;

use crate::Cli;

const REQUIRED_FEATURES_TO_PROTEINS_OPTIONS: &[&str] = &[
    "--parquet",
    "--psm",
    "--output",
    "--sdrf",
    "--quant-method",
    "--min-aa",
    "--min-unique",
    "--keep-contaminants",
    "--run-normalization",
    "--sample-normalization",
    "--normalization-proteins",
    "--fasta",
    "--pibaq-enzyme",
    "--pibaq-max-aa",
    "--pibaq-min-shared",
    "--pibaq-families",
    "--pibaq-min-anchors",
    "--directlfq-min-nonan",
    "--directlfq-num-samples-quadratic",
    "--export-peptides",
    "--export-ions",
    "--batch-correction",
    "--batch-method",
    "--batch-column",
    "--batch-covariate",
    "--batch-nonparametric",
    "--batch-mean-only",
    "--batch-ref",
    "--irs",
    "--irs-reference-sample",
    "--irs-sdrf-column",
    "--irs-sdrf-value",
    "--irs-reference-regex",
    "--irs-stat",
    "--irs-remove-reference",
    "--coverage-threshold",
    "--min-sample-correlation",
    "--ratio-fraction-merge",
    "--impute-method",
    "--impute-quantile",
    "--impute-shift",
    "--impute-scale",
    "--impute-n-neighbors",
    "--de-contrast",
    "--de-contrast-file",
    "--de-method",
    "--de-ensemble-method",
    "--de-ensemble-min-k",
    "--de-log2fc",
    "--de-fdr",
    "--de-fdr-method",
    "--de-output",
    "--memory",
    "--threads",
];

const REMOVED_FEATURES_TO_PROTEINS_OPTIONS: &[&str] = &[
    "--ion-alignment",
    "--output-format",
    "--method",
    "--directlfq-cores",
    "--duckdb-threads",
    "--batch-covariates",
    "--irs-reference-samples",
    "--irs-sdrf-values",
    "--impute",
    "--de",
    "--de-contrasts",
    "--de-contrasts-file",
    "--de-ensemble-methods",
    "--duckdb-memory",
    "--remove-contaminants",
    "--batch-parametric",
    "--impute-method missforest",
    "--plot-dir",
    "--plot-volcano",
    "--plot-heatmap",
    "--plot-pca",
    "--highlight-genes",
    "--interactive-report",
    "--report-output",
];

#[test]
fn top_level_help_lists_compute_commands() {
    let help = render_help(Cli::command());
    for command in ["quantify", "correct-batches"] {
        assert!(
            help.contains(command),
            "missing command `{command}` in help:\n{help}"
        );
    }
    let quantify_help = render_subcommand_help_path(&["quantify"]);
    for command in ["features2proteins", "features2peptides", "peptides2protein"] {
        assert!(
            quantify_help.contains(command),
            "missing command `{command}` in quantify help:\n{quantify_help}"
        );
    }
    // The visualization / tissue-map periphery moved to the Python wheel; the
    // Rust CLI is pure-compute and must not expose those subcommands.
    for command in ["tsne_visualization", "tissuemap", "agentic"] {
        assert!(
            !help.contains(command),
            "periphery command `{command}` must stay out of the Rust CLI:\n{help}"
        );
    }
}

#[test]
fn features2proteins_help_lists_python_option_surface() {
    let help = render_subcommand_help_path(&["quantify", "features2proteins"]);
    for option in REQUIRED_FEATURES_TO_PROTEINS_OPTIONS.iter().copied() {
        assert!(
            help.contains(option),
            "missing option `{option}` in help:\n{help}"
        );
    }
    // The plotting / interactive-report flags moved to the Python wheel; clap
    // must reject them now.
    for option in REMOVED_FEATURES_TO_PROTEINS_OPTIONS.iter().copied() {
        assert!(
            !help_has_option(&help, option),
            "removed plotting option `{option}` must not appear in help:\n{help}"
        );
    }
    // N is spelled in the method name (`top5`), so the companion option is
    // gone from both CLIs and must stay gone.
    assert!(
        !help.contains("--topn"),
        "removed option `--topn` must not appear in help:\n{help}"
    );
    assert!(!help.contains("--ibaq-"));
}

#[test]
fn help_uses_concise_semantic_value_names() {
    let features_help = render_subcommand_help_path(&["quantify", "features2proteins"]);
    for expected in [
        "--parquet <FILE>",
        "--quant-method <METHOD>",
        "--de-contrast <GROUP_A> <GROUP_B>",
        "--memory <SIZE>",
        "--threads <N>",
    ] {
        assert!(
            features_help.contains(expected),
            "missing semantic value name `{expected}` in help:\n{features_help}"
        );
    }
    assert!(
        !features_help.contains("<DE_CONTRAST> <DE_CONTRAST>"),
        "duplicate automatic contrast value names remain in help:\n{features_help}"
    );
    assert!(
        features_help.contains("[possible values: directlfq, pibaq, maxlfq"),
        "dynamic quantification methods do not use the standard possible-values label:\n{features_help}"
    );

    let correct_batches_help = render_subcommand_help_path(&["correct-batches"]);
    for expected in ["--input <PATH>", "--pattern <GLOB>", "--output <FILE>"] {
        assert!(
            correct_batches_help.contains(expected),
            "missing semantic value name `{expected}` in help:\n{correct_batches_help}"
        );
    }
}

#[test]
fn normalization_help_separates_and_aligns_metadata() {
    let features_help = render_subcommand_help_path(&["quantify", "features2proteins"]);
    for expected in [
        "[default: median; directlfq, ratio, peptide-count, spectral-count: none]",
        "[default: global-median; directlfq, ratio, peptide-count, spectral-count: none]",
        "Unavailable for directlfq, ratio, peptide-count, spectral-count",
    ] {
        assert!(
            features_help.contains(expected),
            "missing method-dependent normalization help `{expected}`:\n{features_help}"
        );
    }
    assert!(
        !features_help.contains("none] [possible values"),
        "normalization defaults and possible values must be on separate lines:\n{features_help}"
    );
    for (default_text, values_text) in [
        ("[default: median;", "[possible values: none, mean, median"),
        (
            "[default: global-median;",
            "[possible values: none, global-median",
        ),
    ] {
        let default_column = features_help
            .lines()
            .find(|line| line.contains(default_text))
            .and_then(|line| line.find('['));
        let values_column = features_help
            .lines()
            .find(|line| line.contains(values_text))
            .and_then(|line| line.find('['));
        assert_eq!(
            default_column, values_column,
            "normalization metadata brackets are misaligned:\n{features_help}"
        );
    }
}

#[test]
fn features2peptides_help_lists_python_option_surface() {
    let help = render_subcommand_help_path(&["quantify", "features2peptides"]);
    for option in [
        "--parquet",
        "--sdrf",
        "--min-aa",
        "--min-unique",
        "--keep-shared-peptides",
        "--remove-ids",
        "--remove-decoy-contaminants",
        "--remove-low-frequency-peptides",
        "--output",
        "--skip-normalization",
        "--run-normalization",
        "--sample-normalization",
        "--log2",
        "--save-parquet",
        "--irs-channel",
        "--irs-autodetect-regex",
        "--irs-stat",
        "--irs-scope",
        "--aggregation-level",
        "--filter-config",
        "--generate-filter-config",
        "--filter-min-intensity",
        "--filter-cv-threshold",
        "--filter-charge-state",
        "--filter-max-missed-cleavages",
        "--filter-peptide-fdr",
        "--filter-score",
        "--filter-exclude-modification",
        "--filter-protein-fdr",
        "--filter-min-features",
        "--filter-max-missing-rate",
    ] {
        assert!(
            help.contains(option),
            "missing option `{option}` in help:\n{help}"
        );
    }
}

#[test]
fn peptides2protein_help_lists_python_option_surface() {
    let help = render_subcommand_help_path(&["quantify", "peptides2protein"]);
    for option in [
        "--fasta",
        "--peptides",
        "--quant-method",
        "--enzyme",
        "--normalize",
        "--min-aa",
        "--max-aa",
        "--tpa",
        "--ruler",
        "--ploidy",
        "--organism",
        "--cpc",
        "--output",
        "--qc-report",
        "--threads",
        "--directlfq-min-nonan",
        "--families",
        "--min-shared",
        "--min-anchors",
        "--high-anchor-threshold",
    ] {
        assert!(
            help.contains(option),
            "missing option `{option}` in help:\n{help}"
        );
    }
    // N is spelled in the method name (`--quant-method top5`), so the companion
    // option is gone from both CLIs and must stay gone.
    assert!(
        !help.contains("--topn_n"),
        "removed option `--topn_n` must not appear in help:\n{help}"
    );
    assert!(
        help.contains("[possible values: pibaq, maxlfq, sum, directlfq"),
        "dynamic quantification methods do not use the standard possible-values label:\n{help}"
    );
}

#[test]
fn correct_batches_help_lists_python_option_surface() {
    let correct_batches_help = render_subcommand_help_path(&["correct-batches"]);
    for option in [
        "--input",
        "--pattern",
        "--comment",
        "--sep",
        "--output",
        "--sample-id-column",
        "--protein-id-column",
        "--pibaq-raw-column",
        "--pibaq-corrected-column",
        "--export-anndata",
    ] {
        assert!(
            correct_batches_help.contains(option),
            "missing option `{option}` in help:\n{correct_batches_help}"
        );
    }
    assert!(!correct_batches_help.contains("--ibaq_"));
    assert!(!correct_batches_help.contains("--ibaq-"));
}

#[test]
fn periphery_subcommands_are_removed() {
    // `tsne_visualization` and `tissuemap` moved to the Python wheel; the Rust
    // CLI no longer registers them as subcommands.
    let mut command = Cli::command();
    for name in ["tsne_visualization", "tsne-visualization", "tissuemap"] {
        assert!(
            command.find_subcommand_mut(name).is_none(),
            "periphery subcommand `{name}` must be removed from the Rust CLI"
        );
    }
}

fn render_help(mut command: clap::Command) -> String {
    let mut output = Vec::new();
    if let Err(error) = command.write_long_help(&mut output) {
        panic!("failed to render help: {error}");
    }
    match String::from_utf8(output) {
        Ok(help) => help,
        Err(error) => panic!("help is not valid utf8: {error}"),
    }
}

fn render_subcommand_help_path(path: &[&str]) -> String {
    let mut command = Cli::command();
    for name in path {
        let Some(subcommand) = command.find_subcommand_mut(name) else {
            panic!("missing subcommand `{name}`");
        };
        command = subcommand.clone();
    }
    render_help(command)
}

fn help_has_option(help: &str, option: &str) -> bool {
    help.split_whitespace()
        .any(|token| token.trim_end_matches(',') == option)
}
