use std::path::Path;

use clap::Parser;
use mokume_core::{MokumeError, PreprocessingFilterConfig};

use crate::{Cli, Commands, Features2ProteinsArgs, QuantifyCommands};

const PYTHON_FEATURES_TO_PROTEINS_ARGS: &[&str] = &[
    "mokume",
    "quantify",
    "features2proteins",
    "-p",
    "input.parquet",
    "-o",
    "protein.csv",
    "-s",
    "input.sdrf.tsv",
    "--quant-method",
    "directlfq",
    "--directlfq-min-nonan",
    "3",
    "--batch-correction",
    "--batch-method",
    "column",
    "--batch-column",
    "characteristics[batch]",
    "--batch-covariate",
    "characteristics[sex]",
    "--batch-covariate",
    "characteristics[organism part]",
    "--batch-nonparametric",
    "--batch-mean-only",
    "--batch-ref",
    "2",
    "--irs",
    "--irs-reference-sample",
    "Pool A",
    "--irs-reference-sample",
    "Pool B",
    "--coverage-threshold",
    "0.65",
    "--min-sample-correlation",
    "0.90",
    "--impute-method",
    "knn",
    "--de-method",
    "limma",
    "--de-contrast",
    "NASH-HL",
    "NASH-Ctrl",
    "--de-fdr-method",
    "ihw",
    "--de-output",
    "de.csv",
    "--memory",
    "1GB",
    "--threads",
    "24",
];

#[test]
fn correct_batches_accepts_canonical_column_options() {
    let cli = Cli::parse_from([
        "mokume",
        "correct-batches",
        "-i",
        "in",
        "-o",
        "out",
        "--sample-id-column",
        "MySample",
        "--protein-id-column",
        "MyProtein",
        "--pibaq-raw-column",
        "MyPibaq",
    ]);
    let Commands::CorrectBatches(args) = cli.command else {
        panic!("expected the correct-batches subcommand");
    };
    assert_eq!(args.sample_id_column, "MySample");
    assert_eq!(args.protein_id_column, "MyProtein");
    assert_eq!(args.pibaq_raw_column, "MyPibaq");
}

#[test]
fn parses_python_features2proteins_options() {
    let cli = Cli::parse_from(PYTHON_FEATURES_TO_PROTEINS_ARGS.iter().copied());

    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };

    assert_eq!(config.directlfq.min_nonan, 3);
    assert!(config.batch.enabled);
    assert_eq!(config.batch.method, "column");
    assert_eq!(
        config.batch.column.as_deref(),
        Some("characteristics[batch]")
    );
    assert_eq!(
        config.batch.covariates.as_deref(),
        Some(
            &[
                "characteristics[sex]".to_string(),
                "characteristics[organism part]".to_string(),
            ][..]
        )
    );
    assert!(!config.batch.parametric);
    assert!(config.batch.mean_only);
    assert_eq!(config.batch.ref_batch.as_deref(), Some("2"));
    assert_eq!(
        config.irs.reference_samples.as_deref(),
        Some(&["Pool A".to_string(), "Pool B".to_string()][..])
    );
    assert_eq!(config.coverage_threshold, Some(0.65));
    assert_eq!(config.sample_correlation_threshold, Some(0.90));
    assert!(config.imputation.enabled);
    assert_eq!(config.imputation.method, "knn");
    assert!(config.differential_expression.enabled);
    assert_eq!(config.differential_expression.fdr_method, "ihw");
    assert_eq!(config.runtime.memory.as_deref(), Some("1GB"));
    assert_eq!(config.runtime.threads, Some(24));
}

#[test]
fn canonical_kebab_case_values_map_to_internal_config() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
        "--quant-method",
        "sum",
        "--run-normalization",
        "max-min",
        "--sample-normalization",
        "median-center",
        "--impute-method",
        "most-frequent",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected canonical CLI values to produce a valid config");
    };

    assert_eq!(config.normalization.run_method, "max_min");
    assert_eq!(config.normalization.sample_method, "mediancenter");
    assert_eq!(config.imputation.method, "most_frequent");
}

#[test]
fn rejects_removed_ibaq_method_name() {
    let features = Cli::try_parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--quant-method",
        "ibaq",
    ]);
    assert!(features.is_err());

    let peptides = Cli::try_parse_from([
        "mokume",
        "quantify",
        "peptides2protein",
        "--peptides",
        "peptides.csv",
        "--quant-method",
        "ibaq",
    ]);
    assert!(peptides.is_err());
}

#[test]
fn rejects_nonpositive_memory_budget() {
    let parsed = Cli::try_parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
        "--memory",
        "0GB",
    ]);

    let Err(error) = parsed else {
        panic!("zero memory budget was accepted");
    };
    assert!(error.to_string().contains("invalid memory value `0GB`"));
}

#[test]
fn parses_adaptive_de_options() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "-s",
        "input.sdrf.tsv",
        "--de-method",
        "limma",
        "--de-contrast",
        "A",
        "B",
        "--de-log2fc",
        "auto",
        "--de-fdr-method",
        "storey",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };

    assert_eq!(config.differential_expression.log2fc_threshold, 0.5);
    assert_eq!(
        config.differential_expression.effect_size_gate.as_deref(),
        Some("mixture")
    );
    assert_eq!(config.differential_expression.fdr_method, "storey");
}

#[test]
fn explicit_effect_size_method_uses_numeric_threshold_as_fallback() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--de-log2fc",
        "0.25",
        "--de-effect-size-gate",
        "null-quantile",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };

    assert_eq!(config.differential_expression.log2fc_threshold, 0.25);
    assert_eq!(
        config.differential_expression.effect_size_gate.as_deref(),
        Some("null_quantile")
    );
}

#[test]
fn parses_native_msstats_input() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--msstats",
        "input.csv",
        "--sdrf",
        "input.sdrf.tsv",
        "--output",
        "protein.csv",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };

    assert_eq!(
        config.input.msstats.as_deref(),
        Some(Path::new("input.csv"))
    );
    assert!(config.input.parquet.is_none());
}

#[test]
fn native_msstats_input_requires_sdrf() {
    let Err(error) = Cli::try_parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--msstats",
        "input.csv",
        "--output",
        "protein.csv",
    ]) else {
        panic!("MSstats without SDRF should fail");
    };

    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
}

#[test]
fn filter_protein_fdr_is_off_by_default_and_parsed_when_given() {
    let base = [
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
    ];
    let Ok(config) = features_to_proteins_args(Cli::parse_from(base)).into_config() else {
        panic!("expected a valid features2proteins config");
    };
    assert_eq!(config.protein_fdr_threshold, None);

    let cli = Cli::parse_from(base.iter().copied().chain(["--filter-protein-fdr", "0.01"]));
    let Ok(config) = features_to_proteins_args(cli).into_config() else {
        panic!("expected a valid protein FDR config");
    };
    assert_eq!(config.protein_fdr_threshold, Some(0.01));
}

#[test]
fn filter_protein_fdr_rejects_values_outside_unit_interval() {
    for value in ["1.5", "-0.1", "nan"] {
        let result = Cli::try_parse_from([
            "mokume",
            "quantify",
            "features2proteins",
            "--parquet",
            "input.parquet",
            "--output",
            "protein.csv",
            "--filter-protein-fdr",
            value,
        ]);
        assert!(result.is_err(), "`{value}` should be rejected");
    }
}

#[test]
fn zero_impute_method_enables_imputation() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
        "--impute-method",
        "zero",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid imputation config");
    };
    assert!(config.imputation.enabled);
    assert_eq!(config.imputation.method, "zero");
}

#[test]
fn repeated_de_ensemble_methods_preserve_members() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--de-method",
        "ensemble",
        "--de-ensemble-method",
        "limma",
        "--de-ensemble-method",
        "deqms",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };
    assert_eq!(
        config.differential_expression.ensemble_methods,
        Some(vec!["limma".to_owned(), "deqms".to_owned()])
    );
}

#[test]
fn rejects_ensemble_min_k_for_single_de_method() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--de-method",
        "limma",
        "--de-ensemble-min-k",
        "2",
    ]);
    let args = features_to_proteins_args(cli);

    assert!(matches!(
        args.into_config(),
        Err(MokumeError::InvalidInput { .. })
    ));
}

#[test]
fn repeatable_reference_sample_preserves_commas() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--irs",
        "--irs-reference-sample",
        "Pool, batch A",
        "--irs-reference-sample",
        "Pool B",
    ]);

    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid features2proteins config");
    };

    assert_eq!(
        config.irs.reference_samples.as_deref(),
        Some(&["Pool, batch A".to_string(), "Pool B".to_string()][..])
    );
}

#[test]
fn rejects_removed_plural_reference_sample_option() {
    let Err(error) = Cli::try_parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "input.parquet",
        "-o",
        "protein.csv",
        "--irs-reference-samples",
        "Pool A,Pool B",
    ]) else {
        panic!("removed plural reference-sample option was accepted");
    };

    assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
}

fn features_to_proteins_args(cli: Cli) -> Box<Features2ProteinsArgs> {
    let Commands::Quantify(quantify) = cli.command else {
        panic!("expected quantify command");
    };
    let QuantifyCommands::Features2Proteins(args) = quantify.command else {
        panic!("expected features2proteins command");
    };
    args
}

#[test]
fn filter_config_rejects_unknown_keys() {
    for yaml in [
        "unknown_option: true\n",
        "intensity:\n  min_intensitty: 5\n",
    ] {
        let parsed = serde_yaml::from_str::<PreprocessingFilterConfig>(yaml);
        assert!(parsed.is_err(), "unknown filter key was accepted: {yaml}");
    }
}
