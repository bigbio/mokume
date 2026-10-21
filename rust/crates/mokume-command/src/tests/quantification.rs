use std::path::Path;

use clap::Parser;

use crate::{Cli, Commands, Features2ProteinsArgs, QuantifyCommands};

#[test]
fn stabilization_is_an_opt_in_maxlfq_feature_option() -> mokume_core::Result<()> {
    let argv = [
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "features.parquet",
        "-o",
        "proteins.csv",
    ];
    let config = features_to_proteins_args(Cli::parse_from(argv)).into_config()?;
    assert!(!config.maxlfq.stabilize);
    let config =
        features_to_proteins_args(Cli::parse_from(argv.into_iter().chain(["--stabilize"])))
            .into_config()?;
    assert!(config.maxlfq.stabilize);
    for method in ["directlfq", "sum", "pibaq", "top3"] {
        let cli =
            Cli::parse_from(
                argv.into_iter()
                    .chain(["--quant-method", method, "--stabilize"]),
            );
        let Err(error) = features_to_proteins_args(cli).into_config() else {
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
fn maxlfq_defaults_to_hierarchical_sample_normalization() -> mokume_core::Result<()> {
    let argv = [
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "features.parquet",
        "-o",
        "proteins.csv",
        "--quant-method",
    ];
    let sample_method = |extra: &[&str]| -> mokume_core::Result<String> {
        let cli = Cli::parse_from(argv.into_iter().chain(extra.iter().copied()));
        Ok(features_to_proteins_args(cli)
            .into_config()?
            .normalization
            .sample_method)
    };
    assert_eq!(sample_method(&["maxlfq"])?, "hierarchical");
    assert_eq!(
        sample_method(&["maxlfq", "--normalization-proteins", "np.txt"])?,
        "globalmedian"
    );
    assert_eq!(sample_method(&["sum"])?, "globalmedian");
    Ok(())
}

#[test]
fn directlfq_sample_normalization_stays_on_unless_skipped() -> mokume_core::Result<()> {
    let argv = [
        "mokume",
        "quantify",
        "features2proteins",
        "-p",
        "features.parquet",
        "-o",
        "proteins.csv",
        "--quant-method",
    ];
    let config = features_to_proteins_args(Cli::parse_from(argv.into_iter().chain(["directlfq"])))
        .into_config()?;
    assert!(config.directlfq.normalize_samples);
    let skip = ["directlfq", "--directlfq-no-sample-normalization"];
    let config =
        features_to_proteins_args(Cli::parse_from(argv.into_iter().chain(skip))).into_config()?;
    assert!(!config.directlfq.normalize_samples);
    let cli = Cli::parse_from(
        argv.into_iter()
            .chain(["maxlfq", "--directlfq-no-sample-normalization"]),
    );
    let Err(error) = features_to_proteins_args(cli).into_config() else {
        panic!("MaxLFQ accepted --directlfq-no-sample-normalization");
    };
    assert!(
        error.to_string().contains("only applies to DirectLFQ"),
        "{error}"
    );
    Ok(())
}

#[test]
fn parses_true_spectral_count_psm_input() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--psm",
        "input.psm.parquet",
        "--parquet",
        "input.feature.parquet",
        "--sdrf",
        "input.sdrf.tsv",
        "--output",
        "counts.csv",
        "--quant-method",
        "spectral-count",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid spectral-count config");
    };
    assert_eq!(
        config.input.psm.as_deref(),
        Some(Path::new("input.psm.parquet"))
    );
    assert_eq!(
        config.input.parquet.as_deref(),
        Some(Path::new("input.feature.parquet"))
    );
    assert_eq!(
        config.quantification,
        mokume_core::QuantMethod::SpectralCount
    );
    assert_eq!(config.normalization.run_method, "none");
    assert_eq!(config.normalization.sample_method, "none");
}

#[test]
fn spectral_count_requires_paired_qpx_inputs() {
    let Err(error) = Cli::try_parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--psm",
        "input.psm.parquet",
        "--sdrf",
        "input.sdrf.tsv",
        "--output",
        "counts.csv",
        "--quant-method",
        "spectral-count",
    ]) else {
        panic!("spectral-count accepted a PSM without its feature QPX");
    };
    assert!(error.to_string().contains("--parquet"), "{error}");

    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.feature.parquet",
        "--sdrf",
        "input.sdrf.tsv",
        "--output",
        "counts.csv",
        "--quant-method",
        "spectral-count",
    ]);
    let Err(error) = features_to_proteins_args(cli).into_config() else {
        panic!("spectral-count accepted a feature QPX without its PSM QPX");
    };
    assert!(error.to_string().contains("--psm and --parquet"), "{error}");
}

#[test]
fn feature_input_uses_explicit_peptide_count_name() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.feature.parquet",
        "--output",
        "counts.csv",
        "--quant-method",
        "peptide-count",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid peptide-count config");
    };
    assert_eq!(
        config.quantification,
        mokume_core::QuantMethod::PeptideCount
    );
    assert_eq!(config.normalization.run_method, "none");
    assert_eq!(config.normalization.sample_method, "none");
}

#[test]
fn count_methods_reject_irs() {
    for (method, inputs) in [
        ("peptide-count", vec!["--parquet", "input.feature.parquet"]),
        (
            "spectral-count",
            vec![
                "--psm",
                "input.psm.parquet",
                "--parquet",
                "input.feature.parquet",
            ],
        ),
    ] {
        let mut argv = vec!["mokume", "quantify", "features2proteins"];
        argv.extend(inputs);
        argv.extend([
            "--sdrf",
            "input.sdrf.tsv",
            "--output",
            "counts.csv",
            "--quant-method",
            method,
            "--irs",
            "--irs-reference-sample",
            "Pool",
        ]);
        let cli = Cli::parse_from(argv);
        let args = features_to_proteins_args(cli);
        let Err(error) = args.into_config() else {
            panic!("IRS was accepted for a count method");
        };
        assert!(error.to_string().contains("cannot apply IRS"), "{error}");
    }
}

#[test]
fn pibaq_uses_thirty_aa_and_method_specific_min_unique() {
    let cli = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
        "--quant-method",
        "pibaq",
        "--fasta",
        "proteins.fasta",
    ]);
    let args = features_to_proteins_args(cli);
    let Ok(config) = args.into_config() else {
        panic!("expected a valid piBAQ config");
    };
    assert_eq!(config.pibaq.max_aa, 30);
    assert_eq!(config.filtering.min_unique_peptides, 0);

    let explicit = Cli::parse_from([
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
        "--quant-method",
        "pibaq",
        "--fasta",
        "proteins.fasta",
        "--min-unique",
        "0",
    ]);
    let args = features_to_proteins_args(explicit);
    assert!(args.into_config().is_err());
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

fn features_to_proteins_args(cli: Cli) -> Box<Features2ProteinsArgs> {
    let Commands::Quantify(quantify) = cli.command else {
        panic!("expected quantify command");
    };
    let QuantifyCommands::Features2Proteins(args) = quantify.command else {
        panic!("expected features2proteins command");
    };
    args
}
