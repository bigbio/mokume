use clap::Parser;

use crate::{Cli, Commands, Features2ProteinsArgs, QuantifyCommands};

#[test]
fn transferred_weight_is_off_by_default_and_bounded() {
    let base = [
        "mokume",
        "quantify",
        "features2proteins",
        "--parquet",
        "input.parquet",
        "--output",
        "protein.csv",
    ];
    for (extra, expected) in [
        (&["--quant-method", "maxlfq"][..], None),
        (
            &["--quant-method", "maxlfq", "--transferred-weight", "0.25"][..],
            Some(0.25),
        ),
        (
            &["--quant-method", "top3", "--transferred-weight", "1"][..],
            Some(1.0),
        ),
    ] {
        let cli = Cli::parse_from(base.iter().chain(extra).copied());
        let Ok(config) = features_to_proteins_args(cli).into_config() else {
            panic!("expected a valid config for {extra:?}");
        };
        assert_eq!(config.input.transferred_weight, expected, "{extra:?}");
    }
    for value in ["0", "1.5", "-0.5", "nan"] {
        let argv = base.iter().copied().chain(["--transferred-weight", value]);
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "`{value}` should be rejected"
        );
    }
}

#[test]
fn transferred_weight_needs_weighing_methods_and_kept_transfers() {
    for options in [
        "-p in.parquet --quant-method sum --transferred-weight 0.5",
        "--msstats in.csv -s s.tsv --quant-method maxlfq --transferred-weight 0.5",
        "-p in.parquet --quant-method top3 --transferred-features drop --transferred-weight 0.5",
    ] {
        let argv = [
            "mokume",
            "quantify",
            "features2proteins",
            "-o",
            "protein.csv",
        ]
        .into_iter()
        .chain(options.split_whitespace());
        let Err(error) = features_to_proteins_args(Cli::parse_from(argv)).into_config() else {
            panic!("{options} was accepted");
        };
        assert!(
            error.to_string().contains("--transferred-weight requires"),
            "{options}: {error}"
        );
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
