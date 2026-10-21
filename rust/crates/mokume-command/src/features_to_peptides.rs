use std::{
    fs::{read_to_string, write},
    path::{Path, PathBuf},
};

use clap::Args;
use mokume_core::{
    AggregationLevel, FeatureToPeptidesConfig, FilterConfig, InputConfig, IrsChannelConfig,
    IrsScope, IrsStat, MokumeError, NamedScoreFilterConfig, PreprocessingFilterConfig,
};
use mokume_pipeline::{resolve_irs_autodetect_channel, run_features_to_peptides};

use crate::filter_config_examples::{EXAMPLE_FILTER_CONFIG_JSON, EXAMPLE_FILTER_CONFIG_YAML};
use crate::parsers::{parse_fraction, parse_named_score_filter, parse_nonnegative_f64};

#[allow(dead_code)]
#[derive(Debug, Args)]
pub(crate) struct Features2PeptidesArgs {
    #[arg(
        short = 'p',
        long = "parquet",
        value_name = "FILE",
        required_unless_present = "generate_filter_config"
    )]
    parquet: Option<PathBuf>,

    #[arg(short = 's', long = "sdrf", value_name = "FILE")]
    sdrf: Option<PathBuf>,

    #[arg(long = "min-aa", value_name = "N")]
    min_aa: Option<usize>,

    #[arg(long = "min-unique", value_name = "N")]
    min_unique: Option<usize>,

    #[arg(long = "keep-shared-peptides")]
    keep_shared_peptides: bool,

    #[arg(long = "remove-ids", value_name = "FILE")]
    remove_ids: Option<PathBuf>,

    #[arg(long = "remove-decoy-contaminants")]
    remove_decoy_contaminants: bool,

    #[arg(long = "remove-low-frequency-peptides")]
    remove_low_frequency_peptides: bool,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "FILE",
        required_unless_present = "generate_filter_config"
    )]
    output: Option<PathBuf>,

    #[arg(
        long = "skip-normalization",
        conflicts_with_all = ["run_normalization", "sample_normalization"]
    )]
    skip_normalization: bool,

    #[arg(long = "run-normalization", value_name = "METHOD", value_parser = [
        "none", "mean", "median", "max", "global", "max-min", "iqr",
    ], ignore_case = true)]
    run_normalization: Option<String>,

    #[arg(long = "sample-normalization", value_name = "METHOD", value_parser = [
        "none",
        "global-median",
        "condition-median",
    ], ignore_case = true)]
    sample_normalization: Option<String>,

    #[arg(long = "log2")]
    log2: bool,

    #[arg(long = "save-parquet")]
    save_parquet: bool,

    #[arg(long = "irs-channel", value_name = "NAME")]
    irs_channel: Option<String>,

    #[arg(long = "irs-autodetect-regex", value_name = "REGEX")]
    irs_autodetect_regex: Option<String>,

    #[arg(long = "irs-stat", value_name = "STAT", default_value = "median", value_parser = ["median", "mean"], ignore_case = true)]
    irs_stat: String,

    #[arg(long = "irs-scope", value_name = "SCOPE", default_value = "global", value_parser = ["global", "by-mixture", "two-stage"], ignore_case = true)]
    irs_scope: String,

    #[arg(long = "aggregation-level", value_name = "LEVEL", default_value = "sample", value_parser = ["sample", "run"], ignore_case = true)]
    aggregation_level: String,

    #[arg(long = "filter-config", value_name = "FILE")]
    filter_config: Option<PathBuf>,

    #[arg(long = "generate-filter-config", value_name = "FILE", exclusive = true)]
    generate_filter_config: Option<PathBuf>,

    #[arg(
        long = "filter-min-intensity",
        value_name = "VALUE",
        value_parser = parse_nonnegative_f64
    )]
    filter_min_intensity: Option<f64>,

    #[arg(
        long = "filter-cv-threshold",
        value_name = "VALUE",
        value_parser = parse_nonnegative_f64
    )]
    filter_cv_threshold: Option<f64>,

    #[arg(long = "filter-charge-state", value_name = "CHARGE")]
    filter_charge_state: Vec<i32>,

    #[arg(long = "filter-max-missed-cleavages", value_name = "N")]
    filter_max_missed_cleavages: Option<usize>,

    #[arg(
        long = "filter-peptide-fdr",
        value_name = "FRACTION",
        value_parser = parse_fraction
    )]
    filter_peptide_fdr: Option<f64>,

    #[arg(
        long = "filter-score",
        value_name = "NAME=THRESHOLD",
        value_parser = parse_named_score_filter,
        help = "Direction follows higher_better"
    )]
    filter_score: Option<NamedScoreFilterConfig>,

    #[arg(long = "filter-exclude-modification", value_name = "NAME")]
    filter_exclude_modification: Vec<String>,

    #[arg(
        long = "filter-protein-fdr",
        value_name = "FRACTION",
        value_parser = parse_fraction
    )]
    filter_protein_fdr: Option<f64>,

    #[arg(long = "filter-min-features", value_name = "N")]
    filter_min_features: Option<usize>,

    #[arg(
        long = "filter-max-missing-rate",
        value_name = "FRACTION",
        value_parser = parse_fraction
    )]
    filter_max_missing_rate: Option<f64>,
}

pub(crate) fn dispatch(args: &Features2PeptidesArgs) -> mokume_core::Result<()> {
    if let Some(path) = &args.generate_filter_config {
        generate_filter_config(path)?;
        return Ok(());
    }
    let parquet = required_parquet(args)?;
    validate_optional_input(args.sdrf.as_ref())?;
    validate_optional_input(args.remove_ids.as_ref())?;
    validate_irs_options(args)?;
    let config = feature_to_peptides_config(
        args,
        parquet,
        required_output(args)?,
        resolve_irs_config(args)?,
        build_filter_pipeline(args)?,
    );
    run_features_to_peptides(&config)
}

fn required_parquet(args: &Features2PeptidesArgs) -> mokume_core::Result<PathBuf> {
    let parquet = args
        .parquet
        .as_ref()
        .ok_or_else(|| MokumeError::InvalidInput {
            message: "features2peptides requires --parquet".to_owned(),
        })?;
    validate_optional_input(Some(parquet))?;
    Ok(parquet.clone())
}

fn required_output(args: &Features2PeptidesArgs) -> mokume_core::Result<PathBuf> {
    args.output
        .clone()
        .ok_or_else(|| MokumeError::InvalidInput {
            message: "features2peptides requires --output".to_owned(),
        })
}

fn validate_optional_input(path: Option<&PathBuf>) -> mokume_core::Result<()> {
    if let Some(path) = path {
        if !path.exists() {
            return Err(MokumeError::MissingInput { path: path.clone() });
        }
    }
    Ok(())
}

fn validate_irs_options(args: &Features2PeptidesArgs) -> mokume_core::Result<()> {
    if args.irs_channel.is_some() && args.irs_autodetect_regex.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "choose either --irs-channel or --irs-autodetect-regex, not both".to_owned(),
        });
    }
    if args.irs_autodetect_regex.is_some() && args.sdrf.is_none() {
        return Err(MokumeError::InvalidInput {
            message: "--irs-autodetect-regex requires --sdrf".to_owned(),
        });
    }
    let irs_requested = args.irs_channel.is_some() || args.irs_autodetect_regex.is_some();
    if !irs_requested
        && (!args.irs_stat.eq_ignore_ascii_case("median")
            || !args.irs_scope.eq_ignore_ascii_case("global"))
    {
        return Err(MokumeError::InvalidInput {
            message: "--irs-stat/--irs-scope require --irs-channel or --irs-autodetect-regex"
                .to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn validate_options(args: &Features2PeptidesArgs) -> mokume_core::Result<()> {
    validate_irs_options(args)
}

fn resolve_irs_config(
    args: &Features2PeptidesArgs,
) -> mokume_core::Result<Option<IrsChannelConfig>> {
    let channel = match (&args.irs_channel, &args.irs_autodetect_regex, &args.sdrf) {
        (Some(channel), _, _) => Some(channel.clone()),
        (None, Some(regex), Some(sdrf)) => {
            Some(resolve_irs_autodetect_channel(sdrf, regex)?.ok_or_else(|| {
                MokumeError::InvalidInput {
                    message: format!(
                        "--irs-autodetect-regex `{regex}` matched no reference channel"
                    ),
                }
            })?)
        }
        _ => None,
    };
    Ok(channel.map(|channel| IrsChannelConfig {
        channel,
        stat: if args.irs_stat.eq_ignore_ascii_case("mean") {
            IrsStat::Mean
        } else {
            IrsStat::Median
        },
        scope: parse_irs_scope(&args.irs_scope),
    }))
}

fn parse_irs_scope(scope: &str) -> IrsScope {
    if scope.eq_ignore_ascii_case("by-mixture") {
        IrsScope::ByMixture
    } else if scope.eq_ignore_ascii_case("two-stage") {
        IrsScope::TwoStage
    } else {
        IrsScope::Global
    }
}

fn feature_to_peptides_config(
    args: &Features2PeptidesArgs,
    parquet: PathBuf,
    output: PathBuf,
    irs: Option<IrsChannelConfig>,
    filter_pipeline: Option<PreprocessingFilterConfig>,
) -> FeatureToPeptidesConfig {
    let defaults = FilterConfig::default();
    FeatureToPeptidesConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: args.sdrf.clone(),
            fasta: None,
        },
        output,
        filtering: FilterConfig {
            min_aa: args.min_aa.unwrap_or(defaults.min_aa),
            min_unique_peptides: args.min_unique.unwrap_or(defaults.min_unique_peptides),
            remove_contaminants: args.remove_decoy_contaminants,
        },
        remove_ids: args.remove_ids.clone(),
        remove_low_frequency_peptides: args.remove_low_frequency_peptides,
        keep_shared_peptides: args.keep_shared_peptides,
        skip_normalization: args.skip_normalization,
        run_normalization: args
            .run_normalization
            .clone()
            .map_or_else(|| "median".to_owned(), |value| value.replace('-', "_")),
        sample_normalization: args
            .sample_normalization
            .clone()
            .map_or_else(|| "globalmedian".to_owned(), |value| value.replace('-', "")),
        log2: args.log2,
        save_parquet: args.save_parquet,
        aggregation_level: if args.aggregation_level.eq_ignore_ascii_case("run") {
            AggregationLevel::Run
        } else {
            AggregationLevel::Sample
        },
        filter_pipeline,
        irs,
    }
}

fn build_filter_pipeline(
    args: &Features2PeptidesArgs,
) -> mokume_core::Result<Option<PreprocessingFilterConfig>> {
    let mut config = match &args.filter_config {
        Some(path) => load_filter_config(path)?,
        None if has_filter_override(args) => {
            let mut config = PreprocessingFilterConfig {
                name: "cli_config".to_string(),
                ..PreprocessingFilterConfig::default()
            };
            // A CLI-only override starts from the base filtering contract (explicit
            // --min-aa/--min-unique are applied below), not the preprocessing
            // defaults for settings such as --min_unique or contaminant removal.
            config.peptide.min_peptide_length = FilterConfig::default().min_aa;
            config.protein.min_unique_peptides = FilterConfig::default().min_unique_peptides;
            config.protein.remove_contaminants = args.remove_decoy_contaminants;
            config
        }
        None => return Ok(None),
    };
    apply_filter_overrides(args, &mut config)?;
    Ok(Some(config))
}

fn apply_filter_overrides(
    args: &Features2PeptidesArgs,
    config: &mut PreprocessingFilterConfig,
) -> mokume_core::Result<()> {
    if let Some(value) = args.filter_min_intensity {
        config.intensity.min_intensity = value;
    }
    if let Some(value) = args.filter_cv_threshold {
        config.intensity.cv_threshold = Some(value);
    }
    if !args.filter_charge_state.is_empty() {
        config.peptide.allowed_charge_states = Some(args.filter_charge_state.clone());
    }
    if let Some(value) = args.filter_max_missed_cleavages {
        config.peptide.max_missed_cleavages = Some(value);
    }
    if let Some(value) = &args.filter_score {
        config.peptide.score = Some(value.clone());
    }
    apply_fdr_overrides(args, config);
    apply_list_and_qc_overrides(args, config);
    if let Some(value) = args.min_aa {
        config.peptide.min_peptide_length = value;
    }
    if let Some(value) = args.min_unique {
        config.protein.min_unique_peptides = value;
    }
    Ok(())
}

fn apply_list_and_qc_overrides(
    args: &Features2PeptidesArgs,
    config: &mut PreprocessingFilterConfig,
) {
    if !args.filter_exclude_modification.is_empty() {
        config.peptide.exclude_modifications = args.filter_exclude_modification.clone();
    }
    if let Some(value) = args.filter_min_features {
        config.run_qc.min_identified_features = value;
    }
    if let Some(value) = args.filter_max_missing_rate {
        config.run_qc.max_missing_rate = value;
    }
}

fn has_filter_override(args: &Features2PeptidesArgs) -> bool {
    args.filter_min_intensity.is_some()
        || args.filter_cv_threshold.is_some()
        || !args.filter_charge_state.is_empty()
        || args.filter_max_missed_cleavages.is_some()
        || args.filter_peptide_fdr.is_some()
        || args.filter_score.is_some()
        || !args.filter_exclude_modification.is_empty()
        || args.filter_protein_fdr.is_some()
        || args.filter_min_features.is_some()
        || args.filter_max_missing_rate.is_some()
}

fn apply_fdr_overrides(args: &Features2PeptidesArgs, config: &mut PreprocessingFilterConfig) {
    if let Some(value) = args.filter_peptide_fdr {
        config.peptide.fdr_threshold = Some(value);
    }
    if let Some(value) = args.filter_protein_fdr {
        config.protein.fdr_threshold = Some(value);
    }
}

fn load_filter_config(path: &Path) -> mokume_core::Result<PreprocessingFilterConfig> {
    if !path.exists() {
        return Err(MokumeError::MissingInput {
            path: path.to_path_buf(),
        });
    }
    let text = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let is_json = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
    if is_json {
        serde_json::from_str(&text).map_err(|source| MokumeError::InvalidInput {
            message: format!(
                "failed to parse filter config '{}': {source}",
                path.display()
            ),
        })
    } else {
        serde_yaml::from_str(&text).map_err(|source| MokumeError::InvalidInput {
            message: format!(
                "failed to parse filter config '{}': {source}",
                path.display()
            ),
        })
    }
}

fn generate_filter_config(path: &Path) -> mokume_core::Result<()> {
    let content = if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    {
        EXAMPLE_FILTER_CONFIG_JSON
    } else {
        EXAMPLE_FILTER_CONFIG_YAML
    };
    write(path, content).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::build_filter_pipeline;
    use crate::{Cli, Commands, QuantifyCommands};

    #[test]
    fn cli_filter_override_preserves_base_filter_contract() {
        let cli = Cli::parse_from([
            "mokume",
            "quantify",
            "features2peptides",
            "--parquet",
            "input.parquet",
            "--output",
            "output.csv",
            "--min-unique",
            "5",
            "--filter-min-intensity",
            "0",
        ]);
        let Commands::Quantify(quantify) = cli.command else {
            panic!("expected quantify command");
        };
        let QuantifyCommands::Features2Peptides(args) = quantify.command else {
            panic!("expected features2peptides command");
        };
        let Ok(Some(pipeline)) = build_filter_pipeline(&args) else {
            panic!("expected an enabled valid filter pipeline");
        };
        assert_eq!(pipeline.peptide.min_peptide_length, 7);
        assert_eq!(pipeline.protein.min_unique_peptides, 5);
        assert!(!pipeline.protein.remove_contaminants);
    }

    #[test]
    fn generated_filter_config_is_exclusive() {
        let parsed = Cli::try_parse_from([
            "mokume",
            "quantify",
            "features2peptides",
            "--generate-filter-config",
            "filters.yaml",
            "--filter-min-intensity",
            "10",
        ]);
        assert!(parsed.is_err());
    }
}
