"""
CLI command for converting features to peptides.
"""

import math

import click

from mokume.normalization.peptide import peptide_normalization
from mokume.preprocessing.filters.io import generate_example_config, load_filter_config
from ..model.filters import NamedScoreFilterConfig, PreprocessingFilterConfig


def _parse_named_score_option(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> NamedScoreFilterConfig | None:
    if value is None:
        return None
    try:
        name, raw_threshold = value.rsplit("=", maxsplit=1)
        threshold = float(raw_threshold)
    except (ValueError, TypeError) as exc:
        raise click.BadParameter("expected NAME=THRESHOLD") from exc
    name = name.strip()
    if not name:
        raise click.BadParameter("score name cannot be empty")
    if not math.isfinite(threshold):
        raise click.BadParameter("threshold must be finite")
    return NamedScoreFilterConfig(name=name, threshold=threshold)


def _command_line_value(ctx: click.Context, name: str) -> bool:
    return ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE


def _validate_required_io(ctx: click.Context) -> None:
    params = ctx.params
    if params["parquet"] is None:
        raise click.UsageError(
            "--parquet is required unless --generate-filter-config is used"
        )
    if params["output"] is None:
        raise click.UsageError(
            "--output is required unless --generate-filter-config is used"
        )


def _validate_normalization_options(ctx: click.Context) -> None:
    params = ctx.params
    irs_names = ("irs_channel", "irs_autodetect_regex", "irs_stat", "irs_scope")
    if params["skip_normalization"] and any(
        _command_line_value(ctx, name) for name in irs_names
    ):
        raise click.UsageError(
            "--skip_normalization conflicts with features2peptides IRS options"
        )
    active_normalization = any(
        _command_line_value(ctx, name) and params[name].lower() != "none"
        for name in ("run_normalization", "sample_normalization")
    )
    if params["skip_normalization"] and active_normalization:
        raise click.UsageError(
            "--skip_normalization conflicts with active run/sample normalization"
        )


def _validate_irs_options(ctx: click.Context) -> None:
    params = ctx.params
    if params["irs_channel"] and params["irs_autodetect_regex"]:
        raise click.UsageError(
            "Choose either --irs_channel or --irs_autodetect_regex, not both"
        )
    if params["irs_autodetect_regex"] and not params["sdrf"]:
        raise click.UsageError("--irs_autodetect_regex requires --sdrf")
    if not (params["irs_channel"] or params["irs_autodetect_regex"]) and any(
        _command_line_value(ctx, name) for name in ("irs_stat", "irs_scope")
    ):
        raise click.UsageError(
            "--irs_stat/--irs_scope require --irs_channel or --irs_autodetect_regex"
        )


def _validate_features2peptides_options(ctx: click.Context) -> None:
    _validate_required_io(ctx)
    _validate_normalization_options(ctx)
    _validate_irs_options(ctx)


def _filter_overrides(
    ctx: click.Context,
    peptide_fdr: float | None = None,
    score: NamedScoreFilterConfig | None = None,
    protein_fdr: float | None = None,
    max_missing_rate: float | None = None,
) -> dict:
    params = ctx.params
    overrides = {}
    direct = {
        "filter_min_intensity": "min_intensity",
        "filter_cv_threshold": "cv_threshold",
        "filter_max_missed_cleavages": "max_missed_cleavages",
        "filter_min_unique_peptides": "min_unique_peptides",
        "filter_min_features": "min_features",
    }
    for source, target in direct.items():
        if params[source] is not None:
            overrides[target] = params[source]
    if peptide_fdr is not None:
        overrides["peptide_fdr"] = peptide_fdr
    if score is not None:
        overrides["score"] = score
    if protein_fdr is not None:
        overrides["protein_fdr"] = protein_fdr
    if max_missing_rate is not None:
        overrides["max_missing_rate"] = max_missing_rate
    if params["filter_charge_states"] is not None:
        overrides["charge_states"] = [
            int(value.strip()) for value in params["filter_charge_states"].split(",")
        ]
    if params["filter_exclude_modifications"] is not None:
        overrides["exclude_modifications"] = [
            value.strip() for value in params["filter_exclude_modifications"].split(",")
        ]
    return overrides


def _preprocessing_config(
    ctx: click.Context,
    peptide_fdr: float | None = None,
    score: NamedScoreFilterConfig | None = None,
    protein_fdr: float | None = None,
    max_missing_rate: float | None = None,
) -> PreprocessingFilterConfig | None:
    overrides = _filter_overrides(
        ctx, peptide_fdr, score, protein_fdr, max_missing_rate
    )
    config_path = ctx.params["filter_config"]
    if config_path:
        config = load_filter_config(config_path)
    elif overrides:
        config = PreprocessingFilterConfig(name="cli_config")
    else:
        return None
    if overrides:
        config.apply_overrides(overrides)
    return config


def _resolved_filter_thresholds(
    ctx: click.Context,
    config: PreprocessingFilterConfig | None,
) -> tuple[int, int]:
    if ctx.params["filter_min_unique_peptides"] is not None and _command_line_value(
        ctx, "min_unique"
    ):
        raise click.UsageError(
            "Choose either --min_unique or --filter-min-unique-peptides, not both"
        )
    min_aa = ctx.params["min_aa"]
    min_unique = ctx.params["min_unique"]
    if config is not None and not _command_line_value(ctx, "min_aa"):
        min_aa = config.peptide.min_peptide_length
    if config is not None and not _command_line_value(ctx, "min_unique"):
        min_unique = config.protein.min_unique_peptides
    return min_aa, min_unique


@click.command("features2peptides", short_help="Convert features to parquet file.")
@click.option(
    "-p",
    "--parquet",
    help="Parquet file import generated by quantms.io",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "-s",
    "--sdrf",
    help="SDRF file import generated by quantms",
    default=None,
    type=click.Path(),
)
@click.option(
    "--min_aa",
    help="Minimum number of amino acids to filter peptides",
    type=int,
    default=7,
)
@click.option(
    "--min_unique",
    help="Minimum number of unique peptides to filter proteins",
    default=2,
    type=int,
)
@click.option(
    "--keep-shared-peptides",
    "keep_shared_peptides",
    help="Keep shared/non-unique peptide rows and skip the unique-peptide gate",
    is_flag=True,
    default=False,
)
@click.option(
    "--remove_ids",
    help="Remove specific protein ids from the analysis using a file with one id per line",
    type=click.Path(exists=True),
)
@click.option(
    "--remove_decoy_contaminants",
    help="Remove decoy and contaminants proteins from the analysis",
    is_flag=True,
    default=False,
)
@click.option(
    "--remove_low_frequency_peptides",
    help="Remove peptides that are present in less than 20% of the samples",
    is_flag=True,
    default=False,
)
@click.option(
    "-o",
    "--output",
    help="Peptide intensity file including other all properties for normalization",
    type=click.Path(),
)
@click.option(
    "--skip_normalization", help="Skip normalization step", is_flag=True, default=False
)
@click.option(
    "--run-normalization",
    "run_normalization",
    help="Normalization method for technical replicates/runs within each sample",
    default="median",
    type=click.Choice(
        ["none", "mean", "median", "max", "global", "iqr"],
        case_sensitive=False,
    ),
)
@click.option(
    "--sample-normalization",
    "sample_normalization",
    help="Streaming sample normalization method",
    default="globalMedian",
    type=click.Choice(
        ["none", "globalmedian", "conditionmedian"], case_sensitive=False
    ),
)
@click.option(
    "--log2",
    help="Transform normalized peptide intensities to log2 before writing",
    is_flag=True,
)
@click.option(
    "--save_parquet",
    help="Save normalized peptides to parquet",
    is_flag=True,
)
@click.option(
    "--irs_channel",
    help="IRS reference channel label for TMT/ITRAQ normalization (e.g., TMT126)",
    default=None,
)
@click.option(
    "--irs_autodetect_regex",
    help="Regex to autodetect pooled/reference sample in SDRF to derive IRS channel",
    default=None,
)
@click.option(
    "--irs_stat",
    help="Statistic for IRS per-run metric",
    type=click.Choice(["median", "mean"], case_sensitive=False),
    default="median",
)
@click.option(
    "--irs_scope",
    help="IRS scaling scope",
    type=click.Choice(["global", "by_mixture", "two_stage"], case_sensitive=False),
    default="global",
)
@click.option(
    "--aggregation_level",
    help="Level at which to aggregate intensities: 'sample' (default) aggregates runs to samples, "
    "'run' preserves run-level granularity for downstream quantification (similar to DIA-NN)",
    type=click.Choice(["sample", "run"], case_sensitive=False),
    default="sample",
)
# Filter configuration options
@click.option(
    "--filter-config",
    "filter_config",
    type=click.Path(exists=True),
    help="Path to YAML/JSON filter configuration file",
)
@click.option(
    "--generate-filter-config",
    "generate_filter_config",
    type=click.Path(),
    help="Generate example filter config file at specified path and exit",
)
# Filter override options (override config file values)
@click.option(
    "--filter-min-intensity",
    "filter_min_intensity",
    type=float,
    help="Override: minimum intensity threshold",
)
@click.option(
    "--filter-cv-threshold",
    "filter_cv_threshold",
    type=float,
    help="Override: maximum CV threshold across replicates",
)
@click.option(
    "--filter-charge-states",
    "filter_charge_states",
    type=str,
    help="Override: allowed charge states (comma-separated, e.g., '2,3,4')",
)
@click.option(
    "--filter-max-missed-cleavages",
    "filter_max_missed_cleavages",
    type=int,
    help="Override: maximum missed cleavages",
)
@click.option(
    "--filter-peptide-fdr",
    "filter_peptide_fdr",
    type=click.FloatRange(0.0, 1.0),
    help="Override: maximum QPX peptide q-value",
)
@click.option(
    "--filter-score",
    "filter_score",
    metavar="NAME=THRESHOLD",
    callback=_parse_named_score_option,
    help="Filter one exact QPX additional score using its higher_better direction",
)
@click.option(
    "--filter-exclude-modifications",
    "filter_exclude_modifications",
    type=str,
    help="Override: modifications to exclude (comma-separated, e.g., 'Oxidation,Deamidated')",
)
@click.option(
    "--filter-min-unique-peptides",
    "filter_min_unique_peptides",
    type=int,
    help="Override: minimum unique peptides per protein",
)
@click.option(
    "--filter-protein-fdr",
    "filter_protein_fdr",
    type=click.FloatRange(0.0, 1.0),
    help="Override: maximum QPX protein-group q-value",
)
@click.option(
    "--filter-min-features",
    "filter_min_features",
    type=int,
    help="Override: minimum identified features per run",
)
@click.option(
    "--filter-max-missing-rate",
    "filter_max_missing_rate",
    type=click.FloatRange(0.0, 1.0),
    help="Override: maximum missing feature fraction per run",
)
@click.option(
    "--run",
    type=click.Path(exists=True),
    help="QPX run.parquet sample mapping; conflicts with --sdrf",
)
@click.pass_context
def features2parquet(
    ctx,
    parquet: str,
    sdrf: str,
    run: str,
    min_aa: int,
    min_unique: int,
    keep_shared_peptides: bool,
    remove_ids: str,
    remove_decoy_contaminants: bool,
    remove_low_frequency_peptides: bool,
    output: str,
    skip_normalization: bool,
    run_normalization: str,
    sample_normalization: str,
    log2: bool,
    save_parquet: bool,
    irs_channel: str,
    irs_autodetect_regex: str,
    irs_stat: str,
    irs_scope: str,
    aggregation_level: str,
    # Filter options
    filter_config: str,
    generate_filter_config: str,
    filter_min_intensity: float,
    filter_cv_threshold: float,
    filter_charge_states: str,
    filter_max_missed_cleavages: int,
    filter_peptide_fdr: float,
    filter_score: NamedScoreFilterConfig,
    filter_exclude_modifications: str,
    filter_min_unique_peptides: int,
    filter_protein_fdr: float,
    filter_min_features: int,
    filter_max_missing_rate: float,
) -> None:
    """
    Convert feature data to a parquet file with optional normalization and filtering steps.

    Supports filter configuration via YAML/JSON files or CLI overrides.
    Use --generate-filter-config to create an example configuration file.
    """
    if generate_filter_config:
        generate_example_config(generate_filter_config)
        click.echo(f"Generated example filter config at: {generate_filter_config}")
        return

    _validate_features2peptides_options(ctx)
    preprocessing_config = _preprocessing_config(
        ctx,
        filter_peptide_fdr,
        filter_score,
        filter_protein_fdr,
        filter_max_missing_rate,
    )
    min_aa, min_unique = _resolved_filter_thresholds(ctx, preprocessing_config)

    peptide_normalization(
        parquet=parquet,
        sdrf=sdrf,
        run=run,
        min_aa=min_aa,
        min_unique=min_unique,
        keep_shared_peptides=keep_shared_peptides,
        remove_ids=remove_ids,
        remove_decoy_contaminants=remove_decoy_contaminants,
        remove_low_frequency_peptides=remove_low_frequency_peptides,
        output=output,
        skip_normalization=skip_normalization,
        nmethod=run_normalization,
        pnmethod=sample_normalization,
        log2=log2,
        save_parquet=save_parquet,
        irs_channel=irs_channel,
        irs_autodetect_regex=irs_autodetect_regex,
        irs_stat=irs_stat,
        irs_scope=irs_scope,
        aggregation_level=aggregation_level,
        filter_config=preprocessing_config,
    )
