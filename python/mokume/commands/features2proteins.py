"""
CLI command for unified features to proteins pipeline.

This command provides a single-step workflow from feature-level parquet
files to protein intensities, supporting multiple quantification methods.
"""

import re

import click

from mokume.model.normalization import PeptideNormalizationMethod

from ._features2proteins_options import (
    _parse_de_contrasts,
    _resolved_imputation_method,
    _resolved_normalizations,
    _split_csv,
    _validate_batch_options,
    _validate_de_options,
    _validate_plot_options,
    _validate_quantification_options,
    _validate_reference_options,
)


# Build choices for sample normalization (including hierarchical)
SAMPLE_NORM_CHOICES = [p.name.lower() for p in PeptideNormalizationMethod]

# Quantification methods with a fixed name. ``top<N>`` is handled separately
# because N is part of the method name and can be any integer >= 1.
FIXED_QUANT_METHODS = [
    "directlfq",
    "pibaq",
    "maxlfq",
    "sum",
    "median",
    "ratio",
    "abd",
    "intensity",
    "peptide_count",
]

_TOPN_METHOD_RE = re.compile(r"^top(\d+)$")


class QuantMethodParamType(click.ParamType):
    """Accept the fixed quantification methods plus any ``top<N>`` form.

    ``top<N>`` (top1, top3, top5, top10, ...) carries N in the method name, so
    TopN needs no companion option. Bare ``topn`` keeps the placeholder letter
    and is normalized to ``top3``, the canonical named method (Silva 2006) and
    the factory's own default when a name carries no digits. A ``top`` name with
    no arabic numeral (``topa``) is rejected rather than silently defaulted.
    Matching is case-insensitive and values are normalized to lower case.
    """

    name = "quant_method"

    def get_metavar(self, param, ctx=None):
        return "[" + "|".join(FIXED_QUANT_METHODS) + "|top<N>]"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            self.fail(f"{value!r} is not a valid quantification method", param, ctx)

        normalized = value.strip().lower()
        if normalized in FIXED_QUANT_METHODS:
            return normalized

        # ``topn`` keeps the placeholder letter and means the canonical Top3
        # (Silva 2006). Normalizing it here leaves exactly one internal spelling
        # of every TopN request, so downstream only ever sees ``top<digits>``.
        if normalized == "topn":
            return "top3"

        match = _TOPN_METHOD_RE.match(normalized)
        if match:
            if int(match.group(1)) >= 1:
                return normalized
            self.fail(
                f"'{value}' is not a valid quantification method: N in 'top<N>' "
                "must be an integer >= 1 (e.g. top1, top3, top5).",
                param,
                ctx,
            )

        self.fail(
            f"'{value}' is not a valid quantification method. Choose one of "
            + ", ".join(FIXED_QUANT_METHODS)
            + ", or use 'top<N>' with an integer N >= 1 (e.g. top3, top5, top10).",
            param,
            ctx,
        )
        return None  # pragma: no cover - ``fail`` always raises


QUANT_METHOD = QuantMethodParamType()


@click.command(
    "features2proteins", short_help="Quantify proteins from feature parquet file."
)
@click.option(
    "-p",
    "--parquet",
    help="Parquet file (quantms.io/qpx format)",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "--msstats",
    help="Legacy quantms *_msstats_in.csv file (requires --sdrf)",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output",
    help="Output file for protein intensities",
    required=True,
    type=click.Path(),
)
@click.option(
    "-s",
    "--sdrf",
    help="SDRF file for sample metadata",
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--quant-method",
    "quant_method",
    help=(
        "Quantification method: directlfq, pibaq, maxlfq, top<N> (e.g. top3, "
        "top5, top10), sum, median, ratio, abd (TMT abundance), "
        "intensity (TMT reporter), peptide_count (distinct canonical peptides)"
    ),
    type=QUANT_METHOD,
    default="maxlfq",
    show_default=True,
)
# Filtering options
@click.option(
    "--min-aa",
    "min_aa",
    help="Minimum number of amino acids for peptides",
    type=int,
    default=7,
    show_default=True,
)
@click.option(
    "--min-unique",
    "min_unique",
    help="Minimum number of unique peptides per protein",
    type=int,
    default=2,
    show_default=True,
)
@click.option(
    "--remove-contaminants/--keep-contaminants",
    "remove_contaminants",
    help="Remove contaminants and decoys",
    default=True,
    show_default=True,
)
# Normalization options
@click.option(
    "--run-normalization",
    "run_normalization",
    help="Run/technical replicate normalization",
    type=click.Choice(
        ["none", "mean", "median", "max", "global", "iqr"],
        case_sensitive=False,
    ),
    default=None,
)
@click.option(
    "--sample-normalization",
    "sample_normalization",
    help="Sample normalization method. "
    "Use 'hierarchical' for DirectLFQ-style clustering-based normalization.",
    type=click.Choice(SAMPLE_NORM_CHOICES, case_sensitive=False),
    default=None,
)
@click.option(
    "--normalization-proteins",
    "normalization_proteins",
    help="File with protein IDs to use for normalization (one per line)",
    type=click.Path(exists=True),
    default=None,
)
# Method-specific options
@click.option(
    "--fasta",
    "fasta_file",
    help="FASTA file (required for piBAQ)",
    type=click.Path(exists=True),
    default=None,
)
# piBAQ-specific options (paralog-aware iBAQ)
@click.option(
    "--pibaq-enzyme",
    "pibaq_enzyme",
    help="Protease used to digest the FASTA for piBAQ",
    default="Trypsin",
    show_default=True,
)
@click.option(
    "--pibaq-max-aa",
    "pibaq_max_aa",
    help="Maximum peptide length from the FASTA digest for piBAQ",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
)
@click.option(
    "--pibaq-min-shared",
    "pibaq_min_shared",
    help="Minimum distinct peptides two proteins must share to co-cluster "
    "into one piBAQ family",
    type=int,
    default=2,
    show_default=True,
)
@click.option(
    "--pibaq-families",
    "pibaq_families_yaml",
    help="YAML file with explicit piBAQ family overrides",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--pibaq-min-anchors",
    "pibaq_min_anchors",
    help="Unique-anchor threshold; if no family member reaches it, shared "
    "signal is split equally",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--pibaq-high-anchor-threshold",
    "pibaq_high_anchor_threshold",
    help="Minimum anchor count (weakest member) for a family to be labelled "
    "EvidenceLevel='high'",
    type=int,
    default=3,
    show_default=True,
)
# LFQ-specific options
@click.option(
    "--stabilize",
    is_flag=True,
    default=False,
    help="Enable large-ratio stabilization (MaxLFQ only; default: off)",
)
@click.option(
    "--maxlfq-min-ratio-count",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="MaxLFQ: minimum shared peptide species per sample pair",
)
@click.option(
    "--directlfq-cores",
    "directlfq_cores",
    help="Number of CPU cores for DirectLFQ",
    type=click.IntRange(min=1),
    default=None,
)
@click.option(
    "--directlfq-min-nonan",
    "directlfq_min_nonan",
    help="Minimum non-NaN values for DirectLFQ",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
# Optional exports
@click.option(
    "--export-peptides",
    "export_peptides",
    help="Export normalized peptides to this file (for debugging/analysis)",
    type=click.Path(),
    default=None,
)
@click.option(
    "--export-ions",
    "export_ions",
    help="Export the normalized DirectLFQ ion matrix to this CSV file",
    type=click.Path(),
    default=None,
)
@click.option(
    "--batch-correction",
    "batch_correction",
    help="Enable ComBat batch correction after quantification",
    is_flag=True,
    default=False,
)
@click.option(
    "--batch-method",
    "batch_method",
    help="Batch detection method",
    type=click.Choice(["sample_prefix", "column"], case_sensitive=False),
    default="sample_prefix",
    show_default=True,
)
@click.option(
    "--batch-column",
    "batch_column",
    help="SDRF column to use when --batch-method=column",
    default=None,
)
@click.option(
    "--batch-covariates",
    "batch_covariates",
    help="Comma-separated SDRF columns to preserve as biological covariates",
    default=None,
)
@click.option(
    "--batch-parametric/--batch-nonparametric",
    "batch_parametric",
    help="Use parametric or non-parametric ComBat estimation",
    default=True,
    show_default=True,
)
@click.option(
    "--batch-mean-only",
    "batch_mean_only",
    help="Only adjust batch means, not individual effects",
    is_flag=True,
    default=False,
)
@click.option(
    "--batch-ref",
    "batch_ref",
    help="Reference batch ID for ComBat",
    type=int,
    default=None,
)
# IRS normalization options
@click.option(
    "--irs",
    "irs",
    help="Enable IRS (Internal Reference Scaling) normalization for multi-plex TMT data",
    is_flag=True,
    default=False,
)
@click.option(
    "--irs-reference-samples",
    "irs_reference_samples",
    help="Comma-separated list of reference sample names (source names)",
    default=None,
)
@click.option(
    "--irs-sdrf-column",
    "irs_sdrf_column",
    help="SDRF column to select reference samples from (e.g., 'factor value[disease]')",
    default=None,
)
@click.option(
    "--irs-sdrf-values",
    "irs_sdrf_values",
    help="Comma-separated values in SDRF column that indicate reference samples",
    default=None,
)
@click.option(
    "--irs-reference-regex",
    "irs_reference_regex",
    help="Regex to auto-detect reference samples across SDRF columns",
    default="pool|powder|ref|reference|bridge",
    show_default=True,
)
@click.option(
    "--irs-stat",
    "irs_stat",
    help="Statistic for computing reference intensity per plex",
    type=click.Choice(["median", "mean"], case_sensitive=False),
    default="median",
    show_default=True,
)
@click.option(
    "--irs-remove-reference",
    "irs_remove_reference",
    help="Remove reference samples from the final output",
    is_flag=True,
    default=False,
)
# Coverage filter
@click.option(
    "--coverage-threshold",
    "coverage_threshold",
    help="Minimum fraction of non-missing values per condition to keep a protein (e.g., 0.65)",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
)
@click.option(
    "--min-sample-correlation",
    "sample_correlation_threshold",
    help="Drop samples below mean Pearson correlation to same-condition peers",
    type=click.FloatRange(min=-1.0, max=1.0),
    default=None,
)
# Ratio quantification options
@click.option(
    "--ratio-fraction-merge",
    "ratio_fraction_merge",
    help="How to merge fractions in ratio quantification: mean (PS protocol) or max",
    type=click.Choice(["mean", "max"], case_sensitive=False),
    default="mean",
    show_default=True,
)
# Imputation options
@click.option(
    "--impute",
    "impute",
    help="Enable missing-value imputation on the protein matrix",
    is_flag=True,
    default=False,
)
@click.option(
    "--impute-method",
    "impute_method",
    help="Imputation method (operates in log2 space)",
    type=click.Choice(
        [
            "knn",
            "minprob",
            "mindet",
            "qrilc",
            "missforest",
            "seqknn",
            "impseq",
        ],
        case_sensitive=False,
    ),
    default=None,
)
@click.option(
    "--impute-quantile",
    "impute_quantile",
    help="Observed-value quantile used by MinProb and MinDet",
    type=float,
    default=0.01,
    show_default=True,
)
@click.option(
    "--impute-seed",
    "impute_seed",
    help="Seed of the MinProb/QRILC draws",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
)
@click.option(
    "--impute-tune-sigma",
    "impute_tune_sigma",
    help="MinProb/QRILC multiplier of the fitted standard deviation",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    "--impute-n-neighbors",
    "impute_n_neighbors",
    help="Number of neighbours for KNN/SeqKNN imputation",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
)
# Differential expression options
@click.option(
    "--de",
    "differential_expression",
    help="Enable differential expression analysis",
    is_flag=True,
    default=False,
)
@click.option(
    "--de-contrasts",
    "de_contrasts",
    help="Comma-separated contrasts (e.g., 'A vs B,A vs C' or 'A-B,A-C')",
    default=None,
)
@click.option(
    "--de-contrasts-file",
    "de_contrasts_file",
    help="TSV file with contrasts (columns: group1, group2)",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--de-method",
    "de_method",
    help="DE statistical method (use 'ensemble' for top-k consensus across methods)",
    type=click.Choice(
        [
            "auto",
            "limrots",
            "limma",
            "deqms",
            "proda",
            "rots",
            "ensemble",
        ],
        case_sensitive=False,
    ),
    default="auto",
    show_default=True,
)
@click.option(
    "--de-ensemble-methods",
    "de_ensemble_methods",
    help="Comma-separated DE methods used by --de-method=ensemble (default: limrots,deqms,proda)",
    default=None,
)
@click.option(
    "--de-ensemble-min-k",
    "de_ensemble_min_k",
    help="Minimum number of ensemble members that must agree on direction",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
)
@click.option(
    "--de-log2fc",
    "de_log2fc_threshold",
    help="Minimum absolute log2 fold change for significance",
    type=float,
    default=0.5,
    show_default=True,
)
@click.option(
    "--de-fdr",
    "de_fdr_threshold",
    help="Maximum adjusted p-value (FDR) for significance",
    type=float,
    default=0.05,
    show_default=True,
)
@click.option(
    "--de-fdr-method",
    "de_fdr_method",
    help=(
        "FDR correction method: bh (Benjamini-Hochberg), ihw, or the adaptive-pi0 "
        "procedures bky (Benjamini-Krieger-Yekutieli) / storey (q-values), which "
        "fall back to bh when pi0 is untrustworthy"
    ),
    type=click.Choice(["bh", "ihw", "bky", "storey"], case_sensitive=False),
    default="bh",
    show_default=True,
)
@click.option(
    "--de-output",
    "de_output",
    help="Output file for DE results",
    type=click.Path(),
    default=None,
)
# Plotting options
@click.option(
    "--plot-dir",
    "plot_output_dir",
    help="Output directory for plots",
    type=click.Path(),
    default=None,
)
@click.option(
    "--plot-volcano",
    "plot_volcano",
    help="Generate volcano plot from DE results",
    is_flag=True,
    default=False,
)
@click.option(
    "--plot-heatmap",
    "plot_heatmap",
    help="Generate heatmap of top variable proteins",
    is_flag=True,
    default=False,
)
@click.option(
    "--plot-pca",
    "plot_pca",
    help="Generate PCA plot colored by condition",
    is_flag=True,
    default=False,
)
@click.option(
    "--highlight-genes",
    "highlight_genes",
    help="Comma-separated gene names to highlight on volcano plot",
    default=None,
)
# Interactive report options
@click.option(
    "--interactive-report",
    "interactive_report",
    help="Generate interactive HTML report with plotly (requires mokume-py[reports])",
    is_flag=True,
    default=False,
)
@click.option(
    "--report-output",
    "report_output",
    help="Output path for interactive HTML report (default: <plot-dir>/report_<contrast>.html)",
    type=click.Path(),
    default=None,
)
# DuckDB resource limits (propagated through to mokume.io.feature.Feature)
@click.option(
    "--duckdb-memory",
    "duckdb_memory",
    help=(
        "DuckDB memory limit (e.g. '80GB', '16384MB'). Caps DuckDB's "
        "internal buffer pool only -- the surrounding Python process "
        "(PyArrow / polars / pandas) is not limited by this value and "
        "peak RSS can be 2-3x larger on wide pivots. Use cgroup MemoryMax, "
        "SLURM --mem, or container resources.limits for a hard cap. "
        "Default: DuckDB autoconfig (~80%% of total RAM)."
    ),
    default=None,
)
@click.option(
    "--duckdb-threads",
    "duckdb_threads",
    help="Number of threads DuckDB may use. Default: all available cores.",
    type=click.IntRange(min=1),
    default=None,
)
@click.pass_context
def features2proteins(
    ctx,
    parquet: str,
    msstats: str,
    output: str,
    sdrf: str,
    quant_method: str,
    min_aa: int,
    min_unique: int,
    remove_contaminants: bool,
    run_normalization: str,
    sample_normalization: str,
    normalization_proteins: str,
    fasta_file: str,
    pibaq_enzyme: str,
    pibaq_max_aa: int,
    pibaq_min_shared: int,
    pibaq_families_yaml: str,
    pibaq_min_anchors: int,
    pibaq_high_anchor_threshold: int,
    directlfq_cores: int,
    directlfq_min_nonan: int,
    maxlfq_min_ratio_count: int,
    stabilize: bool,
    export_peptides: str,
    export_ions: str,
    # Batch correction
    batch_correction: bool,
    batch_method: str,
    batch_column: str,
    batch_covariates: str,
    batch_parametric: bool,
    batch_mean_only: bool,
    batch_ref: int,
    # IRS
    irs: bool,
    irs_reference_samples: str,
    irs_sdrf_column: str,
    irs_sdrf_values: str,
    irs_reference_regex: str,
    irs_stat: str,
    irs_remove_reference: bool,
    # Coverage filter
    coverage_threshold: float,
    # Sample correlation QC
    sample_correlation_threshold: float,
    # Ratio
    ratio_fraction_merge: str,
    # Imputation
    impute: bool,
    impute_method: str,
    impute_quantile: float,
    impute_seed: int,
    impute_tune_sigma: float,
    impute_n_neighbors: int,
    # DE
    differential_expression: bool,
    de_contrasts: str,
    de_contrasts_file: str,
    de_method: str,
    de_ensemble_methods: str,
    de_ensemble_min_k: int,
    de_log2fc_threshold: float,
    de_fdr_threshold: float,
    de_fdr_method: str,
    de_output: str,
    # Plots
    plot_output_dir: str,
    plot_volcano: bool,
    plot_heatmap: bool,
    plot_pca: bool,
    highlight_genes: str,
    # Interactive report
    interactive_report: bool,
    report_output: str,
    # DuckDB resource limits (cap DuckDB engine only, NOT total process RSS)
    duckdb_memory: str,
    duckdb_threads: int,
) -> None:
    """
    Quantify proteins from QPX feature parquet or legacy SDRF+MSstats input.

    This is the recommended unified command that handles the full pipeline
    from features to proteins in one step.

    \b
    QUANTIFICATION METHODS:
      directlfq  - DirectLFQ (uses directlfq package for everything)
      pibaq      - Paralog-aware iBAQ with shared-peptide allocation (requires --fasta)
      maxlfq     - MaxLFQ algorithm
      top<N>     - Average of the N most intense peptides, N taken from the
                   method name (top3 = Silva 2006, also top1/top5/top10/...)
      sum        - Sum of all peptides
      median     - Median of peptides
      peptide_count - Distinct canonical peptides per protein/sample
      ratio      - PS protocol: log2(sample/reference) per plex (requires --sdrf)

    \b
    NORMALIZATION:
      DirectLFQ, Ratio, and peptide_count require normalization 'none'.
      For other methods, mokume applies normalization:
      - Run normalization: normalizes technical replicates within samples
      - Sample normalization: normalizes samples relative to each other
        Use 'hierarchical' for DirectLFQ-style clustering normalization
        combined with other quantification methods (e.g., piBAQ).

    \b
    IRS NORMALIZATION (multi-plex TMT):
      Use --irs to enable Internal Reference Scaling. Reference channels
      are auto-detected from SDRF 'characteristics[pooled sample]' column,
      or specified via --irs-reference-samples, --irs-sdrf-column/values,
      or --irs-reference-regex.

    \b
    DIFFERENTIAL EXPRESSION:
      Use --de to enable DE analysis. Contrasts must be specified
      via --de-contrasts (e.g., 'NASH vs HL') or --de-contrasts-file (TSV).

    \b
    EXAMPLES:
      # TMT with IRS normalization + DE + volcano plot
      mokume features2proteins -p data.parquet -o proteins.csv -s sdrf.tsv \\
        --quant-method median --irs --irs-remove-reference \\
        --de --de-contrasts NASH-HL --de-output de_results.csv \\
        --plot-dir plots/ --plot-volcano --plot-pca

      # DirectLFQ quantification (uses directlfq package)
      mokume features2proteins -p data.parquet -o proteins.csv --quant-method directlfq

      # piBAQ with hierarchical normalization
      mokume features2proteins -p data.parquet -o proteins.csv \\
        --quant-method pibaq --sample-normalization hierarchical --fasta uniprot.fasta

      # Ratio quantification (PS protocol) with coverage filter + DE
      mokume features2proteins -p data.parquet -o proteins.csv -s sdrf.tsv \\
        --quant-method ratio --coverage-threshold 0.65 \\
        --de --de-method deqms --de-contrasts NASH-HL
    """
    from mokume.pipeline import features_to_proteins as run_pipeline

    quant_method_lower, run_normalization, sample_normalization = (
        _resolved_normalizations(ctx)
    )
    _validate_quantification_options(ctx, quant_method_lower)
    _validate_batch_options(ctx)
    _validate_reference_options(ctx, quant_method_lower)
    impute_method = _resolved_imputation_method(ctx)
    _validate_de_options(ctx, quant_method_lower)
    _validate_plot_options(ctx)

    # 'top<N>' carries N in the method name; the engine parses it out.
    topn_match = _TOPN_METHOD_RE.match(quant_method)
    if topn_match:
        click.echo(f"Using Top{int(topn_match.group(1))} quantification method")

    # Parse comma-separated CLI values
    parsed_irs_ref_samples = _split_csv(irs_reference_samples)
    parsed_irs_sdrf_values = _split_csv(irs_sdrf_values)
    parsed_batch_covariates = _split_csv(batch_covariates)
    parsed_de_contrasts = _parse_de_contrasts(de_contrasts, de_contrasts_file)
    parsed_highlight_genes = _split_csv(highlight_genes)
    parsed_de_ensemble_methods = (
        [item.strip() for item in de_ensemble_methods.split(",")]
        if de_ensemble_methods is not None
        else None
    )

    # Run the pipeline
    run_pipeline(
        parquet=parquet,
        msstats=msstats,
        output=output,
        sdrf=sdrf,
        quant_method=quant_method,
        min_aa=min_aa,
        min_unique_peptides=min_unique,
        remove_contaminants=remove_contaminants,
        run_normalization=run_normalization,
        sample_normalization=sample_normalization,
        normalization_proteins_file=normalization_proteins,
        fasta_file=fasta_file,
        pibaq_enzyme=pibaq_enzyme,
        pibaq_max_aa=pibaq_max_aa,
        pibaq_min_shared=pibaq_min_shared,
        pibaq_families_yaml=pibaq_families_yaml,
        pibaq_min_anchors=pibaq_min_anchors,
        pibaq_high_anchor_threshold=pibaq_high_anchor_threshold,
        directlfq_num_cores=directlfq_cores,
        directlfq_min_nonan=directlfq_min_nonan,
        maxlfq_min_ratio_count=maxlfq_min_ratio_count,
        stabilize=stabilize,
        export_peptides=export_peptides,
        export_ions=export_ions,
        # Batch correction
        batch_correction=batch_correction,
        batch_method=batch_method,
        batch_column=batch_column,
        batch_covariates=parsed_batch_covariates,
        batch_parametric=batch_parametric,
        batch_mean_only=batch_mean_only,
        batch_ref=batch_ref,
        # IRS
        irs=irs,
        irs_reference_samples=parsed_irs_ref_samples,
        irs_sdrf_column=irs_sdrf_column,
        irs_sdrf_values=parsed_irs_sdrf_values,
        irs_reference_regex=irs_reference_regex,
        irs_stat=irs_stat,
        irs_remove_reference=irs_remove_reference,
        # DE
        differential_expression=differential_expression,
        de_contrasts=parsed_de_contrasts,
        de_method=de_method,
        de_log2fc_threshold=de_log2fc_threshold,
        de_fdr_threshold=de_fdr_threshold,
        de_fdr_method=de_fdr_method,
        de_output=de_output,
        de_ensemble_methods=parsed_de_ensemble_methods,
        de_ensemble_min_k=de_ensemble_min_k,
        # Coverage filter
        coverage_threshold=coverage_threshold,
        # Sample correlation QC
        sample_correlation_threshold=sample_correlation_threshold,
        # Ratio
        ratio_fraction_merge=ratio_fraction_merge,
        # Imputation
        impute=impute,
        impute_method=impute_method,
        impute_quantile=impute_quantile,
        impute_seed=impute_seed,
        impute_tune_sigma=impute_tune_sigma,
        impute_n_neighbors=impute_n_neighbors,
        # Plots
        plot_output_dir=plot_output_dir,
        plot_volcano=plot_volcano,
        plot_heatmap=plot_heatmap,
        plot_pca=plot_pca,
        highlight_genes=parsed_highlight_genes,
        # Interactive report
        interactive_report=interactive_report,
        report_output=report_output,
        # DuckDB resource limits (cap DuckDB engine only, NOT total process RSS)
        duckdb_memory=duckdb_memory,
        duckdb_threads=duckdb_threads,
    )

    click.echo(f"Protein intensities saved to: {output}")
