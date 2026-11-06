"""Validation and value parsing for the features2proteins CLI."""

import csv

import click


def _supplied(ctx: click.Context, *names: str) -> bool:
    return any(
        ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
        for name in names
    )


def _resolved_normalizations(ctx: click.Context) -> tuple[str, str, str]:
    params = ctx.params
    quant_method = params["quant_method"].lower()
    manages_normalization = quant_method in {"directlfq", "ratio"}
    is_count_method = quant_method == "peptide_count"
    requires_none = manages_normalization or is_count_method
    run_method = params["run_normalization"] or ("none" if requires_none else "median")
    sample_method = params["sample_normalization"] or (
        "none" if requires_none else "globalmedian"
    )
    explicitly_active = any(
        _supplied(ctx, name) and method.lower() != "none"
        for name, method in (
            ("run_normalization", run_method),
            ("sample_normalization", sample_method),
        )
    )
    if requires_none and explicitly_active:
        reason = (
            "does not use intensity normalization"
            if is_count_method
            else "manages normalization internally"
        )
        raise click.UsageError(
            f"{params['quant_method']} {reason}; "
            "use explicit 'none' or omit the normalization options"
        )
    if params["normalization_proteins"] and sample_method.lower() != "hierarchical":
        raise click.UsageError(
            "--normalization-proteins requires hierarchical sample normalization "
            "in mokume-py"
        )
    return quant_method, run_method, sample_method


def _validate_quantification_input(ctx: click.Context) -> None:
    params = ctx.params
    if (params["parquet"] is None) == (params["msstats"] is None):
        raise click.UsageError("Provide exactly one of --parquet or --msstats")
    if params["msstats"] and not params["sdrf"]:
        raise click.UsageError("--msstats requires --sdrf")


def _validate_pibaq_options(ctx: click.Context, quant_method: str) -> None:
    params = ctx.params
    pibaq_options = (
        "fasta_file",
        "pibaq_enzyme",
        "pibaq_max_aa",
        "pibaq_min_shared",
        "pibaq_families_yaml",
        "pibaq_min_anchors",
        "pibaq_high_anchor_threshold",
    )
    if quant_method == "pibaq" and not params["fasta_file"]:
        raise click.UsageError("piBAQ quantification requires --fasta option")
    if quant_method != "pibaq" and _supplied(ctx, *pibaq_options):
        raise click.UsageError(
            "piBAQ FASTA/digestion options require --quant-method pibaq"
        )


def _validate_directlfq_options(ctx: click.Context, quant_method: str) -> None:
    if quant_method != "maxlfq" and ctx.params.get("stabilize", False):
        raise click.UsageError("--stabilize requires --quant-method maxlfq")
    if quant_method != "maxlfq" and _supplied(ctx, "maxlfq_min_ratio_count"):
        raise click.UsageError(
            "--maxlfq-min-ratio-count requires --quant-method maxlfq"
        )
    if quant_method != "directlfq" and _supplied(
        ctx, "directlfq_cores", "directlfq_min_nonan", "export_ions"
    ):
        raise click.UsageError(
            "--directlfq-cores/--directlfq-min-nonan/--export-ions require "
            "--quant-method directlfq"
        )


def _validate_ratio_and_output_options(ctx: click.Context, quant_method: str) -> None:
    params = ctx.params
    if params["export_peptides"] and quant_method in {"directlfq", "ratio"}:
        raise click.UsageError(
            f"--export-peptides is not supported by the "
            f"{params['quant_method']} pipeline"
        )
    if quant_method == "ratio" and not params["sdrf"]:
        raise click.UsageError("Ratio quantification requires --sdrf option")
    if quant_method != "ratio" and _supplied(ctx, "ratio_fraction_merge"):
        raise click.UsageError("--ratio-fraction-merge requires --quant-method ratio")
    if params["coverage_threshold"] is not None and not params["sdrf"]:
        raise click.UsageError("--coverage-threshold requires --sdrf")
    if params["sample_correlation_threshold"] is not None and not params["sdrf"]:
        raise click.UsageError("--min-sample-correlation requires --sdrf")


def _validate_quantification_options(ctx: click.Context, quant_method: str) -> None:
    _validate_quantification_input(ctx)
    _validate_pibaq_options(ctx, quant_method)
    _validate_directlfq_options(ctx, quant_method)
    _validate_ratio_and_output_options(ctx, quant_method)


def _validate_batch_options(ctx: click.Context) -> None:
    params = ctx.params
    if (
        params["batch_correction"]
        and params["batch_method"].lower() == "column"
        and not params["batch_column"]
    ):
        raise click.UsageError(
            "Batch correction with method 'column' requires --batch-column option"
        )
    if (
        params["batch_correction"]
        and (params["batch_column"] or params["batch_covariates"])
        and not params["sdrf"]
    ):
        raise click.UsageError(
            "Batch correction with --batch-column or --batch-covariates "
            "requires --sdrf option"
        )
    batch_options = (
        "batch_method",
        "batch_column",
        "batch_covariates",
        "batch_parametric",
        "batch_mean_only",
        "batch_ref",
    )
    if not params["batch_correction"] and _supplied(ctx, *batch_options):
        raise click.UsageError("batch options require --batch-correction")
    if (
        params["batch_correction"]
        and params["batch_method"].lower() != "column"
        and params["batch_column"]
    ):
        raise click.UsageError("--batch-column requires --batch-method column")


def _validate_reference_options(ctx: click.Context, quant_method: str) -> None:
    params = ctx.params
    selectors = (
        _supplied(ctx, "irs_reference_samples"),
        _supplied(ctx, "irs_sdrf_column", "irs_sdrf_values"),
        _supplied(ctx, "irs_reference_regex"),
    )
    if sum(selectors) > 1:
        raise click.UsageError(
            "Choose one IRS/reference selector: samples, SDRF column+values, or regex"
        )
    if bool(params["irs_sdrf_column"]) != bool(params["irs_sdrf_values"]):
        raise click.UsageError(
            "--irs-sdrf-column and --irs-sdrf-values must be provided together"
        )
    irs_only_options = (
        "irs_sdrf_column",
        "irs_sdrf_values",
        "irs_stat",
        "irs_remove_reference",
    )
    if quant_method == "peptide_count" and params["irs"]:
        raise click.UsageError("peptide_count quantification cannot apply IRS")
    if quant_method == "ratio":
        if params["irs"]:
            raise click.UsageError(
                "Ratio quantification already performs reference scaling; "
                "--irs is not applicable"
            )
        if _supplied(ctx, *irs_only_options):
            raise click.UsageError(
                "Ratio accepts --irs-reference-samples or --irs-reference-regex; "
                "IRS column/stat/remove options require IRS normalization"
            )
    elif not params["irs"] and _supplied(
        ctx, "irs_reference_samples", "irs_reference_regex", *irs_only_options
    ):
        raise click.UsageError("IRS options require --irs")
    if params["irs"] and not params["sdrf"]:
        raise click.UsageError("IRS requires --sdrf")


def _resolved_imputation_method(ctx: click.Context) -> str:
    params = ctx.params
    impute_options = (
        "impute_method",
        "impute_quantile",
        "impute_seed",
        "impute_tune_sigma",
        "impute_n_neighbors",
    )
    if not params["impute"] and _supplied(ctx, *impute_options):
        raise click.UsageError("imputation options require --impute")
    if not params["impute"]:
        return "none"
    if params["impute_method"] is None:
        raise click.UsageError("--impute requires --impute-method")
    method = params["impute_method"].lower()
    if _supplied(ctx, "impute_quantile") and method not in {"mindet", "minprob"}:
        raise click.UsageError("--impute-quantile only applies to mindet/minprob")
    if _supplied(ctx, "impute_seed", "impute_tune_sigma") and method not in {
        "minprob",
        "qrilc",
    }:
        raise click.UsageError(
            "--impute-seed/--impute-tune-sigma only apply to minprob/qrilc"
        )
    if _supplied(ctx, "impute_n_neighbors") and method not in {"knn", "seqknn"}:
        raise click.UsageError("--impute-n-neighbors only applies to knn/seqknn")
    return method


def _validate_de_options(ctx: click.Context, quant_method: str) -> None:
    params = ctx.params
    de_options = (
        "de_contrasts",
        "de_contrasts_file",
        "de_method",
        "de_ensemble_methods",
        "de_ensemble_min_k",
        "de_log2fc_threshold",
        "de_fdr_threshold",
        "de_fdr_method",
        "de_output",
        "de_precision_weights",
    )
    if not params["differential_expression"] and _supplied(ctx, *de_options):
        raise click.UsageError("differential-expression options require --de")
    if params["differential_expression"] and not any(
        (params["de_output"], params["plot_volcano"], params["interactive_report"])
    ):
        raise click.UsageError(
            "--de requires --de-output, --plot-volcano, or --interactive-report"
        )
    if params["de_method"].lower() != "ensemble" and _supplied(
        ctx, "de_ensemble_methods", "de_ensemble_min_k"
    ):
        raise click.UsageError(
            "--de-ensemble-methods/--de-ensemble-min-k require --de-method ensemble"
        )
    requested_method = params["de_method"].lower()
    resolved_method = (
        "deqms"
        if requested_method == "auto" and quant_method == "directlfq"
        else "limrots"
        if requested_method == "auto"
        else requested_method
    )
    if resolved_method in {"rots", "limrots"} and _supplied(ctx, "de_fdr_method"):
        raise click.UsageError(
            f"--de-fdr-method does not apply to {resolved_method}, "
            "which retains its permutation FDR"
        )
    if (
        resolved_method in {"rots", "limrots"}
        and params["de_fdr_method"].lower() != "bh"
    ):
        raise click.UsageError(
            f"--de-fdr-method {params['de_fdr_method']} does not apply to "
            f"{resolved_method}, which retains its permutation FDR"
        )


def _validate_precision_weights(ctx: click.Context, quant_method: str) -> None:
    # Data-aware `auto` may pick another method, so DEqMS must be explicit.
    params = ctx.params
    if not params["de_precision_weights"]:
        return
    if quant_method not in {"maxlfq", "directlfq"}:
        raise click.UsageError(
            "--de-precision-weights requires --quant-method maxlfq or directlfq"
        )
    if params["de_method"].lower() != "deqms":
        raise click.UsageError("--de-precision-weights requires --de-method deqms")


def _validate_plot_options(ctx: click.Context) -> None:
    params = ctx.params
    if params["plot_volcano"] and not params["differential_expression"]:
        raise click.UsageError("--plot-volcano requires --de")
    if (
        any((params["plot_volcano"], params["plot_heatmap"], params["plot_pca"]))
        and not params["plot_output_dir"]
    ):
        raise click.UsageError("plot options require --plot-dir")
    if params["highlight_genes"] and not params["plot_volcano"]:
        raise click.UsageError("--highlight-genes requires --plot-volcano")
    if params["interactive_report"] and not params["differential_expression"]:
        raise click.UsageError("--interactive-report requires --de")
    if params["interactive_report"] and not params["sdrf"]:
        raise click.UsageError("--interactive-report requires --sdrf")
    if params["report_output"] and not params["interactive_report"]:
        raise click.UsageError("--report-output requires --interactive-report")
    if params["plot_output_dir"] and not any(
        (
            params["plot_volcano"],
            params["plot_heatmap"],
            params["plot_pca"],
            params["interactive_report"],
        )
    ):
        raise click.UsageError("--plot-dir requires a plot or report output option")


def _split_csv(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",")] if value else None


def _parse_de_contrasts(inline: str | None, path: str | None) -> list[str] | None:
    contrasts = _split_csv(inline) or []
    if not path:
        return contrasts or None
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"group1", "group2"}.issubset(
            reader.fieldnames
        ):
            raise click.UsageError(
                f"Contrasts file '{path}' must have 'group1' and 'group2' "
                f"columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            group1 = row.get("group1", "").strip()
            group2 = row.get("group2", "").strip()
            if group1 and group2:
                contrasts.append(f"{group1} vs {group2}")
    click.echo(f"Loaded {len(contrasts)} contrasts (inline + file: {path})")
    return contrasts or None
