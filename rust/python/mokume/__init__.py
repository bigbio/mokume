"""mokume — a proteomics quantification toolkit.

The compute-heavy pipeline (expression matrix, normalization, imputation,
differential expression, batch correction) lives in a Rust kernel, exposed here
through the compiled ``mokume._mokume`` extension that maturin builds from
``crates/mokume-py``. The Python periphery lives in :mod:`mokume.commands` as
first-class modules. Plotting and reporting consume kernel tables, TissueMap
derives downstream atlas outputs from QPX data, and explicit fallbacks compute
operations the kernel does not provide.

This is the PyO3/maturin layout used by projects such as polars and
pydantic-core (Python imports a compiled Rust extension). The compute commands
run *in-process* (no subprocess) through the same clap parsing + dispatch as the
installed console command, so the flag handling stays single-sourced in Rust.

Each wrapper validates keyword names and translates them according to that
command's real CLI contract, including CSV versus repeatable options and
reverse booleans. For full control, call :func:`run` with explicit argv.
"""

import importlib
import importlib.metadata
import sys
import warnings

flags_for = getattr(importlib.import_module(f"{__name__}._command_flags"), "flags_for")
_NATIVE_EXTENSION = importlib.import_module("mokume._mokume")
_differential_expression = getattr(_NATIVE_EXTENSION, "differential_expression")
_impute_matrix = getattr(_NATIVE_EXTENSION, "impute_matrix")
normalize_matrix = getattr(_NATIVE_EXTENSION, "normalize_matrix")
_native_run = getattr(_NATIVE_EXTENSION, "run")
version = getattr(_NATIVE_EXTENSION, "version")
_pibaq_digest_request = getattr(_NATIVE_EXTENSION, "pibaq_digest_request")
_native_run_cli = getattr(_NATIVE_EXTENSION, "run_cli")
_native_run_cli_with_pibaq_digest = getattr(
    _NATIVE_EXTENSION, "run_cli_with_pibaq_digest"
)
_native_run_with_pibaq_digest = getattr(_NATIVE_EXTENSION, "run_with_pibaq_digest")

# `mokume` (this Rust kernel) and `mokume-py` (pure Python) both install the
# `mokume` import package, so pip silently overwrites files when both are
# present. Warn so the user keeps only one.
try:
    importlib.metadata.distribution("mokume-py")
    warnings.warn(
        "Both 'mokume' (Rust kernel) and 'mokume-py' (pure-Python) are installed; "
        "they share the 'mokume' import name and overwrite each other's files. "
        "Keep only one: uninstall the other (`pip uninstall mokume-py`).",
        RuntimeWarning,
        stacklevel=2,
    )
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "version",
    "run",
    "features2peptides",
    "features2proteins",
    "peptides2protein",
    "correct_batches",
    "normalize_matrix",
    "impute_matrix",
    "differential_expression",
    "protease_catalog",
    "tsne_visualization",
    "tissuemap",
    "peptides2protein_qc",
    "peptides2protein_pibaq",
    "de_plots",
    "interactive_report",
    "qc_report",
    "workflow_comparison",
    "impute",
]
__version__ = version()

_BOOTSTRAPPED = []


def _bootstrap():
    """Apply the periphery's warning filters + logging once, lazily.

    Kept out of import time so a pure ``import mokume`` for the compute path does
    not pull in the logging / warnings setup the plotting commands want.
    """
    if _BOOTSTRAPPED:
        return

    # Lazy: importing mokume.core would pull in the periphery's heavier modules.
    from mokume.core.logging_config import initialize_logging

    # Suppress the numpy matrix deprecation and the pyopenms OPENMS_DATA_PATH
    # false positive, matching the upstream package import behaviour.
    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        module="numpy.matrixlib.defmatrix",
    )
    warnings.filterwarnings("ignore", message=".*OPENMS_DATA_PATH.*")
    initialize_logging()
    _BOOTSTRAPPED.append(True)


def _build_args(command, kwargs):
    """Prefix the subcommand name to the flags built from ``kwargs``."""
    path = (
        ["quantify", command]
        if command in {"features2peptides", "features2proteins", "peptides2protein"}
        else [command]
    )
    return [*path, *flags_for(command, kwargs)]


def _prepare_pibaq_digest(args):
    request = _pibaq_digest_request(args)
    if request is None:
        return None
    module = importlib.import_module("mokume._pibaq_digest")
    return getattr(module, "build_pibaq_digest")(request)


def _pibaq_provenance_tuple(provenance):
    return (
        provenance["pyopenms_version"],
        provenance["enzyme"],
        provenance["catalog_hash"],
        provenance["min_aa"],
        provenance["max_aa"],
        provenance["missed_cleavages"],
    )


def _run(args):
    args = list(args)
    try:
        payload = _prepare_pibaq_digest(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"piBAQ digestion failed: {exc}") from None
    if payload is None:
        result = _native_run(args)
    else:
        accession_peptides, provenance = payload
        result = _native_run_with_pibaq_digest(
            args,
            accession_peptides,
            _pibaq_provenance_tuple(provenance),
        )
    entrypoint = importlib.import_module("mokume.__main__")
    getattr(entrypoint, "_render_requested_pibaq_qc")(args, sys.modules[__name__])
    return result


def _run_cli(args):
    args = list(args)
    try:
        payload = _prepare_pibaq_digest(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"piBAQ digestion failed: {exc}", file=sys.stderr)
        return 1
    if payload is None:
        return _native_run_cli(args)
    accession_peptides, provenance = payload
    return _native_run_cli_with_pibaq_digest(
        args,
        accession_peptides,
        _pibaq_provenance_tuple(provenance),
    )


def run(args):
    """Run a mokume subcommand in-process from an explicit argument list.

    Example::

        mokume.run(["quantify", "features2proteins", "--parquet", "x.parquet",
                    "--output", "y.csv"])
    """
    _run(list(args))


def protease_catalog():
    """Return every protease registered by the installed pyOpenMS runtime."""
    module = importlib.import_module("mokume._pibaq_digest")
    return getattr(module, "installed_protease_catalog")()


def features2peptides(**kwargs):
    """Run ``quantify features2peptides`` (feature parquet -> peptide output)."""
    _run(_build_args("features2peptides", kwargs))


def features2proteins(**kwargs):
    """Run ``quantify features2proteins`` from QPX or MSstats input."""
    _run(_build_args("features2proteins", kwargs))


def peptides2protein(**kwargs):
    """Run ``quantify peptides2protein`` (peptide input -> protein quantities)."""
    _run(_build_args("peptides2protein", kwargs))


def correct_batches(**kwargs):
    """Run ``correct-batches`` (ComBat on piBAQ output, or ``method="bridle"``)."""
    _run(_build_args("correct-batches", kwargs))


def impute_matrix(values, method, **options):
    """Run matrix-level Rust imputation without QPX I/O."""
    return _impute_matrix(values, method, options or None)


def differential_expression(proteins, values, n_a, n_b, method, **options):
    """Run matrix-level Rust differential expression without QPX I/O.

    ``values`` is a row-major linear-intensity matrix whose first ``n_a``
    columns are group A and next ``n_b`` columns are group B. Missing cells may
    be ``None`` or non-finite floats. ``peptide_counts`` is required for DEqMS
    and ensembles containing DEqMS. Other options include ``ensemble_methods``,
    thresholds, condition labels, and ``threads``.
    """
    return _differential_expression(
        list(proteins),
        values,
        n_a,
        n_b,
        method,
        options or None,
    )


def _run_command(module_name, argv):
    """Run a :mod:`mokume.commands` periphery command, raising on failure.

    The command modules signal failure with a non-zero return or ``SystemExit``;
    both are surfaced here as a ``RuntimeError`` so library callers get a normal
    exception instead of an interpreter exit.
    """
    _bootstrap()
    module = importlib.import_module(f"mokume.commands.{module_name}")
    try:
        code = module.main(list(argv))
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            raise RuntimeError(code) from None
        if code not in (0, None):
            raise RuntimeError(f"{module_name} failed with exit code {code}") from None
        return
    if code not in (0, None):
        raise RuntimeError(f"{module_name} failed with exit code {code}")


def tsne_visualization(**kwargs):
    """Render the t-SNE plot for a folder of protein files (``plotting`` extra)."""
    _run_command("visualize", flags_for("visualize", kwargs))


def tissuemap(**kwargs):
    """Run the per-dataset tissue proteome analysis (``tissuemap`` extra)."""
    _run_command("tissuemap", flags_for("tissuemap", kwargs))


def peptides2protein_qc(**kwargs):
    """Render the piBAQ QC report from a protein table (``plotting`` extra)."""
    _run_command("peptides2protein_qc", flags_for("peptides2protein_qc", kwargs))


def peptides2protein_pibaq(**kwargs):
    """Run the native-backed piBAQ compatibility command."""
    _run_command("peptides2protein_pibaq", flags_for("peptides2protein_pibaq", kwargs))


def de_plots(args):
    """Render features2proteins DE plots from kernel-written CSVs.

    Takes an explicit argument list (the per-contrast ``--contrast KEY A B CSV``
    flag repeats, which keyword arguments cannot express). See
    ``mokume plot de --help``.
    """
    _run_command("de_plots", args)


def interactive_report(args):
    """Render the features2proteins interactive HTML report from kernel CSVs.

    Takes an explicit argument list (see ``de_plots`` for why) — run
    ``mokume interactive-report --help`` for the flags.
    """
    _run_command("interactive_report", args)


def qc_report(
    protein_matrix,
    sdrf=None,
    output="qc_report.html",
    de_results=None,
    title="QC Report",
    is_log2=False,
):
    """Build a single-matrix QC HTML report (``analysis`` extra).

    Computes PCA / t-SNE / silhouette / variance decomposition / CV / missing-value
    and DE-quality metrics from a protein matrix CSV and writes an interactive HTML
    report. ``sdrf`` supplies the sample -> condition grouping; ``de_results`` is an
    optional DE result CSV. Returns the output path. For volcano gene-highlighting
    call :func:`mokume.reports.qc_report.generate_qc_report` directly.
    """
    _bootstrap()
    import pandas as pd

    from mokume.reports.qc_report import generate_qc_report

    protein_df = pd.read_csv(protein_matrix)
    sample_to_condition = {}
    if sdrf:
        from mokume.normalization.irs import detect_condition_from_sdrf

        sample_to_condition = detect_condition_from_sdrf(sdrf)
    de_df = (
        pd.read_csv(de_results, float_precision="round_trip") if de_results else None
    )
    return generate_qc_report(
        protein_df,
        sample_to_condition,
        de_results=de_df,
        output_html=str(output),
        title=title,
        is_log2=is_log2,
    )


def workflow_comparison(
    workflows,
    output="workflow_comparison.html",
    title="Workflow Comparison",
    marker_genes=None,
):
    """Build an HTML report comparing several quantification workflows (``analysis`` extra).

    ``workflows`` is a list of dicts, one per workflow, each with ``name`` and
    either ``protein_df`` (a DataFrame) or ``protein_matrix`` (a CSV path); plus
    optional ``de_results`` (DataFrame or CSV path), ``sample_to_condition`` (dict)
    or ``sdrf`` (path), and ``is_log2``. Returns the output path.
    """
    _bootstrap()
    import pandas as pd

    from mokume.reports.workflow_comparison import generate_comparison_report

    built = []
    for workflow in workflows:
        entry = {"name": workflow["name"], "is_log2": workflow.get("is_log2", False)}
        protein = workflow.get("protein_df")
        entry["protein_df"] = (
            protein if protein is not None else pd.read_csv(workflow["protein_matrix"])
        )
        de_results = workflow.get("de_results")
        if de_results is not None:
            entry["de_results"] = (
                de_results
                if hasattr(de_results, "columns")
                else pd.read_csv(de_results, float_precision="round_trip")
            )
        condition = workflow.get("sample_to_condition")
        if condition is None and workflow.get("sdrf"):
            from mokume.normalization.irs import detect_condition_from_sdrf

            condition = detect_condition_from_sdrf(workflow["sdrf"])
        entry["sample_to_condition"] = condition or {}
        built.append(entry)
    return generate_comparison_report(
        built, output_html=str(output), title=title, marker_genes=marker_genes
    )


def impute(matrix, method, output=None, **kwargs):
    """Impute missing values with the pure-Python imputers (``analysis`` extra).

    Reaches the imputers the Rust kernel does not reproduce (``missforest`` wraps
    a scikit-learn estimator), plus every other supported method. ``matrix`` is a
    wide protein matrix CSV or DataFrame; writes ``output`` if given and returns
    the imputed DataFrame.
    """
    _bootstrap()
    import pandas as pd

    from mokume.imputation import impute_missing_values

    frame = matrix if hasattr(matrix, "columns") else pd.read_csv(matrix, index_col=0)
    result = impute_missing_values(frame, method=method, **kwargs)
    if output:
        result.to_csv(output)
    return result
