"""
Pipeline runner with metadata-driven flow dispatch.

The runner resolves the quantification method from the PluginRegistry,
selects the appropriate flow based on the method's ``input_level``
property, executes it, and runs common post-processing.

Third-party plugins can register new flows by adding entries to
``FLOW_DISPATCH``.
"""

from typing import Dict

from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.core.registry import PluginRegistry
from mokume.model.normalization import parse_normalization_methods
from mokume.pipeline.config import PipelineConfig, validate_de_config
from mokume.pipeline import flows

logger = get_logger("mokume.pipeline.runner")


# Map input_level -> flow module.
# Each flow module must have a run(method, config) -> QpxDataset function.
FLOW_DISPATCH: Dict[str, object] = {
    "peptides": flows.standard,  # piBAQ, TopN, sum, median
    "psms": flows.ratio,  # Ratio quantification (PS protocol)
    "peptides_raw": flows.directlfq,  # DirectLFQ (handles its own normalization)
}


def run_pipeline(config: PipelineConfig) -> QpxDataset:
    """Execute the quantification pipeline.

    Resolves the quantification method, selects the appropriate flow,
    and runs common post-processing.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    QpxDataset
        Dataset with proteins populated and optional DE results in uns.
    """
    validate_de_config(config.de)
    parse_normalization_methods(
        config.normalization.run_method,
        config.normalization.sample_method,
    )
    quant_method_name = config.quantification.method.lower()
    logger.info(f"Starting pipeline with quant_method={quant_method_name}")

    if quant_method_name in {"directlfq", "ratio", "peptide_count"} and (
        config.normalization.run_method.lower() != "none"
        or config.normalization.sample_method.lower() != "none"
    ):
        reason = (
            "does not use intensity normalization"
            if quant_method_name == "peptide_count"
            else "manages normalization internally"
        )
        raise ValueError(
            f"{quant_method_name} {reason}; use run/sample normalization 'none'"
        )
    if quant_method_name == "peptide_count" and config.irs.enabled:
        raise ValueError("peptide_count quantification cannot apply IRS")

    if config.input.msstats and quant_method_name == "ratio":
        raise ValueError(
            "Ratio quantification requires PSM-level QPX input; "
            "MSstats feature tables do not contain the required PSM evidence"
        )
    if (
        config.quantification.sample_correlation_threshold is not None
        and not config.input.sdrf
    ):
        raise ValueError("Sample correlation filtering requires an SDRF file")
    if config.output.export_ions and quant_method_name != "directlfq":
        raise ValueError("export_ions is supported only by the directlfq pipeline")

    # Ensure built-in methods are registered
    import mokume.quantification  # noqa: F401

    # Resolve method from registry
    method = PluginRegistry.get("quantification", quant_method_name)

    # Select flow based on method's declared input_level
    flow = FLOW_DISPATCH.get(method.input_level)
    if flow is None:
        raise ValueError(
            f"No pipeline flow registered for input_level='{method.input_level}'. "
            f"Available flows: {list(FLOW_DISPATCH.keys())}"
        )

    # Execute the flow
    dataset = flow.run(method, config)

    # Common post-processing
    dataset = _postprocess(dataset, config)

    return dataset


def _postprocess(dataset: QpxDataset, config: PipelineConfig) -> QpxDataset:
    """Run common post-processing steps.

    These steps apply regardless of which flow was used:
    - IRS normalization (multi-plex TMT)
    - Coverage filter
    - Imputation
    - Batch correction
    - Differential expression
    - Plotting and reports

    Parameters
    ----------
    dataset : QpxDataset
        Dataset with proteins populated.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    QpxDataset
        Updated dataset.
    """
    from mokume.pipeline.stages import (
        ImputationStage,
        NormalizationStage,
        PostprocessingStage,
    )

    protein_df = dataset.proteins
    if protein_df is None:
        logger.warning("No protein data after quantification, skipping post-processing")
        return dataset

    quant_method = config.quantification.method.lower()

    # Create stages once (reused across steps)
    norm_stage = NormalizationStage(config)
    post_stage = PostprocessingStage(config)

    # IRS normalization (skip for ratio — handles cross-plex via reference division)
    if config.irs.enabled and quant_method != "ratio":
        protein_df = norm_stage.apply_irs(protein_df, dataset=dataset)
        dataset.proteins = protein_df
        dataset.record_step("irs_normalization", method=config.irs.stat)

    protein_df = _apply_sample_correlation_filter(
        protein_df, norm_stage, dataset, config
    )

    # Coverage filter
    if config.quantification.coverage_threshold is not None:
        protein_df = norm_stage.apply_coverage_filter(protein_df, dataset=dataset)
        dataset.proteins = protein_df
        dataset.record_step(
            "coverage_filter",
            threshold=config.quantification.coverage_threshold,
            rows_out=len(protein_df),
        )

    # Imputation (after the coverage filter, before batch correction)
    if config.imputation.enabled:
        protein_df = ImputationStage(config).impute(protein_df)
        dataset.proteins = protein_df
        dataset.record_step("imputation", method=config.imputation.method)

    # Batch correction (after imputation)
    if config.batch.enabled:
        protein_df = post_stage.apply_batch_correction(protein_df, dataset=dataset)
        dataset.proteins = protein_df
        dataset.record_step("batch_correction", method=config.batch.method)

    # Differential expression
    de_results = None
    if config.de.enabled:
        de_results = post_stage.run_differential_expression(
            protein_df,
            dataset=dataset,
            precision=dataset.layers.get("cell_precision"),
        )
        if de_results:
            dataset.uns["de_results"] = {
                k: v["df"].to_dict(orient="records") for k, v in de_results.items()
            }

    # Plotting
    if config.output.plot_dir and any(
        [
            config.output.plot_volcano,
            config.output.plot_heatmap,
            config.output.plot_pca,
        ]
    ):
        post_stage.generate_plots(protein_df, de_results, dataset=dataset)

    # Interactive report
    if config.output.interactive_report and de_results:
        post_stage.generate_interactive_report(protein_df, de_results, dataset=dataset)

    # AnnData export — uses QPX naming convention via qpx.Dataset.save_anndata()
    if config.output.export_anndata and config.input.qpx_dir:
        _export_anndata(dataset, config)

    return dataset


def _apply_sample_correlation_filter(
    protein_df, norm_stage, dataset: QpxDataset, config: PipelineConfig
):
    """Apply matrix QC and record it only when the option is enabled."""
    threshold = config.quantification.sample_correlation_threshold
    filtered = norm_stage.apply_sample_correlation_filter(protein_df)
    if threshold is not None:
        dataset.proteins = filtered
        dataset.record_step(
            "sample_correlation_filter",
            threshold=threshold,
            columns_out=len(filtered.columns) - 1,
        )
    return filtered


def _export_anndata(dataset: QpxDataset, config: PipelineConfig) -> None:
    """Export the dataset as AnnData using QPX's save_anndata API.

    If an AnnData file already exists for this view, the new quantification
    is added as a layer (keyed by the quant method name) instead of
    overwriting the file.
    """
    try:
        import anndata as ad
        import qpx
    except ImportError:
        logger.warning(
            "AnnData export requires the 'qpx' and 'anndata' packages. "
            "Install with: pip install qpx anndata"
        )
        return
    import numpy as np
    from pathlib import Path

    qpx_ds = qpx.Dataset(config.input.qpx_dir)
    view = config.output.anndata_view or "ae"
    prefix = Path(config.input.qpx_dir).name
    existing_path = Path(config.input.qpx_dir) / f"{prefix}.{view}.h5ad"

    quant_method = config.quantification.method.lower()

    if existing_path.exists():
        # Append as a layer to existing AnnData
        logger.info(
            f"Existing AnnData found at {existing_path}, "
            f"adding '{quant_method}' as layer"
        )
        adata = ad.read_h5ad(existing_path)
        new_adata = dataset.to_anndata(level="proteins", value_col="Intensity")

        # Align the new matrix to the existing obs/var indices
        import pandas as pd

        new_wide = pd.DataFrame(
            new_adata.X,
            index=new_adata.obs.index,
            columns=new_adata.var.index,
        )
        aligned = new_wide.reindex(index=adata.obs.index, columns=adata.var.index)
        adata.layers[quant_method] = aligned.values.astype(np.float32)

        # Also add log2 layer
        log2_vals = np.log2(
            np.where(aligned.values > 0, aligned.values, np.nan)
        ).astype(np.float32)
        adata.layers[f"{quant_method}_log2"] = log2_vals
    else:
        # Create new AnnData with X = current quantification
        adata = dataset.to_anndata(level="proteins", value_col="Intensity")

        # Add log2 layer
        x_vals = adata.X.copy()
        log2_vals = np.log2(np.where(x_vals > 0, x_vals, np.nan)).astype(np.float32)
        adata.layers[f"{quant_method}_log2"] = log2_vals

    output_path = qpx_ds.save_anndata(adata, view=view)
    logger.info(f"AnnData saved to {output_path}")
