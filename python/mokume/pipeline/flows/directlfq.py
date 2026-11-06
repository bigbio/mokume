"""
DirectLFQ quantification flow.

Delegates normalization and quantification entirely to the DirectLFQ
package. Uses adapter logic to convert between qpx parquet format
and DirectLFQ's expected input (input_level="peptides_raw").

Flow:
    Load qpx parquet -> QpxDataset(.features)
    Convert to DirectLFQ format (adapter)
    Run DirectLFQ normalization + estimation
    Convert back -> QpxDataset(.proteins)
"""

import gc

from mokume.core.constants import load_sdrf
from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.pipeline.config import PipelineConfig
from mokume.pipeline.directlfq_streaming import (
    estimate_protein_intensities_streamed,
)
from mokume.quantification.base import QuantificationMethod
from mokume.quantification.precision import precision_matrix

logger = get_logger("mokume.pipeline.flows.directlfq")


def run(method: QuantificationMethod, config: PipelineConfig) -> QpxDataset:
    """Execute the DirectLFQ quantification flow.

    Mirrors ``QuantificationPipeline._run_directlfq_pipeline`` so that
    ``run_pipeline`` yields a protein matrix identical to
    ``run_dataset``: it loads through the streaming Arrow reader, runs
    DirectLFQ sample normalization, then estimates protein intensities via
    the streaming helper (which returns linear-scale intensities). Requested
    normalized ion output is written incrementally by the same helper.

    Parameters
    ----------
    method : QuantificationMethod
        The resolved DirectLFQ method (used for metadata).
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    QpxDataset
        Dataset with proteins populated.
    """
    try:
        import directlfq.normalization as lfq_norm
        import directlfq.config as lfq_config
    except ImportError as exc:
        raise ImportError(
            "DirectLFQ quantification requires the directlfq package.\n"
            "Install with: pip install directlfq\n"
            "Or: pip install mokume-py[directlfq]"
        ) from exc

    from mokume.pipeline.stages import LoadingStage

    dataset = QpxDataset()

    # Load and filter data as a streaming Arrow Table (single return value).
    loading = LoadingStage(config)
    logger.info("Loading and filtering data for DirectLFQ...")
    filtered_table = loading.load_for_directlfq()
    logger.info("Filtered data: %d features", filtered_table.num_rows)

    # Sample metadata comes from the SDRF (best-effort), not the loading method.
    if config.input.sdrf:
        try:
            dataset.sample_info = load_sdrf(config.input.sdrf)
        except (OSError, ValueError) as exc:  # pragma: no cover - best-effort
            logger.debug("Could not load SDRF sample_info: %s", exc)
    dataset.record_step("loading", method="directlfq", rows_out=filtered_table.num_rows)

    # Convert to DirectLFQ format (zero-copy polars from the Arrow Table).
    logger.info("Converting to DirectLFQ format...")
    directlfq_input = loading.convert_to_directlfq_format(filtered_table)
    logger.info("DirectLFQ input shape: %s", directlfq_input.shape)
    del filtered_table
    gc.collect()

    lfq_config.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
    lfq_config.set_compile_normalized_ion_table(False)

    # Run DirectLFQ normalization
    logger.info("Running DirectLFQ sample normalization...")
    normed_df = lfq_norm.NormalizationManagerSamplesOnSelectedProteins(
        directlfq_input,
        num_samples_quadratic=config.quantification.directlfq_num_samples_quadratic,
    ).complete_dataframe
    del directlfq_input
    gc.collect()

    # Run DirectLFQ protein estimation (streaming; returns linear intensities).
    logger.info("Running DirectLFQ protein estimation...")
    protein_df = _estimate_proteins(normed_df, config, dataset)

    dataset.proteins = protein_df
    dataset.record_step(
        "quantification",
        method="directlfq",
        rows_out=len(protein_df),
    )

    logger.info(f"DirectLFQ complete: {len(protein_df)} proteins")
    return dataset


def _estimate_proteins(normed_df, config: PipelineConfig, dataset: QpxDataset):
    """Estimate protein intensities, keeping cell precisions for weighted DEqMS."""
    residuals = [] if config.de.precision_weights else None
    protein_df = estimate_protein_intensities_streamed(
        normed_df,
        min_nonan=config.quantification.directlfq_min_nonan,
        num_samples_quadratic=config.quantification.directlfq_num_samples_quadratic,
        num_cores=config.quantification.directlfq_num_cores,
        export_ions=config.output.export_ions,
        residuals=residuals,
    )
    if residuals is not None:
        dataset.layers["cell_precision"] = precision_matrix(
            residuals, normed_df.columns, protein_df.columns[0]
        )
    return protein_df
