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

from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.pipeline.config import PipelineConfig
from mokume.quantification.base import QuantificationMethod

logger = get_logger("mokume.pipeline.flows.directlfq")


def run(method: QuantificationMethod, config: PipelineConfig) -> QpxDataset:
    """Execute the DirectLFQ quantification flow.

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
        import directlfq.protein_intensity_estimation as lfq_estimation
        import directlfq.normalization as lfq_norm
        import directlfq.config as lfq_config
    except ImportError:
        raise ImportError(
            "DirectLFQ quantification requires the directlfq package.\n"
            "Install with: pip install directlfq\n"
            "Or: pip install mokume[directlfq]"
        )

    from mokume.pipeline.stages import LoadingStage

    dataset = QpxDataset()

    # Load and filter data
    loading = LoadingStage(config)
    logger.info("Loading and filtering data for DirectLFQ...")
    filtered_df, sample_metadata = loading.load_for_directlfq()
    dataset.features = filtered_df
    if sample_metadata is not None:
        dataset.sample_info = sample_metadata
    dataset.record_step("loading", method="directlfq", rows_out=len(filtered_df))
    logger.info(f"Filtered data: {len(filtered_df)} features")

    # Validate schema
    errors = dataset.validate_level("features")
    if errors:
        logger.warning("Schema validation warnings for features: %s", errors)

    # Convert to DirectLFQ format
    logger.info("Converting to DirectLFQ format...")
    directlfq_input = loading.convert_to_directlfq_format(filtered_df)
    logger.info(f"DirectLFQ input shape: {directlfq_input.shape}")

    # Configure DirectLFQ
    lfq_config.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
    lfq_config.set_compile_normalized_ion_table(
        config.output.export_ions is not None
    )

    # Run DirectLFQ normalization
    logger.info("Running DirectLFQ sample normalization...")
    normed_df = lfq_norm.NormalizationManagerSamplesOnSelectedProteins(
        directlfq_input,
        num_samples_quadratic=config.quantification.directlfq_num_samples_quadratic,
    ).complete_dataframe

    # Run DirectLFQ protein estimation
    logger.info("Running DirectLFQ protein estimation...")
    protein_df, ion_df = lfq_estimation.estimate_protein_intensities(
        normed_df,
        min_nonan=config.quantification.directlfq_min_nonan,
        num_samples_quadratic=10,
        num_cores=config.quantification.directlfq_num_cores,
    )

    # Export ions if requested
    if config.output.export_ions and ion_df is not None:
        logger.info(f"Exporting ions to {config.output.export_ions}")
        ion_df.to_csv(config.output.export_ions)

    dataset.proteins = protein_df
    dataset.record_step(
        "quantification",
        method="directlfq",
        rows_out=len(protein_df),
    )

    logger.info(f"DirectLFQ complete: {len(protein_df)} proteins")
    return dataset
