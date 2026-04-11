"""
Standard quantification flow.

Handles: iBAQ, TopN, sum, median quantification methods.
These all work from peptide-level data (input_level="peptides").

Flow:
    Load qpx parquet -> QpxDataset(.features)
    Apply filters
    Feature normalization
    Assemble peptides -> QpxDataset(.peptides)
    Sample normalization
    Quantification -> QpxDataset(.proteins)
"""

from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.pipeline.config import PipelineConfig
from mokume.pipeline.stages import (
    LoadingStage,
    NormalizationStage,
    QuantificationStage,
)
from mokume.quantification.base import QuantificationMethod

logger = get_logger("mokume.pipeline.flows.standard")


def run(method: QuantificationMethod, config: PipelineConfig) -> QpxDataset:
    """Execute the standard quantification flow.

    Parameters
    ----------
    method : QuantificationMethod
        The resolved quantification method (used for metadata only here;
        the actual dispatch happens in QuantificationStage).
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    QpxDataset
        Dataset with proteins populated.
    """
    dataset = QpxDataset()

    # Load and process peptides (filtering + normalization)
    loading = LoadingStage(config)
    logger.info("Loading and filtering data...")
    peptide_df, sample_metadata = loading.load_for_mokume()
    dataset.peptides = peptide_df
    if sample_metadata is not None:
        dataset.sample_info = sample_metadata
    dataset.record_step("loading", method="standard", rows_out=len(peptide_df))
    logger.info(f"Processed peptides: {len(peptide_df)} rows")

    # Validate schema
    errors = dataset.validate_level("peptides")
    if errors:
        logger.warning("Schema validation warnings for peptides: %s", errors)

    # Export peptides if requested
    if config.output.export_peptides:
        logger.info(f"Exporting peptides to {config.output.export_peptides}")
        peptide_df.to_csv(config.output.export_peptides, index=False)

    # Quantify proteins
    quant_stage = QuantificationStage(config)
    logger.info(f"Quantifying proteins with method: {config.quantification.method}")
    protein_df = quant_stage.quantify(peptide_df)
    dataset.proteins = protein_df
    dataset.record_step(
        "quantification",
        method=config.quantification.method,
        rows_out=len(protein_df),
    )
    logger.info(f"Quantification complete: {len(protein_df)} proteins")

    return dataset
