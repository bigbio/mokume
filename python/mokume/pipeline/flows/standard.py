"""
Standard quantification flow.

Handles: piBAQ, TopN, sum, median quantification methods.
These all work from peptide-level data (input_level="peptides").

Flow:
    Load qpx parquet -> QpxDataset(.features)
    Apply filters
    Feature normalization
    Assemble peptides -> QpxDataset(.peptides)
    Sample normalization
    Quantification -> QpxDataset(.proteins)
"""

from mokume.core.constants import load_sdrf
from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.pipeline.config import PipelineConfig
from mokume.pipeline.stages import (
    LoadingStage,
    QuantificationStage,
)
from mokume.quantification.base import QuantificationMethod

logger = get_logger("mokume.pipeline.flows.standard")


def _attach_sample_info(dataset: QpxDataset, config: PipelineConfig) -> None:
    """Best-effort SDRF sample metadata attachment.

    Mirrors ``QuantificationPipeline.run_dataset``: a malformed or absent
    SDRF must not sink an otherwise successful quantification run, so
    file/parse errors are logged and ``sample_info`` is left unset.
    """
    if not config.input.sdrf:
        return
    try:
        dataset.sample_info = load_sdrf(config.input.sdrf)
    except (OSError, ValueError) as exc:  # pragma: no cover - best-effort
        logger.debug("Could not load SDRF sample_info: %s", exc)


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

    # Load and process peptides (filtering + normalization). ``load_for_mokume``
    # returns a single peptide DataFrame; sample metadata comes from the SDRF,
    # not from the loading method (mirrors ``run_dataset``).
    loading = LoadingStage(config)
    logger.info("Loading and filtering data...")
    peptide_df = loading.load_for_mokume()
    dataset.peptides = peptide_df
    _attach_sample_info(dataset, config)
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
    if quant_stage.cell_precision is not None:
        dataset.layers["cell_precision"] = quant_stage.cell_precision
    dataset.record_step(
        "quantification",
        method=config.quantification.method,
        rows_out=len(protein_df),
    )
    logger.info(f"Quantification complete: {len(protein_df)} proteins")

    return dataset
