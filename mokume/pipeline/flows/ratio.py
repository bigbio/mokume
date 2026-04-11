"""
Ratio quantification flow (PS protocol).

Handles ratio-based quantification where each sample is normalized
against a per-plex reference channel. Works at PSM level
(input_level="psms"). Reference samples and plex mapping come from
the qpx sample table only.

Flow:
    Load qpx parquet at PSM level -> QpxDataset(.psms)
    Reference/plex from qpx sample table (--irs-reference-*, --plex-column)
    Ratio quantification (log2(sample/reference) per plex)
    -> QpxDataset(.proteins)
"""

from mokume.core.dataset import QpxDataset
from mokume.core.logger import get_logger
from mokume.pipeline.config import PipelineConfig
from mokume.quantification.base import QuantificationMethod

logger = get_logger("mokume.pipeline.flows.ratio")


def run(method: QuantificationMethod, config: PipelineConfig) -> QpxDataset:
    """Execute the ratio quantification flow.

    Parameters
    ----------
    method : QuantificationMethod
        The resolved ratio quantification method.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    QpxDataset
        Dataset with proteins populated (log2 ratios).
    """
    from mokume.quantification.ratio import RatioQuantification
    from mokume.pipeline.stages import LoadingStage

    dataset = QpxDataset()

    logger.info("Running ratio-based quantification (PS protocol)...")

    # Load PSM data and get reference/plex from qpx sample table
    loading = LoadingStage(config)
    psm_df, ref_samples, sample_to_plex, sample_metadata = loading.load_for_ratio()
    dataset.psms = psm_df
    if sample_metadata is not None:
        dataset.sample_info = sample_metadata
    dataset.record_step("loading", method="ratio", rows_out=len(psm_df))

    # Validate schema
    errors = dataset.validate_level("psms")
    if errors:
        logger.warning("Schema validation warnings for psms: %s", errors)

    # Run ratio quantification
    ratio_quant = RatioQuantification(
        reference_samples=ref_samples,
        sample_to_plex=sample_to_plex,
    )
    protein_df = ratio_quant.quantify(psm_df)

    # Remove reference samples from output columns (log2(ref/ref) = 0)
    protein_col = protein_df.columns[0]
    seen = set()
    unique_cols = []
    for c in protein_df.columns:
        if c == protein_col or c not in ref_samples:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)
    protein_df = protein_df[unique_cols]

    dataset.proteins = protein_df
    dataset.record_step(
        "quantification",
        method="ratio",
        rows_out=len(protein_df),
        reference_samples=ref_samples,
    )

    # Store ratio-specific metadata
    dataset.uns["ratio_config"] = {
        "reference_samples": ref_samples,
    }

    logger.info(f"Ratio pipeline complete: {len(protein_df)} proteins")
    return dataset
