"""
Peptide-level normalization implementations.

This module provides the main peptide_normalization function that orchestrates
the full normalization workflow. Data access, filtering, and aggregation
utilities have been moved to dedicated modules:

- mokume.io.feature: Feature class and SQLFilterBuilder (DuckDB parquet access)
- mokume.preprocessing.sdrf: analyse_sdrf() and SDRF helpers
- mokume.preprocessing.aggregation: Filtering and aggregation functions
"""

import os
import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from mokume.model.labeling import QuantificationCategory
from mokume.model.normalization import (
    PeptideNormalizationMethod,
    parse_normalization_methods,
)
from mokume.core.constants import (
    FRACTION,
    INTENSITY,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PROTEIN_NAME,
    TECHREPLICATE,
    PARQUET_COLUMNS,
    AGGREGATION_LEVEL_SAMPLE,
)

from mokume.core.write_queue import WriteParquetTask, WriteCSVTask
from mokume.core.logger import get_logger, log_execution_time

# Re-export from new locations for backward compatibility
from mokume.io.feature import Feature, SQLFilterBuilder, unique_peptide_rows
from mokume.preprocessing.sdrf import analyse_sdrf
from mokume.preprocessing.aggregation import (
    parse_uniprot_accession,
    get_canonical_peptide,
    remove_contaminants_entrapments_decoys,
    remove_protein_by_ids,
    reformat_quantms_feature_table_quant_labels,
    apply_initial_filtering,
    merge_fractions,
    get_peptidoform_normalize_intensities,
    sum_peptidoform_intensities,
)

__all__ = [
    "Feature",
    "SQLFilterBuilder",
    "analyse_sdrf",
    "parse_uniprot_accession",
    "get_canonical_peptide",
    "remove_contaminants_entrapments_decoys",
    "remove_protein_by_ids",
    "reformat_quantms_feature_table_quant_labels",
    "apply_initial_filtering",
    "merge_fractions",
    "get_peptidoform_normalize_intensities",
    "sum_peptidoform_intensities",
]

if TYPE_CHECKING:
    from mokume.model.filters import PreprocessingFilterConfig

# Get a logger for this module
logger = get_logger("mokume.peptide_normalization")


def _ensure_qvalue_columns(dataset_df):
    """Add nullable QPX q-value fields for legacy/custom feature frames."""
    for column in ("peptide_qvalue", "pg_global_qvalue"):
        if column not in dataset_df.columns:
            dataset_df[column] = None


@log_execution_time(logger)
def peptide_normalization(
    parquet: str,
    sdrf: str,
    min_aa: int,
    min_unique: int,
    remove_ids: str,
    remove_decoy_contaminants: bool,
    remove_low_frequency_peptides: bool,
    output: str,
    skip_normalization: bool,
    nmethod: str,
    pnmethod: str,
    log2: bool,
    save_parquet: bool,
    irs_channel: str = None,
    irs_autodetect_regex: str = None,
    irs_stat: str = "median",
    irs_scope: str = "global",
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
    filter_config: Optional["PreprocessingFilterConfig"] = None,
    keep_shared_peptides: bool = False,
) -> None:
    """
    Perform peptide normalization on a proteomics dataset.

    Parameters
    ----------
    parquet : str
        Path to the Parquet file containing the dataset.
    sdrf : str
        Path to the SDRF file for quantification details.
    min_aa : int
        Minimum number of amino acids required for peptides.
    min_unique : int
        Minimum number of unique peptides per protein.
    remove_ids : str
        Path to a file with protein IDs to remove.
    remove_decoy_contaminants : bool
        Whether to remove decoys and contaminants.
    remove_low_frequency_peptides : bool
        Whether to remove low-frequency peptides.
    output : str
        Path to the output file for saving results.
    skip_normalization : bool
        Whether to skip normalization steps.
    nmethod : str
        Method for feature-level normalization.
    pnmethod : str
        Method for peptide-level normalization.
    log2 : bool
        Whether to apply log2 transformation to intensities.
    save_parquet : bool
        Whether to save results in Parquet format.
    irs_channel : str, optional
        IRS reference channel label for TMT/ITRAQ normalization.
    irs_autodetect_regex : str, optional
        Regex to autodetect pooled/reference sample in SDRF.
    irs_stat : str, optional
        Statistic for IRS per-run metric (median or mean).
    irs_scope : str, optional
        IRS scaling scope (global, by_mixture, or two_stage).
    aggregation_level : str, optional
        Level at which to aggregate intensities ("sample" or "run").
    filter_config : PreprocessingFilterConfig, optional
        Configuration for preprocessing filters.
    keep_shared_peptides : bool, optional
        Keep shared/non-unique peptide rows. When enabled, the unique-peptide
        row filter and per-protein ``min_unique`` gate are skipped so a later
        piBAQ step can allocate shared peptide intensity.
    """

    feature_normalization, peptide_normalized = parse_normalization_methods(
        nmethod, pnmethod
    )

    if os.path.exists(output):
        raise FileExistsError("The output file already exists.")

    if parquet is None:
        raise FileNotFoundError("The file does not exist.")

    logger.info("Loading data from %s...", parquet)

    # Create filter builder for pre-computations (median maps, peptide frequencies)
    if filter_config is not None and filter_config.enabled:
        contaminant_patterns = filter_config.protein.active_contaminant_patterns()
        named_score = filter_config.peptide.score
        filter_builder = SQLFilterBuilder(
            remove_contaminants=bool(contaminant_patterns),
            contaminant_patterns=contaminant_patterns,
            min_intensity=filter_config.intensity.min_intensity,
            min_peptide_length=min_aa,
            require_unique=not keep_shared_peptides,
            named_score_name=getattr(named_score, "name", None),
            named_score_threshold=getattr(named_score, "threshold", None),
        )
    else:
        filter_builder = SQLFilterBuilder(
            remove_contaminants=remove_decoy_contaminants,
            contaminant_patterns=["CONTAMINANT", "CONTAM_", "ENTRAP", "DECOY"],
            min_intensity=0.0,
            min_peptide_length=min_aa,
            require_unique=not keep_shared_peptides,
        )

    feature = Feature(parquet, filter_builder=filter_builder)

    if sdrf:
        feature.enrich_with_sdrf(sdrf)
        technical_repetitions, label, sample_names, choice = analyse_sdrf(sdrf)
    else:
        technical_repetitions, label, sample_names, choice = (
            feature.experimental_inference
        )

    if remove_low_frequency_peptides:
        low_frequency_peptides = feature.get_low_frequency_peptides()

    med_map = {}
    if (
        not skip_normalization
        and peptide_normalized == PeptideNormalizationMethod.GlobalMedian
    ):
        med_map = feature.get_median_map()
    elif (
        not skip_normalization
        and peptide_normalized == PeptideNormalizationMethod.ConditionMedian
    ):
        med_map = feature.get_median_map_to_condition()

    # IRS normalization pre-computation
    irs_scale_by_techrep: dict[int, float] = {}
    irs_requested = irs_channel is not None or irs_autodetect_regex is not None
    if skip_normalization and irs_requested:
        raise ValueError("skip_normalization conflicts with IRS options")
    if irs_requested and label not in (
        QuantificationCategory.TMT,
        QuantificationCategory.ITRAQ,
    ):
        raise ValueError("IRS normalization requires TMT or ITRAQ input")
    try:
        if not skip_normalization and label in (
            QuantificationCategory.TMT,
            QuantificationCategory.ITRAQ,
        ):
            if irs_channel is None and irs_autodetect_regex and sdrf:
                from mokume.core.constants import load_sdrf as _load_sdrf

                sdrf_df = _load_sdrf(sdrf)
                ref_mask = sdrf_df["source name"].str.contains(
                    irs_autodetect_regex, case=False, na=False
                )
                ref_labels = sdrf_df.loc[ref_mask, "comment[label]"]
                if not ref_labels.empty:
                    irs_channel = ref_labels.mode().iloc[0]
                else:
                    raise ValueError(
                        f"IRS autodetect regex '{irs_autodetect_regex}' found no pooled sample"
                    )

            if irs_channel is not None:
                irs_scale_by_techrep = feature.get_irs_scaling_factors(
                    irs_channel=irs_channel,
                    irs_stat=irs_stat,
                    irs_scope=irs_scope,
                )
                if not irs_scale_by_techrep:
                    raise ValueError(
                        f"IRS channel '{irs_channel}' produced no scaling factors"
                    )
    except Exception as e:
        raise ValueError(f"IRS normalization pre-computation failed: {e}") from e

    # Start writers only after every input-dependent validation/pre-computation
    # has succeeded, so a rejected IRS request cannot leave an empty output.
    write_csv = True
    if write_csv:
        write_csv_task = WriteCSVTask(output)
        write_csv_task.start()

    if save_parquet:
        writer_parquet_task = WriteParquetTask(output)
        writer_parquet_task.start()

    # Initialize filter pipeline if config provided
    filter_pipeline = None
    if filter_config is not None and filter_config.enabled:
        from mokume.preprocessing.filters import get_filter_pipeline

        filter_pipeline = get_filter_pipeline(filter_config)
        if len(filter_pipeline) > 0:
            logger.info(
                "Filter pipeline '%s' initialized with %d filters",
                filter_config.name,
                len(filter_pipeline),
            )

    for samples, df in feature.iter_samples():
        df.dropna(subset=["pg_accessions"], inplace=True)
        for sample in samples:
            logger.info("%s: Data preprocessing...", str(sample).upper())
            dataset_df = df[df["sample_accession"] == sample].copy()

            if not keep_shared_peptides:
                dataset_df = dataset_df[unique_peptide_rows(dataset_df)]
            _ensure_qvalue_columns(dataset_df)
            dataset_df = dataset_df[PARQUET_COLUMNS]

            dataset_df = reformat_quantms_feature_table_quant_labels(
                dataset_df, label, choice
            )

            dataset_df = apply_initial_filtering(dataset_df, min_aa, aggregation_level)

            if not keep_shared_peptides:
                dataset_df = dataset_df.groupby(PROTEIN_NAME).filter(
                    lambda x: len(set(x[PEPTIDE_CANONICAL])) >= min_unique
                )

            if remove_decoy_contaminants:
                dataset_df = remove_contaminants_entrapments_decoys(dataset_df)

            if remove_ids is not None:
                dataset_df = remove_protein_by_ids(dataset_df, remove_ids)
            dataset_df.rename(columns={INTENSITY: NORM_INTENSITY}, inplace=True)

            # Apply filter pipeline if configured
            if filter_pipeline is not None and len(filter_pipeline) > 0:
                initial_count = len(dataset_df)
                dataset_df, filter_results = filter_pipeline.apply(dataset_df)
                if filter_config.log_filtered_counts:
                    for result in filter_results:
                        if result.removed_count > 0:
                            logger.info(
                                "%s: %s removed %d items (%.1f%%)",
                                str(sample).upper(),
                                result.filter_name,
                                result.removed_count,
                                result.removal_rate * 100,
                            )
                    total_removed = initial_count - len(dataset_df)
                    if total_removed > 0:
                        logger.info(
                            "%s: Filter pipeline removed %d/%d items total",
                            str(sample).upper(),
                            total_removed,
                            initial_count,
                        )

            if (
                not skip_normalization
                and nmethod not in ("none", None)
                and technical_repetitions > 1
            ):
                start_time = time.time()
                logger.info(
                    "%s: Normalizing intensities of features using method %s...",
                    str(sample).upper(),
                    nmethod,
                )
                dataset_df = feature_normalization(dataset_df, technical_repetitions)
                elapsed = time.time() - start_time
                logger.info(
                    "%s: Number of features after normalization: %d (completed in %.2f seconds)",
                    str(sample).upper(),
                    len(dataset_df.index),
                    elapsed,
                )

            if irs_scale_by_techrep:
                if TECHREPLICATE in dataset_df.columns:
                    scale_series = (
                        dataset_df[TECHREPLICATE].map(irs_scale_by_techrep).fillna(1.0)
                    )
                    dataset_df.loc[:, NORM_INTENSITY] = (
                        dataset_df[NORM_INTENSITY] * scale_series
                    )
                else:
                    raise ValueError(
                        f"TECHREPLICATE is missing; cannot apply IRS scaling to sample {sample}"
                    )

            dataset_df = get_peptidoform_normalize_intensities(dataset_df)
            logger.info(
                "%s: Number of peptides after peptidoform selection: %d",
                str(sample).upper(),
                len(dataset_df.index),
            )

            if len(dataset_df[FRACTION].unique().tolist()) > 1:
                start_time = time.time()
                logger.info(
                    "%s: Merging features across fractions...", str(sample).upper()
                )
                dataset_df = merge_fractions(dataset_df)
                elapsed = time.time() - start_time
                logger.info(
                    "%s: Number of features after merging fractions: %d "
                    "(completed in %.2f seconds)",
                    str(sample).upper(),
                    len(dataset_df.index),
                    elapsed,
                )

            if not skip_normalization:
                dataset_df = peptide_normalized(dataset_df, sample, med_map)

            if remove_low_frequency_peptides and len(sample_names) > 1:
                dataset_df.set_index(
                    [PROTEIN_NAME, PEPTIDE_CANONICAL], drop=True, inplace=True
                )
                dataset_df = dataset_df[
                    ~dataset_df.index.isin(low_frequency_peptides)
                ].reset_index()
                logger.info(
                    "%s: Peptides after removing low frequency peptides: %d",
                    str(sample).upper(),
                    len(dataset_df.index),
                )

            start_time = time.time()
            logger.info(
                "%s: Summing all peptidoforms per %s...",
                str(sample).upper(),
                aggregation_level,
            )
            dataset_df = sum_peptidoform_intensities(dataset_df, aggregation_level)
            elapsed = time.time() - start_time
            logger.info(
                "%s: Number of peptides after selection: %d (completed in %.2f seconds)",
                str(sample).upper(),
                len(dataset_df.index),
                elapsed,
            )

            if log2:
                dataset_df[NORM_INTENSITY] = np.log2(dataset_df[NORM_INTENSITY])

            logger.info(
                "%s: Saving the normalized peptide intensities...", str(sample).upper()
            )

            if save_parquet:
                writer_parquet_task.write(dataset_df)
            if write_csv:
                write_csv_task.write(dataset_df)

    if write_csv:
        write_csv_task.close()
    if save_parquet:
        writer_parquet_task.close()
