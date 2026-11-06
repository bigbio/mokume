"""
Pipeline stage classes for the quantification pipeline.

Each stage handles a distinct phase of the proteomics quantification workflow:
- LoadingStage: Data loading and filtering
- NormalizationStage: Run-level and sample-level normalization
- QuantificationStage: Protein quantification methods
- PostprocessingStage: Batch correction, DE, plotting, reports
"""

import importlib
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Optional

from mokume.io.feature import Feature, SQLFilterBuilder, unique_peptide_rows
from mokume.preprocessing.sdrf import analyse_sdrf
from mokume.preprocessing.aggregation import (
    remove_contaminants_entrapments_decoys,
    apply_initial_filtering,
    get_peptidoform_normalize_intensities,
    sum_peptidoform_intensities,
    reformat_quantms_feature_table_quant_labels,
)
from mokume.normalization.hierarchical import HierarchicalSampleNormalizer
from mokume.normalization import irs as irs_normalization
from mokume.model.normalization import (
    FeatureNormalizationMethod,
    PeptideNormalizationMethod,
)
from mokume.model.labeling import QuantificationCategory
from mokume.core.constants import (
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    NORM_INTENSITY,
    SAMPLE_ID,
    INTENSITY,
    PARQUET_COLUMNS,
    AGGREGATION_LEVEL_SAMPLE,
    CONDITION,
    PIBAQ,
    load_sdrf,
)
from mokume.core.logger import get_logger
from mokume.postprocessing.batch_correction import (
    is_batch_correction_available,
    detect_batches,
    extract_covariates_from_sdrf,
    apply_batch_correction,
)
from mokume.model.batch_correction import BatchDetectionMethod
from mokume.pipeline.config import PipelineConfig, validate_de_config
from mokume.pipeline.sample_qc import filter_samples_by_correlation

logger = get_logger("mokume.pipeline")

_DEFAULT_REFERENCE_REGEX = "pool|powder|ref|reference|bridge"

# Per-(sample, tech_rep) replicate metric expressions for SQL-first run
# normalization. Each value must produce a scalar metric per group so that the
# cross-replicate factor ``intensity * sample_avg / run_metric`` matches the
# legacy ``normalize_sample`` / ``normalize_replicates`` contract from
# :mod:`mokume.model.normalization`. ``max_min`` is intentionally absent because
# its replicate function returns a per-row Series rather than a scalar metric;
# workflows using it fall back to the legacy pandas loop.
_SQL_RUN_METRIC_EXPR: dict[str, str] = {
    "median": 'MEDIAN("Intensity")',
    "mean": 'AVG("Intensity")',
    "max": 'MAX("Intensity")',
    "global": 'SUM("Intensity")',
    "iqr": (
        '(QUANTILE_CONT("Intensity", 0.75) + QUANTILE_CONT("Intensity", 0.25)) / 2'
    ),
}
_SQL_RUN_NORM_SUPPORTED: frozenset[str] = frozenset({"none", ""}) | frozenset(
    _SQL_RUN_METRIC_EXPR
)

_SQLFIRST_LOAD_QUERY_PREFIX = """
        WITH base AS (
            SELECT
                array_to_string(
                    list_transform(
                        """

_SQLFIRST_LOAD_QUERY_AFTER_MEMBERS = """,
                        lambda x: CASE
                            WHEN length(string_split(x, '|')) = 3
                                THEN string_split(x, '|')[2]
                            ELSE x
                        END
                    ),
                    ';'
                ) AS "ProteinName",
                sequence       AS "PeptideCanonical",
                peptidoform    AS "PeptideSequence",
                charge         AS "PrecursorCharge",
                intensity      AS "Intensity",
                COALESCE(condition, 'Empty') AS "Condition",
                COALESCE(CAST(biological_replicate AS INTEGER), 1) AS "BioReplicate",
                sample_accession AS "SampleID",
                run            AS "Run"
            FROM parquet_db
            WHERE ("""

_SQLFIRST_LOAD_QUERY_AFTER_WHERE = """)
              AND pg_accessions IS NOT NULL
              AND length(sequence) >= ?
              AND (condition IS NULL OR condition != 'Empty')
              AND run IS NOT NULL
        ),
        with_n_unique AS (
            SELECT *,
                COUNT(DISTINCT "PeptideCanonical") OVER (
                    PARTITION BY "ProteinName", "SampleID"
                ) AS _n_unique_pep
            FROM base
        ),
        min_unique_kept AS (
            SELECT * FROM with_n_unique WHERE _n_unique_pep >= ?
        ),
        peptidoform_max AS (
            SELECT * FROM (
                SELECT """

_SQLFIRST_LOAD_QUERY_AFTER_SELECT = """,
                    ROW_NUMBER() OVER (
                        PARTITION BY mu."PeptideSequence", mu."PrecursorCharge",
                                     mu."SampleID", mu."Condition", mu."BioReplicate"
                        ORDER BY """

_SQLFIRST_LOAD_QUERY_AFTER_ORDERBY = """ DESC, mu."Run"
                    ) AS _rn
                FROM min_unique_kept mu
                """

_SQLFIRST_LOAD_QUERY_SUFFIX = """
            )
            WHERE _rn = 1
        )
        SELECT
            "ProteinName",
            "PeptideCanonical",
            "SampleID",
            "BioReplicate",
            "Condition",
            SUM("Intensity") AS "NormIntensity"
        FROM peptidoform_max
        GROUP BY "ProteinName", "PeptideCanonical", "SampleID",
                 "BioReplicate", "Condition"
        """

_LFQ_DIRECTLFQ_QUERY_SUFFIX = """
            )
            WHERE _rn = 1
        )
        SELECT
            "ProteinName" AS protein,
            "PeptideCanonical" AS ion,
            "SampleID" AS sample_accession,
            SUM("Intensity") AS intensity
        FROM peptidoform_max
        GROUP BY "ProteinName", "PeptideCanonical", "SampleID"
        """

_MAXLFQ_SPECIES_QUERY_SUFFIX = """
            ) WHERE _rn = 1
        )
        SELECT "ProteinName", "PeptideCanonical", "PeptideSequence", "PrecursorCharge",
               "SampleID", "BioReplicate", "Condition", SUM("Intensity") AS "NormIntensity"
        FROM peptidoform_max
        GROUP BY "ProteinName", "PeptideCanonical", "PeptideSequence", "PrecursorCharge",
                 "SampleID", "BioReplicate", "Condition"
        """

_RUN_SCALE_QUERY_PREFIX = """
        WITH base AS (
            SELECT
                array_to_string(
                    list_transform(
                        """

_RUN_SCALE_QUERY_AFTER_MEMBERS = """,
                        lambda x: CASE
                            WHEN length(string_split(x, '|')) = 3
                                THEN string_split(x, '|')[2]
                            ELSE x
                        END
                    ),
                    ';'
                ) AS "ProteinName",
                sequence         AS "PeptideCanonical",
                intensity        AS "Intensity",
                sample_accession AS "SampleID",
                run              AS "Run"
            FROM parquet_db
            WHERE ("""

_RUN_SCALE_QUERY_AFTER_WHERE = """)
              AND pg_accessions IS NOT NULL
              AND length(sequence) >= ?
              AND (condition IS NULL OR condition != 'Empty')
              AND run IS NOT NULL
        ),
        with_n_unique AS (
            SELECT *,
                COUNT(DISTINCT "PeptideCanonical") OVER (
                    PARTITION BY "ProteinName", "SampleID"
                ) AS _n_unique_pep
            FROM base
        ),
        kept AS (
            SELECT "SampleID", "Run", "Intensity"
            FROM with_n_unique WHERE _n_unique_pep >= ?
        ),
        sample_run_meta AS (
            SELECT
                "SampleID",
                BOOL_AND("Run" LIKE '%\\_%' ESCAPE '\\') AS all_have_underscore,
                BOOL_AND(
                    TRY_CAST(split_part("Run", '_', -1) AS INTEGER) IS NOT NULL
                ) AS all_parse_last,
                BOOL_AND(TRY_CAST("Run" AS INTEGER) IS NOT NULL) AS all_parse_full
            FROM kept
            GROUP BY "SampleID"
        ),
        with_tech_rep AS (
            SELECT
                m."SampleID",
                m."Run",
                m."Intensity",
                CASE
                    WHEN srm.all_have_underscore AND srm.all_parse_last
                        THEN CAST(split_part(m."Run", '_', -1) AS INTEGER)
                    WHEN srm.all_parse_full
                        THEN CAST(m."Run" AS INTEGER)
                    ELSE
                        DENSE_RANK() OVER (
                            PARTITION BY m."SampleID" ORDER BY m."Run"
                        )
                END AS tech_rep
            FROM kept m
            JOIN sample_run_meta srm USING ("SampleID")
        ),
        run_metric AS (
            SELECT "SampleID", tech_rep, """

_RUN_SCALE_QUERY_SUFFIX = """ AS rm
            FROM with_tech_rep
            GROUP BY "SampleID", tech_rep
        ),
        sample_avg AS (
            SELECT "SampleID", AVG(rm) AS sa, COUNT(*) AS n_tech_reps
            FROM run_metric
            GROUP BY "SampleID"
        ),
        run_to_tech_rep AS (
            SELECT DISTINCT "SampleID", "Run", tech_rep
            FROM with_tech_rep
        )
        SELECT
            r."SampleID" AS "SampleID",
            r."Run"         AS "Run",
            CASE
                WHEN sa.n_tech_reps > 1
                    THEN sa.sa / NULLIF(rm.rm, 0)
                ELSE CAST(1.0 AS DOUBLE)
            END AS scale_factor
        FROM run_to_tech_rep r
        JOIN run_metric rm USING ("SampleID", tech_rep)
        JOIN sample_avg  sa USING ("SampleID")
        """


def _build_sqlfirst_load_query(
    filter_builder: SQLFilterBuilder,
    scale_select: str,
    scale_orderby: str,
    scale_join_clause: str,
    query_suffix: str = _SQLFIRST_LOAD_QUERY_SUFFIX,
) -> tuple[str, list]:
    """Build the SQL-first peptide loading query and its filter parameters."""
    members, member_params = filter_builder.build_protein_members()
    where_clause, where_params = filter_builder.build_where_clause()
    query = "".join(
        [
            _SQLFIRST_LOAD_QUERY_PREFIX,
            members,
            _SQLFIRST_LOAD_QUERY_AFTER_MEMBERS,
            where_clause,
            _SQLFIRST_LOAD_QUERY_AFTER_WHERE,
            scale_select,
            _SQLFIRST_LOAD_QUERY_AFTER_SELECT,
            scale_orderby,
            _SQLFIRST_LOAD_QUERY_AFTER_ORDERBY,
            scale_join_clause,
            query_suffix,
        ]
    )
    return query, [*member_params, *where_params]


def _build_run_scale_query(
    filter_builder: SQLFilterBuilder, metric_expr: str
) -> tuple[str, list]:
    """Build the run normalization scale-factor query and its filter parameters."""
    members, member_params = filter_builder.build_protein_members()
    where_clause, where_params = filter_builder.build_where_clause()
    query = "".join(
        [
            _RUN_SCALE_QUERY_PREFIX,
            members,
            _RUN_SCALE_QUERY_AFTER_MEMBERS,
            where_clause,
            _RUN_SCALE_QUERY_AFTER_WHERE,
            metric_expr,
            _RUN_SCALE_QUERY_SUFFIX,
        ]
    )
    return query, [*member_params, *where_params]


def _sum_maxlfq_species(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Combine contextual maxima without merging modification/charge species."""
    species_keys = [
        PROTEIN_NAME,
        PEPTIDE_CANONICAL,
        "PeptideSequence",
        "PrecursorCharge",
        SAMPLE_ID,
        CONDITION,
        "BioReplicate",
    ]
    dataset_df = dataset_df.groupby(species_keys, observed=True, as_index=False)[
        NORM_INTENSITY
    ].sum()
    return dataset_df


def _require_rows_passing_filters(
    feature: Feature, filter_builder: SQLFilterBuilder
) -> None:
    """Fail before loading, with the reason, when the filters keep no rows."""
    where_clause, where_params = filter_builder.build_where_clause()
    sql = "".join(["SELECT 1 FROM parquet_db WHERE ", where_clause, " LIMIT 1"])
    if feature.parquet_db.execute(sql, where_params).fetchone() is None:
        raise ValueError(
            "No features pass the input filters (intensity > 0, peptide "
            "length, unique peptides, contaminants, decoys and scores)"
        )


class LoadingStage:
    """Load and filter QPX parquet or native SDRF+MSstats data."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _open_feature(self, filter_builder: SQLFilterBuilder) -> Feature:
        """Open the configured QPX parquet or native SDRF+MSstats input."""
        if self.config.input.msstats:
            return Feature.from_msstats(
                self.config.input.msstats,
                self.config.input.sdrf,
                filter_builder=filter_builder,
                duckdb_limits=(
                    self.config.runtime.duckdb_memory,
                    self.config.runtime.duckdb_threads,
                ),
            )
        return Feature(
            self.config.input.parquet,
            filter_builder=filter_builder,
            duckdb_memory=self.config.runtime.duckdb_memory,
            duckdb_threads=self.config.runtime.duckdb_threads,
        )

    def load_for_mokume(self) -> pd.DataFrame:
        """Load data and apply normalization for mokume quantification methods.

        Dispatcher that routes to a SQL-first path when the workflow allows
        pushing the entire filter+aggregation pipeline into a single DuckDB query
        (LFQ label, no per-run normalization needed); otherwise falls back to the
        legacy per-sample pandas loop. Both paths produce equivalent peptide-level
        DataFrames (validated on PXD003539: rows + keys identical, NormIntensity
        agreeing to relative 1e-7, the precision limit of the float32 intensity
        column stored in parquet).
        """
        keep_shared_peptides = self.config.quantification.method.lower() == "pibaq"
        filter_builder = SQLFilterBuilder(
            remove_contaminants=self.config.filtering.remove_contaminants,
            min_peptide_length=self.config.filtering.min_aa,
            require_unique=not keep_shared_peptides,
        )

        feature = self._open_feature(filter_builder)

        if self.config.input.sdrf:
            feature.enrich_with_sdrf(self.config.input.sdrf)
            technical_repetitions, label, _sample_names, choice = analyse_sdrf(
                self.config.input.sdrf
            )
        else:
            (
                technical_repetitions,
                label,
                _sample_names,
                choice,
            ) = feature.experimental_inference

        _require_rows_passing_filters(feature, filter_builder)

        # Build normalization config snapshot (shared between both load paths).
        run_norm = self.config.normalization.run_method.lower()
        sample_norm = self.config.normalization.sample_method.lower()
        sample_norm_method = PeptideNormalizationMethod.from_str(sample_norm)

        med_map: dict = {}
        if sample_norm == "globalmedian":
            med_map = feature.get_median_map()
        elif sample_norm == "conditionmedian":
            med_map = feature.get_median_map_to_condition()

        # SQL-first path is eligible when (a) the label is LFQ (TMT/iTRAQ need
        # channel-aware reformatting that the SQL prototype does not implement),
        # and (b) the requested run normalization method has a SQL translation
        # registered in :data:`_SQL_RUN_METRIC_EXPR`. Workflows using ``max_min``
        # (whose replicate function returns a per-row Series) fall back to the
        # legacy pandas loop.
        sqlfirst_eligible = (
            label == QuantificationCategory.LFQ and run_norm in _SQL_RUN_NORM_SUPPORTED
        )

        if sqlfirst_eligible:
            logger.info(
                "Loading with SQL-first path (label=%s, tech_reps=%d, run_norm=%s)",
                label,
                technical_repetitions,
                run_norm,
            )
            combined_df = self._load_for_mokume_sqlfirst(
                feature,
                sample_norm,
                sample_norm_method,
                med_map,
                run_norm=run_norm,
                technical_repetitions=technical_repetitions,
            )
        else:
            logger.info(
                "Loading with legacy per-sample pandas path "
                "(label=%s, tech_reps=%d, run_norm=%s)",
                label,
                technical_repetitions,
                run_norm,
            )
            combined_df = self._load_for_mokume_legacy_pandas(
                feature,
                label,
                choice,
                technical_repetitions,
                sample_norm,
                sample_norm_method,
                med_map,
            )

        return self._apply_dataset_normalization(combined_df, sample_norm_method)

    def _load_for_mokume_sqlfirst(
        self,
        feature: Feature,
        sample_norm: str,
        sample_norm_method: PeptideNormalizationMethod,
        med_map: dict,
        run_norm: str = "none",
        technical_repetitions: int = 1,
    ) -> pd.DataFrame:
        """Push the full filter+aggregation pipeline into a single DuckDB query.

        Translates the steps that the legacy pandas loop performs per sample
        (parse_uniprot_accession, length / condition / run filters,
        contaminant/decoy removal, per-sample min-unique-peptides filter,
        optional per-(sample, tech_rep) run normalization, peptidoform-max
        keep, sum aggregation at the sample level) into one SQL query against
        the SDRF-enriched ``parquet_db`` view. Only the small post-aggregation
        peptide-level table lands in pandas, where per-sample normalization is
        applied if needed.

        Run normalization is enabled when both ``run_norm`` resolves to a
        registered SQL metric and ``technical_repetitions > 1`` (matching the
        global guard in :meth:`FeatureNormalizationMethod.normalize_runs`).
        """
        min_aa = self.config.filtering.min_aa
        min_unique = (
            0
            if self.config.quantification.method.lower() == "pibaq"
            else self.config.filtering.min_unique_peptides
        )

        run_norm_active = run_norm in _SQL_RUN_METRIC_EXPR and technical_repetitions > 1

        # Run normalization is applied via a small per-(SampleID, Run)
        # scale-factor lookup table registered with DuckDB and joined into the
        # main query, rather than materialising a full-sized scaled intensity
        # CTE (which on PXD030304-scale inputs forced ~95 GB of disk spill).
        scale_df = None
        scale_join_clause = ""
        scale_select = "mu.*"
        scale_orderby = 'mu."Intensity"'
        if run_norm_active:
            scale_df = self._compute_run_scale_map(feature, run_norm)
            feature.parquet_db.register("run_scale_map", scale_df)
            scale_join_clause = (
                "LEFT JOIN run_scale_map rsm "
                'ON rsm."SampleID" = mu."SampleID" '
                'AND rsm."Run" = mu."Run"'
            )
            scale_expr = '(mu."Intensity" * COALESCE(rsm.scale_factor, 1.0))'
            scale_select = f'mu.* REPLACE ({scale_expr} AS "Intensity")'
            scale_orderby = scale_expr

        # Protein-accession parsing is shared with _compute_run_scale_map via
        # the SQL query builders above.
        query, filter_params = _build_sqlfirst_load_query(
            feature.filter_builder,
            scale_select,
            scale_orderby,
            scale_join_clause,
            query_suffix=(
                _MAXLFQ_SPECIES_QUERY_SUFFIX
                if self.config.quantification.method.lower() == "maxlfq"
                else _SQLFIRST_LOAD_QUERY_SUFFIX
            ),
        )
        query_params = [*filter_params, min_aa, min_unique]
        try:
            combined_df = feature.parquet_db.execute(query, query_params).df()
        finally:
            if scale_df is not None:
                feature.parquet_db.unregister("run_scale_map")

        # Per-sample normalization (skip dataset-level methods that operate on
        # the full table later in _apply_dataset_normalization).
        if not sample_norm_method.is_dataset_level and sample_norm != "none":
            chunks = []
            for sample_id, sample_df in combined_df.groupby(
                SAMPLE_ID, observed=True, sort=False
            ):
                chunks.append(
                    sample_norm_method(sample_df.copy(), str(sample_id), med_map)
                )
            combined_df = pd.concat(chunks, ignore_index=True)

        # Match legacy column dtypes (apply_initial_filtering casts these to
        # Categorical in the pandas path; downstream consumers may rely on it).
        combined_df[CONDITION] = pd.Categorical(combined_df[CONDITION])
        combined_df[SAMPLE_ID] = pd.Categorical(combined_df[SAMPLE_ID])
        return combined_df

    def _compute_run_scale_map(self, feature: Feature, run_norm: str) -> pd.DataFrame:
        """Compute the per-``(SampleID, Run)`` multiplicative scale factor
        applied by :meth:`FeatureNormalizationMethod.normalize_runs`.

        Returns a small DataFrame (at most ``n_runs`` rows) with columns
        ``SampleID``, ``Run`` and ``scale_factor``. The TechReplicate ladder
        mirrors :func:`apply_initial_filtering` precisely: per sample, prefer
        the integer trailing run-name segment after ``_``; fall back to casting
        the full run name; fall back further to a per-sample DENSE_RANK over
        unique run names. ``scale_factor`` is ``sample_avg_metric / run_metric``
        for samples with more than one TechReplicate, and ``1.0`` otherwise
        (single-tech-rep samples pass through unchanged), so the caller can
        unconditionally multiply Intensity by ``COALESCE(scale_factor, 1.0)``.

        Compared to the previous in-query CTE chain this aggregates straight
        from the filtered base to a small ``(SampleID, tech_rep) -> metric``
        table and never materialises a full-sized scaled-intensity CTE, which
        is what drove the >95 GB peak resident set on PXD030304.
        """
        min_aa = self.config.filtering.min_aa
        min_unique = (
            0
            if self.config.quantification.method.lower() == "pibaq"
            else self.config.filtering.min_unique_peptides
        )
        metric_expr = _SQL_RUN_METRIC_EXPR[run_norm]

        # NOTE: we project only the columns needed for the run-norm aggregate
        # (SampleID, Run, sequence-for-min-unique gating, Intensity) so DuckDB
        # never has to carry the peptidoform/charge/condition columns through
        # the window-function pipeline used to enforce min_unique_peptides.
        #
        # Protein-accession parsing mirrors _load_for_mokume_sqlfirst through
        # the shared SQL query builders.
        sql, filter_params = _build_run_scale_query(feature.filter_builder, metric_expr)
        sql_params = [*filter_params, min_aa, min_unique]
        return feature.parquet_db.execute(sql, sql_params).df()

    def _load_for_mokume_legacy_pandas(
        self,
        feature: Feature,
        label,
        choice,
        technical_repetitions: int,
        sample_norm: str,
        sample_norm_method: PeptideNormalizationMethod,
        med_map: dict,
    ) -> pd.DataFrame:
        """Original per-sample pandas loop, kept verbatim for fallback paths.

        Used when the SQL-first dispatcher cannot apply (TMT/iTRAQ labels or
        an active per-run normalization stage that needs row-level intensities
        before peptidoform-max).
        """
        all_peptides = []

        for samples, df in feature.iter_samples():
            df.dropna(subset=["pg_accessions"], inplace=True)

            for sample in samples:
                dataset_df = df[df["sample_accession"] == sample].copy()
                if self.config.quantification.method.lower() != "pibaq":
                    dataset_df = dataset_df[unique_peptide_rows(dataset_df)]
                dataset_df = dataset_df[PARQUET_COLUMNS]

                dataset_df = reformat_quantms_feature_table_quant_labels(
                    dataset_df, label, choice
                )
                dataset_df = apply_initial_filtering(
                    dataset_df, self.config.filtering.min_aa, AGGREGATION_LEVEL_SAMPLE
                )

                # Filter by min unique peptides
                if self.config.quantification.method.lower() != "pibaq":
                    dataset_df = dataset_df.groupby(PROTEIN_NAME).filter(
                        lambda x: (
                            len(set(x[PEPTIDE_CANONICAL]))
                            >= self.config.filtering.min_unique_peptides
                        )
                    )

                if self.config.filtering.remove_contaminants:
                    dataset_df = remove_contaminants_entrapments_decoys(dataset_df)

                dataset_df.rename(columns={INTENSITY: NORM_INTENSITY}, inplace=True)

                # Apply run normalization
                run_norm = self.config.normalization.run_method.lower()
                if run_norm not in ("none", "") and technical_repetitions > 1:
                    run_method = FeatureNormalizationMethod.from_str(run_norm)
                    dataset_df = run_method(dataset_df, technical_repetitions)

                dataset_df = get_peptidoform_normalize_intensities(dataset_df)
                if self.config.quantification.method.lower() == "maxlfq":
                    dataset_df = _sum_maxlfq_species(dataset_df)
                else:
                    dataset_df = sum_peptidoform_intensities(
                        dataset_df, AGGREGATION_LEVEL_SAMPLE
                    )

                # Apply per-sample normalization (skip dataset-level methods)
                if not sample_norm_method.is_dataset_level and sample_norm != "none":
                    dataset_df = sample_norm_method(dataset_df, sample, med_map)

                all_peptides.append(dataset_df)

        return pd.concat(all_peptides, ignore_index=True)

    def _apply_dataset_normalization(
        self,
        combined_df: pd.DataFrame,
        sample_norm_method: PeptideNormalizationMethod,
    ) -> pd.DataFrame:
        """Apply dataset-level normalization (operates on the combined table).

        Dispatches to the matching ``NormalizationStage`` method when the
        configured peptide normalization is dataset-level (Hierarchical,
        Quantile, MedianCenter, MeanCenter, Rlr, Loess, TMM).
        """
        if sample_norm_method == PeptideNormalizationMethod.Hierarchical:
            combined_df = NormalizationStage(self.config).apply_hierarchical(
                combined_df
            )
        elif sample_norm_method == PeptideNormalizationMethod.Quantile:
            combined_df = NormalizationStage(self.config).apply_quantile(combined_df)
        elif sample_norm_method == PeptideNormalizationMethod.MedianCenter:
            combined_df = NormalizationStage(self.config).apply_median_center(
                combined_df
            )
        elif sample_norm_method == PeptideNormalizationMethod.MeanCenter:
            combined_df = NormalizationStage(self.config).apply_mean_center(combined_df)
        elif sample_norm_method == PeptideNormalizationMethod.Rlr:
            combined_df = NormalizationStage(self.config).apply_rlr(combined_df)
        elif sample_norm_method == PeptideNormalizationMethod.Loess:
            combined_df = NormalizationStage(self.config).apply_loess(combined_df)
        elif sample_norm_method == PeptideNormalizationMethod.TMM:
            combined_df = NormalizationStage(self.config).apply_tmm(combined_df)

        return combined_df

    def convert_to_directlfq_format(self, table: pa.Table) -> pd.DataFrame:
        """Pivot aggregated DirectLFQ ion rows into the wide log2 input matrix."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "polars is required for DirectLFQ streaming conversion. "
                "Install it with `pip install polars` or "
                "`pip install mokume-py[directlfq]`."
            ) from exc

        pdf = pl.from_arrow(table)
        if isinstance(pdf, pl.Series):
            pdf = pdf.to_frame()

        summed = pdf.group_by(["protein", "ion", "sample_accession"]).agg(
            pl.col("intensity").cast(pl.Float64).sum()
        )
        wide_pl = summed.pivot(
            index=["protein", "ion"],
            on="sample_accession",
            values="intensity",
            aggregate_function=None,
        ).sort(["protein", "ion"])

        sample_cols = sorted(c for c in wide_pl.columns if c not in ("protein", "ion"))
        wide_pl = wide_pl.select(["protein", "ion", *sample_cols])
        wide_pl = wide_pl.with_columns(
            [
                pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c).log(2)).alias(c)
                for c in sample_cols
            ]
        )

        wide = wide_pl.to_pandas().set_index(["protein", "ion"])
        wide.index.names = ["protein", "ion"]
        return wide

    def load_for_directlfq(self) -> pa.Table:
        """Load canonical-peptide rows for DirectLFQ as a streaming Arrow Table.

        Streams the aggregated long-format query result through DuckDB's Arrow
        reader instead of materialising an additional pandas long table.

        The shared SQL-first feature aggregation preserves the full protein
        group, applies ``min_unique_peptides`` per protein/sample, keeps the
        maximum intensity for each peptidoform/charge ion, and then sums those
        ions to the canonical-peptide level.

        Returns
        -------
        pa.Table
            Long-format Arrow Table with columns
            ``[protein, ion, sample_accession, intensity]``.
        """
        filter_builder = SQLFilterBuilder(
            remove_contaminants=self.config.filtering.remove_contaminants,
            min_peptide_length=self.config.filtering.min_aa,
            require_unique=True,
        )

        feature = self._open_feature(filter_builder)

        try:
            if self.config.input.sdrf:
                feature.enrich_with_sdrf(self.config.input.sdrf)

            query, filter_params = _build_sqlfirst_load_query(
                filter_builder,
                "mu.*",
                'mu."Intensity"',
                "",
                query_suffix=_LFQ_DIRECTLFQ_QUERY_SUFFIX,
            )
            query_params = [
                *filter_params,
                self.config.filtering.min_aa,
                self.config.filtering.min_unique_peptides,
            ]
            result = feature.parquet_db.execute(query, query_params)
            reader = result.to_arrow_reader(batch_size=1_000_000)
            # ``from_batches`` eagerly consumes the reader, materialising every
            # batch into Arrow buffers that own their memory independently of
            # the DuckDB connection, so it is safe to close the connection in
            # the ``finally`` below even though we return the Table.
            return pa.Table.from_batches(reader, schema=reader.schema)
        finally:
            feature.parquet_db.close()

    def load_for_ratio(self) -> tuple:
        """Load PSM data and detect references for ratio quantification.

        Returns
        -------
        tuple
            (psm_df, ref_samples, sample_to_plex)
        """
        from mokume.quantification.ratio import load_psm_data
        from mokume.normalization.irs import (
            detect_pooled_from_sdrf,
            detect_reference_by_regex,
            detect_plexes_from_sdrf,
        )

        if not self.config.input.sdrf:
            raise ValueError(
                "Ratio quantification requires an SDRF file for reference "
                "sample detection. Use --sdrf to provide one."
            )

        # Explicit selectors take precedence over metadata autodetection.
        ref_samples = self.config.irs.reference_samples
        if ref_samples:
            logger.info("Using explicit reference samples: %s", ref_samples)
        elif self.config.irs.reference_regex != _DEFAULT_REFERENCE_REGEX:
            ref_samples = detect_reference_by_regex(
                self.config.input.sdrf, self.config.irs.reference_regex
            )
        else:
            ref_samples = detect_pooled_from_sdrf(self.config.input.sdrf)
            if not ref_samples:
                ref_samples = detect_reference_by_regex(
                    self.config.input.sdrf, _DEFAULT_REFERENCE_REGEX
                )

        if not ref_samples:
            raise ValueError(
                "Ratio quantification requires reference samples. "
                "None detected from SDRF. Use --irs-reference-samples to specify."
            )

        # Detect plexes
        sample_to_plex = detect_plexes_from_sdrf(self.config.input.sdrf)

        # Load PSM data
        psm_df = load_psm_data(
            parquet_path=self.config.input.parquet,
            sdrf_path=self.config.input.sdrf,
            min_aa=self.config.filtering.min_aa,
            min_unique_peptides=self.config.filtering.min_unique_peptides,
            remove_contaminants=self.config.filtering.remove_contaminants,
        )

        return psm_df, ref_samples, sample_to_plex


class NormalizationStage:
    """Applies run-level and sample-level normalization."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _peptide_index(self, df: pd.DataFrame) -> list[str]:
        if self.config.quantification.method.lower() == "maxlfq" and {
            "PeptideSequence",
            "PrecursorCharge",
        }.issubset(df.columns):
            return [
                PROTEIN_NAME,
                PEPTIDE_CANONICAL,
                "PeptideSequence",
                "PrecursorCharge",
            ]
        return [PROTEIN_NAME, PEPTIDE_CANONICAL]

    def apply_hierarchical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hierarchical sample normalization."""
        logger.info("Applying hierarchical sample normalization...")

        # Convert to wide format for normalization. observed=True avoids the
        # Cartesian-product blowup when SAMPLE_ID is Categorical.
        wide = df.pivot_table(
            index=self._peptide_index(df),
            columns=SAMPLE_ID,
            values=NORM_INTENSITY,
            aggfunc="sum",
            observed=True,
        )

        # Log2 transform for normalization
        wide = wide.replace(0, np.nan)
        wide_log2 = np.log2(wide)

        # Load selected proteins if specified
        selected_proteins = None
        if self.config.normalization.proteins_file:
            with open(self.config.normalization.proteins_file, encoding="utf-8") as f:
                selected_proteins = [line.strip() for line in f if line.strip()]
            logger.info(
                "Using %d selected proteins for normalization", len(selected_proteins)
            )

        # Apply hierarchical normalization
        normalizer = HierarchicalSampleNormalizer(
            num_samples_quadratic=self.config.quantification.directlfq_num_samples_quadratic,
            selected_proteins=selected_proteins,
        )

        normalized_log2 = normalizer.fit_transform(wide_log2)

        # Convert back to linear scale
        normalized_wide = 2**normalized_log2

        # Convert back to long format
        normalized_long = normalized_wide.reset_index().melt(
            id_vars=self._peptide_index(df),
            var_name=SAMPLE_ID,
            value_name=NORM_INTENSITY,
        )

        # Remove NaN rows
        normalized_long = normalized_long.dropna(subset=[NORM_INTENSITY])

        logger.info(
            "Hierarchical normalization complete: %d rows", len(normalized_long)
        )

        return normalized_long

    def _apply_dataset_normalizer(
        self,
        df: pd.DataFrame,
        normalizer,
        log_space: bool,
        name: str,
    ) -> pd.DataFrame:
        """Pivot peptide-level long DataFrame to wide, fit_transform, melt back."""
        logger.info("Applying %s sample normalization...", name)
        wide = df.pivot_table(
            index=self._peptide_index(df),
            columns=SAMPLE_ID,
            values=NORM_INTENSITY,
            aggfunc="sum",
            observed=True,
        )
        if log_space:
            wide = np.log2(wide.replace(0, np.nan))
        normalized_wide = normalizer.fit_transform(wide)
        if log_space:
            normalized_wide = 2**normalized_wide
        normalized_long = (
            normalized_wide.reset_index()
            .melt(
                id_vars=self._peptide_index(df),
                var_name=SAMPLE_ID,
                value_name=NORM_INTENSITY,
            )
            .dropna(subset=[NORM_INTENSITY])
        )
        logger.info("%s normalization complete: %d rows", name, len(normalized_long))
        return normalized_long

    def apply_quantile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quantile sample normalization."""
        from mokume.normalization.distribution import QuantileNormalizer

        return self._apply_dataset_normalizer(
            df, QuantileNormalizer(), log_space=False, name="Quantile"
        )

    def apply_median_center(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sample-level median centering in log2 space."""
        from mokume.normalization.distribution import MedianCenterNormalizer

        return self._apply_dataset_normalizer(
            df, MedianCenterNormalizer(), log_space=True, name="MedianCenter"
        )

    def apply_mean_center(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sample-level mean centering in log2 space."""
        from mokume.normalization.distribution import MeanCenterNormalizer

        return self._apply_dataset_normalizer(
            df, MeanCenterNormalizer(), log_space=True, name="MeanCenter"
        )

    def apply_rlr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Robust Linear Regression normalization (NormalyzerDE)."""
        from mokume.normalization.rlr import RlrNormalizer

        return self._apply_dataset_normalizer(
            df,
            RlrNormalizer(log_transform=False),
            log_space=False,
            name="RLR",
        )

    def apply_loess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply LOESS regression-based normalization (in log2 space)."""
        from mokume.normalization.loess import LOESSNormalizer

        return self._apply_dataset_normalizer(
            df, LOESSNormalizer(), log_space=True, name="LOESS"
        )

    def apply_tmm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Trimmed Mean of M-values normalization (on linear intensities)."""
        from mokume.normalization.tmm import TMMNormalizer

        return self._apply_dataset_normalizer(
            df, TMMNormalizer(), log_space=False, name="TMM"
        )

    def apply_irs(self, protein_df: pd.DataFrame, dataset=None) -> pd.DataFrame:
        """Apply Internal Reference Scaling normalization for multi-plex TMT data.

        The optional ``dataset`` parameter is accepted for compatibility with
        the metadata-driven pipeline runner (``pipeline.runner``); it is not
        used by the IRS algorithm itself.

        Detection priority for reference samples:
        1. characteristics[pooled sample] column in SDRF
        2. Explicit sample names (--irs-reference-samples)
        3. Explicit column + values (--irs-sdrf-column + --irs-sdrf-values)
        4. Regex scan across factor/characteristic columns
        """
        from mokume.normalization.irs import (
            IRSNormalizer,
            detect_pooled_from_sdrf,
            detect_reference_by_column,
            detect_reference_by_regex,
            detect_plexes_from_sdrf,
        )

        if not self.config.input.sdrf:
            raise ValueError("IRS normalization requires an SDRF file (--sdrf)")

        # Explicit selectors take precedence over pooled-sample autodetection.
        if self.config.irs.reference_samples:
            ref_samples = self.config.irs.reference_samples
            logger.info("Using explicit reference samples: %s", ref_samples)
        elif self.config.irs.sdrf_column and self.config.irs.sdrf_values:
            ref_samples = detect_reference_by_column(
                self.config.input.sdrf,
                self.config.irs.sdrf_column,
                self.config.irs.sdrf_values,
            )
        elif self.config.irs.reference_regex != _DEFAULT_REFERENCE_REGEX:
            ref_samples = detect_reference_by_regex(
                self.config.input.sdrf, self.config.irs.reference_regex
            )
        else:
            ref_samples = detect_pooled_from_sdrf(self.config.input.sdrf)
            if not ref_samples:
                ref_samples = detect_reference_by_regex(
                    self.config.input.sdrf, _DEFAULT_REFERENCE_REGEX
                )

        if not ref_samples:
            raise ValueError("IRS normalization found no reference samples in the SDRF")

        # Detect plexes
        sample_to_plex = detect_plexes_from_sdrf(self.config.input.sdrf)
        if not sample_to_plex:
            raise ValueError("IRS normalization found no plex assignments in the SDRF")

        # Apply IRS
        normalizer = IRSNormalizer(
            reference_samples=ref_samples, stat=self.config.irs.stat
        )
        protein_df = normalizer.fit_transform(protein_df, sample_to_plex)

        logger.info("IRS normalization complete: %d proteins", len(protein_df))

        # Optionally remove reference samples from output
        if self.config.irs.remove_reference:
            protein_col = protein_df.columns[0]
            cols_to_keep = [protein_col] + [
                c
                for c in protein_df.columns
                if c == protein_col or c not in ref_samples
            ]
            # Deduplicate while preserving order
            seen = set()
            unique_cols = []
            for c in cols_to_keep:
                if c not in seen:
                    seen.add(c)
                    unique_cols.append(c)
            protein_df = protein_df[unique_cols]
            logger.info("Removed %d reference samples from output", len(ref_samples))

        return protein_df

    def apply_coverage_filter(
        self, protein_df: pd.DataFrame, dataset=None
    ) -> pd.DataFrame:
        """Apply coverage filter using condition mapping from SDRF.

        The optional ``dataset`` parameter is accepted for compatibility with
        the pipeline runner (``pipeline.runner``) and is otherwise unused.
        """
        from mokume.quantification.ratio import apply_coverage_filter
        from mokume.normalization.irs import detect_condition_from_sdrf

        if not self.config.input.sdrf:
            logger.warning("Coverage filter requires SDRF file, skipping")
            return protein_df

        sample_to_condition = detect_condition_from_sdrf(self.config.input.sdrf)

        # Filter out reference/powder conditions for coverage computation
        protein_col = protein_df.columns[0]
        available_samples = [c for c in protein_df.columns if c != protein_col]
        sample_to_condition = {
            s: c
            for s, c in sample_to_condition.items()
            if s in available_samples
            and "powder" not in c.lower()
            and "pool" not in c.lower()
        }
        if not sample_to_condition:
            raise ValueError(
                "Coverage filtering found no non-reference samples with condition metadata"
            )

        return apply_coverage_filter(
            protein_df,
            sample_to_condition,
            self.config.quantification.coverage_threshold,
        )

    def apply_sample_correlation_filter(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """Filter samples by mean within-condition protein correlation."""
        threshold = self.config.quantification.sample_correlation_threshold
        if threshold is None:
            return protein_df
        if not self.config.input.sdrf:
            raise ValueError("Sample correlation filtering requires an SDRF file")
        sample_to_condition = irs_normalization.detect_condition_from_sdrf(
            self.config.input.sdrf
        )
        return filter_samples_by_correlation(
            protein_df,
            sample_to_condition,
            threshold,
            values_are_log2=self.config.quantification.method.lower()
            in {"abd", "abundance", "tmtabundance", "ratio"},
        )


class QuantificationStage:
    """Routes to the appropriate quantification method.

    With DE precision weights, a MaxLFQ run leaves its cell precisions in
    ``cell_precision`` (protein matrix layout).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cell_precision = None

    def quantify(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Apply protein quantification method."""
        from mokume.quantification import get_quantification_method, TopNQuantification

        quant_method = self.config.quantification.method.lower()

        if quant_method == "pibaq":
            return self._quantify_pibaq(peptide_df)
        if quant_method in (
            "maxlfq",
            "sum",
            "all",
            "abd",
            "abundance",
            "intensity",
            "reporter",
            "peptide_count",
            "peptidecount",
        ):
            # Keep per-protein MaxLFQ workers inside the pipeline resource budget.
            quant_kwargs = self._quantification_options(quant_method)
            effective = self.config.runtime.effective_workers()
            if effective is not None:
                quant_kwargs["n_jobs"] = effective
                quant_kwargs["num_cores"] = effective
            method = get_quantification_method(quant_method, **quant_kwargs)
            result = method.quantify(
                peptide_df,
                protein_column=PROTEIN_NAME,
                peptide_column=PEPTIDE_CANONICAL,
                intensity_column=NORM_INTENSITY,
                sample_column=SAMPLE_ID,
            )
            self._keep_cell_precision(result)
            return self.to_wide_format(result, quant_method)
        if quant_method.startswith("top"):
            match = re.match(r"top(\d+)", quant_method)
            if match:
                n = int(match.group(1))
            else:
                n = 3
            method = TopNQuantification(n=n)
            # TopN itself is single-process; nothing extra to thread.
            result = method.quantify(
                peptide_df,
                protein_column=PROTEIN_NAME,
                peptide_column=PEPTIDE_CANONICAL,
                intensity_column=NORM_INTENSITY,
                sample_column=SAMPLE_ID,
            )
            return self.to_wide_format(result, quant_method)
        if quant_method == "median":
            return self._quantify_median(peptide_df)
        raise ValueError(f"Unknown quantification method: {quant_method}")

    def _keep_cell_precision(self, result: pd.DataFrame) -> None:
        """Keep MaxLFQ's cell precisions, if requested, as a protein matrix."""
        if "Precision" in result:
            self.cell_precision = result.pivot(
                index=PROTEIN_NAME, columns=SAMPLE_ID, values="Precision"
            ).reset_index()

    def _quantification_options(self, method: str) -> dict:
        """Method-specific solver settings, independent of worker allocation."""
        if method == "maxlfq":
            return {
                "min_ratio_count": self.config.quantification.maxlfq_min_ratio_count,
                "stabilize": self.config.quantification.stabilize,
                "cell_precision": self.config.de.precision_weights,
            }
        return {}

    def to_wide_format(self, long_df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """Convert long format quantification results to wide format."""
        _ = method_name
        intensity_col = "Intensity"

        if intensity_col not in long_df.columns:
            # Fallback: use the last numeric column
            numeric_cols = long_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                intensity_col = numeric_cols[-1]

        logger.debug("Using intensity column: %s", intensity_col)

        # Pivot to wide format
        wide_df = long_df.pivot(
            index=PROTEIN_NAME,
            columns=SAMPLE_ID,
            values=intensity_col,
        )

        return wide_df.reset_index()

    def _pibaq_references(self):
        """Digest the configured FASTA and resolve piBAQ families lazily."""
        fasta_module = importlib.import_module("mokume.io.fasta")
        family_module = importlib.import_module("mokume.quantification.families")
        quant_cfg = self.config.quantification
        accession_to_peptides, peptide_to_accessions, _ = getattr(
            fasta_module, "digest_fasta_full"
        )(
            fasta=self.config.input.fasta_file,
            enzyme=quant_cfg.pibaq_enzyme,
            min_aa=self.config.filtering.min_aa,
            max_aa=quant_cfg.pibaq_max_aa,
            canonicalize_isoforms=True,
            compute_mw=False,
        )
        families = getattr(family_module, "discover_families")(
            accession_to_peptides,
            peptide_to_accessions,
            min_shared=quant_cfg.pibaq_min_shared,
        )
        if quant_cfg.pibaq_families_yaml:
            overrides = getattr(family_module, "load_families_yaml")(
                Path(quant_cfg.pibaq_families_yaml)
            )
            families = getattr(family_module, "merge_overrides")(families, overrides)
        return accession_to_peptides, peptide_to_accessions, families

    def _quantify_pibaq(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Quantify using piBAQ (paralog-aware iBAQ).

        Delegates to :func:`mokume.quantification.pibaq.compute_pibaq`, the
        same core used by the standalone ``peptides2protein`` CLI. The two
        entry points share that core, so they agree on piBAQ values **when
        configured identically** -- the digestion enzyme, ``min_aa`` /
        ``max_aa``, family ``min_shared`` and any YAML family overrides must
        all match. Those knobs are sourced here from
        :class:`~mokume.pipeline.config.QuantificationConfig`
        (``pibaq_enzyme`` / ``pibaq_max_aa`` / ``pibaq_min_shared`` /
        ``pibaq_families_yaml`` / ``pibaq_min_anchors`` /
        ``pibaq_high_anchor_threshold``) and :attr:`FilterConfig.min_aa`.

        TPA is intentionally **not** emitted on this path: the pipeline
        contract returns a single wide protein x sample piBAQ matrix, which
        has no column for a parallel TPA value. Use the ``peptides2protein``
        CLI with ``--tpa`` when a TPA table is needed. Accordingly
        ``mw_map`` is left ``None`` here.
        """
        compute_pibaq = getattr(
            importlib.import_module("mokume.quantification.pibaq"),
            "compute_pibaq",
        )

        if not self.config.input.fasta_file:
            raise ValueError(
                "piBAQ quantification requires a FASTA file. Use --fasta to provide one."
            )

        quant_cfg = self.config.quantification
        logger.info(
            "Computing piBAQ for %d proteins using FASTA "
            "(enzyme=%s, max_aa=%d, min_shared=%d)...",
            peptide_df[PROTEIN_NAME].nunique(),
            quant_cfg.pibaq_enzyme,
            quant_cfg.pibaq_max_aa,
            quant_cfg.pibaq_min_shared,
        )

        accession_to_peptides, peptide_to_accessions, families = (
            self._pibaq_references()
        )

        long_df = compute_pibaq(
            peptide_df,
            accession_to_peptides,
            peptide_to_accessions,
            families,
            mw_map=None,
            min_anchors=quant_cfg.pibaq_min_anchors,
            high_anchor_threshold=quant_cfg.pibaq_high_anchor_threshold,
        )

        if long_df.empty:
            logger.info("piBAQ complete: 0 proteins")
            return pd.DataFrame(columns=[PROTEIN_NAME])

        result_wide = long_df.pivot(
            index=PROTEIN_NAME,
            columns=SAMPLE_ID,
            values=PIBAQ,
        )
        logger.info("piBAQ complete: %d proteins", len(result_wide))
        return result_wide.reset_index()

    def _quantify_median(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Quantify using median of peptides."""
        result = (
            peptide_df.groupby([PROTEIN_NAME, SAMPLE_ID], observed=False)[
                NORM_INTENSITY
            ]
            .median()
            .reset_index()
        )

        result_wide = result.pivot(
            index=PROTEIN_NAME, columns=SAMPLE_ID, values=NORM_INTENSITY
        )

        return result_wide.reset_index()


class ImputationStage:
    """Impute missing values on the wide protein-level matrix.

    The matrix is transformed to log2 space before imputation and back to
    linear space afterwards because most imputation methods assume
    log-normal intensities (notably MinProb/MinDet/QRILC).
    """

    _SUPPORTED_METHODS = (
        "none",
        "knn",
        "minprob",
        "mindet",
        "qrilc",
        "missforest",
        "seqknn",
        "impseq",
    )

    def __init__(self, config: PipelineConfig):
        self.config = config

    def impute(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to the wide protein matrix."""
        if not self.config.imputation.enabled:
            return protein_df

        method = self.config.imputation.method.lower()
        if method == "none":
            return protein_df

        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown imputation method: {method}. "
                f"Supported: {', '.join(self._SUPPORTED_METHODS)}"
            )

        if PROTEIN_NAME in protein_df.columns:
            protein_col = PROTEIN_NAME
        elif "protein" in protein_df.columns:
            protein_col = "protein"
        else:
            protein_col = protein_df.columns[0]

        sample_cols = [c for c in protein_df.columns if c != protein_col]
        wide = protein_df.set_index(protein_col)[sample_cols]
        if np.isinf(wide.to_numpy(dtype=float)).any():
            raise ValueError("Protein matrix contains an infinite intensity")

        wide_log2 = np.log2(wide.replace(0, np.nan))
        n_before = int(wide_log2.isna().sum().sum())
        if n_before == 0:
            logger.info("Imputation (%s): no missing values, skipping", method)
            return protein_df

        imputed_log2 = self._dispatch(method, wide_log2)
        imputed_values = imputed_log2.to_numpy(dtype=float)
        if np.isinf(imputed_values).any():
            raise ValueError(f"Imputation method '{method}' produced an infinite value")
        observed = np.isfinite(wide_log2.to_numpy(dtype=float))
        if not np.isfinite(imputed_values[observed]).all():
            raise ValueError(
                f"Imputation method '{method}' replaced an observed value with missing data"
            )
        n_after = int(imputed_log2.isna().sum().sum())
        logger.info(
            "Imputation (%s): %d -> %d missing values", method, n_before, n_after
        )

        try:
            with np.errstate(over="raise", invalid="raise"):
                imputed_linear = np.exp2(imputed_log2)
        except FloatingPointError as exc:
            raise ValueError(
                f"Imputation method '{method}' overflowed on the linear scale"
            ) from exc
        return imputed_linear.reset_index()

    def _dispatch(self, method: str, wide: pd.DataFrame) -> pd.DataFrame:
        """Dispatch to the matching imputation backend."""
        cfg = self.config.imputation

        if method == "knn":
            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=cfg.n_neighbors, keep_empty_features=True)
            arr = imputer.fit_transform(wide)
            return pd.DataFrame(arr, index=wide.index, columns=wide.columns)

        if method == "minprob":
            from mokume.imputation.censored import impute_minprob

            return impute_minprob(
                wide,
                quantile=cfg.quantile,
                tune_sigma=cfg.tune_sigma,
                random_state=cfg.seed,
            )

        if method == "mindet":
            from mokume.imputation.censored import impute_mindet

            return impute_mindet(wide, quantile=cfg.quantile)

        if method == "qrilc":
            from mokume.imputation.qrilc import impute_qrilc

            return impute_qrilc(wide, tune_sigma=cfg.tune_sigma, random_state=cfg.seed)

        if method == "missforest":
            from mokume.imputation.missforest import impute_missforest

            return impute_missforest(wide)

        if method == "seqknn":
            from mokume.imputation.seqknn import impute_seqknn

            return impute_seqknn(wide, k=cfg.n_neighbors)

        if method == "impseq":
            from mokume.imputation.impseq import impute_impseq

            return impute_impseq(wide)

        raise ValueError(f"Unknown imputation method: {method}")


def _auto_select_de_method(
    sample_to_condition: dict[str, str],
    quant: str,
) -> str:
    """Pick a DE method from replicate counts and quantification metadata.

    With tiny groups (min < 3), prefer the permutation-based ROTS; otherwise
    keep the quantification-based default (DEqMS for directLFQ, LimROTS
    otherwise).
    """
    min_grp = min(Counter(sample_to_condition.values()).values(), default=3)
    if min_grp < 3:
        return "rots"
    return "deqms" if quant == "directlfq" else "limrots"


class PostprocessingStage:
    """Batch correction, DE, plotting, reports."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _protein_and_sample_columns(self, protein_df: pd.DataFrame) -> tuple[str, list]:
        """Return the protein identifier column and quantitative sample columns."""
        if PROTEIN_NAME in protein_df.columns:
            protein_col = PROTEIN_NAME
        elif "protein" in protein_df.columns:
            protein_col = "protein"
        else:
            protein_col = protein_df.columns[0]

        return protein_col, [c for c in protein_df.columns if c != protein_col]

    def _batch_method(self) -> BatchDetectionMethod:
        """Resolve the configured batch-detection method."""
        return BatchDetectionMethod.from_str(self.config.batch.method)

    def _detect_batch_indices(self, sample_cols: list) -> list:
        """Detect and validate batch assignments for sample columns."""
        batch_indices = detect_batches(
            sample_ids=sample_cols,
            method=self._batch_method(),
            batch_column_values=(
                self._get_batch_column_values(sample_cols)
                if self.config.batch.column
                else None
            ),
        )

        unique_batches = len(set(batch_indices))
        logger.info("Detected %d batches for batch correction", unique_batches)
        if unique_batches < 2:
            raise ValueError("Batch correction requires at least two batches")

        min_samples = min(Counter(batch_indices).values())
        if min_samples < 2:
            raise ValueError(
                "Batch correction requires at least two samples in every batch; "
                f"minimum observed batch size is {min_samples}"
            )
        return batch_indices

    def _batch_covariates(self, sample_cols: list) -> Optional[list]:
        """Extract configured SDRF covariates for batch correction."""
        if not (self.config.input.sdrf and self.config.batch.covariates):
            return None

        covariates = extract_covariates_from_sdrf(
            self.config.input.sdrf,
            sample_cols,
            self.config.batch.covariates,
        )
        if covariates:
            logger.info(
                "Extracted %d covariates to preserve biological signal",
                len(self.config.batch.covariates),
            )
        return covariates

    def _validate_combat_request(
        self,
        batch_indices: list[int],
        covariates: Optional[list],
    ) -> None:
        """Reject invalid reference labels and singular ComBat designs."""
        ref_batch = self.config.batch.ref_batch
        if ref_batch is not None and (
            ref_batch < 0 or ref_batch not in set(batch_indices)
        ):
            raise ValueError(
                f"Batch reference '{ref_batch}' is not present; "
                f"detected labels are {sorted(set(batch_indices))}"
            )
        if covariates is None:
            return
        covariate_matrix = np.asarray(covariates, dtype=float)
        if covariate_matrix.ndim != 2 or covariate_matrix.shape[0] != len(
            batch_indices
        ):
            raise ValueError(
                "Batch covariate row count must match the matrix sample count"
            )
        labels = sorted(set(batch_indices))
        batch_rows = np.vstack(
            [np.asarray(batch_indices) == label for label in labels]
        ).astype(float)
        if ref_batch is not None:
            batch_rows[labels.index(ref_batch), :] = 1.0
        design = np.vstack([batch_rows, covariate_matrix.T])
        if np.linalg.matrix_rank(design) < design.shape[0]:
            raise ValueError(
                "Batch covariates are confounded with batches or each other; "
                "ComBat design is singular"
            )

    def _complete_batch_matrix(
        self,
        intensity_matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split complete and incomplete rows before ComBat correction."""
        has_nan = intensity_matrix.isna().any(axis=1)
        complete_matrix = intensity_matrix[~has_nan]
        incomplete_matrix = intensity_matrix[has_nan]

        if len(complete_matrix) == 0:
            raise ValueError(
                "Batch correction requires at least one protein observed in every sample"
            )

        if len(incomplete_matrix) > 0:
            logger.info(
                "ComBat: %d complete proteins, %d with missing values "
                "(kept uncorrected)",
                len(complete_matrix),
                len(incomplete_matrix),
            )
        return complete_matrix, incomplete_matrix

    def _rebuild_corrected_df(
        self,
        corrected_matrix: pd.DataFrame,
        incomplete_matrix: pd.DataFrame,
        protein_col: str,
    ) -> pd.DataFrame:
        """Recombine corrected complete rows with uncorrected incomplete rows."""
        if len(incomplete_matrix) > 0:
            corrected_matrix = pd.concat([corrected_matrix, incomplete_matrix])

        corrected_df = corrected_matrix.reset_index()
        return corrected_df.rename(columns={"index": protein_col})

    def apply_batch_correction(
        self, protein_df: pd.DataFrame, dataset=None
    ) -> pd.DataFrame:
        """Apply batch correction after protein quantification.

        Batch = technical variation to REMOVE (from runs/files)
        Covariates = biological signal to PRESERVE (from SDRF characteristics)

        The optional ``dataset`` parameter is accepted for compatibility with
        the pipeline runner (``pipeline.runner``) and is otherwise unused.
        """
        if not is_batch_correction_available():
            raise ImportError(
                "Batch correction requires inmoose package. "
                "Install with: pip install mokume-py[batch-correction]"
            )

        protein_col, sample_cols = self._protein_and_sample_columns(protein_df)
        if len(sample_cols) < 2:
            raise ValueError("Batch correction requires at least two matrix samples")

        intensity_matrix = protein_df.set_index(protein_col)[sample_cols]
        batch_indices = self._detect_batch_indices(sample_cols)
        covariates = self._batch_covariates(sample_cols)
        self._validate_combat_request(batch_indices, covariates)

        matrices = self._complete_batch_matrix(intensity_matrix)
        complete_matrix, incomplete_matrix = matrices

        logger.info("Applying ComBat batch correction...")
        try:
            corrected_matrix = apply_batch_correction(
                df=complete_matrix,
                batch=batch_indices,
                covs=covariates,
                kwargs={
                    "par_prior": self.config.batch.parametric,
                    "mean_only": self.config.batch.mean_only,
                    "ref_batch": self.config.batch.ref_batch,
                },
            )

            corrected_df = self._rebuild_corrected_df(
                corrected_matrix,
                incomplete_matrix,
                protein_col,
            )
            logger.info(
                "Batch correction complete: %d proteins, %d samples",
                len(corrected_df),
                len(sample_cols),
            )
            return corrected_df

        except Exception as e:
            raise ValueError(f"Batch correction failed: {e}") from e

    def _parse_de_contrasts(self, sample_to_condition: dict) -> list[tuple[str, str]]:
        """Parse configured DE contrasts into condition pairs."""
        if not self.config.de.contrasts:
            conditions = sorted(set(sample_to_condition.values()))
            raise ValueError(
                "Differential expression requires explicit contrasts "
                "via --de-contrasts.\n"
                f"Available conditions ({len(conditions)}): {conditions}\n"
                "Format: --de-contrasts 'GroupA vs GroupB,GroupA vs GroupC'"
            )

        contrasts = []
        for contrast_text in self.config.de.contrasts:
            if " vs " in contrast_text:
                parts = contrast_text.split(" vs ", 1)
            elif "-" in contrast_text:
                parts = contrast_text.split("-", 1)
            else:
                raise ValueError(
                    f"Invalid contrast format '{contrast_text}', expected 'A vs B' or 'A-B'"
                )
            contrasts.append((parts[0].strip(), parts[1].strip()))
        return contrasts

    def _resolve_de_method(
        self,
        protein_df: Optional[pd.DataFrame] = None,
        sample_to_condition: Optional[dict] = None,
    ) -> str:
        """Resolve the configured DE method, including data-aware auto selection.

        When ``auto`` and sample metadata are available, use replicate counts
        before falling back to the quantification-based default.
        """
        de_method = self.config.de.method.strip().lower()
        if de_method != "auto":
            return de_method

        quant = self.config.quantification.method.lower()
        if protein_df is not None and sample_to_condition:
            active_conditions = {
                sample: sample_to_condition.get(sample, "unknown")
                for sample in protein_df.columns[1:]
            }
            choice = _auto_select_de_method(active_conditions, quant)
            min_grp = min(Counter(active_conditions.values()).values(), default=0)
            logger.info(
                "Auto-selected DE method: %s (n_proteins=%d, min_group=%d, quant=%s)",
                choice,
                len(protein_df),
                min_grp,
                quant,
            )
            return choice

        de_method = "deqms" if quant == "directlfq" else "limrots"
        logger.info(
            "Auto-selected DE method: %s (quant=%s, no profile)", de_method, quant
        )
        return de_method

    def _load_de_peptide_counts(self, de_method: str) -> Optional[pd.Series]:
        """Load protein-level unique peptide counts when DEqMS needs them."""
        explicit_ensemble_methods = None
        if de_method == "ensemble" and self.config.de.ensemble_methods is not None:
            explicit_ensemble_methods = validate_de_config(self.config.de)
        needs_counts = de_method == "deqms" or (
            de_method == "ensemble"
            and explicit_ensemble_methods is not None
            and "deqms" in explicit_ensemble_methods
        )
        if not (needs_counts and self.config.input.parquet):
            return None

        try:
            pep_df = pd.read_parquet(
                self.config.input.parquet,
                columns=["anchor_protein", "sequence"],
            )
            return pep_df.groupby("anchor_protein")["sequence"].nunique()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("Could not load peptide counts for DEqMS: %s", exc)
            return None

    def _de_output_path(self, contrast_key: str, n_contrasts: int) -> Optional[str]:
        """Return the output path for a DE contrast, if output is enabled."""
        if not self.config.de.output:
            return None
        if n_contrasts == 1:
            return self.config.de.output

        base, ext = os.path.splitext(self.config.de.output)
        return f"{base}_{contrast_key}{ext}" if ext else f"{base}_{contrast_key}.csv"

    def _save_de_result(
        self,
        result: pd.DataFrame,
        contrast_key: str,
        n_contrasts: int,
    ) -> None:
        """Save a DE result table when configured."""
        output_path = self._de_output_path(contrast_key, n_contrasts)
        if output_path is None:
            return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("DE results saved to %s", output_path)

    def _run_ensemble_de(
        self,
        protein_df: pd.DataFrame,
        sample_to_condition: dict,
        contrasts: list[tuple[str, str]],
        peptide_counts: Optional[pd.Series],
    ) -> dict:
        """Run ensemble DE for all configured contrasts."""
        from mokume.analysis.ensemble import run_ensemble

        ensemble_methods = self.config.de.ensemble_methods
        all_results = {}
        for contrast in contrasts:
            key = f"{contrast[0]}-{contrast[1]}"
            logger.info(
                "Running ensemble DE: %s (methods=%s, min_k=%d)",
                key,
                ensemble_methods,
                self.config.de.ensemble_min_k,
            )
            result = run_ensemble(
                protein_df,
                sample_to_condition,
                contrast,
                methods=ensemble_methods,
                min_k=self.config.de.ensemble_min_k,
                fdr_method=self.config.de.fdr_method,
                fdr_threshold=self.config.de.fdr_threshold,
                log2fc_threshold=self.config.de.log2fc_threshold,
                peptide_counts=peptide_counts,
            )
            all_results[key] = {"df": result, "conditions": contrast}
            self._save_de_result(result, key, len(contrasts))
        return all_results

    def _run_single_de_method(
        self,
        protein_df: pd.DataFrame,
        sample_to_condition: dict,
        contrasts: list[tuple[str, str]],
        de_method: str,
        de_inputs: tuple[Optional[pd.Series], Optional[pd.DataFrame]],
    ) -> dict:
        """Run a non-ensemble DE method for all configured contrasts.

        ``de_inputs`` holds the peptide counts and the cell precisions.
        """
        from mokume.analysis.differential_expression import DifferentialExpression

        peptide_counts, precision = de_inputs
        de = DifferentialExpression(
            method=de_method,
            log2fc_threshold=self.config.de.log2fc_threshold,
            fdr_threshold=self.config.de.fdr_threshold,
            fdr_method=self.config.de.fdr_method,
            skip_log2=(self.config.quantification.method.lower() == "ratio"),
            peptide_counts=peptide_counts,
            precision=precision,
        )

        all_results = {}
        for contrast in contrasts:
            key = f"{contrast[0]}-{contrast[1]}"
            logger.info("Running DE: %s", key)
            result = de.run(protein_df, sample_to_condition, contrast)
            all_results[key] = {"df": result, "conditions": contrast}
            self._save_de_result(result, key, len(contrasts))
        return all_results

    def run_differential_expression(
        self, protein_df: pd.DataFrame, dataset=None, precision=None
    ) -> Optional[dict]:
        """Run differential expression analysis.

        The optional ``dataset`` parameter is accepted for compatibility with
        the pipeline runner (``pipeline.runner``) and is otherwise unused.
        ``precision`` holds the quantification's cell precisions for
        precision-weighted DEqMS.
        """
        from mokume.normalization.irs import detect_condition_from_sdrf

        validate_de_config(self.config.de)

        if not self.config.input.sdrf:
            raise ValueError("Differential expression requires an SDRF file (--sdrf)")

        sample_to_condition = detect_condition_from_sdrf(self.config.input.sdrf)
        contrasts = self._parse_de_contrasts(sample_to_condition)
        if not contrasts:
            return None

        de_method = self._resolve_de_method(protein_df, sample_to_condition)
        if (
            de_method in {"rots", "limrots"}
            and self.config.de.fdr_method.lower() != "bh"
        ):
            raise ValueError(
                f"FDR method '{self.config.de.fdr_method}' does not apply to "
                f"{de_method}, which retains its permutation FDR"
            )
        if self.config.de.precision_weights:
            self._validate_precision_weights(de_method)
        peptide_counts = self._load_de_peptide_counts(de_method)

        if de_method == "ensemble":
            return self._run_ensemble_de(
                protein_df,
                sample_to_condition,
                contrasts,
                peptide_counts,
            )

        return self._run_single_de_method(
            protein_df,
            sample_to_condition,
            contrasts,
            de_method,
            (peptide_counts, precision),
        )

    def _validate_precision_weights(self, de_method: str) -> None:
        """Precision weights need MaxLFQ/DirectLFQ values tested by DEqMS."""
        if self.config.quantification.method.lower() not in ("maxlfq", "directlfq"):
            raise ValueError(
                "DE precision weights require maxlfq or directlfq quantification"
            )
        if de_method != "deqms":
            raise ValueError(
                f"DE precision weights require the deqms DE method, not {de_method}"
            )

    def _plot_sample_conditions(self, protein_df: pd.DataFrame) -> Optional[dict]:
        """Return sample-to-condition mapping for plotting when SDRF is available."""
        from mokume.normalization.irs import detect_condition_from_sdrf

        if not self.config.input.sdrf:
            return None

        sample_to_condition = detect_condition_from_sdrf(self.config.input.sdrf)
        if not self.config.irs.remove_reference:
            return sample_to_condition

        protein_col = protein_df.columns[0]
        available_samples = [c for c in protein_df.columns if c != protein_col]
        return {
            sample: condition
            for sample, condition in sample_to_condition.items()
            if sample in available_samples
        }

    def _generate_volcano_plots(
        self, de_results: Optional[dict], plot_dir: Path
    ) -> None:
        """Generate configured volcano plots."""
        if not (self.config.output.plot_volcano and de_results):
            return

        from mokume.plotting.differential_expression import plot_volcano

        for contrast_name, de_entry in de_results.items():
            output_file = str(plot_dir / f"volcano_{contrast_name}.png")
            plot_volcano(
                de_entry["df"],
                log2fc_threshold=self.config.de.log2fc_threshold,
                fdr_threshold=self.config.de.fdr_threshold,
                highlight_genes=self.config.output.highlight_genes,
                title=f"Volcano Plot: {contrast_name}",
                output_file=output_file,
                conditions=tuple(de_entry["conditions"]),
            )
            logger.info("Volcano plot saved to %s", output_file)

    def _significant_de_proteins(self, de_df: pd.DataFrame) -> pd.DataFrame:
        """Return significant DE rows under configured thresholds."""
        return de_df[
            (de_df["adj_pvalue"] < self.config.de.fdr_threshold)
            & (de_df["log2FC"].abs() > self.config.de.log2fc_threshold)
        ]

    def _generate_heatmap_plots(
        self,
        protein_df: pd.DataFrame,
        de_results: Optional[dict],
        sample_to_condition: Optional[dict],
        plot_dir: Path,
    ) -> None:
        """Generate configured per-contrast DE heatmaps."""
        if not (self.config.output.plot_heatmap and sample_to_condition and de_results):
            return

        from mokume.plotting.differential_expression import plot_heatmap

        protein_col = protein_df.columns[0]
        for contrast_name, de_entry in de_results.items():
            sig = self._significant_de_proteins(de_entry["df"])
            if sig.empty:
                logger.info(
                    "Heatmap skipped for %s: no significant proteins",
                    contrast_name,
                )
                continue

            cond_a, cond_b = de_entry["conditions"]
            sig_proteins = (
                sig.assign(abs_log2FC=sig["log2FC"].abs())
                .sort_values(["adj_pvalue", "abs_log2FC"], ascending=[True, False])[
                    "ProteinName"
                ]
                .head(50)
                .tolist()
            )
            contrast_samples = [
                sample
                for sample in protein_df.columns
                if sample != protein_col
                and sample_to_condition.get(sample) in (cond_a, cond_b)
            ]
            contrast_mapping = {
                sample: condition
                for sample, condition in sample_to_condition.items()
                if condition in (cond_a, cond_b)
            }
            output_file = str(plot_dir / f"heatmap_{contrast_name}.png")
            plot_heatmap(
                protein_df[[protein_col] + contrast_samples],
                contrast_mapping,
                proteins=sig_proteins,
                title=(
                    f"DE Heatmap: {contrast_name} "
                    f"(top {len(sig_proteins)}/{len(sig)} sig)"
                ),
                output_file=output_file,
            )
            logger.info("Heatmap saved to %s", output_file)

    def _generate_pca_plot(
        self,
        protein_df: pd.DataFrame,
        sample_to_condition: Optional[dict],
        plot_dir: Path,
    ) -> None:
        """Generate configured PCA-by-condition plot."""
        if not (self.config.output.plot_pca and sample_to_condition):
            return

        from mokume.plotting.differential_expression import plot_pca_conditions

        output_file = str(plot_dir / "pca_conditions.png")
        plot_pca_conditions(
            protein_df,
            sample_to_condition,
            title="PCA by Condition",
            output_file=output_file,
        )
        logger.info("PCA plot saved to %s", output_file)

    def generate_plots(
        self,
        protein_df: pd.DataFrame,
        de_results: Optional[dict] = None,
        dataset=None,
    ):
        """Generate requested plots.

        The optional ``dataset`` parameter is accepted for compatibility with
        the pipeline runner (``pipeline.runner``) and is otherwise unused.
        """
        from mokume.plotting import is_plotting_available

        if not is_plotting_available():
            logger.warning(
                "Plotting dependencies not available. "
                "Install with: pip install mokume-py[plotting]"
            )
            return

        plot_dir = Path(self.config.output.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        sample_to_condition = self._plot_sample_conditions(protein_df)
        self._generate_volcano_plots(de_results, plot_dir)
        self._generate_heatmap_plots(
            protein_df,
            de_results,
            sample_to_condition,
            plot_dir,
        )
        self._generate_pca_plot(protein_df, sample_to_condition, plot_dir)

    def generate_interactive_report(
        self, protein_df: pd.DataFrame, de_results: dict, dataset=None
    ):
        """Generate interactive HTML report for DE results.

        The optional ``dataset`` parameter is accepted for compatibility with
        the pipeline runner (``pipeline.runner``) and is otherwise unused.
        """
        from mokume.reports import is_interactive_available

        if not is_interactive_available():
            logger.warning(
                "Interactive report dependencies (plotly) not available. "
                "Install with: pip install mokume-py[reports]"
            )
            return

        from mokume.reports.interactive import generate_de_report
        from mokume.normalization.irs import detect_condition_from_sdrf

        if not self.config.input.sdrf:
            logger.warning("Interactive report requires SDRF file, skipping")
            return

        sample_to_condition = detect_condition_from_sdrf(self.config.input.sdrf)

        contrast_conditions = {
            f"{a}-{b}": (a, b) for a, b in self._parse_de_contrasts(sample_to_condition)
        }
        for contrast_name, de_entry in de_results.items():
            de_df = de_entry["df"] if isinstance(de_entry, dict) else de_entry
            if self.config.output.report_output:
                if len(de_results) == 1:
                    output_html = self.config.output.report_output
                else:
                    base = self.config.output.report_output.rsplit(".", 1)
                    output_html = f"{base[0]}_{contrast_name}.html"
            else:
                plot_dir = self.config.output.plot_dir or "."
                output_html = str(Path(plot_dir) / f"report_{contrast_name}.html")

            generate_de_report(
                de_results=de_df,
                protein_df=protein_df,
                sample_to_condition={
                    sample: condition
                    for sample, condition in sample_to_condition.items()
                    if condition in contrast_conditions[contrast_name]
                },
                output_html=output_html,
                title=f"DE Report: {contrast_name}",
                highlight_genes=self.config.output.highlight_genes,
                log2fc_threshold=self.config.de.log2fc_threshold,
                fdr_threshold=self.config.de.fdr_threshold,
                conditions=contrast_conditions[contrast_name],
            )
            logger.info("Interactive report saved to %s", output_html)

    def _get_batch_column_values(self, sample_ids: list) -> Optional[list]:
        """Get batch values from SDRF for explicit batch column."""
        if not self.config.input.sdrf or not self.config.batch.column:
            return None

        sdrf = load_sdrf(self.config.input.sdrf)
        batch_col = self.config.batch.column.lower()
        if batch_col not in sdrf.columns:
            raise ValueError(f"Batch column '{self.config.batch.column}' not in SDRF")
        sample_to_batch = dict(zip(sdrf["source name"], sdrf[batch_col]))
        missing = [sample for sample in sample_ids if sample not in sample_to_batch]
        if missing:
            raise ValueError(
                f"Batch SDRF column '{self.config.batch.column}' has no values for: {missing}"
            )
        return [sample_to_batch[sample] for sample in sample_ids]
