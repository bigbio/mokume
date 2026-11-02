"""
QPX parquet data loading for TissueMap.

Handles both LFQ and TMT datasets with automatic GIS detection
for internal-standard normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Patterns that identify GIS / reference / pool channels in organism_part
_GIS_PATTERNS = re.compile(
    r"global\s*internal\s*standard|reference|pool", re.IGNORECASE
)

_NON_REFERENCE_POOLED_VALUES = frozenset(
    {
        "",
        "not pooled",
        "not applicable",
        "not available",
        "none",
        "na",
        "n/a",
    }
)

# A LATERAL UNNEST materialises the expansion and cannot spill; this form streams.
_FLAT_VIEW_QUERY = """
    CREATE TEMP VIEW flat_input AS
    SELECT
        protein,
        run_file_name,
        reporter.label AS tmt_label,
        CAST(reporter.intensity AS DOUBLE) AS intensity
    FROM (
        SELECT
            anchor_protein AS protein,
            run_file_name,
            UNNEST(intensities) AS reporter
        FROM feature_input
        WHERE NOT COALESCE(is_decoy, FALSE)
    )
    WHERE reporter.intensity > 0
"""

_TMT_GIS_AGGREGATION_QUERY = """
    WITH mapped AS (
        SELECT
            flat_input.protein,
            flat_input.run_file_name,
            mapping.sample_accession,
            mapping.is_gis,
            flat_input.intensity
        FROM flat_input
        INNER JOIN sample_map AS mapping
            USING (run_file_name, tmt_label)
    ),
    gis AS (
        SELECT
            protein,
            run_file_name,
            AVG(intensity) AS gis_intensity
        FROM mapped
        WHERE is_gis
        GROUP BY protein, run_file_name
    )
    SELECT
        mapped.protein,
        mapped.sample_accession,
        SUM(mapped.intensity / gis.gis_intensity) AS intensity
    FROM mapped
    INNER JOIN gis USING (protein, run_file_name)
    WHERE NOT mapped.is_gis
      AND gis.gis_intensity > 0
    GROUP BY mapped.protein, mapped.sample_accession
"""

_TMT_RAW_AGGREGATION_QUERY = """
    SELECT
        flat_input.protein,
        mapping.sample_accession,
        SUM(flat_input.intensity) AS intensity
    FROM flat_input
    INNER JOIN sample_map AS mapping
        USING (run_file_name, tmt_label)
    GROUP BY flat_input.protein, mapping.sample_accession
"""

# DuckDB orders NaN above every number, so the positive filter alone keeps it.
_LFQ_AGGREGATION_QUERY = """
    SELECT
        flat_input.protein,
        mapping.sample_accession,
        flat_input.tmt_label,
        SUM(flat_input.intensity) AS intensity
    FROM flat_input
    LEFT JOIN sample_map AS mapping USING (run_file_name)
    WHERE NOT isnan(flat_input.intensity)
    GROUP BY flat_input.protein, mapping.sample_accession, flat_input.tmt_label
"""


@dataclass(frozen=True)
class TissueLoadOptions:
    """Options applied while loading one tissue dataset."""

    min_samples: int = 1
    warning_threshold: int = 3
    n_jobs: int = 8


def _parse_samples_col(samples_val) -> list[dict]:
    """Parse the ``run.samples`` column (list/ndarray of dicts) into a list."""
    if samples_val is None:
        return []
    if isinstance(samples_val, (list, np.ndarray)):
        return list(samples_val)
    return []


def build_run_to_sample_map(ds_dir: Path, ds_id: str) -> pd.DataFrame:
    """Return DataFrame: run_file_name, tmt_label, sample_accession, fraction, batch."""
    run = pq.read_table(ds_dir / "qpx_output" / f"{ds_id}.run.parquet").to_pandas()
    rows: list[dict] = []
    for _, r in run.iterrows():
        for channel_index, s in enumerate(_parse_samples_col(r["samples"]), start=1):
            raw_label = s.get("label", "LFQ")
            label = raw_label.replace("AC=MS:1002038;NT=", "").strip()
            # Normalize common LFQ label variants
            if label.lower() in ("label free sample", "label-free", "lfq"):
                label = "LFQ"
            rows.append(
                {
                    "run_file_name": r["run_file_name"],
                    "fraction": r.get("fraction", 1),
                    "sample_accession": s["sample_accession"],
                    "tmt_label": label,
                    "channel_index": str(channel_index),
                    "batch": r["run_file_name"],
                }
            )
    return pd.DataFrame(rows)


def _detect_gis_accessions(
    ds_dir: Path,
    ds_id: str,
) -> set[str]:
    """Auto-detect GIS / reference sample accessions from sample.parquet.

    Flags samples whose ``organism_part`` matches a known GIS pattern or whose
    QPX ``pooled_sample`` metadata describes a real pool.  ``not pooled`` and
    other SDRF sentinel values are explicitly excluded.

    Returns the set of ``sample_accession`` strings that are GIS channels.
    """
    smp = pq.read_table(ds_dir / "qpx_output" / f"{ds_id}.sample.parquet").to_pandas()
    if "sample_accession" not in smp.columns:
        return set()

    gis_mask = pd.Series(False, index=smp.index)
    if "organism_part" in smp.columns:
        organism_part = smp["organism_part"].fillna("").astype(str)
        gis_mask |= organism_part.str.contains(_GIS_PATTERNS)

    if "pooled_sample" in smp.columns:
        pooled_sample = (
            smp["pooled_sample"].fillna("").astype(str).str.strip().str.lower()
        )
        gis_mask |= ~pooled_sample.isin(_NON_REFERENCE_POOLED_VALUES)

    gis_accessions = set(smp.loc[gis_mask, "sample_accession"])
    if gis_accessions:
        logger.info(
            "[%s] GIS auto-detect: %d reference accessions found",
            ds_id,
            len(gis_accessions),
        )
    return gis_accessions


def _labels_indicate_tmt(labels: pd.Series) -> bool:
    """Return whether any canonical run label identifies TMT/iTRAQ."""
    normalized = labels.fillna("").astype(str)
    return bool(normalized.str.contains(r"TMT|iTRAQ", case=False, regex=True).any())


def _tmt_label_aliases(run_map: pd.DataFrame) -> pd.DataFrame:
    """Return canonical and positional channel labels for each run/sample.

    Some quantms protein-group Parquet files encode reporter channels as
    ``1`` .. ``N`` even though QPX ``run.parquet`` stores canonical labels
    such as ``TMT126`` .. ``TMT131``.  The sample list order comes from the
    SDRF channel order, so the positional alias provides a deterministic join.
    Canonical labels always win when a numeric label already exists, preventing
    positional aliases from duplicating or overriding an explicit QPX mapping.
    """
    columns = ["run_file_name", "tmt_label", "sample_accession"]
    key_columns = ["run_file_name", "tmt_label"]
    canonical = run_map[columns].drop_duplicates().copy()
    positional = (
        run_map[["run_file_name", "channel_index", "sample_accession"]]
        .rename(columns={"channel_index": "tmt_label"})
        .drop_duplicates()
    )

    canonical_keys = (
        canonical[key_columns].drop_duplicates().assign(_canonical_label=True)
    )
    positional = positional.merge(canonical_keys, on=key_columns, how="left")
    positional = positional[positional["_canonical_label"].isna()].drop(
        columns="_canonical_label"
    )

    aliases = pd.concat([canonical, positional], ignore_index=True).drop_duplicates()
    conflicts = (
        aliases.groupby(key_columns, dropna=False)["sample_accession"]
        .nunique(dropna=False)
        .loc[lambda counts: counts > 1]
    )
    if not conflicts.empty:
        examples = ", ".join(
            f"{run_name}/{label}" for run_name, label in conflicts.index[:5]
        )
        raise ValueError(f"ambiguous QPX TMT channel mappings: {examples}")
    return aliases


def _resolve_gis_accessions(
    ds_dir: Path,
    ds_id: str,
    run_map: pd.DataFrame,
) -> set[str]:
    """Resolve reference accessions from metadata, with legacy label fallback."""
    gis_accessions = _detect_gis_accessions(ds_dir, ds_id)
    if gis_accessions:
        return gis_accessions

    fallback_labels = {"TMT126", "TMT131"}
    gis_rows = run_map[run_map["tmt_label"].isin(fallback_labels)]
    gis_accessions = set(gis_rows["sample_accession"])
    logger.warning(
        "[%s] GIS auto-detect found nothing, falling back to labels %s (%d accessions)",
        ds_id,
        fallback_labels,
        len(gis_accessions),
    )
    return gis_accessions


def _execute_aggregation(
    parquet_path: Path,
    sample_map: pd.DataFrame,
    workers: int,
    query: str,
) -> pd.DataFrame:
    """Execute a bounded DuckDB expansion and aggregation query."""
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"SET threads = {workers}")
        con.execute("SET preserve_insertion_order = false")
        con.register("sample_map", sample_map)
        con.read_parquet(str(parquet_path)).create_view("feature_input")
        con.execute(_FLAT_VIEW_QUERY)
        return con.execute(query).df()
    finally:
        con.close()


def _aggregate_tmt_with_duckdb(
    ds_dir: Path,
    ds_id: str,
    feature_prefix: str | None,
    run_map: pd.DataFrame,
    n_jobs: int,
) -> pd.DataFrame:
    """Stream TMT reporter intensities into protein/sample aggregates.

    DuckDB performs the reporter-channel expansion, run/channel metadata join,
    per-run GIS normalization, and fraction aggregation without materialising
    tens of millions of reporter rows in Pandas.
    """
    prefix = feature_prefix or ds_id
    parquet_path = ds_dir / "qpx_output" / f"{prefix}.feature.parquet"
    logger.info("[%s] Reading TMT input %s with DuckDB", ds_id, parquet_path.name)

    gis_accessions = _resolve_gis_accessions(ds_dir, ds_id, run_map)
    label_map = _tmt_label_aliases(run_map)
    label_map["is_gis"] = label_map["sample_accession"].isin(gis_accessions)

    workers = max(1, int(n_jobs))
    if not gis_accessions:
        logger.warning("[%s] No GIS data found; aggregating raw TMT intensities", ds_id)
    aggregated = _execute_aggregation(
        parquet_path,
        label_map,
        workers,
        _TMT_GIS_AGGREGATION_QUERY if gis_accessions else _TMT_RAW_AGGREGATION_QUERY,
    )

    if aggregated.empty:
        raise ValueError(
            f"{ds_id}: no TMT intensities matched QPX run/channel metadata"
        )
    logger.info(
        "[%s] TMT aggregation: %d protein/sample values, %d GIS accessions",
        ds_id,
        len(aggregated),
        len(gis_accessions),
    )
    return aggregated


def _aggregate_lfq_with_duckdb(
    ds_dir: Path,
    ds_id: str,
    feature_prefix: str | None,
    run_map: pd.DataFrame,
    n_jobs: int,
) -> pd.DataFrame:
    """Stream LFQ intensities into protein/sample/label aggregates.

    Unmatched runs stay in the result with a null sample so that their labels
    still count towards detecting TMT input; the caller drops them.
    """
    prefix = feature_prefix or ds_id
    parquet_path = ds_dir / "qpx_output" / f"{prefix}.feature.parquet"
    logger.info("[%s] Reading %s with DuckDB", ds_id, parquet_path.name)
    aggregated = _execute_aggregation(
        parquet_path,
        run_map.drop_duplicates("run_file_name")[["run_file_name", "sample_accession"]],
        max(1, int(n_jobs)),
        _LFQ_AGGREGATION_QUERY,
    )
    if aggregated.empty:
        raise ValueError(f"{ds_id}: no valid intensity values")
    return aggregated


def _pivot_and_annotate(
    merged: pd.DataFrame,
    ds_dir: Path,
    ds_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate tissues, pivot to matrix, and build sample metadata."""
    prot_samp = (
        merged.groupby(["protein", "sample_accession"])["intensity"].sum().reset_index()
    )

    smp = pq.read_table(ds_dir / "qpx_output" / f"{ds_id}.sample.parquet").to_pandas()
    smp = smp[["sample_accession", "organism_part"]].rename(
        columns={"organism_part": "tissue"}
    )
    prot_samp = prot_samp.merge(smp, on="sample_accession", how="left")
    prot_samp = prot_samp[
        prot_samp["tissue"].notna()
        & ~prot_samp["tissue"]
        .str.lower()
        .isin(["not available", "n/a", "", "global internal standard"])
    ]

    mat = prot_samp.pivot_table(
        index="protein",
        columns="sample_accession",
        values="intensity",
        aggfunc="sum",
    )
    tissue_meta = (
        prot_samp[["sample_accession", "tissue"]]
        .drop_duplicates("sample_accession")
        .set_index("sample_accession")
    )
    return mat, tissue_meta


def _filter_low_sample_tissues(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    ds_id: str,
    min_tissue_samples: int,
    low_sample_warning_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop tissues with too few samples and warn about borderline ones."""
    tissue_counts = meta["tissue"].value_counts()
    if min_tissue_samples > 1:
        valid_tissues = tissue_counts[tissue_counts >= min_tissue_samples].index
        dropped = tissue_counts[tissue_counts < min_tissue_samples]
        if len(dropped) > 0:
            logger.warning(
                "[%s] Dropped %d tissues with < %d samples: %s",
                ds_id,
                len(dropped),
                min_tissue_samples,
                ", ".join(dropped.index[:5]),
            )
        valid_samples = meta[meta["tissue"].isin(valid_tissues)].index
        mat = mat[mat.columns.intersection(valid_samples)]
        meta = meta.loc[meta.index.intersection(valid_samples)]

    if low_sample_warning_threshold > 0:
        low_sample = tissue_counts[tissue_counts < low_sample_warning_threshold]
        if len(low_sample) > 0:
            logger.warning(
                "[%s] %d tissues have < %d samples (TS scores may be unreliable): %s",
                ds_id,
                len(low_sample),
                low_sample_warning_threshold,
                ", ".join(f"{t}({c})" for t, c in low_sample.items()),
            )
    return mat, meta


def _attach_metadata(
    tissue_meta: pd.DataFrame,
    run_map: pd.DataFrame,
    quant_type: str,
    ds_id: str,
) -> None:
    """Attach batch, quant_type, and dataset columns to tissue metadata."""
    batch_map = run_map.drop_duplicates("sample_accession").set_index(
        "sample_accession"
    )["batch"]
    tissue_meta["batch"] = tissue_meta.index.map(batch_map).fillna("unknown")
    tissue_meta["quant_type"] = quant_type
    tissue_meta["dataset"] = ds_id


def load_dataset(
    ds_dir: Path,
    ds_id: str,
    *,
    is_tmt: bool | None = None,
    feature_prefix: str | None = None,
    options: TissueLoadOptions = TissueLoadOptions(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single QPX dataset and return (mat, meta).

    Parameters
    ----------
    ds_dir : Path
        Directory containing ``qpx_output/`` with parquet files.
    ds_id : str
        Dataset identifier (e.g. "PXD016999").
    is_tmt : bool | None
        Force TMT mode.  ``None`` = auto-detect from labels.
    feature_prefix : str | None
        Prefix for feature parquet (defaults to *ds_id*).
    options : TissueLoadOptions
        Sample-count thresholds and threads used for streaming TMT aggregation.

    Returns
    -------
    mat : pd.DataFrame
        Proteins (rows) x samples (columns), raw intensities.
    meta : pd.DataFrame
        Index = sample_accession, columns = tissue, batch, quant_type.
    """
    run_map = build_run_to_sample_map(ds_dir, ds_id)
    source = (ds_dir, ds_id, feature_prefix, run_map, options.n_jobs)
    quant_type = "TMT"
    if is_tmt is True or (
        is_tmt is None and _labels_indicate_tmt(run_map["tmt_label"])
    ):
        merged = _aggregate_tmt_with_duckdb(*source)
    else:
        merged = _aggregate_lfq_with_duckdb(*source)
        if is_tmt is None and _labels_indicate_tmt(merged["tmt_label"]):
            merged = _aggregate_tmt_with_duckdb(*source)
        else:
            quant_type = "LFQ"
            merged = merged.dropna(subset=["sample_accession"])

    mat, tissue_meta = _pivot_and_annotate(merged, ds_dir, ds_id)
    _attach_metadata(tissue_meta, run_map, quant_type, ds_id)

    mat, tissue_meta = _filter_low_sample_tissues(
        mat,
        tissue_meta,
        ds_id,
        options.min_samples,
        options.warning_threshold,
    )

    logger.info(
        "[%s] Loaded: %d proteins x %d samples, %d tissues (%s)",
        ds_id,
        mat.shape[0],
        mat.shape[1],
        tissue_meta["tissue"].nunique(),
        quant_type,
    )
    return mat, tissue_meta
