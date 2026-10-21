"""
Equivalence tests for the SQL-first ``LoadingStage.load_for_mokume`` path.

Build a small synthetic LFQ parquet + matching SDRF, then verify that the
SQL-first branch (whole filter+aggregation pipeline pushed into a single
DuckDB query) produces a peptide-level DataFrame equivalent to the legacy
per-sample pandas loop. Equivalence is required for rows, keys, dtypes, and
``NormIntensity`` values (within a small float tolerance, since parquet stores
intensities as float32 while the SQL path keeps them in float64).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mokume.core.constants import (
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PROTEIN_NAME,
    SAMPLE_ID,
)
from mokume.pipeline.config import (
    FilterConfig,
    InputConfig,
    NormalizationConfig,
    PipelineConfig,
    QuantificationConfig,
)
from mokume.pipeline.features_to_proteins import QuantificationPipeline
from mokume.pipeline.stages import LoadingStage
from mokume.io.feature import Feature, SQLFilterBuilder
from mokume.preprocessing.sdrf import analyse_sdrf


_NEW_INTENSITIES_TYPE = pa.list_(
    pa.struct([("label", pa.string()), ("intensity", pa.float32())])
)
_PG_PROTEIN_TYPE = pa.list_(
    pa.struct(
        [
            ("accession", pa.string()),
            ("start", pa.int32()),
            ("end", pa.int32()),
            ("pre", pa.string()),
            ("post", pa.string()),
        ]
    )
)
_LFQ_SCHEMA = pa.schema(
    [
        ("sequence", pa.string()),
        ("peptidoform", pa.string()),
        ("pg_accessions", _PG_PROTEIN_TYPE),
        ("anchor_protein", pa.string()),
        ("charge", pa.int16()),
        ("run_file_name", pa.string()),
        ("unique", pa.bool_()),
        ("is_decoy", pa.bool_()),
        ("intensities", _NEW_INTENSITIES_TYPE),
    ]
)


def _pg(accession: str) -> List[dict]:
    """Build a single-entry pg_accessions list with the QPX struct schema."""
    return [{"accession": accession, "start": 1, "end": 10, "pre": "K", "post": "A"}]


def _intensities(value: float) -> List[dict]:
    return [{"label": "LFQ", "intensity": float(value)}]


def _make_lfq_parquet(path: str) -> List[str]:
    """Create a tiny LFQ parquet exercising every branch of the load pipeline.

    Returns the ordered list of ``run_file_name`` values used (one per sample).
    """
    runs = ["run_S1", "run_S2", "run_S3", "run_S4"]

    rows: List[Tuple] = []

    # Protein P12345 (sp|P12345|GOOD_HUMAN): 3 distinct peptides (>=2 unique) in
    # every sample, several with two charges so peptidoform-max kicks in.
    p12345 = "sp|P12345|GOOD_HUMAN"
    for sample_run in runs:
        rows.append(("PEPTIDEAR", "PEPTIDEAR", p12345, 2, sample_run, 100.0 + 1.0))
        # Same peptide with a different charge → peptidoform-max should keep the
        # higher-intensity row.
        rows.append(("PEPTIDEAR", "PEPTIDEAR", p12345, 3, sample_run, 300.0 + 2.0))
        rows.append(("PEPTIDEBR", "PEPTIDEBR", p12345, 2, sample_run, 200.0 + 3.0))
        rows.append(("PEPTIDECR", "PEPTIDECR", p12345, 2, sample_run, 400.0 + 4.0))

    # Protein P67890: only 1 unique peptide → filtered out by min_unique_peptides=2.
    p67890 = "sp|P67890|LONE_HUMAN"
    for sample_run in runs:
        rows.append(("ONLYONEPEP", "ONLYONEPEP", p67890, 2, sample_run, 500.0))

    # Protein P99999 (bare accession, no pipes): 2 unique peptides → kept.
    p99999 = "P99999"
    for sample_run in runs:
        rows.append(("KEPTPEPAR", "KEPTPEPAR", p99999, 2, sample_run, 600.0))
        rows.append(("KEPTPEPBR", "KEPTPEPBR", p99999, 3, sample_run, 700.0))

    # DIA-NN contaminant prefix → must be filtered out by remove_contaminants.
    contam = "sp|CONTAM_P00761|TRYP_PIG"
    for sample_run in runs:
        rows.append(("CONTAMPEPA", "CONTAMPEPA", contam, 2, sample_run, 800.0))
        rows.append(("CONTAMPEPB", "CONTAMPEPB", contam, 2, sample_run, 900.0))

    # Decoy accession → filtered. Use the canonical DECOY_ prefix so it stays
    # detectable after parse_uniprot_accession strips the surrounding fields.
    decoy = "sp|DECOY_P11111|REV_HUMAN"
    for sample_run in runs:
        rows.append(("DECOYPEPAA", "DECOYPEPAA", decoy, 2, sample_run, 50.0))
        rows.append(("DECOYPEPBB", "DECOYPEPBB", decoy, 2, sample_run, 60.0))

    # Entrapment accession → filtered (same naming convention as above).
    entrap = "sp|ENTRAP_P22222|ENT_HUMAN"
    for sample_run in runs:
        rows.append(("ENTRAPPEPA", "ENTRAPPEPA", entrap, 2, sample_run, 70.0))
        rows.append(("ENTRAPPEPB", "ENTRAPPEPB", entrap, 2, sample_run, 80.0))

    # A short peptide (< 7 AA) attached to the good protein → filtered by min_aa.
    for sample_run in runs:
        rows.append(("SHORT", "SHORT", p12345, 2, sample_run, 999.0))

    # A row with unique=False on the good protein → filtered by require_unique.
    # Stored separately so we can flip the ``unique`` flag.
    non_unique_rows = [
        ("PEPTIDEAR", "PEPTIDEAR", p12345, 2, run_S, 12345.0) for run_S in runs
    ]

    data = {
        "sequence": [r[0] for r in rows] + [r[0] for r in non_unique_rows],
        "peptidoform": [r[1] for r in rows] + [r[1] for r in non_unique_rows],
        "pg_accessions": [_pg(r[2]) for r in rows]
        + [_pg(r[2]) for r in non_unique_rows],
        "anchor_protein": [r[2] for r in rows] + [r[2] for r in non_unique_rows],
        "charge": [r[3] for r in rows] + [r[3] for r in non_unique_rows],
        "run_file_name": [r[4] for r in rows] + [r[4] for r in non_unique_rows],
        "unique": [True] * len(rows) + [False] * len(non_unique_rows),
        "is_decoy": [False] * (len(rows) + len(non_unique_rows)),
        "intensities": [_intensities(r[5]) for r in rows]
        + [_intensities(r[5]) for r in non_unique_rows],
    }
    table = pa.table(data, schema=_LFQ_SCHEMA)
    pq.write_table(table, path)
    return runs


def _make_lfq_sdrf(path: str, runs: List[str]) -> None:
    """Create a matching SDRF where each run maps to its own sample_accession."""
    header = (
        "source name\tcomment[data file]\tcomment[label]\t"
        "characteristics[organism part]\tfactor value[disease]\t"
        "characteristics[biological replicate]\tcomment[fraction identifier]\t"
        "comment[technical replicate]\n"
    )
    lines = [header]
    for i, run in enumerate(runs, start=1):
        cond = "normal" if i % 2 == 1 else "disease"
        sample = f"Sample_{i}"
        lines.append(f"{sample}\t{run}\tLabel free sample\tbrain\t{cond}\t{i}\t1\t1\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def lfq_dataset(tmp_path):
    parquet = tmp_path / "lfq_input.parquet"
    sdrf = tmp_path / "lfq_input.sdrf.tsv"
    runs = _make_lfq_parquet(str(parquet))
    _make_lfq_sdrf(str(sdrf), runs)
    return str(parquet), str(sdrf)


def _make_config(parquet: str, sdrf: str) -> PipelineConfig:
    """Build a minimal PipelineConfig that disables every normalization stage so
    the two load paths are directly comparable.
    """
    return PipelineConfig(
        input=InputConfig(parquet=parquet, sdrf=sdrf),
        filtering=FilterConfig(
            remove_contaminants=True, min_aa=7, min_unique_peptides=2
        ),
        normalization=NormalizationConfig(run_method="none", sample_method="none"),
    )


def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to plain strings and sort by the key tuple so
    that two DataFrames produced by different paths line up row-by-row.
    """
    df = df.copy()
    for col in [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID, "Condition"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "BioReplicate" in df.columns:
        df["BioReplicate"] = df["BioReplicate"].astype(int)
    sort_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID]
    sort_cols += [c for c in ["PeptideSequence", "PrecursorCharge"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _legacy_path_output(cfg: PipelineConfig) -> pd.DataFrame:
    """Build the same feature/SDRF state as the dispatcher and force-call the
    legacy pandas loop directly so we can diff it against the SQL-first output.
    """
    loader = LoadingStage(cfg)
    filter_builder = SQLFilterBuilder(
        remove_contaminants=cfg.filtering.remove_contaminants,
        min_peptide_length=cfg.filtering.min_aa,
        require_unique=True,
    )
    feature = Feature(cfg.input.parquet, filter_builder=filter_builder)
    feature.enrich_with_sdrf(cfg.input.sdrf)
    technical_repetitions, label, _samples, choice = analyse_sdrf(cfg.input.sdrf)
    from mokume.model.normalization import PeptideNormalizationMethod

    sample_norm = cfg.normalization.sample_method.lower()
    sample_norm_method = PeptideNormalizationMethod.from_str(sample_norm)
    return loader._load_for_mokume_legacy_pandas(
        feature,
        label,
        choice,
        technical_repetitions,
        sample_norm,
        sample_norm_method,
        med_map={},
    )


def test_dispatcher_routes_lfq_no_run_norm_to_sqlfirst(lfq_dataset, caplog):
    """LFQ + run_method=none must take the SQL-first path."""
    import logging

    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    with caplog.at_level(logging.INFO, logger="mokume.pipeline"):
        LoadingStage(cfg).load_for_mokume()
    paths = [r.getMessage() for r in caplog.records if "Loading with" in r.getMessage()]
    assert any("SQL-first path" in m for m in paths), (
        f"expected SQL-first dispatch, got log records: {paths}"
    )


def test_sqlfirst_filters_unique_min_aa_decoy_contam_entrap(lfq_dataset):
    """SQL-first output must drop short peptides, decoys, contaminants,
    entrapments, and the single-peptide protein P67890."""
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    df = LoadingStage(cfg).load_for_mokume()

    proteins = set(df[PROTEIN_NAME].astype(str))
    # P12345 keeps after parse_uniprot_accession; bare P99999 stays as-is.
    assert proteins == {"P12345", "P99999"}, proteins

    peptides = set(df[PEPTIDE_CANONICAL].astype(str))
    # SHORT (< 7 AA), ONLYONEPEP (only 1 unique per protein), CONTAM*, DECOY*,
    # ENTRAP* must all be gone.
    forbidden = {
        "SHORT",
        "ONLYONEPEP",
        "CONTAMPEPA",
        "CONTAMPEPB",
        "DECOYPEPAA",
        "DECOYPEPBB",
        "ENTRAPPEPA",
        "ENTRAPPEPB",
    }
    assert forbidden.isdisjoint(peptides), forbidden & peptides
    assert {"PEPTIDEAR", "PEPTIDEBR", "PEPTIDECR", "KEPTPEPAR", "KEPTPEPBR"}.issubset(
        peptides
    )


def test_null_unique_flag_means_a_one_protein_group(lfq_dataset):
    """qpx < 1.1.3 leaves ``unique`` null; it then means a one-protein group."""
    parquet, sdrf = lfq_dataset
    table = pq.read_table(parquet)
    groups = table["pg_accessions"].to_pylist()
    for row, sequence in enumerate(table["sequence"].to_pylist()):
        if sequence in {"KEPTPEPAR", "KEPTPEPBR"}:
            groups[row] = groups[row] + [{**groups[row][0], "accession": "P00002"}]
    table = table.set_column(
        table.schema.get_field_index("pg_accessions"),
        "pg_accessions",
        pa.array(groups, type=_PG_PROTEIN_TYPE),
    )
    column = table.schema.get_field_index("unique")
    nulls = pa.nulls(table.num_rows, pa.bool_())
    pq.write_table(table.set_column(column, "unique", nulls), parquet)
    cfg = _make_config(parquet, sdrf)

    # Both P99999 peptides are now shared with P00002, so they are dropped.
    for df in (LoadingStage(cfg).load_for_mokume(), _legacy_path_output(cfg)):
        assert set(df[PROTEIN_NAME].astype(str)) == {"P12345"}


def test_contaminant_twin_group_is_named_without_contaminant_members(lfq_dataset):
    """A kept ``CONTAM_<ACC>;<ACC>`` group is named ``<ACC>``, as in Rust."""
    parquet, sdrf = lfq_dataset
    table = pq.read_table(parquet)
    groups = table["pg_accessions"].to_pylist()
    for row, sequence in enumerate(table["sequence"].to_pylist()):
        if sequence in {"KEPTPEPAR", "KEPTPEPBR"}:
            twin = {**groups[row][0], "accession": "CONTAM_P99999"}
            groups[row] = [twin] + groups[row]
    column = table.schema.get_field_index("pg_accessions")
    groups = pa.array(groups, type=_PG_PROTEIN_TYPE)
    pq.write_table(table.set_column(column, "pg_accessions", groups), parquet)
    cfg = _make_config(parquet, sdrf)

    for df in (LoadingStage(cfg).load_for_mokume(), _legacy_path_output(cfg)):
        assert set(df[PROTEIN_NAME].astype(str)) == {"P12345", "P99999"}


def test_load_fails_when_no_feature_passes_the_filters(lfq_dataset):
    """Input that the filters empty fails early with a message naming them."""
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    cfg.filtering.min_aa = 50

    with pytest.raises(ValueError, match="No features pass the input filters"):
        LoadingStage(cfg).load_for_mokume()


def test_sqlfirst_respects_disabled_contaminant_filter(request):
    """The SQL-first path must not apply a second hard-coded name filter."""
    parquet, sdrf = request.getfixturevalue("lfq_dataset")
    cfg = _make_config(parquet, sdrf)
    cfg.filtering.remove_contaminants = False

    proteins = set(LoadingStage(cfg).load_for_mokume()[PROTEIN_NAME].astype(str))

    assert "CONTAM_P00761" in proteins


def test_sqlfirst_peptidoform_sums_charges_per_sample(lfq_dataset):
    """After peptidoform-max keeps one row per (peptidoform, charge, sample),
    sum_peptidoform_intensities sums intensities at the peptide-canonical level.
    For PEPTIDEAR in our fixture each sample contributes charge-2 (101) and
    charge-3 (302), so the aggregated NormIntensity must equal 403."""
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    cfg.quantification.method = "sum"
    df = LoadingStage(cfg).load_for_mokume()

    p_ar = df[
        (df[PROTEIN_NAME].astype(str) == "P12345")
        & (df[PEPTIDE_CANONICAL].astype(str) == "PEPTIDEAR")
    ]
    assert len(p_ar) == 4, p_ar  # one row per sample
    assert np.allclose(p_ar[NORM_INTENSITY].astype(float).values, 403.0, atol=1e-3), (
        p_ar[NORM_INTENSITY].tolist()
    )


def test_sqlfirst_pibaq_keeps_shared_rows_and_skips_min_unique(lfq_dataset):
    parquet, sdrf = lfq_dataset
    cfg = PipelineConfig(
        input=InputConfig(parquet=parquet, sdrf=sdrf),
        filtering=FilterConfig(
            remove_contaminants=True, min_aa=7, min_unique_peptides=2
        ),
        normalization=NormalizationConfig(run_method="none", sample_method="none"),
        quantification=QuantificationConfig(method="pibaq"),
    )
    df = LoadingStage(cfg).load_for_mokume()

    proteins = set(df[PROTEIN_NAME].astype(str))
    assert "P67890" in proteins

    peptides = set(df[PEPTIDE_CANONICAL].astype(str))
    assert "ONLYONEPEP" in peptides

    p_ar = df[
        (df[PROTEIN_NAME].astype(str) == "P12345")
        & (df[PEPTIDE_CANONICAL].astype(str) == "PEPTIDEAR")
    ]
    assert len(p_ar) == 4, p_ar
    assert np.allclose(
        p_ar[NORM_INTENSITY].astype(float).values, 12345.0 + 302.0, atol=1e-3
    ), p_ar[NORM_INTENSITY].tolist()


def test_sqlfirst_per_sample_normalization_matches_legacy(lfq_dataset):
    """With ``sample_method="globalMedian"`` the SQL-first path normalizes each
    sample's slice of the post-aggregation DataFrame, while the legacy path
    normalizes each sample inside the per-sample loop. Both call the same
    ``PeptideNormalizationMethod.GlobalMedian`` callable with the same
    ``med_map``, so the resulting NormIntensity must agree row-for-row.

    This covers the ``_load_for_mokume_sqlfirst`` branch that is reached for
    the default ``NormalizationConfig`` (sample_method=``globalMedian``), and
    that was uncovered by the simpler ``none``-normalization tests.
    """
    from mokume.model.normalization import PeptideNormalizationMethod

    parquet, sdrf = lfq_dataset
    cfg = PipelineConfig(
        input=InputConfig(parquet=parquet, sdrf=sdrf),
        filtering=FilterConfig(
            remove_contaminants=True, min_aa=7, min_unique_peptides=2
        ),
        normalization=NormalizationConfig(
            run_method="none", sample_method="globalMedian"
        ),
    )

    loader = LoadingStage(cfg)
    filter_builder = SQLFilterBuilder(
        remove_contaminants=True, min_peptide_length=7, require_unique=True
    )
    sample_norm = "globalmedian"
    sample_norm_method = PeptideNormalizationMethod.from_str(sample_norm)

    feature_sql = Feature(parquet, filter_builder=filter_builder)
    feature_sql.enrich_with_sdrf(sdrf)
    med_map = feature_sql.get_median_map()
    assert med_map, "expected non-empty med_map for globalMedian normalization"

    feature_legacy = Feature(parquet, filter_builder=filter_builder)
    feature_legacy.enrich_with_sdrf(sdrf)
    technical_repetitions, label, _samples, choice = analyse_sdrf(sdrf)

    sql_df = loader._load_for_mokume_sqlfirst(
        feature_sql, sample_norm, sample_norm_method, med_map
    )
    legacy_df = loader._load_for_mokume_legacy_pandas(
        feature_legacy,
        label,
        choice,
        technical_repetitions,
        sample_norm,
        sample_norm_method,
        med_map,
    )

    s = _normalize_for_compare(sql_df)
    lg = _normalize_for_compare(legacy_df)
    assert len(s) == len(lg), (len(s), len(lg))

    key_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID, "BioReplicate", "Condition"]
    for col in key_cols:
        assert (s[col].values == lg[col].values).all(), col

    # The normalization is a single multiplicative factor per sample, so the
    # ratio (sql_value / legacy_value) must be exactly 1 for every row (up to
    # float32 rounding).
    diff = (s[NORM_INTENSITY].astype(float) - lg[NORM_INTENSITY].astype(float)).abs()
    rel = diff / lg[NORM_INTENSITY].astype(float).abs().clip(lower=1e-9)
    assert (rel.max() <= 1e-5) or (diff.max() <= 1e-3), (
        f"max abs={diff.max():.3g}, max rel={rel.max():.3g}"
    )


def test_sqlfirst_equivalent_to_legacy_pandas(lfq_dataset):
    """SQL-first output must match the legacy per-sample pandas loop row-for-row
    on the synthetic LFQ fixture (no normalization stages active)."""
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)

    sql_df = LoadingStage(cfg).load_for_mokume()
    legacy_df = _legacy_path_output(cfg)
    # Legacy path skips the final dataset-level normalization helper when called
    # directly; we want a peptide-level apples-to-apples comparison, so do the
    # same to the SQL output (sample_method=none means the helper is a no-op).

    assert set(sql_df.columns) == set(legacy_df.columns)
    s = _normalize_for_compare(sql_df)
    lg = _normalize_for_compare(legacy_df)
    assert len(s) == len(lg), (len(s), len(lg))

    key_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID, "BioReplicate", "Condition"]
    for col in key_cols:
        assert (s[col].values == lg[col].values).all(), col

    diff = (s[NORM_INTENSITY].astype(float) - lg[NORM_INTENSITY].astype(float)).abs()
    rel = diff / lg[NORM_INTENSITY].astype(float).abs().clip(lower=1e-9)
    # float32 intensities in parquet → relative drift up to ~1e-6 between
    # float64 (DuckDB SUM) and float32 (pandas SUM).
    assert (rel.max() <= 1e-5) or (diff.max() <= 1e-3), (
        f"max abs={diff.max():.3g}, max rel={rel.max():.3g}"
    )


def _make_lfq_multi_techrep_parquet(path: str) -> List[str]:
    """Build a tiny LFQ parquet with 2 samples x 3 runs each so the SQL-first
    run-normalization CTEs (TechReplicate derivation, per-(sample, tech_rep)
    metric, scale) actually fire. Peptide intensities differ per run by a
    deterministic multiplier so the per-run median is meaningful and the
    legacy ``FeatureNormalizationMethod.Median`` factor is non-trivial.

    Returns the ordered list of run_file_name values.
    """
    runs = ["run_SA_1", "run_SA_2", "run_SA_3", "run_SB_1", "run_SB_2", "run_SB_3"]
    # Run multipliers within each sample so that per-(sample, tech_rep) median
    # is genuinely different and run normalization has a measurable effect.
    run_mult = {
        "run_SA_1": 2.0,
        "run_SA_2": 1.0,
        "run_SA_3": 4.0,
        "run_SB_1": 1.5,
        "run_SB_2": 0.5,
        "run_SB_3": 3.0,
    }
    # Four peptides on the same protein → satisfies min_unique_peptides >= 2.
    peptides = [
        ("PEPTIDEAR", 100.0),
        ("PEPTIDEBR", 200.0),
        ("PEPTIDECR", 300.0),
        ("PEPTIDEDR", 400.0),
    ]
    p12345 = "sp|P12345|GOOD_HUMAN"

    rows: List[Tuple] = []
    for run in runs:
        for pep, base in peptides:
            rows.append((pep, pep, p12345, 2, run, base * run_mult[run]))

    data = {
        "sequence": [r[0] for r in rows],
        "peptidoform": [r[1] for r in rows],
        "pg_accessions": [_pg(r[2]) for r in rows],
        "anchor_protein": [r[2] for r in rows],
        "charge": [r[3] for r in rows],
        "run_file_name": [r[4] for r in rows],
        "unique": [True] * len(rows),
        "is_decoy": [False] * len(rows),
        "intensities": [_intensities(r[5]) for r in rows],
    }
    table = pa.table(data, schema=_LFQ_SCHEMA)
    pq.write_table(table, path)
    return runs


def _make_lfq_multi_techrep_sdrf(path: str, runs: List[str]) -> None:
    """Two-sample SDRF where Sample_A owns run_SA_* and Sample_B owns run_SB_*."""
    header = (
        "source name\tcomment[data file]\tcomment[label]\t"
        "characteristics[organism part]\tfactor value[disease]\t"
        "characteristics[biological replicate]\tcomment[fraction identifier]\t"
        "comment[technical replicate]\n"
    )
    lines = [header]
    for run in runs:
        sample = "Sample_A" if "_SA_" in run else "Sample_B"
        cond = "normal" if sample == "Sample_A" else "disease"
        tech_rep = run.rsplit("_", 1)[-1]
        lines.append(
            f"{sample}\t{run}\tLabel free sample\tbrain\t{cond}\t1\t1\t{tech_rep}\n"
        )
    Path(path).write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def lfq_multi_techrep_dataset(tmp_path):
    parquet = tmp_path / "lfq_multi_techrep.parquet"
    sdrf = tmp_path / "lfq_multi_techrep.sdrf.tsv"
    runs = _make_lfq_multi_techrep_parquet(str(parquet))
    _make_lfq_multi_techrep_sdrf(str(sdrf), runs)
    return str(parquet), str(sdrf)


def test_sqlfirst_run_normalization_matches_legacy(lfq_multi_techrep_dataset):
    """SQL-first run normalization (median) must match the legacy pandas
    ``FeatureNormalizationMethod.Median`` row-for-row when each sample has
    multiple technical replicates.

    Together with the global guard ``technical_repetitions > 1`` this
    exercises the full ``_build_run_norm_ctes`` SQL block: TechReplicate
    derivation from the trailing ``_<int>`` segment, per-(sample, tech_rep)
    median, per-sample mean of those medians, and the per-row
    ``intensity * sample_avg / run_metric`` scaling.
    """
    from mokume.model.normalization import PeptideNormalizationMethod

    parquet, sdrf = lfq_multi_techrep_dataset
    cfg = PipelineConfig(
        input=InputConfig(parquet=parquet, sdrf=sdrf),
        filtering=FilterConfig(
            remove_contaminants=True, min_aa=7, min_unique_peptides=2
        ),
        normalization=NormalizationConfig(run_method="median", sample_method="none"),
    )

    sample_norm = "none"
    sample_norm_method = PeptideNormalizationMethod.from_str(sample_norm)
    filter_builder = SQLFilterBuilder(
        remove_contaminants=True, min_peptide_length=7, require_unique=True
    )

    feature_sql = Feature(parquet, filter_builder=filter_builder)
    feature_sql.enrich_with_sdrf(sdrf)
    technical_repetitions, label, _samples, choice = analyse_sdrf(sdrf)
    assert technical_repetitions > 1, (
        "fixture must produce technical_repetitions > 1 to exercise the "
        f"run normalization branch (got {technical_repetitions})"
    )

    feature_legacy = Feature(parquet, filter_builder=filter_builder)
    feature_legacy.enrich_with_sdrf(sdrf)

    loader = LoadingStage(cfg)
    sql_df = loader._load_for_mokume_sqlfirst(
        feature_sql,
        sample_norm,
        sample_norm_method,
        med_map={},
        run_norm="median",
        technical_repetitions=technical_repetitions,
    )
    legacy_df = loader._load_for_mokume_legacy_pandas(
        feature_legacy,
        label,
        choice,
        technical_repetitions,
        sample_norm,
        sample_norm_method,
        med_map={},
    )

    s = _normalize_for_compare(sql_df)
    lg = _normalize_for_compare(legacy_df)
    assert len(s) == len(lg), (len(s), len(lg))

    key_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID, "BioReplicate", "Condition"]
    for col in key_cols:
        assert (s[col].values == lg[col].values).all(), col

    diff = (s[NORM_INTENSITY].astype(float) - lg[NORM_INTENSITY].astype(float)).abs()
    rel = diff / lg[NORM_INTENSITY].astype(float).abs().clip(lower=1e-9)
    assert (rel.max() <= 1e-5) or (diff.max() <= 1e-3), (
        f"max abs={diff.max():.3g}, max rel={rel.max():.3g}"
    )


def test_dispatcher_routes_lfq_with_run_norm_to_sqlfirst(
    lfq_multi_techrep_dataset, caplog
):
    """LFQ + run_method in {median, mean, max, global, iqr} should now stay on
    the SQL-first path — the dispatcher must no longer punt to legacy when
    technical_repetitions > 1."""
    import logging

    parquet, sdrf = lfq_multi_techrep_dataset
    cfg = PipelineConfig(
        input=InputConfig(parquet=parquet, sdrf=sdrf),
        filtering=FilterConfig(
            remove_contaminants=True, min_aa=7, min_unique_peptides=2
        ),
        normalization=NormalizationConfig(run_method="median", sample_method="none"),
    )
    with caplog.at_level(logging.INFO, logger="mokume.pipeline"):
        LoadingStage(cfg).load_for_mokume()
    paths = [r.getMessage() for r in caplog.records if "Loading with" in r.getMessage()]
    assert any("SQL-first path" in m for m in paths), (
        f"expected SQL-first dispatch with run_method=median, got log records: {paths}"
    )


@pytest.mark.parametrize("normalization", ["none", "quantile", "loess"])
def test_maxlfq_normalization_preserves_charge_species(lfq_dataset, normalization):
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    cfg.quantification.method = "maxlfq"
    cfg.normalization.sample_method = normalization
    frame = LoadingStage(cfg).load_for_mokume()
    selected = frame[
        (frame[PROTEIN_NAME].astype(str) == "P12345")
        & (frame[PEPTIDE_CANONICAL] == "PEPTIDEAR")
    ]
    assert len(selected) == 8
    assert set(selected["PrecursorCharge"]) == {2, 3}
    if normalization == "none":
        np.testing.assert_allclose(
            sorted(selected[NORM_INTENSITY]), [101] * 4 + [302] * 4
        )


def test_maxlfq_pipeline_uses_own_solver(monkeypatch, lfq_dataset):
    parquet, sdrf = lfq_dataset
    cfg = _make_config(parquet, sdrf)
    cfg.quantification.method = "maxlfq"
    cfg.runtime.duckdb_threads = 1
    features_module = importlib.import_module("mokume.pipeline.features_to_proteins")

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("MaxLFQ pipeline must not enter DirectLFQ")

    monkeypatch.setattr(
        features_module, "estimate_protein_intensities_streamed", unexpected_call
    )
    result = QuantificationPipeline(cfg).run()
    row = result.set_index(PROTEIN_NAME).loc["P12345"]
    # All four samples have the same four species: 101 + 302 + 203 + 404.
    np.testing.assert_allclose(row.to_numpy(dtype=float), [1010] * 4)
