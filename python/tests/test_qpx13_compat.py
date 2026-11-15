"""Regression coverage for column-based QPX 1.3 feature compatibility."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mokume.core.constants import PEPTIDE_SEQUENCE
from mokume.io import feature as qpx_feature
from mokume.io._protein_groups import protein_group_names, protein_peptide_counts
from mokume.pipeline.config import InputConfig
from mokume.pipeline.features_to_proteins import features_to_proteins
from mokume.preprocessing.filters.peptide import ModificationFilter
from mokume.quantification.ratio import _load_qpx_psm_rows as load_psm_rows
from mokume.tissuemap.loader import _execute_aggregation as execute_aggregation

from .test_qpx_format_compat import (
    _make_scored_qpx_parquet as make_scored_qpx_parquet,
    _make_new_qpx_parquet as make_new_qpx_parquet,
)


def _qpx13_scores(path, values, layout=True):
    """qpx13 scores."""
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).drop(["additional_scores"])
    scores = pa.array(
        values, type=pa.struct([("quality", pa.float32()), ("qvalue", pa.float64())])
    )
    table = table.append_column("additional_scores", scores)
    if layout:
        metadata = {
            b"qpx.struct_lists": json.dumps(
                {
                    "additional_scores": [
                        {"name": "quality", "higher_better": True},
                        {"name": "qvalue", "higher_better": False},
                    ]
                }
            ).encode()
        }
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)
    return table


@pytest.mark.parametrize("name,threshold", [("quality", 0.5), ("qvalue", 0.05)])
def test_struct_score_direction(tmp_path, name, threshold):
    """Struct score direction."""
    path = tmp_path / "struct.parquet"
    _qpx13_scores(
        path, [{"quality": 0.9, "qvalue": 0.01}, {"quality": 0.1, "qvalue": 0.2}]
    )
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name=name, named_score_threshold=threshold
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        report = feature.get_report_from_database(feature.samples)
        assert report["sequence"].tolist() == ["PEPTIDEA"]
    finally:
        feature.parquet_db.close()


def test_unused_optional_columns_are_not_carried_into_reports(tmp_path):
    """Default reads must not expand unused QPX metadata for every intensity."""
    path = tmp_path / "optional.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).append_column("global_qvalue", pa.array([0.01, 0.2]))
    table = table.append_column(
        "modifications",
        pa.array(
            [
                [{"name": "Oxidation", "accession": "UNIMOD:35"}],
                None,
            ]
        ),
    )
    pq.write_table(table, path)
    feature = qpx_feature.Feature(str(path), qpx_feature.SQLFilterBuilder())
    try:
        report = feature.get_report_from_database(feature.samples)
        assert not {"global_qvalue", "modifications"}.intersection(report.columns)
        assert len(report) == 2
    finally:
        feature.parquet_db.close()


@pytest.mark.parametrize(
    "layout,name,error",
    [
        (False, "quality", "requires qpx.struct_lists footer"),
        (True, "missing", "not in qpx.struct_lists layout"),
    ],
)
def test_struct_score_requires_layout(tmp_path, layout, name, error):
    """Struct score requires layout."""
    path = tmp_path / "struct.parquet"
    _qpx13_scores(path, [None, None], layout)
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name=name, named_score_threshold=0.5
    )
    with pytest.raises(ValueError, match=error):
        qpx_feature.Feature(str(path), builder)


@pytest.mark.parametrize("value", [None, float("inf"), float("nan")])
def test_struct_score_invalid_field(tmp_path, value):
    """Struct score invalid field."""
    path = tmp_path / "struct.parquet"
    _qpx13_scores(path, [{"quality": value, "qvalue": 0.01}, None])
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name="quality", named_score_threshold=0.5
    )
    with pytest.raises(ValueError, match="null or non-finite"):
        qpx_feature.Feature(str(path), builder)


def test_null_struct_is_missing_score(tmp_path):
    """Null struct is missing score."""
    path = tmp_path / "struct.parquet"
    _qpx13_scores(path, [{"quality": 0.9, "qvalue": 0.01}, None])
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name="quality", named_score_threshold=0.5
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        assert len(feature.get_report_from_database(feature.samples)) == 1
    finally:
        feature.parquet_db.close()


@pytest.mark.parametrize(
    "name", ["peptide_qvalue", "precursor_qvalue", "global_qvalue"]
)
@pytest.mark.parametrize("top", [[0.01, 0.2], [0.01, None]])
def test_typed_qvalue_accepts_nulls_without_additional_scores(tmp_path, name, top):
    """Typed q values take precedence even when only some rows have values."""
    path = tmp_path / "qvalue.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).drop(["additional_scores"])
    table = table.append_column(name, pa.array(top, type=pa.float64()))
    pq.write_table(table, path, row_group_size=1)
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name=name, named_score_threshold=0.05
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        assert builder.named_score_source == (f'"{name}"', False)
        assert feature.get_report_from_database(feature.samples)[
            "sequence"
        ].tolist() == ["PEPTIDEA"]
    finally:
        feature.parquet_db.close()


def test_all_null_typed_qvalue_is_missing_score(tmp_path):
    """An empty typed column must not fall back to additional scores."""
    path = tmp_path / "empty.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).append_column(
        "peptide_qvalue", pa.array([None, None], type=pa.float64())
    )
    pq.write_table(table, path)
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name="peptide_qvalue", named_score_threshold=0.05
    )
    with pytest.raises(
        ValueError, match="no non-null values for QPX score `peptide_qvalue`"
    ):
        qpx_feature.Feature(str(path), builder)


def test_null_decoy_is_target(tmp_path):
    """A null decoy flag retains a target protein group."""
    path = tmp_path / "decoy.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("is_decoy"),
        "is_decoy",
        pa.array([None, True], type=pa.bool_()),
    )
    pq.write_table(table, path)
    builder = qpx_feature.SQLFilterBuilder()
    feature = qpx_feature.Feature(str(path), builder)
    try:
        clause, params = builder.build_where_clause()
        rows = feature.parquet_db.execute(
            f"SELECT sequence FROM parquet_db WHERE {clause}", params
        ).fetchall()
        assert rows == [("PEPTIDEA",)]
    finally:
        feature.parquet_db.close()


def test_imputed_labels_are_removed_from_feature_and_ratio(tmp_path):
    """Imputed labels are removed from feature and ratio."""
    path = tmp_path / "imputed.parquet"
    make_new_qpx_parquet(str(path))
    table = pq.read_table(path).append_column(
        "imputed_labels", pa.array([["TMT126"], None, []], type=pa.list_(pa.string()))
    )
    pq.write_table(table, path)
    builder = qpx_feature.SQLFilterBuilder(
        remove_contaminants=False, require_unique=False
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        assert (
            feature.parquet_db.execute("SELECT count(*) FROM parquet_db").fetchone()[0]
            == 5
        )
    finally:
        feature.parquet_db.close()
    count = execute_aggregation(
        path, pd.DataFrame({"unused": [1]}), 1, "SELECT count(*) AS n FROM flat_input"
    )
    assert count["n"].tolist() == [5]
    frame, _ = load_psm_rows(str(path), builder)
    assert len(frame) == 5
    assert not (
        (frame["run_file_name"] == "run1")
        & (frame["sequence"] == "PEPTIDEK")
        & (frame["label"] == "TMT126")
    ).any()


def test_structured_modifications_are_excluded_on_input(tmp_path):
    """Names and accessions match exactly; null lists and the list itself stay out."""
    path = tmp_path / "modifications.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).append_column(
        "modifications",
        pa.array(
            [
                [{"name": "Carbamidomethyl", "accession": "UNIMOD:4"}],
                None,
            ],
            type=pa.list_(
                pa.struct([("name", pa.string()), ("accession", pa.string())])
            ),
        ),
    )
    pq.write_table(table, path)
    for values, expected in [
        (["carbamidomethyl"], ["PEPTIDEB"]),
        (["unimod:4"], ["PEPTIDEB"]),
        (["Carbamido"], ["PEPTIDEA", "PEPTIDEB"]),
    ]:
        builder = qpx_feature.SQLFilterBuilder(exclude_modifications=values)
        feature = qpx_feature.Feature(str(path), filter_builder=builder)
        try:
            frame = feature.get_report_from_database(feature.samples)
            assert sorted(set(frame["sequence"])) == expected
            assert "modifications" not in frame
        finally:
            feature.parquet_db.close()


def test_legacy_accession_exclusion_uses_complete_token():
    """Legacy accession exclusion uses complete token."""
    frame = pd.DataFrame({PEPTIDE_SEQUENCE: ["M[UNIMOD:3]K", "M[UNIMOD:35]K"]})
    filtered, _ = ModificationFilter(["UNIMOD:3"]).apply(frame)
    assert filtered[PEPTIDE_SEQUENCE].tolist() == ["M[UNIMOD:35]K"]


def test_group_identity_merges_orders_but_preserves_single_order():
    """Only an observed alternative order can change a group's name."""
    groups = pd.Series([["P2", "P1", "P2"], ["P1", "P2"], ["P4", "P3"]])
    assert protein_group_names(groups).tolist() == ["P1;P2", "P1;P2", "P4;P3"]
    assert protein_group_names(groups[::-1]).sort_index().tolist() == [
        "P1;P2",
        "P1;P2",
        "P4;P3",
    ]


def _shared_anchor_groups(path):
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("pg_accessions"),
        "pg_accessions",
        pa.array([["P1", "P2"], ["P1", "P3"]]),
    )
    pq.write_table(table, path)


def test_low_frequency_counts_distinct_groups_with_shared_anchor(tmp_path):
    """An annotation anchor must not pool two groups' sample coverage."""
    path = tmp_path / "groups.parquet"
    _shared_anchor_groups(path)
    feature = qpx_feature.Feature(str(path))
    try:
        assert set(feature.get_low_frequency_peptides(percentage=0.75)) == {
            ("P1;P2", "PEPTIDEA"),
            ("P1;P3", "PEPTIDEB"),
        }
    finally:
        feature.parquet_db.close()


def test_deqms_counts_distinct_groups_with_shared_anchor(tmp_path):
    """DEqMS counts use matrix group identities rather than anchors."""
    path = tmp_path / "groups.parquet"
    _shared_anchor_groups(path)
    assert protein_peptide_counts(path).to_dict() == {"P1;P2": 1, "P1;P3": 1}


def test_tissuemap_keeps_groups_with_shared_anchor(tmp_path):
    """TissueMap aggregates by the complete group identity."""
    path = tmp_path / "groups.parquet"
    _shared_anchor_groups(path)
    result = execute_aggregation(
        path,
        pd.DataFrame({"unused": [1]}),
        1,
        "SELECT protein, sum(intensity) AS total FROM flat_input GROUP BY protein",
    ).set_index("protein")
    assert result["total"].to_dict() == {"P1;P2": 10.0, "P1;P3": 20.0}


def _run_feature_fixture(tmp_path, tmt):
    """Write matched feature and run views for two channel/fraction cases."""
    path = tmp_path / "feature.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path)
    runs = ["run_a", "run_a"] if tmt else ["run_a", "run_b"]
    labels = ["TMT126", "TMT127"] if tmt else ["LFQ", "LFQ"]
    table = table.set_column(
        table.schema.get_field_index("run_file_name"), "run_file_name", pa.array(runs)
    )
    table = table.set_column(
        table.schema.get_field_index("intensities"),
        "intensities",
        pa.array(
            [
                [{"label": label, "intensity": value}]
                for label, value in zip(labels, [10.0, 20.0])
            ]
        ),
    )
    pq.write_table(table, path)
    samples = [
        {
            "sample_accession": f"S{i}" if tmt else "S",
            "label": label,
            "biological_replicate": 2,
            "technical_replicate": 3,
        }
        for i, label in enumerate(labels)
    ]
    records = (
        [{"run_file_name": "run_a", "fraction": "1", "samples": samples}]
        if tmt
        else [
            {"run_file_name": run, "fraction": str(i + 1), "samples": [sample]}
            for i, (run, sample) in enumerate(zip(runs, samples))
        ]
    )
    run_path = tmp_path / "run.parquet"
    pq.write_table(pa.Table.from_pylist(records), run_path)
    return path, run_path, runs


@pytest.mark.parametrize("tmt", [False, True])
def test_run_mapping_preserves_samples_and_replicates(tmp_path, tmt):
    """Run metadata resolves multiplexed channels and fractionated LFQ."""
    path, run_path, _ = _run_feature_fixture(tmp_path, tmt)
    feature = qpx_feature.Feature(str(path))
    try:
        run_table = feature.enrich_with_run(str(run_path))
        result = feature.get_report_from_database(feature.samples).sort_values(
            "intensity"
        )
        assert result["sample_accession"].tolist() == (
            ["S0", "S1"] if tmt else ["S", "S"]
        )
        assert result["biological_replicate"].tolist() == [2, 2]
        assert run_table["comment[technical replicate]"].tolist() == [3, 3]
        assert "technical_replicate" not in result
        assert result["fraction"].tolist() == (["1", "1"] if tmt else ["1", "2"])
        assert result["condition"].tolist() == result["sample_accession"].tolist()
    finally:
        feature.parquet_db.close()


def test_run_conflicts_with_sdrf():
    """The two alternative sample maps cannot be combined."""
    with pytest.raises(ValueError, match="--run conflicts with --sdrf"):
        InputConfig(parquet="features.parquet", sdrf="samples.tsv", run="run.parquet")


@pytest.mark.parametrize(
    "name", ["peptide_qvalue", "precursor_qvalue", "global_qvalue"]
)
def test_legacy_qvalue_score_entry_without_typed_column(tmp_path, name):
    """QPX files without a typed q value retain named-list score lookup."""
    path = tmp_path / "legacy.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path)
    scores = [
        [{"score_name": name, "score_value": value, "higher_better": False}]
        for value in [0.01, 0.2]
    ]
    table = table.set_column(
        table.schema.get_field_index("additional_scores"),
        "additional_scores",
        pa.array(scores),
    )
    pq.write_table(table, path)
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name=name, named_score_threshold=0.05
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        assert feature.get_report_from_database(feature.samples)[
            "sequence"
        ].tolist() == ["PEPTIDEA"]
    finally:
        feature.parquet_db.close()


@pytest.mark.parametrize("tmt", [False, True])
def test_run_mapping_reaches_protein_matrix(tmp_path, tmt):
    """Quantification uses mapped sample columns for both run-view layouts."""
    path, run_path, _ = _run_feature_fixture(tmp_path, tmt)
    result = features_to_proteins(
        str(path),
        str(tmp_path / "proteins.csv"),
        run=str(run_path),
        quant_method="sum",
        min_unique_peptides=1,
        run_normalization="none",
        sample_normalization="none",
    )
    assert set(result.columns) == (
        {"ProteinName", "S0", "S1"} if tmt else {"ProteinName", "S"}
    )
