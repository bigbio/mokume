"""Regression coverage for column-based QPX 1.3 feature compatibility."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mokume.core.constants import PEPTIDE_SEQUENCE
from mokume.io import feature as qpx_feature
from mokume.preprocessing.filters.peptide import ModificationFilter
from mokume.quantification.ratio import _load_qpx_psm_rows as load_psm_rows
from mokume.tissuemap.loader import _execute_aggregation as execute_aggregation

from .test_qpx_format_compat import (
    _make_scored_qpx_parquet as make_scored_qpx_parquet,
    _make_new_qpx_parquet as make_new_qpx_parquet,
    _ADDITIONAL_SCORES_TYPE as ADDITIONAL_SCORES_TYPE,
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


@pytest.mark.parametrize("top,fast", [([0.01, 0.2], True), ([0.01, None], False)])
def test_top_qvalue_requires_all_row_groups_nonnull(tmp_path, top, fast):
    """Top qvalue requires all row groups nonnull."""
    path = tmp_path / "qvalue.parquet"
    make_scored_qpx_parquet(str(path))
    table = pq.read_table(path).append_column(
        "global_qvalue", pa.array(top, type=pa.float64())
    )
    # The fast path must not need the score list at all. A partial column must.
    if fast:
        table = table.drop(["additional_scores"])
    else:
        values = [
            [
                {
                    "score_name": "global_qvalue",
                    "score_value": value,
                    "higher_better": False,
                }
            ]
            for value in [0.01, 0.2]
        ]
        table = table.set_column(
            table.schema.get_field_index("additional_scores"),
            "additional_scores",
            pa.array(values, type=ADDITIONAL_SCORES_TYPE),
        )
    pq.write_table(table, path, row_group_size=1)
    builder = qpx_feature.SQLFilterBuilder(
        named_score_name="global_qvalue", named_score_threshold=0.05
    )
    feature = qpx_feature.Feature(str(path), builder)
    try:
        assert (builder.named_score_source is not None) == fast
        assert feature.get_report_from_database(feature.samples)[
            "sequence"
        ].tolist() == ["PEPTIDEA"]
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
