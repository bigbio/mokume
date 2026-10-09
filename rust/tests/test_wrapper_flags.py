"""Tests for Python keyword-to-CLI flag conversion."""

from __future__ import annotations

import pytest

from mokume._command_flags import flags_for


def test_rejects_unknown_wrapper_keyword() -> None:
    """Unknown Python wrapper keywords fail instead of becoming ignored flags."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        flags_for("features2proteins", {"not_a_real_option": 1})


def test_paired_and_repeatable_options_use_their_real_shapes() -> None:
    """Paired and repeatable values expand into repeated canonical flags."""
    assert flags_for(
        "features2proteins",
        {
            "de_contrast": [("A", "B"), ("C", "D")],
            "irs_reference_sample": ["Pool A", "Pool B"],
        },
    ) == [
        "--de-contrast",
        "A",
        "B",
        "--de-contrast",
        "C",
        "D",
        "--irs-reference-sample",
        "Pool A",
        "--irs-reference-sample",
        "Pool B",
    ]


def test_boolean_uses_canonical_positive_flag() -> None:
    """A true boolean emits its positive flag while false emits nothing."""
    assert flags_for("features2proteins", {"keep_contaminants": True}) == [
        "--keep-contaminants"
    ]
    assert not flags_for("features2proteins", {"keep_contaminants": False})


def test_tissuemap_repeatable_option_uses_singular_flag() -> None:
    """Each TissueMap dataset is emitted with the singular repeatable flag."""
    assert flags_for("tissuemap", {"tmt_dataset": ["PXD1", "PXD2"]}) == [
        "--tmt-dataset",
        "PXD1",
        "--tmt-dataset",
        "PXD2",
    ]


def test_nonboolean_value_option_rejects_bool() -> None:
    """Booleans cannot silently stand in for scalar option values."""
    with pytest.raises(TypeError, match="expects a value"):
        flags_for("features2proteins", {"min_unique": False})


def test_python_name_matches_real_families_flag() -> None:
    """The Python families keyword maps to the installed CLI spelling."""
    assert flags_for("peptides2protein", {"families": "families.yaml"}) == [
        "--families",
        "families.yaml",
    ]


def test_features2proteins_protein_fdr_keyword_maps_to_flag() -> None:
    """The protein FDR keyword reaches the features2proteins CLI flag."""
    assert flags_for("features2proteins", {"filter_protein_fdr": 0.01}) == [
        "--filter-protein-fdr",
        "0.01",
    ]
