"""Cox Eq. 3 and public MaxLFQ input-contract regressions."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from mokume.mokume_cli import cli
from mokume.quantification import get_quantification_method
from mokume.quantification.maxlfq import MaxLFQQuantification, _maxlfq_solve_protein


def test_inconsistent_triangle_uses_all_pairwise_ratios():
    # Edges A->B=1, B->C=1, A->C=3 in log2 units. The least-squares
    # solution has differences 4/3, 4/3, 8/3, independent of a reference peptide.
    matrix = np.array([[1, 2, np.nan], [np.nan, 1, 2], [1, np.nan, 8]])
    result = _maxlfq_solve_protein(matrix, min_ratio_count=1)
    np.testing.assert_allclose(
        np.log2(result / result[0]), [0, 4 / 3, 8 / 3], atol=1e-12
    )
    assert result.sum() == pytest.approx(15)


def test_ratio_threshold_is_per_pair_and_isolated_signal_is_not_filled():
    matrix = np.array([[1, 2, np.nan], [2, 4, np.nan], [np.nan, 1, 7]])
    result = _maxlfq_solve_protein(matrix)
    np.testing.assert_allclose(result, [17 / 3, 34 / 3, 0])
    # The unsupported third sample still contributes to Cox's total scaling.
    assert result.sum() == pytest.approx(np.nansum(matrix))


def test_disconnected_components_report_unidentifiable_offsets(caplog):
    matrix = np.array(
        [
            [1, 2, np.nan, np.nan],
            [2, 4, np.nan, np.nan],
            [np.nan, np.nan, 10, 40],
            [np.nan, np.nan, 20, 80],
        ]
    )
    np.testing.assert_allclose(_maxlfq_solve_protein(matrix), [3, 6, 30, 120])
    assert "between-component ratios are not identifiable" in caplog.text


@pytest.mark.parametrize(
    "matrix", [np.empty((0, 3)), [[1], [2]], [[0, np.nan], [-1, np.inf]]]
)
def test_no_supported_pair_stays_unquantified(matrix):
    result = _maxlfq_solve_protein(np.asarray(matrix))
    assert np.all(result == 0)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_ratio_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        MaxLFQQuantification(min_ratio_count=count)


def test_even_log_median_and_large_dynamic_range():
    result = _maxlfq_solve_protein(np.array([[1, 1], [1, 9]]))
    assert result[1] / result[0] == pytest.approx(3)
    result = _maxlfq_solve_protein(np.array([[1e-200, 1e200], [1e-200, 1e200]]))
    np.testing.assert_allclose(result, [2e-200, 2e200], rtol=1e-12, atol=0)


def _species_frame():
    return pd.DataFrame(
        [
            ("P", "PEPTIDEAK", "PEPTIDEAK", 2, "S1", 1.0),
            ("P", "PEPTIDEAK", "PEPTIDEAK", 3, "S1", 1.0),
            ("P", "ANOTHERAK", "ANOTHERAK", 2, "S1", 1.0),
            ("P", "PEPTIDEAK", "PEPTIDEAK", 2, "S2", 1.0),
            ("P", "PEPTIDEAK", "PEPTIDEAK", 3, "S2", 9.0),
            ("P", "ANOTHERAK", "ANOTHERAK", 2, "S2", 3.0),
        ],
        columns=[
            "ProteinName",
            "PeptideCanonical",
            "PeptideSequence",
            "PrecursorCharge",
            "SampleID",
            "NormIntensity",
        ],
    )


def test_species_identity_and_duplicate_counts_are_preserved():
    quantifier = get_quantification_method("maxlfq", threads=1)
    result = quantifier.quantify(_species_frame())
    # Species ratios 1, 9, 3 have median 3; total input intensity is 16.
    np.testing.assert_allclose(result["Intensity"], [4, 12])
    assert not quantifier.using_directlfq
    assert get_quantification_method("maxlfq", threads=3).threads == 3
    duplicates = pd.DataFrame(
        [
            ("P", "pep", "S1", 1),
            ("P", "pep", "S1", 1),
            ("P", "pep", "S2", 2),
            ("P", "pep", "S2", 2),
        ],
        columns=["ProteinName", "PeptideCanonical", "SampleID", "NormIntensity"],
    )
    assert quantifier.quantify(duplicates).empty
    allowed = get_quantification_method("maxlfq", threads=1, min_ratio_count=1)
    np.testing.assert_allclose(allowed.quantify(duplicates)["Intensity"], [2, 4])


def test_invalid_charges_cannot_create_shared_species():
    frame = _species_frame()
    invalid = frame.copy()
    invalid["PrecursorCharge"] = [0, np.inf, 2.5, 0, np.inf, 2.5]
    invalid["NormIntensity"] = 1e9
    combined = pd.concat([frame, invalid], ignore_index=True)
    result = MaxLFQQuantification(threads=1).quantify(combined)
    np.testing.assert_allclose(result["Intensity"], [4, 12])


@pytest.mark.parametrize("force_builtin", [False, True])
def test_optional_directlfq_never_changes_dispatch(monkeypatch, force_builtin):
    from mokume.quantification.directlfq import DirectLFQQuantification

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("MaxLFQ must not delegate to DirectLFQ")

    monkeypatch.setattr(DirectLFQQuantification, "quantify", unexpected_call)
    result = MaxLFQQuantification(threads=1, force_builtin=force_builtin).quantify(
        _species_frame()
    )
    np.testing.assert_allclose(result["Intensity"], [4, 12])


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_peptides_command_preserves_species_and_ratio_threshold(tmp_path, extension):
    source = tmp_path / f"peptides.{extension}"
    output = tmp_path / "proteins.tsv"
    frame = _species_frame()
    if extension == "parquet":
        frame.to_parquet(source, index=False)
    else:
        frame.to_csv(source, index=False)
    args = [
        "peptides2protein",
        "--method",
        "maxlfq",
        "-p",
        str(source),
        "-o",
        str(output),
        "--threads",
        "1",
        "--maxlfq-min-ratio-count",
    ]
    result = CliRunner().invoke(cli, [*args, "2"])
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(pd.read_csv(output, sep="\t")["Intensity"], [4, 12])
    result = CliRunner().invoke(cli, [*args, "4"])
    assert result.exit_code == 0, result.output
    assert pd.read_csv(output, sep="\t").empty


@pytest.mark.parametrize(
    "species,expected",
    [(5, 10), (6, 10**0.8 * 30**0.2), (10, 50), (20, 100)],
)
def test_stabilization_boundaries_interpolation_and_total_scaling(species, expected):
    matrix = np.zeros((species, 2))
    matrix[:, 0] = 10
    matrix[:2, 1] = 1
    standard = _maxlfq_solve_protein(matrix)
    enabled = _maxlfq_solve_protein(matrix, stabilize=True)
    assert standard[0] / standard[1] == pytest.approx(10)
    assert enabled[0] / enabled[1] == pytest.approx(expected)
    assert enabled.sum() == pytest.approx(matrix.sum())
    np.testing.assert_allclose(
        _maxlfq_solve_protein(matrix[:, ::-1], stabilize=True), enabled[::-1]
    )
    assert np.all(_maxlfq_solve_protein(matrix, min_ratio_count=3, stabilize=True) == 0)


def test_stabilization_cannot_bridge_disconnected_samples():
    matrix = np.eye(3)
    assert np.all(_maxlfq_solve_protein(matrix, min_ratio_count=1, stabilize=True) == 0)


@pytest.mark.parametrize("bad", ["false", 1, None])
def test_stabilization_rejects_non_boolean_values(bad):
    with pytest.raises(ValueError, match="boolean"):
        MaxLFQQuantification(stabilize=bad)


def test_peptides_cli_stabilizes_and_rejects_other_methods(tmp_path):
    rows = [("P", f"PEPTIDE{i}", "S1", 10) for i in range(20)]
    rows += [("P", f"PEPTIDE{i}", "S2", 1) for i in range(2)]
    frame = pd.DataFrame(
        rows, columns=["ProteinName", "PeptideCanonical", "SampleID", "NormIntensity"]
    )
    source, output = tmp_path / "peptides.csv", tmp_path / "proteins.tsv"
    frame.to_csv(source, index=False)
    args = ["peptides2protein", "-p", str(source), "-o", str(output), "--threads", "1"]
    for enabled, expected in [(False, 10), (True, 100)]:
        options = [*args, "--method", "maxlfq", *(["--stabilize"] if enabled else [])]
        result = CliRunner().invoke(cli, options)
        assert result.exit_code == 0, result.output
        values = pd.read_csv(output, sep="\t")["Intensity"].to_numpy()
        assert values[0] / values[1] == pytest.approx(expected)
    for method in ["directlfq", "sum", "pibaq", "top3"]:
        result = CliRunner().invoke(
            cli, [*args[:-2], "--method", method, "--stabilize"]
        )
        assert result.exit_code == 2, result.output
        assert "--stabilize only applies to --method maxlfq" in result.output
