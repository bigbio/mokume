"""Matrix DEqMS accepts per-cell quantification precision weights."""

import csv
from pathlib import Path

import pytest

import mokume

FEATURE_PARQUET = Path(__file__).parent / "example" / "feature_wide_dense.parquet"


def test_matrix_deqms_precision_weights():
    """Precision 2 floors to unit weights; other methods and shapes are rejected."""
    matrix = [
        [2.0, 4.0, 8.0, 32.0, 64.0, 128.0],
        [8.0, 16.0, 32.0, 8.0, 16.0, 32.0],
        [32.0, 30.0, 34.0, 4.0, 8.0, 16.0],
        [5.0, 6.0, 7.0, 5.5, 6.5, None],
    ]
    proteins = ["P1", "P2", "P3", "P4"]
    counts = [1, 2, 3, 4]
    plain = mokume.differential_expression(
        proteins, matrix, 3, 3, "deqms", peptide_counts=counts
    )
    unit = mokume.differential_expression(
        proteins,
        matrix,
        3,
        3,
        "deqms",
        peptide_counts=counts,
        precision=[[2.0] * 6 for _ in matrix],
    )
    assert unit == plain

    uneven = [[1.0, 8.0, 2.0, 4.0, 0.0, 3.0] for _ in matrix]
    weighted = mokume.differential_expression(
        proteins, matrix, 3, 3, "deqms", peptide_counts=counts, precision=uneven
    )
    assert [row["pvalue"] for row in weighted] != [row["pvalue"] for row in plain]

    with pytest.raises(RuntimeError, match="precision weights require DEqMS"):
        mokume.differential_expression(
            proteins, matrix, 3, 3, "limma", precision=uneven
        )
    with pytest.raises(RuntimeError, match="shape of the matrix"):
        mokume.differential_expression(
            proteins, matrix, 3, 3, "deqms", peptide_counts=counts, precision=uneven[:3]
        )


@pytest.mark.parametrize("method", ["maxlfq", "directlfq"])
def test_export_precision_follows_the_protein_matrix(tmp_path, method):
    """Every quantified cell has a positive precision in the matrix layout."""
    output, precision = tmp_path / "proteins.csv", tmp_path / "precision.csv"
    mokume.features2proteins(
        parquet=str(FEATURE_PARQUET),
        output=str(output),
        quant_method=method,
        export_precision=str(precision),
    )

    def read(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    values, weights = read(output), read(precision)
    assert weights[0] == values[0]
    assert [row[0] for row in weights] == [row[0] for row in values]
    for value_row, weight_row in zip(values[1:], weights[1:]):
        for value, weight in zip(value_row[1:], weight_row[1:]):
            quantified = value not in ("", "0")
            assert (weight != "") == quantified
            assert not quantified or float(weight) > 0
