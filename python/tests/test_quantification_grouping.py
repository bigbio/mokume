"""Regression tests for sample- and run-level protein grouping."""

import pandas as pd
import pytest

from mokume.quantification.all_peptides import AllPeptidesQuantification
from mokume.quantification.topn import TopNQuantification


@pytest.mark.parametrize(
    ("quantifier", "sample_intensity", "run_intensities"),
    [
        (AllPeptidesQuantification(), 330.0, [30.0, 300.0]),
        (TopNQuantification(n=2), 150.0, [15.0, 150.0]),
    ],
    ids=["sum", "topn"],
)
def test_run_column_keeps_technical_runs_separate(
    quantifier, sample_intensity, run_intensities
):
    """TopN and sum must not merge runs when a run column is requested."""
    peptides = pd.DataFrame(
        {
            "ProteinName": ["P1"] * 4,
            "SampleID": ["S1"] * 4,
            "Run": ["R1", "R1", "R2", "R2"],
            "PeptideCanonical": ["PEP1", "PEP2", "PEP1", "PEP2"],
            "NormIntensity": [10.0, 20.0, 100.0, 200.0],
        }
    )

    by_sample = quantifier.quantify(peptides)
    by_run = quantifier.quantify(peptides, run_column="Run").sort_values("Run")

    assert by_sample.columns.tolist() == ["ProteinName", "SampleID", "Intensity"]
    assert by_sample["Intensity"].tolist() == [sample_intensity]
    assert by_run.columns.tolist() == [
        "ProteinName",
        "SampleID",
        "Run",
        "Intensity",
    ]
    assert by_run["Run"].tolist() == ["R1", "R2"]
    assert by_run["Intensity"].tolist() == run_intensities
