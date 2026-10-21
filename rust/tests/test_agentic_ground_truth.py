"""Regression tests for agentic ground-truth boundaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mokume.agentic.evaluator import compute_score_ground_truth, evaluate
from mokume.agentic.service import (
    EvaluationRequest,
    InspectionRequest,
    RecommendationService,
)
from mokume.agentic.state import CandidateConfig

REPOSITORY = Path(__file__).resolve().parents[2]
SERVICE = RecommendationService()
METADATA = {
    "data_type": "LFQ",
    "quantification": "directlfq",
    "upstream_engine": "quantms",
}


def _write_lfq_inputs(tmp_path: Path) -> tuple[Path, Path]:
    proteins = [f"P{index}" for index in range(30)]
    baseline = [100.0 + index for index in range(30)]
    matrix = pd.DataFrame(
        {
            "ProteinName": proteins,
            "A1": baseline,
            "A2": [value * 1.01 for value in baseline],
            "A3": [value * 0.99 for value in baseline],
            "B1": [
                value * 4.0 if index < 5 else value
                for index, value in enumerate(baseline)
            ],
            "B2": [
                value * 4.04 if index < 5 else value
                for index, value in enumerate(baseline)
            ],
            "B3": [
                value * 3.96 if index < 5 else value
                for index, value in enumerate(baseline)
            ],
        }
    )
    protein_matrix = tmp_path / "proteins.tsv"
    matrix.to_csv(protein_matrix, sep="\t", index=False)
    sdrf = tmp_path / "samples.sdrf.tsv"
    pd.DataFrame(
        {
            "source name": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "factor value[condition]": ["A", "A", "A", "B", "B", "B"],
        }
    ).to_csv(sdrf, sep="\t", index=False)
    return protein_matrix, sdrf


def test_ground_truth_without_matrix_overlap_is_rejected(tmp_path: Path) -> None:
    """A mismatched identifier namespace must fail before candidate execution."""
    protein_matrix, sdrf = _write_lfq_inputs(tmp_path)
    truth = tmp_path / "mismatched-truth.txt"
    truth.write_text("NOT_IN_MATRIX\n", encoding="utf-8")
    recommendation = SERVICE.inspect_dataset(
        InspectionRequest(
            str(protein_matrix),
            str(sdrf),
            "linear",
            ["A", "B"],
            None,
            METADATA,
        )
    )["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]
    output_dir = tmp_path / "mismatched-truth"

    with pytest.raises(
        ValueError,
        match=r"Ground-truth proteins do not overlap the tested matrix \(0/1 identifiers\)",
    ):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(protein_matrix),
                str(sdrf),
                ["A", "B"],
                recommendation,
                str(output_dir),
                {
                    **METADATA,
                    "ground_truth": str(truth),
                    "expected_direction": "UP",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )

    assert not output_dir.exists()


def test_score_a_handles_truth_absent_from_candidate_results() -> None:
    """Candidate filtering may remove truth that exists in the input matrix."""
    de_table = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "pvalue": [0.01, 0.8],
            "significance": ["UP", "NOT_DE"],
            "log2FC": [2.0, 0.0],
        }
    )
    protein_matrix = pd.DataFrame(
        {
            "protein": ["P1", "P2", "P3"],
            "A1": [100.0, 120.0, 80.0],
            "A2": [101.0, 121.0, 81.0],
            "B1": [400.0, 120.0, 82.0],
            "B2": [404.0, 121.0, 83.0],
        }
    )
    result = evaluate(
        CandidateConfig(name="filtered-truth"),
        de_table,
        protein_matrix,
        {"A1": "A", "A2": "A", "B1": "B", "B2": "B"},
        ({"P3"}, "UP"),
    )

    assert result.truth_metrics.recall_emp_fdr_curve is None
    assert compute_score_ground_truth(result) is None
