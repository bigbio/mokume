"""Public contract tests for the plugin-owned Mokume agentic workflow."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import mokume.agentic.service as service_module
from mokume.agentic.evaluator import evaluate, method_sensitivity
from mokume.agentic.knowledge import load_knowledge_graph
from mokume.agentic.service import (
    EvaluationRequest,
    InspectionRequest,
    RecommendationService,
)
from mokume.agentic.state import CandidateConfig

REPOSITORY = Path(__file__).resolve().parents[2]
PLUGIN = REPOSITORY / "plugins" / "mokume"
KNOWLEDGE_BUNDLE = (
    REPOSITORY / "rust" / "python" / "mokume" / "agentic" / "knowledge_bundle"
)
BUNDLED_KNOWLEDGE = KNOWLEDGE_BUNDLE / "knowledge.yaml"
MCP_TOOLS = {"inspect_dataset", "evaluate_recommendation", "search_knowledge"}
CODEX_MANIFEST = PLUGIN / ".codex-plugin" / "plugin.json"
CLAUDE_MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
SERVICE = RecommendationService()


@pytest.fixture(name="lfq_inputs")
def _lfq_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write a small 3-vs-3 LFQ matrix with five known changes."""
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
                value * 4.04 if index < 5 else value * 1.01
                for index, value in enumerate(baseline)
            ],
            "B3": [
                value * 3.96 if index < 5 else value * 0.99
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
    truth = tmp_path / "truth.txt"
    truth.write_text("\n".join(proteins[:5]) + "\n", encoding="utf-8")
    peptide_counts = tmp_path / "peptide_counts.tsv"
    pd.DataFrame(
        {
            "protein": proteins,
            "peptide_count": [index % 9 + 1 for index in range(30)],
        }
    ).to_csv(peptide_counts, sep="\t", index=False)
    return {
        "matrix": protein_matrix,
        "sdrf": sdrf,
        "truth": truth,
        "peptide_counts": peptide_counts,
    }


def _inspect(inputs: dict[str, Path]) -> dict:
    return SERVICE.inspect_dataset(
        InspectionRequest(
            str(inputs["matrix"]),
            str(inputs["sdrf"]),
            "linear",
            ["A", "B"],
            None,
            {
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
            },
        )
    )


def test_service_ranks_only_with_ground_truth(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Score A selects a winner; an unlabelled dataset remains unranked."""
    inspected = _inspect(lfq_inputs)
    assert len(inspected["context"]["knowledge_fingerprint"]) == 64
    diagnostics = next(
        block["content"]
        for block in inspected["context"]["blocks"]
        if block["id"] == "diagnostics"
    )
    assert "OUTSIDE_BENCHMARK_PROFILE" in {
        diagnostic["code"] for diagnostic in diagnostics
    }
    recommendation = inspected["policy_recommendation"]
    assert recommendation["confidence"] == "low"
    recommendation["configs"] = recommendation["configs"][:1]

    ranked = SERVICE.evaluate_recommendation(
        EvaluationRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            ["A", "B"],
            recommendation,
            str(tmp_path / "ranked"),
            {
                "ground_truth": str(lfq_inputs["truth"]),
                "expected_direction": "UP",
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
                "input_scale": "linear",
                "threads": 24,
            },
        )
    )
    assert ranked["status"] == "ranked"
    assert ranked["ranking_objective"] == "benchmark_mean_rank"
    assert ranked["ranking"][0]["score_a"] is not None
    assert ranked["ranking"][0]["benchmark_mean_rank"] == 1.0
    assert ranked["ranking"][0]["nmcc"] is not None
    assert ranked["ranking"][0]["gmean"] is not None
    assert ranked["best_config"] == recommendation["configs"][0]["name"]
    assert (tmp_path / "ranked" / "evaluation.json").is_file()

    exploratory = SERVICE.evaluate_recommendation(
        EvaluationRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            ["A", "B"],
            recommendation,
            str(tmp_path / "exploratory"),
            {
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
                "input_scale": "linear",
                "threads": 24,
            },
        )
    )
    assert exploratory["status"] == "exploratory_unranked"
    assert exploratory["ranking_objective"] is None
    assert exploratory["ranking"] == []
    assert exploratory["results"][0]["score_a"] is None
    assert exploratory["best_config"] is None
    assert exploratory["method_sensitivity"]["comparison_available"] is False
    assert exploratory["method_sensitivity_artifact"] == "method_sensitivity.tsv"
    assert "stability" not in exploratory
    assert (tmp_path / "exploratory" / "method_sensitivity.tsv").is_file()


def test_peptide_count_sidecar_reaches_deqms(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The public sidecar contract enables count-aware DEqMS execution."""
    metadata = {
        "data_type": "LFQ",
        "quantification": "directlfq",
        "upstream_engine": "quantms",
    }
    inspected = SERVICE.inspect_dataset(
        InspectionRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            "linear",
            ["A", "B"],
            str(lfq_inputs["peptide_counts"]),
            metadata,
        )
    )
    assert inspected["profile"]["has_peptide_counts"] is True
    diagnostics = next(
        block["content"]
        for block in inspected["context"]["blocks"]
        if block["id"] == "diagnostics"
    )
    assert "DEQMS_COUNTS_REQUIRED" not in {
        diagnostic["code"] for diagnostic in diagnostics
    }
    recommendation = inspected["policy_recommendation"]
    recommendation["configs"] = [
        config for config in recommendation["configs"] if config["de_method"] == "deqms"
    ][:1]
    assert len(recommendation["configs"]) == 1
    output_dir = tmp_path / "count-aware-deqms"

    SERVICE.evaluate_recommendation(
        EvaluationRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            ["A", "B"],
            recommendation,
            str(output_dir),
            {
                **metadata,
                "input_scale": "linear",
                "peptide_counts": str(lfq_inputs["peptide_counts"]),
                "threads": 24,
            },
        )
    )

    de_table = pd.read_csv(next(output_dir.glob("*.de.tsv")), sep="\t")
    expected = pd.read_csv(lfq_inputs["peptide_counts"], sep="\t").set_index("protein")[
        "peptide_count"
    ]
    observed = de_table.set_index(de_table.columns[0])["peptide_count"].astype(int)
    assert observed.to_dict() == expected.reindex(observed.index).to_dict()


@pytest.mark.parametrize(
    ("de_method", "ensemble"),
    [("deqms", "none"), ("ensemble", "limma,deqms,proda")],
)
def test_deqms_candidates_require_peptide_count_sidecar(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
    de_method: str,
    ensemble: str,
) -> None:
    """DEqMS cannot run directly or in an ensemble without peptide counts."""
    inspected = _inspect(lfq_inputs)
    diagnostics = next(
        block["content"]
        for block in inspected["context"]["blocks"]
        if block["id"] == "diagnostics"
    )
    assert "DEQMS_COUNTS_REQUIRED" in {diagnostic["code"] for diagnostic in diagnostics}
    recommendation = inspected["policy_recommendation"]
    assert all(
        config["de_method"] != "deqms" and "deqms" not in config["ensemble"].split(",")
        for config in recommendation["configs"]
    )
    config = dict(recommendation["configs"][0])
    config.update(
        name=f"missing-counts-{de_method}",
        de_method=de_method,
        ensemble=ensemble,
        ensemble_k=2 if de_method == "ensemble" else None,
    )
    recommendation["configs"] = [config]
    output_dir = tmp_path / f"missing-counts-{de_method}"

    with pytest.raises(
        ValueError,
        match="options.peptide_counts is required for DEqMS",
    ):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output_dir),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )

    assert not output_dir.exists()


def test_log2_inspection_computes_cv_on_linear_intensities(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Declared log2 matrices report the same CV as their linear source."""
    metadata = {
        "data_type": "LFQ",
        "quantification": "directlfq",
        "upstream_engine": "quantms",
    }
    linear = SERVICE.inspect_dataset(
        InspectionRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            "linear",
            ["A", "B"],
            None,
            metadata,
        )
    )["profile"]
    log2_matrix = tmp_path / "proteins_log2.tsv"
    frame = pd.read_csv(lfq_inputs["matrix"], sep="\t")
    frame.iloc[:, 1:] = np.log2(frame.iloc[:, 1:].astype(float))
    frame.to_csv(log2_matrix, sep="\t", index=False)

    log2_profile = SERVICE.inspect_dataset(
        InspectionRequest(
            str(log2_matrix),
            str(lfq_inputs["sdrf"]),
            "log2",
            ["A", "B"],
            None,
            metadata,
        )
    )["profile"]

    assert log2_profile["is_log_transformed"] is True
    assert log2_profile["median_cv"] == pytest.approx(linear["median_cv"])
    assert log2_profile["per_condition_median_cv"] == pytest.approx(
        linear["per_condition_median_cv"]
    )


def test_linear_zero_values_share_missing_semantics_with_evaluation(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Linear zeros are missing throughout; log2 zeros remain observed."""
    frame = pd.read_csv(lfq_inputs["matrix"], sep="\t")
    frame.loc[0, "A1"] = 0.0
    frame.loc[1, "B1"] = 0.0
    frame.to_csv(lfq_inputs["matrix"], sep="\t", index=False)

    inspected = _inspect(lfq_inputs)
    expected_missing = 2 / (len(frame) * (len(frame.columns) - 1))
    assert inspected["profile"]["missing_rate"] == round(expected_missing, 4)
    assert inspected["profile"]["intensity_range"][0] > 0.0

    recommendation = inspected["policy_recommendation"]
    recommendation["configs"] = [
        config
        for config in recommendation["configs"]
        if config["de_method"] == "limma" and config["imputation"] == "none"
    ][:1]
    assert len(recommendation["configs"]) == 1
    evaluated = SERVICE.evaluate_recommendation(
        EvaluationRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            ["A", "B"],
            recommendation,
            str(tmp_path / "linear-zero-missing"),
            {
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
                "input_scale": "linear",
                "threads": 24,
            },
        )
    )
    assert evaluated["results"][0]["missing_rate"] == pytest.approx(expected_missing)

    log2_frame = frame.copy()
    values = log2_frame.iloc[:, 1:].mask(log2_frame.iloc[:, 1:] == 0.0, 1.0)
    log2_frame.iloc[:, 1:] = np.log2(values)
    log2_matrix = tmp_path / "zero-valid-log2.tsv"
    log2_frame.to_csv(log2_matrix, sep="\t", index=False)
    log2_profile = SERVICE.inspect_dataset(
        InspectionRequest(
            str(log2_matrix),
            str(lfq_inputs["sdrf"]),
            "log2",
            ["A", "B"],
            metadata={
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
            },
        )
    )["profile"]
    assert log2_profile["missing_rate"] == 0.0
    assert log2_profile["intensity_range"][0] == 0.0


def test_inspection_and_evaluation_scope_to_requested_contrast(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated singleton conditions cannot block or influence a contrast."""
    frame = pd.read_csv(lfq_inputs["matrix"], sep="\t")
    frame["C1"] = np.linspace(50.0, 80.0, len(frame))
    frame.to_csv(lfq_inputs["matrix"], sep="\t", index=False)
    sdrf = pd.read_csv(lfq_inputs["sdrf"], sep="\t")
    sdrf.loc[len(sdrf)] = ["C1", "C"]
    sdrf.to_csv(lfq_inputs["sdrf"], sep="\t", index=False)

    inspected = _inspect(lfq_inputs)
    profile = inspected["profile"]
    assert profile["n_samples"] == 6
    assert profile["n_conditions"] == 2
    assert profile["samples_per_condition"] == {"A": 3, "B": 3}
    assert set(profile["pairwise_median_abs_log2fc"]) == {"A_vs_B"}
    diagnostics = next(
        block["content"]
        for block in inspected["context"]["blocks"]
        if block["id"] == "diagnostics"
    )
    assert "INSUFFICIENT_REPLICATES" not in {
        diagnostic["code"] for diagnostic in diagnostics
    }

    recommendation = inspected["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]
    original = service_module.run_experiment
    observed_columns: list[str] = []

    def record_columns(*args, **kwargs):
        observed_columns.extend(str(column) for column in args[1].columns)
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_experiment", record_columns)
    SERVICE.evaluate_recommendation(
        EvaluationRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            ["A", "B"],
            recommendation,
            str(tmp_path / "contrast-scoped"),
            {
                "data_type": "LFQ",
                "quantification": "directlfq",
                "upstream_engine": "quantms",
                "input_scale": "linear",
                "threads": 24,
            },
        )
    )
    assert observed_columns == ["ProteinName", "A1", "A2", "A3", "B1", "B2", "B3"]


def test_method_sensitivity_reports_signed_call_agreement() -> None:
    """Exploratory comparison reports sharing without creating a ranking."""
    first = pd.DataFrame(
        {
            "protein": ["P1", "P2", "P3"],
            "significance": ["UP", "DOWN", "NOT_DE"],
        }
    )
    second = pd.DataFrame(
        {
            "protein": ["P1", "P2", "P3"],
            "significance": ["UP", "UP", "DOWN"],
        }
    )

    table, summary = method_sensitivity({"first": first, "second": second})

    assert summary == {
        "comparison_available": True,
        "candidate_count": 2,
        "signed_call_union": 4,
        "shared_signed_calls": 1,
        "method_sensitive_signed_calls": 3,
        "interpretation": (
            "Call sharing describes sensitivity to the tested methods and is not "
            "evidence of biological truth."
        ),
    }
    calls = table.set_index(["protein", "direction"])
    assert calls.loc[("P1", "UP"), "classification"] == "shared"
    assert calls.loc[("P2", "DOWN"), "classification"] == "method_sensitive"
    assert calls.loc[("P2", "UP"), "classification"] == "method_sensitive"


def test_bundled_knowledge_rejects_modified_source_artifacts(tmp_path: Path) -> None:
    """Every bundled benchmark artifact is bound to its declared hash."""
    copied = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_BUNDLE, copied)
    summaries = copied / "sources" / "spike-in-mokumebench" / "datasets.yaml"
    summaries.write_text(
        summaries.read_text(encoding="utf-8") + "\n# modified\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Knowledge artifact hash mismatch"):
        load_knowledge_graph(copied / "knowledge.yaml")


def test_bundled_knowledge_rejects_nonfinite_profile_ranges(tmp_path: Path) -> None:
    """Applicability ranges cannot contain YAML NaN or infinities."""
    copied = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_BUNDLE, copied)
    knowledge = copied / "knowledge.yaml"
    contents = knowledge.read_text(encoding="utf-8").replace(
        "missing_rate: [0.03068, 0.246267]",
        "missing_rate: [.nan, 0.246267]",
        1,
    )
    knowledge.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="finite.*numeric range"):
        load_knowledge_graph(knowledge)


def test_agentic_catalog_rejects_version_branches(tmp_path: Path) -> None:
    """The runtime catalog is updated in place rather than schema-versioned."""
    copied = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_BUNDLE, copied)
    knowledge = copied / "knowledge.yaml"
    knowledge.write_text(
        "version: 1\n" + knowledge.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog fields must be exactly"):
        load_knowledge_graph(knowledge)


def test_evaluation_refuses_to_overwrite_existing_output(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A repeated round must use a new output directory."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output_dir already exists"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )


def test_candidate_failure_leaves_no_partial_output(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed round can retry the same target without partial artifacts."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:2]
    assert len(recommendation["configs"]) == 2
    output = tmp_path / "atomic-evaluation"
    original = service_module.run_experiment
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected candidate failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "run_experiment", fail_second)
    request = EvaluationRequest(
        str(lfq_inputs["matrix"]),
        str(lfq_inputs["sdrf"]),
        ["A", "B"],
        recommendation,
        str(output),
        {
            "data_type": "LFQ",
            "quantification": "directlfq",
            "upstream_engine": "quantms",
            "input_scale": "linear",
            "threads": 24,
        },
    )

    with pytest.raises(RuntimeError, match="injected candidate failure"):
        SERVICE.evaluate_recommendation(request)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))

    monkeypatch.setattr(service_module, "run_experiment", original)
    recommendation["configs"] = recommendation["configs"][:1]
    SERVICE.evaluate_recommendation(request)
    assert (output / "evaluation.json").is_file()


def test_generic_sample_names_require_declared_data_type(
    lfq_inputs: dict[str, Path],
) -> None:
    """Generic sample IDs abstain instead of being guessed as LFQ."""
    inspected = SERVICE.inspect_dataset(
        InspectionRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            "linear",
            ["A", "B"],
        )
    )

    assert inspected["profile"]["data_type"] == "unknown"
    assert inspected["profile"]["data_type_source"] == "inferred"
    assert inspected["policy_recommendation"]["configs"] == []
    diagnostics = next(
        block["content"]
        for block in inspected["context"]["blocks"]
        if block["id"] == "diagnostics"
    )
    data_type = next(
        item for item in diagnostics if item["code"] == "PROFILE_DATA_TYPE_INFERRED"
    )
    assert data_type["severity"] == "error"


@pytest.mark.parametrize("alias", ["DIANN", "DIA NN", "DIA-NN"])
def test_upstream_engine_alias_matches_catalog_evidence(
    lfq_inputs: dict[str, Path],
    alias: str,
) -> None:
    """Common DIA-NN spellings bind to the catalog's canonical engine."""
    inspected = SERVICE.inspect_dataset(
        InspectionRequest(
            str(lfq_inputs["matrix"]),
            str(lfq_inputs["sdrf"]),
            "linear",
            ["A", "B"],
            metadata={"data_type": "DIA", "upstream_engine": alias},
        )
    )

    assert inspected["profile"]["upstream_engine"] == "DIA-NN"
    assert (
        "opdea-diann-dia"
        in inspected["context"]["generation_scope"]["allowed_evidence_refs"]
    )


def test_inspection_rejects_matrix_samples_absent_from_sdrf(
    lfq_inputs: dict[str, Path],
) -> None:
    """Every matrix sample must bind to authoritative SDRF metadata."""
    matrix = pd.read_csv(lfq_inputs["matrix"], sep="\t").rename(
        columns={"B3": "UNMAPPED"}
    )
    matrix.to_csv(lfq_inputs["matrix"], sep="\t", index=False)

    with pytest.raises(ValueError, match="missing from the SDRF mapping: UNMAPPED"):
        _inspect(lfq_inputs)


def test_generated_recommendation_requires_complete_unique_configs(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Host-generated candidates need rationale and unique executable settings."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    first = recommendation["configs"][0]
    first["reasoning"] = ""
    output = tmp_path / "blank-reasoning"

    with pytest.raises(ValueError, match="reasoning must be a non-empty string"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )
    assert not output.exists()

    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    duplicate = dict(recommendation["configs"][0])
    duplicate["name"] = "same-settings-under-another-name"
    recommendation["configs"] = [recommendation["configs"][0], duplicate]
    output = tmp_path / "duplicate-settings"

    with pytest.raises(ValueError, match="unique executable settings"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )
    assert not output.exists()


def test_abstention_contract_is_unambiguous(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An abstention cannot carry candidates, evidence, or elevated confidence."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation.update(
        {
            "configs": [],
            "evidence_refs": [],
            "confidence": "high",
            "abstain_reason": "Host declined to propose a candidate.",
        }
    )

    with pytest.raises(ValueError, match="Abstention requires empty configs"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(tmp_path / "invalid-abstention"),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )


def test_score_a_counts_truth_only_within_the_tested_universe() -> None:
    """Undetected truth identifiers are not false negatives for a tested matrix."""
    de_table = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "pvalue": [0.001, 0.8],
            "significance": ["UP", "NOT_DE"],
            "log2FC": [2.0, 0.0],
        }
    )
    protein_matrix = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "A1": [100.0, 120.0],
            "A2": [101.0, 121.0],
            "B1": [400.0, 120.0],
            "B2": [404.0, 121.0],
        }
    )

    result = evaluate(
        CandidateConfig(name="tested-universe"),
        de_table,
        protein_matrix,
        {"A1": "A", "A2": "A", "B1": "B", "B2": "B"},
        ({"P1", "NOT_IN_MATRIX"}, "UP"),
    )

    assert result.truth_metrics.tp == 1
    assert result.truth_metrics.fn == 0


def test_ground_truth_requires_expected_direction(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A labelled benchmark must declare the direction of its truth set."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]

    with pytest.raises(ValueError, match="expected_direction is required"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(tmp_path / "missing-direction"),
                {
                    "ground_truth": str(lfq_inputs["truth"]),
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )


def test_expected_direction_requires_ground_truth(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An unlabelled evaluation cannot declare a truth direction."""
    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]
    output_dir = tmp_path / "direction-without-truth"

    with pytest.raises(ValueError, match="must be null when ground_truth is null"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output_dir),
                {
                    "ground_truth": None,
                    "expected_direction": "UP",
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "input_scale": "linear",
                    "threads": 24,
                },
            )
        )

    assert not output_dir.exists()


def test_input_scale_must_be_explicit(
    lfq_inputs: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Inspection and evaluation reject the removed auto-scale contract."""
    with pytest.raises(ValueError, match="input_scale must be linear or log2"):
        SERVICE.inspect_dataset(
            InspectionRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                "auto",
                ["A", "B"],
            )
        )

    recommendation = _inspect(lfq_inputs)["policy_recommendation"]
    recommendation["configs"] = recommendation["configs"][:1]
    output_dir = tmp_path / "missing-input-scale"
    with pytest.raises(ValueError, match="options.input_scale is required"):
        SERVICE.evaluate_recommendation(
            EvaluationRequest(
                str(lfq_inputs["matrix"]),
                str(lfq_inputs["sdrf"]),
                ["A", "B"],
                recommendation,
                str(output_dir),
                {
                    "data_type": "LFQ",
                    "quantification": "directlfq",
                    "upstream_engine": "quantms",
                    "threads": 24,
                },
            )
        )
    assert not output_dir.exists()


def test_inspection_rejects_contrast_without_two_conditions(
    lfq_inputs: dict[str, Path],
) -> None:
    """A one-condition SDRF cannot satisfy a two-condition inspection."""
    pd.DataFrame(
        {
            "source name": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "factor value[condition]": ["A"] * 6,
        }
    ).to_csv(lfq_inputs["sdrf"], sep="\t", index=False)

    with pytest.raises(
        ValueError,
        match="Contrast requires at least two matrix samples per condition: A=6, B=0",
    ):
        _inspect(lfq_inputs)


def test_plugin_manifests_share_one_skill_and_wheel_runtime() -> None:
    """Host adapters share one skill while evidence ships in the wheel."""
    codex = json.loads(CODEX_MANIFEST.read_text(encoding="utf-8"))
    claude = json.loads(CLAUDE_MANIFEST.read_text(encoding="utf-8"))

    assert codex["name"] == claude["name"] == "mokume"
    assert codex["version"] == claude["version"]
    assert codex["skills"] == "./skills/"
    assert (PLUGIN / "skills" / "analyze-proteomics" / "SKILL.md").is_file()
    assert BUNDLED_KNOWLEDGE.is_file()
    assert not (PLUGIN / "knowledge").exists()

    codex_marketplace = json.loads(
        (REPOSITORY / ".agents" / "plugins" / "marketplace.json").read_text(
            encoding="utf-8"
        )
    )
    claude_marketplace = json.loads(
        (REPOSITORY / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    codex_plugins = {entry["name"]: entry for entry in codex_marketplace["plugins"]}
    claude_plugins = {entry["name"]: entry for entry in claude_marketplace["plugins"]}
    assert codex_plugins["mokume"]["source"] == {
        "source": "local",
        "path": "./plugins/mokume",
    }
    assert claude_plugins["mokume"]["source"] == "./plugins/mokume"


def test_plugin_registers_the_local_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both host adapters start the same three public local tools."""
    pytest.importorskip("mcp")
    create_server = getattr(
        importlib.import_module("mokume.agentic.mcp_server"),
        "create_server",
    )

    codex_manifest = json.loads(CODEX_MANIFEST.read_text(encoding="utf-8"))
    server_config = codex_manifest["mcpServers"]["mokume"]
    assert server_config["command"] == "mokume"
    assert server_config["args"] == ["mcp", "serve"]
    assert server_config["cwd"] == "."

    claude = json.loads(CLAUDE_MANIFEST.read_text(encoding="utf-8"))
    claude_server = claude["mcpServers"]["mokume"]
    assert claude_server == {
        "command": "mokume",
        "args": ["mcp", "serve"],
    }

    monkeypatch.chdir(tmp_path)
    server = create_server()
    tools = asyncio.run(server.list_tools())
    tools_by_name = {tool.name: tool for tool in tools}
    assert set(tools_by_name) == MCP_TOOLS
    inspection_schema = tools_by_name["inspect_dataset"].inputSchema
    assert set(inspection_schema["properties"]) == {
        "protein_matrix",
        "sdrf",
        "contrast",
        "options",
    }
    assert set(inspection_schema["required"]) == {
        "protein_matrix",
        "sdrf",
        "contrast",
        "options",
    }
    evaluation_schema = tools_by_name["evaluate_recommendation"].inputSchema
    assert set(evaluation_schema["properties"]) == {
        "protein_matrix",
        "sdrf",
        "contrast",
        "recommendation",
        "options",
    }
    assert set(evaluation_schema["required"]) == {
        "protein_matrix",
        "sdrf",
        "contrast",
        "recommendation",
        "options",
    }
