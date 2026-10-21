"""Dataset benchmark summaries bundled in the form the service shows them."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

import yaml

from mokume.agentic.contract import DATA_TYPES

if TYPE_CHECKING:
    from mokume.agentic.knowledge import SourceEnvelope


DATASET_ARTIFACT = "datasets.yaml"
_RECORD_FIELDS = frozenset(
    {
        "id",
        "project_accession",
        "study",
        "data_type",
        "url",
        "preset_eligible",
        "design",
        "ground_truth",
        "benchmark",
        "held_out_evaluation",
    }
)
_DESIGN_FIELDS = frozenset({"family", "factor_column", "condition_map", "contrasts"})
_CONTRAST_FIELDS = frozenset(
    {
        "id",
        "group1",
        "group2",
        "role",
        "expected_direction",
        "expected_log2fc",
        "expected_log2fc_by_component",
    }
)
_TRUTH_FIELDS = frozenset({"entity_count", "source"})
_BENCHMARK_COUNTS = (
    "declared_candidates",
    "successful_candidates",
    "failed_candidates",
    "contrast_count",
)
_BENCHMARK_FIELDS = frozenset({"contract_id", "grid_complete", *_BENCHMARK_COUNTS})
_HELD_OUT_NUMBERS = ("held_out_rank_spearman", "held_out_deficit", "deficit_regret")
_HELD_OUT_COUNTS = ("training_datasets", "ranked_workflows", "held_out_benchmark_rank")
_HELD_OUT_FIELDS = frozenset(
    {
        "held_out_study",
        "selected_pipeline",
        "oracle_config_id",
        *_HELD_OUT_NUMBERS,
        *_HELD_OUT_COUNTS,
    }
)
_PIPELINE_TEXT = (
    "id",
    "quantification",
    "normalization",
    "imputation",
    "de_method",
    "fdr_method",
)
_PIPELINE_FIELDS = frozenset({*_PIPELINE_TEXT, "log2fc_threshold"})


class DatasetEvidence(NamedTuple):
    """One searchable dataset summary that is never a recommendation prior."""

    id: str
    source_id: str
    dataset_id: str
    project_accession: str
    study: str
    data_type: str
    url: str
    preset_eligible: bool
    design: dict[str, Any]
    ground_truth: dict[str, Any]
    benchmark: dict[str, Any]
    held_out_evaluation: dict[str, Any] | None

    def to_context_dict(self, source: SourceEnvelope) -> dict[str, Any]:
        """Serialize bounded dataset evidence for explanation-only search."""
        limitations = [
            "Dataset summaries are explanation-only and are not recommendation priors."
        ]
        if not self.preset_eligible:
            limitations.append(
                "This dataset is excluded from cross-dataset preset aggregation."
            )
        if self.held_out_evaluation is None:
            limitations.append("No held-out evaluation is available.")
        return {
            "id": self.id,
            "kind": "dataset_benchmark_summary",
            "status": "complete" if self.benchmark["grid_complete"] else "incomplete",
            "eligible_as_prior": False,
            "preset_eligible": self.preset_eligible,
            "applicability": {"data_type": self.data_type},
            "dataset": {
                "id": self.dataset_id,
                "project_accession": self.project_accession,
                "study": self.study,
                "url": self.url,
            },
            "design": copy.deepcopy(self.design),
            "ground_truth": copy.deepcopy(self.ground_truth),
            "benchmark": dict(self.benchmark),
            "held_out_evaluation": copy.deepcopy(self.held_out_evaluation),
            "limitations": limitations,
            "source": source.to_dict(),
        }

    def method_values(self) -> tuple[str, ...]:
        """Return methods selected in this dataset's held-out evaluation."""
        if self.held_out_evaluation is None:
            return ()
        selected = self.held_out_evaluation["selected_pipeline"]
        return tuple(str(value) for value in selected.values() if value is not None)


def load_dataset_evidence(source_id: str, path: Path) -> dict[str, DatasetEvidence]:
    """Validate the dataset summaries of one benchmark source."""
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{path.name} must be a list of dataset summaries")
    records: dict[str, DatasetEvidence] = {}
    for item in items:
        record = _dataset_record(source_id, item)
        if record.id in records:
            raise ValueError(f"Duplicate dataset evidence id: {record.id}")
        records[record.id] = record
    return records


def _dataset_record(source_id: str, item: Any) -> DatasetEvidence:
    dataset_id = _text(_fields(item, _RECORD_FIELDS, "dataset summary")["id"], "id")
    data_type = _text(item["data_type"], f"{dataset_id} data type")
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unsupported data type for {dataset_id}: {data_type!r}")
    held_out = item["held_out_evaluation"]
    return DatasetEvidence(
        id=f"dataset-{dataset_id}",
        source_id=source_id,
        dataset_id=dataset_id,
        project_accession=_text(item["project_accession"], "project accession"),
        study=_text(item["study"], "study"),
        data_type=data_type,
        url=_text(item["url"], "dataset URL"),
        preset_eligible=_bool(item["preset_eligible"], "preset_eligible"),
        design=_design(item["design"], dataset_id),
        ground_truth=_ground_truth(item["ground_truth"], dataset_id),
        benchmark=_benchmark(item["benchmark"], dataset_id),
        held_out_evaluation=(
            None if held_out is None else _held_out(held_out, dataset_id)
        ),
    )


def _design(value: Any, dataset_id: str) -> dict[str, Any]:
    design = _fields(value, _DESIGN_FIELDS, f"{dataset_id} design")
    _text(design["family"], "design family")
    _text(design["factor_column"], "factor column")
    if not all(
        isinstance(key, str) and isinstance(entry, str)
        for key, entry in _mapping(design["condition_map"], "condition map").items()
    ):
        raise ValueError(f"{dataset_id} condition map must contain text")
    contrasts = design["contrasts"]
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError(f"{dataset_id} contrasts must be a non-empty list")
    for contrast in contrasts:
        _fields(contrast, _CONTRAST_FIELDS, f"{dataset_id} contrast")
        for key in ("id", "group1", "group2", "role", "expected_direction"):
            _text(contrast[key], f"contrast {key}")
        if contrast["expected_log2fc"] is not None:
            _number(contrast["expected_log2fc"], "expected log2fc")
        by_component = _mapping(contrast["expected_log2fc_by_component"], "log2fc")
        for entry in by_component.values():
            _number(entry, "component log2fc")
    return design


def _ground_truth(value: Any, dataset_id: str) -> dict[str, Any]:
    truth = _fields(value, _TRUTH_FIELDS, f"{dataset_id} ground truth")
    _integer(truth["entity_count"], "ground truth count")
    source = _mapping(truth["source"], f"{dataset_id} ground truth source")
    if not all(isinstance(entry, (str, int, float)) for entry in source.values()):
        raise ValueError(f"{dataset_id} ground truth source must hold scalars")
    return truth


def _benchmark(value: Any, dataset_id: str) -> dict[str, Any]:
    benchmark = _fields(value, _BENCHMARK_FIELDS, f"{dataset_id} benchmark")
    _text(benchmark["contract_id"], "contract id")
    _bool(benchmark["grid_complete"], "grid_complete")
    for key in _BENCHMARK_COUNTS:
        _integer(benchmark[key], key)
    return benchmark


def _held_out(value: Any, dataset_id: str) -> dict[str, Any]:
    held_out = _fields(value, _HELD_OUT_FIELDS, f"{dataset_id} held-out evaluation")
    for key in ("held_out_study", "oracle_config_id"):
        _text(held_out[key], key)
    for key in _HELD_OUT_COUNTS:
        _integer(held_out[key], key)
    for key in _HELD_OUT_NUMBERS:
        _number(held_out[key], key)
    pipeline = _fields(
        held_out["selected_pipeline"], _PIPELINE_FIELDS, "selected pipeline"
    )
    for key in _PIPELINE_TEXT:
        _text(pipeline[key], f"selected {key}")
    _number(pipeline["log2fc_threshold"], "selected log2fc")
    return held_out


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _fields(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if set(_mapping(value, name)) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value
