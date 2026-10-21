"""Typed, provenance-aware knowledge graph for agentic recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, NamedTuple, TYPE_CHECKING

import yaml

from mokume.agentic.contract import (
    CONFIDENCE_LEVELS,
    DATA_TYPES,
    DE_METHODS,
    ENSEMBLE_PRESETS,
    FDR_METHODS,
    IMPUTATION_METHODS,
    NORMALIZATION_METHODS,
    is_supported_quantification,
    validate_config_values,
)
from mokume.agentic.dataset_knowledge import (
    DATASET_ARTIFACT,
    DatasetEvidence,
    load_dataset_evidence,
)

if TYPE_CHECKING:
    from mokume.agentic.profiler import DataProfile


_KNOWLEDGE_ENV = "MOKUME_AGENTIC_KNOWLEDGE"
_BUNDLED_KNOWLEDGE = Path(__file__).with_name("knowledge_bundle") / "knowledge.yaml"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceEnvelope(NamedTuple):
    """Provenance envelope for one external or generated evidence source."""

    id: str
    kind: str
    title: str
    locator: str
    captured_at: str
    trust: str
    status: str
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source for a bound context block."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "locator": self.locator,
            "captured_at": self.captured_at,
            "trust": self.trust,
            "status": self.status,
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class Applicability:
    """Dataset characteristics under which an evidence record applies.

    A record with a quantification applies only to a matrix that declares it.
    """

    data_type: str
    setting: str | None = None
    upstream_engine: str | None = None
    quantification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize applicability fields."""
        return {
            "data_type": self.data_type,
            "setting": self.setting,
            "upstream_engine": self.upstream_engine,
            "quantification": self.quantification,
        }


@dataclass(frozen=True)
class ReferenceProfile:
    """Observed profile envelope for the datasets supporting one preset."""

    projects: tuple[str, ...]
    n_proteins: tuple[int, int]
    n_samples: tuple[int, int]
    missing_rate: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the benchmark coverage envelope."""
        return {
            "projects": list(self.projects),
            "n_proteins": list(self.n_proteins),
            "n_samples": list(self.n_samples),
            "missing_rate": list(self.missing_rate),
        }


class PipelineConfig(NamedTuple):
    """A full pipeline configuration retained as benchmark evidence."""

    quantification: str | None
    normalization: str
    imputation: str
    de_method: str
    fdr_method: str
    log2fc_threshold: float | str
    ensemble: str = "none"
    ensemble_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full pipeline without dropping frozen axes."""
        return {
            "quantification": self.quantification,
            "normalization": self.normalization,
            "imputation": self.imputation,
            "de_method": self.de_method,
            "fdr_method": self.fdr_method,
            "log2fc_threshold": self.log2fc_threshold,
            "ensemble": self.ensemble,
            "ensemble_k": self.ensemble_k,
        }


class EvidenceRecord(NamedTuple):
    """One benchmark-derived pipeline observation in the knowledge graph."""

    id: str
    source_id: str
    kind: str
    status: str
    confidence: str
    priority: int
    eligible_as_prior: bool
    applicability: Applicability
    pipeline: PipelineConfig
    metrics: dict[str, Any]
    reference_profile: ReferenceProfile | None = None
    limitations: tuple[str, ...] = ()

    def to_context_dict(self, source: SourceEnvelope) -> dict[str, Any]:
        """Serialize only the traceable fields permitted in model context."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "confidence": self.confidence,
            "applicability": self.applicability.to_dict(),
            "reference_profile": (
                self.reference_profile.to_dict()
                if self.reference_profile is not None
                else None
            ),
            "pipeline": self.pipeline.to_dict(),
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "source": source.to_dict(),
        }


@dataclass(frozen=True)
class GraphEdge:
    """Directed relationship between two typed knowledge objects."""

    source: str
    relationship: str
    target: str

    def to_dict(self) -> dict[str, str]:
        """Serialize a graph edge."""
        return {
            "source": self.source,
            "relationship": self.relationship,
            "target": self.target,
        }


@dataclass(frozen=True)
class KnowledgeGraph:
    """Validated source and evidence objects with explicit relationships."""

    fingerprint: str
    sources: dict[str, SourceEnvelope]
    evidence: dict[str, EvidenceRecord]
    datasets: dict[str, DatasetEvidence]
    edges: tuple[GraphEdge, ...]

    def matching(self, profile: DataProfile) -> list[EvidenceRecord]:
        """Return eligible evidence for the profile's data type and quantification.

        Records for the declared quantification replace the general records of
        their source; without a declaration only general records apply.
        """
        data_type = profile.data_type.upper()
        candidates = [
            record
            for record in self.evidence.values()
            if record.eligible_as_prior
            and record.applicability.data_type.upper() == data_type
        ]
        specific = {
            record.source_id
            for record in candidates
            if profile.quantification is not None
            and record.applicability.quantification == profile.quantification
        }
        matches = [
            record
            for record in candidates
            if (
                record.source_id not in specific
                if record.applicability.quantification is None
                else record.applicability.quantification == profile.quantification
            )
        ]
        return sorted(matches, key=lambda record: (record.priority, record.id))


def load_knowledge_graph(path: str | Path | None = None) -> KnowledgeGraph:
    """Load an explicit, environment-selected, or bundled knowledge graph."""
    if path is not None:
        selected = path
    else:
        selected = os.environ.get(_KNOWLEDGE_ENV, _BUNDLED_KNOWLEDGE)
    return _load_knowledge_graph(str(Path(selected).expanduser().resolve()))


@lru_cache(maxsize=8)
def _load_knowledge_graph(path: str) -> KnowledgeGraph:
    """Load one resolved knowledge snapshot and cache its validated graph."""
    knowledge_path = Path(path)
    content = knowledge_path.read_bytes()
    raw = yaml.safe_load(content.decode("utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) != {"sources", "evidence"}:
        raise ValueError(
            "knowledge catalog fields must be exactly ['evidence', 'sources']"
        )
    sources = _parse_sources(raw["sources"], knowledge_path.parent)
    evidence = _parse_evidence(raw["evidence"], sources)
    datasets = _parse_dataset_sources(sources, knowledge_path.parent)
    edges = tuple(
        GraphEdge(record.id, "supported_by", record.source_id)
        for record in evidence.values()
    ) + tuple(
        GraphEdge(record.id, "applies_to", record.applicability.data_type)
        for record in evidence.values()
    )
    return KnowledgeGraph(
        fingerprint=hashlib.sha256(content).hexdigest(),
        sources=sources,
        evidence=evidence,
        datasets=datasets,
        edges=edges,
    )


def _parse_dataset_sources(
    sources: dict[str, SourceEnvelope], knowledge_root: Path
) -> dict[str, DatasetEvidence]:
    """Load the dataset summaries bundled with validated sources."""
    datasets: dict[str, DatasetEvidence] = {}
    for source in sources.values():
        if DATASET_ARTIFACT not in source.artifacts:
            continue
        artifact = (knowledge_root / source.locator / DATASET_ARTIFACT).resolve()
        parsed = load_dataset_evidence(source.id, artifact)
        duplicates = set(datasets) & set(parsed)
        if duplicates:
            raise ValueError(f"Duplicate dataset evidence ids: {sorted(duplicates)}")
        datasets.update(parsed)
    return datasets


def _parse_sources(
    items: list[dict[str, Any]],
    knowledge_root: Path,
) -> dict[str, SourceEnvelope]:
    """Build unique source envelopes from YAML data."""
    if not isinstance(items, list):
        raise ValueError("knowledge sources must be a list")
    sources: dict[str, SourceEnvelope] = {}
    for item in items:
        required = {"id", "kind", "title", "locator", "captured_at", "trust", "status"}
        allowed = required | {"artifacts"}
        if (
            not isinstance(item, dict)
            or not required <= set(item)
            or set(item) - allowed
        ):
            raise ValueError(
                "knowledge source fields must contain exactly the source contract"
            )
        source = SourceEnvelope(
            id=str(item["id"]),
            kind=str(item["kind"]),
            title=str(item["title"]),
            locator=str(item["locator"]),
            captured_at=str(item["captured_at"]),
            trust=str(item["trust"]),
            status=str(item["status"]),
            artifacts=dict(item.get("artifacts", {})),
        )
        _validate_source(source, knowledge_root)
        if source.id in sources:
            raise ValueError(f"Duplicate knowledge source id: {source.id}")
        sources[source.id] = source
    return sources


def _parse_evidence(
    items: list[dict[str, Any]],
    sources: dict[str, SourceEnvelope],
) -> dict[str, EvidenceRecord]:
    """Build and validate unique evidence records from YAML data."""
    if not isinstance(items, list):
        raise ValueError("knowledge evidence must be a list")
    records: dict[str, EvidenceRecord] = {}
    for item in items:
        source_id, priority, eligible, metrics, limitations = _evidence_metadata(
            item, sources
        )
        applicability = Applicability(**item["applicability"])
        reference_profile = _parse_reference_profile(item.get("reference_profile"))
        pipeline = PipelineConfig(**item["pipeline"])
        _validate_pipeline(pipeline)
        record = EvidenceRecord(
            id=str(item["id"]),
            source_id=source_id,
            kind=str(item["kind"]),
            status=str(item["status"]),
            confidence=str(item["confidence"]),
            priority=priority,
            eligible_as_prior=eligible,
            applicability=applicability,
            pipeline=pipeline,
            metrics=dict(metrics),
            reference_profile=reference_profile,
            limitations=tuple(limitations),
        )
        if record.id in records:
            raise ValueError(f"Duplicate evidence id: {record.id}")
        if record.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"Unsupported evidence confidence {record.confidence!r}: {record.id}"
            )
        _validate_evidence(record)
        records[record.id] = record
    return records


def _evidence_metadata(
    item: dict[str, Any],
    sources: dict[str, SourceEnvelope],
) -> tuple[str, int, bool, dict[str, Any], list[Any]]:
    """Validate the scalar and collection fields of one evidence item."""
    required = {
        "id",
        "source_id",
        "kind",
        "status",
        "confidence",
        "priority",
        "eligible_as_prior",
        "applicability",
        "pipeline",
        "metrics",
        "limitations",
    }
    allowed = required | {"reference_profile"}
    if not isinstance(item, dict) or not required <= set(item) or set(item) - allowed:
        raise ValueError(
            "knowledge evidence fields must contain exactly the evidence contract"
        )
    source_id = str(item["source_id"])
    if source_id not in sources:
        raise ValueError(f"Unknown source_id {source_id!r}")
    metrics = item["metrics"]
    limitations = item["limitations"]
    if not isinstance(metrics, dict):
        raise ValueError("evidence metrics must be an object")
    if not isinstance(limitations, list):
        raise ValueError("evidence limitations must be a list")
    eligible = item["eligible_as_prior"]
    priority = item["priority"]
    if not isinstance(eligible, bool):
        raise ValueError("eligible_as_prior must be boolean")
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        raise ValueError("evidence priority must be a non-negative integer")
    return source_id, priority, eligible, metrics, limitations


def _parse_reference_profile(item: Any) -> ReferenceProfile | None:
    """Parse one optional benchmark coverage envelope."""
    if item is None:
        return None
    if not isinstance(item, dict):
        raise ValueError("reference_profile must be an object")
    required = {"projects", "n_proteins", "n_samples", "missing_rate"}
    if set(item) != required:
        raise ValueError(f"reference_profile fields must be exactly {sorted(required)}")
    projects = item["projects"]
    if (
        not isinstance(projects, list)
        or not projects
        or not all(isinstance(project, str) and project.strip() for project in projects)
    ):
        raise ValueError("reference_profile projects must be non-empty strings")
    n_proteins = _integer_range(item["n_proteins"], "n_proteins")
    n_samples = _integer_range(item["n_samples"], "n_samples")
    missing_rate = _numeric_range(item["missing_rate"], "missing_rate")
    if not 0.0 <= missing_rate[0] <= missing_rate[1] <= 1.0:
        raise ValueError("reference_profile missing_rate must be within [0, 1]")
    return ReferenceProfile(
        tuple(projects),
        n_proteins,
        n_samples,
        missing_rate,
    )


def _integer_range(item: Any, name: str) -> tuple[int, int]:
    """Validate a two-value positive integer range."""
    if (
        not isinstance(item, list)
        or len(item) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
        or item[0] <= 0
        or item[0] > item[1]
    ):
        raise ValueError(
            f"reference_profile {name} must be an increasing integer range"
        )
    return item[0], item[1]


def _numeric_range(item: Any, name: str) -> tuple[float, float]:
    """Validate a two-value finite numeric range."""
    if (
        not isinstance(item, list)
        or len(item) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in item
        )
        or any(not math.isfinite(value) for value in item)
        or item[0] > item[1]
    ):
        raise ValueError(
            f"reference_profile {name} must be a finite increasing numeric range"
        )
    return float(item[0]), float(item[1])


def _validate_source(source: SourceEnvelope, knowledge_root: Path) -> None:
    """Reject incomplete provenance or unverifiable local artifacts."""
    required = (
        source.id,
        source.kind,
        source.title,
        source.locator,
        source.captured_at,
        source.trust,
        source.status,
    )
    if any(not value.strip() for value in required):
        raise ValueError("Knowledge source provenance fields must be non-empty")
    invalid = {
        name: digest
        for name, digest in source.artifacts.items()
        if not isinstance(name, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    }
    if invalid:
        raise ValueError(f"Knowledge source has invalid SHA-256 artifacts: {invalid}")
    if source.artifacts:
        _validate_local_artifacts(source, knowledge_root)


def _validate_local_artifacts(source: SourceEnvelope, knowledge_root: Path) -> None:
    """Require every bundled source artifact and verify its exact content."""
    locator = Path(source.locator)
    if locator.is_absolute() or "://" in source.locator:
        raise ValueError("Knowledge sources with artifacts require a relative locator")
    knowledge_root = knowledge_root.resolve()
    source_root = (knowledge_root / locator).resolve()
    if not source_root.is_relative_to(knowledge_root) or not source_root.is_dir():
        raise ValueError(f"Knowledge artifact directory not found: {source.locator}")
    declared = set(source.artifacts)
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if declared != actual:
        raise ValueError(
            f"Knowledge artifact manifest mismatch for {source.id}: "
            f"missing={sorted(declared - actual)}, undeclared={sorted(actual - declared)}"
        )
    for name, expected in source.artifacts.items():
        artifact = (source_root / name).resolve()
        if not artifact.is_relative_to(source_root):
            raise ValueError(f"Knowledge artifact escapes its source directory: {name}")
        actual_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_digest != expected:
            raise ValueError(f"Knowledge artifact hash mismatch: {source.id}/{name}")


def _validate_evidence(record: EvidenceRecord) -> None:
    """Validate ontology fields not covered by the method contract."""
    if not all(
        value.strip()
        for value in (record.id, record.kind, record.status, record.source_id)
    ):
        raise ValueError("Evidence identity fields must be non-empty")
    if record.applicability.data_type.upper() not in DATA_TYPES:
        raise ValueError(
            f"Unsupported evidence data type: {record.applicability.data_type!r}"
        )
    quantification = record.applicability.quantification
    if quantification is not None and quantification != record.pipeline.quantification:
        raise ValueError(
            f"Evidence {record.id} applies to {quantification!r} matrices but its "
            f"pipeline uses {record.pipeline.quantification!r}"
        )
    if any(
        not isinstance(item, str) or not item.strip() for item in record.limitations
    ):
        raise ValueError(f"Evidence limitations must be non-empty strings: {record.id}")


def _validate_pipeline(pipeline: PipelineConfig) -> None:
    """Ensure knowledge records name methods supported by Mokume."""
    if pipeline.quantification is not None and not is_supported_quantification(
        pipeline.quantification
    ):
        raise ValueError(
            f"Unsupported quantification in knowledge: {pipeline.quantification}"
        )
    allowed = {
        "normalization": NORMALIZATION_METHODS,
        "imputation": IMPUTATION_METHODS,
        "de_method": DE_METHODS,
        "fdr_method": FDR_METHODS,
    }
    for field_name, values in allowed.items():
        value = getattr(pipeline, field_name)
        if value not in values:
            raise ValueError(f"Unsupported {field_name} in knowledge: {value}")
    if pipeline.ensemble not in ENSEMBLE_PRESETS:
        raise ValueError(f"Unsupported ensemble in knowledge: {pipeline.ensemble}")
    validate_config_values(pipeline.to_dict())
