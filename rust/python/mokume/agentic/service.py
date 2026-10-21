"""Deterministic service boundary used by the Mokume Plugin MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from mokume.agentic.context import (
    BoundContext,
    GeneratedRecommendationBlock,
    bind_context,
    validate_generated_recommendation,
)
from mokume.agentic.contract import GENERATED_CONFIG_FIELDS, requires_peptide_counts
from mokume.agentic.evaluator import (
    compute_score_ground_truth,
    evaluate,
    method_sensitivity,
)
from mokume.agentic.knowledge import KnowledgeGraph, load_knowledge_graph
from mokume.agentic.profiler import (
    AcquisitionMetadata,
    DataProfile,
    detect_sdrf_batch_fields,
    profile_data,
)
from mokume.agentic.ranking import build_ranking_payload
from mokume.agentic.rules import rule_propose
from mokume.agentic.runner import (
    ExperimentContext,
    PreprocessCache,
    run_experiment,
)
from mokume.agentic.state import CandidateConfig, EvaluationResult
from mokume.normalization.irs import detect_condition_from_sdrf

from ._service_io import (
    as_linear as _as_linear,
    atomic_output_dir as _atomic_output_dir,
    audit_fields as _audit_fields,
    canonicalize_missing as _canonicalize_missing,
    load_ground_truth as _load_ground_truth,
    load_peptide_counts as _load_peptide_counts,
    output_path as _output_path,
    parse_contrast as _parse_contrast,
    read_matrix as _read_matrix,
    require_file as _require_file,
    scope_contrast as _scope_contrast,
    slug as _slug,
    validate_contrast_samples as _validate_contrast_samples,
    validate_input_scale as _validate_input_scale,
    validate_runtime_options as _validate_runtime_options,
    validated_object as _validated_object,
    write_evaluation as _write_evaluation,
)

_METADATA_FIELDS = frozenset(
    {"data_type", "quantification", "upstream_engine", "factor_column"}
)
_OPTION_FIELDS = _METADATA_FIELDS | {
    "ground_truth",
    "expected_direction",
    "fdr_threshold",
    "input_scale",
    "peptide_counts",
    "threads",
}


@dataclass(frozen=True)
class DatasetMetadata:
    """Declared acquisition facts and the SDRF factor selection."""

    acquisition: AcquisitionMetadata
    factor_column: str | None = None


@dataclass(frozen=True)
class EvaluationOptions:
    """Validated optional settings for one recommendation round."""

    metadata: DatasetMetadata
    input_scale: str
    peptide_counts: str | None = None
    ground_truth: str | None = None
    expected_direction: str | None = None
    fdr_threshold: float = 0.05
    threads: int = 24


@dataclass(frozen=True)
class InspectionRequest:
    """Inputs for one dataset-inspection tool call."""

    protein_matrix: str
    sdrf: str
    input_scale: str
    contrast: list[str]
    peptide_counts: str | None = None
    metadata: dict[str, str | None] | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    """Required and optional inputs for one evaluation tool call."""

    protein_matrix: str
    sdrf: str
    contrast: list[str]
    recommendation: dict[str, Any]
    output_dir: str
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvaluationRun:
    """Inputs shared by every candidate in one evaluation round."""

    experiment: ExperimentContext
    ground_truth: set[str] | None
    expected_direction: str | None
    output_dir: Path


class PreparedEvaluation(NamedTuple):
    """Validated request state ready for execution or abstention."""

    runtime: EvaluationOptions
    contrast: tuple[str, str]
    destination: Path
    frame: pd.DataFrame
    conditions: dict[str, str]
    peptide_counts: pd.Series | None
    context: BoundContext
    generated: GeneratedRecommendationBlock


class ProfileInputs(NamedTuple):
    """Inputs shared by inspection and evaluation profiling."""

    frame: pd.DataFrame
    sdrf: str
    metadata: DatasetMetadata
    input_scale: str
    peptide_counts: pd.Series | None


def _validate_ground_truth_overlap(
    frame: pd.DataFrame,
    ground_truth: set[str] | None,
) -> None:
    """Reject a ground-truth namespace absent from the tested matrix."""
    if ground_truth is None:
        return
    tested_proteins = set(frame.iloc[:, 0].astype(str))
    if ground_truth.isdisjoint(tested_proteins):
        raise ValueError(
            "Ground-truth proteins do not overlap the tested matrix "
            f"(0/{len(ground_truth)} identifiers)"
        )


class RecommendationService:
    """Bind one immutable knowledge graph to the two public MCP operations."""

    def __init__(self, knowledge: str | Path | None = None) -> None:
        if knowledge is not None:
            knowledge_path = str(Path(knowledge).expanduser().resolve())
            _require_file(knowledge_path, "knowledge")
            knowledge = knowledge_path
        self._graph = load_knowledge_graph(knowledge)

    def inspect_dataset(
        self,
        request: InspectionRequest,
    ) -> dict[str, Any]:
        """Profile a matrix and bind only compatible, traceable evidence."""
        _require_file(request.protein_matrix, "protein_matrix")
        _require_file(request.sdrf, "sdrf")
        _validate_input_scale(request.input_scale, "input_scale")
        contrast = _parse_contrast(request.contrast)
        frame = _canonicalize_missing(
            _read_matrix(request.protein_matrix), request.input_scale
        )
        counts = _load_peptide_counts(request.peptide_counts, frame)
        profile, _, context, _ = self._profile(
            ProfileInputs(
                frame,
                request.sdrf,
                _parse_dataset_metadata(request.metadata),
                request.input_scale,
                counts,
            ),
            contrast,
        )
        return {
            "profile": profile.to_dict(),
            "context": context.to_dict(),
            "policy_recommendation": _policy_recommendation(profile, context),
            "ranking_contract": "unranked_without_ground_truth",
        }

    def evaluate_recommendation(
        self,
        request: EvaluationRequest,
    ) -> dict[str, Any]:
        """Validate and evaluate one host-generated recommendation block."""
        prepared = self._prepare_evaluation(request)
        if prepared.generated.abstain_reason is not None:
            with _atomic_output_dir(prepared.destination) as output_dir:
                payload = _write_evaluation(
                    output_dir,
                    _abstention_payload(self._graph, prepared),
                )
            return payload

        linear_frame, resolved_scale = _as_linear(
            prepared.frame,
            prepared.runtime.input_scale,
        )
        experiment = ExperimentContext(
            sample_to_condition=prepared.conditions,
            contrast=prepared.contrast,
            peptide_counts=prepared.peptide_counts,
            threads=prepared.runtime.threads,
        )
        ground_truth = _load_ground_truth(prepared.runtime.ground_truth)
        _validate_ground_truth_overlap(linear_frame, ground_truth)
        candidates = [
            _candidate(item, prepared.generated, prepared.runtime.fdr_threshold)
            for item in prepared.generated.configs
        ]
        with _atomic_output_dir(prepared.destination) as output_dir:
            run = EvaluationRun(
                experiment=experiment,
                ground_truth=ground_truth,
                expected_direction=prepared.runtime.expected_direction,
                output_dir=output_dir,
            )
            payload = _run_candidates(candidates, linear_frame, run)
            payload.update(
                _audit_fields(
                    self._graph,
                    prepared.generated,
                    prepared.context,
                    input_scale=resolved_scale,
                )
            )
            safe_payload = _write_evaluation(output_dir, payload)
        return safe_payload

    def _prepare_evaluation(
        self,
        request: EvaluationRequest,
    ) -> PreparedEvaluation:
        """Validate and bind one evaluation request without writing output."""
        runtime = _parse_evaluation_options(request.options)
        contrast = _parse_contrast(request.contrast)
        _require_file(request.protein_matrix, "protein_matrix")
        _require_file(request.sdrf, "sdrf")
        if runtime.ground_truth is not None:
            _require_file(runtime.ground_truth, "ground_truth")
        destination = _output_path(request.output_dir)
        frame = _canonicalize_missing(
            _read_matrix(request.protein_matrix), runtime.input_scale
        )
        peptide_counts = _load_peptide_counts(runtime.peptide_counts, frame)
        _, conditions, context, frame = self._profile(
            ProfileInputs(
                frame,
                request.sdrf,
                runtime.metadata,
                runtime.input_scale,
                peptide_counts,
            ),
            contrast,
        )
        generated = validate_generated_recommendation(
            request.recommendation,
            context,
        )
        if peptide_counts is None and any(
            requires_peptide_counts(item["de_method"], item["ensemble"])
            for item in generated.configs
        ):
            raise ValueError(
                "options.peptide_counts is required for DEqMS and ensembles "
                "containing DEqMS"
            )
        return PreparedEvaluation(
            runtime,
            contrast,
            destination,
            frame,
            conditions,
            peptide_counts,
            context,
            generated,
        )

    def _profile(
        self,
        inputs: ProfileInputs,
        contrast: tuple[str, str],
    ) -> tuple[DataProfile, dict[str, str], BoundContext, pd.DataFrame]:
        """Build the shared profile and policy context for one request."""
        conditions = detect_condition_from_sdrf(
            inputs.sdrf,
            inputs.metadata.factor_column,
        )
        missing = [
            str(sample)
            for sample in inputs.frame.columns[1:]
            if str(sample) not in conditions
        ]
        if missing:
            raise ValueError(
                "Protein-matrix samples are missing from the SDRF mapping: "
                + ", ".join(missing)
            )
        _validate_contrast_samples(inputs.frame, conditions, contrast)
        frame, conditions = _scope_contrast(inputs.frame, conditions, contrast)
        sdrf_batch_fields = detect_sdrf_batch_fields(inputs.sdrf, list(conditions))
        profile = profile_data(
            frame,
            conditions,
            inputs.peptide_counts,
            input_scale=inputs.input_scale,
            metadata=inputs.metadata.acquisition,
        )
        profile = profile._replace(
            batch_fields=list(
                dict.fromkeys([*sdrf_batch_fields, *profile.batch_fields])
            )
        )
        return profile, conditions, bind_context(profile, self._graph), frame


def _abstention_payload(
    graph: KnowledgeGraph,
    prepared: PreparedEvaluation,
) -> dict[str, Any]:
    """Build the persisted result for a policy-required abstention."""
    payload = {
        "status": "abstained",
        "abstain_reason": prepared.generated.abstain_reason,
        "ranking_objective": None,
        "results": [],
        "ranking": [],
        "best_config": None,
        "cache": {"hits": 0, "misses": 0, "unique_combos": 0},
    }
    payload.update(
        _audit_fields(
            graph,
            prepared.generated,
            prepared.context,
            input_scale=None,
        )
    )
    return payload


def _run_candidates(
    candidates: list[CandidateConfig],
    protein_df: pd.DataFrame,
    run: EvaluationRun,
) -> dict[str, Any]:
    """Run a bounded candidate set and persist its exact result tables."""
    slugs = _candidate_slugs(candidates)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    cache = PreprocessCache(run.experiment.threads)
    evaluations: list[EvaluationResult] = []
    de_tables: dict[str, pd.DataFrame] = {}
    for config, filename in zip(candidates, slugs):
        measured, de_table = _evaluate_candidate(config, protein_df, run, cache)
        evaluations.append(measured)
        de_tables[config.name] = de_table
        de_table.to_csv(
            run.output_dir / f"{filename}.de.tsv",
            sep="\t",
            index=False,
        )

    sensitivity_table, sensitivity_summary = method_sensitivity(
        de_tables,
        run.expected_direction if run.ground_truth is not None else None,
    )
    sensitivity_table.to_csv(
        run.output_dir / "method_sensitivity.tsv",
        sep="\t",
        index=False,
    )
    rows = [measured.to_dict() for measured in evaluations]
    payload = build_ranking_payload(
        rows,
        run.ground_truth is not None,
        cache.stats(),
    )
    payload["method_sensitivity"] = sensitivity_summary
    payload["method_sensitivity_artifact"] = "method_sensitivity.tsv"
    return payload


def _evaluate_candidate(
    config: CandidateConfig,
    protein_df: pd.DataFrame,
    run: EvaluationRun,
    cache: PreprocessCache,
) -> tuple[EvaluationResult, pd.DataFrame]:
    """Execute and measure one candidate against the shared round context."""
    de_table = run_experiment(config, protein_df, run.experiment, cache=cache)
    processed = cache.get_or_compute(
        config.normalization,
        config.imputation,
        protein_df,
    )
    measured = evaluate(
        config,
        de_table,
        processed,
        run.experiment.sample_to_condition,
        (
            (run.ground_truth, run.expected_direction)
            if run.ground_truth is not None and run.expected_direction is not None
            else None
        ),
    )
    if run.ground_truth is not None:
        measured = measured.with_score_a(compute_score_ground_truth(measured))
    return measured, de_table


def _candidate_slugs(candidates: list[CandidateConfig]) -> list[str]:
    """Validate candidate identities before creating partial output."""
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    slugs = [_slug(name) for name in names]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Candidate names must map to distinct output filenames")
    return slugs


def _candidate(
    item: dict[str, Any],
    generated: GeneratedRecommendationBlock,
    fdr_threshold: float,
) -> CandidateConfig:
    """Attach block-level provenance to one validated executable config."""
    return CandidateConfig(
        **item,
        fdr_threshold=fdr_threshold,
        evidence_refs=list(generated.evidence_refs),
        confidence=generated.confidence,
        limitations=list(generated.limitations),
        generated_by="host",
    )


def _policy_recommendation(
    profile: DataProfile,
    context: BoundContext,
) -> dict[str, Any]:
    """Return an exact recommendation block from deterministic policy."""
    if not context.scope.generation_allowed:
        errors = [
            item.message for item in context.diagnostics if item.severity == "error"
        ]
        block = {
            "configs": [],
            "evidence_refs": [],
            "confidence": "low",
            "limitations": errors,
            "abstain_reason": "; ".join(errors),
        }
        validate_generated_recommendation(block, context)
        return block

    candidates = rule_propose(profile, context)
    evidence_refs = list(
        dict.fromkeys(
            reference
            for candidate in candidates
            for reference in candidate.evidence_refs
        )
    )
    limitations = list(
        dict.fromkeys(
            limitation
            for candidate in candidates
            for limitation in candidate.limitations
        )
    )
    block = {
        "configs": [
            {field: candidate.to_dict()[field] for field in GENERATED_CONFIG_FIELDS}
            for candidate in candidates
        ],
        "evidence_refs": evidence_refs,
        "confidence": "low",
        "limitations": limitations,
        "abstain_reason": None,
    }
    validate_generated_recommendation(block, context)
    return block


def _parse_dataset_metadata(
    payload: dict[str, Any] | None,
) -> DatasetMetadata:
    """Validate declared acquisition metadata from an MCP object."""
    values = _validated_object(payload, _METADATA_FIELDS, "metadata")
    for name, value in values.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"metadata.{name} must be a non-empty string or null")
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            values.get("data_type"),
            values.get("quantification"),
            values.get("upstream_engine"),
        ),
        factor_column=values.get("factor_column"),
    )


def _parse_evaluation_options(
    payload: dict[str, Any] | None,
) -> EvaluationOptions:
    """Validate the optional execution block from the MCP boundary."""
    values = _validated_object(payload, _OPTION_FIELDS, "options")
    metadata = _parse_dataset_metadata(
        {name: values[name] for name in _METADATA_FIELDS if name in values}
    )
    ground_truth = values.get("ground_truth")
    if ground_truth is not None and not isinstance(ground_truth, str):
        raise ValueError("options.ground_truth must be a string or null")
    expected_direction = values.get("expected_direction")
    fdr_threshold = values.get("fdr_threshold", 0.05)
    if "input_scale" not in values:
        raise ValueError("options.input_scale is required")
    input_scale = values["input_scale"]
    peptide_counts = values.get("peptide_counts")
    if peptide_counts is not None and (
        not isinstance(peptide_counts, str) or not peptide_counts.strip()
    ):
        raise ValueError("options.peptide_counts must be a non-empty string or null")
    threads = values.get("threads", 24)
    if isinstance(fdr_threshold, bool) or not isinstance(fdr_threshold, (int, float)):
        raise ValueError("options.fdr_threshold must be numeric")
    _validate_runtime_options(
        float(fdr_threshold),
        input_scale,
        threads,
        expected_direction,
    )
    if ground_truth is not None and expected_direction is None:
        raise ValueError(
            "options.expected_direction is required when ground_truth is supplied"
        )
    if ground_truth is None and expected_direction is not None:
        raise ValueError(
            "options.expected_direction must be null when ground_truth is null"
        )
    return EvaluationOptions(
        metadata=metadata,
        input_scale=input_scale,
        peptide_counts=peptide_counts,
        ground_truth=ground_truth,
        expected_direction=expected_direction,
        fdr_threshold=float(fdr_threshold),
        threads=threads,
    )
