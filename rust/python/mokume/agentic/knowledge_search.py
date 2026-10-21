"""Bounded text search over validated Mokume knowledge records."""

from __future__ import annotations

from collections.abc import Collection, Iterable
import re
from typing import Any, NamedTuple

from mokume.agentic.contract import DATA_TYPES
from mokume.agentic.dataset_knowledge import DatasetEvidence
from mokume.agentic.knowledge import EvidenceRecord, KnowledgeGraph


SearchRecord = EvidenceRecord | DatasetEvidence
RankedRecord = tuple[int, int, str, SearchRecord]
MAX_SEARCH_RESULTS = 5
_SEARCH_TOKEN = re.compile(r"[^\W_]+(?:[.+-][^\W_]+)*", re.UNICODE)


class _SearchFilters(NamedTuple):
    """Validated search text and filters."""

    query: str
    data_type: str | None
    method: str | None
    tokens: set[str]
    method_filter: str | None


def method_key(value: str) -> str:
    """Normalize a method name for exact method filtering."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def search_knowledge(
    graph: KnowledgeGraph,
    query: str,
    *,
    data_type: str | None = None,
    method: str | None = None,
    limit: int = MAX_SEARCH_RESULTS,
) -> dict[str, Any]:
    """Search method evidence and dataset summaries for explanation only."""
    filters = _search_filters(query, data_type, method, limit)
    results = []
    for record in matching_knowledge_records(
        graph, filters.tokens, filters.data_type, filters.method_filter, limit
    ):
        result = record.to_context_dict(graph.sources[record.source_id])
        if isinstance(record, EvidenceRecord):
            result["eligible_as_prior"] = record.eligible_as_prior
        results.append(result)
    return {
        "scope": "explanation_only",
        "execution_authority": False,
        "knowledge_fingerprint": graph.fingerprint,
        "query": filters.query,
        "filters": {"data_type": filters.data_type, "method": filters.method},
        "count": len(results),
        "results": results,
    }


def matching_knowledge_records(
    graph: KnowledgeGraph,
    query_tokens: set[str],
    data_type: str | None,
    selected_method: str | None,
    limit: int,
) -> list[SearchRecord]:
    """Rank method evidence and dataset summaries for one validated query."""
    requested = _requested_datasets(graph.datasets.values(), query_tokens)
    ranked = _rank_evidence(
        graph, query_tokens, data_type, selected_method, requested.values()
    )
    ranked.extend(
        _rank_datasets(graph, query_tokens, data_type, selected_method, requested)
    )
    ranked.sort(key=lambda item: item[:3])
    return [record for _, _, _, record in ranked[:limit]]


def _search_filters(
    query: str, data_type: str | None, method: str | None, limit: int
) -> _SearchFilters:
    normalized_query = _search_value(query, "query", 200)
    normalized_type = (
        _search_value(data_type, "data_type", 3).upper()
        if data_type is not None
        else None
    )
    if normalized_type is not None and normalized_type not in DATA_TYPES:
        raise ValueError("data_type must be DIA, LFQ, or TMT")
    normalized_method = (
        _search_value(method, "method", 64) if method is not None else None
    )
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_SEARCH_RESULTS
    ):
        raise ValueError(f"limit must be an integer from 1 to {MAX_SEARCH_RESULTS}")
    query_tokens = _tokens(normalized_query)
    if not query_tokens:
        raise ValueError("query must contain searchable text")
    selected_method = method_key(normalized_method) if normalized_method else None
    if normalized_method and not selected_method:
        raise ValueError("method must contain searchable text")
    return _SearchFilters(
        normalized_query,
        normalized_type,
        normalized_method,
        query_tokens,
        selected_method,
    )


def _search_value(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    if "\x00" in normalized or len(normalized) > maximum:
        raise ValueError(f"{name} is invalid or too long")
    return normalized


def _tokens(text: str) -> set[str]:
    """Whole search words, so 'dia' does not match inside 'median'."""
    return {token.casefold() for token in _SEARCH_TOKEN.findall(text)}


def _requested_datasets(
    records: Iterable[DatasetEvidence], query_tokens: set[str]
) -> dict[str, DatasetEvidence]:
    return {
        record.id: record
        for record in records
        if _dataset_identifiers(record) & query_tokens
    }


def _rank_evidence(
    graph: KnowledgeGraph,
    query_tokens: set[str],
    data_type: str | None,
    selected_method: str | None,
    requested: Collection[DatasetEvidence],
) -> list[RankedRecord]:
    ranked: list[RankedRecord] = []
    for record in graph.evidence.values():
        if requested and not _record_references_datasets(record, requested):
            continue
        if data_type and record.applicability.data_type.upper() != data_type:
            continue
        if selected_method and selected_method not in _record_method_keys(record):
            continue
        score = len(query_tokens & _tokens(_record_search_text(graph, record)))
        if score:
            ranked.append((-score, record.priority, record.id, record))
    return ranked


def _rank_datasets(
    graph: KnowledgeGraph,
    query_tokens: set[str],
    data_type: str | None,
    selected_method: str | None,
    requested: dict[str, DatasetEvidence],
) -> list[RankedRecord]:
    ranked: list[RankedRecord] = []
    for record in graph.datasets.values():
        if requested and record.id not in requested:
            continue
        if data_type and record.data_type != data_type:
            continue
        identifier_match = bool(_dataset_identifiers(record) & query_tokens)
        if selected_method and (
            not identifier_match or selected_method not in _dataset_method_keys(record)
        ):
            continue
        score = len(query_tokens & _tokens(_dataset_search_text(graph, record)))
        if score:
            ranked.append((-score - 10 * identifier_match, 30, record.id, record))
    return ranked


def _dataset_identifiers(record: DatasetEvidence) -> set[str]:
    return {record.dataset_id.casefold(), record.project_accession.casefold()}


def _record_method_keys(record: EvidenceRecord) -> set[str]:
    values = [
        record.pipeline.quantification,
        record.pipeline.normalization,
        record.pipeline.imputation,
        record.pipeline.de_method,
        record.pipeline.fdr_method,
        record.pipeline.ensemble,
        record.applicability.setting,
        record.applicability.upstream_engine,
    ]
    return {method_key(str(value)) for value in values if value is not None}


def _record_search_text(graph: KnowledgeGraph, record: EvidenceRecord) -> str:
    source = graph.sources[record.source_id]
    values = [
        record.id,
        record.source_id,
        record.kind,
        record.status,
        record.confidence,
        *record.applicability.to_dict().values(),
        *record.pipeline.to_dict().values(),
        *record.metrics.keys(),
        *record.metrics.values(),
        *record.limitations,
        source.id,
        source.kind,
        source.title,
        source.trust,
        source.status,
    ]
    if record.reference_profile is not None:
        values.extend(record.reference_profile.projects)
    return " ".join(str(value) for value in values if value is not None)


def _record_references_datasets(
    record: EvidenceRecord, datasets: Collection[DatasetEvidence]
) -> bool:
    if record.reference_profile is None:
        return False
    referenced = {project.casefold() for project in record.reference_profile.projects}
    return any(_dataset_identifiers(dataset) & referenced for dataset in datasets)


def _dataset_method_keys(record: DatasetEvidence) -> set[str]:
    return {method_key(value) for value in record.method_values()}


def _dataset_search_text(graph: KnowledgeGraph, record: DatasetEvidence) -> str:
    source = graph.sources[record.source_id]
    values = [
        record.id,
        record.dataset_id,
        record.project_accession,
        record.study,
        record.data_type,
        record.design,
        record.ground_truth,
        record.benchmark,
        record.held_out_evaluation,
        source.id,
        source.kind,
        source.title,
        source.trust,
        source.status,
    ]
    return " ".join(str(value) for value in values if value is not None)
