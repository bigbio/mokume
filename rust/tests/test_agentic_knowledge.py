"""Knowledge-location contracts of the Plugin runtime."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from mokume.agentic.knowledge import load_knowledge_graph
from mokume.agentic.knowledge_search import search_knowledge
from mokume.agentic.service import RecommendationService

REPOSITORY = Path(__file__).resolve().parents[2]
KNOWLEDGE_BUNDLE = (
    REPOSITORY / "rust" / "python" / "mokume" / "agentic" / "knowledge_bundle"
)
BUNDLED_KNOWLEDGE = KNOWLEDGE_BUNDLE / "knowledge.yaml"


def test_knowledge_resolution_prefers_explicit_then_environment_then_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Development overrides must not replace the installed default."""
    monkeypatch.delenv("MOKUME_AGENTIC_KNOWLEDGE", raising=False)
    bundled = load_knowledge_graph()
    assert bundled.fingerprint == load_knowledge_graph(BUNDLED_KNOWLEDGE).fingerprint
    assert RecommendationService() is not None

    copied = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_BUNDLE, copied)
    override = copied / "knowledge.yaml"
    override.write_text(
        override.read_text(encoding="utf-8").replace(
            "title: Evaluating differential expression methods for proteomics data",
            "title: Environment-selected knowledge snapshot",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOKUME_AGENTIC_KNOWLEDGE", str(override))

    environment = load_knowledge_graph()
    assert environment.fingerprint != bundled.fingerprint
    assert load_knowledge_graph(BUNDLED_KNOWLEDGE).fingerprint == bundled.fingerprint


def test_bundled_knowledge_exposes_dataset_summaries_without_promoting_them() -> None:
    """All benchmark datasets are searchable but never recommendation priors."""
    graph = load_knowledge_graph(BUNDLED_KNOWLEDGE)
    dataset = graph.datasets["dataset-PXD071205_DI__DI_5min_250pg"]

    assert len(graph.datasets) == 88
    assert dataset.data_type == "DIA"
    assert dataset.preset_eligible is False
    assert dataset.benchmark["successful_candidates"] == 744
    assert dataset.benchmark["failed_candidates"] == 0
    assert dataset.held_out_evaluation is None
    assert all(
        record.id in graph.evidence
        for record in graph.matching(
            SimpleNamespace(data_type="DIA", quantification=None)
        )
    )


@pytest.mark.parametrize(
    ("quantification", "grid_preset"),
    [
        (None, "grid-lfq-preset"),
        ("maxlfq", "grid-lfq-maxlfq-preset"),
        ("top3", "grid-lfq-top3-preset"),
    ],
)
def test_declared_quantification_selects_its_grid_preset(
    quantification: str | None, grid_preset: str
) -> None:
    """A frozen quantification gets the Grid preset measured on its matrices."""
    graph = load_knowledge_graph(BUNDLED_KNOWLEDGE)
    profile = SimpleNamespace(data_type="LFQ", quantification=quantification)

    assert [record.id for record in graph.matching(profile)] == [
        grid_preset,
        "opdea-fragpipe-dda",
        "opdea-maxquant-dda",
    ]


def test_search_matches_whole_words() -> None:
    """A DIA query does not match records through words such as 'median'."""
    result = search_knowledge(load_knowledge_graph(BUNDLED_KNOWLEDGE), "DIA")

    assert result["count"] == 5
    assert {item["applicability"]["data_type"] for item in result["results"]} == {"DIA"}


def test_mcp_main_uses_bundle_and_accepts_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hidden CLI starts without plugin-root knowledge arguments."""
    module = importlib.import_module("mokume.agentic.mcp_server")
    observed: list[tuple[str, str | None]] = []

    def run(*, transport: str) -> None:
        observed.append(("transport", transport))

    def create_server(knowledge: str | None = None) -> SimpleNamespace:
        observed.append(("knowledge", knowledge))
        return SimpleNamespace(run=run)

    monkeypatch.setattr(module, "create_server", create_server)

    assert module.main([]) == 0
    assert observed == [("knowledge", None), ("transport", "stdio")]

    observed.clear()
    assert module.main(["--knowledge", str(BUNDLED_KNOWLEDGE)]) == 0
    assert observed == [
        ("knowledge", str(BUNDLED_KNOWLEDGE.resolve())),
        ("transport", "stdio"),
    ]


def test_mcp_search_returns_explanation_only_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plugin hosts search bundled evidence without gaining execution authority."""
    pytest.importorskip("mcp")
    create_server = getattr(
        importlib.import_module("mokume.agentic.mcp_server"),
        "create_server",
    )
    monkeypatch.chdir(tmp_path)
    server = create_server()
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    schema = tools["search_knowledge"].inputSchema
    assert set(schema["properties"]) == {"query", "data_type", "method", "limit"}
    assert schema["required"] == ["query"]

    _, payload = asyncio.run(
        server.call_tool("search_knowledge", {"query": "PXD020815", "limit": 2})
    )

    assert payload["scope"] == "explanation_only"
    assert payload["execution_authority"] is False
    assert [item["id"] for item in payload["results"]] == [
        "dataset-PXD020815",
        "grid-tmt-preset",
    ]
