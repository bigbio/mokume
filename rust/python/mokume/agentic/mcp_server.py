"""Local stdio MCP server for the installable Mokume Plugin."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


def create_server(knowledge: str | None = None):
    """Build the MCP server around one immutable knowledge snapshot."""
    try:
        fast_mcp = getattr(importlib.import_module("mcp.server.fastmcp"), "FastMCP")
        service_module = importlib.import_module("mokume.agentic.service")
        service_type = getattr(service_module, "RecommendationService")
        inspection_type = getattr(service_module, "InspectionRequest")
        request_type = getattr(service_module, "EvaluationRequest")
        load_graph = getattr(
            importlib.import_module("mokume.agentic.knowledge"), "load_knowledge_graph"
        )
        search = getattr(
            importlib.import_module("mokume.agentic.knowledge_search"),
            "search_knowledge",
        )
    except ImportError as exc:
        raise RuntimeError(
            "Mokume MCP dependencies are missing; install 'mokume[plugin]'"
        ) from exc

    server = fast_mcp("mokume")
    service = service_type(knowledge)
    # The service loads the same cached snapshot.
    graph = load_graph(knowledge)

    @server.tool()
    def inspect_dataset(
        protein_matrix: str,
        sdrf: str,
        contrast: list[str],
        options: dict[str, str | None],
    ) -> dict[str, Any]:
        """Profile a protein matrix and return policy-filtered Mokume evidence."""
        inspection_options = dict(options)
        input_scale = inspection_options.pop("input_scale", None)
        if not isinstance(input_scale, str) or not input_scale:
            raise ValueError("options.input_scale must be a non-empty string")
        peptide_counts = inspection_options.pop("peptide_counts", None)
        return service.inspect_dataset(
            inspection_type(
                protein_matrix,
                sdrf,
                input_scale,
                contrast,
                peptide_counts,
                inspection_options,
            )
        )

    @server.tool()
    def evaluate_recommendation(
        protein_matrix: str,
        sdrf: str,
        contrast: list[str],
        recommendation: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and run a scoped candidate block with the Rust kernel."""
        runtime_options = dict(options)
        output_dir = runtime_options.pop("output_dir", None)
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("options.output_dir must be a non-empty string")
        return service.evaluate_recommendation(
            request_type(
                protein_matrix,
                sdrf,
                contrast,
                recommendation,
                output_dir,
                runtime_options,
            )
        )

    @server.tool()
    def search_knowledge(
        query: str,
        data_type: str | None = None,
        method: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search benchmark evidence and dataset summaries by method, type, or PXD.

        Results explain choices; they never authorize a configuration.
        """
        return search(graph, query, data_type=data_type, method=method, limit=limit)

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the plugin MCP server over stdio."""
    parser = argparse.ArgumentParser(prog="mokume mcp serve")
    parser.add_argument("--knowledge", metavar="<FILE>")
    args = parser.parse_args(argv)
    knowledge = None
    if args.knowledge is not None:
        path = Path(args.knowledge).expanduser().resolve()
        if not path.is_file():
            parser.error(f"knowledge catalog not found: {path}")
        knowledge = str(path)
    create_server(knowledge).run(transport="stdio")
    return 0
