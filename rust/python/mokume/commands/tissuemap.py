#!/usr/bin/env python3
"""Per-dataset tissue proteome analysis command (``mokume.tissuemap``).

A periphery command in the mokume wheel: it re-implements no compute kernel; it
drives the tissue-atlas pipeline (PCA / t-SNE / UMAP, AdaTiSS specificity,
markers, atlas plots) over a dataset directory. Exposed as ``mokume tissuemap``
and in-process as ``mokume.tissuemap``.

Run requirements: the ``tissuemap`` extra (scanpy, umap-learn, combat, anndata,
matplotlib, seaborn, ...): ``pip install mokume[tissuemap]``. The
``mokume.imputation`` subpackage that ``tissuemap`` depends on ships in the
wheel.

argv contract:
    --input/-i          PATH    dataset directory (required unless --generate-config)
    --outdir/-o         PATH    output directory (default: tissuemap_output)
    --config            PATH    YAML configuration file
    --generate-config   PATH    write default YAML template and exit
    --tmt-dataset       STR     TMT dataset id (repeatable)
    --threads/-t        INT     threads (default: 8)
    --dpi               INT     plot DPI
    --impute-method     STR
    --embedding-method  {tsne,umap}
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional


_IMPUTATION_CHOICES = (
    "mindet",
    "minprob",
    "qrilc",
    "knn",
    "seqknn",
    "missforest",
    "impseq",
    "median",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mokume tissuemap",
        description="Build a tissue proteome atlas.",
    )
    parser.add_argument(
        "-i", "--input", dest="scan_dir", metavar="<DIR>", type=Path, default=None
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="output_dir",
        metavar="<DIR>",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config", dest="config_path", metavar="<FILE>", type=Path, default=None
    )
    parser.add_argument("--generate-config", metavar="<FILE>", type=Path, default=None)
    parser.add_argument(
        "--tmt-dataset",
        dest="tmt_datasets",
        metavar="<ID>",
        action="append",
        default=[],
    )
    parser.add_argument(
        "-t", "--threads", dest="n_jobs", metavar="<N>", type=int, default=None
    )
    parser.add_argument("--dpi", metavar="<N>", type=int, default=None)
    parser.add_argument(
        "--impute-method",
        dest="imputation_method",
        metavar="<METHOD>",
        type=str.lower,
        choices=_IMPUTATION_CHOICES,
        default=None,
        help="[possible values: " + ", ".join(_IMPUTATION_CHOICES) + "]",
    )
    parser.add_argument(
        "--embedding-method",
        dest="embedding_method",
        metavar="<METHOD>",
        type=str.lower,
        choices=("tsne", "umap"),
        default=None,
        help="[possible values: tsne, umap]",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace):
    """Mirror ``mokume.commands.tissuemap._build_config``."""
    from mokume.tissuemap.config import (
        EmbeddingConfig,
        InputConfig,
        OutputConfig,
        PlottingConfig,
        TissueMapConfig,
        load_config,
    )

    scan_dir: Optional[Path] = args.scan_dir
    output_dir: Optional[Path] = args.output_dir
    config_path: Optional[Path] = args.config_path
    tmt_datasets = tuple(args.tmt_datasets)
    n_jobs: Optional[int] = args.n_jobs
    dpi: Optional[int] = args.dpi
    imputation_method: Optional[str] = args.imputation_method
    embedding_method: Optional[str] = args.embedding_method

    if config_path is not None:
        overrides: dict[str, object] = {}
        if scan_dir is not None:
            overrides["input.scan_dir"] = str(scan_dir)
        if output_dir is not None:
            overrides["output.output_dir"] = str(output_dir)
        if n_jobs is not None:
            overrides["n_jobs"] = n_jobs
        if tmt_datasets:
            overrides["input.tmt_datasets"] = list(tmt_datasets)
        if dpi is not None:
            overrides["plotting.dpi"] = dpi
        if imputation_method is not None:
            overrides["embedding.imputation_method"] = imputation_method.lower()
        if embedding_method is not None:
            overrides["embedding.embedding_method"] = embedding_method.lower()
        return load_config(config_path, overrides=overrides)

    embedding_kwargs: dict[str, str] = {}
    if imputation_method is not None:
        embedding_kwargs["imputation_method"] = imputation_method.lower()
    if embedding_method is not None:
        embedding_kwargs["embedding_method"] = embedding_method.lower()

    return TissueMapConfig(
        n_jobs=n_jobs if n_jobs is not None else 8,
        input=InputConfig(scan_dir=scan_dir, tmt_datasets=list(tmt_datasets)),
        plotting=PlottingConfig(dpi=dpi if dpi is not None else 300),
        output=OutputConfig(
            output_dir=output_dir
            if output_dir is not None
            else Path("tissuemap_output")
        ),
        embedding=EmbeddingConfig(**embedding_kwargs),
    )


def _handle_generate_config(args, generate_default_yaml):
    if args.generate_config is None:
        return None
    if any(
        (
            args.scan_dir is not None,
            args.output_dir is not None,
            args.config_path is not None,
            bool(args.tmt_datasets),
            args.n_jobs is not None,
            args.dpi is not None,
            args.imputation_method is not None,
            args.embedding_method is not None,
        )
    ):
        print(
            "error: --generate-config cannot be combined with run/config overrides",
            file=sys.stderr,
        )
        return 2
    generate_default_yaml(args.generate_config)
    print(f"Default config written to {args.generate_config}")
    return 0


def _config_error(config):
    if not config.input.scan_dir.exists() or not config.input.scan_dir.is_dir():
        return (
            "TissueMap scan directory does not exist or is not a directory: "
            f"{config.input.scan_dir}"
        )
    if config.n_jobs < 1:
        return "n_jobs must be greater than zero"
    if config.plotting.dpi < 1:
        return "plotting.dpi must be greater than zero"
    return None


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    try:
        from mokume.tissuemap.config import generate_default_yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "error: failed to import mokume tissuemap configuration: "
            f"{exc}\nInstall them with: pip install mokume[tissuemap]",
            file=sys.stderr,
        )
        return 1

    generate_result = _handle_generate_config(args, generate_default_yaml)
    if generate_result is not None:
        return generate_result

    if args.scan_dir is None and args.config_path is None:
        print(
            "error: --input is required unless it is defined by --config.",
            file=sys.stderr,
        )
        return 2

    config = _build_config(args)
    config_error = _config_error(config)
    if config_error:
        print(f"error: {config_error}", file=sys.stderr)
        return 2

    try:
        pipeline_class = getattr(
            importlib.import_module("mokume.tissuemap.pipeline"), "TissueMapPipeline"
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "error: failed to import mokume tissuemap dependencies: "
            f"{exc}\nInstall them with: pip install mokume[tissuemap]",
            file=sys.stderr,
        )
        return 1

    pipeline_class(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
