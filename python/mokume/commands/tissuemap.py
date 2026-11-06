"""
CLI command for the TissueMap pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click


@click.command("tissuemap", short_help="Per-dataset tissue proteome analysis")
@click.option(
    "--scan-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing QPX dataset(s). Can be a single PXD directory "
    "or a parent with PXD* subdirectories. [required unless --generate-config]",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML configuration file. Use --generate-config to create a template.",
)
@click.option(
    "--generate-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write default YAML config template to this path and exit.",
)
@click.option(
    "--tmt-dataset",
    "tmt_datasets",
    multiple=True,
    default=(),
    help="Dataset ID(s) that are TMT (repeat for multiple). "
    "Auto-detected if not specified.",
)
@click.option(
    "--n-jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Threads for t-SNE and parallel dataset processing.",
)
@click.option(
    "--dpi",
    type=int,
    default=None,
    help="Plot resolution (DPI).  [default: 300]",
)
@click.option(
    "--imputation-method",
    "imputation_method",
    type=click.Choice(
        [
            "mindet",
            "minprob",
            "qrilc",
            "knn",
            "seqknn",
            "missforest",
            "impseq",
            "median",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Imputation for the PCA input. Defaults to 'mindet' (MNAR-aware, "
    "recommended for proteomics). Overrides any value in --config.",
)
@click.option(
    "--embedding-method",
    "embedding_method",
    type=click.Choice(["tsne", "umap"], case_sensitive=False),
    default=None,
    help="Dimensionality reduction backend after PCA. Defaults to 'tsne'. "
    "'umap' requires the optional 'umap-learn' package.",
)
def tissuemap_cmd(
    scan_dir: Optional[Path],
    output_dir: Optional[Path],
    config_path: Optional[Path],
    generate_config: Optional[Path],
    tmt_datasets: tuple[str, ...],
    n_jobs: Optional[int],
    dpi: Optional[int],
    imputation_method: Optional[str],
    embedding_method: Optional[str],
) -> None:
    """Per-dataset tissue proteome analysis.

    \b
    Quick start:
      mokume tissuemap --scan-dir ./data --output-dir ./results

    \b
    Advanced (with config file):
      mokume tissuemap --generate-config tissuemap.yaml
      mokume tissuemap --scan-dir ./data --config tissuemap.yaml
    """
    from mokume.tissuemap.config import generate_default_yaml
    from mokume.tissuemap.pipeline import TissueMapPipeline

    if generate_config is not None:
        if any(
            (
                scan_dir is not None,
                output_dir is not None,
                config_path is not None,
                bool(tmt_datasets),
                n_jobs is not None,
                dpi is not None,
                imputation_method is not None,
                embedding_method is not None,
            )
        ):
            raise click.UsageError(
                "--generate-config cannot be combined with run/config overrides"
            )
        generate_default_yaml(generate_config)
        click.echo(f"Default config written to {generate_config}")
        return

    config = _build_config(
        scan_dir,
        output_dir,
        config_path,
        tmt_datasets,
        n_jobs,
        dpi,
        imputation_method,
        embedding_method,
    )
    if scan_dir is None and config_path is None:
        raise click.UsageError(
            "--scan-dir is required unless it is defined by --config"
        )
    if not config.input.scan_dir.is_dir():
        raise click.UsageError(
            f"TissueMap scan directory does not exist: {config.input.scan_dir}"
        )
    if config.n_jobs < 1:
        raise click.UsageError("n_jobs must be greater than zero")
    if config.plotting.dpi < 1:
        raise click.UsageError("plotting.dpi must be greater than zero")
    TissueMapPipeline(config).run()


def _build_config(
    scan_dir: Optional[Path],
    output_dir: Optional[Path],
    config_path: Optional[Path],
    tmt_datasets: tuple[str, ...],
    n_jobs: Optional[int],
    dpi: Optional[int],
    imputation_method: Optional[str],
    embedding_method: Optional[str],
):
    """Build TissueMapConfig from CLI arguments."""
    from mokume.tissuemap.config import (
        EmbeddingConfig,
        InputConfig,
        OutputConfig,
        PlottingConfig,
        TissueMapConfig,
        load_config,
    )

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
