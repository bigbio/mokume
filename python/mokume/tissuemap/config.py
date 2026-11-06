"""
Configuration dataclasses for the TissueMap pipeline.

Supports loading from YAML via :func:`load_config` and generating annotated
default templates via :func:`generate_default_yaml`.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class InputConfig:
    """Data input configuration."""

    scan_dir: Path = Path(".")
    tmt_datasets: list[str] = field(default_factory=list)
    feature_prefix: Optional[str] = None
    min_tissue_samples: int = 1
    low_sample_warning_threshold: int = 0


@dataclass
class FilteringConfig:
    """Protein filtering configuration."""

    max_nan_frac: float = 0.95
    remove_contaminants: bool = True
    contaminant_pattern: str = r"CONTAM|ENTRAP|DECOY"


@dataclass
class TissueSpecificityConfig:
    """AdaTiSS tissue specificity scoring configuration."""

    use_pure_mad: bool = True
    sigma_floor: Optional[float] = None
    ts_enriched_threshold: Optional[float] = None
    ts_specific_threshold: Optional[float] = None
    ts_housekeeping_threshold: Optional[float] = None


@dataclass
class EmbeddingConfig:
    """Dimensionality reduction configuration.

    The PCA input is built by selecting low-NaN proteins and then imputing the
    remaining missing values via :func:`mokume.imputation.impute_missing_values`.
    Defaults to MNAR-aware ``mindet`` because proteomics missingness is mostly
    not-at-random (low-abundance proteins drop below LOD) — left-censored
    methods preserve the tissue-specific low-abundance signal that MAR
    imputers (median, mean, knn) tend to wash out.
    """

    max_nan_frac_for_pca: Optional[float] = None
    pca_components: int = 50
    tsne_perplexity: float = 15.0
    random_state: int = 42

    # Imputation: any method exposed by mokume.imputation
    # Examples: "mindet", "minprob", "qrilc", "median", "mean", "knn",
    # "seqknn", "missforest", "impseq".
    imputation_method: str = "mindet"
    imputation_n_neighbors: int = 5

    # Dimensionality reduction: "tsne" (default) or "umap"
    embedding_method: str = "tsne"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


@dataclass
class PlottingConfig:
    """Plotting configuration."""

    dpi: int = 300
    save_pdf: bool = True
    n_marker_top: int = 10


@dataclass
class OutputConfig:
    """Output configuration."""

    output_dir: Path = Path("tissuemap_output")


@dataclass
class TissueMapConfig:
    """Top-level configuration for the TissueMap pipeline."""

    n_jobs: int = 8
    input: InputConfig = field(default_factory=InputConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    tissue_specificity: TissueSpecificityConfig = field(
        default_factory=TissueSpecificityConfig
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ── Section → dataclass mapping ──────────────────────────────────────
_SECTION_MAP: dict[str, type] = {
    "input": InputConfig,
    "filtering": FilteringConfig,
    "tissue_specificity": TissueSpecificityConfig,
    "embedding": EmbeddingConfig,
    "plotting": PlottingConfig,
    "output": OutputConfig,
}


def _coerce_value(cls: type, key: str, value: Any) -> Any:
    """Cast YAML values to the types declared on the dataclass field."""
    for f in dataclasses.fields(cls):
        if f.name == key:
            if f.type in ("Path", "pathlib.Path") or f.type is Path:
                return Path(value) if value is not None else value
            return value
    return value


def _parse_sections(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse YAML sections into dataclass instances."""
    sections: dict[str, Any] = {}
    for section_name, cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if not isinstance(section_data, dict):
            section_data = {}
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {
            k: _coerce_value(cls, k, v)
            for k, v in section_data.items()
            if k in valid_fields
        }
        unknown = set(section_data) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown keys in '{section_name}' section: {', '.join(sorted(unknown))}"
            )
        sections[section_name] = cls(**kwargs)
    return sections


def _apply_overrides(cfg: TissueMapConfig, overrides: dict[str, Any]) -> None:
    """Apply dotted CLI overrides to the config in-place."""
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        if "." not in dotted_key:
            if hasattr(cfg, dotted_key):
                object.__setattr__(cfg, dotted_key, value)
            continue
        section_name, field_name = dotted_key.split(".", 1)
        if section_name not in _SECTION_MAP:
            continue
        sub = getattr(cfg, section_name)
        if hasattr(sub, field_name):
            coerced = _coerce_value(type(sub), field_name, value)
            object.__setattr__(sub, field_name, coerced)


def load_config(
    yaml_path: Path, overrides: dict[str, Any] | None = None
) -> TissueMapConfig:
    """Load a :class:`TissueMapConfig` from a YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML configuration file.
    overrides : dict, optional
        CLI overrides applied **after** YAML loading.
        Keys are ``"section.field"`` strings, e.g. ``"embedding.pca_components"``.

    Returns
    -------
    TissueMapConfig
    """
    with open(yaml_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    known_top_level = {"n_jobs", *_SECTION_MAP}
    unknown_top_level = sorted(set(raw) - known_top_level)
    if unknown_top_level:
        raise ValueError(
            f"Unknown TissueMap config keys: {', '.join(unknown_top_level)}"
        )

    sections = _parse_sections(raw)

    top_kwargs: dict[str, Any] = {}
    if "n_jobs" in raw:
        top_kwargs["n_jobs"] = int(raw["n_jobs"])

    cfg = TissueMapConfig(**top_kwargs, **sections)

    if overrides:
        _apply_overrides(cfg, overrides)

    return cfg


_YAML_TEMPLATE = """\
# ─────────────────────────────────────────────────────────────────────
# TissueMap pipeline configuration
# Generated by: mokume tissuemap --generate-config
#
# All values below are defaults. Uncomment and modify as needed.
# CLI options (--input, --outdir, etc.) override this file.
#
# Advanced parameters not shown here can still be added manually.
# See full reference: mokume.tissuemap.config dataclass docstrings.
# ─────────────────────────────────────────────────────────────────────

# ── Pipeline ─────────────────────────────────────────────────────
# n_jobs: 8                                 # Threads for t-SNE & parallel datasets

# ── Data input ───────────────────────────────────────────────────────
# input:
#   scan_dir: "."                      # Directory with PXD dataset(s)
#   tmt_datasets: []                   # Force TMT for these dataset IDs
#   feature_prefix: null               # Custom feature parquet prefix
#   min_tissue_samples: 1              # Min samples per tissue

# ── Protein filtering ────────────────────────────────────────────────
# filtering:
#   max_nan_frac: 0.95                 # Max NaN fraction to keep protein
#   remove_contaminants: true
#   contaminant_pattern: "CONTAM|ENTRAP|DECOY"

# ── AdaTiSS tissue specificity scoring ───────────────────────────────
# tissue_specificity:
#   use_pure_mad: true                 # false → use DPD fitting
#   sigma_floor: null                  # null → auto-detect from data
#   ts_enriched_threshold: null        # null → GMM auto (floor 2.5)
#   ts_specific_threshold: null        # null → GMM auto (floor 4.0)
#   ts_housekeeping_threshold: null    # null → GMM auto (floor 2.0)

# ── Dimensionality reduction ─────────────────────────────────────────
# embedding:
#   # (always runs)
#   max_nan_frac_for_pca: null         # null → auto (ensure enough proteins for PCA)
#   pca_components: 50
#   tsne_perplexity: 15.0
#   random_state: 42                   # Seed for reproducibility
#   # Imputation for the PCA input (MNAR-aware default for proteomics):
#   #   mindet | minprob | qrilc | knn | seqknn | missforest | bpca | gms |
#   #   impseq | impseqrob | median
#   imputation_method: mindet
#   imputation_n_neighbors: 5          # Used by knn / seqknn
#   # Backend after PCA: tsne (default) or umap (requires umap-learn)
#   embedding_method: tsne
#   umap_n_neighbors: 15
#   umap_min_dist: 0.1

# ── Plotting ─────────────────────────────────────────────────────────
# plotting:
#   dpi: 300
#   save_pdf: true
#   n_marker_top: 10                   # Top markers per tissue to save

# ── Output ───────────────────────────────────────────────────────────
# output:
#   output_dir: "tissuemap_output"
"""


def generate_default_yaml(out_path: Path) -> None:
    """Write the annotated default YAML template to *out_path*."""
    out_path.write_text(_YAML_TEMPLATE, encoding="utf-8")
    logger.info("Default config written to %s", out_path)
