"""Data profiling for agentic analysis."""

from dataclasses import dataclass
import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mokume.core.constants import load_sdrf
from mokume.core.logger import get_logger

logger = get_logger("mokume.agentic.profiler")

_UPSTREAM_ENGINE_ALIASES = {
    "diann": "DIA-NN",
    "fragpipe": "FragPipe",
    "maxquant": "MaxQuant",
    "quantms": "quantms",
    "spectronaut": "Spectronaut",
}


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Declared upstream facts used for evidence matching."""

    data_type: str | None = None
    quantification: str | None = None
    upstream_engine: str | None = None


class DataProfile(NamedTuple):
    """Characterization of a protein intensity matrix."""

    n_proteins: int
    n_samples: int
    n_conditions: int
    samples_per_condition: dict[str, int]
    missing_rate: float
    missing_per_sample: dict[str, float]
    median_cv: float | None
    has_peptide_counts: bool
    data_type: str
    batch_fields: list[str]
    intensity_range: tuple[float, float]
    is_log_transformed: bool
    pca_explained_variance: list[float]
    per_condition_median_cv: dict[str, float]
    pairwise_median_abs_log2fc: dict[str, float]
    data_type_source: str
    quantification: str | None
    upstream_engine: str | None

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dictionary."""
        return {
            "n_proteins": self.n_proteins,
            "n_samples": self.n_samples,
            "n_conditions": self.n_conditions,
            "samples_per_condition": self.samples_per_condition,
            "missing_rate": round(self.missing_rate, 4),
            "median_cv": (
                round(self.median_cv, 4) if self.median_cv is not None else None
            ),
            "has_peptide_counts": self.has_peptide_counts,
            "data_type": self.data_type,
            "data_type_source": self.data_type_source,
            "quantification": self.quantification,
            "upstream_engine": self.upstream_engine,
            "batch_fields": self.batch_fields,
            "intensity_range": [round(v, 2) for v in self.intensity_range],
            "is_log_transformed": self.is_log_transformed,
            "pca_explained_variance": [
                round(v, 4) for v in self.pca_explained_variance
            ],
            "per_condition_median_cv": {
                k: round(v, 4) for k, v in self.per_condition_median_cv.items()
            },
            "pairwise_median_abs_log2fc": {
                k: round(v, 4) for k, v in self.pairwise_median_abs_log2fc.items()
            },
        }


def compute_median_cv(
    matrix: pd.DataFrame,
    sample_to_condition: dict[str, str],
) -> float | None:
    """Compute median within-condition CV across proteins."""
    conditions = set(sample_to_condition.values())
    all_cvs = []
    for cond in conditions:
        cols = [s for s in matrix.columns if sample_to_condition.get(s) == cond]
        if len(cols) < 2:
            continue
        subset = matrix[cols]
        row_mean = subset.mean(axis=1)
        row_std = subset.std(axis=1, ddof=1)
        cv = row_std / row_mean.replace(0, np.nan)
        all_cvs.extend(cv.dropna().tolist())
    return float(np.median(all_cvs)) if all_cvs else None


def _linear_cv_matrix(matrix: pd.DataFrame, input_scale: str) -> pd.DataFrame:
    """Return linear intensities for coefficient-of-variation diagnostics."""
    if input_scale == "linear":
        return matrix
    with np.errstate(over="raise", invalid="raise"):
        values = np.exp2(matrix.to_numpy(dtype=float))
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def _compute_per_condition_cv(
    matrix: pd.DataFrame,
    sample_to_condition: dict[str, str],
) -> dict[str, float]:
    """Return per-condition median CV (proteins with >=2 samples in condition)."""
    conditions = sorted(set(sample_to_condition.values()))
    out: dict[str, float] = {}
    for cond in conditions:
        cols = [s for s in matrix.columns if sample_to_condition.get(s) == cond]
        if len(cols) < 2:
            continue
        subset = matrix[cols]
        row_mean = subset.mean(axis=1)
        row_std = subset.std(axis=1, ddof=1)
        cv = (row_std / row_mean.replace(0, np.nan)).dropna()
        if len(cv) > 0:
            out[cond] = float(cv.median())
    return out


def _compute_pca_evr(matrix: pd.DataFrame, n_components: int = 2) -> list[float]:
    """Return explained-variance ratios for the first N principal components.

    Drops proteins with missing values; falls back to an empty list when too
    few samples / proteins remain for a stable decomposition.
    """
    if matrix.shape[1] < 3 or matrix.shape[0] < 5:
        return []
    complete = matrix.dropna(how="any")
    if complete.shape[0] < 5:
        return []
    data = complete.values.T  # samples x proteins
    n_comp = min(n_components, data.shape[0] - 1, data.shape[1])
    if n_comp < 1:
        return []
    pca = PCA(n_components=n_comp, random_state=0)
    try:
        pca.fit(data)
    except (ValueError, np.linalg.LinAlgError):
        return []
    return [float(v) for v in pca.explained_variance_ratio_]


def _compute_pairwise_log2fc(
    matrix: pd.DataFrame,
    sample_to_condition: dict[str, str],
    is_log: bool,
) -> dict[str, float]:
    """Median |log2FC| of condition-mean profiles for each unordered pair."""
    conditions = sorted(set(sample_to_condition.values()))
    if len(conditions) < 2:
        return {}
    cond_means: dict[str, pd.Series] = {}
    for cond in conditions:
        cols = [s for s in matrix.columns if sample_to_condition.get(s) == cond]
        if not cols:
            continue
        cond_means[cond] = matrix[cols].mean(axis=1, skipna=True)
    out: dict[str, float] = {}
    keys = sorted(cond_means)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            diff = _pairwise_log2fc(cond_means[a], cond_means[b], is_log)
            if len(diff) > 0:
                out[f"{a}_vs_{b}"] = float(np.median(np.abs(diff)))
    return out


def _pairwise_log2fc(
    mean_a: pd.Series,
    mean_b: pd.Series,
    is_log: bool,
) -> pd.Series:
    """Return finite log2 fold changes between two condition means."""
    if is_log:
        return (mean_a - mean_b).dropna()
    ratio = (mean_a / mean_b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio > 0).dropna()
    return np.log2(ratio) if len(ratio) > 0 else pd.Series(dtype=float)


def _detect_data_type(
    sample_to_condition: dict[str, str],
) -> str:
    """Heuristic data type detection from sample naming patterns."""
    joined = " ".join(sample_to_condition).casefold()
    tokens = re.sub(r"[^a-z0-9]+", " ", joined).split()
    if any(
        re.fullmatch(r"(?:tmt\d*(?:plex)?|\d+plex|plex)", token) for token in tokens
    ):
        return "TMT"
    if "dia" in tokens or "diann" in tokens:
        return "DIA"
    if "lfq" in tokens:
        return "LFQ"
    return "unknown"


def _canonicalize_upstream_engine(value: str | None) -> str | None:
    """Return the catalog spelling for a declared upstream engine."""
    if value is None:
        return None
    stripped = value.strip()
    key = re.sub(r"[^a-z0-9]+", "", stripped.casefold())
    return _UPSTREAM_ENGINE_ALIASES.get(key, stripped)


def _detect_batch_fields(sample_ids: list[str]) -> list[str]:
    """Detect potential batch structure from sample ID prefixes."""
    prefixes = {s.split("-")[0] for s in sample_ids if "-" in s}
    if len(prefixes) > 1:
        return ["sample_prefix"]
    return []


def detect_sdrf_batch_fields(sdrf_path: str, sample_ids: list[str]) -> list[str]:
    """Return varying SDRF columns explicitly identified as batch fields."""
    sdrf = load_sdrf(sdrf_path)
    selected = {str(sample) for sample in sample_ids}
    matched = sdrf["source name"].astype(str).isin(selected)
    data_file = "comment[data file]"
    if data_file in sdrf.columns:
        raw_names = sdrf[data_file].astype(str)
        stems = raw_names.str.replace(
            r"\.(raw|mzml|mzxml|mgf|wiff|wiff2|wiff\.scan|baf|yep|fid|d|d\.zip|raw\.zip)$",
            "",
            regex=True,
            flags=re.IGNORECASE,
        )
        selected_upper = {sample.upper() for sample in selected}
        matched |= raw_names.isin(selected) | stems.isin(selected)
        matched |= raw_names.str.upper().isin(selected_upper)
        matched |= stems.str.upper().isin(selected_upper)

    scoped = sdrf.loc[matched]
    fields = []
    for column in scoped.columns:
        if "batch" not in column.casefold():
            continue
        values = scoped[column].dropna().astype(str).str.strip()
        values = values[
            ~values.str.casefold().isin({"", "not available", "not applicable"})
        ]
        if values.nunique() > 1:
            fields.append(column)
    return fields


def _condition_counts(
    sample_cols: list[str],
    sample_to_condition: dict[str, str],
) -> dict[str, int]:
    """Count matrix columns assigned to each condition."""
    counts: dict[str, int] = {}
    for sample in sample_cols:
        condition = sample_to_condition.get(sample, "unknown")
        counts[condition] = counts.get(condition, 0) + 1
    return counts


def _missingness(matrix: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Return overall and per-sample missingness rates."""
    missing_count = int(matrix.isna().sum().sum())
    overall = missing_count / matrix.size if matrix.size else 0.0
    per_sample = {column: float(matrix[column].isna().mean()) for column in matrix}
    return overall, per_sample


def _intensity_range(values: np.ndarray) -> tuple[float, float]:
    """Return the finite intensity range or zeros for an empty matrix."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def profile_data(
    protein_df: pd.DataFrame,
    sample_to_condition: dict[str, str],
    peptide_counts: pd.Series | None = None,
    *,
    input_scale: str,
    metadata: AcquisitionMetadata | None = None,
) -> DataProfile:
    """Profile a protein intensity matrix for agentic method selection."""
    if input_scale not in {"linear", "log2"}:
        raise ValueError("input_scale must be linear or log2")
    metadata = metadata or AcquisitionMetadata()
    protein_col = protein_df.columns[0]
    sample_cols = [c for c in protein_df.columns if c != protein_col]
    matrix = protein_df.set_index(protein_col)[sample_cols]

    cond_counts = _condition_counts(sample_cols, sample_to_condition)
    missing_rate, missing_per_sample = _missingness(matrix)
    is_log = input_scale == "log2"
    cv_matrix = _linear_cv_matrix(matrix, input_scale)
    resolved_data_type = (
        metadata.data_type.strip().upper()
        if metadata.data_type is not None
        else _detect_data_type(sample_to_condition)
    )
    return DataProfile(
        n_proteins=len(matrix),
        n_samples=len(sample_cols),
        n_conditions=len(cond_counts),
        samples_per_condition=cond_counts,
        missing_rate=missing_rate,
        missing_per_sample=missing_per_sample,
        median_cv=compute_median_cv(cv_matrix, sample_to_condition),
        has_peptide_counts=peptide_counts is not None,
        data_type=resolved_data_type,
        batch_fields=_detect_batch_fields(sample_cols),
        intensity_range=_intensity_range(matrix.values.astype(float)),
        is_log_transformed=is_log,
        pca_explained_variance=_compute_pca_evr(matrix),
        per_condition_median_cv=_compute_per_condition_cv(
            cv_matrix, sample_to_condition
        ),
        pairwise_median_abs_log2fc=_compute_pairwise_log2fc(
            matrix, sample_to_condition, is_log
        ),
        data_type_source=("declared" if metadata.data_type is not None else "inferred"),
        quantification=(
            metadata.quantification.strip().lower() if metadata.quantification else None
        ),
        upstream_engine=_canonicalize_upstream_engine(metadata.upstream_engine),
    )
