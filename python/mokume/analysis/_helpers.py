"""
Pure-Python helper utilities for differential expression workflows.

Provides BH adjustment, per-group summary, testability filtering, and
result finalisation helpers used by the various DE methods.
"""

from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "bh_adjust",
    "filter_testable",
    "finalize_de_result",
    "load_multipletests",
    "per_group_summary",
]


def load_multipletests():
    """Lazily load statsmodels' multiple-testing implementation."""
    try:
        module = importlib.import_module("statsmodels.stats.multitest")
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for differential-expression FDR correction. "
            "Install it with: pip install mokume-py[analysis]"
        ) from exc
    return module.multipletests


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg correction, tolerating NaN/Inf p-values."""
    valid = np.isfinite(pvalues)
    adj = np.full_like(pvalues, np.nan, dtype=float)
    if valid.any():
        adj[valid] = load_multipletests()(pvalues[valid], method="fdr_bh")[1]
    return adj


def per_group_summary(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    cond_a: str,
    cond_b: str,
) -> pd.DataFrame:
    """Compute per-protein mean and observation counts for both groups."""
    sub_a = log2_matrix[samples_a]
    sub_b = log2_matrix[samples_b]
    finite_a = np.isfinite(sub_a)
    finite_b = np.isfinite(sub_b)
    return pd.DataFrame(
        {
            "ProteinName": log2_matrix.index,
            f"mean_{cond_a}": sub_a.where(finite_a).mean(axis=1).values,
            f"mean_{cond_b}": sub_b.where(finite_b).mean(axis=1).values,
            "n_a": finite_a.sum(axis=1).values,
            "n_b": finite_b.sum(axis=1).values,
        }
    )


def filter_testable(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Drop proteins lacking enough finite observations in both groups."""
    finite_matrix = log2_matrix.where(np.isfinite(log2_matrix))
    nobs_a = finite_matrix[samples_a].notna().sum(axis=1)
    nobs_b = finite_matrix[samples_b].notna().sum(axis=1)
    keep = (nobs_a >= min_per_group) & (nobs_b >= min_per_group)
    return finite_matrix.loc[keep]


def finalize_de_result(
    raw: pd.DataFrame,
    sub_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    cond_a: str,
    cond_b: str,
    extra_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Merge group summary, BH-adjust, select columns, and sort.

    A method that brings its own ``adj_pvalue`` (the ROTS permutation FDR)
    keeps it instead of a BH adjustment.
    """
    summary = per_group_summary(sub_matrix, samples_a, samples_b, cond_a, cond_b)
    merged = raw.merge(summary, on="ProteinName", how="left")
    if "adj_pvalue" not in merged.columns:
        merged["adj_pvalue"] = bh_adjust(merged["pvalue"].values)
    columns = [
        "ProteinName",
        "log2FC",
        "pvalue",
        "adj_pvalue",
        *extra_cols,
        f"mean_{cond_a}",
        f"mean_{cond_b}",
        "n_a",
        "n_b",
    ]
    available = [c for c in columns if c in merged.columns]
    return merged[available].sort_values("adj_pvalue").reset_index(drop=True)
