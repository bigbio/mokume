"""Enrichment classification and AnnData construction for TS scores."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Literature floor thresholds (Jiang et al. 2020, PMID 32916130)
FLOOR_ENRICHED: float = 2.5
FLOOR_SPECIFIC: float = 4.0
FLOOR_HOUSEKEEPING: float = 2.0


def _classify_enrichment(
    ts_matrix: np.ndarray,
    tissue_labels_all: np.ndarray,
    unique_tissues: list[str],
    log2_matrix: np.ndarray,
    ts_enriched: float = FLOOR_ENRICHED,
    ts_specific: float = FLOOR_SPECIFIC,
    ts_housekeeping: float = FLOOR_HOUSEKEEPING,
) -> list[str]:
    """Classify each protein into enrichment categories (vectorized)."""
    n_proteins = ts_matrix.shape[0]
    n_tissues = len(unique_tissues)

    # Pre-compute per-protein stats across tissues (all vectorized)
    n_valid = np.sum(~np.isnan(ts_matrix), axis=1)  # (n_proteins,)
    max_ts = np.nanmax(ts_matrix, axis=1)  # (n_proteins,)
    n_above_enriched = np.nansum(ts_matrix >= ts_enriched, axis=1)
    n_above_specific = np.nansum(ts_matrix >= ts_specific, axis=1)
    n_in_gap = np.nansum(
        (ts_matrix >= ts_enriched) & (ts_matrix < ts_specific),
        axis=1,
    )
    all_below_hk = np.all(
        np.isnan(ts_matrix) | (np.abs(ts_matrix) < ts_housekeeping),
        axis=1,
    )

    # Pre-compute per-tissue detection mask (vectorized)
    # detected[p, t] = True if protein p has at least one non-NaN in tissue t
    detected = np.zeros((n_proteins, n_tissues), dtype=bool)
    for t_idx, tissue in enumerate(unique_tissues):
        t_mask = tissue_labels_all == tissue
        tissue_data = log2_matrix[t_mask, :]  # (n_tissue_samples, n_proteins)
        detected[:, t_idx] = np.any(~np.isnan(tissue_data), axis=0)
    detected_all = detected.sum(axis=1) == n_tissues

    # Classify using vectorized boolean conditions
    cats = np.full(n_proteins, "other", dtype=object)

    # House-keeping: detected in all tissues AND all |TS| < hk AND enough tissues
    hk_mask = (n_valid >= 3) & detected_all & all_below_hk
    cats[hk_mask] = "house-keeping"

    # Tissue-enriched: at least one TS >= enriched (overrides hk)
    enriched_mask = (n_valid >= 3) & (n_above_enriched >= 1)
    cats[enriched_mask] = "tissue-enriched"

    # Tissue-specific: max >= specific, exactly 1 above specific, 0 in gap
    specific_mask = (
        (n_valid >= 3)
        & (max_ts >= ts_specific)
        & (n_above_specific == 1)
        & (n_in_gap == 0)
    )
    cats[specific_mask] = "tissue-specific"

    # Sparse proteins stay "other"
    cats[n_valid < 3] = "other"

    return cats.tolist()


def build_ts_anndata(
    ts_df: pd.DataFrame,
    unique_tissues: list[str],
) -> ad.AnnData:
    """Build an AnnData where obs = tissues, var = proteins, X = TS scores.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Output of :func:`compute_ts_scores`.
    unique_tissues : list[str]
        Tissue names (becomes obs index).

    Returns
    -------
    ad.AnnData
        Tissues x proteins AnnData with TS scores in X and enrichment
        metadata in ``var``.
    """
    tissue_cols = [c for c in ts_df.columns if c in unique_tissues]
    ts_matrix = ts_df[tissue_cols].values.T  # tissues x proteins

    var_df = ts_df[
        ["mu", "sigma", "pi", "enrichment_category", "max_tissue", "max_ts"]
    ].copy()
    var_df["anchor_protein"] = ts_df["anchor_protein"]
    var_df.index = ts_df.index
    var_df.index.name = "protein"

    obs_df = pd.DataFrame(index=tissue_cols)
    obs_df.index.name = "tissue"

    adata = ad.AnnData(
        X=ts_matrix.astype(np.float32),
        obs=obs_df,
        var=var_df,
    )
    return adata
