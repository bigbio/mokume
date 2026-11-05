"""
Multi-method ensemble for differential expression analysis.

Runs multiple DE methods on the same data and combines results
using a top-k consensus strategy: a protein is deemed significant
only when at least *k* methods agree on its direction.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from mokume.analysis.differential_expression import DifferentialExpression
from mokume.analysis.ensemble_config import (
    DEFAULT_ENSEMBLE_METHODS,
    resolve_ensemble_methods,
    resolve_combined_results,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.analysis.ensemble")


def run_ensemble(
    protein_df: pd.DataFrame,
    sample_to_condition: dict[str, str],
    contrast: tuple[str, str],
    methods: Sequence[str] | None = DEFAULT_ENSEMBLE_METHODS,
    min_k: int = 2,
    fdr_method: str = "bh",
    fdr_threshold: float = 0.05,
    log2fc_threshold: float | str = 0.5,
    peptide_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Run multiple DE methods and combine with top-k consensus.

    Parameters
    ----------
    protein_df : pd.DataFrame
        Protein intensity matrix (first column = protein IDs).
    sample_to_condition : dict[str, str]
        Sample-to-condition mapping.
    contrast : tuple[str, str]
        Condition pair (A, B) for comparison.
    methods : sequence of str or None
        DE methods to run. ``None`` uses limrots, deqms, and proda.
    min_k : int
        Minimum number of methods that must agree on significance
        for a protein to be called DE (default 2).
    fdr_method : str
        FDR correction method ("bh", "bky", "storey"). Applied both by each
        sub-method and by the ensemble combination layer.
    fdr_threshold : float
        FDR threshold for each sub-method and for the combined results.
    log2fc_threshold : float or "auto"
        Log2 fold-change threshold. ``"auto"`` estimates the gate from the data
        (each member from its own fold changes, the ensemble from the median).
    peptide_counts : pd.Series or None
        Peptide counts (passed to DEqMS if included).

    Returns
    -------
    pd.DataFrame
        Ensemble DE results with columns: ProteinName, log2FC (median),
        pvalue (Cauchy combined), adj_pvalue, significance,
        n_methods_up, n_methods_down, methods_significant.

    Raises
    ------
    ValueError
        If the ensemble configuration or a shared member input is invalid.
    KeyError
        If a required shared input field is missing.
    """
    resolved_methods = resolve_ensemble_methods(methods, min_k)
    individual_results: dict[str, pd.DataFrame] = {}
    for method in resolved_methods:
        try:
            de = DifferentialExpression(
                method=method,
                fdr_method=fdr_method,
                fdr_threshold=fdr_threshold,
                log2fc_threshold=log2fc_threshold,
                peptide_counts=peptide_counts,
            )
            result = de.run(protein_df, sample_to_condition, contrast)
            if not result.empty:
                individual_results[method] = result
                logger.info("Ensemble member %s: %d proteins", method, len(result))
        except (RuntimeError, ArithmeticError) as exc:
            logger.warning("Ensemble member %s failed: %s", method, exc)

    if not individual_results:
        logger.warning("No ensemble members produced results")
        return pd.DataFrame()

    return combine_de_results(
        individual_results,
        min_k=min_k,
        log2fc_threshold=log2fc_threshold,
        fdr_method=fdr_method,
        fdr_threshold=fdr_threshold,
    )


def _resolve_log2fc_gate(log2fc: np.ndarray, threshold: float | str) -> float:
    """Return the ensemble effect-size gate: a fixed float, or "auto" from data.

    Mirrors ``DifferentialExpression._resolve_log2fc_gate``, but estimates the
    gate from the ensemble's median fold changes so that the combination layer
    re-classifies on the same scale it filters. A fixed gate calibrated for
    label-free is too strict for compressed isobaric (TMT) ratios; ``"auto"``
    lets the gate track the data.
    """
    if isinstance(threshold, str) and threshold.lower() == "auto":
        from mokume.analysis.effect_size_gate import (  # pylint: disable=import-outside-toplevel
            estimate_effect_size_gate,
        )

        gate = estimate_effect_size_gate(log2fc)
        logger.info("Ensemble auto effect-size gate estimated from data: %.3f", gate)
        return gate
    return float(threshold)


def _combine_member_pvalues(p_values: list[float]) -> float:
    """Combine member p-values with the equal-weight Cauchy combination.

    The members test the same contrast on the same matrix, so their p-values
    are strongly dependent. Fisher's method assumes independence and gives
    p-values that are too small; the Cauchy combination (Liu & Xie 2020) stays
    valid under arbitrary dependence for small p-values.
    """
    if len(p_values) >= 2:
        # cot(p pi) avoids cancellation in 0.5 - p for tiny p-values.
        terms = [
            1.0 / math.tan(max(p, np.finfo(float).tiny) * math.pi) for p in p_values
        ]
        # Upper tail of the standard Cauchy, 0.5 - atan(t) / pi, kept precise
        # for large t.
        return math.atan2(1.0, sum(terms) / len(terms)) / math.pi
    if p_values:
        return float(p_values[0])
    return np.nan


def _combined_protein_row(
    protein: str,
    results: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Combine member evidence for one protein before global FDR correction."""
    fc_values: list[float] = []
    p_values: list[float] = []
    n_up = 0
    n_down = 0
    contributing: list[str] = []

    for method, frame in results.items():
        match = frame[frame["ProteinName"] == protein]
        if match.empty:
            continue
        row = match.iloc[0]
        significance = row.get("significance", "Unchanged")
        fold_change = row["log2FC"]
        if significance == "NotTested" or not np.isfinite(fold_change):
            continue
        fc_values.append(fold_change)
        pvalue = row["pvalue"]
        if np.isfinite(pvalue) and 0 <= pvalue <= 1:
            p_values.append(pvalue)
        if significance == "UP":
            n_up += 1
            contributing.append(method)
        elif significance == "DOWN":
            n_down += 1
            contributing.append(method)

    return {
        "ProteinName": protein,
        "log2FC": float(np.median(fc_values)) if fc_values else np.nan,
        "pvalue": _combine_member_pvalues(p_values),
        "n_methods_up": n_up,
        "n_methods_down": n_down,
        "methods_significant": ",".join(contributing),
    }


def _adjust_ensemble_pvalues(
    ensemble: pd.DataFrame,
    fdr_method: str,
    fdr_threshold: float,
) -> np.ndarray:
    """Apply the requested FDR method to finite combined p-values."""
    valid = np.isfinite(ensemble["pvalue"].values)
    adjusted_pvalues = np.full(len(ensemble), np.nan)
    if not valid.any():
        return adjusted_pvalues

    from mokume.analysis.adaptive_fdr import (  # pylint: disable=import-outside-toplevel
        adjust_pvalues,
    )

    adjusted, method_used = adjust_pvalues(
        ensemble.loc[valid, "pvalue"].values,
        method=fdr_method,
        alpha=fdr_threshold,
    )
    adjusted_pvalues[valid] = adjusted
    if method_used != fdr_method:
        logger.info("Ensemble FDR: requested %s, applied %s", fdr_method, method_used)
    return adjusted_pvalues


def combine_de_results(
    results: dict[str, pd.DataFrame],
    min_k: int = 2,
    log2fc_threshold: float | str = 0.5,
    fdr_method: str = "bh",
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Combine DE results from multiple methods using top-k consensus.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping of method name to its DE result DataFrame.
        Each must have columns: ProteinName, log2FC, pvalue, adj_pvalue,
        significance.
    min_k : int
        Minimum agreement count.
    log2fc_threshold : float or "auto"
        Log2 fold-change threshold for re-classification. ``"auto"`` estimates
        the gate from the ensemble's median fold changes.
    fdr_method : str
        FDR correction applied to the Cauchy-combined p-values ("bh", "bky",
        "storey"). Adaptive procedures fall back to BH when pi0 is unreliable.
    fdr_threshold : float
        FDR threshold for re-classification.

    Returns
    -------
    pd.DataFrame
        Combined results.

    Raises
    ------
    ValueError
        If ``min_k`` or a result label is invalid or ambiguous.
    """
    results = resolve_combined_results(results, min_k)
    all_proteins: set[str] = set()
    for df in results.values():
        all_proteins.update(df["ProteinName"].tolist())

    rows = [_combined_protein_row(protein, results) for protein in sorted(all_proteins)]

    ensemble = pd.DataFrame(rows)
    if ensemble.empty:
        return ensemble

    # FDR correction on the Cauchy-combined p-values. The requested method must
    # reach the combination layer too: correcting the members adaptively but the
    # ensemble with BH would silently discard the adaptive-pi0 gain here.
    ensemble["adj_pvalue"] = _adjust_ensemble_pvalues(
        ensemble, fdr_method, fdr_threshold
    )

    gate = _resolve_log2fc_gate(
        ensemble["log2FC"].to_numpy(dtype=float), log2fc_threshold
    )

    tested = (
        np.isfinite(ensemble["pvalue"].to_numpy(dtype=float))
        & np.isfinite(ensemble["adj_pvalue"].to_numpy(dtype=float))
        & np.isfinite(ensemble["log2FC"].to_numpy(dtype=float))
    )

    # Consensus significance
    ensemble["significance"] = "NotTested"
    ensemble.loc[tested, "significance"] = "Unchanged"
    ensemble.loc[
        tested
        & (ensemble["n_methods_up"] >= min_k)
        & (ensemble["adj_pvalue"] < fdr_threshold)
        & (ensemble["log2FC"] > gate),
        "significance",
    ] = "UP"
    ensemble.loc[
        tested
        & (ensemble["n_methods_down"] >= min_k)
        & (ensemble["adj_pvalue"] < fdr_threshold)
        & (ensemble["log2FC"] < -gate),
        "significance",
    ] = "DOWN"

    ensemble = ensemble.sort_values("adj_pvalue").reset_index(drop=True)

    significance = ensemble["significance"]
    logger.info(
        "Ensemble results: %d proteins, %d NotTested, %d UP, %d DOWN (k>=%d)",
        len(ensemble),
        (significance == "NotTested").sum(),
        (significance == "UP").sum(),
        (significance == "DOWN").sum(),
        min_k,
    )
    return ensemble
