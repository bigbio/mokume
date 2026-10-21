"""
Reproducibility-Optimised Test Statistic (ROTS) differential expression
(pure Python).

Reimplements the ROTS bootstrap optimization algorithm using numpy,
without rpy2 or R dependencies.

Reference
---------
Suomi T, Seyednasrollah F, Jaakkola MK, Faux T, Elo LL. ROTS: An R package
for reproducibility-optimized statistical testing. *PLoS Comput Biol*.
2017;13(5):e1005562.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mokume.analysis._helpers import (
    filter_testable,
    finalize_de_result,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.analysis.rots")


# ---------------------------------------------------------------------------
# Core ROTS algorithm
# ---------------------------------------------------------------------------


def _compute_d_stat(
    fc: np.ndarray,
    s: np.ndarray,
    a1: float,
    a2: float,
) -> np.ndarray:
    """Compute the ROTS d-statistic: d = fc / (a1 + a2 * s)."""
    denom = a1 + a2 * s
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return fc / denom


def _group_stats(
    mat: np.ndarray,
    n_a: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d and s matching R ROTS::testStatistic (unpaired, two-group)."""
    x = mat[:, :n_a]
    y = mat[:, n_a:]
    mx = np.nanmean(x, axis=1)
    my = np.nanmean(y, axis=1)
    d = my - mx

    sx = np.nansum((x - mx[:, np.newaxis]) ** 2, axis=1)
    sy = np.nansum((y - my[:, np.newaxis]) ** 2, axis=1)
    nx = np.sum(np.isfinite(x), axis=1)
    ny = np.sum(np.isfinite(y), axis=1)

    denom = nx + ny - 2
    denom = np.maximum(denom, 1)
    s = np.sqrt(
        ((sx + sy) / denom) * (1.0 / np.maximum(nx, 1) + 1.0 / np.maximum(ny, 1))
    )

    bad = (nx < 2) | (ny < 2)
    d[bad] = 0.0
    s[bad] = 1.0

    return d, s


def _reproducibility_score(
    ranks1: np.ndarray,
    ranks2: np.ndarray,
    k: int,
) -> float:
    """Compute overlap-based reproducibility for top-k features."""
    top1 = set(np.where(ranks1 <= k)[0])
    top2 = set(np.where(ranks2 <= k)[0])
    if k == 0:
        return 0.0
    return len(top1 & top2) / k


def _build_ssq_grid() -> np.ndarray:
    """Build R ROTS ssq parameter grid: c((0:20)/100, (11:50)/50, (6:25)/5)."""
    return np.concatenate(
        [
            np.arange(0, 21) / 100.0,
            np.arange(11, 51) / 50.0,
            np.arange(6, 26) / 5.0,
        ]
    )


def _build_n_grid(k_max: int) -> np.ndarray:
    """Build R ROTS N (top-list) grid."""
    n = np.concatenate(
        [
            np.arange(1, 21) * 5,
            np.arange(11, 51) * 10,
            np.arange(21, 41) * 25,
            np.arange(11, 1001) * 100,
        ]
    )
    return n[n < k_max]


def _bootstrap_optimize(
    mat: np.ndarray,
    n_a: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Find optimal (a1, a2, k) matching R ROTS optimization."""
    n_genes = mat.shape[0]
    n_b = mat.shape[1] - n_a
    n_total = n_a + n_b
    k_max = n_genes // 4
    if k_max < 1:
        return 0.0, 1.0, 1

    ssq = _build_ssq_grid()
    n_grid = _build_n_grid(k_max)
    if len(n_grid) == 0:
        n_grid = np.array([1])
    n_ssq = len(ssq)
    n_k = len(n_grid)
    two_b = 2 * n_boot

    D = np.zeros((n_genes, two_b))
    S = np.zeros((n_genes, two_b))
    pD = np.zeros((n_genes, two_b))
    pS = np.zeros((n_genes, two_b))

    cl = np.concatenate([np.ones(n_a, dtype=int), np.full(n_b, 2, dtype=int)])

    for i in range(two_b):
        boot_idx = np.empty(n_total, dtype=int)
        for label in [1, 2]:
            pos = np.where(cl == label)[0]
            boot_idx[pos] = rng.choice(pos, len(pos), replace=True)
        boot_mat = mat[:, boot_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_b, s_b = _group_stats(boot_mat, n_a)
        D[:, i] = d_b
        S[:, i] = s_b

        perm_idx = rng.permutation(n_total)
        perm_mat = mat[:, perm_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_p, s_p = _group_stats(perm_mat, n_a)
        pD[:, i] = d_p
        pS[:, i] = s_p

    reprotable = np.zeros((n_ssq + 1, n_k))
    reprotable_p = np.zeros((n_ssq + 1, n_k))
    reprotable_sd = np.zeros((n_ssq + 1, n_k))

    for si in range(n_ssq):
        overlaps = np.zeros((n_boot, n_k))
        overlaps_p = np.zeros((n_boot, n_k))
        for b in range(n_boot):
            d1 = D[:, 2 * b] / (ssq[si] + S[:, 2 * b])
            d2 = D[:, 2 * b + 1] / (ssq[si] + S[:, 2 * b + 1])
            dp = pD[:, 2 * b] / (ssq[si] + pS[:, 2 * b])
            r1 = _rank_abs(d1)
            r2 = _rank_abs(d2)
            rp = _rank_abs(dp)
            for ki, k in enumerate(n_grid):
                overlaps[b, ki] = _reproducibility_score(r1, r2, k)
                overlaps_p[b, ki] = _reproducibility_score(r1, rp, k)
        reprotable[si] = np.mean(overlaps, axis=0)
        reprotable_p[si] = np.mean(overlaps_p, axis=0)
        sd = np.std(overlaps, axis=0, ddof=1)
        sd[sd < 1e-10] = 1e-10
        reprotable_sd[si] = sd

    si = n_ssq
    overlaps = np.zeros((n_boot, n_k))
    overlaps_p = np.zeros((n_boot, n_k))
    for b in range(n_boot):
        d1 = D[:, 2 * b]
        d2 = D[:, 2 * b + 1]
        dp = pD[:, 2 * b]
        r1 = _rank_abs(d1)
        r2 = _rank_abs(d2)
        rp = _rank_abs(dp)
        for ki, k in enumerate(n_grid):
            overlaps[b, ki] = _reproducibility_score(r1, r2, k)
            overlaps_p[b, ki] = _reproducibility_score(r1, rp, k)
    reprotable[si] = np.mean(overlaps, axis=0)
    reprotable_p[si] = np.mean(overlaps_p, axis=0)
    sd = np.std(overlaps, axis=0, ddof=1)
    sd[sd < 1e-10] = 1e-10
    reprotable_sd[si] = sd

    ztable = (reprotable - reprotable_p) / reprotable_sd
    zt_finite = ztable.copy()
    zt_finite[~np.isfinite(zt_finite)] = -np.inf
    best_idx = np.unravel_index(np.argmax(zt_finite), zt_finite.shape)
    best_si, best_ki = best_idx

    if best_si < n_ssq:
        best_a1 = float(ssq[best_si])
        best_a2 = 1.0
    else:
        best_a1 = 1.0
        best_a2 = 0.0
    best_k = int(n_grid[best_ki])

    return best_a1, best_a2, best_k


def _rank_abs(x: np.ndarray) -> np.ndarray:
    """Rank by absolute value (1 = largest)."""
    order = np.argsort(-np.abs(x))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def _calculate_p(
    observed: np.ndarray,
    permuted: np.ndarray,
) -> np.ndarray:
    """Compute p-values matching R ROTS::calculateP.

    Uses sorted global pool of all permuted |d| values.
    """
    n = len(observed)
    obs_order = np.argsort(-np.abs(observed))
    obs_sorted = np.abs(observed[obs_order])
    perm_flat = np.sort(np.abs(permuted.ravel()))[::-1]

    n_perm = len(perm_flat)
    p = np.zeros(n)
    j = 0
    for i in range(n):
        while j < n_perm and perm_flat[j] >= obs_sorted[i]:
            j += 1
        p[i] = j / n_perm

    result = np.zeros(n)
    for i in range(n):
        result[obs_order[i]] = p[i]
    return result


def _calculate_fdr(observed: np.ndarray, permuted: np.ndarray) -> np.ndarray:
    """Native permutation FDR matching R ROTS::calculateFDR.

    For each permutation (a column of ``permuted``), the statistic ranked k-th
    by ``|d|`` gets the number of permuted ``|d|`` above it (an equal value
    counts once) divided by k. The FDR is the median over permutations, capped
    at 1 and made monotone from the least extreme statistic upwards. Unlike a
    pooled estimate, the median is not floored by the permutations that
    reproduce the observed grouping, which are frequent in small designs.
    """
    fdr = np.full(len(observed), np.nan)
    finite = np.flatnonzero(~np.isnan(observed))
    if len(finite) == 0 or permuted.shape[1] == 0:
        return fdr
    order = finite[np.argsort(-np.abs(observed[finite]), kind="stable")]
    values = np.abs(observed[order])
    ranks = np.arange(1, len(order) + 1)
    ratios = np.empty((len(order), permuted.shape[1]))
    for column in range(permuted.shape[1]):
        null = np.abs(permuted[:, column])
        null = np.sort(null[~np.isnan(null)])
        above = len(null) - np.searchsorted(null, values, side="right")
        tied = np.isin(values, null)
        ratios[:, column] = (above + tied) / ranks
    median = np.minimum(np.median(ratios, axis=1), 1.0)
    fdr[order] = np.minimum.accumulate(median[::-1])[::-1]
    return fdr


# ---------------------------------------------------------------------------
# Public API: run_rots
# ---------------------------------------------------------------------------


def run_rots(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    cond_a: str,
    cond_b: str,
    n_boot: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run ROTS differential expression (pure Python implementation)."""
    sub_matrix = filter_testable(log2_matrix, samples_a, samples_b)
    if sub_matrix.empty:
        logger.warning("No proteins passed DE filtering for ROTS")
        return pd.DataFrame()

    sample_order = list(samples_a) + list(samples_b)
    mat = sub_matrix[sample_order].values
    gene_names = list(sub_matrix.index)
    n_a = len(samples_a)

    rng = np.random.default_rng(seed)

    logger.info("ROTS: %d bootstrap iterations", n_boot)
    best_a1, best_a2, best_k = _bootstrap_optimize(
        mat,
        n_a,
        n_boot,
        rng,
    )
    logger.info("ROTS optimal: a1=%.4f, a2=%.4f, k=%d", best_a1, best_a2, best_k)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_obs, s_obs = _group_stats(mat, n_a)
    d_stat = _compute_d_stat(d_obs, s_obs, best_a1, best_a2)

    n_total = mat.shape[1]
    two_b = 2 * n_boot
    pD_mat = np.zeros((len(gene_names), two_b))
    for i in range(two_b):
        perm_idx = rng.permutation(n_total)
        perm_mat = mat[:, perm_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dp, sp = _group_stats(perm_mat, n_a)
        pD_mat[:, i] = _compute_d_stat(dp, sp, best_a1, best_a2)

    pvalues = _calculate_p(d_stat, pD_mat)

    a_data = mat[:, :n_a]
    b_data = mat[:, n_a:]
    logfc = np.nanmean(a_data, axis=1) - np.nanmean(b_data, axis=1)

    raw = pd.DataFrame(
        {
            "ProteinName": gene_names,
            "log2FC": logfc,
            "d_stat": d_stat,
            "pvalue": pvalues,
            "adj_pvalue": _calculate_fdr(d_stat, pD_mat),
        }
    )

    return finalize_de_result(
        raw,
        sub_matrix,
        samples_a,
        samples_b,
        cond_a,
        cond_b,
        extra_cols=("d_stat",),
    )
