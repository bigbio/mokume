"""
LimROTS differential expression analysis (pure Python).

Reimplements the LimROTS hybrid method: limma empirical Bayes posterior
variance combined with ROTS-style bootstrap optimization of the moderated
d-statistic, without rpy2 or R dependencies.

Reference
---------
Anwar AM, Jeba A, Lahti L, Coffey E. LimROTS: a hybrid method integrating
empirical Bayes and reproducibility-optimized statistics for robust
differential expression analysis. *Bioinformatics*. 2025.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mokume.analysis._helpers import (
    filter_testable,
    per_group_summary,
)
from mokume.analysis.limma import (
    _contrasts_fit,
    _ebayes,
    _lm_fit_with_na,
)
from mokume.analysis.rots import (
    _calculate_fdr,
    _rank_abs,
    _reproducibility_score,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.analysis.limrots")


_LIMROTS_OPTION_DEFAULTS = {
    "n_boot": 100,
    "seed": 42,
    "robust": True,
    "trend": True,
}


# ---------------------------------------------------------------------------
# LimROTS core: eBayes posterior variance + ROTS bootstrap
# ---------------------------------------------------------------------------


def _limrots_d_stat(
    beta: np.ndarray,
    s_post: np.ndarray,
    u: np.ndarray,
    a1: float,
    a2: float,
) -> np.ndarray:
    """Compute the LimROTS moderated d-statistic.

    d_alpha(i) = beta_i / (a1 + a2 * s_post_i * u)
    """
    denom = a1 + a2 * s_post * u
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return beta / denom


def _limrots_bootstrap_optimize(
    mat: np.ndarray,
    design: np.ndarray,
    contrasts: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    gene_names: list[str],
    coef_names: list[str],
) -> tuple[float, float]:
    """Find optimal (a1, a2) by bootstrap reproducibility."""
    n_a = int(design[:, 0].sum())

    a1_grid = np.concatenate(
        [
            np.arange(0, 21) / 100.0,
            np.arange(11, 51) / 50.0,
            np.arange(6, 26) / 5.0,
        ]
    )
    a2_grid = np.ones_like(a1_grid)

    n_params = len(a1_grid)
    k_max = max(1, mat.shape[0] // 4)
    k_grid = np.concatenate(
        [
            np.arange(1, 21) * 5,
            np.arange(11, 51) * 10,
            np.arange(21, 41) * 25,
            np.arange(11, 1001) * 100,
        ]
    )
    k_values = k_grid[k_grid < k_max]
    if len(k_values) == 0:
        k_values = np.array([max(1, k_max)])
    n_k = len(k_values)

    repro_real = np.zeros((n_boot, n_params, n_k))
    repro_perm = np.zeros((n_boot, n_params, n_k))

    n_total = design.shape[0]
    n_b = n_total - n_a

    for b in range(n_boot):
        idx1 = np.concatenate(
            [
                rng.choice(n_a, n_a, replace=True),
                rng.choice(n_b, n_b, replace=True) + n_a,
            ]
        )
        idx2 = np.concatenate(
            [
                rng.choice(n_a, n_a, replace=True),
                rng.choice(n_b, n_b, replace=True) + n_a,
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b1, s1, u1 = _boot_ebayes(
                mat, design, contrasts, idx1, gene_names, coef_names
            )
            b2, s2, u2 = _boot_ebayes(
                mat, design, contrasts, idx2, gene_names, coef_names
            )

        perm_idx = rng.permutation(n_total)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bp, sp, up = _boot_ebayes(
                mat,
                design[perm_idx],
                contrasts,
                np.arange(n_total),
                gene_names,
                coef_names,
            )

        for pi in range(n_params):
            d1 = _limrots_d_stat(b1, s1, u1, a1_grid[pi], a2_grid[pi])
            d2 = _limrots_d_stat(b2, s2, u2, a1_grid[pi], a2_grid[pi])
            dp = _limrots_d_stat(bp, sp, up, a1_grid[pi], a2_grid[pi])

            r1 = _rank_abs(d1)
            r2 = _rank_abs(d2)
            rp = _rank_abs(dp)

            for ki, k in enumerate(k_values):
                repro_real[b, pi, ki] = _reproducibility_score(r1, r2, k)
                repro_perm[b, pi, ki] = _reproducibility_score(r1, rp, k)

    mean_real = np.mean(repro_real, axis=0)
    mean_perm = np.mean(repro_perm, axis=0)
    std_real = np.std(repro_real, axis=0, ddof=1)
    std_real = np.where(std_real < 1e-10, 1e-10, std_real)

    z_scores = (mean_real - mean_perm) / std_real
    best_idx = np.unravel_index(np.nanargmax(z_scores), z_scores.shape)
    return float(a1_grid[best_idx[0]]), float(a2_grid[best_idx[0]])


def _boot_ebayes(
    mat: np.ndarray,
    design: np.ndarray,
    contrasts: np.ndarray,
    sample_idx: np.ndarray,
    gene_names: list[str],
    coef_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run lmFit + eBayes on a bootstrap sample, return (beta, s_post, u)."""
    boot_mat = mat[:, sample_idx]
    boot_design = design[sample_idx]

    fit = _lm_fit_with_na(boot_mat, boot_design, gene_names, coef_names)
    fit_c = _contrasts_fit(fit, contrasts)
    fit_eb = _ebayes(fit_c)

    beta = fit_eb.coefficients[:, 0]
    s_post = np.sqrt(fit_eb.s2_post)
    u = fit_eb.stdev_unscaled[:, 0]
    return beta, s_post, u


# ---------------------------------------------------------------------------
# p-value from permutation null
# ---------------------------------------------------------------------------


def _pvalue_permutation(
    d_obs: np.ndarray,
    mat: np.ndarray,
    design: np.ndarray,
    contrasts: np.ndarray,
    a1: float,
    a2: float,
    n_perm: int,
    rng: np.random.Generator,
    gene_names: list[str],
    coef_names: list[str],
) -> np.ndarray:
    """Permutation p-values of the LimROTS d-statistic, and the null statistics.

    The null statistics hold one column per permutation for the native FDR.
    """
    n_genes = len(d_obs)
    abs_d = np.abs(d_obs)
    count = np.ones(n_genes)
    null = np.empty((n_genes, n_perm))

    for permutation in range(n_perm):
        perm_idx = rng.permutation(design.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bp, sp, up = _boot_ebayes(
                mat,
                design[perm_idx],
                contrasts,
                np.arange(design.shape[0]),
                gene_names,
                coef_names,
            )
        d_p = _limrots_d_stat(bp, sp, up, a1, a2)
        null[:, permutation] = d_p
        count += np.sum(np.abs(d_p)[:, np.newaxis] >= abs_d[np.newaxis, :], axis=0)

    pvalues = count / (n_genes * (n_perm + 1))
    return np.minimum(pvalues, 1.0), null


# ---------------------------------------------------------------------------
# Public API: run_limrots
# ---------------------------------------------------------------------------


def run_limrots(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    contrast: tuple[str, str],
    **options,
) -> pd.DataFrame:
    """Run LimROTS differential expression (pure Python implementation)."""
    unknown = set(options) - set(_LIMROTS_OPTION_DEFAULTS)
    if unknown:
        unknown_args = ", ".join(sorted(unknown))
        raise TypeError(f"Unexpected LimROTS options: {unknown_args}")
    settings = {**_LIMROTS_OPTION_DEFAULTS, **options}

    cond_a, cond_b = contrast
    sub_matrix = filter_testable(log2_matrix, samples_a, samples_b)
    if sub_matrix.empty:
        logger.warning("No proteins passed DE filtering for LimROTS")
        return pd.DataFrame()

    sample_order = list(samples_a) + list(samples_b)
    mat = sub_matrix[sample_order].values
    gene_names = list(sub_matrix.index)
    n_a = len(samples_a)

    design = np.zeros((len(sample_order), 2))
    design[:n_a, 0] = 1.0
    design[n_a:, 1] = 1.0
    coef_names = [cond_a, cond_b]
    contrasts_mat = np.array([[1.0], [-1.0]])

    rng = np.random.default_rng(settings["seed"])

    logger.info("LimROTS: %d bootstrap iterations", settings["n_boot"])
    best_a1, best_a2 = _limrots_bootstrap_optimize(
        mat,
        design,
        contrasts_mat,
        settings["n_boot"],
        rng,
        gene_names,
        coef_names,
    )
    logger.info("LimROTS optimal: a1=%.4f, a2=%.4f", best_a1, best_a2)

    fit = _lm_fit_with_na(mat, design, gene_names, coef_names)
    fit_c = _contrasts_fit(fit, contrasts_mat)
    fit_eb = _ebayes(fit_c)

    beta = fit_eb.coefficients[:, 0]
    s_post = np.sqrt(fit_eb.s2_post)
    u = fit_eb.stdev_unscaled[:, 0]
    d_stat = _limrots_d_stat(beta, s_post, u, best_a1, best_a2)

    n_perm = min(settings["n_boot"], 100)
    pvalues, null = _pvalue_permutation(
        d_stat,
        mat,
        design,
        contrasts_mat,
        best_a1,
        best_a2,
        n_perm,
        rng,
        gene_names,
        coef_names,
    )

    raw = pd.DataFrame(
        {
            "ProteinName": gene_names,
            "log2FC": beta,
            "t_stat": d_stat,
            "pvalue": pvalues,
        }
    )

    summary = per_group_summary(sub_matrix, samples_a, samples_b, cond_a, cond_b)
    merged = raw.merge(summary, on="ProteinName", how="left")
    merged["adj_pvalue"] = _calculate_fdr(merged["t_stat"].to_numpy(dtype=float), null)

    columns = [
        "ProteinName",
        "log2FC",
        "pvalue",
        "adj_pvalue",
        "t_stat",
        f"mean_{cond_a}",
        f"mean_{cond_b}",
        "n_a",
        "n_b",
    ]
    available = [c for c in columns if c in merged.columns]
    return merged[available].sort_values("adj_pvalue").reset_index(drop=True)
