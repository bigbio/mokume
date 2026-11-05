"""
DEqMS-style differential expression analysis (pure Python).

Reimplements the DEqMS pipeline: limma (lmFit → contrasts.fit → eBayes) plus
spectraCounteBayes variance moderation, without rpy2 or R dependencies.

Reference
---------
Zhu Y, Orre LM, Zhou Tran Y, et al. DEqMS: a method for accurate variance
estimation in differential protein expression analysis. *Mol Cell Proteomics*.
2020;19(6):1047-1057.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy.stats import t as t_dist

from mokume.analysis._deqms_loess import loess_predict
from mokume.analysis._helpers import (
    bh_adjust,
    filter_testable,
    per_group_summary,
)
from mokume.analysis.limma import (
    _contrasts_fit,
    _ebayes,
    _lm_fit_with_na,
    EBayesResult,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.analysis.deqms")

# The quadratic count fit needs floor(.75*n) >= 3 valid proteins.
MIN_VALID_POINTS = 4


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    tiny = 1e-300
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, 301):
        m = float(iteration)
        m2 = 2.0 * m
        term = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        term = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return result


def _log_regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute ``log(I_x(a, b))`` without underflow."""
    if x <= 0.0:
        return -math.inf
    if x >= 1.0:
        return 0.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        fraction = _beta_continued_fraction(a, b, x)
        return log_front + math.log(fraction) - math.log(a)

    fraction = _beta_continued_fraction(b, a, 1.0 - x)
    complement = min(math.exp(log_front) * fraction / b, 1.0)
    return math.log1p(-complement)


def _two_sided_t_log_pvalue(
    t_statistic: np.ndarray,
    degrees_freedom: np.ndarray,
) -> np.ndarray:
    """Return two-sided Student-t log p-values without float underflow."""
    t_abs, df = np.broadcast_arrays(
        np.abs(np.asarray(t_statistic, dtype=float)),
        np.asarray(degrees_freedom, dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(2.0) + t_dist.logsf(t_abs, df)

    flat_result = result.ravel()
    flat_t = t_abs.ravel()
    flat_df = df.ravel()
    underflow = np.isneginf(flat_result) & np.isfinite(flat_t) & (flat_df > 0.0)
    for index in np.flatnonzero(underflow):
        x = flat_df[index] / (flat_df[index] + flat_t[index] ** 2)
        flat_result[index] = _log_regularized_incomplete_beta(
            flat_df[index] / 2.0,
            0.5,
            x,
        )
    return np.minimum(result, 0.0)


# ---------------------------------------------------------------------------
# spectraCounteBayes: count-aware variance moderation (Zhu et al. 2020)
# ---------------------------------------------------------------------------


def _spectra_count_ebayes(
    fit: EBayesResult,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply spectra-count-aware Bayes moderation (R DEqMS::spectraCounteBayes).

    The log variance follows R's Gaussian quadratic LOESS on log2 counts, as
    the Rust ``spectra_count_ebayes``. Returns (sca_t, sca_pvalue, log_pvalue,
    sca_var_post, d0).
    """
    sigma2 = fit.sigma**2
    log_var = np.log(sigma2)
    df = fit.df_residual.copy()
    n_genes = len(log_var)

    df_valid = df.copy()
    df_valid[df_valid == 0] = np.nan

    counts_f = counts.astype(float).copy()
    if np.nanmin(counts_f) == 0:
        counts_f = counts_f + 1

    x = np.log2(counts_f)

    valid = np.isfinite(log_var) & np.isfinite(x) & np.isfinite(df_valid)
    if valid.sum() < MIN_VALID_POINTS:
        logger.warning(
            "spectraCounteBayes: too few valid points (%d), "
            "falling back to standard eBayes",
            valid.sum(),
        )
        return _standard_ebayes_statistics(fit)

    if np.ptp(x[valid]) < 1e-10:
        logger.warning(
            "spectraCounteBayes: all counts identical, falling back to standard eBayes",
        )
        return _standard_ebayes_statistics(fit)

    y_pred = loess_predict(log_var, x, valid)
    if not np.isfinite(y_pred[valid]).all():
        logger.warning(
            "spectraCounteBayes: non-finite count trend (tied counts), "
            "falling back to standard eBayes",
        )
        return _standard_ebayes_statistics(fit)
    eg = log_var - digamma(df_valid / 2) + np.log(df_valid / 2)
    egpred = y_pred - digamma(df_valid / 2) + np.log(df_valid / 2)

    myfct = (eg - egpred) ** 2 - polygamma(1, df_valid / 2)
    d0 = _grid_search_d0(float(np.mean(myfct[np.isfinite(myfct)])), n_genes)

    s02 = np.exp(egpred + digamma(d0 / 2) - np.log(d0 / 2))

    post_var = (d0 * s02 + df * sigma2) / (d0 + df)
    post_df = d0 + df

    sca_t = fit.coefficients[:, 0] / (fit.stdev_unscaled[:, 0] * np.sqrt(post_var))
    sca_pvalue = 2.0 * t_dist.sf(np.abs(sca_t), post_df)
    log_pvalue = _two_sided_t_log_pvalue(sca_t, post_df)

    return sca_t, sca_pvalue, log_pvalue, post_var, d0


def _standard_ebayes_statistics(
    fit: EBayesResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the plain eBayes fallback with stable log p-values."""
    df_total = np.minimum(
        fit.df_residual + fit.df_prior,
        np.nansum(fit.df_residual),
    )
    return (
        fit.t_stat[:, 0],
        fit.p_value[:, 0],
        _two_sided_t_log_pvalue(fit.t_stat[:, 0], df_total),
        fit.s2_post,
        fit.df_prior,
    )


def _grid_search_d0(mean_myfct: float, n_genes: int) -> float:
    """Grid search for prior df d0 (R DEqMS internal algorithm)."""
    max_iter = n_genes * 10
    best_d0 = 0.1
    best_diff = np.inf
    prev_prev = np.inf
    prev = np.inf
    for i in range(1, max_iter + 1):
        test_d0 = i / 10.0
        diff = abs(mean_myfct - float(polygamma(1, test_d0 / 2)))
        if diff < best_diff:
            best_diff = diff
            best_d0 = test_d0
        if i > 2 and prev_prev < prev:
            break
        prev_prev = prev
        prev = diff
    return best_d0


# ---------------------------------------------------------------------------
# Count vector builder (shared logic with deqms.py)
# ---------------------------------------------------------------------------


def _build_count_vector(
    proteins: list[str],
    peptide_counts: pd.Series | None,
) -> np.ndarray:
    """Align peptide counts with the matrix row order; default to 1.

    Notes
    -----
    DEqMS's defining feature is the PSM/peptide-count covariate fed into
    the spectra-count empirical-Bayes adjustment. If callers do not pass
    ``peptide_counts`` (or no protein matches), every protein gets a
    count of ``1`` and ``_spectra_count_ebayes`` degenerates to limma
    plus uniform variance shrinkage - the DEqMS result becomes
    effectively identical to limma. We emit a single warning so users
    aware of this trade-off can choose to provide counts, while still
    letting the call succeed (matching the original R DEqMS behaviour
    when only intensity data is available).
    """
    counts = np.ones(len(proteins), dtype=int)
    if peptide_counts is None:
        logger.warning(
            "DEqMS: no peptide_counts provided; falling back to count=1 per "
            "protein. Results will be equivalent to limma + uniform EBayes. "
            "Pass peptide_counts (unique peptides per protein) to enable the "
            "spectra-count ebayes adjustment."
        )
        return counts
    aligned = peptide_counts.reindex(proteins)
    valid = aligned.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        logger.warning(
            "DEqMS: peptide_counts had zero overlap with the protein matrix "
            "(checked %d proteins); falling back to count=1. Make sure the "
            "Series index uses the same protein accessions as the matrix "
            "(e.g. anchor_protein/UniProt IDs).",
            len(proteins),
        )
    elif n_valid < len(proteins):
        logger.info(
            "DEqMS: peptide_counts covered %d/%d proteins; missing entries "
            "default to count=1.",
            n_valid,
            len(proteins),
        )
    counts[valid.values] = aligned[valid].astype(int).values
    return counts


def _precision_weights(values: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Observation weights from each cell's quantification precision (1/SE^2).

    A cell without a precision (zero or NaN, e.g. an imputed value) takes the
    5th percentile of the observed precisions. Each weight is then floored as
    ``1/(1/w + tau2)``, ``tau2`` being the median observed ``1/w``, so the
    biological variance bounds how much one well-measured cell can count.
    Mirrors the Rust ``deqms_two_group_weighted``.
    """
    measured = np.isfinite(values) & np.isfinite(precision) & (precision > 0)
    observed = precision[measured]
    if not observed.size:
        logger.warning(
            "DEqMS: no finite positive precision overlaps observed intensities; "
            "using unit weights. Check protein and sample alignment."
        )
    tau2 = float(np.median(1.0 / observed)) if observed.size else 0.0
    unmeasured = float(np.quantile(observed, 0.05)) if observed.size else 1.0
    return 1.0 / (1.0 / np.where(measured, precision, unmeasured) + tau2)


# ---------------------------------------------------------------------------
# Public API: run_deqms
# ---------------------------------------------------------------------------


def run_deqms(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    contrast: tuple[str, str],
    **options,
) -> pd.DataFrame:
    """Run DEqMS differential expression (pure Python implementation).

    ``precision``, a protein-by-sample frame of quantification precisions,
    weights each cell by :func:`_precision_weights`; the weighted fit reports
    weighted group means.
    """
    peptide_counts = options.pop("peptide_counts", None)
    precision = options.pop("precision", None)
    if options:
        unknown = ", ".join(sorted(options))
        raise TypeError(f"Unexpected DEqMS options: {unknown}")

    cond_a, cond_b = contrast
    sub_matrix = filter_testable(log2_matrix, samples_a, samples_b)
    if sub_matrix.empty:
        logger.warning("No proteins passed DE filtering for DEqMS")
        return pd.DataFrame()

    sample_order = list(samples_a) + list(samples_b)
    mat = sub_matrix[sample_order].values
    gene_names = list(sub_matrix.index)
    weights = None
    if precision is not None:
        # Fill and floor over every row of the contrast, as the Rust DEqMS.
        weights = _precision_weights(
            log2_matrix[sample_order].to_numpy(dtype=float),
            precision.reindex(index=log2_matrix.index, columns=sample_order).to_numpy(
                dtype=float
            ),
        )[log2_matrix.index.get_indexer(sub_matrix.index)]

    design = np.zeros((len(sample_order), 2))
    design[: len(samples_a), 0] = 1.0
    design[len(samples_a) :, 1] = 1.0
    coef_names = [cond_a, cond_b]

    fit = _lm_fit_with_na(mat, design, gene_names, coef_names, weights)
    contrasts = np.array([[1.0], [-1.0]])
    fit_c = _contrasts_fit(fit, contrasts)
    fit_eb = _ebayes(fit_c)

    counts = _build_count_vector(gene_names, peptide_counts)

    sca_t, sca_pvalue, log_pvalue, _, _ = _spectra_count_ebayes(fit_eb, counts)

    raw = pd.DataFrame(
        {
            "ProteinName": gene_names,
            "log2FC": fit_eb.coefficients[:, 0],
            "sca_t": sca_t,
            "sca_pvalue": sca_pvalue,
            "peptide_count": counts,
            "log_pvalue": log_pvalue,
        }
    )

    summary = per_group_summary(sub_matrix, samples_a, samples_b, cond_a, cond_b)
    merged = raw.merge(summary, on="ProteinName", how="left")
    if weights is not None:
        merged[f"mean_{cond_a}"] = fit.coefficients[:, 0]
        merged[f"mean_{cond_b}"] = fit.coefficients[:, 1]
    return _ordered_result(merged, cond_a, cond_b)


def _ordered_result(merged, cond_a, cond_b):
    """Add the BH-adjusted p-values and order the DEqMS columns and rows."""
    merged["pvalue"] = merged["sca_pvalue"]
    merged["adj_pvalue"] = bh_adjust(merged["sca_pvalue"].values)
    merged["sca_adj_pvalue"] = merged["adj_pvalue"]

    columns = [
        "ProteinName",
        "log2FC",
        "pvalue",
        "adj_pvalue",
        "sca_t",
        "sca_pvalue",
        "sca_adj_pvalue",
        f"mean_{cond_a}",
        f"mean_{cond_b}",
        "n_a",
        "n_b",
        "peptide_count",
        "log_pvalue",
    ]
    available = [c for c in columns if c in merged.columns]
    return merged[available].sort_values("adj_pvalue").reset_index(drop=True)
