"""
Limma moderated t-test differential expression analysis (pure Python).

Reimplements the core limma pipeline (lmFit → contrasts.fit → eBayes →
topTable) using numpy/scipy, without rpy2 or R dependencies.

Reference
---------
Smyth GK. Linear models and empirical Bayes methods for assessing
differential expression in microarray experiments. *Stat Appl Genet
Mol Biol*. 2004;3:Article3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln, polygamma

from mokume.analysis._helpers import (
    filter_testable,
    finalize_de_result,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.analysis.limma")


# ---------------------------------------------------------------------------
# Data container for fitted linear model (mirrors R MArrayLM)
# ---------------------------------------------------------------------------


@dataclass
class LmFitResult:
    """Fitted linear model results for all genes."""

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    df_residual: np.ndarray
    design: np.ndarray
    gene_names: list[str] = field(default_factory=list)
    coef_names: list[str] = field(default_factory=list)


@dataclass
class EBayesResult(LmFitResult):
    """eBayes-moderated fit results."""

    s2_prior: float = 0.0
    df_prior: float = 0.0
    s2_post: np.ndarray = field(default_factory=lambda: np.array([]))
    t_stat: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value: np.ndarray = field(default_factory=lambda: np.array([]))
    lods: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Trigamma inverse (Newton iteration, Smyth 2004 Appendix)
# ---------------------------------------------------------------------------


def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y via Newton iteration."""
    if not np.isfinite(x) or x <= 0:
        return np.inf
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(polygamma(1, y))
        tetra = float(polygamma(2, y))
        delta = tri * (1.0 - tri / x) / tetra
        y += delta
        if abs(delta) < 1e-8:
            break
    return y


# ---------------------------------------------------------------------------
# fitFDist: estimate prior df and variance (Smyth 2004 Section 6)
# ---------------------------------------------------------------------------


def _logmdigamma(x: np.ndarray | float) -> np.ndarray | float:
    """Compute log(x) - digamma(x), matching R statmod::logmdigamma."""
    return np.log(x) - digamma(x)


def _fit_f_dist(var: np.ndarray, df1: np.ndarray) -> tuple[float, float]:
    """Fit a scaled F-distribution (R limma::fitFDist, legacy path).

    Matches R exactly when all df1 values are equal.
    """
    df1 = np.atleast_1d(np.asarray(df1, dtype=float))
    var = np.asarray(var, dtype=float)
    n = len(var)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(var[0]), 0.0

    ok = np.isfinite(df1) & (df1 > 1e-15) if len(df1) == n else np.ones(n, dtype=bool)
    ok = ok & np.isfinite(var) & (var > -1e-15)
    if ok.sum() < 2:
        return float(var[ok][0]) if ok.sum() == 1 else (np.nan, np.nan)

    x = var[ok]
    d1 = df1[ok] if len(df1) == n else df1

    x = np.maximum(x, 0.0)
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)

    z = np.log(x)
    e = z + _logmdigamma(d1 / 2.0)

    nok = len(x)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / (nok - 1))

    evar -= float(np.mean(polygamma(1, d1 / 2.0)))

    if evar > 0:
        df2 = 2.0 * _trigamma_inverse(evar)
        s20 = float(np.exp(emean - _logmdigamma(df2 / 2.0)))
    else:
        df2 = np.inf
        s20 = float(np.mean(x))

    return s20, df2


def _fit_f_dist_unequal_df1(
    var: np.ndarray, df1: np.ndarray
) -> tuple[float | np.ndarray, float]:
    """Fit scaled F-distribution with unequal df1 (R limma::fitFDistUnequalDF1).

    Uses maximum likelihood optimization, matching R's optimize() approach.
    """
    var = np.asarray(var, dtype=float)
    df1 = np.atleast_1d(np.asarray(df1, dtype=float))

    ok = np.isfinite(var) & (var > 0) & np.isfinite(df1) & (df1 > 0.01)
    if ok.sum() < 2:
        return np.nan, np.nan

    x = var[ok]
    d1_full = df1[ok]

    m = np.median(x)
    xpos = np.maximum(x, 1e-12 * m)
    z = np.log(xpos)
    d1 = d1_full / 2.0

    e = z + _logmdigamma(d1)
    w = 1.0 / polygamma(1, d1)
    emean = float(np.sum(w * e) / np.sum(w))

    d1x = d1 * xpos

    def neg2loglik(par: float) -> float:
        d2 = par / (1.0 - par)
        d2s20 = d2 * np.exp(emean - _logmdigamma(d2))
        ll = np.sum(
            -(d1 + d2) * np.log1p(d1x / d2s20)
            - d1 * np.log(d2s20)
            + gammaln(d1 + d2)
            - gammaln(d2)
        )
        return -2.0 * ll

    result = minimize_scalar(neg2loglik, bounds=(0.5, 0.9998), method="bounded")
    par_opt = result.x
    d2 = par_opt / (1.0 - par_opt)
    s20 = np.exp(emean - _logmdigamma(d2))

    return float(s20), float(2.0 * d2)


# ---------------------------------------------------------------------------
# squeezeVar: empirical Bayes variance moderation
# ---------------------------------------------------------------------------


def _squeeze_var(var: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Moderate sample variances toward a common prior (R limma::squeezeVar)."""
    var = np.asarray(var, dtype=float)
    df = np.atleast_1d(np.asarray(df, dtype=float))
    n = len(var)
    if n < 3:
        return var.copy(), float(np.median(var)), 0.0

    if len(df) > 1:
        var = np.where(df == 0, 0.0, var)

    dfp = df[df > 0]
    legacy = dfp.min() == dfp.max()

    if legacy:
        s2_prior, df_prior = _fit_f_dist(var, df)
    else:
        s2_prior, df_prior = _fit_f_dist_unequal_df1(var, df)

    if np.isfinite(df_prior):
        var_post = (df * var + df_prior * s2_prior) / (df + df_prior)
    else:
        var_post = np.full(n, s2_prior)

    return var_post, float(s2_prior), float(df_prior)


# ---------------------------------------------------------------------------
# lmFit: per-gene ordinary least squares
# ---------------------------------------------------------------------------


def _lm_fit(
    data: np.ndarray,
    design: np.ndarray,
    gene_names: list[str],
    coef_names: list[str],
) -> LmFitResult:
    """Fit per-gene linear models by ordinary least squares."""
    n_genes, n_samples = data.shape
    n_coefs = design.shape[1]

    XtX_inv = np.linalg.pinv(design.T @ design)
    beta = (XtX_inv @ design.T @ data.T).T

    fitted = data @ design @ XtX_inv @ design.T
    residuals = data - fitted @ np.eye(n_samples)
    residuals = data - (design @ beta.T).T

    rss = np.nansum(residuals**2, axis=1)
    n_obs = np.sum(np.isfinite(data), axis=1)
    df_residual = n_obs - n_coefs

    sigma = np.where(
        df_residual > 0,
        np.sqrt(rss / df_residual),
        np.nan,
    )

    stdev_unscaled = np.sqrt(np.diag(XtX_inv))
    stdev_unscaled = np.broadcast_to(stdev_unscaled, (n_genes, n_coefs)).copy()

    return LmFitResult(
        coefficients=beta,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual.astype(float),
        design=design,
        gene_names=gene_names,
        coef_names=coef_names,
    )


def _lm_fit_with_na(
    data: np.ndarray,
    design: np.ndarray,
    gene_names: list[str],
    coef_names: list[str],
    weights: np.ndarray | None = None,
) -> LmFitResult:
    """Fit per-gene least squares handling NaN values per gene.

    ``weights`` (limma's ``lmFit(weights =)``) gives each cell a positive
    observation weight.
    """
    n_genes, _ = data.shape
    n_coefs = design.shape[1]

    beta = np.full((n_genes, n_coefs), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_res = np.full(n_genes, np.nan)
    stdev_us = np.full((n_genes, n_coefs), np.nan)

    for i in range(n_genes):
        obs = np.isfinite(data[i])
        n_obs = obs.sum()
        if n_obs <= n_coefs:
            continue
        y = data[i, obs]
        X = design[obs]
        w = np.ones(n_obs) if weights is None else weights[i, obs]
        try:
            XtX_inv = np.linalg.inv((X.T * w) @ X)
        except np.linalg.LinAlgError:
            continue
        b = XtX_inv @ (X.T * w) @ y
        resid = y - X @ b
        df = n_obs - n_coefs
        beta[i] = b
        sigma[i] = np.sqrt(np.sum(w * resid**2) / df) if df > 0 else np.nan
        df_res[i] = df
        stdev_us[i] = np.sqrt(np.diag(XtX_inv))

    return LmFitResult(
        coefficients=beta,
        stdev_unscaled=stdev_us,
        sigma=sigma,
        df_residual=df_res,
        design=design,
        gene_names=gene_names,
        coef_names=coef_names,
    )


# ---------------------------------------------------------------------------
# contrasts.fit: apply contrast to fitted model
# ---------------------------------------------------------------------------


def _contrasts_fit(fit: LmFitResult, contrasts: np.ndarray) -> LmFitResult:
    """Apply a contrast matrix to a fitted linear model (R limma::contrasts.fit).

    For orthogonal designs: new_stdev = sqrt(old_stdev^2 %*% contrasts^2),
    preserving per-gene stdev_unscaled information (critical for NA handling).
    """
    contrasts = np.asarray(contrasts)
    new_coef = fit.coefficients @ contrasts

    new_stdev = np.sqrt(fit.stdev_unscaled**2 @ contrasts**2)

    return LmFitResult(
        coefficients=new_coef,
        stdev_unscaled=new_stdev,
        sigma=fit.sigma,
        df_residual=fit.df_residual,
        design=fit.design,
        gene_names=fit.gene_names,
        coef_names=[f"contrast_{i}" for i in range(contrasts.shape[1])],
    )


# ---------------------------------------------------------------------------
# eBayes: empirical Bayes moderation
# ---------------------------------------------------------------------------


def _ebayes(fit: LmFitResult) -> EBayesResult:
    """Apply empirical Bayes moderation to a fitted linear model."""
    var = fit.sigma**2
    var_post, s2_prior, df_prior = _squeeze_var(var, fit.df_residual)

    s_post = np.sqrt(var_post)
    t_stat = fit.coefficients / (fit.stdev_unscaled * s_post[:, np.newaxis])

    df_total = fit.df_residual + df_prior
    df_pooled = np.nansum(fit.df_residual)
    df_total = np.minimum(df_total, df_pooled)

    from scipy.stats import t as t_dist

    p_value = 2.0 * t_dist.sf(np.abs(t_stat), df_total[:, np.newaxis])

    return EBayesResult(
        coefficients=fit.coefficients,
        stdev_unscaled=fit.stdev_unscaled,
        sigma=fit.sigma,
        df_residual=fit.df_residual,
        design=fit.design,
        gene_names=fit.gene_names,
        coef_names=fit.coef_names,
        s2_prior=s2_prior,
        df_prior=df_prior,
        s2_post=var_post,
        t_stat=t_stat,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# topTable: extract results for one coefficient
# ---------------------------------------------------------------------------


def _top_table(fit: EBayesResult, amean: np.ndarray, coef: int = 0) -> pd.DataFrame:
    """Extract a results table from an eBayes fit.

    ``amean`` is the per-protein average expression (row means of the
    pre-contrast expression matrix), matching limma's ``AveExpr``.
    """
    return pd.DataFrame(
        {
            "ProteinName": fit.gene_names,
            "log2FC": fit.coefficients[:, coef],
            "AveExpr": amean,
            "t_stat": fit.t_stat[:, coef],
            "pvalue": fit.p_value[:, coef],
            "B": np.zeros(len(fit.gene_names)),
        }
    )


# ---------------------------------------------------------------------------
# Public API: run_limma
# ---------------------------------------------------------------------------


def run_limma(
    log2_matrix: pd.DataFrame,
    samples_a: list[str],
    samples_b: list[str],
    cond_a: str,
    cond_b: str,
) -> pd.DataFrame:
    """Run limma moderated t-test (pure Python implementation).

    Reimplements the limma pipeline (lmFit → contrasts.fit → eBayes →
    topTable) using numpy/scipy, without rpy2 or R dependencies.
    """
    sub_matrix = filter_testable(log2_matrix, samples_a, samples_b)
    if sub_matrix.empty:
        logger.warning("No proteins passed DE filtering for limma")
        return pd.DataFrame()

    sample_order = list(samples_a) + list(samples_b)
    mat = sub_matrix[sample_order].values
    gene_names = list(sub_matrix.index)

    group = np.array([0] * len(samples_a) + [1] * len(samples_b))
    design = np.zeros((len(sample_order), 2))
    design[group == 0, 0] = 1.0
    design[group == 1, 1] = 1.0
    coef_names = [cond_a, cond_b]

    has_na = np.any(np.isnan(mat))
    if has_na:
        fit = _lm_fit_with_na(mat, design, gene_names, coef_names)
    else:
        fit = _lm_fit(mat, design, gene_names, coef_names)

    contrasts = np.array([[1.0], [-1.0]])
    fit_c = _contrasts_fit(fit, contrasts)
    fit_eb = _ebayes(fit_c)
    amean = np.nanmean(mat, axis=1)
    raw = _top_table(fit_eb, amean, coef=0)

    return finalize_de_result(
        raw,
        sub_matrix,
        samples_a,
        samples_b,
        cond_a,
        cond_b,
        extra_cols=("t_stat", "AveExpr", "B"),
    )
