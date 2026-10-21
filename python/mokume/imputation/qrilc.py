"""
QRILC (Quantile Regression Imputation of Left-Censored data).

Follows ``imputeLCMD::impute.QRILC`` 2.1 as the Rust kernel does. For each
sample (column) a normal is fitted to the observed values by regressing
their quantiles on the standard-normal quantiles they would have if the
missing fraction ``pNA`` were the left tail; missing entries are then drawn
from that normal truncated above its ``pNA + 0.001`` quantile.

References
----------
- Wei R, et al. Missing Value Imputation Approach for Mass
  Spectrometry-based Metabolomics Data. Sci Rep. 2018;8:663.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mokume.core.logger import get_logger
from mokume.imputation.stochastic import (
    check_tune_sigma,
    normal_cdf,
    normal_quantile,
    uniforms,
)

logger = get_logger("mokume.imputation.qrilc")


def impute_qrilc(
    data: pd.DataFrame,
    tune_sigma: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Impute left-censored missing values with imputeLCMD QRILC.

    Every sample draws a value for every protein, by inverting the truncated
    normal at a uniform, and keeps the draws of its missing cells.

    Parameters
    ----------
    data : pd.DataFrame
        Protein intensity matrix (rows=proteins, columns=samples) in
        **log2 space** with NaN for missing values.
    tune_sigma : float
        Multiplier applied to the fitted standard deviation before
        drawing (default 1.0).
    random_state : int
        Seed for the draws (the kernel's ``seed``). Without it repeated runs
        on the same matrix disagree, and a benchmark reads that spread as a
        difference between imputation methods rather than as noise within one.

    Returns
    -------
    pd.DataFrame
        Filled matrix with the same shape and index.
    """
    check_tune_sigma(tune_sigma)
    matrix = data.to_numpy(dtype=float)
    observed = np.isfinite(matrix)
    n_rows, n_cols = matrix.shape
    if n_rows == 0:
        return data.copy()

    draws = uniforms(random_state, matrix.size).reshape(n_cols, n_rows).T
    result = matrix.copy()
    for col in range(n_cols):
        missing = ~observed[:, col]
        result[missing, col] = _truncated_draws(
            matrix[observed[:, col], col], n_rows, draws[missing, col], tune_sigma
        )

    logger.info("QRILC imputation: %d values imputed", int((~observed).sum()))
    return pd.DataFrame(result, index=data.index, columns=data.columns)


def _truncated_draws(
    values: np.ndarray, n_rows: int, uniform: np.ndarray, tune_sigma: float
) -> np.ndarray:
    """Draws at ``uniform`` from one sample's fitted normal, truncated above."""
    missing_fraction = 1.0 - values.size / n_rows
    mean, sd = _distribution_parameters(values, missing_fraction)
    upper = mean + sd * normal_quantile(missing_fraction + 0.001)
    # tmvtnorm 1.7's univariate Gibbs branch passes sigma directly as SD.
    draw_sd = sd * tune_sigma
    upper_probability = normal_cdf((upper - mean) / draw_sd)
    return mean + draw_sd * normal_quantile(uniform * upper_probability)


def _distribution_parameters(
    values: np.ndarray, missing_fraction: float
) -> tuple[float, float]:
    """Mean and SD from the quantile profile of one sample's observed values."""
    if values.size < 2 or missing_fraction >= 0.99:
        raise ValueError(
            "QRILC needs at least two observations and less than 99% missingness "
            "per sample"
        )
    steps = np.arange(100)
    x = normal_quantile(
        missing_fraction + 0.001 + (0.99 - missing_fraction) * steps / 99.0
    )
    y = np.quantile(values, 0.001 + steps * 0.01)
    dx = x - x.mean()
    sd = float(np.sum(dx * (y - y.mean())) / np.sum(dx * dx))
    mean = float(y.mean() - sd * x.mean())
    if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0.0):
        raise ValueError("QRILC fitted distribution has nonpositive or invalid scale")
    return mean, sd
