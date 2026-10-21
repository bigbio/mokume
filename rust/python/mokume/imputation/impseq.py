"""Sequential imputation (pure Python).

Reimplements ``rrcovNA::impSeq`` as the Rust kernel does, without rpy2 or R
dependencies. Rows with missing values are imputed one at a time, fewest
missing values first (ties in row order), from the conditional normal given
their observed values; the mean and covariance come from the complete rows
plus the rows imputed so far. When those rows number at most the column
count, the covariance gets ``0.01 I`` as in rrcovNA.

Reference
---------
Béguin C, Hulliger B. The BACON-EEM Algorithm for Multivariate Outlier
Detection in Incomplete Survey Data. *Survey Methodology*. 2008.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mokume.core.logger import get_logger

logger = get_logger("mokume.imputation.impseq")


class _Moments:
    """Column sums, means and scatter (summed centred cross-products) of the
    rows used so far, updated in place as each imputed row is added."""

    def __init__(self, rows: np.ndarray):
        self.rows = rows.shape[0]
        self.sums = rows.sum(axis=0)
        self.mean = self.sums / self.rows
        centred = rows - self.mean
        self.scatter = centred.T @ centred

    def covariance(self) -> np.ndarray:
        """Sample covariance (ddof = 1)."""
        return self.scatter / (self.rows - 1)

    def push(self, row: np.ndarray) -> None:
        """Add one row (Welford update of the scatter)."""
        self.rows += 1
        before = row - self.mean
        self.sums = self.sums + row
        self.mean = self.sums / self.rows
        self.scatter = self.scatter + np.outer(before, row - self.mean)


def _cond_normal_impute(
    row: np.ndarray,
    obs_idx: np.ndarray,
    miss_idx: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Conditional normal mean ``mu_m + Sigma_mo Sigma_oo^-1 (x_o - mu_o)``."""
    try:
        z = np.linalg.solve(cov[np.ix_(obs_idx, obs_idx)], row[obs_idx] - mu[obs_idx])
    except np.linalg.LinAlgError as exc:
        raise ValueError("impSeq observed covariance is singular") from exc
    return mu[miss_idx] + cov[np.ix_(miss_idx, obs_idx)] @ z


def _impseq_core(data: np.ndarray) -> np.ndarray:
    """Sequential imputation matching R rrcovNA::impSeq."""
    p = data.shape[1]
    observed = np.isfinite(data)
    result = np.where(observed, data, np.nan)
    complete_mask = observed.all(axis=1)
    if complete_mask.all():
        return result
    if complete_mask.sum() < 2:
        raise ValueError(
            "impSeq requires at least two complete protein rows to estimate covariance"
        )

    moments = _Moments(result[complete_mask])
    incomplete_idx = np.flatnonzero(~complete_mask)
    n_miss_per_row = (~observed[incomplete_idx]).sum(axis=1)
    order = incomplete_idx[np.argsort(n_miss_per_row, kind="stable")]
    for row_i in order:
        obs_j = np.flatnonzero(observed[row_i])
        miss_j = np.flatnonzero(~observed[row_i])
        if obs_j.size == 0:
            result[row_i, miss_j] = moments.mean[miss_j]
        else:
            cov = moments.covariance()
            if moments.rows <= p:
                cov = cov + 0.01 * np.eye(p)
            result[row_i, miss_j] = _cond_normal_impute(
                result[row_i], obs_j, miss_j, moments.mean, cov
            )
        moments.push(result[row_i])

    return result


def impute_impseq(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Impute missing values using sequential regression (pure Python).

    Parameters
    ----------
    data : pd.DataFrame
        Matrix with NaN for missing values; at least two rows must be
        complete.

    Returns
    -------
    pd.DataFrame
        Filled matrix with the same shape and index.
    """
    n_missing = int(data.isna().sum().sum())
    filled = _impseq_core(data.to_numpy(dtype=float))
    result = pd.DataFrame(filled, index=data.index, columns=data.columns)
    logger.info("impSeq imputation: %d values imputed", n_missing)
    return result
