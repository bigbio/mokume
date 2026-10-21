"""
GMS (Gaussian Mixture Sampling) imputation for proteomics data.

Fits a two-component Gaussian mixture per sample (column) to separate the
abundance distribution from the missing-not-at-random (MNAR) component and
fills the sample's missing values with the centre of the lower component, as
the Rust kernel does. The fill is deterministic: the centre is the expected
value of a draw from that component.

References
----------
- Lazar C, et al. Accounting for the Multiple Natures of Missing
  Values in Label-Free Quantitative Proteomics Data Sets.
  J Proteome Res. 2016;15(4):1116-25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mokume.core.logger import get_logger

logger = get_logger("mokume.imputation.gms")

_N_COMPONENTS = 2
# scikit-learn GaussianMixture defaults: reg_covar, tol and max_iter.
_REG_COVAR = 1e-6
_EM_TOL = 1e-3
_EM_MAX_ITER = 100


def impute_gms(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using Gaussian Mixture Sampling.

    A sample with fewer than three observed values is filled with their mean.

    Parameters
    ----------
    data : pd.DataFrame
        Protein intensity matrix (rows=proteins, columns=samples) with NaN.

    Returns
    -------
    pd.DataFrame
        Filled matrix with the same shape and index.
    """
    matrix = data.to_numpy(dtype=float)
    observed = np.isfinite(matrix)
    result = matrix.copy()
    for col in range(matrix.shape[1]):
        values = matrix[observed[:, col], col]
        missing = ~observed[:, col]
        if values.size == 0 or not missing.any():
            continue
        if values.size < _N_COMPONENTS + 1:
            fill = values.mean()
        else:
            fill = _lower_component_mean(values)
        if np.isfinite(fill):
            result[missing, col] = fill

    logger.info("GMS imputation: %d values imputed", int((~observed).sum()))
    return pd.DataFrame(result, index=data.index, columns=data.columns)


def _lower_component_mean(values: np.ndarray) -> float:
    """Mean of the lower component of a two-component mixture fitted by EM.

    EM starts from the sorted values split at their median index, each half
    giving a component's mean, variance and weight; it stops when the average
    log-likelihood changes by less than ``_EM_TOL``.
    """
    points = np.sort(values)[:, np.newaxis]
    count = points.shape[0]
    split = min(max(count // 2, 1), count - 1)
    halves = (points[:split, 0], points[split:, 0])
    means = np.array([half.mean() for half in halves])
    variances = np.array(
        [np.mean((half - mean) ** 2) for half, mean in zip(halves, means)]
    )
    variances = variances + _REG_COVAR
    weights = np.array([split, count - split]) / count

    previous = -np.inf
    for _ in range(_EM_MAX_ITER):
        weights, means, variances, average = _em_step(points, weights, means, variances)
        if abs(average - previous) < _EM_TOL:
            break
        previous = average
    return float(means[0] if means[0] <= means[1] else means[1])


def _em_step(
    points: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One EM iteration: the new weights, means and variances, and the
    average log-likelihood under the old ones."""
    tiny = np.finfo(float).tiny
    delta = points - means
    densities = weights * (
        np.exp(-0.5 * delta * delta / variances) / np.sqrt(2.0 * np.pi * variances)
    )
    total = densities[:, 0] + densities[:, 1]
    total = np.where(total > 0.0, total, tiny)
    gamma = densities / total[:, np.newaxis]
    effective = gamma.sum(axis=0)
    denominator = np.where(effective > 0.0, effective, tiny)
    means = (gamma * points).sum(axis=0) / denominator
    delta = points - means
    variances = (gamma * delta * delta).sum(axis=0) / denominator + _REG_COVAR
    count = points.shape[0]
    return effective / count, means, variances, float(np.log(total).sum() / count)
