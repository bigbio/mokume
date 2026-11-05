"""Per-cell quantification precision from a protein's ion residuals.

Each ion's log2 differences to the quantified protein profile are centred on
their median, which removes the ion's response factor. The residual sum of
squares of a protein, on ``observations - ions - (samples - 1)`` degrees of
freedom, is shrunk towards the dataset's median residual variance with
``SHRINK_DF`` prior degrees of freedom. A cell's precision (1/SE^2) is its
observed ion count over that variance, so a value supported by more and more
consistent ions weighs more in a precision-weighted test. Mirrors the Rust
``mokume_quant::precision`` module.
"""

import numpy as np
import pandas as pd

# Prior degrees of freedom that shrink a protein's residual variance towards
# the dataset median.
SHRINK_DF = 4.0
# Proteins need this many residual degrees of freedom to enter the median.
MIN_PRIOR_DOF = 3.0
# Residual variance used when no protein reaches MIN_PRIOR_DOF.
FALLBACK_VARIANCE = 0.04


def log2_positive(values):
    """log2 of the positive finite entries; everything else becomes NaN."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    np.log2(values, out=result, where=np.isfinite(values) & (values > 0))
    return result


def profile_residuals(ions, profile):
    """Residual sum of squares, degrees of freedom and per-sample ion counts.

    ``ions`` is an ion-by-sample log2 array and ``profile`` the protein's log2
    value per sample; non-finite entries are missing.
    """
    ions = np.asarray(ions, dtype=float)
    profile = np.asarray(profile, dtype=float)
    levels = np.isfinite(profile)
    differences = np.where(np.isfinite(ions) & levels, ions - profile, np.nan)
    differences = differences[np.isfinite(differences).sum(axis=1) > 1]
    observed = np.isfinite(differences)
    centred = differences - np.nanmedian(differences, axis=1, keepdims=True)
    rss = float(np.sum(centred[observed] ** 2))
    free = observed.sum() - len(differences) - (levels.sum() - 1.0)
    return rss, max(0.0, float(free)), observed.sum(axis=0)


def precision_matrix(residuals, samples, protein_column):
    """:func:`cell_precision` as a protein matrix, names in ``protein_column``."""
    return (
        pd.Series(cell_precision(residuals, samples), dtype=float)
        .unstack()
        .rename_axis(index=protein_column, columns=None)
        .reset_index()
    )


def cell_precision(residuals, samples):
    """Precision of every quantified cell, keyed by ``(protein, sample)``.

    ``residuals`` pairs each protein with its :func:`profile_residuals` over
    ``samples``.
    """
    variances = [rss / dof for _, (rss, dof, _) in residuals if dof >= MIN_PRIOR_DOF]
    prior = float(np.median(variances)) if variances else FALLBACK_VARIANCE
    precision = {}
    for protein, (rss, dof, counts) in residuals:
        variance = (rss + SHRINK_DF * prior) / (dof + SHRINK_DF)
        precision.update(
            ((protein, sample), count / variance)
            for sample, count in zip(samples, counts)
            if count
        )
    return precision
