"""
Censored-aware missing value imputation for proteomics data.

Implements multiple imputation strategies that account for the
non-random nature of missingness in mass spectrometry data:

- **MinProb**: Draws replacement values from a normal centred on a low
  quantile of each sample (imputeLCMD ``impute.MinProb``).
- **MinDet**: Replaces missing values with a fixed quantile of the
  per-sample observed values.

References
----------
- Lazar C, et al. Accounting for the Multiple Natures of Missing Values
  in Label-Free Quantitative Proteomics. J Proteome Res. 2016;15(4):1116-25.

"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from mokume.core.logger import get_logger
from mokume.imputation.stochastic import check_tune_sigma, normals

logger = get_logger("mokume.imputation.censored")


def impute_minprob(
    data: pd.DataFrame,
    quantile: float = 0.01,
    tune_sigma: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perform imputeLCMD MinProb imputation, as the Rust kernel does.

    Each missing entry of a sample is drawn from ``N(q, sd**2)``: ``q`` is the
    sample's ``quantile``-th observed value and ``sd`` is ``tune_sigma`` times
    the median standard deviation of the proteins observed in more than half
    of the samples. As in imputeLCMD, every sample draws a value for every
    protein and keeps the draws of its missing cells.

    ``random_state`` seeds the draws (the kernel's ``seed``) so repeated runs
    on the same matrix agree. Without it a benchmark cannot rank this method
    against a deterministic one: the run-to-run spread gets read as a
    difference between methods.
    """
    check_tune_sigma(tune_sigma)
    matrix = data.to_numpy(dtype=float)
    observed = np.isfinite(matrix)
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return data.copy()

    sd = _median_protein_sd(np.where(observed, matrix, np.nan), observed) * tune_sigma
    draws = normals(random_state, matrix.size).reshape(n_cols, n_rows).T
    result = matrix.copy()
    for col in range(n_cols):
        values = matrix[observed[:, col], col]
        if values.size == 0 or not 0.0 <= quantile <= 1.0:
            raise ValueError(
                "MinProb needs observed values in every sample and a quantile in [0,1]"
            )
        missing = ~observed[:, col]
        result[missing, col] = np.quantile(values, quantile) + sd * draws[missing, col]

    logger.info("MinProb imputation: %d values imputed", int((~observed).sum()))
    return pd.DataFrame(result, index=data.index, columns=data.columns)


def _median_protein_sd(values: np.ndarray, observed: np.ndarray) -> float:
    """Median SD of the proteins with values in more than half of the samples."""
    counts = observed.sum(axis=1)
    eligible = (counts * 2 > values.shape[1]) & (counts >= 2)
    if not eligible.any():
        raise ValueError(
            "MinProb cannot estimate SD: no proteins with more than 50% observed "
            "samples and at least two values"
        )
    return float(np.median(np.nanstd(values[eligible], axis=1, ddof=1)))


def impute_mindet(
    data: pd.DataFrame,
    quantile: float = 0.01,
) -> pd.DataFrame:
    """Apply MinDet imputation using a per-sample observed quantile."""
    result = data.copy()
    n_imputed = 0

    for col in data.columns:
        missing = data[col].isna()
        if not missing.any():
            continue
        observed = data[col].dropna().values
        if len(observed) == 0:
            continue
        fill_val = np.quantile(observed, quantile)
        result.loc[missing, col] = fill_val
        n_imputed += missing.sum()

    logger.info("MinDet imputation: %d values imputed (q=%.3f)", n_imputed, quantile)
    return result


def _knn_impute(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """KNN imputation wrapper for the dispatcher."""
    imputer = KNNImputer(
        n_neighbors=kwargs.get("n_neighbors", 5),
        weights=kwargs.get("weights", "uniform"),
    )
    result = pd.DataFrame(
        imputer.fit_transform(data),
        index=data.index,
        columns=data.columns,
    )
    n_imputed = data.isna().sum().sum()
    logger.info("KNN imputation: %d values imputed", n_imputed)
    return result


def _dispatch_seqknn(data, **kwargs):
    from mokume.imputation.seqknn import impute_seqknn

    return impute_seqknn(data, **kwargs)


def _dispatch_missforest(data, **kwargs):
    from mokume.imputation.missforest import impute_missforest

    return impute_missforest(data, **kwargs)


def _dispatch_impseq(data, **kwargs):
    from mokume.imputation.impseq import impute_impseq

    return impute_impseq(data, **kwargs)


def _dispatch_impseqrob(data, **kwargs):
    from mokume.imputation.impseqrob import impute_impseqrob

    return impute_impseqrob(data, **kwargs)


def _dispatch_bpca(data, **kwargs):
    from mokume.imputation.bpca import impute_bpca

    return impute_bpca(data, **kwargs)


def _dispatch_qrilc(data, **kwargs):
    from mokume.imputation.qrilc import impute_qrilc

    return impute_qrilc(data, **kwargs)


def _dispatch_gms(data, **kwargs):
    from mokume.imputation.gms import impute_gms

    return impute_gms(data, **kwargs)


_DISPATCH = {
    "minprob": impute_minprob,
    "mindet": impute_mindet,
    "knn": _knn_impute,
    "seqknn": _dispatch_seqknn,
    "missforest": _dispatch_missforest,
    "impseq": _dispatch_impseq,
    "impseqrob": _dispatch_impseqrob,
    "bpca": _dispatch_bpca,
    "qrilc": _dispatch_qrilc,
    "gms": _dispatch_gms,
}


def impute_censored(
    data: pd.DataFrame,
    method: str = "minprob",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch censored-aware imputation by method name."""
    method = method.lower()
    if method == "none":
        logger.info("No imputation applied (method='none')")
        return data.copy()
    handler = _DISPATCH.get(method)
    if handler is None:
        valid = ", ".join(sorted([*_DISPATCH, "none"]))
        raise ValueError(f"Unknown imputation method '{method}'. Choose from: {valid}")
    return handler(data, **kwargs)
