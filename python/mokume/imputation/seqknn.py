"""
Sequential KNN (SeqKNN) imputation.

Implements the sequential k-nearest neighbours imputation method of
Kim, Kim & Yi (2004) BMC Bioinformatics 5:160, with the rules of the Rust
kernel (``--impute-method seqknn``).

Algorithm
---------
1. Order rows (proteins) by missing-value count, ascending; ties keep the
   row order.
2. The rows without missing values form the initial complete set; at least
   one is required.
3. For each remaining row, take the ``k`` complete rows nearest by squared
   Euclidean distance over the row's observed columns (as the original
   SeqKnn 1.0.1 ``nnmiss``; ties by complete-set order) and fill each
   missing entry with their mean weighted by ``1 / (distance + 1e-15)``.
4. After a row is imputed, append it to the complete set so it can
   serve as a neighbour for subsequent rows.

This sequential structure is the key difference from the standard
sklearn ``KNNImputer`` (a single pass over all rows simultaneously)
and matches the Bioconductor ``SeqKnn`` semantics.

References
----------
- Kim, K.-Y., Kim, B.-J., & Yi, G.-S. (2004). Reuse of imputed data
  in microarray analysis increases imputation efficiency.
  *BMC Bioinformatics*, 5, 160.
"""

import numpy as np
import pandas as pd

from mokume.core.logger import get_logger

logger = get_logger("mokume.imputation.seqknn")


def _weighted_neighbour_mean(
    target_row: np.ndarray,
    complete_rows: np.ndarray,
    k: int,
) -> np.ndarray:
    """Impute missing entries of ``target_row`` using k nearest complete rows."""
    missing_mask = np.isnan(target_row)
    observed_mask = ~missing_mask
    diffs = complete_rows[:, observed_mask] - target_row[observed_mask]
    distances = np.sum(diffs * diffs, axis=1)
    nearest_idx = np.argsort(distances, kind="stable")[:k]
    weights = 1.0 / (distances[nearest_idx] + 1e-15)
    neighbours = complete_rows[nearest_idx][:, missing_mask]
    weighted_sum = np.sum(neighbours * weights[:, None], axis=0)

    result = target_row.copy()
    result[missing_mask] = weighted_sum / weights.sum()
    return result


def impute_seqknn(data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Sequentially impute missing values with weighted k-NN.

    ``data`` is a DataFrame with NaN-encoded missings; ``k`` is the
    neighbour count. Returns a same-shape DataFrame with NaNs filled.
    """
    if data.empty:
        return data.copy()
    if k < 1:
        raise ValueError("k must be >= 1")

    matrix = data.to_numpy(dtype=float, copy=True)
    missing_per_row = np.sum(np.isnan(matrix), axis=1)
    if missing_per_row.max() == 0:
        return data.copy()

    sort_order = np.argsort(missing_per_row, kind="stable")
    sorted_matrix = matrix[sort_order]
    n_complete = int(np.sum(missing_per_row == 0))
    if n_complete == 0:
        raise ValueError("SeqKNN requires an initial set of complete protein rows")

    for idx in range(n_complete, len(sorted_matrix)):
        sorted_matrix[idx] = _weighted_neighbour_mean(
            sorted_matrix[idx], sorted_matrix[:idx], k
        )

    restored = sorted_matrix[np.argsort(sort_order)]
    logger.info(
        "SeqKNN: imputed %d rows with k=%d neighbours",
        int(np.sum(missing_per_row > 0)),
        k,
    )
    return pd.DataFrame(restored, index=data.index, columns=data.columns)
