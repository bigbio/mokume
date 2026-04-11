"""
KNN imputation implementation.

Wraps sklearn.impute.KNNImputer and registers it with the PluginRegistry.
"""

import pandas as pd
from sklearn.impute import KNNImputer

from mokume.core.registry import PluginRegistry
from mokume.imputation.base import ImputationMethod


@PluginRegistry.register("imputation", "knn")
class KNNImputation(ImputationMethod):
    """K-Nearest Neighbors imputation.

    Uses sklearn's KNNImputer to fill missing values based on the
    values of the nearest neighbors in the feature space.

    Parameters
    ----------
    n_neighbors : int
        Number of neighboring samples to use. Default 5.
    weights : str
        Weight function for prediction: "uniform" or "distance". Default "uniform".
    metric : str
        Distance metric for neighbor search. Default "nan_euclidean".
    keep_empty_features : bool
        Whether to keep features that are entirely NaN. Default True.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        keep_empty_features: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.keep_empty_features = keep_empty_features

    @property
    def name(self) -> str:
        return "KNN"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors.

        Parameters
        ----------
        df : pd.DataFrame
            Wide-format numeric matrix with NaN values.
        **kwargs
            Ignored (reserved for future use).

        Returns
        -------
        pd.DataFrame
            Matrix with missing values imputed.
        """
        imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            keep_empty_features=self.keep_empty_features,
        )
        imputed = imputer.fit_transform(df)
        return pd.DataFrame(imputed, columns=df.columns, index=df.index)

    def __repr__(self) -> str:
        return f"KNNImputation(n_neighbors={self.n_neighbors})"
