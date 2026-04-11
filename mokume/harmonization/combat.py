"""
ComBat batch correction implementation.

Wraps the existing batch correction logic and registers it with the PluginRegistry.

Requires: pip install mokume[batch-correction]
"""

from typing import List, Optional

import pandas as pd

from mokume.harmonization.base import BatchCorrector
from mokume.core.registry import PluginRegistry


@PluginRegistry.register("harmonization", "combat")
class ComBatCorrector(BatchCorrector):
    """ComBat batch correction using inmoose.

    Removes batch effects while optionally preserving biological signal
    specified via covariates (e.g., sex, tissue from SDRF).

    Parameters
    ----------
    parametric : bool
        Use parametric empirical Bayes estimation. Default True.
    mean_only : bool
        Only adjust batch means. Default False.
    """

    def __init__(self, parametric: bool = True, mean_only: bool = False):
        self.parametric = parametric
        self.mean_only = mean_only

    @property
    def name(self) -> str:
        return "ComBat"

    def correct(
        self,
        df: pd.DataFrame,
        batch: List[int],
        covariates: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply ComBat batch correction.

        Delegates to mokume.harmonization.correction.apply_batch_correction().

        Parameters
        ----------
        df : pd.DataFrame
            Wide-format protein intensity matrix (proteins x samples).
        batch : list[int]
            Batch assignment for each sample.
        covariates : list[list[int]], optional
            Biological covariates to preserve.
        **kwargs
            Additional keyword arguments passed to apply_batch_correction.

        Returns
        -------
        pd.DataFrame
            Batch-corrected intensity matrix.
        """
        from mokume.harmonization.correction import apply_batch_correction

        return apply_batch_correction(
            df=df,
            batch=batch,
            covs=covariates,
            kwargs={
                "par_prior": self.parametric,
                "mean_only": self.mean_only,
                **kwargs,
            },
        )
