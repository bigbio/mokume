"""Plugin registrations for the advanced (dev-authored) imputers.

The advanced imputers (bpca, qrilc, seqknn, impseq, impseqrob, gms,
missforest) are implemented as plain functions in their own modules.
Several of those modules are byte-locked, so this module does NOT modify
them: it imports each ``impute_*`` function and wraps it in a thin
:class:`~mokume.imputation.base.ImputationMethod` adapter that delegates
straight to the dev function.

Each adapter follows the same idiom as
:mod:`mokume.imputation.simple`: a ``name`` property plus an
``impute(df, **kwargs)`` method that pulls the dev function's own
parameters out of ``kwargs`` (falling back to the dev defaults) and
returns the imputed wide-format matrix unchanged.

Importing this module registers all seven methods; it is imported for
its side effects.
"""

from __future__ import annotations

import pandas as pd

from mokume.core.registry import PluginRegistry
from mokume.imputation.base import ImputationMethod

# Delegate to dev-authored implementations (do not reimplement the math).
from mokume.imputation.bpca import impute_bpca
from mokume.imputation.gms import impute_gms
from mokume.imputation.impseq import impute_impseq
from mokume.imputation.impseqrob import impute_impseqrob
from mokume.imputation.missforest import impute_missforest
from mokume.imputation.qrilc import impute_qrilc
from mokume.imputation.seqknn import impute_seqknn


@PluginRegistry.register("imputation", "bpca")
class BPCAImputation(ImputationMethod):
    """Bayesian PCA imputation (delegates to ``impute_bpca``).

    Low-rank probabilistic imputation for MAR-style missingness.

    Parameters (via ``impute`` kwargs)
    ----------------------------------
    n_pcs : int
        Number of principal components (default 2).
    """

    @property
    def name(self) -> str:
        return "BPCA"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return impute_bpca(df, n_pcs=kwargs.get("n_pcs", 2))


@PluginRegistry.register("imputation", "qrilc")
class QRILCImputation(ImputationMethod):
    """imputeLCMD QRILC imputation (delegates to ``impute_qrilc``).

    Left-censored (MNAR) fill; expects log-space input.

    Parameters (via ``impute`` kwargs)
    ----------------------------------
    tune_sigma : float
        Standard-deviation multiplier before drawing (default 1.0).
    """

    @property
    def name(self) -> str:
        return "QRILC"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return impute_qrilc(df, tune_sigma=kwargs.get("tune_sigma", 1.0))


@PluginRegistry.register("imputation", "seqknn")
class SeqKNNImputation(ImputationMethod):
    """Sequential KNN imputation (delegates to ``impute_seqknn``).

    Kim et al. (2004) sequential weighted k-NN fill.

    Parameters (via ``impute`` kwargs)
    ----------------------------------
    k : int
        Neighbour count (default 5). ``n_neighbors`` is accepted as an
        alias for compatibility with the sklearn-style callers.
    """

    @property
    def name(self) -> str:
        return "SeqKNN"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        k = kwargs.get("k", kwargs.get("n_neighbors", 5))
        return impute_seqknn(df, k=k)


@PluginRegistry.register("imputation", "impseq")
class ImpSeqImputation(ImputationMethod):
    """Sequential regression imputation (delegates to ``impute_impseq``).

    rrcovNA-style iterative conditional-normal imputation. Takes no
    tuning parameters.
    """

    @property
    def name(self) -> str:
        return "ImpSeq"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return impute_impseq(df)


@PluginRegistry.register("imputation", "impseqrob")
class ImpSeqRobImputation(ImputationMethod):
    """Robust sequential imputation (delegates to ``impute_impseqrob``).

    rrcovNA-style robust iterative imputation with MCD covariance.

    Parameters (via ``impute`` kwargs)
    ----------------------------------
    alpha : float
        MCD support fraction (default 0.9).
    max_variables : int or None
        Column-count guard against the O(p^3) blow-up (default 2000);
        pass ``None`` to disable. Omitted from the call unless supplied
        so the dev default is honoured.
    """

    @property
    def name(self) -> str:
        return "ImpSeqRob"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        call_kwargs = {"alpha": kwargs.get("alpha", 0.9)}
        if "max_variables" in kwargs:
            call_kwargs["max_variables"] = kwargs["max_variables"]
        return impute_impseqrob(df, **call_kwargs)


@PluginRegistry.register("imputation", "gms")
class GMSImputation(ImputationMethod):
    """Gaussian Mixture Sampling imputation (delegates to ``impute_gms``).

    Per-sample two-component GMM MNAR fill with the lower component's
    centre. Takes no tuning parameters.
    """

    @property
    def name(self) -> str:
        return "GMS"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return impute_gms(df)


@PluginRegistry.register("imputation", "missforest")
class MissForestImputation(ImputationMethod):
    """Random-forest iterative imputation (delegates to ``impute_missforest``).

    Stekhoven & Buehlmann (2011) via sklearn IterativeImputer.

    Parameters (via ``impute`` kwargs)
    ----------------------------------
    n_estimators : int
        Trees per forest (default 100).
    max_iter : int
        Imputation passes (default 10).
    random_state : int or None
        Seed for reproducibility (default 0).
    n_jobs : int
        Parallel workers, -1 = all cores (default -1).
    """

    @property
    def name(self) -> str:
        return "MissForest"

    def impute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return impute_missforest(
            df,
            n_estimators=kwargs.get("n_estimators", 100),
            max_iter=kwargs.get("max_iter", 10),
            random_state=kwargs.get("random_state", 0),
            n_jobs=kwargs.get("n_jobs", -1),
        )
