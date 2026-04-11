"""
Batch correction utilities for the mokume package.

This module provides batch effect correction using ComBat (via inmoose).

Key Concepts:
- Batch: Technical variation to REMOVE (e.g., different runs, labs, processing days)
- Covariates: Biological variables to PRESERVE (e.g., sex, tissue from SDRF characteristics)

Note: This module requires the optional 'inmoose' dependency.
Install it with: pip install mokume[batch-correction]
"""

import logging
import warnings
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="numpy.matrixlib.defmatrix"
)

from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA

from mokume.plotting import is_plotting_available
from mokume.harmonization.models import BatchDetectionMethod

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_inmoose_available() -> bool:
    """Check if inmoose is installed."""
    try:
        import inmoose
        return True
    except ImportError:
        return False


def is_batch_correction_available() -> bool:
    """
    Check if batch correction dependencies are installed.

    Returns
    -------
    bool
        True if inmoose is installed, False otherwise.

    Notes
    -----
    Install batch correction support with: pip install mokume[batch-correction]
    """
    return is_inmoose_available()


def compute_pca(df, n_components=5) -> pd.DataFrame:
    """Compute principal components for a given dataframe."""
    pca = PCA(n_components=n_components)
    pca.fit(df)
    df_pca = pca.transform(df)
    df_pca = pd.DataFrame(
        df_pca, index=df.index, columns=[f"PC{i}" for i in range(1, n_components + 1)]
    )
    return df_pca


def get_batch_info_from_sample_names(sample_list: List[str]) -> List[int]:
    """Get batch indices from sample names (legacy function, use detect_batches instead)."""
    samples = [s.split("-")[0] for s in sample_list]
    batches = list(set(samples))
    index = {i: batches.index(i) for i in batches}
    return [index[i] for i in samples]


def detect_batches(
    sample_ids: List[str],
    method: Union[BatchDetectionMethod, str] = BatchDetectionMethod.SAMPLE_PREFIX,
    run_info: Optional[Dict[str, str]] = None,
    batch_column_values: Optional[List[str]] = None,
) -> List[int]:
    """
    Detect batch assignments for samples.

    Parameters
    ----------
    sample_ids : List[str]
        Sample identifiers.
    method : BatchDetectionMethod or str
        How to determine batches. Options:
        - "sample_prefix": Extract from sample name prefix (PXD001-S1 → PXD001)
        - "run": Use run/reference file name
        - "fraction": Use fraction identifier
        - "techreplicate": Use technical replicate identifier
        - "column": Use explicit batch values
    run_info : Optional[Dict[str, str]]
        Mapping of sample_id → run_name (for "run" method).
    batch_column_values : Optional[List[str]]
        Explicit batch values for each sample (for "column" method).

    Returns
    -------
    List[int]
        Batch index for each sample (0-indexed).

    Raises
    ------
    ValueError
        If required parameters are missing for the selected method.

    Examples
    --------
    >>> samples = ["PXD001-S1", "PXD001-S2", "PXD002-S1", "PXD002-S2"]
    >>> detect_batches(samples, method="sample_prefix")
    [0, 0, 1, 1]

    >>> detect_batches(samples, method="column", batch_column_values=["A", "A", "B", "B"])
    [0, 0, 1, 1]
    """
    if isinstance(method, str):
        method = BatchDetectionMethod.from_str(method)

    if method == BatchDetectionMethod.SAMPLE_PREFIX:
        # Extract batch prefix from sample names.
        # Supports multiple conventions:
        #   PXD001-S1  → PXD001  (hyphen-separated)
        #   p1_1       → p1      (quantms TMT plex prefix: letters + digits before _digit)
        import re
        prefixes = []
        for s in sample_ids:
            if "-" in s:
                prefixes.append(s.split("-")[0])
            else:
                # Match leading non-numeric + optional digits as prefix (e.g. p1_1 → p1)
                m = re.match(r"^([a-zA-Z]+\d+)_", s)
                if m:
                    prefixes.append(m.group(1))
                else:
                    prefixes.append(s)
        indices, _ = pd.factorize(pd.array(prefixes))
        return indices.tolist()

    elif method == BatchDetectionMethod.RUN_NAME:
        if run_info is None:
            raise ValueError("run_info required for RUN_NAME method")
        runs = [run_info.get(s, s) for s in sample_ids]
        indices, _ = pd.factorize(pd.array(runs))
        return indices.tolist()

    elif method == BatchDetectionMethod.FRACTION:
        # Requires fraction info - fall back to sample prefix if not available
        if run_info is None:
            logger.warning("No fraction info provided, falling back to sample_prefix")
            return detect_batches(sample_ids, BatchDetectionMethod.SAMPLE_PREFIX)
        fractions = [run_info.get(s, "1") for s in sample_ids]
        indices, _ = pd.factorize(pd.array(fractions))
        return indices.tolist()

    elif method == BatchDetectionMethod.TECHREPLICATE:
        # Requires tech rep info - fall back to sample prefix if not available
        if run_info is None:
            logger.warning("No tech rep info provided, falling back to sample_prefix")
            return detect_batches(sample_ids, BatchDetectionMethod.SAMPLE_PREFIX)
        tech_reps = [run_info.get(s, "1") for s in sample_ids]
        indices, _ = pd.factorize(pd.array(tech_reps))
        return indices.tolist()

    elif method == BatchDetectionMethod.EXPLICIT_COLUMN:
        if batch_column_values is None:
            raise ValueError("batch_column_values required for EXPLICIT_COLUMN method")
        if len(batch_column_values) != len(sample_ids):
            raise ValueError(
                f"batch_column_values length ({len(batch_column_values)}) "
                f"must match sample_ids length ({len(sample_ids)})"
            )
        indices, _ = pd.factorize(pd.array(batch_column_values))
        return indices.tolist()

    else:
        raise ValueError(f"Unknown batch detection method: {method}")


def extract_covariates_from_sdrf(
    sdrf_path: str,
    sample_ids: List[str],
    covariate_columns: List[str],
) -> Optional[List[List[int]]]:
    """
    Extract categorical covariates from SDRF for batch correction.

    Covariates represent biological variables whose signal should be PRESERVED
    during batch correction. For example, if samples from different batches
    share the same sex or tissue type, ComBat will preserve this biological
    signal while removing technical batch effects.

    Parameters
    ----------
    sdrf_path : str
        Path to SDRF file.
    sample_ids : List[str]
        Sample IDs matching the protein matrix columns (in order).
    covariate_columns : List[str]
        SDRF columns to use as covariates.
        e.g., ["characteristics[sex]", "characteristics[organism part]"]

    Returns
    -------
    List[List[int]] or None
        Covariate matrix as list of lists (samples × covariates) with
        categorical encoding, or None if no valid covariates found.

    Notes
    -----
    - Covariates MUST be categorical (ComBat requirement)
    - Samples in covariate matrix must match protein matrix column order
    - Signal from these variables is PRESERVED after batch correction

    Examples
    --------
    SDRF with columns:
        source name | characteristics[sex] | characteristics[tissue]
        Sample1     | male                 | liver
        Sample2     | female               | liver
        Sample3     | male                 | brain

    >>> extract_covariates_from_sdrf(
    ...     "experiment.sdrf.tsv",
    ...     ["Sample1", "Sample2", "Sample3"],
    ...     ["characteristics[sex]", "characteristics[tissue]"]
    ... )
    [[0, 0], [1, 0], [0, 1]]  # [sex_encoded, tissue_encoded] per sample
    """
    if not covariate_columns:
        return None

    try:
        sdrf = pd.read_csv(sdrf_path, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to read SDRF file: {e}")
        return None

    sdrf.columns = [c.lower() for c in sdrf.columns]

    # Find the sample name column
    sample_col = None
    for col in ["source name", "sample name", "source_name", "sample_name"]:
        if col in sdrf.columns:
            sample_col = col
            break

    if sample_col is None:
        logger.warning("Could not find sample name column in SDRF")
        return None

    # Build sample → row index mapping
    sdrf_samples = sdrf[sample_col].tolist()

    covar_data = []
    valid_columns = []

    for col in covariate_columns:
        col_lower = col.lower()

        # Find matching column (exact or partial match)
        matched_col = None
        if col_lower in sdrf.columns:
            matched_col = col_lower
        else:
            # Try partial match for characteristics columns
            for sdrf_col in sdrf.columns:
                if col_lower in sdrf_col or sdrf_col in col_lower:
                    matched_col = sdrf_col
                    break

        if matched_col is None:
            logger.warning(f"Covariate column '{col}' not found in SDRF, skipping")
            continue

        # Create sample → value mapping
        sample_to_value = dict(zip(sdrf[sample_col], sdrf[matched_col]))

        # Get values for our samples in order
        values = []
        for sample_id in sample_ids:
            value = sample_to_value.get(sample_id)
            if value is None:
                # Try partial match
                for sdrf_sample in sdrf_samples:
                    if sample_id in sdrf_sample or sdrf_sample in sample_id:
                        value = sample_to_value.get(sdrf_sample)
                        break
            values.append(value if value is not None else "unknown")

        # Check if all values are the same (no information)
        unique_values = set(values)
        if len(unique_values) <= 1:
            logger.warning(
                f"Covariate '{col}' has only one unique value, skipping (no information)"
            )
            continue

        # Encode as categorical integers
        encoded, _ = pd.factorize(pd.array(values))
        covar_data.append(encoded.tolist())
        valid_columns.append(col)
        logger.info(f"Extracted covariate '{col}' with {len(unique_values)} unique values")

    if not covar_data:
        return None

    # Transpose: from [covariates][samples] to [samples][covariates]
    # pycombat expects covar_mod as (n_samples, n_covariates)
    n_samples = len(sample_ids)
    n_covariates = len(covar_data)
    result = [[covar_data[j][i] for j in range(n_covariates)] for i in range(n_samples)]

    logger.info(f"Extracted {n_covariates} covariates for {n_samples} samples: {valid_columns}")
    return result


def remove_single_sample_batches(df: pd.DataFrame, batch: list) -> pd.DataFrame:
    """Remove batches with only one sample."""
    batch_dict = dict(zip(df.columns, batch))
    single_sample_batch = [
        k for k, v in batch_dict.items() if list(batch_dict.values()).count(v) == 1
    ]
    df_single_batches_removed = df.drop(single_sample_batch, axis=1)
    return df_single_batches_removed


class TooFewSamplesInBatch(ValueError):
    def __init__(self, batches):
        super().__init__(
            f"Batches must contain at least two samples, the following batch factors did not: {batches}"
        )


def apply_batch_correction(
    df: pd.DataFrame,
    batch: List[int],
    covs: Optional[List[int]] = None,
    kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply batch correction using pycombat from inmoose.

    Note: Requires the optional 'inmoose' dependency.
    Install it with: pip install mokume[inmoose]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples as columns and features as rows.
    batch : List[int]
        Batch indices for each sample.
    covs : Optional[List[int]]
        Covariate indices for each sample.
    kwargs : Optional[dict]
        Additional arguments for pycombat_norm.

    Returns
    -------
    pd.DataFrame
        Batch-corrected DataFrame.

    Raises
    ------
    ImportError
        If inmoose is not installed.
    ValueError
        If sample counts don't match batch/covariate counts.
    TooFewSamplesInBatch
        If any batch has fewer than 2 samples.
    """
    if not is_inmoose_available():
        raise ImportError(
            "inmoose is required for batch correction but is not installed. "
            "Install it with: pip install mokume[inmoose]"
        )

    if kwargs is None:
        kwargs = {}

    if len(df.columns) != len(batch):
        raise ValueError(
            f"The number of samples should match the number of batch "
            f"indices. There were {len(batch)} batch indices and {len(df.columns)} samples"
        )

    if any([batch.count(i) < 2 for i in set(batch)]):
        short_batches = [i for i in set(batch) if batch.count(i) < 2]
        raise TooFewSamplesInBatch(short_batches)

    if covs:
        if len(df.columns) != len(covs):
            raise ValueError(
                f"The number of samples should match the number of covariates. "
                f"There were {len(covs)} batch indices and {len(df.columns)} samples"
            )

    from inmoose.pycombat import pycombat_norm

    df_co = pycombat_norm(counts=df, batch=batch, covar_mod=covs, **kwargs)
    return df_co


def find_clusters(df, min_cluster_size, min_samples) -> pd.DataFrame:
    """Compute clusters for a given dataframe using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=True,
        cluster_selection_epsilon=0.01,
    )
    clusterer.fit(df)
    df["cluster"] = clusterer.labels_
    return df


def iterative_outlier_removal(
    df: pd.DataFrame,
    batch: List[int],
    n_components: int = 5,
    min_cluster_size: int = 10,
    min_samples: int = 10,
    n_iter: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Iteratively remove outliers using PCA and HDBSCAN clustering."""
    batch_dict = dict(zip(df.columns, batch))

    # Check plotting availability once if verbose
    can_plot = verbose and is_plotting_available()
    if verbose and not can_plot:
        logger.warning(
            "Plotting skipped: plotting dependencies not installed. "
            "Install with: pip install mokume[plotting]"
        )

    for i in range(n_iter):
        logger.info("Running iteration: {}".format(i + 1))

        df_pca = compute_pca(df.T, n_components=n_components)
        df_clusters = find_clusters(
            df_pca, min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        logger.info(df_clusters)

        outliers = df_clusters[df_clusters["cluster"] == -1].index.tolist()
        df_filtered_outliers = df.drop(outliers, axis=1)
        logger.info(f"Number of outliers in iteration {i + 1}: {len(outliers)}")
        logger.info(f"Outliers in iteration {i + 1}: {str(outliers)}")

        batch_dict = {col: batch_dict[col] for col in df_filtered_outliers.columns}
        df = df_filtered_outliers

        if can_plot:
            from mokume.plotting import plot_pca

            plot_pca(
                df_clusters,
                output_file=f"iterative_outlier_removal_{i + 1}.png",
                x_col="PC1",
                y_col="PC2",
                hue_col="cluster",
                title=f"Iteration {i + 1}: Number of outliers: {len(outliers)}",
            )

        if len(outliers) == 0:
            break

    return df
