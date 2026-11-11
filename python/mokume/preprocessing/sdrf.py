"""
SDRF analysis utilities.

This module provides functions for analyzing SDRF (Sample and Data Relationship Format)
files to determine quantification details for proteomics experiments.
"""

from typing import Optional

import pandas as pd

from mokume.model.labeling import QuantificationCategory, IsobaricLabel
from mokume.core.constants import CHANNEL, load_sdrf


def analyse_sdrf(
    sdrf_path: str,
) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
    """
    Analyzes an SDRF file to determine quantification details.

    Parameters
    ----------
    sdrf_path : str
        The file path to the SDRF file.

    Returns
    -------
    tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]
        A tuple containing the number of technical repetitions, the quantification category,
        a list of unique sample names, and the isobaric label scheme if applicable.
    """
    sdrf_df = (
        sdrf_path.copy()
        if isinstance(sdrf_path, pd.DataFrame)
        else load_sdrf(sdrf_path)
    )

    labels = set(sdrf_df["comment[label]"])
    # Determine label type
    label, channel_set = QuantificationCategory.classify(labels)
    if label in (QuantificationCategory.TMT, QuantificationCategory.ITRAQ):
        choice_df = (
            pd.DataFrame.from_dict(
                channel_set.channels(), orient="index", columns=[CHANNEL]
            )
            .reset_index()
            .rename(columns={"index": "comment[label]"})
        )
        sdrf_df = sdrf_df.merge(choice_df, on="comment[label]", how="left")
    sample_names = sdrf_df["source name"].unique().tolist()
    technical_repetitions = len(sdrf_df["comment[technical replicate]"].unique())
    return technical_repetitions, label, sample_names, channel_set
