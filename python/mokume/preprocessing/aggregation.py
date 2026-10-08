"""
Data aggregation and filtering utilities for proteomics preprocessing.

This module provides functions for filtering, merging, and aggregating
peptide-level data before quantification.
"""

import re
from typing import Optional

import pandas as pd

from mokume.model.labeling import QuantificationCategory, IsobaricLabel
from mokume.core.constants import (
    BIOREPLICATE,
    TECHREPLICATE,
    CHANNEL,
    CONDITION,
    FRACTION,
    INTENSITY,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
    PEPTIDE_SEQUENCE,
    PROTEIN_NAME,
    RUN,
    SAMPLE_ID,
    parquet_map,
    AGGREGATION_LEVEL_SAMPLE,
    AGGREGATION_LEVEL_RUN,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.preprocessing.aggregation")


def parse_uniprot_accession(uniprot_id: str) -> str:
    """
    Parse a UniProt accession string to extract and return the core accession numbers.

    Parameters
    ----------
    uniprot_id : str
        A string containing one or more UniProt accessions.

    Returns
    -------
    str
        A semicolon-separated string of core accession numbers.
    """
    uniprot_list = uniprot_id.split(";")
    result_uniprot_list = []
    for accession in uniprot_list:
        if accession.count("|") == 2:
            accession = accession.split("|")[1]
        result_uniprot_list.append(accession)
    return ";".join(result_uniprot_list)


def get_canonical_peptide(peptide_sequence: str) -> str:
    """
    Remove modifications and special characters from a peptide sequence.

    Parameters
    ----------
    peptide_sequence : str
        The peptide sequence to be cleaned.

    Returns
    -------
    str
        The cleaned canonical peptide sequence.
    """
    clean_peptide = re.sub(r"[\(\[].*?[\)\]]", "", peptide_sequence)
    clean_peptide = clean_peptide.replace(".", "").replace("-", "")
    return clean_peptide


def remove_contaminants_entrapments_decoys(
    dataset: pd.DataFrame, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    Remove rows from the dataset that contain contaminants, entrapments, or decoys.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing protein data.
    protein_field : str
        The column name in the DataFrame to check for contaminants.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the contaminants, entrapments, and decoys removed.
    """
    from mokume.core.contaminants import (
        DEFAULT_CONTAMINANT_PATTERNS,
        contaminant_group_mask,
    )

    mask = contaminant_group_mask(
        dataset[protein_field], DEFAULT_CONTAMINANT_PATTERNS, case_sensitive=True
    )
    return dataset[~mask]


def remove_protein_by_ids(
    dataset: pd.DataFrame, protein_file: str, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    Remove proteins from a dataset based on a list of protein IDs.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset containing protein information.
    protein_file : str
        Path to the file containing protein IDs to be removed.
    protein_field : str
        The field in the dataset to check for protein IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the specified proteins removed.
    """
    with open(protein_file, "r") as contaminants_reader:
        contaminants = contaminants_reader.read().split("\n")
    contaminants = [cont for cont in contaminants if cont.strip()]
    if not contaminants:
        return dataset
    cregex = "|".join(re.escape(cont) for cont in contaminants)
    return dataset[~dataset[protein_field].str.contains(cregex, regex=True)]


def reformat_quantms_feature_table_quant_labels(
    data_df: pd.DataFrame,
    label: QuantificationCategory,
    choice: Optional[IsobaricLabel],
) -> pd.DataFrame:
    """
    Reformats a DataFrame containing quantification labels for QuantMS features.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing quantification data.
    label : QuantificationCategory
        The quantification category (e.g., LFQ, TMT, ITRAQ).
    choice : Optional[IsobaricLabel]
        The isobaric label scheme, if applicable.

    Returns
    -------
    pd.DataFrame
        The reformatted DataFrame with updated column names and channel information.
    """
    data_df = data_df.rename(columns=parquet_map)
    data_df[PROTEIN_NAME] = data_df[PROTEIN_NAME].str.join(";")
    if label == QuantificationCategory.LFQ:
        data_df.drop(CHANNEL, inplace=True, axis=1)
    else:
        data_df[CHANNEL] = data_df[CHANNEL].map(choice.channels())

    return data_df


def apply_initial_filtering(
    data_df: pd.DataFrame,
    min_aa: int,
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
) -> pd.DataFrame:
    """
    Apply initial filtering to a DataFrame containing peptide data.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing peptide data.
    min_aa : int
        The minimum number of amino acids required for peptides.
    aggregation_level : str
        Level at which to aggregate intensities. Options:
        - "sample": Aggregate at sample level (default)
        - "run": Aggregate at run level, preserving run information

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame with relevant columns.
    """
    # Remove 0 intensity signals from the data
    data_df = data_df[data_df[INTENSITY] > 0]

    data_df = data_df[
        (data_df["Condition"] != "Empty") | (data_df["Condition"].isnull())
    ]

    # "Run" is NA for reference files not found in the SDRF file.
    if data_df[RUN].isna().any():
        missing_files = (
            data_df.loc[data_df[RUN].isna(), "Reference"].drop_duplicates().tolist()
        )

        logger.warning(
            f"Reference files {missing_files} are not present in the SDRF file. Skipping calculation."
        )
        data_df.dropna(subset=[RUN], inplace=True)

    # Filter peptides with less amino acids than min_aa (default: 7)
    data_df = data_df[data_df[PEPTIDE_CANONICAL].str.len() >= min_aa]
    data_df[PROTEIN_NAME] = data_df[PROTEIN_NAME].apply(parse_uniprot_accession)
    if FRACTION not in data_df.columns:
        data_df[FRACTION] = 1

    # Try to extract technical replicate from run name
    try:
        if data_df[RUN].str.contains("_").all():
            # Get the last part after underscore (e.g., "S1_Brain_2" -> "2")
            last_parts = data_df[RUN].str.split("_").str.get(-1)
            data_df[TECHREPLICATE] = last_parts.astype("int")
        else:
            data_df[TECHREPLICATE] = data_df[RUN].astype("int")
    except (ValueError, TypeError):
        # Fall back to using run index
        unique_runs = data_df[RUN].unique().tolist()
        run_to_index = {run: i + 1 for i, run in enumerate(unique_runs)}
        data_df[TECHREPLICATE] = data_df[RUN].map(run_to_index)

    # Define columns to keep based on aggregation level
    columns_to_keep = [
        PROTEIN_NAME,
        PEPTIDE_SEQUENCE,
        PEPTIDE_CANONICAL,
        PEPTIDE_CHARGE,
        INTENSITY,
        CONDITION,
        TECHREPLICATE,
        BIOREPLICATE,
        FRACTION,
        SAMPLE_ID,
    ]

    # Include RUN column for run-level aggregation
    if aggregation_level == AGGREGATION_LEVEL_RUN:
        columns_to_keep.append(RUN)

    data_df = data_df[columns_to_keep]
    data_df[CONDITION] = pd.Categorical(data_df[CONDITION])
    data_df[SAMPLE_ID] = pd.Categorical(data_df[SAMPLE_ID])

    return data_df


def merge_fractions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fractions in the dataset by grouping and aggregating normalized intensity.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptide data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with merged fractions and maximum normalized intensity.
    """
    dataset = dataset.dropna(subset=[NORM_INTENSITY])
    dataset = dataset.groupby(
        [
            PROTEIN_NAME,
            PEPTIDE_SEQUENCE,
            PEPTIDE_CANONICAL,
            PEPTIDE_CHARGE,
            CONDITION,
            BIOREPLICATE,
            TECHREPLICATE,
            SAMPLE_ID,
        ],
        observed=True,
    ).agg({NORM_INTENSITY: "max"})
    dataset = dataset.reset_index()
    return dataset


def get_peptidoform_normalize_intensities(
    dataset: pd.DataFrame, higher_intensity: bool = True
) -> pd.DataFrame:
    """
    Normalize peptide intensities in a dataset by selecting the highest intensity.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptide data.
    higher_intensity : bool
        If True, selects the row with the highest normalized intensity for each group.

    Returns
    -------
    pd.DataFrame
        A DataFrame with normalized intensities.
    """
    dataset = dataset.dropna(subset=[NORM_INTENSITY])
    if higher_intensity:
        dataset = dataset.loc[
            dataset.groupby(
                [PEPTIDE_SEQUENCE, PEPTIDE_CHARGE, SAMPLE_ID, CONDITION, BIOREPLICATE],
                observed=True,
            )[NORM_INTENSITY].idxmax()
        ]
    dataset = dataset.reset_index(drop=True)
    return dataset


def sum_peptidoform_intensities(
    dataset: pd.DataFrame,
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
) -> pd.DataFrame:
    """
    Aggregate normalized intensities for each unique peptidoform.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptidoform data with normalized intensities.
    aggregation_level : str
        Level at which to aggregate intensities. Options:
        - "sample": Aggregate at sample level (default, original behavior)
        - "run": Aggregate at run level, preserving run information

    Returns
    -------
    pd.DataFrame
        A DataFrame with summed normalized intensities for each unique peptidoform entry.
    """
    dataset = dataset.dropna(subset=[NORM_INTENSITY])

    # Define columns based on aggregation level
    base_columns = [
        PROTEIN_NAME,
        PEPTIDE_CANONICAL,
        SAMPLE_ID,
        BIOREPLICATE,
        CONDITION,
        NORM_INTENSITY,
    ]

    groupby_columns = [
        PROTEIN_NAME,
        PEPTIDE_CANONICAL,
        SAMPLE_ID,
        BIOREPLICATE,
        CONDITION,
    ]

    # If run-level aggregation, include RUN/TECHREPLICATE columns
    if aggregation_level == AGGREGATION_LEVEL_RUN:
        if RUN in dataset.columns:
            base_columns.insert(-1, RUN)
            groupby_columns.append(RUN)
        if TECHREPLICATE in dataset.columns:
            base_columns.insert(-1, TECHREPLICATE)
            groupby_columns.append(TECHREPLICATE)

    dataset = dataset[[c for c in base_columns if c in dataset.columns]]

    dataset.loc[:, NORM_INTENSITY] = dataset.groupby(
        [c for c in groupby_columns if c in dataset.columns],
        observed=True,
    )[NORM_INTENSITY].transform("sum")
    dataset = dataset.drop_duplicates()
    dataset.reset_index(inplace=True, drop=True)
    return dataset
