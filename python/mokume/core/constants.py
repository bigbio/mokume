"""
Constants and common utilities for the mokume package.

This module defines column names, mappings, and utility functions used
throughout the package for data processing and analysis.
"""

import os
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

# Column name constants
PROTEIN_NAME = "ProteinName"
PEPTIDE_SEQUENCE = "PeptideSequence"
PEPTIDE_CANONICAL = "PeptideCanonical"
PEPTIDE_CHARGE = "PrecursorCharge"
CHANNEL = "Channel"
CONDITION = "Condition"
BIOREPLICATE = "BioReplicate"
TECHREPLICATE = "TechReplicate"
RUN = "Run"
FRACTION = "Fraction"
INTENSITY = "Intensity"
NORM_INTENSITY = "NormIntensity"
REFERENCE = "Reference"
SAMPLE_ID = "SampleID"
SAMPLE_ID_REGEX = r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$"
PIBAQ = "PiBAQ"
PIBAQ_NORMALIZED = "PiBAQNorm"
PIBAQ_LOG = "PiBAQLog"
PIBAQ_BEC = "PiBAQBec"
PIBAQ_PPB = "PiBAQPpb"
TPA = "TPA"
MOLECULARWEIGHT = "MolecularWeight"
# piBAQ family metadata. FamilyId holds the representative
# canonical accession (the family member with the most digested
# peptides); EvidenceLevel takes one of 'high' (every member reaches the
# high-anchor threshold), 'medium' (at least one member reaches the minimum),
# or 'family_only' (no member reaches the minimum);
# FamilySize is the count of canonical members in the family.
FAMILY_ID = "FamilyId"
EVIDENCE_LEVEL = "EvidenceLevel"
FAMILY_SIZE = "FamilySize"
EVIDENCE_HIGH = "high"
EVIDENCE_MEDIUM = "medium"
EVIDENCE_FAMILY_ONLY = "family_only"
COPYNUMBER = "CopyNumber"
CONCENTRATION_NM = "Concentration[nM]"
WEIGHT_NG = "Weight[ng]"
MOLES_NMOL = "Moles[nmol]"
GLOBALMEDIAN = "globalMedian"
CONDITIONMEDIAN = "conditionMedian"

# Aggregation level constants
AGGREGATION_LEVEL_SAMPLE = "sample"
AGGREGATION_LEVEL_RUN = "run"


# Parquet column names (QPX compatible)
PARQUET_COLUMNS = [
    "pg_accessions",
    "peptidoform",
    "sequence",
    "charge",
    "channel",
    "condition",
    "biological_replicate",
    "run",
    "fraction",
    "intensity",
    "run_file_name",
    "sample_accession",
    "peptide_qvalue",
    "pg_global_qvalue",
]


# Mapping from parquet column names to internal column names
parquet_map = {
    "pg_accessions": PROTEIN_NAME,
    "peptidoform": PEPTIDE_SEQUENCE,
    "sequence": PEPTIDE_CANONICAL,
    "charge": PEPTIDE_CHARGE,
    "channel": CHANNEL,
    "condition": CONDITION,
    "biological_replicate": BIOREPLICATE,
    "run": RUN,
    "fraction": FRACTION,
    "intensity": INTENSITY,
    "run_file_name": REFERENCE,
    "sample_accession": SAMPLE_ID,
}


def get_accession(identifier: str) -> str:
    """
    Get protein accession from the identifier.

    Supports multiple formats:
    - Standard UniProt: ``sp|P12345|PROT_NAME`` or ``tr|Q12345|PROT_NAME`` → ``P12345``
    - UPS database prefix: ``UPS|P02768ups|ALBU_HUMAN_UPS`` → ``P02768ups``
    - Non-standard 2-part: ``P02768ups|ALBU_HUMAN_UPS`` → ``P02768ups``
    - Plain accession: ``O13547`` → ``O13547``

    Parameters
    ----------
    identifier : str
        Protein identifier.

    Returns
    -------
    str
        Protein accession.
    """
    _DB_PREFIXES = {"sp", "tr", "sw", "nxp", "ups"}
    identifier_lst = identifier.split("|")
    if len(identifier_lst) == 1:
        return identifier_lst[0]
    if identifier_lst[0].lower() in _DB_PREFIXES:
        return identifier_lst[1]
    return identifier_lst[0]


def build_accession_map(
    proteins: Iterable[str],
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Build a mapping from normalized accessions to original protein names.

    Parameters
    ----------
    proteins : Iterable[str]
        Protein identifiers in any supported format.

    Returns
    -------
    tuple[dict[str, list[str]], set[str]]
        A tuple of (acc_to_originals dict, protein_accessions set).
    """
    acc_to_originals = {}
    for p in proteins:
        acc = get_accession(p)
        acc_to_originals.setdefault(acc, []).append(p)
    return acc_to_originals, set(acc_to_originals.keys())


# Functions needed by Combiner
def load_sdrf(sdrf_path: str) -> pd.DataFrame:
    """
    Load SDRF TSV as a dataframe.

    Parameters
    ----------
    sdrf_path : str
        Path to SDRF TSV.

    Returns
    -------
    pd.DataFrame
        Loaded SDRF data.
    """
    if not os.path.exists(sdrf_path):
        raise FileNotFoundError(f"{sdrf_path} does not exist!")
    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = [col.lower() for col in sdrf_df.columns]
    return sdrf_df


def load_feature(feature_path: str) -> pd.DataFrame:
    """
    Load feature file as a dataframe.

    Parameters
    ----------
    feature_path : str
        Path to feature file.

    Returns
    -------
    pd.DataFrame
        Loaded feature data.

    Raises
    ------
    ValueError
        If the provided file's suffix is not supported, either "parquet" or "csv".
    """
    suffix = os.path.splitext(feature_path)[1][1:]
    if suffix == "parquet":
        return pd.read_parquet(feature_path)
    elif suffix == "csv":
        return pd.read_csv(feature_path)
    else:
        raise ValueError(
            f"{suffix} is not allowed as input, please provide msstats_in or feature parquet."
        )


def is_parquet(path: str) -> bool:
    """
    Check if a file is in Parquet format.

    This function attempts to open the specified file and read its header
    to determine if it matches the Parquet file signature.

    Parameters
    ----------
    path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file is a Parquet file, False otherwise.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(4)
        return header == b"PAR1"
    except IOError:
        return False
