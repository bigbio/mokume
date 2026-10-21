"""
Internal Reference Scaling (IRS) normalization for multi-plex TMT experiments.

IRS corrects between-plex batch effects by scaling protein intensities so that
reference channels (pooled samples present across all plexes) have equal
intensity across plexes.

References
----------
Plubell et al. (2017). Extended Multiplexing of TMT Labeling Reveals Age and
High Fat Diet Specific Proteome Changes in Mouse Epididymal Adipose Tissue.
Molecular & Cellular Proteomics, 16(5), 873-890.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

from mokume.core.constants import load_sdrf
from mokume.core.logger import get_logger

logger = get_logger("mokume.normalization.irs")


class IRSNormalizer:
    """
    Internal Reference Scaling normalizer for multi-plex TMT data.

    For each protein, IRS computes the reference intensity per plex (using
    reference channels), then scales all samples in each plex so that
    reference intensities are equalized across plexes.

    Parameters
    ----------
    reference_samples : list[str]
        Sample names (source names) that are reference channels.
    stat : str
        Statistic to compute reference intensity: "median" or "mean".
    """

    def __init__(self, reference_samples: list[str], stat: str = "median"):
        self.reference_samples = reference_samples
        self.stat = stat
        self.scaling_factors_ = None

    def fit(self, protein_df: pd.DataFrame, sample_to_plex: dict[str, str]):
        """
        Compute per-protein, per-plex IRS scaling factors.

        Parameters
        ----------
        protein_df : pd.DataFrame
            Wide-format protein matrix with a protein column and sample columns.
        sample_to_plex : dict
            Mapping from sample name to plex identifier.
        """
        protein_col = protein_df.columns[0]
        sample_cols = [c for c in protein_df.columns if c != protein_col]

        # Build plex->reference_samples mapping
        plexes = sorted(set(sample_to_plex.values()))
        plex_ref_samples = {}
        for plex in plexes:
            refs = [s for s in self.reference_samples if sample_to_plex.get(s) == plex]
            if not refs:
                raise ValueError(
                    f"No reference samples found in plex '{plex}'. "
                    f"Reference samples: {self.reference_samples}, "
                    f"Plex samples: {[s for s, p in sample_to_plex.items() if p == plex]}"
                )
            plex_ref_samples[plex] = refs

        logger.info(
            "IRS fit: %d plexes, %d reference samples, stat=%s",
            len(plexes),
            len(self.reference_samples),
            self.stat,
        )
        for plex, refs in plex_ref_samples.items():
            logger.info("  Plex '%s': reference channels = %s", plex, refs)

        # Compute reference intensity per protein per plex
        intensity_matrix = protein_df.set_index(protein_col)[sample_cols]
        plex_ref_intensity = {}

        for plex, refs in plex_ref_samples.items():
            ref_cols = [r for r in refs if r in intensity_matrix.columns]
            if not ref_cols:
                raise ValueError(
                    f"Reference samples {refs} not found in protein matrix columns"
                )
            # Non-positive intensities are missing: additive methods such as
            # DirectLFQ store a missing cell as 0.
            ref_data = intensity_matrix[ref_cols].where(intensity_matrix[ref_cols] > 0)
            if self.stat == "median":
                plex_ref_intensity[plex] = ref_data.median(axis=1)
            else:
                plex_ref_intensity[plex] = ref_data.mean(axis=1)

        # Compute global reference intensity (geometric mean across plexes)
        ref_df = pd.DataFrame(plex_ref_intensity)
        # Replace 0 and negatives with NaN for geometric mean
        ref_df[ref_df <= 0] = np.nan
        log_refs = np.log(ref_df)
        global_ref = np.exp(log_refs.mean(axis=1))

        # Compute scaling factors: global_ref / plex_ref for each plex
        scaling_factors = {}
        for plex in plexes:
            scaling_factors[plex] = global_ref / plex_ref_intensity[plex]

        self.scaling_factors_ = scaling_factors
        self._sample_to_plex = sample_to_plex
        self._protein_col = protein_col

        return self

    def transform(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply IRS scaling to protein intensities.

        Parameters
        ----------
        protein_df : pd.DataFrame
            Wide-format protein matrix.

        Returns
        -------
        pd.DataFrame
            IRS-normalized protein matrix.
        """
        if self.scaling_factors_ is None:
            raise RuntimeError("Must call fit() before transform()")

        protein_col = self._protein_col
        result = protein_df.copy()
        result = result.set_index(protein_col)

        for sample, plex in self._sample_to_plex.items():
            if sample in result.columns and plex in self.scaling_factors_:
                result[sample] = result[sample] * self.scaling_factors_[plex]

        return result.reset_index()

    def fit_transform(
        self, protein_df: pd.DataFrame, sample_to_plex: dict[str, str]
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(protein_df, sample_to_plex).transform(protein_df)


# ---------------------------------------------------------------------------
# SDRF helper functions for reference channel and plex detection
# ---------------------------------------------------------------------------


def detect_pooled_from_sdrf(sdrf_path: str) -> Optional[list[str]]:
    """
    Detect reference samples from SDRF ``characteristics[pooled sample]`` column.

    Per the SDRF-Proteomics specification, samples with value ``pooled`` or
    ``SN=sample1;SN=sample2;...`` are reference channels.

    Parameters
    ----------
    sdrf_path : str
        Path to the SDRF TSV file.

    Returns
    -------
    list[str] or None
        List of reference sample source names, or None if column not present.
    """
    sdrf = load_sdrf(sdrf_path)

    pooled_col = "characteristics[pooled sample]"
    if pooled_col not in sdrf.columns:
        return None

    # Pooled if value is "pooled" or starts with "SN=" (not "not pooled")
    lowered = sdrf[pooled_col].str.lower()
    mask = (lowered == "pooled") | (lowered.str.startswith("sn=", na=False))
    ref_samples = sdrf.loc[mask, "source name"].unique().tolist()

    if ref_samples:
        logger.info(
            "Detected %d pooled reference samples from SDRF '%s' column: %s",
            len(ref_samples),
            pooled_col,
            ref_samples,
        )
        return ref_samples

    return None


def detect_reference_by_column(
    sdrf_path: str, column: str, values: list[str]
) -> list[str]:
    """
    Detect reference samples by matching values in a specific SDRF column.

    Parameters
    ----------
    sdrf_path : str
        Path to the SDRF TSV file.
    column : str
        SDRF column name (e.g., ``"factor value[disease]"``).
    values : list[str]
        Values that indicate a reference sample.

    Returns
    -------
    list[str]
        List of reference sample source names.
    """
    sdrf = load_sdrf(sdrf_path)
    column_lower = column.lower()

    if column_lower not in sdrf.columns:
        raise ValueError(
            f"Column '{column}' not found in SDRF. Available: {list(sdrf.columns)}"
        )

    values_lower = [v.lower() for v in values]
    mask = sdrf[column_lower].str.lower().isin(values_lower)
    ref_samples = sdrf.loc[mask, "source name"].unique().tolist()

    logger.info(
        "Detected %d reference samples from column '%s' with values %s: %s",
        len(ref_samples),
        column,
        values,
        ref_samples,
    )
    return ref_samples


def detect_reference_by_regex(
    sdrf_path: str, regex: str = r"pool|powder|ref|reference|bridge"
) -> list[str]:
    """
    Detect reference samples by regex scan across all factor/characteristic columns.

    Parameters
    ----------
    sdrf_path : str
        Path to the SDRF TSV file.
    regex : str
        Regex pattern to match reference sample values.

    Returns
    -------
    list[str]
        List of reference sample source names.
    """
    sdrf = load_sdrf(sdrf_path)

    # Scan all factor value[*] and characteristics[*] columns
    scan_cols = [
        c
        for c in sdrf.columns
        if c.startswith("factor value[") or c.startswith("characteristics[")
    ]

    pattern = re.compile(regex, re.IGNORECASE)
    ref_mask = pd.Series(False, index=sdrf.index)

    for col in scan_cols:
        col_match = sdrf[col].astype(str).str.contains(pattern, na=False)
        ref_mask = ref_mask | col_match

    ref_samples = sdrf.loc[ref_mask, "source name"].unique().tolist()

    logger.info(
        "Detected %d reference samples by regex '%s' across %d columns: %s",
        len(ref_samples),
        regex,
        len(scan_cols),
        ref_samples,
    )
    return ref_samples


def detect_plexes_from_sdrf(sdrf_path: str) -> dict[str, str]:
    """
    Detect plex membership from SDRF ``source name`` prefix convention.

    Parses quantms-style source names (e.g., ``p1_1`` -> plex ``p1``,
    ``p2_1`` -> plex ``p2``).

    Parameters
    ----------
    sdrf_path : str
        Path to the SDRF TSV file.

    Returns
    -------
    dict[str, str]
        Mapping from sample name (source name) to plex identifier.
    """
    sdrf = load_sdrf(sdrf_path)

    sample_to_plex = {}
    for source_name in sdrf["source name"].unique():
        mixture = re.search(r"(?:^|_)(mixture\d+)(?:_|$)", source_name, re.IGNORECASE)
        if mixture:
            sample_to_plex[source_name] = mixture.group(1).lower()
            continue
        # Parse prefix: p1_1 -> p1, p2_10 -> p2
        match = re.match(r"^(p\d+)_\d+[NC]?$", source_name, re.IGNORECASE)
        if match:
            sample_to_plex[source_name] = match.group(1)
        else:
            # Fallback: use everything before the last underscore+digit
            parts = source_name.rsplit("_", 1)
            if len(parts) == 2 and re.fullmatch(r"\d+[NC]?", parts[1], re.IGNORECASE):
                sample_to_plex[source_name] = parts[0]
            else:
                sample_to_plex[source_name] = "plex1"
                logger.warning(
                    "Could not parse plex from source name '%s', assigning to 'plex1'",
                    source_name,
                )

    plexes = sorted(set(sample_to_plex.values()))
    logger.info("Detected %d plexes: %s", len(plexes), plexes)
    for plex in plexes:
        members = [s for s, p in sample_to_plex.items() if p == plex]
        logger.info("  Plex '%s': %s", plex, sorted(members))

    return sample_to_plex


def _shorten_factor_label(raw_label: str) -> str:
    """Shorten verbose SDRF ontology-annotated factor value labels.

    Handles formats like
    ``CT=mixture;QY=12500 amol;CN=UPS1;CV=Standards Research Group``
    and returns the most informative short token (e.g. ``12500 amol``).
    Plain-text labels are returned unchanged.
    """
    if ";" not in raw_label or "=" not in raw_label:
        return raw_label
    parts: dict[str, str] = {}
    for token in raw_label.split(";"):
        if "=" in token:
            k, v = token.split("=", 1)
            parts[k.strip()] = v.strip()
    for key in ("QY", "CN", "NT"):
        if key in parts:
            return parts[key]
    return next(iter(parts.values()), raw_label)


def detect_condition_from_sdrf(
    sdrf_path: str,
    factor_col: str | None = None,
) -> dict[str, str]:
    """
    Detect sample-to-condition mapping from an SDRF column.

    By default the first ``factor value[*]`` column is used. Pass
    ``factor_col`` to point at any other column (e.g. ``"characteristics[
    disease]"``) when the dataset's primary factor is not biologically
    informative for the contrast of interest (this happens with cell-line
    panels where ``factor value[cell line]`` has only one sample per group).

    Parameters
    ----------
    sdrf_path : str
        Path to the SDRF TSV file.
    factor_col : str, optional
        Explicit SDRF column to use as the condition source. When ``None``
        (default), the first ``factor value[*]`` column is used.

    Returns
    -------
    dict[str, str]
        Mapping from sample/run name to condition.  Keys include source
        names and, when ``comment[data file]`` is present, run file names
        (with and without extension, both original and upper-cased) so that
        the mapping works regardless of whether downstream code uses source
        names or run identifiers as column headers.
    """
    sdrf = load_sdrf(sdrf_path)

    if factor_col is None:
        factor_cols = [c for c in sdrf.columns if c.startswith("factor value[")]
        if not factor_cols:
            raise ValueError("No 'factor value[*]' columns found in SDRF")
        factor_col = factor_cols[0]
    elif factor_col not in sdrf.columns:
        raise ValueError(
            f"Column '{factor_col}' not in SDRF; available factor/characteristics: "
            + ", ".join(
                c
                for c in sdrf.columns
                if c.startswith("factor value[") or c.startswith("characteristics[")
            )
        )
    sample_to_condition = {}
    for _, row in sdrf.drop_duplicates(subset=["source name"]).iterrows():
        sample_to_condition[row["source name"]] = _shorten_factor_label(
            str(row[factor_col])
        )

    # Also map run file names to conditions for run-level protein matrices
    data_file_col = "comment[data file]"
    if data_file_col in sdrf.columns:
        for _, row in sdrf.iterrows():
            condition = _shorten_factor_label(str(row[factor_col]))
            raw_name = str(row[data_file_col])
            # Strip any of the common MS raw-file extensions (single or
            # archive-wrapped) so the stem matches run-level matrix columns.
            stem = re.sub(
                r"\.(raw|mzML|mzXML|mgf|wiff|wiff2|wiff\.scan|baf|yep|fid|d|d\.zip|raw\.zip)$",
                "",
                raw_name,
                flags=re.IGNORECASE,
            )
            for variant in (raw_name, stem, raw_name.upper(), stem.upper()):
                if variant and variant not in sample_to_condition:
                    sample_to_condition[variant] = condition

    logger.info(
        "Condition mapping from '%s': %d entries",
        factor_col,
        len(sample_to_condition),
    )
    return sample_to_condition
