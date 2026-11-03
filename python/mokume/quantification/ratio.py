"""
Ratio-based quantification for TMT experiments.

Implements the Proteome Sciences (PS) post-processing protocol where
quantification operates in **ratio space**: PSM intensity / reference
intensity -> log2, with median aggregation up to protein level.

This is a self-contained pipeline (like DirectLFQ) that does NOT go
through the standard _load_and_process_peptides() -> _quantify() flow.

References
----------
Proteome Sciences post-processing protocol for TMT data.
"""

import duckdb
import numpy as np
import pandas as pd
import re

from mokume.core.constants import (
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
    SAMPLE_ID,
)
from mokume.core.logger import get_logger
from mokume.core.registry import PluginRegistry
from mokume.io._qpx import measured_intensity_clause
from mokume.io.feature import SQLFilterBuilder

logger = get_logger("mokume.quantification.ratio")


@PluginRegistry.register("quantification", "ratio")
class RatioQuantification:
    """
    Ratio-based TMT quantification following the PS protocol.

    Computes log2(sample / reference) per PSM per plex, then aggregates
    PSM -> peptide -> protein using median. The output is a wide protein
    x sample matrix of log2 ratios (already in log2 space).

    Parameters
    ----------
    reference_samples : list[str]
        Sample names (source names) that are reference channels.
    sample_to_plex : dict[str, str]
        Mapping from sample name to plex identifier.
    fraction_merge_method : str
        How to merge fractions: "mean" (the PS protocol behaviour, and the
        default) or "max". Compared verbatim, so it must be lowercase;
        QuantificationConfig folds the case for callers coming through the
        pipeline.
    """

    # Data level this method consumes; maps to the ``ratio`` flow in the
    # pipeline runner's ``FLOW_DISPATCH``. Ratio quantification is
    # instantiated directly by ``pipeline.flows.ratio`` (not via the
    # PluginRegistry), so this is pure dispatch metadata.
    input_level: str = "psms"

    def __init__(
        self,
        reference_samples: list[str],
        sample_to_plex: dict[str, str],
        fraction_merge_method: str = "mean",
    ):
        self.reference_samples = reference_samples
        self.sample_to_plex = sample_to_plex
        self.fraction_merge_method = fraction_merge_method

    def quantify(self, psm_long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full ratio quantification pipeline.

        Parameters
        ----------
        psm_long_df : pd.DataFrame
            Long-format PSM data with columns: ProteinName, PeptideSequence
            (or PeptideCanonical), PrecursorCharge, SampleID, Fraction, Intensity.

        Returns
        -------
        pd.DataFrame
            Wide protein x sample matrix of log2 ratios, with ProteinName
            as the first column.
        """
        # Normalize column names
        df = psm_long_df.copy()
        if "PeptideSequence" in df.columns and PEPTIDE_CANONICAL not in df.columns:
            df.rename(columns={"PeptideSequence": PEPTIDE_CANONICAL}, inplace=True)
        if "Fraction" not in df.columns:
            df["Fraction"] = "1"

        n_psms_before = len(df)
        logger.info("Ratio quantification: %d PSM rows", n_psms_before)

        # 1. Average fractions
        df = self._average_fractions(df)
        logger.info("After fraction averaging: %d rows", len(df))

        # 2. Compute reference intensity per plex per PSM group
        ref_df = self._compute_reference_intensity(df)

        # 3. Compute log2 ratios
        df = self._compute_log2_ratios(df, ref_df)
        logger.info("After log2 ratio computation: %d rows", len(df))

        # 4. Median aggregate PSM -> peptide (by sequence, ignoring charge)
        df = self._aggregate_psm_to_peptide(df)
        logger.info("After PSM->peptide aggregation: %d rows", len(df))

        # 5. Median aggregate peptide -> protein
        df = self._aggregate_peptide_to_protein(df)
        logger.info("After peptide->protein aggregation: %d rows", len(df))

        # 6. Pivot to wide format. observed=True avoids the pandas Cartesian
        # product blowup when SAMPLE_ID is Categorical (LoadingStage casts it
        # for legacy dtype parity).
        wide = df.pivot_table(
            index=PROTEIN_NAME,
            columns=SAMPLE_ID,
            values="log2ratio",
            aggfunc="median",
            observed=True,
        )

        n_proteins = len(wide)
        n_samples = len(wide.columns)
        logger.info(
            "Ratio quantification complete: %d proteins, %d samples",
            n_proteins,
            n_samples,
        )

        return wide.reset_index()

    def _average_fractions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average (or max) intensities across fractions per PSM per sample.

        Matches R's rowMeans(na.rm=TRUE) for the PS protocol.
        """
        group_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, PEPTIDE_CHARGE, SAMPLE_ID]

        if self.fraction_merge_method == "max":
            result = df.groupby(group_cols, as_index=False)["Intensity"].max()
        else:
            # mean — matches R's rowMeans(na.rm=TRUE)
            result = df.groupby(group_cols, as_index=False)["Intensity"].mean()

        return result

    def _compute_reference_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute reference intensity per plex per PSM group.

        For each plex, reference intensity = mean of reference channel
        intensities per (ProteinName, PeptideCanonical, PrecursorCharge).
        """
        # Assign plex to each row
        df["_plex"] = df[SAMPLE_ID].map(self.sample_to_plex)

        # Filter to reference samples only
        ref_mask = df[SAMPLE_ID].isin(self.reference_samples)
        ref_data = df[ref_mask]

        if len(ref_data) == 0:
            raise ValueError(
                f"No reference sample data found. Reference samples: {self.reference_samples}, "
                f"Available samples: {sorted(df[SAMPLE_ID].unique())}"
            )

        # Mean of reference channels per plex per PSM group
        group_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, PEPTIDE_CHARGE, "_plex"]
        ref_intensity = (
            ref_data.groupby(group_cols, as_index=False)["Intensity"]
            .mean()
            .rename(columns={"Intensity": "_ref_intensity"})
        )

        plexes = sorted(ref_intensity["_plex"].unique())
        for plex in plexes:
            plex_refs = [
                s for s in self.reference_samples if self.sample_to_plex.get(s) == plex
            ]
            logger.info("  Plex '%s': reference channels = %s", plex, plex_refs)

        return ref_intensity

    def _compute_log2_ratios(
        self, df: pd.DataFrame, ref_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute log2(sample_intensity / reference_intensity) per PSM per plex."""
        df["_plex"] = df[SAMPLE_ID].map(self.sample_to_plex)

        # Remove reference samples from the output (log2(ref/ref) ~ 0)
        non_ref_mask = ~df[SAMPLE_ID].isin(self.reference_samples)
        df = df[non_ref_mask]

        # Merge reference intensity
        merge_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, PEPTIDE_CHARGE, "_plex"]
        df = df.merge(ref_df, on=merge_cols, how="left")

        # Replace zeros with NaN before division
        df["Intensity"] = df["Intensity"].replace(0, np.nan)
        df["_ref_intensity"] = df["_ref_intensity"].replace(0, np.nan)

        # Compute ratio and log2
        ratio = df["Intensity"] / df["_ref_intensity"]
        df["log2ratio"] = np.log2(ratio)

        # Replace inf/-inf with NaN
        df["log2ratio"] = df["log2ratio"].replace([np.inf, -np.inf], np.nan)

        # Drop helper columns
        df = df.drop(columns=["_plex", "_ref_intensity", "Intensity"])

        # Drop rows with NaN log2ratio
        df = df.dropna(subset=["log2ratio"])

        return df

    def _aggregate_psm_to_peptide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median aggregate PSM -> peptide (by sequence, ignoring charge)."""
        group_cols = [PROTEIN_NAME, PEPTIDE_CANONICAL, SAMPLE_ID]
        result = df.groupby(group_cols, as_index=False)["log2ratio"].median()
        return result

    def _aggregate_peptide_to_protein(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median aggregate peptide -> protein."""
        group_cols = [PROTEIN_NAME, SAMPLE_ID]
        result = df.groupby(group_cols, as_index=False)["log2ratio"].median()
        return result


def apply_coverage_filter(
    protein_df: pd.DataFrame,
    sample_to_condition: dict[str, str],
    threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Remove proteins with insufficient coverage in any condition.

    A protein passes if, in ALL conditions, the fraction of non-missing
    (and non-zero) values >= threshold.

    Matches R logic:
        all(sapply(gene_by_group, \\(gv) sum(!is.na(gv) & gv != 0) / length(gv) >= 0.65))

    Parameters
    ----------
    protein_df : pd.DataFrame
        Wide-format protein matrix (ProteinName | sample1 | sample2 | ...).
    sample_to_condition : dict
        Mapping from sample name to condition.
    threshold : float
        Minimum fraction of non-missing values required per condition.

    Returns
    -------
    pd.DataFrame
        Filtered protein matrix.
    """
    protein_col = protein_df.columns[0]
    sample_cols = [c for c in protein_df.columns if c != protein_col]

    # Group sample columns by condition
    condition_samples = {}
    for sample in sample_cols:
        cond = sample_to_condition.get(sample)
        if cond is not None:
            condition_samples.setdefault(cond, []).append(sample)

    if not condition_samples:
        logger.warning("No condition mapping for samples, skipping coverage filter")
        return protein_df

    intensity_matrix = protein_df.set_index(protein_col)

    # For each protein, check coverage in each condition
    keep_mask = pd.Series(True, index=intensity_matrix.index)

    for cond, samples in condition_samples.items():
        cond_data = intensity_matrix[samples]
        # Count non-missing and non-zero values
        valid_count = ((cond_data.notna()) & (cond_data != 0)).sum(axis=1)
        coverage = valid_count / len(samples)
        keep_mask = keep_mask & (coverage >= threshold)

    n_before = len(protein_df)
    result = protein_df[keep_mask.values]
    n_after = len(result)

    logger.info(
        "Coverage filter (%.0f%%): %d -> %d proteins (%d removed)",
        threshold * 100,
        n_before,
        n_after,
        n_before - n_after,
    )

    return result.reset_index(drop=True)


def _strip_raw_ext(name: str) -> str:
    """Normalize a run file path/name before SDRF matching."""
    return re.sub(
        r"\.(raw|mzML|mzml|d|wiff|RAW)$",
        "",
        str(name).replace("\\", "/").rsplit("/", maxsplit=1)[-1],
    )


def _load_sdrf(sdrf_path: str) -> pd.DataFrame:
    """Load SDRF with lower-cased column names."""
    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = [c.lower() for c in sdrf_df.columns]
    return sdrf_df


def _parquet_columns(conn: duckdb.DuckDBPyConnection, parquet_path: str) -> list[str]:
    """Return QPX parquet column names via DuckDB."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [parquet_path],
        ).fetchall()
    ]


def _pg_accessions_projection(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    columns: list[str],
) -> str:
    """Return the SQL projection for pg_accessions across QPX formats."""
    if "pg_accessions" not in columns:
        return "pg_accessions"

    try:
        type_str = (
            conn.execute(
                "SELECT typeof(pg_accessions) FROM read_parquet(?) LIMIT 1",
                [parquet_path],
            )
            .fetchone()[0]
            .lower()
        )
    except Exception as exc:
        logger.debug("Could not detect pg_accessions type: %s", exc)
        return "pg_accessions"

    if "struct" in type_str:
        return "list_transform(pg_accessions, lambda x: x.accession) as pg_accessions"
    return "pg_accessions"


def _qpx_psm_base_query(pg_col: str, is_new_qpx: bool) -> str:
    """Return the trusted base query for the detected QPX layout."""
    if is_new_qpx:
        return "".join(
            [
                "SELECT ",
                pg_col,
                ", sequence,",
                " charge as precursor_charge,",
                " run_file_name as run_file_name,",
                " unnest.label as label,",
                " unnest.intensity as intensity",
                " FROM (SELECT *, UNNEST(intensities) AS unnest FROM read_parquet(?))",
                " WHERE unnest.intensity IS NOT NULL AND ",
            ]
        )

    return "".join(
        [
            "SELECT ",
            pg_col,
            ", sequence,",
            " precursor_charge as precursor_charge,",
            " unnest.sample_accession as sample_accession,",
            " reference_file_name as run_file_name,",
            " unnest.channel as label,",
            " unnest.intensity as intensity",
            " FROM (SELECT *, UNNEST(intensities) AS unnest FROM read_parquet(?))",
            " WHERE unnest.intensity IS NOT NULL AND ",
        ]
    )


def _load_qpx_psm_rows(
    parquet_path: str,
    filter_builder: SQLFilterBuilder,
) -> tuple[pd.DataFrame, bool]:
    """Load filtered long-format PSM rows from QPX parquet."""
    conn = duckdb.connect()
    try:
        columns = _parquet_columns(conn, parquet_path)
        is_new_qpx = "charge" in columns or "run_file_name" in columns
        if "is_decoy" in columns:
            filter_builder.has_is_decoy = True

        where_clause, where_params = filter_builder.build_where_clause()
        pg_col = _pg_accessions_projection(conn, parquet_path, columns)
        measured = measured_intensity_clause(columns)
        query = "".join(
            (_qpx_psm_base_query(pg_col, is_new_qpx), measured, " AND ", where_clause)
        )
        return conn.execute(query, [parquet_path] + where_params).df(), is_new_qpx
    finally:
        conn.close()


def _sdrf_run_metadata(sdrf_df: pd.DataFrame) -> pd.DataFrame:
    """Build run/label to sample/fraction metadata from SDRF."""
    sdrf_run_file = (
        sdrf_df["comment[data file]"].apply(_strip_raw_ext)
        if "comment[data file]" in sdrf_df.columns
        else sdrf_df.get("source name", pd.Series())
    )
    sdrf_label = sdrf_df.get("comment[label]", pd.Series(dtype=str))
    sdrf_fraction = sdrf_df.get(
        "comment[fraction identifier]", pd.Series("1", index=sdrf_df.index)
    )
    return pd.DataFrame(
        {
            "run_file_name": sdrf_run_file.values,
            "label": sdrf_label.values if len(sdrf_label) > 0 else "",
            "source_name": sdrf_df["source name"].values,
            "fraction": sdrf_fraction.values,
        }
    )


def _legacy_fraction_map(sdrf_df: pd.DataFrame) -> dict:
    """Build run-file to fraction mapping for legacy QPX rows."""
    fraction_map = {}
    if "comment[fraction identifier]" not in sdrf_df.columns:
        return fraction_map

    for _, row in sdrf_df.iterrows():
        for col in ["comment[data file]", "comment[spectrum file]"]:
            if col in sdrf_df.columns and pd.notna(row.get(col)):
                fraction_map[row[col]] = str(row["comment[fraction identifier]"])
                break
    return fraction_map


def _apply_sdrf_metadata(
    df: pd.DataFrame,
    sdrf_df: pd.DataFrame,
    is_new_qpx: bool,
) -> pd.DataFrame:
    """Attach sample accession and fraction columns from SDRF metadata."""
    if is_new_qpx:
        df = df.merge(
            _sdrf_run_metadata(sdrf_df),
            on=["run_file_name", "label"],
            how="left",
        )
        df["sample_accession"] = df["source_name"].fillna(df["run_file_name"])
        df["Fraction"] = df["fraction"].fillna("1")
        return df

    fraction_map = _legacy_fraction_map(sdrf_df)
    if fraction_map:
        df["Fraction"] = df["run_file_name"].map(fraction_map).fillna("1")
    else:
        df["Fraction"] = "1"
    return df


def _finalize_psm_data(df: pd.DataFrame, min_unique_peptides: int) -> pd.DataFrame:
    """Rename, select and filter PSM rows for ratio quantification."""
    first_acc = df["pg_accessions"].str[0].fillna("")
    df[PROTEIN_NAME] = np.where(
        first_acc.str.contains("|", regex=False),
        first_acc.str.split("|").str[1],
        first_acc,
    )
    df[PEPTIDE_CANONICAL] = df["sequence"]
    df[PEPTIDE_CHARGE] = df["precursor_charge"]
    df[SAMPLE_ID] = df["sample_accession"]
    df["Intensity"] = df["intensity"]

    df = df[
        [
            PROTEIN_NAME,
            PEPTIDE_CANONICAL,
            PEPTIDE_CHARGE,
            SAMPLE_ID,
            "Fraction",
            "Intensity",
        ]
    ]
    df = df[~df[PROTEIN_NAME].str.contains(";")]

    peptide_counts = df.groupby(PROTEIN_NAME)[PEPTIDE_CANONICAL].nunique()
    valid_proteins = peptide_counts[peptide_counts >= min_unique_peptides].index
    return df[df[PROTEIN_NAME].isin(valid_proteins)]


def _log_psm_summary(df: pd.DataFrame) -> None:
    """Log loaded PSM matrix dimensions."""
    logger.info(
        "Loaded PSM data: %d rows, %d proteins, %d peptides, %d samples",
        len(df),
        df[PROTEIN_NAME].nunique(),
        df[PEPTIDE_CANONICAL].nunique(),
        df[SAMPLE_ID].nunique(),
    )


def load_psm_data(
    parquet_path: str,
    sdrf_path: str,
    min_aa: int = 7,
    min_unique_peptides: int = 2,
    remove_contaminants: bool = True,
) -> pd.DataFrame:
    """
    Load PSM data from QPX parquet, unnest intensities, join SDRF metadata.

    Uses DuckDB directly following the existing Feature class pattern.

    Parameters
    ----------
    parquet_path : str
        Path to the QPX parquet file.
    sdrf_path : str
        Path to the SDRF TSV file.
    min_aa : int
        Minimum peptide sequence length.
    min_unique_peptides : int
        Minimum unique peptides per protein.
    remove_contaminants : bool
        Whether to remove contaminants and decoys.

    Returns
    -------
    pd.DataFrame
        Long-format PSM data with columns: ProteinName, PeptideCanonical,
        PrecursorCharge, SampleID, Fraction, Intensity.
    """
    filter_builder = SQLFilterBuilder(
        remove_contaminants=remove_contaminants,
        min_peptide_length=min_aa,
        require_unique=True,
    )
    sdrf_df = _load_sdrf(sdrf_path)
    df, is_new_qpx = _load_qpx_psm_rows(parquet_path, filter_builder)

    if len(df) == 0:
        raise ValueError("No PSM data after filtering. Check parquet file and filters.")

    df = _apply_sdrf_metadata(df, sdrf_df, is_new_qpx)
    df = _finalize_psm_data(df, min_unique_peptides)
    _log_psm_summary(df)
    return df
