"""
Peptide-level preprocessing filters.
"""

import re
from typing import Tuple, List

import pandas as pd

from mokume.core.logger import get_logger
from mokume.core.constants import (
    PEPTIDE_SEQUENCE,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
)
from mokume.preprocessing.filters.base import BaseFilter, FilterResult
from mokume.preprocessing.filters.enums import FilterLevel


logger = get_logger("mokume.preprocessing.filters.peptide")
PEPTIDE_QVALUE = "peptide_qvalue"


class PeptideLengthFilter(BaseFilter):
    """Filter peptides by length."""

    def __init__(
        self,
        min_length: int = 7,
        max_length: int = 50,
        sequence_column: str = PEPTIDE_CANONICAL,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        min_length : int, optional
            Minimum peptide length in amino acids.
        max_length : int, optional
            Maximum peptide length in amino acids.
        sequence_column : str, optional
            Column name containing peptide sequences.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.sequence_column = sequence_column

    @property
    def name(self) -> str:
        return "PeptideLengthFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        col = self.sequence_column
        if col not in df.columns and PEPTIDE_CANONICAL in df.columns:
            col = PEPTIDE_CANONICAL
        elif col not in df.columns and PEPTIDE_SEQUENCE in df.columns:
            col = PEPTIDE_SEQUENCE

        if col not in df.columns:
            logger.warning("%s: Sequence column not found, skipping filter", self.name)
            return df, self._create_result(input_count, input_count)

        # Calculate sequence length (removing modifications if present)
        def get_aa_length(seq):
            if pd.isna(seq):
                return 0
            # Remove common modification patterns
            clean_seq = re.sub(r"\[.*?\]", "", str(seq))
            clean_seq = re.sub(r"\(.*?\)", "", clean_seq)
            return len(clean_seq)

        lengths = df[col].apply(get_aa_length)
        mask = (lengths >= self.min_length) & (lengths <= self.max_length)
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d peptides outside length range [%d, %d]",
            self.name,
            input_count - output_count,
            self.min_length,
            self.max_length,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {"min_length": self.min_length, "max_length": self.max_length},
        )


class ChargeStateFilter(BaseFilter):
    """Filter peptides by charge state."""

    def __init__(
        self,
        allowed_charges: List[int],
        charge_column: str = PEPTIDE_CHARGE,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        allowed_charges : list[int]
            List of allowed charge states (e.g., [2, 3, 4]).
        charge_column : str, optional
            Column name containing charge state values.
        """
        self.allowed_charges = allowed_charges
        self.charge_column = charge_column

    @property
    def name(self) -> str:
        return "ChargeStateFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        if self.charge_column not in df.columns:
            logger.warning(
                "%s: Charge column '%s' not found, skipping filter",
                self.name,
                self.charge_column,
            )
            return df, self._create_result(input_count, input_count)

        mask = df[self.charge_column].isin(self.allowed_charges)
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d peptides with charge not in %s",
            self.name,
            input_count - output_count,
            self.allowed_charges,
        )

        return filtered_df, self._create_result(
            input_count, output_count, {"allowed_charges": self.allowed_charges}
        )


def _modification_pattern(value: str) -> str:
    """Accessions match complete bracket tokens in legacy peptidoforms."""
    if re.fullmatch(r"[^:\s]+:[0-9]+", value):
        value = f"[{value}]"
    return re.escape(value)


class ModificationFilter(BaseFilter):
    """Filter peptides by modifications."""

    def __init__(
        self,
        exclude_modifications: List[str],
        sequence_column: str = PEPTIDE_SEQUENCE,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        exclude_modifications : list[str]
            List of modification names to exclude (e.g., ["Oxidation", "Deamidated"]).
        sequence_column : str, optional
            Column name containing modified peptide sequences.
        """
        self.exclude_modifications = exclude_modifications
        self.sequence_column = sequence_column

    @property
    def name(self) -> str:
        return "ModificationFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        if self.sequence_column not in df.columns:
            logger.warning(
                "%s: Sequence column '%s' not found, skipping filter",
                self.name,
                self.sequence_column,
            )
            return df, self._create_result(input_count, input_count)

        # Build pattern for modifications to exclude
        if not self.exclude_modifications:
            return df, self._create_result(input_count, input_count)

        patterns = "|".join(
            _modification_pattern(mod) for mod in self.exclude_modifications
        )

        mask = ~df[self.sequence_column].fillna("").astype(str).str.contains(
            patterns, case=False, regex=True
        )
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d peptides with modifications %s",
            self.name,
            input_count - output_count,
            self.exclude_modifications,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {"exclude_modifications": self.exclude_modifications},
        )


class MissedCleavageFilter(BaseFilter):
    """Filter peptides by number of missed cleavages."""

    def __init__(
        self,
        max_missed_cleavages: int,
        sequence_column: str = PEPTIDE_CANONICAL,
        enzyme: str = "trypsin",
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        max_missed_cleavages : int
            Maximum number of missed cleavages allowed.
        sequence_column : str, optional
            Column name containing peptide sequences.
        enzyme : str, optional
            Enzyme used for digestion (currently only trypsin supported).
        """
        self.max_missed_cleavages = max_missed_cleavages
        self.sequence_column = sequence_column
        self.enzyme = enzyme.lower()

    @property
    def name(self) -> str:
        return "MissedCleavageFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        col = self.sequence_column
        if col not in df.columns and PEPTIDE_CANONICAL in df.columns:
            col = PEPTIDE_CANONICAL

        if col not in df.columns:
            logger.warning("%s: Sequence column not found, skipping filter", self.name)
            return df, self._create_result(input_count, input_count)

        def count_missed_cleavages(seq):
            if pd.isna(seq):
                return 0
            seq_str = str(seq).upper()
            # Remove last character (C-terminal cleavage site doesn't count)
            if len(seq_str) > 1:
                seq_str = seq_str[:-1]

            if self.enzyme == "trypsin":
                # Count K and R not followed by P
                count = 0
                for i, aa in enumerate(seq_str):
                    if aa in ["K", "R"]:
                        # Check if followed by P
                        if i + 1 < len(seq_str) and seq_str[i + 1] != "P":
                            count += 1
                        elif i + 1 >= len(seq_str):
                            # Last position doesn't count as missed
                            pass
                return count
            else:
                logger.warning("Enzyme '%s' not supported", self.enzyme)
                return 0

        missed = df[col].apply(count_missed_cleavages)
        mask = missed <= self.max_missed_cleavages
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d peptides with > %d missed cleavages",
            self.name,
            input_count - output_count,
            self.max_missed_cleavages,
        )

        return filtered_df, self._create_result(
            input_count,
            output_count,
            {"max_missed_cleavages": self.max_missed_cleavages, "enzyme": self.enzyme},
        )


class SequencePatternFilter(BaseFilter):
    """Filter peptides by sequence patterns (regex)."""

    def __init__(
        self,
        exclude_patterns: List[str],
        sequence_column: str = PEPTIDE_CANONICAL,
    ):
        """
        Initialize the filter.

        Parameters
        ----------
        exclude_patterns : list[str]
            List of regex patterns to exclude.
        sequence_column : str, optional
            Column name containing peptide sequences.
        """
        self.exclude_patterns = exclude_patterns
        self.sequence_column = sequence_column
        # Pre-compile patterns
        self._compiled_patterns = [re.compile(p) for p in exclude_patterns]

    @property
    def name(self) -> str:
        return "SequencePatternFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)

        col = self.sequence_column
        if col not in df.columns and PEPTIDE_CANONICAL in df.columns:
            col = PEPTIDE_CANONICAL

        if col not in df.columns:
            logger.warning("%s: Sequence column not found, skipping filter", self.name)
            return df, self._create_result(input_count, input_count)

        # Combine compiled patterns into single regex for vectorized matching
        if not self._compiled_patterns:
            return df, self._create_result(input_count, input_count)

        combined_pattern = "|".join(p.pattern for p in self._compiled_patterns)
        mask = ~df[col].fillna("").astype(str).str.contains(
            combined_pattern, regex=True
        )
        filtered_df = df[mask].copy()

        output_count = len(filtered_df)

        logger.debug(
            "%s: Removed %d peptides matching patterns %s",
            self.name,
            input_count - output_count,
            self.exclude_patterns,
        )

        return filtered_df, self._create_result(
            input_count, output_count, {"exclude_patterns": self.exclude_patterns}
        )


class PeptideFDRFilter(BaseFilter):
    """Filter peptide rows by the dedicated QPX peptide q-value."""

    def __init__(self, fdr_threshold: float, fdr_column: str = PEPTIDE_QVALUE) -> None:
        self.fdr_threshold = fdr_threshold
        self.fdr_column = fdr_column

    @property
    def name(self) -> str:
        return "PeptideFDRFilter"

    @property
    def level(self) -> FilterLevel:
        return FilterLevel.PEPTIDE

    def apply(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, FilterResult]:
        input_count = len(df)
        if self.fdr_column not in df.columns or not df[self.fdr_column].notna().any():
            raise ValueError(
                "peptide FDR filtering requires a populated QPX "
                f"'{self.fdr_column}' column"
            )
        filtered_df = df[df[self.fdr_column] <= self.fdr_threshold].copy()
        return filtered_df, self._create_result(
            input_count,
            len(filtered_df),
            {"fdr_threshold": self.fdr_threshold},
        )
