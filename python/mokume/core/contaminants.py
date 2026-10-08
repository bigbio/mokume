"""Group-level contaminant rule shared by the pandas and SQL filters."""

import re
from typing import Iterable

import pandas as pd

DEFAULT_CONTAMINANT_PATTERNS = ["CONTAMINANT", "CONTAM_", "ENTRAP", "DECOY"]


def _is_decoy_pattern(pattern: str) -> bool:
    return pattern.upper() == "DECOY"


def contaminant_group_mask(
    groups: pd.Series,
    patterns: Iterable[str],
    case_sensitive: bool = False,
) -> pd.Series:
    """True where a ``;``-joined group is a decoy (any member) or all-contaminant."""
    patterns = list(patterns)
    contam = [p for p in patterns if not _is_decoy_pattern(p)]
    decoy = [p for p in patterns if _is_decoy_pattern(p)]
    flags = 0 if case_sensitive else re.IGNORECASE
    contam_re = (
        re.compile("|".join(re.escape(p) for p in contam), flags) if contam else None
    )
    decoy_re = (
        re.compile("|".join(re.escape(p) for p in decoy), flags) if decoy else None
    )

    def is_contaminant(group) -> bool:
        if not isinstance(group, str):
            return False
        members = [m.strip() for m in group.split(";") if m.strip()]
        if not members:
            return False
        if decoy_re is not None and any(decoy_re.search(m) for m in members):
            return True
        return contam_re is not None and all(contam_re.search(m) for m in members)

    return groups.map(is_contaminant).astype(bool)
