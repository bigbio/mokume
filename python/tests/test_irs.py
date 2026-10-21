"""Regression tests for IRS metadata helpers."""

import pandas as pd
import pytest

from mokume.normalization.irs import IRSNormalizer, detect_plexes_from_sdrf


def test_detect_plexes_accepts_tmt_n_and_c_channel_suffixes(tmp_path):
    """TMT channel labels must not create a synthetic ``plex1`` batch."""
    sdrf = tmp_path / "tmt.sdrf.tsv"
    sdrf.write_text(
        "source name\tcomment[label]\n"
        "sample_Mixture1_126\tTMT126\n"
        "condition_a_Mixture1_127N\tTMT127N\n"
        "condition_b_Mixture1_127C\tTMT127C\n"
        "p2_127N\tTMT127N\n",
        encoding="utf-8",
    )

    assert detect_plexes_from_sdrf(str(sdrf)) == {
        "sample_Mixture1_126": "mixture1",
        "condition_a_Mixture1_127N": "mixture1",
        "condition_b_Mixture1_127C": "mixture1",
        "p2_127N": "p2",
    }


def test_zero_reference_intensities_count_as_missing():
    """A missing protein written as 0 must not pull the plex reference down."""
    proteins = pd.DataFrame(
        {"protein": ["P1"], "r1": [0.0], "r2": [400.0], "r3": [100.0]}
    )
    plexes = {"r1": "a", "r2": "a", "r3": "b"}
    for stat in ("median", "mean"):
        normalizer = IRSNormalizer(["r1", "r2", "r3"], stat=stat).fit(proteins, plexes)
        # Plex references are 400 and 100, so the global reference is 200.
        assert normalizer.scaling_factors_["a"].iloc[0] == pytest.approx(0.5)
        assert normalizer.scaling_factors_["b"].iloc[0] == pytest.approx(2.0)
