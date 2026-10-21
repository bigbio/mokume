"""Unit tests for TMT quantification methods and ratio exposure."""

import pandas as pd
import pytest

from mokume.model.quantification import QuantificationMethod
from mokume.quantification import (
    RatioQuantification,
    TMTAbundanceQuantification,
    TMTReporterIntensityQuantification,
    get_quantification_method,
)


@pytest.fixture
def simple_peptide_df():
    """Synthetic peptide-level intensity table.

    Layout: 2 proteins (P1, P2) x 2 peptides x 3 samples = 12 rows.
    Intensities are powers of two so log2/median/sum yield round numbers
    that are easy to assert without floating-point noise.
    """
    intensities = {
        ("P1", "PEPA", "S1"): 4.0,
        ("P1", "PEPA", "S2"): 8.0,
        ("P1", "PEPA", "S3"): 16.0,
        ("P1", "PEPB", "S1"): 16.0,
        ("P1", "PEPB", "S2"): 32.0,
        ("P1", "PEPB", "S3"): 64.0,
        ("P2", "PEPC", "S1"): 1.0,
        ("P2", "PEPC", "S2"): 2.0,
        ("P2", "PEPC", "S3"): 4.0,
        ("P2", "PEPD", "S1"): 4.0,
        ("P2", "PEPD", "S2"): 8.0,
        ("P2", "PEPD", "S3"): 16.0,
    }
    rows = [
        {
            "ProteinName": prot,
            "PeptideCanonical": pep,
            "SampleID": samp,
            "NormIntensity": value,
        }
        for (prot, pep, samp), value in intensities.items()
    ]
    return pd.DataFrame(rows)


class TestTMTAbundance:
    """``abd`` quantifier: median of log2 peptide intensities."""

    def test_name(self):
        assert TMTAbundanceQuantification().name == "TMTAbundance"

    def test_median_of_log2(self, simple_peptide_df):
        result = TMTAbundanceQuantification().quantify(
            simple_peptide_df, intensity_column="NormIntensity"
        )
        indexed = result.set_index(["ProteinName", "SampleID"])

        # P1 S1: log2([4, 16]) = [2, 4] -> median = 3
        # P1 S2: log2([8, 32]) = [3, 5] -> median = 4
        # P1 S3: log2([16, 64]) = [4, 6] -> median = 5
        # P2 S1: log2([1, 4]) = [0, 2] -> median = 1
        # P2 S2: log2([2, 8]) = [1, 3] -> median = 2
        # P2 S3: log2([4, 16]) = [2, 4] -> median = 3
        expected = {
            ("P1", "S1"): 3.0,
            ("P1", "S2"): 4.0,
            ("P1", "S3"): 5.0,
            ("P2", "S1"): 1.0,
            ("P2", "S2"): 2.0,
            ("P2", "S3"): 3.0,
        }
        for key, value in expected.items():
            assert indexed.loc[key, "Intensity"] == pytest.approx(value)

    def test_zero_intensity_excluded_from_median(self):
        df = pd.DataFrame(
            {
                "ProteinName": ["P1", "P1", "P1"],
                "PeptideCanonical": ["A", "B", "C"],
                "SampleID": ["S1", "S1", "S1"],
                "NormIntensity": [0.0, 4.0, 16.0],
            }
        )
        result = TMTAbundanceQuantification().quantify(
            df, intensity_column="NormIntensity"
        )
        # log2([4, 16]) = [2, 4] -> median = 3 (zero excluded)
        assert result["Intensity"].iloc[0] == pytest.approx(3.0)


class TestTMTReporterIntensity:
    """``intensity`` quantifier: linear-space protein-level sum."""

    def test_name(self):
        assert TMTReporterIntensityQuantification().name == "TMTReporterIntensity"

    def test_sum_per_protein_sample(self, simple_peptide_df):
        result = TMTReporterIntensityQuantification().quantify(
            simple_peptide_df, intensity_column="NormIntensity"
        )
        indexed = result.set_index(["ProteinName", "SampleID"])

        # P1 S1: 4 + 16 = 20; P1 S2: 8 + 32 = 40; P1 S3: 16 + 64 = 80
        # P2 S1: 1 + 4 = 5;  P2 S2: 2 + 8 = 10;  P2 S3: 4 + 16 = 20
        expected = {
            ("P1", "S1"): 20.0,
            ("P1", "S2"): 40.0,
            ("P1", "S3"): 80.0,
            ("P2", "S1"): 5.0,
            ("P2", "S2"): 10.0,
            ("P2", "S3"): 20.0,
        }
        for key, value in expected.items():
            assert indexed.loc[key, "Intensity"] == pytest.approx(value)


class TestQuantificationFactory:
    """``get_quantification_method`` accepts the new method aliases."""

    def test_all_alias(self):
        method = get_quantification_method("all")
        assert type(method).__module__ == "mokume.quantification.all_peptides"
        assert type(method).__name__ == "AllPeptidesQuantification"

    @pytest.mark.parametrize("key", ["abd", "abundance", "TMTAbundance"])
    def test_abd_aliases(self, key):
        assert isinstance(get_quantification_method(key), TMTAbundanceQuantification)

    @pytest.mark.parametrize("key", ["intensity", "reporter", "TMTReporterIntensity"])
    def test_intensity_aliases(self, key):
        assert isinstance(
            get_quantification_method(key), TMTReporterIntensityQuantification
        )

    def test_ratio_requires_reference_kwargs(self):
        with pytest.raises(ValueError, match="reference_samples"):
            get_quantification_method("ratio")

    def test_ratio_with_kwargs_returns_instance(self):
        method = get_quantification_method(
            "ratio",
            reference_samples=["ref1"],
            sample_to_plex={"s1": "p1", "ref1": "p1"},
            fraction_merge_method="max",
        )
        assert isinstance(method, RatioQuantification)
        assert method.reference_samples == ["ref1"]
        assert method.fraction_merge_method == "max"


class TestQuantificationMethodEnum:
    """``QuantificationMethod`` enum exposes the new members."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("tmt_abundance", QuantificationMethod.TMT_ABUNDANCE),
            ("TMT_REPORTER_INTENSITY", QuantificationMethod.TMT_REPORTER_INTENSITY),
            ("ratio", QuantificationMethod.RATIO),
        ],
    )
    def test_from_str(self, name, expected):
        assert QuantificationMethod.from_str(name) == expected

    def test_removed_ibaq_name_is_rejected(self):
        with pytest.raises(KeyError):
            QuantificationMethod.from_str("ibaq")

    def test_description_mentions_algorithm(self):
        assert "log2" in QuantificationMethod.TMT_ABUNDANCE.description
        assert (
            "reporter"
            in QuantificationMethod.TMT_REPORTER_INTENSITY.description.lower()
        )
        assert "ratio" in QuantificationMethod.RATIO.description.lower()
