"""
Tests for protein accession normalization and FASTA matching.

Covers get_accession, build_accession_map, and the normalization
path in extract_fasta / get_protein_molecular_weights.
"""

from mokume.core.constants import build_accession_map, get_accession
from mokume.io.fasta import _strip_nonstandard_aa


# ---------------------------------------------------------------------------
# get_accession
# ---------------------------------------------------------------------------


class TestGetAccession:
    """Tests for get_accession with standard and non-standard formats."""

    def test_standard_uniprot_sp(self):
        assert get_accession("sp|P12345|PROT_NAME") == "P12345"

    def test_standard_uniprot_tr(self):
        assert get_accession("tr|Q67890|PROT_NAME") == "Q67890"

    def test_standard_uniprot_sw(self):
        assert get_accession("sw|A0A0K3|PROT_NAME") == "A0A0K3"

    def test_standard_uniprot_nxp(self):
        assert get_accession("nxp|NX_P12345|PROT_NAME") == "NX_P12345"

    def test_ups_database_prefix(self):
        """A UPS| prefix is a database tag, so the accession is the second field."""
        assert get_accession("UPS|P02768ups|ALBU_HUMAN_UPS") == "P02768ups"

    def test_prefix_case_insensitive(self):
        assert get_accession("SP|P12345|PROT_NAME") == "P12345"
        assert get_accession("Tr|Q67890|PROT_NAME") == "Q67890"

    def test_nonstandard_two_part(self):
        assert get_accession("P02768ups|ALBU_HUMAN_UPS") == "P02768ups"

    def test_nonstandard_ups1_format(self):
        assert get_accession("P01031ups|CO5_HUMAN_UPS") == "P01031ups"

    def test_plain_accession(self):
        assert get_accession("O13547") == "O13547"

    def test_plain_accession_with_suffix(self):
        assert get_accession("P12345ups") == "P12345ups"

    def test_empty_string(self):
        assert get_accession("") == ""


# ---------------------------------------------------------------------------
# build_accession_map
# ---------------------------------------------------------------------------


class TestBuildAccessionMap:
    """Tests for build_accession_map normalization helper."""

    def test_standard_proteins(self):
        proteins = ["sp|P12345|PROT_A", "sp|Q67890|PROT_B"]
        acc_map, acc_set = build_accession_map(proteins)
        assert acc_set == {"P12345", "Q67890"}
        assert acc_map["P12345"] == ["sp|P12345|PROT_A"]
        assert acc_map["Q67890"] == ["sp|Q67890|PROT_B"]

    def test_nonstandard_proteins(self):
        proteins = ["P02768ups|ALBU_HUMAN_UPS", "P01031ups|CO5_HUMAN_UPS"]
        acc_map, acc_set = build_accession_map(proteins)
        assert acc_set == {"P02768ups", "P01031ups"}
        assert acc_map["P02768ups"] == ["P02768ups|ALBU_HUMAN_UPS"]

    def test_mixed_formats(self):
        proteins = [
            "sp|P12345|PROT_A",
            "P02768ups|ALBU_HUMAN_UPS",
            "O13547",
        ]
        _, acc_set = build_accession_map(proteins)
        assert acc_set == {"P12345", "P02768ups", "O13547"}

    def test_multiple_originals_same_accession(self):
        proteins = ["sp|P12345|PROT_A", "tr|P12345|PROT_B"]
        acc_map, acc_set = build_accession_map(proteins)
        assert acc_set == {"P12345"}
        assert len(acc_map["P12345"]) == 2

    def test_empty_list(self):
        acc_map, acc_set = build_accession_map([])
        assert not acc_map
        assert not acc_set

    def test_accepts_iterable(self):
        proteins = iter(["sp|P12345|A", "O99999"])
        _, acc_set = build_accession_map(proteins)
        assert acc_set == {"P12345", "O99999"}


# ---------------------------------------------------------------------------
# _strip_nonstandard_aa
# ---------------------------------------------------------------------------


class TestStripNonstandardAA:
    """Tests for _strip_nonstandard_aa helper in io.fasta."""

    def test_no_nonstandard(self):
        assert _strip_nonstandard_aa("ACDEFG") == "ACDEFG"

    def test_removes_known_nonstandard(self):
        assert _strip_nonstandard_aa("AXBZCDEFJU") == "ACDEF"

    def test_empty_string(self):
        assert _strip_nonstandard_aa("") == ""

    def test_all_nonstandard(self):
        assert _strip_nonstandard_aa("XBZJUO") == ""
