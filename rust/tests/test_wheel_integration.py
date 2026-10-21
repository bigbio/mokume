"""Integration tests for the mokume wheel API using real fixture data.

Exercises the Rust compute kernel through the Python wrappers with actual
peptide/feature parquet and CSV files, verifying that the end-to-end pipeline
produces valid output.
"""

import os

import pytest

import mokume

EXAMPLE = os.path.join(os.path.dirname(__file__), "example")
PEPTIDES_CSV = os.path.join(EXAMPLE, "peptides_small.csv")
FEATURE_PARQUET = os.path.join(EXAMPLE, "feature_wide.parquet")
NEW_QPX_PARQUET = os.path.join(EXAMPLE, "feature_wide_new_qpx.parquet")
NEW_QPX_SDRF = os.path.join(EXAMPLE, "sdrf_new_qpx.tsv")

pd = pytest.importorskip("pandas")


@pytest.fixture
def peptides_parquet(tmp_path):
    pytest.importorskip("pyarrow")  # pandas needs an engine to write parquet
    df = pd.read_csv(PEPTIDES_CSV)
    path = tmp_path / "peptides.parquet"
    df.to_parquet(str(path), index=False)
    return str(path)


class TestPeptides2Protein:
    """peptides2protein with various quantification methods."""

    @pytest.mark.parametrize("method", ["sum", "top3"])
    def test_basic_methods(self, tmp_path, method):
        output = str(tmp_path / f"{method}.tsv")
        mokume.peptides2protein(
            peptides=PEPTIDES_CSV,
            quant_method=method,
            output=output,
        )
        assert os.path.exists(output)
        df = pd.read_csv(output, sep="\t")
        assert len(df) > 0
        assert "ProteinName" in df.columns

    def test_parquet_input(self, tmp_path, peptides_parquet):
        output = str(tmp_path / "from_parquet.tsv")
        mokume.peptides2protein(
            peptides=peptides_parquet,
            quant_method="sum",
            output=output,
        )
        assert os.path.exists(output)
        df = pd.read_csv(output, sep="\t")
        assert len(df) > 0

    def test_csv_vs_parquet_consistent(self, tmp_path, peptides_parquet):
        csv_out = str(tmp_path / "csv.tsv")
        pq_out = str(tmp_path / "pq.tsv")
        mokume.peptides2protein(
            peptides=PEPTIDES_CSV,
            quant_method="sum",
            output=csv_out,
        )
        mokume.peptides2protein(
            peptides=peptides_parquet,
            quant_method="sum",
            output=pq_out,
        )
        df_csv = pd.read_csv(csv_out, sep="\t")
        df_pq = pd.read_csv(pq_out, sep="\t")
        assert set(df_csv["ProteinName"]) == set(df_pq["ProteinName"])

    def test_topn_with_custom_n(self, tmp_path):
        # N is spelled in the method name. This used to read `quant_method="top3",
        # topn_n=5` and still produce Top5 -- the digits in the name were inert
        # and the companion option won. Both spellings of N are gone now.
        output = str(tmp_path / "top5.tsv")
        mokume.peptides2protein(
            peptides=PEPTIDES_CSV,
            quant_method="top5",
            output=output,
        )
        assert os.path.exists(output)
        df = pd.read_csv(output, sep="\t")
        assert len(df) > 0


class TestFeatures2Proteins:
    """features2proteins with QPX-format feature parquet."""

    def test_basic_run(self, tmp_path):
        output = str(tmp_path / "proteins.csv")
        mokume.features2proteins(
            parquet=NEW_QPX_PARQUET,
            output=output,
            sdrf=NEW_QPX_SDRF,
            quant_method="sum",
            min_unique=1,
        )
        assert os.path.exists(output)
        df = pd.read_csv(output)
        assert len(df) > 0


class TestFeatures2Peptides:
    """features2peptides converts feature parquet to peptide-level output."""

    def test_basic_run(self, tmp_path):
        output = str(tmp_path / "peptides.csv")
        mokume.features2peptides(
            parquet=FEATURE_PARQUET,
            output=output,
        )
        assert os.path.exists(output)
        df = pd.read_csv(output)
        assert len(df) > 0


class TestRunGeneric:
    """mokume.run() with explicit argument lists."""

    def test_run_peptides2protein(self, tmp_path):
        output = str(tmp_path / "run_out.tsv")
        mokume.run(
            [
                "quantify",
                "peptides2protein",
                "--peptides",
                PEPTIDES_CSV,
                "--quant-method",
                "sum",
                "--output",
                output,
            ]
        )
        assert os.path.exists(output)
