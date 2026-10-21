"""Periphery rendering tests: the folded ``mokume.plotting`` draws real files.

Skipped unless the ``analysis`` extra (matplotlib + pandas) is installed.
"""

import glob
import importlib
import os

import pytest

import mokume

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_peptides2protein_qc_writes_pdf(tmp_path):
    pdf = tmp_path / "qc.pdf"
    mokume.peptides2protein_qc(
        protein_table=os.path.join(DATA, "proteins_qc.tsv"),
        qc_report=str(pdf),
        plot_column="PiBAQ",
    )
    assert pdf.exists()
    assert pdf.stat().st_size > 0


def test_de_plots_writes_volcano_png(tmp_path):
    plot_dir = tmp_path / "plots"
    mokume.de_plots(
        [
            "--protein-matrix",
            os.path.join(DATA, "proteins_matrix.csv"),
            "--outdir",
            str(plot_dir),
            "--volcano",
            "--contrast",
            "c1",
            "CondA",
            "CondB",
            os.path.join(DATA, "de.csv"),
        ]
    )
    assert glob.glob(str(plot_dir / "*.png"))


def test_de_plots_preserves_tiny_kernel_pvalues(tmp_path, monkeypatch):
    de_csv = tmp_path / "de.csv"
    de_csv.write_text(
        "ProteinName,log2FC,adj_pvalue,significance\n"
        "P1,1.0,0.00000000000000000000000000009010904590617499,UP\n",
        encoding="utf-8",
    )
    observed = {}

    def capture_volcano(de_results, **_kwargs):
        observed["adj_pvalue"] = de_results.loc[0, "adj_pvalue"]

    plotting = importlib.import_module("mokume.plotting.differential_expression")
    monkeypatch.setattr(plotting, "plot_volcano", capture_volcano)

    de_plots = importlib.import_module("mokume.commands.de_plots")
    assert (
        de_plots.main(
            [
                "--protein-matrix",
                os.path.join(DATA, "proteins_matrix.csv"),
                "--outdir",
                str(tmp_path / "plots"),
                "--volcano",
                "--contrast",
                "c1",
                "CondA",
                "CondB",
                str(de_csv),
            ]
        )
        == 0
    )
    assert observed["adj_pvalue"] > 0.0
    assert observed["adj_pvalue"] == pytest.approx(
        9.010904590617499e-29, rel=1e-15, abs=0.0
    )
