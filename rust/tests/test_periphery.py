"""Periphery rendering tests: the folded ``mokume.plotting`` draws real files.

Skipped unless the ``analysis`` extra (matplotlib + pandas) is installed.
"""

import glob
import importlib
import json
import os
from types import SimpleNamespace

import pytest

import mokume

pytest.importorskip("matplotlib")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
plt = pytest.importorskip("matplotlib.pyplot")

profiles = pytest.importorskip("mokume.plotting.profile")
qc_diagnostics = pytest.importorskip("mokume.reports._qc_diagnostics")
diagnostics = pytest.importorskip("mokume.plotting.diagnostics")
ma_plots = pytest.importorskip("mokume.plotting.ma")
de_diagnostics = pytest.importorskip("mokume.commands._de_diagnostics")
distributions = pytest.importorskip("mokume.plotting.distributions")
style = pytest.importorskip("mokume.plotting.style")

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
    de = pd.read_csv(os.path.join(DATA, "de.csv"))
    de["pvalue"] = [0.0001, 0.0002, 0.1, 0.8]
    de_csv = tmp_path / "de.csv"
    de.to_csv(de_csv, index=False)
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
            str(de_csv),
        ]
    )
    assert glob.glob(str(plot_dir / "*.png"))


def test_static_plot_restores_defaults_and_exports_editable_pdf(tmp_path, caplog):
    """Static exports scope typography, preserve caller fonts and avoid Type 3."""
    plotting = importlib.import_module("mokume.plotting.differential_expression")
    de = pd.read_csv(os.path.join(DATA, "de.csv"))
    de["pvalue"] = [0.0001, 0.0002, 0.1, 0.8]
    output = tmp_path / "volcano.pdf"
    with plt.rc_context(
        {"font.family": ["DejaVu Sans"], "axes.grid": True, "pdf.fonttype": 3}
    ):
        before = dict(plt.rcParams)
        figure = plotting.plot_volcano(de, output_file=str(output))
        assert dict(plt.rcParams) == before
    axis = figure.axes[0]
    assert axis.title.get_fontsize() == 11
    assert axis.title.get_fontweight() == "normal"
    assert axis.xaxis.label.get_fontsize() == 10
    assert "DejaVu Sans" in axis.title.get_fontfamily()
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert not any(line.get_visible() for line in axis.get_ygridlines())
    assert axis.collections[0].get_rasterized()
    assert b"/Subtype /Type3" not in output.read_bytes()
    assert b"/FontFile2" in output.read_bytes()
    assert "findfont: Font family" not in caplog.text


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


def test_volcano_height_and_guide_use_raw_pvalues(tmp_path):
    """The adjusted-p cutoff includes small-FC proteins when placing the guide."""
    plotting = importlib.import_module("mokume.plotting.differential_expression")
    report = pytest.importorskip("mokume.reports.interactive")
    de = pd.DataFrame(
        {
            "ProteinName": ["P1", "P2", "P3"],
            "log2FC": [2.0, 0.1, -2.0],
            "pvalue": [0.001, 0.007, 0.03],
            "adj_pvalue": [0.01, 0.04, 0.2],
            "significance": ["UP", "Unchanged", "Unchanged"],
        }
    )
    for threshold, expected in [(0.05, -np.log10(0.007)), (0.001, None)]:
        figure = plotting.plot_volcano(de, fdr_threshold=threshold)
        axis = figure.axes[0]
        np.testing.assert_allclose(
            axis.collections[0].get_offsets()[:, 1], -np.log10(de["pvalue"])
        )
        guides = [line for line in axis.lines if list(line.get_xdata()) == [0, 1]]
        output = tmp_path / "report.html"
        report.generate_de_report(
            de,
            pd.DataFrame({"ProteinName": ["P1"], "S1": [10]}),
            {"S1": "A"},
            str(output),
            fdr_threshold=threshold,
        )
        arguments = output.read_text(encoding="utf-8").split(
            "Plotly.newPlot('volcano-plot', "
        )[1]
        _, layout = json.loads("[" + arguments.split(", plotConfig)")[0] + "]")
        shapes = [shape for shape in layout["shapes"] if shape.get("xref") == "paper"]
        if expected is None:
            assert not guides and not shapes
        else:
            assert guides[0].get_ydata()[0] == pytest.approx(expected)
            assert shapes[0]["y0"] == pytest.approx(expected)
        plt.close(figure)


def test_de_heatmap_selects_adjusted_p_before_absolute_fold_change(
    tmp_path, monkeypatch
):
    """Large fold changes cannot displace stronger adjusted-p evidence."""
    de_plots = importlib.import_module("mokume.commands.de_plots")
    de = pd.DataFrame({"ProteinName": [f"P{i}" for i in range(53)], "log2FC": 1.0})
    de["adj_pvalue"] = np.linspace(0.002, 0.007, 53)
    de.loc[:2] = [
        ["large_fc", 100, 0.04],
        ["tie_low", 1, 0.001],
        ["tie_high", -2, 0.001],
    ]
    de_csv = tmp_path / "de.csv"
    de.to_csv(de_csv, index=False)
    matrix = pd.DataFrame({"ProteinName": de["ProteinName"], "S1": 1, "S2": 2, "S3": 3})
    protein_csv = tmp_path / "proteins.csv"
    matrix.to_csv(protein_csv, index=False)
    sdrf = tmp_path / "samples.sdrf.tsv"
    sdrf.write_text("source name\tfactor value[condition]\nS1\tA\nS2\tB\nS3\tC\n")
    observed = {}

    def capture_heatmap(frame, mapping, **kwargs):
        observed.update(frame=frame, mapping=mapping, **kwargs)

    plotting = importlib.import_module("mokume.plotting.differential_expression")
    monkeypatch.setattr(plotting, "plot_heatmap", capture_heatmap)
    assert (
        de_plots.main(
            [
                "-p",
                str(protein_csv),
                "-o",
                str(tmp_path),
                "-s",
                str(sdrf),
                "--heatmap",
                "--contrast",
                "A-B",
                "A",
                "B",
                str(de_csv),
            ]
        )
        == 0
    )
    assert observed["proteins"] == [
        "tie_high",
        "tie_low",
        *[f"P{i}" for i in range(3, 51)],
    ]
    assert list(observed["frame"].columns) == ["ProteinName", "S1", "S2"]
    assert observed["mapping"] == {"S1": "A", "S2": "B"}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_peptide_profile_conditions_come_from_sdrf(tmp_path, with_metadata):
    """Four-column exports and stale six-column metadata use the same SDRF order."""
    frame = pd.DataFrame(
        {
            "ProteinName": ["P1"] * 3,
            "PeptideCanonical": ["PEPTIDE"] * 3,
            "SampleID": ["S1", "S2", "S3"],
            "NormIntensity": [8, 16, 32],
        }
    )
    if with_metadata:
        frame = frame.assign(BioReplicate=1, Condition="wrong")
    path = tmp_path / "peptides.csv"
    frame.to_csv(path, index=False)
    values = profiles.peptide_matrix(path, {"S1": "B", "S2": "A"})
    assert list(values.columns) == ["S2", "S1"]
    np.testing.assert_array_equal(values.loc[("P1", "PEPTIDE")], [4, 3])


def test_qc_imputation_mask_and_completeness_counts():
    """Only newly observed cells are imputed; complete and all-missing rows count."""
    before = pd.DataFrame(
        [[1, np.nan, 3], [4, 5, 6], [np.nan] * 3], index=["P1", "P2", "P3"]
    )
    after = before.copy()
    after.loc["P1", 1] = 2
    observed, filled = qc_diagnostics.imputation_values(before, after.iloc[::-1])
    np.testing.assert_array_equal(np.sort(observed), [1, 3, 4, 5, 6])
    np.testing.assert_array_equal(filled, [2])
    assert qc_diagnostics.completeness_counts(before).to_dict() == {
        0: 1,
        1: 0,
        2: 1,
        3: 1,
    }


def test_upset_counts_exclusive_intersections():
    """A protein belongs to exactly one observed membership pattern."""
    counts = diagnostics.upset_counts(
        {"A": {"p1", "p2", "p3"}, "B": {"p2", "p3", "p4"}, "C": {"p3"}}
    )
    assert counts == {("A",): 1, ("A", "B"): 1, ("A", "B", "C"): 1, ("B",): 1}


def test_protein_groups_and_expected_ratio_references(tmp_path):
    """Read composition groups and reject unknown expected-ratio references."""
    path = tmp_path / "groups.tsv"
    path.write_text("ProteinName\tgroup\nP1\thuman\nP2\tyeast\n")
    groups = ma_plots.read_protein_groups(path)
    assert groups.to_dict() == {"P1": "human", "P2": "yeast"}
    args = SimpleNamespace(
        contrast=[("B-A", "B", "A", "de.csv")], expected_log2fc=[("B-A", "yeast", "1")]
    )
    assert de_diagnostics.expected_ratios(args, groups) == {"B-A": {"yeast": 1.0}}
    for key, group in [("missing", "yeast"), ("B-A", "missing")]:
        args.expected_log2fc = [(key, group, "1")]
        with pytest.raises(ValueError, match="unknown"):
            de_diagnostics.expected_ratios(args, groups)


def test_distribution_order_and_large_condition_palette():
    """Sort samples within conditions and extend the palette past six."""
    frame = pd.DataFrame(
        {
            "SampleID": ["S1", "S3", "S2"] * 2,
            "Condition": ["B", "A", "A"] * 2,
            "PiBAQ": [1, 2, 3, 4, 5, 6],
        }
    )
    figure = distributions.plot_box_plot(frame, "PiBAQ", "SampleID")
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
        "S2",
        "S3",
        "S1",
    ]
    plt.close(figure)
    frame = pd.DataFrame(
        {
            "SampleID": [f"S{i}" for i in range(7)],
            "Condition": [f"C{i}" for i in range(7)],
            "PiBAQ": 1,
        }
    )
    _, colors, palette = distributions.distribution_groups(frame, "SampleID")
    assert tuple(palette.values()) == style.category_colors(7)
    assert len(set(colors.values())) == 7
