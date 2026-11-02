"""Regression tests for optional TissueMap embedding dependencies."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

from mokume.tissuemap.embedding import _impute_for_pca
from mokume.plotting._common import bottom_legend, place_labels
from mokume.plotting.style import CONDITION_COLORS, category_colors
from mokume.plotting.differential_expression import plot_pca_conditions, plot_volcano
from mokume.plotting.profile import plot_profile
from mokume.reports.qc_report import generate_qc_report
from mokume.tissuemap.plotting.atlas import (
    _draw_dendrogram_panel,
    _group_styles,
    _label_groups,
    _name_cluster,
    plot_slide_atlas_dendrogram,
)
from mokume.tissuemap.plotting import markers as marker_plots
from mokume.tissuemap.plotting.markers import _protein_label
from mokume.tissuemap.plotting.markers import _significant_markers
from mokume.tissuemap.plotting.specificity import plot_ts_distribution
from mokume.tissuemap.tissue_specificity import _compute_ts_vectorized_mad


PYTHON_ROOT = Path(__file__).parents[1]


def test_tsne_module_import_does_not_load_optional_umap(tmp_path):
    """A broken optional UMAP install must not block the independent t-SNE path."""
    (tmp_path / "umap.py").write_text(
        "raise RuntimeError('UMAP should be loaded only when requested')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(PYTHON_ROOT)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mokume.tissuemap.embedding; "
            "assert 'umap' not in sys.modules",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_embedding_mindet_uses_each_samples_protein_distribution():
    """Embedding adapts its sample-by-protein matrix to the imputer contract."""
    matrix = np.array(
        [
            [1.0, 10.0, np.nan],
            [2.0, np.nan, 100.0],
            [np.nan, 30.0, 1000.0],
        ]
    )

    result = _impute_for_pca(matrix, "mindet", n_neighbors=2)

    assert result[0, 2] == pytest.approx(np.quantile([1.0, 10.0], 0.01))
    assert result[1, 1] == pytest.approx(np.quantile([2.0, 100.0], 0.01))
    assert result[2, 0] == pytest.approx(np.quantile([30.0, 1000.0], 0.01))


def test_pure_mad_pi_uses_observed_values_as_denominator():
    """Missing values are not counted as population-model outliers."""
    matrix = np.array([[0.0], [1.0], [2.0], [np.nan], [np.nan]])

    _scores, params = _compute_ts_vectorized_mad(
        matrix,
        np.array(["A", "A", "A", "B", "B"]),
        ["A", "B"],
        sigma_floor=0.01,
    )

    assert params[0].pi == pytest.approx(1.0)


def test_marker_display_requires_effect_and_adjusted_significance():
    """Top-ranked non-significant proteins are not labelled as markers."""
    result = pd.DataFrame(
        {
            "names": ["kept", "weak_fdr", "weak_effect"],
            "pvals_adj": [0.01, 0.1, 0.01],
            "logfoldchanges": [0.6, 1.0, 0.2],
        }
    )

    assert _significant_markers(result)["names"].tolist() == ["kept"]


def test_atlas_dendrogram_skips_single_tissue(tmp_path, caplog):
    """A single-tissue dataset has no meaningful dendrogram to plot."""
    adata = SimpleNamespace(
        obsm={"X_tsne": np.array([[0.0, 0.0], [1.0, 1.0]])},
        obs=pd.DataFrame({"tissue": ["liver", "liver"]}),
    )

    with caplog.at_level(logging.WARNING):
        plot_slide_atlas_dendrogram(adata, tmp_path)

    assert "requires at least two tissues" in caplog.text
    assert not any(tmp_path.iterdir())


def test_atlas_stores_each_tissues_group_color(tmp_path, monkeypatch):
    """Marker plots color tissue labels with the group colors the atlas stores."""
    rng = np.random.default_rng(0)
    tissues = ["brain", "brain", "lung", "lung", "skin", "skin"]
    adata = ad.AnnData(
        rng.normal(size=(6, 4)),
        obs=pd.DataFrame({"tissue": tissues}, index=[f"s{i}" for i in range(6)]),
    )
    adata.obsm["X_tsne"] = rng.normal(size=(6, 2))
    monkeypatch.setattr(Figure, "savefig", lambda *_args, **_kwargs: None)

    plot_slide_atlas_dendrogram(adata, tmp_path)

    colors = adata.uns["tissue_colors"]
    assert set(colors) == set(tissues)
    assert set(colors.values()) <= set(
        _group_styles(adata.uns["proteomic_groups"]).values()
    )


@pytest.mark.parametrize(
    ("members", "used", "expected"),
    [
        (["brain, cortex", "brain, cerebellum", "lung"], set(), "Brain"),
        (["brain, cortex", "lung"], set(), "Group 3"),
        (["brain, cortex", "brain, cerebellum"], {"Brain"}, "Group 3"),
    ],
)
def test_atlas_names_require_a_unique_strict_organ_majority(members, used, expected):
    """A minority family or a previously used organ name cannot name a group."""
    assert _name_cluster(members, 3, used) == expected


def test_atlas_groups_use_distinct_palette_colors():
    """Atlas and dendrogram groups share distinct colors beyond six."""
    small = _group_styles({f"Group {i}": [] for i in range(6)})
    large = _group_styles({f"Group {i}": [] for i in range(10)})

    assert tuple(small.values()) == CONDITION_COLORS
    assert len(set(large.values())) == 10
    assert tuple(large.values()) == category_colors(10)


@pytest.mark.parametrize(
    ("identifier", "label"),
    [
        ("sp|P12345|ABC_HUMAN", "ABC"),
        ("tr|A0A123|ABC_SUB_HUMAN", "ABC_SUB"),
        ("db|ACC|ENTRY", "db|ACC|ENTRY"),
        ("P12345", "P12345"),
        ("ABC_HUMAN", "ABC_HUMAN"),
    ],
)
def test_marker_labels_shorten_only_uniprot_entries(identifier, label):
    """Plain identifiers retain their original labels."""
    assert _protein_label(identifier) == label


def test_marker_tsne_shares_a_protein_relative_scale_without_changing_values(
    tmp_path, monkeypatch
):
    """Differently scaled proteins share one colorbar; missing samples stay missing."""
    matrix = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [np.nan, np.nan]])
    adata = ad.AnnData(
        matrix.copy(),
        obs=pd.DataFrame({"tissue": ["a"] * 5}, index=list("abcde")),
        var=pd.DataFrame(index=["p1", "p2"]),
    )
    adata.obsm["X_tsne"] = np.arange(10).reshape(5, 2)
    adata.uns["tissue_markers"] = {}
    monkeypatch.setattr(
        marker_plots, "_select_showcase_markers", lambda *_: [("p1", "a"), ("p2", "a")]
    )
    figures = []
    monkeypatch.setattr(
        Figure, "savefig", lambda fig, *_args, **_kwargs: figures.append(fig)
    )

    marker_plots.plot_marker_tsne(adata, ["a"], tmp_path, save_pdf=False)

    fig = figures[0]
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    for ax in visible[:2]:
        detected = ax.collections[1]
        np.testing.assert_allclose(
            detected.get_array(), [0, 0.97 / 2.94, 1.97 / 2.94, 1]
        )
        assert detected.get_clim() == (0, 1)
    assert "within each protein" in visible[-1].get_ylabel()
    np.testing.assert_array_equal(adata.X, matrix)


@pytest.mark.parametrize("group_count", [10, 15])
def test_atlas_group_labels_stay_inside_axes_without_overlapping(group_count):
    """Labels near an embedding boundary must not cross the title or each other."""
    fig, (ax, _) = plt.subplots(1, 2, figsize=(8, 6))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    _label_groups(
        ax, {f"Group {i}": np.array([0.95, 0.95]) for i in range(group_count)}
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent()
    boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]

    assert all(bounds.contains(*box.min) and bounds.contains(*box.max) for box in boxes)
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    plt.close(fig)


def test_atlas_leaders_stay_off_other_labels():
    """A leader into a dense cloud must not run through another group's label."""
    cloud = np.random.default_rng(0).normal(size=(900, 2))
    central = cloud[np.linalg.norm(cloud, axis=1) < 1.5][:10]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.scatter(*cloud.T, s=12)
    ax.set(xlim=(-8, 8), ylim=(-8, 8))
    _label_groups(ax, {f"Group {i}": xy for i, xy in enumerate(central)})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]
    for text, box in zip(ax.texts, boxes):
        if not text.arrow_patch.get_visible():
            continue
        start = ax.transData.transform(text.xy)
        path = np.linspace(start, (box.min + box.max) / 2, 200)
        others = [other for other in boxes if other is not box]
        assert not any(other.count_contains(path) for other in others)
    plt.close(fig)


def test_category_label_leaders_stay_off_other_labels():
    """Omittable labels must not send a leader through another label."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-6, 6, (40, 2))
    cloud = np.vstack(
        [center + rng.normal(scale=0.2, size=(5, 2)) for center in centers]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.scatter(*cloud.T, s=12)
    ax.set(xlim=(-8, 8), ylim=(-8, 8))
    anchors = {f"Category name {i}": xy for i, xy in enumerate(centers)}
    place_labels(ax, {}, optional=anchors, fit="near")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]
    assert any(text.arrow_patch.get_visible() for text in ax.texts)
    for text, box in zip(ax.texts, boxes):
        if not text.arrow_patch.get_visible():
            continue
        start = ax.transData.transform(text.xy)
        path = np.linspace(start, (box.min + box.max) / 2, 200)
        others = [other for other in boxes if other is not box]
        assert not any(other.count_contains(path) for other in others)
    plt.close(fig)


def test_dendrogram_group_markers_do_not_cover_leaf_labels():
    """Marker spacing in points must not depend on the dendrogram width."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_dendrogram_panel(
        ax,
        np.array([[0, 1, 1.0, 2]]),
        ["brain", "skin"],
        {"brain": "Group 1", "skin": "Group 1"},
        {"Group 1": CONDITION_COLORS[0]},
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, marker in zip(ax.get_yticklabels(), ax.collections[1:]):
        center = marker.get_offset_transform().transform(marker.get_offsets())[0]
        radius = (np.sqrt(marker.get_sizes()[0]) + marker.get_linewidths()[0]) / 2
        assert label.get_window_extent(renderer).x1 < center[0] - radius * fig.dpi / 72
    plt.close(fig)


@pytest.mark.parametrize("condition_count", [40, 120])
def test_pca_supports_many_conditions_and_category_labels(tmp_path, condition_count):
    """Category labels that do not fit are omitted; the legend keeps every category."""
    rng = np.random.default_rng(42)
    samples = [f"S{i}" for i in range(condition_count * 2)]
    frame = pd.DataFrame(
        np.exp2(rng.normal(20, 1, (40, len(samples)))), columns=samples
    )
    frame.insert(0, "ProteinName", [f"P{i}" for i in range(40)])
    conditions = {sample: f"C{i // 2:02}" for i, sample in enumerate(samples)}
    output = tmp_path / "pca.png"

    fig = plot_pca_conditions(frame, conditions, output_file=output)

    assert output.is_file()
    assert len(fig.legends[0].get_texts()) == condition_count
    assert {text.get_text() for text in fig.axes[0].texts} <= set(conditions.values())
    # Saving closes the figure, and closing can detach its Agg canvas.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in fig.axes[0].texts]
    assert boxes
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    axes_size = fig.axes[0].get_window_extent().size / fig.dpi
    np.testing.assert_allclose(axes_size, [6.4, 4.5])
    plt.close(fig)


def test_volcano_supports_fifteen_highlighted_proteins(tmp_path):
    """Highlight labels do not depend on a figure-level category legend."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "ProteinName": [f"P{i:04}" for i in range(500)],
            "log2FC": rng.normal(size=500),
            "pvalue": rng.uniform(0.001, 1, 500),
            "adj_pvalue": 1.0,
            "significance": "Unchanged",
        }
    )
    output = tmp_path / "volcano.png"

    fig = plot_volcano(
        frame,
        highlight_genes=frame["ProteinName"].iloc[:15].tolist(),
        output_file=output,
    )

    assert output.is_file()
    assert len(fig.axes[0].texts) == 15
    plt.close(fig)


def test_qc_peptide_diagnostics_support_more_than_six_conditions(tmp_path):
    """Peptide boxes and report scripts must use cycling category styles."""
    rng = np.random.default_rng(42)
    samples = [f"S{i}" for i in range(16)]
    frame = pd.DataFrame(np.exp2(rng.normal(20, 1, (12, 16))), columns=samples)
    frame.insert(0, "ProteinName", [f"P{i}" for i in range(12)])
    conditions = {sample: f"C{i // 2}" for i, sample in enumerate(samples)}
    peptides = pd.DataFrame(
        {
            "ProteinName": "P0",
            "PeptideCanonical": "PEPTIDE",
            "SampleID": samples,
            "NormIntensity": np.arange(16) + 100,
        }
    )
    output = tmp_path / "qc.html"

    generate_qc_report(
        frame,
        conditions,
        output_html=output,
        peptides=peptides,
        unnormalized_peptides=peptides,
    )

    report = output.read_text(encoding="utf-8")
    assert "normalization-plot" in report
    assert "const condColors = {" in report
    assert "styleIndices" not in report
    assert "shapeByPlex" not in report


def test_category_palette_extends_with_unique_stable_prefixes():
    """The six seeds remain fixed while larger palettes extend deterministically."""
    assert category_colors(6) == CONDITION_COLORS
    colors = category_colors(40)
    assert len(colors) == len(set(colors)) == 40
    assert colors[:6] == CONDITION_COLORS
    assert colors[:20] == category_colors(20)


def _saved_sizes(monkeypatch):
    """Record each saved figure's size by output file name."""
    sizes = {}

    def savefig(figure, path, **_kwargs):
        sizes[Path(path).name] = tuple(figure.get_size_inches())

    monkeypatch.setattr(Figure, "savefig", savefig)
    return sizes


def test_ts_distribution_height_follows_the_bars_it_draws(tmp_path, monkeypatch):
    """The panel shows the top 25 tissues, so more tissues must not stretch it."""
    sizes = _saved_sizes(monkeypatch)
    tissues = [f"tissue {i}" for i in range(114)]
    ts_df = pd.DataFrame(
        {
            "max_ts": np.linspace(1, 8, 228),
            "max_tissue": tissues * 2,
            "enrichment_category": "tissue-specific",
        }
    )
    plot_ts_distribution(ts_df, tissues, tmp_path, save_pdf=False)
    assert sizes["ts_distribution.png"][1] == pytest.approx(25 * 0.14 + 1.4)


def test_marker_charts_grow_with_many_tissues(tmp_path, monkeypatch):
    """Heatmap columns and dot plot rows keep room for every tissue name."""
    rng = np.random.default_rng(0)
    tissue_count = 60
    tissues = [f"tissue {i:02d}" for i in range(tissue_count) for _ in range(6)]
    values = rng.normal(size=(len(tissues), 120)) + 10
    for index in range(tissue_count):
        values[index * 6 : index * 6 + 6, index] += 8
    adata = ad.AnnData(
        values.astype(np.float32),
        obs=pd.DataFrame(
            {"tissue": tissues}, index=[f"s{i}" for i in range(len(tissues))]
        ),
        var=pd.DataFrame(index=[f"P{i:03d}" for i in range(120)]),
    )
    marker_plots.compute_markers(adata)
    sizes = _saved_sizes(monkeypatch)
    order = sorted(set(tissues))
    marker_plots.plot_marker_heatmap(adata, order, tmp_path, save_pdf=False)
    marker_plots.plot_dotplot(adata, order, tmp_path)
    assert sizes["marker_heatmap.png"][0] == pytest.approx(tissue_count * 0.14 + 1.5)
    assert sizes["marker_dotplot.png"][1] == pytest.approx(tissue_count * 0.16 + 1.5)


@pytest.mark.parametrize("category_count", [20, 400])
def test_bottom_legend_stays_below_the_axis_title(category_count):
    """A legend's height grows the canvas without pushing it into the axis title."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.set_xlabel("PC1")
    for index in range(category_count):
        ax.plot([], [], "o", label=f"Category {index}")

    bottom_legend(ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    assert legend.y1 < ax.xaxis.label.get_window_extent(renderer).y0
    assert legend.y0 >= 0
    plt.close(fig)


@pytest.mark.parametrize(
    ("sample_count", "condition_count", "line_count"),
    [(12, 3, 2), (60, 3, 2), (60, 20, 0)],
)
def test_profile_thins_out_many_samples(
    tmp_path, monkeypatch, sample_count, condition_count, line_count
):
    """Many samples get dots without lines, many conditions no separators."""
    samples = [f"S{index}" for index in range(sample_count)]
    proteins = (
        pd.DataFrame(np.full((1, sample_count), 1024.0), columns=samples, index=["P1"])
        .rename_axis("ProteinName")
        .reset_index()
    )
    peptides = pd.DataFrame(
        {
            "ProteinName": "P1",
            "PeptideCanonical": "PEPTIDE",
            "SampleID": samples,
            "NormIntensity": 512.0,
        }
    )
    conditions = {
        sample: f"C{index % condition_count}" for index, sample in enumerate(samples)
    }
    plotted = []
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda fig, *_, **__: plotted.append(fig)
    )

    plot_profile(proteins, peptides, conditions, "P1", tmp_path / "profile.png")

    ax = plotted[0].axes[0]
    separators = [line for line in ax.get_lines() if len(set(line.get_xdata())) == 1]
    data = [line for line in ax.get_lines() if line not in separators]
    assert len(separators) == line_count
    assert all((line.get_linestyle() == "None") == (sample_count > 40) for line in data)
