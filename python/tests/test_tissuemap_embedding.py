"""Regression tests for optional TissueMap embedding dependencies."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mokume.tissuemap.embedding import _impute_for_pca
from mokume.tissuemap.plotting.atlas import plot_slide_atlas_dendrogram
from mokume.tissuemap.plotting.markers import _significant_markers
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
