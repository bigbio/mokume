"""Frozen compatibility and official TMM regressions for ``features2proteins``.

The compiled Rust kernel (``mokume._mokume``) and the pure-Python pipeline are
two implementations of the same protein-quantification contract. They share the
import name ``mokume`` and are mutually exclusive in a single interpreter, and CI
installs them in separate jobs with different dependency sets (the wheel job has
``_mokume`` but not ``duckdb``; the pure-Python job has ``duckdb`` but not
``_mokume``). Running both live in one process is therefore impossible.

Instead we compare the Rust kernel's output against *frozen compatibility
goldens* checked in beside the fixture. The goldens were produced by the
pure-Python ``QuantificationPipeline(...).run()`` (float64); they pin accepted
behavior for these covered paths without making Python authoritative. The Rust
kernel now preserves the input precision, while these compatibility tests retain
their original ``rtol=1e-6`` and ``atol=1e-2`` thresholds.

Regenerate the goldens only after an intentional shared-contract change that is
meant to update both implementations, not after an isolated pure-Python change::

    cd python && python -c "\
from mokume.pipeline.config import PipelineConfig, InputConfig, QuantificationConfig; \
from mokume.pipeline.features_to_proteins import QuantificationPipeline; \
QuantificationPipeline(PipelineConfig(\
input=InputConfig(parquet='../rust/tests/example/feature_wide.parquet'), \
quantification=QuantificationConfig(method='sum'))).run().to_csv(\
'../rust/tests/example/feature_wide_sum_python.csv', index=False)"

TMM uses independent edgeR 4.8.2 oracles in ``fixtures/tmm_edger_4_8_2``.
``provenance.json`` records input SHA256, the canonical-peptide input contract,
official parameters and full-precision factors. The original Parquet intensities
are collapsed by canonical peptide after the default protein/sample filtering;
``edgeR::calcNormFactors(method="TMM")`` fits sample factors on this linear-scale
matrix, with absent entries zero-filled only for fitting. Protein sum/median is
then computed after division by the centered composition factors. This is not
edgeR CPM normalization. Sparse factors must follow edgeR's pairwise support and
global centering, which differ from the old Python >=10-valid-feature fallback.

The ``feature_wide_multiform`` fixture is ``feature_wide_dense`` with a second
peptidoform per canonical (``"<seq>(Oxidation)"``, intensity ``* 0.6 + 1000``),
so every canonical carries two peptidoforms and the peptidoform -> canonical
collapse actually merges rows (the 1:1 dense/sparse fixtures never do). Its
TMM cases use the same independent edgeR construction. The RLR compatibility
golden remains ``feature_wide_multiform_median_rlr_python.csv``, produced with
``normalization=NormalizationConfig(sample_method='rlr')``.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

mokume = pytest.importorskip("mokume")

REPO = Path(__file__).resolve().parents[2]
FEATURE_PARQUET = REPO / "rust" / "tests" / "example" / "feature_wide.parquet"
DENSE_PARQUET = REPO / "rust" / "tests" / "example" / "feature_wide_dense.parquet"
MULTIFORM_PARQUET = (
    REPO / "rust" / "tests" / "example" / "feature_wide_multiform.parquet"
)
GOLDEN_DIR = REPO / "rust" / "tests" / "example"
EDGER_GOLDEN_DIR = REPO / "rust" / "tests" / "fixtures" / "tmm_edger_4_8_2"


def _rust_features2proteins(
    method: str,
    out_path: Path,
    parquet: Path = FEATURE_PARQUET,
    sample_normalization: Optional[str] = None,
) -> pd.DataFrame:
    """Quantify via the compiled Rust kernel (in-process)."""
    if not hasattr(mokume, "_mokume"):
        pytest.skip("mokume._mokume (Rust kernel) not available in this interpreter")
    args = [
        "quantify",
        "features2proteins",
        "--parquet",
        str(parquet),
        "--quant-method",
        method,
        "--output",
        str(out_path),
    ]
    if sample_normalization is not None:
        args += ["--sample-normalization", sample_normalization]
    mokume.run(args)
    return pd.read_csv(out_path)


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Protein x sample matrix indexed by protein with columns sorted."""
    df = df.rename(columns={df.columns[0]: "ProteinName"})
    df = df.set_index("ProteinName").sort_index()
    return df.reindex(sorted(df.columns), axis=1)


@pytest.mark.parametrize("method", ["sum", "median"])
def test_rust_matches_python_golden(method, tmp_path):
    """Rust output matches the frozen compatibility golden within f32 tolerance."""
    golden_path = GOLDEN_DIR / f"feature_wide_{method}_python.csv"
    rust_df = _canonical(
        _rust_features2proteins(method, tmp_path / f"rust_{method}.csv")
    )
    py_df = _canonical(pd.read_csv(golden_path))

    assert list(rust_df.index) == list(py_df.index), "protein sets differ"
    assert list(rust_df.columns) == list(py_df.columns), "sample columns differ"

    rust_missing = rust_df.isna().to_numpy()
    py_missing = py_df.isna().to_numpy()
    assert np.array_equal(rust_missing, py_missing), "missing-value masks differ"
    common = ~rust_missing
    rust_vals = rust_df.to_numpy(dtype=float)[common]
    py_vals = py_df.to_numpy(dtype=float)[common]
    assert np.allclose(rust_vals, py_vals, rtol=1e-6, atol=1e-2), (
        f"{method}: Rust output diverges from the pure-Python golden beyond f32 "
        f"tolerance (max abs diff {np.max(np.abs(rust_vals - py_vals)):.4g})"
    )


@pytest.mark.parametrize("method", ["sum", "median"])
@pytest.mark.parametrize(
    "fixture, parquet, golden_prefix",
    [
        ("sparse", FEATURE_PARQUET, "feature_wide"),
        ("dense", DENSE_PARQUET, "feature_wide_dense"),
        ("multiform", MULTIFORM_PARQUET, "feature_wide_multiform"),
    ],
)
def test_rust_tmm_matches_edger_golden(
    method, fixture, parquet, golden_prefix, tmp_path
):
    """The full Rust pipeline matches edgeR factors plus protein aggregation.

    Covers sparse pairwise support, dense rescaling, and multiple peptidoforms
    per canonical peptide with both protein sum and median aggregation.
    """
    golden_path = EDGER_GOLDEN_DIR / f"{golden_prefix}_{method}_tmm_edger.csv"
    rust_df = _canonical(
        _rust_features2proteins(
            method,
            tmp_path / f"rust_{fixture}_{method}_tmm.csv",
            parquet=parquet,
            sample_normalization="tmm",
        )
    )
    upstream_df = _canonical(pd.read_csv(golden_path, float_precision="round_trip"))

    assert list(rust_df.index) == list(upstream_df.index), "protein sets differ"
    assert list(rust_df.columns) == list(upstream_df.columns), "sample columns differ"

    rust_missing = rust_df.isna().to_numpy()
    upstream_missing = upstream_df.isna().to_numpy()
    assert np.array_equal(rust_missing, upstream_missing), "missing-value masks differ"
    common = ~rust_missing
    rust_vals = rust_df.to_numpy(dtype=float)[common]
    upstream_vals = upstream_df.to_numpy(dtype=float)[common]
    assert np.allclose(rust_vals, upstream_vals, rtol=1e-6, atol=1e-2), (
        f"tmm/{fixture}/{method}: Rust output diverges from the edgeR 4.8.2 "
        f"oracle beyond the original tolerance "
        f"(max abs diff {np.max(np.abs(rust_vals - upstream_vals)):.4g})"
    )


@pytest.mark.parametrize("method", ["sum", "median"])
def test_rust_tmm_is_not_a_noop_on_dense(method, tmp_path):
    """On the dense fixture, Rust TMM must differ from ``none`` (regression guard).

    Before the Rust TMM implementation, ``--sample-normalization tmm`` was
    silently equivalent to ``none``; this asserts the two now diverge on a
    fixture whose samples share 20 canonical peptides.
    """
    none_df = _canonical(
        _rust_features2proteins(
            method,
            tmp_path / f"rust_dense_{method}_none.csv",
            parquet=DENSE_PARQUET,
            sample_normalization="none",
        )
    )
    tmm_df = _canonical(
        _rust_features2proteins(
            method,
            tmp_path / f"rust_dense_{method}_tmm.csv",
            parquet=DENSE_PARQUET,
            sample_normalization="tmm",
        )
    )

    assert list(none_df.index) == list(tmm_df.index)
    assert list(none_df.columns) == list(tmm_df.columns)
    none_missing = none_df.isna().to_numpy()
    tmm_missing = tmm_df.isna().to_numpy()
    assert np.array_equal(none_missing, tmm_missing), "TMM changed missing-value mask"
    common = ~none_missing
    max_diff = np.max(
        np.abs(none_df.to_numpy(float)[common] - tmm_df.to_numpy(float)[common])
    )
    assert max_diff > 1.0, (
        f"tmm/{method}: TMM produced output identical to none on the dense "
        f"fixture (max abs diff {max_diff:.4g}) -- TMM is a no-op"
    )


@pytest.mark.parametrize(
    "quant, norm",
    [
        ("median", "rlr"),
    ],
)
def test_rust_matches_python_multiform_collapse(quant, norm, tmp_path):
    """Rust matches pure-Python when a canonical carries multiple peptidoforms.

    The ``feature_wide`` and ``feature_wide_dense`` fixtures carry a single
    peptidoform per canonical peptide, so the peptidoform -> canonical collapse
    that ``finalize`` performs on the ``median`` quant path is a no-op there and
    never exercises how the dataset normalizers write canonical-keyed values back
    into the cells. ``feature_wide_multiform`` packs two peptidoforms per
    canonical, so the collapse actually merges rows.

    This pins the writeback family (``rlr`` here as the representative) when
    ``finalize`` re-collapses the already-canonical values. A regression that
    reintroduced a double-collapse would diverge from the pure-Python golden.
    """
    golden = GOLDEN_DIR / f"feature_wide_multiform_{quant}_{norm}_python.csv"
    rust_df = _canonical(
        _rust_features2proteins(
            quant,
            tmp_path / f"rust_multiform_{quant}_{norm}.csv",
            parquet=MULTIFORM_PARQUET,
            sample_normalization=norm,
        )
    )
    py_df = _canonical(pd.read_csv(golden))

    assert list(rust_df.index) == list(py_df.index), "protein sets differ"
    assert list(rust_df.columns) == list(py_df.columns), "sample columns differ"

    rust_missing = rust_df.isna().to_numpy()
    py_missing = py_df.isna().to_numpy()
    assert np.array_equal(rust_missing, py_missing), "missing-value masks differ"
    common = ~rust_missing
    rust_vals = rust_df.to_numpy(dtype=float)[common]
    py_vals = py_df.to_numpy(dtype=float)[common]
    assert np.allclose(rust_vals, py_vals, rtol=1e-6, atol=1e-2), (
        f"{norm}/{quant}: Rust diverges from the pure-Python golden on the "
        f"multi-peptidoform collapse path "
        f"(max abs diff {np.max(np.abs(rust_vals - py_vals)):.4g})"
    )
