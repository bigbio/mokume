"""
Tests for analysis, imputation, and normalization modules.

Covers: DifferentialExpression (limrots, deqms, proda, ihw),
        censored imputation (minprob, mindet, knn),
        LOESS normalization.

All DE methods are pure-Python reimplementations; no rpy2/R required.
"""

import numpy as np
import pandas as pd
import pytest

import mokume.analysis.ensemble as ensemble_module
from mokume.analysis._helpers import filter_testable, per_group_summary
from mokume.analysis.differential_expression import DifferentialExpression
from mokume.analysis.deqms import _two_sided_t_log_pvalue, run_deqms
from mokume.analysis.limma import run_limma
from mokume.analysis.limrots import run_limrots
from mokume.analysis.proda import run_proda
from mokume.analysis.rots import _calculate_fdr, run_rots
from mokume.analysis.ensemble import combine_de_results, run_ensemble
from mokume.pipeline.config import (
    DEConfig,
    ImputationConfig,
    InputConfig,
    PipelineConfig,
)
from mokume.pipeline.features_to_proteins import QuantificationPipeline
from mokume.pipeline.runner import run_pipeline
from mokume.pipeline.stages import ImputationStage, PostprocessingStage
from mokume.imputation.censored import (
    impute_minprob,
    impute_mindet,
    impute_censored,
)
from mokume.normalization.loess import LOESSNormalizer, loess_normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_protein_matrix(n_proteins=100, n_samples_per_group=4, seed=42):
    """Create a synthetic protein intensity matrix with known DE proteins."""
    rng = np.random.default_rng(seed)

    n_de = 15  # first 15 proteins are DE
    n_total = n_samples_per_group * 2

    # Background ~ N(23, 1)
    data = rng.normal(23, 1, size=(n_proteins, n_total))

    # Inject fold change for DE proteins in group A
    data[:n_de, :n_samples_per_group] += 3.0  # log2FC ~ 3

    proteins = [f"Protein_{i:03d}" for i in range(n_proteins)]
    samples_a = [f"A_{i}" for i in range(n_samples_per_group)]
    samples_b = [f"B_{i}" for i in range(n_samples_per_group)]

    df = pd.DataFrame(data, index=proteins, columns=samples_a + samples_b)

    # Sprinkle some NaN (5% missing, light enough for R packages)
    mask = rng.random(df.shape) < 0.05
    df[mask] = np.nan

    return df, samples_a, samples_b, proteins[:n_de]


def _make_wide_df(matrix, samples_a, samples_b):
    """Convert index-based matrix to wide-format with ProteinName column."""
    wide = matrix.copy()
    wide.insert(0, "ProteinName", wide.index)
    wide = wide.reset_index(drop=True)
    s2c = {s: "A" for s in samples_a}
    s2c.update({s: "B" for s in samples_b})
    return wide, s2c


# ---------------------------------------------------------------------------
# DifferentialExpression — LimROTS
# ---------------------------------------------------------------------------


class TestDELimROTS:
    def test_limrots_returns_results(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="limrots",
            log2fc_threshold=1.0,
            skip_log2=True,
            n_boot=20,
        )
        result = de.run(wide, s2c, ("A", "B"))
        assert len(result) > 0
        assert "adj_pvalue" in result.columns

    def test_limrots_detects_de(self):
        mat, sa, sb, _de_proteins = _make_protein_matrix(n_samples_per_group=5)
        wide, s2c = _make_wide_df(mat, sa, sb)
        res_lr = DifferentialExpression(
            method="limrots",
            log2fc_threshold=1.0,
            skip_log2=True,
            n_boot=20,
        ).run(wide, s2c, ("A", "B"))
        sig = res_lr[res_lr["significance"] == "UP"]["ProteinName"].tolist()
        assert len(sig) > 0

    def test_run_limrots_direct(self):
        mat, sa, sb, _ = _make_protein_matrix()
        log2_mat = np.log2(mat.clip(lower=1))
        result = run_limrots(log2_mat, sa, sb, ("A", "B"), n_boot=10)
        assert len(result) > 0
        assert "pvalue" in result.columns


# ---------------------------------------------------------------------------
# DifferentialExpression — DEqMS
# ---------------------------------------------------------------------------


class TestDEDEqMS:
    def test_deqms_returns_results(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="deqms",
            log2fc_threshold=1.0,
            skip_log2=True,
        )
        result = de.run(wide, s2c, ("A", "B"))
        assert len(result) > 0
        assert "adj_pvalue" in result.columns

    def test_run_deqms_direct(self):
        mat, sa, sb, _ = _make_protein_matrix()
        log2_mat = np.log2(mat.clip(lower=1))
        result = run_deqms(log2_mat, sa, sb, ("A", "B"))
        assert len(result) > 0
        assert "pvalue" in result.columns
        assert "log_pvalue" in result.columns

    def test_deqms_log_pvalue_survives_float_underflow(self):
        log_pvalue = _two_sided_t_log_pvalue(
            np.array([40.837, 87.944]),
            np.array([10_000.0, 10_000.0]),
        )

        assert np.all(np.isfinite(log_pvalue))
        assert log_pvalue == pytest.approx(
            [-775.0382247148757, -2868.950716179171],
            abs=3e-11,
        )

    def test_deqms_with_peptide_counts(self):
        rng = np.random.default_rng(42)
        mat, sa, sb, _ = _make_protein_matrix()
        log2_mat = np.log2(mat.clip(lower=1))
        counts = pd.Series(
            rng.integers(2, 10, len(log2_mat)),
            index=log2_mat.index,
        )
        result = run_deqms(log2_mat, sa, sb, ("A", "B"), peptide_counts=counts)
        assert len(result) > 0
        assert "peptide_count" in result.columns


# ---------------------------------------------------------------------------
# DifferentialExpression — proDA
# ---------------------------------------------------------------------------


class TestDEProDA:
    def test_proda_returns_results(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="proda", log2fc_threshold=1.0, skip_log2=True
        )
        result = de.run(wide, s2c, ("A", "B"))
        assert len(result) > 0
        assert "adj_pvalue" in result.columns

    def test_run_proda_direct_uses_moderated_df(self):
        """Use the per-protein moderated df instead of a constant residual df."""
        mat, sa, sb, _ = _make_protein_matrix()
        result = run_proda(np.log2(mat.clip(lower=1)), sa, sb, "A", "B")
        assert len(result) > 0
        assert {
            "ProteinName",
            "log2FC",
            "pvalue",
            "adj_pvalue",
            "df",
        }.issubset(result.columns)
        naive_df = (len(sa) + len(sb)) - 2  # p = intercept + group
        df_values = result["df"].values
        assert np.all(df_values > 0)
        assert not np.allclose(df_values, naive_df)


# ---------------------------------------------------------------------------
# IHW correction
# ---------------------------------------------------------------------------


class TestIHW:
    def test_ihw_differs_from_bh(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        res_bh = DifferentialExpression(
            method="deqms",
            log2fc_threshold=1.0,
            fdr_method="bh",
            skip_log2=True,
        ).run(wide, s2c, ("A", "B"))
        res_ihw = DifferentialExpression(
            method="deqms",
            log2fc_threshold=1.0,
            fdr_method="ihw",
            skip_log2=True,
        ).run(wide, s2c, ("A", "B"))
        assert len(res_ihw) > 0
        assert "adj_pvalue" in res_ihw.columns
        # IHW should produce different adjusted p-values than BH
        merged = res_bh.merge(res_ihw, on="ProteinName", suffixes=("_bh", "_ihw"))
        assert not np.allclose(
            merged["adj_pvalue_bh"].values,
            merged["adj_pvalue_ihw"].values,
            equal_nan=True,
        )

    def test_ihw_fallback_on_small_data(self):
        # Very few proteins → should fall back to BH
        mat, sa, sb, _ = _make_protein_matrix(n_proteins=5)
        wide, s2c = _make_wide_df(mat, sa, sb)
        res_bh = DifferentialExpression(
            method="deqms",
            log2fc_threshold=1.0,
            fdr_method="bh",
            skip_log2=True,
        ).run(wide, s2c, ("A", "B"))
        res_ihw = DifferentialExpression(
            method="deqms",
            log2fc_threshold=1.0,
            fdr_method="ihw",
            skip_log2=True,
        ).run(wide, s2c, ("A", "B"))
        assert len(res_ihw) > 0
        # With too few proteins, IHW falls back to BH → same results
        merged = res_bh.merge(res_ihw, on="ProteinName", suffixes=("_bh", "_ihw"))
        assert np.allclose(
            merged["adj_pvalue_bh"].values,
            merged["adj_pvalue_ihw"].values,
            equal_nan=True,
        )

    def test_limrots_fdr_unchanged_by_ihw(self):
        # LimROTS carries its own permutation-based FDR in ``adj_pvalue``;
        # requesting IHW must not overwrite it, so the adjusted p-values are
        # identical under bh and ihw (same seed → same permutation FDR).
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        res_bh = DifferentialExpression(
            method="limrots",
            log2fc_threshold=1.0,
            fdr_method="bh",
            skip_log2=True,
            n_boot=20,
        ).run(wide, s2c, ("A", "B"))
        res_ihw = DifferentialExpression(
            method="limrots",
            log2fc_threshold=1.0,
            fdr_method="ihw",
            skip_log2=True,
            n_boot=20,
        ).run(wide, s2c, ("A", "B"))
        merged = res_bh.merge(res_ihw, on="ProteinName", suffixes=("_bh", "_ihw"))
        assert np.allclose(
            merged["adj_pvalue_bh"].values,
            merged["adj_pvalue_ihw"].values,
            equal_nan=True,
        )

    def test_rots_fdr_unchanged_by_ihw(self):
        # Same guarantee for ROTS: its permutation FDR survives an IHW request.
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        res_bh = DifferentialExpression(
            method="rots",
            log2fc_threshold=1.0,
            fdr_method="bh",
            skip_log2=True,
            n_boot=20,
        ).run(wide, s2c, ("A", "B"))
        res_ihw = DifferentialExpression(
            method="rots",
            log2fc_threshold=1.0,
            fdr_method="ihw",
            skip_log2=True,
            n_boot=20,
        ).run(wide, s2c, ("A", "B"))
        merged = res_bh.merge(res_ihw, on="ProteinName", suffixes=("_bh", "_ihw"))
        assert np.allclose(
            merged["adj_pvalue_bh"].values,
            merged["adj_pvalue_ihw"].values,
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# Censored imputation
# ---------------------------------------------------------------------------


class TestCensoredImputation:
    def test_minprob_fills_nan(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(23, 1, (20, 6)),
            columns=[f"S{i}" for i in range(6)],
        )
        data.iloc[0, 0] = np.nan
        data.iloc[5, 3] = np.nan
        result = impute_minprob(data)
        assert result.isna().sum().sum() == 0

    def test_mindet_fills_nan(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(23, 1, (20, 6)),
            columns=[f"S{i}" for i in range(6)],
        )
        data.iloc[0, 0] = np.nan
        result = impute_mindet(data)
        assert result.isna().sum().sum() == 0

    def test_impute_censored_dispatcher(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(23, 1, (20, 6)))
        data.iloc[0, 0] = np.nan
        for method in ("minprob", "mindet", "knn", "none"):
            result = impute_censored(data, method=method)
            if method != "none":
                assert result.isna().sum().sum() == 0

    def test_impute_censored_invalid_method(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute_censored(data, method="invalid")


# ---------------------------------------------------------------------------
# LOESS normalization
# ---------------------------------------------------------------------------


class TestLOESS:
    def test_loess_normalize_shape(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(23, 1, (100, 6)),
            columns=[f"S{i}" for i in range(6)],
        )
        # Add systematic bias to one sample
        data["S0"] += np.linspace(0, 3, 100)
        result = loess_normalize(data)
        assert result.shape == data.shape
        assert result.isna().sum().sum() == 0

    def test_loess_normalizer_class(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            rng.normal(23, 1, (100, 6)),
            columns=[f"S{i}" for i in range(6)],
        )
        norm = LOESSNormalizer(frac=0.75, reference="median")
        result = norm.fit_transform(data)
        assert result.shape == data.shape

    def test_loess_reference_mean(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(23, 1, (100, 6)))
        result = loess_normalize(data, reference="mean")
        assert result.shape == data.shape

    def test_loess_invalid_reference(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown reference"):
            loess_normalize(data, reference="invalid")


# ---------------------------------------------------------------------------
# run_comparisons
# ---------------------------------------------------------------------------


class TestRunComparisons:
    def test_multiple_contrasts(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="deqms", log2fc_threshold=1.0, skip_log2=True
        )
        results = de.run_comparisons(wide, s2c, [("A", "B")])
        assert "A-B" in results
        assert len(results["A-B"]) > 0


# ---------------------------------------------------------------------------
# DifferentialExpression — limma
# ---------------------------------------------------------------------------


class TestDELimma:
    def test_limma_returns_results(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="limma", log2fc_threshold=1.0, skip_log2=True
        )
        result = de.run(wide, s2c, ("A", "B"))
        assert len(result) > 0
        assert "adj_pvalue" in result.columns

    def test_run_limma_direct(self):
        mat, sa, sb, _ = _make_protein_matrix()
        result = run_limma(np.log2(mat.clip(lower=1)), sa, sb, "A", "B")
        assert len(result) > 0
        assert "pvalue" in result.columns

    def test_limma_has_expected_columns(self):
        mat, sa, sb, _ = _make_protein_matrix()
        result = run_limma(np.log2(mat.clip(lower=1)), sa, sb, "A", "B")
        for col in ("ProteinName", "log2FC", "pvalue", "adj_pvalue", "t_stat"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# DifferentialExpression — ROTS
# ---------------------------------------------------------------------------


class TestDEROTS:
    def test_rots_returns_results(self):
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        de = DifferentialExpression(
            method="rots", log2fc_threshold=1.0, skip_log2=True, n_boot=20
        )
        result = de.run(wide, s2c, ("A", "B"))
        assert len(result) > 0
        assert "adj_pvalue" in result.columns

    def test_run_rots_direct(self):
        mat, sa, sb, _ = _make_protein_matrix()
        result = run_rots(np.log2(mat.clip(lower=1)), sa, sb, "A", "B", n_boot=20)
        assert len(result) > 0
        assert "pvalue" in result.columns

    def test_rots_has_expected_columns(self):
        mat, sa, sb, _ = _make_protein_matrix()
        result = run_rots(np.log2(mat.clip(lower=1)), sa, sb, "A", "B", n_boot=20)
        for col in ("ProteinName", "log2FC", "pvalue", "adj_pvalue", "d_stat"):
            assert col in result.columns

    def test_native_fdr_follows_rots_calculate_fdr(self):
        """The FDR is the median over permutations of above-count over rank."""
        # Ranked |d|: 3, 2, 1. Per permutation, (above + tie) / rank:
        # [0, 1/2, 2/3], [1, 1/2, 1/3], [0, 0, 1/3]; medians 0, 1/2, 1/3;
        # the cumulative minimum from the bottom gives 0, 1/3, 1/3.
        permuted = np.array([[0.5, -3.0, 1.0], [2.5, 0.0, 1.0], [1.0, 0.1, -1.0]])
        fdr = _calculate_fdr(np.array([3.0, -1.0, 2.0]), permuted)
        np.testing.assert_allclose(fdr, [0.0, 1 / 3, 1 / 3])

    def test_native_fdr_is_not_floored_by_a_reproduced_grouping(self):
        """A permutation that reproduces the observed grouping sets no floor."""
        observed = np.array([4.0, 3.0, 0.2, 0.1])
        null = np.array([0.3, 0.2, 0.1, 0.0])
        fdr = _calculate_fdr(observed, np.column_stack([null, null, observed]))
        np.testing.assert_allclose(fdr[:2], [0.0, 0.0])
        assert fdr[2] > 0.0


# ---------------------------------------------------------------------------
# DifferentialExpression — non-finite log2FC guard
# ---------------------------------------------------------------------------


class TestNonFiniteLog2FCGuard:
    """A protein imputed to a degenerate value in one group can yield an
    Inf/-Inf log2FC; it must be coerced to NaN and excluded from significance
    rather than passing the ``|log2FC| > threshold`` filter."""

    def _finalize(self, log2fc, pvalue):
        de = DifferentialExpression(
            method="limma", log2fc_threshold=0.5, fdr_method="bh"
        )
        df = pd.DataFrame(
            {
                "ProteinName": [f"P{i}" for i in range(len(log2fc))],
                "log2FC": log2fc,
                "pvalue": pvalue,
            }
        )
        return de._finalize_results(df)

    def test_inf_log2fc_coerced_to_nan(self):
        out = self._finalize([np.inf, -np.inf, np.nan, 3.0], [1e-6, 1e-6, 1e-6, 1e-6])
        out = out.set_index("ProteinName")
        assert np.isnan(out.loc["P0", "log2FC"])  # +inf -> NaN
        assert np.isnan(out.loc["P1", "log2FC"])  # -inf -> NaN
        assert np.isfinite(out.loc["P3", "log2FC"])  # finite untouched

    def test_inf_log2fc_not_counted_significant(self):
        out = self._finalize([np.inf, -np.inf, 3.0, -3.0], [1e-6, 1e-6, 1e-6, 1e-6])
        out = out.set_index("ProteinName")
        # non-finite folds are non-estimable, not evidence of no change
        assert out.loc["P0", "significance"] == "NotTested"
        assert out.loc["P1", "significance"] == "NotTested"
        # genuine finite folds are still classified
        assert out.loc["P2", "significance"] == "UP"
        assert out.loc["P3", "significance"] == "DOWN"

    def test_nonfinite_pvalue_is_not_tested(self):
        out = self._finalize([2.0, 2.0], [np.inf, 1e-6]).set_index("ProteinName")
        assert np.isnan(out.loc["P0", "pvalue"])
        assert np.isnan(out.loc["P0", "adj_pvalue"])
        assert out.loc["P0", "significance"] == "NotTested"
        assert out.loc["P1", "significance"] == "UP"


def test_de_helpers_count_only_finite_observations():
    matrix = pd.DataFrame(
        {
            "A1": [1.0, np.inf, np.inf],
            "A2": [2.0, 2.0, 2.0],
            "A3": [3.0, 4.0, np.nan],
            "B1": [3.0, 3.0, 3.0],
            "B2": [4.0, 4.0, 4.0],
            "B3": [5.0, 5.0, 5.0],
        },
        index=["finite", "infinite", "insufficient"],
    )

    samples_a = ["A1", "A2", "A3"]
    samples_b = ["B1", "B2", "B3"]
    testable = filter_testable(matrix, samples_a, samples_b)
    assert list(testable.index) == ["finite", "infinite"]
    assert np.isnan(testable.loc["infinite", "A1"])

    summary = per_group_summary(matrix, samples_a, samples_b, "A", "B")
    infinite = summary.set_index("ProteinName").loc["infinite"]
    assert infinite["mean_A"] == 3.0
    assert infinite["n_a"] == 2
    assert infinite["mean_B"] == 4.0
    assert infinite["n_b"] == 3


def test_differential_expression_rejects_infinite_input():
    matrix = pd.DataFrame(
        {
            "ProteinName": ["P1"],
            "A1": [1.0],
            "A2": [np.inf],
            "B1": [2.0],
            "B2": [3.0],
        }
    )
    sample_to_condition = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}

    with pytest.raises(ValueError, match="contains an infinite intensity"):
        DifferentialExpression(method="limma").run(
            matrix, sample_to_condition, ("A", "B")
        )


def test_pipeline_imputation_rejects_linear_scale_overflow(monkeypatch):
    config = PipelineConfig(
        input=InputConfig(parquet="unused.parquet"),
        imputation=ImputationConfig(enabled=True, method="mindet"),
    )
    stage = ImputationStage(config)
    matrix = pd.DataFrame(
        {
            "ProteinName": ["P1", "P2"],
            "S1": [100.0, np.nan],
            "S2": [200.0, 300.0],
        }
    )
    overflow = pd.DataFrame(
        [[np.log2(100.0), np.log2(200.0)], [2048.0, np.log2(300.0)]],
        index=["P1", "P2"],
        columns=["S1", "S2"],
    )
    monkeypatch.setattr(stage, "_dispatch", lambda method, wide: overflow)

    with pytest.raises(ValueError, match="overflowed on the linear scale"):
        stage.impute(matrix)


def test_pipeline_imputation_rejects_infinite_input():
    config = PipelineConfig(
        input=InputConfig(parquet="unused.parquet"),
        imputation=ImputationConfig(enabled=True, method="mindet"),
    )
    matrix = pd.DataFrame(
        {
            "ProteinName": ["P1", "P2"],
            "S1": [100.0, np.inf],
            "S2": [200.0, 300.0],
        }
    )

    with pytest.raises(ValueError, match="contains an infinite intensity"):
        ImputationStage(config).impute(matrix)


# ---------------------------------------------------------------------------
# DifferentialExpression — Ensemble
# ---------------------------------------------------------------------------


class TestDEEnsemble:
    @pytest.mark.parametrize(
        ("methods", "min_k"),
        [
            ([], 1),
            ([""], 1),
            (["unknown"], 1),
            (["auto"], 1),
            (["ensemble"], 1),
            ("limma", 1),
            (["limma", " LIMMA "], 1),
            (["limma"], 0),
            (["limma"], 2),
        ],
    )
    def test_invalid_configuration_fails_before_member_execution(
        self,
        monkeypatch,
        methods,
        min_k,
    ):
        calls = []

        class RecordingDE:
            def __init__(self, method, **options):
                calls.append((method, options))

            def run(self, protein_df, sample_to_condition, contrast):
                return pd.DataFrame()

        monkeypatch.setattr(ensemble_module, "DifferentialExpression", RecordingDE)

        with pytest.raises(ValueError, match="ensemble"):
            run_ensemble(
                pd.DataFrame(),
                {},
                ("A", "B"),
                methods=methods,
                min_k=min_k,
            )

        assert calls == []

    def test_member_names_are_canonicalized_once(self, monkeypatch):
        calls = []

        class RecordingDE:
            def __init__(self, method, **options):
                calls.append((method, options))

            def run(self, protein_df, sample_to_condition, contrast):
                return pd.DataFrame()

        monkeypatch.setattr(ensemble_module, "DifferentialExpression", RecordingDE)

        result = run_ensemble(
            pd.DataFrame(),
            {},
            ("A", "B"),
            methods=[" LimMA ", "DEQMS"],
            min_k=np.int64(1),
        )

        assert result.empty
        assert [method for method, _ in calls] == ["limma", "deqms"]

    @pytest.mark.parametrize("error_type", [ValueError, KeyError])
    def test_shared_member_input_error_is_not_silenced(self, monkeypatch, error_type):
        class InvalidInputDE:
            def __init__(self, method, **options):
                pass

            def run(self, protein_df, sample_to_condition, contrast):
                raise error_type("shared member input")

        monkeypatch.setattr(ensemble_module, "DifferentialExpression", InvalidInputDE)

        with pytest.raises(error_type, match="shared member input"):
            run_ensemble(
                pd.DataFrame(),
                {},
                ("A", "B"),
                methods=["limma"],
                min_k=1,
            )

    def test_method_runtime_failure_keeps_other_members(self, monkeypatch):
        calls = []

        class PartiallyFailingDE:
            def __init__(self, method, **options):
                self.method = method

            def run(self, protein_df, sample_to_condition, contrast):
                calls.append(self.method)
                if self.method == "limma":
                    raise RuntimeError("method fit failed")
                return pd.DataFrame()

        monkeypatch.setattr(
            ensemble_module, "DifferentialExpression", PartiallyFailingDE
        )

        result = run_ensemble(
            pd.DataFrame(),
            {},
            ("A", "B"),
            methods=["limma", "deqms"],
            min_k=1,
        )

        assert result.empty
        assert calls == ["limma", "deqms"]

    @pytest.mark.parametrize(
        "results,min_k",
        [
            ({}, 0),
            (
                {
                    "limma": pd.DataFrame(columns=["ProteinName"]),
                    " LIMMA ": pd.DataFrame(columns=["ProteinName"]),
                },
                1,
            ),
        ],
    )
    def test_combiner_rejects_invalid_consensus_configuration(self, results, min_k):
        with pytest.raises(ValueError, match="ensemble"):
            combine_de_results(results, min_k=min_k)

    def test_combiner_canonicalizes_result_method_names(self):
        result = pd.DataFrame(
            {
                "ProteinName": ["P1"],
                "log2FC": [2.0],
                "pvalue": [0.001],
                "adj_pvalue": [0.001],
                "significance": ["UP"],
            }
        )

        combined = combine_de_results({" LimMA ": result}, min_k=np.int64(1))

        assert combined.loc[0, "methods_significant"] == "limma"

    @pytest.mark.parametrize("entrypoint", ["oop", "runner"])
    def test_pipeline_entrypoints_validate_ensemble_before_input(
        self,
        tmp_path,
        entrypoint,
    ):
        config = PipelineConfig(
            input=InputConfig(parquet=str(tmp_path / "missing.parquet")),
            de=DEConfig(
                enabled=True,
                method="ensemble",
                contrasts=["A vs B"],
                ensemble_methods=["unknown"],
                ensemble_min_k=1,
            ),
        )

        with pytest.raises(ValueError, match="unknown ensemble"):
            if entrypoint == "oop":
                QuantificationPipeline(config)
            else:
                run_pipeline(config)

    def test_pipeline_rejects_ensemble_options_for_single_method(self, tmp_path):
        config = PipelineConfig(
            input=InputConfig(parquet=str(tmp_path / "missing.parquet")),
            de=DEConfig(
                enabled=True,
                method="limma",
                contrasts=["A vs B"],
                ensemble_methods=["deqms"],
                ensemble_min_k=1,
            ),
        )

        with pytest.raises(ValueError, match="ensemble methods only apply"):
            QuantificationPipeline(config)

    def test_pipeline_rejects_ensemble_options_when_de_is_disabled(self, tmp_path):
        config = PipelineConfig(
            input=InputConfig(parquet=str(tmp_path / "missing.parquet")),
            de=DEConfig(
                enabled=False,
                method="ensemble",
                ensemble_methods=["unknown"],
                ensemble_min_k=0,
            ),
        )

        with pytest.raises(ValueError, match="options require DE to be enabled"):
            QuantificationPipeline(config)

    def test_pipeline_uses_canonical_explicit_deqms_member_counts(self, monkeypatch):
        config = PipelineConfig(
            input=InputConfig(parquet="explicit.parquet"),
            de=DEConfig(
                enabled=True,
                method=" Ensemble ",
                ensemble_methods=[" DEQMS "],
                ensemble_min_k=1,
            ),
        )
        stage = PostprocessingStage(config)
        peptide_rows = pd.DataFrame(
            {
                "anchor_protein": ["P1", "P1", "P2"],
                "sequence": ["AAA", "BBB", "CCC"],
            }
        )
        monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: peptide_rows)

        de_method = stage._resolve_de_method()
        counts = stage._load_de_peptide_counts(de_method)

        assert de_method == "ensemble"
        assert counts.to_dict() == {"P1": 2, "P2": 1}

    def test_default_ensemble_does_not_activate_peptide_count_loading(
        self, monkeypatch
    ):
        config = PipelineConfig(
            input=InputConfig(parquet="default.parquet"),
            de=DEConfig(enabled=True, method="ensemble"),
        )
        stage = PostprocessingStage(config)

        def unexpected_read(*args, **kwargs):
            raise AssertionError("default ensemble must not load peptide counts")

        monkeypatch.setattr(pd, "read_parquet", unexpected_read)

        assert stage._load_de_peptide_counts("ensemble") is None

    def test_ensemble_combines_without_protein_keyerror(self):
        # Regression: run_ensemble previously crashed with KeyError 'Protein'
        # because combine_de_results read df["Protein"] while every member
        # method emits "ProteinName". It must now run and return a combined
        # table keyed by ProteinName with the consensus columns.
        mat, sa, sb, _ = _make_protein_matrix()
        wide, s2c = _make_wide_df(mat, sa, sb)
        result = run_ensemble(
            wide,
            s2c,
            ("A", "B"),
            methods=("limma", "deqms"),
            min_k=1,
            log2fc_threshold=1.0,
        )
        assert not result.empty
        assert "ProteinName" in result.columns
        assert "Protein" not in result.columns
        for col in (
            "log2FC",
            "pvalue",
            "adj_pvalue",
            "significance",
            "n_methods_up",
            "n_methods_down",
            "methods_significant",
        ):
            assert col in result.columns

    def test_ensemble_preserves_not_tested_semantics(self):
        member = pd.DataFrame(
            {
                "ProteinName": ["P1"],
                "log2FC": [np.nan],
                "pvalue": [np.nan],
                "adj_pvalue": [np.nan],
                "significance": ["NotTested"],
            }
        )

        result = combine_de_results({"limma": member}, min_k=1)

        assert len(result) == 1
        assert result.loc[0, "significance"] == "NotTested"
        assert np.isnan(result.loc[0, "log2FC"])
        assert np.isnan(result.loc[0, "pvalue"])
