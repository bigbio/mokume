"""
Tests for TMM (Trimmed Mean of M-values) normalization.

Tests cover:
- Basic functionality
- Edge cases (zeros, missing values)
- Geometric mean property
- Known value validation
"""

import os

import numpy as np
import pandas as pd
import pytest

from mokume.normalization.tmm import TMMNormalizer, tmm_normalize

# Cecilia fixture (small real PXD020192 LFQ slice), reused for the pipeline
# wiring test below.
_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "example")
_PARQUET = os.path.join(_EXAMPLE_DIR, "feature_wide.parquet")
_SDRF = os.path.join(_EXAMPLE_DIR, "PXD020192.sdrf.tsv")


class TestTMMNormalizer:
    """Tests for TMMNormalizer class."""

    @pytest.fixture
    def simple_data(self):
        """Simple test data with known properties (50 proteins for proper TMM)."""
        np.random.seed(42)
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)
        return pd.DataFrame(
            {
                "S1": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S2": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S3": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S4": base * (1 + np.random.randn(n_proteins) * 0.1),
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

    def test_fit_creates_factors(self, simple_data):
        """TMM fit should create normalization factors for each sample."""
        tmm = TMMNormalizer()
        tmm.fit(simple_data)

        assert tmm.norm_factors_ is not None
        assert len(tmm.norm_factors_) == 4
        assert all(f > 0 for f in tmm.norm_factors_)

    def test_geometric_mean_is_one(self, simple_data):
        """Geometric mean of factors should be 1."""
        tmm = TMMNormalizer()
        tmm.fit(simple_data)

        geom_mean = np.exp(np.mean(np.log(tmm.norm_factors_)))
        assert np.isclose(geom_mean, 1.0, atol=1e-6)

    def test_transform_preserves_shape(self, simple_data):
        """Transform should preserve data shape."""
        tmm = TMMNormalizer()
        result = tmm.fit_transform(simple_data)

        assert result.shape == simple_data.shape
        assert list(result.columns) == list(simple_data.columns)
        assert list(result.index) == list(simple_data.index)

    def test_reference_sample_selection(self, simple_data):
        """Should select valid reference sample."""
        tmm = TMMNormalizer()
        tmm.fit(simple_data)

        assert tmm.ref_sample_ in simple_data.columns

    def test_custom_reference_sample(self, simple_data):
        """Should use custom reference sample if provided."""
        tmm = TMMNormalizer(ref_sample="S2")
        tmm.fit(simple_data)

        assert tmm.ref_sample_ == "S2"

    def test_invalid_reference_raises(self, simple_data):
        """Should raise error for invalid reference sample."""
        tmm = TMMNormalizer(ref_sample="InvalidSample")

        with pytest.raises(ValueError, match="not found"):
            tmm.fit(simple_data)

    def test_handles_zeros(self):
        """TMM should handle zero values gracefully."""
        np.random.seed(42)
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        data = pd.DataFrame(
            {
                "S1": base.copy(),
                "S2": base.copy(),
                "S3": base.copy(),
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        # Add some zeros
        data.iloc[0, 0] = 0
        data.iloc[5, 1] = 0

        tmm = TMMNormalizer()
        result = tmm.fit_transform(data)

        # Should not have all NaN
        assert not result.isnull().all().any()

    def test_handles_nan(self):
        """TMM should handle NaN values gracefully."""
        np.random.seed(42)
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        data = pd.DataFrame(
            {
                "S1": base.copy(),
                "S2": base.copy(),
                "S3": base.copy(),
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        # Add some NaN values
        data.iloc[0, 0] = np.nan
        data.iloc[5, 1] = np.nan

        tmm = TMMNormalizer()
        result = tmm.fit_transform(data)

        assert result.isna().equals(data.isna())
        observed = result.to_numpy()[~data.isna().to_numpy()]
        assert np.isfinite(observed).all()

    def test_trim_parameters(self, simple_data):
        """Different trim parameters should affect results."""
        tmm1 = TMMNormalizer(m_trim=0.1, a_trim=0.01)
        tmm2 = TMMNormalizer(m_trim=0.4, a_trim=0.1)

        result1 = tmm1.fit_transform(simple_data)
        result2 = tmm2.fit_transform(simple_data)

        assert not np.allclose(result1.to_numpy(), result2.to_numpy())

    def test_transform_without_fit_raises(self, simple_data):
        """Transform without fit should raise error."""
        tmm = TMMNormalizer()

        with pytest.raises(ValueError, match="not been fitted"):
            tmm.transform(simple_data)

    def test_get_norm_factors(self, simple_data):
        """get_norm_factors should return copy of factors."""
        tmm = TMMNormalizer()
        tmm.fit(simple_data)

        factors1 = tmm.get_norm_factors()
        factors2 = tmm.get_norm_factors()

        # Should be equal but not same object
        pd.testing.assert_series_equal(factors1, factors2)

    def test_get_effective_library_sizes(self, simple_data):
        """get_effective_library_sizes should return adjusted sizes."""
        tmm = TMMNormalizer()
        tmm.fit(simple_data)

        eff_sizes = tmm.get_effective_library_sizes(simple_data)

        assert len(eff_sizes) == len(simple_data.columns)
        assert all(s > 0 for s in eff_sizes)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise error."""
        tmm = TMMNormalizer()

        with pytest.raises(ValueError, match="empty"):
            tmm.fit(pd.DataFrame())

    def test_single_sample(self):
        """Single sample should return unchanged (cannot normalize)."""
        data = pd.DataFrame(
            {
                "S1": [100, 200, 300],
            },
            index=["P1", "P2", "P3"],
        )

        tmm = TMMNormalizer()
        _ = tmm.fit_transform(data)

        # Factor should be 1.0
        assert np.isclose(tmm.norm_factors_["S1"], 1.0)


class TestTMMConvenienceFunction:
    """Tests for tmm_normalize convenience function."""

    def test_basic_usage(self):
        """tmm_normalize should work with default parameters."""
        np.random.seed(42)
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        data = pd.DataFrame(
            {
                "S1": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S2": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S3": base * (1 + np.random.randn(n_proteins) * 0.1),
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        result = tmm_normalize(data)

        assert result.shape == data.shape
        assert not result.isnull().all().any()

    def test_with_parameters(self):
        """tmm_normalize should accept custom parameters."""
        np.random.seed(42)
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        data = pd.DataFrame(
            {
                "S1": base * (1 + np.random.randn(n_proteins) * 0.1),
                "S2": base * (1 + np.random.randn(n_proteins) * 0.1),
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        result = tmm_normalize(data, m_trim=0.2, a_trim=0.1, ref_sample="S1")
        expected = TMMNormalizer(m_trim=0.2, a_trim=0.1, ref_sample="S1").fit_transform(
            data
        )

        pd.testing.assert_frame_equal(result, expected)


class TestTMMNumericalAccuracy:
    """Tests for numerical accuracy of TMM implementation."""

    def test_symmetric_data_factors_near_one(self):
        """For symmetric data, all factors should be close to 1."""
        # Create perfectly symmetric data with enough proteins
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        data = pd.DataFrame(
            {
                "S1": base,
                "S2": base,
                "S3": base,
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        tmm = TMMNormalizer()
        tmm.fit(data)

        # All factors should be exactly 1.0
        for factor in tmm.norm_factors_:
            assert np.isclose(factor, 1.0, atol=1e-6)

    def test_composition_bias_affects_factors(self):
        """Composition bias should result in different normalization factors."""
        n_proteins = 50
        base = np.linspace(100, 1000, n_proteins)

        # S1 and S3 are normal
        # S2 has composition bias: 10% of proteins are 10x higher
        s1 = base.copy()
        s2 = base.copy()
        s2[-5:] = s2[-5:] * 10  # Last 5 proteins are 10x higher (composition bias)
        s3 = base.copy()

        data = pd.DataFrame(
            {
                "S1": s1,
                "S2": s2,
                "S3": s3,
            },
            index=[f"P{i}" for i in range(n_proteins)],
        )

        tmm = TMMNormalizer()
        tmm.fit(data)

        # S2 should have different factor due to composition bias
        # (the outlier proteins inflate the library size but are trimmed in M calculation)
        factors = tmm.get_norm_factors()

        # S1 and S3 should have similar factors
        assert np.isclose(factors["S1"], factors["S3"], atol=0.1)
        assert not np.isclose(factors["S2"], factors["S1"])

        # After normalization, the non-outlier proteins in S2 should be adjusted
        result = tmm.transform(data)

        assert not np.allclose(result["S2"].iloc[:-5], data["S2"].iloc[:-5])


class TestTMMSampleNormalizationWiring:
    """TMM is wired into the peptide/sample normalization enum + pipeline."""

    def test_from_str_resolves_tmm(self):
        """PeptideNormalizationMethod.from_str('tmm') must return the TMM member."""
        from mokume.model.normalization import PeptideNormalizationMethod

        method = PeptideNormalizationMethod.from_str("tmm")
        assert method is PeptideNormalizationMethod.TMM
        # TMM needs the full matrix, so it must be dataset-level.
        assert method.is_dataset_level is True

    def test_tmm_in_cli_sample_normalization_choices(self):
        """Both features2* commands derive choices from the enum names."""
        from mokume.model.normalization import PeptideNormalizationMethod

        choices = [p.name.lower() for p in PeptideNormalizationMethod]
        assert "tmm" in choices

    def test_pipeline_runs_with_tmm_sample_normalization(self):
        """End-to-end: the pipeline runs TMM and yields a finite protein matrix.

        Also asserts the values differ from the ``none`` baseline, so a
        no-op regression (TMM silently skipped) would be caught.
        """
        from mokume.pipeline.config import (
            PipelineConfig,
            InputConfig,
            FilterConfig,
            NormalizationConfig,
            QuantificationConfig,
            IRSConfig,
            OutputConfig,
        )
        from mokume.pipeline.features_to_proteins import QuantificationPipeline

        def _make(sample_method: str) -> PipelineConfig:
            return PipelineConfig(
                input=InputConfig(parquet=_PARQUET, sdrf=_SDRF),
                filtering=FilterConfig(
                    min_aa=7,
                    min_unique_peptides=1,
                    remove_contaminants=True,
                ),
                normalization=NormalizationConfig(
                    run_method="median",
                    sample_method=sample_method,
                ),
                quantification=QuantificationConfig(method="median"),
                irs=IRSConfig(enabled=False),
                output=OutputConfig(),
            )

        ds_tmm = QuantificationPipeline(_make("tmm")).run_dataset()
        assert ds_tmm.proteins is not None
        assert len(ds_tmm.proteins) > 0

        tmm_wide = ds_tmm.proteins.set_index(ds_tmm.proteins.columns[0]).select_dtypes(
            include=[np.number]
        )
        tmm_vals = tmm_wide.values
        # At least some quantified values, and every non-NaN value finite.
        assert np.isfinite(tmm_vals).any()
        assert np.isfinite(tmm_vals[~np.isnan(tmm_vals)]).all()

        # TMM must actually change the matrix relative to no normalization.
        ds_none = QuantificationPipeline(_make("none")).run_dataset()
        none_wide = ds_none.proteins.set_index(
            ds_none.proteins.columns[0]
        ).select_dtypes(include=[np.number])

        common_idx = tmm_wide.index.intersection(none_wide.index)
        common_cols = [c for c in tmm_wide.columns if c in none_wide.columns]
        diff = (
            tmm_wide.loc[common_idx, common_cols].values
            - none_wide.loc[common_idx, common_cols].values
        )
        diff = diff[~np.isnan(diff)]
        assert diff.size > 0
        assert np.nanmax(np.abs(diff)) > 0.0
