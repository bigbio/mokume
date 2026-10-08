import logging
from pathlib import Path

import pandas as pd
import pytest
from mokume.model.labeling import QuantificationCategory
from mokume.normalization.peptide import (
    peptide_normalization,
    SQLFilterBuilder,
    Feature,
)

TESTS_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TestSQLFilterBuilder:
    """Tests for the SQLFilterBuilder class."""

    def test_default_where_clause(self):
        """Test that default filter builder generates expected WHERE clause."""
        builder = SQLFilterBuilder()
        where_clause, params = builder.build_where_clause()

        # Should include intensity > 0
        assert "intensity > 0" in where_clause
        # Should include peptide length filter (parameterized)
        assert 'LENGTH("sequence") >= ?' in where_clause
        assert 7 in params
        # Should include unique peptide filter
        assert '"unique" = 1' in where_clause
        # Should include contaminant filters (parameterized with ? placeholders)
        assert "list_bool_and(list_transform(pg_accessions" in where_clause
        assert "CONTAMINANT" in params
        assert "CONTAM_" in params
        assert "DECOY" in params
        assert "ENTRAP" in params

    def test_custom_contaminant_patterns(self):
        """Test filter builder with custom contaminant patterns."""
        builder = SQLFilterBuilder(
            contaminant_patterns=["CONTAM", "REV_"],
            min_peptide_length=5,
        )
        where_clause, params = builder.build_where_clause()

        assert "CONTAM" in params
        assert "REV_" in params
        assert "DECOY" not in params
        assert 'LENGTH("sequence") >= ?' in where_clause
        assert 5 in params

    def test_disable_contaminant_filter(self):
        """Test that contaminant filter can be disabled."""
        builder = SQLFilterBuilder(remove_contaminants=False)
        where_clause, _params = builder.build_where_clause()

        assert "pg_accessions" not in where_clause
        # Other filters should still be present
        assert "intensity > 0" in where_clause

    def test_decoy_only_filter_patterns(self):
        """A config can remove decoys without removing contaminants."""
        from mokume.model.filters import PreprocessingFilterConfig

        config = PreprocessingFilterConfig(name="decoys_only")
        config.protein.remove_contaminants = False
        config.protein.remove_decoys = True

        builder = SQLFilterBuilder(
            remove_contaminants=True,
            contaminant_patterns=config.protein.active_contaminant_patterns(),
        )
        _where_clause, params = builder.build_where_clause()

        assert "DECOY" in params
        assert "CONTAMINANT" not in params
        assert "CONTAM_" not in params
        assert "ENTRAP" not in params

    def test_min_intensity_threshold(self):
        """Test that min intensity threshold is applied."""
        builder = SQLFilterBuilder(min_intensity=1000.0)
        where_clause, params = builder.build_where_clause()

        assert "intensity >= ?" in where_clause
        assert 1000.0 in params

    def test_disable_unique_requirement(self):
        """Test that unique peptide requirement can be disabled."""
        builder = SQLFilterBuilder(require_unique=False)
        where_clause, _params = builder.build_where_clause()

        assert '"unique" = 1' not in where_clause


class TestFeatureWideFormat:
    """Tests for Feature class with wide format parquet (quantms.io/qpx)."""

    @pytest.fixture
    def feature_path(self):
        """Path to test feature parquet file (wide format)."""
        return str(TESTS_DIR / "example/feature_wide.parquet")

    @pytest.fixture
    def sdrf_path(self):
        """Path to test SDRF file."""
        return str(TESTS_DIR / "example/PXD020192.sdrf.tsv")

    def test_feature_loads_wide_format(self, feature_path):
        """Test that Feature class can load wide format parquet with UNNEST."""
        feature = Feature(feature_path)

        # Should have samples from UNNEST
        samples = feature.get_unique_samples()
        assert len(samples) > 0

    def test_feature_with_filter_builder(self, feature_path):
        """Test Feature class accepts and stores filter_builder."""
        builder = SQLFilterBuilder(
            remove_contaminants=True,
            min_peptide_length=7,
        )
        feature = Feature(feature_path, filter_builder=builder)

        assert feature.filter_builder is not None
        assert feature.filter_builder.remove_contaminants is True
        assert feature.filter_builder.min_peptide_length == 7

    def test_get_median_map(self, feature_path):
        """Test that get_median_map works with wide format."""
        feature = Feature(feature_path)
        med_map = feature.get_median_map()

        # Should return results
        assert len(med_map) > 0
        # All values should be positive
        for sample, factor in med_map.items():
            assert factor > 0

    def test_get_median_map_with_filter(self, feature_path):
        """Test that get_median_map uses filter_builder when provided."""
        # Without filter
        feature_no_filter = Feature(feature_path)
        med_map_unfiltered = feature_no_filter.get_median_map()

        # With filter (excluding contaminants)
        builder = SQLFilterBuilder(remove_contaminants=True)
        feature_filtered = Feature(feature_path, filter_builder=builder)
        med_map_filtered = feature_filtered.get_median_map()

        # Both should return results
        assert len(med_map_unfiltered) > 0
        assert len(med_map_filtered) > 0

        # The samples should be the same (filtering applies to features, not samples)
        assert set(med_map_unfiltered.keys()) == set(med_map_filtered.keys())

    def test_get_low_frequency_peptides(self, feature_path):
        """Test that get_low_frequency_peptides works with wide format."""
        feature = Feature(feature_path)
        low_freq = feature.get_low_frequency_peptides()

        # Should return a tuple
        assert isinstance(low_freq, tuple)

    def test_get_median_map_to_condition(self, feature_path):
        """Test that get_median_map_to_condition works with wide format."""
        feature = Feature(feature_path)
        med_map = feature.get_median_map_to_condition()

        # Should return dict of dicts
        assert isinstance(med_map, dict)
        for condition, samples in med_map.items():
            assert isinstance(samples, dict)

    def test_enrich_with_sdrf(self, feature_path, sdrf_path):
        """Test that enrich_with_sdrf enriches data with SDRF metadata."""
        feature = Feature(feature_path)

        # Before enrichment, condition should equal sample_accession
        conditions_before = feature.get_unique_conditions()
        _ = feature.get_unique_samples()
        # Conditions default to sample_accession
        assert len(conditions_before) > 0

        # Enrich with SDRF
        feature.enrich_with_sdrf(sdrf_path)

        # After enrichment, conditions should be from SDRF
        conditions_after = feature.get_unique_conditions()
        assert len(conditions_after) > 0

    def test_iter_samples(self, feature_path):
        """Test that iter_samples works with wide format."""
        feature = Feature(feature_path)

        count = 0
        for samples, df in feature.iter_samples(sample_num=5):
            assert len(samples) <= 5
            assert len(df) > 0
            count += 1

        assert count > 0


class TestFeatureNewQPXFormat:
    """Tests for Feature class with new QPX format parquet (charge, run_file_name, {label, intensity})."""

    @pytest.fixture
    def feature_path(self):
        return str(TESTS_DIR / "example/feature_wide_new_qpx.parquet")

    @pytest.fixture
    def sdrf_path(self):
        return str(TESTS_DIR / "example/sdrf_new_qpx.tsv")

    def test_loads_new_qpx_format(self, feature_path):
        """Test that Feature detects and loads new QPX schema."""
        feature = Feature(feature_path)
        assert feature._is_new_qpx is True
        assert feature._charge_col == "charge"
        assert feature._run_col == "run_file_name"

        samples = feature.get_unique_samples()
        assert len(samples) > 0

    def test_unnested_columns_present(self, feature_path):
        """Test that unnested view has expected column names."""
        feature = Feature(feature_path)
        df = feature.parquet_db.sql("SELECT * FROM parquet_db LIMIT 1").df()
        for col in [
            "charge",
            "run_file_name",
            "sample_accession",
            "channel",
            "intensity",
            "condition",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enrich_with_sdrf_maps_sample_accession(self, feature_path, sdrf_path):
        """Test that enrich_with_sdrf correctly maps (run_file_name, label) -> source name."""
        feature = Feature(feature_path)
        feature.enrich_with_sdrf(sdrf_path)

        samples = feature.get_unique_samples()
        # After SDRF enrichment, samples should be SDRF source names
        assert "Sample_A_126" in samples, (
            f"'Sample_A_126' not found in samples: {samples}"
        )
        assert "Sample_A_127N" in samples, (
            f"'Sample_A_127N' not found in samples: {samples}"
        )

    def test_enrich_with_sdrf_maps_condition(self, feature_path, sdrf_path):
        """Test that enrich_with_sdrf maps conditions from SDRF factor values."""
        feature = Feature(feature_path)
        feature.enrich_with_sdrf(sdrf_path)

        conditions = feature.get_unique_conditions()
        assert "normal" in conditions, f"'normal' not found in conditions: {conditions}"
        assert "disease" in conditions, (
            f"'disease' not found in conditions: {conditions}"
        )

    def test_get_median_map(self, feature_path):
        """Test get_median_map works with new QPX format."""
        feature = Feature(feature_path)
        med_map = feature.get_median_map()
        assert len(med_map) > 0
        for sample, factor in med_map.items():
            assert factor > 0


def _shared_peptide_feature_rows() -> pd.DataFrame:
    """Build feature rows containing one unique and one shared peptide."""
    return pd.DataFrame.from_records(
        [
            (["P1"], "UNIQUEPEP", 100.0, True),
            (["P1", "P2"], "SHAREDPEP", 80.0, False),
        ],
        columns=["pg_accessions", "sequence", "intensity", "unique"],
    ).assign(
        peptidoform=lambda df: df["sequence"],
        charge=2,
        channel="raw",
        condition="case",
        biological_replicate=1,
        run="run1",
        fraction="1",
        run_file_name="run1",
        sample_accession="Sample1",
    )


def _make_fake_feature(feature_rows, captured_filter_builders):
    """Build a Feature replacement that returns deterministic rows."""

    class FakeFeature:
        """Feature test double for peptide_normalization."""

        def __init__(self, _parquet, filter_builder=None):
            """Capture the filter builder created by peptide_normalization."""
            captured_filter_builders.append(filter_builder)

        @property
        def experimental_inference(self):
            """Return LFQ metadata for the fake sample."""
            return (1, QuantificationCategory.LFQ, ["Sample1"], None)

        def iter_samples(self):
            """Yield one sample with a copy of the input rows."""
            yield ["Sample1"], feature_rows.copy()

    return FakeFeature


def _run_keep_shared_case(output, keep_shared_peptides, min_unique=1) -> pd.DataFrame:
    """Run peptide_normalization for the shared peptide regression case."""
    peptide_normalization(
        parquet="unused.parquet",
        sdrf=None,
        min_aa=7,
        min_unique=min_unique,
        remove_ids=None,
        remove_decoy_contaminants=False,
        remove_low_frequency_peptides=False,
        output=str(output),
        skip_normalization=True,
        nmethod="none",
        pnmethod="none",
        log2=False,
        save_parquet=False,
        keep_shared_peptides=keep_shared_peptides,
    )
    return pd.read_csv(output)


class TestPeptideNormalizationWideFormat:
    """Tests for peptide_normalization with wide format parquet."""

    def test_keep_shared_peptides_retains_non_unique_rows(self, monkeypatch, tmp_path):
        """Keep shared peptide rows when --keep-shared-peptides is enabled."""
        captured_filter_builders = []
        fake_feature = _make_fake_feature(
            _shared_peptide_feature_rows(),
            captured_filter_builders,
        )
        monkeypatch.setattr("mokume.normalization.peptide.Feature", fake_feature)
        unique_only_df = _run_keep_shared_case(
            tmp_path / "unique_only.csv",
            keep_shared_peptides=False,
        )
        assert set(unique_only_df["ProteinName"]) == {"P1"}

        keep_shared_df = _run_keep_shared_case(
            tmp_path / "keep_shared.csv",
            keep_shared_peptides=True,
            min_unique=2,
        )
        assert set(keep_shared_df["ProteinName"]) == {"P1", "P1;P2"}
        assert captured_filter_builders[-1].require_unique is False

    def test_normalization_without_sdrf(self):
        """Test peptide normalization without SDRF (uses defaults)."""
        args = {
            "parquet": str(TESTS_DIR / "example/feature_wide.parquet"),
            "sdrf": None,
            "min_aa": 7,
            "min_unique": 1,
            "remove_ids": None,
            "remove_decoy_contaminants": True,
            "remove_low_frequency_peptides": False,
            "output": str(TESTS_DIR / "example" / "out" / "PXD020192-no-sdrf.csv"),
            "skip_normalization": False,
            "nmethod": "median",
            "pnmethod": "globalMedian",
            "log2": True,
            "save_parquet": False,
        }

        out = Path(args["output"])
        if out.exists():
            out.unlink()

        # Should complete without errors
        peptide_normalization(**args)

        # Output should exist
        assert out.exists()

        # Clean up
        if out.exists():
            out.unlink()

    def test_normalization_with_sdrf(self):
        """Test peptide normalization with SDRF enrichment."""
        args = {
            "parquet": str(TESTS_DIR / "example/feature_wide.parquet"),
            "sdrf": str(TESTS_DIR / "example/PXD020192.sdrf.tsv"),
            "min_aa": 7,
            "min_unique": 1,
            "remove_ids": None,
            "remove_decoy_contaminants": True,
            "remove_low_frequency_peptides": False,
            "output": str(TESTS_DIR / "example" / "out" / "PXD020192-with-sdrf.csv"),
            "skip_normalization": False,
            "nmethod": "median",
            "pnmethod": "globalMedian",
            "log2": True,
            "save_parquet": False,
        }

        out = Path(args["output"])
        if out.exists():
            out.unlink()

        # Should complete without errors
        peptide_normalization(**args)

        # Output should exist
        assert out.exists()

        # Clean up
        if out.exists():
            out.unlink()

    def test_normalization_with_filter_config(self):
        """Test peptide normalization with PreprocessingFilterConfig."""
        from mokume.model.filters import PreprocessingFilterConfig

        # Create a filter config
        filter_config = PreprocessingFilterConfig(
            name="test_config",
            enabled=True,
        )
        # Set contaminant removal
        filter_config.protein.remove_contaminants = True
        filter_config.protein.remove_decoys = True

        args = {
            "parquet": str(TESTS_DIR / "example/feature_wide.parquet"),
            "sdrf": str(TESTS_DIR / "example/PXD020192.sdrf.tsv"),
            "min_aa": 7,
            "min_unique": 1,
            "remove_ids": None,
            "remove_decoy_contaminants": True,
            "remove_low_frequency_peptides": False,
            "output": str(TESTS_DIR / "example" / "out" / "PXD020192-filtered.csv"),
            "skip_normalization": False,
            "nmethod": "median",
            "pnmethod": "globalMedian",
            "log2": True,
            "save_parquet": False,
            "filter_config": filter_config,
        }

        out = Path(args["output"])
        if out.exists():
            out.unlink()

        # Should complete without errors
        peptide_normalization(**args)

        # Output should exist
        assert out.exists()

        # Clean up
        if out.exists():
            out.unlink()
