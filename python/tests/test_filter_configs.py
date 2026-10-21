"""
Tests for preprocessing filter configurations.

This module tests that all example filter configurations can be loaded
and that the filter system works correctly.
"""

import pytest
from pathlib import Path

from mokume.preprocessing.filters import (
    load_filter_config,
    get_filter_pipeline,
    generate_example_config,
)
from mokume.model.filters import PreprocessingFilterConfig


# Path to example filter configurations
EXAMPLE_FILTERS_DIR = Path(__file__).parent / "example" / "filters"


def get_example_config_files():
    """Get all example filter configuration files."""
    if not EXAMPLE_FILTERS_DIR.exists():
        return []
    return list(EXAMPLE_FILTERS_DIR.glob("*.yaml"))


class TestFilterConfigurations:
    """Tests for filter configuration loading and validation."""

    @pytest.mark.parametrize(
        "config_file",
        get_example_config_files(),
        ids=lambda p: p.stem,
    )
    def test_load_example_config(self, config_file):
        """Test that example configurations can be loaded."""
        config = load_filter_config(config_file)

        assert config is not None
        assert isinstance(config, PreprocessingFilterConfig)
        assert config.name is not None
        assert config.enabled is True

    @pytest.mark.parametrize(
        "config_file",
        get_example_config_files(),
        ids=lambda p: p.stem,
    )
    def test_create_pipeline_from_config(self, config_file):
        """Test that filter pipelines can be created from configurations."""
        config = load_filter_config(config_file)
        pipeline = get_filter_pipeline(config)

        assert pipeline.name == config.name
        assert len(pipeline) > 0

    def test_basic_qc_config(self):
        """Test the basic QC configuration specifically."""
        config_path = EXAMPLE_FILTERS_DIR / "basic_qc.yaml"
        config = load_filter_config(config_path)

        assert config.name == "basic_qc"
        assert config.protein.min_unique_peptides == 2
        assert config.protein.remove_contaminants is True
        assert config.peptide.min_peptide_length == 7

    def test_stringent_filtering_config(self):
        """Test the stringent filtering configuration specifically."""
        config_path = EXAMPLE_FILTERS_DIR / "stringent_filtering.yaml"
        config = load_filter_config(config_path)

        assert config.name == "stringent_filtering"
        assert config.intensity.min_intensity == 1000.0
        assert config.intensity.cv_threshold == 0.3
        assert config.peptide.allowed_charge_states == [2, 3, 4]
        assert "Oxidation" in config.peptide.exclude_modifications

    def test_generate_example_config(self, tmp_path):
        """Test that example config generation works."""
        yaml_path = tmp_path / "test_config.yaml"
        generate_example_config(yaml_path)

        assert yaml_path.exists()

        # Load and verify
        config = load_filter_config(yaml_path)
        assert config.name == "example_config"

    def test_generate_json_config(self, tmp_path):
        """Test that JSON config generation works."""
        json_path = tmp_path / "test_config.json"
        generate_example_config(json_path, format="json")

        assert json_path.exists()

        # Load and verify
        config = load_filter_config(json_path)
        assert config is not None

    def test_config_apply_overrides(self):
        """Test that CLI overrides work correctly."""
        config = PreprocessingFilterConfig(name="test")

        config.apply_overrides(
            {
                "min_intensity": 500.0,
                "cv_threshold": 0.25,
                "min_replicate_agreement": 2,
                "charge_states": [2, 3],
                "peptide_fdr": 0.02,
                "min_unique_peptides": 3,
                "protein_fdr": 0.03,
                "max_missing_rate": 0.4,
            }
        )

        assert config.intensity.min_intensity == 500.0
        assert config.intensity.cv_threshold == 0.25
        assert config.intensity.min_replicate_agreement == 2
        assert config.peptide.allowed_charge_states == [2, 3]
        assert config.peptide.fdr_threshold == 0.02
        assert config.protein.min_unique_peptides == 3
        assert config.protein.fdr_threshold == 0.03
        assert config.run_qc.max_missing_rate == 0.4

    def test_unknown_top_level_key_is_rejected(self):
        """A misspelled config key must not become a silent no-op."""
        with pytest.raises(ValueError, match="Unknown preprocessing filter keys"):
            PreprocessingFilterConfig.from_dict({"min_intensitty": 100.0})

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            (None, "strict_mode"),
            ("intensity", "remove_zero_intensity"),
            ("peptide", "min_search_score"),
            ("peptide", "require_unique_peptides"),
            ("protein", "min_coverage"),
            ("protein", "protein_grouping"),
            ("run_qc", "min_sample_correlation"),
        ],
    )
    def test_unimplemented_filter_options_are_rejected(self, section, key):
        """Filter configs must reject options with no execution path."""
        data = {key: False} if section is None else {section: {key: False}}

        with pytest.raises((TypeError, ValueError)):
            PreprocessingFilterConfig.from_dict(data)

    @pytest.mark.parametrize(
        ("remove_contaminants", "remove_decoys", "removed", "retained"),
        [
            (False, True, "DECOY_P1", "CONTAMINANT_P2"),
            (True, False, "CONTAMINANT_P2", "DECOY_P1"),
        ],
    )
    def test_contaminant_and_decoy_switches_are_independent(
        self,
        remove_contaminants,
        remove_decoys,
        removed,
        retained,
    ):
        """Each protein-filter switch changes the applied pattern set."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="pattern_switch")
        config.protein.remove_contaminants = remove_contaminants
        config.protein.remove_decoys = remove_decoys
        pipeline = get_filter_pipeline(config)
        contaminant_filter = next(
            item for item in pipeline.filters if item.name == "ContaminantFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["DECOY_P1", "CONTAMINANT_P2", "P3"],
                "PeptideCanonical": ["PEPTIDEA", "PEPTIDEB", "PEPTIDEC"],
            }
        )

        filtered, _result = contaminant_filter.apply(frame)

        assert removed not in set(filtered["ProteinName"])
        assert retained in set(filtered["ProteinName"])


class TestFilterPipeline:
    """Tests for filter pipeline functionality."""

    def test_empty_pipeline(self):
        """Test that disabled config creates empty pipeline."""
        config = PreprocessingFilterConfig(name="disabled", enabled=False)
        pipeline = get_filter_pipeline(config)

        assert len(pipeline) == 0

    def test_pipeline_with_intensity_filters(self):
        """Test pipeline with intensity filters."""
        config = PreprocessingFilterConfig(name="test")
        config.intensity.min_intensity = 100.0
        config.intensity.cv_threshold = 0.5

        pipeline = get_filter_pipeline(config)

        # Should have at least MinIntensityFilter and CVThresholdFilter
        filter_names = [f.name for f in pipeline.filters]
        assert "MinIntensityFilter" in filter_names
        assert "CVThresholdFilter" in filter_names

    def test_pipeline_with_protein_filters(self):
        """Test pipeline with protein filters."""
        config = PreprocessingFilterConfig(name="test")
        config.protein.min_unique_peptides = 2
        config.protein.remove_contaminants = True

        pipeline = get_filter_pipeline(config)

        filter_names = [f.name for f in pipeline.filters]
        assert "ContaminantFilter" in filter_names
        assert "MinPeptideFilter" in filter_names

    def test_explicit_qpx_fdr_filters_apply_and_require_values(self):
        """Opt-in FDR cutoffs must filter dedicated QPX q-value fields."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="qpx_fdr")
        config.peptide.fdr_threshold = 0.01
        config.protein.fdr_threshold = 0.01
        pipeline = get_filter_pipeline(config)
        peptide_filter = next(
            item for item in pipeline.filters if item.name == "PeptideFDRFilter"
        )
        protein_filter = next(
            item for item in pipeline.filters if item.name == "ProteinFDRFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["P1", "P1", "P2", "P2"],
                "PeptideCanonical": ["A", "B", "C", "D"],
                "peptide_qvalue": [0.005, 0.02, 0.005, 0.005],
                "pg_global_qvalue": [0.02, 0.02, 0.02, 0.005],
            }
        )

        peptide_rows, _result = peptide_filter.apply(frame)
        assert set(peptide_rows["PeptideCanonical"]) == {"A", "C", "D"}
        protein_rows, _result = protein_filter.apply(frame)
        assert set(protein_rows["ProteinName"]) == {"P2"}
        with pytest.raises(ValueError, match="populated QPX 'peptide_qvalue'"):
            peptide_filter.apply(frame.drop(columns="peptide_qvalue"))

    def test_min_peptides_is_applied_when_stricter_than_unique_threshold(self):
        """The total-peptide threshold must not be ignored when sequences exist."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="min_peptide_threshold")
        config.protein.remove_contaminants = False
        config.protein.remove_decoys = False
        config.protein.min_peptides = 3
        config.protein.min_unique_peptides = 1
        pipeline = get_filter_pipeline(config)
        peptide_filter = next(
            item for item in pipeline.filters if item.name == "MinPeptideFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["P1", "P1", "P2", "P2", "P2"],
                "PeptideCanonical": ["A", "B", "A", "B", "C"],
            }
        )

        filtered, _result = peptide_filter.apply(frame)

        assert set(filtered["ProteinName"]) == {"P2"}

    def test_replicate_agreement_counts_technical_replicates(self):
        """Per-sample filtering must not group on the constant SampleID."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="replicate_agreement")
        config.intensity.min_replicate_agreement = 2
        pipeline = get_filter_pipeline(config)
        replicate_filter = next(
            item for item in pipeline.filters if item.name == "ReplicateAgreementFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["P1", "P1", "P2"],
                "PeptideCanonical": ["A", "A", "B"],
                "Condition": ["control", "control", "control"],
                "TechReplicate": [1, 2, 1],
                "NormIntensity": [10.0, 11.0, 12.0],
            }
        )

        filtered, _result = replicate_filter.apply(frame)

        assert set(filtered["ProteinName"]) == {"P1"}

    def test_missing_rate_uses_the_complete_feature_universe(self):
        """Absent long-format rows must contribute to per-run missingness."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="missing_rate")
        config.run_qc.max_missing_rate = 0.4
        pipeline = get_filter_pipeline(config)
        missing_filter = next(
            item for item in pipeline.filters if item.name == "MissingRateFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["P1", "P2", "P3", "P1"],
                "PeptideCanonical": ["A", "B", "C", "A"],
                "TechReplicate": [1, 1, 1, 2],
                "NormIntensity": [10.0, 11.0, 12.0, 9.0],
            }
        )

        filtered, _result = missing_filter.apply(frame)

        assert set(filtered["TechReplicate"]) == {1}

    def test_min_features_counts_distinct_features_per_technical_run(self):
        """Duplicate rows must not inflate a run's identified-feature count."""
        import pandas as pd

        config = PreprocessingFilterConfig(name="min_features")
        config.run_qc.min_identified_features = 3
        pipeline = get_filter_pipeline(config)
        feature_filter = next(
            item for item in pipeline.filters if item.name == "MinFeaturesFilter"
        )
        frame = pd.DataFrame(
            {
                "ProteinName": ["P1", "P1", "P2", "P1", "P2", "P3"],
                "PeptideCanonical": ["A", "A", "B", "A", "B", "C"],
                "TechReplicate": [1, 1, 1, 2, 2, 2],
                "NormIntensity": [10.0, 10.0, 11.0, 9.0, 12.0, 13.0],
            }
        )

        filtered, _result = feature_filter.apply(frame)

        assert set(filtered["TechReplicate"]) == {2}
