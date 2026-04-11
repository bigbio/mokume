"""Tests for QpxDataset."""

import json
import os
import tempfile

import pandas as pd
import pytest

from mokume.core.constants import PROTEIN_NAME, SAMPLE_ID, PEPTIDE_CANONICAL, NORM_INTENSITY
from mokume.core.dataset import QpxDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_peptide_df():
    """Minimal peptide-level DataFrame."""
    return pd.DataFrame({
        PROTEIN_NAME: ["P1", "P1", "P2", "P2"],
        PEPTIDE_CANONICAL: ["PEPTIDE", "ANOTHERPEP", "THIRDPEP", "FOURTHPEP"],
        SAMPLE_ID: ["S1", "S2", "S1", "S2"],
        NORM_INTENSITY: [100.0, 200.0, 300.0, 400.0],
    })


@pytest.fixture
def sample_protein_df():
    """Minimal protein-level DataFrame."""
    return pd.DataFrame({
        PROTEIN_NAME: ["P1", "P2"],
        SAMPLE_ID: ["S1", "S1"],
        "Intensity": [500.0, 600.0],
    })


@pytest.fixture
def dataset_with_peptides(sample_peptide_df):
    ds = QpxDataset()
    ds.peptides = sample_peptide_df
    return ds


# ---------------------------------------------------------------------------
# Tests: data level access
# ---------------------------------------------------------------------------

class TestDataLevelAccess:
    def test_get_level_returns_dataframe(self, dataset_with_peptides):
        df = dataset_with_peptides.get_level("peptides")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_get_level_none_when_empty(self):
        ds = QpxDataset()
        assert ds.get_level("peptides") is None

    def test_get_level_invalid_raises(self):
        ds = QpxDataset()
        with pytest.raises(ValueError, match="Unknown data level"):
            ds.get_level("invalid_level")

    def test_set_level(self, sample_peptide_df):
        ds = QpxDataset()
        ds.set_level("peptides", sample_peptide_df)
        assert ds.peptides is not None
        assert len(ds.get_level("peptides")) == 4

    def test_set_level_invalid_raises(self, sample_peptide_df):
        ds = QpxDataset()
        with pytest.raises(ValueError, match="Unknown data level"):
            ds.set_level("invalid", sample_peptide_df)

    def test_populated_levels(self, sample_peptide_df, sample_protein_df):
        ds = QpxDataset()
        assert ds.populated_levels == []
        ds.peptides = sample_peptide_df
        assert ds.populated_levels == ["peptides"]
        ds.proteins = sample_protein_df
        assert set(ds.populated_levels) == {"peptides", "proteins"}


# ---------------------------------------------------------------------------
# Tests: schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_validate_valid_peptides(self, dataset_with_peptides):
        errors = dataset_with_peptides.validate_level("peptides")
        assert errors == []

    def test_validate_missing_columns(self):
        ds = QpxDataset()
        ds.peptides = pd.DataFrame({"A": [1]})
        errors = ds.validate_level("peptides")
        assert len(errors) > 0
        assert any("Missing required column" in e for e in errors)

    def test_validate_unpopulated_level(self):
        ds = QpxDataset()
        errors = ds.validate_level("proteins")
        assert len(errors) == 1
        assert "not populated" in errors[0]


# ---------------------------------------------------------------------------
# Tests: wide matrix
# ---------------------------------------------------------------------------

class TestToWideMatrix:
    def test_pivot_proteins(self, sample_protein_df):
        ds = QpxDataset()
        ds.proteins = sample_protein_df
        wide = ds.to_wide_matrix(level="proteins")
        assert isinstance(wide, pd.DataFrame)
        assert wide.index.name == PROTEIN_NAME

    def test_wide_empty_level_raises(self):
        ds = QpxDataset()
        with pytest.raises(ValueError, match="not populated"):
            ds.to_wide_matrix(level="proteins")


# ---------------------------------------------------------------------------
# Tests: subsetting
# ---------------------------------------------------------------------------

class TestSubsetting:
    def test_subset_samples(self, sample_peptide_df):
        ds = QpxDataset()
        ds.peptides = sample_peptide_df
        subset = ds.subset_samples(["S1"])
        df = subset.get_level("peptides")
        assert list(df[SAMPLE_ID].unique()) == ["S1"]

    def test_subset_proteins(self, sample_peptide_df):
        ds = QpxDataset()
        ds.peptides = sample_peptide_df
        subset = ds.subset_proteins(["P1"])
        df = subset.get_level("peptides")
        assert list(df[PROTEIN_NAME].unique()) == ["P1"]


# ---------------------------------------------------------------------------
# Tests: save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, sample_peptide_df, sample_protein_df):
        ds = QpxDataset()
        ds.peptides = sample_peptide_df
        ds.proteins = sample_protein_df
        ds.uns = {"pipeline": "test", "version": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "peptides.parquet"))
            assert os.path.exists(os.path.join(tmpdir, "proteins.parquet"))
            assert os.path.exists(os.path.join(tmpdir, "uns.json"))

            # Load back
            loaded = QpxDataset.load(tmpdir)
            assert loaded.get_level("peptides") is not None
            assert len(loaded.get_level("peptides")) == 4
            assert loaded.get_level("proteins") is not None
            assert loaded.uns["pipeline"] == "test"

    def test_save_with_layers(self, sample_peptide_df):
        ds = QpxDataset()
        ds.peptides = sample_peptide_df
        ds.layers["normalized"] = sample_peptide_df.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save(tmpdir)
            loaded = QpxDataset.load(tmpdir)
            assert "normalized" in loaded.layers
            assert len(loaded.layers["normalized"]) == 4


# ---------------------------------------------------------------------------
# Tests: provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_record_step(self):
        ds = QpxDataset()
        ds.record_step("loading", method="standard", rows_out=100)
        assert "provenance" in ds.uns
        assert len(ds.uns["provenance"]["steps"]) == 1
        step = ds.uns["provenance"]["steps"][0]
        assert step["name"] == "loading"
        assert step["method"] == "standard"
        assert step["rows_out"] == 100

    def test_record_multiple_steps(self):
        ds = QpxDataset()
        ds.record_step("loading")
        ds.record_step("normalization")
        ds.record_step("quantification")
        assert len(ds.uns["provenance"]["steps"]) == 3


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_empty(self):
        ds = QpxDataset()
        r = repr(ds)
        assert "QpxDataset(" in r

    def test_repr_with_data(self, sample_peptide_df):
        ds = QpxDataset()
        ds.peptides = sample_peptide_df
        r = repr(ds)
        assert "peptides:" in r
        assert "4 rows" in r
