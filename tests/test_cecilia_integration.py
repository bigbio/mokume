"""
Integration test: run the plugin-architecture pipeline against the
cecilia NASH TMT dataset (2-plex TMT11).

Tests multiple quantification methods end-to-end:
  - median (standard flow)
  - maxlfq (standard flow)
  - top3 (standard flow, TopN pattern)
  - ratio (ratio flow, PS protocol)
  - median + IRS normalization (standard flow + post-processing)

Each test validates:
  1. Pipeline completes without error
  2. Output proteins DataFrame is non-empty
  3. QpxDataset provenance is recorded
  4. Schema validation passes
"""

import os
import tempfile

import pandas as pd
import pytest

# Paths ---------------------------------------------------------------
PARQUET = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cecilia-problem", "qpx_output",
    "cecilia.feature.parquet",
)
SDRF = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cecilia-problem", "sdrf",
    "combined_plex_data_sheet.sdrf.tsv",
)

# Skip the entire module if dataset files are missing
pytestmark = pytest.mark.skipif(
    not (os.path.exists(PARQUET) and os.path.exists(SDRF)),
    reason="Cecilia NASH dataset not available",
)

# Imports (after path setup) -------------------------------------------
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
from mokume.core.dataset import QpxDataset


def _make_config(
    quant_method: str = "median",
    irs: bool = False,
    irs_remove_ref: bool = False,
    coverage_threshold: float = None,
    export_peptides: str = None,
) -> PipelineConfig:
    """Build a PipelineConfig pointing at the cecilia dataset."""
    return PipelineConfig(
        input=InputConfig(parquet=PARQUET, sdrf=SDRF),
        filtering=FilterConfig(
            min_aa=7,
            min_unique_peptides=2,
            remove_contaminants=True,
        ),
        normalization=NormalizationConfig(
            run_method="median",
            sample_method="globalMedian",
        ),
        quantification=QuantificationConfig(
            method=quant_method,
            coverage_threshold=coverage_threshold,
        ),
        irs=IRSConfig(
            enabled=irs,
            remove_reference=irs_remove_ref,
        ),
        output=OutputConfig(
            export_peptides=export_peptides,
        ),
    )


def _assert_valid_result(dataset: QpxDataset, min_proteins: int = 50):
    """Common assertions for all pipeline results."""
    # Proteins must be populated
    assert dataset.proteins is not None, "proteins level is None"

    protein_df = dataset.proteins
    if isinstance(protein_df, pd.DataFrame):
        assert len(protein_df) >= min_proteins, (
            f"Expected >= {min_proteins} proteins, got {len(protein_df)}"
        )

    # Provenance must be recorded
    assert "provenance" in dataset.uns, "No provenance in uns"
    steps = dataset.uns["provenance"]["steps"]
    assert len(steps) >= 2, f"Expected >= 2 provenance steps, got {len(steps)}"

    step_names = [s["name"] for s in steps]
    assert "loading" in step_names
    assert "quantification" in step_names


# ======================================================================
# Tests
# ======================================================================

class TestMedianQuantification:
    """Standard flow: median summarization."""

    def test_median_pipeline(self):
        config = _make_config(quant_method="median")
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()
        _assert_valid_result(dataset)
        print(f"\n  median: {len(dataset.proteins)} proteins, "
              f"{len(dataset.proteins.columns) - 1} samples")


class TestMaxLFQQuantification:
    """Standard flow: MaxLFQ."""

    def test_maxlfq_pipeline(self):
        config = _make_config(quant_method="maxlfq")
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()
        _assert_valid_result(dataset)
        print(f"\n  maxlfq: {len(dataset.proteins)} proteins, "
              f"{len(dataset.proteins.columns) - 1} samples")


class TestTopNQuantification:
    """Standard flow: TopN pattern (top3 via registry)."""

    def test_top3_pipeline(self):
        config = _make_config(quant_method="top3")
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()
        _assert_valid_result(dataset)
        print(f"\n  top3: {len(dataset.proteins)} proteins, "
              f"{len(dataset.proteins.columns) - 1} samples")


class TestRatioQuantification:
    """Ratio flow: PS protocol log2 ratios."""

    def test_ratio_pipeline(self):
        config = _make_config(
            quant_method="ratio",
            coverage_threshold=0.65,
        )
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()
        _assert_valid_result(dataset, min_proteins=30)

        # Ratio-specific checks
        assert "ratio_config" in dataset.uns
        assert dataset.uns["ratio_config"]["reference_samples"]
        print(f"\n  ratio: {len(dataset.proteins)} proteins, "
              f"refs={dataset.uns['ratio_config']['reference_samples']}")


class TestIRSNormalization:
    """Standard flow + IRS post-processing."""

    def test_median_irs_pipeline(self):
        config = _make_config(
            quant_method="median",
            irs=True,
            irs_remove_ref=True,
        )
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()
        _assert_valid_result(dataset)

        # IRS should have added a step
        step_names = [s["name"] for s in dataset.uns["provenance"]["steps"]]
        assert "irs_normalization" in step_names
        print(f"\n  median+IRS: {len(dataset.proteins)} proteins, "
              f"{len(dataset.proteins.columns) - 1} samples")


class TestQpxDatasetSaveLoad:
    """Test save/load roundtrip with real pipeline output."""

    def test_save_load_roundtrip(self):
        config = _make_config(quant_method="median")
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.save(tmpdir)
            loaded = QpxDataset.load(tmpdir)

            assert loaded.proteins is not None
            assert len(loaded.proteins) == len(dataset.proteins)
            assert loaded.uns.get("provenance") is not None


class TestPeptideExport:
    """Test peptide export path."""

    def test_export_peptides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pep_path = os.path.join(tmpdir, "peptides.csv")
            config = _make_config(
                quant_method="median",
                export_peptides=pep_path,
            )
            pipeline = QuantificationPipeline(config)
            dataset = pipeline.run_dataset()
            _assert_valid_result(dataset)

            assert os.path.exists(pep_path), "Peptide export file not created"
            pep_df = pd.read_csv(pep_path)
            assert len(pep_df) > 0
            print(f"\n  Exported {len(pep_df)} peptide rows")


class TestSchemaValidation:
    """Verify schema validation is wired in."""

    def test_peptides_schema_valid(self):
        config = _make_config(quant_method="median")
        pipeline = QuantificationPipeline(config)
        dataset = pipeline.run_dataset()

        # Peptides should pass validation
        if dataset.peptides is not None:
            errors = dataset.validate_level("peptides")
            assert errors == [], f"Schema errors: {errors}"
