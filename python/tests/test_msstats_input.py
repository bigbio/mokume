"""Regression coverage for native SDRF and MSstats input."""

from inspect import signature
from pathlib import Path

import pandas as pd
import pytest

from mokume.io.feature import Feature
from mokume.pipeline.config import (
    InputConfig,
    NormalizationConfig,
    PipelineConfig,
    QuantificationConfig,
)
from mokume.pipeline.features_to_proteins import features_to_proteins
from mokume.pipeline.runner import run_pipeline


def test_msstats_parameter_preserves_positional_api():
    """Adding MSstats input must not shift existing positional arguments."""
    arguments = (
        signature(features_to_proteins)
        .bind("input.parquet", "output.csv", "input.sdrf.tsv", "sum")
        .arguments
    )

    assert arguments["quant_method"] == "sum"
    assert "msstats" not in arguments


def _write_sdrf(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_lfq_msstats_runs_end_to_end(tmp_path):
    """LFQ MSstats rows resolve through SDRF and reach protein output."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    output = tmp_path / "proteins.csv"
    _write_sdrf(
        sdrf,
        [
            {
                "source name": "S1",
                "comment[data file]": "run.raw.one.raw",
                "comment[label]": "AC=MS:1002038;NT=label free sample",
                "comment[technical replicate]": "1",
                "comment[fraction identifier]": "1",
                "characteristics[biological replicate]": "1",
                "factor value[condition]": "A",
            },
            {
                "source name": "S2",
                "comment[data file]": "run.raw.two.mzML",
                "comment[label]": "AC=MS:1002038;NT=label free sample",
                "comment[technical replicate]": "1",
                "comment[fraction identifier]": "1",
                "characteristics[biological replicate]": "1",
                "factor value[condition]": "B",
            },
        ],
    )
    pd.DataFrame(
        [
            {
                "ProteinName": "P1",
                "PeptideSequence": "PEPTIDEK",
                "PrecursorCharge": 2,
                "Run": 1,
                "Intensity": 100.0,
                "Reference": r"C:\data\run.raw.one.raw",
            },
            {
                "ProteinName": "P1",
                "PeptideSequence": "PEPTIDEK",
                "PrecursorCharge": 2,
                "Run": 2,
                "Intensity": 200.0,
                "Reference": "run.raw.two.mzML",
            },
        ]
    ).to_csv(msstats, index=False)

    proteins = features_to_proteins(
        parquet=None,
        msstats=str(msstats),
        sdrf=str(sdrf),
        output=str(output),
        quant_method="sum",
        min_unique_peptides=1,
        run_normalization="none",
        sample_normalization="none",
    )

    row = proteins.set_index("ProteinName").loc["P1"]
    assert row["S1"] == 100.0
    assert row["S2"] == 200.0


def test_numeric_factor_and_pooled_replicate_load(tmp_path):
    """SDRF numeric factor values and pooled biological replicates load."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    output = tmp_path / "proteins.csv"
    _write_sdrf(
        sdrf,
        [
            {
                "source name": sample,
                "comment[data file]": f"{sample}.raw",
                "comment[label]": "label free sample",
                "comment[technical replicate]": "1",
                "comment[fraction identifier]": "1",
                "characteristics[biological replicate]": replicate,
                "factor value[dose]": dose,
            }
            for sample, replicate, dose in [("S1", "1", 10), ("S2", "pooled", 20)]
        ],
    )
    pd.DataFrame(
        [
            {
                "ProteinName": "P1",
                "PeptideSequence": "PEPTIDEK",
                "PrecursorCharge": 2,
                "Intensity": intensity,
                "Reference": f"{sample}.mzML",
            }
            for sample, intensity in [("S1", 100.0), ("S2", 200.0)]
        ]
    ).to_csv(msstats, index=False)

    proteins = features_to_proteins(
        parquet=None,
        msstats=str(msstats),
        sdrf=str(sdrf),
        output=str(output),
        quant_method="sum",
        min_unique_peptides=1,
        run_normalization="none",
        sample_normalization="none",
    )

    row = proteins.set_index("ProteinName").loc["P1"]
    assert row["S1"] == 100.0
    assert row["S2"] == 200.0


def test_tmt10_channel_positions_use_sdrf_labels(tmp_path):
    """TMT channel positions map to canonical SDRF labels."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    labels = [
        "TMT126",
        "TMT127N",
        "TMT127C",
        "TMT128N",
        "TMT128C",
        "TMT129N",
        "TMT129C",
        "TMT130N",
        "TMT130C",
        "TMT131",
    ]
    _write_sdrf(
        sdrf,
        [
            {
                "source name": f"S{position}",
                "comment[data file]": "plex.raw",
                "comment[label]": label,
                "comment[technical replicate]": "1",
                "comment[fraction identifier]": "1",
                "characteristics[biological replicate]": str(position),
                "factor value[condition]": "A",
            }
            for position, label in enumerate(labels, start=1)
        ],
    )
    pd.DataFrame(
        [
            {
                "ProteinName": "P1",
                "PeptideSequence": ".(TMT6plex)PEPTIDEK",
                "Charge": 2,
                "Channel": channel,
                "Run": "1_1_1",
                "Intensity": intensity,
                "Reference": "plex.mzML_controllerType=0 controllerNumber=1 scan=1",
            }
            for channel, intensity in [(1, 100.0), (10, 1000.0)]
        ]
    ).to_csv(msstats, index=False)

    feature = Feature.from_msstats(str(msstats), str(sdrf))
    try:
        feature.enrich_with_sdrf(str(sdrf))
        rows = feature.parquet_db.execute(
            "SELECT channel, sample_accession, sequence, intensity "
            "FROM parquet_db ORDER BY channel"
        ).fetchall()
    finally:
        feature.parquet_db.close()

    assert rows == [
        ("TMT126", "S1", "PEPTIDEK", 100.0),
        ("TMT131", "S10", "PEPTIDEK", 1000.0),
    ]


def test_partial_tmt11_uses_declared_channel_positions(tmp_path):
    """A TMT11 plex remains identifiable when its final channel is unused."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    labels = [
        "TMT126",
        "TMT127N",
        "TMT127C",
        "TMT128N",
        "TMT128C",
        "TMT129N",
        "TMT129C",
        "TMT130N",
        "TMT130C",
        "TMT131N",
    ]
    _write_sdrf(
        sdrf,
        [
            {
                "source name": f"S{position}",
                "comment[data file]": "plex.raw",
                "comment[label]": label,
            }
            for position, label in enumerate(labels, start=1)
        ],
    )
    pd.DataFrame(
        [
            {
                "ProteinName": "P1",
                "PeptideSequence": "PEPTIDEK",
                "Charge": 2,
                "Channel": 10,
                "Run": "plex",
                "Intensity": 100.0,
            }
        ]
    ).to_csv(msstats, index=False)

    feature = Feature.from_msstats(str(msstats), str(sdrf))
    try:
        row = feature.parquet_db.execute(
            "SELECT intensities[1].label FROM parquet_db_raw"
        ).fetchone()
    finally:
        feature.parquet_db.close()

    assert row == ("TMT131N",)


def test_reference_names_the_file_when_run_is_a_design_index(tmp_path):
    """quantms Run is a design index, so numeric file names must not claim it."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    _write_sdrf(
        sdrf,
        [
            {
                "source name": sample,
                "comment[data file]": data_file,
                "comment[label]": "label free sample",
            }
            for sample, data_file in [("S1", "2.raw"), ("S2", "2_20221202203139.raw")]
        ],
    )
    pd.DataFrame(
        [
            {
                "ProteinName": "P1",
                "PeptideSequence": "PEPTIDEK",
                "Charge": 2,
                "Run": run,
                "Intensity": intensity,
                "Reference": reference,
            }
            for run, reference, intensity in [
                (1, "2.mzML", 100.0),
                (2, "2_20221202203139.mzML", 200.0),
            ]
        ]
    ).to_csv(msstats, index=False)

    feature = Feature.from_msstats(str(msstats), str(sdrf))
    try:
        rows = feature.parquet_db.execute(
            "SELECT intensities[1].intensity, run_file_name "
            "FROM parquet_db_raw ORDER BY 1"
        ).fetchall()
    finally:
        feature.parquet_db.close()

    assert rows == [(100.0, "2.raw"), (200.0, "2_20221202203139.raw")]


def test_msstats_rejects_nonfinite_intensity(tmp_path):
    """Non-finite source values must not become quantified zeroes."""
    sdrf = tmp_path / "input.sdrf.tsv"
    msstats = tmp_path / "input_msstats_in.csv"
    _write_sdrf(
        sdrf,
        [
            {
                "source name": "S1",
                "comment[data file]": "run.raw",
                "comment[label]": "label free sample",
            }
        ],
    )
    msstats.write_text(
        "ProteinName,PeptideSequence,Charge,Reference,Intensity\n"
        "P1,PEPTIDEK,2,run,NaN\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid required values"):
        Feature.from_msstats(str(msstats), str(sdrf))


def test_ratio_rejects_feature_level_msstats_input():
    """Ratio quantification rejects MSstats without PSM evidence."""
    config = PipelineConfig(
        input=InputConfig(msstats="input.csv", sdrf="input.sdrf.tsv"),
        normalization=NormalizationConfig(run_method="none", sample_method="none"),
        quantification=QuantificationConfig(method="ratio"),
    )

    with pytest.raises(ValueError, match="PSM-level QPX input"):
        run_pipeline(config)
