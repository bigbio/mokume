"""CLI tests for features2peptides."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from mokume.mokume_cli import cli
from mokume.commands.features2proteins import features2proteins
from mokume.commands.features2peptides import features2parquet

from .test_qpx_format_compat import _make_scored_qpx_parquet as make_scored_qpx_parquet


def _patch_peptide_normalization(monkeypatch):
    """Patch peptide_normalization and return captured keyword arguments."""
    captured = {}

    def fake_peptide_normalization(**kwargs):
        """Capture CLI arguments passed to peptide_normalization."""
        captured.update(kwargs)

    monkeypatch.setattr(
        "mokume.commands.features2peptides.peptide_normalization",
        fake_peptide_normalization,
    )
    return captured


def _invoke_features2peptides(tmp_path: Path, parquet: Path):
    """Run features2peptides with --keep-shared-peptides enabled."""
    return CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--skip_normalization",
            "--keep-shared-peptides",
        ],
    )


def test_features2peptides_passes_keep_shared_peptides(monkeypatch, tmp_path: Path):
    """The CLI forwards --keep-shared-peptides to peptide normalization."""
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    captured = _patch_peptide_normalization(monkeypatch)

    result = _invoke_features2peptides(tmp_path, parquet)

    assert result.exit_code == 0
    assert captured["keep_shared_peptides"] is True


def test_features2peptides_rejects_unscoped_irs_option(tmp_path: Path):
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--irs_scope",
            "two_stage",
        ],
    )

    assert result.exit_code != 0
    assert "require --irs_channel or --irs_autodetect_regex" in result.output


def test_filter_config_controls_duplicate_length_and_unique_thresholds(
    monkeypatch,
    tmp_path: Path,
):
    """Filter config values replace defaults shared with legacy CLI options."""
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    config = tmp_path / "filters.json"
    config.write_text(
        json.dumps(
            {
                "peptide": {"min_peptide_length": 11},
                "protein": {"min_unique_peptides": 4},
            }
        ),
        encoding="utf-8",
    )
    captured = _patch_peptide_normalization(monkeypatch)

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--filter-config",
            str(config),
        ],
    )

    assert result.exit_code == 0
    assert captured["min_aa"] == 11
    assert captured["min_unique"] == 4


def test_explicit_legacy_thresholds_override_filter_config(monkeypatch, tmp_path: Path):
    """Explicit legacy CLI thresholds remain authoritative over config values."""
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    config = tmp_path / "filters.json"
    config.write_text(
        json.dumps(
            {
                "peptide": {"min_peptide_length": 11},
                "protein": {"min_unique_peptides": 4},
            }
        ),
        encoding="utf-8",
    )
    captured = _patch_peptide_normalization(monkeypatch)

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--filter-config",
            str(config),
            "--min_aa",
            "9",
            "--min_unique",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert captured["min_aa"] == 9
    assert captured["min_unique"] == 3


def test_duplicate_min_unique_cli_options_are_rejected(tmp_path: Path):
    """Two CLI spellings must not silently override each other."""
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--min_unique",
            "3",
            "--filter-min-unique-peptides",
            "4",
        ],
    )

    assert result.exit_code != 0
    assert "Choose either --min_unique or --filter-min-unique-peptides" in result.output


def test_filter_max_missing_rate_reaches_filter_config(monkeypatch, tmp_path: Path):
    """The CLI override must activate the run-level missing-rate filter."""
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    captured = _patch_peptide_normalization(monkeypatch)

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--filter-max-missing-rate",
            "0.4",
        ],
    )

    assert result.exit_code == 0
    assert captured["filter_config"].run_qc.max_missing_rate == 0.4


def test_named_score_cli_reaches_filter_config(monkeypatch, tmp_path: Path):
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    captured = _patch_peptide_normalization(monkeypatch)

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--filter-score",
            "diann_ms1_profile_corr=0.8",
        ],
    )

    assert result.exit_code == 0
    score = captured["filter_config"].peptide.score
    assert score.name == "diann_ms1_profile_corr"
    assert score.threshold == 0.8


@pytest.mark.parametrize("method", ["globalmedian", "conditionmedian"])
def test_sample_without_passing_rows_is_skipped(tmp_path: Path, method):
    """A score filter that empties one sample leaves the other samples' output."""
    parquet = tmp_path / "input.parquet"
    make_scored_qpx_parquet(str(parquet))
    table = pq.read_table(parquet)
    index = table.schema.get_field_index("intensities")
    intensities = [[{"label": "LFQ", "intensity": value}] for value in (10.0, 20.0)]
    table = table.set_column(
        index, "intensities", pa.array(intensities, type=table.schema.field(index).type)
    )
    pq.write_table(table, parquet)
    output = tmp_path / "peptides.csv"

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(output),
            "--filter-score",
            "qvalue=0.05",
            "--filter-min-unique-peptides",
            "1",
            "--sample-normalization",
            method,
        ],
    )

    assert result.exit_code == 0, result.output
    assert pd.read_csv(output)["SampleID"].tolist() == ["run_a"]


def test_named_score_cli_rejects_ambiguous_value(tmp_path: Path):
    parquet = tmp_path / "input.parquet"
    parquet.write_text("placeholder", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "features2peptides",
            "-p",
            str(parquet),
            "-o",
            str(tmp_path / "peptides.csv"),
            "--filter-score",
            "0.8",
        ],
    )

    assert result.exit_code != 0
    assert "expected NAME=THRESHOLD" in result.output


def test_run_and_sdrf_conflict_before_reading(tmp_path):
    """Both commands reach pipeline validation before reading input."""

    feature_path = tmp_path / "feature.parquet"
    feature_path.touch()
    sdrf = tmp_path / "samples.tsv"
    sdrf.touch()
    run_path = tmp_path / "run.parquet"
    run_path.touch()
    for command in [features2proteins, features2parquet]:
        result = CliRunner().invoke(
            command,
            [
                "-p",
                str(feature_path),
                "-o",
                str(tmp_path / "out.csv"),
                "--sdrf",
                str(sdrf),
                "--run",
                str(run_path),
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "--run conflicts with --sdrf"
