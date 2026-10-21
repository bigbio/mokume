"""Tests for the machine-readable native command contract."""

import pytest

from mokume import command_schema, validate_args


def test_native_command_schema_exposes_leaf_contract():
    """The public schema exposes executable leaves and their flag metadata."""
    schema = command_schema()
    commands = {tuple(command["path"]): command for command in schema}

    command = commands[("quantify", "features2proteins")]
    flags = {flag["id"]: flag for flag in command["flags"]}
    assert flags["parquet"]["short"] == "p"
    assert flags["parquet"]["value_arity"] == {"min": 1, "max": 1}
    assert "msstats" in flags["parquet"]["conflicts"]
    assert flags["de_contrast"]["repeat"] is True
    assert flags["de_contrast"]["value_names"] == ["GROUP_A", "GROUP_B"]
    assert flags["log_level"]["global"] is True


def test_native_validate_args_never_dispatches(tmp_path):
    """Argument validation rejects invalid contracts without running commands."""
    output = tmp_path / "must-not-exist.csv"
    assert (
        validate_args(
            [
                "quantify",
                "features2proteins",
                "--parquet",
                "missing-but-parseable.parquet",
                "--output",
                str(output),
            ]
        )
        is None
    )
    assert not output.exists()

    with pytest.raises(RuntimeError, match="not-a-command"):
        validate_args(["not-a-command"])

    with pytest.raises(RuntimeError, match="--threads only applies"):
        validate_args(
            [
                "quantify",
                "peptides2protein",
                "--peptides",
                "missing-but-parseable.csv",
                "--quant-method",
                "sum",
                "--threads",
                "2",
                "--output",
                str(tmp_path / "unused.tsv"),
            ]
        )
