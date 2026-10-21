from pathlib import Path

import mokume.pipeline as pipeline
from click.testing import CliRunner

from mokume.mokume_cli import cli


def _make_input_files(tmp_path: Path) -> tuple[str, str]:
    parquet = tmp_path / "input.parquet"
    sdrf = tmp_path / "input.sdrf.tsv"
    parquet.write_text("placeholder", encoding="utf-8")
    sdrf.write_text("source name\nSample1\n", encoding="utf-8")
    return str(parquet), str(sdrf)


def test_features_cli_stabilization_default_forwarding_and_method_scope(
    monkeypatch, tmp_path
):
    parquet, _ = _make_input_files(tmp_path)
    captured = {}
    monkeypatch.setattr(
        pipeline, "features_to_proteins", lambda **kwargs: captured.update(kwargs)
    )
    args = ["features2proteins", "-p", parquet, "-o", str(tmp_path / "proteins.csv")]
    for enabled in [False, True]:
        result = CliRunner().invoke(cli, [*args, *(["--stabilize"] if enabled else [])])
        assert result.exit_code == 0, result.output
        assert captured["stabilize"] is enabled
    for method in ["directlfq", "sum", "pibaq", "top3"]:
        fasta = tmp_path / "proteome.fasta"
        fasta.write_text(">P\nPEPTIDEK\n", encoding="utf-8")
        method_options = ["--fasta", str(fasta)] if method == "pibaq" else []
        result = CliRunner().invoke(
            cli, [*args, "--quant-method", method, "--stabilize", *method_options]
        )
        assert result.exit_code == 2, result.output
        assert "--stabilize requires --quant-method maxlfq" in result.output


def test_features2proteins_uses_standard_pibaq_max_aa_default(monkeypatch, tmp_path):
    """The pure-Python CLI shares the canonical 30-aa piBAQ default."""
    parquet, sdrf = _make_input_files(tmp_path)
    fasta = tmp_path / "proteome.fasta"
    fasta.write_text(">P1\nPEPTIDEK\n", encoding="utf-8")
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "features_to_proteins", fake_run_pipeline)
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--quant-method",
            "pibaq",
            "--fasta",
            str(fasta),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["pibaq_max_aa"] == 30


def test_features2proteins_passes_directlfq_and_batch_options(monkeypatch, tmp_path):
    parquet, sdrf = _make_input_files(tmp_path)
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "features_to_proteins", fake_run_pipeline)

    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--quant-method",
            "directlfq",
            "--directlfq-min-nonan",
            "3",
            "--export-ions",
            "ions.csv",
            "--batch-correction",
            "--batch-method",
            "column",
            "--batch-column",
            "characteristics[batch]",
            "--batch-covariates",
            "characteristics[sex],characteristics[organism part]",
            "--batch-nonparametric",
            "--batch-mean-only",
            "--batch-ref",
            "2",
            "--de",
            "--de-contrasts",
            "A vs B",
            "--de-output",
            "de.csv",
            "--de-fdr-method",
            "ihw",
        ],
    )

    assert result.exit_code == 0
    assert captured["directlfq_min_nonan"] == 3
    assert captured["export_ions"] == "ions.csv"
    assert captured["batch_correction"] is True
    assert captured["batch_method"] == "column"
    assert captured["batch_column"] == "characteristics[batch]"
    assert captured["batch_covariates"] == [
        "characteristics[sex]",
        "characteristics[organism part]",
    ]
    assert captured["batch_parametric"] is False
    assert captured["batch_mean_only"] is True
    assert captured["batch_ref"] == 2
    assert captured["de_fdr_method"] == "ihw"


def test_features2proteins_passes_sample_correlation_threshold(monkeypatch, tmp_path):
    parquet, sdrf = _make_input_files(tmp_path)
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "features_to_proteins", fake_run_pipeline)
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--min-sample-correlation",
            "0.85",
        ],
    )

    assert result.exit_code == 0
    assert captured["sample_correlation_threshold"] == 0.85


def test_features2proteins_requires_sdrf_for_sample_correlation(tmp_path):
    parquet, _ = _make_input_files(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "--min-sample-correlation",
            "0.85",
        ],
    )

    assert result.exit_code != 0
    assert "--min-sample-correlation requires --sdrf" in result.output


def test_features2proteins_rejects_ion_export_for_non_directlfq(tmp_path):
    parquet, _ = _make_input_files(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "--quant-method",
            "sum",
            "--export-ions",
            "ions.csv",
        ],
    )

    assert result.exit_code != 0
    assert "--export-ions require --quant-method directlfq" in result.output


def test_features2proteins_requires_batch_column_for_column_method(tmp_path):
    parquet, sdrf = _make_input_files(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--batch-correction",
            "--batch-method",
            "column",
        ],
    )

    assert result.exit_code != 0
    assert "requires --batch-column option" in result.output


def test_features2proteins_preserves_empty_ensemble_members(monkeypatch, tmp_path):
    parquet, sdrf = _make_input_files(tmp_path)
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "features_to_proteins", fake_run_pipeline)

    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--de",
            "--de-method",
            "ensemble",
            "--de-contrasts",
            "A vs B",
            "--de-output",
            "de.csv",
            "--de-ensemble-methods",
            "",
        ],
    )

    assert result.exit_code == 0
    assert captured["de_ensemble_methods"] == [""]


def test_features2proteins_rejects_ensemble_k_for_single_method(tmp_path):
    parquet, sdrf = _make_input_files(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            parquet,
            "-o",
            "out.csv",
            "-s",
            sdrf,
            "--de",
            "--de-method",
            "limma",
            "--de-contrasts",
            "A vs B",
            "--de-output",
            "de.csv",
            "--de-ensemble-min-k",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert "require --de-method ensemble" in result.output


def test_features2proteins_accepts_msstats_with_sdrf(monkeypatch, tmp_path):
    """The CLI accepts MSstats only as the selected feature input."""
    _, sdrf = _make_input_files(tmp_path)
    msstats = tmp_path / "input_msstats_in.csv"
    msstats.write_text("ProteinName,PeptideSequence,Intensity\n", encoding="utf-8")
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "features_to_proteins", fake_run_pipeline)
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "--msstats",
            str(msstats),
            "--sdrf",
            sdrf,
            "--output",
            "out.csv",
        ],
    )

    assert result.exit_code == 0
    assert captured["parquet"] is None
    assert captured["msstats"] == str(msstats)
