"""Precision-weighted DEqMS in the pure-Python package (mirrors the Rust tests)."""

import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from mokume.analysis.deqms import _precision_weights, run_deqms
from mokume.analysis import differential_expression
from mokume.analysis.differential_expression import DifferentialExpression
from mokume.mokume_cli import cli
from mokume.pipeline.config import (
    DEConfig,
    FilterConfig,
    InputConfig,
    NormalizationConfig,
    PipelineConfig,
    QuantificationConfig,
)
from mokume.pipeline.features_to_proteins import QuantificationPipeline
from mokume.pipeline.runner import run_pipeline
from mokume.quantification.maxlfq import MaxLFQQuantification
from mokume.quantification.precision import cell_precision, profile_residuals

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "example")
PARQUET = os.path.join(EXAMPLE_DIR, "feature_wide.parquet")
SDRF = os.path.join(EXAMPLE_DIR, "PXD020192.sdrf.tsv")


def test_residuals_remove_each_ion_offset():
    # Ion 1 tracks the profile with a constant offset; ion 2 deviates by
    # +0.1 / -0.1 around its median offset in two of three samples.
    rss, dof, counts = profile_residuals(
        [[8.0, 9.0, 10.0], [9.1, 10.0, 10.9]], [10.0, 11.0, 12.0]
    )
    assert rss == pytest.approx(0.02, abs=1e-12)
    # 6 observations - 2 ions - (3 samples - 1).
    assert dof == 2.0
    assert list(counts) == [2, 2, 2]


def test_missing_profile_samples_get_no_precision():
    _, dof, counts = profile_residuals([[8.0, 9.0]], [10.0, np.nan])
    assert list(counts) == [0, 0]
    assert dof == 0.0


def test_single_observation_ions_do_not_inflate_precision():
    """An ion without a within-ion ratio must not increase any cell's support."""
    baseline = profile_residuals([[8.0, 9.0, 10.0]], [10.0, 11.0, 12.0])
    with_singleton = profile_residuals(
        [[8.0, 9.0, 10.0], [7.0, np.nan, np.nan]], [10.0, 11.0, 12.0]
    )
    assert baseline[:2] == with_singleton[:2]
    np.testing.assert_array_equal(baseline[2], with_singleton[2])


def test_misaligned_precision_emits_a_warning(caplog):
    """Missing protein alignment cannot silently turn a weighted request off."""
    log2, group_a, group_b, counts = _fixture()
    precision = pd.DataFrame(2.0, index=["missing"], columns=log2.columns)
    result = run_deqms(
        log2, group_a, group_b, ("A", "B"), peptide_counts=counts, precision=precision
    )
    assert np.isfinite(result["pvalue"]).all()
    assert "no finite positive precision" in caplog.text


def test_precision_counts_ions_over_the_shrunk_variance():
    fits = [
        ("a", (0.3, 3.0, [2, 0])),
        ("b", (1.5, 5.0, [4, 0])),
        ("c", (0.0, 1.0, [0, 1])),
    ]
    precision = cell_precision(fits, ["s0", "s1"])
    # Prior = median(0.1, 0.3) = 0.2; "c" has too few dof to enter it.
    assert precision[("a", "s0")] == pytest.approx(2.0 / ((0.3 + 4.0 * 0.2) / 7.0))
    assert precision[("c", "s1")] == pytest.approx(1.0 / ((0.0 + 4.0 * 0.2) / 5.0))
    assert len(precision) == 3


def test_precision_weights_fill_unmeasured_cells_and_floor():
    weights = _precision_weights(
        np.array([[1.0, 2.0, 3.0, np.nan]]), np.array([[1.0, 4.0, 0.0, 4.0]])
    )
    tau2 = (1.0 + 0.25) / 2.0
    unmeasured = 1.0 + (4.0 - 1.0) * 0.05
    expected = 1.0 / (1.0 / np.array([1.0, 4.0, unmeasured, unmeasured]) + tau2)
    np.testing.assert_allclose(weights[0], expected, rtol=0, atol=1e-12)


def _fixture():
    rows = []
    for index in range(15):
        shift = 1.5 if index % 3 == 0 else 0.0
        rows.append(
            [
                5.0
                + index
                + 0.1 * (((index * 7 + column * 5) % 11) - 5.0)
                + (shift if column < 3 else 0.0)
                for column in range(6)
            ]
        )
    proteins = [f"P{index:02d}" for index in range(15)]
    samples = [f"S{column}" for column in range(6)]
    log2 = pd.DataFrame(rows, index=proteins, columns=samples)
    log2.iloc[3, 1] = np.nan
    counts = pd.Series([1 + index % 7 for index in range(15)], index=proteins)
    return log2, samples[:3], samples[3:], counts


def test_unit_precision_weights_reproduce_deqms():
    # Precision 2 everywhere floors to weight 1/(1/2 + 1/2) = 1.
    log2, group_a, group_b, counts = _fixture()
    plain = run_deqms(log2, group_a, group_b, ("A", "B"), peptide_counts=counts)
    weighted = run_deqms(
        log2,
        group_a,
        group_b,
        ("A", "B"),
        peptide_counts=counts,
        precision=pd.DataFrame(2.0, index=log2.index, columns=log2.columns),
    )
    pd.testing.assert_frame_equal(plain, weighted)


def test_uneven_precision_reweights_the_test_and_group_means():
    log2, group_a, group_b, counts = _fixture()
    precision = pd.DataFrame(
        [[1.0, 8.0, 2.0, 4.0, 0.0, 3.0]] * len(log2),
        index=log2.index,
        columns=log2.columns,
    )
    plain = run_deqms(log2, group_a, group_b, ("A", "B"), peptide_counts=counts)
    weighted = run_deqms(
        log2, group_a, group_b, ("A", "B"), peptide_counts=counts, precision=precision
    ).set_index("ProteinName")
    assert not np.allclose(
        weighted.loc[plain["ProteinName"], "pvalue"], plain["pvalue"]
    )
    weights = _precision_weights(log2.to_numpy(), precision.to_numpy())
    row = log2.loc["P00", group_a].to_numpy()
    assert weighted.loc["P00", "mean_A"] == pytest.approx(
        np.average(row, weights=weights[0, :3])
    )


def test_precision_weights_require_deqms():
    log2, *_ = _fixture()
    with pytest.raises(ValueError, match="deqms"):
        DifferentialExpression(method="limma", precision=log2.reset_index())


def test_maxlfq_cell_precision_keeps_intensities():
    rng = np.random.default_rng(0)
    peptides = pd.DataFrame(
        [
            {
                "ProteinName": protein,
                "PeptideCanonical": f"{protein}_{species}",
                "SampleID": f"S{sample}",
                "NormIntensity": 2 ** (20 + species + sample + rng.normal(0, 0.1)),
            }
            for protein in ("P1", "P2")
            for species in range(4)
            for sample in range(4)
        ]
    )
    plain = MaxLFQQuantification(threads=1).quantify(peptides)
    with_precision = MaxLFQQuantification(threads=1, cell_precision=True).quantify(
        peptides
    )
    pd.testing.assert_frame_equal(plain, with_precision.drop(columns="Precision"))
    assert (with_precision["Precision"] > 0).all()


def _config(quant_method, de_output, precision_weights):
    directlfq = quant_method == "directlfq"
    return PipelineConfig(
        input=InputConfig(parquet=PARQUET, sdrf=SDRF),
        filtering=FilterConfig(
            min_aa=7, min_unique_peptides=1, remove_contaminants=True
        ),
        normalization=NormalizationConfig(
            run_method="none" if directlfq else "median",
            sample_method="none" if directlfq else "globalMedian",
        ),
        quantification=QuantificationConfig(method=quant_method),
        de=DEConfig(
            enabled=True,
            contrasts=["Brain vs Placenta"],
            method="deqms",
            output=str(de_output),
            precision_weights=precision_weights,
        ),
    )


@pytest.mark.parametrize("quant_method", ["maxlfq", "directlfq"])
def test_pipelines_pass_cell_precision_to_deqms(tmp_path, monkeypatch, quant_method):
    if quant_method == "directlfq":
        pytest.importorskip("directlfq")
        pytest.importorskip("polars")
    received = {}
    parquet, sdrf = _variable_precision_input(tmp_path)

    def spy(*args, **options):
        received[name] = options.get("precision")
        if received[name] is not None:
            pd.testing.assert_index_equal(received[name].index, args[0].index)
            pd.testing.assert_index_equal(received[name].columns, args[0].columns)
        return run_deqms(*args, **options)

    monkeypatch.setattr(differential_expression, "run_deqms", spy)
    outputs = {}
    for name, weighted in (("plain", False), ("weighted", True), ("runner", True)):
        config = _config(quant_method, tmp_path / f"{name}.csv", weighted)
        config.input.parquet, config.input.sdrf = str(parquet), str(sdrf)
        config.de.contrasts = ["A vs B"]
        if name == "runner":
            run_pipeline(config)
        else:
            QuantificationPipeline(config).run()
        outputs[name] = pd.read_csv(config.de.output)
    assert received["plain"] is None
    assert received["weighted"].notna().to_numpy().any()
    pd.testing.assert_frame_equal(received["weighted"], received["runner"])
    pd.testing.assert_frame_equal(outputs["weighted"], outputs["runner"])
    assert received["weighted"].stack().nunique() > 1
    plain = outputs["plain"].set_index("ProteinName")
    weighted = outputs["weighted"].set_index("ProteinName").reindex(plain.index)
    assert np.max(np.abs(plain["pvalue"] - weighted["pvalue"])) > 1e-5


def _variable_precision_input(tmp_path):
    """Vary ion counts and sample coverage so dropping weights changes the fit."""
    rows = []
    for protein, letter in enumerate("ACDEFGHIKLMNPQR"):
        for ion in range(2 + protein % 4):
            peptide = letter + "A" * ion + "PEPTIDEK"
            for sample in range(6):
                if sample == 0 and ion > 0:
                    continue
                value = 20 + protein / 3 + ion + (sample < 3) * (protein % 3 == 0)
                value += 0.1 * ((protein * 7 + ion * 3 + sample * 5) % 11 - 5)
                rows.append(
                    {
                        "sequence": peptide,
                        "peptidoform": peptide,
                        "charge": 2,
                        "anchor_protein": f"P{protein:02}",
                        "pg_accessions": [f"P{protein:02}"],
                        "run_file_name": f"run{sample}",
                        "unique": True,
                        "is_decoy": False,
                        "intensities": [{"label": "LFQ", "intensity": 2.0**value}],
                    }
                )
    parquet, sdrf = tmp_path / "features.parquet", tmp_path / "samples.tsv"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    pd.DataFrame(
        {
            "source name": [f"S{i}" for i in range(6)],
            "comment[data file]": [f"run{i}" for i in range(6)],
            "comment[label]": "label free sample",
            "comment[technical replicate]": "1",
            "comment[fraction identifier]": "1",
            "characteristics[biological replicate]": range(1, 7),
            "factor value[condition]": ["A"] * 3 + ["B"] * 3,
        }
    ).to_csv(sdrf, sep="\t", index=False)
    return parquet, sdrf


def test_pipeline_rejects_precision_weights_without_maxlfq_or_directlfq(tmp_path):
    with pytest.raises(ValueError, match="maxlfq or directlfq"):
        QuantificationPipeline(_config("sum", tmp_path / "de.csv", True)).run()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--quant-method", "top3", "--de-method", "deqms"], "maxlfq or directlfq"),
        (["--quant-method", "maxlfq"], "--de-method deqms"),
    ],
)
def test_cli_validates_precision_weights(tmp_path, extra, message):
    result = CliRunner().invoke(
        cli,
        [
            "features2proteins",
            "-p",
            PARQUET,
            "-s",
            SDRF,
            "-o",
            str(tmp_path / "proteins.csv"),
            "--de",
            "--de-contrasts",
            "Brain vs Placenta",
            "--de-output",
            str(tmp_path / "de.csv"),
            "--de-precision-weights",
            *extra,
        ],
    )
    assert result.exit_code != 0
    assert message in result.output
