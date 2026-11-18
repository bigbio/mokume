"""Normalization names fail before Python pipeline input or output work."""

from pathlib import Path

import pytest

from mokume.normalization.peptide import peptide_normalization
from mokume.pipeline.config import (
    InputConfig,
    NormalizationConfig,
    PipelineConfig,
    QuantificationConfig,
)
from mokume.pipeline.features_to_proteins import (
    QuantificationPipeline,
    features_to_proteins,
)
from mokume.pipeline.runner import run_pipeline


def _config(parquet: Path, run_method: str, sample_method: str) -> PipelineConfig:
    """Build the smallest config accepted by both protein pipeline entry points."""
    return PipelineConfig(
        input=InputConfig(parquet=str(parquet)),
        normalization=NormalizationConfig(
            run_method=run_method,
            sample_method=sample_method,
        ),
    )


def _invoke(
    entrypoint: str,
    parquet: Path,
    output: Path,
    run_method: str,
    sample_method: str,
) -> None:
    """Invoke one public Python normalization boundary."""
    if entrypoint == "pipeline":
        QuantificationPipeline(_config(parquet, run_method, sample_method))
    elif entrypoint == "runner":
        run_pipeline(_config(parquet, run_method, sample_method))
    elif entrypoint == "proteins":
        features_to_proteins(
            parquet=str(parquet),
            output=str(output),
            run_normalization=run_method,
            sample_normalization=sample_method,
        )
    else:
        peptide_normalization(
            parquet=str(parquet),
            sdrf=None,
            min_aa=7,
            min_unique=1,
            remove_ids=None,
            remove_decoy_contaminants=False,
            remove_low_frequency_peptides=False,
            output=str(output),
            skip_normalization=False,
            nmethod=run_method,
            pnmethod=sample_method,
            log2=False,
            save_parquet=False,
        )


@pytest.mark.parametrize("entrypoint", ["pipeline", "runner", "proteins", "peptides"])
@pytest.mark.parametrize(
    ("run_method", "sample_method", "kind"),
    [("bogus", "none", "run"), ("none", "bogus", "sample")],
)
def test_invalid_normalization_fails_before_input_or_output(
    entrypoint,
    run_method,
    sample_method,
    kind,
    tmp_path,
) -> None:
    """Every callable entry point rejects one invalid name without touching data."""
    parquet = tmp_path / "missing.parquet"
    output = tmp_path / "result.csv"

    with pytest.raises(
        ValueError,
        match=rf"Unknown {kind} normalization method: 'bogus'",
    ):
        _invoke(entrypoint, parquet, output, run_method, sample_method)

    assert not output.exists()


@pytest.mark.parametrize(
    ("quant_method", "expected"),
    [
        ("maxlfq", ("median", "hierarchical")),
        ("top3", ("median", "hierarchical")),
        ("sum", ("median", "hierarchical")),
        ("median", ("median", "hierarchical")),
        ("pibaq", ("median", "globalmedian")),
        ("directlfq", ("none", "none")),
        ("peptide_count", ("none", "none")),
    ],
)
def test_unset_normalization_takes_the_quant_method_default(
    quant_method, expected
) -> None:
    """PipelineConfig fills unset methods as the features2proteins command does."""
    config = PipelineConfig(
        input=InputConfig(parquet="features.parquet"),
        quantification=QuantificationConfig(method=quant_method),
    )
    normalization = config.normalization
    assert (normalization.run_method, normalization.sample_method) == expected

    explicit = PipelineConfig(
        input=InputConfig(parquet="features.parquet"),
        normalization=NormalizationConfig(sample_method="globalMedian"),
        quantification=QuantificationConfig(method=quant_method),
    )
    assert explicit.normalization.sample_method == "globalMedian"
