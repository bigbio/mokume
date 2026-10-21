"""Tests for :class:`mokume.pipeline.config.QuantificationConfig` validation.

``ratio_fraction_merge`` is read by the Python and Rust implementations, which
disagree on case, so the config layer folds it. See the ``__post_init__``
comment for why that has to happen here rather than in either reader.
"""

from __future__ import annotations

import pytest

from mokume.pipeline.config import QuantificationConfig
from mokume.quantification import get_quantification_method


def test_stabilization_is_opt_in_and_maxlfq_only():
    assert not QuantificationConfig().stabilize
    assert QuantificationConfig(stabilize=True).stabilize
    assert get_quantification_method("maxlfq", stabilize=True).stabilize
    for method in ["sum", "directlfq", "top3", "pibaq"]:
        with pytest.raises(ValueError, match="stabilize only applies to MaxLFQ"):
            QuantificationConfig(method=method, stabilize=True)
        with pytest.raises(ValueError, match="stabilize only applies to MaxLFQ"):
            get_quantification_method(method, stabilize=True)


class TestRatioFractionMerge:
    def test_default_is_mean(self) -> None:
        assert QuantificationConfig().ratio_fraction_merge == "mean"

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("mean", "mean"),
            ("max", "max"),
            ("Max", "max"),
            ("MEAN", "mean"),
        ],
    )
    def test_case_is_folded(self, given: str, expected: str) -> None:
        config = QuantificationConfig(ratio_fraction_merge=given)
        assert config.ratio_fraction_merge == expected

    @pytest.mark.parametrize("bad", ["median", "sum", "", "maximum"])
    def test_rejects_unknown_methods(self, bad: str) -> None:
        with pytest.raises(ValueError, match="must be 'mean' or 'max'"):
            QuantificationConfig(ratio_fraction_merge=bad)

    def test_error_names_the_value_the_user_wrote(self) -> None:
        with pytest.raises(ValueError, match="'Median'"):
            QuantificationConfig(ratio_fraction_merge="Median")


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_sample_correlation_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        QuantificationConfig(sample_correlation_threshold=threshold)
