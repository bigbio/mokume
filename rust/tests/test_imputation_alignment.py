"""Public Rust imputation options preserve reproducible stochastic models."""

import numpy as np
import pytest

import mokume


@pytest.mark.parametrize("method", ["minprob", "qrilc"])
def test_stochastic_imputation_seed_and_spread_are_forwarded(method):
    """Both options reach the kernel and measured cells remain unchanged."""
    matrix = [
        [10.0, 20.0, 30.0],
        [20.0, None, 45.0],
        [35.0, 50.0, None],
        [None, 70.0, 80.0],
        [60.0, 85.0, 95.0],
        [70.0, None, 120.0],
    ]
    first = mokume.impute_matrix(matrix, method, seed=42, tune_sigma=1.0, threads=1)
    assert first == mokume.impute_matrix(matrix, method, seed=42, threads=1)
    assert first != mokume.impute_matrix(matrix, method, seed=43, threads=1)
    assert first != mokume.impute_matrix(
        matrix, method, seed=42, tune_sigma=0.7, threads=1
    )
    observed = np.asarray(matrix, dtype=float)
    observed_mask = np.isfinite(observed)
    np.testing.assert_array_equal(
        np.asarray(first)[observed_mask], observed[observed_mask]
    )


def test_minprob_rejects_parameters_of_the_removed_shifted_constant_model():
    """Old kwargs must not silently become ineffective on the new algorithm."""
    with pytest.raises(TypeError, match="tune_sigma"):
        mokume.impute_matrix([[1.0, None], [2.0, 3.0]], "minprob", shift=1.6)
    with pytest.raises(TypeError, match="only apply"):
        mokume.impute_matrix([[1.0, None], [2.0, 3.0]], "mean", seed=42)
