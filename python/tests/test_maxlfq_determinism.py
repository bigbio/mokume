"""MaxLFQ must not depend on the order its input rows happen to arrive in.

Regression tests for bigbio/mokume#58: the reference peptide was chosen with
``np.argmax``, which breaks ties by row position, so an unordered upstream read
made protein quantities change between identical runs (observed: >2 log2 units on
one protein, and the differential-expression call count moving between runs).
"""

import numpy as np
import pandas as pd
import pytest

from mokume.quantification.maxlfq import (
    MaxLFQQuantification,
    _maxlfq_solve_protein,
)


def _matrix():
    # 4 peptides x 4 samples; several share the same number of valid values so the
    # tie-break is exercised, and one has a missing value.
    return np.array(
        [
            [100.0, 200.0, 400.0, 800.0],
            [50.0, 100.0, 200.0, 400.0],
            [10.0, 20.0, 40.0, np.nan],
            [400.0, 800.0, 1600.0, 3200.0],
        ]
    )


@pytest.mark.parametrize("seed", range(8))
def test_solve_is_invariant_to_row_permutation(seed):
    """Permuting peptide rows must not change the protein quantities."""
    m = _matrix()
    base = _maxlfq_solve_protein(m)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m.shape[0])
    permuted = _maxlfq_solve_protein(m[perm, :])
    np.testing.assert_allclose(permuted, base, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(5))
def test_quantify_is_invariant_to_dataframe_row_order(seed):
    """Shuffling rows must not change sample- or run-level output."""
    log2_matrix = np.array(
        [[6, 2, 1, 3], [5, 6, 1, 0], [2, 0, 6, 4], [3, 4, 5, 0]],
        dtype=float,
    )
    rows = [
        {
            "ProteinName": "P1",
            "PeptideCanonical": peptide,
            "SampleID": sample,
            "Run": "R1",
            "NormIntensity": 2.0 ** log2_matrix[pep_idx, sample_idx],
        }
        for pep_idx, peptide in enumerate(("A", "B", "C", "D"))
        for sample_idx, sample in enumerate(("S1", "S2", "S3", "S4"))
    ]
    df = pd.DataFrame(rows)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    quantifier = MaxLFQQuantification(min_peptides=2, threads=1, force_builtin=True)

    for run_column in (None, "Run"):
        sort_by = ["SampleID", *(["Run"] if run_column else [])]
        expected = quantifier.quantify(df, run_column=run_column).sort_values(sort_by)
        actual = quantifier.quantify(shuffled, run_column=run_column).sort_values(
            sort_by
        )
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True)
        )
