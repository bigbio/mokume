"""The pure-Python imputers reproduce the R oracles that pin the Rust kernel.

The fixture is the Rust crate's (imputeLCMD 2.1, rrcovNA 0.5-3, pcaMethods
2.2.0 and SeqKnn 1.0.1 under R 4.5.3, log2 values). QRILC and MinProb used the
kernel's seeded draws (seed 42), which the Python imputers share.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mokume.imputation import impute_censored

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "rust/crates/mokume-imputation/tests/fixtures/upstream.tsv"
)
CASES = {
    name: rows
    for name, rows in pd.read_csv(FIXTURE, sep="\t", na_values=["NA"]).groupby("case")
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_imputer_matches_r_oracle(name):
    rows = CASES[name]
    matrix = np.full((rows["row"].max() + 1, rows["column"].max() + 1), np.nan)
    matrix[rows["row"], rows["column"]] = rows["input_log2"]
    method = rows["method"].iloc[0]
    options = {"seqknn": {"k": 2}}.get(method, {})
    if method in {"minprob", "qrilc"}:
        options = {"tune_sigma": rows["tune_sigma"].iloc[0]}

    result = impute_censored(pd.DataFrame(matrix), method=method, **options).to_numpy()

    observed = np.isfinite(matrix)
    np.testing.assert_array_equal(result[observed], matrix[observed])
    expected = rows.dropna(subset=["expected_fill_log2"])
    assert len(expected) == (~observed).sum()
    np.testing.assert_allclose(
        result[expected["row"], expected["column"]],
        expected["expected_fill_log2"],
        rtol=0,
        atol=1e-8,
    )
