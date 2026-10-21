from pathlib import Path
import pandas as pd

from mokume.core.constants import PIBAQ_NORMALIZED, SAMPLE_ID
from mokume.postprocessing.reshape import (
    remove_samples_low_protein_number,
    remove_missing_values,
    describe_expression_metrics,
)

TESTS_DIR = Path(__file__).parent


def _load_pibaq_fixture() -> pd.DataFrame:
    """Load the historical iBAQ fixture under the canonical piBAQ schema."""
    return pd.read_csv(
        TESTS_DIR / "example/PXD017834-example-ibaq.tsv", sep="\t"
    ).rename(columns={"IbaqNorm": PIBAQ_NORMALIZED})


def test_remove_samples_low_protein_number():
    """Samples below the requested unique-protein count are removed."""
    pibaq_df = _load_pibaq_fixture()
    new_pibaq = remove_samples_low_protein_number(pibaq_df, min_protein_num=286)

    protein_counts = new_pibaq.groupby(SAMPLE_ID)["ProteinName"].nunique()
    assert protein_counts.ge(286).all()
    assert set(new_pibaq[SAMPLE_ID]) == {
        "PXD017834-Sample-1",
        "PXD017834-Sample-2",
        "PXD017834-Sample-3",
        "PXD017834-Sample-4",
        "PXD017834-Sample-5",
    }


def test_remove_missing_values():
    """Samples above the requested expression missingness are removed."""
    pibaq_df = _load_pibaq_fixture()
    new_pibaq = remove_missing_values(
        pibaq_df, missingness_percentage=1, expression_column=PIBAQ_NORMALIZED
    )

    assert set(new_pibaq[SAMPLE_ID]) == {
        "PXD017834-Sample-1",
        "PXD017834-Sample-3",
        "PXD017834-Sample-4",
    }


def test_describe_expression_metrics():
    """Expression summaries contain per-sample counts for supported metrics."""
    pibaq_df = _load_pibaq_fixture()

    metrics = describe_expression_metrics(pibaq_df)

    assert metrics.index.tolist() == sorted(pibaq_df[SAMPLE_ID].unique())
    assert metrics[(PIBAQ_NORMALIZED, "count")].to_dict() == {
        sample: float(count)
        for sample, count in pibaq_df.groupby(SAMPLE_ID)[PIBAQ_NORMALIZED]
        .count()
        .items()
    }
