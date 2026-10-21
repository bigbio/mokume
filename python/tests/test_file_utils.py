import pandas as pd

from mokume.io.parquet import combine_pibaq_tsv_files, create_anndata
from mokume.core.constants import (
    SAMPLE_ID,
    PROTEIN_NAME,
    PIBAQ,
    PIBAQ_NORMALIZED,
    PIBAQ_LOG,
)


def test_combine_pibaq_tsv_files(tmp_path):
    """Combine matching piBAQ tables without losing rows or columns."""
    columns = [SAMPLE_ID, PROTEIN_NAME, PIBAQ]
    pd.DataFrame([["S1", "P1", 10.0]], columns=columns).to_csv(
        tmp_path / "batch-a.tsv", sep="\t", index=False
    )
    pd.DataFrame([["S2", "P2", 20.0]], columns=columns).to_csv(
        tmp_path / "batch-b.tsv", sep="\t", index=False
    )

    combined = combine_pibaq_tsv_files(
        dir_path=str(tmp_path), pattern="batch-*.tsv", sep="\t"
    ).sort_values(SAMPLE_ID, ignore_index=True)

    pd.testing.assert_frame_equal(
        combined,
        pd.DataFrame([["S1", "P1", 10.0], ["S2", "P2", 20.0]], columns=columns),
    )


def test_create_anndata():
    """Create an AnnData matrix with requested layers and observation metadata."""
    df = pd.DataFrame(
        {
            SAMPLE_ID: ["S1", "S1", "S2", "S2"],
            PROTEIN_NAME: ["P1", "P2", "P1", "P2"],
            PIBAQ: [10.0, 20.0, 30.0, 40.0],
            PIBAQ_NORMALIZED: [0.1, 0.2, 0.3, 0.4],
            PIBAQ_LOG: [1.0, 2.0, 3.0, 4.0],
            "Condition": ["A", "A", "B", "B"],
        }
    )
    adata = create_anndata(
        df=df,
        obs_col=SAMPLE_ID,
        var_col=PROTEIN_NAME,
        value_col=PIBAQ,
        layer_cols=[PIBAQ_NORMALIZED, PIBAQ_LOG],
        obs_metadata_cols=["Condition"],
        var_metadata_cols=[],
    )

    assert adata.shape == (2, 2)
    assert adata.layers[PIBAQ_NORMALIZED].shape == (2, 2)
    assert adata.layers[PIBAQ_LOG].shape == (2, 2)
    assert adata.obs["Condition"].to_dict() == {"S1": "A", "S2": "B"}
