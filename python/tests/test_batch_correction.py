import pytest
import pandas as pd

from mokume.commands.batch_correct import (
    get_batch_id_from_sample_names,
    run_batch_correction,
)
from mokume.core.constants import PIBAQ, PIBAQ_BEC, PROTEIN_NAME, SAMPLE_ID


@pytest.fixture(name="batch_input_dir")
def _batch_input_dir(tmp_path):
    """Create two batches with two samples and five complete proteins each."""
    folder = tmp_path / "batches"
    folder.mkdir()
    samples = ["PXD1-A", "PXD1-B", "PXD2-A", "PXD2-B"]
    intensities = {
        "P1": [10.0, 11.0, 20.0, 22.0],
        "P2": [30.0, 33.0, 45.0, 46.0],
        "P3": [5.0, 8.0, 12.0, 15.0],
        "P4": [100.0, 90.0, 150.0, 140.0],
        "P5": [50.0, 55.0, 70.0, 80.0],
    }
    rows = [
        {PROTEIN_NAME: protein, SAMPLE_ID: sample, PIBAQ: values[index]}
        for protein, values in intensities.items()
        for index, sample in enumerate(samples)
    ]
    pd.DataFrame(rows).to_csv(folder / "synthetic.ibaq.tsv", sep="\t", index=False)
    return folder


@pytest.mark.parametrize(
    "samples",
    [
        ["PXD000001-A", "PXD000001-B", "PXD000002-A"],
        pd.Index(["PXD000001-A", "PXD000001-B", "PXD000002-A"]),
    ],
)
def test_get_batch_id_supports_python_and_pandas_sequences(samples):
    """Batch labels remain stable for Python and pandas sequences."""
    assert get_batch_id_from_sample_names(samples).tolist() == [0, 0, 1]


def test_correct_batches(tmp_path, batch_input_dir):
    """Batch correction writes corrected long data and matching AnnData."""
    pytest.importorskip("inmoose", reason="inmoose is required for batch correction")
    anndata = pytest.importorskip(
        "anndata", reason="anndata is required for batch correction"
    )

    source = pd.read_csv(batch_input_dir / "synthetic.ibaq.tsv", sep="\t")
    output_path = tmp_path / "pibaq_corrected_combined.tsv"
    corrected = run_batch_correction(
        folder=batch_input_dir,
        pattern="*ibaq.tsv",
        comment="#",
        sep="\t",
        output=output_path,
        pibaq_raw_column=PIBAQ,
        export_anndata=True,
    )

    assert output_path.is_file()
    assert corrected.shape == (20, 4)
    assert corrected[PIBAQ_BEC].notna().all()
    identity_columns = [PROTEIN_NAME, SAMPLE_ID]
    assert set(corrected[identity_columns].itertuples(index=False, name=None)) == set(
        source[identity_columns].itertuples(index=False, name=None)
    )
    pd.testing.assert_frame_equal(
        corrected[[*identity_columns, PIBAQ]]
        .sort_values(identity_columns)
        .reset_index(drop=True),
        source[[*identity_columns, PIBAQ]]
        .sort_values(identity_columns)
        .reset_index(drop=True),
    )

    adata_path = output_path.with_suffix(".h5ad")
    assert adata_path.is_file()
    adata = anndata.read_h5ad(adata_path)
    assert adata.shape == (4, 5)
    assert adata.layers[PIBAQ_BEC].shape == (4, 5)
    assert set(adata.obs_names) == set(source[SAMPLE_ID])
    assert set(adata.var_names) == set(source[PROTEIN_NAME])


def test_correct_batches_rejects_invalid_sample_ids(tmp_path, batch_input_dir):
    """Reject sample identifiers that cannot be assigned to a batch."""
    input_path = batch_input_dir / "synthetic.ibaq.tsv"
    data = pd.read_csv(input_path, sep="\t")
    data.loc[data[SAMPLE_ID] == "PXD1-A", SAMPLE_ID] = "invalid sample"
    data.to_csv(input_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="Invalid sample IDs found in the data"):
        run_batch_correction(
            folder=batch_input_dir,
            pattern="*ibaq.tsv",
            comment="#",
            sep="\t",
            output=tmp_path / "unused.tsv",
            pibaq_raw_column=PIBAQ,
        )


def test_correct_batches_rejects_missing_required_column(tmp_path, batch_input_dir):
    """Reject a configured sample column that is absent from the input."""
    with pytest.raises(ValueError, match="NonexistentColumn"):
        run_batch_correction(
            folder=batch_input_dir,
            pattern="*ibaq.tsv",
            comment="#",
            sep="\t",
            output=tmp_path / "unused.tsv",
            sample_id_column="NonexistentColumn",
            pibaq_raw_column=PIBAQ,
        )


def test_correct_batches_rejects_empty_input_directory(tmp_path):
    """Reject an input directory with no matching quantification files."""
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(ValueError, match="Failed to load input files: No files found"):
        run_batch_correction(
            folder=folder,
            pattern="*ibaq.tsv",
            comment="#",
            sep="\t",
            output=tmp_path / "unused.tsv",
        )
