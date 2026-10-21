from pathlib import Path

import pandas as pd

from mokume.core.constants import PIBAQ, PROTEIN_NAME, SAMPLE_ID
from mokume.quantification.pibaq import peptides_to_protein

TESTS_DIR = Path(__file__).parent


def test_pibaq_compute(tmp_path):
    """The file-oriented piBAQ workflow writes quantified data and its QC report."""
    output = tmp_path / "PXD017834-pibaq.tsv"
    qc_report = tmp_path / "QCprofile.pdf"
    args = {
        "fasta": str(
            TESTS_DIR
            / "example/Homo-sapiens-uniprot-reviewed-contaminants-decoy-202210.fasta"
        ),
        "peptides": str(TESTS_DIR / "example/PXD017834-peptides.csv"),
        "enzyme": "Trypsin",
        "normalize": True,
        "min_aa": 7,
        "max_aa": 30,
        "tpa": True,
        "ruler": True,
        "ploidy": 2,
        "cpc": 200,
        "organism": "human",
        "output": str(output),
        "verbose": True,
        "qc_report": str(qc_report),
    }

    peptides_to_protein(**args)

    result = pd.read_csv(output, sep="\t")
    assert not result.empty
    assert {PROTEIN_NAME, SAMPLE_ID, PIBAQ}.issubset(result.columns)
    assert qc_report.is_file()
