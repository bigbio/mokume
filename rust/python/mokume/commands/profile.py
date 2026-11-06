"""Render one peptide/protein profile PNG per requested protein."""

import argparse
import importlib
from pathlib import Path


def main(argv=None):
    """Run ``mokume plot profile`` using exported peptide intensities."""
    parser = argparse.ArgumentParser(prog="mokume plot profile", description=__doc__)
    importlib.import_module("mokume.commands._de_cli").add_protein_matrix_argument(
        parser
    )
    parser.add_argument("--peptides", required=True, metavar="<FILE>")
    parser.add_argument("-s", "--sdrf", required=True, metavar="<FILE>")
    parser.add_argument(
        "--protein", action="append", required=True, metavar="<PROTEIN>"
    )
    parser.add_argument("-o", "--output", required=True, metavar="<DIR>")
    args = parser.parse_args(argv)

    pd = importlib.import_module("pandas")
    irs = importlib.import_module("mokume.normalization.irs")
    profiles = importlib.import_module("mokume.plotting.profile")

    matrix = pd.read_csv(args.protein_matrix)
    peptides = pd.read_csv(args.peptides)
    mapping = irs.detect_condition_from_sdrf(args.sdrf)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, protein in enumerate(args.protein):
        # Prefix with the request index; accessions may contain path separators.
        filename = output / f"profile_{index + 1}_{protein.replace('/', '_')}.png"
        profiles.plot_profile(matrix, peptides, mapping, protein, filename)
        print(f"Protein profile saved to {filename}")
    return 0
