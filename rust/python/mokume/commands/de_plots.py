#!/usr/bin/env python3
"""DE-plotting command (volcano / heatmap / PCA) for the mokume wheel.

The Rust ``features2proteins`` pipeline owns the numbers: it writes the protein
intensity matrix (``--output``) and one differential-expression (DE) result CSV
per contrast (``--de-output`` / the path the CLI synthesizes inside the plot
directory). This script is the thin "copy-py" glue that reproduces the volcano /
heatmap / PCA plots the Python ``features_to_proteins`` path emits via
``mokume/pipeline/stages.py`` ``generate_plots`` (lines 2061-2085) and its
helpers, calling the same shared plotting functions
``mokume.plotting.differential_expression.plot_volcano`` / ``plot_heatmap`` /
``plot_pca_conditions``.

Design (matches ``peptides2protein_qc.py``: Rust writes the tables, Python
only draws): we never recompute DE or the protein matrix here. We read the CSVs
Rust produced and call the shared mokume plotting helpers, so the cells in the
plots are the cells in the Rust CSVs. No mokume algorithm is duplicated -- a
system ``pip install mokume[analysis]`` provides the helpers.

argv contract:

    mokume plot de \
        -p <proteins.csv> -o <plots/> \
        [--sdrf <sdrf.tsv>] \
        [--volcano] [--heatmap] \
        [--log2fc 0.5] [--fdr 0.05] \
        [--highlight-protein PROTEIN] ... \
        [--contrast <key> <condA> <condB> <de_csv>] ...

    mokume plot pca -p <proteins.csv> -s <sdrf.tsv> -o <pca.png>

Each ``--contrast`` flag carries one contrast: its output-file key (Python's
``f"{condA}-{condB}"`` or the single-contrast ``--de-output`` stem), the two
condition labels, and the path to the DE result CSV Rust wrote. Volcano and
heatmap plots are per-contrast; the PCA plot is one figure over all conditions
and needs no DE table.

Exit code 0 on success; non-zero (with a message on stderr) otherwise.
"""

from __future__ import annotations

import argparse
import sys

from mokume.commands._de_cli import (
    add_de_result_arguments,
    add_protein_matrix_argument,
)


def _parse_args(argv, mode):
    description = (
        "Render a PCA-by-condition plot."
        if mode == "pca"
        else "Render volcano plots and DE heatmaps."
    )
    parser = argparse.ArgumentParser(
        prog=f"mokume plot {mode}", description=description
    )
    add_protein_matrix_argument(parser)
    if mode == "pca":
        parser.add_argument("-s", "--sdrf", metavar="<FILE>", required=True)
        parser.add_argument("-o", "--output", metavar="<FILE>", required=True)
        return parser.parse_args(argv)
    if mode != "de":
        raise SystemExit(f"unknown plot mode: {mode}")
    parser.add_argument("-o", "--outdir", metavar="<DIR>", required=True)
    parser.add_argument(
        "-s",
        "--sdrf",
        metavar="<FILE>",
        default=None,
        help="Required for heatmap annotations.",
    )
    parser.add_argument("--volcano", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    add_de_result_arguments(parser)
    return parser.parse_args(argv)


def _plot_sample_conditions(sample_to_condition, protein_df):
    """Restrict SDRF conditions to samples present in the protein matrix."""
    protein_col = protein_df.columns[0]
    available = [column for column in protein_df.columns if column != protein_col]
    return {
        sample: condition
        for sample, condition in sample_to_condition.items()
        if sample in available
    }


def _significant_de_proteins(de_df, log2fc_threshold, fdr_threshold):
    """Mirror ``stages._significant_de_proteins`` (adj_pvalue + |log2FC| gate)."""
    return de_df[
        (de_df["adj_pvalue"] < fdr_threshold)
        & (de_df["log2FC"].abs() > log2fc_threshold)
    ]


def _generate_volcano_plots(pd, contrasts, plot_dir, args):
    """Mirror ``stages._generate_volcano_plots``: one volcano per contrast."""
    from mokume.plotting.differential_expression import plot_volcano

    highlight = args.highlight_protein or None
    for key, cond_a, cond_b, de_csv in contrasts:
        de_df = pd.read_csv(de_csv, float_precision="round_trip")
        output_file = str(plot_dir / "volcano_{0}.png".format(key))
        plot_volcano(
            de_df,
            log2fc_threshold=args.log2fc,
            fdr_threshold=args.fdr,
            highlight_genes=highlight,
            title=f"Volcano Plot: {key} ({cond_a} vs {cond_b})",
            output_file=output_file,
        )
        print("Volcano plot saved to {0}".format(output_file))


def _generate_heatmap_plots(
    pd, contrasts, protein_df, sample_to_condition, plot_dir, args
):
    """Mirror ``stages._generate_heatmap_plots``: per-contrast DE heatmap."""
    from mokume.plotting.differential_expression import plot_heatmap

    protein_col = protein_df.columns[0]
    for key, cond_a, cond_b, de_csv in contrasts:
        de_df = pd.read_csv(de_csv, float_precision="round_trip")
        sig = _significant_de_proteins(de_df, args.log2fc, args.fdr)
        if sig.empty:
            print("Heatmap skipped for {0}: no significant proteins".format(key))
            continue

        sig_proteins = (
            sig.sort_values(
                by="log2FC", key=lambda series: series.abs(), ascending=False
            )["ProteinName"]
            .head(50)
            .tolist()
        )
        contrast_samples = [
            sample
            for sample in protein_df.columns
            if sample != protein_col
            and sample_to_condition.get(sample) in (cond_a, cond_b)
        ]
        contrast_mapping = {
            sample: condition
            for sample, condition in sample_to_condition.items()
            if condition in (cond_a, cond_b)
        }
        output_file = str(plot_dir / "heatmap_{0}.png".format(key))
        plot_heatmap(
            protein_df[[protein_col] + contrast_samples],
            contrast_mapping,
            proteins=sig_proteins,
            title="DE Heatmap: {0} (top {1}/{2} sig)".format(
                key, len(sig_proteins), len(sig)
            ),
            output_file=output_file,
        )
        print("Heatmap saved to {0}".format(output_file))


def _generate_pca_plot(protein_df, sample_to_condition, output_file):
    """Mirror ``stages._generate_pca_plot``: one PCA figure over all conditions."""
    from mokume.plotting.differential_expression import plot_pca_conditions

    plot_pca_conditions(
        protein_df,
        sample_to_condition,
        title="PCA by Condition",
        output_file=output_file,
    )
    print("PCA plot saved to {0}".format(output_file))


def _validate_de_args(args):
    if not any((args.volcano, args.heatmap)):
        raise SystemExit("DE plots aborted: select --volcano or --heatmap")
    if (args.volcano or args.heatmap) and not args.contrast:
        raise SystemExit(
            "DE plots aborted: --volcano/--heatmap require at least one --contrast"
        )
    if args.heatmap and not args.sdrf:
        raise SystemExit("DE plots aborted: --heatmap requires --sdrf")
    if args.sdrf and not args.heatmap:
        raise SystemExit("DE plots aborted: --sdrf only applies to --heatmap")
    if args.highlight_protein and not args.volcano:
        raise SystemExit("DE plots aborted: --highlight-protein requires --volcano")
    if args.log2fc < 0:
        raise SystemExit("DE plots aborted: --log2fc must be non-negative")
    if not 0 <= args.fdr <= 1:
        raise SystemExit("DE plots aborted: --fdr must be between 0 and 1")


def main(argv=None, mode="de"):
    """Run the DE plotting command."""
    args = _parse_args(sys.argv[1:] if argv is None else argv, mode)
    if mode == "de":
        _validate_de_args(args)

    from pathlib import Path

    # Import pandas / mokume.plotting lazily so a clear message is printed when
    # the analysis extra is missing, rather than an opaque ImportError trace.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "DE plots aborted: pandas is not installed ({0}). "
            "Install with: pip install mokume[analysis]".format(exc)
        )

    try:
        from mokume.plotting import is_plotting_available
    except ImportError as exc:
        raise SystemExit(
            "DE plots aborted: mokume plotting dependencies are not installed "
            "({0}). Install with: pip install mokume[analysis]".format(exc)
        )

    if not is_plotting_available():
        raise SystemExit(
            "DE plots aborted: mokume plotting dependencies (matplotlib, seaborn) "
            "are not installed. Install with: pip install mokume[analysis]"
        )

    protein_df = pd.read_csv(args.protein_matrix)

    # Sample -> condition mapping (heatmap annotation + PCA coloring) mirrors
    # ``stages`` and is restricted to columns present in the protein matrix.
    sample_to_condition = {}
    if mode == "pca" or args.sdrf:
        from mokume.normalization.irs import detect_condition_from_sdrf

        sample_to_condition = detect_condition_from_sdrf(args.sdrf)
        if not sample_to_condition:
            raise SystemExit("DE plots aborted: the SDRF yielded no sample conditions")
    sample_to_condition = _plot_sample_conditions(sample_to_condition, protein_df)

    if mode == "pca":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _generate_pca_plot(protein_df, sample_to_condition, str(output))
        return 0

    plot_dir = Path(args.outdir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.volcano and args.contrast:
        _generate_volcano_plots(pd, args.contrast, plot_dir, args)
    if args.heatmap and args.contrast and sample_to_condition:
        _generate_heatmap_plots(
            pd, args.contrast, protein_df, sample_to_condition, plot_dir, args
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
