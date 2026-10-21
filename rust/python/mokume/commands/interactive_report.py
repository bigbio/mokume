#!/usr/bin/env python3
"""Interactive HTML DE-report command for the mokume wheel.

The Rust ``features2proteins`` pipeline owns the numbers: it writes the protein
intensity matrix (``--output``) and one differential-expression (DE) result CSV
per contrast (``--de-output`` / the path the CLI synthesizes). This script is the
thin "copy-py" glue that reproduces the standalone interactive HTML report the
Python ``features_to_proteins`` path emits via ``mokume/pipeline/stages.py``
``generate_interactive_report`` (lines 2087-2129), calling the same shared
``mokume.reports.interactive.generate_de_report``.

Design (matches ``peptides2protein_qc.py`` / ``de_plots.py``: Rust writes
the tables, Python only draws): we never recompute DE or the protein matrix here.
We read the CSVs Rust produced and call the shared mokume report builder, so the
cells in the report are the cells in the Rust CSVs. No mokume algorithm is
duplicated -- a system ``pip install mokume[analysis]`` provides the builder
(plotly is loaded from a CDN by the generated HTML; the Python builder itself
needs only pandas + numpy, but the availability gate matches Python and checks
for plotly).

argv contract (runnable via ``mokume interactive-report``):

    mokume interactive-report \
        --protein-matrix <proteins.csv> \
        --sdrf <sdrf.tsv> \
        [-o <report.html>] \
        [--log2fc 0.5] [--fdr 0.05] \
        [--highlight-protein PROTEIN] ... \
        [--contrast <key> <condA> <condB> <de_csv>] ...

Each ``--contrast`` flag carries one contrast: its output-file key, the two
condition labels, and the path to the DE result CSV Rust wrote. The report is
per-contrast. The output path mirrors ``stages.generate_interactive_report``:
with ``--output`` and a single contrast the path is used verbatim; with
multiple contrasts the contrast key is inserted before the extension; without
``--output`` the report lands in ``./report_<key>.html``.

Exit code 0 on success; non-zero (with a message on stderr) otherwise.
"""

from __future__ import annotations

import argparse
import sys

from ._de_cli import (
    add_de_result_arguments,
    add_protein_matrix_argument,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="mokume interactive-report",
        description="Build a DE HTML report.",
    )
    add_protein_matrix_argument(parser)
    parser.add_argument(
        "-s",
        "--sdrf",
        metavar="<FILE>",
        required=True,
        help="Sample metadata.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="<FILE>",
        default=None,
        help="Default: ./report_<key>.html.",
    )
    add_de_result_arguments(parser)
    return parser.parse_args(argv)


def _resolve_output_html(output, key, n_contrasts):
    """Mirror the output-path logic in ``stages.generate_interactive_report``."""
    from pathlib import Path

    if output:
        if n_contrasts == 1:
            return output
        base = output.rsplit(".", 1)
        return "{0}_{1}.html".format(base[0], key)
    return str(Path(".") / f"report_{key}.html")


def _validate_args(args):
    if args.log2fc < 0:
        raise SystemExit("Interactive report aborted: --log2fc must be non-negative")
    if not 0 <= args.fdr <= 1:
        raise SystemExit("Interactive report aborted: --fdr must be between 0 and 1")


def _require_sample_conditions(sample_to_condition):
    if not sample_to_condition:
        raise SystemExit(
            "Interactive report aborted: the SDRF yielded no sample conditions"
        )


def _validate_contrast_conditions(sample_to_condition, cond_a, cond_b):
    missing_conditions = [
        condition
        for condition in (cond_a, cond_b)
        if condition not in set(sample_to_condition.values())
    ]
    if missing_conditions:
        raise SystemExit(
            "Interactive report aborted: contrast conditions absent from SDRF: "
            + ", ".join(missing_conditions)
        )


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate_args(args)

    # Import pandas / mokume.reports lazily so a clear message is printed when
    # the analysis extra is missing, rather than an opaque ImportError trace.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Interactive report aborted: pandas is not installed ({0}). "
            "Install with: pip install mokume[analysis]".format(exc)
        )

    try:
        from mokume.reports import is_interactive_available
    except ImportError as exc:
        raise SystemExit(
            "Interactive report aborted: mokume reports dependencies are not "
            "installed ({0}). Install with: pip install mokume[analysis]".format(exc)
        )

    if not is_interactive_available():
        raise SystemExit(
            "Interactive report aborted: report dependencies (plotly) are not "
            "installed. Install with: pip install mokume[analysis]"
        )

    if not args.contrast:
        raise SystemExit(
            "Interactive report aborted: no contrasts supplied. The report is "
            "generated per differential-expression contrast."
        )

    from mokume.reports.interactive import generate_de_report
    from mokume.normalization.irs import detect_condition_from_sdrf

    protein_df = pd.read_csv(args.protein_matrix)
    sample_to_condition = detect_condition_from_sdrf(args.sdrf)
    _require_sample_conditions(sample_to_condition)
    highlight = args.highlight_protein or None

    n_contrasts = len(args.contrast)
    for key, cond_a, cond_b, de_csv in args.contrast:
        _validate_contrast_conditions(sample_to_condition, cond_a, cond_b)
        de_df = pd.read_csv(de_csv, float_precision="round_trip")
        output_html = _resolve_output_html(args.output, key, n_contrasts)
        generate_de_report(
            de_results=de_df,
            protein_df=protein_df,
            sample_to_condition=sample_to_condition,
            output_html=output_html,
            title=f"DE Report: {key} ({cond_a} vs {cond_b})",
            highlight_genes=highlight,
            log2fc_threshold=args.log2fc,
            fdr_threshold=args.fdr,
        )
        print("Interactive report saved to {0}".format(output_html))

    return 0


if __name__ == "__main__":
    sys.exit(main())
