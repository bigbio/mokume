#!/usr/bin/env python3
"""piBAQ QC-report command (density + box plots) for the mokume wheel.

The Rust ``peptides2protein`` piBAQ path computes the protein table itself and
writes it as a tab-separated file. This script is the thin "copy-py" glue that
reproduces the QC report that the Python ``peptides_to_protein(verbose=True)``
path emits (``mokume/quantification/pibaq.py``): density (KDE) and
box plots, log2-scaled, one pair per quantification column, written into a single
PDF.

Design (blueprint section 2.5, option ii -- "Rust writes the TSV, Python only
draws"): the Rust kernel stays the single source of the numbers, so the cells in
the QC plots are byte-for-byte the cells in the Rust TSV. We do **not** recompute
piBAQ here; we only read the table Rust produced and call the shared plotting
helpers ``mokume.plotting.plot_distributions`` / ``plot_box_plot`` (exactly
the functions the Python verbose block uses). No mokume algorithm code lives in
Rust -- the first-class ``mokume.plotting`` helpers draw the figures (their
third-party deps come from the ``analysis`` extra).

argv contract (runnable via ``python -m mokume.commands.peptides2protein_qc``):

    python -m mokume.commands.peptides2protein_qc \
        --protein-table <proteins.tsv> \
        --qc-report <QCprofile.pdf> \
        [--plot-column PiBAQ|PiBAQPpb] \
        [--tpa] [--ruler]

Exit code 0 on success; non-zero (with a message on stderr) otherwise.
"""

import argparse
import sys


# Column headers, mirroring ``mokume.core.constants`` so the command reads the
# same output schema the Rust kernel writes (kept inline to avoid importing the
# mokume extension just for string constants -- the plotting import below is the
# only hard mokume need).
SAMPLE_ID = "SampleID"
PIBAQ = "PiBAQ"
PIBAQ_PPB = "PiBAQPpb"
TPA = "TPA"
COPYNUMBER = "CopyNumber"
CONCENTRATION_NM = "Concentration[nM]"


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Render a piBAQ QC report.")
    parser.add_argument(
        "--protein-table",
        metavar="<FILE>",
        required=True,
        help="piBAQ TSV input.",
    )
    parser.add_argument(
        "--qc-report",
        metavar="<FILE>",
        required=True,
        help="PDF output.",
    )
    parser.add_argument(
        "--plot-column",
        metavar="<COLUMN>",
        default=PIBAQ,
        help="Default: PiBAQ.",
    )
    parser.add_argument(
        "--tpa",
        action="store_true",
        help="Include the TPA distribution.",
    )
    parser.add_argument(
        "--ruler",
        action="store_true",
        help="Include CopyNumber and Concentration[nM] distributions.",
    )
    return parser.parse_args(argv)


def _require_column(frame, column, table_path):
    if column not in frame.columns:
        raise SystemExit(
            "QC report aborted: column '{0}' is not present in '{1}'. "
            "Available columns: {2}".format(
                column, table_path, ", ".join(map(str, frame.columns))
            )
        )


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Import pandas / mokume.plotting lazily so a clear message is printed when
    # the analysis extra is missing, rather than an opaque ImportError trace.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "QC report aborted: pandas is not installed ({0}). "
            "Install with: pip install mokume[analysis]".format(exc)
        )

    try:
        from mokume.plotting import PdfPages, plot_box_plot, plot_distributions
    except ImportError as exc:
        raise SystemExit(
            "QC report aborted: mokume plotting dependencies are not "
            "installed ({0}). Install with: pip install mokume[analysis]".format(exc)
        )

    res = pd.read_csv(args.protein_table, sep="\t")
    _require_column(res, SAMPLE_ID, args.protein_table)
    _require_column(res, args.plot_column, args.protein_table)

    # ``plot_width = len(set(SampleID)) * 0.5 + 10`` -- identical to
    # Same sample-count-based plot width as the canonical piBAQ module.
    plot_width = len(set(res[SAMPLE_ID])) * 0.5 + 10

    pdf = PdfPages(args.qc_report)
    try:
        # Primary piBAQ column: density + box, matching the canonical module.
        density = plot_distributions(
            res,
            args.plot_column,
            SAMPLE_ID,
            log2=True,
            width=plot_width,
            title="{0} Distribution".format(args.plot_column),
        )
        box = plot_box_plot(
            res,
            args.plot_column,
            SAMPLE_ID,
            log2=True,
            width=plot_width,
            title="{0} Distribution".format(args.plot_column),
            violin=False,
        )
        pdf.savefig(density, bbox_inches="tight")
        pdf.savefig(box, bbox_inches="tight")

        if args.tpa:
            _require_column(res, TPA, args.protein_table)
            density_tpa = plot_distributions(
                res,
                TPA,
                SAMPLE_ID,
                log2=True,
                width=plot_width,
                title="TPA Distribution",
            )
            box_tpa = plot_box_plot(
                res,
                TPA,
                SAMPLE_ID,
                log2=True,
                width=plot_width,
                title="{0} Distribution".format(TPA),
                violin=False,
            )
            pdf.savefig(density_tpa, bbox_inches="tight")
            pdf.savefig(box_tpa, bbox_inches="tight")

        if args.ruler:
            for column in (COPYNUMBER, CONCENTRATION_NM):
                _require_column(res, column, args.protein_table)
                density_col = plot_distributions(
                    res,
                    column,
                    SAMPLE_ID,
                    width=plot_width,
                    log2=True,
                    title="{0} Distribution".format(column),
                )
                box_col = plot_box_plot(
                    res,
                    column,
                    SAMPLE_ID,
                    width=plot_width,
                    log2=True,
                    title="{0} Distribution".format(column),
                    violin=False,
                )
                pdf.savefig(density_col, bbox_inches="tight")
                pdf.savefig(box_col, bbox_inches="tight")
    finally:
        pdf.close()

    print(
        "QC report written to {0} ({1} sample(s))".format(
            args.qc_report, len(set(res[SAMPLE_ID]))
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
