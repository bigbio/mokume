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
import importlib
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
        from mokume import plotting
    except ImportError as exc:
        raise SystemExit(
            "QC report aborted: mokume plotting dependencies are not "
            "installed ({0}). Install with: pip install mokume[analysis]".format(exc)
        )

    res = pd.read_csv(args.protein_table, sep="\t")
    _require_column(res, SAMPLE_ID, args.protein_table)
    _require_column(res, args.plot_column, args.protein_table)

    _render_pdf(res, args, plotting)

    print(
        "QC report written to {0} ({1} sample(s))".format(
            args.qc_report, len(set(res[SAMPLE_ID]))
        )
    )
    return 0


def _render_pdf(frame, args, plotting):
    """Render distribution pairs and a primary-abundance rank page."""
    plt = importlib.import_module("matplotlib.pyplot")
    common = importlib.import_module("mokume.plotting._common")
    diagnostics = importlib.import_module("mokume.plotting.diagnostics")

    columns = [args.plot_column]
    if args.tpa:
        columns.append(TPA)
    if args.ruler:
        columns.extend((COPYNUMBER, CONCENTRATION_NM))
    width = min(16, frame[SAMPLE_ID].nunique() * 0.5 + 6)
    with common.static_style(), plotting.PdfPages(args.qc_report) as pdf:
        for column in columns:
            _require_column(frame, column, args.protein_table)
            options = {
                "log2": True,
                "width": width,
                "title": f"{column} Distribution",
            }
            for plot in (plotting.plot_distributions, plotting.plot_box_plot):
                figure = plot(frame, column, SAMPLE_ID, **options)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        matrix = frame.pivot_table(
            index="ProteinName", columns=SAMPLE_ID, values=args.plot_column
        ).reset_index()
        figure = diagnostics.plot_abundance_rank(common.log2_matrix(matrix))
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    sys.exit(main())
