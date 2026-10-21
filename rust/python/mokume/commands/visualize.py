#!/usr/bin/env python3
"""t-SNE visualization command (``mokume.tsne_visualization``).

A pure-Python periphery command in the mokume wheel: it re-implements no compute
kernel; it reads the protein files and draws the t-SNE plot through the mokume
plotting helpers. Exposed as ``mokume plot tsne`` and in-process as
``mokume.tsne_visualization``.

Run requirements: the ``analysis`` extra (matplotlib + seaborn + scikit-learn):
``pip install mokume[analysis]``.

argv contract:
    --input/-i    PATH   folder that contains the protein files (required)
    --pattern/-p  STR    protein file glob pattern (default: proteins.tsv)
    --output/-o   PATH   destination PDF (required)
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mokume plot tsne",
        description="Render a t-SNE plot from protein tables.",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="<DIR>",
        required=True,
        help="Protein-table directory.",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        metavar="<GLOB>",
        default="proteins.tsv",
        help="Default: proteins.tsv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="<FILE>",
        required=True,
        help="PDF output.",
    )
    return parser.parse_args(argv)


def _compute_tsne_compat(
    df_pca, n_components=2, perplexity=30, learning_rate=200, n_iter=2000
):
    """sklearn-version-robust reproduction of ``mokume.commands.visualize.compute_tsne``.

    scikit-learn renamed ``TSNE``'s ``n_iter`` argument to ``max_iter`` in 1.5 and
    removed ``n_iter`` outright in 1.7, but the installed ``mokume.compute_tsne``
    still passes ``n_iter=`` and therefore raises ``TypeError`` on modern sklearn.
    Pick whichever keyword the installed ``TSNE`` accepts so the command runs
    across sklearn versions; the parameters are identical to mokume's.
    """
    import inspect

    import numpy as np
    import pandas as pd
    from sklearn.manifold import TSNE

    iter_kw = "n_iter" if "n_iter" in inspect.signature(TSNE).parameters else "max_iter"
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        **{iter_kw: n_iter},
    )
    tsne_results = tsne.fit_transform(np.asarray(df_pca))
    tsne_cols = [f"tSNE{i + 1}" for i in range(n_components)]
    df_tsne = pd.DataFrame(data=tsne_results, columns=tsne_cols)
    df_tsne.index = df_pca.index
    return df_tsne


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    try:
        import glob

        import pandas as pd

        constants = importlib.import_module("mokume.core.constants")
        from mokume.plotting import is_plotting_available
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "error: failed to import mokume plotting dependencies: "
            f"{exc}\nInstall them with: pip install mokume[analysis]",
            file=sys.stderr,
        )
        return 1

    if not is_plotting_available():
        print(
            "error: plotting dependencies (matplotlib, seaborn) are not "
            "installed. Install them with: pip install mokume[analysis]",
            file=sys.stderr,
        )
        return 1

    from mokume.plotting import compute_pca_with_plot, plot_tsne

    # Reproduce the upstream tsne_visualization body verbatim so the plotting
    # stays single-sourced in the mokume package.
    files = glob.glob(f"{args.input}/*{args.pattern}")
    if not files:
        print(
            f"error: no files matched '{args.input}/*{args.pattern}'",
            file=sys.stderr,
        )
        return 1

    dfs = []
    for f in files:
        reanalysis = (f.split("/")[-1].split("_")[0]).replace("-proteins.tsv", "")
        dfs += [
            pd.read_csv(
                f,
                usecols=[
                    constants.PROTEIN_NAME,
                    constants.SAMPLE_ID,
                    constants.PIBAQ_LOG,
                ],
                sep=None,
                engine="python",
            ).assign(reanalysis=reanalysis)
        ]

    total_proteins = pd.concat(dfs, ignore_index=True)

    normalize_df = pd.pivot_table(
        total_proteins,
        index=[constants.SAMPLE_ID, "reanalysis"],
        columns=constants.PROTEIN_NAME,
        values=constants.PIBAQ_LOG,
    )
    normalize_df = normalize_df.fillna(0)
    df_pca = compute_pca_with_plot(normalize_df, n_components=30)
    df_tsne = _compute_tsne_compat(df_pca)

    batch = df_tsne.index.get_level_values("reanalysis").tolist()
    df_tsne["batch"] = batch

    plot_tsne(
        df_tsne,
        "tSNE1",
        "tSNE2",
        "batch",
        args.output,
    )
    print(f"t-SNE plot written; input shape {total_proteins.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
