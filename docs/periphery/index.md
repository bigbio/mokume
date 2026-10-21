# Python Periphery

The periphery is the Python half of the toolkit: plotting, tissue maps,
interactive reports, the piBAQ QC report, and the pure-Python method fallbacks.
It lives in
`rust/python/mokume/commands/` (plus `rust/python/mokume/reports/`,
`rust/python/mokume/normalization/`, `rust/python/mokume/imputation/` for the
analysis fallbacks) and is reached **only** through the
`pip install mokume` wheel. The Rust compute extension does not import this
periphery.

## Reporting reads kernel output; fallbacks are explicit

Plotting, QC, and reporting commands consume tables the **kernel** wrote without
re-running kernel-supported computation. TissueMap instead derives its
documented downstream normalization, batch correction, tissue-specificity, and
atlas outputs from QPX data.

One explicit fallback computes an operation the Rust kernel does not provide:
`mokume.impute(method='missforest')` runs the scikit-learn estimator. It is
documented under [Rust Wheel](../rust-wheel.md) and
[Analysis Fallbacks](analysis-fallbacks.md), rather than being presented as
kernel-produced results.

## Extras matrix

The base wheel depends on pyOpenMS because both Rust-backed piBAQ commands use
the installed `ProteaseDB` catalog for FASTA digestion. pyOpenMS itself installs
numpy, pandas, and matplotlib; the extras select each command's additional
libraries. Install just the extra for the command you run.

| Command | Extra | Third-party libraries |
| --- | --- | --- |
| `mokume plot tsne` | `analysis` | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn |
| `mokume.peptides2protein_qc` | `analysis` | numpy, pandas, matplotlib, seaborn |
| `mokume plot pca` / `mokume plot de` | `analysis` | numpy, pandas, matplotlib, seaborn, scikit-learn |
| `mokume interactive-report` | `analysis` | numpy, pandas, plotly |
| `mokume tissuemap` | `tissuemap` | scanpy, anndata, umap-learn, combat, matplotlib, seaborn, pyarrow |
| `mokume.qc_report` / `mokume.workflow_comparison` | `analysis` | numpy, pandas, scipy, scikit-learn |
| `mokume.impute` | `analysis` | numpy, pandas, scipy, scikit-learn |

```bash
pip install mokume                 # compute kernel + Python API
pip install "mokume[analysis]"     # + plots, reports, QC, comparison, and missforest
pip install "mokume[tissuemap]"    # + per-dataset tissue proteome analysis
pip install "mokume[plugin]"       # + local MCP service
pip install "mokume[all]"          # everything
```

The exact dependency lists are declared in `pyproject.toml`'s
`[project.optional-dependencies]`. DirectLFQ and ComBat are native Rust, while
pyOpenMS-backed FASTA digestion is part of the base piBAQ capability.

## CLI and Python entry points

The public plotting, report, and TissueMap workflows are available directly
from the wheel console command:

```bash
mokume plot tsne --input ./proteins --pattern proteins.tsv --output tsne.pdf
mokume tissuemap --input ./data --outdir ./out
mokume plot de --help
mokume interactive-report --help
```

Most also have an ergonomic wrapper on the top-level `mokume` package:

```python
import mokume

# kwarg wrappers (the kwargs -> flags rule applies):
mokume.tsne_visualization(
    input="./proteins", pattern="proteins.tsv", output="tsne.pdf"
)
mokume.tissuemap(input="./data", outdir="./out")
mokume.peptides2protein_qc(protein_table="proteins.tsv", qc_report="QC.pdf")

# de_plots / interactive_report take an explicit argv (the per-contrast
# --contrast KEY A B CSV flag repeats, which keyword arguments cannot express):
mokume.de_plots(["--protein-matrix", "proteins.csv", "--outdir", "plots",
                 "--volcano", "--contrast", "c1", "A", "B", "de.csv"])

# QC / comparison reports + the pure-Python method fallbacks (analysis extra):
mokume.qc_report(protein_matrix="proteins.csv", sdrf="x.sdrf.tsv", output="qc.html")
mokume.impute("proteins.csv", method="missforest", output="imputed.csv")
```

The kwarg wrappers follow the validated per-command
[kwargs &rarr; flags rule](../rust-wheel.md). `de_plots` and
`interactive_report` take a literal argument list because their per-contrast
flag repeats in a shape keyword arguments cannot express.

## What lives where

- **Tissue Proteome Atlas** &rarr; [`mokume tissuemap`](tissuemap.md) — PCA /
  t-SNE / UMAP / AdaTiSS tissue-specificity scoring, markers, atlas plots.
- **Visualization & Reports** &rarr;
  [t-SNE / DE plots / interactive report](visualization.md).
- **Analysis Fallbacks** &rarr; [`mokume.impute`,
  `mokume.qc_report`, `mokume.workflow_comparison`](analysis-fallbacks.md) —
  methods the Rust kernel does not reproduce, plus the QC / comparison reports.
