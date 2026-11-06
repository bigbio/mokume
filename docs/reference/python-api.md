# Python API

`pip install mokume` gives you a thin Python wheel over the Rust compute
kernel, in the PyO3/maturin layout used by projects such as polars and
pydantic-core (Python imports a compiled Rust extension). This wheel does not
expose the separately installed pure-Python package's rich class-based API. Its
kernel-supported compute runs in-process through the `mokume._mokume` extension;
plotting and reporting read the kernel's tables, TissueMap derives downstream
atlas outputs from QPX data, and explicitly documented Python-only fallbacks
compute operations the kernel does not provide.

The package has two layers:

- **Compute wrappers** — full commands such as `mokume.features2proteins(...)`
  plus matrix-level `normalize_matrix`, `impute_matrix`, and
  `differential_expression` calls. They run **in-process, with no subprocess**.
- **Periphery and fallbacks** — plotting, tissue maps, DE plots, interactive
  reports, piBAQ QC, and explicit pure-Python fallbacks such as `missforest`.
  These live in `mokume.commands.*` / `mokume.reports.*` and are reached through
  the ergonomic wrappers below. Each needs an
  [install extra](#install-extras).

```python
import mokume

mokume.version()   # the kernel version string
```

---

## Compute wrappers

Each compute wrapper validates keyword arguments against that command's schema,
maps them to the exact flags documented in the [CLI Reference](cli.md), and runs
the kernel in-process. Unknown keywords and invalid boolean/list shapes fail
before dispatch instead of being converted into accidental CLI flags.

```python
import mokume

# feature parquet -> protein matrix (+ optional DE)
mokume.features2proteins(parquet="features.parquet", output="proteins.csv")

# feature parquet -> peptide-level output
mokume.features2peptides(parquet="features.parquet", output="peptides.csv")

# peptide-level input -> protein quantities
mokume.peptides2protein(quant_method="pibaq", peptides="peptides.parquet",
                        fasta="proteome.fasta", output="proteins.tsv")

# ComBat batch-effect correction on piBAQ output
mokume.correct_batches(input="pibaq_dir", output="corrected.tsv")
```

### kwargs → flags rule

Each wrapper translates its documented `**kwargs` into a CLI argument list:

| keyword form | becomes | example |
|--------------|---------|---------|
| `key=value` | `--key value` (`_` → `-`) | `quant_method="pibaq"` → `--quant-method pibaq` |
| `key=True` | `--key` (a bare flag) | `batch_correction=True` → `--batch-correction` |
| `key=[a, b]` | repeated singular flags | `batch_covariate=["sex", "tissue"]` → `--batch-covariate sex --batch-covariate tissue` |
| `key=[(a, b)]` | repeated two-value flags | `de_contrast=[("A", "B")]` → `--de-contrast A B` |
| `key=None` / `key=False` | skipped | omitted entirely |

```python
mokume.features2proteins(
    parquet="features.parquet",
    output="proteins.csv",
    sdrf="experiment.sdrf.tsv",
    quant_method="maxlfq",
    batch_correction=True,
    batch_covariate=["characteristics[sex]"],
    de_contrast=[("NASH", "HL")],
    threads=24,
)
```

The wrappers accept canonical keyword names only. A sequence is accepted only
for repeatable fields; all other sequence-valued keywords raise `TypeError`.

### `mokume.run([...])` — full control

When you need flags a keyword cannot express (e.g. a repeated `--contrast KEY A B CSV`), pass the argument vector verbatim. `run` accepts the subcommand name as the first element:

```python
mokume.run(["quantify", "features2proteins", "--parquet", "x.parquet",
            "--output", "y.csv"])
mokume.run(["correct-batches", "--input", "pibaq_dir", "--output", "corrected.tsv"])
```

`mokume.run` and the four wrappers raise on a dispatch failure and surface clap's usage errors; they never tear down the hosting interpreter.

Rust tracing is process-global. Put `--log-level` and `--log-file` in an explicit
`mokume.run([...])` call when you need non-default logging, and repeat that exact
configuration on later in-process calls. Requesting a different level or file
raises a clear error instead of silently ignoring the new values.

---

## Matrix-level compute

These calls reuse the same Rust implementations as the full pipeline without
rerunning QPX loading or protein aggregation. Matrices are row-major linear
intensities (`values[protein][sample]`); use `None` or a non-finite float for a
missing cell. Every call accepts an explicit thread count.

```python
import mokume

values = [
    [100.0, 120.0, None, 240.0],
    [400.0, 420.0, 800.0, 820.0],
]

normalized = mokume.normalize_matrix(
    values, "median", ["A1", "A2", "B1", "B2"], threads=24
)
imputed = mokume.impute_matrix(normalized, "mindet", threads=24)
de_rows = mokume.differential_expression(
    ["P1", "P2"],
    imputed,
    2,
    2,
    "limma",
    condition_a="A",
    condition_b="B",
    threads=24,
)
```

`normalize_matrix` supports `none`, `median`, `mean`, `quantile`, `rlr`,
`loess`, `hierarchical`, and `tmm`. `impute_matrix` supports every native Rust
imputer; `missforest` remains the explicit Python fallback. The DE call supports
`limma`, `deqms`, `rots`, `limrots`, `proda`, and `ensemble`, returning a list
of dictionaries with the same result-column names as the corresponding command
table. Because matrix-level DE has no quantification context, it requires a
concrete method and rejects `auto`. Protein identifiers must be non-empty and
unique. `deqms` and ensembles containing `deqms` also require
`peptide_counts=[...]`, aligned one-to-one with the protein rows and containing
finite positive integers. For two-group `deqms`, `precision=[...]` adds the
[precision weights](../concepts/differential-expression.md#precision-weights):
a matrix shaped like the values with a positive precision for every quantified
cell and `None` elsewhere, as written by `--export-precision`.

---

## Periphery

The plotting and reporting periphery reads the tables the kernel wrote without
recomputing them, so the cells in its plots match the kernel output. TissueMap
derives downstream atlas outputs from QPX data, while explicit fallbacks compute
operations unsupported by the Rust kernel. Each command lives in
`mokume.commands.<name>` with a `main(argv)` entry point, is routed by the
unified `mokume` console command, and usually has an ergonomic wrapper on the
top-level package.

### Visualization

```python
import mokume

# t-SNE over a folder of protein files (analysis extra)
mokume.tsne_visualization(input="./proteins", pattern="proteins.tsv",
                          output="tsne.pdf")

# per-dataset tissue proteome analysis (tissuemap extra)
mokume.tissuemap(input="./data", outdir="./out", threads=24)

# piBAQ QC report from a protein table (analysis extra)
mokume.peptides2protein_qc(protein_table="proteins.tsv", qc_report="QC.pdf")
```

`de_plots` and `interactive_report` take an explicit argv (the per-contrast `--contrast KEY A B CSV` flag repeats, which keyword arguments cannot express):

```python
# DE volcano / heatmap from kernel-written CSVs (analysis extra)
mokume.de_plots(["--protein-matrix", "proteins.csv", "--outdir", "plots",
                 "--volcano", "--contrast", "c1", "A", "B", "de.csv"])

# interactive HTML report from kernel CSVs (analysis extra)
mokume.interactive_report([
    "--protein-matrix", "proteins.csv",
    "--sdrf", "experiment.sdrf.tsv",
    "--output", "report.html",
    "--contrast", "c1", "A", "B", "de.csv",
])
```

Run `mokume plot de --help` / `mokume interactive-report --help` for the flags.

### QC and workflow-comparison reports

```python
# single-matrix QC report: PCA / t-SNE / silhouette / CV / missing-value / DE-quality
path = mokume.qc_report(
    protein_matrix="proteins.csv",
    sdrf="experiment.sdrf.tsv",
    output="qc.html",
    de_results="de.csv",     # optional
)

# compare several quantification workflows in one HTML report
path = mokume.workflow_comparison(
    workflows=[
        {"name": "maxlfq", "protein_matrix": "maxlfq.csv", "sdrf": "x.sdrf.tsv"},
        {"name": "pibaq",   "protein_matrix": "pibaq.csv",   "sdrf": "x.sdrf.tsv"},
    ],
    output="comparison.html",
)
```

Both need the `analysis` extra. `qc_report` also accepts `highlight_genes` and the
optional `peptides`, `unnormalized_peptides` and `unimputed_matrix` exports
(see [Visualization](../periphery/visualization.md#single-matrix-qc-report)).

### Pure-Python method fallback

A method not reproducible bit-for-bit in the Rust kernel: the kernel's `features2proteins` errors point here (needs the `analysis` extra):

```python
# missforest — wraps scikit-learn's IterativeImputer
mokume.impute("proteins.csv", method="missforest", output="imputed.csv")
```

`mokume.impute` also reaches every other supported method (`knn`, `minprob`, `qrilc`, ...); it accepts a wide protein-matrix CSV path or a DataFrame and returns the imputed DataFrame, writing `output` if given.

### Runtime piBAQ protease catalog

```python
catalog = mokume.protease_catalog()
print([(entry["name"], entry["regex"]) for entry in catalog])
```

This reads the installed pyOpenMS `ProteaseDB` at call time. Both Rust-backed
piBAQ commands accept every entry in this catalog; Python supplies the complete
theoretical-peptide map and Rust performs the aggregation.

---

## Install extras { #install-extras }

pyOpenMS is a base dependency used by both piBAQ commands and itself depends on
numpy, pandas, and matplotlib. Install only the extra for any additional
periphery command you run:

```bash
pip install mokume                 # compute kernel + Python API
pip install "mokume[analysis]"     # + plots, reports, QC, comparison, and missforest
pip install "mokume[tissuemap]"    # + per-dataset tissue proteome analysis
pip install "mokume[plugin]"       # + local MCP service for the Mokume Plugin
pip install "mokume[all]"          # everything
```

| Wrapper | Extra | Third-party libraries |
|---------|-------|------------------------|
| `mokume.tsne_visualization` | `analysis` | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn |
| `mokume.peptides2protein_qc` | `analysis` | numpy, pandas, matplotlib, seaborn |
| `mokume.de_plots` | `analysis` | numpy, pandas, matplotlib, seaborn, scikit-learn |
| `mokume.interactive_report` | `analysis` | numpy, pandas, plotly |
| `mokume.tissuemap` | `tissuemap` | scanpy, anndata, umap-learn, combat, matplotlib, seaborn, pyarrow |
| `mokume.qc_report` / `mokume.workflow_comparison` | `analysis` | numpy, pandas, scipy, scikit-learn |
| `mokume.impute` | `analysis` | numpy, pandas, scipy, scikit-learn |
| Mokume Plugin MCP service | `plugin` | mcp, numpy, pandas, scipy, scikit-learn, statsmodels, PyYAML |

The exact dependency lists are declared in `pyproject.toml`. DirectLFQ and ComBat
are native Rust, while pyOpenMS-backed digestion is part of the base piBAQ path.
