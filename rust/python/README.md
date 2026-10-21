# mokume — Python wheel

`pip install mokume` gives you a Rust compute kernel with a Python periphery,
in the PyO3/maturin layout used by projects such as polars and pydantic-core
(Python imports a compiled Rust extension).

- The compute-heavy pipeline (expression matrix, normalization, imputation,
  differential expression, batch correction) lives in the Rust crates under
  `crates/` and is exposed in-process through the compiled `mokume._mokume`
  extension that maturin builds from `crates/mokume-py`.
- The periphery lives here in `mokume/commands/`. Plotting and reporting consume
  kernel tables, TissueMap derives downstream atlas outputs from QPX data, and
  explicit Python-only fallbacks handle operations the kernel does not provide.

The wheel also installs a `mokume` console command. Both that command and the
Python API run the compiled extension in-process; the periphery is reached only
through this wheel.

## Install

```bash
pip install mokume              # compute kernel + Python API
pip install "mokume[analysis]"  # + plots, reports, QC, comparison, and missforest
pip install "mokume[tissuemap]" # + per-dataset tissue proteome analysis
pip install "mokume[plugin]"    # + local MCP service
pip install "mokume[all]"       # everything
```

From a source checkout, build the extension with `maturin develop` (or
`maturin build`); the periphery is plain Python and needs no build step.

## Compute API

The compute commands and installed console command use the same clap parsing,
so flags stay single-sourced in Rust:

```python
import mokume

mokume.features2proteins(parquet="features.parquet", output="proteins.csv")
mokume.peptides2protein(quant_method="pibaq", peptides="peptides.parquet",
                        fasta="proteome.fasta", output="proteins.tsv")
catalog = mokume.protease_catalog()  # installed pyOpenMS ProteaseDB
# full control over the argument vector:
mokume.run(["correct-batches", "--input", "pibaq", "--output", "corrected.tsv"])
```

Each wrapper maps keyword arguments to CLI flags (`key=value` → `--key value`
with `_` rewritten to `-`; `key=True` → `--key`; a list repeats the flag; `None`
/ `False` are skipped).

For Python pipelines that already have a protein matrix, the wheel also exposes
the kernel stages directly. The input is a row-major linear-intensity list;
`None` represents a missing cell:

```python
values = [[100.0, 120.0, None, 240.0], [400.0, 420.0, 800.0, 820.0]]
normalized = mokume.normalize_matrix(
    values, "median", ["A1", "A2", "B1", "B2"], threads=24
)
imputed = mokume.impute_matrix(normalized, "mindet", threads=24)
rows = mokume.differential_expression(
    ["P1", "P2"], imputed, 2, 2, "limma", threads=24
)
```

## Periphery commands

Each lives in `mokume.commands.<name>` with a `main(argv)` entry point, is
runnable as `python -m mokume.commands.<name>`, and most have an ergonomic
wrapper on the top-level package:

```python
mokume.tsne_visualization(input="./proteins", pattern="proteins.tsv")
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

The plotting and reporting functions read the tables the kernel wrote without
recomputing them. pyOpenMS is a base dependency (and itself installs numpy,
pandas, and matplotlib): both piBAQ commands query its complete runtime protease
catalog and pass the theoretical-peptide map into Rust.
`impute(method="missforest")` remains an explicit Python fallback.

| Command                       | Extra        | Third-party libraries                                  |
| ----------------------------- | ------------ | ------------------------------------------------------ |
| `mokume.tsne_visualization`   | `analysis`   | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn |
| `mokume.peptides2protein_qc`  | `analysis`   | numpy, pandas, matplotlib, seaborn                     |
| `mokume.de_plots`             | `analysis`   | numpy, pandas, matplotlib, seaborn, scikit-learn       |
| `mokume.interactive_report`   | `analysis`   | numpy, pandas, plotly                                  |
| `mokume.tissuemap`            | `tissuemap`  | scanpy, anndata, umap-learn, combat, matplotlib, seaborn, pyarrow |
| `mokume.qc_report` / `mokume.workflow_comparison` | `analysis` | numpy, pandas, scipy, scikit-learn         |
| `mokume.impute`                                   | `analysis` | numpy, pandas, scipy, scikit-learn         |

The exact dependency lists are declared in `pyproject.toml`'s
`[project.optional-dependencies]`. `mokume.impute(method="missforest")` reaches the
pure-Python imputer the Rust kernel does not reproduce (it wraps a scikit-learn
estimator); the kernel's `features2proteins` errors point here.
