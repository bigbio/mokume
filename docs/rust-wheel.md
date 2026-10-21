# Rust Wheel

`mokume` is the Rust distribution of mokume. It packages the compiled
`mokume._mokume` extension together with a Python API, a `mokume` console
command, and the Python periphery for plots, tissue maps, reports, and selected
fallbacks. pyOpenMS is a base dependency that supplies runtime FASTA digestion
for piBAQ. There is no separately distributed Rust binary.

## Three interfaces, one installed wheel

All compute interfaces execute the same Rust kernel in-process. The console
command starts Python and imports the extension; it does not launch another
executable.

=== "Console command"

    ```bash
    mokume quantify features2proteins \
      --parquet features.parquet \
      --sdrf experiment.sdrf.tsv \
      --output proteins.csv \
      --quant-method maxlfq \
      --threads 24
    ```

=== "Python kwargs"

    ```python
    import mokume

    mokume.features2proteins(
        parquet="features.parquet",
        sdrf="experiment.sdrf.tsv",
        output="proteins.csv",
        quant_method="maxlfq",
        threads=24,
    )
    ```

=== "Explicit argv"

    ```python
    import mokume

    mokume.run([
        "quantify",
        "features2proteins",
        "--parquet", "features.parquet",
        "--sdrf", "experiment.sdrf.tsv",
        "--output", "proteins.csv",
        "--quant-method", "maxlfq",
        "--threads", "24",
    ])
    ```

## Choosing an interface

| Your need | Interface |
| --- | --- |
| Shell pipelines and CI | `mokume` console command |
| Python scripting | `mokume.features2proteins(...)` or another keyword wrapper |
| Flags not expressible as keyword arguments | `mokume.run([...])` |
| Plots | `mokume plot tsne`, `mokume plot pca`, or `mokume plot de` |
| Tissue proteome maps | `mokume tissuemap` |
| Interactive HTML DE report | `mokume interactive-report` |
| piBAQ QC report PDF | `mokume.peptides2protein_qc` |
| `missforest` imputation | `mokume.impute(method=...)` |
| piBAQ with any installed pyOpenMS protease | `mokume.peptides2protein(...)` or `mokume.features2proteins(...)` |
| Agent-host method recommendation | Mokume Plugin + local MCP service |

The unified wheel CLI groups three Rust-native quantification commands under
`mokume quantify`, alongside standalone batch correction and the
optional Python periphery commands. The latter are still part of the installed
`mokume` command, but they do not move plotting or TissueMap computation into
Rust. The hidden `mokume mcp serve` service belongs to the `plugin` extra and
is started by the installed Mokume Plugin; users do not configure or launch it
manually.

## The kwargs &rarr; flags rule

The compute wrappers validate keyword arguments against a per-command schema
and then translate them into command flags:

| Python keyword argument | Command flag |
| --- | --- |
| `key="value"` | `--key value` (`_` becomes `-`) |
| `key=True` | `--key` |
| a repeatable `key=[a, b]` | `--key a --key b` |
| a paired repeatable `key=[(a, b)]` | `--key a b` |
| `key=None` or `key=False` | skipped |

For example, `quant_method="maxlfq"` becomes `--quant-method maxlfq` and
`de_contrast=[("A", "B"), ("C", "D")]` becomes repeated
`--de-contrast A B` flags. The wrapper accepts only canonical names; unknown
keywords and list values on scalar-only fields raise
`TypeError` before kernel dispatch.

## Capability locations

| Capability | Implementation | Public entry point |
| --- | --- | --- |
| `features2proteins` | Rust kernel | `mokume.features2proteins` |
| `features2peptides` | Rust kernel | `mokume.features2peptides` |
| `peptides2protein` | Rust kernel | `mokume.peptides2protein` |
| `correct-batches` | Rust kernel | `mokume.correct_batches` |
| Matrix normalization | Rust kernel | `mokume.normalize_matrix` |
| Matrix imputation | Rust kernel | `mokume.impute_matrix` |
| Matrix differential expression | Rust kernel | `mokume.differential_expression` |
| t-SNE, PCA, and DE plots | Python periphery | `mokume plot tsne` / `mokume plot pca` / `mokume plot de` |
| piBAQ-QC plots | Python periphery | `peptides2protein --qc-report` |
| Tissue proteome maps | Python periphery | `mokume tissuemap` |
| Interactive HTML DE report | Python periphery | `mokume interactive-report` |
| QC and workflow comparison | Python periphery | report wrappers |
| `missforest` imputation | Python fallback | `mokume.impute` |
| piBAQ FASTA digestion | pyOpenMS runtime catalog feeding the Rust kernel | `mokume.peptides2protein` / `mokume.features2proteins` |

DirectLFQ and ComBat are native Rust. pyOpenMS and its numpy, pandas, and
matplotlib dependencies are installed with the base wheel; install only the
periphery extras needed for additional presentation or fallback functionality.
