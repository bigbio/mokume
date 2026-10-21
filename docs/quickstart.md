# Quick Start

This guide shows how to go from raw feature data to protein intensities using mokume.

![The mokume quantify features2proteins pipeline: source data through quantify, normalize, impute, batch-correct, and differential expression](assets/pipeline.svg){ width="100%" }

## Prerequisites

You need:

1. A **parquet file** in quantms.io/qpx format (output from quantms pipeline)
2. Optionally, an **SDRF file** for sample metadata

For most workflows, `pip install mokume` is enough. The wheel runs the Rust
compute kernel in-process and installs the `mokume` console command. If you want
the TissueMap periphery command, install `mokume[tissuemap]` first.

For evidence-bound method recommendation, install `mokume[plugin]` and the
[Mokume Plugin](user-guide/agentic-plugin.md). Do not configure a second MCP
entry or put a model API key in Mokume.

## One-Step Pipeline (Recommended)

The `features2proteins` command handles everything: loading, filtering, normalization, and quantification.

=== "CLI"

    ```bash
    # MaxLFQ quantification (default)
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        -s experiment.sdrf.tsv

    # With TMT IRS normalization + differential expression
    # (the kernel writes one DE result CSV per contrast via --de-output)
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        -s experiment.sdrf.tsv \
        --quant-method median \
        --irs --irs-remove-reference \
        --de-contrast "NASH" "HL" \
        --de-output de_results.csv

    # DirectLFQ (native Rust)
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        --quant-method directlfq

    # piBAQ (requires FASTA)
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        --quant-method pibaq \
        --fasta proteome.fasta
    ```

=== "Python (wheel)"

    ```python
    import mokume

    # The wheel runs the same Rust kernel in-process (no subprocess) and
    # validates kwargs against the command's exact CLI schema.

    # Simple MaxLFQ
    mokume.features2proteins(
        parquet="features.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="maxlfq",
    )

    # TMT with IRS + DE
    mokume.features2proteins(
        parquet="features.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="median",
        irs=True,
        irs_remove_reference=True,
        de_contrast=[("NASH", "HL")],
        de_output="de_results.csv",
    )
    ```

=== "Python (package)"

    The pure-Python `mokume-py` package (`pip install mokume-py` or
    `pip install ./python`) exposes a
    class-based API. Build a `PipelineConfig`, run it, and read the protein
    matrix off the returned `QpxDataset`:

    ```python
    from mokume.pipeline.config import PipelineConfig, InputConfig, QuantificationConfig
    from mokume.pipeline.runner import run_pipeline

    config = PipelineConfig(
        input=InputConfig(
            parquet="features.parquet",
            sdrf="experiment.sdrf.tsv",
        ),
        quantification=QuantificationConfig(method="maxlfq"),
    )
    dataset = run_pipeline(config)                    # QpxDataset with .proteins populated
    protein_matrix = dataset.get_level("proteins")   # protein x sample DataFrame
    ```

    See [Python API (package)](reference/python-api-package.md) for the full
    OOP surface (`QpxDataset`, runtime resource controls, the plugin registry).

!!! note "Plots and reports are periphery commands"

    The compute kernel writes tables: the protein matrix and, with `--de-output`,
    the DE result CSVs. Render figures from those tables with the Python
    periphery: `mokume plot de` for volcano/heatmap, `mokume plot pca` for PCA,
    and `mokume interactive-report` for the HTML report. These commands need the
    `analysis` extra.

## Two-Step Pipeline

For more control, use the peptide normalization step separately:

```bash
# Step 1: Normalize peptides
mokume quantify features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --run-normalization median \
    --sample-normalization global-median \
    --output peptides.csv

# Step 2: Quantify proteins
mokume quantify peptides2protein \
    --quant-method maxlfq \
    -p peptides.csv \
    -o proteins.tsv
```

## Tissue Atlas Workflow

Use `tissuemap` when your goal is tissue atlas analysis rather than standard
protein quantification. It is a wheel CLI command backed by the Python
periphery, rather than a Rust-native compute command:

```bash
# Install the optional dependencies first: pip install "mokume[tissuemap]"
mokume tissuemap \
    --input QPX_data/tissues-mq/PXD016999 \
    --outdir ./tissuemap_results
```

This workflow generates batch-corrected AnnData outputs, tissue-specificity scores, and atlas-style plots.

## What's Next?

- [Quantification Methods](concepts/quantification.md) — understand piBAQ, MaxLFQ, TopN, and more
- [Normalization](concepts/normalization.md) — learn about the normalization pipeline
- [Unified Pipeline](user-guide/features2proteins.md) — full reference for features2proteins
- [Tissue Proteome Atlas](periphery/tissuemap.md) — run the per-dataset TissueMap periphery command
