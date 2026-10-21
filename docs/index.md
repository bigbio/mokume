# mokume

[![Python application](https://github.com/bigbio/mokume/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/mokume.svg)](https://badge.fury.io/py/mokume)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mokume)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Rust-first proteomics quantification toolkit for the quantms ecosystem.**

The name comes from [mokume-gane](https://en.wikipedia.org/wiki/Mokume-gane) (木目金), a Japanese metalworking technique that fuses multiple metal layers into distinctive patterns — similar to how this toolkit melds peptide intensities into unified protein expression profiles.

The leading Rust implementation ships as a PyO3/maturin wheel (`pip install
mokume`). The wheel exposes the compiled `mokume._mokume` extension through
both a Python API and an installed `mokume` console command. The repository also
contains the separately maintained pure-Python `mokume-py` distribution, whose
class-based pipeline uses its own implementation. Select one distribution per
environment; they share the `mokume` import package and cannot be installed
together safely.

![The mokume quantify features2proteins pipeline: source data through quantify, normalize, impute, batch-correct, and differential expression, with the best-known methods at each stage](assets/pipeline.svg){ width="100%" }

---

<div class="grid cards" markdown>

-   :material-test-tube:{ .lg .middle } **Multiple Quantification Methods**

    ---

    piBAQ (paralog-aware iBAQ with exact shared-peptide allocation), TopN, MaxLFQ, DirectLFQ, Sum, Ratio — choose the right method for your experiment.

    [:octicons-arrow-right-24: Quantification methods](concepts/quantification.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Flexible Normalization**

    ---

    Feature-level, sample-level, and hierarchical normalization with a unified pipeline.

    [:octicons-arrow-right-24: Normalization](concepts/normalization.md)

-   :material-filter-variant:{ .lg .middle } **Batch Correction**

    ---

    Remove technical variation while preserving biological signal using native Rust ComBat (oracle-verified vs inmoose).

    [:octicons-arrow-right-24: Batch correction](concepts/batch-correction.md)

-   :material-flask-outline:{ .lg .middle } **IRS for Multi-plex TMT**

    ---

    Internal Reference Scaling with automatic reference detection from SDRF.

    [:octicons-arrow-right-24: IRS normalization](concepts/irs.md)

-   :material-cog-outline:{ .lg .middle } **Preprocessing Filters**

    ---

    Comprehensive QC filters configurable via YAML or CLI.

    [:octicons-arrow-right-24: Preprocessing](concepts/preprocessing.md)

-   :material-chart-scatter-plot-hexbin:{ .lg .middle } **Differential Expression**

    ---

    LimROTS, DEqMS, proDA, limma, and ROTS with BH, IHW, BKY, or Storey FDR correction — choose by discovery vs precision priority.

    [:octicons-arrow-right-24: Differential Expression](concepts/differential-expression.md)

-   :material-map-marker-path:{ .lg .middle } **Tissue Proteome Atlas**

    ---

    Build per-dataset tissue atlases with AdaTiSS tissue-specificity scoring, AnnData outputs, and atlas plots — a Python periphery command (`tissuemap` extra).

    [:octicons-arrow-right-24: TissueMap workflow](periphery/tissuemap.md)

-   :material-rocket-launch-outline:{ .lg .middle } **One-Step Pipeline**

    ---

    Go from feature parquet to protein intensities in a single command.

    [:octicons-arrow-right-24: Quick start](quickstart.md)

-   :material-robot-outline:{ .lg .middle } **Traceable Method Recommendation**

    ---

    Install the Mokume Plugin to inspect data and evaluate bounded candidate
    settings through the Rust kernel.

    [:octicons-arrow-right-24: Mokume Plugin](user-guide/agentic-plugin.md)

</div>

---

## Choose Your Workflow

- **Standard LFQ / TMT quantification** — start with [`features2proteins`](user-guide/features2proteins.md)
- **Need more control before protein summarization** — use the two-step path via [`features2peptides`](user-guide/features2peptides.md) and [`peptides2protein`](user-guide/peptides2protein.md)
- **Tissue atlas analysis** — use the [`tissuemap`](periphery/tissuemap.md) periphery command (wheel only, `tissuemap` extra)
- **Agent-host method recommendation** — use the [Mokume Plugin](user-guide/agentic-plugin.md)

## Quick Example

=== "CLI"

    ```bash
    # MaxLFQ quantification with normalization
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        -s experiment.sdrf.tsv \
        --quant-method maxlfq
    ```

=== "Python (wheel)"

    ```python
    import mokume

    # The wheel runs the same Rust kernel in-process (no subprocess) and
    # validates kwargs against the command's exact CLI schema.
    mokume.features2proteins(
        parquet="features.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="maxlfq",
    )
    ```

---

## Part of the quantms Ecosystem

mokume is a core component of the [quantms](https://quantms.org) proteomics analysis platform, providing the quantification engine that powers protein-level analysis, normalization, and tissue atlas workflows from mass spectrometry data.

| Ecosystem Tool | Purpose |
|---|---|
| [quantms](https://quantms.org) | Nextflow pipeline for quantitative proteomics |
| [qpx](https://qpx.quantms.org) | Data format and conversion tools |
| **mokume** | Protein quantification, normalization, and tissue atlas analysis |
