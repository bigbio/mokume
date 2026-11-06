# mokume

[![Rust](https://github.com/bigbio/mokume/actions/workflows/rust.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/rust.yml)
[![Wheels](https://github.com/bigbio/mokume/actions/workflows/wheels.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/wheels.yml)
[![PyPI version](https://badge.fury.io/py/mokume.svg)](https://badge.fury.io/py/mokume)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mokume)

**mokume** is a proteomics quantification toolkit: it turns peptide-level mass
spectrometry intensities into protein expression matrices, with built-in
normalization, imputation, batch correction, and differential expression. It is
designed for the [quantms](https://github.com/bigbio/quantms) ecosystem and works
through both Python and an installed console command.

It supports piBAQ, TopN, MaxLFQ, and DirectLFQ quantification and ships as the
`mokume` wheel.

## Why mokume?

*Mokume-gane* (木目金, "wood-grain metal") is a Japanese metalworking technique
that fuses layers of different metals into a single piece with a distinctive
flowing pattern. mokume does the same with proteomics data: it melds many noisy,
overlapping peptide intensities into one coherent protein expression profile.

mokume is an evolution of [ibaqpy](https://github.com/bigbio/ibaqpy), extended
well beyond the original iBAQ workflow to a broader range of quantification,
normalization, and
differential-expression methods, with the heavy lifting rewritten in Rust.

## Installation

The Python wheel is the recommended way to install mokume:

```bash
pip install mokume
```

pyOpenMS and its numpy, pandas, and matplotlib dependencies are installed with
the base wheel so piBAQ can use every protease in the runtime catalog. Optional
analysis, tissue-map, and Plugin stacks remain extras:

```bash
pip install "mokume[analysis]"   # plots, reports, QC, comparison, and missforest
pip install "mokume[tissuemap]"  # tissue-map + AnnData export
pip install "mokume[plugin]"     # local MCP service for the Mokume Plugin
pip install "mokume[all]"        # everything above
```

Mokume and the optional Plugin/MCP workflow require Python 3.10 or newer.

The wheel also installs the `mokume` console command. A conda/mamba environment
is provided in `environment.yaml`. See
[Installation](https://github.com/bigbio/mokume/blob/main/docs/installation.md) for details.

## Quick start

Run the full pipeline from a quantms feature table to a protein matrix:

```bash
mokume quantify features2proteins \
  --parquet features.parquet \
  --sdrf samples.sdrf.tsv \
  --quant-method maxlfq \
  --output proteins.csv
```

Add differential expression by passing one or more contrasts:

```bash
mokume quantify features2proteins \
  --parquet features.parquet \
  --sdrf samples.sdrf.tsv \
  --quant-method maxlfq \
  --de-contrast "Treatment" "Control" \
  --output proteins.csv --de-output de_results.csv
```

The same kernel runs in-process from Python. Each keyword argument maps to a CLI
flag, and the command writes its result to `--output`:

```python
import mokume

mokume.features2proteins(
    parquet="features.parquet",
    sdrf="samples.sdrf.tsv",
    quant_method="maxlfq",
    output="proteins.csv",
)
```

The Python API and console command run the identical compute kernel, so a result
produced through either interface is the same.

## Commands

| Command | What it does |
| --- | --- |
| `quantify features2proteins` | Full pipeline: feature table to a protein quantification matrix |
| `quantify features2peptides` | Aggregate features to peptide-level intensities |
| `quantify peptides2protein` | Roll peptide intensities up to protein quantities |
| `correct-batches` | Standalone ComBat batch correction (with AnnData export) |

## Quantification and methods

mokume covers the common protein quantification strategies and the analysis
steps around them. The full catalog of methods is documented in the
[user guide](https://github.com/bigbio/mokume/tree/main/docs/user-guide).

`features2proteins` runs these stages in order:

<p align="center">
  <img src="https://raw.githubusercontent.com/bigbio/mokume/main/docs/assets/pipeline.svg" alt="The mokume quantify features2proteins pipeline: source data through quantify, normalize, impute, batch-correct, and differential expression, with the best-known methods at each stage" width="100%">
</p>

- **Quantification:** `maxlfq`, `directlfq`, `pibaq`, `top<N>`, `sum`
  (also `median`, `ratio`, `abd`, `intensity`, `peptide-count`,
  `spectral-count`). `peptide-count` uses feature QPX; true `spectral-count`
  pairs PSM QPX (`--psm`) with its feature QPX (`--parquet`). piBAQ requires a
  FASTA; TopN, MaxLFQ, and Sum do not. Both count methods require run/sample
  normalization `none` and reject IRS.
- **Normalization:** run-level and sample-level options including `median`,
  `quantile`, `rlr`, and `loess`.
- **Imputation:** a focused set of imputers, from simple (`mindet`, `knn`) to
  model-based (`qrilc`, `impseq`).
- **Batch correction:** native ComBat (parametric, non-parametric, and
  covariate-aware) that removes technical batch effects while preserving
  biological signal.
- **Differential expression:** `limma`, `deqms`, `rots`, `limrots`,
  `proda`, and an ensemble, with BH (default), IHW, BKY, or Storey FDR control
  and fixed or data-driven effect-size gates. LimROTS gives
  the best sensitivity on MaxLFQ data, DEqMS controls false positives better on
  noisier DirectLFQ data, and proDA models missing values as informative
  dropouts.

## How it works

mokume's compute lives once in a Rust kernel shipped in the `mokume` wheel.
The compiled extension runs in-process, and the installed `mokume` console
command reaches that extension through Python without a subprocess.

A thin Python periphery adds plots and reports on top. You never need to know it
is Rust to use it. The repository also contains the separately maintained
pure-Python `mokume-py` distribution. The Rust kernel leads new native
computation work. For the full design and maintenance policy, see
[Architecture](https://github.com/bigbio/mokume/blob/main/docs/architecture.md) and
[Maintenance scope](https://github.com/bigbio/mokume/blob/main/docs/maintenance-scope.md).

The installable Mokume Plugin calls the deterministic profile, policy, and
evaluation services, and uses the model and credentials owned by Codex or
Claude Code. See the [Mokume Plugin guide](https://github.com/bigbio/mokume/blob/main/docs/user-guide/agentic-plugin.md).

## Example

A full run on real data: **PXD030304**, 228.72 million DIA-NN QPX feature rows
from 5,798 label-free runs representing 949 cancer cell lines. Native Rust
DirectLFQ writes the 9,530 × 949 protein matrix; Mokume's Python periphery reads
that result to embed the samples, inspect detection depth, score AdaTiSS tissue
specificity, and find tissue markers.

<p align="center">
  <img src="https://raw.githubusercontent.com/bigbio/mokume/main/docs/assets/pxd030304_rust_overview.png" alt="Six-panel PXD030304 Rust DirectLFQ overview with PCA, t-SNE, tissue representation, detection depth, and variance panels" width="100%">
</p>

*All 949 cell lines: PCA and t-SNE by annotated tissue, tissue representation,
technical-run depth versus protein detection, detection across major tissues,
and the PCA variance profile. PC1 explains 19.5% and PC2 6.8% of the variance.*

<p align="center">
  <img src="https://raw.githubusercontent.com/bigbio/mokume/main/docs/assets/pxd030304_rust_biology.png" alt="Six-panel PXD030304 tissue-specificity and marker showcase" width="100%">
</p>

*Biological panels use the 815 cell lines in the 34 tissues with at least five
samples: Wilcoxon marker profiles, AdaTiSS scores, tissue-specific protein
counts, and three marker-expression maps. The complete Rust command, renderer,
result interpretation, and numerical comparison with the previous kernel are in
[`docs/examples/pxd030304-cell-lines.md`](https://github.com/bigbio/mokume/blob/main/docs/examples/pxd030304-cell-lines.md).*

## Documentation

- [Documentation home](https://github.com/bigbio/mokume/blob/main/docs/index.md)
- [Quick start](https://github.com/bigbio/mokume/blob/main/docs/quickstart.md)
- [Installation](https://github.com/bigbio/mokume/blob/main/docs/installation.md)
- [Rust wheel](https://github.com/bigbio/mokume/blob/main/docs/rust-wheel.md)
- [Architecture](https://github.com/bigbio/mokume/blob/main/docs/architecture.md)
- [Maintenance scope](https://github.com/bigbio/mokume/blob/main/docs/maintenance-scope.md)
- [Benchmarks](https://github.com/bigbio/mokume/tree/main/benchmarks)

## Citation

mokume is part of the [quantms](https://github.com/bigbio/quantms) ecosystem and
evolves [ibaqpy](https://github.com/bigbio/ibaqpy). Until a dedicated mokume
paper is available, please cite ibaqpy and quantms — see those repositories for
their current citation details and DOIs.

## Credits, contributing, and license

mokume is developed by the [bigbio](https://github.com/bigbio) community as part
of the quantms ecosystem. Contributions are welcome; see
[the community guide](https://github.com/bigbio/mokume/blob/main/docs/community.md) for development setup and guidelines.

Licensed under the [MIT License](https://github.com/bigbio/mokume/blob/main/LICENSE).
