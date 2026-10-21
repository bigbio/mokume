# Mokume Benchmarks

Benchmarking studies for mokume protein quantification methods.

## Available Benchmarks

- [quant-hela-method-comparison](./quant-hela-method-comparison/) compares six
  current Rust quantification methods across 20 human datasets and the paired
  PXD007683 TMT/LFQ inputs. It reports metric-specific behavior without a
  ground-truth winner.
- [batch-quartet-multilab](./batch-quartet-multilab/) evaluates four current
  Rust quantification methods and native Rust ComBat on the 72-sample Quartet
  design. Batch diagnostics use a matched 53-protein universe; coverage is
  reported separately.
- [quant-pxd007683-tmt-vs-lfq](./quant-pxd007683-tmt-vs-lfq/) is the PXD007683
  spike-in benchmark. Its current refresh covers the LFQ arm: DirectLFQ and
  MaxLFQ reach about 6.8% median CV on the common universe, while piBAQ has the
  widest coverage.

## Quick Summary

### quant-hela-method-comparison

Evaluates piBAQ, MaxLFQ, DirectLFQ, Top3, Top10, and Sum across 20 public human
datasets. The paired PXD007683 TMT/LFQ check uses 5,312 common proteins. CV is
descriptive sample dispersion, not a technical-replicate ranking.

### batch-quartet-multilab

Benchmarks piBAQ, MaxLFQ, DirectLFQ, Top3, and native Rust ComBat across four
laboratories, six acquisition/lab batches, and 72 samples. ComBat improves all
four matched-universe diagnostics; no single method is ranked across coverage
and batch metrics.

### quant-pxd007683-tmt-vs-lfq

Recomputes the 11-sample PXD007683 LFQ arm with the current Rust kernel.

- **Methods:** piBAQ, MaxLFQ, DirectLFQ, Sum, Top3, Top5, and Top10
- **Scope:** LFQ was refreshed; the checked-in TMT and cross-technology assets were not regenerated in that run
- **Interpretation:** CV, coverage, and spike-in recovery are reported separately; no global winner is inferred from CV alone

## Running Benchmarks

```bash
pip install "mokume[analysis]"      # Rust kernel + plotting dependencies
cd benchmarks/<benchmark-name>
# Follow the input paths and entry point documented in that benchmark's README.
```

Quantification calls go through the Rust kernel via `mokume.peptides2protein()` /
`mokume.features2proteins()` (file-based, in-process through PyO3). Normalization,
imputation, and filter benchmarks that test individual components use numpy / scipy
/ scikit-learn directly (the Rust kernel bundles these internally but does not
expose them as standalone Python APIs).

## Data Policy

Large data files (>50MB) are **NOT** committed to git. Each benchmark README documents data sources and download scripts.

## Benchmark Structure

```
benchmarks/<benchmark-name>/
├── README.md           # Overview, results, methodology (expandable)
├── scripts/            # Benchmark scripts
├── data/               # GIT-IGNORED: large data files
├── results/            # Versioned metric tables; each benchmark documents provenance
└── figures/            # PNG visualizations
```
