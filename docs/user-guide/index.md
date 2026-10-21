# User Guide

This section covers the practical usage of each mokume command. The
`mokume` wheel installs a `mokume` console command backed by the compiled
`mokume._mokume` extension and the wheel's Python periphery. Compute commands
run the leading Rust kernel in-process; plotting, reporting, and TissueMap
commands route to the periphery. The separately maintained pure-Python
distribution is documented under
[Python API (package)](../reference/python-api-package.md).

## Rust-native compute commands

The console command exposes exactly four compute subcommands. Each can also be
driven through a thin keyword wrapper (`mokume.<command>(**kwargs)`) or with an
explicit argument list (`mokume.run([...])`).

### [features2proteins: Unified Pipeline](features2proteins.md)

The recommended entry point. Takes raw feature data and produces a protein quantification matrix in one step, with optional normalization, batch correction, IRS, imputation, and differential expression.

### [features2peptides: Peptide Normalization](features2peptides.md)

Normalizes and filters feature-level data into peptide intensities. Use this when you need fine-grained control over the normalization step before quantification.

### [peptides2protein: Protein Quantification](peptides2protein.md)

Quantifies proteins from normalized peptide data. Supports piBAQ (with TPA and ProteomicRuler), TopN, MaxLFQ, DirectLFQ, and Sum.

### [correct-batches: Batch Correction](batch-correct.md)

Standalone batch correction for already-quantified protein data. Combines multiple files and applies native Rust ComBat (oracle-verified vs inmoose) correction.

## Python periphery commands

The same `mokume` console command exposes `tsne-visualization`, `tissuemap`,
`de-plots`, and `interactive-report`. These commands live in the Python
periphery of the wheel: plotting and reporting consume kernel tables, while
TissueMap derives its downstream atlas from QPX data. Install the relevant
extra (`analysis`, `tissuemap`, or `all`) to pull in their libraries.
The top-level Python wrappers remain available for scripts. Evidence-bound
method selection is available through the
[Mokume Plugin](agentic-plugin.md), which uses the default Rust-backed wheel
and the deterministic recommendation service.
