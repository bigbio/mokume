# Examples

Worked, copy-pasteable examples for each kind of analysis mokume supports. Most
commands and snippets on these pages run against the small fixtures shipped in
the repository (`python/tests/example/`), so you can reproduce them without your
own data:

- `feature_wide.parquet` — a 500-feature QPX slice from PXD020192 (10 samples).
- `PXD020192.sdrf.tsv` — the matching SDRF metadata.
- `PXD017834-peptides.csv` — a normalized peptide table used for piBAQ.
- `Homo-sapiens-uniprot-reviewed-contaminants-decoy-202210.fasta` — a human
  UniProt FASTA used as the piBAQ digestion reference.

Each page uses the same three tabs you saw in the [Quick Start](../quickstart.md):

- **CLI** — the `mokume` console command installed by the wheel.
- **Python (wheel)** — the thin keyword wrappers exposed by `pip install mokume`
  (`mokume.features2proteins(...)`, `mokume.peptides2protein(...)`). These call
  the same Rust kernel in-process.
- **Python (package)** — the pure-Python `mokume-py` package (`pip install mokume-py`),
  which exposes the object-oriented `PipelineConfig` / `QpxDataset` API and
  `mokume.analysis`.

See [Rust Wheel](../rust-wheel.md) for the full picture of which surface does
what.

## Analysis pages

### [Quantification](quantification.md)

Turn features into a protein matrix with MaxLFQ, DirectLFQ, TopN, Sum, or Median.
Covers all three surfaces and how the quant method is selected.

### [Differential Expression](differential-expression.md)

Test proteins for abundance changes between conditions. Covers the kernel's
`--de-contrast` flags, the Python `DifferentialExpression` class, standalone `run_*`
functions, and the separate Mokume Plugin workflow.

### [Absolute Expression](absolute-expression.md)

Estimate per-protein copy numbers and concentrations with piBAQ, TPA, and the
ProteomicRuler. Explains every computed column
(`PiBAQ`, `PiBAQNorm`, `TPA`, `CopyNumber`, `EvidenceLevel`, ...).

### [Full Pipeline](pipeline.md)

Wire quantification, normalization, imputation, batch correction, IRS, and DE
into one `PipelineConfig` and run it through `run_pipeline` /
`QuantificationPipeline.run_dataset`. Shows the `QpxDataset` result API and how
to choose between the separate Python and Rust distributions.

### [CPTAC UCEC total proteome](cptac-ucec.md)

Run a full PDC000125 analysis on 4.13 million QPX feature rows: construct an
API-validated SDRF, quantify TMT reporter abundance, apply global-median and IRS
normalization with condition-wise coverage filtering, run limma, and reproduce
the three six-panel README composites.

### [PXD030304 cancer cell-line atlas](pxd030304-cell-lines.md)

Run native Rust DirectLFQ on 228.72 million DIA-NN QPX feature rows from 949
cell lines, then reproduce the README overview and tissue-biology composites
from the written matrix.
