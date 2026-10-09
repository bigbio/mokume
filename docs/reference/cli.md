# CLI Reference

The `mokume` console command is installed by the Rust-backed wheel. Native
quantification commands live under `mokume quantify`; optional Python periphery
workflows provide TissueMap, plots, and interactive reports. Most users start
with `mokume quantify features2proteins`.
Use `mokume --help` or `mokume help <COMMAND> [SUBCOMMAND]` for details.

In generated help, angle brackets name a value that follows an option; they are
not typed literally. For example, `--output <FILE>` expects one file path, while
`--de-contrast <GROUP_A> <GROUP_B>` expects two condition labels.

The compute flag surface below is single-sourced in Rust and shared with the
wheel's thin Python API. Periphery commands dispatch to modules packaged in the
same wheel and state their required extra in root help.

!!! note "One CLI, two implementation layers"
    `quantify features2proteins`, `quantify features2peptides`,
    `quantify peptides2protein`, and `correct-batches` run in the Rust kernel.
    `tissuemap`, `plot`, and `interactive-report` run in the wheel's Python periphery.
    Plotting and reporting consume kernel tables; TissueMap performs its
    documented downstream analysis from QPX data.

## features2proteins

The unified pipeline: features to protein quantification in one step.

```bash
mokume quantify features2proteins [OPTIONS]
```

### Input & Output

| Option | Default | Description |
|--------|---------|-------------|
| `-p/--parquet` | none | Input quantms.io/QPX feature parquet; pair it with `--psm` for `spectral-count` |
| `--msstats` | none | Input MSstats CSV; mutually exclusive with QPX inputs and requires `--sdrf` |
| `--psm` | none | PSM-level QPX parquet for `spectral-count`; requires matching `--parquet` and `--sdrf` |
| `-o/--output` | required | Output protein intensities CSV |

The output schema is fixed: the protein identifier column is `ProteinName`.
There is no alternate output-format switch.

Provide `--parquet`, `--msstats`, or the paired `--psm` and `--parquet` inputs.
The paired QPX inputs are used only by true `spectral-count` and require SDRF.
MSstats input requires
`ProteinName`, `PeptideSequence`, `Intensity`, `Charge` or `PrecursorCharge`,
and `Run` or `Reference`; isobaric data also requires `Channel`. Ratio
quantification requires PSM-level QPX evidence and does not accept MSstats
feature tables.

### Metadata & Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `-s/--sdrf` | none | SDRF file for sample metadata |
| `--min-aa` | 7 | Minimum amino acid length |
| `--min-unique` | 2 | Minimum unique peptides per protein/sample; piBAQ uses its method-specific value 0 and rejects this option |
| `--keep-contaminants` | off | Keep contaminants (a group is a contaminant only if all its accessions are); rows marked `is_decoy=true` are always removed |

### Quantification

| Option | Default | Description |
|--------|---------|-------------|
| `--quant-method` | `maxlfq` | Method: maxlfq, directlfq, pibaq, `top<N>`, sum, median, ratio, abd, intensity, peptide-count, spectral-count |
| `-f/--fasta` | none | FASTA file (required for piBAQ) |
| `-t/--threads` | auto | Shared Rust worker count for all methods, including DirectLFQ |
| `--directlfq-min-nonan` | 1 | Min non-NaN values for DirectLFQ |
| `--directlfq-num-samples-quadratic` | 50 | Maximum samples in DirectLFQ's quadratic global-alignment subset |
| `--pibaq-enzyme` | `Trypsin` | Protease name from the installed pyOpenMS catalog |
| `--pibaq-max-aa` | 30 | Maximum theoretical peptide length |
| `--pibaq-missed-cleavages` | 0 | Missed cleavages in the theoretical digest (peptide-to-protein mapping and, by default, the denominator) |
| `--pibaq-denominator-missed-cleavages` | `--pibaq-missed-cleavages` | Max missed cleavages of theoretical peptides counted in the denominator (Trypsin only) |
| `--pibaq-mc-mode` | `drop` | Observed peptides not in the digest: `drop`, or `parent` (add the intensity, split equally, to the fully cleaved digest peptides it contains, or to its Met-retaining N-terminal form; Trypsin only) |
| `--pibaq-min-shared` | 2 | Minimum shared peptides for automatic family discovery |
| `--pibaq-families` | none | YAML file with explicit family overrides |
| `--pibaq-min-anchors` | 1 | Minimum unique-peptide anchors (in the dataset) for a family to have member-level evidence |
| `--pibaq-shared` | `unique` | Shared-peptide handling: `unique` (member values from unique peptides only), `stable-ratio` (unique peptides plus shared signal split by cross-sample member ratios), `proportional` (legacy per-sample split) |
| `--pibaq-family-rows` | `family-only` | Family-level rows (`A;B;C`): `family-only` (families with no unique-peptide evidence), `all`, `none`; `none` with `proportional` |
| `--pibaq-evidence` | none | Write a per-row TSV side-car: row type, family, evidence class, theoretical/observed unique and shared peptide counts |

!!! note "Write the N in the method name: `--quant-method top<N>`"
    TopN quantification takes its N from the method name — `--quant-method top3`,
    `top5`, `top10`, and so on for any N ≥ 1.

### Normalization

| Option | Default | Description |
|--------|---------|-------------|
| `--run-normalization` | method-dependent | Run-level: median, mean, max, global, max-min, iqr, none |
| `--sample-normalization` | method-dependent | Sample-level: global-median, condition-median, hierarchical, quantile, median-center, mean-center, rlr, loess, tmm, none |
| `--normalization-proteins` | none | Protein-ID file for applicable intensity normalizers; unavailable for directlfq, ratio, peptide-count, and spectral-count |

DirectLFQ and Ratio manage normalization internally. `peptide-count` and
`spectral-count` count evidence identities and do not use intensity
normalization. All four therefore default both normalization layers to `none`;
passing a non-`none` value is rejected. Other methods default to `median` /
`global-median`.

### IRS (Multi-Plex TMT)

| Option | Default | Description |
|--------|---------|-------------|
| `--irs` | off | Enable IRS normalization |
| `--irs-reference-sample` | auto | Reference sample name; repeat for multiple samples |
| `--irs-sdrf-column` | auto | SDRF column for reference detection |
| `--irs-sdrf-value` | auto | Value indicating a reference sample; repeat for multiple values |
| `--irs-reference-regex` | `pool\|powder\|ref\|reference\|bridge` | Regex for reference auto-detection |
| `--irs-stat` | `median` | Plex reference statistic: median or mean |
| `--irs-remove-reference` | off | Remove reference samples from output |
| `--irs-plex-column` | auto | SDRF column naming each channel's plex (default: channels sharing data files) |

IRS options require `--irs` and an SDRF. Reference detection must find usable
reference samples and plex assignments; otherwise the command fails. IRS is
rejected for `peptide-count` and `spectral-count` because scaling count evidence
would no longer represent an integer evidence count.

### Matrix QC and Coverage Filter

| Option | Default | Description |
|--------|---------|-------------|
| `--min-sample-correlation` | none | Drop samples whose mean pairwise Pearson correlation to same-condition peers is below the threshold; uses pairwise-complete log2 protein intensities and requires SDRF metadata |
| `--coverage-threshold` | none | Min fraction of non-missing values per condition |

### Ratio Quantification

| Option | Default | Description |
|--------|---------|-------------|
| `--ratio-fraction-merge` | `mean` | Fraction merge strategy: mean or max |

### Batch Correction

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-correction` | off | Enable ComBat batch correction |
| `--batch-method` | `sample-prefix` | Detection method: sample-prefix or column |
| `--batch-column` | none | SDRF column used when `--batch-method=column` |
| `--batch-covariate` | none | SDRF column to preserve; repeat for multiple columns |
| `--batch-nonparametric` | off | Use non-parametric ComBat instead of the parametric default |
| `--batch-mean-only` | off | Only adjust batch means |
| `--batch-ref` | none | Original sample-prefix or SDRF-column batch label |

ComBat is a native Rust implementation, oracle-verified against inmoose
(parametric ~1e-6, covariate / non-parametric / `ref_batch` / `mean_only`
paths included). Numeric covariates keep their values, while nominal covariates
use k-1 one-hot indicators. It runs on proteins with no missing cells; the rest
are kept uncorrected. Invalid batch layouts and an empty complete-protein subset
are errors rather than successful uncorrected outputs.

### Imputation

| Option | Default | Description |
|--------|---------|-------------|
| `--impute-method` | none | Select a method and enable imputation; use `zero` for zero filling |
| `--impute-quantile` | 0.01 | Quantile for MinProb/MinDet |
| `--impute-shift` | 1.6 | MinProb shift in standard deviations |
| `--impute-scale` | 0.3 | MinProb scale factor for the imputation distribution sigma |
| `--impute-n-neighbors` | 5 | Number of neighbours for KNN/SeqKNN |

Imputation runs in log2 space (the matrix is transformed before imputation and back to linear afterwards) so that censored-aware methods like MinProb/MinDet/QRILC behave correctly. The imputation step is applied after coverage filtering and before batch correction in the pipeline.

`missforest` is intentionally absent from this Rust compute command because it
wraps scikit-learn. Install `mokume[analysis]` and call the wheel's Python
`mokume.impute` API when that method is required.

### Differential Expression

| Option | Default | Description |
|--------|---------|-------------|
| `--de-contrast` | — | Two condition labels; repeat for multiple contrasts |
| `--de-contrast-file` | — | TSV file with columns `group1`, `group2` |
| `--de-method` | `auto` | Method: auto, limrots, limma, deqms, proda, rots, ensemble |
| `--de-ensemble-method` | `limrots`, `deqms`, `proda` | Ensemble member; repeat to override the defaults |
| `--de-ensemble-min-k` | 2 | Minimum ensemble members that must agree on direction |
| `--de-log2fc` | 0.5 | Minimum absolute log2 fold change, or `auto` for the data-driven mixture gate |
| `--de-effect-size-gate` | none | Explicit data-driven gate: mixture or null-quantile; a numeric log2FC value is its fallback |
| `--de-fdr` | 0.05 | Maximum FDR threshold |
| `--de-fdr-method` | `bh` | FDR correction: bh, ihw, bky, or storey |
| `--de-output` | required for DE | DE results file; with multiple contrasts each is written as `<stem>_<A-B>.<ext>` |

Any DE option enables differential expression. At least one contrast and an
output path must be supplied, so a completed calculation cannot be discarded.
Inline and file contrasts can be combined.

`--de-method auto` selects `deqms` for `directlfq` quantification and `limrots`
for other quantification methods. All methods run in the native Rust kernel —
no R or rpy2 required. ROTS and LimROTS retain permutation FDR and therefore
reject alternative `--de-fdr-method` values. Deterministic methods (limma /
deqms) are cell-exact against frozen Python-generated compatibility output;
RNG/optimizer-driven methods (rots / limrots / proda) match log2 fold change
cell-exactly and p-values at rank level.

`--de-method ensemble` runs each member method on the same contrast and combines the per-protein verdicts via top-k consensus: a protein is called UP/DOWN only when at least `--de-ensemble-min-k` members agree on direction and the Fisher-combined p-value passes the FDR threshold. Eligible non-ROTS members use the requested correction; ROTS and LimROTS retain their native permutation FDR. The Fisher-combined p-values use BH by default, with BKY or Storey applied when requested and reliable; IHW remains a member-level correction because the combined rows have no IHW covariate.

### Plots & Reports

`features2proteins` is pure compute and writes no figures. The kernel emits the protein-matrix CSV and (with `--de-output`) one DE result CSV per contrast; render plots and HTML reports from those CSVs with the wheel periphery:

```bash
# PCA from the protein matrix (plotting extra)
mokume plot pca --protein-matrix proteins.csv \
    --sdrf experiment.sdrf.tsv --output pca.pdf

# DE volcano / heatmap from the kernel CSVs (plotting extra)
mokume plot de --protein-matrix proteins.csv --outdir plots \
    --sdrf experiment.sdrf.tsv --volcano --heatmap \
    --contrast c1 A B de.csv

# Interactive HTML report (reports extra)
mokume interactive-report --protein-matrix proteins.csv \
    --sdrf experiment.sdrf.tsv --output report.html \
    --contrast c1 A B de.csv
```

Pass `--help` to either command for the full flag set.

### Export

| Option | Default | Description |
|--------|---------|-------------|
| `--export-peptides` | none | Export normalized peptides (not supported with DirectLFQ) |
| `--export-ions` | none | Export normalized ions (DirectLFQ only) |

`--export-peptides` also rejects dataset-level sample normalization for
non-cell-based aggregation methods, because those peptide values would not
represent the normalized protein matrix.

### Runtime Resource Controls

| Option | Default | Description |
|--------|---------|-------------|
| `-t/--threads` | Rayon default | Size the Rayon thread pool used by parallel Rust sections |
| `--memory` | none | Cross-platform soft process resident-memory budget, such as `512MB` or `1GB`; reduces QPX batch size/read-ahead and fails when a checkpoint observes usage above the budget |

`--memory` is a planner and runtime guard, not an operating-system hard limit.
The Rust pipeline checks resident memory at startup, after decoded input
batches, and between major phases. It reads RSS on Unix-like systems and the
process Working Set on Windows. A batch may transiently cross the requested
value, and dataset-sized aggregation state is not spilled to disk; if that
state no longer fits, the command exits with an explicit error instead of
continuing to grow unchecked. The smaller synchronous batches may reduce
throughput. Use systemd/cgroup, SLURM, Windows Job Objects, or container limits
when the process must never exceed a hard ceiling.

The Rust path does not expose DuckDB-specific resource options because it does
not use DuckDB.
For piBAQ, runtime pyOpenMS FASTA digestion happens before dispatch to the Rust
pipeline and therefore lies outside this soft budget.

---

## features2peptides

Feature-level to peptide-level normalization.

```bash
mokume quantify features2peptides [OPTIONS]
```

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p/--parquet` | required | Input parquet file |
| `-s/--sdrf` | none | SDRF file for metadata |
| `-o/--output` | required | Output peptide intensity file |
| `--min-aa` | 7 | Minimum amino acid length |
| `--min-unique` | 2 | Minimum unique peptides per protein |
| `--keep-shared-peptides` | off | Keep shared/non-unique peptides and skip the unique-peptide gate |
| `--remove-ids` | none | File with protein IDs to exclude |
| `--remove-decoy-contaminants` | off | Remove decoys and contaminants |
| `--remove-low-frequency-peptides` | off | Remove peptides in <20% of samples |

### Normalization

| Option | Default | Description |
|--------|---------|-------------|
| `--run-normalization` | `median` | Feature normalization: median, mean, max, global, max-min, iqr, none |
| `--sample-normalization` | `global-median` | Sample normalization: global-median, condition-median, none |
| `--skip-normalization` | off | Skip all normalization |
| `--log2` | off | Log2 transform output |
| `--save-parquet` | off | Save output as parquet |

Dataset-level sample normalizers are not accepted by `features2peptides`; run
them through `features2proteins`, where the full matrix exists.

### TMT / ITRAQ

| Option | Default | Description |
|--------|---------|-------------|
| `--irs-channel` | none | Explicit pooled/reference channel label |
| `--irs-autodetect-regex` | none | Regex to detect pooled samples from SDRF |
| `--irs-stat` | `median` | IRS per-run statistic |
| `--irs-scope` | `global` | IRS scaling scope: global, by-mixture, or two-stage |
| `--aggregation-level` | `sample` | Aggregate at sample or run level |

!!! note "Channel-based IRS in `features2peptides`"
    Choose exactly one of `--irs-channel` and `--irs-autodetect-regex`.
    Autodetection requires an SDRF and must match a reference channel; a
    requested channel must produce scaling factors. `--skip-normalization`
    conflicts with IRS. SDRF-driven multi-plex IRS is also available in
    `features2proteins` (`--irs` with repeated `--irs-reference-sample` /
    `--irs-sdrf-column`).

### Filter Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--filter-config` | none | YAML/JSON filter configuration file |
| `--generate-filter-config` | none | Generate example config and exit |
| `--filter-min-intensity` | none | Min intensity threshold (override) |
| `--filter-cv-threshold` | none | Max CV across replicates (override) |
| `--filter-charge-state` | none | Charge state override; repeat for multiple states |
| `--filter-max-missed-cleavages` | none | Max missed cleavages (override) |
| `--filter-peptide-fdr` | none | Max QPX peptide q-value (override) |
| `--filter-score NAME=THRESHOLD` | none | Named QPX score threshold; direction comes from `higher_better` |
| `--filter-exclude-modification` | none | Modification override; repeat for multiple names |
| `--filter-protein-fdr` | none | Max QPX protein-group q-value (override) |
| `--filter-min-features` | none | Min features per run (override) |
| `--filter-max-missing-rate` | none | Max missing feature fraction per technical run (override) |

Only implemented filters appear in the generated example. FDR filtering is
disabled by default and requires populated `peptide_qvalue` or
`pg_global_qvalue` input. Missing rate uses the complete distinct
`(protein, peptide)` universe among the surviving technical runs in each sample.
Named-score filtering is applied before normalization and aggregation, matches
`additional_scores.score_name` exactly, and fails if the requested score is
missing or has inconsistent `higher_better` values.
Active unsupported filter-config values are rejected instead of being logged and
ignored.

---

## peptides2protein

Protein quantification from normalized peptide data.

```bash
mokume quantify peptides2protein [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p/--peptides` | required | Input peptide intensity file |
| `-f/--fasta` | none | FASTA file (required for piBAQ) |
| `--quant-method` | `pibaq` | Method: pibaq, `top<N>` (top3, top5, top10, ...), maxlfq, sum, directlfq |
| `--enzyme` | `Trypsin` | Enzyme for in-silico digestion |
| `--normalize` | off | Normalize quantification values |
| `--min-aa` | 7 | Min amino acid length |
| `--max-aa` | 30 | Max amino acid length |
| `--tpa` | off | Calculate TPA (piBAQ only) |
| `--ruler` | off | ProteomicRuler (piBAQ only) |
| `--ploidy` | 2 with `--ruler` | Positive ploidy number (ruler only) |
| `--organism` | `human` with `--ruler` | Organism for histone data (ruler only) |
| `--cpc` | 200 with `--ruler` | Positive cellular protein concentration in g/L (ruler only) |
| `-t/--threads` | auto | Positive worker count for MaxLFQ and DirectLFQ |
| `--directlfq-min-nonan` | 1 | Min non-NaN for DirectLFQ |
| `--families` | none | YAML file with explicit family overrides (piBAQ only; see [user guide](../user-guide/peptides2protein.md#family-discovery-tuning)) |
| `--min-shared` | 2 | Minimum shared peptides for auto-family discovery (piBAQ only) |
| `--min-anchors` | 1 | Anchor threshold; if no member reaches it, shared signal is split equally and evidence is `family_only` (piBAQ only) |
| `--high-anchor-threshold` | 3 | Anchors every member must reach for `EvidenceLevel=high` (piBAQ only) |
| `-o/--output` | required | Output file path |
| `--qc-report` | none | Generate the piBAQ QC PDF at this path after native quantification |

Use `--quant-method top5` or `--quant-method top10` for Top5 or Top10-style quantification; `top3` is the named method from [Silva et al. 2006](https://doi.org/10.1074/mcp.M500230-MCP200). `--output` is required for every method.

!!! note "piBAQ uses the installed pyOpenMS catalog"
    Both piBAQ commands query the installed pyOpenMS `ProteaseDB` at runtime and
    support its complete catalog. Python digests the FASTA and passes the full
    protein-to-theoretical-peptide map into Rust. At `debug` or `info` log level,
    the run log records the pyOpenMS version, canonical enzyme, catalog SHA-256,
    peptide-length bounds, and missed-cleavage count.

!!! note "`--qc-report` plots the native result table"
    Rust writes the piBAQ table first, then the wheel renders the density and box
    plots from those exact values. Install `mokume[plotting]` to use this option.

---

## correct-batches

Standalone batch correction for pre-quantified data.

```bash
mokume correct-batches [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-i/--input` | required | Directory with TSV files |
| `-p/--pattern` | `*pibaq.tsv` | File matching pattern |
| `-o/--output` | required | Output file path |
| `--sample-id-column` | `SampleID` | Sample ID column |
| `--protein-id-column` | `ProteinName` | Protein ID column |
| `--pibaq-raw-column` | `PiBAQ` | Raw intensity column |
| `--pibaq-corrected-column` | `PiBAQBec` | Corrected intensity column |
| `--comment` | `#` | Comment character |
| `--sep` | `\t` | Field separator |
| `--export-anndata` | off | Export to AnnData h5ad |

ComBat here is the native Rust implementation, oracle-verified against inmoose.
The combined protein × sample matrix must be complete and finite; structural
gaps, blanks, `NaN`, and infinities fail instead of being silently filled with
zero. Explicit numeric zero remains a valid observation. The `.h5ad` written by
`--export-anndata` is Rust-native and verified to round-trip through
`anndata.read_h5ad`. This command does not expose batch-method or covariate
options; for those, use `features2proteins --batch-correction`.

---

## Periphery commands

These commands ship in the `pip install mokume` wheel but remain implemented in
Python rather than the Rust compute kernel:

```bash
pip install "mokume[plotting]"
mokume plot tsne --input ./proteins --pattern proteins.tsv --output tsne.pdf

pip install "mokume[tissuemap]"
mokume tissuemap --input ./data --outdir ./out --threads 24
```

t-SNE visualization reads protein tables, while TissueMap derives a downstream
atlas from QPX data. See the [Python API](python-api.md) for the corresponding
in-process functions and the full periphery surface.
