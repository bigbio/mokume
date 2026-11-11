# features2proteins: Unified Pipeline

The `features2proteins` command is the recommended way to go from raw feature data to protein quantification. It handles loading, filtering, normalization, quantification, batch correction, IRS, imputation, and differential expression in a single step. The compute is single-sourced in the Rust kernel; plotting and interactive reports are a separate wheel periphery step (see [Plots and Reports](#plots-and-reports)).

![The features2proteins pipeline stages: source data, quantify, normalize, impute, batch-correct, differential expression](../assets/pipeline.svg){ width="100%" }

## Basic Usage

=== "CLI"

    ```bash
    mokume quantify features2proteins \
        -p features.parquet \
        -o proteins.csv \
        -s experiment.sdrf.tsv \
        --quant-method maxlfq
    ```

=== "Python (wheel)"

    The wheel wrapper validates documented keyword arguments, maps them to the
    command's exact CLI flags, and runs the same kernel in-process:

    ```python
    import mokume

    mokume.features2proteins(
        parquet="features.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="maxlfq",
    )
    ```

=== "Python (explicit argv)"

    ```python
    import mokume

    mokume.run([
        "quantify",
        "features2proteins",
        "--parquet", "features.parquet",
        "--output", "proteins.csv",
        "--sdrf", "experiment.sdrf.tsv",
        "--quant-method", "maxlfq",
    ])
    ```

## Input Formats

Choose one input mode:

- `--parquet` accepts a quantms.io/QPX feature parquet file. It is also the
  required protein-group mapping paired with `--psm` for `spectral-count`.
  Intensities that a QPX 1.3 row lists in `imputed_labels` (imputed by the
  upstream tool, such as Spectronaut) are dropped before quantification.
- `--msstats` accepts a native MSstats CSV and requires `--sdrf` so runs and
  channels can be mapped to samples.
- `--psm` accepts a PSM-level QPX parquet for true `spectral-count`; it requires
  both the matching feature QPX via `--parquet` and sample metadata via `--sdrf`.

```bash
mokume quantify features2proteins \
    --msstats msstats.csv \
    --sdrf experiment.sdrf.tsv \
    --output proteins.csv \
    --quant-method maxlfq
```

An MSstats CSV must contain `ProteinName`, `PeptideSequence`, `Intensity`,
`Charge` or `PrecursorCharge`, and `Run` or `Reference`. If both `Run` and
`Reference` are present, `Reference` names the data file, because quantms
writes `Run` as a design index. Isobaric data also requires `Channel`.
`--quant-method ratio` requires PSM-level QPX evidence and therefore cannot
use an MSstats feature table.

### Runs and samples

With `--sdrf`, the SDRF `source name` column defines the samples. All runs
(`comment[data file]`) that carry the same source name are combined into one
sample, which is how technical replicates and fractions are meant to be
combined: their features are pooled before sample-level normalization and
protein quantification, so the output has one column for that sample. For
Median, Abd, TopN and peptide count, the highest intensity of each peptidoform
and charge across the sample's runs is kept and these values are summed into
the canonical peptide before the protein value is computed.

Sample-level normalization (such as the global median) and the cross-sample
methods (MaxLFQ, DirectLFQ) use every sample, so merging runs into one sample
also changes the values of the other samples. A duplicated source name in the
SDRF therefore shifts the whole matrix, not only the affected sample.

mokume logs each source name that maps to more than one data file. Runs that
differ in `comment[technical replicate]` or `comment[fraction identifier]` are
the normal replicate or fraction design and are logged at INFO level. If the runs
also have the same technical replicate and fraction identifier (identical or
missing values), mokume logs a WARNING, because this usually indicates a
duplicated source name rather than technical replicates. Give every distinct
sample its own source name. Neither message changes the output.

## Quantification Methods

| Method | CLI Flag | FASTA Required | Description |
|--------|----------|:--------------:|-------------|
| MaxLFQ | `--quant-method maxlfq` | No | Delayed normalization (default) |
| DirectLFQ | `--quant-method directlfq` | No | Hierarchical alignment (native Rust) |
| piBAQ | `--quant-method pibaq` | Yes | Absolute quantification |
| TopN | `--quant-method top3` / `top5` / any `top<N>` | No | Average of the N most intense peptides |
| Sum | `--quant-method sum` | No | Sum of all peptides |
| Median | `--quant-method median` | No | Median peptide intensity |
| Ratio | `--quant-method ratio` | No | Log2 sample/reference (TMT) |
| TMT Abundance | `--quant-method abd` | No | Median of log2 peptide intensities (TMT) |
| TMT Reporter Intensity | `--quant-method intensity` | No | Sum of raw reporter intensities (TMT) |
| Peptide Count | `--quant-method peptide-count` | No | Distinct canonical peptides from feature QPX |
| Spectral Count | `--quant-method spectral-count --psm PSM --parquet FEATURE` | No | Unique QPX `psm_id` values from feature-linked PSMs; requires SDRF |

In practice:

- Use `maxlfq` as the default starting point for standard LFQ workflows.
- Use `directlfq` when you explicitly want the native Rust DirectLFQ estimator
  to handle normalization and quantification together.
- Use `pibaq` when you need absolute-style quantification and have a FASTA
  file. The pipeline delegates to the same piBAQ algorithm as
  `peptides2protein` -- see
  [Quantification Methods → piBAQ](../concepts/quantification.md#pibaq-paralog-aware-ibaq)
  for family discovery and exact shared-peptide allocation. The wide-format
  pipeline output retains per-protein `PiBAQ` values but does not surface the
  `FamilyId` / `EvidenceLevel` metadata columns.
- Use `ratio` for TMT PS-style reference-based analysis.
- Use `peptide-count` for distinct peptide identifications from feature QPX;
  use `spectral-count` for unique feature-linked PSMs from matching PSM and
  feature QPX files. Both are integer evidence counts, so run/sample intensity
  normalization and IRS are rejected.
- Use `top<N>` for the classic Top3-style summary; `top3` is the method from
  Silva et al. 2006, and any other N works the same way (`top5`, `top10`, ...).

```bash
# piBAQ (requires FASTA)
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method pibaq --fasta proteome.fasta

# TopN — the N lives in the method name (top3, top5, top10, ...)
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method top5

# DirectLFQ (native Rust, no extra dependency)
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method directlfq --threads 24
```

## Memory & Performance for Large Studies

The Rust kernel reads QPX parquet data in Arrow record batches and accumulates
the compact peptide, protein, and sample structures needed by the selected
method. It does not load the input through DuckDB or build a pandas pivot.

Use `--threads` to size the Rayon thread pool used by parallel Rust sections:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method directlfq \
    --threads 24 \
    --memory 1GB
```

`--memory` sets a cross-platform soft process resident-memory budget. It reduces
the QPX Arrow batch size, disables read-ahead, and checks RSS on Unix-like
systems or the process Working Set on Windows after input batches and major
pipeline phases. If in-memory aggregation state cannot fit, Mokume exits with a
clear budget-exceeded error rather than silently ignoring the option. The guard
can observe a transient overshoot only at the next checkpoint, and smaller
synchronous batches may reduce throughput. Use an external cgroup, scheduler,
Windows Job Object, or container limit when a hard ceiling is required.

Runtime pyOpenMS FASTA digestion for piBAQ occurs before the Rust pipeline
starts, so it is not covered by `--memory`.

## Protein FDR Filter

QPX feature tables are not always filtered at the same protein FDR: DIA-NN
exports typically keep protein groups up to a 5% global q-value, while OpenMS
DDA exports are usually 1%. To put every experiment on the same footing before
quantification (for example when building atlases from many datasets), pass
`--filter-protein-fdr`:

```bash
mokume quantify features2proteins \
    -p features.parquet \
    -o proteins.csv \
    -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --filter-protein-fdr 0.01
```

A protein group is kept when its best (minimum) `pg_global_qvalue` across all
of its rows is `<=` the threshold; every row of a failing group, and every
group without a protein q-value, is dropped before the `--min-unique` gate.
The semantics match `features2peptides --filter-protein-fdr`. The filter is
off by default. The command fails with a clear error when the QPX input has
no populated `pg_global_qvalue` column, and it is rejected for MSstats input
and `spectral-count`. The log reports the protein groups before/after the
filter for every run and for the whole experiment.

## Transferred Features (Match Between Runs)

With match between runs, a DDA workflow quantifies a precursor in runs where
it was not identified by transferring the identification from another run.
QPX 1.3 records this per row in `identification_type`: only `transferred`
counts as a transfer, while `direct` and `secondary` (a co-fragmented
identification from the run's own MS/MS) do not. Rows without a type fall back
to `id_run_file_name`, which is null (or names another run) for transferred
features. Pass `--transferred-features drop` to quantify only features
identified in their own run:

```bash
mokume quantify features2proteins \
    -p features.parquet \
    -o proteins.csv \
    -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --transferred-features drop
```

The default is `keep`. Dropped features are removed before normalization and
quantification, and the log reports how many were dropped. The command fails
when the QPX input has neither an `identification_type` nor an
`id_run_file_name` column, when a row without a type lists several runs,
or when `drop` removes every feature. Check those columns if every feature
is classified as transferred. `drop` is rejected for MSstats input and
`spectral-count`; an explicit `keep` is accepted and has no effect.
quantms DIA-NN and isobaric exports set `id_run_file_name` to the feature's
own run, so `drop` keeps every feature there.

Keep the default unless you need identification-only quantities. On 39
label-free DDA spike-in datasets (18 studies, 2–54% transferred features),
`drop` removed a median 18% of the MaxLFQ proteins and lowered the Score A of
the leading `directlfq` and `maxlfq` configurations by about 0.02, also on
the proteins that `drop` still quantifies; `sum`, `top3` and `pibaq` were
about unchanged.

## Normalization Options

### Run-Level Normalization

Adjusts for intensity differences between MS runs within each sample.

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --run-normalization median  # median, mean, max, global, max-min, iqr, none
```

### Sample-Level Normalization

Adjusts for systematic differences across samples.

```bash
# Global median (default)
mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization global-median

# Hierarchical (DirectLFQ-style)
mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization hierarchical

# With specific normalization proteins
mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization global-median \
    --normalization-proteins housekeeping.txt

# Quantile / MedianCenter / MeanCenter / RLR / LOESS / TMM (dataset-level)
mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization quantile

mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization loess

mokume quantify features2proteins -p data.parquet -o out.csv \
    --sample-normalization tmm
```

`--normalization-proteins` is available for applicable intensity normalizers.
It is unavailable for DirectLFQ, Ratio, peptide-count, and spectral-count.

!!! note "Dataset-level normalizers"
    `quantile`, `median-center`, `mean-center`, `rlr`, `loess`, and `tmm` are
    dataset-level normalizers applied after peptide aggregation, all native in
    the Rust kernel. `tmm` (Trimmed Mean of M-values,
    `mokume.normalization.tmm.TMMNormalizer`) is robust to composition bias from
    highly abundant proteins.

    MaxLFQ accepts `quantile` and `hierarchical` as dataset-level methods,
    applied to its peptide species, and piBAQ accepts `quantile`; requesting
    RLR, LOESS, centering, or TMM with either method, or `hierarchical` with
    piBAQ, is rejected before input loading. For MaxLFQ + quantile or
    hierarchical, Mokume selects the built-in MaxLFQ path so the requested
    normalization is actually applied.

- `global-median` is the default and a good general-purpose starting point.
- `hierarchical` is useful when you want DirectLFQ-style normalization with a non-DirectLFQ quantification method.

## IRS Normalization (Multi-Plex TMT)

For TMT experiments with shared reference channels across plexes:

```bash
# Auto-detect references from SDRF
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method median \
    --irs --irs-remove-reference

# Explicit reference samples
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method median \
    --irs --irs-reference-sample "p1_11" \
    --irs-reference-sample "p2_11"

# Custom regex for reference detection
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --irs --irs-reference-regex "pool|bridge|control"
```

| IRS Option | Default | Description |
|------------|---------|-------------|
| `--irs` | off | Enable IRS normalization |
| `--irs-reference-sample` | auto | Reference sample name; repeat for multiple samples |
| `--irs-sdrf-column` | auto | SDRF column for reference detection |
| `--irs-sdrf-value` | auto | Value indicating reference samples; repeat for multiple values |
| `--irs-reference-regex` | `pool\|powder\|ref\|reference\|bridge` | Regex for auto-detection |
| `--irs-stat` | `median` | Statistic for plex reference: median or mean |
| `--irs-remove-reference` | off | Remove reference samples from output |
| `--irs-plex-column` | auto | SDRF column naming each channel's plex (default: channels sharing data files) |

Every IRS sub-option requires `--irs` and an SDRF. If reference detection finds
no usable sample/plex mapping or no finite scale, the command fails rather than
returning an unscaled matrix. IRS is not applicable to `peptide-count` or
`spectral-count` and is rejected for both methods.

## Sample Correlation QC

Use `--min-sample-correlation` to remove a sample when its mean Pearson
correlation to the other samples in the same SDRF condition falls below a
threshold:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --min-sample-correlation 0.8
```

The comparison is one-shot on the normalized protein matrix: positive finite
linear intensities are log2-transformed, while the already-log2 `abd` and
`ratio` outputs are used directly. Each pair uses its shared proteins, and all
sample scores are computed before any sample is removed. Conditions with fewer
than two samples and pairs with fewer than three usable proteins are rejected
because the requested correlation cannot be evaluated. Pooled/powder reference
conditions are retained but are not scored.

## Ratio Quantification (TMT PS Protocol)

For multi-plex TMT with per-plex reference division:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method ratio \
    --coverage-threshold 0.65 \
    --ratio-fraction-merge mean
```

!!! info
    Ratio quantification handles cross-plex normalization inherently via
    per-plex reference division. Combining it with `--irs` is rejected.

## Batch Correction

ComBat runs as a native Rust kernel (oracle-verified vs inmoose), so no extra dependency is needed.

=== "CLI"

    ```bash
    mokume quantify features2proteins \
        -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
        --quant-method maxlfq \
        --batch-correction \
        --batch-method sample-prefix \
        --batch-covariate "characteristics[sex]" \
        --batch-covariate "characteristics[organism part]"
    ```

=== "Python (wheel)"

    ```python
    import mokume

    mokume.features2proteins(
        parquet="features.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="maxlfq",
        batch_correction=True,
        batch_method="sample-prefix",
        batch_covariate=[
            "characteristics[sex]",
            "characteristics[organism part]",
        ],
    )
    ```

Only `sample-prefix` and `column` (with `--batch-column`) are exposed in this
protein-matrix flow. ComBat requires at least two batches with two samples each
and at least one protein observed in every sample; otherwise it fails instead
of returning unchanged values. PCA + HDBSCAN outlier removal is not ported.

## Differential Expression

Contrasts must be explicitly specified via repeated `--de-contrast GROUP_A GROUP_B`
options or `--de-contrast-file` (TSV). Both can be combined, and any DE option
enables differential expression without a separate switch.

=== "Inline contrasts"

    ```bash
    mokume quantify features2proteins \
        -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
        --quant-method maxlfq \
        --de-contrast "NASH" "HL" \
        --de-contrast "NASH" "Control" \
        --de-method deqms \
        --de-fdr-method ihw \
        --de-output de_results.csv
    ```

=== "Contrasts file"

    ```bash
    mokume quantify features2proteins \
        -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
        --quant-method maxlfq \
        --de-contrast-file contrasts.tsv \
        --de-method deqms \
        --de-fdr-method ihw \
        --de-output de_results.csv
    ```

    Where `contrasts.tsv` is a two-column TSV:

    ```
    group1    group2
    NASH      HL
    NASH      Control
    HL        Control
    ```

| DE Option | Default | Description |
|-----------|---------|-------------|
| `--de-contrast` | — | Two condition labels; repeat the option for multiple contrasts |
| `--de-contrast-file` | — | TSV file with columns `group1`, `group2` |
| `--de-method` | `auto` | Method: auto, limrots, limma, deqms, proda, rots, ensemble |
| `--de-covariate` | — | SDRF column added to the limma/DEqMS design; repeat for several columns |
| `--de-block` | — | SDRF column whose levels are random blocks (e.g. the subject) in the limma/DEqMS design |
| `--de-precision-weights` | off | Weight DEqMS by each cell's MaxLFQ/DirectLFQ quantification precision |
| `--de-ensemble-method` | `limrots`, `deqms`, `proda` | Ensemble member; repeat to override the defaults |
| `--de-ensemble-min-k` | 2 | Minimum ensemble members that must agree on direction (ensemble only) |
| `--de-log2fc` | 0.5 | Minimum absolute log2 fold change, or `auto` for the data-driven mixture gate |
| `--de-effect-size-gate` | — | Explicit data-driven gate: `mixture` or `null-quantile`; a numeric `--de-log2fc` becomes its fallback |
| `--de-fdr` | 0.05 | Maximum FDR threshold |
| `--de-fdr-method` | `bh` | FDR correction: bh, ihw, bky, or storey |
| `--de-output` | — | DE results file; with multiple contrasts each is written as `<stem>_<A-B>.<ext>` |

!!! warning "Contrasts are required"
    If DE options are supplied without `--de-contrast` or `--de-contrast-file`,
    the pipeline raises an error. Each inline contrast receives two distinct
    arguments, so hyphenated condition names are unambiguous.
    A DE output path is also required, so completed results cannot be discarded.

!!! tip
    `--de-method auto` chooses `deqms` for `directlfq`
    quantification and `limrots` for all others; LimROTS needs a
    complete matrix, so a protein matrix with missing values also
    gets `deqms`. All methods
    run in the native Rust kernel — no R or rpy2 required.
    BKY and Storey fall back to BH when their pi0 estimate is not reliable.
    ROTS and LimROTS retain their own permutation FDR, so alternative
    `--de-fdr-method` values are rejected for those methods. Ensemble applies
    the selected correction to the combined result. On a contrast with
    missing values, the default ensemble members run `limma` in place of
    `limrots`; an explicit `--de-ensemble-method limrots` member needs a
    complete (for example, imputed) matrix.
    See [Differential Expression
    concepts](../concepts/differential-expression.md) for a
    detailed comparison of methods.

### Covariates and paired designs

`--de-covariate` adds SDRF columns to a limma or DEqMS design
(`~0 + condition + covariates`). All samples with a condition are fitted
together, and each contrast compares two condition coefficients. Numeric
columns stay numeric; other columns are categorical. A paired design passes the
subject column:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --de-contrast "treated" "control" \
    --de-method limma \
    --de-covariate "characteristics[individual]" \
    --de-covariate "comment[batch]" \
    --de-output de_results.csv
```

A constant, missing, or confounded covariate is rejected. With covariates
`--de-method auto` selects `deqms`; ROTS, LimROTS, proDA, and ensemble reject
the option.

`--de-block` models a column's levels as random blocks instead of fixed effects
(limma's `duplicateCorrelation`), for example the subject of a repeated-measures
design or the biological replicate of technical replicates. The consensus
intra-block correlation is written to the log, and the same method rules apply:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --de-contrast "treated" "control" \
    --de-method limma \
    --de-block "characteristics[individual]" \
    --de-output de_results.csv
```

```bash
# Top-k consensus across multiple methods
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --de-contrast "NASH" "HL" \
    --de-method ensemble \
    --de-ensemble-method limrots \
    --de-ensemble-method deqms \
    --de-ensemble-method proda \
    --de-ensemble-min-k 2 \
    --de-output ensemble_de.csv
```

## Imputation

Proteomics data is sparse. The pipeline can fill missing values on the
protein matrix after coverage filtering and before batch correction.
Imputation runs in log2 space so that censored-aware methods (MinProb,
MinDet, QRILC) behave correctly.

```bash
# MinProb low-quantile draw (imputeLCMD)
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --impute-method minprob \
    --impute-quantile 0.01 --impute-tune-sigma 1.0

# KNN imputation
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --impute-method knn --impute-n-neighbors 5
```

| Imputation Option | Default | Description |
|-------------------|---------|-------------|
| `--impute-method` | none | Select a method and enable imputation; use `zero` for zero filling |
| `--impute-quantile` | 0.01 | Quantile for MinProb/MinDet |
| `--impute-tune-sigma` | 1.0 | MinProb/QRILC multiplier of the fitted SD |
| `--impute-seed` | 42 | Seed of the MinProb/QRILC draws |
| `--impute-n-neighbors` | 5 | Neighbours for KNN/SeqKNN |

Imputation tuning options are method-scoped: quantile applies to MinDet/MinProb,
tune sigma and seed to MinProb/QRILC, and neighbour count to KNN/SeqKNN.
Passing them to another method is rejected.

!!! warning "`missforest` is Python analysis periphery"
    `missforest` is not offered by the Rust compute command because it wraps
    scikit-learn. Install the Rust-backed wheel's `analysis` dependencies and
    call the Python API:

    ```python
    import mokume

    mokume.impute("proteins.csv", method="missforest", output="imputed.csv")
    ```

## Plots and Reports

Plotting and the interactive HTML report are **not** part of the `features2proteins`
CLI — there are no `--plot-*` / `--interactive-report` flags. The kernel writes the
protein matrix and, with `--de-output`, one DE result CSV per contrast; the Python
periphery then reads those CSVs and renders the figures. Install the `analysis`
extra:

```python
import mokume

# DE plots (volcano / heatmap) — explicit argv: the per-contrast
# --contrast KEY A B CSV flag repeats, which keyword arguments cannot express.
mokume.de_plots([
    "--protein-matrix", "proteins.csv",
    "--outdir", "plots",
    "--sdrf", "experiment.sdrf.tsv",
    "--volcano", "--heatmap",
    "--contrast", "NASH-HL", "NASH", "HL", "de_results.csv",
])

# Interactive HTML report from the same kernel CSVs (analysis extra).
mokume.interactive_report([
    "--protein-matrix", "proteins.csv",
    "--sdrf", "experiment.sdrf.tsv",
    "--output", "qc_report.html",
    "--contrast", "NASH-HL", "NASH", "HL", "de_results.csv",
])
```

The figures read the kernel's output tables, so the cells in the plots match the
cells in the kernel matrix.

## Exporting Intermediate Data

```bash
# Export normalized peptides from a non-DirectLFQ aggregation
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method sum \
    --export-peptides peptides.csv

# Export the normalized ion table used by DirectLFQ
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv \
    --quant-method directlfq \
    --export-ions ions.csv
```

MaxLFQ supports peptide export with its default hierarchical normalization and
with quantile normalization. The export sums the normalized modification/charge
species actually used by MaxLFQ into protein × canonical peptide × sample rows.
Python's MaxLFQ export retains the individual species instead.

The Rust CLI also provides two diagnostic exports:

```bash
# Matched peptide tables before and after sample normalization
mokume quantify features2proteins \
    -p features.parquet -s experiment.sdrf.tsv -o proteins.csv \
    --export-peptides peptides.csv \
    --export-unnormalized-peptides peptides.before-sample-normalization.csv

# Protein matrix immediately before imputation
mokume quantify features2proteins \
    -p features.parquet -s experiment.sdrf.tsv -o proteins.csv \
    --impute-method minprob --export-unimputed proteins.before-imputation.csv

# Quantification precision of every MaxLFQ cell
mokume quantify features2proteins \
    -p features.parquet -s experiment.sdrf.tsv -o proteins.csv \
    --quant-method maxlfq --export-precision proteins.precision.csv
```

`--export-unnormalized-peptides` retains feature filtering and run normalization,
including channel IRS scaling when enabled, but precedes both ingest-time sample
factors (such as global/condition median) and dataset-level sample normalization.
It is collected during the same parquet read. Its rows and columns match
`--export-peptides` exactly. Dataset-level exports use the four columns
`ProteinName,PeptideCanonical,SampleID,NormIntensity`; other configurations also
include `BioReplicate` and `Condition`.

`--export-unimputed` requires `--impute-method`. It uses the final matrix's CSV
format after protein-level IRS, sample correlation filtering and coverage
filtering, but before imputation and batch correction. Neither diagnostic export
changes quantification, differential expression, or any existing output.

`--export-precision` writes the precisions behind `--de-precision-weights` in
the final protein matrix's format, with the same proteins and samples. Cells
that MaxLFQ or DirectLFQ did not quantify, including imputed ones, are empty.
Pass the table to `mokume.differential_expression(..., precision=...)` to weight
DEqMS on a matrix outside the command. It does not change any other output.

DirectLFQ, Ratio and SpectralCount reject both peptide exports. `--export-ions`
is DirectLFQ-only. piBAQ peptide export with dataset-level sample normalization
remains unsupported.

## Full Example

A complete TMT multi-plex analysis (kernel compute; render plots separately with
the wheel periphery as shown in [Plots and Reports](#plots-and-reports)):

```bash
mokume quantify features2proteins \
    -p features.parquet \
    -o proteins.csv \
    -s experiment.sdrf.tsv \
    --quant-method median \
    --run-normalization median \
    --sample-normalization global-median \
    --min-unique 2 \
    --irs --irs-remove-reference \
    --batch-correction --batch-method sample-prefix \
    --de-contrast "NASH" "HL" --de-method deqms --de-fdr-method ihw \
    --de-output de_results.csv
```
