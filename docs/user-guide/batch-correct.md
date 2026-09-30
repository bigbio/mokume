# correct-batches: Batch Correction

The `correct-batches` command applies native Rust ComBat batch correction (oracle-verified vs inmoose) to already-quantified protein data. It reads multiple TSV files from a folder, combines them, and removes batch effects. Because ComBat is native in the kernel, no extra dependency is needed.

!!! tip "Prefer the integrated pipeline"
    For most use cases, batch correction is easier to apply via `features2proteins --batch-correction`. Use this standalone command when you have pre-existing protein quantification files that need correction.

This page documents the `mokume correct-batches` CLI subcommand.

The combined long table must contain one finite piBAQ value for every
protein × sample cell. Structural gaps, blank values, `NaN`, and infinities are
rejected with examples of the affected cells; Mokume never turns them into
zero silently. Impute or otherwise resolve missing values explicitly before
running this command. An explicit numeric zero remains a valid observed value.

## Basic Usage

=== "CLI"

    ```bash
    mokume correct-batches \
        -i pibaq_folder/ \
        -p "*pibaq.tsv" \
        -o corrected_pibaq.tsv
    ```

=== "Python (wheel)"

    The wheel wrapper validates documented keyword arguments, maps them to the
    command's exact CLI flags, and runs the same kernel in-process:

    ```python
    import mokume

    mokume.correct_batches(
        input="pibaq_folder/",
        pattern="*pibaq.tsv",
        output="corrected_pibaq.tsv",
    )
    ```

=== "Python (explicit argv)"

    ```python
    import mokume

    mokume.run([
        "correct-batches",
        "--input", "pibaq_folder/",
        "--pattern", "*pibaq.tsv",
        "--output", "corrected_pibaq.tsv",
    ])
    ```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `combat` | `combat` (this page) or `bridle` (see [BRIDLE](#bridle-multi-dataset-collections)) |
| `-i/--input` | required | Folder containing TSV files (`bridle`: one long-format file) |
| `-p/--pattern` | `*pibaq.tsv` | File matching pattern |
| `-o/--output` | required | Output file path |
| `--sample-id-column` | `SampleID` | Sample ID column name |
| `--protein-id-column` | `ProteinName` | Protein ID column name |
| `--pibaq-raw-column` | `PiBAQ` | Raw intensity column |
| `--pibaq-corrected-column` | `PiBAQBec` | Corrected intensity column |
| `--comment` | `#` | Comment character in files |
| `--sep` | `\t` | Field separator |
| `--export-anndata` | off | Export to AnnData h5ad format |

## With Covariates

To preserve biological signal during batch correction, supply covariates. The
standalone `correct-batches` command runs ComBat on the combined piBAQ folder and
does **not** expose batch-method or covariate options. Covariate-aware correction
is driven from the [`features2proteins`](features2proteins.md#batch-correction)
flow, which extracts the covariates from the SDRF:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --batch-correction \
    --batch-method sample-prefix \
    --batch-covariate "characteristics[sex]" \
    --batch-covariate "characteristics[tissue]"
```

```python
import mokume

mokume.features2proteins(
    parquet="features.parquet",
    output="proteins.csv",
    sdrf="experiment.sdrf.tsv",
    quant_method="maxlfq",
    batch_correction=True,
    batch_method="sample-prefix",
    batch_covariate=["characteristics[sex]", "characteristics[tissue]"],
)
```

!!! warning
    Without covariates, batch correction may remove biological signal that correlates with batches. See [Batch Correction concepts](../concepts/batch-correction.md) for details.

## AnnData Export

Export corrected data to AnnData format for downstream analysis with scanpy or other single-cell/proteomics tools:

```bash
    mokume correct-batches \
    -i pibaq_folder/ \
    -p "*pibaq.tsv" \
    -o corrected_pibaq.tsv \
    --export-anndata
```

This creates a `.h5ad` file alongside the TSV output.

## BRIDLE: multi-dataset collections

BRIDLE (**B**atch **R**emoval via **I**ntrinsic **D**etectability and
**L**atent **E**stimation) integrates a *collection* of datasets that measure
overlapping biological units onto one scale. A unit measured by >= 2 datasets
is an *anchor sample*: a cell line, a reference material, a pooled QC, or the
same patient across cohorts. Each dataset's offset is predicted from intrinsic
protein detectability features and refined on its anchor samples, while the
shared biology is a latent low-rank estimate. Unlike ComBat it does not need a
complete matrix: missing values stay missing, nothing is imputed and no protein
is dropped. It is the Rust port of the `lim_lin` prototype that won the Cell
Line Collection integration benchmark. `--method lim` is still accepted as a
deprecated alias of `--method bridle` (it logs a warning).

```text
y[i,g]     = theta[anchor,g] + A[dataset,g] + c[i] + P[plex,g] + eps
theta[l,g] = m_g + Lin[lineage(l),g] + U_l . V_g + R[l,g]     (shared biology, rank --rank)
A[s,g]     = f(s, x_g) + r[s,g]                               (A[--reference] = 0)
```

* `f` is a per-dataset ridge regression on technical protein features `x_g`
  (reference-abundance spline plus sequence features from `--fasta`: length,
  tryptic peptides, uniqueness, GRAVY, pI, charge, missed-cleavage context,
  amino-acid composition). No functional annotation is used.
* `r` is an empirical-Bayes gene-specific offset learned from *anchor samples*
  (units measured in >= 2 datasets). Datasets with 1 to 19 anchor samples are
  cross-fitted, so a single-sample dataset keeps its own signal. A dataset that
  shares no anchor sample with the rest gets only the intrinsic-detectability
  correction `f` (a warning is logged).
* `P` are TMT plex effects, `c` a per-profile loading; noise variance is
  modelled per dataset as a function of abundance.
* The output is `v = y - A - c - P` for every observed input cell
  (`imputed = false`). `--theta-output` writes the pooled per-anchor biology.

```bash
mokume correct-batches --method bridle \
    -i raw_long.parquet -o values.parquet \
    --reference ProCan \
    --fasta Homo-sapiens-uniprot-reviewed-contaminants.fasta --fasta-organism HUMAN \
    --lineage-table DepMap/Model.csv \
    --plex-column plex --plex-table ccle_plexes.tsv \
    --theta-output integrated.parquet --report fit_info.json
```

Input: `.parquet`, `.tsv` or `.csv` with one row per observed (dataset, anchor,
gene) cell and log2 values (collapse replicates per dataset and anchor first).

**Plexes.** With `--plex-column`, plex / mixture ids come from the input column
or from `--plex-table` (dataset, anchor, plex; e.g. derived from the SDRF: one
plex = one set of fraction files sharing a TMT mixture). Profiles without an id
in a plexed dataset are attached to the nearest plex by their own missingness
pattern. Without `--plex-column`, plexes are inferred from shared missingness
(Jaccard distance, average linkage, cut 0.05; datasets with >= 20 profiles),
exactly as the prototype. `--no-plex` disables the block.

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-column` / `--anchor-column` / `--gene-column` / `--value-column` | `ds` / `cvcl` / `gene` / `v` | Input columns (also used for the output); `--line-column` is a deprecated alias of `--anchor-column` |
| `--reference` | largest dataset | Dataset with `A = 0` |
| `--lineage-table` | none | Lineage per anchor, e.g. DepMap `Model.csv` |
| `--lineage-key-column` / `--lineage-column` | `RRID` / `OncotreeLineage` | Columns of the lineage table |
| `--plex-column` / `--plex-table` / `--no-plex` | inferred | Plex handling (see above) |
| `--fasta` / `--fasta-organism` | none | Sequence features; gene names from `GN=` of Swiss-Prot entries |
| `--rank` | `16` | Rank of the shared biological low-rank term |
| `--sweeps` / `--seed` | `60` / `0` | Fit sweeps (early stop on a 1% monitor hold-out) and seed |
| `--theta-output` / `--report` | none | Pooled biology and JSON fit report (per dataset: `n_anchor_samples`, `anchored`, `cross_fitted`, ...) |

The fit is deterministic (fixed seed, results independent of the thread
count) and multi-threaded; set `RAYON_NUM_THREADS` to limit cores.
