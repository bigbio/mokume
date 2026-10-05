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
is dropped. The model is the Rust port of the `lim_lin` prototype of the Cell
Line Collection integration benchmark; the defaults are the configuration that
won the follow-up benchmark (2026-10, 21 cell-line datasets, arm
`A3gnc_phdelta_all`, see [Defaults and benchmark](#defaults-and-benchmark)).
`--method lim` is still accepted as a deprecated alias of `--method bridle`
(it logs a warning).

```text
y[i,g]     = theta[anchor,g] + A[dataset,g] + c[i] + P[plex,g] + eps
theta[l,g] = m_g + Lin[lineage(l),g] + U_l . V_g + R[l,g]     (shared biology, rank --rank)
A[s,g]     = a0[s,g] + r[s,g]                                 (A[--reference] = 0)
a0[s,g]    = graph prior offset where one exists, else f(s, x_g)
```

* `f` is a per-dataset ridge regression on technical protein features `x_g`
  (reference-abundance spline plus sequence features from `--fasta`: length,
  tryptic peptides, uniqueness, GRAVY, pI, charge, missed-cleavage context,
  amino-acid composition). No functional annotation is used.
* The **graph prior** (default; `--no-graph-prior` disables it) gives `A` a
  starting value and prior mean from the anchors themselves. Per gene, batches
  (datasets, split by plex where there are plexes) are fitted by alternating
  means on anchor cells (values of anchors observed in >= 2 profiles); two
  batches are linked when they share >= 3 anchor samples, and a batch linked,
  possibly through other batches, to the reference gets the offset relative to
  it. A batch that is not linked falls back to centring onto the reference's
  gene mean when it has >= 5 profiles and >= 3 values of the gene. The prior
  of a dataset is the mean over its profiles; where neither applies (e.g. a
  single-sample dataset) the prior stays the feature model `f`. The report
  gives `graph_prior_genes` per dataset.
* `r` is an empirical-Bayes gene-specific offset around `a0` learned from
  *anchor samples* (units measured in >= 2 datasets). Datasets with 1 to 19
  anchor samples are cross-fitted, so a single-sample dataset keeps its own
  signal. A dataset that shares no anchor sample with the rest gets only
  `a0` (a warning is logged).
* `P` are TMT plex effects, `c` a per-profile loading; noise variance is
  modelled per dataset as a function of abundance.
* The fit runs up to `--sweeps` (400) sweeps and, after `--min-sweeps` (200),
  stops when the mean |change| between two sweeps of the fitted offsets
  `A + c + P` over observed cells falls below `--stop-tol` (1e-5). `c` counts
  even though it is kept in the output by default, and the post-fit plex
  rescale is not part of it. The per-sweep change is in the report's
  `history` (`out_change`). `--stop-rule monitor --sweeps 60` restores the
  previous rule (hold-out MSE change < 2e-4 over 3 sweeps after 8 sweeps),
  `--stop-rule none` always runs `--sweeps`.
* The output is `v = y - A - P` for every observed input cell (`imputed =
  false`): the sample loading `c` is **kept** by default;
  `--remove-sample-loading` outputs `v = y - A - c - P`.
  `--theta-output` writes the pooled per-anchor biology.
* **Plex rescale** (default; `--no-plex-rescale` disables it). Additive
  offsets cannot fix a batch whose spread is compressed or inflated (TMT ratio
  compression), so each (dataset x plex, gene) - or (dataset, gene) without
  plexes - is finally rescaled around its own mean: `v' = mean_b + (v -
  mean_b) / delta`. `delta` is an empirical-Bayes ComBat scale estimated from
  **all** values of the batch, never from anchor identity: the log ratio of the
  batch's variance to the pooled within-batch variance of the gene (chi-square
  bias corrected, >= 5 values), shrunk towards a per-batch mean that is itself
  shrunk to 0 (prior sd 0.5), `delta = exp(l / 2)` clipped to [0.25, 4].
  Batches with < 50 estimable genes take the median of their dataset's other
  plexes (else no rescale). Per-batch `mu` (log variance ratio) and median
  `delta` are written to the `--report` under `plex_rescale`.
* `--anchor-scale` (opt-in, not recommended) rescales each dataset that shares
  >= 20 anchor samples with the reference onto the reference's spread:
  `v' = refmean_g + (v - studymean_g) / b_s`, where `b_s` is the median over
  genes of `sd(dataset) / sd(reference)` on the shared anchors (genes observed
  on >= 10 of them, reference SD > 0.3, Pearson r > 0.5). Genes without an
  anchor mean on one side are scaled around the dataset's own mean, so no
  value is lost. Datasets below the threshold keep `b_s = 1`; `b_s` per
  dataset is written to the `--report` under `anchor_scale`. It runs before
  the plex rescale; the combination was not benchmarked.
* `--lineage-table` (opt-in, not recommended) adds a shared per-lineage
  effect `Lin` to the biology.

```bash
mokume correct-batches --method bridle \
    -i raw_long.parquet -o values.parquet \
    --reference ProCan \
    --fasta Homo-sapiens-uniprot-reviewed-contaminants.fasta --fasta-organism HUMAN \
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
exactly as the prototype. `--no-plex` disables the block. The same plexes
define the batches of the graph prior and of the plex rescale (the reference
dataset is one batch in the graph prior).

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-column` / `--anchor-column` / `--gene-column` / `--value-column` | `ds` / `cvcl` / `gene` / `v` | Input columns (also used for the output); `--line-column` is a deprecated alias of `--anchor-column` |
| `--reference` | largest dataset | Dataset with `A = 0` |
| `--lineage-table` | off | Lineage per anchor, e.g. DepMap `Model.csv` (not recommended, see below) |
| `--lineage-key-column` / `--lineage-column` | `RRID` / `OncotreeLineage` | Columns of the lineage table |
| `--plex-column` / `--plex-table` / `--no-plex` | inferred | Plex handling (see above) |
| `--fasta` / `--fasta-organism` | none | Sequence features; gene names from `GN=` of Swiss-Prot entries |
| `--rank` | `16` | Rank of the shared biological low-rank term |
| `--sweeps` / `--seed` | `400` / `0` | Maximum fit sweeps and seed |
| `--stop-rule` | `output` | `output` (mean change of `A + c + P` < `--stop-tol`), `monitor` (previous rule), `none`; the report records the value as given (`params.stop_rule`) |
| `--stop-tol` / `--min-sweeps` | `1e-5` / `200` (`monitor`: `2e-4` / `8`) | Early-stop tolerance and minimum sweeps |
| `--no-graph-prior` | prior on | Use the feature model `f` alone as the prior of `A` |
| `--remove-sample-loading` | `c` kept | Output `y - A - c - P` |
| `--no-plex-rescale` | rescale on | Skip the post-fit (dataset x plex, gene) rescale |
| `--anchor-scale` | off | Per-dataset scale `b_s` from anchor samples shared with the reference (see above) |
| `--theta-output` / `--report` | none | Pooled biology and JSON fit report (per dataset: `n_anchor_samples`, `anchored`, `cross_fitted`, `graph_prior_genes`, `delta_median`, ...; `history`, `plex_rescale`; `dataset_value` only when the value report is computed) |
| `--value-report` | off | Compute the value report and add it to the `--report` JSON (`dataset_value`); implied by the four options below |
| `--dataset-report` / `--profile-report` | none | Per-dataset / per-profile value report (TSV), see [Per-dataset value report](#per-dataset-value-report) |
| `--identity` | off | Add the profile-vs-profile identity check to the value report (quadratic in profiles) |
| `--identity-reference` | none | Per-anchor reference profiles (e.g. DepMap RNA) for the identity check; implies `--identity` |

### Per-dataset value report

What does each dataset contribute, and does it agree with the rest? With
`--value-report`, `--dataset-report`, `--profile-report`, `--identity` or
`--identity-reference`, the same fit (no refits) is summarised per dataset
(JSON `dataset_value` in `--report`, TSV `--dataset-report`) and per profile
(`--profile-report`). `--report` alone does not compute it.

**Memory.** The disagreement and redundancy metrics keep every cross-dataset
cell pair (same anchor and gene), so the report's memory grows with
sum over anchors of C(m, 2) x genes (m = datasets measuring the anchor; 5
floats per pair), on top of the fit. It can exceed the fit itself: on the
review probe the fit peaked at 586 MB and fit + report at 1.47 GB. The
identity checks are also quadratic, in time: `--identity` correlates every
profile with every profile of the other datasets, and `--identity-reference`
every profile with every reference anchor. A *shared* anchor is measured by >= 2
datasets; `v` is the corrected output. By default only the cheap metrics are
computed; the identity check against other datasets (`id_self_r`, `id_rank`,
..., per profile `self_r`, `self_rank`, `best_line`, ...) correlates every
profile with every profile of the other datasets and is opt-in with
`--identity` (implied by `--identity-reference`). Without it those columns are
`NaN` (`best_line` empty); `abund_rho` is always computed.

| Group | Columns | Definition |
|-------|---------|------------|
| Coverage | `n_lines`, `n_cells`, `n_genes`, `genes_per_profile` | Size of the dataset |
| | `uniq_lines` | Anchors only this dataset measures |
| | `anchor_lines` / `bridge_lines` | Shared anchors / anchors shared with exactly one other dataset (single-source without this one) |
| | `anchor_partners` / `partners_any` | Datasets sharing >= 3 / >= 1 anchors |
| | `uniq_genes` / `uniq_gene_cells` | Genes / (anchor, gene) cells observed in no other dataset |
| Fit | `noise_var` | Median fitted noise variance `sig2[s,g]` over its genes |
| | `c_abs` / `offset_sd` | Median \|sample loading\|; SD of the offset `A[s,g]` over its genes (0 for the reference) |
| | `delta_median` / `delta_extreme` | Plex rescale `delta` over its (batch, gene) cells: median, fraction outside [0.67, 1.5] |
| | `agree_med` / `disagree_var` | Over cross-dataset differences `e = v_a - v_b` (same anchor and gene) involving the dataset: median \|e\|, median(e^2)/0.4549 |
| | `excess_var` | The dataset's share `d_s` of that variance: non-negative least squares of median(e^2)/0.4549 = `d_a + d_b` over dataset pairs with >= 200 cells. No replicate-noise floor is subtracted, so it includes the dataset's own noise. Only identifiable where the datasets' pair graph has an odd cycle (e.g. three datasets that all share anchors): with 2 datasets, chains such as A-REF-C or even cycles the split `d_a + d_b` is not unique, so `excess_var` is `NaN` (a warning is logged) |
| Identity | `abund_rho` / `abund_rho_min` | Spearman of each raw profile with the gene's median level in the *other* datasets (fit-free); median / min over profiles. Low = distorted abundance shape (enrichment, wrong unit) |
| | `id_self_r`, `id_rank`, `id_top1`, `id_best_is_self` | Only with `--identity`: on gene-centred `v`, Pearson with every profile of the other datasets (>= 200 shared genes): mean r with the same anchor, rank of the own anchor (1 = best), fraction ranked 1st, fraction whose best match is the own anchor |
| | `id_r_max_any` / `id_r_med_any` | Only with `--identity`: best / median r with any other-dataset profile |
| | `id_rna_self`, `id_rna_rank`, `id_rna_top1`, `id_rna_top5` | Only with `--identity-reference`: Pearson of the gene-centred profile with each reference anchor (reference gene-centred across anchors), own-anchor r and rank |
| Redundancy | `marg_shift`, `marg_se_gain`, `wshare` | Per shared anchor and gene with >= 2 sources, consensus `sum(w v)/sum(w)`, `w = 1/sig2[s,g]`: \|consensus - consensus without this dataset\|, `1 - SE_with/SE_without`, weight share; medians over genes, then over its shared profiles |
| | `redund_ge3` / `other_src_med` | Fraction of its shared profiles whose anchor has >= 3 other sources / median other sources |
| Flag | `no_anchors_cannot_audit` | No shared anchor: identity, agreement and redundancy do not exist, and problems such as an enrichment artefact are invisible here (also logged as a warning) |

The definitions are the single-fit ("cheap") metrics of the 2026-10 dataset
value analysis. Differences from that analysis: values are the in-sample
corrected output (not leave-lines-out predictions), disagreement uses plain
cross-dataset differences instead of the benchmark scorer, no replicate-noise
floor is available, and redundancy weights are the fit's per-cell noise
variances. The per-profile table (`self_rank`, `best_line`, `abund_rho`, ...)
points at the individual profiles behind a poor dataset summary.

Not implemented yet: leave-one-dataset-out refits that measure each dataset's
influence on the others (planned as an opt-in `--influence`; about one full
fit per dataset with >= 2 anchors, plus a placebo).

### Defaults and benchmark

Measured on 21 cell-line datasets (5-fold leave-lines-out, held-out lines
relabelled per dataset; accuracy = median |error| of a held-out line's
cross-dataset difference, lower is better; biology guards = cis RNA / copy
number correlation, deletion AUCs, CORUM co-complex AUROC, proliferation and
EMT signatures):

| Step | Default | Measured effect |
|------|---------|-----------------|
| 400 sweeps / output-change stop | on | 60 sweeps were not converged: accuracy -0.027 [-0.034, -0.021] at 400 sweeps, better agreement, no biology lost. The benchmark arms ran a fixed 400 sweeps, while the defaults may stop from sweep 200 once the change is below `1e-5`; the output differs negligibly from a fixed 400 (measured median |difference| 1.7e-6). The tolerance is a safeguard, not a tuned value |
| Graph prior | on | Leave-one-dataset-out accuracy 1.062 -> 0.797 (ahead of graph offsets alone, 0.804); leave-lines-out equivalent (+0.005) |
| Keep `c` | on | 0.011 less accurate than removing `c`, but CORUM +0.010 and better deletion / cis signal: `c` carries biology |
| Plex rescale | on | Accuracy 0.786 vs 0.799 for BERT (-0.013 [-0.021, -0.007]), better on cis RNA, EMT and proliferation; 0.007 less accurate than without the rescale but better on 7 biology guards |
| Lineage table | off | No measurable effect |
| `--anchor-scale` | off | Hurts accuracy (about +0.08, like other monotone rescalings) |

Caveats:

* **Post-hoc selection.** The plex rescale (and its "all rows of the batch"
  variant) was chosen after a diagnostic on fold 1 of the same benchmark, so
  its margin over BERT is optimistic; the spec was fixed before scoring the
  other folds.
* **Rescale and outliers.** A few hundred cells with absurd values (near-zero
  TMT reporter intensities, |v| up to ~47 log2) are stretched further by the
  rescale; medians are unaffected but RMSE-type summaries inflate. Filter such
  values upstream.
* **Datasets without anchors.** Offsets predicted from features (`f`) were
  validated only for single-sample datasets. For a whole multi-line dataset
  without anchors (leave-one-dataset-out) BRIDLE was 0.005-0.015 less accurate
  than BERT / per-batch centring, so do not rely on predicted offsets alone
  for such datasets.
* **Lineage table.** Integrating with lineage labels and then studying
  lineage differences in the output is circular; it had no effect on accuracy.
* `BridleParams::legacy()` (Rust API) reproduces the pre-benchmark
  configuration: 60 sweeps with the monitor rule, no graph prior, `c` removed,
  no rescale. On the command line that takes all five flags:
  `--stop-rule monitor --sweeps 60 --no-graph-prior --remove-sample-loading
  --no-plex-rescale`.

The fit is deterministic (fixed seed, results independent of the thread
count) and multi-threaded; set `RAYON_NUM_THREADS` to limit cores.
