# features2proteins Python/Rust Parity Notes

This document tracks compatibility coverage for overlapping Python and Rust
`features2proteins` paths. Rust is the leading implementation; frozen outputs
captured from Python remain compatibility baselines only for the paths covered
here, and behavior must also be checked against the public contract and algorithm
intent for each parameter.

## Parameter Alignment

| Parameter | Stage | Python Behavior | Rust Behavior | Status |
| --- | --- | --- | --- | --- |
| `--parquet` | Input | Required QPX feature parquet input. | Required QPX feature parquet input. | Implemented |
| `--sdrf` | Input | Optional SDRF sample metadata. Required for ratio and metadata-aware stages. | Optional SDRF sample metadata. Required for ratio and metadata-aware stages. | Implemented subset |
| `--output` | Output | Required protein matrix output path. | Required protein matrix output path. | Implemented |
| `--quant-method` | Quantification | Feature-level `peptide_count` counts distinct peptides. | `peptide-count` uses the same feature-level definition; `spectral-count` pairs PSM and feature QPX, resolves groups through `feature_id`, and counts unique QPX `psm_id` values. Missing cells for both count methods are additive zeroes in observed samples. | Intentional contract split |
| `top<N>` method syntax | Quantification | Accepts N in the method name, such as `--quant-method top5`. | Same. | Implemented |
| `--min-aa` | Filtering | Removes peptide sequences shorter than the threshold. | Removes peptide sequences shorter than the threshold. | Implemented |
| `--min-unique` | Filtering | Requires a minimum number of unique peptides per protein/sample cell. | Requires a minimum number of unique peptides per protein/sample cell for non-piBAQ methods. | Implemented subset |
| `--filter-protein-fdr` | Filtering | Not available (Python exposes it only on `features2peptides`). | Keeps protein groups whose best QPX `pg_global_qvalue` is `<=` the threshold, same group-minimum rule as `features2peptides`. | Rust-only extension |
| contaminant policy | Filtering | `--remove-contaminants/--keep-contaminants` controls removal. | Removes by default; `--keep-contaminants` opts out. The redundant positive flag is not exposed. | Implemented subset |
| `--run-normalization` | Normalization | Supports none, mean, median, max, global, max_min, and iqr. | Supports none, mean, median, max, global, max_min, and iqr. | Implemented subset |
| `--sample-normalization` | Normalization | Supports none, globalmedian, conditionmedian, hierarchical, quantile, mediancenter, meancenter, rlr, and loess. | Supports none, globalmedian, conditionmedian, quantile, mediancenter, meancenter, rlr, loess, and hierarchical. hierarchical has non-identity real-path golden oracles and is cell-exact; loess is ~2e-3 vs statsmodels lowess; mediancenter/meancenter/hierarchical are real-data cell-exact on PXD003539. | Implemented subset |
| worker count | Runtime | `--duckdb-threads` caps DuckDB workers and `--directlfq-cores` caps DirectLFQ workers. | `--threads` configures the shared Rayon global pool. | Intentional CLI difference |
| memory budget | Runtime | `--duckdb-memory` caps DuckDB memory. | `--memory` is a cross-platform soft process resident-memory planner/guard: it reduces QPX batch/read-ahead memory and checks RSS or Working Set between batches/phases. It cannot replace an operating-system, scheduler, or container hard limit. Runtime pyOpenMS piBAQ digestion occurs before the guarded Rust dispatch. | Intentional CLI difference |
| `--export-peptides` | Output | Writes normalized peptide-level intermediates. | Writes Python-shaped peptide intermediates for non-DirectLFQ methods; DirectLFQ peptide export still returns `NotImplemented`. | Implemented subset |
| `--export-ions` | Output | Streams DirectLFQ's normalized, within-protein-aligned ion matrix in linear intensity space. | Writes the same normalized, within-protein-aligned linear ion matrix; non-DirectLFQ methods return `NotImplemented`. | Implemented subset |
| `--normalization-proteins` | Normalization | Restricts normalization proteins to the provided list for selected dataset-level normalizers without filtering the final matrix. | Implemented sample normalizers use the file to restrict factor inputs and still apply the factors to the full output matrix. It is unavailable for directlfq, ratio, peptide-count, and spectral-count. Empty files or lists with no matching features fail clearly. | Implemented subset |
| `--coverage-threshold` | Postprocessing | Drops proteins below per-condition non-missing coverage. | Applies a per-condition coverage filter when SDRF condition metadata is available. | Implemented subset |
| `--min-sample-correlation` | Postprocessing | Drops samples below mean within-condition Pearson correlation on the normalized protein matrix. | Same one-shot pairwise-complete log2 correlation and sample-column removal. | Implemented |
| `--impute` and basic imputation options | Postprocessing | Supports a broad imputation catalog. | Supports mindet, minprob, mean, median, constant, zero, most_frequent, knn, seqknn, qrilc, impseq, gms, bpca, and impseqrob. `--impute` requires an explicit method. `missforest` is absent from the compute CLI and remains available through `mokume.impute`. | Implemented subset |
| batch correction options | Postprocessing | Runs ComBat when optional dependencies are available. | Native ComBat supports parametric/non-parametric, covariates, ref_batch, and mean_only. The protein-matrix CLI exposes only sample_prefix and explicit SDRF column detection. Invalid batch layouts and a matrix with no complete protein row fail instead of returning unchanged values. | Implemented subset |
| differential expression options | Postprocessing | Runs DE methods and writes DE output. | The Rust dispatcher supports limma, deqms, rots, limrots, proda, and ensemble. Conditions resolve from SDRF source names and data-file stems. Deterministic kernels are cell-exact; RNG/optimizer methods use tiered compatibility. BH/IHW/BKY/Storey apply where the method exposes raw p-values; ROTS/LimROTS retain permutation FDR and reject alternative corrections. DE requires explicit contrasts and output. Ensemble-only parameters are rejected for single methods. | Implemented (tiered) |
| plotting and report options | Output | Writes plots and optional HTML reports. | Not exposed by the Rust compute CLI; plotting/report APIs consume its CSV outputs. | Python periphery |

## Real-Data Parity Matrix

Use the local cell-line QPX data for Rust/Python comparisons.

| Dataset | Role | Feature Parquet | SDRF |
| --- | --- | --- | --- |
| `PXD003539` | First-pass baseline | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/qpx/PXD003539.feature.parquet` | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/mokume/sdrf/PXD003539.sdrf.tsv` |
| `PXD004701` | Normalization and TopN baseline | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD004701/qpx/PXD004701.feature.parquet` | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD004701/mokume/sdrf/PXD004701.sdrf.tsv` |
| `PXD041421` | Normalization and TopN baseline | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD041421/qpx/PXD041421.feature.parquet` | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD041421/mokume/sdrf/PXD041421.sdrf.tsv` |
| `PXD030304` | Stress baseline | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD030304/qpx/PXD030304.feature.parquet` | `/home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD030304/mokume/sdrf/PXD030304.sdrf.tsv` |

## Required Comparison Metrics

Every real-data parity report must include:

- protein count
- sample count
- protein set differences
- sample set differences
- shared matrix cell count
- median absolute error
- 95th percentile absolute error
- median relative error
- Spearman correlation per sample
- total intensity difference per sample
- largest protein/sample cell differences

## First-Pass Command Shape

Python baseline:

```bash
python -m mokume.mokume_cli features2proteins \
  --parquet /home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/qpx/PXD003539.feature.parquet \
  --sdrf /home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/mokume/sdrf/PXD003539.sdrf.tsv \
  --output /tmp/mokume-parity/PXD003539/python/sum_none.csv \
  --quant-method sum \
  --run-normalization none \
  --sample-normalization none \
  --duckdb-threads 24
```

`mokume` wheel baseline:

```bash
mokume quantify features2proteins \
  --parquet /home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/qpx/PXD003539.feature.parquet \
  --sdrf /home/shenyufei/Git-repository/Bigbio/Bigbio_data/cell-lines/PXD003539/mokume/sdrf/PXD003539.sdrf.tsv \
  --output /tmp/mokume-parity/PXD003539/rust/sum_none.csv \
  --quant-method sum \
  --run-normalization none \
  --sample-normalization none \
  --threads 24
```

Comparison:

```bash
python scripts/compare_protein_matrices.py \
  --python /tmp/mokume-parity/PXD003539/python/sum_none.csv \
  --rust /tmp/mokume-parity/PXD003539/rust/sum_none.csv \
  --output /tmp/mokume-parity/PXD003539/sum_none_report.json
```

## Verification Basis and Known Gaps

As of the 2026-06-20 alignment audit, 26 computation-core methods are parity-checked
and considered aligned: the quant methods reachable through `features2proteins`
(sum, median, `top<N>`, intensity, abd, peptide_count, directlfq, maxlfq — which
delegates to the directlfq path by default, matching Python — and the piBAQ
family-allocation core), the run-level and sample-level normalizers listed above, and
14 imputation methods.

Feature-level `peptide_count` and paired PSM/feature `spectral_count` have
separate input contracts and are validated independently.

### DirectLFQ feature contract

The Python reference loader and the Rust feature path prepare DirectLFQ ions with
the same contract:

- `ProteinName` is the complete semicolon-joined protein group from
  `pg_accessions`; the first accession is not used as a lossy group key.
- A contextual precursor is identified by peptidoform, charge, sample,
  condition, and biological replicate. Duplicate feature rows for that context
  contribute their maximum intensity. Distinct contextual precursors are then
  summed into one canonical-peptide intensity per `(protein group, sample)`.
- `min_unique_peptides` is applied per `(protein group, sample)` using distinct
  canonical peptides before ions enter the solver.
- Protein groups, canonical ion sequences, and sample names receive lexical
  deterministic ordering before DirectLFQ constructs its matrix. Rust maps its
  temporary lexical IDs back to the original registries after solving, so input
  row encounter order cannot change the result.
- `--export-ions` is produced by the same DirectLFQ calculation as the protein
  matrix. It contains the sample-normalized and within-protein-aligned ion
  profiles in linear intensity space, not the raw input traces.

Real-data parity (2026-06-21, Wave 1-C): `scripts/run_parity_matrix.sh` was run across
all four cell-line datasets (PXD003539/004701/041421/030304) for the Phase-1
deterministic combos — **24/24 combos cell-exact** (rel max 0.0 for
sum/median/topn/runmedian/conditionmedian; ~5e-8 float-level for globalmedian; protein
and sample sets exact py==rust). The DE family was real-data verified on PXD004701
(limma/deqms cell-exact, 100% significance-call agreement; rots/limrots rank-level).

The corrected DirectLFQ/MaxLFQ feature contract above supersedes comparisons that
used first-accession grouping or raw feature sums. It was rerun on 2026-08-30 by
comparing the native feature route with the native
`features2peptides -> peptides2protein` route under identical filters
(`min_unique=1`) and no external normalization. Protein/sample sets, nonzero
support, and every positive cell were exact in all four LFQ checks:

- PXD002099 DirectLFQ: 1,701 proteins, 5 samples, 7,542 positive cells.
- PXD007683 DirectLFQ: 8,249 proteins, 11 samples, 78,541 positive cells.
- PXD002099 delegated MaxLFQ: 1,239 proteins, 5 samples, 5,353 positive cells.
- PXD007683 delegated MaxLFQ: 6,500 proteins, 11 samples, 60,117 positive cells.

The maximum relative error was `0.0` and support mismatch was zero for every
check. On PXD002099, enabling DirectLFQ ion export left the protein matrix
byte-equivalent and produced 10,097 unique, finite, nonnegative protein/ion rows.
These checks establish parity between the two native entry paths; they do not
substitute for a separate Python-versus-Rust runtime comparison.
`ratio` has no real-data parity (cell-line data is label-free / no plex); it is
covered by a synthetic plexed golden test cell-exact vs Python
`RatioQuantification`.

Since the 2026-06-20 audit: the DE family (limma/deqms/rots/limrots/proda/
ensemble), all four FDR choices (BH/IHW/BKY/Storey), both data-driven
effect-size gates (mixture/null_quantile), and covariate/non-parametric ComBat
were ported and wired;
`most_frequent` imputation and the piBAQ extras (TPA/ProteomicRuler/normalize_pibaq)
landed; `peptides2protein`, `correct-batches`, and `features2peptides` (filters +
peptide normalization) are wired; `ratio` switched to an SDRF-driven `sample_to_plex`
map; `maxlfq` and `directlfq` share the corrected feature-input contract above.
Deterministic non-LFQ methods are cell-exact; RNG/optimizer DE methods are
faithful-not-bit-exact (tiered tolerance).

Remaining gaps (irreducible or niche):

- Python-delegated computation methods (functionally available via Python mokume;
  the Rust kernel does not reproduce them, and its error points the user to
  Python): `missforest` (wraps scikit-learn `IterativeImputer` +
  `RandomForestRegressor`; tree + RNG internals not reproducible cross-language —
  a Rust ML crate would not align either).
- `features2peptides` runs the filtering, run/sample factor normalization,
  intermediate exports, and all three channel-IRS scopes. Its sample-normalizer
  choices are intentionally limited to `none`, `globalmedian`, and
  `conditionmedian`; full-matrix methods belong to `features2proteins`. Active
  filter settings that QPX cannot evaluate and unknown config keys are rejected.
  IRS must resolve a real reference channel and scaling factors.
