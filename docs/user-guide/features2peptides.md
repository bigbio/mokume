# features2peptides: Peptide Normalization

The `features2peptides` command normalizes feature-level mass spectrometry data into peptide intensities. This is the first step of the two-step pipeline, giving you fine-grained control over normalization before protein quantification.

## Basic Usage

=== "CLI"

    ```bash
    mokume quantify features2peptides \
        -p features.parquet \
        -s experiment.sdrf.tsv \
        --run-normalization median \
        --sample-normalization global-median \
        --output peptides.csv
    ```

=== "Python (wheel)"

    The wheel wrapper validates documented keyword arguments, maps them to the
    command's exact CLI flags, and runs the same kernel in-process:

    ```python
    import mokume

    mokume.features2peptides(
        parquet="features.parquet",
        sdrf="experiment.sdrf.tsv",
        run_normalization="median",
        sample_normalization="global-median",
        output="peptides.csv",
    )
    ```

=== "Python (explicit argv)"

    ```python
    import mokume

    mokume.run([
        "quantify",
        "features2peptides",
        "--parquet", "features.parquet",
        "--sdrf", "experiment.sdrf.tsv",
        "--run-normalization", "median",
        "--sample-normalization", "global-median",
        "--output", "peptides.csv",
    ])
    ```

## Processing Steps

The command performs these steps in order:

1. Parse protein identifiers and retain unique peptides
2. Remove entries with empty intensity or condition
3. Filter peptides by minimum amino acid length
4. Remove low-confidence proteins (< min unique peptides)
5. Optionally remove decoys, contaminants, and specified proteins
6. Normalize at feature level between MS runs (`--run-normalization`)
7. Merge peptidoforms across fractions and technical replicates
8. Normalize at sample level (`--sample-normalization`)
9. Remove low-frequency peptides (optional)
10. Assemble peptidoforms to peptides
11. Optional log2 transformation

### Runs and samples

With `--sdrf`, all runs (`comment[data file]`) that carry the same SDRF
`source name` form one sample. In step 7 the highest intensity of each
peptidoform and charge across those runs is kept, and step 10 sums the
peptidoforms into the canonical peptide; `--aggregation-level run` keeps the
runs apart instead. Sample-level normalization uses every sample, so merging
runs into one sample also changes the values of the other samples.

mokume logs each source name that maps to more than one data file: at INFO
level for runs that differ in `comment[technical replicate]` or
`comment[fraction identifier]` (the normal replicate or fraction design), and as
a WARNING when those runs also share technical replicate and fraction identifier
(identical or missing values), which usually indicates a duplicated source name
rather than technical replicates. Neither message changes the output.

## Normalization Methods

### Feature-Level (`--run-normalization`)

| Method | Description |
|--------|-------------|
| `median` | Normalize by median across MS runs (default) |
| `mean` | Normalize by mean across MS runs |
| `max` | Normalize by the maximum intensity within each run |
| `global` | Normalize by total intensity within each run |
| `max-min` | Apply min-max scaling |
| `iqr` | Normalize by interquartile range |
| `none` | Skip feature normalization |

### Sample-Level (`--sample-normalization`)

| Method | Description |
|--------|-------------|
| `global-median` | Adjust all samples to global median (default) |
| `condition-median` | Adjust samples within each condition |
| `none` | Skip sample normalization |

!!! note "Only scalar-per-sample methods are accepted"
    The peptide flow accepts only `none`, `global-median`, and
    `condition-median`. Dataset-level methods need the full matrix and are
    rejected here instead of being accepted as no-ops. They remain available
    in [`features2proteins`](features2proteins.md#normalization-options).

## Filtering Options

```bash
mokume quantify features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --min-aa 7 \
    --min-unique 2 \
    --remove-decoy-contaminants \
    --remove-low-frequency-peptides \
    --output peptides.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--min-aa` | 7 | Minimum amino acid length |
| `--min-unique` | 2 | Minimum unique peptides per protein |
| `--keep-shared-peptides` | off | Keep shared/non-unique peptides and skip the unique-peptide gate |
| `--remove-decoy-contaminants` | off | Remove decoys and contaminants |
| `--remove-low-frequency-peptides` | off | Remove peptides in <20% of samples |
| `--remove-ids` | none | File with protein IDs to exclude |

## TMT / ITRAQ Options

For labeled datasets, `features2peptides` also supports IRS-style scaling and control over aggregation level:

| Option | Default | Description |
|--------|---------|-------------|
| `--irs-channel` | none | Explicit pooled/reference channel label |
| `--irs-autodetect-regex` | none | Regex to detect pooled samples from SDRF |
| `--irs-stat` | `median` | IRS per-run statistic: median or mean |
| `--irs-scope` | `global` | IRS scaling scope: global, by-mixture, or two-stage |
| `--aggregation-level` | `sample` | Aggregate intensities at sample or run level |

!!! note "Channel-based IRS"
    The channel IRS path (`--irs-channel` / `--irs-autodetect-regex`) scales on
    the TMT `mixture` / `channel` columns and is implemented for all three scopes
    (`--irs-scope global` / `by-mixture` / `two-stage`). The reference channel is
    taken from `--irs-channel`, or auto-detected from the SDRF when
    `--irs-autodetect-regex` is given. For cross-plex reference scaling driven by
    the SDRF, see
    [`features2proteins`](features2proteins.md#irs-normalization-multi-plex-tmt).

## Preprocessing Filters

For more advanced filtering, use a YAML/JSON configuration file:

```bash
# Generate example configuration
mokume quantify features2peptides --generate-filter-config filters.yaml

# Use filter configuration
mokume quantify features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --output peptides.csv

# CLI overrides (take precedence over config file)
mokume quantify features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --filter-min-intensity 1000 \
    --filter-cv-threshold 0.3 \
    --filter-charge-state 2 \
    --filter-charge-state 3 \
    --filter-charge-state 4 \
    --filter-score diann_ms1_profile_corr=0.8 \
    --filter-protein-fdr 0.01 \
    --output peptides.csv
```

### CLI Filter Overrides

| Option | Description |
|--------|-------------|
| `--filter-min-intensity` | Minimum intensity threshold |
| `--filter-cv-threshold` | Maximum CV across replicates |
| `--filter-charge-state` | Allowed charge state; repeat for multiple states |
| `--filter-max-missed-cleavages` | Maximum missed cleavages |
| `--filter-peptide-fdr` | Maximum QPX `peptide_qvalue` |
| `--filter-score` | Named QPX score threshold (`NAME=THRESHOLD`) |
| `--filter-exclude-modification` | Modification name or accession to exclude; repeat for multiple values |
| `--filter-protein-fdr` | Maximum QPX `pg_global_qvalue` per protein group |
| `--filter-min-features` | Minimum identified features per run |
| `--filter-max-missing-rate` | Maximum missing feature fraction per technical run |

!!! note "Group-level filter support"
    The per-row filters (min-intensity floor, peptide length, charge states,
    excluded modifications, missed cleavages) and the per-`(protein, sample)`
    unique-peptide gate are wired in the kernel and oracle-locked vs Python.
    FDR filtering is opt-in: peptide FDR uses `peptide_qvalue`, while protein
    FDR keeps groups whose minimum `pg_global_qvalue` passes. A requested but
    unpopulated q-value field is an error rather than a no-op.
    `--filter-score` matches one exact score name and reads its QPX
    `higher_better` flag, so higher-better scores use `>=` and lower-better
    scores use `<=`. Missing, invalid, or direction-inconsistent scores fail
    before output is written.
    In the struct form that `qpxc transform relayout` writes, the score field
    and its direction come from the `qpx.struct_lists` parquet footer; a struct
    column without that footer, or a name the footer does not list, is an
    error. `precursor_qvalue` and `global_qvalue` are read from the top-level
    QPX 1.3 columns when the parquet statistics show no nulls in any row
    group, and from the score entry otherwise.
    `--filter-exclude-modification` compares each value with the `name` and
    `accession` of the QPX `modifications` list, ignoring case, so
    `Carbamidomethyl` and `UNIMOD:4` remove the same rows. Inputs without that
    column match the peptidoform instead: an accession such as `UNIMOD:35`
    matches only the complete bracketed token, and other values match as
    substrings.
    Among the group-level filters, CV threshold (`--filter-cv-threshold`),
    quantile outlier removal, and the run-QC checks `--filter-min-features` /
    min-total-intensity / min-proteins / `--filter-max-missing-rate` are
    implemented via a pre-pass. Run-QC rejects individual technical runs;
    missing rate uses the complete distinct `(protein, peptide)` universe among
    the surviving runs in that sample. Replicate agreement reproduces the Python
    per-sample behaviour. Filter-config keys that the QPX streaming model cannot
    evaluate (including sample correlation, search score, and protein coverage)
    are rejected when active. Unknown YAML/JSON keys are also rejected, so a typo
    cannot silently pass through.

See [Preprocessing Filters](../concepts/preprocessing.md) for the full filter reference.

## Output Options

```bash
# Standard CSV output
mokume quantify features2peptides -p data.parquet -o peptides.csv

# Parquet output
mokume quantify features2peptides -p data.parquet -o peptides.csv --save-parquet

# Log2 transform
mokume quantify features2peptides -p data.parquet -o peptides.csv --log2

# Skip normalization entirely
mokume quantify features2peptides -p data.parquet -o peptides.csv --skip-normalization
```

`--skip-normalization` conflicts with channel IRS options. IRS autodetection
requires an SDRF and must match a reference channel; otherwise the command
fails instead of writing an unscaled result.

## Python API

The wheel exposes the same command as a thin wrapper; keyword arguments map to CLI
validated command flags; unknown keywords and invalid value shapes fail before
dispatch.

```python
import mokume

mokume.features2peptides(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    min_aa=7,
    min_unique=2,
    remove_decoy_contaminants=True,
    remove_low_frequency_peptides=True,
    output="peptides-norm.csv",
    run_normalization="median",
    sample_normalization="global-median",
    log2=True,
    save_parquet=True,
)
```

### With Preprocessing Filters

Point the wrapper at a YAML/JSON filter config, or pass the per-filter overrides
directly:

```python
import mokume

mokume.features2peptides(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    output="peptides.csv",
    run_normalization="median",
    sample_normalization="global-median",
    filter_config="filters.yaml",
    filter_min_intensity=1000,
    filter_charge_state=[2, 3, 4],
    min_unique=2,
)
```
