# Preprocessing Filters

mokume provides a configurable filter system for quality control, driven by YAML/JSON files or CLI options on the `features2peptides` command.

## QPX and SDRF sample identity

When an SDRF is supplied for the current QPX list-of-label/intensity schema, every positive intensity must resolve to exactly one SDRF record before preprocessing starts. Run keys use the filename basename, ignore ASCII case, and remove one trailing `.raw`, `.mzml`, `.d`, or `.wiff` extension. Labels ignore surrounding whitespace and ASCII case, and SDRF controlled-vocabulary values use their `NT=` term. `LFQ`, `label-free`, and `label free sample` are the only equivalent label aliases.

Label-free QPX entries whose intensity labels are run filenames use each intensity label as its owning run. Isobaric entries use the row run together with the reporter label. Duplicate normalized SDRF keys, unmatched runs, missing reporter channels, and ambiguous run-only matches are errors; mokume does not fall back to the first SDRF row or a run name when an SDRF mapping is invalid.

QPX 1.3 rows can list in `imputed_labels` the labels whose intensity the upstream tool imputed instead of measuring (for example Spectronaut `EG.IsImputed`). mokume drops those intensities before preprocessing and quantification; inputs without the column, and rows where it is null, are unchanged.

## Filter Categories

### Intensity Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| MinIntensityFilter | `min_intensity` | 0.0 | Remove features below threshold |
| CVThresholdFilter | `cv_threshold` | null | Max CV across replicates |
| ReplicateAgreementFilter | `min_replicate_agreement` | 1 | Min replicates with detection |
| QuantileFilter | `quantile_lower/upper` | 0.0/1.0 | Remove intensity outliers |

### Peptide Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| PeptideLengthFilter | `min/max_peptide_length` | 7/50 | Peptide length range |
| ChargeStateFilter | `allowed_charge_states` | null | Allowed charges (e.g., [2,3,4]) |
| ModificationFilter | `exclude_modifications` | [] | Remove rows carrying these modifications (name or accession) |
| MissedCleavageFilter | `max_missed_cleavages` | null | Max missed cleavages |
| SequencePatternFilter | `exclude_sequence_patterns` | [] | Regex patterns to exclude |

### Protein Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| ContaminantFilter | `remove_contaminants/decoys` | true | Remove contaminants/decoys |
| MinPeptideFilter | `min_unique_peptides` | 2 | Min unique peptides per protein/sample cell |
| RazorPeptideFilter | `razor_peptide_handling` | "keep" | Handle shared peptides |

### Run/Sample QC Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| RunIntensityFilter | `min_total_intensity` | 0.0 | Min total intensity per run |
| MinFeaturesFilter | `min_identified_features` | 0 | Min features per run |
| MissingRateFilter | `max_missing_rate` | 1.0 | Max missing value rate |

!!! note "Group-level filter support"
    The per-row filters (min-intensity floor, peptide length, charge states,
    excluded modifications, missed cleavages) and the per-`(protein, sample)`
    unique-peptide gate are wired and oracle-locked in the Rust kernel. Among the
    **group-level** filters, CV threshold, quantile outlier removal, and the
    run-QC checks (min-features, min-total-intensity, min-proteins and missing
    rate) are implemented via a technical-run pre-pass. Missing rate is the absent
    fraction of the complete distinct `(protein, peptide)` universe among the
    surviving runs in each sample. Replicate agreement reproduces the Python
    per-sample behaviour. Settings that cannot be evaluated from QPX streaming
    input (sample correlation, search score, and coverage) are rejected when active.
    Unknown YAML/JSON keys and unsupported razor handling are also rejected.

## Configuration

### YAML Configuration File

Generate an example configuration:

```bash
mokume quantify features2peptides --generate-filter-config filters.yaml
```

Example `basic_qc.yaml`:

```yaml
name: basic_qc
enabled: true

intensity:
  min_intensity: 0.0

peptide:
  fdr_threshold: null
  min_peptide_length: 7
  max_peptide_length: 50

protein:
  fdr_threshold: null
  min_unique_peptides: 2
  remove_contaminants: true
  remove_decoys: true
  contaminant_patterns:
    - CONTAMINANT
    - CONTAM_
    - ENTRAP
    - DECOY
```

### Using Filter Configurations

```bash
# From config file
mokume quantify features2peptides \
    -p features.parquet -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --output peptides.csv

# CLI overrides (take precedence over config file)
mokume quantify features2peptides \
    -p features.parquet -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --filter-min-intensity 1000 \
    --filter-cv-threshold 0.3 \
    --filter-protein-fdr 0.01 \
    --output peptides.csv

# CLI-only filtering (no config file)
mokume quantify features2peptides \
    -p features.parquet -s experiment.sdrf.tsv \
    --filter-min-intensity 500 \
    --min-unique 2 \
    --output peptides.csv
```

Peptide and protein FDR thresholds are explicit opt-ins. They use the dedicated
QPX `peptide_qvalue` and `pg_global_qvalue` fields; requesting a threshold when
the corresponding field is absent or entirely null fails before output is
written.

### Python (wheel)

The filter config is built and applied entirely inside the kernel; the wheel
wrapper passes the same flags as the CLI:

```python
import mokume

# Generate an example config
mokume.features2peptides(generate_filter_config="filters.yaml")

# Run with a config file plus CLI overrides
mokume.features2peptides(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    filter_config="filters.yaml",
    filter_min_intensity=1000,
    min_unique=2,
    output="peptides.csv",
)
```

Charge states and excluded modifications are Python sequences, which the wrapper
translates to repeated `--filter-charge-state` and
`--filter-exclude-modification` flags.

## Pre-configured Templates

mokume includes templates for common scenarios in `tests/example/filters/`:

| Configuration | Use Case | Description |
|---------------|----------|-------------|
| `basic_qc.yaml` | General QC | Minimal filtering for standard experiments |
| `stringent_filtering.yaml` | Publication | High-confidence results with strict thresholds |
| `tmt_labeling.yaml` | TMT/iTRAQ | Optimized for multiplexed labeling |
| `dia_analysis.yaml` | DIA | Optimized for DIA-NN, Spectronaut |
| `exploratory_analysis.yaml` | Exploration | Minimal filtering for data exploration |
