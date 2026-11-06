# mokume-py

[![Python application](https://github.com/bigbio/mokume/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/python-app.yml)
[![Publish mokume-py](https://github.com/bigbio/mokume/actions/workflows/python-publish.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/mokume-py.svg)](https://badge.fury.io/py/mokume-py)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mokume-py)

## Why "mokume"?

The name comes from [mokume-gane](https://en.wikipedia.org/wiki/Mokume-gane) (木目金), a Japanese metalworking technique that fuses multiple metal layers into distinctive patterns - similar to how this library melds peptide intensities into unified protein expression profiles.

## Overview

**mokume-py** is the pure-Python distribution of mokume. It supports protein
quantification methods including piBAQ, TopN, MaxLFQ, and DirectLFQ, while the
installed import package and console command remain named `mokume`. This
library is an evolution of [ibaqpy](https://github.com/bigbio/ibaqpy).

This directory contains the separately maintained pure-Python implementation.
The Rust kernel under `../rust/` leads new native computation work, so supported
options can differ between the two implementations. See the repository's
[maintenance policy](https://github.com/bigbio/mokume/blob/main/docs/maintenance-scope.md).

## Installation

```bash
pip install mokume-py
```

The base install covers the **core LFQ and piBAQ workflows** — MaxLFQ, Top3/TopN,
FASTA digestion through pyOpenMS, feature/peptide normalization, and parquet/SDRF
I/O. Optional features that require scikit-learn, statsmodels, or anndata remain
available through extras.

```bash
# Differential expression, FDR, DEqMS, LOESS / RLR normalization (statsmodels)
pip install mokume-py[analysis]

# KNN / missForest imputation (scikit-learn)
pip install mokume-py[imputation]

# DirectLFQ quantification method
pip install mokume-py[directlfq]

# Plotting support (QC reports and visualizations)
pip install mokume-py[plotting]

# .h5ad / AnnData export
pip install mokume-py[h5ad]

# TissueMap pipeline (tissue specificity analysis)
pip install mokume-py[tissuemap]

# ComBat batch correction
pip install mokume-py[batch-correction]

# Everything (all optional dependencies)
pip install mokume-py[all]
```

Or install from source:

```bash
git clone https://github.com/bigbio/mokume
cd mokume
pip install ./python
```

Using conda:

```bash
mamba env create -f python/environment.yaml
conda activate mokume
pip install ./python
```

## Library Structure

The import package is `mokume`, organised by stage: `quantification`,
`normalization`, `imputation`, `analysis`, `pipeline`, `preprocessing`,
`tissuemap`, and others. The
[Python API (package) reference](https://github.com/bigbio/mokume/blob/main/docs/reference/python-api-package.md)
lists the modules and their public functions.

## Quantification Methods

| Method | Description | Requires FASTA | Class | Optional |
|--------|-------------|----------------|-------|----------|
| **piBAQ** | Paralog-aware iBAQ | Yes | `peptides_to_protein()` | No |
| **TopN** | Average of N most intense peptides (configurable N) | No | `TopNQuantification` | No |
| **MaxLFQ** | Delayed normalization with parallelization | No | `MaxLFQQuantification` | No |
| **DirectLFQ** | Intensity traces with hierarchical alignment | No | `DirectLFQQuantification` | Yes (`directlfq`) |
| **Sum** | Sum of all peptide intensities | No | `AllPeptidesQuantification` | No |
| **Median** | Median of peptide intensities | No | Built-in | No |
| **Ratio** | Log2 sample/reference per plex (PS protocol, TMT) | No | `RatioQuantification` | No |
| **TMT Abundance** (`abd`) | Median of log2 peptide intensities | No | `TMTAbundanceQuantification` | No |
| **TMT Reporter** (`intensity`) | Sum of raw reporter intensities | No | `TMTReporterIntensityQuantification` | No |
| **Spectral Count** | Count of PSM rows per (protein, sample) | No | `SpectralCountQuantification` | No |
| **Peptide Count** | Count of distinct peptides per (protein, sample) | No | `PeptideCountQuantification` | No |

MaxLFQ and DirectLFQ are separate methods: `maxlfq` never dispatches to the
DirectLFQ package.

> **Note:** `Top3Quantification` is an alias for `TopNQuantification(n=3)`.

### MaxLFQ Algorithm Details

`MaxLFQQuantification` is an independent MaxLFQ ratio solver for prepared
linear intensities:

- Computes the median log ratio for each sample pair from the peptide species
  they share, requiring `min_ratio_count` shared species per pair (default 2;
  `min_peptides` is its legacy alias)
- Solves the pairwise ratios for one profile per protein and scales it to the
  protein's total input intensity; samples without a valid ratio stay missing
  rather than falling back to a median
- Uses `PeptideSequence` + `PrecursorCharge` as the species when both columns
  are present
- Runs proteins in parallel with joblib (`threads`, `-1` for all cores);
  `stabilize=True` enables the overlap-dependent large-ratio stabilization of
  Eq. 5 in the MaxLFQ paper

`force_builtin` is accepted only for compatibility and has no effect.

## Differential Expression Analysis

mokume provides modern empirical Bayes and bootstrap-based statistical methods for identifying differentially expressed proteins with high sensitivity while controlling false positives. All methods are **pure-Python** reimplementations — no R or rpy2 required.

### DE Methods

- **LimROTS** — Combines limma empirical Bayes variance stabilization with ROTS bootstrap-optimized test statistic and permutation-based FDR. Best overall sensitivity; recommended for MaxLFQ data.
- **DEqMS** — Extends limma with peptide-count weighting via spectra count eBayes. Better false-positive control on noisy data; recommended for DirectLFQ data.
- **proDA** — Probabilistic dropout-aware analysis that models missing values as informative dropout events rather than discarding them.
- **limma** — Stable empirical Bayes baseline (`lmFit → contrasts.fit → eBayes → topTable`).
- **ROTS** — Bootstrap-optimised reproducibility-oriented test statistic without empirical Bayes shrinkage.
- **Ensemble** — Top-k consensus across multiple DE methods with Cauchy-combined p-values.

FDR correction: **BH** (Benjamini-Hochberg, default) or **IHW** (Independent Hypothesis Weighting).

### Python API

```python
from mokume.analysis import DifferentialExpression

import pandas as pd
protein_df = pd.read_csv("maxlfq_proteins.csv")
sample_to_condition = {
    "Control_R1": "Control", "Control_R2": "Control", "Control_R3": "Control",
    "Treatment_R1": "Treatment", "Treatment_R2": "Treatment", "Treatment_R3": "Treatment",
}

# LimROTS (recommended for MaxLFQ)
de = DifferentialExpression(method="limrots", log2fc_threshold=1.0, fdr_threshold=0.05)
result = de.run(protein_df, sample_to_condition, ("Treatment", "Control"))

# DEqMS (recommended for DirectLFQ, requires peptide counts)
de = DifferentialExpression(method="deqms", peptide_counts=peptide_counts)
result = de.run(protein_df, sample_to_condition, ("Treatment", "Control"))

# Multiple contrasts
contrasts = [("Treatment", "Control"), ("Drug", "Control")]
results = de.run_comparisons(protein_df, sample_to_condition, contrasts)
# Returns dict: {"Treatment-Control": DataFrame, "Drug-Control": DataFrame}
```

Output columns: `ProteinName`, `log2FC`, `pvalue`, `adj_pvalue`, `significance` (`UP` / `DOWN` / `Unchanged` / `NotTested`).

### Missing Value Imputation

The `features2proteins` pipeline integrates 13 imputation methods via
`--impute --impute-method <name>` (operates on the protein matrix in
log2 space, after coverage filtering and before batch correction).

Standalone API examples:

```python
from mokume.imputation.censored import impute_censored

# MinProb: draw from a normal at a low per-sample quantile (recommended for MNAR)
imputed = impute_censored(log2_matrix, method="minprob", quantile=0.01, tune_sigma=1.0)

# MinDet: replace NaN with per-column quantile
imputed = impute_censored(log2_matrix, method="mindet", quantile=0.01)

# KNN: k-nearest neighbor imputation
imputed = impute_censored(log2_matrix, method="knn", n_neighbors=10)
```

### LOESS Normalization

```python
from mokume.normalization import loess_normalize

# Correct intensity-dependent biases via locally-weighted regression
normalized = loess_normalize(log2_matrix)
```

## End-to-End Pipeline (`features2proteins`)

The `features2proteins` command runs the complete workflow from quantms feature files to protein quantification with optional differential expression:

```bash
# MaxLFQ quantification + automatic DE (LimROTS)
mokume features2proteins \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --de \
    --de-contrasts "Treatment-Control,Drug-Control" \
    -o output_dir/

# DirectLFQ quantification + automatic DE (DEqMS)
mokume features2proteins \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --quant-method directlfq \
    --de \
    --de-contrasts "Treatment-Control" \
    --de-log2fc 1.0 \
    --de-fdr 0.05 \
    -o output_dir/

# Explicit DE method override
mokume features2proteins \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --de \
    --de-method deqms \
    --de-fdr-method ihw \
    --de-contrasts "Treatment-Control" \
    -o output_dir/
```

## CLI Usage

### Peptides to Protein Quantification

```bash
# Using piBAQ (default) - requires FASTA
mokume peptides2protein --method pibaq \
    -f proteome.fasta \
    -p peptides.csv \
    -o proteins-pibaq.tsv

# Using TopN - no FASTA required (N can be any number: top3, top5, top10, etc.)
mokume peptides2protein --method top3 \
    -p peptides.csv \
    -o proteins-top3.tsv

# Using TopN with different N values
mokume peptides2protein --method top5 \
    -p peptides.csv \
    -o proteins-top5.tsv

mokume peptides2protein --method top10 \
    -p peptides.csv \
    -o proteins-top10.tsv

# Using MaxLFQ with parallelization
mokume peptides2protein --method maxlfq \
    --threads 4 \
    -p peptides.csv \
    -o proteins-maxlfq.tsv

# Using DirectLFQ (requires: pip install mokume-py[directlfq])
mokume peptides2protein --method directlfq \
    -p peptides.csv \
    -o proteins-directlfq.tsv

# Using Sum of all peptides
mokume peptides2protein --method sum \
    -p peptides.csv \
    -o proteins-sum.tsv
```

### Full piBAQ with TPA and ProteomicRuler

```bash
mokume peptides2protein \
    -f proteome.fasta \
    -p peptides.csv \
    -e Trypsin \
    --normalize \
    --tpa \
    --ruler \
    --ploidy 2 \
    --cpc 200 \
    --organism human \
    --output proteins-pibaq.tsv \
    --verbose \
    --qc_report QC.pdf
```

### Features to Peptides

```bash
mokume features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --remove_decoy_contaminants \
    --remove_low_frequency_peptides \
    --run-normalization median \
    --sample-normalization globalMedian \
    --output peptides-norm.csv
```

### Preprocessing Filters

mokume supports comprehensive preprocessing filters via YAML/JSON configuration files or CLI options.

**Generate example filter configuration:**

```bash
mokume features2peptides --generate-filter-config filters.yaml
```

**Use filter configuration file:**

```bash
mokume features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --output peptides-filtered.csv
```

**CLI filter overrides (take precedence over config file):**

```bash
mokume features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --filter-config filters.yaml \
    --filter-min-intensity 1000 \
    --filter-cv-threshold 0.3 \
    --filter-charge-states "2,3,4" \
    --filter-max-missed-cleavages 2 \
    --filter-protein-fdr 0.01 \
    --output peptides-filtered.csv
```

**CLI-only filtering (no config file):**

```bash
mokume features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --filter-min-intensity 500 \
    --filter-min-unique-peptides 2 \
    --filter-max-missing-rate 0.5 \
    --output peptides-filtered.csv
```

### Batch Correction

```bash
# Standalone batch correction (post-quantification)
mokume correct-batches \
    -f pibaq_folder/ \
    -p "*pibaq.tsv" \
    -o corrected_pibaq.tsv \
    --export_anndata
```

> **Note:** For integrated batch correction during quantification, use the Python API
> with `PipelineConfig(batch_correction=True, ...)`. See Python API section below.

### TissueMap Pipeline

Per-dataset tissue proteome analysis with AdaTiSS tissue specificity scoring.

```bash
# Generate default configuration template
mokume tissuemap --generate-config tissuemap.yaml

# Run on a single dataset
mokume tissuemap \
    --scan-dir QPX_data/tissues-mq/PXD016999 \
    --output-dir ./results

# Run on multiple datasets (auto-discovers PXD* subdirectories)
mokume tissuemap \
    --scan-dir QPX_data/tissues-mq \
    --tmt-dataset PXD016999 \
    --output-dir ./results \
    --n-jobs 8

# Run with custom YAML configuration
mokume tissuemap \
    --scan-dir QPX_data/tissues-mq \
    --config tissuemap.yaml \
    --output-dir ./results
```

**Outputs per dataset:**

| File | Description |
|------|-------------|
| `<ds_id>.corrected.h5ad` | Batch-corrected AnnData with embeddings |
| `<ds_id>.ts_scores.h5ad` | Tissue specificity score matrix |
| `protein_ts_scores.csv` | Per-protein TS scores and enrichment categories |
| `plots/` | PCA scree, tissue atlas, dendrogram, marker heatmap, dotplot, TS distribution |

### t-SNE Visualization

```bash
mokume tsne-visualization \
    -f protein_folder/ \
    -o proteins.tsv
```

## Python API

### Quantification Methods

```python
import pandas as pd
from mokume.quantification import (
    TopNQuantification,
    MaxLFQQuantification,
    AllPeptidesQuantification,
    get_quantification_method,
    is_directlfq_available,
    peptides_to_protein,  # piBAQ function
)

# Load peptide data
peptides = pd.read_csv("peptides.csv")

# --- TopN Quantification (generic, configurable N) ---
# Top3 quantification
top3 = TopNQuantification(n=3)
result = top3.quantify(
    peptides,
    protein_column="ProteinName",
    peptide_column="PeptideSequence",
    intensity_column="NormIntensity",
    sample_column="SampleID",
)

# Top5 quantification
top5 = TopNQuantification(n=5)
result = top5.quantify(peptides, protein_column="ProteinName", ...)

# Top10 quantification
top10 = TopNQuantification(n=10)
result = top10.quantify(peptides, protein_column="ProteinName", ...)

# --- MaxLFQ Quantification ---
maxlfq = MaxLFQQuantification(
    min_ratio_count=2,      # Shared peptide species required for each sample pair
    threads=4,              # Use 4 parallel cores (-1 for all cores)
)
result = maxlfq.quantify(peptides, protein_column="ProteinName", ...)

# Run-level quantification
result = maxlfq.quantify(
    peptides,
    protein_column="ProteinName",
    sample_column="SampleID",
    run_column="Run",  # Optional: quantify at run level instead of sample level
)

# --- DirectLFQ Quantification (standalone, optional dependency) ---
if is_directlfq_available():
    from mokume.quantification import DirectLFQQuantification
    directlfq = DirectLFQQuantification(min_nonan=2)
    result = directlfq.quantify(peptides, protein_column="ProteinName", ...)

# --- Sum of All Peptides ---
sum_quant = AllPeptidesQuantification()
result = sum_quant.quantify(peptides, protein_column="ProteinName", ...)

# --- Factory Function ---
# The factory function automatically parses TopN from method name
method = get_quantification_method("top3")   # Creates TopNQuantification(n=3)
method = get_quantification_method("top5")   # Creates TopNQuantification(n=5)
method = get_quantification_method("top10")  # Creates TopNQuantification(n=10)
method = get_quantification_method("maxlfq", min_peptides=2, threads=-1)
result = method.quantify(peptides, ...)

# --- Check available methods ---
from mokume.quantification import list_quantification_methods
print(list_quantification_methods())
# Includes pibaq, topN, maxlfq, directlfq, sum, and the TMT methods.

# --- piBAQ with Full Pipeline ---
peptides_to_protein(
    fasta="proteome.fasta",
    peptides="peptides.csv",
    enzyme="Trypsin",
    normalize=True,
    tpa=True,
    ruler=True,
    ploidy=2,
    cpc=200,
    organism="human",
    output="proteins-pibaq.tsv",
    min_aa=7,
    max_aa=30,
    verbose=True,
    qc_report="QC.pdf",
)
```

### Normalization

```python
from mokume.normalization.peptide import peptide_normalization

# Full peptide normalization pipeline
peptide_normalization(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    min_aa=7,
    min_unique=2,
    remove_ids=None,
    remove_decoy_contaminants=True,
    remove_low_frequency_peptides=True,
    output="peptides-norm.csv",
    skip_normalization=False,
    nmethod="median",      # Feature normalization: mean, median, max, global, max_min, iqr, none
    pnmethod="globalMedian",  # Sample normalization: globalMedian, conditionMedian, hierarchical, none
    log2=True,
    save_parquet=False,
)
```

### Preprocessing Filters

```python
from mokume.preprocessing.filters import (
    load_filter_config,
    save_filter_config,
    generate_example_config,
    get_filter_pipeline,
    FilterPipeline,
)
from mokume.model.filters import PreprocessingFilterConfig

# Generate example configuration file
generate_example_config("filters.yaml")

# Load configuration from file
config = load_filter_config("filters.yaml")

# Create configuration programmatically
config = PreprocessingFilterConfig(
    name="custom_filters",
    enabled=True,
    log_filtered_counts=True,
)
config.intensity.min_intensity = 1000.0
config.intensity.cv_threshold = 0.3
config.peptide.allowed_charge_states = [2, 3, 4]
config.peptide.exclude_modifications = ["Oxidation"]
config.protein.min_unique_peptides = 2
config.run_qc.max_missing_rate = 0.5

# Apply CLI-style overrides
config.apply_overrides({
    "min_intensity": 500,
    "charge_states": [2, 3],
    "max_missing_rate": 0.3,
})

# Save configuration
save_filter_config(config, "my_filters.yaml")

# Create and use filter pipeline directly
pipeline = get_filter_pipeline(config)

import pandas as pd
df = pd.read_csv("peptides.csv")
filtered_df, results = pipeline.apply(df)

# Check filter results
for result in results:
    print(f"{result.filter_name}: removed {result.removed_count} ({result.removal_rate:.1%})")

# Get pipeline summary
summary = pipeline.summary(results)
print(f"Total removed: {summary['total_removed']} / {summary['total_input']}")

# Use filters with peptide_normalization
from mokume.normalization.peptide import peptide_normalization

peptide_normalization(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    output="peptides-filtered.csv",
    nmethod="median",
    pnmethod="globalMedian",
    filter_config=config,  # Pass filter configuration
)
```

### Batch Correction

Batch effects are systematic technical variations introduced during sample processing that can obscure true biological differences. Common sources include:

- Different processing days or times
- Different instruments or operators
- Different reagent lots
- Samples processed in different labs

**mokume** uses the ComBat algorithm (via [inmoose](https://github.com/epigenelabs/inmoose)) to remove batch effects while preserving biological signal.

#### Key Concepts

| Term | Definition | Example |
|------|------------|---------|
| **Batch** | Technical variation to **REMOVE** | Samples from Lab A vs Lab B |
| **Covariate** | Biological signal to **PRESERVE** | Tissue type, sex, disease status |

**Why covariates matter:** Without covariates, batch correction may accidentally remove biological signal that correlates with batch assignments. For example, if all liver samples were processed on Day 1 and all brain samples on Day 2, naive batch correction would remove the tissue-specific signal. By specifying tissue as a covariate, ComBat preserves this biological variation.

```text
Without covariates:  Batch effect removed, but tissue signal also reduced
With covariates:     Batch effect removed, tissue signal preserved
```

**Requires optional dependency:** `pip install mokume-py[batch-correction]`

#### Integrated Pipeline Approach (Recommended)

```python
from mokume.pipeline import QuantificationPipeline, PipelineConfig
from mokume.pipeline.config import (
    InputConfig, QuantificationConfig, BatchCorrectionConfig,
)

# Batch correction with covariates from SDRF
config = PipelineConfig(
    input=InputConfig(
        parquet="data.parquet",
        sdrf="experiment.sdrf.tsv",
    ),
    quantification=QuantificationConfig(method="maxlfq"),
    batch=BatchCorrectionConfig(
        enabled=True,
        method="sample_prefix",
        covariates=["characteristics[sex]", "characteristics[organism part]"],
    ),
)

pipeline = QuantificationPipeline(config)
proteins = pipeline.run()  # Returns batch-corrected protein matrix
```

#### Low-Level API

```python
from mokume.postprocessing import (
    apply_batch_correction,
    detect_batches,
    extract_covariates_from_sdrf,
    pivot_wider,
    pivot_longer,
)

# Reshape to wide format (proteins x samples)
df_wide = pivot_wider(df, row_name="ProteinName", col_name="SampleID", values="PiBAQ")

# Detect batches from sample names
batch_indices = detect_batches(
    sample_ids=df_wide.columns.tolist(),
    method="sample_prefix",  # PXD001-S1 → batch=PXD001
)

# Extract covariates from SDRF (biological signal to preserve)
covariates = extract_covariates_from_sdrf(
    "experiment.sdrf.tsv",
    sample_ids=df_wide.columns.tolist(),
    covariate_columns=["characteristics[sex]", "characteristics[tissue]"],
)

# Apply ComBat batch correction
df_corrected = apply_batch_correction(
    df=df_wide,
    batch=batch_indices,
    covs=covariates,  # Preserve biological signal
)
```

#### Batch Detection Methods

| Method | Description | Example |
|--------|-------------|---------|
| `sample_prefix` | Extract from sample name prefix | `PXD001-S1` → `PXD001` |
| `run` | Use run/reference file name | Each file is a batch |
| `column` | Explicit values from SDRF column | User-specified |

#### When to Use Batch Correction

**Recommended scenarios:**

- Combining datasets from multiple studies (e.g., PXD001 + PXD002)
- Samples processed on different days/instruments
- Multi-site studies with different labs

**Requirements:**

- At least 2 samples per batch (ComBat requirement)
- At least 2 batches to correct

#### Best Practices

1. **Always specify covariates** when biological groups correlate with batches
2. **Use SDRF characteristics** to identify biological variables to preserve
3. **Apply at protein level** (after quantification) for best results
4. **Verify results** by checking that biological signal (e.g., tissue clustering) is preserved

### Data Reshaping

```python
from mokume.postprocessing.reshape import (
    pivot_wider,
    pivot_longer,
    remove_samples_low_protein_number,
    remove_missing_values,
    describe_expression_metrics,
)

# Long to wide format
df_wide = pivot_wider(df, row_name="ProteinName", col_name="SampleID", values="PiBAQ")

# Wide to long format
df_long = pivot_longer(df_wide, row_name="ProteinName", col_name="SampleID", values="PiBAQ")

# Quality filtering
df_filtered = remove_samples_low_protein_number(df, min_protein_num=100)
df_filtered = remove_missing_values(df, missingness_percentage=20, expression_column="PiBAQ")

# Get expression statistics
metrics = describe_expression_metrics(df)
```

### Creating AnnData Objects

```python
from mokume.io.parquet import create_anndata

# Create AnnData from long-format DataFrame
adata = create_anndata(
    df,
    obs_col="SampleID",           # Observation (sample) column
    var_col="ProteinName",        # Variable (protein) column
    value_col="PiBAQ",             # Main data values
    layer_cols=["PiBAQNorm", "PiBAQLog", "PiBAQBec"],  # Additional layers
    obs_metadata_cols=["Condition"],  # Sample metadata
    var_metadata_cols=["GeneName"],   # Protein metadata
)

# Save to h5ad
adata.write("proteins.h5ad")
```

### FASTA Handling

```python
from mokume.io.fasta import (
    load_fasta,
    digest_protein,
    extract_fasta,
    get_protein_molecular_weights,
)

# Load FASTA file
proteins = load_fasta("proteome.fasta")

# Digest a single protein sequence
peptides = digest_protein(
    sequence="MKWVTFISLLFLFSSAYS...",
    enzyme="Trypsin",
    min_aa=7,
    max_aa=30,
)

# Extract info for specific proteins
unique_peptide_counts, mw_dict, found_proteins = extract_fasta(
    fasta="proteome.fasta",
    enzyme="Trypsin",
    proteins=["P12345", "P67890"],
    min_aa=7,
    max_aa=30,
    tpa=True,
)

# Get molecular weights
mw_dict = get_protein_molecular_weights("proteome.fasta", ["P12345", "P67890"])
```

### Organism Metadata

```python
from mokume.model.organism import OrganismDescription

# Get available organisms
organisms = OrganismDescription.registered_organisms()
# ['human', 'mouse', 'yeast', 'drome', 'caeel', 'schpo']

# Get organism description
human = OrganismDescription.get("human")
print(human.genome_size)      # Genome size in base pairs
print(human.histone_entries)  # List of histone protein accessions
```

## Computed Values Reference

| Column | Description | Method |
|--------|-------------|--------|
| `PiBAQ` | Proteotypic plus allocated shared intensity / owned theoretical peptides | piBAQ |
| `PiBAQNorm` | `PiBAQ / sum(PiBAQ)` per sample | piBAQ |
| `PiBAQLog` | `10 + log10(PiBAQNorm)` | piBAQ |
| `PiBAQPpb` | `PiBAQNorm * 100,000,000` | piBAQ |
| `PiBAQBec` | Batch effect corrected | piBAQ + ComBat |
| `TPA` | `NormIntensity / MolecularWeight` | piBAQ |
| `CopyNumber` | Protein copies per cell | ProteomicRuler |
| `Concentration[nM]` | Protein concentration | ProteomicRuler |
| `TopNIntensity` | Average of top N peptides (e.g., Top3Intensity, Top5Intensity) | TopN |
| `MaxLFQIntensity` | MaxLFQ algorithm result | MaxLFQ |
| `DirectLFQIntensity` | DirectLFQ intensity traces | DirectLFQ |
| `SumIntensity` | Sum of all peptides | Sum |

## Normalization Methods

### Feature-Level Normalization (`--run-normalization`)

| Method | Description |
|--------|-------------|
| `median` | Normalize by median across MS runs |
| `mean` | Normalize by mean across MS runs |
| `max` | Normalize by maximum intensity within each run |
| `global` | Normalize by total intensity within each run |
| `max_min` | Apply min-max scaling |
| `iqr` | Normalize by interquartile range |
| `none` | Skip feature normalization |

### Sample-Level Normalization (`--sample-normalization`)

| Method | Description |
|--------|-------------|
| `globalMedian` | Adjust all samples to global median |
| `conditionMedian` | Adjust samples within each condition to median |
| `hierarchical` | DirectLFQ-style hierarchical clustering normalization |
| `none` | Skip sample normalization |

## Preprocessing Filters

mokume provides a comprehensive filter system for quality control. Filters can be configured via YAML/JSON files or CLI options.

### Filter Categories

#### Intensity Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| MinIntensityFilter | `min_intensity` | 0.0 | Remove features below threshold |
| CVThresholdFilter | `cv_threshold` | null | Max CV across replicates |
| ReplicateAgreementFilter | `min_replicate_agreement` | 1 | Min replicates with detection |
| QuantileFilter | `quantile_lower/upper` | 0.0/1.0 | Remove intensity outliers |

#### Peptide Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| PeptideLengthFilter | `min/max_peptide_length` | 7/50 | Peptide length range |
| ChargeStateFilter | `allowed_charge_states` | null | Allowed charges (e.g., [2,3,4]) |
| ModificationFilter | `exclude_modifications` | [] | Remove specific modifications |
| MissedCleavageFilter | `max_missed_cleavages` | null | Max missed cleavages |
| SequencePatternFilter | `exclude_sequence_patterns` | [] | Regex patterns to exclude |

#### Protein Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| ContaminantFilter | `remove_contaminants/decoys` | true | Remove contaminants/decoys |
| MinPeptideFilter | `min_unique_peptides` | 2 | Min unique peptides per protein |
| RazorPeptideFilter | `razor_peptide_handling` | "keep" | Handle shared peptides |

#### Run/Sample QC Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| RunIntensityFilter | `min_total_intensity` | 0.0 | Min total intensity per run |
| MinFeaturesFilter | `min_identified_features` | 0 | Min features per run |
| MissingRateFilter | `max_missing_rate` | 1.0 | Max missing value rate |

### Example Filter Configurations

We provide several pre-configured filter templates for common use cases in [`tests/example/filters/`](https://github.com/bigbio/mokume/tree/main/python/tests/example/filters):

| Configuration | Use Case | Description |
|---------------|----------|-------------|
| [`basic_qc.yaml`](https://github.com/bigbio/mokume/blob/main/python/tests/example/filters/basic_qc.yaml) | General QC | Minimal filtering for standard experiments |
| [`stringent_filtering.yaml`](https://github.com/bigbio/mokume/blob/main/python/tests/example/filters/stringent_filtering.yaml) | Publication | High-confidence results with strict thresholds |
| [`tmt_labeling.yaml`](https://github.com/bigbio/mokume/blob/main/python/tests/example/filters/tmt_labeling.yaml) | TMT/iTRAQ | Optimized for multiplexed labeling experiments |
| [`dia_analysis.yaml`](https://github.com/bigbio/mokume/blob/main/python/tests/example/filters/dia_analysis.yaml) | DIA | Optimized for DIA-NN, Spectronaut analysis |
| [`exploratory_analysis.yaml`](https://github.com/bigbio/mokume/blob/main/python/tests/example/filters/exploratory_analysis.yaml) | Exploration | Minimal filtering for data exploration |

#### Example: Basic QC Configuration

```yaml
# basic_qc.yaml - Minimal filtering for standard experiments
name: basic_qc
enabled: true

intensity:
  min_intensity: 0.0

peptide:
  min_peptide_length: 7
  max_peptide_length: 50

protein:
  min_unique_peptides: 2
  remove_contaminants: true
  remove_decoys: true
  contaminant_patterns:
    - CONTAMINANT
    - CONTAM_
    - ENTRAP
    - DECOY
```

Use these configurations directly:

```bash
mokume features2peptides \
    -p features.parquet \
    --filter-config tests/example/filters/basic_qc.yaml \
    -o peptides.csv
```

## Data Processing Pipeline

### Feature Processing (`features2peptides`)

1. Parse protein identifiers and retain unique peptides
2. Remove entries with empty intensity or condition
3. Filter peptides by minimum amino acids
4. Remove low-confidence proteins (< min_unique peptides)
5. Optionally remove decoys, contaminants, and specified proteins
6. Normalize at feature level between MS runs
7. Merge peptidoforms across fractions and technical replicates
8. Normalize at sample level
9. Remove low-frequency peptides
10. Assemble peptidoforms to peptides
11. Optional log2 transformation

### piBAQ Calculation

1. Load peptide intensity data
2. Digest the FASTA and discover shared-peptide protein families
3. Allocate each shared-peptide intensity proportionally to unique-anchor
   signal, or equally when no anchor signal exists
4. Add each member's proteotypic and allocated shared intensities
5. Divide by that member's owned theoretical-peptide count
6. Optionally calculate normalized piBAQ, TPA, copy number, and concentration

### TissueMap Pipeline API

```python
from pathlib import Path

from mokume.tissuemap.config import (
    TissueMapConfig, InputConfig, OutputConfig, load_config,
)
from mokume.tissuemap.pipeline import TissueMapPipeline

# --- Option 1: Programmatic configuration ---
config = TissueMapConfig(
    n_jobs=8,
    input=InputConfig(
        scan_dir=Path("QPX_data/tissues-mq/PXD016999"),
        tmt_datasets=["PXD016999"],
    ),
    output=OutputConfig(output_dir=Path("./results")),
)
pipeline = TissueMapPipeline(config)
pipeline.run()

# --- Option 2: YAML configuration ---
config = load_config(Path("tissuemap.yaml"), overrides={
    "input.scan_dir": "QPX_data/tissues-mq",
    "output.output_dir": "./results",
})
pipeline = TissueMapPipeline(config)
pipeline.run()
```

#### TissueMap Configuration Reference

| Section | Parameter | Default | Description |
| ------- | --------- | ------- | ----------- |
| `input` | `scan_dir` | `.` | PXD dataset directory |
| `input` | `tmt_datasets` | `[]` | Force TMT dataset IDs |
| `input` | `min_tissue_samples` | `1` | Min samples per tissue |
| `filtering` | `max_nan_frac` | `0.95` | Max NaN fraction |
| `filtering` | `remove_contaminants` | `true` | Remove contaminants |
| `tissue_specificity` | `use_pure_mad` | `true` | MAD-based estimation |
| `tissue_specificity` | `ts_enriched_threshold` | `null` | Auto, floor 2.5 |
| `tissue_specificity` | `ts_specific_threshold` | `null` | Auto, floor 4.0 |
| `tissue_specificity` | `ts_housekeeping_threshold` | `null` | Auto, floor 2 |
| `embedding` | `pca_components` | `50` | PCA components |
| `embedding` | `tsne_perplexity` | `15.0` | t-SNE perplexity |
| `plotting` | `dpi` | `250` | Plot resolution |
| `plotting` | `save_pdf` | `true` | Also save PDF |
| `output` | `output_dir` | `tissuemap_output` | Output directory |

#### Enrichment Categories

| Category | Criteria |
| -------- | -------- |
| **tissue-specific** | TS ≥ 4.0 in one tissue only |
| **tissue-enriched** | At least one TS ≥ 2.5 |
| **house-keeping** | All tissues, all \|TS\| < 2.0 |
| **other** | Everything else |

## Agentic Recommendation

Agentic recommendation is provided by the installable Mokume Plugin over the
default Rust-backed `mokume[plugin]` distribution. The plugin host owns the
model and credentials. See the
[Mokume Plugin guide](https://github.com/bigbio/mokume/blob/main/docs/user-guide/agentic-plugin.md).

## Citation

> Zheng P, Audain E, Webel H, Dai C, Klein J, Hitz MP, Sachsenberg T, Bai M, Perez-Riverol Y. Ibaqpy: A scalable Python package for baseline quantification in proteomics leveraging SDRF metadata. J Proteomics. 2025;317:105440. doi: [10.1016/j.jprot.2025.105440](https://doi.org/10.1016/j.jprot.2025.105440).

> Wang H, Dai C, Pfeuffer J, Sachsenberg T, Sanchez A, Bai M, Perez-Riverol Y. Tissue-based absolute quantification using large-scale TMT and LFQ experiments. Proteomics. 2023;23(20):e2300188. doi: [10.1002/pmic.202300188](https://doi.org/10.1002/pmic.202300188).

## License

MIT License - see [LICENSE](https://github.com/bigbio/mokume/blob/main/LICENSE) for details.

## Credits

- [Julianus Pfeuffer](https://github.com/jpfeuffer)
- [Yasset Perez-Riverol](https://github.com/ypriverol)
- [Hong Wang](https://github.com/WangHong007)
- [Ping Zheng](https://github.com/zprobot)
- [Joshua Klein](https://github.com/mobiusklein)
- [Enrique Audain](https://github.com/enriquea)
