# Batch Correction

Batch effects are systematic technical variations introduced during sample processing that can obscure true biological differences.

## What Are Batch Effects?

Common sources include:

- Different processing days or times
- Different instruments or operators
- Different reagent lots
- Multi-site studies with different labs

mokume uses the **ComBat algorithm** to remove batch effects while preserving biological signal. ComBat is a native Rust implementation, oracle-verified against [inmoose](https://github.com/epigenelabs/inmoose) (~1e-6 / 1e-9). It needs no third-party Python dependency.

## Key Concepts

| Term | Definition | Example |
|------|------------|---------|
| **Batch** | Technical variation to **remove** | Samples from Lab A vs Lab B |
| **Covariate** | Biological signal to **preserve** | Tissue type, sex, disease status |

!!! warning "Why covariates matter"
    Without covariates, batch correction may accidentally remove biological signal that correlates with batch assignments.

    For example, if all liver samples were processed on Day 1 and all brain samples on Day 2, naive batch correction would remove the tissue-specific signal. By specifying tissue as a covariate, ComBat preserves this biological variation.

    ```
    Without covariates:  Batch effect removed, but tissue signal also reduced
    With covariates:     Batch effect removed, tissue signal preserved
    ```

## Using Batch Correction

### Integrated Pipeline (Recommended)

Run ComBat as part of `features2proteins` with `--batch-correction`. Batches are
detected from the sample-name prefix (or an explicit SDRF column), and
Repeat `--batch-covariate` for each SDRF column whose biological signal to preserve.

=== "CLI"

    ```bash
    mokume quantify features2proteins \
        -p data.parquet -o proteins.csv -s experiment.sdrf.tsv \
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
        parquet="data.parquet",
        output="proteins.csv",
        sdrf="experiment.sdrf.tsv",
        quant_method="maxlfq",
        batch_correction=True,
        batch_method="sample-prefix",
        batch_covariate=["characteristics[sex]", "characteristics[organism part]"],
    )
    ```

The covariate columns are extracted from the SDRF with a sample-substring
fallback. Finite numeric columns keep their numeric values; nominal columns
use k-1 one-hot indicators, so categories are not treated as ordered numbers.
A column with any non-numeric value is nominal, even if some labels look like
numbers (for example the cell line 5637). Constant columns are rejected.

Integrated ComBat corrects proteins observed in every matrix sample and leaves
incomplete protein rows unchanged. Linear intensities are corrected on the
log2 scale and returned to the linear scale, so a zero or negative intensity
counts as missing; log2-scale quantifications (`abd`, `ratio`) are corrected as
they are. A protein row that ComBat cannot correct (for example when only one
protein is complete, so its priors are undefined) keeps its measured values.

### Standalone piBAQ correction

To correct already-written piBAQ tables (e.g. when merging datasets), use the
dedicated `correct-batches` command, which runs the sample-prefix ComBat flow
over a folder of piBAQ TSVs:

=== "CLI"

    ```bash
    mokume correct-batches \
        --input ./pibaq_outputs --pattern "*pibaq.tsv" \
        --output corrected.tsv
    ```

=== "Python (wheel)"

    ```python
    import mokume

    mokume.correct_batches(input="./pibaq_outputs", pattern="*pibaq.tsv",
                           output="corrected.tsv")
    ```

Unlike the integrated path, this standalone long-table command requires the
entire protein × sample matrix to be complete and finite. It rejects missing
cells rather than silently filling them with zero.

## Batch Detection Methods

| Method | Description | Example |
|--------|-------------|---------|
| `sample-prefix` | Extract from sample name prefix | `PXD001-S1` &rarr; batch `PXD001` |
| `column` | Explicit values from SDRF column (`--batch-column`) | User-specified |

The protein-matrix CLI exposes only methods with available sample-level
metadata. If the selected data do not contain at least two batches with two
samples each, or contain no complete protein row for ComBat, the command fails
instead of returning the uncorrected matrix as a successful result.

## When to Use Batch Correction

**Recommended scenarios:**

- Combining datasets from multiple studies (e.g., PXD001 + PXD002)
- Samples processed on different days/instruments
- Multi-site studies with different labs

**Requirements:**

- At least 2 samples per batch
- At least 2 batches

## Best Practices

1. **Always specify covariates** when biological groups correlate with batches
2. **Use SDRF characteristics** to identify biological variables to preserve
3. **Apply at protein level** (after quantification) for best results
4. **Verify results** by checking that biological signal (e.g., condition clustering) is preserved
