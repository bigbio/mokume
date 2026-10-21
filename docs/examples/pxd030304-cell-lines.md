# PXD030304 cancer cell-line atlas

This full-size example starts from the local QPX 1.1.4 export for **PXD030304**:
228,721,586 DIA-NN feature rows from 5,798 label-free runs representing 949
cancer cell lines. The native Rust kernel builds one DirectLFQ protein matrix;
the Python periphery then reads that matrix to render the atlas, marker, and
tissue-specificity panels. The plotting step does not re-quantify the data.

The source data are not committed to this repository. Set `CELL_LINE_ROOT` to a
local PXD030304 directory containing the QPX feature table and its SDRF.

## Install the runtime

```bash
python -m pip install "mokume[tissuemap]"
```

## Build the Rust DirectLFQ matrix

```bash
export CELL_LINE_ROOT=/path/to/PXD030304
export CELL_LINE_OUT=/tmp/mokume-pxd030304
export CELL_LINE_QPX="$CELL_LINE_ROOT/qpx/PXD030304.feature.parquet"
export CELL_LINE_SDRF="$CELL_LINE_ROOT/mokume/sdrf/PXD030304.sdrf.tsv"

mkdir -p "$CELL_LINE_OUT"
RAYON_NUM_THREADS=24 mokume quantify features2proteins \
    --parquet "$CELL_LINE_QPX" \
    --sdrf "$CELL_LINE_SDRF" \
    --output "$CELL_LINE_OUT/PXD030304.directlfq.csv" \
    --quant-method directlfq \
    --run-normalization none \
    --sample-normalization none \
    --threads 24
```

DirectLFQ performs its own peptide-profile alignment and protein summarization.
The two `none` options mean that no additional run-level or sample-level
normalization is layered onto that result.

## Render the complete showcase

```bash
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
NUMEXPR_NUM_THREADS=24 MPLCONFIGDIR="$CELL_LINE_OUT/matplotlib" \
python docs/examples/render_pxd030304.py \
    "$CELL_LINE_OUT/PXD030304.directlfq.csv" \
    "$CELL_LINE_SDRF" \
    "$CELL_LINE_OUT/showcase" \
    --threads 24 \
    --min-tissue-samples 5
```

The renderer changes positive DirectLFQ values to log2 scale, retains proteins
observed in at least 5% of the 949 cell lines, applies MinDet only for the
PCA/t-SNE embedding, and fixes the random seed at 42. All 949 cell lines enter
the overview. Marker testing and AdaTiSS use the 815 cell lines in the 34
tissues with at least five samples, so singleton and very small tissue groups
cannot drive the biological panels.

## Reproduced result

The Rust run accepted 209,153,286 feature measurements and wrote a 9,530 × 949
protein matrix. Treating its zero sentinel as missing gives 51.93% missing
cells. The visualization filter retains 7,222 proteins, of which 4,468 meet the
stricter missingness threshold used for PCA and t-SNE. PC1 explains 19.5% and
PC2 6.8% of the variance.

![PXD030304 Rust DirectLFQ overview](../assets/pxd030304_rust_overview.png)

*PCA and t-SNE for all 949 cell lines, tissue representation, technical-run
depth versus protein detection, detection distributions for the major tissues,
and the PCA variance profile.*

![PXD030304 tissue specificity and marker panels](../assets/pxd030304_rust_biology.png)

*Wilcoxon marker profiles, the AdaTiSS distribution, tissue-specific protein
counts, and three marker-expression maps. Across the 34 adequately represented
tissues, AdaTiSS classifies 81 proteins as tissue-specific, 905 as
tissue-enriched, 2,554 as housekeeping, and 3,682 as other.*

## Numerical cross-check against the previous kernel

The same QPX input and SDRF were also quantified with the previous Rust kernel
(commit `93c076e`); that table was used only for this comparison and was never
an input to the new matrix or figures. The new table has 22 additional
proteins, all human proteins whose group also names a contaminant entry (for
example albumin P02768, catalase P04040, and cytochrome c P99999); the current
kernel keeps these human entries instead of dropping the whole group. On the
9,508 shared proteins and 949 samples, detection status agrees for all
9,023,092 protein-sample cells. Across the 4,334,238 cells positive in both
tables, the log2 Pearson and Spearman correlations both exceed 0.99999, the
median log2 offset is −0.0003, and the largest absolute log2 difference is
0.0065. These checks show no global scale, sample-order, protein-order, or
detection-pattern failure in the new Rust result; they do not claim
bit-for-bit identity between kernel versions.
