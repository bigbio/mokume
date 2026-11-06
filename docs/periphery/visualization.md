# Visualization & Reports

mokume's plotting and reporting live in the Python periphery of the wheel: static
plots (matplotlib/seaborn) and interactive HTML reports (plotly) for quality
control and differential expression analysis. The Rust kernel is pure compute —
it writes the protein matrix and the per-contrast DE result CSVs, and the
periphery commands **read those tables** to render figures, so the cells in the
plots always match the cells in the kernel output (the periphery never
recomputes the numbers).

!!! note "Analysis extra"
    Plotting and reports are exposed by the wheel's unified CLI but remain
    implemented in its Python periphery. Install the shared analysis extra:
    ```bash
    pip install "mokume[analysis]"  # plots, reports, QC, comparison, and missforest
    pip install "mokume[all]"       # every optional capability
    ```

## Static Figure Style

Matplotlib figures use the first installed font from Arial, Helvetica,
Liberation Sans and Nimbus Sans. The caller's configured fonts remain available
for missing glyphs, including Chinese sample names. Body text and axis labels
are 10 pt, ticks and legends 9 pt, and regular-weight titles 11 pt. Dense labels
may use smaller type, down to 6 pt. Multi-panel letters are lowercase and bold.

Static plots have no background grid. Only the left and bottom axes are shown,
with outward ticks. Bitmap exports default to 300 dpi; TissueMap's `--dpi` can
override this. PDF and PostScript embed TrueType fonts, and SVG keeps text as
text. Dense protein and peptide scatter layers and marker t-SNE layers are
rasterized; PCA and atlas sample markers, labels and axes remain vector graphics.
Default canvases match journal columns: about 3.5 in for single panels and at
most 7.2 in for multi-panel figures, while heatmaps and marker grids grow with
their rows and columns. Protein points are small, translucent and unringed so
overlaps read as density. No marker has a white ring: sample points, profile
and variance-curve markers and UpSet dots have no outline, and highlighted
proteins are larger with a thin dark outline. These settings apply only while
Mokume creates and exports a figure; matplotlib defaults are restored.
Interactive HTML reports follow the same point rules but keep their own fonts
and layout.

## DE Plots from features2proteins Output

First run the kernel to write the protein matrix and one DE result CSV per
contrast:

```bash
mokume quantify features2proteins \
    -p features.parquet -o proteins.csv -s experiment.sdrf.tsv \
    --quant-method maxlfq \
    --de-contrast "NASH" "HL" \
    --de-contrast "NASH" "Control" \
    --de-output de_results
```

Then render the plots from those CSVs with the periphery command. Repeat
`--contrast KEY A B CSV` for every comparison:

```bash
mokume plot de \
    --protein-matrix proteins.csv \
    --outdir plots \
    --sdrf experiment.sdrf.tsv \
    --volcano --heatmap --ma \
    --highlight-protein COL10A1 \
    --highlight-protein FN1 \
    --highlight-protein ALB \
    --contrast NASH-HL NASH HL de_results_NASH-HL.csv \
    --contrast NASH-Control NASH Control de_results_NASH-Control.csv
```

This generates:

- `plots/volcano_NASH-HL.png` -- Volcano plot for each contrast
- `plots/heatmap_NASH-HL.png` -- Per-contrast heatmap showing the top 50 significant proteins by ascending adjusted p-value, breaking ties by descending |log2FC|. Only the two compared conditions are included. Skipped if no significant proteins exist for that contrast.

Volcano heights are `-log10(p-value)`, using the raw `pvalue` column. Point
colors use the adjusted p-value and |log2FC| significance cutoffs. The horizontal
guide marks the largest raw p-value among proteins with adjusted p-value below
`--fdr`, including those below the fold-change cutoff. It is omitted when no
protein passes the adjusted p-value cutoff.

The `--volcano` / `--heatmap` / `--ma` flags select which plots to render;
`--log2fc` (default 0.5) and `--fdr` (default 0.05) mirror the
kernel's `--de-log2fc` / `--de-fdr` so the significance cutoffs match. Volcano,
MA and heatmap require at least one `--contrast`; heatmap requires `--sdrf`.
Invalid option scopes are rejected instead of producing an empty plot directory.

Render PCA as a separate command because it produces one output file:

```bash
mokume plot pca \
    --protein-matrix proteins.csv \
    --sdrf experiment.sdrf.tsv \
    --output pca_conditions.png
```

PCA uses complete-case proteins and reports their count below the title.
It uses `random_state=0` so repeated runs are reproducible. Small
plots (at most 30 samples) include sample labels, placed from the outermost
sample inward; a label that would cover a placed label or another sample's
point is omitted. HTML PCA plots in the QC and comparison reports apply the
same rule after each zoom or resize, and hover always names the sample.
Conditions are distinguished only by color. All category markers are small
circles with opacity 0.85 and no outline: static scatter area is 20 pt² and
HTML marker diameter is 7 px. The first six colors use the shared palette.
Additional colors are selected greedily from an in-gamut OKLCH grid
(lightness 0.43-0.77, chroma 0.10-0.17, the seeds' range) to maximize
separation in OKLab. Palettes are deterministic and preserve their prefixes as
the number of categories grows. Plex information remains in hover text.

Above 12 categories, static and HTML PCA label each category at the sample
nearest its coordinate-wise median, replacing sample-name labels. Legends move
below the plot in columns. Static PCA uses a 7.2-inch-wide figure with an axes
area of about 6.4 by 4.5 inches. Labels stay inside that axes area; the canvas
grows downwards for the legend. Long category names may wrap. TissueMap also
places legends below the atlas above 12 proteomic groups.

Category, atlas-group and highlighted-protein labels try nearby positions in
eight directions, moving outwards from the anchor. They avoid earlier labels
and prefer positions clear of points. All static labels stay inside their own
axes. A label more than 8 pt from its anchor gets a thin leader to it. If no
gap remains, labels may overlap. HTML leaders end at the label edge. HTML
labels are repositioned after zooms, resizing and legend changes.

## Interactive DE Report

Generate one interactive HTML report per contrast from the same kernel CSVs:

```bash
mokume interactive-report \
    --protein-matrix proteins.csv \
    --sdrf experiment.sdrf.tsv \
    --output de_report.html \
    --highlight-protein COL10A1 \
    --highlight-protein FN1 \
    --contrast NASH-HL NASH HL de_results/NASH_vs_HL.csv
```

With one contrast, `--output` is used as given; with several, each report is
written as `<stem>_<key>.html`. Omit `--output` to write `report_<key>.html` in
the current directory. The report embeds plotly.js, so it opens without a
network connection.

Each report includes:

- **Summary** -- proteins tested, up, down, unchanged and not tested, with the
  fold-change and FDR cutoffs
- **Volcano plot** -- hover details and highlighted proteins; click a point to
  show that protein below
- **MA plot** -- A is the average of the two groups' mean log2 intensities;
  M is log2FC. Colors and hover details match the volcano plot. DE `mean_<condition>`
  columns are already log2; if absent, A is computed from the protein matrix.
- **Protein expression** -- log2 intensities as sample points grouped by
  condition, with neutral gray points and a black mean marker for each condition.
  Only samples from the two
  conditions in the current contrast are included.
- **Results table** -- raw and adjusted p-values, searchable and filterable by
  significance

For PCA, sample correlation, CV, missing values and variance
decomposition, use the QC report below.

## Protein and Peptide Profiles

Supply `--peptides peptides.csv` to `interactive-report` to add a selected-protein
profile beside the grouped expression points. Both four-column and six-column
peptide exports are accepted; conditions always come from SDRF. Only the current
contrast's two conditions are included. Thin gray lines show individual canonical
peptides; the main colored line shows the protein summary. Their agreement helps
identify inconsistent peptide trajectories; their absolute levels need not match.

For one static PNG per protein:

```bash
mokume plot profile -p proteins.csv --peptides peptides.csv \
    -s experiment.sdrf.tsv --protein COL10A1 --protein FN1 -o profiles
```

For known-composition spike-ins, color an MA plot by a `ProteinName` / `group`
TSV and add contrast-specific expected log2 ratios:

```bash
mokume plot de -p proteins.csv -s experiment.sdrf.tsv -o plots --ma \
    --protein-groups protein_groups.tsv --expected-log2fc B-A yeast 1 \
    --contrast B-A B A de_B-A.csv
```

The mapping may omit proteins; they are drawn as gray `Unassigned` points
behind the groups and do not count as a group. Each `ProteinName` must occur
only once. Duplicate protein entries are rejected.

MA plots with more than three composition groups use one panel per group,
sorted by group name in three columns with shared axes. Each panel emphasizes
its group over gray background proteins and shows that group's expected ratio,
when supplied. All group points are circles; these panels have no legend.
Plots without composition groups or with up to three groups keep one panel.

Unknown contrast or group references fail. With two or more contrasts, `plot de`
automatically writes `upset_significant.png` with exclusive significant-protein
intersections. With `--highlight-protein`, it also writes
`log2fc_across_contrasts.png`: each protein is a colored dot, side by side
within each contrast and not joined, since contrasts have no order. Filled dots
are significant and open dots are not. More than six proteins use small
multiples. Confidence intervals are omitted because DE tables contain no
standard errors.

## Single-Matrix QC Report

For a QC report built directly from a protein matrix CSV (with optional DE
results and SDRF grouping), use `mokume.qc_report` (`analysis` extra). It computes
PCA / silhouette / variance decomposition / CV / missing-value and
DE-quality metrics and writes an interactive HTML report:

Sample correlation uses a light-to-dark sequential blue scale, from the minimum
off-diagonal correlation (rounded down to 0.01) to 1. The zero-centered DE
z-score heatmap uses a diverging RdBu scale. Missing heatmap cells remain gray.

```python
import mokume

mokume.qc_report(
    protein_matrix="proteins.csv",
    sdrf="experiment.sdrf.tsv",   # optional sample -> condition grouping
    de_results="de_results/NASH_vs_HL.csv",  # optional
    output="qc_report.html",
    title="My Experiment QC Report",
    is_log2=False,
    peptides="peptides.csv",                    # optional normalized export
    unnormalized_peptides="peptides_before.csv", # optional pre-normalization export
    unimputed_matrix="proteins_before.csv",     # optional pre-imputation export
    highlight_genes=["COL10A1"],
)
```

Optional intermediate tables come from `features2proteins --export-peptides`,
`--export-unnormalized-peptides` and `--export-unimputed`. The QC report adds:

- Peptide boxplots before and after sample normalization, with a shared log2 axis.
- Missingness versus observed mean abundance: complete and incomplete proteins
  have separate density curves. Lower abundance among incomplete proteins is
  evidence compatible with MNAR, not proof of the missingness mechanism.
- Observed versus filled-cell intensity distributions when an unimputed matrix
  is supplied. Filled cells were missing before imputation and observed afterward.
- Quantified protein counts per sample, a completeness histogram (observed in k
  samples), and a binary missingness heatmap. Missingness uses the unimputed
  matrix when available. More than 500 incomplete proteins are sampled by
  missing-count strata for the heatmap.
- An abundance-rank curve with optional `highlight_genes` labels. The piBAQ QC
  PDF also includes this curve.

Diagnostic distributions use observed values and log2 intensities. If the main
matrix has already been filled, supply `unimputed_matrix` to recover missingness.
The MNAR panel is omitted when its source has no missing values. QC reports
retain the raw p-value histogram. The standalone `plot tsne` command remains
available.

## Workflow Comparison Report

Compare multiple quantification workflows side by side (`analysis` extra). Pass a
list of workflow dicts, each with a `name` and either a `protein_df` DataFrame or
a `protein_matrix` CSV path (plus optional `de_results`, `sample_to_condition` /
`sdrf`, and `is_log2`):

```python
import mokume

mokume.workflow_comparison(
    workflows=[
        {"name": "IRS_RLR",     "protein_matrix": "irs_rlr.csv",
         "de_results": "irs_rlr_de.csv",     "sdrf": "experiment.sdrf.tsv"},
        {"name": "Ratio_ComBat", "protein_matrix": "ratio_combat.csv",
         "de_results": "ratio_combat_de.csv", "sdrf": "experiment.sdrf.tsv"},
    ],
    output="comparison.html",
    title="Method Comparison",
    marker_genes={
        "Q03692": {"name": "COL10A1", "expected": "UP"},
        "P02751": {"name": "FN1", "expected": "UP"},
    },
)
```

This generates a report with:

- Summary metrics table (silhouette, CV, DE count, pi1)
- PCA grids for each workflow
- DE concordance analysis (Jaccard, log2FC correlation, CAT curves)
- Marker gene results across workflows, with each marker's sample profile
  centered at that workflow's own protein median. Missing markers are noted.

## t-SNE Visualization Command

A standalone periphery command for t-SNE visualization from a folder of protein
files (`analysis` extra). It is exposed by the wheel CLI while remaining outside
the Rust compute kernel:

```bash
mokume plot tsne \
    --input protein_folder/ \
    --pattern proteins.tsv \
    --output tsne.pdf
```

The equivalent Python APIs remain available as `mokume.de_plots([...])`,
`mokume.interactive_report([...])`, and `mokume.tsne_visualization(**kwargs)`.

## Plotting Library API

Besides the canned commands above, the `mokume.plotting` module exposes the
individual figure functions as a callable library (`analysis` extra). Use these
when you want to build figures directly from a kernel CSV in your own script
instead of running `de_plots`. Each takes a pandas `DataFrame` and either
returns a matplotlib `Figure` or writes to an `output_file` / `file_name` path.

| Function | Purpose |
|----------|---------|
| `plot_volcano(de_results, log2fc_threshold=0.5, fdr_threshold=0.05, highlight_genes=None, output_file=None, ...)` | Volcano plot from a per-contrast DE result frame |
| `plot_heatmap(protein_df, sample_to_condition, proteins=None, top_n=50, output_file=None, ...)` | Clustered heatmap of protein intensities |
| `plot_pca_conditions(protein_df, sample_to_condition, output_file=None, ...)` | PCA scatter coloured by experimental condition |
| `plot_pca(df_pca, output_file, x_col="PC1", y_col="PC2", hue_col="batch", ...)` | PCA scatter from a precomputed PC table |
| `compute_pca_with_plot(df, n_components=5)` | Fit PCA, show the variance-explained scree, return the PC frame |
| `plot_tsne(df, x_col, y_col, hue_col, file_name)` | t-SNE scatter from a frame with embedding columns |
| `plot_distributions(dataset, field, class_field, log2=True, ...)` | Per-class quantile/distribution plot |
| `plot_box_plot(dataset, field, class_field, violin=False, ...)` | Box (or violin) plot of a field grouped by class |
| `is_plotting_available()` | `True` if matplotlib/seaborn are installed |

`plot_distributions` and `plot_box_plot` use the frame's `Condition` column when
present and sort samples by condition then name. The palette expands for
additional conditions. Without condition information, sample curves and boxes
are neutral and have no legend.
`plot_pca` and `plot_tsne` use small circles and the expanding category palette;
`plot_pca`'s `palette` argument is ignored.

```python
import pandas as pd
from mokume.plotting import plot_volcano, plot_heatmap

de = pd.read_csv("proteins.de.Treatment_vs_Control.csv")
plot_volcano(de, fdr_threshold=0.05, log2fc_threshold=1.0,
             output_file="volcano.png")

proteins = pd.read_csv("proteins.csv", index_col=0)
sample_to_condition = {"S1": "Treatment", "S2": "Control"}  # ...
plot_heatmap(proteins, sample_to_condition, top_n=50, output_file="heatmap.png")
```

Functions that return a `Figure` (volcano, heatmap, PCA-by-condition,
distributions, box plot) also save to `output_file`/`file_name` when given one;
without it they return the figure for you to compose or save yourself.
