# Analysis Fallbacks

A handful of methods stay **Python-only**: the Rust kernel does not reproduce
them, so the wheel ships them and the kernel's errors point here. They all live
in the `analysis` extra:

```bash
pip install "mokume[analysis]"   # numpy, pandas, scipy, scikit-learn
```

This page covers the three `analysis`-extra entry points:

- `mokume.impute(method='missforest')` — the scikit-learn imputer
- `mokume.qc_report` — single-matrix QC HTML report
- `mokume.workflow_comparison` — multi-workflow comparison HTML report

## `mokume.impute(method=...)` — missforest

The pure-Python imputer, reaching `missforest` (which the Rust kernel does not
reproduce) plus every other supported method. `matrix` is a wide protein matrix
CSV or DataFrame; it writes `output` if given and returns the imputed DataFrame.

```python
import mokume

mokume.impute("proteins.csv", method="missforest", output="imputed.csv")
```

!!! warning "missforest is wheel-only"
    `missforest` wraps scikit-learn's `IterativeImputer`, driven by
    `RandomForestRegressor`. Its output is the artifact of sklearn's exact
    tree-building internals **plus** its RNG — the model differs structurally,
    not just in RNG draws, so no cross-language tolerance tier is reachable (a
    Rust ML crate would not align either). The Rust `features2proteins` CLI does
    not advertise `missforest`; call `mokume.impute(..., method="missforest")`
    from the Python analysis periphery instead.

## `mokume.qc_report` — single-matrix QC report

Builds an interactive QC HTML report from one protein matrix: PCA,
silhouette, variance decomposition, CV, missing-value, and DE-quality metrics.
`sdrf` supplies the sample &rarr; condition grouping; `de_results` is an optional
DE result CSV. Returns the output path.

```python
import mokume

mokume.qc_report(
    protein_matrix="proteins.csv",
    sdrf="experiment.sdrf.tsv",
    output="qc.html",
)
```

## `mokume.workflow_comparison` — multi-workflow comparison report

Builds an HTML report comparing several quantification workflows. `workflows` is
a list of dicts, one per workflow, each with a `name` and either a `protein_df`
(DataFrame) or a `protein_matrix` (CSV path); plus optional `de_results`,
`sample_to_condition` or `sdrf`, and `is_log2`. Returns the output path.

```python
import mokume

mokume.workflow_comparison(
    workflows=[
        {"name": "maxlfq", "protein_matrix": "maxlfq.csv", "sdrf": "x.sdrf.tsv"},
        {"name": "pibaq",   "protein_matrix": "pibaq.csv",   "sdrf": "x.sdrf.tsv"},
    ],
    output="comparison.html",
)
```

## Why these are Python-only

The QC and workflow-comparison reports were never a Rust porting target — they
summarize and visualize the kernel's output rather than computing the
single-sourced numbers. `missforest` is different: it is a genuine computation
method whose **RNG / estimator internals cannot be matched cross-language**
(scikit-learn's exact tree internals plus RNG). Rather than reproduce a method
unfaithfully, the Rust kernel fails fast with a `NotImplemented` error that
points here, so a requested method never looks applied when it is not. Install
the `analysis` extra and run the wheel fallback on the kernel's output matrix.
