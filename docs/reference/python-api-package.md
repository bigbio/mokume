# Python API (package)

The pure-Python `mokume-py` package (`pip install mokume-py`) exposes a class-based API
for quantification and differential expression, built on an OOP data layer
(`QpxDataset`) with a pluggable method registry and runtime resource controls.
The reference below is rendered directly from the package source, so it always
matches the installed version rather than a hand-written copy.

For the Rust-accelerated wheel's thin in-process API, see
[Python API (wheel)](python-api.md).

## Quantification

::: mokume.quantification.get_quantification_method

::: mokume.quantification.topn.TopNQuantification

::: mokume.quantification.maxlfq.MaxLFQQuantification

::: mokume.quantification.all_peptides.AllPeptidesQuantification

## Differential expression

::: mokume.analysis.DifferentialExpression

## Data layer and pipeline

A `QpxDataset` is the container the pipeline produces. It holds the data levels
(`psms`, `features`, `peptides`, `proteins`), sample metadata, and a provenance
log of the steps applied. `run_pipeline` resolves the configured quantification
method from the `PluginRegistry`, dispatches to the flow matching the method's
`input_level`, and returns a populated `QpxDataset`.

```python
from mokume.pipeline.config import PipelineConfig, InputConfig, QuantificationConfig
from mokume.pipeline.runner import run_pipeline

config = PipelineConfig(
    input=InputConfig(parquet="features.parquet"),
    quantification=QuantificationConfig(method="maxlfq"),
)
dataset = run_pipeline(config)                 # QpxDataset with .proteins populated
protein_matrix = dataset.get_level("proteins")  # protein x sample DataFrame
```

`QuantificationPipeline(config).run_dataset()` returns the same `QpxDataset` from
the pipeline object directly; `run()` returns the bare protein matrix.

### QpxDataset

::: mokume.core.dataset.QpxDataset

### PluginRegistry

The registry maps a name to a method class within one of the plugin groups
(`quantification`, `normalization.feature`, `normalization.sample`, `imputation`,
`harmonization`, `filter`). Register a class with the `@PluginRegistry.register`
decorator; resolve one with `get`; list a group with `available`.

```python
from mokume.core.registry import PluginRegistry

PluginRegistry.available("quantification")
# ['directlfq', 'maxlfq', 'median', 'peptide_count', 'pibaq', 'ratio', 'spectral_count', 'sum', ...]
method = PluginRegistry.get("quantification", "maxlfq")
```

::: mokume.core.registry.PluginRegistry

### run_pipeline

::: mokume.pipeline.runner.run_pipeline

### DirectLFQ ion export

The pure-Python command can stream the normalized ion matrix produced during
DirectLFQ protein estimation without retaining the full ion result in memory:

```bash
mokume features2proteins \
    --parquet features.parquet \
    --output proteins.csv \
    --quant-method directlfq \
    --export-ions normalized-ions.csv
```

The CSV contains `protein`, `ion`, and one linear-intensity column per sample.
The same output is available through `OutputConfig(export_ions=...)` when using
`run_pipeline` or `QuantificationPipeline` directly. Other quantification
methods reject `export_ions`.

## Runtime resources

`RuntimeConfig` controls the DuckDB memory and thread hints used by the
pure-Python pipeline. `run_pipeline` always uses the pure-Python implementation;
it does not dispatch into `mokume`. To use the Rust implementation, install
`mokume` in a separate environment and use its [thin Python API](python-api.md)
or installed console command.

```python
from mokume.pipeline.config import RuntimeConfig

config = PipelineConfig(
    input=InputConfig(parquet="features.parquet"),
    quantification=QuantificationConfig(method="sum"),
    runtime=RuntimeConfig(duckdb_memory="80GB", duckdb_threads=24),
)
dataset = run_pipeline(config)
```

::: mokume.pipeline.config.RuntimeConfig

## Agentic recommendation

Agentic recommendation is not part of `mokume-py`. Install the default
Rust-backed distribution with the
[Mokume Plugin](../user-guide/agentic-plugin.md) (`mokume[plugin]`). Use it in a
separate environment because both distributions provide the same `mokume`
import name.
