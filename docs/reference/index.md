# Reference

Reference documentation for the unified wheel CLI, its Rust-wheel and
pure-Python APIs, output columns, and periphery workflows.

## [CLI Reference](cli.md)

The Rust-native compute commands and Python periphery commands exposed by
`pip install mokume`, with their options and dependency extras.

## [Python API (wheel)](python-api.md)

The thin `pip install mokume` wheel: the in-process compute wrappers (`mokume.features2proteins(...)`, `mokume.run([...])`, ...) and the Python periphery functions (t-SNE, tissue maps, DE plots, interactive reports, QC, the missforest fallback).

## [Python API (package)](python-api-package.md)

The separately maintained pure-Python `mokume` package: its class-based
pipeline, `QpxDataset`, plugin registry, differential-expression API, and
runtime resource controls.

Evidence-bound recommendation is provided by the default Rust-backed wheel
through the [Mokume Plugin](../user-guide/agentic-plugin.md), not by `mokume-py`.

## [Computed Values](computed-values.md)

Column names and formulas for all output values (piBAQ, TopN, MaxLFQ, TPA, CopyNumber, etc.).
