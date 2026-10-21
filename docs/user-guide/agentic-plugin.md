# Mokume Plugin

The Mokume Plugin adds traceable proteomics method recommendation to Codex and
Claude Code. The host model performs the reasoning; Mokume supplies the
knowledge snapshot, deterministic policy checks, validation, and the
Rust-backed normalization, imputation, and differential-expression tools.

The plugin does not ask for, store, or call a model API key. It bundles a local
stdio MCP configuration, so enabling the plugin starts the Mokume MCP server
automatically for the task. Do not add a second custom MCP entry or run the
server manually.

## Install

Install the default Rust-backed distribution with the plugin dependencies:

```bash
pip install "mokume[plugin]"
```

The Rust-backed `mokume` distribution and this optional MCP workflow require
Python 3.10 or newer.

### Codex

Add the repository marketplace and install the plugin:

```bash
codex plugin marketplace add bigbio/mokume
codex plugin add mokume@bigbio
```

In the Codex app, the same setup is available under **Plugins → Add
marketplace**. Use `https://github.com/bigbio/mokume` as the Git source and
leave the sparse path empty. Start a new task after installation so Codex loads
the skill and its MCP tools.

The `mokume` executable must remain available on `PATH` to the Codex process.
Installation is complete when the task exposes `mokume.inspect_dataset`,
`mokume.evaluate_recommendation`, and `mokume.search_knowledge`.

### Claude Code

Add the same repository as a Claude Code marketplace and install the same
plugin bundle:

```bash
claude plugin marketplace add bigbio/mokume
claude plugin install mokume@bigbio
```

Start a new Claude Code session after installation. The plugin discovers the
shared `analyze-proteomics` skill and starts the Mokume MCP server, which loads
its knowledge snapshot from the installed wheel. The `mokume` executable must
be on the `PATH` inherited by Claude Code. No separate `/mcp` setup is required.

## Use

Ask Codex to use `$mokume:analyze-proteomics`, or invoke
`/mokume:analyze-proteomics` in Claude Code, and provide:

- an absolute protein-matrix path;
- an absolute SDRF path;
- an exact two-condition contrast from the selected SDRF factor;
- the explicit protein-matrix scale (`linear` or `log2`);
- an absolute peptide-count sidecar when testing DEqMS directly or in an ensemble;
- the contrast and, when known, the `LFQ` (label-free DDA), `DIA`, or `TMT`
  data type;
- the upstream quantification and engine when known; a declared quantification
  selects the benchmark preset measured on that quantification;
- an absolute output directory; and
- optionally, a ground-truth protein list for a spike-in benchmark.

The workflow first calls `inspect_dataset` with that contrast. This profiles only
the two requested conditions and binds compatible evidence into typed context
blocks; unrelated SDRF conditions do not affect diagnostics or preprocessing.
The host may then propose at most five configurations under the returned contract
and pass the exact block to `evaluate_recommendation`.

`search_knowledge` answers questions about methods, benchmark evidence, and the
benchmark result of a named PXD dataset. Its results are explanation-only: a
recommendation may cite only the `allowed_evidence_refs` that `inspect_dataset`
returns.

The comma- or tab-delimited protein matrix must use its first column for
non-empty, unique protein identifiers and provide at least two numeric sample
columns. Column names must be non-empty and unique, every sample column must map
to the SDRF, and sample cells may contain finite values or missing values (`NaN`)
but not positive or negative infinity. A linear matrix requires at least one
positive finite intensity and treats non-positive cells as missing. A log2 matrix
requires at least one finite intensity and preserves finite zero and negative
values as observations. The declared `input_scale` selects these semantics.

The MCP schema keeps file paths, contrast, and the generated recommendation
explicit while grouping the scale, optional sidecar, declared acquisition facts,
and runtime controls in `options`; evaluation also places its required
`output_dir` in `options`.
`input_scale` must be explicitly declared as `linear` or `log2`; Mokume does not
guess it from intensity magnitude. The peptide-count sidecar is supplied as
`options.peptide_counts` during both inspection and evaluation. It
contains exactly two comma- or tab-delimited columns named `protein` and
`peptide_count`. Protein IDs are unique, counts are positive integers, and at
least one ID must match the matrix's first column. It is optional for
count-independent candidates, but mandatory for `deqms` and every ensemble that
contains `deqms`. Without it, policy omits those candidates and evaluation rejects
any host-added one. Individual matrix proteins absent from a supplied sidecar use
a count of one.

When `data_type` is omitted, Mokume only recognizes explicit `LFQ`, `DIA`/`DIA-NN`,
or `TMT`/`plex` sample-name markers. Generic names such as `S1` do not imply LFQ:
the profile returns `unknown`, policy abstains, and the host must ask for a supported
declaration. Engine aliases are normalized to the knowledge catalog, so `DIANN`,
`DIA NN`, and `DIA-NN` all bind as `DIA-NN`.

Inspection requires a two-item list using the canonical condition labels derived
from the selected SDRF factor. Reuse the returned
`profile.samples_per_condition` keys, in the same order, for evaluation; do not
replace them with longer raw SDRF values. Unknown object fields are rejected
rather than silently ignored.
The output path must be absolute and must not already exist; Mokume never
overwrites a previous evaluation round. Artifacts are written to a sibling staging
directory and the requested path appears only after the full round succeeds. A
failed candidate leaves no partial target, so the same path can be retried. Repeat
the inspection scale, sidecar, and metadata inside evaluation so both calls bind
the same policy context.

With relevant ground truth and an explicit `UP` or `DOWN` expected direction,
the tool ranks candidates separately on pAUC at 1%, 5%, and 10% false-positive
cutoffs, normalized MCC, and G-mean, then selects the lowest five-metric
`benchmark_mean_rank`. The returned `score_a` remains the absolute arithmetic
mean of the three pAUC values and normalized MCC; it is not the winner field.
Without ground truth, set the expected direction to null; the tool returns
exploratory diagnostics and no winner. For a ground-truth result, report each
candidate's tested universe (`TP + FP + FN + TN`). If those totals differ, the
ranking is candidate-universe-specific and must not be presented as general
method superiority.
Quantification is recorded as provenance but remains frozen because this entry
point starts from an existing protein matrix.

The Plugin accepts only `fdr_method="bh"` for standalone `rots` and `limrots`
candidates. Both methods retain their native permutation FDR, so alternative
values would otherwise create duplicate candidate signatures with identical
results. Ensemble candidates may still use the full FDR catalog: eligible
non-ROTS members and the combination layer use the requested correction, while
ROTS and LimROTS members retain their native permutation FDR.

Each evaluated round writes one DE table per candidate,
`method_sensitivity.tsv`, and `evaluation.json`. The sensitivity table separates
signed calls shared by every tested candidate from calls that depend on the
chosen method; it is descriptive and never selects a winner. The JSON artifact
records the knowledge fingerprint, evidence references, confidence, limitations,
input scale, diagnostics, measurements, and ranking status.

## Repository layout

```text
.agents/plugins/marketplace.json       # Codex marketplace
.claude-plugin/marketplace.json        # Claude Code marketplace
plugins/mokume/
├── .codex-plugin/                      # Codex manifest and MCP adapter
├── .claude-plugin/plugin.json         # Claude Code manifest and MCP adapter
└── skills/analyze-proteomics/          # shared workflow and output contract
rust/python/mokume/agentic/
├── knowledge_bundle/                   # evidence index and source artifacts
└── ...                                 # deterministic MCP service
```

Update the knowledge index separately from model prompts. Every eligible
record must retain its source, captured date, artifact hashes, applicability,
metrics, confidence, status, and limitations. Single-dataset oracle results are
not eligible priors.
