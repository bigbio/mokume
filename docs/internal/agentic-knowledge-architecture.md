# Agentic Knowledge Architecture

Mokume agentic recommendation is an installable host plugin, not an embedded
LLM client. The host owns the model, conversation, and credentials. Mokume owns
the scientific contract and deterministic local execution:

```text
wheel-bundled knowledge snapshot
  -> typed KnowledgeGraph
  -> deterministic PolicyDecision + Diagnostic[]
  -> ProfileBlock + ContractBlock + DiagnosticBlock + EvidenceBlock
  -> host-written GeneratedRecommendationBlock
  -> semantic validation
  -> Rust-backed normalization / imputation / DEA
  -> five-metric benchmark mean rank or exploratory unranked results
```

## Plugin boundary

`plugins/mokume/` is the distributable host adapter for Codex and Claude Code.
Each host has a thin manifest and discovers the same skill, while the installed
Mokume wheel owns the MCP service and committed knowledge snapshot. Enabling
the plugin makes either host start the equivalent of:

```text
mokume mcp serve
```

The MCP service and knowledge bundle live in `rust/python/mokume/agentic/` and
are installed by the default `mokume[plugin]` wheel. There is no custom provider
configuration, Mokume API key, internal model call, or standalone agentic TUI.
The server exposes `inspect_dataset`, `evaluate_recommendation`, and
`search_knowledge`. Search returns evidence records and dataset summaries for
explanation only; it never authorizes a configuration.

## Knowledge and provenance

`rust/python/mokume/agentic/knowledge_bundle/knowledge.yaml` is the runtime
evidence index. It stores `SourceEnvelope`, `EvidenceRecord`, `PipelineConfig`,
applicability metadata, and the profile envelope observed in supporting
datasets. Bundled source artifacts live beside it under `sources/`. A source
bundles only what the service shows: a benchmark source may hold `datasets.yaml`,
one summary per benchmark dataset in the form search returns it; a source
without per-dataset summaries declares no artifacts. Loading fails
on an unknown source reference, unsupported method, missing or undeclared
artifact, or artifact hash mismatch. An explicit `--knowledge` path
or the `MOKUME_AGENTIC_KNOWLEDGE` environment variable may override the bundled
snapshot for development and validation.

Evidence retains complete upstream quantification provenance. The current MCP
entry point starts from a protein matrix, so quantification, run normalization,
batch correction, IRS, and upstream filters are frozen axes. They may constrain
applicability but cannot silently become executable candidate fields. A
benchmark therefore gives each data type a general preset and one preset per
quantification method. A preset with `applicability.quantification` applies
only when the matrix declares that quantification, and it then replaces the
general preset of its source; without a declaration, only general presets apply.

## Policy and scoped generation

`inspect_dataset` profiles only the requested two-condition contrast before the
host recommends anything, so unrelated SDRF groups cannot change its diagnostics.
`policy.py` checks replicate count, evidence applicability, known
quantification or engine mismatch, provisional presets, held-out transfer,
DEqMS count fallback, and relevant controls. Stable diagnostic codes make these
decisions auditable outside natural-language prompting.

`context.py` exposes only four read-only blocks and one writable
`GeneratedRecommendationBlock`. The block has an exact field set, at most five
configs, an evidence allowlist, a confidence ceiling, required limitations, and
abstention semantics. `evaluate_recommendation` reruns the validator; a host
cannot bypass policy by directly calling the execution tool.

## Evaluation

The service calls the default wheel's Rust matrix APIs for normalization,
native imputation, and differential expression. `missforest` remains the
documented Python fallback. Each round writes candidate DE tables and a strict
JSON audit artifact containing the knowledge fingerprint, evidence references,
confidence, limitations, diagnostics, input scale, measurements, and cache
statistics. Exploratory rounds also write a signed-call method-sensitivity table.

Ground-truth datasets rank pAUC001, pAUC005, pAUC01, normalized MCC, and G-mean
separately and average those ranks. The four-metric `score_a` remains an
absolute compatibility summary and never selects the winner. If all ranking
metrics are not measurable, the service does not manufacture a winner. Datasets
without ground truth are always
`exploratory_unranked`; method sensitivity, DE count, CV, and missingness remain
diagnostics only.

## Updating evidence

An evidence update must retain the source locator, captured timestamp, artifact
hashes, status, confidence, applicability, metrics, and limitations. A new Grid
preset remains a review candidate until cross-dataset held-out evaluation
supports promotion. Single-dataset oracle results cannot be eligible priors.
