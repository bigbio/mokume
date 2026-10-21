---
name: analyze-proteomics
description: Inspect protein expression matrices with SDRF metadata, select traceable Mokume normalization, imputation, and differential-expression candidates, and evaluate those candidates through the bundled Mokume MCP server. Use for proteomics method selection, spike-in benchmarking, DEA configuration, dataset diagnostics, or explaining why Mokume can or cannot recommend a preset.
---

# Analyze Proteomics

Use the bundled `mokume` MCP tools for data access and computation. Keep biological
reasoning in the host model; never request or store a model API key inside Mokume.
If `inspect_dataset` or `evaluate_recommendation` is unavailable, stop and report a
host/plugin integration error. Do not replace the MCP workflow with a hand-written stdio client or a
different Mokume CLI path.

## Workflow

1. Resolve the protein matrix, SDRF, available peptide-count sidecar, optional
   ground-truth list, and output directory to absolute paths. Identify the
   contrast, explicit matrix scale (`linear` or `log2`), data type (`LFQ` for
   label-free DDA, `DIA`, or `TMT`), upstream quantification, and upstream engine
   when known. A declared quantification selects the benchmark preset measured on
   that quantification. Ask for the
   scale when it is unknown and ask for the data type when generic sample names do
   not identify it; do not infer either fact from intensity magnitude or sample
   count.
2. Call `mokume.inspect_dataset` before recommending or running anything. Put
   the exact two-item SDRF contrast at the tool's top level. Put the explicit
   `input_scale`, optional `peptide_counts` path, declared acquisition facts, and
   any SDRF factor override in its `options` object. Inspection profiles only
   those two conditions; repeat the canonical labels returned in
   `profile.samples_per_condition` during evaluation.
3. Read every returned diagnostic. If policy disallows generation because the data
   type is `unknown`, ask for a supported declaration and inspect again. For any
   other policy error, report the abstention and do not manufacture a configuration.
4. Start from `policy_recommendation`. Change its `configs` only when the returned
   profile or evidence supports the change. Preserve the exact block contract in
   [recommendation-contract.md](references/recommendation-contract.md). Without a
   peptide-count sidecar, do not add `deqms` or an ensemble containing `deqms`;
   deterministic policy omits those candidates. Standalone `rots` and `limrots`
   candidates must use `fdr_method="bh"` because their native permutation FDR is
   not replaced by another correction; this restriction does not apply to an
   `ensemble` candidate.
5. Call `mokume.evaluate_recommendation` with exactly two canonical condition
   labels returned by inspection and in the same order used for inspection. Do
   not substitute longer raw SDRF values when inspection normalized them. Put a
   new absolute `output_dir` and all runtime settings in its `options` object.
   Include the same peptide-count sidecar path, input scale, declared acquisition
   metadata, and factor override used during inspection in that object. The
   sidecar is mandatory when any candidate uses `deqms` directly or through an
   ensemble; evaluation rejects the whole round when it is absent. Keep
   `threads=24` unless the user explicitly chooses another value.
6. Compare results. With ground truth, use the returned five-metric
   `benchmark_mean_rank` ranking; `score_a` is an absolute compatibility summary,
   not the winner field. Without ground truth, inspect `method_sensitivity.tsv`
   and report shared versus
   method-sensitive signed calls without selecting or implying a winner. For a
   ground-truth ranking, report each candidate's tested universe
   (`TP + FP + FN + TN`); if those totals differ, qualify the ranking as
   candidate-universe-specific.
7. Report the knowledge fingerprint, evidence references, confidence, limitations,
   input scale, output paths, and whether the result is ranked or exploratory.

## Scientific Boundaries

- Treat quantification as frozen metadata at the protein-matrix entry point. Do not
  claim that this workflow re-ran MaxLFQ, DirectLFQ, piBAQ, or another aggregator.
- Never rank configurations from DE count, CV, missingness, method agreement, or
  permutation diagnostics.
- Interpret `missing_rate` as a fraction and CV values as unitless ratios. For
  example, `median_cv=0.0852` means a median CV of about `8.52%`, not `0.0852%`.
- Do not call a benchmark candidate optimal for the current dataset before it is
  evaluated with relevant ground truth.
- Keep FDR threshold as a user-controlled operating point; do not hide a changed
  threshold inside a candidate recommendation.
- Never use `input_scale=auto`; evaluation requires an explicit `linear` or
  `log2` declaration.
- For `input_scale=linear`, Mokume treats non-positive values as missing to match
  the Rust normalization, imputation, and DE kernels. For `input_scale=log2`,
  finite zero and negative values remain valid observations.
- Preserve evidence IDs and required limitations verbatim. Do not cite knowledge
  records outside the `allowed_evidence_refs` returned by `inspect_dataset`.
- Use `mokume.search_knowledge` to explain methods, benchmark evidence, or the
  benchmark result of a named PXD dataset. Its results are explanation-only and
  never a recommendation source.
- Treat an inferred data type, unknown quantification, incompatible upstream engine,
  provisional preset, or weak held-out transfer as a material limitation.

## Iteration

Evaluate at most five candidates per round. On later rounds, change no more than two
executable axes per candidate and explain each change using the previous result and
allowed evidence. Stop when the user-requested budget is exhausted, policy abstains,
or further candidates would only repeat an existing executable signature.
