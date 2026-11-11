# Changelog

## 0.2.0 (unreleased)

The Rust-backed `mokume` wheel replaces the pure-Python 0.1.0 release as the
default distribution. The pure-Python implementation continues as `mokume-py`.

### Changed

- `--quant-method maxlfq` and `MaxLFQQuantification` run the MaxLFQ solver
  (Cox et al., 2014, Eq. 3): a least-squares fit to the median log ratios of the
  peptide species that each sample pair shares. In 0.1.0 they ran DirectLFQ when
  it was installed and otherwise a DirectLFQ-style trace alignment with median
  aggregation, so results differ between 0.1.0 and 0.2.0 under the same name.
- `features2proteins` defaults the sample normalization of MaxLFQ, `top<N>`,
  `sum` and `median` to `hierarchical`, in the Rust CLI and in mokume-py. The
  Rust CLI keeps `global-median` when `--normalization-proteins` is given. The
  global median follows each run's detection depth and match-between-runs
  transfers; pass `--sample-normalization global-median` to keep it. With the
  new default, `sum` reports a protein that has no peptide in a sample as
  missing instead of 0.
- mokume-py's `NormalizationConfig` leaves both methods unset by default, and
  `PipelineConfig` fills them with the same defaults per quantification method,
  so DirectLFQ, ratio and `peptide_count` no longer need an explicit `none`.
- mokume-py's MaxLFQ marks samples without a supported pair ratio as missing
  (NaN) instead of zero. Protein tables already left these samples out.
- `--filter-score` reads the typed `peptide_qvalue`, `precursor_qvalue` and
  `global_qvalue` columns whenever the file has them, also when some rows are
  null; a null value fails the filter. QPX files without these columns still use
  `additional_scores`. mokume-py keeps rows with a null `is_decoy` as targets,
  like the Rust kernel.
- Protein groups are identified by their member set, so the same accessions in
  another stored order form one group, named by the smallest observed order.
  mokume-py's low-frequency peptide filter, DEqMS peptide counts, ratio
  quantification and TissueMap use the group instead of `anchor_protein`;
  TissueMap keeps `anchor_protein` as an annotation column.

### Added

- `features2proteins --transferred-weight` and `peptides2protein
  --weight-column` weigh measurements in MaxLFQ pair medians and in `top<N>`
  selection (Rust kernel only).
- `features2proteins --run` and `features2peptides --run` take the samples
  from a QPX run view (`run.samples`, matched by exact run and label) instead
  of an SDRF; each sample is its own condition.
