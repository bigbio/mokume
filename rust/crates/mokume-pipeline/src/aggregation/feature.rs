use crate::accessions::{
    first_protein_name, peptide_key, protein_group_name, rename_contaminant_twins,
};
use crate::aggregation::{
    AggregationMeasurement, FeatureAggregation, FeatureWeights, MaxLfqFeatureAggregation,
    PibaqAggregation, PibaqFeatureAggregation, RatioAggregation,
};
use crate::from_peptides::PibaqDigest;
use crate::peptide_normalization::{
    apply_dataset_norm_to_peptide_cells, collapse_to_canonical, push_peptide_max,
    sum_peptide_values, top_n_intensities, writes_canonical_cells,
};
use crate::protein_matrix::ProteinValues;
use crate::solvers::{
    directlfq_cells, prepare_directlfq_ions, solve_directlfq, write_directlfq_ions,
};
use crate::state::CellKey;
use crate::{
    DirectLfqRetain, FeatureToProteinsConfig, HashMap, HashSet, PeptideId, ProteinId,
    QpxFeatureRecord, QuantMethod, Result, SampleId, SampleNormalizationMethod, SdrfRawTable,
    SdrfTable, StringIdRegistry, direct_lfq_aligned_with, invalid_input, median, median_finite,
};

impl FeatureAggregation {
    pub(crate) fn from_config(
        config: &FeatureToProteinsConfig,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
        pibaq_digest: Option<PibaqDigest>,
    ) -> Result<Self> {
        match config.quantification {
            QuantMethod::Sum | QuantMethod::Intensity => Ok(Self::Sum(HashMap::new())),
            QuantMethod::Median => Ok(Self::Median(HashMap::new())),
            QuantMethod::Abd => Ok(Self::Abd(HashMap::new())),
            QuantMethod::PeptideCount => Ok(Self::PeptideCount(HashMap::new())),
            QuantMethod::SpectralCount => Err(invalid_input(
                "spectral_count requires the PSM-level aggregation path",
            )),
            QuantMethod::Pibaq => Ok(Self::Pibaq(Box::new(PibaqFeatureAggregation::new(
                PibaqAggregation::from_digest(
                    config,
                    pibaq_digest.ok_or_else(|| {
                        invalid_input("piBAQ requires a runtime pyOpenMS FASTA digest")
                    })?,
                )?,
            )))),
            QuantMethod::Ratio => Ok(Self::Ratio(RatioAggregation::from_config(
                config, sdrf, raw_sdrf,
            )?)),
            QuantMethod::TopN => Self::top_n(config),
            QuantMethod::MaxLfq | QuantMethod::DirectLfq => Ok(Self::Lfq {
                method: config.quantification,
                route_to_directlfq: config.quantification == QuantMethod::DirectLfq,
                maxlfq_min_ratio_count: config.maxlfq.min_ratio_count,
                stabilize: config.maxlfq.stabilize,
                directlfq_min_nonan: config.directlfq.min_nonan,
                directlfq_num_samples_quadratic: config.directlfq.num_samples_quadratic,
                directlfq_normalize_samples: config.directlfq.normalize_samples,
                traces: MaxLfqFeatureAggregation::weighted(
                    config
                        .input
                        .transferred_weight
                        .filter(|_| config.quantification == QuantMethod::MaxLfq),
                ),
                cell_precision: config.differential_expression.precision_weights
                    || config.output.export_precision.is_some(),
                cached_directlfq: None,
            }),
        }
    }

    fn top_n(config: &FeatureToProteinsConfig) -> Result<Self> {
        if config.topn_peptides == 0 {
            return Err(invalid_input("topn must be greater than 0"));
        }
        Ok(Self::TopN {
            topn: config.topn_peptides,
            cells: HashMap::new(),
            weights: config.input.transferred_weight.map(FeatureWeights::new),
        })
    }

    pub(crate) fn push(&mut self, measurement: AggregationMeasurement<'_>) -> Result<()> {
        let cell = CellKey {
            protein: measurement.protein,
            sample: measurement.sample,
        };
        match self {
            Self::Sum(cells)
            | Self::Median(cells)
            | Self::Abd(cells)
            | Self::PeptideCount(cells) => {
                push_peptide_max(cells, cell, measurement.peptide, measurement.intensity);
            }
            Self::Pibaq(pibaq) => {
                pibaq.push(measurement);
            }
            Self::Ratio(ratio) => {
                ratio.push(measurement);
            }
            Self::TopN { cells, weights, .. } => {
                push_peptide_max(cells, cell, measurement.peptide, measurement.intensity);
                if let Some(weights) = weights {
                    weights.push(cell, measurement.peptide, measurement.transferred);
                }
            }
            Self::Lfq { traces, .. } => {
                traces.push(measurement);
            }
        }
        Ok(())
    }

    pub(crate) fn applies_dataset_normalization(&self) -> bool {
        match self {
            Self::Sum(_)
            | Self::Median(_)
            | Self::Abd(_)
            | Self::PeptideCount(_)
            | Self::TopN { .. }
            | Self::Pibaq(_) => true,
            // MaxLFQ uses explicitly configured normalization on peptide species.
            // DirectLFQ performs its own internal sample normalization.
            Self::Lfq {
                route_to_directlfq, ..
            } => !route_to_directlfq,
            Self::Ratio(_) => false,
        }
    }

    /// Apply a dataset-level sample normalization (one that needs the full
    /// protein x sample matrix, not a per-sample factor). Quantile runs
    /// on the canonical peptide cells; piBAQ supports quantile and MaxLFQ supports
    /// quantile and hierarchical on their own structures. Other dataset methods
    /// are gated out before this point.
    pub(crate) fn apply_dataset_normalization(
        &mut self,
        method: SampleNormalizationMethod,
        allowed_cells: &HashSet<CellKey>,
        peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
        samples: &StringIdRegistry<SampleId>,
    ) {
        match self {
            Self::Sum(cells)
            | Self::Median(cells)
            | Self::Abd(cells)
            | Self::PeptideCount(cells)
            | Self::TopN { cells, .. } => {
                apply_dataset_norm_to_peptide_cells(
                    cells,
                    method,
                    allowed_cells,
                    peptide_to_canonical,
                    samples,
                );
            }
            Self::Pibaq(pibaq) => {
                if method == SampleNormalizationMethod::Quantile {
                    pibaq.apply_quantile_normalization();
                }
            }
            Self::Lfq {
                route_to_directlfq: false,
                traces,
                ..
            } => match method {
                SampleNormalizationMethod::Quantile => {
                    traces.apply_quantile_normalization(allowed_cells);
                }
                SampleNormalizationMethod::Hierarchical => {
                    traces.apply_hierarchical_normalization(allowed_cells, samples);
                }
                _ => {}
            },
            Self::Ratio(_) | Self::Lfq { .. } => {}
        }
        self.collapse_top_n_weights(method, peptide_to_canonical);
    }

    /// Top-N weights follow its cells onto canonical peptides.
    fn collapse_top_n_weights(
        &mut self,
        method: SampleNormalizationMethod,
        peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
    ) {
        if let Self::TopN {
            weights: Some(weights),
            ..
        } = self
            && writes_canonical_cells(method)
        {
            weights.collapse(peptide_to_canonical);
        }
    }

    pub(crate) fn write_intermediate_exports(
        &mut self,
        config: &FeatureToProteinsConfig,
        proteins: &StringIdRegistry<ProteinId>,
        canonical_peptides: &StringIdRegistry<PeptideId>,
        samples: &StringIdRegistry<SampleId>,
        allowed_cells: &HashSet<CellKey>,
    ) -> Result<()> {
        let Self::Lfq {
            method: QuantMethod::DirectLfq,
            directlfq_min_nonan,
            directlfq_num_samples_quadratic,
            directlfq_normalize_samples,
            traces,
            cell_precision,
            cached_directlfq,
            ..
        } = self
        else {
            return Ok(());
        };
        let Some(path) = config.output.export_ions.as_deref() else {
            return Ok(());
        };

        let mut prepared = prepare_directlfq_ions(
            traces.directlfq_sums(),
            allowed_cells,
            proteins,
            canonical_peptides,
            samples,
        );
        let result = direct_lfq_aligned_with(
            std::mem::take(&mut prepared.ions),
            *directlfq_min_nonan,
            *directlfq_num_samples_quadratic,
            *directlfq_normalize_samples,
            DirectLfqRetain::ions(*cell_precision),
        );
        write_directlfq_ions(
            path,
            &result.normalized_ions,
            &prepared,
            proteins,
            canonical_peptides,
            samples,
        )?;
        *cached_directlfq = Some(directlfq_cells(result, &prepared));
        Ok(())
    }

    /// piBAQ only: rename `CONTAM_` twins of digest accessions to the non-contaminant entry.
    pub(crate) fn resolve_contaminant_twins(&self, protein_group: &str) -> Option<String> {
        let Self::Pibaq(pibaq) = self else {
            return None;
        };
        rename_contaminant_twins(protein_group, &pibaq.core.contaminant_twins)
    }

    pub(crate) fn keeps_shared_peptides(&self) -> bool {
        matches!(self, Self::Pibaq(_))
    }

    /// The same method without data, for a pipelined ingest: it answers
    /// ingest's per-feature queries while this one is filled on a worker
    /// thread. `None` for piBAQ and Ratio, whose `push` reads names.
    pub(crate) fn empty_like(&self) -> Option<Self> {
        Some(match self {
            Self::Sum(_) => Self::Sum(HashMap::new()),
            Self::Median(_) => Self::Median(HashMap::new()),
            Self::Abd(_) => Self::Abd(HashMap::new()),
            Self::PeptideCount(_) => Self::PeptideCount(HashMap::new()),
            Self::TopN { topn, weights, .. } => Self::TopN {
                topn: *topn,
                cells: HashMap::new(),
                weights: weights
                    .as_ref()
                    .map(|weights| FeatureWeights::new(weights.transferred)),
            },
            Self::Lfq {
                method,
                route_to_directlfq,
                maxlfq_min_ratio_count,
                stabilize,
                directlfq_min_nonan,
                directlfq_num_samples_quadratic,
                directlfq_normalize_samples,
                traces,
                cell_precision,
                ..
            } => Self::Lfq {
                method: *method,
                route_to_directlfq: *route_to_directlfq,
                maxlfq_min_ratio_count: *maxlfq_min_ratio_count,
                stabilize: *stabilize,
                directlfq_min_nonan: *directlfq_min_nonan,
                directlfq_num_samples_quadratic: *directlfq_num_samples_quadratic,
                directlfq_normalize_samples: *directlfq_normalize_samples,
                traces: MaxLfqFeatureAggregation::weighted(
                    traces
                        .weights
                        .as_ref()
                        .map(|weights| weights.ions.transferred),
                ),
                cell_precision: *cell_precision,
                cached_directlfq: None,
            },
            Self::Pibaq(_) | Self::Ratio(_) => return None,
        })
    }

    pub(crate) fn preserves_loading_rows(&self) -> bool {
        matches!(
            self,
            Self::Sum(_)
                | Self::Median(_)
                | Self::Abd(_)
                | Self::PeptideCount(_)
                | Self::TopN { .. }
                | Self::Lfq { .. }
        )
    }

    pub(crate) fn preserves_contextual_canonical_rows(&self) -> bool {
        matches!(
            self,
            Self::Median(_) | Self::Abd(_) | Self::PeptideCount(_) | Self::TopN { .. }
        )
    }

    pub(crate) fn peptide_key(&self, feature: &QpxFeatureRecord) -> String {
        match self {
            Self::Ratio(_) => feature.sequence.clone(),
            _ => peptide_key(feature),
        }
    }

    pub(crate) fn protein_name(&self, feature: &QpxFeatureRecord) -> Option<String> {
        match self {
            // Ratio retains its first-accession contract. LFQ methods keep the
            // complete protein group, matching the peptide intermediate.
            Self::Ratio(_) => first_protein_name(&feature.protein_accessions),
            _ => protein_group_name(&feature.protein_accessions),
        }
    }

    /// The protein values and, for `cell_precision`, each cell's precision.
    pub(crate) fn finalize(
        self,
        allowed_cells: &HashSet<CellKey>,
        proteins: &mut StringIdRegistry<ProteinId>,
        peptide_to_canonical: &HashMap<PeptideId, PeptideId>,
        canonical_peptides: &StringIdRegistry<PeptideId>,
        samples: &StringIdRegistry<SampleId>,
    ) -> Result<(ProteinValues, HashMap<CellKey, f64>)> {
        let values = match self {
            Self::Sum(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        allowed_cells
                            .contains(&key)
                            .then(|| (key, sum_peptide_values(&peptides)))
                    })
                    .collect(),
            ),
            Self::Median(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let mut intensities = canonical.into_values().collect::<Vec<_>>();
                        median(&mut intensities).map(|value| (key, value))
                    })
                    .collect(),
            ),
            Self::Abd(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let mut log2_intensities = canonical
                            .into_values()
                            .filter(|value| value.is_finite() && *value > 0.0)
                            .map(f64::log2)
                            .collect::<Vec<_>>();
                        median_finite(&mut log2_intensities).map(|value| (key, value))
                    })
                    .collect(),
            ),
            Self::PeptideCount(cells) => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        Some((key, canonical.len() as f64))
                    })
                    .collect(),
            ),
            Self::Pibaq(pibaq) => ProteinValues::Cells(pibaq.finalize(proteins)?),
            Self::Ratio(ratio) => ProteinValues::Cells(ratio.finalize()),
            Self::TopN {
                topn,
                cells,
                mut weights,
            } => ProteinValues::Cells(
                cells
                    .into_iter()
                    .filter_map(|(key, peptides)| {
                        if !allowed_cells.contains(&key) {
                            return None;
                        }
                        let canonical = collapse_to_canonical(peptides, peptide_to_canonical);
                        let weights = weights
                            .as_mut()
                            .map(|weights| weights.take_cell(&key, peptide_to_canonical));
                        let selected = top_n_intensities(canonical, weights.as_ref(), topn);
                        if selected.is_empty() {
                            return None;
                        }
                        let value = selected.iter().sum::<f64>() / selected.len() as f64;
                        Some((key, value))
                    })
                    .collect(),
            ),
            Self::Lfq {
                route_to_directlfq: true,
                directlfq_min_nonan,
                directlfq_num_samples_quadratic,
                directlfq_normalize_samples,
                traces,
                cell_precision,
                cached_directlfq,
                ..
            } => {
                if let Some((values, precision)) = cached_directlfq {
                    return Ok((ProteinValues::Rows(values), precision));
                }
                let prepared = prepare_directlfq_ions(
                    traces.into_directlfq_sums(),
                    allowed_cells,
                    proteins,
                    canonical_peptides,
                    samples,
                );
                let (values, precision) = solve_directlfq(
                    prepared,
                    directlfq_min_nonan,
                    directlfq_num_samples_quadratic,
                    directlfq_normalize_samples,
                    cell_precision,
                );
                return Ok((ProteinValues::Rows(values), precision));
            }
            Self::Lfq {
                traces,
                maxlfq_min_ratio_count,
                stabilize,
                cell_precision,
                ..
            } => {
                return traces.into_protein_values(
                    allowed_cells,
                    proteins,
                    samples,
                    maxlfq_min_ratio_count,
                    stabilize,
                    cell_precision,
                );
            }
        };
        Ok((values, HashMap::new()))
    }
}
