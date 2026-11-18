use crate::aggregation::{
    AggregationMeasurement, FeatureWeights, MaxLfqFeatureAggregation, TraceWeights,
};
use crate::exports::CanonicalExportValues;
use crate::features_to_proteins::COLLAPSE_BLOCK;
use crate::peptide_normalization::lfq_traces::{
    apply_hierarchical_to_lfq_traces, apply_quantile_to_lfq_traces,
};
use crate::peptide_normalization::{collapse_to_canonical, push_peptide_max};
use crate::protein_matrix::ProteinValues;
use crate::solvers::{maxlfq_residuals, maxlfq_values_and_precision};
use crate::state::{CellKey, DirectLfqCellKey, PeptideCellKey};
use crate::{
    HashMap, HashSet, IntoParallelIterator, ParallelIterator, PeptideMeasurement, ProteinId,
    Result, SampleId, StringIdRegistry, maxlfq, solve_max_lfq_with_stabilization, warn,
};

impl MaxLfqFeatureAggregation {
    /// Empty traces that weigh transferred features by `transferred_weight`.
    pub(crate) fn weighted(transferred_weight: Option<f64>) -> Self {
        Self {
            weights: transferred_weight.map(|weight| {
                Box::new(TraceWeights {
                    ions: FeatureWeights::new(weight),
                    traces: HashMap::new(),
                })
            }),
            ..Self::default()
        }
    }

    /// Read the normalized species traces without changing the solver input.
    pub(crate) fn normalized_canonical_values(&self) -> CanonicalExportValues {
        let species_to_canonical = self
            .ion_to_species
            .iter()
            .map(|(ion, species)| (*species, self.ion_to_canonical[ion]))
            .collect::<HashMap<_, _>>();
        let mut traces = self.peptide_traces.iter().collect::<Vec<_>>();
        traces.sort_by_key(|(key, _)| (key.protein, key.sample, key.peptide));
        let mut values = HashMap::new();
        for (key, intensity) in traces {
            let canonical = species_to_canonical[&key.peptide];
            *values
                .entry((key.protein, key.sample, canonical))
                .or_insert(0.0) += intensity;
        }
        values
    }

    pub(crate) fn into_protein_values(
        self,
        allowed_cells: &HashSet<CellKey>,
        proteins: &StringIdRegistry<ProteinId>,
        samples: &StringIdRegistry<SampleId>,
        min_ratio_count: usize,
        stabilize: bool,
        cell_precision: bool,
    ) -> Result<(ProteinValues, HashMap<CellKey, f64>)> {
        let (traces, weights) = self.into_weighted_species_traces();
        let mut sample_ids = allowed_cells
            .iter()
            .map(|cell| cell.sample)
            .collect::<Vec<_>>();
        sample_ids.sort_by(|left, right| samples.resolve(*left).cmp(&samples.resolve(*right)));
        sample_ids.dedup();
        let mut grouped = HashMap::<ProteinId, Vec<PeptideMeasurement>>::new();
        for (key, intensity) in traces {
            let cell = CellKey {
                protein: key.protein,
                sample: key.sample,
            };
            if allowed_cells.contains(&cell) {
                grouped
                    .entry(key.protein)
                    .or_default()
                    .push(PeptideMeasurement {
                        peptide: key.peptide,
                        sample: key.sample,
                        intensity,
                        weight: weights.get(&key).copied().unwrap_or(1.0),
                    });
            }
        }
        let rows = grouped.into_iter().collect::<Vec<_>>().into_par_iter()
            .map(|(protein, measurements)| {
                let result = solve_max_lfq_with_stabilization(&measurements, &sample_ids, min_ratio_count, stabilize)?;
                if result.components.len() > 1 {
                    warn!(protein = ?proteins.resolve(protein), components = result.components.len(),
                        "MaxLFQ sample graph is disconnected; between-component ratios are not identifiable");
                }
                let values = result.intensities.into_iter().filter(|(_, x)| *x > 0.0).collect::<Vec<_>>();
                let residuals = cell_precision.then(|| maxlfq_residuals(measurements, &values));
                Ok((protein, values, result.stabilized_pairs, residuals))
            }).collect::<Result<Vec<_>>>()?;
        maxlfq::log_stabilization(min_ratio_count, stabilize, rows.iter().map(|row| row.2));
        Ok(maxlfq_values_and_precision(rows))
    }

    pub(crate) fn push(&mut self, measurement: AggregationMeasurement<'_>) {
        self.ion_to_species
            .entry(measurement.peptide)
            .or_insert(measurement.species);
        self.ion_to_canonical
            .entry(measurement.peptide)
            .or_insert(measurement.canonical);
        let cell = CellKey {
            protein: measurement.protein,
            sample: measurement.sample,
        };
        push_peptide_max(
            &mut self.ion_cells,
            cell,
            measurement.peptide,
            measurement.intensity,
        );
        if let Some(weights) = &mut self.weights {
            weights
                .ions
                .push(cell, measurement.peptide, measurement.transferred);
        }
    }

    fn collapse_ions(&mut self, preserve_species: bool) {
        let mut cells = std::mem::take(&mut self.ion_cells)
            .into_iter()
            .collect::<Vec<_>>();
        cells.sort_by_key(|(cell, _)| (cell.protein.get(), cell.sample.get()));
        let mapping = if preserve_species {
            // A cell has one loading context, so its ions are distinct species:
            // reserve them all rather than hold old and new tables on growth.
            self.peptide_traces
                .reserve(cells.iter().map(|(_, peptides)| peptides.len()).sum());
            &self.ion_to_species
        } else {
            &self.ion_to_canonical
        };
        // Collapse a block of cells in parallel, then insert it in cell order;
        // blocks bound the memory of collapsed cells awaiting insertion.
        let mut cells = cells.into_iter();
        loop {
            let block = cells.by_ref().take(COLLAPSE_BLOCK).collect::<Vec<_>>();
            if block.is_empty() {
                break;
            }
            let collapsed = block
                .into_par_iter()
                .map(|(cell, peptides)| (cell, collapse_to_canonical(peptides, mapping)))
                .collect::<Vec<_>>();
            for (cell, canonical) in collapsed {
                for (peptide, intensity) in canonical {
                    self.peptide_traces.insert(
                        PeptideCellKey {
                            protein: cell.protein,
                            sample: cell.sample,
                            peptide,
                        },
                        intensity,
                    );
                }
            }
        }
        if let Some(weights) = &mut self.weights {
            weights.collapse_ions(mapping);
        }
    }

    pub(crate) fn apply_quantile_normalization(&mut self, allowed_cells: &HashSet<CellKey>) {
        self.collapse_ions(true);
        apply_quantile_to_lfq_traces(&mut self.peptide_traces, allowed_cells);
    }

    pub(crate) fn apply_hierarchical_normalization(
        &mut self,
        allowed_cells: &HashSet<CellKey>,
        samples: &StringIdRegistry<SampleId>,
    ) {
        self.collapse_ions(true);
        apply_hierarchical_to_lfq_traces(&mut self.peptide_traces, allowed_cells, samples);
    }

    fn into_peptide_traces(mut self) -> HashMap<PeptideCellKey, f64> {
        self.collapse_ions(false);
        self.peptide_traces
    }

    /// Species traces and their weights (empty without `--transferred-weight`).
    fn into_weighted_species_traces(
        mut self,
    ) -> (HashMap<PeptideCellKey, f64>, HashMap<PeptideCellKey, f64>) {
        self.collapse_ions(true);
        let weights = self.weights.map(|weights| weights.traces);
        (self.peptide_traces, weights.unwrap_or_default())
    }

    pub(crate) fn into_directlfq_sums(self) -> Vec<(DirectLfqCellKey, f64)> {
        self.into_peptide_traces()
            .into_iter()
            .map(|(key, intensity)| {
                (
                    DirectLfqCellKey {
                        protein: key.protein,
                        canonical: key.peptide,
                        sample: key.sample,
                    },
                    intensity,
                )
            })
            .collect()
    }

    pub(crate) fn directlfq_sums(&self) -> Vec<(DirectLfqCellKey, f64)> {
        let mut sums = self
            .peptide_traces
            .iter()
            .map(|(key, intensity)| {
                (
                    DirectLfqCellKey {
                        protein: key.protein,
                        canonical: key.peptide,
                        sample: key.sample,
                    },
                    *intensity,
                )
            })
            .collect::<HashMap<_, _>>();
        let mut cells = self.ion_cells.iter().collect::<Vec<_>>();
        cells.sort_by_key(|(cell, _)| (cell.protein.get(), cell.sample.get()));
        for (cell, peptides) in cells {
            for (canonical, intensity) in
                collapse_to_canonical(peptides.clone(), &self.ion_to_canonical)
            {
                sums.insert(
                    DirectLfqCellKey {
                        protein: cell.protein,
                        canonical,
                        sample: cell.sample,
                    },
                    intensity,
                );
            }
        }
        sums.into_iter().collect()
    }
}

pub(crate) fn lexical_id_remap<I>(registry: &StringIdRegistry<I>) -> (HashMap<I, I>, HashMap<I, I>)
where
    I: Copy + Eq + std::hash::Hash + From<u32> + Into<u32>,
{
    let numeric_ids = registry.iter().map(|(id, _)| id).collect::<Vec<_>>();
    let mut by_name = registry.iter().collect::<Vec<_>>();
    by_name.sort_by(|left, right| left.1.cmp(right.1));

    let mut to_lexical = HashMap::with_capacity(by_name.len());
    let mut from_lexical = HashMap::with_capacity(by_name.len());
    for ((original, _), lexical) in by_name.into_iter().zip(numeric_ids) {
        to_lexical.insert(original, lexical);
        from_lexical.insert(lexical, original);
    }
    (to_lexical, from_lexical)
}
