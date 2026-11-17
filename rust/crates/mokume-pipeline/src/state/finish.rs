use crate::aggregation::FeatureAggregation;
use crate::exports::{CanonicalExportValues, PeptideExportNames, PeptideExportOptions};
use crate::peptide_normalization::{
    apply_dataset_norm_to_peptide_cells, collapse_to_canonical, writes_canonical_cells,
};
use crate::protein_matrix::ProteinMatrix;
use crate::state::{CellKey, FeatureToProteinState, LowFrequencyTracker};
use crate::{
    FeatureToProteinsConfig, HashMap, HashSet, PeptideId, Result, SampleNormalizationMethod,
};
impl FeatureToProteinState {
    pub(crate) fn write_intermediate_exports(
        &mut self,
        config: &FeatureToProteinsConfig,
        allowed_cells: &HashSet<CellKey>,
        options: PeptideExportOptions,
        dataset_normalization: Option<SampleNormalizationMethod>,
    ) -> Result<()> {
        let low_frequency_peptides = self
            .low_frequency
            .as_ref()
            .map(LowFrequencyTracker::low_frequency_peptides)
            .unwrap_or_default();
        // Read MaxLFQ's already-normalized species traces. Cell-based methods
        // reuse their normalization helper on cells rebuilt from export rows.
        let dataset_normalized = dataset_normalization
            .map(|method| self.export_dataset_normalized_values(method, allowed_cells));
        self.export_rows.write(
            config,
            allowed_cells,
            &low_frequency_peptides,
            options,
            dataset_normalized.as_ref(),
            &PeptideExportNames {
                proteins: self.proteins.iter().map(|(_, name)| name).collect(),
                samples: self.samples.iter().map(|(_, name)| name).collect(),
                canonicals: self
                    .canonical_peptides
                    .iter()
                    .map(|(_, name)| name)
                    .collect(),
            },
        )?;
        self.aggregation.write_intermediate_exports(
            config,
            &self.proteins,
            &self.canonical_peptides,
            &self.samples,
            allowed_cells,
        )
    }

    /// Dataset-normalized canonical values for `--export-peptides`. MaxLFQ
    /// reads its actual solver traces; cell-based methods rebuild the same
    /// max-merged `(protein, sample) -> {peptide: intensity}` cells the
    /// aggregation holds (both come from `push_peptide_max` over the same
    /// ingest), then apply the identical [`apply_dataset_norm_to_peptide_cells`]
    /// helper, so the exported peptides match the protein-matrix normalization.
    fn export_dataset_normalized_values(
        &self,
        method: SampleNormalizationMethod,
        allowed_cells: &HashSet<CellKey>,
    ) -> CanonicalExportValues {
        let mut result = HashMap::new();
        if !self.export_rows.peptides {
            return result;
        }
        if let FeatureAggregation::Lfq {
            route_to_directlfq: false,
            traces,
            ..
        } = &self.aggregation
        {
            return traces.normalized_canonical_values();
        }
        let mut cells = self.export_peptide_cells();
        apply_dataset_norm_to_peptide_cells(
            &mut cells,
            method,
            allowed_cells,
            &self.index.peptide_to_canonical,
            &self.samples,
        );
        for (cell, values) in cells {
            // TMM scales contextual ions in place; the other normalizers
            // already replace them with canonical values.
            let canonical_values = if method == SampleNormalizationMethod::Tmm {
                collapse_to_canonical(values, &self.index.peptide_to_canonical)
            } else {
                values
            };
            for (canonical, value) in canonical_values {
                result.insert((cell.protein, cell.sample, canonical), value);
            }
        }
        result
    }

    fn export_peptide_cells(&self) -> HashMap<CellKey, HashMap<PeptideId, f64>> {
        let mut cells: HashMap<CellKey, HashMap<PeptideId, f64>> = HashMap::new();
        for (key, ions) in &self.export_rows.rows.rows {
            let cell = CellKey {
                protein: key.protein,
                sample: key.sample,
            };
            let target = cells.entry(cell).or_default();
            for (peptide, values) in ions.iter() {
                let current = target.entry(peptide).or_insert(0.0);
                if values.intensity > *current {
                    *current = values.intensity;
                }
            }
        }
        cells
    }

    pub(crate) fn into_matrix(
        mut self,
        config: &FeatureToProteinsConfig,
        min_unique_peptides: usize,
        dataset_normalization: Option<SampleNormalizationMethod>,
    ) -> Result<ProteinMatrix> {
        let allowed_cells = self.allowed_cells(min_unique_peptides);
        // Per-(protein, sample) unique-canonical-peptide counts for the DEqMS DE
        // path, captured BEFORE the `min_unique_peptides` cell filter.
        let peptide_counts = self.cell_peptide_counts();
        // The per-cell unique-peptide sets only feed the two lookups above.
        self.index.unique_peptides = HashMap::new();
        if let Some(method) = dataset_normalization
            && self.aggregation.applies_dataset_normalization()
        {
            self.aggregation.apply_dataset_normalization(
                method,
                &allowed_cells,
                &self.index.peptide_to_canonical,
                &self.samples,
            );
        }
        let options = PeptideExportOptions::default();
        self.write_intermediate_exports(config, &allowed_cells, options, dataset_normalization)?;
        let empty_mapping = HashMap::new();
        // TMM scales contextual peptidoforms in place, so `finalize` must still
        // collapse them; the other dataset normalizers write canonical cells.
        let collapse_mapping = if dataset_normalization.is_some_and(writes_canonical_cells)
            && self.aggregation.applies_dataset_normalization()
        {
            &empty_mapping
        } else {
            &self.index.peptide_to_contextual_canonical
        };
        let (values, precision) = self.aggregation.finalize(
            &allowed_cells,
            &mut self.proteins,
            collapse_mapping,
            &self.canonical_peptides,
            &self.samples,
        )?;
        let allowed_proteins = values.protein_ids();
        Ok(ProteinMatrix {
            proteins: self.proteins,
            samples: self.samples,
            allowed_proteins,
            excluded_samples: HashSet::new(),
            peptide_counts,
            precision,
            values,
        })
    }

    /// Distinct canonical peptides (from `feature.sequence`) per (protein,
    /// sample) cell over the whole ingested set, with no `min_unique_peptides`
    /// filter applied.
    fn cell_peptide_counts(&self) -> HashMap<CellKey, usize> {
        self.index
            .unique_peptides
            .iter()
            .map(|(cell, peptides)| (*cell, peptides.len()))
            .collect()
    }

    pub(crate) fn allowed_cells(&self, min_unique_peptides: usize) -> HashSet<CellKey> {
        self.index
            .unique_peptides
            .iter()
            .filter_map(|(cell, peptides)| (peptides.len() >= min_unique_peptides).then_some(*cell))
            .collect()
    }
}
