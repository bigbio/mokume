pub(crate) mod batch;
mod correlation;
pub(crate) mod covariates;
mod irs;
use crate::plex::{condition_by_sample, is_reference_condition};
use crate::state::CellKey;
use crate::{
    File, HashMap, HashSet, ImputationConfig, MokumeError, OutputFormat, Path, ProteinId, Result,
    SampleId, SdrfTable, StringIdRegistry, WriterBuilder, coverage_filtered_proteins,
    create_parent_dir, csv_error, format_float, imputed_values, invalid_input, matrix,
};

#[derive(Debug)]
pub(crate) enum ProteinValues {
    Cells(HashMap<CellKey, f64>),
    Rows(HashMap<ProteinId, Vec<(SampleId, f64)>>),
}

impl ProteinValues {
    pub(crate) fn protein_ids(&self) -> HashSet<ProteinId> {
        match self {
            Self::Cells(values) => values.keys().map(|key| key.protein).collect(),
            Self::Rows(rows) => rows.keys().copied().collect(),
        }
    }

    fn sample_ids(&self) -> HashSet<SampleId> {
        match self {
            Self::Cells(values) => values.keys().map(|key| key.sample).collect(),
            Self::Rows(rows) => rows
                .values()
                .flat_map(|values| values.iter().map(|(sample, _)| *sample))
                .collect(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ProteinMatrix {
    pub(crate) proteins: StringIdRegistry<ProteinId>,
    pub(crate) samples: StringIdRegistry<SampleId>,
    pub(crate) allowed_proteins: HashSet<ProteinId>,
    pub(crate) excluded_samples: HashSet<SampleId>,
    /// Per-(protein, sample) unique-canonical-peptide count, captured at ingest
    /// before the `min_unique_peptides` cell filter. Consumed only by the DEqMS
    /// DE path (the count-aware `spectraCounteBayes` moderation); every other
    /// stage ignores it.
    pub(crate) peptide_counts: HashMap<CellKey, usize>,
    /// Per-(protein, sample) quantification precision (1/SE^2) of MaxLFQ and
    /// DirectLFQ cells, filled only for `--de-precision-weights`.
    pub(crate) precision: HashMap<CellKey, f64>,
    pub(crate) values: ProteinValues,
}
impl ProteinMatrix {
    pub(crate) fn value(&self, protein: ProteinId, sample: SampleId) -> Option<f64> {
        match &self.values {
            ProteinValues::Cells(values) => values.get(&CellKey { protein, sample }).copied(),
            ProteinValues::Rows(rows) => rows.get(&protein).and_then(|values| {
                values
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == sample).then_some(*value))
            }),
        }
    }

    /// Build the log2-transformed protein x sample matrix over `samples`
    /// (column order preserved) for differential expression. Each cell is
    /// `log2(value)` when the linear intensity is finite and strictly positive;
    /// zero, missing, and non-positive values become `NaN`, while infinities
    /// are rejected. Rows are emitted for the kept proteins in matrix output
    /// order (sorted by accession) so protein identity lines up with the
    /// returned names.
    pub(crate) fn log2_rows(&self, samples: &[SampleId]) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));

        let mut names = Vec::with_capacity(proteins.len());
        let mut rows = Vec::with_capacity(proteins.len());
        for (protein, accession) in proteins {
            names.push(accession.to_owned());
            let mut row = Vec::with_capacity(samples.len());
            for sample in samples {
                let value = match self.value(protein, *sample) {
                    Some(value) if value.is_infinite() => {
                        return Err(invalid_input(format!(
                            "differential-expression matrix contains an infinite intensity for protein `{accession}`"
                        )));
                    }
                    Some(value) if value.is_finite() && value > 0.0 => value.log2(),
                    _ => f64::NAN,
                };
                row.push(value);
            }
            rows.push(row);
        }
        Ok((names, rows))
    }

    /// Whether any kept protein lacks a finite, strictly positive value in one
    /// of `samples`, i.e. whether [`Self::log2_rows`] over them would emit `NaN`.
    pub(crate) fn has_missing_values(&self, samples: &[SampleId]) -> bool {
        self.proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .any(|(protein, _)| {
                samples.iter().any(|sample| {
                    !self
                        .value(protein, *sample)
                        .is_some_and(|value| value.is_finite() && value > 0.0)
                })
            })
    }

    /// Fewest unique canonical peptides of the protein named `name` (the
    /// accession string `log2_rows` emits) in any of `samples`; a sample without
    /// the protein, or an unknown name, counts 0.
    pub(crate) fn min_sample_peptide_count(&self, name: &str, samples: &[SampleId]) -> usize {
        let Some(protein) = self.proteins.get(name) else {
            return 0;
        };
        samples
            .iter()
            .map(|&sample| {
                self.peptide_counts
                    .get(&CellKey { protein, sample })
                    .copied()
                    .unwrap_or(0)
            })
            .min()
            .unwrap_or(0)
    }

    /// Write each cell's quantification precision in the layout of the protein
    /// matrix; a cell without one stays empty.
    pub(crate) fn write_precision_csv(
        &mut self,
        path: &Path,
        output_format: OutputFormat,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let mut values = ProteinValues::Cells(std::mem::take(&mut self.precision));
        std::mem::swap(&mut self.values, &mut values);
        let written = self.write_csv(path, output_format, drop_empty_samples, None);
        std::mem::swap(&mut self.values, &mut values);
        if let ProteinValues::Cells(precision) = values {
            self.precision = precision;
        }
        written
    }

    /// Cell precisions of protein `name` over `samples`, zero where none.
    pub(crate) fn sample_precisions(&self, name: &str, samples: &[SampleId]) -> Vec<f64> {
        let protein = self.proteins.get(name);
        samples
            .iter()
            .map(|&sample| {
                protein
                    .and_then(|protein| self.precision.get(&CellKey { protein, sample }))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }

    fn set_value(&mut self, protein: ProteinId, sample: SampleId, value: f64) {
        match &mut self.values {
            ProteinValues::Cells(values) => {
                values.insert(CellKey { protein, sample }, value);
            }
            ProteinValues::Rows(rows) => {
                let values = rows.entry(protein).or_default();
                if let Some((_, current)) = values
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == sample)
                {
                    *current = value;
                } else {
                    values.push((sample, value));
                    values.sort_by_key(|(sample, _)| sample.get());
                }
            }
        }
    }
}
impl ProteinMatrix {
    pub(crate) fn apply_coverage_filter(
        &mut self,
        sdrf: Option<&SdrfTable>,
        threshold: f64,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("coverage filtering requires --sdrf option"));
        };
        let condition_by_sample = condition_by_sample(sdrf);
        let mut samples_by_condition = HashMap::<String, Vec<SampleId>>::new();
        for (sample, sample_name) in self.sample_columns(drop_empty_samples) {
            let Some(condition) = condition_by_sample.get(sample_name) else {
                continue;
            };
            if is_reference_condition(condition) {
                continue;
            }
            samples_by_condition
                .entry(condition.clone())
                .or_default()
                .push(sample);
        }
        if samples_by_condition.is_empty() {
            return Err(invalid_input(
                "coverage filtering found no non-reference samples with condition metadata",
            ));
        }

        self.allowed_proteins = coverage_filtered_proteins(
            self.allowed_proteins.iter().copied(),
            &samples_by_condition,
            threshold,
            |protein, sample| self.value(protein, sample),
        );
        Ok(())
    }

    pub(crate) fn apply_imputation(
        &mut self,
        config: &ImputationConfig,
        drop_empty_samples: bool,
    ) -> Result<()> {
        let (proteins, samples) = self.imputation_axes(drop_empty_samples);
        if proteins.iter().any(|protein| {
            samples
                .iter()
                .any(|sample| self.value(*protein, *sample).is_some_and(f64::is_infinite))
        }) {
            return Err(invalid_input(
                "imputation matrix contains an infinite intensity",
            ));
        }
        // Match the Python pipeline, which imputes in log2 space and converts
        // back to linear: positive finite values are log2-transformed, missing
        // and non-positive values stay missing, and infinities are rejected.
        let fills = imputed_values(config, &proteins, &samples, |protein, sample| {
            self.value(protein, sample)
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(f64::log2)
        })?;
        for (protein, sample, value) in fills {
            let intensity = matrix::checked_imputed_intensity(value)?;
            self.set_value(protein, sample, intensity);
        }
        Ok(())
    }

    fn imputation_axes(&self, drop_empty_samples: bool) -> (Vec<ProteinId>, Vec<SampleId>) {
        let mut proteins = self.allowed_proteins.iter().copied().collect::<Vec<_>>();
        proteins.sort_by_key(|protein| protein.get());
        let samples = self
            .sample_columns(drop_empty_samples)
            .into_iter()
            .map(|(sample, _)| sample)
            .collect::<Vec<_>>();
        (proteins, samples)
    }

    pub(crate) fn write_csv(
        &self,
        path: &Path,
        output_format: OutputFormat,
        drop_empty_samples: bool,
        missing_fill: Option<f64>,
    ) -> Result<()> {
        create_parent_dir(path)?;
        let file = File::create(path).map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = WriterBuilder::new().from_writer(file);
        let samples = self.sample_columns(drop_empty_samples);
        let sample_index = samples
            .iter()
            .enumerate()
            .map(|(index, (sample, _))| (*sample, index))
            .collect::<HashMap<_, _>>();
        let column_missing = self.missing_column_values(&samples, missing_fill);
        let mut header = Vec::with_capacity(samples.len() + 1);
        header.push(output_format.protein_column().to_owned());
        header.extend(samples.iter().map(|(_, sample)| (*sample).to_owned()));
        writer
            .write_record(header)
            .map_err(|source| csv_error(path, source))?;

        self.write_protein_csv_rows(&mut writer, path, &sample_index, &column_missing)?;
        writer.flush().map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    fn missing_column_values(
        &self,
        samples: &[(SampleId, &str)],
        missing_fill: Option<f64>,
    ) -> Vec<String> {
        // Per-column fill for cells with no observed (positive, finite) value.
        // Mirrors Python's `pivot_table(fill_value=0)` over the observed samples:
        // the additive methods (sum / intensity / peptide_count / spectral_count /
        // directlfq) write `0` for a missing cell, the average/ratio methods
        // (median / topn / abd / maxlfq / ratio) leave it empty (NaN). A sample
        // with no observations at all stays empty even for the additive methods,
        // because Python's pivot never densifies it (it is re-added as a NaN
        // column), not 0.
        let observed = self.observed_samples();
        samples
            .iter()
            .map(|(sample, _)| match missing_fill {
                Some(value) if observed.contains(sample) => format_float(value),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
    }

    fn write_protein_csv_rows(
        &self,
        writer: &mut csv::Writer<File>,
        path: &Path,
        sample_index: &HashMap<SampleId, usize>,
        column_missing: &[String],
    ) -> Result<()> {
        let mut proteins = self
            .proteins
            .iter()
            .filter(|(protein, _)| self.allowed_proteins.contains(protein))
            .collect::<Vec<_>>();
        proteins.sort_by(|left, right| left.1.cmp(right.1));
        for (protein, accession) in proteins {
            if !self.allowed_proteins.contains(&protein) {
                continue;
            }
            let mut row_values = column_missing.to_vec();
            self.fill_row_values(protein, sample_index, &mut row_values);
            let mut row = Vec::with_capacity(column_missing.len() + 1);
            row.push(accession.to_owned());
            row.extend(row_values);
            writer
                .write_record(row)
                .map_err(|source| csv_error(path, source))?;
        }
        Ok(())
    }

    /// Samples carrying at least one observed (finite, strictly positive) value.
    /// These are the columns Python's `pivot_table(fill_value=0)` densifies; a
    /// sample absent from this set has no observations and stays empty even for the
    /// additive (0-fill) methods.
    fn observed_samples(&self) -> HashSet<SampleId> {
        let mut observed = HashSet::new();
        match &self.values {
            ProteinValues::Cells(values) => {
                for (key, value) in values {
                    if value.is_finite() && *value > 0.0 {
                        observed.insert(key.sample);
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                for values in rows.values() {
                    for (sample, value) in values {
                        if value.is_finite() && *value > 0.0 {
                            observed.insert(*sample);
                        }
                    }
                }
            }
        }
        observed
    }

    fn fill_row_values(
        &self,
        protein: ProteinId,
        sample_index: &HashMap<SampleId, usize>,
        row_values: &mut [String],
    ) {
        // Only a finite, strictly positive intensity counts as observed; a missing
        // cell (absent, or the `0` directlfq/maxlfq emit for an unquantified sample)
        // keeps the caller's per-column `column_missing` default. This is what lets
        // the same dense directlfq solver feed `directlfq` (missing -> `0`) and
        // `maxlfq` (missing -> empty) without re-running the quantification.
        match &self.values {
            ProteinValues::Cells(values) => {
                for (sample, index) in sample_index {
                    let value = values.get(&CellKey {
                        protein,
                        sample: *sample,
                    });
                    if let Some(value) = value
                        && value.is_finite()
                        && *value > 0.0
                    {
                        row_values[*index] = format_float(*value);
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                if let Some(values) = rows.get(&protein) {
                    for (sample, value) in values {
                        if let Some(index) = sample_index.get(sample)
                            && value.is_finite()
                            && *value > 0.0
                        {
                            row_values[*index] = format_float(*value);
                        }
                    }
                }
            }
        }
    }
}
