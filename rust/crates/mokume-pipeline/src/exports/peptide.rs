use crate::accessions::register_id;
use crate::aggregation::tech_replicate_of;
use crate::exports::{
    CanonicalExportValues, ExportIons, IntermediateMeasurement, IonValues, PeptideExport,
    PeptideExportContext, PeptideExportKey, PeptideExportNames, PeptideExportRow, RazorHandling,
    parquet_sibling,
};
use crate::state::CellKey;
use crate::{
    File, HashMap, HashSet, MokumeError, Path, PeptideId, PeptideParquetRow, ProteinId, Result,
    SampleId, WriterBuilder, create_parent_dir, csv_error, format_float, invalid_input,
    write_peptide_parquet,
};

impl PeptideExport {
    fn context(&mut self, measurement: IntermediateMeasurement<'_>) -> Result<u32> {
        let bio = measurement
            .sdrf_record
            .and_then(|record| record.biological_replicate)
            .unwrap_or(1);
        // On an SDRF miss Python's `_create_unnest_view` / `enrich_with_sdrf`
        // fall the Condition column back to the `sa_fallback` (`run_file_name`
        // for new QPX, the unnested `sample_accession` for legacy), which is
        // exactly the value `sample_name()` already resolves to. This is
        // `sample_condition` without allocating for every row.
        let condition = measurement
            .sdrf_record
            .and_then(|record| record.condition.as_deref())
            .unwrap_or(measurement.sample_name);
        let condition_id = register_id(&mut self.context_labels, condition, "export condition")?;
        let (run, tech) = if self.run_mode {
            (
                Some(measurement.run_file_name),
                measurement
                    .sdrf_record
                    .and_then(|record| record.technical_replicate)
                    .map(i64::from)
                    .unwrap_or_else(|| tech_replicate_of(measurement.run_file_name)),
            )
        } else {
            (None, 0)
        };
        let run_id = run
            .map(|name| register_id(&mut self.context_labels, name, "export run"))
            .transpose()?;
        let key = (bio, condition_id, run_id, tech);
        if let Some(id) = self.context_ids.get(&key) {
            return Ok(*id);
        }
        let id = u32::try_from(self.contexts.len())
            .map_err(|_| invalid_input("export context id overflow"))?;
        self.contexts.push(PeptideExportContext {
            bio_replicate: bio.to_string(),
            condition: condition.to_owned(),
            run: run.map(str::to_owned),
            tech_replicate: tech,
        });
        self.context_ids.insert(key, id);
        Ok(id)
    }

    pub(crate) fn push(
        &mut self,
        measurement: IntermediateMeasurement<'_>,
        unnormalized: f64,
    ) -> Result<()> {
        let key = PeptideExportKey {
            protein: measurement.protein,
            sample: measurement.sample,
            canonical: measurement.canonical,
            context: self.context(measurement)?,
        };
        // Record the first-appearance order of proteins per `(sample, canonical)`
        // for the `AssignToTop` tie-break. Only this mode needs it, so the other
        // modes pay nothing here. Push is called in parquet row order, so the vec
        // ends up in pandas `unique()` order (first occurrence wins).
        if self.razor_handling == RazorHandling::AssignToTop {
            let order = self
                .razor_order
                .entry((measurement.sample, measurement.peptide_canonical.to_owned()))
                .or_default();
            if !order.iter().any(|name| name == measurement.protein_name) {
                order.push(measurement.protein_name.to_owned());
            }
        }
        let peptide = measurement.peptide;
        let values = IonValues {
            intensity: measurement.intensity,
            unnormalized,
        };
        self.rows
            .entry(key)
            .and_modify(|ions| ions.merge(peptide, values))
            .or_insert(ExportIons::One(peptide, values));
        Ok(())
    }

    /// Filter and sort the row keys shared by both exports. Drops cells failing
    /// the unique-peptide gate and the `(ProteinName, PeptideCanonical)`
    /// low-frequency set (Python `peptide.py:373-378`; empty when
    /// `--remove_low_frequency_peptides` is off).
    pub(crate) fn ordered_keys<'a>(
        &'a self,
        allowed_cells: &HashSet<CellKey>,
        low_frequency_peptides: &HashSet<(String, String)>,
        names: &PeptideExportNames<'_>,
    ) -> Vec<&'a PeptideExportKey> {
        let low_frequency = low_frequency_peptides
            .iter()
            .map(|(protein, canonical)| (protein.as_str(), canonical.as_str()))
            .collect::<HashSet<_>>();
        let mut keys = self
            .rows
            .keys()
            .filter(|key| {
                allowed_cells.contains(&CellKey {
                    protein: key.protein,
                    sample: key.sample,
                }) && !low_frequency.contains(&(
                    names.proteins[key.protein.get() as usize],
                    names.canonicals[key.canonical.get() as usize],
                ))
            })
            .collect::<Vec<_>>();
        self.filter_razor(&mut keys, names);
        keys.sort_by(|left, right| names.row(left, self).cmp(&names.row(right, self)));
        keys
    }

    /// Yield rows for the CSV and parquet writers without keeping them, so both
    /// serialisations are value-identical. Mirrors `sum_peptidoform_intensities`
    /// (the per-cell sum) followed by the optional `--log2`.
    pub(crate) fn exported_rows<'a>(
        &'a self,
        keys: &'a [&'a PeptideExportKey],
        unnormalized: bool,
        log2: bool,
        dataset_normalized: Option<&'a CanonicalExportValues>,
        names: &'a PeptideExportNames<'a>,
    ) -> impl Iterator<Item = (PeptideExportRow<'a>, f64)> + 'a {
        keys.iter().map(move |key| {
            // When a dataset-level sample normalization was applied, use the
            // normalized canonical value (Python normalizes the peptide table
            // before export) instead of the raw sum, summed in `PeptideId` order.
            let raw = dataset_normalized
                .and_then(|values| values.get(&(key.protein, key.sample, key.canonical)))
                .copied()
                .unwrap_or_else(|| {
                    self.rows[*key]
                        .iter()
                        .map(|(_, values)| values.value(unnormalized))
                        .sum()
                });
            (names.row(key, self), if log2 { raw.log2() } else { raw })
        })
    }

    /// RazorPeptideFilter (Python protein.py:348-378), applied after the
    /// `min_unique` gate as in the Python pipeline. A razor peptide is a
    /// canonical mapping to more than one protein within a sample; `Remove`
    /// drops all its rows. Python re-applies no `min_unique` gate afterwards.
    fn filter_razor(&self, keys: &mut Vec<&PeptideExportKey>, names: &PeptideExportNames<'_>) {
        if self.razor_handling == RazorHandling::Keep {
            return;
        }
        let mut canonical_proteins: HashMap<(SampleId, PeptideId), HashSet<ProteinId>> =
            HashMap::new();
        for key in keys.iter() {
            canonical_proteins
                .entry((key.sample, key.canonical))
                .or_default()
                .insert(key.protein);
        }
        if self.razor_handling == RazorHandling::Remove {
            keys.retain(|key| canonical_proteins[&(key.sample, key.canonical)].len() <= 1);
        } else {
            self.assign_razor(keys, &canonical_proteins, names);
        }
    }

    /// `assign_to_top`: for a razor canonical, keep only the rows of the protein
    /// with the most unique peptides in the sample. Ties go to the protein that
    /// arrived earliest in parquet row order, matching pandas `unique()` ordering
    /// and `max`'s keep-first behaviour. Non-razor canonicals stay untouched.
    fn assign_razor(
        &self,
        keys: &mut Vec<&PeptideExportKey>,
        canonical_proteins: &HashMap<(SampleId, PeptideId), HashSet<ProteinId>>,
        names: &PeptideExportNames<'_>,
    ) {
        let mut protein_canonicals: HashMap<(SampleId, &str), HashSet<PeptideId>> = HashMap::new();
        for key in keys.iter() {
            protein_canonicals
                .entry((key.sample, names.proteins[key.protein.get() as usize]))
                .or_default()
                .insert(key.canonical);
        }
        keys.retain(|key| {
            if canonical_proteins[&(key.sample, key.canonical)].len() <= 1 {
                return true;
            }
            let canonical = names.canonicals[key.canonical.get() as usize];
            let order = self.razor_order.get(&(key.sample, canonical.to_owned()));
            let top = order.and_then(|order| {
                let mut best: Option<(&str, usize)> = None;
                for name in order {
                    let count = protein_canonicals
                        .get(&(key.sample, name.as_str()))
                        .map_or(0, HashSet::len);
                    if best.is_none_or(|(_, best_count)| count > best_count) {
                        best = Some((name.as_str(), count));
                    }
                }
                best.map(|(name, _)| name)
            });
            // If the order is somehow missing, drop the razor rows like `remove`.
            top.is_some_and(|top| names.proteins[key.protein.get() as usize] == top)
        });
    }

    fn csv_header(&self, omit_replicate_condition: bool) -> Vec<&'static str> {
        let mut header = vec!["ProteinName", "PeptideCanonical", "SampleID"];
        if !omit_replicate_condition {
            header.extend(["BioReplicate", "Condition"]);
        }
        if self.run_mode {
            header.extend(["Run", "TechReplicate"]);
        }
        header.push("NormIntensity");
        header
    }

    pub(crate) fn write_csv<'a>(
        &self,
        path: &Path,
        rows: impl Iterator<Item = (PeptideExportRow<'a>, f64)>,
        omit_replicate_condition: bool,
        save_parquet: bool,
    ) -> Result<()> {
        create_parent_dir(path)?;
        let file = File::create(path).map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = WriterBuilder::new().from_writer(file);
        writer
            .write_record(self.csv_header(omit_replicate_condition))
            .map_err(|source| csv_error(path, source))?;
        let mut parquet_rows = Vec::new();
        for (row, value) in rows {
            row.write_csv(
                &mut writer,
                path,
                value,
                omit_replicate_condition,
                self.run_mode,
            )?;
            if save_parquet {
                parquet_rows.push(row.parquet(value));
            }
        }
        writer.flush().map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if save_parquet {
            write_peptide_parquet(&parquet_sibling(path), &parquet_rows, self.run_mode)?;
        }
        Ok(())
    }
}

impl PeptideExportRow<'_> {
    fn write_csv(
        &self,
        writer: &mut csv::Writer<File>,
        path: &Path,
        value: f64,
        omit: bool,
        run_mode: bool,
    ) -> Result<()> {
        let bio = self.bio_replicate.parse::<i32>().unwrap_or(1).to_string();
        let intensity = format_float(value);
        let mut record = vec![self.protein_name, self.peptide_canonical, self.sample_name];
        if !omit {
            record.extend([bio.as_str(), self.condition]);
        }
        let tech = if self.run.is_some() {
            self.tech_replicate
        } else {
            1
        }
        .to_string();
        if run_mode {
            record.extend([self.run.unwrap_or_default(), tech.as_str()]);
        }
        record.push(&intensity);
        writer
            .write_record(record)
            .map_err(|source| csv_error(path, source))
    }

    fn parquet(&self, value: f64) -> PeptideParquetRow {
        PeptideParquetRow {
            protein_name: self.protein_name.to_owned(),
            peptide_canonical: self.peptide_canonical.to_owned(),
            sample_id: self.sample_name.to_owned(),
            bio_replicate: self.bio_replicate.parse::<i32>().unwrap_or(1),
            condition: self.condition.to_owned(),
            run: self.run.map(str::to_owned),
            tech_replicate: self.run.map(|_| self.tech_replicate),
            norm_intensity: value,
        }
    }
}
