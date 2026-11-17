use crate::pibaq::allocation::plex_median_factors;
use crate::plex::resolve_irs_plexes;
use crate::protein_matrix::{ProteinMatrix, ProteinValues};
use crate::ratio::resolve_irs_reference_samples;
use crate::{
    BTreeSet, HashMap, IrsConfig, IrsMissingReference, ProteinId, Result, SampleId, SdrfRawTable,
    SdrfTable, info, invalid_input, irs_scaling_factors,
};
impl ProteinMatrix {
    pub(crate) fn apply_irs(
        &mut self,
        sdrf: Option<&SdrfTable>,
        raw_sdrf: Option<&SdrfRawTable>,
        config: &IrsConfig,
    ) -> Result<()> {
        let Some(sdrf) = sdrf else {
            return Err(invalid_input("IRS normalization requires --sdrf option"));
        };
        let raw_sdrf = raw_sdrf.ok_or_else(|| {
            invalid_input("IRS normalization requires readable raw SDRF metadata")
        })?;
        let reference_samples = resolve_irs_reference_samples(sdrf, raw_sdrf, config)?;
        if reference_samples.is_empty() {
            return Err(invalid_input(
                "IRS normalization found no reference samples in the SDRF",
            ));
        }

        let (sample_to_plex, plexes) = resolve_irs_plexes(sdrf, raw_sdrf, config)?;

        let sample_by_name = self
            .samples
            .iter()
            .map(|(sample, name)| (name.to_owned(), sample))
            .collect::<HashMap<_, _>>();
        let mut refs_by_plex = HashMap::<String, Vec<SampleId>>::new();
        for reference in &reference_samples {
            let Some(sample) = sample_by_name.get(reference).copied() else {
                continue;
            };
            let Some(plex) = sample_to_plex.get(reference) else {
                continue;
            };
            refs_by_plex.entry(plex.clone()).or_default().push(sample);
        }
        for plex in &plexes {
            if !refs_by_plex.contains_key(plex) {
                return Err(invalid_input(format!(
                    "No reference samples found in plex `{plex}`"
                )));
            }
        }

        let mut factors = irs_scaling_factors(
            self.allowed_proteins.iter().copied(),
            &plexes,
            &refs_by_plex,
            &config.stat,
            |protein, sample| self.value(protein, sample),
        );
        if factors.is_empty() {
            return Err(invalid_input(
                "IRS normalization could not compute any finite scaling factors",
            ));
        }
        let missing = self.irs_unreferenced_pairs(&sample_to_plex, &factors);
        info!(
            protein_plex_pairs = missing.len(),
            mode = config.missing_reference.label(),
            "IRS: proteins with values but no reference value in a plex"
        );
        if config.missing_reference == IrsMissingReference::PlexMedian {
            let medians = plex_median_factors(&factors);
            for (protein, plex) in &missing {
                if let Some(median) = medians.get(plex) {
                    factors
                        .entry(*protein)
                        .or_default()
                        .insert(plex.clone(), *median);
                }
            }
        }
        self.scale_by_irs(&sample_to_plex, &factors);
        if config.missing_reference == IrsMissingReference::Drop {
            self.drop_protein_plex_values(&sample_to_plex, &missing);
        }

        if config.remove_reference {
            for reference in reference_samples {
                if let Some(sample) = sample_by_name.get(&reference).copied() {
                    self.excluded_samples.insert(sample);
                }
            }
        }
        Ok(())
    }

    /// (protein, plex) pairs that have values but no IRS factor.
    fn irs_unreferenced_pairs(
        &self,
        sample_to_plex: &HashMap<String, String>,
        factors: &HashMap<ProteinId, HashMap<String, f64>>,
    ) -> BTreeSet<(ProteinId, String)> {
        let plex_of = self.sample_plex_ids(sample_to_plex);
        let mut missing = BTreeSet::new();
        let mut visit = |protein: ProteinId, sample: SampleId, value: f64| {
            if !value.is_finite() {
                return;
            }
            let Some(plex) = plex_of.get(&sample) else {
                return;
            };
            if !factors
                .get(&protein)
                .is_some_and(|by_plex| by_plex.contains_key(plex))
            {
                missing.insert((protein, plex.clone()));
            }
        };
        match &self.values {
            ProteinValues::Cells(values) => {
                for (key, value) in values {
                    visit(key.protein, key.sample, *value);
                }
            }
            ProteinValues::Rows(rows) => {
                for (protein, values) in rows {
                    for (sample, value) in values {
                        visit(*protein, *sample, *value);
                    }
                }
            }
        }
        missing
    }

    fn sample_plex_ids(
        &self,
        sample_to_plex: &HashMap<String, String>,
    ) -> HashMap<SampleId, String> {
        self.samples
            .iter()
            .filter_map(|(sample, name)| {
                sample_to_plex.get(name).map(|plex| (sample, plex.clone()))
            })
            .collect()
    }

    fn drop_protein_plex_values(
        &mut self,
        sample_to_plex: &HashMap<String, String>,
        pairs: &BTreeSet<(ProteinId, String)>,
    ) {
        if pairs.is_empty() {
            return;
        }
        let plex_of = self.sample_plex_ids(sample_to_plex);
        let keep = |protein: ProteinId, sample: &SampleId| {
            plex_of
                .get(sample)
                .is_none_or(|plex| !pairs.contains(&(protein, plex.clone())))
        };
        match &mut self.values {
            ProteinValues::Cells(values) => values.retain(|key, _| keep(key.protein, &key.sample)),
            ProteinValues::Rows(rows) => {
                for (protein, values) in rows.iter_mut() {
                    values.retain(|(sample, _)| keep(*protein, sample));
                }
            }
        }
    }

    fn scale_by_irs(
        &mut self,
        sample_to_plex: &HashMap<String, String>,
        factors: &HashMap<ProteinId, HashMap<String, f64>>,
    ) {
        let sample_to_plex = self
            .samples
            .iter()
            .filter_map(|(sample, name)| {
                sample_to_plex.get(name).map(|plex| (sample, plex.clone()))
            })
            .collect::<HashMap<_, _>>();
        match &mut self.values {
            ProteinValues::Cells(values) => {
                for (key, value) in values {
                    if let Some(factor) = sample_to_plex.get(&key.sample).and_then(|plex| {
                        factors
                            .get(&key.protein)
                            .and_then(|by_plex| by_plex.get(plex))
                    }) {
                        *value *= *factor;
                    }
                }
            }
            ProteinValues::Rows(rows) => {
                for (protein, values) in rows {
                    for (sample, value) in values {
                        if let Some(factor) = sample_to_plex.get(sample).and_then(|plex| {
                            factors.get(protein).and_then(|by_plex| by_plex.get(plex))
                        }) {
                            *value *= *factor;
                        }
                    }
                }
            }
        }
    }
}
