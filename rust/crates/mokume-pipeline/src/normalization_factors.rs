use crate::accessions::{
    matches_sql_contaminant, parse_protein_accession, protein_group_name, sample_name, sdrf_record,
};
use crate::aggregation::IntensityFactors;
use crate::qpx_stream::stream_prepass_features;
use crate::{
    FeatureToProteinsConfig, FilterConfig, HashMap, HashSet, MemoryPlan, MokumeError,
    NamedScoreFilterConfig, Path, QpxFeatureRecord, QuantMethod, Result, RunCellKey,
    RunNormalizationMethod, SampleNormalizationMethod, SdrfLookupCache, SdrfRecord, SdrfTable,
    condition_median_sample_factors, global_median_sample_factors, invalid_input,
    parse_run_normalization_method, parse_sample_normalization_method, read_to_string,
    run_normalization_transforms, tmt,
};

pub(crate) fn collect_intensity_factors(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    contaminant_patterns: &[String],
    keep_shared_peptides: bool,
    memory: &MemoryPlan,
    named_score: Option<&NamedScoreFilterConfig>,
    tmt: Option<&mut tmt::TmtRowProcessor>,
) -> Result<IntensityFactors> {
    if matches!(
        config.quantification,
        QuantMethod::DirectLfq | QuantMethod::Ratio
    ) {
        return Ok(IntensityFactors::default());
    }

    let run_method = parse_run_normalization_method(&config.normalization.run_method)?;
    let sample_method = parse_sample_normalization_method(&config.normalization.sample_method)?;
    let sample_uses_factors = sample_method_uses_factors(sample_method);
    if run_method.is_none() && !sample_uses_factors {
        return Ok(IntensityFactors::default());
    }
    let normalization_proteins = config
        .normalization
        .normalization_proteins
        .as_deref()
        .filter(|_| sample_uses_factors)
        .map(load_normalization_proteins)
        .transpose()?;
    let mut collector = NormalizationFactorCollector::new(
        run_method,
        sample_method,
        sdrf,
        normalization_proteins,
        contaminant_patterns,
    );
    // `keep_shared_peptides` mirrors Python's `require_unique = not keep_shared`
    // in `SQLFilterBuilder` (`peptide.py:168/176`, `stages.py:329`): when shared
    // peptides are kept, the median pre-pass must include non-unique rows so the
    // per-sample median (and the resulting sample/run factors) matches Python.
    // The caller decides the value: the protein path keeps the legacy
    // `quantification == Pibaq` rule (`stages.py:325`), while the peptide path
    // forwards the `--keep-shared-peptides` flag from `FeatureToPeptidesConfig`,
    // which `peptide_export_config` cannot represent (it pins quantification to
    // `Sum`, so the old in-function `== Pibaq` derivation always read `false`).
    stream_prepass_features(&config.input, sdrf, memory, named_score, tmt, |feature| {
        collector.push(feature, config.filtering, keep_shared_peptides)
    })?;
    if collector.normalization_proteins.is_some()
        && sample_uses_factors
        && !collector.has_sample_factor_values()
    {
        return Err(invalid_input(
            "no features matched --normalization-proteins for sample normalization",
        ));
    }
    Ok(collector.into_factors())
}

fn sample_method_uses_factors(method: Option<SampleNormalizationMethod>) -> bool {
    matches!(
        method,
        Some(SampleNormalizationMethod::GlobalMedian | SampleNormalizationMethod::ConditionMedian)
    )
}

#[derive(Debug)]
pub(crate) struct NormalizationFactorCollector<'a> {
    run_method: Option<RunNormalizationMethod>,
    sample_method: Option<SampleNormalizationMethod>,
    sdrf: Option<SdrfLookupCache<'a>>,
    normalization_proteins: Option<HashSet<String>>,
    /// Custom contaminant patterns for the median pre-pass (Python
    /// `SQLFilterBuilder.contaminant_patterns`). Matched against the **raw**
    /// `pg_accessions` via [`matches_sql_contaminant`], which falls back to the
    /// default [`is_contaminant`] when this equals the default list.
    contaminant_patterns: &'a [String],
    sample_values: HashMap<String, Vec<f64>>,
    condition_sample_values: HashMap<String, HashMap<String, Vec<f64>>>,
    run_values: HashMap<RunCellKey, Vec<f64>>,
}

impl<'a> NormalizationFactorCollector<'a> {
    pub(crate) fn new(
        run_method: Option<RunNormalizationMethod>,
        sample_method: Option<SampleNormalizationMethod>,
        sdrf: Option<&'a SdrfTable>,
        normalization_proteins: Option<HashSet<String>>,
        contaminant_patterns: &'a [String],
    ) -> Self {
        Self {
            run_method,
            sample_method,
            sdrf: sdrf.map(SdrfLookupCache::new),
            normalization_proteins,
            contaminant_patterns,
            sample_values: HashMap::new(),
            condition_sample_values: HashMap::new(),
            run_values: HashMap::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        feature: QpxFeatureRecord,
        filtering: FilterConfig,
        keep_shared_peptides: bool,
    ) -> Result<()> {
        if !passes_feature_filter(&feature, filtering, keep_shared_peptides) {
            return Ok(());
        }
        // Contaminant removal in the median pre-pass mirrors Python's
        // `SQLFilterBuilder`: match the **raw** `pg_accessions` (not the parsed
        // group), as a case-sensitive literal substring. Structured `is_decoy`
        // flags are handled independently by `passes_feature_filter`.
        if filtering.remove_contaminants
            && matches_sql_contaminant(&feature.protein_accessions, self.contaminant_patterns)
        {
            return Ok(());
        }
        let Some(protein_group) = protein_group_name(&feature.protein_accessions) else {
            return Ok(());
        };

        let sdrf_record = sdrf_record(&feature, self.sdrf.as_mut())?;
        let sample = sample_name(&feature, sdrf_record);
        if self
            .normalization_proteins
            .as_ref()
            .is_none_or(|proteins| proteins.contains(&protein_group))
        {
            match self.sample_method {
                Some(SampleNormalizationMethod::GlobalMedian) => {
                    self.sample_values
                        .entry(sample.clone())
                        .or_default()
                        .push(feature.intensity);
                }
                // MedianCenter/MeanCenter are dataset-level (centered on the
                // summed-canonical-peptide matrix after ingest), so they collect
                // no ingest-time factor here, matching the other cell-path
                // methods below.
                Some(
                    SampleNormalizationMethod::Quantile
                    | SampleNormalizationMethod::Rlr
                    | SampleNormalizationMethod::Loess
                    | SampleNormalizationMethod::Hierarchical
                    | SampleNormalizationMethod::MedianCenter
                    | SampleNormalizationMethod::MeanCenter
                    | SampleNormalizationMethod::Tmm,
                ) => {}
                Some(SampleNormalizationMethod::ConditionMedian) => {
                    let condition = sample_condition(&sample, sdrf_record);
                    self.condition_sample_values
                        .entry(condition)
                        .or_default()
                        .entry(sample.clone())
                        .or_default()
                        .push(feature.intensity);
                }
                None => {}
            }
        }

        if self.run_method.is_some() {
            self.run_values
                .entry(RunCellKey {
                    sample,
                    run: feature.run_file_name,
                })
                .or_default()
                .push(feature.intensity);
        }
        Ok(())
    }

    fn into_factors(self) -> IntensityFactors {
        let mut factors = IntensityFactors::default();
        if let Some(method) = self.run_method {
            factors.run = run_normalization_transforms(method, self.run_values);
        }
        match self.sample_method {
            Some(SampleNormalizationMethod::GlobalMedian) => {
                factors.sample = global_median_sample_factors(self.sample_values);
            }
            Some(SampleNormalizationMethod::ConditionMedian) => {
                factors.sample = condition_median_sample_factors(self.condition_sample_values);
            }
            // MedianCenter/MeanCenter are applied on the cell path post-ingest,
            // so they produce no per-sample factor here.
            Some(
                SampleNormalizationMethod::Quantile
                | SampleNormalizationMethod::Rlr
                | SampleNormalizationMethod::Loess
                | SampleNormalizationMethod::Hierarchical
                | SampleNormalizationMethod::MedianCenter
                | SampleNormalizationMethod::MeanCenter
                | SampleNormalizationMethod::Tmm,
            ) => {}
            None => {}
        }
        factors
    }

    pub(crate) fn has_sample_factor_values(&self) -> bool {
        !self.sample_values.is_empty() || !self.condition_sample_values.is_empty()
    }
}

/// Read exact accessions from `--remove_ids`, one per non-empty line.
pub(crate) fn load_remove_ids(path: &Path) -> Result<HashSet<String>> {
    let contents = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(contents
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(parse_protein_accession)
        .filter(|accession| !accession.is_empty())
        .map(str::to_owned)
        .collect())
}

pub(crate) fn has_removed_accession(accessions: &[String], removed: &HashSet<String>) -> bool {
    !removed.is_empty()
        && accessions
            .iter()
            .any(|accession| removed.contains(parse_protein_accession(accession)))
}

pub(crate) fn load_normalization_proteins(path: &Path) -> Result<HashSet<String>> {
    let contents = read_to_string(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let proteins = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(parse_protein_accession)
        .filter(|protein| !protein.is_empty())
        .map(str::to_owned)
        .collect::<HashSet<_>>();
    if proteins.is_empty() {
        return Err(invalid_input(format!(
            "normalization proteins file `{}` does not contain protein IDs",
            path.display()
        )));
    }
    Ok(proteins)
}

pub(crate) fn sample_condition(sample: &str, sdrf_record: Option<&SdrfRecord>) -> String {
    sdrf_record
        .and_then(|record| record.condition.clone())
        .unwrap_or_else(|| sample.to_owned())
}

pub(crate) fn unsupported(stage: &'static str) -> Result<()> {
    Err(MokumeError::NotImplemented { stage })
}

pub(crate) fn passes_feature_filter(
    feature: &QpxFeatureRecord,
    filtering: FilterConfig,
    keep_shared_peptides: bool,
) -> bool {
    feature.sequence.len() >= filtering.min_aa
        && !feature.is_decoy.unwrap_or(false)
        // qpx < 1.1.3 leaves `unique` null in OpenMS-derived features; it then
        // means a one-protein group, which is how qpx >= 1.1.3 sets it.
        && (keep_shared_peptides || feature.unique.unwrap_or(feature.protein_accessions.len() == 1))
        && feature.intensity.is_finite()
        && feature.intensity > 0.0
        && !feature.protein_accessions.is_empty()
}
