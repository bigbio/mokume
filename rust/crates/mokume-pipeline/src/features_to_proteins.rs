use crate::accessions::register_id;
use crate::aggregation::{FeatureAggregation, IntensityFactors};
use crate::features_to_peptides::collect_fdr_filter_state;
use crate::from_peptides::PibaqDigest;
use crate::normalization_factors::collect_intensity_factors;
use crate::protein_matrix::{ProteinMatrix, ProteinValues};
use crate::qpx_stream::{build_tmt_processor, stream_input_features, stream_tmt_features};
use crate::state::{CellKey, ContextualIds, FeatureToProteinState, IngestRecord, PeptideIndex};
use crate::validation::{
    dataset_sample_normalization_method, expand_de_contrasts_file, input_sample_table,
    required_parquet, resolve_de_method, validate_de_ensemble_options,
    validate_features_to_proteins, validate_implemented_subset,
};
use crate::{
    BTreeSet, FeatureToProteinsConfig, HashMap, HashSet, MemoryPlan, ProteinId, QuantMethod,
    Result, RuntimeConfig, SampleId, SdrfLookupCache, SdrfRawTable, SdrfTable, StringIdRegistry,
    de, info, invalid_input, spectral_count, sync_channel, threading,
};

pub fn run_features_to_proteins(config: &FeatureToProteinsConfig) -> Result<()> {
    if config.quantification == QuantMethod::Pibaq {
        validate_features_to_proteins(config)?;
        return Err(invalid_input(
            "piBAQ requires the Python wheel's runtime pyOpenMS FASTA digest",
        ));
    }
    let threads = configured_threads(config);
    threading::install(threads, || run_features_to_proteins_inner(config, None))
}

/// Run `features2proteins` with a complete runtime pyOpenMS digest for piBAQ.
pub fn run_features_to_proteins_with_pibaq_digest(
    config: &FeatureToProteinsConfig,
    digest: PibaqDigest,
) -> Result<()> {
    let threads = configured_threads(config);
    threading::install(threads, move || {
        run_features_to_proteins_inner(config, Some(digest))
    })
}

fn configured_threads(config: &FeatureToProteinsConfig) -> Option<usize> {
    config.runtime.threads
}

fn initialize_memory_plan(runtime: &RuntimeConfig) -> Result<MemoryPlan> {
    let memory = MemoryPlan::from_runtime(runtime)?;
    memory.check("startup")?;
    if let Some(limit_bytes) = memory.limit_bytes() {
        info!(
            limit_bytes,
            qpx_batch_size = memory.qpx_batch_size(),
            qpx_window = memory.qpx_window(),
            qpx_read_ahead = memory.channel_capacity(),
            "using soft runtime memory budget"
        );
    }
    Ok(memory)
}

fn run_features_to_proteins_inner(
    config: &FeatureToProteinsConfig,
    pibaq_digest: Option<PibaqDigest>,
) -> Result<()> {
    // Fold `--de-contrast-file` into the contrast list up front (mirroring
    // Python's CLI) so validation and the DE stage see one resolved list; the
    // owned, expanded config then shadows the borrowed one for the rest of the run.
    let expanded;
    let config = if config.differential_expression.contrasts_file.is_some() {
        expanded = expand_de_contrasts_file(config)?;
        &expanded
    } else {
        config
    };

    validate_de_ensemble_options(config)?;
    validate_features_to_proteins(config)?;
    validate_implemented_subset(config)?;
    let memory = initialize_memory_plan(&config.runtime)?;
    let sdrf = input_sample_table(&config.input)?;
    let raw_sdrf = config
        .input
        .sdrf
        .as_ref()
        .map(SdrfRawTable::from_path)
        .transpose()?;
    if config.quantification == QuantMethod::SpectralCount {
        let matrix = spectral_count_matrix(config, sdrf.as_ref(), &memory)?;
        return finish_protein_matrix(config, sdrf.as_ref(), raw_sdrf.as_ref(), &memory, matrix);
    }
    // The protein pipeline has no custom contaminant patterns; an empty slice
    // makes the median pre-pass fall back to the default `is_contaminant` path.
    // The median pre-pass keeps shared peptides only for piBAQ (Python
    // `stages.py:325`: `keep_shared_peptides = method == "pibaq"`).
    // `--filter-protein-fdr`: resolve the passing protein groups (and fail on a
    // missing q-value column) before any other pre-pass or output is produced.
    // Like `features2peptides`, the filter applies at ingest only; the
    // normalization median pre-pass is unaffected.
    let protein_fdr_allowed = match config.protein_fdr_threshold {
        Some(threshold) => collect_fdr_filter_state(
            required_parquet(&config.input)?,
            None,
            Some(threshold),
            None,
        )?,
        None => None,
    };
    let mut tmt_processor = build_tmt_processor(config, sdrf.as_ref(), raw_sdrf.as_ref(), &memory)?;
    let intensity_factors = collect_intensity_factors(
        config,
        sdrf.as_ref(),
        &[],
        config.quantification == QuantMethod::Pibaq,
        &memory,
        None,
        tmt_processor.as_mut(),
    )?;
    memory.check("normalization pre-pass")?;
    let intensity_factors = (!intensity_factors.is_empty()).then_some(&intensity_factors);
    let dataset_normalization = dataset_sample_normalization_method(config)?;
    let mut state =
        FeatureToProteinState::new(config, sdrf.as_ref(), raw_sdrf.as_ref(), pibaq_digest)?;
    state.protein_fdr_allowed = protein_fdr_allowed;
    if let Some(processor) = tmt_processor.as_mut() {
        let mut sdrf_lookup = sdrf.as_ref().map(SdrfLookupCache::new);
        stream_tmt_features(
            &config.input,
            sdrf.as_ref(),
            &memory,
            processor,
            "quantification",
            |feature| {
                state.ingest(
                    &feature,
                    sdrf_lookup.as_mut(),
                    config.filtering,
                    intensity_factors,
                    None,
                )
            },
        )?;
    } else {
        ingest_features(
            &mut state,
            config,
            sdrf.as_ref(),
            &memory,
            intensity_factors,
        )?;
    }
    memory.check("feature aggregation")?;

    info!(
        accepted_features = state.accepted_features,
        accepted_measurements = state.accepted_measurements,
        proteins = state.proteins.len(),
        samples = state.samples.len(),
        quant_method = %config.quantification,
        output = %config.output.protein_matrix.display(),
        "features2proteins aggregation finished"
    );

    let min_unique_peptides = if config.quantification == QuantMethod::Pibaq {
        0
    } else {
        config.filtering.min_unique_peptides
    };
    let matrix = state.into_matrix(config, min_unique_peptides, dataset_normalization)?;
    memory.check("protein matrix materialization")?;
    finish_protein_matrix(config, sdrf.as_ref(), raw_sdrf.as_ref(), &memory, matrix)
}

/// Accepted features per message of a pipelined ingest.
const INGEST_BATCH: usize = 8192;

/// Protein-sample cells per parallel block of [`MaxLfqFeatureAggregation::collapse_ions`].
pub(crate) const COLLAPSE_BLOCK: usize = 65_536;

/// Ingest every feature into `state`. With three or more threads and an
/// aggregation whose `push` reads only ids, ingest runs as a pipeline: this
/// thread resolves filters, names and ids while [`IngestSinks::fill`] fills
/// the aggregation and the peptide index. Each structure still receives every
/// feature in file order, so the result equals the serial ingest.
fn ingest_features(
    state: &mut FeatureToProteinState,
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    intensity_factors: Option<&IntensityFactors>,
) -> Result<()> {
    let mut sdrf_lookup = sdrf.map(SdrfLookupCache::new);
    let placeholder = if rayon::current_num_threads() >= 3 {
        state.aggregation.empty_like()
    } else {
        None
    };
    let Some(placeholder) = placeholder else {
        return stream_input_features(&config.input, sdrf, memory, None, |feature| {
            state.ingest(
                &feature,
                sdrf_lookup.as_mut(),
                config.filtering,
                intensity_factors,
                None,
            )
        });
    };
    let mut sinks = IngestSinks::take(state, placeholder);
    let result = sinks.fill(|send| {
        let mut batch = Vec::with_capacity(INGEST_BATCH);
        stream_input_features(&config.input, sdrf, memory, None, |feature| {
            state.ingest(
                &feature,
                sdrf_lookup.as_mut(),
                config.filtering,
                intensity_factors,
                Some(&mut batch),
            )?;
            if batch.len() < INGEST_BATCH {
                return Ok(());
            }
            send(std::mem::replace(
                &mut batch,
                Vec::with_capacity(INGEST_BATCH),
            ))
        })
        .and_then(|()| send(batch))
    });
    sinks.restore(state);
    result
}

/// The structures that a pipelined ingest fills off the reading thread.
struct IngestSinks {
    aggregation: FeatureAggregation,
    index: PeptideIndex,
    contextual_ids: ContextualIds,
    /// Whether the aggregation keeps contextual canonical peptide rows.
    contextual: bool,
    /// Whether the contextual ids moved here; `ingest` assigns them itself
    /// when it exports peptide rows.
    defer_context: bool,
}

impl IngestSinks {
    fn take(state: &mut FeatureToProteinState, placeholder: FeatureAggregation) -> Self {
        let defer_context = !state.export_rows.has_peptides();
        Self {
            contextual: state.aggregation.preserves_contextual_canonical_rows(),
            aggregation: std::mem::replace(&mut state.aggregation, placeholder),
            index: std::mem::take(&mut state.index),
            contextual_ids: if defer_context {
                std::mem::take(&mut state.contextual)
            } else {
                ContextualIds::default()
            },
            defer_context,
        }
    }

    fn restore(self, state: &mut FeatureToProteinState) {
        state.aggregation = self.aggregation;
        state.index = self.index;
        if self.defer_context {
            state.contextual = self.contextual_ids;
        }
    }

    /// Run `produce`, which sends batches of accepted features, while one
    /// worker assigns their contextual ids and fills the aggregation, then
    /// hands each batch to another worker that fills the peptide index.
    fn fill<P>(&mut self, produce: P) -> Result<()>
    where
        P: FnOnce(&mut dyn FnMut(Vec<IngestRecord>) -> Result<()>) -> Result<()>,
    {
        let (aggregation, index) = (&mut self.aggregation, &mut self.index);
        let (contextual_ids, contextual) = (&mut self.contextual_ids, self.contextual);
        std::thread::scope(|scope| {
            let (to_aggregation, aggregation_batches) = sync_channel::<Vec<IngestRecord>>(4);
            let (to_index, index_batches) = sync_channel::<Vec<IngestRecord>>(4);
            let aggregation_worker = scope.spawn(move || -> Result<()> {
                for mut batch in aggregation_batches {
                    for record in &mut batch {
                        contextual_ids.resolve(record, contextual)?;
                        // These methods read only the ids of a measurement.
                        aggregation.push(record.measurement("", ""))?;
                    }
                    if to_index.send(batch).is_err() {
                        break;
                    }
                }
                Ok(())
            });
            let index_worker = scope.spawn(move || index.add_batches(index_batches, contextual));
            let mut send = move |batch: Vec<IngestRecord>| {
                to_aggregation
                    .send(batch)
                    .map_err(|_| invalid_input("pipelined ingest worker stopped"))
            };
            let produced = produce(&mut send);
            // Closing the channels lets the workers finish before they are joined.
            drop(send);
            let aggregated = aggregation_worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            index_worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            // A worker error is the cause of a failed send, so it is reported first.
            aggregated.and(produced)
        })
    }
}

fn finish_protein_matrix(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    raw_sdrf: Option<&SdrfRawTable>,
    memory: &MemoryPlan,
    mut matrix: ProteinMatrix,
) -> Result<()> {
    let drop_empty_samples = config.quantification == QuantMethod::Ratio;
    apply_protein_postprocessing(config, sdrf, raw_sdrf, drop_empty_samples, &mut matrix)?;
    memory.check("matrix post-processing")?;
    // Python's dataset-level normalization pivots through sparse long form, so
    // absent cells remain missing even for additive quantification methods.
    let missing_fill = missing_protein_value(config)?;
    matrix.write_csv(
        &config.output.protein_matrix,
        config.output.format,
        drop_empty_samples,
        missing_fill,
    )?;
    if let Some(path) = &config.output.export_precision {
        matrix.write_precision_csv(path, config.output.format, drop_empty_samples)?;
    }
    if config.differential_expression.enabled {
        let mut differential_expression = config.differential_expression.clone();
        differential_expression.method = resolve_de_method(config, || {
            let samples = matrix
                .sample_columns(drop_empty_samples)
                .into_iter()
                .map(|(sample, _)| sample)
                .collect::<Vec<_>>();
            !matrix.has_missing_values(&samples)
        });
        de::run_differential_expression(
            &matrix,
            sdrf,
            raw_sdrf,
            &differential_expression,
            drop_empty_samples,
        )?;
    }
    Ok(())
}

fn apply_protein_postprocessing(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    raw_sdrf: Option<&SdrfRawTable>,
    drop_empty_samples: bool,
    matrix: &mut ProteinMatrix,
) -> Result<()> {
    if config.irs.enabled {
        matrix.apply_irs(sdrf, raw_sdrf, &config.irs)?;
    }
    let values_are_log2 = matches!(config.quantification, QuantMethod::Abd | QuantMethod::Ratio);
    if let Some(threshold) = config.sample_correlation_threshold {
        matrix.apply_sample_correlation_filter(
            sdrf,
            threshold,
            drop_empty_samples,
            values_are_log2,
        )?;
    }
    if let Some(threshold) = config.coverage_threshold {
        matrix.apply_coverage_filter(sdrf, threshold, drop_empty_samples)?;
    }
    if let Some(path) = &config.output.export_unimputed {
        let missing_fill = missing_protein_value(config)?;
        matrix.write_csv(path, config.output.format, drop_empty_samples, missing_fill)?;
    }
    if config.imputation.enabled {
        matrix.apply_imputation(&config.imputation, drop_empty_samples)?;
    }
    if config.batch.enabled {
        matrix.apply_batch_correction(
            &config.batch,
            config.input.sdrf.as_deref(),
            drop_empty_samples,
            values_are_log2,
        )?;
    }
    Ok(())
}

fn missing_protein_value(config: &FeatureToProteinsConfig) -> Result<Option<f64>> {
    if dataset_sample_normalization_method(config)?.is_some() {
        return Ok(None);
    }
    // Missing protein x sample cells follow Python's per-method convention: the
    // additive methods write `0`, the average/ratio methods leave the cell empty.
    Ok(match config.quantification {
        QuantMethod::Sum
        | QuantMethod::Intensity
        | QuantMethod::PeptideCount
        | QuantMethod::SpectralCount
        | QuantMethod::DirectLfq => Some(0.0),
        QuantMethod::Median
        | QuantMethod::TopN
        | QuantMethod::Abd
        | QuantMethod::MaxLfq
        | QuantMethod::Ratio
        | QuantMethod::Pibaq => None,
    })
}

fn spectral_count_matrix(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
) -> Result<ProteinMatrix> {
    let sdrf = sdrf.ok_or_else(|| invalid_input("spectral_count requires --sdrf option"))?;
    let counted = spectral_count::count(config, sdrf, memory)?;
    let (proteins, samples) = spectral_registries(&counted.cells)?;

    let mut values = HashMap::new();
    let mut peptide_counts = HashMap::new();
    for cell in counted.cells {
        let protein_id = proteins
            .get(&cell.protein_group)
            .ok_or_else(|| invalid_input("spectral-count protein registry is inconsistent"))?;
        let sample_id = samples
            .get(&cell.sample)
            .ok_or_else(|| invalid_input("spectral-count sample registry is inconsistent"))?;
        let key = CellKey {
            protein: protein_id,
            sample: sample_id,
        };
        values.insert(key, cell.psms as f64);
        peptide_counts.insert(key, cell.sequences.len());
    }
    let allowed_proteins = values.keys().map(|cell| cell.protein).collect();
    info!(
        target_psms = counted.target_psms,
        unique_psms = counted.unique_psms,
        proteins = proteins.len(),
        samples = samples.len(),
        "PSM spectral counting finished"
    );
    Ok(ProteinMatrix {
        proteins,
        samples,
        allowed_proteins,
        excluded_samples: HashSet::new(),
        peptide_counts,
        precision: HashMap::new(),
        values: ProteinValues::Cells(values),
    })
}

fn spectral_registries(
    cells: &[spectral_count::SpectralCountCell],
) -> Result<(StringIdRegistry<ProteinId>, StringIdRegistry<SampleId>)> {
    let protein_names = cells
        .iter()
        .map(|cell| cell.protein_group.as_str())
        .collect::<BTreeSet<_>>();
    let sample_names = cells
        .iter()
        .map(|cell| cell.sample.as_str())
        .collect::<BTreeSet<_>>();
    let mut proteins = StringIdRegistry::new();
    let mut samples = StringIdRegistry::new();
    for protein in protein_names {
        register_id(&mut proteins, protein, "protein")?;
    }
    for sample in sample_names {
        register_id(&mut samples, sample, "sample")?;
    }
    Ok((proteins, samples))
}
