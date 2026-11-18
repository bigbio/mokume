use crate::plex::derive_sample_plexes;
use crate::{
    FeatureToProteinsConfig, HashMap, InputConfig, IntoParallelIterator, MemoryPlan, MsstatsReader,
    NamedScoreFilterConfig, ParallelIterator, Path, ProteinGroupOrders, QpxFeatureRecord,
    QpxParquetReader, QpxScoreValue, Result, SdrfRawTable, SdrfTable, TransferredFeatures, info,
    invalid_input, tmt,
};

/// Streams QPX features through `consume`, overlapping the parallelizable
/// decode + flatten work with the serial consumer.
///
/// A background thread reads raw Arrow batches serially (a single Parquet
/// reader cannot be shared) while the calling thread flattens each window with
/// the invocation-scoped Rayon pool. Batches and rows keep reader order, so the
/// floating-point accumulation order is identical to a fully serial pass and
/// the protein matrix stays cell-for-cell identical. The consumer mutates
/// shared state and therefore stays serial on the calling thread.
/// Stream QPX features in file order. With `score`, only rows that pass the
/// named `additional_scores` threshold reach `consume`, and a file without that
/// score is rejected.
pub(crate) fn stream_qpx_features<F>(
    parquet: &Path,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    stream_selected_qpx_features(parquet, memory, score, false, consume)
}

pub(crate) fn stream_selected_qpx_features<F>(
    parquet: &Path,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    with_modifications: bool,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let mut reader = QpxParquetReader::open(
        parquet,
        memory.qpx_batch_size(),
        score.map(|score| score.name.as_str()),
        false,
        with_modifications,
    )?;
    let groups = ProteinGroupOrders::from_path(parquet)?;
    if groups.is_identity() {
        return stream_qpx_reader(&mut reader, memory, score, &mut consume);
    }
    stream_qpx_reader(&mut reader, memory, score, &mut |mut feature| {
        feature.protein_accessions = groups.resolve(&feature.protein_accessions);
        consume(feature)
    })
}

/// [`stream_qpx_features`] for quantification input: `TransferredFeatures::Drop`
/// skips features without an identification in their run, and a transferred
/// weight keeps them marked as transferred.
fn stream_quant_qpx_features<F>(
    parquet: &Path,
    input: &InputConfig,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let drop = input.transferred_features == TransferredFeatures::Drop;
    if !drop && input.transferred_weight.is_none() {
        return stream_qpx_features(parquet, memory, score, consume);
    }
    let mut reader = QpxParquetReader::open(
        parquet,
        memory.qpx_batch_size(),
        score.map(|score| score.name.as_str()),
        true,
        false,
    )?;
    let groups = ProteinGroupOrders::from_path(parquet)?;
    if !drop {
        return stream_qpx_reader(&mut reader, memory, score, &mut |mut feature| {
            if !groups.is_identity() {
                feature.protein_accessions = groups.resolve(&feature.protein_accessions);
            }
            consume(feature)
        });
    }
    stream_direct_qpx_features(&mut reader, &groups, memory, score, consume)
}

fn stream_direct_qpx_features<F>(
    reader: &mut QpxParquetReader,
    groups: &ProteinGroupOrders,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let mut dropped = 0usize;
    let mut kept = 0usize;
    stream_qpx_reader(
        reader,
        memory,
        score,
        &mut |mut feature: QpxFeatureRecord| {
            if !groups.is_identity() {
                feature.protein_accessions = groups.resolve(&feature.protein_accessions);
            }
            if feature.transferred {
                dropped += 1;
                return Ok(());
            }
            kept += 1;
            consume(feature)
        },
    )?;
    info!(dropped, "dropped transferred QPX features");
    if dropped > 0 && kept == 0 {
        return Err(invalid_input(
            "--transferred-features drop removed all QPX features; check identification_type and id_run_file_name, or use keep",
        ));
    }
    Ok(())
}

fn stream_qpx_reader<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let Some(score) = score else {
        return read_qpx_features(reader, memory, None, consume);
    };
    let mut filter = ScoreFilter {
        score,
        seen: 0,
        higher_better: None,
        missing: 0,
        missing_transferred: 0,
    };
    read_qpx_features(reader, memory, Some(score.name.as_str()), &mut |feature| {
        let Some(value) = feature.selected_score else {
            filter.record_missing(&feature);
            return Ok(());
        };
        if filter.passes(value)? {
            consume(feature)?;
        }
        Ok(())
    })?;
    filter.log_missing();
    if filter.seen == 0 {
        return Err(invalid_input(format!(
            "--filter-score found no non-null values for QPX score `{}`",
            score.name
        )));
    }
    Ok(())
}

/// `--filter-score` state for one pass: every row carrying the score must agree
/// on its direction.
struct ScoreFilter<'a> {
    score: &'a NamedScoreFilterConfig,
    seen: usize,
    higher_better: Option<bool>,
    missing: usize,
    missing_transferred: usize,
}

impl ScoreFilter<'_> {
    fn record_missing(&mut self, feature: &QpxFeatureRecord) {
        self.missing += usize::from(feature.row_start);
        self.missing_transferred += usize::from(feature.row_start && feature.transferred);
    }

    fn log_missing(&self) {
        if self.missing > 0 {
            info!(
                score = %self.score.name,
                dropped_rows = self.missing,
                transferred_rows = self.missing_transferred,
                "dropped QPX feature rows because the filter score is null"
            );
        }
    }

    fn passes(&mut self, value: QpxScoreValue) -> Result<bool> {
        self.seen += 1;
        if *self.higher_better.get_or_insert(value.higher_better) != value.higher_better {
            return Err(invalid_input(format!(
                "QPX score `{}` has inconsistent higher_better values",
                self.score.name
            )));
        }
        Ok(if value.higher_better {
            value.value >= self.score.threshold
        } else {
            value.value <= self.score.threshold
        })
    }
}

fn read_qpx_features<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    if memory.channel_capacity() == 0 {
        stream_qpx_features_synchronously(reader, memory, score_name, consume)
    } else {
        stream_qpx_features_buffered(reader, memory, score_name, consume)
    }
}

fn stream_qpx_features_synchronously<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    while let Some(batch) = reader.next_raw_batch() {
        let batch = batch?;
        let features = match score_name {
            Some(name) => mokume_io::flatten_qpx_batch_with_score(&batch, name)?,
            None => mokume_io::flatten_qpx_batch(&batch)?,
        };
        for feature in features {
            consume(feature)?;
        }
        memory.check("QPX streaming")?;
    }
    Ok(())
}

fn stream_qpx_features_buffered<F>(
    reader: &mut QpxParquetReader,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    let window_size = memory.qpx_window();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<mokume_io::RecordBatch>>>(
        memory.channel_capacity(),
    );

    std::thread::scope(|scope| -> Result<()> {
        scope.spawn(move || {
            loop {
                let mut window: Vec<mokume_io::RecordBatch> = Vec::with_capacity(window_size);
                for _ in 0..window_size {
                    match reader.next_raw_batch() {
                        Some(Ok(batch)) => window.push(batch),
                        Some(Err(error)) => {
                            let _ = tx.send(Err(error));
                            return;
                        }
                        None => break,
                    }
                }
                if window.is_empty() {
                    return;
                }
                if tx.send(Ok(window)).is_err() {
                    return;
                }
            }
        });

        let result = consume_qpx_windows(&rx, memory, score_name, consume);
        // A consumer or RSS-budget error must drop the receiver before the
        // scoped reader thread is joined. This wakes a sender blocked by the
        // bounded channel instead of deadlocking on scope exit.
        drop(rx);
        result
    })
}

fn consume_qpx_windows<F>(
    receiver: &std::sync::mpsc::Receiver<Result<Vec<mokume_io::RecordBatch>>>,
    memory: &MemoryPlan,
    score_name: Option<&str>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    for message in receiver {
        let flattened: Result<Vec<Vec<QpxFeatureRecord>>> = message?
            .into_par_iter()
            .map(|batch| match score_name {
                Some(name) => mokume_io::flatten_qpx_batch_with_score(&batch, name),
                None => mokume_io::flatten_qpx_batch(&batch),
            })
            .collect();
        for features in flattened? {
            for feature in features {
                consume(feature)?;
            }
            memory.check("QPX streaming")?;
        }
    }
    Ok(())
}

/// Stream input features grouped by source row (all reporter channels of one row).
fn stream_feature_rows<F>(
    input: &InputConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    mut consume_row: F,
) -> Result<()>
where
    F: FnMut(Vec<QpxFeatureRecord>) -> Result<()>,
{
    let mut row = Vec::new();
    stream_input_features(input, sdrf, memory, None, |feature| {
        if feature.row_start && !row.is_empty() {
            consume_row(std::mem::take(&mut row))?;
        }
        row.push(feature);
        Ok(())
    })?;
    if !row.is_empty() {
        consume_row(row)?;
    }
    Ok(())
}

/// Stream input features through the TMT row corrections, one source row at a time.
pub(crate) fn stream_tmt_features<F>(
    input: &InputConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    processor: &mut tmt::TmtRowProcessor,
    pass: &str,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    stream_feature_rows(input, sdrf, memory, |row| {
        processor
            .process(row, sdrf)
            .into_iter()
            .try_for_each(&mut consume)
    })?;
    processor.log_pass(pass);
    Ok(())
}

/// Stream the normalization pre-pass features, through the TMT row corrections
/// when they are active.
pub(crate) fn stream_prepass_features<F>(
    input: &InputConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    named_score: Option<&NamedScoreFilterConfig>,
    tmt: Option<&mut tmt::TmtRowProcessor>,
    consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    match tmt {
        Some(processor) => stream_tmt_features(
            input,
            sdrf,
            memory,
            processor,
            "normalization pre-pass",
            consume,
        ),
        None => stream_input_features(input, sdrf, memory, named_score, consume),
    }
}

/// Build the TMT row processor (and run its pre-pass) when any TMT option is set.
pub(crate) fn build_tmt_processor(
    config: &FeatureToProteinsConfig,
    sdrf: Option<&SdrfTable>,
    raw_sdrf: Option<&SdrfRawTable>,
    memory: &MemoryPlan,
) -> Result<Option<tmt::TmtRowProcessor>> {
    if !config.tmt.is_active() {
        return Ok(None);
    }
    let sample_plex = match sdrf {
        Some(table) => derive_sample_plexes(table, raw_sdrf, config.irs.plex_column.as_deref())?.0,
        None => HashMap::new(),
    };
    let mut processor =
        tmt::TmtRowProcessor::new(&config.tmt, sample_plex, config.input.parquet.as_deref())?;
    if processor.needs_prepass() {
        stream_feature_rows(&config.input, sdrf, memory, |row| {
            processor.observe(row, sdrf);
            Ok(())
        })?;
        processor.finish_prepass();
        memory.check("TMT pre-pass")?;
    }
    Ok(Some(processor))
}

pub(crate) fn stream_input_features<F>(
    input: &InputConfig,
    sdrf: Option<&SdrfTable>,
    memory: &MemoryPlan,
    score: Option<&NamedScoreFilterConfig>,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(QpxFeatureRecord) -> Result<()>,
{
    match (&input.parquet, &input.msstats) {
        (Some(parquet), None) => stream_quant_qpx_features(parquet, input, memory, score, consume),
        (None, Some(msstats)) => {
            let sdrf = sdrf.ok_or_else(|| invalid_input("MSstats input requires --sdrf option"))?;
            for (index, feature) in MsstatsReader::open(msstats, sdrf)?.enumerate() {
                consume(feature?)?;
                if index % 4096 == 0 {
                    memory.check("MSstats streaming")?;
                }
            }
            Ok(())
        }
        _ => Err(invalid_input(
            "provide exactly one feature input: --parquet or --msstats",
        )),
    }
}
