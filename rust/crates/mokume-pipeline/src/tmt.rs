//! Per-row reporter-ion corrections for isobaric (TMT) QPX features.

mod floor;
mod impurity;

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use mokume_core::{MokumeError, Result, TmtConfig, TmtRowMerge};
use mokume_io::{QpxFeatureRecord, SdrfTable};
use tracing::{info, warn};

use floor::{plex_model, PlexModel, PlexSample};
use impurity::{channel_key, ImpurityMatrix};

const FLOOR_CLAMP: f64 = 0.01;

#[derive(Debug, Default, Clone, Copy)]
struct TmtStats {
    rows: u64,
    impurity_rows: u64,
    floor_rows: u64,
    clamped_channels: u64,
    dropped_rows: u64,
}

/// Applies impurity correction, the interference floor and best-row selection to rows.
#[derive(Debug)]
pub(crate) struct TmtRowProcessor {
    config: TmtConfig,
    impurity: Option<ImpurityMatrix>,
    sample_plex: HashMap<String, String>,
    run_plex: HashMap<String, String>,
    samples: BTreeMap<String, PlexSample>,
    models: HashMap<String, PlexModel>,
    best: HashMap<u64, f64>,
    stats: TmtStats,
    unknown_labels: u64,
}

impl TmtRowProcessor {
    pub(crate) fn new(config: &TmtConfig, sample_plex: HashMap<String, String>) -> Result<Self> {
        let impurity = config
            .impurity_table
            .as_deref()
            .map(ImpurityMatrix::from_path)
            .transpose()?;
        Ok(Self {
            config: config.clone(),
            impurity,
            sample_plex,
            run_plex: HashMap::new(),
            samples: BTreeMap::new(),
            models: HashMap::new(),
            best: HashMap::new(),
            stats: TmtStats::default(),
            unknown_labels: 0,
        })
    }

    /// Whether a full pre-pass over the input is required before [`Self::process`].
    pub(crate) fn needs_prepass(&self) -> bool {
        self.config.row_merge == TmtRowMerge::BestRow || self.config.interference_floor.is_some()
    }

    fn plex(&mut self, row: &[QpxFeatureRecord], sdrf: Option<&SdrfTable>) -> String {
        let Some(first) = row.first() else {
            return String::new();
        };
        if let Some(plex) = self.run_plex.get(&first.run_file_name) {
            return plex.clone();
        }
        let plex = row
            .iter()
            .find_map(|record| {
                let sample = sdrf?
                    .lookup(&record.run_file_name, record.label.as_deref())
                    .ok()?;
                self.sample_plex.get(&sample.sample_accession).cloned()
            })
            .unwrap_or_else(|| first.run_file_name.clone());
        self.run_plex
            .insert(first.run_file_name.clone(), plex.clone());
        plex
    }

    fn apply_impurity(&mut self, row: &mut [QpxFeatureRecord]) {
        let Some(matrix) = &self.impurity else {
            return;
        };
        let mut observed = vec![None; matrix.fraction.len()];
        let mut positions = Vec::with_capacity(row.len());
        for record in row.iter() {
            let position = record
                .label
                .as_deref()
                .and_then(channel_key)
                .and_then(|key| matrix.index.get(&key).copied());
            if let Some(position) = position {
                observed[position] = Some(record.intensity);
            } else {
                self.unknown_labels += 1;
            }
            positions.push(position);
        }
        let corrected = matrix.correct(&observed);
        for (record, position) in row.iter_mut().zip(positions) {
            if let Some(position) = position {
                record.intensity = corrected[position];
            }
        }
        self.stats.impurity_rows += 1;
    }

    fn row_key(row: &[QpxFeatureRecord], plex: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        if let Some(first) = row.first() {
            first.peptidoform.hash(&mut hasher);
            first.charge.hash(&mut hasher);
        }
        plex.hash(&mut hasher);
        hasher.finish()
    }

    fn is_reporter_row(row: &[QpxFeatureRecord]) -> bool {
        row.first().is_some_and(|record| record.label.is_some())
    }

    /// Pre-pass: collect best-row totals and a per-plex row sample.
    pub(crate) fn observe(&mut self, mut row: Vec<QpxFeatureRecord>, sdrf: Option<&SdrfTable>) {
        if !Self::is_reporter_row(&row) {
            return;
        }
        let plex = self.plex(&row, sdrf);
        self.apply_impurity(&mut row);
        if self.config.row_merge == TmtRowMerge::BestRow {
            let total = row_total(&row);
            let best = self.best.entry(Self::row_key(&row, &plex)).or_insert(total);
            if total > *best {
                *best = total;
            }
        }
        if self.config.interference_floor.is_some() {
            let sample = self.samples.entry(plex).or_insert_with(PlexSample::seeded);
            let values = row
                .iter()
                .filter(|record| record.intensity.is_finite() && record.intensity > 0.0)
                .filter_map(|record| {
                    record
                        .label
                        .as_deref()
                        .map(|label| (sample.label_index(label), record.intensity))
                })
                .collect::<Vec<_>>();
            sample.offer(values);
        }
    }

    /// Turn the pre-pass sample into per-plex loading factors and floors.
    pub(crate) fn finish_prepass(&mut self) {
        self.reset_stats();
        let Some(floor) = self.config.interference_floor else {
            return;
        };
        let samples = std::mem::take(&mut self.samples);
        let mut summary = Vec::new();
        for (plex, sample) in samples {
            let model = plex_model(&sample, floor, self.config.floor_quantile);
            summary.push(format!("{plex}={:.4}", model.floor));
            self.models.insert(plex, model);
        }
        info!(
            plexes = summary.len(),
            floors = %summary.join(" "),
            "TMT interference floor per plex"
        );
    }

    fn reset_stats(&mut self) {
        self.stats = TmtStats::default();
        self.unknown_labels = 0;
    }

    fn apply_floor(&mut self, row: &mut [QpxFeatureRecord], plex: &str) {
        let Some(model) = self.models.get(plex) else {
            return;
        };
        let floor = model.floor;
        if floor <= 0.0 {
            return;
        }
        let loading = |record: &QpxFeatureRecord| {
            record
                .label
                .as_deref()
                .and_then(|label| model.loading.get(label).copied())
                .unwrap_or(1.0)
        };
        let adjusted = row
            .iter()
            .filter(|record| record.intensity.is_finite())
            .map(|record| record.intensity / loading(record))
            .collect::<Vec<_>>();
        if adjusted.is_empty() {
            return;
        }
        let mean = adjusted.iter().sum::<f64>() / adjusted.len() as f64;
        if mean <= 0.0 {
            return;
        }
        let mut clamped = 0;
        for record in row.iter_mut() {
            if !record.intensity.is_finite() {
                continue;
            }
            let background = mean * loading(record);
            let corrected = (record.intensity - floor * background) / (1.0 - floor);
            let minimum = FLOOR_CLAMP * background;
            if corrected < minimum {
                clamped += 1;
            }
            record.intensity = corrected.max(minimum);
        }
        self.stats.floor_rows += 1;
        self.stats.clamped_channels += clamped;
    }

    /// Main pass: corrected rows, or nothing when best-row selection drops the row.
    pub(crate) fn process(
        &mut self,
        mut row: Vec<QpxFeatureRecord>,
        sdrf: Option<&SdrfTable>,
    ) -> Vec<QpxFeatureRecord> {
        if !Self::is_reporter_row(&row) {
            return row;
        }
        self.stats.rows += 1;
        let plex = self.plex(&row, sdrf);
        self.apply_impurity(&mut row);
        if self.config.row_merge == TmtRowMerge::BestRow {
            let total = row_total(&row);
            if self
                .best
                .get(&Self::row_key(&row, &plex))
                .is_some_and(|best| total < *best)
            {
                self.stats.dropped_rows += 1;
                return Vec::new();
            }
        }
        if self.config.interference_floor.is_some() {
            self.apply_floor(&mut row, &plex);
        }
        row
    }

    pub(crate) fn log_pass(&mut self, pass: &str) {
        info!(
            pass,
            rows = self.stats.rows,
            impurity_corrected_rows = self.stats.impurity_rows,
            floor_corrected_rows = self.stats.floor_rows,
            clamped_channels = self.stats.clamped_channels,
            best_row_dropped_rows = self.stats.dropped_rows,
            row_merge = self.config.row_merge.label(),
            "TMT reporter corrections"
        );
        if self.unknown_labels > 0 {
            warn!(
                records = self.unknown_labels,
                "TMT impurity table has no entry for some reporter labels; left uncorrected"
            );
        }
        self.reset_stats();
    }
}

fn row_total(row: &[QpxFeatureRecord]) -> f64 {
    row.iter()
        .map(|record| record.intensity)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum()
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mokume_core::{TmtConfig, TmtRowMerge};
    use mokume_io::QpxFeatureRecord;

    use super::TmtRowProcessor;

    fn row(peptidoform: &str, values: &[(&str, f64)]) -> Vec<QpxFeatureRecord> {
        values
            .iter()
            .enumerate()
            .map(|(index, (label, intensity))| QpxFeatureRecord {
                sequence: peptidoform.to_owned(),
                peptidoform: peptidoform.to_owned(),
                charge: 2,
                run_file_name: "run1".to_owned(),
                sample_accession: None,
                protein_accessions: vec!["P1".to_owned()],
                anchor_protein: None,
                unique: Some(true),
                is_decoy: Some(false),
                peptide_qvalue: None,
                pg_global_qvalue: None,
                selected_score: None,
                label: Some((*label).to_owned()),
                intensity: *intensity,
                row_start: index == 0,
            })
            .collect()
    }

    #[test]
    fn best_row_keeps_only_the_highest_reporter_sum() -> mokume_core::Result<()> {
        let config = TmtConfig {
            row_merge: TmtRowMerge::BestRow,
            ..TmtConfig::default()
        };
        let mut processor = TmtRowProcessor::new(&config, HashMap::new())?;
        let low = row("PEPTIDEK", &[("TMT126", 10.0), ("TMT127N", 90.0)]);
        let high = row("PEPTIDEK", &[("TMT126", 100.0), ("TMT127N", 50.0)]);
        processor.observe(low.clone(), None);
        processor.observe(high.clone(), None);
        processor.finish_prepass();
        assert!(processor.process(low, None).is_empty());
        let kept = processor.process(high, None);
        assert_eq!(
            kept.iter()
                .map(|record| record.intensity)
                .collect::<Vec<_>>(),
            vec![100.0, 50.0]
        );
        Ok(())
    }
}
