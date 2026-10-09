//! Per-row reporter-ion corrections for isobaric (TMT) QPX features.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fs::read_to_string;
use std::hash::{Hash, Hasher};
use std::path::Path;

use mokume_core::{MokumeError, Result, TmtConfig, TmtInterferenceFloor, TmtRowMerge};
use mokume_io::{QpxFeatureRecord, SdrfTable};
use tracing::{info, warn};

const RESERVOIR_ROWS: usize = 20_000;
const MIN_FLOOR_ROWS: usize = 100;
const FLOOR_CLAMP: f64 = 0.01;
const MAX_FLOOR: f64 = 0.5;

/// Reporter channel key, e.g. `TMT127N` / `127N` -> `127N`.
pub(crate) fn channel_key(label: &str) -> Option<String> {
    let upper = label.trim().to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut end = bytes.len();
    let suffix = match bytes.last() {
        Some(b'N') | Some(b'C') => {
            end -= 1;
            Some(bytes[end] as char)
        }
        _ => None,
    };
    let digits = upper[..end]
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>();
    if digits.len() != 3 {
        return None;
    }
    let mass = digits.into_iter().rev().collect::<String>();
    Some(match suffix {
        Some(letter) => format!("{mass}{letter}"),
        None => mass,
    })
}

/// Isotope-impurity matrix: `fraction[i][j]` of true channel `i` observed in channel `j`.
#[derive(Debug, Clone)]
pub(crate) struct ImpurityMatrix {
    index: HashMap<String, usize>,
    fraction: Vec<Vec<f64>>,
}

impl ImpurityMatrix {
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        let text = read_to_string(path).map_err(|source| MokumeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
            .map_err(|message| invalid_input(format!("{}: {message}", path.display())))
    }

    pub(crate) fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut rows = Vec::<(String, [f64; 4])>::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields = line
                .split(|c: char| c == '\t' || c == ',' || c == ';' || c.is_whitespace())
                .filter(|field| !field.is_empty())
                .collect::<Vec<_>>();
            let Some(key) = fields.first().and_then(|label| channel_key(label)) else {
                continue;
            };
            let values = fields[1..]
                .iter()
                .take(4)
                .map(|value| value.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>();
            let Ok(values) = values else {
                continue;
            };
            let [m2, m1, p1, p2] = values[..] else {
                return Err(format!(
                    "channel {key} needs 4 impurity values (-2, -1, +1, +2)"
                ));
            };
            if [m2, m1, p1, p2]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(format!(
                    "channel {key} has a negative or non-finite impurity"
                ));
            }
            rows.push((key, [m2, m1, p1, p2]));
        }
        if rows.len() < 2 {
            return Err("impurity table needs at least two channels".to_owned());
        }
        let uses_suffix = rows
            .iter()
            .any(|(key, _)| key.ends_with('N') || key.ends_with('C'));
        let channel = |key: &str| -> (i32, char) {
            let mass = key[..3].parse::<i32>().unwrap_or(0);
            let kind = match key.chars().nth(3) {
                Some(letter) => letter,
                None if uses_suffix && mass == 126 => 'C',
                None if uses_suffix => 'N',
                None => ' ',
            };
            (mass, kind)
        };
        let index = rows
            .iter()
            .enumerate()
            .map(|(position, (key, _))| (key.clone(), position))
            .collect::<HashMap<_, _>>();
        if index.len() != rows.len() {
            return Err("impurity table lists a channel twice".to_owned());
        }
        let by_channel = rows
            .iter()
            .enumerate()
            .map(|(position, (key, _))| (channel(key), position))
            .collect::<HashMap<_, _>>();
        let size = rows.len();
        let mut fraction = vec![vec![0.0; size]; size];
        for (position, (key, impurities)) in rows.iter().enumerate() {
            let (mass, kind) = channel(key);
            let total = impurities.iter().sum::<f64>() / 100.0;
            if total >= 1.0 {
                return Err(format!("channel {key} impurities sum to 100% or more"));
            }
            fraction[position][position] = 1.0 - total;
            for (offset, impurity) in [-2, -1, 1, 2].into_iter().zip(impurities) {
                if let Some(target) = by_channel.get(&(mass + offset, kind)) {
                    fraction[position][*target] += impurity / 100.0;
                }
            }
        }
        Ok(Self { index, fraction })
    }

    /// Non-negative least squares for the true channel intensities of one row.
    pub(crate) fn correct(&self, observed: &[Option<f64>]) -> Vec<f64> {
        let y = observed
            .iter()
            .map(|value| value.unwrap_or(0.0))
            .collect::<Vec<_>>();
        let rhs = self
            .fraction
            .iter()
            .map(|row| row.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>())
            .collect::<Vec<_>>();
        let gram = self
            .fraction
            .iter()
            .map(|left| {
                self.fraction
                    .iter()
                    .map(|right| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scale = y.iter().copied().fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        let mut x = y.clone();
        for _ in 0..500 {
            let mut change = 0.0_f64;
            for (i, (row, target)) in gram.iter().zip(&rhs).enumerate() {
                let diagonal = row[i];
                if diagonal <= 0.0 {
                    continue;
                }
                let residual = target - row.iter().zip(&x).map(|(g, v)| g * v).sum::<f64>();
                let updated = (x[i] + residual / diagonal).max(0.0);
                change = change.max((updated - x[i]).abs());
                x[i] = updated;
            }
            if change <= 1e-12 * scale {
                break;
            }
        }
        x
    }
}

#[derive(Debug, Default)]
struct PlexSample {
    labels: Vec<String>,
    rows: Vec<Vec<(usize, f64)>>,
    seen: u64,
    rng: u64,
}

impl PlexSample {
    fn label_index(&mut self, label: &str) -> usize {
        if let Some(position) = self.labels.iter().position(|known| known == label) {
            return position;
        }
        self.labels.push(label.to_owned());
        self.labels.len() - 1
    }

    fn offer(&mut self, row: Vec<(usize, f64)>) {
        self.seen += 1;
        if self.rows.len() < RESERVOIR_ROWS {
            self.rows.push(row);
            return;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let slot = self.rng % self.seen;
        if let Ok(slot) = usize::try_from(slot) {
            if slot < RESERVOIR_ROWS {
                self.rows[slot] = row;
            }
        }
    }
}

/// Loading factors (mean 1) and additive floor of one plex.
#[derive(Debug, Clone, Default)]
struct PlexModel {
    loading: HashMap<String, f64>,
    floor: f64,
}

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
            let sample = self.samples.entry(plex).or_insert_with(|| PlexSample {
                rng: 0x9E37_79B9_7F4A_7C15,
                ..PlexSample::default()
            });
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
        self.reset_stats();
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

fn plex_model(sample: &PlexSample, floor: TmtInterferenceFloor, quantile: f64) -> PlexModel {
    let channels = sample.labels.len();
    let complete = sample
        .rows
        .iter()
        .filter(|row| row.len() == channels && channels >= 2)
        .collect::<Vec<_>>();
    if complete.len() < MIN_FLOOR_ROWS {
        warn!(
            rows = complete.len(),
            "TMT interference floor: too few complete rows in a plex; floor not applied"
        );
        return PlexModel::default();
    }
    let mut ratios = vec![Vec::with_capacity(complete.len()); channels];
    for row in &complete {
        let mean = row.iter().map(|(_, value)| value).sum::<f64>() / channels as f64;
        for (label, value) in row.iter() {
            ratios[*label].push(value / mean);
        }
    }
    let mut loading = ratios
        .iter_mut()
        .map(|values| quantile_of(values, 0.5))
        .collect::<Vec<_>>();
    let loading_mean = loading.iter().sum::<f64>() / channels as f64;
    for value in &mut loading {
        *value /= loading_mean;
    }
    let floor = match floor {
        TmtInterferenceFloor::Fixed(value) => value,
        TmtInterferenceFloor::Auto => {
            let mut totals = complete
                .iter()
                .map(|row| row.iter().map(|(_, value)| value).sum::<f64>())
                .collect::<Vec<_>>();
            let cutoff = quantile_of(&mut totals, 0.5);
            let mut adjusted = Vec::new();
            for row in &complete {
                let total = row.iter().map(|(_, value)| value).sum::<f64>();
                if total < cutoff {
                    continue;
                }
                let mean = row
                    .iter()
                    .map(|(label, value)| value / loading[*label])
                    .sum::<f64>()
                    / channels as f64;
                for (label, value) in row.iter() {
                    adjusted.push(value / (loading[*label] * mean));
                }
            }
            quantile_of(&mut adjusted, quantile).clamp(0.0, MAX_FLOOR - 1e-6)
        }
    };
    PlexModel {
        loading: sample
            .labels
            .iter()
            .cloned()
            .zip(loading)
            .collect::<HashMap<_, _>>(),
        floor,
    }
}

fn quantile_of(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    values[lower] * (1.0 - weight) + values[upper] * weight
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mokume_core::{TmtConfig, TmtInterferenceFloor, TmtRowMerge};
    use mokume_io::QpxFeatureRecord;

    use super::{channel_key, plex_model, ImpurityMatrix, PlexSample, TmtRowProcessor};

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

    #[test]
    fn interference_floor_is_subtracted_relative_to_the_row_mean() {
        // Two channels at equal loading; one channel carries only a 5% floor in half the rows.
        let mut sample = PlexSample::default();
        let a = sample.label_index("TMT126");
        let b = sample.label_index("TMT127N");
        for index in 0..400 {
            let level = 1000.0 + index as f64;
            if index % 2 == 0 {
                sample.rows.push(vec![(a, level), (b, level)]);
            } else {
                sample.rows.push(vec![(a, 1.9 * level), (b, 0.1 * level)]);
            }
        }
        let fixed = plex_model(&sample, TmtInterferenceFloor::Fixed(0.05), 0.01);
        assert!((fixed.floor - 0.05).abs() < 1e-12);
        let auto = plex_model(&sample, TmtInterferenceFloor::Auto, 0.01);
        assert!(auto.floor > 0.0 && auto.floor < 0.5, "{}", auto.floor);
        let loading_sum = auto.loading.values().sum::<f64>();
        assert!((loading_sum - 2.0).abs() < 1e-9);
    }

    #[test]
    fn channel_keys_strip_prefixes() {
        assert_eq!(channel_key("TMT127N").as_deref(), Some("127N"));
        assert_eq!(channel_key("tmt126").as_deref(), Some("126"));
        assert_eq!(channel_key("131C").as_deref(), Some("131C"));
        assert_eq!(channel_key("sample"), None);
    }

    #[test]
    fn impurity_correction_recovers_true_intensities() -> Result<(), String> {
        let matrix = ImpurityMatrix::parse(
            "channel\t-2\t-1\t+1\t+2\n126\t0\t0\t5\t0\n127N\t0\t0\t4\t0\n127C\t0\t2\t3\t0\n128N\t0\t1\t0\t0\n",
        )?;
        // 126 leaks 5% into 127C (13C), 127C leaks 2% back into 126.
        let truth = [1000.0, 500.0, 0.0, 200.0];
        let mut observed = [0.0; 4];
        for (i, value) in truth.iter().enumerate() {
            for (j, slot) in observed.iter_mut().enumerate() {
                *slot += value * matrix.fraction[i][j];
            }
        }
        assert!((observed[2] - 50.0).abs() < 1e-9);
        let corrected = matrix.correct(&observed.map(Some));
        for (value, expected) in corrected.iter().zip(truth) {
            assert!((value - expected).abs() < 1e-6, "{corrected:?}");
        }
        Ok(())
    }

    #[test]
    fn impurity_table_rejects_bad_rows() {
        assert!(ImpurityMatrix::parse("126\t0\t0\t5\n127N\t0\t0\t4\t0\n").is_err());
        assert!(ImpurityMatrix::parse("126\t0\t0\t-5\t0\n127N\t0\t0\t4\t0\n").is_err());
    }
}
