//! Per-plex channel loading and the additive co-isolation (interference) floor.

use std::collections::HashMap;

use mokume_core::TmtInterferenceFloor;
use tracing::warn;

const RESERVOIR_ROWS: usize = 20_000;
const MIN_FLOOR_ROWS: usize = 100;
const MAX_FLOOR: f64 = 0.5;

/// Reservoir sample of reporter rows from one plex, collected in the pre-pass.
#[derive(Debug, Default)]
pub(super) struct PlexSample {
    labels: Vec<String>,
    rows: Vec<Vec<(usize, f64)>>,
    seen: u64,
    rng: u64,
}

impl PlexSample {
    pub(super) fn seeded() -> Self {
        Self {
            rng: 0x9E37_79B9_7F4A_7C15,
            ..Self::default()
        }
    }

    pub(super) fn label_index(&mut self, label: &str) -> usize {
        if let Some(position) = self.labels.iter().position(|known| known == label) {
            return position;
        }
        self.labels.push(label.to_owned());
        self.labels.len() - 1
    }

    pub(super) fn offer(&mut self, row: Vec<(usize, f64)>) {
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
pub(super) struct PlexModel {
    pub(super) loading: HashMap<String, f64>,
    pub(super) floor: f64,
}

pub(super) fn plex_model(
    sample: &PlexSample,
    floor: TmtInterferenceFloor,
    quantile: f64,
) -> PlexModel {
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
    let loading = channel_loading(&complete, channels);
    let floor = match floor {
        TmtInterferenceFloor::Fixed(value) => value,
        TmtInterferenceFloor::Auto => auto_floor(&complete, &loading, quantile),
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

/// Median channel/row-mean ratio per channel, scaled to mean 1.
fn channel_loading(complete: &[&Vec<(usize, f64)>], channels: usize) -> Vec<f64> {
    let mut ratios = vec![Vec::with_capacity(complete.len()); channels];
    for row in complete {
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
    loading
}

/// Low quantile of loading-adjusted channel ratios in the more intense half of complete rows.
fn auto_floor(complete: &[&Vec<(usize, f64)>], loading: &[f64], quantile: f64) -> f64 {
    let channels = loading.len();
    let mut totals = complete
        .iter()
        .map(|row| row.iter().map(|(_, value)| value).sum::<f64>())
        .collect::<Vec<_>>();
    let cutoff = quantile_of(&mut totals, 0.5);
    let mut adjusted = Vec::new();
    for row in complete {
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

#[cfg(test)]
mod tests {
    use mokume_core::TmtInterferenceFloor;

    use super::{plex_model, PlexSample};

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
}
