//! Reporter channel keys and the isotope-impurity correction.

use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

use mokume_core::{MokumeError, Result};

use super::invalid_input;

/// Corrected values at or below this fraction of the row maximum are set to zero.
const RESIDUE_FRACTION: f64 = 1e-6;

/// Reporter channel key, e.g. `TMT127N` / `127N` -> `127N`.
pub(crate) fn channel_key(label: &str) -> Option<String> {
    let upper = label.trim().to_ascii_uppercase();
    let suffix = upper.chars().last().filter(|letter| "NC".contains(*letter));
    let body = &upper[..upper.len() - suffix.map_or(0, char::len_utf8)];
    let digits = body
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
    pub(super) index: HashMap<String, usize>,
    pub(super) fraction: Vec<Vec<f64>>,
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
        let rows = impurity_rows(text)?;
        if rows.len() < 2 {
            return Err("impurity table needs at least two channels".to_owned());
        }
        let index = rows
            .iter()
            .enumerate()
            .map(|(position, (key, _))| (key.clone(), position))
            .collect::<HashMap<_, _>>();
        if index.len() != rows.len() {
            return Err("impurity table lists a channel twice".to_owned());
        }
        let fraction = leakage_fractions(&rows)?;
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
        // Solver residue is not a measured intensity: report it as an empty channel.
        for value in &mut x {
            if *value <= RESIDUE_FRACTION * scale {
                *value = 0.0;
            }
        }
        x
    }
}

/// Channel lines of an impurity table; lines that do not start with a channel name are headers.
fn impurity_rows(text: &str) -> std::result::Result<Vec<(String, [f64; 4])>, String> {
    let mut rows = Vec::new();
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
        let Ok(values) = <[&str; 4]>::try_from(&fields[1..]) else {
            return Err(format!(
                "channel {key} needs exactly 4 impurity values (-2, -1, +1, +2)"
            ));
        };
        let values = values.map(|value| value.parse::<f64>().unwrap_or(f64::NAN));
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(format!(
                "channel {key} has a non-numeric, negative or non-finite impurity"
            ));
        }
        rows.push((key, values));
    }
    Ok(rows)
}

/// Spread each channel's impurities onto the channels 2 and 1 Da lighter and 1 and 2 Da heavier.
fn leakage_fractions(rows: &[(String, [f64; 4])]) -> std::result::Result<Vec<Vec<f64>>, String> {
    // Keys are three digits plus an optional N/C suffix.
    let uses_suffix = rows.iter().any(|(key, _)| key.len() > 3);
    let by_channel = rows
        .iter()
        .enumerate()
        .map(|(position, (key, _))| (mass_and_kind(key, uses_suffix), position))
        .collect::<HashMap<_, _>>();
    let size = rows.len();
    let mut fraction = vec![vec![0.0; size]; size];
    for (position, (key, impurities)) in rows.iter().enumerate() {
        let (mass, kind) = mass_and_kind(key, uses_suffix);
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
    Ok(fraction)
}

/// Nominal mass and isotope kind; in an N/C table the bare 126 is a C channel and other bare keys are N.
fn mass_and_kind(key: &str, uses_suffix: bool) -> (i32, &str) {
    let mass = key[..3].parse::<i32>().unwrap_or(0);
    let kind = match &key[3..] {
        "" if uses_suffix && mass == 126 => "C",
        "" if uses_suffix => "N",
        kind => kind,
    };
    (mass, kind)
}

#[cfg(test)]
mod tests {
    use super::{channel_key, ImpurityMatrix};

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
    fn empty_and_missing_channels_stay_exactly_zero() -> Result<(), String> {
        let matrix = ImpurityMatrix::parse(
            "126\t0\t0\t5\t0\n127N\t0\t0\t4\t0\n127C\t0\t2\t3\t0\n128N\t0\t1\t0\t0\n",
        )?;
        let truth = [1000.0, 500.0, 0.0, 200.0];
        let mut observed = [0.0; 4];
        for (i, value) in truth.iter().enumerate() {
            for (j, slot) in observed.iter_mut().enumerate() {
                *slot += value * matrix.fraction[i][j];
            }
        }
        assert_eq!(matrix.correct(&observed.map(Some))[2], 0.0);
        let missing = [
            Some(observed[0]),
            Some(observed[1]),
            None,
            Some(observed[3]),
        ];
        assert_eq!(matrix.correct(&missing)[2], 0.0);
        Ok(())
    }

    #[test]
    fn impurity_table_rejects_bad_rows() {
        assert!(ImpurityMatrix::parse("126\t0\t0\t5\n127N\t0\t0\t4\t0\n").is_err());
        assert!(ImpurityMatrix::parse("126\t0\t0\t-5\t0\n127N\t0\t0\t4\t0\n").is_err());
        // A channel line with an unreadable value or a fifth value is an error, not a skipped channel.
        assert!(
            ImpurityMatrix::parse("126\t0\t0\t5\t0\n127N\t0\tn/a\t4\t0\n127C\t0\t2\t3\t0\n")
                .is_err()
        );
        assert!(ImpurityMatrix::parse("126\t0\t0\t5\t0\t0\n127N\t0\t0\t4\t0\n").is_err());
    }
}
