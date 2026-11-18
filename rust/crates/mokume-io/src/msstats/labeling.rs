use std::collections::{HashMap, HashSet};

use mokume_core::Result;

use super::invalid_input;
use crate::{SdrfTable, normalize_label_key};

const TMT6: &[&str] = &["TMT126", "TMT127", "TMT128", "TMT129", "TMT130", "TMT131"];
pub(super) const TMT10: &[&str] = &[
    "TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N",
    "TMT130C", "TMT131",
];
pub(super) const TMT11: &[&str] = &[
    "TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N",
    "TMT130C", "TMT131N", "TMT131C",
];
const TMT16: &[&str] = &[
    "TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N",
    "TMT130C", "TMT131N", "TMT131C", "TMT132N", "TMT132C", "TMT133N", "TMT133C", "TMT134N",
];
const ITRAQ4: &[&str] = &["ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117"];
const ITRAQ8: &[&str] = &[
    "ITRAQ113", "ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117", "ITRAQ118", "ITRAQ119", "ITRAQ121",
];

#[derive(Debug)]
pub(super) enum Labeling {
    LabelFree,
    Isobaric {
        channels: &'static [&'static str],
        declared: HashMap<String, String>,
    },
}

impl Labeling {
    pub(super) fn from_sdrf(sdrf: &SdrfTable, observed_channels: &HashSet<String>) -> Result<Self> {
        let labels = sdrf
            .records()
            .iter()
            .filter_map(|record| record.label.as_deref())
            .map(normalize_label_key)
            .filter(|label| !label.is_empty())
            .collect::<HashSet<_>>();

        if labels.len() == 1 && labels.contains("label free sample") {
            return Ok(Self::LabelFree);
        }

        let channels = if !labels.is_empty() && labels.iter().all(|label| label.starts_with("tmt"))
        {
            infer_isobaric_channels(&labels, observed_channels, &[TMT6, TMT10, TMT11, TMT16])?
        } else if !labels.is_empty() && labels.iter().all(|label| label.starts_with("itraq")) {
            infer_isobaric_channels(&labels, observed_channels, &[ITRAQ4, ITRAQ8])?
        } else {
            return Err(invalid_input(format!(
                "cannot infer LFQ, TMT, or iTRAQ labeling from SDRF labels: {labels:?}"
            )));
        };

        let canonical = channels
            .iter()
            .map(|label| (normalize_label_key(label), (*label).to_owned()))
            .collect::<HashMap<_, _>>();
        let declared = labels
            .into_iter()
            .filter_map(|label| canonical.get(&label).cloned().map(|value| (label, value)))
            .collect();
        Ok(Self::Isobaric { channels, declared })
    }

    pub(super) fn resolve_channel(&self, value: Option<&str>) -> Result<Option<String>> {
        let Self::Isobaric { channels, declared } = self else {
            return Ok(None);
        };
        let text = value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                invalid_input("MSstats input is missing a channel value for an isobaric row")
            })?;
        let canonical = channel_position(text)
            .and_then(|position| channels.get(position - 1))
            .map(|label| (*label).to_owned())
            .or_else(|| declared.get(&normalize_label_key(text)).cloned())
            .ok_or_else(|| {
                invalid_input(format!(
                    "MSstats channel `{text}` is not declared by the SDRF plex"
                ))
            })?;
        Ok(Some(canonical))
    }
}

fn infer_isobaric_channels(
    labels: &HashSet<String>,
    observed_channels: &HashSet<String>,
    schemes: &[&'static [&'static str]],
) -> Result<&'static [&'static str]> {
    let candidates = compatible_schemes(labels, schemes);
    if let Some(exact) = candidates
        .iter()
        .copied()
        .find(|scheme| scheme.len() == labels.len())
    {
        return Ok(exact);
    }

    let positions = numeric_positions(observed_channels);
    let candidates = candidates
        .into_iter()
        .filter(|scheme| scheme_matches_positions(scheme, labels, &positions))
        .collect::<Vec<_>>();
    let Some(first) = candidates.first().copied() else {
        return Err(invalid_input(
            "cannot infer the isobaric plex from SDRF labels and MSstats channels",
        ));
    };

    let expected = scheme_signature(first, &positions);
    if candidates
        .iter()
        .skip(1)
        .any(|scheme| scheme_signature(scheme, &positions) != expected)
    {
        return Err(invalid_input(
            "the isobaric plex is ambiguous for the supplied SDRF and MSstats channels",
        ));
    }
    candidates
        .into_iter()
        .min_by_key(|scheme| scheme.len())
        .ok_or_else(|| invalid_input("cannot infer the isobaric plex"))
}

fn compatible_schemes(
    labels: &HashSet<String>,
    schemes: &[&'static [&'static str]],
) -> Vec<&'static [&'static str]> {
    schemes
        .iter()
        .copied()
        .filter(|scheme| {
            labels
                .iter()
                .all(|label| scheme.iter().any(|known| known.eq_ignore_ascii_case(label)))
        })
        .collect()
}

fn numeric_positions(observed_channels: &HashSet<String>) -> Vec<usize> {
    let mut positions = observed_channels
        .iter()
        .filter_map(|value| channel_position(value))
        .collect::<Vec<_>>();
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn scheme_matches_positions(
    scheme: &[&str],
    labels: &HashSet<String>,
    positions: &[usize],
) -> bool {
    positions.iter().all(|position| {
        scheme
            .get(position - 1)
            .is_some_and(|label| labels.contains(&normalize_label_key(label)))
    })
}

fn scheme_signature<'a>(scheme: &'a [&str], positions: &[usize]) -> Vec<&'a str> {
    positions
        .iter()
        .map(|position| scheme[*position - 1])
        .collect()
}

fn channel_position(value: &str) -> Option<usize> {
    let number = value.trim().parse::<f64>().ok()?;
    (number.is_finite() && number.fract() == 0.0 && number >= 1.0).then_some(number as usize)
}
