use std::collections::HashMap;

use mokume_core::stats::median_finite;
use mokume_core::{ProteinId, SampleId};

pub fn irs_scaling_factors<F>(
    proteins: impl IntoIterator<Item = ProteinId>,
    plexes: &[String],
    refs_by_plex: &HashMap<String, Vec<SampleId>>,
    stat: &str,
    mut value_at: F,
) -> HashMap<ProteinId, HashMap<String, f64>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let mut factors = HashMap::<ProteinId, HashMap<String, f64>>::new();
    for protein in proteins {
        let mut refs = Vec::<(String, f64)>::new();
        for plex in plexes {
            let Some(samples) = refs_by_plex.get(plex) else {
                continue;
            };
            if let Some(reference) = irs_reference_value(protein, samples, stat, &mut value_at) {
                refs.push((plex.clone(), reference));
            }
        }
        let positive_refs = refs
            .iter()
            .filter_map(|(_, value)| ((*value > 0.0) && value.is_finite()).then_some((*value).ln()))
            .collect::<Vec<_>>();
        if positive_refs.is_empty() {
            continue;
        }
        let global_ref = (positive_refs.iter().sum::<f64>() / positive_refs.len() as f64).exp();
        for (plex, reference) in refs {
            if reference > 0.0 && reference.is_finite() {
                factors
                    .entry(protein)
                    .or_default()
                    .insert(plex, global_ref / reference);
            }
        }
    }
    factors
}

fn irs_reference_value<F>(
    protein: ProteinId,
    samples: &[SampleId],
    stat: &str,
    value_at: &mut F,
) -> Option<f64>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    // Non-positive intensities are missing: additive methods such as DirectLFQ
    // store a missing cell as 0, which must not pull the reference down.
    let mut values = samples
        .iter()
        .filter_map(|sample| value_at(protein, *sample))
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    if stat.eq_ignore_ascii_case("mean") {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    } else {
        median_finite(&mut values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_reference_intensities_count_as_missing() {
        let protein = ProteinId::new(0);
        let samples = [SampleId::new(0), SampleId::new(1), SampleId::new(2)];
        let plexes = ["a".to_owned(), "b".to_owned()];
        let refs_by_plex = HashMap::from([
            (plexes[0].clone(), vec![samples[0], samples[1]]),
            (plexes[1].clone(), vec![samples[2]]),
        ]);
        let values = [0.0, 400.0, 100.0];
        for stat in ["median", "mean"] {
            let factors = irs_scaling_factors([protein], &plexes, &refs_by_plex, stat, |_, s| {
                values.get(s.get() as usize).copied()
            });
            // Plex references are 400 and 100, so the global reference is 200.
            assert!((factors[&protein]["a"] - 0.5).abs() < 1e-12, "{stat}");
            assert!((factors[&protein]["b"] - 2.0).abs() < 1e-12, "{stat}");
        }
    }
}
