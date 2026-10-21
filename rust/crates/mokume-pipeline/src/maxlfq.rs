use std::collections::BTreeMap;

use mokume_core::{PeptideId, Result, SampleId};
use mokume_quant::{solve_max_lfq_with_stabilization, PeptideMeasurement};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::{threading, LfqPeptideObservation, LfqProteinIntensity};

/// Quantify prepared peptide-species observations with Cox et al.'s Eq. 3.
/// Normalization must be performed explicitly upstream. Peptide identifiers
/// are used as supplied; a canonical-only table cannot recover charge/PTM data.
pub fn run_maxlfq_from_peptides_with_threads(
    observations: &[LfqPeptideObservation],
    min_ratio_count: usize,
    stabilize: bool,
    threads: Option<usize>,
) -> Result<Vec<LfqProteinIntensity>> {
    threading::install(threads, || {
        quantify(observations, min_ratio_count, stabilize)
    })
}

fn quantify(
    observations: &[LfqPeptideObservation],
    min_ratio_count: usize,
    stabilize: bool,
) -> Result<Vec<LfqProteinIntensity>> {
    let mut grouped = BTreeMap::<&str, Vec<&LfqPeptideObservation>>::new();
    let names = observations
        .iter()
        .map(|row| row.sample.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sample_index = names
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, SampleId::new(i as u32)))
        .collect::<BTreeMap<_, _>>();
    let samples = (0..names.len())
        .map(|i| SampleId::new(i as u32))
        .collect::<Vec<_>>();
    // Validate the configuration even when no protein observations are supplied.
    solve_max_lfq_with_stabilization(&[], &samples, min_ratio_count, stabilize)?;
    for row in observations {
        grouped.entry(&row.protein).or_default().push(row);
    }
    let output = grouped.into_iter().collect::<Vec<_>>().into_par_iter().map(|(protein, rows)| {
        let mut peptides = BTreeMap::new();
        let measurements = rows.iter().map(|row| {
            let next = PeptideId::new(peptides.len() as u32);
            PeptideMeasurement {
                peptide: *peptides.entry(row.peptide.as_str()).or_insert(next),
                sample: sample_index[row.sample.as_str()],
                intensity: row.intensity,
            }
        }).collect::<Vec<_>>();
        let result = solve_max_lfq_with_stabilization(&measurements, &samples, min_ratio_count, stabilize)?;
        if result.components.len() > 1 {
            warn!(protein, components = result.components.len(),
                "MaxLFQ sample graph is disconnected; between-component ratios are not identifiable");
        }
        Ok((result.intensities.into_iter().filter(|(_, x)| *x > 0.0).map(|(sample, intensity)| LfqProteinIntensity {
            protein: protein.to_owned(), sample: names[sample.get() as usize].to_owned(), intensity,
        }).collect::<Vec<_>>(), result.stabilized_pairs))
    }).collect::<Result<Vec<_>>>()?;
    log_stabilization(min_ratio_count, stabilize, output.iter().map(|row| row.1));
    Ok(output.into_iter().flat_map(|(values, _)| values).collect())
}

pub(crate) fn log_stabilization(
    min_ratio_count: usize,
    stabilize: bool,
    pairs_per_protein: impl Iterator<Item = usize>,
) {
    let (stabilized_protein_groups, stabilized_sample_pairs) = pairs_per_protein
        .fold((0_usize, 0_usize), |(proteins, pairs), count| {
            (proteins + usize::from(count > 0), pairs + count)
        });
    info!(
        min_ratio_count,
        stabilize,
        stabilized_protein_groups,
        stabilized_sample_pairs,
        "MaxLFQ quantification finished; counts describe positive stabilization weights"
    );
}
