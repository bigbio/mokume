//! Maximal peptide ratio extraction: Cox et al. (2014), Eq. 3 and Fig. 2.
//!
//! Input is a single protein's positive, linear peptide-species intensities.
//! Species must distinguish sequence, modification and charge. Normalization
//! (including the paper's separate delayed-normalization stage) is upstream.

use std::collections::{BTreeMap, HashMap};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, Par};
use mokume_core::{MokumeError, Result, SampleId};

use crate::{median, sample_order, PeptideMeasurement};

/// The minimum number of shared species supporting a sample-pair ratio.
pub const DEFAULT_MAXLFQ_MIN_RATIO_COUNT: usize = 2;

#[derive(Debug, Clone)]
pub struct MaxLfqResult {
    /// Linear intensities in the requested sample order; zero means unquantified.
    pub intensities: Vec<(SampleId, f64)>,
    /// Independently solved sample components. Ratios across components are not
    /// identifiable from shared peptides; their offsets use observed totals.
    pub components: Vec<Vec<SampleId>>,
    /// Valid sample pairs assigned a positive large-ratio stabilization weight.
    pub stabilized_pairs: usize,
}

pub fn max_lfq(measurements: &[PeptideMeasurement]) -> Result<Vec<(SampleId, f64)>> {
    max_lfq_with_samples(measurements, &sample_order(measurements))
}

pub fn max_lfq_with_samples(
    measurements: &[PeptideMeasurement],
    samples: &[SampleId],
) -> Result<Vec<(SampleId, f64)>> {
    Ok(solve_max_lfq(measurements, samples, DEFAULT_MAXLFQ_MIN_RATIO_COUNT)?.intensities)
}

/// Minimize the unweighted squared residuals of all valid pairwise log ratios.
/// Each pair's ratio is the median of its shared species' log ratios; when the
/// measurements carry unequal weights it is the weighted median, each species'
/// ratio weighted by the product of its two measurement weights.
/// Duplicate species/sample observations are summed before taking logarithms
/// and keep their largest weight.
/// Missing, nonpositive and nonfinite observations do not support a ratio.
/// A sample without a valid edge stays zero (Cox et al., Fig. 2D).
pub fn solve_max_lfq(
    measurements: &[PeptideMeasurement],
    samples: &[SampleId],
    min_ratio_count: usize,
) -> Result<MaxLfqResult> {
    solve_max_lfq_with_stabilization(measurements, samples, min_ratio_count, false)
}

/// Apply Cox et al. Eq. 5 to low-overlap ratios when `stabilize` is enabled.
/// Shared-species requirements and disconnected-component handling are unchanged.
pub fn solve_max_lfq_with_stabilization(
    measurements: &[PeptideMeasurement],
    samples: &[SampleId],
    min_ratio_count: usize,
    stabilize: bool,
) -> Result<MaxLfqResult> {
    if min_ratio_count == 0 {
        return Err(invalid("minimum ratio count must be positive"));
    }
    let sample_index: HashMap<_, _> = samples.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    if sample_index.len() != samples.len() {
        return Err(invalid("sample identifiers must be unique"));
    }
    let (rows, weights) = intensity_rows(measurements, &sample_index)?;
    let columns = SampleColumns::new(&rows, weights.as_deref(), samples.len());
    let components = connected_components(
        columns
            .linked_pairs(min_ratio_count)
            .map(|(i, j, _)| (i, j)),
        samples.len(),
    );
    let (relatives, stabilized_pairs) =
        solve_components(&rows, &columns, &components, min_ratio_count, stabilize)?;
    let mut log_profile = vec![f64::NEG_INFINITY; samples.len()];
    for (component, relative) in components.iter().zip(relatives) {
        let total = log_total(&rows, component.iter().copied());
        let shift = total - log_sum_exp2(&relative);
        for (&sample, value) in component.iter().zip(relative) {
            log_profile[sample] = value + shift;
        }
    }
    // Eq. 3 fixes ratios, not the absolute scale. Rescale the entire supported
    // profile to the original summed intensity, including isolated observations.
    if !components.is_empty() {
        let shift = log_total(&rows, 0..samples.len()) - log_sum_exp2(&log_profile);
        for value in &mut log_profile {
            *value += shift;
        }
    }
    Ok(MaxLfqResult {
        intensities: linear_intensities(samples, log_profile)?,
        stabilized_pairs,
        components: components
            .into_iter()
            .map(|c| c.into_iter().map(|i| samples[i]).collect())
            .collect(),
    })
}

fn linear_intensities(samples: &[SampleId], log_profile: Vec<f64>) -> Result<Vec<(SampleId, f64)>> {
    samples
        .iter()
        .zip(log_profile)
        .map(|(&sample, log_value)| {
            let value = log_value.exp2();
            if !value.is_finite() || (log_value.is_finite() && value == 0.0) {
                return Err(invalid("protein intensity is outside the finite f64 range"));
            }
            Ok((sample, value))
        })
        .collect()
}

/// Values by species (rows) and sample (columns).
type SpeciesRows = Vec<Vec<f64>>;

/// Species-by-sample intensities and, when any measurement is weighted, the
/// matching weights.
fn intensity_rows(
    measurements: &[PeptideMeasurement],
    sample_index: &HashMap<SampleId, usize>,
) -> Result<(SpeciesRows, Option<SpeciesRows>)> {
    let mut ordered = measurements
        .iter()
        .filter(|m| {
            m.intensity.is_finite() && m.intensity > 0.0 && sample_index.contains_key(&m.sample)
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        (a.peptide.get(), a.sample.get())
            .cmp(&(b.peptide.get(), b.sample.get()))
            .then_with(|| a.intensity.total_cmp(&b.intensity))
    });
    let weighted = ordered.iter().any(|m| m.weight != 1.0);
    let mut rows = BTreeMap::<u32, (Vec<f64>, Vec<f64>)>::new();
    for measurement in ordered {
        let (values, weights) = rows
            .entry(measurement.peptide.get())
            .or_insert_with(|| (vec![0.0; sample_index.len()], vec![0.0; sample_index.len()]));
        let slot = sample_index[&measurement.sample];
        values[slot] += measurement.intensity;
        weights[slot] = weights[slot].max(measurement.weight);
        if !values[slot].is_finite() {
            return Err(invalid(
                "summed species intensity is outside the finite f64 range",
            ));
        }
    }
    let (rows, weights): (Vec<_>, Vec<_>) = rows.into_values().unzip();
    Ok((rows, weighted.then_some(weights)))
}

/// Sample-major logs and per-sample bitsets of observed species: each pair
/// reads two contiguous columns and skips pairs that share too few species.
struct SampleColumns {
    species: usize,
    words: usize,
    logs: Vec<f64>,
    /// Sample-major measurement weights, present only for weighted input.
    weights: Option<Vec<f64>>,
    observed: Vec<u64>,
    /// Samples observing at least one species; no other sample can pair.
    present: Vec<usize>,
}

impl SampleColumns {
    fn new(rows: &[Vec<f64>], weight_rows: Option<&[Vec<f64>]>, n: usize) -> Self {
        let species = rows.len();
        let words = species.div_ceil(64);
        let weights = weight_rows.map(|weight_rows| {
            let mut weights = vec![0.0; n * species];
            for (row_index, row) in weight_rows.iter().enumerate() {
                for (sample, &weight) in row.iter().enumerate() {
                    weights[sample * species + row_index] = weight;
                }
            }
            weights
        });
        let mut logs = vec![0.0; n * species];
        let mut observed = vec![0_u64; n * words];
        for (row_index, row) in rows.iter().enumerate() {
            for (sample, value) in row.iter().enumerate() {
                let log = value.log2();
                logs[sample * species + row_index] = log;
                if log.is_finite() {
                    observed[sample * words + row_index / 64] |= 1 << (row_index % 64);
                }
            }
        }
        let present = (0..n)
            .filter(|&sample| {
                observed[sample * words..(sample + 1) * words]
                    .iter()
                    .any(|&word| word != 0)
            })
            .collect();
        Self {
            species,
            words,
            logs,
            weights,
            observed,
            present,
        }
    }

    /// Pairs `(i, j, shared)` with `i < j`, in ascending order, that share at
    /// least `min_count` species.
    fn linked_pairs(&self, min_count: usize) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.present
            .iter()
            .enumerate()
            .flat_map(move |(position, &i)| {
                self.present[position + 1..].iter().filter_map(move |&j| {
                    let shared = self.shared(i, j);
                    (shared >= min_count).then_some((i, j, shared))
                })
            })
    }

    fn seen(&self, sample: usize) -> &[u64] {
        &self.observed[sample * self.words..(sample + 1) * self.words]
    }

    fn shared(&self, i: usize, j: usize) -> usize {
        self.seen(i)
            .iter()
            .zip(self.seen(j))
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// Call `visit` with each species both samples observe, in species order.
    fn for_each_shared(&self, i: usize, j: usize, mut visit: impl FnMut(usize)) {
        for (word, (a, b)) in self.seen(i).iter().zip(self.seen(j)).enumerate() {
            let mut both = a & b;
            while both != 0 {
                visit(word * 64 + both.trailing_zeros() as usize);
                both &= both - 1;
            }
        }
    }

    fn column<'a>(&self, values: &'a [f64], sample: usize) -> &'a [f64] {
        &values[sample * self.species..(sample + 1) * self.species]
    }

    /// Log ratios of `j` over `i` for the species both observe, in species order.
    fn ratios_into(&self, i: usize, j: usize, ratios: &mut Vec<f64>) {
        let (logs_i, logs_j) = (self.column(&self.logs, i), self.column(&self.logs, j));
        ratios.clear();
        self.for_each_shared(i, j, |row| ratios.push(logs_j[row] - logs_i[row]));
    }

    /// [`Self::ratios_into`] paired with the product of the two measurement weights.
    fn weighted_ratios_into(
        &self,
        weights: &[f64],
        i: usize,
        j: usize,
        pairs: &mut Vec<(f64, f64)>,
    ) {
        let (logs_i, logs_j) = (self.column(&self.logs, i), self.column(&self.logs, j));
        let (weights_i, weights_j) = (self.column(weights, i), self.column(weights, j));
        pairs.clear();
        self.for_each_shared(i, j, |row| {
            pairs.push((logs_j[row] - logs_i[row], weights_i[row] * weights_j[row]));
        });
    }

    /// The pair's median log ratio, weighted when the input is weighted.
    fn pair_ratio(
        &self,
        (i, j): (usize, usize),
        ratios: &mut Vec<f64>,
        pairs: &mut Vec<(f64, f64)>,
    ) -> Option<f64> {
        match &self.weights {
            Some(weights) => {
                self.weighted_ratios_into(weights, i, j, pairs);
                weighted_median(pairs, ratios)
            }
            None => {
                self.ratios_into(i, j, ratios);
                median(ratios)
            }
        }
    }
}

/// Weighted median with the midpoint convention: the value at which the
/// cumulative weight first exceeds half the total, or the mean of two values
/// when half the total falls exactly between them. Equal weights take
/// [`median`] itself, so they reproduce the unweighted result bit for bit.
fn weighted_median(pairs: &mut [(f64, f64)], values: &mut Vec<f64>) -> Option<f64> {
    let first = pairs.first()?.1;
    if pairs.iter().all(|&(_, weight)| weight == first) {
        values.clear();
        values.extend(pairs.iter().map(|&(value, _)| value));
        return median(values);
    }
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let half = pairs.iter().map(|&(_, weight)| weight).sum::<f64>() / 2.0;
    let mut cumulative = 0.0;
    for (index, &(value, weight)) in pairs.iter().enumerate() {
        cumulative += weight;
        if cumulative == half {
            return Some((value + pairs[index + 1].0) / 2.0);
        }
        if cumulative > half {
            return Some(value);
        }
    }
    None
}

/// Reduced graph Laplacian and right-hand side of one component.
type ComponentSystem = (Mat<f64>, Mat<f64>);

/// Add each pair's median log ratio to its component's system as the pair is
/// visited, in the order a stored edge list would be summed, so no list of all
/// pairwise ratios is held.
fn solve_components(
    rows: &[Vec<f64>],
    columns: &SampleColumns,
    components: &[Vec<usize>],
    min_count: usize,
    stabilize: bool,
) -> Result<(Vec<Vec<f64>>, usize)> {
    // Component and position of every sample (rows hold one value per sample).
    let mut slots = vec![(0, 0); rows.first().map_or(0, Vec::len)];
    for (index, component) in components.iter().enumerate() {
        for (position, &sample) in component.iter().enumerate() {
            slots[sample] = (index, position);
        }
    }
    let mut systems = components
        .iter()
        .map(|component| {
            let n = component.len() - 1;
            (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, 1))
        })
        .collect::<Vec<_>>();
    let mut stabilized_pairs = 0;
    let mut ratios = Vec::with_capacity(rows.len());
    let mut pairs = Vec::new();
    for (left, right, shared) in columns.linked_pairs(min_count) {
        let Some(ratio) = columns.pair_ratio((left, right), &mut ratios, &mut pairs) else {
            continue;
        };
        let (ratio, applied) = if stabilize {
            stabilized_ratio(rows, (left, right), shared, ratio)
        } else {
            (ratio, false)
        };
        stabilized_pairs += usize::from(applied);
        let ((component, i), (_, j)) = (slots[left], slots[right]);
        add_ratio(&mut systems[component], i, j, ratio);
    }
    let relatives = systems
        .into_iter()
        .map(solve_system)
        .collect::<Result<Vec<_>>>()?;
    Ok((relatives, stabilized_pairs))
}

fn stabilized_ratio(
    rows: &[Vec<f64>],
    (i, j): (usize, usize),
    shared: usize,
    median_ratio: f64,
) -> (f64, bool) {
    let count_i = rows.iter().filter(|row| row[i] > 0.0).count();
    let count_j = rows.iter().filter(|row| row[j] > 0.0).count();
    let x = count_i.max(count_j) as f64 / shared as f64;
    let weight = ((x - 2.5) / 2.5).clamp(0.0, 1.0);
    if weight == 0.0 {
        return (median_ratio, false);
    }
    let sum_ratio = log_total(rows, std::iter::once(j)) - log_total(rows, std::iter::once(i));
    ((1.0 - weight) * median_ratio + weight * sum_ratio, true)
}

fn connected_components(pairs: impl Iterator<Item = (usize, usize)>, n: usize) -> Vec<Vec<usize>> {
    // Union-find keeps memory linear in the samples, however many edges there are.
    let mut parent = (0..n).collect::<Vec<_>>();
    let mut linked = vec![false; n];
    for (i, j) in pairs {
        linked[i] = true;
        linked[j] = true;
        let (a, b) = (find_root(&mut parent, i), find_root(&mut parent, j));
        parent[a.max(b)] = a.min(b);
    }
    // Components in the order of their first sample, each sorted ascending.
    let mut slot = vec![usize::MAX; n];
    let mut components = Vec::<Vec<usize>>::new();
    for sample in (0..n).filter(|&sample| linked[sample]) {
        let root = find_root(&mut parent, sample);
        if slot[root] == usize::MAX {
            slot[root] = components.len();
            components.push(Vec::new());
        }
        components[slot[root]].push(sample);
    }
    components
}

fn find_root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Fix the first log intensity to zero. The reduced graph Laplacian is
/// positive definite for a connected component; no ridge or fallback is used.
/// Pairs join ascending samples, so (j, i) lies in the lower triangle, the
/// only part the factorization reads.
fn add_ratio((matrix, rhs): &mut ComponentSystem, i: usize, j: usize, ratio: f64) {
    if i > 0 {
        matrix[(i - 1, i - 1)] += 1.0;
        rhs[(i - 1, 0)] -= ratio;
    }
    if j > 0 {
        matrix[(j - 1, j - 1)] += 1.0;
        rhs[(j - 1, 0)] += ratio;
    }
    if i > 0 && j > 0 {
        matrix[(j - 1, i - 1)] -= 1.0;
    }
}

fn solve_system((mut matrix, mut rhs): ComponentSystem) -> Result<Vec<f64>> {
    let n = matrix.nrows();
    // Proteins are solved in parallel, so each factorization runs on one thread.
    let par = Par::Seq;
    let mut buffer = MemBuffer::new(
        llt::factor::cholesky_in_place_scratch::<f64>(n, par, Default::default())
            .or(llt::solve::solve_in_place_scratch::<f64>(n, 1, par)),
    );
    llt::factor::cholesky_in_place(
        matrix.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    )
    .map_err(|_| invalid("connected-component linear solve failed"))?;
    llt::solve::solve_in_place(
        matrix.as_ref(),
        rhs.as_mut(),
        par,
        MemStack::new(&mut buffer),
    );
    Ok(std::iter::once(0.0)
        .chain(rhs.col(0).iter().copied())
        .collect())
}

fn log_sum_exp2(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|x| (x - max).exp2()).sum::<f64>().log2()
}

fn log_total(rows: &[Vec<f64>], columns: impl Iterator<Item = usize>) -> f64 {
    let columns = columns.collect::<Vec<_>>();
    let mut logs = rows
        .iter()
        .flat_map(|row| columns.iter().map(|&i| row[i].log2()))
        .collect::<Vec<_>>();
    // Stable under peptide and sample permutations, including rounding of sums.
    logs.sort_by(f64::total_cmp);
    log_sum_exp2(&logs)
}

fn invalid(message: &str) -> MokumeError {
    MokumeError::InvalidInput {
        message: format!("MaxLFQ: {message}"),
    }
}

#[cfg(test)]
mod tests;
