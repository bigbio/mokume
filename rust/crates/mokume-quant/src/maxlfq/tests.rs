use mokume_core::{PeptideId, Result, SampleId};

use super::{solve_max_lfq, solve_max_lfq_with_stabilization, MaxLfqResult};
use crate::PeptideMeasurement;

fn observations(rows: &[&[f64]]) -> Vec<PeptideMeasurement> {
    rows.iter()
        .enumerate()
        .flat_map(|(peptide, row)| {
            row.iter()
                .enumerate()
                .map(move |(sample, &intensity)| PeptideMeasurement {
                    peptide: PeptideId::new(peptide as u32),
                    sample: SampleId::new(sample as u32),
                    intensity,
                })
        })
        .collect()
}

fn solve(rows: &[&[f64]], min_count: usize) -> Result<MaxLfqResult> {
    let n = rows.first().map_or(0, |row| row.len());
    let samples = (0..n).map(|i| SampleId::new(i as u32)).collect::<Vec<_>>();
    solve_max_lfq(&observations(rows), &samples, min_count)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1e-300) * 1e-11,
        "{actual} != {expected}"
    );
}

#[test]
fn inconsistent_cycle_minimizes_all_pairwise_residuals() -> Result<()> {
    // Edges have log2 differences 1, 1, 3. Eq. 3 gives [0, 4/3, 8/3]
    // up to an additive constant, not a reference-peptide alignment.
    let result = solve(&[&[1.0, 2.0, 0.0], &[0.0, 1.0, 2.0], &[1.0, 0.0, 8.0]], 1)?;
    let values = result
        .intensities
        .iter()
        .map(|(_, x)| *x)
        .collect::<Vec<_>>();
    close((values[1] / values[0]).log2(), 4.0 / 3.0);
    close((values[2] / values[0]).log2(), 8.0 / 3.0);
    close(values.iter().sum(), 15.0);
    Ok(())
}

#[test]
fn minimum_ratio_count_is_per_pair_not_per_protein() -> Result<()> {
    let rows: &[&[f64]] = &[&[10.0, 20.0, 0.0], &[40.0, 80.0, 0.0], &[0.0, 5.0, 10.0]];
    let strict = solve(rows, 2)?;
    close(strict.intensities[1].1 / strict.intensities[0].1, 2.0);
    assert_eq!(strict.intensities[2].1, 0.0);
    close(strict.intensities.iter().map(|(_, x)| x).sum(), 165.0);
    let relaxed = solve(rows, 1)?;
    close(relaxed.intensities[2].1 / relaxed.intensities[0].1, 4.0);
    Ok(())
}

#[test]
fn disconnected_components_have_separate_offsets() -> Result<()> {
    let result = solve(
        &[
            &[1.0, 2.0, 0.0, 0.0],
            &[2.0, 4.0, 0.0, 0.0],
            &[0.0, 0.0, 10.0, 40.0],
            &[0.0, 0.0, 20.0, 80.0],
        ],
        2,
    )?;
    assert_eq!(result.components.len(), 2);
    for ((_, actual), expected) in result.intensities.iter().zip([3.0, 6.0, 30.0, 120.0]) {
        close(*actual, expected);
    }
    Ok(())
}

#[test]
fn missing_and_invalid_values_never_support_edges() -> Result<()> {
    let result = solve(
        &[
            &[1.0, 2.0, f64::NAN],
            &[2.0, 4.0, f64::INFINITY],
            &[-1.0, 0.0, 7.0],
        ],
        2,
    )?;
    close(result.intensities[1].1 / result.intensities[0].1, 2.0);
    assert_eq!(result.intensities[2].1, 0.0);
    close(result.intensities.iter().map(|(_, x)| x).sum(), 16.0);
    Ok(())
}

#[test]
fn duplicate_species_are_summed_but_do_not_inflate_ratio_counts() -> Result<()> {
    let mut data = observations(&[&[10.0, 20.0], &[5.0, 10.0]]);
    data.extend(observations(&[&[10.0, 20.0]]));
    let samples = [SampleId::new(0), SampleId::new(1)];
    let result = solve_max_lfq(&data, &samples, 2)?;
    close(result.intensities[0].1, 25.0);
    close(result.intensities[1].1, 50.0);
    assert!(solve_max_lfq(&data, &samples, 3)?.components.is_empty());
    Ok(())
}

#[test]
fn order_and_identifier_changes_preserve_the_profile() -> Result<()> {
    let mut data = observations(&[
        &[128.0, 0.0, 2.0, 64.0],
        &[64.0, 0.0, 0.0, 4.0],
        &[2.0, 32.0, 32.0, 0.0],
    ]);
    let samples = (0..4).map(SampleId::new).collect::<Vec<_>>();
    let expected = solve_max_lfq(&data, &samples, 1)?;
    data.reverse();
    for row in &mut data {
        row.peptide = PeptideId::new(100 - row.peptide.get());
    }
    let reversed_samples = samples.iter().copied().rev().collect::<Vec<_>>();
    let actual = solve_max_lfq(&data, &reversed_samples, 1)?;
    for (a, b) in actual.intensities.iter().rev().zip(expected.intensities) {
        assert_eq!(a.0, b.0);
        close(a.1, b.1);
    }
    Ok(())
}

#[test]
fn isolated_single_sample_and_empty_inputs_stay_unquantified() -> Result<()> {
    for rows in [vec![&[1.0][..]], vec![&[1.0, 0.0][..], &[0.0, 2.0][..]]] {
        let result = solve(&rows, 1)?;
        assert!(result.components.is_empty());
        assert!(result.intensities.iter().all(|(_, x)| *x == 0.0));
    }
    assert!(solve(&[], 2)?.intensities.is_empty());
    Ok(())
}

#[test]
fn even_ratio_median_is_symmetric_in_log_space() -> Result<()> {
    let result = solve(&[&[1.0, 1.0], &[1.0, 9.0]], 2)?;
    close(result.intensities[1].1 / result.intensities[0].1, 3.0);
    Ok(())
}

#[test]
fn large_dynamic_range_does_not_overflow_intermediate_ratios() -> Result<()> {
    let result = solve(&[&[1e-200, 1e200], &[1e-200, 1e200]], 2)?;
    close(result.intensities[0].1, 2e-200);
    close(result.intensities[1].1, 2e200);
    Ok(())
}

#[test]
fn invalid_configuration_and_duplicate_overflow_are_errors() {
    let samples = [SampleId::new(0), SampleId::new(1)];
    assert!(solve_max_lfq(&[], &samples, 0).is_err());
    assert!(solve_max_lfq(&[], &[samples[0], samples[0]], 2).is_err());
    let mut data = observations(&[&[f64::MAX, 1.0]]);
    data.extend(data.clone());
    assert!(solve_max_lfq(&data, &samples, 1).is_err());
}

#[test]
fn stabilization_follows_cox_overlap_boundaries_and_log_interpolation() -> Result<()> {
    let samples = [SampleId::new(0), SampleId::new(1)];
    for (species, expected_ratio, pairs) in [
        (5, 10.0, 0),
        (6, 10.0_f64.powf(0.8) * 30.0_f64.powf(0.2), 1),
        (10, 50.0, 1),
        (20, 100.0, 1),
    ] {
        let rows = (0..species)
            .map(|i| [10.0, if i < 2 { 1.0 } else { 0.0 }])
            .collect::<Vec<_>>();
        let refs = rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>();
        let data = observations(&refs);
        let result = solve_max_lfq_with_stabilization(&data, &samples, 2, true)?;
        close(
            result.intensities[0].1 / result.intensities[1].1,
            expected_ratio,
        );
        close(
            result.intensities.iter().map(|(_, x)| x).sum(),
            species as f64 * 10.0 + 2.0,
        );
        assert_eq!(result.stabilized_pairs, pairs);
        let disabled = solve_max_lfq_with_stabilization(&data, &samples, 2, false)?;
        assert_eq!(
            disabled.intensities,
            solve_max_lfq(&data, &samples, 2)?.intensities
        );
        assert_eq!(disabled.stabilized_pairs, 0);
        let reversed = solve_max_lfq_with_stabilization(&data, &[samples[1], samples[0]], 2, true)?;
        for (a, b) in result
            .intensities
            .iter()
            .zip(reversed.intensities.iter().rev())
        {
            close(a.1, b.1);
        }
    }
    Ok(())
}

#[test]
fn stabilization_never_creates_unsupported_edges() -> Result<()> {
    let data = observations(&[&[10.0, 1.0, 0.0], &[10.0, 0.0, 0.0], &[0.0, 0.0, 8.0]]);
    let samples = [SampleId::new(0), SampleId::new(1), SampleId::new(2)];
    let strict = solve_max_lfq_with_stabilization(&data, &samples, 2, true)?;
    assert!(strict.components.is_empty());
    assert_eq!(strict.stabilized_pairs, 0);
    assert!(strict.intensities.iter().all(|(_, value)| *value == 0.0));
    let relaxed = solve_max_lfq_with_stabilization(&data, &samples, 1, true)?;
    assert_eq!(relaxed.intensities[2].1, 0.0);
    assert_eq!(relaxed.components, vec![vec![samples[0], samples[1]]]);
    Ok(())
}
