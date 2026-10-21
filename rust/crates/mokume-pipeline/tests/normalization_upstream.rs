//! Frozen independent upstream outputs: directlfq 0.3.3
//! NormalizationManagerSamples(log2(X), num_samples_quadratic=50), edgeR 4.8.2
//! calcNormFactors(X, method="TMM") with default trimming/weights followed by
//! X/factor, and preprocessCore 1.72.0 normalize.quantiles(X, copy=TRUE).
//! Missing counts are zero only for TMM fitting. All expected outputs were
//! written with 17 significant digits. Fixtures cover the common input range.

use std::{error::Error, path::Path};

use mokume_pipeline::normalize_matrix;

type MatrixFixture = (Vec<String>, Vec<Vec<f64>>);

fn read_matrix(path: &Path) -> Result<MatrixFixture, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let columns = reader
        .headers()?
        .iter()
        .skip(1)
        .map(str::to_owned)
        .collect();
    let values = reader
        .records()
        .map(|record| {
            let record = record?;
            record
                .iter()
                .skip(1)
                .map(|value| value.parse::<f64>().map_err(Into::into))
                .collect()
        })
        .collect::<Result<Vec<Vec<f64>>, Box<dyn Error>>>()?;
    Ok((columns, values))
}

fn compare_fixture(case: &str, method: &str) -> Result<(), Box<dyn Error>> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/normalization");
    let (samples, input) = read_matrix(&base.join(format!("{case}.input.csv")))?;
    let (expected_samples, expected) = read_matrix(&base.join(format!("{case}.expected.csv")))?;
    assert_eq!(samples, expected_samples);
    let actual = normalize_matrix(&input, &samples, method, Some(1))?;
    assert_eq!(actual.len(), expected.len());
    for (row, (actual_row, expected_row)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual_row.len(), expected_row.len());
        for (column, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
            assert_eq!(
                actual.is_nan(),
                expected.is_nan(),
                "{case} ({row}, {column}) missing mask"
            );
            if expected.is_finite() {
                assert_eq!(
                    actual == 0.0,
                    expected == 0.0,
                    "{case} ({row}, {column}) zero state"
                );
                assert!(
                    (actual - expected).abs() <= 1e-8 + 1e-6 * expected.abs(),
                    "{case} ({row}, {column}): {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn hierarchical_matches_official_alignment_and_sample_order() -> Result<(), Box<dyn Error>> {
    for case in [
        "hierarchical_offsets",
        "hierarchical_missing",
        "hierarchical_linear",
    ] {
        compare_fixture(case, "hierarchical")?;
    }
    Ok(())
}

#[test]
fn tmm_matches_edger_with_small_overlap_sparse_reference_and_ties() -> Result<(), Box<dyn Error>> {
    for case in ["tmm_small", "tmm_ties", "tmm_sparse"] {
        compare_fixture(case, "tmm")?;
    }
    Ok(())
}

#[test]
fn quantile_matches_preprocesscore_with_full_row_grid_and_ties() -> Result<(), Box<dyn Error>> {
    for case in [
        "quantile_missing_grid",
        "quantile_complete_ties",
        "quantile_missing_ties",
    ] {
        compare_fixture(case, "quantile")?;
    }
    Ok(())
}

#[test]
fn tmm_rejects_undefined_library_sizes_and_negative_intensities() {
    let samples = vec!["a".to_owned(), "b".to_owned()];
    for input in [
        vec![vec![1.0, 0.0], vec![2.0, 0.0]],
        vec![vec![1.0, -1.0], vec![2.0, 3.0]],
    ] {
        assert!(normalize_matrix(&input, &samples, "tmm", Some(1)).is_err());
    }
}

#[test]
fn hierarchical_masks_singletons_only_in_the_quadratic_subset() -> Result<(), Box<dyn Error>> {
    // directlfq 0.3.3: one observed intensity is discarded by get_normfacts,
    // but can be shifted onto the reference in the >50-sample linear branch.
    for width in [2, 55] {
        let samples: Vec<_> = (0..width).map(|j| format!("s{j:03}")).collect();
        let mut input: Vec<_> = (1..=6).map(|i| vec![i as f64; width]).collect();
        for row in &mut input {
            row[0] = f64::NAN;
        }
        input[0][0] = 16.0;
        let output = normalize_matrix(&input, &samples, "hierarchical", Some(1))?;
        if width == 2 {
            assert!(output.iter().all(|row| row[0].is_nan()));
        } else {
            assert_eq!(output[0][0], 1.0);
            assert!(output.iter().skip(1).all(|row| row[0].is_nan()));
        }
    }
    Ok(())
}

#[test]
fn tmm_all_zero_matrix_uses_edger_degenerate_unity_factors() -> Result<(), Box<dyn Error>> {
    let samples = vec!["a".to_owned(), "b".to_owned()];
    let input = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    assert_eq!(normalize_matrix(&input, &samples, "tmm", Some(1))?, input);
    Ok(())
}

#[test]
fn quantile_rejects_undefined_missing_ranks_but_accepts_a_complete_single_row(
) -> Result<(), Box<dyn Error>> {
    let samples = vec!["a".to_owned(), "b".to_owned()];
    for missing_column in [
        vec![f64::NAN, f64::NAN, f64::NAN],
        vec![1.0, f64::NAN, f64::NAN],
    ] {
        let input: Vec<_> = missing_column
            .iter()
            .enumerate()
            .map(|(i, &v)| vec![v, (i + 2) as f64])
            .collect();
        let Err(error) = normalize_matrix(&input, &samples, "quantile", Some(1)) else {
            panic!("undefined missing-value rank must be rejected");
        };
        assert!(error
            .to_string()
            .contains("rank interpolation is undefined"));
    }
    assert_eq!(
        normalize_matrix(&[vec![1.0, 2.0]], &samples, "quantile", Some(1))?,
        vec![vec![1.5, 1.5]]
    );
    Ok(())
}

#[test]
fn quantile_treats_signed_zero_as_one_tie_group() -> Result<(), Box<dyn Error>> {
    // Independent preprocessCore 1.72.0 output; R ranks +0 and -0 equally.
    let samples = vec!["a".to_owned(), "b".to_owned()];
    let input = vec![vec![-0.0, 10.0], vec![0.0, 20.0], vec![1.0, 30.0]];
    let expected = vec![vec![7.5, 5.0], vec![7.5, 10.0], vec![15.5, 15.5]];
    assert_eq!(
        normalize_matrix(&input, &samples, "quantile", Some(1))?,
        expected
    );
    Ok(())
}
