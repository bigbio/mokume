use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    matrix: Vec<Vec<Option<f64>>>,
    design: Vec<Vec<f64>>,
    block: Vec<usize>,
    consensus: f64,
    atanh_correlations: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn cases() -> Vec<Case> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../tests/data/limma_block_official.json"))
            .unwrap_or_else(|e| panic!("invalid official JSON fixture: {e}"));
    fixture.cases
}

// REML stops at glmgam.fit's tolerance on the same iterates as R, so the
// per-protein estimates agree well beyond it.
const TOL: f64 = 1e-8;

#[test]
fn correlations_match_duplicate_correlation_limma_3_66_0() {
    for case in cases() {
        let rows = case
            .matrix
            .iter()
            .map(|row| row.iter().map(|value| value.unwrap_or(f64::NAN)).collect())
            .collect::<Vec<Vec<f64>>>();
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let estimate = duplicate_correlation(&refs, &case.design, &case.block)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(estimate.warning.is_none(), "{}", case.name);
        for (i, (actual, expected)) in estimate
            .atanh_correlations
            .iter()
            .zip(&case.atanh_correlations)
            .enumerate()
        {
            match expected {
                Some(expected) => assert!(
                    (actual - expected).abs() <= TOL * (1.0 + expected.abs()),
                    "{} protein {i}: {actual} != {expected}",
                    case.name
                ),
                None => assert!(actual.is_nan(), "{} protein {i}: {actual} != NA", case.name),
            }
        }
        assert!(
            (estimate.consensus - case.consensus).abs() <= TOL,
            "{}: {} != {}",
            case.name,
            estimate.consensus,
            case.consensus
        );
    }
}

#[test]
fn singleton_or_designed_blocks_give_zero_correlation() -> Result<()> {
    let rows = [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]];
    let refs = rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>();
    let design = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
    ];
    // Every sample its own block.
    let singletons = duplicate_correlation(&refs, &design, &[0, 1, 2, 3])?;
    // Blocks equal to the conditions.
    let designed = duplicate_correlation(&refs, &design, &[0, 0, 1, 1])?;
    for estimate in [singletons, designed] {
        assert_eq!(estimate.consensus, 0.0);
        assert!(estimate.warning.is_some());
    }
    Ok(())
}
