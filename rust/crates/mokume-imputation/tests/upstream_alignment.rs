//! Regression oracles from imputeLCMD 2.1, rrcovNA 0.5-3, pcaMethods 2.2.0
//! and the authors' archived SeqKnn 1.0.1, generated with R 4.5.3.
//! Inputs/outputs use log2 abundances; all observed cells are preserved.
//! QRILC/MinProb use shared uniform/normal draws (seed 42, tune.sigma=.7),
//! preserving the official function bodies; their ordinary R RNG differs.
use std::collections::BTreeMap;

use mokume_core::{ImputationConfig, ProteinId, SampleId};
use mokume_imputation::imputed_values;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Default)]
struct Case {
    method: String,
    tune_sigma: f64,
    values: BTreeMap<(usize, usize), Option<f64>>,
    expected: BTreeMap<(usize, usize), f64>,
}

#[test]
fn official_imputation_oracles() -> TestResult {
    let mut cases = BTreeMap::<String, Case>::new();
    for line in include_str!("fixtures/upstream.tsv").lines().skip(1) {
        let fields = line.split('\t').collect::<Vec<_>>();
        let case = cases.entry(fields[0].to_owned()).or_default();
        case.method = fields[1].to_owned();
        case.tune_sigma = fields[2].parse()?;
        let key = (fields[3].parse()?, fields[4].parse()?);
        let input: f64 = fields[5].parse()?;
        case.values.insert(key, input.is_finite().then_some(input));
        if fields[6] != "NA" {
            case.expected.insert(key, fields[6].parse()?);
        }
    }
    assert_eq!(cases.len(), 9);
    for (name, case) in cases {
        check_case(&name, &case)?;
    }
    Ok(())
}

fn check_case(name: &str, case: &Case) -> TestResult {
    let rows = case.values.keys().map(|key| key.0).max().unwrap_or(0) + 1;
    let cols = case.values.keys().map(|key| key.1).max().unwrap_or(0) + 1;
    let proteins = (0..rows)
        .map(|i| ProteinId::new(i as u32))
        .collect::<Vec<_>>();
    let samples = (0..cols)
        .map(|j| SampleId::new(j as u32))
        .collect::<Vec<_>>();
    let config = ImputationConfig {
        enabled: true,
        method: case.method.clone(),
        tune_sigma: case.tune_sigma,
        n_neighbors: 2,
        ..Default::default()
    };
    let fills = imputed_values(&config, &proteins, &samples, |p, s| {
        case.values
            .get(&(p.get() as usize, s.get() as usize))
            .copied()
            .flatten()
    })?;
    assert_eq!(fills.len(), case.expected.len(), "{name}: full missing set");
    for (p, s, value) in fills {
        let key = (p.get() as usize, s.get() as usize);
        assert_eq!(case.values[&key], None, "{name}: observed cell emitted");
        let expected = case.expected[&key];
        assert!(
            (value - expected).abs() <= 1e-8,
            "{name} {key:?}: {value:.17} vs {expected:.17}"
        );
    }
    Ok(())
}

#[test]
fn stochastic_seed_is_reproducible_and_changes_draws() -> TestResult {
    let proteins = (0..12).map(ProteinId::new).collect::<Vec<_>>();
    let samples = (0..3).map(SampleId::new).collect::<Vec<_>>();
    for method in ["qrilc", "minprob"] {
        let run = |seed| {
            imputed_values(
                &ImputationConfig {
                    enabled: true,
                    method: method.to_owned(),
                    seed,
                    ..Default::default()
                },
                &proteins,
                &samples,
                |p, s| {
                    if p.get() % 4 == 0 {
                        None
                    } else {
                        Some(10.0 + f64::from(p.get()) + f64::from(s.get()))
                    }
                },
            )
        };
        let a = run(42)?;
        let b = run(42)?;
        let c = run(43)?;
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.windows(2).any(|pair| pair[0].2 != pair[1].2));
    }
    Ok(())
}

#[test]
fn seqknn_uses_squared_distance_and_zero_distance_epsilon() -> TestResult {
    let proteins = (0..3).map(ProteinId::new).collect::<Vec<_>>();
    let samples = (0..2).map(SampleId::new).collect::<Vec<_>>();
    let matrix = [
        [Some(1.), Some(10.)],
        [Some(1.), Some(20.)],
        [Some(1.), None],
    ];
    let fills = imputed_values(
        &ImputationConfig {
            method: "seqknn".into(),
            n_neighbors: 2,
            ..Default::default()
        },
        &proteins,
        &samples,
        |p, s| matrix[p.get() as usize][s.get() as usize],
    )?;
    assert_eq!(fills.len(), 1);
    assert!((fills[0].2 - 15.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn unsupported_initial_sets_and_legacy_parameters_are_explicit_errors() {
    let proteins = [ProteinId::new(0), ProteinId::new(1)];
    let samples = [SampleId::new(0), SampleId::new(1)];
    for method in ["seqknn", "impseq"] {
        let result = imputed_values(
            &ImputationConfig {
                method: method.into(),
                ..Default::default()
            },
            &proteins,
            &samples,
            |p, s| if p.get() == s.get() { Some(10.) } else { None },
        );
        assert!(result.is_err(), "{method}");
    }
}
