use super::minimize;
use serde::Deserialize;
use std::cell::RefCell;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Curve {
    CoupledQuadratic,
    ActiveLowerBound,
    Rosenbrock,
    DifferentScales,
    IndefiniteQuartic,
    CoupledLowerBound,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: Curve,
    start: Vec<f64>,
    lower: Vec<Option<f64>>,
    expected: Expected,
    evaluations: Vec<Point>,
}
#[derive(Deserialize)]
struct Expected {
    par: Vec<f64>,
    iterations: usize,
    evaluations: Vec<usize>,
    status: u8,
}
#[derive(Deserialize)]
struct Point {
    x: Vec<f64>,
    f: f64,
}

fn objective(curve: &Curve, x: &[f64]) -> f64 {
    match curve {
        Curve::CoupledQuadratic => {
            (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2) + 0.4 * (x[0] - 1.0) * (x[1] + 2.0)
        }
        Curve::ActiveLowerBound => (x[0] + 2.0).powi(2) + 3.0 * (x[1] - 1.0).powi(2),
        Curve::Rosenbrock => 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
        Curve::DifferentScales => 0.001 * (x[0] - 30.0).powi(2) + 100.0 * (x[1] - 0.2).powi(2),
        Curve::IndefiniteQuartic => (x[0] * x[0] - 1.0).powi(2) + (x[1] - 0.4).powi(2),
        Curve::CoupledLowerBound => (x[0] - x[1]).powi(2) + 10.0 * (x[1] + 1.0).powi(2),
    }
}
fn gradient(curve: &Curve, x: &[f64]) -> Vec<f64> {
    match curve {
        Curve::CoupledQuadratic => vec![
            2.0 * (x[0] - 1.0) + 0.4 * (x[1] + 2.0),
            4.0 * (x[1] + 2.0) + 0.4 * (x[0] - 1.0),
        ],
        Curve::ActiveLowerBound => vec![2.0 * (x[0] + 2.0), 6.0 * (x[1] - 1.0)],
        Curve::Rosenbrock => vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ],
        Curve::DifferentScales => vec![0.002 * (x[0] - 30.0), 200.0 * (x[1] - 0.2)],
        Curve::IndefiniteQuartic => vec![4.0 * x[0] * (x[0] * x[0] - 1.0), 2.0 * (x[1] - 0.4)],
        Curve::CoupledLowerBound => vec![
            2.0 * (x[0] - x[1]),
            2.0 * (x[1] - x[0]) + 20.0 * (x[1] + 1.0),
        ],
    }
}
fn hessian(curve: &Curve, x: &[f64]) -> Vec<Vec<f64>> {
    match curve {
        Curve::CoupledQuadratic => vec![vec![2.0, 0.4], vec![0.4, 4.0]],
        Curve::ActiveLowerBound => vec![vec![2.0, 0.0], vec![0.0, 6.0]],
        Curve::Rosenbrock => vec![
            vec![1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
            vec![-400.0 * x[0], 200.0],
        ],
        Curve::DifferentScales => vec![vec![0.002, 0.0], vec![0.0, 200.0]],
        Curve::IndefiniteQuartic => vec![vec![12.0 * x[0] * x[0] - 4.0, 0.0], vec![0.0, 2.0]],
        Curve::CoupledLowerBound => vec![vec![2.0, -2.0], vec![-2.0, 22.0]],
    }
}
fn close(actual: f64, expected: f64, curve: &Curve) {
    assert!(
        (actual - expected).abs() <= 1e-9 + 1e-10 * expected.abs(),
        "{curve:?}: {actual} != {expected}"
    );
}
fn check_case(case: &Case) {
    let lower = case
        .lower
        .iter()
        .map(|v| v.unwrap_or(f64::NEG_INFINITY))
        .collect::<Vec<_>>();
    let trace = RefCell::new(Vec::new());
    let result = minimize(
        &|x| {
            let f = objective(&case.name, x);
            trace.borrow_mut().push(Point { x: x.to_vec(), f });
            f
        },
        &|x| gradient(&case.name, x),
        &|x| hessian(&case.name, x),
        &case.start,
        &lower,
    );
    assert!(result.converged, "{:?}", case.name);
    assert_eq!(
        result.iterations, case.expected.iterations,
        "{:?}",
        case.name
    );
    assert_eq!(
        result.evaluations, case.expected.evaluations[0],
        "{:?}",
        case.name
    );
    assert_eq!(result.status, case.expected.status, "{:?}", case.name);
    for (&actual, &expected) in result.x.iter().zip(&case.expected.par) {
        close(actual, expected, &case.name);
    }
    let trace = trace.into_inner();
    assert_eq!(trace.len(), result.evaluations);
    // R's wrapper evaluates the returned solution once more after PORT exits.
    assert_eq!(case.evaluations.len(), trace.len() + 1);
    for (actual, expected) in trace.iter().zip(&case.evaluations) {
        close(actual.f, expected.f, &case.name);
        for (&a, &b) in actual.x.iter().zip(&expected.x) {
            close(a, b, &case.name);
        }
    }
}

#[test]
fn default_nlminb_trajectories_include_bounds_and_indefinite_hessians(
) -> Result<(), serde_json::Error> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../../tests/data/port_official.json"))?;
    for case in &fixture.cases {
        check_case(case);
    }
    Ok(())
}
