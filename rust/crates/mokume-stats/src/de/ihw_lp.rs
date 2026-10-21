//! The IHW Grenander linear program for an ordinal covariate.
use super::{invalid, Result};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

pub(super) fn learn(
    groups: &[Vec<f64>],
    counts: &[f64],
    alpha: f64,
    lambda: f64,
) -> Result<Vec<f64>> {
    if lambda == 0.0 {
        return Ok(vec![1.0; groups.len()]);
    }
    let total = counts.iter().sum::<f64>();
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let y: Vec<Variable> = counts
        .iter()
        .map(|m| problem.add_var(m / total * counts.len() as f64, (0.0, f64::INFINITY)))
        .collect();
    let t: Vec<Variable> = counts
        .iter()
        .map(|_| problem.add_var(0.0, (0.0, f64::INFINITY)))
        .collect();
    for (g, values) in groups.iter().enumerate() {
        for (slope, intercept) in grenander_lines(values)? {
            problem.add_constraint([(y[g], 1.0), (t[g], -slope)], ComparisonOp::Le, intercept);
        }
    }
    if lambda.is_finite() {
        add_variation(&mut problem, &t, counts, lambda);
    }
    let budget: Vec<(Variable, f64)> = (0..groups.len())
        .flat_map(|g| [(y[g], -alpha * counts[g]), (t[g], counts[g])])
        .collect();
    problem.add_constraint(&budget, ComparisonOp::Le, 0.0);
    let solution = problem
        .solve()
        .map_err(|e| invalid(format!("IHW linear program failed: {e}")))?;
    let thresholds: Vec<f64> = t.iter().map(|&v| solution[v].max(0.0)).collect();
    let weighted_sum: f64 = thresholds.iter().zip(counts).map(|(t, m)| t * m).sum();
    if thresholds.iter().all(|&v| v == 0.0) {
        return Ok(vec![1.0; groups.len()]);
    }
    if weighted_sum <= 0.0 || !weighted_sum.is_finite() {
        return Err(invalid("IHW solver produced an invalid threshold budget"));
    }
    Ok(thresholds
        .iter()
        .map(|t| t * total / weighted_sum)
        .collect())
}

fn add_variation(problem: &mut Problem, t: &[Variable], counts: &[f64], lambda: f64) {
    let mut budget: Vec<(Variable, f64)> = Vec::new();
    for pair in t.windows(2) {
        let z = problem.add_var(0.0, (0.0, f64::INFINITY));
        problem.add_constraint(
            [(pair[0], 1.0), (pair[1], -1.0), (z, -1.0)],
            ComparisonOp::Le,
            0.0,
        );
        problem.add_constraint(
            [(pair[0], -1.0), (pair[1], 1.0), (z, -1.0)],
            ComparisonOp::Le,
            0.0,
        );
        budget.push((z, 1.0));
    }
    let total = counts.iter().sum::<f64>();
    budget.extend(t.iter().zip(counts).map(|(&t, m)| (t, -lambda * m / total)));
    problem.add_constraint(&budget, ComparisonOp::Le, 0.0);
}

fn grenander_lines(sorted: &[f64]) -> Result<Vec<(f64, f64)>> {
    if sorted.is_empty() {
        return Err(invalid("IHW training fold has an empty covariate bin"));
    }
    let mut points: Vec<(f64, f64)> = Vec::new();
    for (i, &value) in sorted.iter().enumerate() {
        let p = if value <= 1e-20 { 0.0 } else { value };
        let cdf = (i + 1) as f64 / sorted.len() as f64;
        if let Some(last) = points.last_mut().filter(|last| last.0 == p) {
            last.1 = cdf;
        } else {
            points.push((p, cdf));
        }
    }
    if points[0].0 > 0.0 {
        points.insert(0, (0.0, 0.0));
    }
    if points.last().is_some_and(|p| p.0 < 1.0) {
        points.push((1.0, 1.0));
    }
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for point in points {
        while hull.len() >= 2
            && slope(hull[hull.len() - 2], hull[hull.len() - 1])
                <= slope(hull[hull.len() - 1], point)
        {
            hull.pop();
        }
        hull.push(point);
    }
    Ok(hull
        .windows(2)
        .map(|pair| {
            let slope = slope(pair[0], pair[1]);
            (slope, pair[0].1 - slope * pair[0].0)
        })
        .collect())
}

fn slope(a: (f64, f64), b: (f64, f64)) -> f64 {
    (b.1 - a.1) / (b.0 - a.0)
}
