//! Bound-constrained analytic-Hessian trust-region minimization using the PORT
//! DRMNHB / DG7QTS / DA7SST numerical rules used by R's `nlminb`.
//! This entrypoint uses R's default unit scale, lower bounds, infinite upper
//! bounds, 150 iterations, 200 function evaluations, and fixed tolerances.
//!
//! The numerical procedures are due to David M. Gay and the PORT library
//! (AT&T Bell Laboratories): <https://netlib.org/port/readme>. The control flow
//! was checked against R's `R-4-5-branch` `portsrc.f`, `port.c`, and
//! `R/nlminb.R`; source SHA256 values and R 4.5.3 execution provenance
//! accompany the independent test fixture.
//! This is the analytic-Hessian, fixed-scale subset needed by proDA, not a
//! general replacement for R's configurable `nlminb` interface.
#[path = "port/assess.rs"]
mod assess;
#[path = "port/bounds.rs"]
mod bounds;
#[path = "port/math.rs"]
mod math;
#[path = "port/trust.rs"]
mod trust;
use assess::Assessment;
use math::{dot, norm};
use trust::Trust;

#[derive(Debug)]
pub(super) struct PortResult {
    pub x: Vec<f64>,
    pub converged: bool,
    #[cfg(test)]
    pub iterations: usize,
    #[cfg(test)]
    pub evaluations: usize,
    #[cfg(test)]
    pub status: u8,
}

pub(super) fn minimize<F, G, H>(
    objective: &F,
    gradient: &G,
    hessian: &H,
    start: &[f64],
    lower: &[f64],
) -> PortResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let mut engine = Engine::new(start, lower, objective(start));
    let mut g = gradient(&engine.x);
    let mut h = hessian(&engine.x);
    for iteration in 0..150 {
        engine.iterations = iteration + 1;
        engine.assessment.begin();
        let origin = engine.x.clone();
        let mut cache = Trust::new(engine.alpha, engine.previous_radius);
        if let Some(status) = engine.search(objective, &mut cache, &h, &g, &origin) {
            return engine.finish(status);
        }
        let terminal = (7..=12).contains(&engine.assessment.code);
        if terminal && engine.assessment.f >= engine.assessment.f0 {
            return engine.finish(engine.assessment.code - 4);
        }
        let new_gradient = gradient(&engine.x);
        let new_hessian = hessian(&engine.x);
        if terminal && !engine.released_bound(&origin, &g, &new_gradient) {
            return engine.finish(engine.assessment.code - 4);
        }
        engine.accept(&origin, &g, &h, &new_gradient);
        g = new_gradient;
        h = new_hessian;
    }
    engine.finish(10)
}

struct Engine<'a> {
    x: Vec<f64>,
    lower: &'a [f64],
    saved: Vec<f64>,
    assessment: Assessment,
    radius: f64,
    alpha: f64,
    previous_radius: f64,
    iterations: usize,
    evaluations: usize,
}
impl<'a> Engine<'a> {
    fn new(start: &[f64], lower: &'a [f64], value: f64) -> Self {
        Self {
            x: start.to_vec(),
            lower,
            saved: start.to_vec(),
            assessment: Assessment::new(value),
            radius: 1.0 / 1.1,
            alpha: 0.0,
            previous_radius: 0.0,
            iterations: 0,
            evaluations: 1,
        }
    }
    fn finish(&self, status: u8) -> PortResult {
        PortResult {
            x: self.x.clone(),
            converged: (3..=6).contains(&status),
            #[cfg(test)]
            iterations: self.iterations,
            #[cfg(test)]
            evaluations: self.evaluations,
            #[cfg(test)]
            status,
        }
    }
    fn search<F: Fn(&[f64]) -> f64>(
        &mut self,
        f: &F,
        cache: &mut Trust,
        h: &[Vec<f64>],
        g: &[f64],
        origin: &[f64],
    ) -> Option<u8> {
        loop {
            if self.evaluations >= 200 {
                return Some(9);
            }
            self.propose(cache, h, g, origin);
            let restored = self.restoration_code();
            if self.needs_objective() {
                self.evaluate(f, origin);
            }
            self.assessment.relative_step = relative_distance(&self.x, origin);
            self.assessment.assess(self.evaluations);
            self.restore(origin, restored);
            match self.assessment.code {
                1 | 5 => self.radius = self.assessment.factor * self.assessment.step.norm,
                6 => self.radius = 1.0,
                2..=4 | 7..=12 => return None,
                _ => return Some(64),
            }
        }
    }
    fn propose(&mut self, cache: &mut Trust, h: &[Vec<f64>], g: &[f64], origin: &[f64]) {
        self.assessment.step = bounds::calculate(cache, h, g, origin, self.lower, self.radius);
        self.alpha = self.assessment.step.alpha;
        self.previous_radius = cache.previous_radius();
    }
    fn needs_objective(&self) -> bool {
        self.assessment.code != 6
            && self.assessment.step.norm > 0.0
            && !(self.assessment.code == 5
                && self.assessment.factor > 1.0
                && self.assessment.step.predicted <= 1.2 * self.assessment.difference)
    }
    fn restoration_code(&self) -> u8 {
        if self.assessment.restore != 2 {
            return 3;
        }
        if self.assessment.code == 6 {
            return 2;
        }
        if self.assessment.code == 5 && self.assessment.step.norm > 0.0 && !self.needs_objective() {
            0
        } else {
            3
        }
    }
    fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, f: &F, origin: &[f64]) {
        self.x = origin
            .iter()
            .zip(&self.assessment.step.vector)
            .zip(self.lower)
            .map(|((x, s), lo)| (x + s).max(*lo))
            .collect();
        self.evaluations += 1;
        let value = f(&self.x);
        self.assessment.f = if value.is_nan() { f64::INFINITY } else { value };
        self.assessment.relative_step = relative_distance(&self.x, origin);
    }
    fn restore(&mut self, origin: &[f64], restored: u8) {
        match self.assessment.restore {
            1 => self.x = origin.to_vec(),
            2 => self.saved = self.x.clone(),
            3 => {
                self.x = self.saved.clone();
                self.assessment.step.vector =
                    self.x.iter().zip(origin).map(|(x, o)| x - o).collect();
                self.assessment.relative_step = relative_distance(&self.x, origin);
                self.assessment.restore = restored;
            }
            _ => {}
        }
    }
    fn released_bound(&self, origin: &[f64], old_g: &[f64], new_g: &[f64]) -> bool {
        (0..origin.len()).any(|i| {
            origin[i] <= self.lower[i]
                && old_g[i] > 0.0
                && !(self.x[i] <= self.lower[i] && new_g[i] > 0.0)
        })
    }
    fn accept(&mut self, origin: &[f64], old_g: &[f64], old_h: &[Vec<f64>], new_g: &[f64]) {
        let step: Vec<f64> = self.x.iter().zip(origin).map(|(x, o)| x - o).collect();
        if self.assessment.code == 3 {
            let error: Vec<f64> = old_h
                .iter()
                .zip(old_g)
                .zip(new_g)
                .map(|((row, g), new)| dot(row, &step) + g - new)
                .collect();
            let free_gradient: Vec<_> = (0..new_g.len())
                .filter(|&i| !(self.x[i] <= self.lower[i] && new_g[i] > 0.0))
                .map(|i| new_g[i])
                .collect();
            if norm(&error) <= 0.5 * norm(&free_gradient)
                || dot(new_g, &step) < 0.75 * self.assessment.step.gradient_product
            {
                self.assessment.factor = 2.0;
            }
        }
        let candidate = self.assessment.factor * norm(&step);
        if self.assessment.factor < 1.0 || candidate > self.radius {
            self.radius = candidate;
        }
    }
}
fn relative_distance(x: &[f64], origin: &[f64]) -> f64 {
    let numerator = x
        .iter()
        .zip(origin)
        .map(|(x, o)| (x - o).abs())
        .fold(0.0, f64::max);
    let denominator = x
        .iter()
        .zip(origin)
        .map(|(x, o)| x.abs() + o.abs())
        .fold(0.0, f64::max);
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "port/tests.rs"]
mod tests;
