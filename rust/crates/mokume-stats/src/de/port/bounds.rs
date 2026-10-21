use super::math::{backward, cholesky, dot, forward, norm};
use super::trust::{Step, Trust};

pub(super) fn calculate(
    cache: &mut Trust,
    h: &[Vec<f64>],
    g: &[f64],
    x: &[f64],
    lower: &[f64],
    radius: f64,
) -> Step {
    let free = (0..x.len())
        .filter(|&i| !(x[i] <= lower[i] && g[i] > 0.0))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return Step {
            vector: vec![0.0; x.len()],
            ..Step::default()
        };
    }
    let residual = free.iter().map(|&i| g[i]).collect();
    let mut clipping = Clipping {
        free,
        h,
        current: x.to_vec(),
        lower,
        base: x,
        residual,
        step: vec![0.0; x.len()],
        predicted: 0.0,
        radius,
        step_norm: 0.0,
    };
    let proposal = cache.compute(&clipping.reduced_hessian(), &clipping.residual, radius);
    let newton = (proposal.newton_norm, proposal.newton_predicted);
    let mut proposal = clipping.constrained(cache, proposal);
    proposal.vector = clipping.step;
    proposal.predicted = clipping.predicted;
    proposal.norm = clipping.step_norm;
    proposal.gradient_product = dot(g, &proposal.vector);
    proposal.newton_norm = newton.0;
    proposal.newton_predicted = newton.1;
    proposal
}

struct Clipping<'a> {
    free: Vec<usize>,
    h: &'a [Vec<f64>],
    current: Vec<f64>,
    lower: &'a [f64],
    base: &'a [f64],
    residual: Vec<f64>,
    step: Vec<f64>,
    predicted: f64,
    radius: f64,
    step_norm: f64,
}

impl Clipping<'_> {
    fn constrained(&mut self, cache: &mut Trust, mut proposal: Step) -> Step {
        let mut reduced = false;
        while self.apply(&proposal) {
            // DG7QSB repeats DG7QTS on the remaining free coordinates after a
            // clipped step leaves too little room for the current diagonal shift.
            let remaining = self.radius - self.step_norm;
            *cache = Trust::new(proposal.alpha, cache.previous_radius());
            proposal = cache.compute(&self.reduced_hessian(), &self.residual, remaining);
            reduced = true;
        }
        if reduced {
            // The next proposal starts with the full free-coordinate set.
            *cache = Trust::new(proposal.alpha, cache.previous_radius());
        }
        proposal
    }
    fn apply(&mut self, proposal: &Step) -> bool {
        let initial_free = self.free.len();
        let alpha = proposal.alpha.abs();
        let mut direction = proposal.vector.clone();
        let mut gt = -proposal.gradient_product;
        let mut clipped = Vec::new();
        let mut restart = false;
        loop {
            let (fraction, mut hit) = self.fraction(&direction);
            let candidate = self.candidate(&direction, fraction);
            let candidate_norm = norm(&candidate);
            if candidate_norm > 1.0001 * 1.1 * self.radius {
                if self.free.len() < initial_free {
                    restart = self.step_norm < 0.9 * self.radius;
                    break;
                }
                hit = None;
            }
            self.update(candidate, candidate_norm, &direction, fraction, alpha, gt);
            let Some(hit) = hit else {
                break;
            };
            clipped.push(self.free.remove(hit));
            self.residual.remove(hit);
            if self.free.is_empty() {
                break;
            }
            let (l, failed) = cholesky(&self.reduced_hessian(), alpha);
            if failed.is_some() {
                break;
            }
            let solved = forward(&l, &self.residual);
            gt = dot(&solved, &solved);
            direction = backward(&l, &solved).into_iter().map(|v| -v).collect();
        }
        self.bias_clipped(&clipped);
        restart
    }

    fn reduced_hessian(&self) -> Vec<Vec<f64>> {
        self.free
            .iter()
            .map(|&i| self.free.iter().map(|&j| self.h[i][j]).collect())
            .collect()
    }

    fn candidate(&self, direction: &[f64], fraction: f64) -> Vec<f64> {
        let mut candidate = self.step.clone();
        for (k, &i) in self.free.iter().enumerate() {
            candidate[i] += fraction * direction[k];
        }
        candidate
    }

    fn update(
        &mut self,
        candidate: Vec<f64>,
        candidate_norm: f64,
        direction: &[f64],
        fraction: f64,
        alpha: f64,
        gt: f64,
    ) {
        self.step = candidate;
        self.step_norm = candidate_norm;
        self.predicted += fraction
            * ((1.0 - 0.5 * fraction) * gt + 0.5 * alpha * fraction * dot(direction, direction));
        for (residual, &delta) in self.residual.iter_mut().zip(direction) {
            *residual = (1.0 - fraction) * *residual - fraction * alpha * delta;
        }
    }

    fn bias_clipped(&mut self, clipped: &[usize]) {
        for &i in clipped {
            self.step[i] -= 2.0 * f64::EPSILON * self.current[i].abs().max(self.base[i].abs());
        }
        for (i, value) in self.current.iter_mut().enumerate() {
            *value = self.base[i] + self.step[i];
        }
    }

    fn fraction(&self, direction: &[f64]) -> (f64, Option<usize>) {
        let mut fraction = 1.0;
        let mut hit = None;
        for (k, &i) in self.free.iter().enumerate() {
            let current = self.base[i] + self.step[i];
            if current + direction[k] < self.lower[i] {
                let ratio = (self.lower[i] - current) / direction[k];
                // DS7BQN sets K before comparing TI with T: the bound index
                // is the last violated coordinate, while T is the minimum.
                hit = Some(k);
                if ratio < fraction {
                    fraction = ratio;
                }
            }
        }
        (fraction, hit)
    }
}
