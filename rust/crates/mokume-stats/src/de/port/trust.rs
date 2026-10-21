use super::math::{
    backward, cholesky, dot, eigen_bounds, failed_bound, forward, norm, singular_estimate,
};

#[derive(Clone, Default)]
pub(super) struct Step {
    pub vector: Vec<f64>,
    pub alpha: f64,
    pub norm: f64,
    pub newton_norm: f64,
    pub gradient_product: f64,
    pub predicted: f64,
    pub newton_predicted: f64,
}

#[derive(Default)]
pub(super) struct Trust {
    pub alpha: f64,
    radius: f64,
    dst: f64,
    lower: f64,
    upper: f64,
    emin: f64,
    emax: f64,
    max_entry: f64,
    phipin: f64,
    newton_norm: f64,
    newton_predicted: f64,
    ka: usize,
    initialized: bool,
    q: Vec<f64>,
    chol: Vec<Vec<f64>>,
}

struct Search<'a> {
    h: &'a [Vec<f64>],
    g: &'a [f64],
    radius: f64,
    gradient_norm: f64,
    alpha: f64,
    lower: f64,
    upper: f64,
    old_phi: f64,
    iteration: usize,
    minimum: usize,
    limit: usize,
    q: Vec<f64>,
    chol: Vec<Vec<f64>>,
    gtsta: f64,
    dst: f64,
}

impl Trust {
    pub fn previous_radius(&self) -> f64 {
        self.radius
    }
    pub fn new(alpha: f64, radius: f64) -> Self {
        Self {
            alpha,
            radius,
            ..Self::default()
        }
    }
    pub fn compute(&mut self, h: &[Vec<f64>], g: &[f64], radius: f64) -> Step {
        let mut search = Search {
            h,
            g,
            radius,
            gradient_norm: norm(g),
            alpha: self.alpha.abs(),
            lower: 0.0,
            upper: 0.0,
            old_phi: 0.0,
            iteration: self.ka,
            minimum: self.ka + 3,
            limit: self.ka + 50,
            q: vec![0.0; g.len()],
            chol: vec![],
            gtsta: 0.0,
            dst: 0.0,
        };
        if !self.initialized {
            if let Some(step) = self.initialize(&mut search) {
                return step;
            }
        } else if let Some(step) = self.restart(&mut search) {
            return step;
        }
        self.iterate(&mut search)
    }

    fn initialize(&mut self, s: &mut Search<'_>) -> Option<Step> {
        self.initialized = true;
        self.ka = 0;
        s.iteration = 0;
        s.limit = 50;
        s.minimum = if s.gradient_norm == 0.0 { 0 } else { 3 };
        self.newton_norm = 0.0;
        self.newton_predicted = 0.0;
        self.max_entry = s.h.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        let (chol, failed) = cholesky(s.h, 0.0);
        s.chol = chol;
        if let Some(failed) = failed {
            s.lower = failed_bound(s.chol.clone(), failed);
            self.newton_norm = -s.lower;
        } else {
            self.newton(s);
            if s.dst - s.radius <= 0.1 * s.radius {
                s.alpha = 0.0;
                return Some(self.finish(s));
            }
        }
        (self.emin, self.emax) = eigen_bounds(s.h);
        s.upper = self.upper_limit(s.gradient_norm, s.radius);
        s.lower = s.lower.max(s.gradient_norm / s.radius - self.emax);
        s.alpha = (self.alpha.abs() * self.radius / s.radius)
            .max(s.lower)
            .min(s.upper);
        if failed.is_none() {
            let v = forward(&s.chol, &s.q);
            let t = norm(&v);
            self.phipin = s.radius / t / t;
            s.lower = s.lower.max((s.dst - s.radius) * self.phipin);
        }
        None
    }

    fn newton(&mut self, s: &mut Search<'_>) {
        let solved = forward(&s.chol, s.g);
        s.gtsta = dot(&solved, &solved);
        s.q = backward(&s.chol, &solved);
        s.dst = norm(&s.q);
        self.newton_predicted = 0.5 * s.gtsta;
        self.newton_norm = s.dst;
    }

    fn upper_limit(&self, gnorm: f64, radius: f64) -> f64 {
        let mut upper = gnorm / radius - self.emin;
        if gnorm == 0.0 {
            upper += 0.001 + 0.001 * upper;
        }
        if upper <= 0.0 {
            0.001
        } else {
            upper
        }
    }

    fn restart(&mut self, s: &mut Search<'_>) -> Option<Step> {
        if self.newton_norm > 0.0 && self.newton_norm - s.radius <= 0.1 * s.radius {
            s.iteration += 1;
            let (l, fail) = cholesky(s.h, 0.0);
            s.chol = l;
            if fail.is_none() {
                self.newton(s);
                s.alpha = 0.0;
                return Some(self.finish(s));
            }
        }
        // A previous unshifted Newton step bypasses the spectral bracket.
        // DG7QTS resumes at label 50 when KA=0 and that step no longer fits.
        if self.ka == 0 {
            return self.initialize(s);
        }
        s.minimum = if s.gradient_norm == 0.0 {
            0
        } else {
            self.ka + 3
        };
        s.dst = self.dst;
        s.upper = self.upper_limit(s.gradient_norm, s.radius);
        if s.radius <= self.radius {
            s.lower = if s.alpha > 0.0 { self.lower } else { 0.0 };
            s.lower = s.lower.max(s.gradient_norm / s.radius - self.emax);
        } else {
            if s.alpha > 0.0 {
                s.upper = s.upper.min(self.upper);
            }
            s.lower = 0.0_f64
                .max(-self.newton_norm)
                .max(s.gradient_norm / s.radius - self.emax);
        }
        if self.newton_norm > 0.0 {
            s.lower = s.lower.max((self.newton_norm - s.radius) * self.phipin);
        }
        // DG7QTS retains W(Q) and L, including the hard-case direction;
        // solving again from g would discard that direction on a restart.
        s.chol = self.chol.clone();
        s.q = self.q.clone();
        if s.dst - s.radius < 0.0 {
            s.upper = s.upper.min(s.alpha);
        }
        self.update_alpha(s);
        None
    }

    fn iterate(&mut self, s: &mut Search<'_>) -> Step {
        loop {
            s.iteration += 1;
            if -self.newton_norm >= s.alpha || s.alpha < s.lower || s.alpha >= s.upper {
                s.alpha = s.upper * 0.001_f64.max((s.lower / s.upper).sqrt());
            }
            if s.alpha <= 0.0 {
                s.alpha = 0.5 * s.upper;
            }
            if s.alpha <= 0.0 {
                s.alpha = s.upper;
            }
            let (l, failed) = cholesky(s.h, s.alpha);
            s.chol = l;
            if let Some(index) = failed {
                self.raise_lower(s, index);
                continue;
            }
            let tmp = forward(&s.chol, s.g);
            s.gtsta = dot(&tmp, &tmp);
            s.q = backward(&s.chol, &tmp);
            s.dst = norm(&s.q);
            let phi = s.dst - s.radius;
            if (phi <= 0.1 * s.radius && phi >= -0.1 * s.radius) || phi == s.old_phi {
                return self.finish(s);
            }
            s.old_phi = phi;
            if phi < 0.0 {
                if let Some(step) = self.hard_case(s) {
                    return step;
                }
            }
            if s.iteration >= s.limit {
                return self.finish(s);
            }
            if phi < 0.0 {
                s.upper = s.upper.min(s.alpha);
            }
            self.update_alpha(s);
        }
    }

    fn raise_lower(&mut self, s: &mut Search<'_>, failed: usize) {
        s.lower = s.alpha + failed_bound(s.chol.clone(), failed);
        self.newton_norm = -s.lower;
        s.upper = s.upper.max(s.lower);
        if s.alpha >= s.lower {
            let t = if 0.001 * s.alpha > 0.0 {
                0.001 * s.alpha
            } else {
                0.001
            };
            s.lower = s.alpha + t;
            if s.upper <= s.lower {
                s.upper = s.lower + t;
            }
        }
    }

    fn update_alpha(&self, s: &mut Search<'_>) {
        if s.minimum == 0 {
            return;
        }
        let solved = forward(&s.chol, &s.q);
        let t = norm(&solved);
        s.alpha += ((s.dst - s.radius) / t) * (s.dst / t) * (s.dst / s.radius);
        s.lower = s.lower.max(s.alpha);
        s.alpha = s.lower;
    }

    fn hard_case(&mut self, s: &mut Search<'_>) -> Option<Step> {
        let delta = s.alpha + self.newton_norm.min(0.0);
        let twice = s.alpha * s.dst * s.dst + s.gtsta;
        let psifac = (0.2 / (3.0 * (4.0 * 0.9 * 3.0 + 4.0) * s.radius)) / s.radius;
        if s.iteration >= s.minimum || delta < psifac * twice {
            let (small, left, right) = singular_estimate(&s.chol);
            let scaled: Vec<_> = right.iter().map(|v| small * v).collect();
            let mut direction = backward(&s.chol, &scaled);
            let t2 = 1.0 / norm(&direction);
            for v in &mut direction {
                *v *= t2;
            }
            let t = t2 * small;
            let sw = dot(&s.q, &direction);
            let extra = (s.radius + s.dst) * (s.radius - s.dst);
            let root = (sw * sw + extra).sqrt().copysign(sw);
            let si = extra / (sw + root);
            if (t2 * si).powi(2) <= 0.1 * (s.dst * s.dst + s.alpha * s.radius * s.radius) {
                s.alpha = -s.alpha;
                let gain = si * (s.alpha * sw - 0.5 * si * (s.alpha + t * dot(&left, &direction)));
                let mut out = self.finish(s);
                out.predicted = 0.5 * twice;
                if gain >= 0.1 * twice / 6.0 {
                    out.predicted += gain;
                    out.norm = s.radius;
                    for (i, value) in out.vector.iter_mut().enumerate() {
                        *value -= si * direction[i];
                    }
                    out.gradient_product = dot(s.g, &out.vector);
                }
                self.q = out.vector.clone();
                self.dst = out.norm;
                return Some(out);
            }
            if self.newton_norm <= 0.0 {
                self.newton_norm = self.newton_norm.min(t2 * t2 - s.alpha);
            }
            s.lower = s.lower.max(-self.newton_norm);
        }
        if delta <= 50.0 * f64::EPSILON * self.max_entry {
            return Some(self.finish(s));
        }
        None
    }

    fn finish(&mut self, s: &Search<'_>) -> Step {
        self.alpha = s.alpha;
        self.radius = s.radius;
        self.dst = s.dst;
        self.lower = s.lower;
        self.upper = s.upper;
        self.ka = s.iteration;
        self.q = s.q.clone();
        self.chol = s.chol.clone();
        Step {
            vector: s.q.iter().map(|v| -v).collect(),
            alpha: s.alpha,
            norm: s.dst,
            newton_norm: self.newton_norm,
            gradient_product: -s.gtsta,
            predicted: 0.5 * (s.alpha.abs() * s.dst * s.dst + s.gtsta),
            newton_predicted: self.newton_predicted,
        }
    }
}
