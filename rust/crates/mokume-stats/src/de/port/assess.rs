use super::trust::Step;

/// Single-model PORT step assessment (DA7SST); analytic Hessians use STGLIM=1.
pub(super) struct Assessment {
    pub code: u8,
    pub restore: u8,
    pub factor: f64,
    pub f: f64,
    pub f0: f64,
    pub difference: f64,
    pub relative_step: f64,
    pub step: Step,
    radius_increments: i32,
    saved_f: f64,
    saved_norm: f64,
    saved_pred: f64,
    saved_gt: f64,
    best_evaluation: usize,
    original_code: u8,
    good: bool,
}
impl Assessment {
    pub fn new(value: f64) -> Self {
        Self {
            code: 4,
            restore: 0,
            factor: 1.0,
            f: value,
            f0: value,
            difference: 0.0,
            relative_step: 0.0,
            step: Step::default(),
            radius_increments: 0,
            saved_f: value,
            saved_norm: 0.0,
            saved_pred: 0.0,
            saved_gt: 0.0,
            best_evaluation: 1,
            original_code: 4,
            good: true,
        }
    }
    pub fn begin(&mut self) {
        self.f0 = self.f;
        self.code = 4;
    }
    pub fn assess(&mut self, mut evaluation: usize) {
        self.restore = 0;
        self.good = true;
        let mut restored_scale = 1.0;
        if self.code == 6 {
            self.restore_singular();
            return;
        }
        if matches!(self.code, 3 | 4) {
            self.radius_increments = 0;
            self.saved_f = self.f0;
        }
        if self.code == 5 && self.f >= self.saved_f && self.saved_f < self.f0 {
            let factor = self.step.norm / self.saved_norm;
            restored_scale = factor;
            self.restore_best();
            if evaluation >= self.best_evaluation + 3 {
                self.difference = self.f0 - self.f;
                self.code = 4;
                self.factor = factor;
                return;
            }
            self.good = false;
            evaluation = self.best_evaluation;
        }
        self.difference = self.f0 - self.f;
        if self.difference <= 1e-4 * self.step.predicted && self.radius_increments <= 0 {
            self.negligible(evaluation, restored_scale);
        } else {
            self.best_evaluation = evaluation;
            self.saved_norm = self.step.norm;
            if self.difference <= 0.1 * self.step.predicted {
                self.code = 4;
                self.shrink(1.0);
            } else {
                self.successful();
            }
        }
        self.convergence();
    }
    fn restore_best(&mut self) {
        self.restore = 3;
        self.f = self.saved_f;
        self.step.predicted = self.saved_pred;
        self.step.gradient_product = self.saved_gt;
        self.step.norm = self.saved_norm;
    }
    fn negligible(&mut self, evaluation: usize, scale: f64) {
        if self.f >= self.f0 {
            self.saved_f = self.f;
            self.f = self.f0;
            self.restore = 1;
        } else {
            self.best_evaluation = evaluation;
        }
        self.code = 5;
        self.radius_increments -= 1;
        self.shrink(scale);
    }
    fn shrink(&mut self, scale: f64) {
        self.original_code = self.code;
        let emax = self.step.gradient_product + self.difference;
        self.factor = 0.5 * scale;
        if emax < self.step.gradient_product {
            self.factor = scale * 0.1_f64.max(0.5 * self.step.gradient_product / emax);
        }
        if self.relative_step <= 100.0 * f64::EPSILON {
            self.code = 12;
            return;
        }
        if self.f < self.f0 {
            self.save();
        }
    }
    fn successful(&mut self) {
        if self.difference < -0.75 * self.step.gradient_product
            || self.radius_increments < 0
            || matches!(self.restore, 1 | 3)
        {
            self.factor = 1.0;
            self.code = 3;
            return;
        }
        self.factor = 4.0;
        let gt = self.step.gradient_product;
        if self.difference < (0.5 / self.factor - 1.0) * gt {
            self.factor = 2.0_f64.max(0.5 * gt / (gt + self.difference));
        }
        self.code = 4;
        if self.step.alpha != 0.0
            && !(self.step.newton_norm >= 0.0
                && (self.step.newton_norm < 2.0 * self.step.norm
                    || self.step.newton_predicted < 1.2 * self.difference))
        {
            self.code = 5;
            self.radius_increments += 1;
        }
        if self.code == 5 {
            self.save();
        }
    }
    fn save(&mut self) {
        self.saved_f = self.f;
        if self.restore == 0 {
            self.restore = 2;
        }
        self.saved_norm = self.step.norm;
        self.saved_pred = self.step.predicted;
        self.saved_gt = self.step.gradient_product;
    }
    fn convergence(&mut self) {
        if self.code != 12 {
            self.original_code = self.code;
        }
        if self.restore == 1 && self.saved_f < self.f0 {
            self.restore = 3;
        }
        if 0.5 * self.difference > self.step.predicted {
            return;
        }
        let threshold = 1e-10 * self.f0.abs();
        if self.step.predicted <= threshold && (self.step.norm > 1.0 || self.step.alpha == 0.0) {
            self.code = 11;
        }
        let convergence = self.x_and_relative_convergence(threshold);
        if convergence > 0 {
            self.code = convergence + 6;
        }
        if self.code > 5 && self.code != 12 {
            return;
        }
        if self.step.alpha != 0.0 {
            self.singular_check(threshold);
        }
    }
    fn x_and_relative_convergence(&self, threshold: f64) -> u8 {
        if self.step.newton_norm < 0.0 {
            return 0;
        }
        let mut result = 0;
        if (self.step.newton_predicted > 0.0 && self.step.newton_predicted <= threshold)
            || (self.step.newton_predicted == 0.0 && self.step.predicted == 0.0)
        {
            result = 2;
        }
        if self.step.alpha == 0.0 && self.relative_step <= f64::EPSILON.sqrt() && self.good {
            result += 1;
        }
        result
    }
    fn singular_check(&mut self, threshold: f64) {
        if self.step.norm <= 1.0 {
            if self.step.predicted >= threshold {
                return;
            }
            if self.step.newton_norm > 0.0 && self.step.newton_norm <= 2.0 {
                return;
            }
        } else {
            if self.step.norm <= 2.0 {
                return;
            }
            let scale = 1.0 / self.step.norm;
            if scale * (2.0 - scale) * self.step.predicted >= threshold {
                return;
            }
        }
        if self.step.newton_predicted < 0.0 {
            if -self.step.newton_predicted <= threshold {
                self.code = 11;
            }
            return;
        }
        self.saved_gt = self.step.gradient_product;
        self.saved_norm = self.step.norm;
        if self.code == 12 {
            self.saved_norm = -self.saved_norm;
        }
        self.saved_pred = self.step.predicted;
        self.restore = if self.restore == 3 { 0 } else { 2 };
        self.code = 6;
    }
    fn restore_singular(&mut self) {
        self.step.gradient_product = self.saved_gt;
        self.step.norm = self.saved_norm.abs();
        self.code = if self.saved_norm <= 0.0 {
            12
        } else {
            self.original_code
        };
        self.step.newton_predicted = -self.step.predicted;
        self.step.predicted = self.saved_pred;
        self.restore = 3;
        if -self.step.newton_predicted <= 1e-10 * self.f0.abs() {
            self.code = 11;
        }
    }
}
