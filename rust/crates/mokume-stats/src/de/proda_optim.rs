// R stats::optim's default Nelder-Mead controls for proDA hyperparameters.
// Initial simplex and stopping/evaluation rules follow R's nmmin, including
// a shared .1*max(abs(x0)) step and sqrt(eps)*(abs(f0)+sqrt(eps)) tolerance.

struct Simplex {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    low: usize,
    evaluations: usize,
    old_size: f64,
}

fn finite_objective(f: &impl Fn(&[f64]) -> f64, x: &[f64]) -> f64 {
    let value = f(x);
    if value.is_finite() {
        value
    } else {
        1e35
    }
}

fn initialize(f: &impl Fn(&[f64]) -> f64, x0: &[f64]) -> Simplex {
    let n = x0.len();
    let mut points = vec![x0.to_vec(); n + 1];
    let mut step = x0.iter().map(|v| 0.1 * v.abs()).fold(0.0, f64::max);
    if step == 0.0 {
        step = 0.1;
    }
    let mut size = 0.0;
    for i in 0..n {
        let mut trial = step;
        while points[i + 1][i] == x0[i] {
            points[i + 1][i] = x0[i] + trial;
            trial *= 10.0;
        }
        size += trial;
    }
    let values = points.iter().map(|p| finite_objective(f, p)).collect();
    Simplex {
        points,
        values,
        low: 0,
        evaluations: n + 1,
        old_size: size,
    }
}

/// (parameters, converged). R permits its last iteration to exceed maxit.
fn r_nelder_mead(f: &impl Fn(&[f64]) -> f64, x0: &[f64]) -> (Vec<f64>, bool) {
    let mut simplex = initialize(f, x0);
    let tolerance = f64::EPSILON.sqrt() * (simplex.values[0].abs() + f64::EPSILON.sqrt());
    loop {
        let (lo, hi) = extrema(&simplex);
        simplex.low = lo;
        if simplex.values[hi] <= simplex.values[lo] + tolerance {
            return (simplex.points[lo].clone(), true);
        }
        if !simplex_step(f, &mut simplex, hi) {
            return (simplex.points[lo].clone(), false);
        }
        if simplex.evaluations > 500 {
            return (simplex.points[lo].clone(), false);
        }
    }
}

fn extrema(s: &Simplex) -> (usize, usize) {
    let (mut lo, mut hi) = (s.low, s.low);
    for j in 0..s.points.len() {
        if j != lo {
            if s.values[j] < s.values[lo] {
                lo = j;
            }
            if s.values[j] > s.values[hi] {
                hi = j;
            }
        }
    }
    (lo, hi)
}

fn simplex_center(s: &Simplex, hi: usize) -> Vec<f64> {
    let n = s.points[0].len();
    (0..n)
        .map(|i| {
            let mut sum = -s.points[hi][i];
            for point in &s.points {
                sum += point[i];
            }
            sum / n as f64
        })
        .collect()
}

fn simplex_step(f: &impl Fn(&[f64]) -> f64, s: &mut Simplex, hi: usize) -> bool {
    let center = simplex_center(s, hi);
    let reflected = center
        .iter()
        .zip(&s.points[hi])
        .map(|(c, h)| 2.0 * c - h)
        .collect::<Vec<_>>();
    let vr = finite_objective(f, &reflected);
    s.evaluations += 1;
    let vh = s.values[hi];
    if vr < s.values[s.low] {
        let extended = reflected
            .iter()
            .zip(&center)
            .map(|(r, c)| 2.0 * r - c)
            .collect::<Vec<_>>();
        let ve = finite_objective(f, &extended);
        s.evaluations += 1;
        (s.points[hi], s.values[hi]) = if ve < vr {
            (extended, ve)
        } else {
            (reflected, vr)
        };
        return true;
    }
    if vr < vh {
        s.points[hi] = reflected;
        s.values[hi] = vr;
    }
    let contracted = s.points[hi]
        .iter()
        .zip(&center)
        .map(|(h, c)| 0.5 * h + 0.5 * c)
        .collect::<Vec<_>>();
    let vc = finite_objective(f, &contracted);
    s.evaluations += 1;
    if vc < s.values[hi] {
        s.points[hi] = contracted;
        s.values[hi] = vc;
        return true;
    }
    if vr < vh {
        return true;
    }
    shrink_simplex(f, s)
}

fn shrink_simplex(f: &impl Fn(&[f64]) -> f64, s: &mut Simplex) -> bool {
    let low = s.points[s.low].clone();
    let mut size = 0.0;
    for j in 0..s.points.len() {
        if j == s.low {
            continue;
        }
        for (x, &l) in s.points[j].iter_mut().zip(&low) {
            *x = 0.5 * (*x - l) + l;
            size += (*x - l).abs();
        }
    }
    if size >= s.old_size {
        return false;
    }
    s.old_size = size;
    for j in 0..s.points.len() {
        if j != s.low {
            s.values[j] = finite_objective(f, &s.points[j]);
            s.evaluations += 1;
        }
    }
    true
}

/// R uniroot/R_zeroin2, using its default eps^.25 absolute tolerance.
fn r_uniroot(f: &impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (mut c, mut fc) = (a, fa);
    let tolerance = f64::EPSILON.sqrt().sqrt();
    for _ in 0..=1000 {
        let previous = b - a;
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + tolerance / 2.0;
        let mut step = (c - b) / 2.0;
        if step.abs() <= tol || fb == 0.0 {
            return b;
        }
        if let Some(interpolated) = root_interpolation([a, b, c], [fa, fb, fc], tol, previous) {
            step = interpolated;
        }
        if step.abs() < tol {
            step = if step > 0.0 { tol } else { -tol };
        }
        a = b;
        fa = fb;
        b += step;
        fb = f(b);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    b
}

/// Accept the same secant/inverse-quadratic proposal as R_zeroin2.
fn root_interpolation(x: [f64; 3], f: [f64; 3], tol: f64, previous: f64) -> Option<f64> {
    let [a, b, c] = x;
    let [fa, fb, fc] = f;
    if !(previous.abs() >= tol && fa.abs() > fb.abs()) {
        return None;
    }
    let cb = c - b;
    let (mut p, mut q) = if a == c {
        let ratio = fb / fa;
        (cb * ratio, 1.0 - ratio)
    } else {
        let (q, t1, t2) = (fa / fc, fb / fc, fb / fa);
        (
            t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0)),
            (q - 1.0) * (t1 - 1.0) * (t2 - 1.0),
        )
    };
    if p > 0.0 {
        q = -q;
    } else {
        p = -p;
    }
    if p < 0.75 * cb * q - (tol * q).abs() / 2.0 && p < (previous * q / 2.0).abs() {
        Some(p / q)
    } else {
        None
    }
}
