//! Multivariate normal MLE and conditional draws used by rrcovNA's norm path.
use mokume_core::{MokumeError, Result};

struct Model {
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
}

pub(super) fn impute(matrix: &[Vec<Option<f64>>]) -> Result<Vec<Vec<f64>>> {
    let (center, scale, standardized) = standardize(matrix)?;
    let model = fit(&standardized)?;
    let mut result = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // norm processes missingness patterns in reverse order, preserving row order
    // within a pattern. Its own generator is seeded to 1234567 by impSeqRob.
    let mut order = (0..matrix.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        standardized[b]
            .iter()
            .rev()
            .map(Option::is_none)
            .cmp(standardized[a].iter().rev().map(Option::is_none))
    });
    let mut random = NormRandom::new();
    let _ = random.normal(); // is1n makes one initial, unused gauss() call.
    let mut theta = augmented(&model);
    for i in order {
        let (missing, mean, covariance) = conditional(&standardized[i], &mut theta)?;
        let lower = cholesky(&covariance)?;
        let mut noise = Vec::new();
        for (a, &j) in missing.iter().enumerate() {
            noise.push(random.normal());
            let draw = mean[a] + (0..=a).map(|b| lower[a][b] * noise[b]).sum::<f64>();
            result[i][j] = draw * scale[j] + center[j];
        }
    }
    Ok(result)
}

type Standardized = (Vec<f64>, Vec<f64>, Vec<Vec<Option<f64>>>);

fn standardize(matrix: &[Vec<Option<f64>>]) -> Result<Standardized> {
    let p = matrix[0].len();
    let mut center = vec![0.0; p];
    let mut scale = vec![1.0; p];
    let mut order = (0..matrix.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        matrix[a]
            .iter()
            .rev()
            .map(Option::is_none)
            .cmp(matrix[b].iter().rev().map(Option::is_none))
    });
    for j in 0..p {
        let values = order
            .iter()
            .filter_map(|&i| matrix[i][j])
            .collect::<Vec<_>>();
        if values.is_empty() {
            return Err(invalid(
                "norm initialization requires observations in every column",
            ));
        }
        let sum = values.iter().sum::<f64>();
        center[j] = sum / values.len() as f64;
        let variance = (values.iter().map(|v| v * v).sum::<f64>()
            - sum * sum / values.len() as f64)
            / values.len() as f64;
        if variance > 0.0 {
            scale[j] = variance.sqrt();
        }
    }
    let standardized = matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| v.map(|x| (x - center[j]) / scale[j]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((center, scale, standardized))
}

fn fit(x: &[Vec<Option<f64>>]) -> Result<Model> {
    let p = x[0].len();
    let mut model = Model {
        mean: vec![0.0; p],
        cov: (0..p)
            .map(|i| (0..p).map(|j| f64::from(i == j)).collect())
            .collect(),
    };
    let mut order = (0..x.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        x[a].iter()
            .rev()
            .map(Option::is_none)
            .cmp(x[b].iter().rev().map(Option::is_none))
    });
    let (known_mean, known_second) = observed_statistics(x, &order);
    for _ in 0..1000 {
        let (mut mean, mut second) =
            expected_statistics(x, &order, &model, &known_mean, &known_second)?;
        for value in &mut mean {
            *value /= x.len() as f64;
        }
        for a in 0..p {
            for b in 0..p {
                second[a][b] = second[a][b] / x.len() as f64 - mean[a] * mean[b];
            }
        }
        let change = mean
            .iter()
            .chain(second.iter().flatten())
            .zip(model.mean.iter().chain(model.cov.iter().flatten()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        model = Model { mean, cov: second };
        if change <= 1e-4 {
            break;
        }
    }
    Ok(model)
}

fn expected_statistics(
    x: &[Vec<Option<f64>>],
    order: &[usize],
    model: &Model,
    known_mean: &[f64],
    known_second: &[Vec<f64>],
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut mean = known_mean.to_vec();
    let mut second = known_second.to_vec();
    let mut theta = augmented(model);
    for &i in order {
        let row = &x[i];
        let (missing, conditional_mean, conditional_cov) = conditional(row, &mut theta)?;
        for (a, &i) in missing.iter().enumerate() {
            mean[i] += conditional_mean[a];
            for (j, observed) in row.iter().enumerate() {
                if let Some(value) = observed {
                    second[i][j] += conditional_mean[a] * value;
                    second[j][i] = second[i][j];
                }
            }
            for (b, &j) in missing.iter().enumerate().skip(a) {
                second[i][j] += conditional_mean[a] * conditional_mean[b] + conditional_cov[a][b];
                second[j][i] = second[i][j];
            }
        }
    }
    Ok((mean, second))
}

fn observed_statistics(x: &[Vec<Option<f64>>], order: &[usize]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let p = x[0].len();
    let mut mean = vec![0.0; p];
    let mut second = vec![vec![0.0; p]; p];
    for &i in order {
        for (a, left) in x[i].iter().enumerate() {
            if let Some(left) = left {
                mean[a] += left;
                for (b, right) in x[i].iter().enumerate().skip(a) {
                    if let Some(right) = right {
                        second[a][b] += left * right;
                        second[b][a] = second[a][b];
                    }
                }
            }
        }
    }
    (mean, second)
}

type Conditional = (Vec<usize>, Vec<f64>, Vec<Vec<f64>>);

fn augmented(model: &Model) -> Vec<Vec<f64>> {
    let p = model.mean.len();
    let mut theta = vec![vec![0.0; p + 1]; p + 1];
    theta[0][0] = -1.0;
    for i in 0..p {
        theta[0][i + 1] = model.mean[i];
        theta[i + 1][0] = model.mean[i];
        for j in 0..p {
            theta[i + 1][j + 1] = model.cov[i][j];
        }
    }
    theta
}

/// Carry the swept matrix between missingness patterns as norm::swpobs does.
fn conditional(row: &[Option<f64>], theta: &mut [Vec<f64>]) -> Result<Conditional> {
    for (j, value) in row.iter().enumerate() {
        let swept = theta[j + 1][j + 1] < 0.0;
        if value.is_some() != swept {
            sweep(theta, j + 1, if value.is_some() { 1.0 } else { -1.0 })?;
        }
    }
    let missing = (0..row.len())
        .filter(|&j| row[j].is_none())
        .collect::<Vec<_>>();
    let mean = missing
        .iter()
        .map(|&j| {
            let mut value = theta[0][j + 1];
            for (i, observed) in row.iter().enumerate() {
                if let Some(x) = observed {
                    value += theta[i + 1][j + 1] * x;
                }
            }
            value
        })
        .collect();
    let covariance = missing
        .iter()
        .map(|&i| missing.iter().map(|&j| theta[i + 1][j + 1]).collect())
        .collect();
    Ok((missing, mean, covariance))
}

fn sweep(theta: &mut [Vec<f64>], pivot: usize, direction: f64) -> Result<()> {
    let a = theta[pivot][pivot];
    if a == 0.0 || !a.is_finite() {
        return Err(invalid("norm EM encountered a singular covariance pivot"));
    }
    theta[pivot][pivot] = -1.0 / a;
    for (j, row) in theta.iter_mut().enumerate() {
        if j != pivot {
            row[pivot] = row[pivot] / a * direction;
        }
    }
    let column = theta.iter().map(|row| row[pivot]).collect::<Vec<_>>();
    theta[pivot].copy_from_slice(&column);
    for i in 0..theta.len() {
        for j in i..theta.len() {
            if i != pivot && j != pivot {
                theta[i][j] -= a * theta[i][pivot] * theta[j][pivot];
                theta[j][i] = theta[i][j];
            }
        }
    }
    Ok(())
}

fn cholesky(cov: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = cov.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let residual = cov[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            if i == j {
                if residual <= 0.0 {
                    return Err(invalid(
                        "norm conditional covariance is not positive definite",
                    ));
                }
                lower[i][j] = residual.sqrt();
            } else {
                lower[i][j] = residual / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn invalid(message: &str) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.to_owned(),
    }
}

/// norm 1.0-11.1 uses Park-Miller and single-precision Box-Muller arithmetic.
struct NormRandom {
    state: u64,
    spare: Option<f32>,
}
impl NormRandom {
    fn new() -> Self {
        let mut r = Self {
            state: 1234567,
            spare: None,
        };
        let _ = r.uniform();
        r
    }
    fn uniform(&mut self) -> f32 {
        self.state = self.state * 16807 % 2147483647;
        self.state as f32 * 4.656_613e-10_f32
    }
    fn normal(&mut self) -> f64 {
        if let Some(x) = self.spare.take() {
            return f64::from(x);
        }
        let radius = (-2.0_f32 * self.uniform().ln()).sqrt();
        // Preserve norm's literal 3.141593 as f32, not the mathematical PI constant.
        let angle = 2.0_f32 * f32::from_bits(0x40490fdc) * self.uniform();
        self.spare = Some(radius * angle.sin());
        f64::from(radius * angle.cos())
    }
}
