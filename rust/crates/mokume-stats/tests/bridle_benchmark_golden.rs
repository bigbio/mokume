//! Golden test for the benchmark-winning BRIDLE configuration (the defaults:
//! graph prior, sample loading kept, output-change stop rule, post-fit plex
//! rescale).
//!
//! A synthetic collection with a reference, a TMT dataset of 3 plexes (one
//! with inflated spread), an LFQ dataset, a 4-line and a single-line dataset
//! and a dataset linked by only 2 lines (centring prior). The generator mirrors
//! `rust/scripts/bridle_benchmark_reference.py` draw for draw (NumPy
//! `RandomState` stream); that script runs the benchmark's Python reference
//! (`graphp` prior + `bridle_var.fit_var` with `gprior`/`addc` +
//! `posthoc_delta_all.py`) and its numbers are stored in
//! `tests/fixtures/bridle_benchmark_expected.tsv`.

use std::collections::HashMap;

use mokume_stats::batch::bridle::{
    bridle_fit, build_design, reference_abundance, BridleData, BridleParams, BridleResult,
    NumpyRandomState, PlexMode, SequenceFeatures,
};

const SEED: u32 = 20_261_003;
const G: usize = 200;
const NF: usize = 6;
const NL: usize = 44;
const N_PLEX: usize = 3;
/// Max |rust - python| over the stored prior offsets, fitted values (before
/// and after the rescale) and batch scales. Both sides run in float64; the
/// residual difference is summation order (ridge / ALS solves, alternating
/// means of the graph prior, grouped means of the rescale).
const PY_TOL: f64 = 1e-10;

const ORDER: [&str; 6] = ["REF", "TMT", "LFQ", "SMALL", "SINGLE", "UNL"];

fn members(d: &str) -> Vec<usize> {
    match d {
        "REF" => (0..25).collect(),
        "TMT" => (8..26).collect(),
        "LFQ" => (20..35).collect(),
        "SMALL" => (0..4).collect(),
        "SINGLE" => vec![5],
        _ => [0, 1].into_iter().chain(35..44).collect(),
    }
}

struct Fixture {
    data: BridleData,
    features: SequenceFeatures,
}

fn fixture() -> Fixture {
    let mut rs = NumpyRandomState::new(SEED);
    let x: Vec<Vec<f64>> = (0..G)
        .map(|_| (0..NF).map(|_| rs.next_gauss()).collect())
        .collect();
    let m: Vec<f64> = (0..G).map(|_| 20.0 + 2.0 * rs.next_gauss()).collect();
    let u: Vec<Vec<f64>> = (0..NL)
        .map(|_| (0..3).map(|_| rs.next_gauss()).collect())
        .collect();
    let v: Vec<Vec<f64>> = (0..G)
        .map(|_| (0..3).map(|_| 0.4 * rs.next_gauss()).collect())
        .collect();
    let r: Vec<Vec<f64>> = (0..NL)
        .map(|_| (0..G).map(|_| 0.2 * rs.next_gauss()).collect())
        .collect();
    let theta: Vec<Vec<f64>> = (0..NL)
        .map(|l| {
            (0..G)
                .map(|g| m[g] + (0..3).map(|k| u[l][k] * v[g][k]).sum::<f64>() + r[l][g])
                .collect()
        })
        .collect();
    let mut offsets: HashMap<&str, Vec<f64>> = HashMap::new();
    for d in ORDER {
        let a0 = 0.5 * rs.next_gauss();
        let beta: Vec<f64> = (0..NF).map(|_| 0.3 * rs.next_gauss()).collect();
        let rr: Vec<f64> = (0..G).map(|_| 0.15 * rs.next_gauss()).collect();
        let a: Vec<f64> = (0..G)
            .map(|g| {
                if d == "REF" {
                    0.0
                } else {
                    a0 + (0..NF).map(|j| x[g][j] * beta[j]).sum::<f64>() + rr[g]
                }
            })
            .collect();
        offsets.insert(d, a);
    }
    let mut p: Vec<Vec<f64>> = (0..N_PLEX)
        .map(|_| (0..G).map(|_| 0.3 * rs.next_gauss()).collect())
        .collect();
    for g in 0..G {
        let mean = (0..N_PLEX).map(|k| p[k][g]).sum::<f64>() / N_PLEX as f64;
        for row in p.iter_mut() {
            row[g] -= mean;
        }
    }
    let pmiss: Vec<Vec<bool>> = (0..N_PLEX)
        .map(|_| (0..G).map(|_| rs.next_f64() < 0.2).collect())
        .collect();
    let mut profiles: Vec<(String, String, Vec<f64>, Option<String>)> = Vec::new();
    for d in ORDER {
        for (j, &l) in members(d).iter().enumerate() {
            let c = 0.2 * rs.next_gauss();
            let eps: Vec<f64> = (0..G).map(|_| 0.2 * rs.next_gauss()).collect();
            let mut miss: Vec<bool> = (0..G).map(|_| rs.next_f64() < 0.1).collect();
            let k = (d == "TMT").then_some(j / 6);
            if let Some(k) = k {
                miss.clone_from(&pmiss[k]);
            }
            let scale = if k == Some(1) { 1.6 } else { 1.0 };
            let vals: Vec<f64> = (0..G)
                .map(|g| {
                    if miss[g] {
                        f64::NAN
                    } else {
                        m[g] + scale * (theta[l][g] - m[g])
                            + offsets[d][g]
                            + c
                            + eps[g]
                            + k.map_or(0.0, |k| p[k][g])
                    }
                })
                .collect();
            profiles.push((
                d.to_owned(),
                format!("L{l:02}"),
                vals,
                k.map(|k| format!("P{k}")),
            ));
        }
    }
    // wide matrix in pandas pivot order: sorted by (ds, line)
    profiles.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    let data = BridleData {
        datasets: profiles.iter().map(|p| p.0.clone()).collect(),
        lines: profiles.iter().map(|p| p.1.clone()).collect(),
        lineages: vec![None; profiles.len()],
        plexes: Some(profiles.iter().map(|p| p.3.clone()).collect()),
        genes: (0..G).map(|g| format!("G{g:03}")).collect(),
        values: profiles.iter().flat_map(|p| p.2.clone()).collect(),
    };
    let features = SequenceFeatures {
        names: (0..NF).map(|j| format!("f{j}")).collect(),
        values: x,
        n_missing: 0,
    };
    Fixture { data, features }
}

struct Expected {
    rows: Vec<Vec<String>>,
}

impl Expected {
    fn load() -> Self {
        let text = include_str!("fixtures/bridle_benchmark_expected.tsv");
        let rows = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| l.split('\t').map(str::to_owned).collect())
            .collect();
        Self { rows }
    }

    fn scalar(&self, kind: &str) -> f64 {
        self.rows
            .iter()
            .find(|r| r[0] == kind)
            .and_then(|r| r[4].parse().ok())
            .unwrap_or(f64::NAN)
    }

    fn of(&self, kind: &str) -> impl Iterator<Item = &Vec<String>> {
        let kind = kind.to_owned();
        self.rows.iter().filter(move |r| r[0] == kind)
    }
}

fn params(tol: f64) -> BridleParams {
    BridleParams {
        reference: "REF".to_owned(),
        rank: 4,
        sweeps: 120,
        min_sweeps: 40,
        converge_tol: tol,
        plex_mode: PlexMode::Explicit,
        ..BridleParams::default()
    }
}

fn run(fx: &Fixture, p: &BridleParams) -> BridleResult {
    let ab = reference_abundance(&fx.data, "REF");
    let (design, _, abz) = build_design(&ab, Some(&fx.features));
    match bridle_fit(&fx.data, &design, &abz, p) {
        Ok(r) => r,
        Err(e) => panic!("bridle_fit failed: {e}"),
    }
}

fn row_index(fx: &Fixture, ds: &str, line: &str) -> usize {
    (0..fx.data.n_profiles())
        .find(|&i| fx.data.datasets[i] == ds && fx.data.lines[i] == line)
        .unwrap_or(usize::MAX)
}

fn gene_index(gene: &str) -> usize {
    gene[1..].parse().unwrap_or(usize::MAX)
}

/// Max |got - want| over the expectation rows of `kind`, and their number.
fn max_cell_diff(fx: &Fixture, exp: &Expected, kind: &str, values: &[f64]) -> (f64, usize) {
    let mut max: f64 = 0.0;
    let mut n = 0;
    for r in exp.of(kind) {
        let want: f64 = r[4].parse().unwrap_or(f64::NAN);
        let got = values[row_index(fx, &r[1], &r[2]) * G + gene_index(&r[3])];
        max = max.max((got - want).abs());
        n += 1;
    }
    (max, n)
}

#[test]
fn matches_python_reference() {
    let fx = fixture();
    let exp = Expected::load();
    let tol = exp.scalar("stop_tol");
    let res = run(&fx, &params(tol));

    // graph prior
    let mut prior_max: f64 = 0.0;
    let mut n_prior = 0;
    for r in exp.of("prior") {
        let s = res
            .dataset_names
            .iter()
            .position(|d| *d == r[1])
            .unwrap_or(usize::MAX);
        let want: f64 = r[4].parse().unwrap_or(f64::NAN);
        prior_max = prior_max.max((res.prior_offsets[s * G + gene_index(&r[3])] - want).abs());
        n_prior += 1;
    }
    let n_rust = res.prior_offsets.iter().filter(|x| x.is_finite()).count();
    assert_eq!(
        n_rust, n_prior,
        "graph prior cells rust {n_rust} python {n_prior}"
    );

    // stop rule: same sweep, same output change
    assert_eq!(
        res.report.history.len(),
        exp.scalar("n_sweeps") as usize,
        "sweeps"
    );
    assert!(res.report.converged);
    let dout = res.report.history.last().map_or(f64::NAN, |h| h.out_change);
    let dout_diff = (dout - exp.scalar("dout_last")).abs();

    // fit (sample loading kept) before the rescale: same fit without it
    let pre = run(
        &fx,
        &BridleParams {
            plex_rescale: false,
            ..params(tol)
        },
    );
    let (fit_max, n_fit) = max_cell_diff(&fx, &exp, "fit", &pre.corrected);
    let (cell_max, n_cell) = max_cell_diff(&fx, &exp, "cell", &res.corrected);

    // batch scales
    let mut mu_max: f64 = 0.0;
    let mut n_mu = 0;
    for r in exp.of("mu") {
        let want: f64 = r[4].parse().unwrap_or(f64::NAN);
        let got = res
            .report
            .plex_rescale
            .iter()
            .find(|b| b.batch == r[1])
            .map_or(f64::NAN, |b| b.mu);
        mu_max = mu_max.max((got - want).abs());
        n_mu += 1;
    }
    assert_eq!(n_mu, res.report.plex_rescale.len());
    eprintln!(
        "BRIDLE benchmark golden: max |rust - python| prior {prior_max:.3e} ({n_prior}), \
         fit {fit_max:.3e} ({n_fit}), final {cell_max:.3e} ({n_cell}), mu {mu_max:.3e} ({n_mu}), \
         out_change {dout_diff:.3e}"
    );
    assert!(n_fit > 2000 && n_cell > 2000, "expected cells missing");
    for (what, d) in [
        ("prior", prior_max),
        ("fit", fit_max),
        ("final", cell_max),
        ("mu", mu_max),
        ("out_change", dout_diff),
    ] {
        assert!(d < PY_TOL, "{what}: max |rust - python| = {d}");
    }
}

#[test]
fn defaults_are_the_benchmark_configuration() {
    let d = BridleParams::default();
    assert!(d.graph_prior && d.keep_sample_loading && d.plex_rescale);
    assert!(!d.anchor_scale);
    assert_eq!((d.sweeps, d.min_sweeps), (400, 200));
}

#[test]
fn rescale_equalises_the_inflated_plex_and_keeps_every_value() {
    let fx = fixture();
    let exp = Expected::load();
    let res = run(&fx, &params(exp.scalar("stop_tol")));
    let mu = |b: &str| {
        res.report
            .plex_rescale
            .iter()
            .find(|r| r.batch == b)
            .map_or(f64::NAN, |r| r.mu)
    };
    // P1 was generated with 1.6x the spread of P0 / P2
    assert!(mu("TMT|P1") > mu("TMT|P0") + 0.4 && mu("TMT|P1") > mu("TMT|P2") + 0.2);
    for (v, x) in res.corrected.iter().zip(&fx.data.values) {
        assert_eq!(v.is_finite(), x.is_finite());
    }
}
