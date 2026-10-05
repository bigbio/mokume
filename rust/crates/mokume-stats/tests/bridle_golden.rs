//! Golden test for BRIDLE (`mokume_stats::batch::bridle`).
//!
//! A synthetic collection with known biology, known technical offsets, TMT
//! plexes, a cross-fitted weakly anchored dataset, a single-line dataset and an
//! unanchored dataset, drawn from a NumPy `RandomState`-compatible stream. The
//! expected values in `tests/fixtures/bridle_golden_expected.tsv` were
//! generated once by the Python BRIDLE (LIM) prototype
//! (`research/cellline-integration/lim_lin/lim.py`, the benchmark `lim_lin`
//! variant) on this collection and are now frozen; running the test needs no
//! Python. The generator lived at `rust/scripts/bridle_golden_reference.py`
//! until 7091171 (`git show 4ca3608:rust/scripts/bridle_golden_reference.py`).
//!
//! Checks:
//! 1. Rust matches the frozen expected values cell by cell (tolerance [`TOL`]).
//! 2. Technical offsets are recovered, biology is preserved, the single-line
//!    dataset keeps its own signal, and nothing is imputed.

use std::collections::HashMap;

use mokume_stats::batch::bridle::{
    bridle_fit, build_design, reference_abundance, BridleData, BridleParams, BridleResult,
    NumpyRandomState, PlexMode, SequenceFeatures,
};

const SEED: u32 = 20_260_927;
const G: usize = 240;
const NF: usize = 6;
const NL: usize = 80;
const N_PLEX: usize = 6;
/// Max |v_rust - v_expected| over the stored cells. Both sides run in float64;
/// the residual difference is summation order in the ridge / ALS solves.
const TOL: f64 = 1e-6;

const ORDER: [&str; 6] = ["REF", "DSB", "DSC", "DSD", "SINGLE", "UNB"];

fn members(d: &str) -> Vec<usize> {
    match d {
        "REF" => (0..40).collect(),
        "DSB" => (20..50).collect(),
        "DSC" => (30..60).collect(),
        "DSD" => (0..5).chain(50..62).collect(),
        "SINGLE" => vec![10],
        _ => (62..72).collect(),
    }
}

struct Fixture {
    data: BridleData,
    features: SequenceFeatures,
    /// true theta, lines x genes
    theta: Vec<Vec<f64>>,
    /// true technical offset per dataset
    offsets: HashMap<String, Vec<f64>>,
}

fn lineage(l: usize) -> Option<String> {
    (l % 10 != 9).then(|| format!("LIN{}", l % 4))
}

fn fixture() -> Fixture {
    let mut rs = NumpyRandomState::new(SEED);
    let x: Vec<Vec<f64>> = (0..G)
        .map(|_| (0..NF).map(|_| rs.next_gauss()).collect())
        .collect();
    let m: Vec<f64> = (0..G).map(|_| 20.0 + 2.0 * rs.next_gauss()).collect();
    let lin: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..G).map(|_| 0.5 * rs.next_gauss()).collect())
        .collect();
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
                .map(|g| {
                    let le = if lineage(l).is_some() {
                        lin[l % 4][g]
                    } else {
                        0.0
                    };
                    let uv: f64 = (0..3).map(|k| u[l][k] * v[g][k]).sum();
                    m[g] + le + uv + r[l][g]
                })
                .collect()
        })
        .collect();
    let mut offsets = HashMap::new();
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
        offsets.insert(d.to_owned(), a);
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

    // (ds, line) -> (values, plex)
    let mut profiles: Vec<(String, String, Vec<f64>, Option<String>)> = Vec::new();
    for d in ORDER {
        for (j, &l) in members(d).iter().enumerate() {
            let c = 0.2 * rs.next_gauss();
            let eps: Vec<f64> = (0..G).map(|_| 0.2 * rs.next_gauss()).collect();
            let mut miss: Vec<bool> = (0..G).map(|_| rs.next_f64() < 0.1).collect();
            let k = (d == "DSC").then_some(j / 5);
            if let Some(k) = k {
                miss.clone_from(&pmiss[k]);
            }
            let vals: Vec<f64> = (0..G)
                .map(|g| {
                    if miss[g] {
                        f64::NAN
                    } else {
                        theta[l][g] + offsets[d][g] + c + eps[g] + k.map_or(0.0, |k| p[k][g])
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
    let lineage_of =
        |line: &str| -> Option<String> { line[1..].parse::<usize>().ok().and_then(lineage) };
    let data = BridleData {
        datasets: profiles.iter().map(|p| p.0.clone()).collect(),
        lines: profiles.iter().map(|p| p.1.clone()).collect(),
        lineages: profiles.iter().map(|p| lineage_of(&p.1)).collect(),
        plexes: Some(profiles.iter().map(|p| p.3.clone()).collect()),
        genes: (0..G).map(|g| format!("G{g:03}")).collect(),
        values: profiles.iter().flat_map(|p| p.2.clone()).collect(),
    };
    let features = SequenceFeatures {
        names: (0..NF).map(|j| format!("f{j}")).collect(),
        values: x,
        n_missing: 0,
    };
    Fixture {
        data,
        features,
        theta,
        offsets,
    }
}

fn params(plex_mode: PlexMode) -> BridleParams {
    BridleParams {
        reference: "REF".to_owned(),
        rank: 4,
        sweeps: 30,
        plex_mode,
        // the prototype's configuration; the later defaults are off here
        ..BridleParams::legacy()
    }
}

fn run(fx: &Fixture, plex_mode: PlexMode) -> BridleResult {
    let ab = reference_abundance(&fx.data, "REF");
    let (design, _, abz) = build_design(&ab, Some(&fx.features));
    match bridle_fit(&fx.data, &design, &abz, &params(plex_mode)) {
        Ok(r) => r,
        Err(e) => panic!("bridle_fit failed: {e}"),
    }
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

fn row_index(fx: &Fixture, ds: &str, line: &str) -> usize {
    (0..fx.data.n_profiles())
        .find(|&i| fx.data.datasets[i] == ds && fx.data.lines[i] == line)
        .unwrap_or(usize::MAX)
}

#[test]
fn matches_frozen_reference() {
    let fx = fixture();
    let res = run(&fx, PlexMode::Inferred);
    let text = include_str!("fixtures/bridle_golden_expected.tsv");
    let mut n_cells = 0;
    let mut max_diff: f64 = 0.0;
    for line in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let want: f64 = f[4].parse().unwrap_or(f64::NAN);
        match f[0] {
            "cell" => {
                let i = row_index(&fx, f[1], f[2]);
                let g: usize = f[3][1..].parse().unwrap_or(usize::MAX);
                let got = res.corrected[i * G + g];
                max_diff = max_diff.max((got - want).abs());
                n_cells += 1;
            }
            "ds_mean" | "ds_sd" => {
                let vals: Vec<f64> = (0..fx.data.n_profiles())
                    .filter(|&i| fx.data.datasets[i] == f[1])
                    .flat_map(|i| res.corrected[i * G..(i + 1) * G].to_vec())
                    .filter(|v| v.is_finite())
                    .collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let got = if f[0] == "ds_mean" {
                    mean
                } else {
                    (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64)
                        .sqrt()
                };
                assert!(
                    (got - want).abs() < TOL,
                    "{} {}: rust {got} expected {want}",
                    f[0],
                    f[1]
                );
            }
            "n_sweeps" => assert_eq!(res.report.history.len(), want as usize),
            "hold_mse_last" => {
                let got = res.report.history.last().map_or(f64::NAN, |h| h.hold_mse);
                assert!(
                    (got - want).abs() < TOL,
                    "hold mse rust {got} expected {want}"
                );
            }
            "tauR" => {
                let got = res.report.history.last().map_or(f64::NAN, |h| h.tau_r);
                assert!((got - want).abs() < TOL, "tauR rust {got} expected {want}");
            }
            _ => {}
        }
    }
    eprintln!("BRIDLE golden: max |rust - expected| = {max_diff:.3e} over {n_cells} cells");
    assert!(n_cells > 1000, "expected cells missing ({n_cells})");
    assert!(
        max_diff < TOL,
        "max |rust - expected| = {max_diff} over {n_cells} cells"
    );
}

#[test]
fn nothing_is_imputed_and_no_gene_dropped() {
    let fx = fixture();
    let res = run(&fx, PlexMode::Inferred);
    assert_eq!(res.corrected.len(), fx.data.values.len());
    for (v, x) in res.corrected.iter().zip(&fx.data.values) {
        assert_eq!(v.is_finite(), x.is_finite());
    }
}

#[test]
fn recovers_technical_offsets() {
    let fx = fixture();
    let res = run(&fx, PlexMode::Inferred);
    for (s, name) in res.dataset_names.iter().enumerate() {
        let est = &res.offsets[s * G..(s + 1) * G];
        let truth = &fx.offsets[name];
        if name == "REF" {
            assert!(est.iter().all(|&a| a == 0.0));
            continue;
        }
        // offsets are identified up to a constant shared with c / theta
        let r = corr(est, truth);
        let floor = if name == "UNB" { 0.8 } else { 0.9 };
        assert!(r > floor, "{name}: corr(A_hat, A_true) = {r:.3}");
    }
}

#[test]
fn preserves_biology_and_single_line_signal() {
    let fx = fixture();
    let res = run(&fx, PlexMode::Inferred);
    // per-gene centring by the REF mean of the truth and of the output
    let ref_rows: Vec<usize> = (0..fx.data.n_profiles())
        .filter(|&i| fx.data.datasets[i] == "REF")
        .collect();
    let centre = |vals: &dyn Fn(usize, usize) -> f64, g: usize| -> f64 {
        let v: Vec<f64> = ref_rows
            .iter()
            .map(|&i| vals(i, g))
            .filter(|x| x.is_finite())
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let out = |i: usize, g: usize| res.corrected[i * G + g];
    let truth_row = |i: usize, g: usize| {
        let l: usize = fx.data.lines[i][1..].parse().unwrap_or(0);
        fx.theta[l][g]
    };
    let out_c: Vec<f64> = (0..G).map(|g| centre(&out, g)).collect();
    let tru_c: Vec<f64> = (0..G).map(|g| centre(&truth_row, g)).collect();
    for d in ["DSB", "DSC", "DSD", "SINGLE"] {
        let mut rs = Vec::new();
        for i in (0..fx.data.n_profiles()).filter(|&i| fx.data.datasets[i] == d) {
            let (a, b): (Vec<f64>, Vec<f64>) = (0..G)
                .filter(|&g| out(i, g).is_finite())
                .map(|g| (out(i, g) - out_c[g], truth_row(i, g) - tru_c[g]))
                .unzip();
            rs.push(corr(&a, &b));
        }
        rs.sort_by(f64::total_cmp);
        let med = rs[rs.len() / 2];
        assert!(
            med > 0.7,
            "{d}: median centred corr with true theta = {med:.3}"
        );
    }
    // the single-line dataset is cross-fitted: its own r is zero, A_out = f
    let s = res
        .dataset_names
        .iter()
        .position(|d| d == "SINGLE")
        .unwrap_or(usize::MAX);
    let rep = &res.report.datasets[s];
    assert!(rep.cross_fitted && rep.n_anchor_samples == 1);
    let i = row_index(&fx, "SINGLE", "L10");
    for g in (0..G).filter(|&g| fx.data.values[i * G + g].is_finite()) {
        let a_out = fx.data.values[i * G + g] - res.corrected[i * G + g] - res.sample_loading[i];
        assert!((a_out - res.feature_offsets[s * G + g]).abs() < 1e-9);
    }
}

#[test]
fn explicit_plexes_match_inferred_on_clean_fixture() {
    let fx = fixture();
    let a = run(&fx, PlexMode::Inferred);
    let b = run(&fx, PlexMode::Explicit);
    assert_eq!(a.report.n_plexes, 6);
    assert_eq!(b.report.n_plexes, 6);
    let max = a
        .corrected
        .iter()
        .zip(&b.corrected)
        .filter(|(x, _)| x.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(max < 1e-9, "explicit vs inferred plexes differ by {max}");
}

#[test]
fn anchor_scale_is_opt_in_and_only_touches_well_anchored_datasets() {
    let fx = fixture();
    // off by default: `matches_frozen_reference` checks the golden numbers
    assert!(!params(PlexMode::Inferred).anchor_scale);
    let off = run(&fx, PlexMode::Inferred);
    assert!(off.report.anchor_scale.is_empty());
    let ab = reference_abundance(&fx.data, "REF");
    let (design, _, abz) = build_design(&ab, Some(&fx.features));
    let p = BridleParams {
        anchor_scale: true,
        ..params(PlexMode::Inferred)
    };
    let on = match bridle_fit(&fx.data, &design, &abz, &p) {
        Ok(r) => r,
        Err(e) => panic!("bridle_fit failed: {e}"),
    };
    // DSB shares 20 lines with REF (the threshold) and has no compression
    for r in &on.report.anchor_scale {
        if r.name == "DSB" {
            assert!(
                r.applied && r.n_shared == 20 && (r.b - 1.0).abs() < 0.1,
                "{r:?}"
            );
        } else {
            assert!(!r.applied && r.b == 1.0, "{r:?}");
        }
    }
    for (i, ds) in fx.data.datasets.iter().enumerate() {
        for g in 0..G {
            let (x, y) = (off.corrected[i * G + g], on.corrected[i * G + g]);
            assert_eq!(x.is_finite(), y.is_finite());
            if ds != "DSB" {
                assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
            }
        }
    }
}
