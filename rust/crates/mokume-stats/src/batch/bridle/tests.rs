//! Unit tests for BRIDLE: ridge solves, EM updates, cross-fit, plexes and
//! missing-value handling on small hand-built collections. The end-to-end
//! comparison with the Python prototype lives in `tests/bridle_golden.rs`.

use super::*;

/// Tiny collection: REF (6 lines), B (4 of the same lines, +1.0 offset),
/// S (single line L0, +0.5 offset), U (unanchored, 3 own lines). 60 genes.
fn toy(missing_every: usize) -> BridleData {
    let g_n = 60;
    let mut rs = NumpyRandomState::new(11);
    let base: Vec<f64> = (0..g_n).map(|_| 20.0 + 2.0 * rs.next_gauss()).collect();
    let line_eff: Vec<Vec<f64>> = (0..9)
        .map(|_| (0..g_n).map(|_| 0.5 * rs.next_gauss()).collect())
        .collect();
    let plan: [(&str, Vec<usize>, f64); 4] = [
        ("REF", (0..6).collect(), 0.0),
        ("B", (0..4).collect(), 1.0),
        ("S", vec![0], 0.5),
        ("U", (6..9).collect(), -0.7),
    ];
    let mut data = BridleData {
        datasets: Vec::new(),
        lines: Vec::new(),
        lineages: Vec::new(),
        plexes: None,
        genes: (0..g_n).map(|g| format!("G{g:02}")).collect(),
        values: Vec::new(),
    };
    let mut cell = 0;
    for (d, lines, off) in plan {
        for l in lines {
            data.datasets.push(d.to_owned());
            data.lines.push(format!("L{l}"));
            data.lineages
                .push(Some(if l % 2 == 0 { "even" } else { "odd" }.to_owned()));
            for g in 0..g_n {
                cell += 1;
                let v = if missing_every > 0 && cell % missing_every == 0 {
                    f64::NAN
                } else {
                    base[g] + line_eff[l][g] + off + 0.05 * rs.next_gauss()
                };
                data.values.push(v);
            }
        }
    }
    data
}

fn toy_params() -> BridleParams {
    BridleParams {
        reference: "REF".to_owned(),
        rank: 2,
        sweeps: 15,
        min_f_genes: 10,
        // the core tests check v = y - A - c - P; kept-c is tested separately
        keep_sample_loading: false,
        ..BridleParams::default()
    }
}

fn fit(data: &BridleData, params: &BridleParams) -> BridleResult {
    let ab = reference_abundance(data, &params.reference);
    let (design, _, abz) = build_design(&ab, None);
    match bridle_fit(data, &design, &abz, params) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    }
}

#[test]
fn weighted_ridge_rows_solves_normal_equations() {
    // factors x_j (rank 2), weights w_j, response wz_j = w_j * z_j
    let factors = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let w = [1.0, 2.0, 0.0];
    let z = [3.0, -1.0, 100.0];
    let wz: Vec<f64> = w.iter().zip(&z).map(|(a, b)| a * b).collect();
    let b = weighted_ridge_rows(&w, &wz, &factors, 2, 0.5);
    // (diag(1, 2) + 0.5 I) b = (3, -2)
    assert!((b[0] - 2.0).abs() < 1e-12 && (b[1] + 0.8).abs() < 1e-12);
}

#[test]
fn missing_values_stay_missing_and_no_gene_is_dropped() {
    let mut data = toy(7);
    // a gene observed in a single profile keeps its single value
    let g_n = data.genes.len();
    for i in 1..data.n_profiles() {
        data.values[i * g_n + 5] = f64::NAN;
    }
    let res = fit(&data, &toy_params());
    assert_eq!(res.corrected.len(), data.values.len());
    for (v, x) in res.corrected.iter().zip(&data.values) {
        assert_eq!(v.is_finite(), x.is_finite());
    }
    assert!(res.corrected[5].is_finite());
    assert_eq!(res.report.n_genes, g_n);
}

#[test]
fn em_updates_stay_finite_and_monitor_improves() {
    let res = fit(
        &toy(0),
        &BridleParams {
            holdout_frac: 0.05,
            // cold start: the graph prior already places A near its optimum
            graph_prior: false,
            ..toy_params()
        },
    );
    let h = &res.report.history;
    assert!(h.iter().all(|s| s.tau_r.is_finite() && s.tau_r > 0.0));
    assert!(h.last().map_or(f64::NAN, |s| s.hold_mse) < h[0].hold_mse);
    for d in &res.report.datasets {
        assert!(
            d.tau_s >= 0.02 - 1e-12 && d.tau_s <= 2.0 + 1e-12,
            "{}: tau_s {}",
            d.name,
            d.tau_s
        );
        assert!(d.sig2 > 0.0);
    }
}

#[test]
fn offsets_are_recovered_relative_to_reference() {
    let data = toy(0);
    let res = fit(&data, &toy_params());
    let g_n = data.genes.len();
    let mean_offset = |name: &str| {
        let s = res
            .dataset_names
            .iter()
            .position(|d| d == name)
            .unwrap_or(0);
        res.offsets[s * g_n..(s + 1) * g_n].iter().sum::<f64>() / g_n as f64
    };
    assert_eq!(mean_offset("REF"), 0.0);
    // the applied offset y - v = A_out + c, relative to REF's (a constant can
    // move between m_g and every profile's c), recovers B's +1.0. A_out is
    // cross-fitted: B has 4 anchor samples < cf_max_nb.
    let applied = |name: &str| {
        let rows: Vec<usize> = (0..data.n_profiles())
            .filter(|&i| data.datasets[i] == name)
            .collect();
        rows.iter()
            .flat_map(|&i| (0..g_n).map(move |g| (i, g)))
            .map(|(i, g)| data.values[i * g_n + g] - res.corrected[i * g_n + g])
            .sum::<f64>()
            / (rows.len() * g_n) as f64
    };
    let rel = applied("B") - applied("REF");
    assert!((rel - 1.0).abs() < 0.05, "B offset relative to REF {rel}");
    let b_rows: Vec<usize> = (0..data.n_profiles())
        .filter(|&i| data.datasets[i] == "B")
        .collect();
    // after correction, B's copy of a line agrees with REF's copy
    for i in b_rows {
        let j = (0..data.n_profiles())
            .find(|&j| data.datasets[j] == "REF" && data.lines[j] == data.lines[i])
            .unwrap_or(0);
        let d: f64 = (0..g_n)
            .map(|g| (res.corrected[i * g_n + g] - res.corrected[j * g_n + g]).abs())
            .sum::<f64>()
            / g_n as f64;
        assert!(d < 0.25, "line {} |B - REF| = {d}", data.lines[i]);
    }
}

#[test]
fn cross_fit_keeps_single_line_signal() {
    let data = toy(0);
    let res = fit(&data, &toy_params());
    let g_n = data.genes.len();
    let s = res.dataset_names.iter().position(|d| d == "S").unwrap_or(0);
    let rep = &res.report.datasets[s];
    assert!(rep.anchored && rep.cross_fitted && rep.n_anchor_samples == 1);
    // one anchor sample -> the fold-excluded r is 0: A_out = f exactly
    let i = (0..data.n_profiles())
        .find(|&i| data.datasets[i] == "S")
        .unwrap_or(0);
    for g in 0..g_n {
        let a_out = data.values[i * g_n + g] - res.corrected[i * g_n + g] - res.sample_loading[i];
        assert!((a_out - res.feature_offsets[s * g_n + g]).abs() < 1e-9);
    }
    // B has 4 anchor samples (< cf_max_nb) and is cross-fitted too; REF is not
    let b = res.dataset_names.iter().position(|d| d == "B").unwrap_or(0);
    assert!(res.report.datasets[b].cross_fitted);
    let r = res
        .dataset_names
        .iter()
        .position(|d| d == "REF")
        .unwrap_or(0);
    assert!(!res.report.datasets[r].cross_fitted && !res.report.datasets[r].anchored);
    // unanchored dataset: A = f (no residual offset)
    let u = res.dataset_names.iter().position(|d| d == "U").unwrap_or(0);
    assert!(!res.report.datasets[u].anchored);
    for g in 0..g_n {
        assert_eq!(res.offsets[u * g_n + g], res.feature_offsets[u * g_n + g]);
    }
}

#[test]
fn explicit_plex_ids_are_scoped_per_dataset() {
    let mut data = toy(0);
    let plex: Vec<Option<String>> = (0..data.n_profiles())
        .map(|i| match data.datasets[i].as_str() {
            "REF" => Some(format!("m{}", i % 2)),
            "B" => Some("m0".to_owned()), // a single plex in B -> no plex effect
            _ => None,
        })
        .collect();
    data.plexes = Some(plex);
    let res = fit(
        &data,
        &BridleParams {
            plex_mode: PlexMode::Explicit,
            ..toy_params()
        },
    );
    assert_eq!(res.report.n_plexes, 2);
    let r = res
        .dataset_names
        .iter()
        .position(|d| d == "REF")
        .unwrap_or(0);
    assert_eq!(res.report.datasets[r].n_plexes, 2);
    assert!(res
        .profile_plex
        .iter()
        .enumerate()
        .all(|(i, p)| p.is_some() == (data.datasets[i] == "REF")));
}

#[test]
fn deterministic_across_thread_counts() {
    let data = toy(9);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
        match pool {
            Ok(pool) => pool.install(|| fit(&data, &toy_params())),
            Err(e) => panic!("pool: {e}"),
        }
    };
    let a = run(1);
    let b = run(4);
    for (x, y) in a.corrected.iter().zip(&b.corrected) {
        assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
    }
}

#[test]
fn input_errors_are_reported() {
    let data = toy(0);
    let ab = reference_abundance(&data, "REF");
    let (design, _, abz) = build_design(&ab, None);
    let bad_ref = BridleParams {
        reference: "NOPE".to_owned(),
        ..toy_params()
    };
    assert!(bridle_fit(&data, &design, &abz, &bad_ref).is_err());
    let mut dup = data.clone();
    dup.lines[1] = dup.lines[0].clone();
    assert!(bridle_fit(&dup, &design, &abz, &toy_params()).is_err());
    assert!(bridle_fit(&data, &design[1..], &abz[1..], &toy_params()).is_err());
}

/// REF and B share 30 lines, B is compressed by 0.7 and shifted by +1; B has 5
/// own lines. Gene 0 is never observed by B on a shared line, gene 1 never by
/// REF. C shares only 10 lines with REF (below the threshold).
fn compressed(b_true: f64) -> BridleData {
    let g_n = 80;
    let mut rs = NumpyRandomState::new(5);
    let base: Vec<f64> = (0..g_n).map(|_| 20.0 + 2.0 * rs.next_gauss()).collect();
    let z: Vec<Vec<f64>> = (0..35)
        .map(|_| (0..g_n).map(|_| rs.next_gauss()).collect())
        .collect();
    let mut data = BridleData {
        datasets: Vec::new(),
        lines: Vec::new(),
        lineages: Vec::new(),
        plexes: None,
        genes: (0..g_n).map(|g| format!("G{g:02}")).collect(),
        values: Vec::new(),
    };
    let plan: [(&str, Vec<usize>); 3] = [
        ("B", (0..35).collect()),
        ("C", (20..30).collect()),
        ("REF", (0..30).collect()),
    ];
    for (d, lines) in plan {
        for l in lines {
            data.datasets.push(d.to_owned());
            data.lines.push(format!("L{l:02}"));
            data.lineages.push(None);
            for g in 0..g_n {
                let v = match d {
                    "REF" if g == 1 => f64::NAN,
                    "REF" => base[g] + z[l][g],
                    "B" if g == 0 && l < 30 => f64::NAN,
                    "B" => base[g] + 1.0 + b_true * z[l][g] + 0.05 * rs.next_gauss(),
                    _ => base[g] + 0.5 * z[l][g],
                };
                data.values.push(v);
            }
        }
    }
    data
}

#[test]
fn anchor_scale_recovers_compression() {
    let data = compressed(0.7);
    let g_n = data.genes.len();
    let mut v = data.values.clone();
    let rep = anchor_scale(&data, &mut v, "REF", 20);
    let by = |name: &str| rep.iter().find(|r| r.name == name).cloned();
    let b = by("B").map_or(f64::NAN, |r| r.b);
    assert!((b - 0.7).abs() < 0.02, "b = {b}");
    assert!(by("B").is_some_and(|r| r.applied && r.n_shared == 30));
    // below the threshold and the reference: b = 1, values untouched
    for name in ["C", "REF"] {
        let r = by(name).map(|r| (r.b, r.applied));
        assert_eq!(r, Some((1.0, false)), "{name}");
    }
    for i in (0..data.n_profiles()).filter(|&i| data.datasets[i] != "B") {
        for g in 0..g_n {
            let (x, y) = (data.values[i * g_n + g], v[i * g_n + g]);
            assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        }
    }
    // B's copy of a shared line now matches REF's (offset and spread)
    let row = |d: &str, l: &str| {
        (0..data.n_profiles())
            .find(|&i| data.datasets[i] == d && data.lines[i] == l)
            .unwrap_or(usize::MAX)
    };
    let (i, j) = (row("B", "L03"), row("REF", "L03"));
    for g in 2..g_n {
        let d = (v[i * g_n + g] - v[j * g_n + g]).abs();
        assert!(d < 0.25, "gene {g}: |B - REF| = {d}");
    }
    // gene 0 (no B anchor mean): scaled around B's own mean over its 5 lines
    let own: Vec<usize> = (30..35).map(|l| row("B", &format!("L{l}"))).collect();
    let m = own.iter().map(|&i| data.values[i * g_n]).sum::<f64>() / 5.0;
    for &i in &own {
        let want = m + (data.values[i * g_n] - m) / b;
        assert!((v[i * g_n] - want).abs() < 1e-12);
    }
    // gene 1 (no REF anchor mean): scaled around B's anchor mean
    let anc: Vec<usize> = (0..30).map(|l| row("B", &format!("L{l:02}"))).collect();
    let am = anc.iter().map(|&i| data.values[i * g_n + 1]).sum::<f64>() / 30.0;
    for &i in anc.iter().chain(&own) {
        let want = am + (data.values[i * g_n + 1] - am) / b;
        assert!((v[i * g_n + 1] - want).abs() < 1e-12);
    }
}

#[test]
fn anchor_scale_loses_no_value() {
    let data = compressed(0.7);
    let mut v = data.values.clone();
    anchor_scale(&data, &mut v, "REF", 20);
    assert_eq!(v.len(), data.values.len());
    let n_in = data.values.iter().filter(|x| x.is_finite()).count();
    assert_eq!(v.iter().filter(|x| x.is_finite()).count(), n_in);
    for (x, y) in data.values.iter().zip(&v) {
        assert_eq!(x.is_finite(), y.is_finite());
    }
}

#[test]
fn anchor_scale_without_passing_genes_keeps_b_one() {
    // B is pure noise: no gene correlates with REF
    let mut data = compressed(0.7);
    let mut rs = NumpyRandomState::new(9);
    for i in (0..data.n_profiles()).filter(|&i| data.datasets[i] == "B") {
        let g_n = data.genes.len();
        for x in &mut data.values[i * g_n..(i + 1) * g_n] {
            if x.is_finite() {
                *x = 20.0 + 1e-3 * rs.next_gauss();
            }
        }
    }
    let mut v = data.values.clone();
    let rep = anchor_scale(&data, &mut v, "REF", 20);
    let b = rep.iter().find(|r| r.name == "B").map(|r| (r.b, r.applied));
    assert_eq!(b, Some((1.0, false)));
    assert!(data
        .values
        .iter()
        .zip(&v)
        .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())));
}

#[test]
fn anchor_scale_in_fit_is_opt_in_and_keeps_every_value() {
    let data = toy(7);
    let off = fit(&data, &toy_params());
    assert!(!toy_params().anchor_scale && off.report.anchor_scale.is_empty());
    // default threshold: B has 4 shared anchors < 20 -> b = 1, output identical
    let on = fit(
        &data,
        &BridleParams {
            anchor_scale: true,
            ..toy_params()
        },
    );
    assert!(on.report.anchor_scale.iter().all(|r| r.b == 1.0));
    for (x, y) in off.corrected.iter().zip(&on.corrected) {
        assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
    }
    // compressed collection: B (30 shared anchors) is rescaled, C (10) and REF
    // are untouched, and every observed cell keeps a value
    let data = compressed(0.7);
    let params = BridleParams {
        rank: 2,
        ..toy_params()
    };
    let off = fit(&data, &params);
    let on = fit(
        &data,
        &BridleParams {
            anchor_scale: true,
            ..params
        },
    );
    let b = on.report.anchor_scale.iter().find(|r| r.name == "B");
    assert!(
        b.is_some_and(|r| r.applied && (r.b - 0.7).abs() < 0.1),
        "{b:?}"
    );
    let g_n = data.genes.len();
    for i in 0..data.n_profiles() {
        for g in 0..g_n {
            let (x, y, v) = (
                off.corrected[i * g_n + g],
                on.corrected[i * g_n + g],
                data.values[i * g_n + g],
            );
            assert_eq!(y.is_finite(), v.is_finite());
            if data.datasets[i] != "B" {
                assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
            }
        }
    }
}

#[test]
fn default_convergence_is_output_change_with_long_runs() {
    let p = BridleParams::default();
    assert_eq!(p.stop_rule, StopRule::OutputChange);
    assert_eq!((p.sweeps, p.min_sweeps), (400, 200));
    assert!((p.converge_tol - 1e-5).abs() < 1e-18);
    // the previous rule stays reachable
    let m = BridleParams::default().monitor_stop();
    assert_eq!(m.stop_rule, StopRule::MonitorMse);
    assert_eq!((m.sweeps, m.min_sweeps), (60, 8));
}

#[test]
fn output_change_rule_stops_after_min_sweeps_below_tol() {
    let data = toy(7);
    let base = BridleParams {
        sweeps: 12,
        min_sweeps: 5,
        ..toy_params()
    };
    // a huge tolerance stops at the first eligible sweep (5 sweeps run)
    let res = fit(
        &data,
        &BridleParams {
            converge_tol: 1e9,
            ..base.clone()
        },
    );
    assert!(res.report.converged);
    assert_eq!(res.report.history.len(), 5);
    let h = &res.report.history;
    assert!(h[0].out_change.is_nan());
    assert!(h[1..]
        .iter()
        .all(|s| s.out_change.is_finite() && s.out_change >= 0.0));
    // tolerance 0 never fires: all sweeps run, not converged
    let res = fit(
        &data,
        &BridleParams {
            converge_tol: 0.0,
            ..base.clone()
        },
    );
    assert!(!res.report.converged);
    assert_eq!(res.report.history.len(), 12);
    // the change shrinks as the fit settles
    let h = &res.report.history;
    assert!(h[11].out_change < h[1].out_change);
    // StopRule::Never ignores the tolerance
    let res = fit(
        &data,
        &BridleParams {
            converge_tol: 1e9,
            stop_rule: StopRule::Never,
            ..base
        },
    );
    assert!(!res.report.converged && res.report.history.len() == 12);
}

#[test]
fn out_change_is_the_mean_change_of_the_output() {
    // two fits of 3 and 4 sweeps share their first 3 sweeps; the 4th sweep's
    // out_change is the mean |v4 - v3| of the (uncross-fitted) output on a
    // collection without cross-fitted datasets (REF + U, unanchored)
    let mut data = toy(0);
    let keep: Vec<usize> = (0..data.n_profiles())
        .filter(|&i| data.datasets[i] == "REF" || data.datasets[i] == "U")
        .collect();
    let g_n = data.genes.len();
    data = BridleData {
        datasets: keep.iter().map(|&i| data.datasets[i].clone()).collect(),
        lines: keep.iter().map(|&i| data.lines[i].clone()).collect(),
        lineages: keep.iter().map(|&i| data.lineages[i].clone()).collect(),
        plexes: None,
        genes: data.genes.clone(),
        values: keep
            .iter()
            .flat_map(|&i| data.values[i * g_n..(i + 1) * g_n].to_vec())
            .collect(),
    };
    let p = |sweeps| BridleParams {
        sweeps,
        stop_rule: StopRule::Never,
        holdout_frac: 0.0,
        ..toy_params()
    };
    let a = fit(&data, &p(3));
    let b = fit(&data, &p(4));
    let (s, c) = a
        .corrected
        .iter()
        .zip(&b.corrected)
        .filter(|(x, _)| x.is_finite())
        .fold((0.0, 0_usize), |(s, c), (x, y)| (s + (x - y).abs(), c + 1));
    let want = s / c as f64;
    let got = b.report.history[3].out_change;
    assert!((got - want).abs() < 1e-12, "out_change {got} vs {want}");
}

#[test]
fn graph_prior_is_default_and_covers_linked_datasets_only() {
    assert!(BridleParams::default().graph_prior && !BridleParams::legacy().graph_prior);
    let data = toy(0);
    let g_n = data.genes.len();
    let res = fit(&data, &toy_params());
    let genes = |name: &str| {
        res.report
            .datasets
            .iter()
            .find(|d| d.name == name)
            .map_or(usize::MAX, |d| d.graph_prior_genes)
    };
    // B shares 4 lines with REF (>= 3): linked on every gene; S (1 line) and
    // U (3 own lines, < 5 profiles) get neither a graph nor a centring prior
    assert_eq!(
        (genes("REF"), genes("B"), genes("S"), genes("U")),
        (0, g_n, 0, 0)
    );
    // B's offsets start at, and are shrunk towards, the graph prior (+1.0)
    let b = res.dataset_names.iter().position(|d| d == "B").unwrap_or(0);
    let mean_a = res.offsets[b * g_n..(b + 1) * g_n].iter().sum::<f64>() / g_n as f64;
    let mean_c_b: f64 = (0..data.n_profiles())
        .filter(|&i| data.datasets[i] == "B")
        .map(|i| res.sample_loading[i])
        .sum::<f64>()
        / 4.0;
    assert!(
        (mean_a + mean_c_b - 1.0).abs() < 0.1,
        "A_B {mean_a} c_B {mean_c_b}"
    );
    let off = fit(
        &data,
        &BridleParams {
            graph_prior: false,
            ..toy_params()
        },
    );
    assert!(off.report.datasets.iter().all(|d| d.graph_prior_genes == 0));
    // the prior changes the fit
    assert!(off
        .corrected
        .iter()
        .zip(&res.corrected)
        .any(|(x, y)| x.is_finite() && (x - y).abs() > 1e-9));
}

#[test]
fn sample_loading_is_kept_in_the_output_by_default() {
    assert!(BridleParams::default().keep_sample_loading);
    assert!(!BridleParams::legacy().keep_sample_loading);
    let data = toy(7);
    let g_n = data.genes.len();
    let removed = fit(&data, &toy_params());
    let kept = fit(
        &data,
        &BridleParams {
            keep_sample_loading: true,
            ..toy_params()
        },
    );
    // same fit; the output differs by exactly c on every observed cell
    assert_eq!(removed.sample_loading, kept.sample_loading);
    assert!(kept.sample_loading.iter().any(|c| c.abs() > 1e-6));
    for i in 0..data.n_profiles() {
        for g in 0..g_n {
            let (x, y) = (removed.corrected[i * g_n + g], kept.corrected[i * g_n + g]);
            assert_eq!(x.is_finite(), y.is_finite());
            if x.is_finite() {
                assert!((y - x - kept.sample_loading[i]).abs() < 1e-12);
            }
        }
    }
}
