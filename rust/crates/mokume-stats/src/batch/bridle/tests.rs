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
