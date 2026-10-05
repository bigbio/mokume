//! Tests of the per-dataset value report: known answers on a synthetic
//! collection and parity with the B4 prototype definitions
//! (`rust/scripts/bridle_value_reference.py`, stored in
//! `tests/fixtures/bridle_value_expected.tsv`).

use std::collections::HashMap;

use super::super::{bridle_fit, build_design, reference_abundance, BridleParams, NumpyRandomState};
use super::*;

const SEED: u32 = 20_261_004;
const G: usize = 300;
const NL: usize = 20;
const MISS: f64 = 0.1;
/// Max |rust - python| (both float64; summation order differs).
const PY_TOL: f64 = 1e-9;

struct Study {
    name: &'static str,
    lines: Vec<usize>,
    off: f64,
    sd: f64,
    var: f64,
}

fn plan() -> Vec<Study> {
    let s = |name, lines: Vec<usize>, off, sd, var| Study {
        name,
        lines,
        off,
        sd,
        var,
    };
    vec![
        s("REF", (0..16).collect(), 0.0, 0.2, 0.04),
        s("B", (5..19).collect(), 1.0, 0.2, 0.04),
        s("C", (0..13).collect(), -0.5, 0.3, 0.09),
        // copy of B (+0.01 noise): redundant
        s("DUP", (5..11).collect(), 0.7, 0.0, 0.04),
        // genes reversed: distorted abundance shape, wrong identity
        s("BAD", (0..5).collect(), 0.3, 0.2, 1.0),
        // the only source of L19: no anchors
        s("SINGLE", vec![19], 0.2, 0.2, 0.04),
    ]
}

/// Raw input, corrected values (truth + noise), per-cell variance
/// (`datasets x genes`, sorted names) and the RNA reference; same draws as
/// the Python generator.
struct Fixture {
    data: BridleData,
    corrected: Vec<f64>,
    noise_var: Vec<f64>,
    rna: RnaReference,
}

fn fixture() -> Fixture {
    let mut rs = NumpyRandomState::new(SEED);
    let m: Vec<f64> = (0..G).map(|_| 20.0 + 2.0 * rs.next_gauss()).collect();
    let e: Vec<Vec<f64>> = (0..NL)
        .map(|_| (0..G).map(|_| 0.6 * rs.next_gauss()).collect())
        .collect();
    let mut rna = vec![0.0; NL * G];
    for l in 0..NL {
        for g in 0..G {
            rna[l * G + g] = e[l][g] + 0.5 * rs.next_gauss();
        }
    }
    let truth = |l: usize, g: usize| m[g] + e[l][g];
    let mut data = BridleData {
        datasets: Vec::new(),
        lines: Vec::new(),
        lineages: Vec::new(),
        plexes: None,
        genes: (0..G).map(|g| format!("G{g:03}")).collect(),
        values: Vec::new(),
    };
    let mut corrected = Vec::new();
    let mut b_vals: HashMap<(usize, usize), f64> = HashMap::new();
    let studies = plan();
    let mut var_of: HashMap<&str, f64> = HashMap::new();
    for st in &studies {
        var_of.insert(st.name, st.var);
        let off_g: Vec<f64> = (0..G).map(|_| st.off + 0.3 * rs.next_gauss()).collect();
        for &l in &st.lines {
            data.datasets.push(st.name.to_owned());
            data.lines.push(format!("L{l:02}"));
            data.lineages.push(None);
            for (g, &off) in off_g.iter().enumerate() {
                let z = rs.next_gauss();
                let miss = rs.next_f64() < MISS;
                let v = match st.name {
                    "DUP" => b_vals.get(&(l, g)).copied().unwrap_or(f64::NAN) + 0.01 * z,
                    "BAD" => truth(l, G - 1 - g) + st.sd * z,
                    _ => truth(l, g) + st.sd * z,
                };
                let v = if miss { f64::NAN } else { v };
                if st.name == "B" {
                    b_vals.insert((l, g), v);
                }
                data.values.push(v + off);
                corrected.push(v);
            }
        }
    }
    let mut names: Vec<&str> = studies.iter().map(|s| s.name).collect();
    names.sort_unstable();
    let noise_var = names
        .iter()
        .flat_map(|n| std::iter::repeat_n(var_of[n], G))
        .collect();
    Fixture {
        data,
        corrected,
        noise_var,
        rna: RnaReference {
            lines: (0..NL).map(|l| format!("L{l:02}")).collect(),
            values: rna,
        },
    }
}

/// Report with the opt-in identity check on.
fn report(fx: &Fixture, rna: bool) -> ValueReport {
    report_with(
        fx,
        rna,
        &ValueParams {
            identity: true,
            ..ValueParams::default()
        },
    )
}

fn report_with(fx: &Fixture, rna: bool, params: &ValueParams) -> ValueReport {
    let n = fx.data.n_profiles();
    let s_n = fx.noise_var.len() / G;
    let view = FitView {
        corrected: &fx.corrected,
        noise_var: &fx.noise_var,
        sample_loading: &vec![0.0; n],
        offsets: &vec![0.0; s_n * G],
        delta: vec![(f64::NAN, f64::NAN); s_n],
    };
    value_report(&fx.data, &view, rna.then_some(&fx.rna), params)
}

fn ds<'a>(r: &'a ValueReport, name: &str) -> &'a DatasetValue {
    match r.datasets.iter().find(|d| d.name == name) {
        Some(d) => d,
        None => panic!("no dataset {name}"),
    }
}

#[test]
fn coverage_counts_unique_shared_and_bridge_lines() {
    let r = report(&fixture(), false);
    let b = ds(&r, "B");
    // B: L16-L18 only in B, L13-L15 shared with REF only (bridges)
    assert_eq!(
        (b.n_lines, b.uniq_lines, b.anchor_lines, b.bridge_lines),
        (14, 3, 11, 3)
    );
    // partners with >= 3 shared lines: REF (11), C (8), DUP (6); BAD shares none
    assert_eq!((b.anchor_partners, b.partners_any), (3, 3));
    assert_eq!(ds(&r, "BAD").anchor_partners, 2); // REF, C (5 lines each)
    let single = ds(&r, "SINGLE");
    assert_eq!((single.uniq_lines, single.anchor_lines), (1, 0));
    assert_eq!(single.uniq_gene_cells, single.n_cells);
    assert_eq!(ds(&r, "DUP").uniq_gene_cells, 0);
}

#[test]
fn single_line_study_without_anchors_is_flagged() {
    let r = report(&fixture(), false);
    let single = ds(&r, "SINGLE");
    assert!(single.no_anchors_cannot_audit);
    assert!(single.id_self_r.is_nan() && single.excess_var.is_nan());
    assert!(single.marg_shift.is_nan() && single.agree_med.is_nan());
    assert!(r
        .datasets
        .iter()
        .filter(|d| d.name != "SINGLE")
        .all(|d| !d.no_anchors_cannot_audit));
}

#[test]
fn distorted_study_has_low_abundance_rho_identity_and_high_excess() {
    let r = report(&fixture(), true);
    let bad = ds(&r, "BAD");
    for d in r.datasets.iter().filter(|d| d.name != "BAD") {
        assert!(d.abund_rho > 0.8, "{}: {}", d.name, d.abund_rho);
        assert!(bad.abund_rho < d.abund_rho - 0.5);
        if !d.no_anchors_cannot_audit {
            assert!(d.id_self_r > 0.7 && d.id_rank == 1.0, "{}", d.name);
            assert!(bad.excess_var > 10.0 * d.excess_var);
        }
        assert_eq!(d.id_rna_top1, 1.0, "{}", d.name);
    }
    assert!(bad.id_self_r < 0.1 && bad.id_rank > 5.0);
    assert!(bad.id_best_is_self < 0.5 && bad.id_rna_top1 == 0.0);
    // the profiles' best match is never their own line
    assert!(r
        .profiles
        .iter()
        .filter(|p| p.dataset == "BAD")
        .all(|p| p.best_line != p.line));
}

#[test]
fn duplicated_study_is_redundant() {
    let r = report(&fixture(), false);
    let dup = ds(&r, "DUP");
    // every DUP line has >= 3 other sources and its best match is B's copy
    assert_eq!(dup.redund_ge3, 1.0);
    assert!(dup.id_r_max_any > 0.99);
    assert!(dup.agree_med < ds(&r, "REF").agree_med);
    // a duplicate barely moves the consensus
    assert!(dup.marg_shift < ds(&r, "REF").marg_shift);
    // rna identity is skipped without a reference
    assert!(r.datasets.iter().all(|d| d.id_rna_self.is_nan()));
}

#[test]
fn identity_check_is_opt_in_and_leaves_the_cheap_metrics_unchanged() {
    let fx = fixture();
    assert!(!ValueParams::default().identity);
    let cheap = report_with(&fx, false, &ValueParams::default());
    let full = report(&fx, false);
    for (c, f) in cheap.datasets.iter().zip(&full.datasets) {
        assert!(c.id_self_r.is_nan() && c.id_rank.is_nan() && c.id_top1.is_nan());
        assert!(c.id_best_is_self.is_nan() && c.id_r_max_any.is_nan());
        assert!(c.id_r_med_any.is_nan() && c.id_rna_self.is_nan());
        let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
        for (a, b) in [
            (c.abund_rho, f.abund_rho),
            (c.excess_var, f.excess_var),
            (c.agree_med, f.agree_med),
            (c.marg_shift, f.marg_shift),
            (c.redund_ge3, f.redund_ge3),
        ] {
            assert!(same(a, b), "{}: {a} vs {b}", c.name);
        }
        assert_eq!(c.no_anchors_cannot_audit, f.no_anchors_cannot_audit);
    }
    assert!(cheap.profiles.iter().all(|p| p.best_line.is_empty()));
    // a reference alone still gives the reference identity
    let rna_only = report_with(&fx, true, &ValueParams::default());
    let full_rna = report(&fx, true);
    for (a, b) in rna_only.datasets.iter().zip(&full_rna.datasets) {
        assert_eq!(a.id_rna_top1.to_bits(), b.id_rna_top1.to_bits());
        assert!(a.id_self_r.is_nan());
    }
}

#[test]
fn nnls_matches_known_solution_and_clips_at_zero() {
    // d = (1, 2, 0): rows a+b, a+c, b+c, with a negative target pulling c down
    let x = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let d = nnls(&x, &[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0], 3);
    assert!((d[0] - 1.0).abs() < 1e-9 && (d[1] - 2.0).abs() < 1e-9 && d[2].abs() < 1e-9);
    let d = nnls(&x, &[3.0, 0.0, 1.0], &[1.0, 1.0, 1.0], 3);
    assert!(d.iter().all(|&v| v >= 0.0));
    assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![3.5, 1.0, 3.5, 2.0]);
}

/// `(kind, dataset, line, metric) -> value`.
fn expected() -> HashMap<(String, String, String, String), String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bridle_value_expected.tsv"
    );
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => panic!("{path}: {e}"),
    };
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (
                (f[0].into(), f[1].into(), f[2].into(), f[3].into()),
                f[4].to_owned(),
            )
        })
        .collect()
}

fn close(name: &str, got: f64, want: &str) {
    let want: f64 = match want.parse() {
        Ok(v) => v,
        Err(e) => panic!("{name}: bad expected value {want}: {e}"),
    };
    assert!(
        (got.is_nan() && want.is_nan()) || (got - want).abs() <= PY_TOL * want.abs().max(1.0),
        "{name}: rust {got} vs python {want}"
    );
}

#[test]
fn matches_the_b4_prototype_definitions() {
    let r = report(&fixture(), true);
    let exp = expected();
    let mut checked = 0;
    for d in &r.datasets {
        let metrics: [(&str, f64); 30] = [
            ("n_lines", d.n_lines as f64),
            ("n_cells", d.n_cells as f64),
            ("n_genes", d.n_genes as f64),
            ("genes_per_profile", d.genes_per_profile),
            ("uniq_lines", d.uniq_lines as f64),
            ("anchor_lines", d.anchor_lines as f64),
            ("anchor_partners", d.anchor_partners as f64),
            ("partners_any", d.partners_any as f64),
            ("bridge_lines", d.bridge_lines as f64),
            ("uniq_genes", d.uniq_genes as f64),
            ("uniq_gene_cells", d.uniq_gene_cells as f64),
            (
                "pair_cells",
                if d.pair_cells == 0 {
                    f64::NAN
                } else {
                    d.pair_cells as f64
                },
            ),
            ("agree_med", d.agree_med),
            ("disagree_var", d.disagree_var),
            ("excess_var", d.excess_var),
            ("id_rna_self", d.id_rna_self),
            ("id_rna_rank", d.id_rna_rank),
            ("id_rna_top1", d.id_rna_top1),
            ("id_rna_top5", d.id_rna_top5),
            ("id_self_r", d.id_self_r),
            ("id_rank", d.id_rank),
            ("id_r_max_any", d.id_r_max_any),
            ("id_r_med_any", d.id_r_med_any),
            ("abund_rho", d.abund_rho),
            ("abund_rho_min", d.abund_rho_min),
            ("marg_shift", d.marg_shift),
            ("marg_se_gain", d.marg_se_gain),
            ("wshare", d.wshare),
            ("redund_ge3", d.redund_ge3),
            ("other_src_med", d.other_src_med),
        ];
        for (k, got) in metrics {
            let key = ("dataset".into(), d.name.clone(), String::new(), k.into());
            if let Some(want) = exp.get(&key) {
                close(&format!("{} {k}", d.name), got, want);
                checked += 1;
            }
        }
    }
    for p in &r.profiles {
        let key = |k: &str| {
            (
                "profile".to_owned(),
                p.dataset.clone(),
                p.line.clone(),
                k.to_owned(),
            )
        };
        let metrics = [
            ("abund_rho", p.abund_rho),
            ("self_r", p.self_r),
            ("self_rank", p.self_rank),
            ("best_r", p.best_r),
            ("med_r_any", p.med_r_any),
            ("rna_r_self", p.rna_r_self),
            ("rna_rank", p.rna_rank),
            ("marg_shift", p.marg_shift),
            ("marg_se_gain", p.marg_se_gain),
            ("wshare", p.wshare),
        ];
        for (k, got) in metrics {
            let Some(want) = exp.get(&key(k)) else {
                panic!("missing expected {} {} {k}", p.dataset, p.line);
            };
            close(&format!("{} {} {k}", p.dataset, p.line), got, want);
            checked += 1;
        }
        assert_eq!(
            exp.get(&key("best_line")).map(String::as_str),
            Some(p.best_line.as_str())
        );
    }
    assert_eq!(checked, 6 * 30 + r.profiles.len() * 10);
}

#[test]
fn dataset_value_runs_on_a_fit() {
    let fx = fixture();
    let params = BridleParams {
        reference: "REF".to_owned(),
        rank: 2,
        sweeps: 10,
        min_sweeps: 5,
        min_f_genes: 10,
        ..BridleParams::default()
    };
    let ab = reference_abundance(&fx.data, &params.reference);
    let (design, _, abz) = build_design(&ab, None);
    let res = match bridle_fit(&fx.data, &design, &abz, &params) {
        Ok(r) => r,
        Err(e) => panic!("fit failed: {e}"),
    };
    let vparams = ValueParams {
        identity: true,
        ..ValueParams::default()
    };
    let r = dataset_value(&fx.data, &res, None, &vparams);
    assert_eq!(r.datasets.len(), 6);
    assert_eq!(r.profiles.len(), fx.data.n_profiles());
    let refd = ds(&r, "REF");
    assert_eq!(refd.offset_sd, 0.0);
    assert!(refd.delta_median.is_finite() && refd.noise_var > 0.0);
    assert!(ds(&r, "B").c_abs.is_finite());
    assert!(ds(&r, "SINGLE").no_anchors_cannot_audit);
    let bad = ds(&r, "BAD");
    assert!(bad.id_self_r < 0.2 && bad.abund_rho < 0.3, "{bad:?}");
    assert!(bad.excess_var > ds(&r, "C").excess_var);
}
