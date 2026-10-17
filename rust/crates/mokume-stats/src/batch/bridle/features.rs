//! Technical protein features `x_g` for BRIDLE's structured offset `f(s, x_g)`.
//!
//! Port of the prototype's `feats.py` (sequence features) and `gene_feats`
//! (design preparation). Only sequence-derived, technical properties are used
//! (no GO, pathways, complexes or tissue annotation), so `f` can explain
//! detectability / digestion / ionisation bias without absorbing biology:
//!
//! `log_len, log_ntryp, log_nuniq, cov_obs, gravy, pI, charge7, mc_frac,
//! n_fraction_noM`, 20 amino-acid frequencies and a missing-sequence flag
//! (constant columns, e.g. `aa_I` in an I/L-merged FASTA, are dropped later).
//!
//! Tryptic digestion here is the plain Keil rule (cleave after K/R, not before
//! P) with no missed cleavages, peptides 7-30 aa, exactly as in `feats.py`. It
//! is intentionally independent of the pyOpenMS digest used by piBAQ: these are
//! coarse covariates, not a peptide universe.

use std::collections::HashMap;

/// Names of the sequence feature columns, in output order.
pub const FEATURE_NAMES: [&str; 30] = [
    "log_len",
    "log_ntryp",
    "log_nuniq",
    "cov_obs",
    "gravy",
    "pI",
    "charge7",
    "mc_frac",
    "n_fraction_noM",
    "aa_A",
    "aa_C",
    "aa_D",
    "aa_E",
    "aa_F",
    "aa_G",
    "aa_H",
    "aa_I",
    "aa_K",
    "aa_L",
    "aa_M",
    "aa_N",
    "aa_P",
    "aa_Q",
    "aa_R",
    "aa_S",
    "aa_T",
    "aa_V",
    "aa_W",
    "aa_Y",
    "miss_seq",
];

const AA: &[u8; 20] = b"ACDEFGHIKLMNPQRSTVWY";

/// Kyte-Doolittle hydropathy (0 for non-standard residues).
fn kd(c: u8) -> f64 {
    match c {
        b'A' => 1.8,
        b'R' => -4.5,
        b'N' | b'D' | b'Q' | b'E' => -3.5,
        b'C' => 2.5,
        b'G' => -0.4,
        b'H' => -3.2,
        b'I' => 4.5,
        b'L' => 3.8,
        b'K' => -3.9,
        b'M' => 1.9,
        b'F' => 2.8,
        b'P' => -1.6,
        b'S' => -0.8,
        b'T' => -0.7,
        b'W' => -0.9,
        b'Y' => -1.3,
        b'V' => 4.2,
        _ => 0.0,
    }
}

/// Net charge at `ph` with EMBOSS-like pKa values.
fn charge(counts: &[usize; 256], ph: f64) -> f64 {
    let frac_pos = |pk: f64| 1.0 / (1.0 + 10_f64.powf(ph - pk));
    let frac_neg = |pk: f64| 1.0 / (1.0 + 10_f64.powf(pk - ph));
    let n = |c: u8| counts[c as usize] as f64;
    let pos = frac_pos(8.6)
        + n(b'H') * frac_pos(6.5)
        + n(b'K') * frac_pos(10.8)
        + n(b'R') * frac_pos(12.5);
    let neg = frac_neg(3.6)
        + n(b'C') * frac_neg(8.5)
        + n(b'D') * frac_neg(3.9)
        + n(b'E') * frac_neg(4.1)
        + n(b'Y') * frac_neg(10.1);
    pos - neg
}

/// Isoelectric point by 40 bisection steps on `[0, 14]`.
fn isoelectric_point(counts: &[usize; 256]) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 14.0_f64);
    let mut mid = 7.0;
    for _ in 0..40 {
        mid = (lo + hi) / 2.0;
        if charge(counts, mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    mid
}

/// Tryptic peptides (cleave after K/R unless followed by P), no missed cleavage.
pub fn tryptic_peptides(seq: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..seq.len() {
        let cut = matches!(seq[i], b'K' | b'R') && seq.get(i + 1) != Some(&b'P');
        if cut {
            out.push(&seq[start..=i]);
            start = i + 1;
        }
    }
    if start < seq.len() {
        out.push(&seq[start..]);
    }
    out
}

fn observable(p: &[u8]) -> bool {
    (7..=30).contains(&p.len())
}

/// One FASTA record.
struct Entry {
    header: String,
    sequence: String,
}

fn parse_fasta(text: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut header: Option<String> = None;
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix('>') {
            if let Some(prev) = header.take() {
                out.push(Entry {
                    header: prev,
                    sequence: std::mem::take(&mut seq),
                });
            }
            header = Some(h.to_owned());
            seq.clear();
        } else {
            seq.push_str(line.trim());
        }
    }
    if let Some(prev) = header {
        out.push(Entry {
            header: prev,
            sequence: seq,
        });
    }
    out
}

/// Raw (unstandardised) sequence features, `genes x FEATURE_NAMES`. `NaN` for
/// genes without a sequence (all columns except `miss_seq`, which is `1`).
#[derive(Debug, Clone)]
pub struct SequenceFeatures {
    pub names: Vec<String>,
    /// Row-major `genes x names.len()`.
    pub values: Vec<Vec<f64>>,
    pub n_missing: usize,
}

/// Compute [`SequenceFeatures`] for `genes` from FASTA text.
///
/// A gene maps to the longest Swiss-Prot (`>sp|`) sequence whose header carries
/// `GN=<gene>` (and contains `organism`, when given, e.g. `"HUMAN"` as in the
/// prototype). Genes not found by gene name fall back to a match on the
/// accession or entry name (`sp|ACC|NAME`), so protein-accession inputs also
/// work. Tryptic-peptide uniqueness is counted over every FASTA entry.
pub fn sequence_features(
    fasta_text: &str,
    genes: &[String],
    organism: Option<&str>,
) -> SequenceFeatures {
    let entries = parse_fasta(fasta_text);
    let mut by_gene: HashMap<&str, &str> = HashMap::new();
    let mut by_acc: HashMap<&str, &str> = HashMap::new();
    for e in &entries {
        let first = e.header.split_whitespace().next().unwrap_or("");
        let is_sp = first.starts_with("sp|");
        if is_sp && organism.is_none_or(|o| e.header.contains(o)) {
            if let Some(gn) = e
                .header
                .split("GN=")
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
            {
                let longer = by_gene.get(gn).is_none_or(|s| e.sequence.len() > s.len());
                if longer {
                    by_gene.insert(gn, &e.sequence);
                }
            }
        }
        for id in first.split('|').skip(1) {
            by_acc.entry(id).or_insert(&e.sequence);
        }
    }
    let mut pepcount: HashMap<&[u8], u32> = HashMap::new();
    for e in &entries {
        let mut peps: Vec<&[u8]> = tryptic_peptides(e.sequence.as_bytes())
            .into_iter()
            .filter(|p| observable(p))
            .collect();
        peps.sort_unstable();
        peps.dedup();
        for p in peps {
            *pepcount.entry(p).or_insert(0) += 1;
        }
    }

    let width = FEATURE_NAMES.len();
    let mut values = Vec::with_capacity(genes.len());
    let mut n_missing = 0;
    for g in genes {
        let raw = by_gene.get(g.as_str()).or_else(|| by_acc.get(g.as_str()));
        let clean: Option<Vec<u8>> = raw
            .map(|s| {
                s.bytes()
                    .filter(u8::is_ascii_uppercase)
                    .collect::<Vec<u8>>()
            })
            .filter(|s| !s.is_empty());
        let Some(s) = clean else {
            let mut row = vec![f64::NAN; width];
            row[width - 1] = 1.0;
            values.push(row);
            n_missing += 1;
            continue;
        };
        let len = s.len() as f64;
        let mut counts = [0_usize; 256];
        for &c in &s {
            counts[c as usize] += 1;
        }
        let peps: Vec<&[u8]> = tryptic_peptides(&s)
            .into_iter()
            .filter(|p| observable(p))
            .collect();
        let n_uniq = peps
            .iter()
            .filter(|p| pepcount.get(*p).copied() == Some(1))
            .count();
        let covered: usize = peps.iter().map(|p| p.len()).sum();
        let kr = counts[b'K' as usize] + counts[b'R' as usize];
        let mc = s
            .windows(2)
            .filter(|w| matches!(w[0], b'K' | b'R') && w[1] == b'P')
            .count();
        let no_m = peps.iter().filter(|p| !p.contains(&b'M')).count();
        let gravy = s.iter().map(|&c| kd(c)).sum::<f64>() / len;
        let mut row = vec![
            len.log2(),
            (peps.len() as f64 + 1.0).log2(),
            (n_uniq as f64 + 1.0).log2(),
            covered as f64 / len,
            gravy,
            isoelectric_point(&counts),
            charge(&counts, 7.0) / len * 100.0,
            mc as f64 / kr.max(1) as f64,
            no_m as f64 / peps.len().max(1) as f64,
        ];
        row.extend(AA.iter().map(|&a| counts[a as usize] as f64 / len));
        row.push(0.0);
        values.push(row);
    }
    SequenceFeatures {
        names: FEATURE_NAMES.iter().map(|s| (*s).to_owned()).collect(),
        values,
        n_missing,
    }
}

/// Linear-interpolated quantile of a sorted slice (NumPy's default method).
pub(crate) fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Median of the finite values (average of the two middle values).
pub(crate) fn nan_median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Build the design matrix `D = [1, spline(abundance), z(features)]` used by
/// the per-dataset ridge `f(s, x_g)` (the prototype's `gene_feats` + `FModel`).
///
/// * `abundance[g]`: reference-dataset median (fallback: all-data median);
///   z-scored, expanded to a cubic truncated-power spline with knots at the
///   10/35/65/90% quantiles, each column standardised (population SD).
/// * `features`: raw per-gene technical features (may contain `NaN`); columns
///   with zero sample SD are dropped, `NaN` is filled with the column median,
///   then columns are standardised (sample SD, `ddof = 1`).
///
/// Returns `(design row-major genes x p, column names, abz)` where `abz` is the
/// z-scored abundance used by the noise model.
pub fn build_design(
    abundance: &[f64],
    features: Option<&SequenceFeatures>,
) -> (Vec<Vec<f64>>, Vec<String>, Vec<f64>) {
    let g = abundance.len();
    let finite: Vec<f64> = abundance
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mean = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
    let sd = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len().max(1) as f64)
        .sqrt();
    let sd = if sd > 0.0 { sd } else { 1.0 };
    let abz: Vec<f64> = abundance
        .iter()
        .map(|v| if v.is_finite() { (v - mean) / sd } else { 0.0 })
        .collect();
    let mut sorted = abz.clone();
    sorted.sort_by(f64::total_cmp);
    let knots: Vec<f64> = [0.1, 0.35, 0.65, 0.9]
        .iter()
        .map(|&q| quantile_sorted(&sorted, q))
        .collect();

    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    cols.push(abz.clone());
    for k in &knots {
        cols.push(abz.iter().map(|x| (x - k).max(0.0).powi(3)).collect());
    }
    for (j, c) in cols.iter_mut().enumerate() {
        standardise(c, 0);
        names.push(format!("abund_s{j}"));
    }
    if let Some(f) = features {
        for (j, name) in f.names.iter().enumerate() {
            let mut col: Vec<f64> = f.values.iter().map(|row| row[j]).collect();
            let present: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
            if present.len() < 2 || sample_sd(&present) == 0.0 {
                continue;
            }
            let med = nan_median(present.iter().copied());
            for v in &mut col {
                if !v.is_finite() {
                    *v = med;
                }
            }
            standardise(&mut col, 1);
            cols.push(col);
            names.push(name.clone());
        }
    }
    let p = cols.len() + 1;
    let mut design = vec![vec![0.0; p]; g];
    for (gi, row) in design.iter_mut().enumerate() {
        row[0] = 1.0;
        for (j, c) in cols.iter().enumerate() {
            row[j + 1] = c[gi];
        }
    }
    let mut all_names = vec!["intercept".to_owned()];
    all_names.extend(names);
    (design, all_names, abz)
}

fn sample_sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn standardise(col: &mut [f64], ddof: usize) {
    let n = col.len();
    if n <= ddof {
        return;
    }
    let m = col.iter().sum::<f64>() / n as f64;
    let sd = (col.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - ddof) as f64).sqrt();
    let sd = if sd > 0.0 { sd } else { 1.0 };
    for v in col.iter_mut() {
        *v = (*v - m) / sd;
    }
}

#[cfg(test)]
mod tests {
    use super::{build_design, sequence_features, tryptic_peptides};

    const FASTA: &str = ">sp|P1|A_HUMAN Protein A OS=Homo sapiens GN=GENEA PE=1\n\
MKAAAAAAAKPLLLLLLLRGGGGGGGGK\nWWWWWWWW\n\
>sp|P2|B_HUMAN Protein B OS=Homo sapiens GN=GENEB PE=1\nMAAAAAAAKAAAAAAAR\n\
>sp|P3|B2_HUMAN Protein B long OS=Homo sapiens GN=GENEB PE=1\nMAAAAAAAKAAAAAAARCCCCCCCK\n\
>sp|P9|X_BOVIN contaminant GN=GENEA\nMKK\n";

    #[test]
    fn digestion_follows_keil_rule() {
        let peps: Vec<&[u8]> = tryptic_peptides(b"AKPRGKAR");
        assert_eq!(peps, vec![&b"AKPR"[..], b"GK", b"AR"]);
    }

    #[test]
    fn features_use_longest_organism_entry_and_flag_missing() {
        let genes = vec![
            "GENEA".to_owned(),
            "GENEB".to_owned(),
            "NOPE".to_owned(),
            "P2".to_owned(),
        ];
        let f = sequence_features(FASTA, &genes, Some("HUMAN"));
        assert_eq!(f.n_missing, 1);
        // GENEA -> P1 (the BOVIN entry is excluded by the organism filter).
        let len_a = (28.0_f64 + 8.0).log2();
        assert!((f.values[0][0] - len_a).abs() < 1e-12);
        // GENEB -> the longer P3 entry (25 aa).
        assert!((f.values[1][0] - 25.0_f64.log2()).abs() < 1e-12);
        // Missing gene: NaN features, miss_seq = 1.
        assert!(f.values[2][0].is_nan());
        assert_eq!(f.values[2][29], 1.0);
        // Accession fallback.
        assert!((f.values[3][0] - 17.0_f64.log2()).abs() < 1e-12);
        // pI is in range, KP counted as missed cleavage for GENEA (1 of 4 K/R).
        assert!(f.values[0][5] > 0.0 && f.values[0][5] < 14.0);
        assert!((f.values[0][7] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn design_drops_constant_columns_and_fills_missing() {
        let genes = vec![
            "GENEA".to_owned(),
            "GENEB".to_owned(),
            "NOPE".to_owned(),
            "P2".to_owned(),
        ];
        let f = sequence_features(FASTA, &genes, Some("HUMAN"));
        let ab = [1.0, 2.0, 3.0, 4.0];
        let (d, names, abz) = build_design(&ab, Some(&f));
        assert_eq!(names[0], "intercept");
        assert!(names.iter().all(|n| n != "aa_Y"), "constant column kept");
        assert!(d.iter().flatten().all(|v| v.is_finite()));
        assert!((abz.iter().sum::<f64>()).abs() < 1e-12);
        assert_eq!(d.len(), 4);
        assert_eq!(d[0].len(), names.len());
    }
}
