//! Pure filter primitives for the `features2peptides` preprocessing filter
//! pipeline (`--filter-config` / `--filter-*`), ported from
//! `mokume/preprocessing/filters/`. Each function reproduces one Python filter's
//! per-row or per-group decision exactly; the orchestration that groups rows and
//! threads these through the per-sample flow lives in the `features2peptides`
//! wiring. Keeping the decisions here as small pure functions makes every
//! cross-language edge (the missed-cleavage C-terminal rule, `pd.std` ddof=1,
//! case-insensitive modification matching) directly oracle-testable.

use regex::Regex;

/// Canonical amino-acid length after stripping modification annotations,
/// mirroring `PeptideLengthFilter`'s `get_aa_length`
/// (`preprocessing/filters/peptide.py`): drop every `[...]` group, then every
/// `(...)` group, then count the remaining characters. The two passes run in the
/// same order as Python's sequential `re.sub` so nested/adjacent annotations
/// reduce identically.
pub fn canonical_aa_length(sequence: &str) -> usize {
    let without_brackets = strip_delimited(sequence, '[', ']');
    let without_parens = strip_delimited(&without_brackets, '(', ')');
    without_parens.chars().count()
}

/// Remove each minimal `open..close` span (inclusive) from `text`, matching
/// Python's non-greedy `re.sub(r"\[.*?\]", "", s)`: on an unmatched `open` with
/// no later `close`, the remainder is kept verbatim (Python's regex would not
/// match, so nothing is removed).
fn strip_delimited(text: &str, open: char, close: char) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != open {
            out.push(ch);
            continue;
        }
        // Look for a matching close; buffer the span so an unmatched open is
        // restored verbatim (Python's non-greedy match would not fire).
        let mut span = String::new();
        span.push(ch);
        let mut closed = false;
        for inner in chars.by_ref() {
            span.push(inner);
            if inner == close {
                closed = true;
                break;
            }
        }
        if !closed {
            out.push_str(&span);
        }
    }
    out
}

/// Trypsin missed-cleavage count, mirroring `MissedCleavageFilter`'s
/// `count_missed_cleavages` (`preprocessing/filters/peptide.py`): uppercase the
/// sequence, drop the C-terminal residue (a terminal K/R is the expected cut, not
/// a miss), then count `K`/`R` residues that are followed by a residue other than
/// `P`. A trailing K/R with no following residue is never counted.
pub fn trypsin_missed_cleavages(sequence: &str) -> usize {
    let upper = sequence.to_ascii_uppercase();
    let residues: Vec<char> = upper.chars().collect();
    // Drop the C-terminal residue when the peptide is longer than one residue.
    let scan_len = if residues.len() > 1 {
        residues.len() - 1
    } else {
        residues.len()
    };
    let mut count = 0;
    for index in 0..scan_len {
        let residue = residues[index];
        if residue != 'K' && residue != 'R' {
            continue;
        }
        // The "next" residue must lie within the C-terminal-trimmed sequence
        // (`residues[..scan_len]`); a K/R at the trimmed end is terminal, not a
        // miss. Followed by `P` is also not a cut.
        if index + 1 < scan_len && residues[index + 1] != 'P' {
            count += 1;
        }
    }
    count
}

/// Match structured names/accessions exactly, or legacy peptidoform tokens.
pub fn peptide_has_excluded_modification(
    modified_sequence: &str,
    modifications: Option<&[String]>,
    exclude: &[String],
) -> bool {
    if let Some(modifications) = modifications {
        return exclude.iter().any(|excluded| {
            modifications
                .iter()
                .any(|value| value.eq_ignore_ascii_case(excluded))
        });
    }
    let haystack = modified_sequence.to_ascii_lowercase();
    exclude.iter().any(|modification| {
        let needle = modification.to_ascii_lowercase();
        let accession = needle.split_once(':').is_some_and(|(prefix, suffix)| {
            !prefix.is_empty()
                && !prefix.chars().any(char::is_whitespace)
                && !suffix.is_empty()
                && suffix.bytes().all(|byte| byte.is_ascii_digit())
        });
        if accession {
            haystack.contains(&format!("[{needle}]"))
        } else {
            haystack.contains(&needle)
        }
    })
}

/// Whether a canonical peptide sequence matches any exclusion pattern, mirroring
/// `SequencePatternFilter` (`preprocessing/filters/peptide.py:415-418`): Python
/// drops rows where `str.contains("|".join(patterns), regex=True)` is true, i.e.
/// any pattern is found anywhere in the canonical sequence. Each pattern is a real
/// (un-escaped, case-sensitive) regex; OR-ing the per-pattern searches is
/// equivalent to Python's single alternation regex. An empty pattern set never
/// excludes (Python builds no filter).
pub fn sequence_matches_excluded(canonical: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|pattern| pattern.is_match(canonical))
}

/// Whether a charge passes the allowed-charge-state filter, mirroring
/// `ChargeStateFilter`'s `isin` (`preprocessing/filters/peptide.py`).
pub fn charge_is_allowed(charge: i32, allowed: &[i32]) -> bool {
    allowed.contains(&charge)
}

/// The effective minimum intensity threshold the intensity filter applies,
/// mirroring `create_intensity_filters` (`preprocessing/filters/factory.py`):
/// `remove_zero_intensity` forces a positive floor of `1e-10` even when
/// `min_intensity` is `0`, so zero-intensity rows are always removed.
pub fn effective_min_intensity(min_intensity: f64, remove_zero_intensity: bool) -> f64 {
    let floor = if remove_zero_intensity { 1e-10 } else { 0.0 };
    min_intensity.max(floor)
}

/// Coefficient of variation for a replicate group, mirroring
/// `CVThresholdFilter`'s `calc_cv` (`preprocessing/filters/intensity.py`):
/// `pandas` sample standard deviation (ddof = 1) over the mean, or `None` when the
/// group has fewer than two values or a non-positive mean. `None` means the group
/// passes any threshold (Python keeps `NaN` CVs).
pub fn coefficient_of_variation(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return None;
    }
    // Sample standard deviation (ddof = 1), matching `pandas.Series.std`.
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1.0);
    Some(variance.sqrt() / mean)
}

/// Missing-value rate for a run/sample group, mirroring `MissingRateFilter`'s
/// `missing_rate` (`preprocessing/filters/run_qc.py`): the fraction of values that
/// are `NaN` or exactly `0`. An empty group yields `NaN` (Python's mean of an
/// empty series), so callers treat empty groups as failing any finite threshold.
pub fn missing_rate(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let missing = values
        .iter()
        .filter(|value| value.is_nan() || **value == 0.0)
        .count();
    missing as f64 / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values captured from the Python primitives (`/tmp/mokume-filter-oracle/gen.py`,
    // run in the Bigbio conda env): re/pandas reproduce these exactly.

    #[test]
    fn canonical_aa_length_strips_modifications_like_python() {
        assert_eq!(canonical_aa_length("PEPTIDE"), 7);
        assert_eq!(canonical_aa_length("PEPT[Phospho]IDE"), 7);
        assert_eq!(canonical_aa_length("AC(ox)DEFR"), 6);
        assert_eq!(canonical_aa_length("M(Oxidation)KKR"), 4);
        assert_eq!(canonical_aa_length(""), 0);
        // Unmatched delimiter: Python's non-greedy regex does not match, so the
        // remainder (including the stray bracket) is kept.
        assert_eq!(canonical_aa_length("PEP[TIDE"), "PEP[TIDE".chars().count());
    }

    #[test]
    fn trypsin_missed_cleavages_matches_python_oracle() {
        assert_eq!(trypsin_missed_cleavages("PEPTIDEK"), 0);
        assert_eq!(trypsin_missed_cleavages("PEPTKIDE"), 1);
        assert_eq!(trypsin_missed_cleavages("PEPTKPIDE"), 0); // K before P is not a cut
        assert_eq!(trypsin_missed_cleavages("KKKKK"), 3); // last K dropped, K-before-P none
        assert_eq!(trypsin_missed_cleavages("PEPTIDER"), 0);
        assert_eq!(trypsin_missed_cleavages("PEPTRIDEK"), 1);
        assert_eq!(trypsin_missed_cleavages("K"), 0);
        assert_eq!(trypsin_missed_cleavages("PEPTIDE"), 0);
        assert_eq!(trypsin_missed_cleavages("RPEPTIDE"), 0); // R before P is not a cut
    }

    #[test]
    fn modification_exclusion_uses_structured_identity_and_legacy_tokens() {
        let modifications = vec!["Carbamidomethyl".to_owned(), "UNIMOD:4".to_owned()];
        for name in ["carbamidomethyl", "unimod:4"] {
            assert!(peptide_has_excluded_modification(
                "CK",
                Some(&modifications),
                &[name.to_owned()]
            ));
        }
        assert!(!peptide_has_excluded_modification(
            "CK",
            Some(&modifications),
            &["Carbamido".to_owned()]
        ));
        assert!(!peptide_has_excluded_modification(
            "M[UNIMOD:35]K",
            None,
            &["UNIMOD:3".to_owned()]
        ));
        assert!(peptide_has_excluded_modification(
            "M[UNIMOD:3]K",
            None,
            &["UNIMOD:3".to_owned()]
        ));
        assert!(!peptide_has_excluded_modification(
            "M[UNIMOD:3]K",
            Some(&[]),
            &["UNIMOD:3".to_owned()]
        ));
    }

    #[test]
    fn modification_exclusion_is_case_insensitive_substring() {
        let oxidation = vec!["Oxidation".to_string()];
        assert!(peptide_has_excluded_modification(
            "PEPT(Oxidation)IDE",
            None,
            &oxidation
        ));
        assert!(!peptide_has_excluded_modification(
            "PEPTIDE", None, &oxidation
        ));
        assert!(peptide_has_excluded_modification(
            "pept(oxidation)ide",
            None,
            &oxidation
        ));
        let both = vec!["Oxidation".to_string(), "Phospho".to_string()];
        assert!(peptide_has_excluded_modification(
            "AC(Phospho)DE",
            None,
            &both
        ));
        // Empty exclude list never excludes.
        assert!(!peptide_has_excluded_modification(
            "AC(Phospho)DE",
            None,
            &[]
        ));
    }

    fn compile(patterns: &[&str]) -> Vec<Regex> {
        let compiled: Vec<Regex> = patterns
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect();
        assert_eq!(compiled.len(), patterns.len(), "test patterns must compile");
        compiled
    }

    #[test]
    fn sequence_pattern_exclusion_matches_python_regex_contains() {
        // Anchored start (Python `re`/`str.contains` search semantics).
        let starts_ac = compile(&["^AC"]);
        assert!(sequence_matches_excluded("ACDEFK", &starts_ac));
        assert!(!sequence_matches_excluded("DACEFK", &starts_ac));
        // Anchored end / character class.
        let ends_kr = compile(&["[KR]$"]);
        assert!(sequence_matches_excluded("PEPTIDEK", &ends_kr));
        assert!(sequence_matches_excluded("PEPTIDER", &ends_kr));
        assert!(!sequence_matches_excluded("PEPTIDEA", &ends_kr));
        // Wildcard middle (`P.G` = P, any one char, G — verified via Python
        // `re.search`: "APAGK"/"PQG" match, "APGK"(adjacent)/"APPK"/"PAAG" do not).
        let p_any_g = compile(&["P.G"]);
        assert!(sequence_matches_excluded("APAGK", &p_any_g));
        assert!(sequence_matches_excluded("PQG", &p_any_g));
        assert!(!sequence_matches_excluded("APGK", &p_any_g));
        assert!(!sequence_matches_excluded("APPK", &p_any_g));
        assert!(!sequence_matches_excluded("PAAG", &p_any_g));
        // Multiple patterns: OR of the alternation, equal to Python's joined regex.
        let any_of = compile(&["^AC", "K$"]);
        assert!(sequence_matches_excluded("ACDEF", &any_of));
        assert!(sequence_matches_excluded("PEPTIDEK", &any_of));
        assert!(!sequence_matches_excluded("DEFGH", &any_of));
        // Case-sensitive (Python passes no `case=False`).
        let upper_ac = compile(&["AC"]);
        assert!(sequence_matches_excluded("ACDEF", &upper_ac));
        assert!(!sequence_matches_excluded("acdef", &upper_ac));
        // Empty pattern set never excludes.
        assert!(!sequence_matches_excluded("ACDEF", &[]));
    }

    #[test]
    fn charge_allowed_is_membership() {
        assert!(charge_is_allowed(2, &[2, 3, 4]));
        assert!(!charge_is_allowed(1, &[2, 3, 4]));
    }

    #[test]
    fn effective_min_intensity_floors_zero_when_removing_zeros() {
        assert_eq!(effective_min_intensity(0.0, true), 1e-10);
        assert_eq!(effective_min_intensity(0.0, false), 0.0);
        assert_eq!(effective_min_intensity(5.0, true), 5.0);
        assert_eq!(effective_min_intensity(5.0, false), 5.0);
    }

    #[test]
    fn coefficient_of_variation_matches_pandas_ddof1() {
        let Some(cv) = coefficient_of_variation(&[10.0, 12.0, 14.0]) else {
            panic!("cv should be Some");
        };
        assert!((cv - 0.16666666666666666).abs() < 1e-12);
        assert_eq!(coefficient_of_variation(&[5.0, 5.0, 5.0]), Some(0.0));
        let Some(cv2) = coefficient_of_variation(&[2.0, 8.0]) else {
            panic!("cv should be Some");
        };
        assert!((cv2 - 0.8485281374238569).abs() < 1e-12);
        // Single value or non-positive mean -> None (Python NaN -> passes).
        assert_eq!(coefficient_of_variation(&[100.0]), None);
        assert_eq!(coefficient_of_variation(&[0.0, 0.0]), None);
    }

    #[test]
    fn missing_rate_counts_nan_or_zero() {
        assert_eq!(missing_rate(&[10.0, 0.0, f64::NAN, 5.0]), 0.5);
        assert_eq!(missing_rate(&[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(missing_rate(&[0.0, 0.0]), 1.0);
        assert!(missing_rate(&[]).is_nan());
    }
}
