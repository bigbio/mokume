use std::collections::{HashMap, HashSet};

use mokume_core::SampleId;
use mokume_io::{SdrfRawTable, SdrfTable};

use crate::accessions::{
    drop_contaminant_accessions, find_contaminant_twins, matches_protein_contaminant,
    matches_sql_contaminant, rename_contaminant_twins, strip_contaminant_members,
};
use crate::pibaq::summarize_digest_misses;
use crate::plex::{PlexSource, derive_sample_plexes};

fn owned(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn contaminant_twin_group_is_kept_with_human_accession() {
    assert!(!matches_protein_contaminant("CONTAM_P05787;P05787", &[]));
    assert_eq!(
        strip_contaminant_members("CONTAM_P05787;P05787", &[]).as_deref(),
        Some("P05787")
    );
    assert!(!matches_sql_contaminant(
        &owned(&["sp|CONTAM_P04179|CONTAM_SODM_HUMAN", "sp|P04179|SODM_HUMAN"]),
        &[]
    ));
}

#[test]
fn all_contaminant_group_is_dropped() {
    assert!(matches_protein_contaminant("CONTAM_P00761", &[]));
    assert!(matches_protein_contaminant(
        "CONTAM_P02769;CONTAM_P02768",
        &[]
    ));
    assert_eq!(
        strip_contaminant_members("CONTAM_P02769;CONTAM_P02768", &[]),
        None
    );
    assert!(matches_sql_contaminant(
        &owned(&["sp|CONTAM_P00761|TRYP_PIG"]),
        &[]
    ));
}

#[test]
fn mixed_group_keeps_non_contaminant_members() {
    assert!(!matches_protein_contaminant(
        "P02768;CONTAM_P02769;Q9XXXX",
        &[]
    ));
    assert_eq!(
        strip_contaminant_members("P02768;CONTAM_P02769;Q9XXXX", &[]).as_deref(),
        Some("P02768;Q9XXXX")
    );
    assert_eq!(strip_contaminant_members("P02768;Q9XXXX", &[]), None);
}

#[test]
fn decoy_member_still_drops_group() {
    assert!(matches_protein_contaminant("P12345;DECOY_P99999", &[]));
    assert!(matches_sql_contaminant(
        &owned(&["P12345", "DECOY_P99999"]),
        &[]
    ));
    let custom = owned(&["CONTAM", "DECOY"]);
    assert!(matches_protein_contaminant("P1;DECOY_P2", &custom));
    assert!(!matches_protein_contaminant("CONTAM_P1;P1", &custom));
    assert!(matches_protein_contaminant("contam_P1", &custom));
    assert!(!matches_sql_contaminant(
        &owned(&["CONTAM_P1", "P1"]),
        &custom
    ));
}

#[test]
fn digest_drops_contaminant_accessions() {
    let mut digest = HashMap::<String, HashSet<String>>::new();
    digest.insert(
        "P05787".to_owned(),
        HashSet::from(["LESGMQNMSIHTK".to_owned()]),
    );
    digest.insert(
        "CONTAM_P05787".to_owned(),
        HashSet::from(["LESGMQNMSIHTK".to_owned()]),
    );
    assert_eq!(drop_contaminant_accessions(&mut digest), 1);
    assert!(digest.contains_key("P05787"));
    assert!(!digest.contains_key("CONTAM_P05787"));
}

#[test]
fn contaminant_only_twin_group_is_renamed_to_human() {
    let peptides = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect();
    let mut digest = HashMap::<String, HashSet<String>>::new();
    digest.insert("P02533".to_owned(), peptides(&["AAAK", "BBBK"]));
    digest.insert("CONTAM_P02533".to_owned(), peptides(&["AAAK", "BBBK"]));
    digest.insert("CONTAM_P00761".to_owned(), peptides(&["TRYPK"]));
    digest.insert("P11111".to_owned(), peptides(&["CCCK"]));
    digest.insert("CONTAM_P11111".to_owned(), peptides(&["CCCK", "BOVINEK"]));
    let twins = find_contaminant_twins(&digest);
    assert_eq!(twins.len(), 1);
    assert_eq!(twins["CONTAM_P02533"], "P02533");
    assert_eq!(
        rename_contaminant_twins("CONTAM_P02533", &twins).as_deref(),
        Some("P02533")
    );
    assert_eq!(
        rename_contaminant_twins("CONTAM_P02533;P02533", &twins).as_deref(),
        Some("P02533")
    );
    assert_eq!(rename_contaminant_twins("CONTAM_P00761", &twins), None);
}

#[test]
fn digest_misses_are_counted_by_reason() {
    let mut peptide_accessions = HashMap::<String, HashSet<String>>::new();
    for peptide in ["PEPTIDEK", "MAEPTIDEK", "SECONDPEPK"] {
        peptide_accessions.insert(peptide.to_owned(), HashSet::from(["P1".to_owned()]));
    }
    let owner = peptide_accessions
        .keys()
        .map(|peptide| (peptide.clone(), "F1".to_owned()))
        .collect::<HashMap<_, _>>();
    let sample = SampleId::from(0);
    let observations = vec![
        ("PEPTIDEK".to_owned(), sample, 100.0),
        ("AEPTIDEK".to_owned(), sample, 10.0),
        ("PEPTIDEKSECONDPEPK".to_owned(), sample, 20.0),
        ("A".repeat(31) + "K", sample, 30.0),
        ("SEMITYPTICPEPTIDE".to_owned(), sample, 40.0),
        ("SEMITYPTICPEPTIDE".to_owned(), SampleId::from(1), 0.0),
    ];
    let summary = summarize_digest_misses(&observations, &owner, &peptide_accessions, Some(30));
    assert_eq!(summary.observed_peptides, 5);
    assert_eq!(summary.dropped_peptides, 4);
    assert_eq!(summary.met_removed_n_term, 1);
    assert_eq!(summary.missed_cleavage, 1);
    assert_eq!(summary.too_long, 1);
    assert_eq!(summary.other, 1);
    assert!((summary.dropped_intensity_fraction() - 100.0 / 200.0).abs() < 1e-12);
}

const TMT_HEADER: &str = "source name\tcomment[data file]\tcomment[label]\tcomment[plex]\n";

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn tmt_sdrf(rows: &[(&str, &str, &str, &str)]) -> TestResult<(SdrfTable, SdrfRawTable)> {
    let mut text = TMT_HEADER.to_owned();
    for (sample, file, label, plex) in rows {
        text.push_str(&format!("{sample}\t{file}\t{label}\t{plex}\n"));
    }
    Ok((
        SdrfTable::from_reader(text.as_bytes())?,
        SdrfRawTable::from_reader(text.as_bytes())?,
    ))
}

fn plex_sizes(plexes: &HashMap<String, String>) -> Vec<usize> {
    let mut sizes = HashMap::<&str, usize>::new();
    for plex in plexes.values() {
        *sizes.entry(plex.as_str()).or_default() += 1;
    }
    let mut sizes = sizes.into_values().collect::<Vec<_>>();
    sizes.sort_unstable();
    sizes
}

#[test]
fn set_style_source_names_split_by_data_files() -> TestResult {
    let mut rows = Vec::new();
    for (set, files) in [
        ("Set1", ["s1_f1.raw", "s1_f2.raw"]),
        ("Set2", ["s2_f1.raw", "s2_f2.raw"]),
    ] {
        for label in ["TMT126", "TMT127C", "TMT131"] {
            for file in files {
                rows.push((format!("PXD011896-{set}-{label}"), file, label));
            }
        }
    }
    let rows = rows
        .iter()
        .map(|(sample, file, label)| (sample.as_str(), *file, *label, ""))
        .collect::<Vec<_>>();
    let (sdrf, raw) = tmt_sdrf(&rows)?;
    let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
    assert_eq!(source, PlexSource::DataFiles);
    assert_eq!(plex_sizes(&plexes), vec![3, 3]);
    assert_eq!(
        plexes["PXD011896-Set1-TMT126"],
        plexes["PXD011896-Set1-TMT131"]
    );
    assert_ne!(
        plexes["PXD011896-Set1-TMT126"],
        plexes["PXD011896-Set2-TMT126"]
    );
    Ok(())
}

#[test]
fn sample_style_source_names_split_by_data_files() -> TestResult {
    let (sdrf, raw) = tmt_sdrf(&[
        ("MSV000085836-Sample-1", "a.raw", "TMT126", ""),
        ("MSV000085836-Sample-2", "a.raw", "TMT127N", ""),
        ("MSV000085836-Sample-3", "b.raw", "TMT126", ""),
        ("MSV000085836-Sample-4", "b.raw", "TMT127N", ""),
    ])?;
    let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
    assert_eq!(source, PlexSource::DataFiles);
    assert_eq!(plex_sizes(&plexes), vec![2, 2]);
    assert_ne!(
        plexes["MSV000085836-Sample-1"],
        plexes["MSV000085836-Sample-3"]
    );
    Ok(())
}

#[test]
fn legacy_plex_prefix_is_kept_as_plex_id() -> TestResult {
    let (sdrf, raw) = tmt_sdrf(&[
        ("p1_1", "a.raw", "TMT126", ""),
        ("p1_2", "a.raw", "TMT127N", ""),
        ("p2_1", "b.raw", "TMT126", ""),
        ("p2_2", "b.raw", "TMT127N", ""),
    ])?;
    let (plexes, _) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
    assert_eq!(plexes["p1_2"], "p1");
    assert_eq!(plexes["p2_1"], "p2");
    Ok(())
}

#[test]
fn explicit_plex_column_overrides_data_files() -> TestResult {
    let (sdrf, raw) = tmt_sdrf(&[
        ("S1", "a.raw", "TMT126", "A"),
        ("S2", "a.raw", "TMT127N", "A"),
        ("S3", "b.raw", "TMT126", "B"),
        ("S4", "b.raw", "TMT127N", "B"),
    ])?;
    let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), Some("comment[plex]"))?;
    assert_eq!(source, PlexSource::Column);
    assert_eq!(plexes["S2"], "A");
    assert_eq!(plexes["S3"], "B");
    assert!(derive_sample_plexes(&sdrf, Some(&raw), Some("comment[missing]")).is_err());
    Ok(())
}

#[test]
fn label_free_falls_back_to_source_name_rule() -> TestResult {
    let (sdrf, raw) = tmt_sdrf(&[
        ("p1_1", "a.raw", "label free sample", ""),
        ("p1_2", "b.raw", "label free sample", ""),
    ])?;
    let (plexes, source) = derive_sample_plexes(&sdrf, Some(&raw), None)?;
    assert_eq!(source, PlexSource::SourceName);
    assert_eq!(plexes["p1_2"], "p1");
    Ok(())
}
