use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use mokume_core::{FeatureToProteinsConfig, FilterConfig, Result};
use mokume_io::{
    QpxFeatureProteinGroup, QpxFeatureProteinGroupReader, QpxPsmParquetReader, QpxPsmRecord,
    SdrfTable,
};

use crate::memory::MemoryPlan;

#[derive(Debug)]
pub(crate) struct SpectralCountCell {
    pub protein_group: String,
    pub sample: String,
    pub psms: usize,
    pub sequences: HashSet<String>,
}

#[derive(Debug)]
pub(crate) struct SpectralCountResult {
    pub cells: Vec<SpectralCountCell>,
    pub target_psms: usize,
    pub unique_psms: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpectrumAssignment {
    proteins: BTreeSet<String>,
    sample: String,
    sequence: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FeatureProteinGroup {
    proteins: BTreeSet<String>,
    contaminant: bool,
    is_decoy: bool,
}

pub(crate) fn count(
    config: &FeatureToProteinsConfig,
    sdrf: &SdrfTable,
    memory: &MemoryPlan,
) -> Result<SpectralCountResult> {
    let psm = config
        .input
        .psm
        .as_deref()
        .ok_or_else(|| super::invalid_input("spectral_count requires --psm input"))?;
    let features =
        config.input.parquet.as_deref().ok_or_else(|| {
            super::invalid_input("spectral_count requires --parquet feature mapping")
        })?;
    let feature_ids = collect_feature_ids(psm, memory)?;
    let feature_groups = load_feature_groups(features, &feature_ids, memory)?;
    let reader = QpxPsmParquetReader::open(psm, memory.qpx_batch_size())?;
    let (assignments, target_psms) =
        collect_assignments(config, sdrf, memory, reader, &feature_groups)?;
    if assignments.is_empty() {
        return Err(super::invalid_input(
            "spectral_count found no usable feature-linked target PSM after decoy, peptide-length, contaminant, and protein-group filtering",
        ));
    }
    let unique_psms = assignments.len();
    let cells = collapse_assignments(assignments, config.filtering.min_unique_peptides);
    Ok(SpectralCountResult {
        cells,
        target_psms,
        unique_psms,
    })
}

fn collect_feature_ids(path: &Path, memory: &MemoryPlan) -> Result<HashSet<i64>> {
    let mut reader = QpxPsmParquetReader::open(path, memory.qpx_batch_size())?;
    let mut feature_ids = HashSet::new();
    for batch in &mut reader {
        feature_ids.extend(batch?.into_iter().filter_map(|record| record.feature_id));
        memory.check("PSM feature-link scan")?;
    }
    if feature_ids.is_empty() {
        return Err(super::invalid_input(
            "spectral_count found no PSM linked to a QPX feature_id",
        ));
    }
    Ok(feature_ids)
}

fn load_feature_groups(
    path: &Path,
    requested: &HashSet<i64>,
    memory: &MemoryPlan,
) -> Result<HashMap<i64, FeatureProteinGroup>> {
    let mut reader = QpxFeatureProteinGroupReader::open(path, memory.qpx_batch_size())?;
    let mut groups = HashMap::new();
    for batch in &mut reader {
        for record in batch? {
            if requested.contains(&record.feature_id) {
                insert_feature_group(&mut groups, record)?;
            }
        }
        memory.check("QPX feature protein-group mapping")?;
    }
    if groups.is_empty() {
        return Err(super::invalid_input(
            "spectral_count found no referenced feature_id in the QPX feature mapping",
        ));
    }
    Ok(groups)
}

fn insert_feature_group(
    groups: &mut HashMap<i64, FeatureProteinGroup>,
    record: QpxFeatureProteinGroup,
) -> Result<()> {
    let group = FeatureProteinGroup {
        proteins: protein_set(&record.protein_accessions).unwrap_or_default(),
        contaminant: super::matches_sql_contaminant(&record.protein_accessions, &[]),
        is_decoy: record.is_decoy,
    };
    if let Some(existing) = groups.insert(record.feature_id, group.clone()) {
        if existing != group {
            return Err(super::invalid_input(format!(
                "QPX feature_id {} has conflicting protein-group mappings",
                record.feature_id
            )));
        }
    }
    Ok(())
}

fn collect_assignments(
    config: &FeatureToProteinsConfig,
    sdrf: &SdrfTable,
    memory: &MemoryPlan,
    mut reader: QpxPsmParquetReader,
    feature_groups: &HashMap<i64, FeatureProteinGroup>,
) -> Result<(HashMap<i64, SpectrumAssignment>, usize)> {
    let mut assignments = HashMap::new();
    let mut run_samples = HashMap::new();
    let mut target_psms = 0;
    for batch in &mut reader {
        for record in batch? {
            target_psms += usize::from(add_record(
                &config.filtering,
                sdrf,
                &mut run_samples,
                &mut assignments,
                record,
                feature_groups,
            )?);
        }
        memory.check("PSM spectral-count streaming")?;
    }
    Ok((assignments, target_psms))
}

fn add_record(
    filtering: &FilterConfig,
    sdrf: &SdrfTable,
    run_samples: &mut HashMap<String, String>,
    assignments: &mut HashMap<i64, SpectrumAssignment>,
    record: QpxPsmRecord,
    feature_groups: &HashMap<i64, FeatureProteinGroup>,
) -> Result<bool> {
    if record.is_decoy || record.sequence.len() < filtering.min_aa {
        return Ok(false);
    }
    let Some(group) = record
        .feature_id
        .and_then(|feature_id| feature_groups.get(&feature_id))
    else {
        return Ok(false);
    };
    if group.is_decoy || group.proteins.is_empty() {
        return Ok(false);
    }
    if filtering.remove_contaminants && group.contaminant {
        return Ok(false);
    }
    let mut proteins = group.proteins.clone();
    if filtering.remove_contaminants {
        let kept = proteins
            .iter()
            .filter(|protein| !super::is_contaminant(protein))
            .cloned()
            .collect::<BTreeSet<_>>();
        if !kept.is_empty() {
            proteins = kept;
        }
    }
    let sample = sample_for_run(sdrf, run_samples, &record.run_file_name)?;
    insert_assignment(assignments, record, proteins, sample)?;
    Ok(true)
}

fn sample_for_run(
    sdrf: &SdrfTable,
    cache: &mut HashMap<String, String>,
    run: &str,
) -> Result<String> {
    if let Some(sample) = cache.get(run) {
        return Ok(sample.clone());
    }
    let sample = sdrf.lookup(run, None)?.sample_accession.clone();
    cache.insert(run.to_owned(), sample.clone());
    Ok(sample)
}

fn insert_assignment(
    assignments: &mut HashMap<i64, SpectrumAssignment>,
    record: QpxPsmRecord,
    proteins: BTreeSet<String>,
    sample: String,
) -> Result<()> {
    let psm_id = record.psm_id;
    let assignment = SpectrumAssignment {
        proteins,
        sample,
        sequence: record.sequence,
    };
    if let Some(existing) = assignments.insert(psm_id, assignment.clone()) {
        if existing != assignment {
            return Err(super::invalid_input(format!(
                "QPX psm_id {psm_id} has conflicting spectral-count assignments"
            )));
        }
        return Err(super::invalid_input(format!(
            "QPX psm_id {psm_id} is duplicated"
        )));
    }
    Ok(())
}

fn collapse_assignments(
    assignments: HashMap<i64, SpectrumAssignment>,
    min_unique: usize,
) -> Vec<SpectralCountCell> {
    let mut cells = BTreeMap::<(String, String), (usize, HashSet<String>)>::new();
    for assignment in assignments.into_values() {
        let protein_group = assignment
            .proteins
            .into_iter()
            .collect::<Vec<_>>()
            .join(";");
        let cell = cells.entry((protein_group, assignment.sample)).or_default();
        cell.0 += 1;
        cell.1.insert(assignment.sequence);
    }
    cells
        .into_iter()
        .filter(|(_, (_, sequences))| sequences.len() >= min_unique)
        .map(
            |((protein_group, sample), (psms, sequences))| SpectralCountCell {
                protein_group,
                sample,
                psms,
                sequences,
            },
        )
        .collect()
}

fn protein_set(accessions: &[String]) -> Option<BTreeSet<String>> {
    let proteins = accessions
        .iter()
        .map(|accession| super::parse_protein_accession(accession))
        .filter(|accession| !accession.is_empty())
        .collect::<BTreeSet<_>>();
    (!proteins.is_empty()).then_some(proteins)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use mokume_core::FilterConfig;
    use mokume_io::{QpxPsmRecord, SdrfTable};

    use super::{
        add_record, collapse_assignments, insert_assignment, protein_set, FeatureProteinGroup,
        SpectrumAssignment,
    };

    fn assignment(protein: &str, sample: &str, sequence: &str) -> SpectrumAssignment {
        SpectrumAssignment {
            proteins: BTreeSet::from([protein.to_owned()]),
            sample: sample.to_owned(),
            sequence: sequence.to_owned(),
        }
    }

    #[test]
    fn counts_psms_not_unique_peptides() {
        let assignments = HashMap::from([
            (1, assignment("P1", "S1", "PEPTIDEA")),
            (2, assignment("P1", "S1", "PEPTIDEA")),
            (3, assignment("P1", "S1", "PEPTIDEB")),
        ]);
        let cells = collapse_assignments(assignments, 2);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].psms, 3);
        assert_eq!(cells[0].sequences.len(), 2);
    }

    #[test]
    fn keeps_distinct_psms_from_one_spectrum_separate() -> Result<(), Box<dyn std::error::Error>> {
        let mut assignments = HashMap::new();
        let first = QpxPsmRecord {
            psm_id: 1,
            sequence: "PEPTIDEA".to_owned(),
            run_file_name: "run-a".to_owned(),
            scan: vec![1],
            feature_id: Some(10),
            is_decoy: false,
        };
        insert_assignment(
            &mut assignments,
            first,
            BTreeSet::from(["P1".to_owned()]),
            "S1".to_owned(),
        )?;
        let second = QpxPsmRecord {
            psm_id: 2,
            sequence: "PEPTIDEB".to_owned(),
            run_file_name: "run-a".to_owned(),
            scan: vec![1],
            feature_id: Some(20),
            is_decoy: false,
        };
        insert_assignment(
            &mut assignments,
            second,
            BTreeSet::from(["P2".to_owned()]),
            "S1".to_owned(),
        )?;
        let cells = collapse_assignments(assignments, 1);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].protein_group, "P1");
        assert_eq!(cells[0].psms, 1);
        assert_eq!(cells[1].protein_group, "P2");
        assert_eq!(cells[1].psms, 1);
        Ok(())
    }

    #[test]
    fn shared_protein_set_is_sorted_and_deduplicated() {
        let accessions = ["sp|P2|two", "P1", "P2"].map(str::to_owned);
        assert_eq!(
            protein_set(&accessions),
            Some(BTreeSet::from(["P1".to_owned(), "P2".to_owned()]))
        );
    }

    #[test]
    fn maps_psm_feature_id_to_feature_protein_group() -> Result<(), Box<dyn std::error::Error>> {
        let sdrf =
            SdrfTable::from_reader(b"source name\tcomment[data file]\nS1\trun-a.raw\n".as_slice())?;
        let feature_groups = HashMap::from([(
            10,
            FeatureProteinGroup {
                proteins: BTreeSet::from(["P1".to_owned()]),
                contaminant: false,
                is_decoy: false,
            },
        )]);
        let record = QpxPsmRecord {
            psm_id: 42,
            sequence: "PEPTIDEA".to_owned(),
            run_file_name: "run-a.mzML".to_owned(),
            scan: vec![42],
            feature_id: Some(10),
            is_decoy: false,
        };
        let mut run_samples = HashMap::new();
        let mut assignments = HashMap::new();

        assert!(add_record(
            &FilterConfig::default(),
            &sdrf,
            &mut run_samples,
            &mut assignments,
            record,
            &feature_groups,
        )?);
        let cells = collapse_assignments(assignments, 1);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].protein_group, "P1");
        assert_eq!(cells[0].sample, "S1");
        assert_eq!(cells[0].psms, 1);
        Ok(())
    }
}
