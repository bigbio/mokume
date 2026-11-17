use crate::aggregation::ProteinFamily;
use crate::pibaq::PibaqEvidence;
use crate::pibaq::allocation::{FamilyCounts, PibaqCell, family_row_id};
use crate::state::CellKey;
use crate::{
    File, HashMap, MokumeError, Path, PibaqSharedMode, ProteinId, Result, StringIdRegistry,
    WriterBuilder, create_parent_dir, csv_error,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PibaqRowType {
    Member,
    Family,
    MemberNotQuantified,
}

impl PibaqRowType {
    const fn label(self) -> &'static str {
        match self {
            Self::Member => "member",
            Self::Family => "family",
            Self::MemberNotQuantified => "member_not_quantified",
        }
    }
}

/// One line of the `--pibaq-evidence` side-car.
#[derive(Debug, Clone)]
pub(crate) struct PibaqEvidenceRow {
    pub(crate) row_id: String,
    pub(crate) row_type: PibaqRowType,
    family_id: String,
    family_size: usize,
    family_members: String,
    evidence: PibaqEvidence,
    pub(crate) theoretical_peptides: usize,
    pub(crate) theoretical_unique: usize,
    pub(crate) theoretical_shared: usize,
    observed_unique: usize,
    pub(crate) observed_shared: usize,
    pub(crate) samples_quantified: usize,
}

impl PibaqEvidenceRow {
    fn record(&self, mode: PibaqSharedMode) -> [String; 13] {
        [
            self.row_id.clone(),
            self.row_type.label().to_owned(),
            self.family_id.clone(),
            self.family_size.to_string(),
            self.family_members.clone(),
            self.evidence.label().to_owned(),
            mode.label().to_owned(),
            self.theoretical_peptides.to_string(),
            self.theoretical_unique.to_string(),
            self.theoretical_shared.to_string(),
            self.observed_unique.to_string(),
            self.observed_shared.to_string(),
            self.samples_quantified.to_string(),
        ]
    }
}

const PIBAQ_EVIDENCE_HEADER: [&str; 13] = [
    "protein",
    "row_type",
    "family_id",
    "family_size",
    "family_members",
    "evidence",
    "shared_mode",
    "theoretical_peptides",
    "theoretical_unique",
    "theoretical_shared",
    "observed_unique",
    "observed_shared",
    "samples_quantified",
];

pub(crate) fn family_evidence_rows(
    family: &ProteinFamily,
    counts: &FamilyCounts,
    evidence: PibaqEvidence,
    output: &HashMap<CellKey, PibaqCell>,
    mode: PibaqSharedMode,
    proteins: &StringIdRegistry<ProteinId>,
) -> Vec<PibaqEvidenceRow> {
    let mut samples = HashMap::<ProteinId, usize>::new();
    for key in output.keys() {
        *samples.entry(key.protein).or_insert(0) += 1;
    }
    let quantified = |row_id: &str| {
        proteins
            .get(row_id)
            .and_then(|protein| samples.get(&protein).copied())
            .unwrap_or(0)
    };
    let mut rows = Vec::with_capacity(family.members.len() + 1);
    for (position, member) in family.members.iter().enumerate() {
        rows.push(member_evidence_row(
            family,
            counts,
            position,
            evidence,
            mode,
            quantified(member),
        ));
    }
    let row_id = family_row_id(family);
    let samples_quantified = if family.members.len() > 1 {
        quantified(&row_id)
    } else {
        0
    };
    if samples_quantified > 0 {
        rows.push(family_evidence_row(
            family,
            counts,
            evidence,
            row_id,
            samples_quantified,
        ));
    }
    rows
}

fn member_evidence_row(
    family: &ProteinFamily,
    counts: &FamilyCounts,
    position: usize,
    evidence: PibaqEvidence,
    mode: PibaqSharedMode,
    samples_quantified: usize,
) -> PibaqEvidenceRow {
    let theoretical_peptides = if mode == PibaqSharedMode::Unique {
        counts.theoretical_unique[position]
    } else {
        counts.theoretical_total[position]
    };
    PibaqEvidenceRow {
        row_id: family.members[position].clone(),
        row_type: if samples_quantified > 0 {
            PibaqRowType::Member
        } else {
            PibaqRowType::MemberNotQuantified
        },
        family_id: family.family_id.clone(),
        family_size: family.members.len(),
        family_members: family.members.join(";"),
        evidence,
        theoretical_peptides,
        theoretical_unique: counts.theoretical_unique[position],
        theoretical_shared: counts.theoretical_total[position]
            - counts.theoretical_unique[position],
        observed_unique: counts.observed_unique[position],
        observed_shared: counts.observed_shared[position],
        samples_quantified,
    }
}

fn family_evidence_row(
    family: &ProteinFamily,
    counts: &FamilyCounts,
    evidence: PibaqEvidence,
    row_id: String,
    samples_quantified: usize,
) -> PibaqEvidenceRow {
    let theoretical_unique = counts.theoretical_unique.iter().sum::<usize>();
    PibaqEvidenceRow {
        row_id,
        row_type: PibaqRowType::Family,
        family_id: family.family_id.clone(),
        family_size: family.members.len(),
        family_members: family.members.join(";"),
        evidence,
        theoretical_peptides: counts.owned_count,
        theoretical_unique,
        theoretical_shared: counts.owned_count.saturating_sub(theoretical_unique),
        observed_unique: counts.observed_unique.iter().sum(),
        observed_shared: counts.observed_shared_peptides(),
        samples_quantified,
    }
}

pub(crate) fn write_pibaq_evidence(
    path: &Path,
    rows: &[PibaqEvidenceRow],
    mode: PibaqSharedMode,
) -> Result<()> {
    create_parent_dir(path)?;
    let file = File::create(path).map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_writer(file);
    writer
        .write_record(PIBAQ_EVIDENCE_HEADER)
        .map_err(|source| csv_error(path, source))?;
    let mut sorted = rows.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        (&left.family_id, left.row_type, &left.row_id).cmp(&(
            &right.family_id,
            right.row_type,
            &right.row_id,
        ))
    });
    for row in sorted {
        writer
            .write_record(row.record(mode))
            .map_err(|source| csv_error(path, source))?;
    }
    writer.flush().map_err(|source| MokumeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}
