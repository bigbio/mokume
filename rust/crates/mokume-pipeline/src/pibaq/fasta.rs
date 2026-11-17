use crate::accessions::parse_protein_accession;
use crate::{HashMap, MokumeError, Path, Result, read_to_string};

/// Per-canonical monoisotopic molecular weights for the piBAQ TPA column.
///
/// Mirrors `mokume.io.fasta.digest_fasta_full(..., compute_mw=True)`: every
/// FASTA entry's MW is computed from its non-standard-stripped sequence, then
/// keyed by the canonical (isoform-collapsed) accession with first-seen
/// priority (the canonical entry's own MW wins because FASTA order is
/// preserved). Only consumed by `peptides2protein --tpa`.
pub(crate) fn load_fasta_mw(fasta: &Path) -> Result<HashMap<String, f64>> {
    let contents = read_to_string(fasta).map_err(|source| MokumeError::Io {
        path: fasta.to_path_buf(),
        source,
    })?;
    let mut mw_map = HashMap::<String, f64>::new();
    for (identifier, sequence) in parse_fasta_entries(&contents) {
        let accession = canonicalize_isoform(parse_protein_accession(&identifier));
        let weight = protein_mono_weight(&strip_nonstandard_amino_acids(&sequence));
        mw_map.entry(accession).or_insert(weight);
    }
    Ok(mw_map)
}

fn parse_fasta_entries(contents: &str) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    let mut current_identifier: Option<String> = None;
    let mut current_sequence = String::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            if let Some(identifier) = current_identifier.replace(fasta_identifier(header)) {
                entries.push((identifier, std::mem::take(&mut current_sequence)));
            }
        } else if current_identifier.is_some() {
            current_sequence.push_str(line);
        }
    }
    if let Some(identifier) = current_identifier {
        entries.push((identifier, current_sequence));
    }
    entries
}

fn fasta_identifier(header: &str) -> String {
    header
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn strip_nonstandard_amino_acids(sequence: &str) -> String {
    sequence
        .chars()
        .filter_map(|aa| {
            let aa = aa.to_ascii_uppercase();
            (!matches!(aa, 'X' | 'B' | 'Z' | 'J' | 'U' | 'O')).then_some(aa)
        })
        .collect()
}

fn canonicalize_isoform(accession: &str) -> String {
    let Some((base, suffix)) = accession.rsplit_once('-') else {
        return accession.to_owned();
    };
    if !base.is_empty() && suffix.chars().all(|character| character.is_ascii_digit()) {
        base.to_owned()
    } else {
        accession.to_owned()
    }
}

/// Monoisotopic mass of water (`H2O`), the OpenMS `EmpiricalFormula("H2O")`
/// mono weight added once per peptide/protein by `AASequence::getMonoWeight`.
const WATER_MONO_MASS: f64 = 18.0105650638;

/// Per-residue monoisotopic masses (the internal residue masses pyOpenMS
/// returns from `Residue::getMonoWeight(Internal)`), used to reproduce
/// `AASequence::getMonoWeight` for the piBAQ TPA `MolecularWeight` column.
/// The full-protein mono weight is `WATER_MONO_MASS + sum(residue masses)`.
const fn residue_mono_mass(residue: char) -> Option<f64> {
    let mass = match residue {
        'A' => 71.03711415949999,
        'C' => 103.00918488949999,
        'D' => 115.02694415949999,
        'E' => 129.0425942233,
        'F' => 147.06841428710004,
        'G' => 57.0214640957,
        'H' => 137.0589122233,
        'I' => 113.0840643509,
        'K' => 128.09496338280002,
        'L' => 113.0840643509,
        'M' => 131.04048501709997,
        'N' => 114.04292819140001,
        'P' => 97.05276422329999,
        'Q' => 128.05857825520002,
        'R' => 156.1011113828,
        'S' => 87.0320291595,
        'T' => 101.04767922330001,
        'V' => 99.0684142871,
        'W' => 186.079313319,
        'Y' => 163.06332928710003,
        _ => return None,
    };
    Some(mass)
}

/// Monoisotopic molecular weight of a protein sequence, matching the pyOpenMS
/// `AASequence.fromString(seq).getMonoWeight()` value the Python piBAQ TPA path
/// reads via `digest_fasta_full(..., compute_mw=True)`. The accumulation order
/// (water first, then residues left to right) mirrors how OpenMS sums the
/// empirical formula so the result is bit-for-bit comparable within 1e-9 for
/// the peptide/protein lengths this path handles. Residues outside the
/// 20 standard amino acids are skipped, consistent with the upstream
/// `_strip_nonstandard_aa` applied before the digest.
pub(crate) fn protein_mono_weight(sequence: &str) -> f64 {
    let mut total = WATER_MONO_MASS;
    for residue in sequence.chars() {
        if let Some(mass) = residue_mono_mass(residue) {
            total += mass;
        }
    }
    total
}
