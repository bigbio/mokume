"""
FASTA file handling utilities.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pyopenms

from mokume.core.constants import build_accession_map, get_accession

AASequence = getattr(pyopenms, "AASequence")
FASTAFile = getattr(pyopenms, "FASTAFile")
ProteaseDigestion = getattr(pyopenms, "ProteaseDigestion")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_fasta(fasta_path: str) -> List:
    """
    Load a FASTA file and return the list of protein entries.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.

    Returns
    -------
    List
        List of FASTA protein entries.
    """
    fasta_proteins = []
    FASTAFile().load(fasta_path, fasta_proteins)
    return fasta_proteins


def digest_protein(
    sequence: str,
    enzyme: str,
    min_aa: int = 7,
    max_aa: int = 30,
) -> List[str]:
    """
    Digest a protein sequence using a specified enzyme.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence.
    enzyme : str
        Name of the enzyme to use for digestion.
    min_aa : int
        Minimum peptide length to include.
    max_aa : int
        Maximum peptide length to include.

    Returns
    -------
    List[str]
        List of peptide sequences.
    """
    digestor = ProteaseDigestion()
    digestor.setEnzyme(enzyme)

    digest = []
    digestor.digest(AASequence.fromString(sequence), digest, min_aa, max_aa)
    return [str(pep.toString()) for pep in digest]


_NONSTANDARD_AA = {"X", "B", "Z", "J", "U", "O"}


def _strip_nonstandard_aa(sequence: str) -> str:
    """Remove non-standard amino acids from a protein sequence."""
    for aa in _NONSTANDARD_AA:
        sequence = sequence.replace(aa, "")
    return sequence


def canonicalize_isoform(accession: str) -> str:
    """Strip UniProt isoform suffix to obtain the canonical accession.

    UniProt convention encodes alternative splice isoforms as
    ``<primary_accession>-<isoform_number>`` (e.g. ``P05067-2``). The
    canonical entry shares its sequence with ``<primary_accession>-1`` and
    is also exported simply as ``<primary_accession>``. Non-canonical
    isoforms often share most of their peptides with the canonical entry,
    which makes them indistinguishable in shotgun proteomics. Collapsing
    isoforms to the canonical accession is the conventional UniProt
    accession behaviour and matches the gpGrouper protein grouping
    contract (Saltzman et al. 2018 MCP 17:2270).

    Plain accessions without a numeric suffix and accessions whose suffix
    is not purely numeric (e.g. ``P02768ups``) are returned unchanged.

    Parameters
    ----------
    accession : str
        Protein accession in any UniProt-compatible form.

    Returns
    -------
    str
        Canonical accession (suffix stripped) or ``accession`` if no
        UniProt-style isoform suffix is present.
    """
    if "-" not in accession:
        return accession
    base, suffix = accession.rsplit("-", 1)
    if not base or not suffix.isdigit():
        return accession
    return base


def canonicalize_isoforms_map(
    accession_to_peptides: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Collapse a per-accession peptide map onto canonical accessions.

    For every accession the peptide set is unioned onto the canonical
    accession produced by :func:`canonicalize_isoform`. The second return
    value records the alias mapping so callers can rewrite downstream
    tables (e.g. peptide-protein rows) without re-running digestion.

    Parameters
    ----------
    accession_to_peptides : dict[str, set[str]]
        Mapping from accession to its digested peptide set.

    Returns
    -------
    tuple
        ``(canonical_to_peptides, original_to_canonical)``.
    """
    canonical_to_peptides: Dict[str, Set[str]] = {}
    original_to_canonical: Dict[str, str] = {}
    for accession, peps in accession_to_peptides.items():
        canonical = canonicalize_isoform(accession)
        original_to_canonical[accession] = canonical
        if canonical in canonical_to_peptides:
            canonical_to_peptides[canonical].update(peps)
        else:
            canonical_to_peptides[canonical] = set(peps)
    return canonical_to_peptides, original_to_canonical


def invert_peptide_index(
    accession_to_peptides: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """Build the peptide → accessions inverted index.

    Parameters
    ----------
    accession_to_peptides : dict[str, set[str]]
        Per-accession peptide set, typically from :func:`digest_fasta_full`.

    Returns
    -------
    dict[str, set[str]]
        For every peptide, the set of accessions that contain it after
        digestion.
    """
    peptide_to_accessions: "defaultdict[str, Set[str]]" = defaultdict(set)
    for accession, peps in accession_to_peptides.items():
        for pep in peps:
            peptide_to_accessions[pep].add(accession)
    return dict(peptide_to_accessions)


def extract_fasta(
    fasta: str,
    enzyme: str,
    proteins: List[str],
    min_aa: int,
    max_aa: int,
    tpa: bool = False,
) -> Tuple[Dict[str, int], Dict[str, float], Set[str]]:
    """
    Extract protein information from a FASTA file using a specified enzyme for digestion.

    The number of enzyme-specific theoretical peptides reported per protein
    is the count of peptides that are *unique across the full FASTA database*
    (proteotypic). Counting all theoretical peptides — including those shared
    with homologous proteins — would systematically deflate iBAQ for large
    homologous families (myosin, tubulin, actin, histone, keratin) by 3-20x
    *when paired with the legacy proteotypic-only numerator* (shared peptides
    discarded), because the denominator would then be inflated by shared
    peptides while the numerator only contains proteotypic intensity.

    .. note::
        This proteotypic-only count is the **legacy ibaqpy denominator** and
        is symmetric only with a proteotypic-only numerator. The default
        Mokume piBAQ path
        (:func:`mokume.quantification.pibaq.compute_pibaq`), which re-allocates
        shared-peptide intensity into the numerator and therefore uses a
        *total-potential* denominator (proteotypic + shared) derived from
        :func:`digest_fasta_full` instead of this function. The two
        denominator conventions are mutually exclusive — each is only correct
        next to its matching numerator. This function is retained for direct
        API consumers and is not called by the piBAQ pipeline.

    Sequences with nonstandard amino acids (X/B/Z/J/U/O) are stripped before
    digestion. Callers asking for proteins that are absent from the FASTA are
    silently omitted from ``uniquepepcounts`` and ``mw_dict``; only when
    *no* requested protein is found does the function raise ``ValueError``.

    Parameters
    ----------
    fasta : str
        Path to the FASTA file containing protein sequences.
    enzyme : str
        Name of the enzyme used for protein digestion.
    proteins : List[str]
        List of protein accessions to search for in the FASTA file.
    min_aa : int
        Minimum number of amino acids for peptides to be considered.
    max_aa : int
        Maximum number of amino acids for peptides to be considered.
    tpa : bool
        If True, calculate molecular weights for Total Protein Approach (TPA).

    Returns
    -------
    Tuple[Dict[str, int], Dict[str, float], Set[str]]
        A tuple containing:
        - uniquepepcounts: Dictionary mapping caller-provided protein names to
          cross-protein unique theoretical peptide counts.
        - mw_dict: Dictionary mapping caller-provided protein names to
          molecular weights (empty if ``tpa=False``).
        - found_proteins: Set of caller-provided protein names (i.e. the
          original entries from ``proteins``) that resolved to a FASTA entry.

    Raises
    ------
    ValueError
        If none of the specified proteins are found in the FASTA file.
    """
    acc_to_originals, protein_accessions = build_accession_map(proteins)

    fasta_proteins = load_fasta(fasta)
    digestor = ProteaseDigestion()
    digestor.setEnzyme(enzyme)

    accession_to_peptides, peptide_to_accessions, accession_to_mw = _digest_full_fasta(
        fasta_proteins,
        digestor,
        min_aa,
        max_aa,
        tpa,
        keep_peptides_for=protein_accessions,
    )
    uniquepepcounts, mw_dict, found_proteins = _select_unique_counts_for_callers(
        protein_accessions,
        acc_to_originals,
        accession_to_peptides,
        peptide_to_accessions,
        accession_to_mw,
        tpa,
    )

    if len(found_proteins) == 0:
        raise ValueError(
            "None of the specified proteins were found in the FASTA file. "
            "Please check that the protein accessions match."
        )

    logger.info("Found %d proteins in FASTA file", len(found_proteins))
    return uniquepepcounts, mw_dict, found_proteins


def _digest_one(sequence, digestor: ProteaseDigestion, min_aa: int, max_aa: int):
    """Digest a single sequence and return its peptide set.

    Uses ``pep.toString()`` rather than ``str(pep)`` to keep peptide string
    keys stable across pyOpenMS versions (``__str__`` has varied; ``toString``
    is part of the documented API).
    """
    digest: List = []
    digestor.digest(sequence, digest, min_aa, max_aa)
    return {pep.toString() for pep in digest}


def _digest_full_fasta(
    fasta_proteins,
    digestor: ProteaseDigestion,
    min_aa: int,
    max_aa: int,
    tpa: bool,
    keep_peptides_for: Set[str],
):
    """Digest every protein in the FASTA once and return the inverted index
    needed to compute cross-protein peptide uniqueness.

    The inverted index ``peptide_to_accessions`` is populated for every FASTA
    entry (cross-protein uniqueness requires it). To avoid wasting memory on
    large proteomes, ``accession_to_peptides`` and ``accession_to_mw`` are
    only retained for accessions in ``keep_peptides_for``.

    Per-entry parse, digest, and weight failures are logged and skipped so
    that one bad sequence cannot poison quantification of the rest of the
    proteome.
    """
    accession_to_peptides: Dict[str, Set[str]] = {}
    peptide_to_accessions: "defaultdict[str, Set[str]]" = defaultdict(set)
    accession_to_mw: Dict[str, float] = {}
    for entry in fasta_proteins:
        accession = get_accession(entry.identifier)
        try:
            aa_sequence = AASequence.fromString(_strip_nonstandard_aa(entry.sequence))
            peps = _digest_one(aa_sequence, digestor, min_aa, max_aa)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping %s: %s", accession, exc)
            continue
        for pep in peps:
            peptide_to_accessions[pep].add(accession)
        if accession in keep_peptides_for:
            accession_to_peptides[accession] = peps
            if tpa:
                try:
                    accession_to_mw[accession] = aa_sequence.getMonoWeight()
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Skipping MW for %s: %s", accession, exc)
    return accession_to_peptides, peptide_to_accessions, accession_to_mw


def digest_fasta_full(
    fasta: str,
    enzyme: str,
    min_aa: int,
    max_aa: int,
    *,
    canonicalize_isoforms: bool = True,
    compute_mw: bool = False,
    missed_cleavages: int = 0,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, float]]:
    """Digest every protein in a FASTA and return the full peptide indices.

    Unlike :func:`extract_fasta` (which only emits per-protein unique-peptide
    counts), this function exposes the underlying ``accession_to_peptides`` and
    ``peptide_to_accessions`` maps so that callers can perform downstream
    analyses such as family discovery (shared-peptide connected components),
    proportional shared-peptide allocation, or sequence-aware protein
    inference. It is the entry point for the piBAQ family discovery layer.

    Parameters
    ----------
    fasta : str
        Path to the FASTA file containing protein sequences.
    enzyme : str
        Name of the enzyme used for protein digestion (e.g. ``Trypsin``).
    min_aa : int
        Minimum number of amino acids for peptides to be retained.
    max_aa : int
        Maximum number of amino acids for peptides to be retained.
    canonicalize_isoforms : bool, optional
        When ``True`` (default), UniProt isoform suffixes (``-2``, ``-3``,
        ...) are stripped and peptide sets are unioned onto the canonical
        accession via :func:`canonicalize_isoforms_map`. Set to ``False``
        to preserve every original FASTA accession verbatim.
    compute_mw : bool, optional
        When ``True``, the third return value is populated with per-accession
        monoisotopic molecular weights. Defaults to ``False`` to save the
        digest cost when MW is not needed.
    missed_cleavages : int, optional
        Maximum missed cleavages in the theoretical digest (default ``0``).

    Returns
    -------
    tuple
        ``(accession_to_peptides, peptide_to_accessions, accession_to_mw)``
        where every accession is canonical when ``canonicalize_isoforms`` is
        ``True``. ``accession_to_mw`` is empty when ``compute_mw`` is
        ``False``.
    """
    fasta_proteins = load_fasta(fasta)
    digestor = ProteaseDigestion()
    digestor.setEnzyme(enzyme)
    digestor.setMissedCleavages(missed_cleavages)

    accession_to_peptides: Dict[str, Set[str]] = {}
    accession_to_mw: Dict[str, float] = {}
    for entry in fasta_proteins:
        accession = get_accession(entry.identifier)
        try:
            aa_sequence = AASequence.fromString(_strip_nonstandard_aa(entry.sequence))
            peps = _digest_one(aa_sequence, digestor, min_aa, max_aa)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping %s: %s", accession, exc)
            continue
        if accession in accession_to_peptides:
            accession_to_peptides[accession].update(peps)
        else:
            accession_to_peptides[accession] = peps
        if compute_mw and accession not in accession_to_mw:
            try:
                accession_to_mw[accession] = aa_sequence.getMonoWeight()
            except (ValueError, RuntimeError) as exc:
                logger.warning("Skipping MW for %s: %s", accession, exc)

    if canonicalize_isoforms:
        accession_to_peptides, alias_map = canonicalize_isoforms_map(
            accession_to_peptides
        )
        if compute_mw and accession_to_mw:
            canonical_mw: Dict[str, float] = {}
            for original, mw in accession_to_mw.items():
                canonical = alias_map.get(original, original)
                # Keep the first MW observed per canonical (canonical entry's
                # own MW takes priority when present, since iteration order
                # preserves the FASTA order).
                canonical_mw.setdefault(canonical, mw)
            accession_to_mw = canonical_mw

    peptide_to_accessions = invert_peptide_index(accession_to_peptides)
    logger.info(
        "Digested %d %saccessions into %d distinct peptides",
        len(accession_to_peptides),
        "canonical " if canonicalize_isoforms else "",
        len(peptide_to_accessions),
    )
    return accession_to_peptides, peptide_to_accessions, accession_to_mw


def _emit_mw_for_caller(
    originals: List[str],
    accession: str,
    accession_to_mw: Dict[str, float],
    mw_dict: Dict[str, float],
) -> None:
    """Copy the molecular weight of ``accession`` to each caller name when known."""
    if accession not in accession_to_mw:
        return
    mw = accession_to_mw[accession]
    for orig in originals:
        mw_dict[orig] = mw


def _select_unique_counts_for_callers(
    protein_accessions: Set[str],
    acc_to_originals: Dict[str, List[str]],
    accession_to_peptides: Dict[str, Set[str]],
    peptide_to_accessions: Dict[str, Set[str]],
    accession_to_mw: Dict[str, float],
    tpa: bool,
):
    """Emit per-caller cross-protein unique peptide counts (and MW)."""
    found_proteins: Set[str] = set()
    uniquepepcounts: Dict[str, int] = {}
    mw_dict: Dict[str, float] = {}
    for accession in protein_accessions:
        peps = accession_to_peptides.get(accession)
        if peps is None:
            continue
        originals = acc_to_originals[accession]
        found_proteins.update(originals)
        unique_count = sum(
            1 for pep in peps if peptide_to_accessions[pep] == {accession}
        )
        for orig in originals:
            uniquepepcounts[orig] = unique_count
        if tpa:
            _emit_mw_for_caller(originals, accession, accession_to_mw, mw_dict)
    return uniquepepcounts, mw_dict, found_proteins


def get_protein_molecular_weights(
    fasta: str,
    proteins: List[str],
) -> Dict[str, float]:
    """
    Calculate molecular weights for a list of proteins from a FASTA file.

    Parameters
    ----------
    fasta : str
        Path to the FASTA file.
    proteins : List[str]
        List of protein accessions.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping protein accessions to molecular weights.
    """
    acc_to_originals, protein_accessions = build_accession_map(proteins)

    fasta_proteins = load_fasta(fasta)
    mw_dict: Dict[str, float] = {}

    for entry in fasta_proteins:
        accession = get_accession(entry.identifier)
        if accession in protein_accessions:
            aa_sequence = AASequence.fromString(_strip_nonstandard_aa(entry.sequence))
            mw = aa_sequence.getMonoWeight()
            for orig in acc_to_originals[accession]:
                mw_dict[orig] = mw

    return mw_dict
