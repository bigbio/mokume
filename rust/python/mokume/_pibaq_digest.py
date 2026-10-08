"""Runtime pyOpenMS FASTA digestion for the Rust piBAQ kernel."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Set, Tuple

import pyopenms

from mokume.io.fasta import digest_fasta_full

ProteaseDB = getattr(pyopenms, "ProteaseDB")
ProteaseDigestion = getattr(pyopenms, "ProteaseDigestion")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def installed_protease_catalog() -> List[Dict[str, object]]:
    """Read the complete protease catalog from the installed pyOpenMS runtime."""
    database = ProteaseDB()
    names = []
    database.getAllNames(names)
    catalog = []
    for raw_name in names:
        enzyme = database.getEnzyme(raw_name)
        catalog.append(
            {
                "name": _text(enzyme.getName()),
                "regex": _text(enzyme.getRegEx()),
                "synonyms": sorted(_text(value) for value in enzyme.getSynonyms()),
            }
        )
    return sorted(catalog, key=lambda entry: str(entry["name"]))


def _catalog_hash(catalog: List[Dict[str, object]]) -> str:
    encoded = json.dumps(
        catalog,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pibaq_digest(
    request: Tuple[str, str, int, int, int],
) -> Tuple[Dict[str, Set[str]], Dict[str, object]]:
    """Digest a FASTA with the installed catalog and return Rust-ready payloads."""
    fasta, requested_enzyme, min_aa, max_aa, missed_cleavages = request
    if missed_cleavages < 0:
        raise ValueError("piBAQ missed cleavages must be non-negative")
    digestion = ProteaseDigestion()
    digestion.setEnzyme(requested_enzyme)
    digestion.setMissedCleavages(missed_cleavages)
    enzyme = _text(digestion.getEnzymeName())

    accession_peptides, _, _ = digest_fasta_full(
        fasta,
        enzyme,
        min_aa,
        max_aa,
        canonicalize_isoforms=True,
        compute_mw=False,
        missed_cleavages=missed_cleavages,
    )
    catalog = installed_protease_catalog()
    provenance = {
        "pyopenms_version": pyopenms.__version__,
        "enzyme": enzyme,
        "catalog_hash": _catalog_hash(catalog),
        "min_aa": min_aa,
        "max_aa": max_aa,
        "missed_cleavages": missed_cleavages,
    }
    return accession_peptides, provenance
