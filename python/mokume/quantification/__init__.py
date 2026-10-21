"""
Protein quantification methods for the mokume package.

This module provides implementations for various protein quantification
methods including piBAQ, Top3, TopN, MaxLFQ, DirectLFQ, and AllPeptides.

DirectLFQ is an optional dependency. Install with:
    pip install mokume-py[directlfq]
"""

import importlib
import re
from typing import TYPE_CHECKING

from mokume._lazy import module_api
from mokume.quantification.base import ProteinQuantificationMethod

if TYPE_CHECKING:
    from mokume.quantification.all_peptides import AllPeptidesQuantification
    from mokume.quantification.directlfq import (
        DirectLFQQuantification,
        is_directlfq_available,
    )
    from mokume.quantification.maxlfq import MaxLFQQuantification
    from mokume.quantification.pibaq import (
        ConcentrationWeightByProteomicRuler,
        compute_pibaq,
        extract_fasta,
        normalize_pibaq,
        peptides_to_protein,
    )
    from mokume.quantification.ratio import RatioQuantification
    from mokume.quantification.spectral_count import (
        PeptideCountQuantification,
        SpectralCountQuantification,
    )
    from mokume.quantification.tmt_abundance import TMTAbundanceQuantification
    from mokume.quantification.tmt_reporter import TMTReporterIntensityQuantification
    from mokume.quantification.top3 import Top3Quantification
    from mokume.quantification.topn import TopNQuantification

__all__ = [
    # Base class
    "ProteinQuantificationMethod",
    # piBAQ
    "peptides_to_protein",
    "compute_pibaq",
    "normalize_pibaq",
    "extract_fasta",
    "ConcentrationWeightByProteomicRuler",
    # Quantification methods
    "Top3Quantification",
    "TopNQuantification",
    "MaxLFQQuantification",
    "AllPeptidesQuantification",
    "RatioQuantification",
    "TMTAbundanceQuantification",
    "TMTReporterIntensityQuantification",
    "PeptideCountQuantification",
    "SpectralCountQuantification",
    # DirectLFQ (optional)
    "DirectLFQQuantification",
    "is_directlfq_available",
    # Factory function
    "get_quantification_method",
    "list_quantification_methods",
]


_LAZY_EXPORTS = {
    "peptides_to_protein": ("mokume.quantification.pibaq", "peptides_to_protein"),
    "compute_pibaq": ("mokume.quantification.pibaq", "compute_pibaq"),
    "normalize_pibaq": ("mokume.quantification.pibaq", "normalize_pibaq"),
    "extract_fasta": ("mokume.quantification.pibaq", "extract_fasta"),
    "ConcentrationWeightByProteomicRuler": (
        "mokume.quantification.pibaq",
        "ConcentrationWeightByProteomicRuler",
    ),
    "Top3Quantification": (
        "mokume.quantification.top3",
        "Top3Quantification",
    ),
    "TopNQuantification": (
        "mokume.quantification.topn",
        "TopNQuantification",
    ),
    "MaxLFQQuantification": (
        "mokume.quantification.maxlfq",
        "MaxLFQQuantification",
    ),
    "AllPeptidesQuantification": (
        "mokume.quantification.all_peptides",
        "AllPeptidesQuantification",
    ),
    "RatioQuantification": (
        "mokume.quantification.ratio",
        "RatioQuantification",
    ),
    "TMTAbundanceQuantification": (
        "mokume.quantification.tmt_abundance",
        "TMTAbundanceQuantification",
    ),
    "TMTReporterIntensityQuantification": (
        "mokume.quantification.tmt_reporter",
        "TMTReporterIntensityQuantification",
    ),
    "SpectralCountQuantification": (
        "mokume.quantification.spectral_count",
        "SpectralCountQuantification",
    ),
    "PeptideCountQuantification": (
        "mokume.quantification.spectral_count",
        "PeptideCountQuantification",
    ),
    "DirectLFQQuantification": (
        "mokume.quantification.directlfq",
        "DirectLFQQuantification",
    ),
    "is_directlfq_available": (
        "mokume.quantification.directlfq",
        "is_directlfq_available",
    ),
}

_SIMPLE_METHODS = {
    "all": "AllPeptidesQuantification",
    "sum": "AllPeptidesQuantification",
    "abd": "TMTAbundanceQuantification",
    "abundance": "TMTAbundanceQuantification",
    "tmtabundance": "TMTAbundanceQuantification",
    "intensity": "TMTReporterIntensityQuantification",
    "reporter": "TMTReporterIntensityQuantification",
    "tmtreporterintensity": "TMTReporterIntensityQuantification",
    "peptidecount": "PeptideCountQuantification",
    "peptide_count": "PeptideCountQuantification",
    "spectralcount": "SpectralCountQuantification",
    "spectral_count": "SpectralCountQuantification",
    "count": "SpectralCountQuantification",
}


__getattr__, __dir__ = module_api(_LAZY_EXPORTS, globals(), __name__)


def _maxlfq_options(kwargs: dict) -> dict:
    """Resolve the supported MaxLFQ constructor options and legacy aliases."""
    return {
        "min_peptides": kwargs.get("min_peptides", 2),
        "min_ratio_count": kwargs.get("min_ratio_count"),
        "stabilize": kwargs.get("stabilize", False),
        "threads": kwargs.get("threads", kwargs.get("n_jobs", -1)),
        "verbose": kwargs.get("verbose", 0),
    }


def get_quantification_method(method: str, **kwargs) -> ProteinQuantificationMethod:
    """
    Get a quantification method instance by name.

    Parameters
    ----------
    method : str
        Name of the quantification method. One of:
        'topN' (where N is any number, e.g., 'top3', 'top5', 'top10'),
        'pibaq', 'maxlfq', 'directlfq', 'all', 'sum',
        'abd' / 'abundance', 'intensity' / 'reporter', 'peptide_count',
        'spectral_count', 'ratio'.
    **kwargs
        Additional arguments passed to the quantification method constructor.

        For MaxLFQ:
            - min_ratio_count: int (default 2; shared species per sample pair)
            - stabilize: bool (default False; Cox Eq. 5 large-ratio stabilization)
            - min_peptides: legacy alias for min_ratio_count
            - threads: int (default -1, all cores)

        For TopN:
            - n: int (default 3, can also be parsed from method name)

        For DirectLFQ:
            - min_nonan: int (default 1)
            - num_cores: int (default None)
            - deactivate_normalization: bool (default False)

        For piBAQ:
            - fasta: str (required before quantification)
            - enzyme: str (default "Trypsin")

        For 'ratio' (TMT ratio quantification):
            - reference_samples: list[str] (required)
            - sample_to_plex: dict[str, str] (required)
            - fraction_merge_method: str (default 'mean')

        For 'abd' / 'abundance' and 'intensity' / 'reporter':
            No additional arguments.

    Returns
    -------
    ProteinQuantificationMethod
        An instance of the requested quantification method.

    Raises
    ------
    ValueError
        If the method name is not recognized.
    ImportError
        If DirectLFQ is requested but not installed.

    Examples
    --------
    >>> method = get_quantification_method("maxlfq", min_ratio_count=2, threads=4)
    >>> result = method.quantify(peptide_df, ...)

    >>> # TopN with any N
    >>> method = get_quantification_method("top5")  # Top 5 peptides
    >>> method = get_quantification_method("top10")  # Top 10 peptides

    >>> # DirectLFQ requires optional install
    >>> method = get_quantification_method("directlfq", min_nonan=2)
    """
    method_lower = method.lower()
    if kwargs.get("stabilize", False) and method_lower != "maxlfq":
        raise ValueError("stabilize only applies to MaxLFQ")

    if method_lower == "pibaq":
        registry_module = importlib.import_module("mokume.core.registry")
        registry = getattr(registry_module, "PluginRegistry")
        return registry.get("quantification", "pibaq", **kwargs)

    # Handle topN methods (top3, top5, top10, etc.)
    if method_lower.startswith("top"):
        match = re.match(r"top(\d+)", method_lower)
        if match:
            n = int(match.group(1))
        else:
            n = kwargs.get("n", 3)
        return __getattr__("TopNQuantification")(n=n)

    if method_lower == "maxlfq":
        return __getattr__("MaxLFQQuantification")(**_maxlfq_options(kwargs))

    if method_lower == "directlfq":
        return __getattr__("DirectLFQQuantification")(
            min_nonan=kwargs.get("min_nonan", 1),
            num_cores=kwargs.get("num_cores", None),
            deactivate_normalization=kwargs.get("deactivate_normalization", False),
        )

    method_class = _SIMPLE_METHODS.get(method_lower)
    if method_class is not None:
        return __getattr__(method_class)()

    if method_lower == "ratio":
        try:
            reference_samples = kwargs["reference_samples"]
            sample_to_plex = kwargs["sample_to_plex"]
        except KeyError as exc:
            raise ValueError(
                "RatioQuantification requires 'reference_samples' and "
                "'sample_to_plex' kwargs."
            ) from exc
        return __getattr__("RatioQuantification")(
            reference_samples=reference_samples,
            sample_to_plex=sample_to_plex,
            fraction_merge_method=kwargs.get("fraction_merge_method", "mean"),
        )

    available = (
        "pibaq, topN (e.g., top3, top5, top10), maxlfq, directlfq, all/sum, "
        "abd/abundance, intensity/reporter, peptide_count, spectral_count, ratio"
    )
    raise ValueError(
        f"Unknown quantification method: {method}. Available methods: {available}"
    )


def list_quantification_methods() -> dict:
    """
    List all available quantification methods.

    Returns
    -------
    dict
        Dictionary mapping method names to their availability status.
        For optional dependencies, shows whether they are installed.

    Examples
    --------
    >>> from mokume.quantification import list_quantification_methods
    >>> methods = list_quantification_methods()
    >>> "pibaq" in methods
    True
    """
    return {
        "pibaq": True,
        "top3": True,
        "topn": True,
        "maxlfq": True,
        "directlfq": __getattr__("is_directlfq_available")(),
        "sum": True,
        "abd": True,
        "intensity": True,
        "ratio": True,
        "peptide_count": True,
        "spectral_count": True,
    }
