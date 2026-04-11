"""
Pipeline flow modules.

Each flow handles a different quantification paradigm:
- standard: iBAQ, TopN, sum, median (input_level="peptides")
- ratio: PS protocol log2 ratios (input_level="psms")
- directlfq: DirectLFQ package delegation (input_level="peptides_raw")

Flow dispatch is handled by `pipeline/runner.py`.
"""

from mokume.pipeline.flows import standard, ratio, directlfq

__all__ = ["standard", "ratio", "directlfq"]
