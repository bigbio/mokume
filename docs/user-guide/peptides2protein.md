# peptides2protein: Protein Quantification

The `peptides2protein` command quantifies proteins from normalized peptide data. It supports multiple quantification methods and is the second step of the two-step pipeline.

## Basic Usage

=== "CLI"

    ```bash
    # piBAQ (default, requires FASTA)
    mokume quantify peptides2protein --quant-method pibaq \
        -f proteome.fasta \
        -p peptides.csv \
        -o proteins-pibaq.tsv

    # TopN (no FASTA needed)
    mokume quantify peptides2protein --quant-method top3 \
        -p peptides.csv \
        -o proteins-top3.tsv

    # MaxLFQ with parallelization
    mokume quantify peptides2protein --quant-method maxlfq \
        --threads 4 \
        -p peptides.csv \
        -o proteins-maxlfq.tsv
    ```

=== "Python (wheel)"

    The wheel wrapper validates documented keyword arguments, maps them to the
    command's exact CLI flags, and runs the same kernel in-process:

    ```python
    import mokume

    # piBAQ (requires FASTA)
    mokume.peptides2protein(
        quant_method="pibaq",
        fasta="proteome.fasta",
        peptides="peptides.csv",
        output="proteins-pibaq.tsv",
    )

    # TopN (no FASTA needed)
    mokume.peptides2protein(quant_method="top3", peptides="peptides.csv",
                            output="proteins-top3.tsv")
    ```

## Methods

### piBAQ

piBAQ is Mokume's paralog-aware extension of iBAQ. For each sample,
shared-peptide signal is allocated in proportion to mapped members'
proteotypic-peptide intensities; if all mapped members lack that signal, it is
split equally. Each shared intensity is allocated exactly once. See
[Quantification Methods](../concepts/quantification.md#pibaq-paralog-aware-ibaq)
for the underlying algorithm. **Requires a FASTA file**.

```bash
mokume quantify peptides2protein --quant-method pibaq \
    -f proteome.fasta \
    -p peptides.csv \
    --enzyme Trypsin \
    --normalize \
    --output proteins-pibaq.tsv
```

The output adds three metadata columns -- `FamilyId`, `FamilySize`, `EvidenceLevel` -- so users can audit family support. `family_only` means that no family member reaches the minimum anchor threshold; it does not duplicate one family-level piBAQ across all members. Every reported piBAQ remains a per-member estimate with a per-member theoretical-peptide denominator.

#### Family Discovery Tuning

Families are discovered automatically by collapsing UniProt isoform suffixes (`-2`, `-3`, ...) onto the canonical entry and then grouping proteins on the shared-peptide graph. Two CLI flags tune the auto-discovery; both are optional and rarely need adjustment.

```bash
# Lower the shared-peptide threshold for very tightly homologous families
mokume quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv \
    --min-shared 1 -o out.tsv

# Pin specific families with an audit-friendly YAML override
mokume quantify peptides2protein --quant-method pibaq -f proteome.fasta -p peptides.csv \
    --families families.yaml -o out.tsv
```

The YAML schema is:

```yaml
families:
  - name: ACT
    members: [P60709, P63261, P68133]   # canonical accessions only
  - name: HIST_H2A
    members: [P0C0S5, Q96QV6, P04908]
```

#### Full piBAQ with TPA and ProteomicRuler

```bash
mokume quantify peptides2protein \
    --quant-method pibaq \
    -f proteome.fasta \
    -p peptides.csv \
    --enzyme Trypsin \
    --normalize \
    --tpa \
    --ruler \
    --ploidy 2 \
    --cpc 200 \
    --organism human \
    --output proteins-pibaq.tsv \
    --qc-report QC.pdf
```

```python
import mokume

mokume.peptides2protein(
    fasta="proteome.fasta",
    peptides="peptides.csv",
    enzyme="Trypsin",
    normalize=True,
    tpa=True,
    ruler=True,
    ploidy=2,
    cpc=200,
    organism="human",
    output="proteins-pibaq.tsv",
    qc_report="QC.pdf",
    min_aa=7,
    max_aa=30,
)
```

!!! note "piBAQ enzyme coverage and the QC report"
    piBAQ reads the complete protease catalog from the installed pyOpenMS runtime,
    including context-dependent rules, unspecific cleavage, and no cleavage. Python
    supplies the theoretical-peptide map; Rust performs shared-peptide allocation,
    family handling, denominators, TPA, normalization, and output. With the
    plotting dependencies installed, `--qc-report QC.pdf` writes the PDF after
    the native table has been produced; the path itself enables rendering.

### TopN

Averages the N most intense peptides per protein per sample.

```bash
# Top3 (most common) -- the named method from Silva et al. 2006
mokume quantify peptides2protein --quant-method top3 -p peptides.csv -o out.tsv

# Top5
mokume quantify peptides2protein --quant-method top5 -p peptides.csv -o out.tsv

# Top10
mokume quantify peptides2protein --quant-method top10 -p peptides.csv -o out.tsv
```

N is spelled in the method name, so any N works the same way and there is no
companion option to keep in sync.

`--weight-column` names a column of per-row weights in (0, 1]. Each protein and
sample then takes its N rows by weight, highest first, and by intensity within
a weight; for example, a lower weight for rows that match-between-runs
transferred makes top3 take identified peptides first.

```python
import mokume

mokume.peptides2protein(quant_method="top3", peptides="peptides.csv", output="out.tsv")
mokume.peptides2protein(quant_method="top5", peptides="peptides.csv", output="out.tsv")
```

### MaxLFQ

Maximal peptide ratio extraction (Cox et al., 2014): the log ratio of each sample pair is the median log ratio of the peptide species both samples share (at least `--maxlfq-min-ratio-count`, default 2), and the protein profile is the least-squares fit to all pair ratios. Pass normalized intensities; this command does not normalize. Samples without a supported pair ratio are missing from the output.

With `--weight-column`, the log ratio of each shared species is weighted by the product of its two rows' weights and the pair ratio is the weighted median (the midpoint when half the total weight falls exactly between two ratios). Equal weights reproduce the unweighted result exactly; the summed-intensity ratio that `--stabilize` blends in stays unweighted.

```bash
mokume quantify peptides2protein --quant-method maxlfq \
    --threads 4 \
    -p peptides.csv \
    -o proteins-maxlfq.tsv
```

```python
import mokume

mokume.peptides2protein(quant_method="maxlfq", threads=4, peptides="peptides.csv",
                        output="proteins-maxlfq.tsv")
```

### DirectLFQ

Uses hierarchical normalization with variance-guided pairwise alignment, native in the Rust kernel (no extra dependency).

```bash
mokume quantify peptides2protein --quant-method directlfq \
    --directlfq-min-nonan 2 \
    -p peptides.csv \
    -o proteins-directlfq.tsv
```

```python
import mokume

mokume.peptides2protein(quant_method="directlfq", directlfq_min_nonan=2, peptides="peptides.csv",
                        output="proteins-directlfq.tsv")
```

### Sum

Sums all peptide intensities per protein per sample.

```bash
mokume quantify peptides2protein --quant-method sum \
    -p peptides.csv \
    -o proteins-sum.tsv
```

```python
import mokume

mokume.peptides2protein(quant_method="sum", peptides="peptides.csv",
                        output="proteins-sum.tsv")
```

## CLI Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `-f/--fasta` | none | FASTA file (required for piBAQ) |
| `-p/--peptides` | required | Input peptide intensity file |
| `--quant-method` | `pibaq` | Quantification method: pibaq, `top<N>` (for example top3 or top5), maxlfq, sum, directlfq |
| `--enzyme` | `Trypsin` | Enzyme for in-silico digestion |
| `--normalize` | off | Normalize quantification values |
| `--min-aa` | 7 | Minimum amino acid length |
| `--max-aa` | 30 | Maximum amino acid length |
| `--tpa` | off | Calculate TPA (piBAQ only) |
| `--ruler` | off | Use ProteomicRuler (piBAQ only) |
| `--ploidy` | 2 with `--ruler` | Positive ploidy number (ruler only) |
| `--organism` | `human` with `--ruler` | Organism for histone data (ruler only) |
| `--cpc` | 200 with `--ruler` | Positive cellular protein concentration in g/L (ruler only) |
| `-t/--threads` | automatic | Positive worker count for MaxLFQ and DirectLFQ |
| `--weight-column` | none | Column of per-row weights in (0, 1] for MaxLFQ pair medians and `top<N>` selection |
| `--directlfq-min-nonan` | 1 | Min non-NaN values (DirectLFQ) |
| `--families` | none | Optional YAML file with explicit family overrides (piBAQ only) |
| `--min-shared` | 2 | Minimum shared peptides for auto-family discovery (piBAQ only) |
| `-o/--output` | required | Output file path |
| `--qc-report` | none | Path for the piBAQ QC report PDF; providing it enables rendering |

Method-specific options are rejected for other methods instead of being
accepted and ignored. For example, `--threads` applies only to LFQ methods,
`--directlfq-min-nonan` only to DirectLFQ, `--weight-column` only to MaxLFQ and
`top<N>`, and FASTA/digestion/family/QC options only to piBAQ. `--ploidy`, `--organism`, and `--cpc` require `--ruler`, and invalid or
non-positive values are rejected.
