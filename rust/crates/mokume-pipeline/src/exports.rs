mod peptide;
use crate::aggregation::IntensityFactors;
use crate::state::CellKey;
use crate::{
    FeatureToProteinsConfig, HashMap, HashSet, Path, PathBuf, PeptideId, ProteinId, Result,
    SampleId, SdrfRecord, StringIdRegistry,
};

#[derive(Debug, Default)]
pub(crate) struct IntermediateExports {
    pub(crate) peptides: bool,
    unnormalized_peptides: bool,
    /// Shared by both exports, whose keys and ions always match.
    pub(crate) rows: PeptideExport,
}

/// `RazorPeptideFilter` mode (Python `preprocessing/filters/protein.py:334-378`).
/// A razor peptide is a canonical mapping to more than one protein within a
/// sample (the pipeline runs per-sample). `Keep` is the no-op default; `Remove`
/// drops every row of a razor peptide; `AssignToTop` keeps only the rows of the
/// protein with the most unique peptides in the sample, breaking ties by
/// first-appearance order (the protein seen earliest in parquet row order wins),
/// mirroring Python `protein.py:358-378` (pandas `unique()` order + `max`'s
/// keep-first tie-break).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum RazorHandling {
    #[default]
    Keep,
    Remove,
    AssignToTop,
}

/// Map the configured `razor_peptide_handling` string to [`RazorHandling`].
/// `validate_filter_pipeline_subset` rejects everything but
/// `keep`/`remove`/`assign_to_top` before this runs, so any other value falls
/// back to the no-op `Keep`.
pub(crate) fn parse_razor_handling(handling: &str) -> RazorHandling {
    match handling {
        "remove" => RazorHandling::Remove,
        "assign_to_top" => RazorHandling::AssignToTop,
        _ => RazorHandling::Keep,
    }
}

#[derive(Debug, Default)]
pub(crate) struct PeptideExport {
    pub(crate) rows: HashMap<PeptideExportKey, ExportIons>,
    pub(crate) contexts: Vec<PeptideExportContext>,
    pub(crate) context_ids: HashMap<(u32, u32, Option<u32>, i64), u32>,
    pub(crate) context_labels: StringIdRegistry<u32>,
    /// When set, the export keys on `(Run, TechReplicate)` in addition to the
    /// sample, reproducing Python `aggregation_level == "run"`. Off (sample
    /// level) for the `features2proteins --export-peptides` path, which never
    /// requests run granularity.
    pub(crate) run_mode: bool,
    /// `RazorPeptideFilter` handling (Python protein.py:334-378), applied at
    /// materialization after the `min_unique` gate (Python orders MinPeptide
    /// before Razor). `Keep` for the `features2proteins --export-peptides` path.
    pub(crate) razor_handling: RazorHandling,
    /// First-appearance order of `ProteinName` per `(sample, canonical)`, used
    /// only by `RazorHandling::AssignToTop` to break ties the same way Python's
    /// `max(group[ProteinName].unique(), key=...)` does (pandas `unique()`
    /// preserves first-occurrence order, and `max` keeps the first maximum). The
    /// `rows` map is a `HashMap`, so it cannot recover this order; `push` is
    /// called in parquet row order, so we record the order here as rows arrive.
    /// Populated only when `razor_handling == AssignToTop`, leaving the other
    /// modes' push path untouched.
    pub(crate) razor_order: HashMap<(SampleId, String), Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PeptideExportKey {
    pub(crate) protein: ProteinId,
    pub(crate) sample: SampleId,
    pub(crate) canonical: PeptideId,
    pub(crate) context: u32,
}

/// Per-ion maxima for `--export-peptides` and `--export-unnormalized-peptides`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IonValues {
    pub(crate) intensity: f64,
    pub(crate) unnormalized: f64,
}

/// A canonical cell's ions in `PeptideId` order. Most cells hold a single ion,
/// so it stays inline in the row table.
#[derive(Debug, PartialEq)]
pub(crate) enum ExportIons {
    One(PeptideId, IonValues),
    Many(Box<[(PeptideId, IonValues)]>),
}

#[derive(Debug)]
pub(crate) struct PeptideExportContext {
    bio_replicate: String,
    condition: String,
    run: Option<String>,
    tech_replicate: i64,
}

pub(crate) type CanonicalExportValues = HashMap<(ProteinId, SampleId, PeptideId), f64>;

pub(crate) struct PeptideExportNames<'a> {
    pub(crate) proteins: Vec<&'a str>,
    pub(crate) samples: Vec<&'a str>,
    pub(crate) canonicals: Vec<&'a str>,
}

/// Borrowed metadata in the exact lexical ordering used by both export formats.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PeptideExportRow<'a> {
    protein_name: &'a str,
    peptide_canonical: &'a str,
    sample_name: &'a str,
    bio_replicate: &'a str,
    condition: &'a str,
    run: Option<&'a str>,
    tech_replicate: i64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IntermediateMeasurement<'a> {
    pub(crate) protein: ProteinId,
    pub(crate) sample: SampleId,
    pub(crate) peptide: PeptideId,
    pub(crate) canonical: PeptideId,
    pub(crate) intensity: f64,
    pub(crate) protein_name: &'a str,
    pub(crate) peptide_canonical: &'a str,
    pub(crate) sample_name: &'a str,
    pub(crate) run_file_name: &'a str,
    pub(crate) sdrf_record: Option<&'a SdrfRecord>,
}

/// Options that select how the peptide intermediate is serialised. The
/// `features2proteins --export-peptides` path uses the all-default value (CSV
/// only, sample level); `features2peptides` overrides them from its config.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PeptideExportOptions {
    pub(crate) log2: bool,
    /// Additional parquet sibling to write (Python `--save_parquet`).
    pub(crate) save_parquet: bool,
}

impl IntermediateExports {
    pub(crate) fn has_peptides(&self) -> bool {
        self.peptides || self.unnormalized_peptides
    }

    pub(crate) fn new(config: &FeatureToProteinsConfig) -> Self {
        Self {
            peptides: config.output.export_peptides.is_some(),
            unnormalized_peptides: config.output.export_unnormalized_peptides.is_some(),
            rows: PeptideExport::default(),
        }
    }

    /// Variant of [`IntermediateExports::new`] that requests run-level peptide
    /// keys, used only by `features2peptides --aggregation_level run`.
    pub(crate) fn new_with_run_mode(config: &FeatureToProteinsConfig, run_mode: bool) -> Self {
        Self {
            peptides: config.output.export_peptides.is_some(),
            unnormalized_peptides: false,
            rows: PeptideExport {
                run_mode,
                ..PeptideExport::default()
            },
        }
    }

    pub(crate) fn push(
        &mut self,
        measurement: IntermediateMeasurement<'_>,
        raw_intensity: f64,
        intensity_factors: Option<&IntensityFactors>,
    ) -> Result<()> {
        if !self.has_peptides() {
            return Ok(());
        }
        let unnormalized = match intensity_factors {
            Some(factors) if self.unnormalized_peptides => factors.normalize_run(
                raw_intensity,
                measurement.sample_name,
                measurement.run_file_name,
            ),
            _ => raw_intensity,
        };
        self.rows.push(measurement, unnormalized)
    }

    pub(crate) fn write(
        &self,
        config: &FeatureToProteinsConfig,
        allowed_cells: &HashSet<CellKey>,
        low_frequency_peptides: &HashSet<(String, String)>,
        options: PeptideExportOptions,
        dataset_normalized: Option<&CanonicalExportValues>,
        names: &PeptideExportNames<'_>,
    ) -> Result<()> {
        let keys = self
            .rows
            .ordered_keys(allowed_cells, low_frequency_peptides, names);
        // Both exports keep the same keys and columns. Dataset-level
        // normalization drops BioReplicate/Condition, as in Python's melt.
        let omit_replicate_condition = dataset_normalized.is_some();
        for (path, unnormalized, normalized) in [
            (
                config.output.export_peptides.as_deref(),
                false,
                dataset_normalized,
            ),
            (
                config.output.export_unnormalized_peptides.as_deref(),
                true,
                None,
            ),
        ] {
            let Some(path) = path else {
                continue;
            };
            let rows =
                self.rows
                    .exported_rows(&keys, unnormalized, options.log2, normalized, names);
            self.rows
                .write_csv(path, rows, omit_replicate_condition, options.save_parquet)?;
        }
        Ok(())
    }
}

/// Derive the parquet sibling of the CSV output path, mirroring Python's
/// `WriteParquetTask`, which does `os.path.splitext(output)[0] + ".parquet"`.
fn parquet_sibling(path: &Path) -> PathBuf {
    path.with_extension("parquet")
}

impl PeptideExportNames<'_> {
    fn row<'a>(
        &'a self,
        key: &PeptideExportKey,
        export: &'a PeptideExport,
    ) -> PeptideExportRow<'a> {
        let context = &export.contexts[key.context as usize];
        PeptideExportRow {
            protein_name: self.proteins[key.protein.get() as usize],
            peptide_canonical: self.canonicals[key.canonical.get() as usize],
            sample_name: self.samples[key.sample.get() as usize],
            bio_replicate: &context.bio_replicate,
            condition: &context.condition,
            run: context.run.as_deref(),
            tech_replicate: context.tech_replicate,
        }
    }
}

impl IonValues {
    fn merge(&mut self, other: Self) {
        if other.intensity > self.intensity {
            self.intensity = other.intensity;
        }
        if other.unnormalized > self.unnormalized {
            self.unnormalized = other.unnormalized;
        }
    }

    fn value(self, unnormalized: bool) -> f64 {
        if unnormalized {
            self.unnormalized
        } else {
            self.intensity
        }
    }
}

impl ExportIons {
    fn merge(&mut self, peptide: PeptideId, values: IonValues) {
        match self {
            Self::One(current, current_values) if *current == peptide => {
                current_values.merge(values);
            }
            Self::One(current, current_values) => {
                let first = (*current, *current_values);
                let second = (peptide, values);
                let (low, high) = if first.0 < second.0 {
                    (first, second)
                } else {
                    (second, first)
                };
                *self = Self::Many(Box::new([low, high]));
            }
            Self::Many(ions) => match ions.binary_search_by_key(&peptide, |(id, _)| *id) {
                Ok(index) => ions[index].1.merge(values),
                Err(index) => {
                    let mut grown = std::mem::take(ions).into_vec();
                    grown.reserve_exact(1);
                    grown.insert(index, (peptide, values));
                    *ions = grown.into_boxed_slice();
                }
            },
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (PeptideId, IonValues)> + '_ {
        let (one, many) = match self {
            Self::One(peptide, values) => (Some((*peptide, *values)), &[][..]),
            Self::Many(ions) => (None, &ions[..]),
        };
        one.into_iter().chain(many.iter().copied())
    }
}
