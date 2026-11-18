use crate::IrsStat;
use crate::accessions::parse_protein_accession;
use crate::aggregation::FeatureAggregation;
use crate::aggregation::IntensityFactors;
use crate::aggregation::MaxLfqFeatureAggregation;
use crate::aggregation::ProteinFamily;
use crate::exports::ExportIons;
use crate::exports::IntermediateMeasurement;
use crate::exports::PeptideExport;
use crate::exports::PeptideExportNames;
use crate::features_to_peptides::ProteinFdrReport;
use crate::features_to_peptides::ProteinFdrRunCount;
use crate::features_to_proteins::run_features_to_proteins;
use crate::from_peptides::run_lfq_from_peptides;
use crate::irs::{
    irs_by_mixture_scale_from_runs, irs_global_scale_from_runs, irs_mixture_first_token,
    irs_two_stage_scale_from_runs, resolve_irs_autodetect_channel,
};
use crate::normalization_factors::{
    NormalizationFactorCollector, has_removed_accession, load_normalization_proteins,
    passes_feature_filter,
};
use crate::peptide_normalization::{sum_canonical_cell, sum_peptide_values};
use crate::pibaq::allocation::FamilyCounts;
use crate::pibaq::allocation::finalize_family_resolved;
use crate::pibaq::evidence::PibaqRowType;
use crate::pibaq::{PibaqEvidence, classify_evidence};
use crate::plex::sample_plex;
use crate::protein_matrix::batch::{
    factorize_batch_labels, resolve_batch_method, resolve_reference_batch, validate_batch_sizes,
    validate_combat_design,
};
use crate::protein_matrix::covariates::{
    batch_column_values_for_samples, extract_sdrf_covariates, match_sdrf_column,
};
use crate::protein_matrix::{ProteinMatrix, ProteinValues};
use crate::ratio::{detect_reference_samples, resolve_irs_reference_samples};
use crate::state::CellKey;
use crate::state::DirectLfqCellKey;
use crate::state::PeptideCellKey;
use crate::validation::{
    expand_de_contrasts_file, fails_protein_fdr, resolve_de_method, validate_de_ensemble_options,
    validate_features_to_proteins, validate_implemented_subset,
};

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

use mokume_core::{
    BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, FeatureToProteinsConfig,
    FilterConfig, ImputationConfig, InputConfig, IrsConfig, MaxLfqConfig, MokumeError,
    NormalizationConfig, OutputConfig, OutputFormat, PeptideId, PibaqConfig, PibaqFamilyRows,
    PibaqSharedMode, ProteinId, QuantMethod, RatioConfig, RuntimeConfig, SampleId,
    StringIdRegistry, TmtConfig, TransferredFeatures,
};
use mokume_io::{QpxFeatureRecord, SdrfRawTable, SdrfTable};
use mokume_normalization::{RunCellKey, RunNormalizationTransform, SampleNormalizationMethod};
use shared::{
    assert_close, base_config, batch_test_matrix, combat_on_log2, contextual_peptide_export,
    covariate_fixture, existing_dummy_path, lfq_aggregation, lfq_fixture_protein_ids,
    lfq_min_unique_fixture, multi_mixture_runs, paralog_observations, resolved_values,
    two_member_family_fixture, unique_suffix, write_temp_sdrf,
};

mod shared {
    use super::*;

    pub(crate) fn contextual_peptide_export(
        table: &SdrfTable,
        names: &super::PeptideExportNames<'_>,
    ) -> mokume_core::Result<super::PeptideExport> {
        let mut export = super::PeptideExport {
            run_mode: true,
            ..super::PeptideExport::default()
        };
        for protein in [0, 1] {
            for sample in [0, 1] {
                for canonical in [0, 1] {
                    for record in table.records() {
                        export.push(
                            super::IntermediateMeasurement {
                                protein: ProteinId::new(protein),
                                sample: SampleId::new(sample),
                                canonical: PeptideId::new(canonical),
                                peptide: PeptideId::new(0),
                                intensity: 8.0,
                                protein_name: names.proteins[protein as usize],
                                peptide_canonical: names.canonicals[canonical as usize],
                                sample_name: names.samples[sample as usize],
                                run_file_name: &record.data_file,
                                sdrf_record: Some(record),
                            },
                            8.0,
                        )?;
                    }
                }
            }
        }
        Ok(export)
    }

    type FamilyFixture = (
        super::ProteinFamily,
        HashSet<String>,
        HashMap<String, HashSet<String>>,
        HashMap<String, HashSet<String>>,
    );

    pub(crate) fn two_member_family_fixture() -> FamilyFixture {
        let family = super::ProteinFamily {
            family_id: "A".to_owned(),
            members: vec!["A".to_owned(), "B".to_owned()],
        };
        let owned = HashSet::from([
            "unique_a".to_owned(),
            "unique_b".to_owned(),
            "shared".to_owned(),
        ]);
        let accession_peptides = HashMap::from([
            (
                "A".to_owned(),
                HashSet::from(["unique_a".to_owned(), "shared".to_owned()]),
            ),
            (
                "B".to_owned(),
                HashSet::from(["unique_b".to_owned(), "shared".to_owned()]),
            ),
        ]);
        let peptide_accessions = HashMap::from([
            ("unique_a".to_owned(), HashSet::from(["A".to_owned()])),
            ("unique_b".to_owned(), HashSet::from(["B".to_owned()])),
            (
                "shared".to_owned(),
                HashSet::from(["A".to_owned(), "B".to_owned()]),
            ),
        ]);
        (family, owned, accession_peptides, peptide_accessions)
    }

    pub(crate) fn resolved_values(
        mode: PibaqSharedMode,
        family_rows: PibaqFamilyRows,
        evidence: super::PibaqEvidence,
        observations: &[(String, SampleId, f64)],
    ) -> HashMap<(String, u32), f64> {
        let (family, owned, accession_peptides, peptide_accessions) = two_member_family_fixture();
        let counts = super::FamilyCounts::new(
            &family,
            observations,
            &owned,
            &accession_peptides,
            &peptide_accessions,
        );
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let output = super::finalize_family_resolved(
            &family,
            observations,
            &counts,
            evidence,
            mode,
            family_rows,
            &mut proteins,
        );
        output
            .into_iter()
            .filter_map(|(key, cell)| {
                proteins
                    .resolve(key.protein)
                    .map(|name| ((name.to_owned(), key.sample.get()), cell.pibaq))
            })
            .collect()
    }

    pub(crate) fn paralog_observations() -> Vec<(String, SampleId, f64)> {
        vec![
            ("unique_a".to_owned(), SampleId::new(1), 100.0),
            ("unique_b".to_owned(), SampleId::new(1), 50.0),
            ("shared".to_owned(), SampleId::new(1), 300.0),
            ("unique_a".to_owned(), SampleId::new(2), 80.0),
            ("shared".to_owned(), SampleId::new(2), 200.0),
        ]
    }

    // Multi-mixture synthetic TMT fixture shared by the by_mixture / two_stage
    // scope tests. Two mixtures, two techreplicates each (passed in the stable
    // run-name-sorted order the collect side produces):
    //   mixA_1 -> [100,200,300]  (median 200, mean 200)   techrep 1
    //   mixA_2 -> [ 50,150,250]  (median 150, mean 150)   techrep 2
    //   mixB_3 -> [400,500,600]  (median 500, mean 500)   techrep 3
    //   mixB_4 -> [ 80,120,400]  (median 120, mean 200)   techrep 4
    // Values are locked against Python `get_irs_scaling_factors` run on the
    // matching pyarrow fixture (see scratchpad oracle).
    pub(crate) fn multi_mixture_runs() -> Vec<(String, String, Vec<f64>)> {
        vec![
            (
                "mixA_1".to_owned(),
                "mixA".to_owned(),
                vec![100.0, 200.0, 300.0],
            ),
            (
                "mixA_2".to_owned(),
                "mixA".to_owned(),
                vec![50.0, 150.0, 250.0],
            ),
            (
                "mixB_3".to_owned(),
                "mixB".to_owned(),
                vec![400.0, 500.0, 600.0],
            ),
            (
                "mixB_4".to_owned(),
                "mixB".to_owned(),
                vec![80.0, 120.0, 400.0],
            ),
        ]
    }

    pub(crate) fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    pub(crate) fn write_temp_sdrf(
        name: &str,
        contents: &str,
    ) -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
        let directory = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_irs_autodetect_{name}_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?;
        let path = directory.path().join("autodetect.sdrf.tsv");
        fs::write(&path, contents)?;
        Ok((directory, path))
    }

    pub(crate) fn covariate_fixture() -> Result<SdrfRawTable, Box<dyn std::error::Error>> {
        // Mirror the Python oracle fixture (`gen_oracle.py`).
        let input = concat!(
            "source name\tcharacteristics[sex]\tcharacteristics[organism part]\tbatch\tconst_col\n",
            "S1\tmale\tliver\tb1\tx\n",
            "S2\tfemale\tliver\tb1\tx\n",
            "S3\tmale\tbrain\tb2\tx\n",
            "S4\tfemale\tbrain\tb2\tx\n",
        );
        Ok(SdrfRawTable::from_reader(input.as_bytes())?)
    }

    /// Direct log2-scale ComBat of linear rows, returned to the linear scale.
    pub(crate) fn combat_on_log2(rows: &[Vec<f64>], batch: &[usize]) -> Vec<Vec<f64>> {
        let log2 = rows
            .iter()
            .map(|row| row.iter().map(|value| value.log2()).collect())
            .collect::<Vec<Vec<f64>>>();
        mokume_stats::batch::combat(
            &log2,
            batch,
            None,
            mokume_stats::batch::ComBatParams::default(),
        )
        .into_iter()
        .map(|row| row.into_iter().map(f64::exp2).collect())
        .collect()
    }

    pub(crate) fn batch_test_matrix(
        rows: &[(&str, [Option<f64>; 4])],
    ) -> (ProteinMatrix, Vec<SampleId>, Vec<ProteinId>) {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A-1", "A-2", "B-1", "B-2"]
            .iter()
            .map(|name| samples.get_or_insert(name).expect("sample id"))
            .collect::<Vec<_>>();
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let mut cells = std::collections::HashMap::new();
        let mut ids = Vec::new();
        for (name, values) in rows {
            let protein = proteins.get_or_insert(name).expect("protein id");
            ids.push(protein);
            for (sample, value) in sample_ids.iter().zip(values) {
                if let Some(value) = value {
                    cells.insert(
                        CellKey {
                            protein,
                            sample: *sample,
                        },
                        *value,
                    );
                }
            }
        }
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: ids.iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: std::collections::HashMap::new(),
            precision: std::collections::HashMap::new(),
            values: ProteinValues::Cells(cells),
        };
        (matrix, sample_ids, ids)
    }

    pub(crate) fn existing_dummy_path(
        name: &str,
    ) -> Result<(tempfile::TempDir, PathBuf), Box<dyn std::error::Error>> {
        let directory = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_pipeline_{name}_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?;
        let path = directory.path().join("dummy");
        fs::write(&path, [])?;
        Ok((directory, path))
    }

    pub(crate) fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    }

    pub(crate) fn base_config(parquet: PathBuf) -> FeatureToProteinsConfig {
        FeatureToProteinsConfig {
            input: InputConfig {
                parquet: Some(parquet),
                msstats: None,
                psm: None,
                sdrf: None,
                run: None,
                fasta: None,
                transferred_features: TransferredFeatures::Keep,
                transferred_weight: None,
            },
            output: OutputConfig {
                protein_matrix: PathBuf::from("protein.csv"),
                export_peptides: None,
                export_unnormalized_peptides: None,
                export_unimputed: None,
                export_ions: None,
                export_precision: None,
                format: OutputFormat::PythonCompatible,
            },
            filtering: FilterConfig::default(),
            normalization: NormalizationConfig::default(),
            quantification: QuantMethod::MaxLfq,
            topn_peptides: 3,
            maxlfq: MaxLfqConfig::default(),
            pibaq: PibaqConfig::default(),
            directlfq: DirectLfqConfig::default(),
            batch: BatchCorrectionConfig::default(),
            irs: IrsConfig::default(),
            tmt: TmtConfig::default(),
            coverage_threshold: None,
            sample_correlation_threshold: None,
            protein_fdr_threshold: None,
            ratio: RatioConfig::default(),
            imputation: ImputationConfig::default(),
            differential_expression: DifferentialExpressionConfig::default(),
            runtime: RuntimeConfig {
                memory: None,
                threads: None,
            },
        }
    }

    // Both LFQ methods apply `min_unique_peptides` per `(protein, sample)`.
    pub(crate) fn lfq_aggregation(
        method: super::QuantMethod,
        directlfq_sums: std::collections::HashMap<super::DirectLfqCellKey, f64>,
    ) -> super::FeatureAggregation {
        let peptide_traces = directlfq_sums
            .into_iter()
            .map(|(key, intensity)| {
                (
                    super::PeptideCellKey {
                        protein: key.protein,
                        sample: key.sample,
                        peptide: key.canonical,
                    },
                    intensity,
                )
            })
            .collect();
        super::FeatureAggregation::Lfq {
            method,
            route_to_directlfq: method == super::QuantMethod::DirectLfq,
            maxlfq_min_ratio_count: 2,
            stabilize: false,
            directlfq_min_nonan: 1,
            directlfq_num_samples_quadratic: 50,
            directlfq_normalize_samples: true,
            traces: super::MaxLfqFeatureAggregation {
                peptide_traces,
                ..super::MaxLfqFeatureAggregation::default()
            },
            cell_precision: false,
            cached_directlfq: None,
        }
    }

    pub(crate) struct LfqMinUniqueFixture {
        pub(crate) scattered: ProteinId,
        pub(crate) dense: ProteinId,
        sums: HashMap<super::DirectLfqCellKey, f64>,
        pub(crate) allowed_cells: HashSet<super::CellKey>,
        pub(crate) canonical_peptides: StringIdRegistry<PeptideId>,
    }

    pub(crate) fn lfq_min_unique_fixture() -> LfqMinUniqueFixture {
        let scattered = ProteinId::new(1);
        let dense = ProteinId::new(2);
        let pep_a = PeptideId::new(10);
        let pep_b = PeptideId::new(11);
        let s1 = SampleId::new(1);
        let s2 = SampleId::new(2);
        let s3 = SampleId::new(3);
        let s4 = SampleId::new(4);

        // `scattered`: two distinct peptides, each in two samples, but never
        // both in the same sample (pep_a in s1/s2, pep_b in s3/s4).
        // `dense`: two distinct peptides in every sample.
        let mut sums = std::collections::HashMap::new();
        let cell = |protein, canonical, sample| super::DirectLfqCellKey {
            protein,
            canonical,
            sample,
        };
        sums.insert(cell(scattered, pep_a, s1), 1000.0);
        sums.insert(cell(scattered, pep_a, s2), 1100.0);
        sums.insert(cell(scattered, pep_b, s3), 2000.0);
        sums.insert(cell(scattered, pep_b, s4), 2200.0);
        for &sample in &[s1, s2, s3, s4] {
            sums.insert(cell(dense, pep_a, sample), 1000.0);
            sums.insert(cell(dense, pep_b, sample), 1100.0);
        }

        // `allowed_cells` is the per-(protein, sample) gate: only `dense` ever
        // carries two distinct peptides in a single sample.
        let allowed_cells = [s1, s2, s3, s4]
            .into_iter()
            .map(|sample| super::CellKey {
                protein: dense,
                sample,
            })
            .collect::<HashSet<_>>();
        // Register canonical sequences so `pep_a` (id 10) and `pep_b` (id 11)
        // resolve; `seq_NN` strings keep alphabetical order aligned with the id
        // order, so the DirectLFQ row order is unchanged by the sequence-rank
        // sort under test elsewhere.
        let mut canonical_peptides = super::StringIdRegistry::<PeptideId>::new();
        for index in 0..=11 {
            let _ = canonical_peptides.get_or_insert(&format!("seq_{index:02}"));
        }
        LfqMinUniqueFixture {
            scattered,
            dense,
            sums,
            allowed_cells,
            canonical_peptides,
        }
    }

    pub(crate) fn lfq_fixture_protein_ids(
        method: super::QuantMethod,
        fixture: &LfqMinUniqueFixture,
    ) -> mokume_core::Result<HashSet<ProteinId>> {
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        Ok(lfq_aggregation(method, fixture.sums.clone())
            .finalize(
                &fixture.allowed_cells,
                &mut proteins,
                &HashMap::new(),
                &fixture.canonical_peptides,
                &StringIdRegistry::<SampleId>::new(),
            )?
            .0
            .protein_ids())
    }
}
mod accessions;
mod aggregation;
mod batch;
mod correlation;
mod covariates;
mod exports;
mod features_to_peptides;
mod from_peptides;
mod irs;
mod normalization_factors;
mod peptide_normalization;
mod pibaq;
mod plex;
mod protein_matrix;
mod ratio;
mod state;
mod validation;
