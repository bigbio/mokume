use super::*;
use crate::protein_matrix::batch::detect_batches_for_method;

mod cases {
    use super::*;

    #[test]
    fn combat_design_rejects_batch_confounded_covariate() {
        let batch = [0, 0, 1, 1];
        let confounded = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
        assert!(validate_combat_design(&batch, &confounded, None).is_err());
        let orthogonal = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
        assert!(validate_combat_design(&batch, &orthogonal, None).is_ok());
    }

    #[test]
    fn resolve_batch_method_rejects_unknown_values() {
        use mokume_stats::batch::BatchDetectionMethod;
        // Known names parse (case/separator-insensitive).
        assert_eq!(
            super::resolve_batch_method("Sample-Prefix").ok(),
            Some(BatchDetectionMethod::SamplePrefix)
        );
        assert_eq!(
            super::resolve_batch_method("COLUMN").ok(),
            Some(BatchDetectionMethod::ExplicitColumn)
        );
        assert!(super::resolve_batch_method("nonsense").is_err());
    }

    #[test]
    fn detect_batches_for_method_in_pipeline_flow() {
        use mokume_stats::batch::BatchDetectionMethod;
        let names = ["PXD001-S1", "PXD001-S2", "PXD002-S1", "PXD002-S2"];

        // sample_prefix factorizes the sample prefixes.
        let labels =
            super::detect_batches_for_method(BatchDetectionMethod::SamplePrefix, &names, None);
        assert_eq!(labels.ok(), Some(vec![0, 0, 1, 1]));

        // Without run_info, the pipeline never invokes `run` here (the caller
        // returns an error), but the helper still surfaces the `run_info`
        // requirement for `run` -- mirroring Python's `ValueError`.
        let run = super::detect_batches_for_method(BatchDetectionMethod::RunName, &names, None);
        assert!(run.is_err());

        // The `column` path validates the explicit value length.
        let mismatched = ["A".to_owned(), "B".to_owned()];
        let bad = super::detect_batches_for_method(
            BatchDetectionMethod::ExplicitColumn,
            &names,
            Some(&mismatched),
        );
        assert!(bad.is_err());
    }

    #[test]
    fn validate_batch_sizes_requires_two_batches_each_two_samples() {
        assert_eq!(
            validate_batch_sizes(vec![0, 0, 1, 1]),
            Some(vec![0, 0, 1, 1])
        );
        assert_eq!(validate_batch_sizes(vec![0, 0, 0]), None); // a single batch
        assert_eq!(validate_batch_sizes(vec![0, 0, 1]), None); // batch 1 has one sample
    }

    #[test]
    fn resolves_original_batch_label_before_numeric_compatibility() {
        let labels = ["0", "0", "batch-b", "batch-b"].map(str::to_owned);
        let encoded = [0, 0, 1, 1];
        assert_eq!(
            resolve_reference_batch("batch-b", &labels, &encoded).ok(),
            Some(1)
        );
        assert_eq!(
            resolve_reference_batch("0", &labels, &encoded).ok(),
            Some(0)
        );
        assert!(resolve_reference_batch("missing", &labels, &encoded).is_err());
    }

    #[test]
    fn apply_batch_correction_corrects_complete_rows_and_keeps_incomplete()
    -> Result<(), Box<dyn std::error::Error>> {
        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let (mut matrix, sample_ids, [p1, p2, p3]) = incomplete_batch_matrix(p1_row, p2_row)?;

        matrix.apply_batch_correction(
            &BatchCorrectionConfig {
                enabled: true,
                ..BatchCorrectionConfig::default()
            },
            None,
            false,
            false,
        )?;

        // Complete rows match a direct ComBat call over log2([P1, P2]) with batch
        // [0,0,1,1], returned to the linear scale.
        let expected = combat_on_log2(&[p1_row.to_vec(), p2_row.to_vec()], &[0, 0, 1, 1]);
        for (row, protein) in [p1, p2].into_iter().enumerate() {
            for (col, sample) in sample_ids.iter().enumerate() {
                let got = matrix.value(protein, *sample).expect("corrected value");
                assert!((got - expected[row][col]).abs() < 1e-9 * expected[row][col]);
            }
        }
        // Incomplete row P3 is untouched, and its missing cell stays missing.
        for (col, sample) in sample_ids.iter().take(3).enumerate() {
            let got = matrix.value(p3, *sample).expect("p3 value");
            assert!((got - [7.0, 8.0, 9.0][col]).abs() < 1e-12);
        }
        assert!(matrix.value(p3, sample_ids[3]).is_none());
        Ok(())
    }

    #[test]
    fn apply_batch_correction_skips_non_positive_rows_and_keeps_undefined_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let config = BatchCorrectionConfig {
            enabled: true,
            ..BatchCorrectionConfig::default()
        };
        // P2 holds a zero (DirectLFQ's missing value): it is not a complete row,
        // so it stays as it is and only P1 is corrected. With a single complete
        // row ComBat cannot estimate its priors and returns NaN; P1 then keeps
        // its measured values rather than becoming missing.
        let (mut matrix, samples, ids) = batch_test_matrix(&[
            ("P1", [Some(10.0), Some(12.0), Some(30.0), Some(33.0)]),
            ("P2", [Some(5.0), Some(0.0), Some(20.0), Some(19.0)]),
        ]);
        matrix.apply_batch_correction(&config, None, false, false)?;
        for (protein, row) in ids
            .iter()
            .zip([[10.0, 12.0, 30.0, 33.0], [5.0, 0.0, 20.0, 19.0]])
        {
            for (sample, expected) in samples.iter().zip(row) {
                assert_eq!(matrix.value(*protein, *sample), Some(expected));
            }
        }

        // Log2-scale values (abd, ratio) are corrected as they are, negatives included.
        let rows = [vec![-1.0, -0.5, 2.0, 2.5], vec![0.5, 1.0, 3.0, 2.0]];
        let (mut log2, samples, ids) = log2_batch_matrix(&rows);
        log2.apply_batch_correction(&config, None, false, true)?;
        let expected = mokume_stats::batch::combat(
            &rows,
            &[0, 0, 1, 1],
            None,
            mokume_stats::batch::ComBatParams::default(),
        );
        for (protein, row) in ids.iter().zip(&expected) {
            for (sample, value) in samples.iter().zip(row) {
                let got = log2.value(*protein, *sample).expect("corrected value");
                assert!((got - value).abs() < 1e-9);
            }
        }
        Ok(())
    }

    #[test]
    fn apply_batch_correction_column_method_reads_sdrf_and_errors_on_missing()
    -> Result<(), Box<dyn std::error::Error>> {
        // Sample names that are each their own `sample_prefix` (no '-', no
        // `[a-z]+\d+_`), so prefix detection would see four singleton batches and
        // no-op. The explicit `column` method reads the SDRF `batch_id` column,
        // grouping them into two batches of two so correction actually runs --
        // proving the column path is wired, not the prefix path.
        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let (dir, sdrf_path) = column_batch_sdrf()?;

        let column_config = |method: &str, column: Option<&str>| BatchCorrectionConfig {
            enabled: true,
            method: method.to_owned(),
            column: column.map(ToOwned::to_owned),
            ..BatchCorrectionConfig::default()
        };

        let (mut matrix, sample_ids, p1, p2) =
            two_row_batch_matrix(["X1", "X2", "X3", "X4"], &p1_row, &p2_row)?;
        matrix.apply_batch_correction(
            &column_config("column", Some("batch_id")),
            Some(&sdrf_path),
            false,
            false,
        )?;
        // Corrected rows match a direct log2 ComBat over the column-derived batch [0,0,1,1].
        let expected = combat_on_log2(&[p1_row.to_vec(), p2_row.to_vec()], &[0, 0, 1, 1]);
        for (row, protein) in [p1, p2].into_iter().enumerate() {
            for (col, sample) in sample_ids.iter().enumerate() {
                let got = matrix.value(protein, *sample).expect("corrected value");
                assert!((got - expected[row][col]).abs() < 1e-9 * expected[row][col]);
            }
        }

        // An absent column errors like Python's `_detect_explicit_batches(None)`.
        let (mut missing, ..) = two_row_batch_matrix(["X1", "X2", "X3", "X4"], &p1_row, &p2_row)?;
        assert!(
            missing
                .apply_batch_correction(
                    &column_config("column", Some("nope")),
                    Some(&sdrf_path),
                    false,
                    false
                )
                .is_err()
        );

        // `run` has no run-level mapping in the protein-matrix flow -> error.
        let (mut run_matrix, ..) =
            two_row_batch_matrix(["X1", "X2", "X3", "X4"], &p1_row, &p2_row)?;
        assert!(
            run_matrix
                .apply_batch_correction(&column_config("run", None), Some(&sdrf_path), false, false)
                .is_err()
        );

        fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn apply_batch_correction_rejects_removed_and_unknown_methods()
    -> Result<(), Box<dyn std::error::Error>> {
        let p1_row = [10.0, 12.0, 30.0, 33.0];
        let p2_row = [5.0, 6.0, 20.0, 19.0];
        let method_config = |method: &str| BatchCorrectionConfig {
            enabled: true,
            method: method.to_owned(),
            ..BatchCorrectionConfig::default()
        };

        for method in ["fraction", "techreplicate", "totally-unknown"] {
            let (mut matrix, ..) =
                two_row_batch_matrix(["g1-A", "g1-B", "g2-A", "g2-B"], &p1_row, &p2_row)?;
            assert!(
                matrix
                    .apply_batch_correction(&method_config(method), None, false, false)
                    .is_err()
            );
        }
        Ok(())
    }

    type IncompleteBatchMatrix = (ProteinMatrix, Vec<SampleId>, [ProteinId; 3]);

    fn incomplete_batch_matrix(
        p1_row: [f64; 4],
        p2_row: [f64; 4],
    ) -> Result<IncompleteBatchMatrix, Box<dyn std::error::Error>> {
        // 4 samples -> 2 batches of 2 via sample_prefix ("A-*" vs "B-*"). P1/P2 are
        // complete (ComBat-corrected); P3 has a missing cell (kept uncorrected).
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A-1", "A-2", "B-1", "B-2"]
            .iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let p1 = proteins.get_or_insert("P1").expect("p1");
        let p2 = proteins.get_or_insert("P2").expect("p2");
        let p3 = proteins.get_or_insert("P3").expect("p3");

        let mut cells: std::collections::HashMap<CellKey, f64> = std::collections::HashMap::new();
        insert_batch_row(&mut cells, &sample_ids, p1, &p1_row);
        insert_batch_row(&mut cells, &sample_ids, p2, &p2_row);
        // P3: only the first three samples present (the fourth stays missing).
        insert_batch_row(&mut cells, &sample_ids, p3, &[7.0, 8.0, 9.0]);

        let allowed_proteins = [p1, p2, p3].into_iter().collect();
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins,
            excluded_samples: HashSet::new(),
            peptide_counts: std::collections::HashMap::new(),
            precision: std::collections::HashMap::new(),
            values: ProteinValues::Cells(cells),
        };
        Ok((matrix, sample_ids, [p1, p2, p3]))
    }

    fn log2_batch_matrix(rows: &[Vec<f64>; 2]) -> (ProteinMatrix, Vec<SampleId>, Vec<ProteinId>) {
        batch_test_matrix(&[
            (
                "P1",
                rows[0]
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("row"),
            ),
            (
                "P2",
                rows[1]
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("row"),
            ),
        ])
    }

    type BuiltMatrix = (ProteinMatrix, Vec<SampleId>, ProteinId, ProteinId);

    fn two_row_batch_matrix(
        names: [&str; 4],
        p1_row: &[f64; 4],
        p2_row: &[f64; 4],
    ) -> Result<BuiltMatrix, Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = names
            .iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let p1 = proteins.get_or_insert("P1").expect("p1");
        let p2 = proteins.get_or_insert("P2").expect("p2");
        let mut cells: std::collections::HashMap<CellKey, f64> = std::collections::HashMap::new();
        insert_batch_row(&mut cells, &sample_ids, p1, p1_row);
        insert_batch_row(&mut cells, &sample_ids, p2, p2_row);
        let allowed_proteins = [p1, p2].into_iter().collect();
        Ok((
            ProteinMatrix {
                proteins,
                samples,
                allowed_proteins,
                excluded_samples: HashSet::new(),
                peptide_counts: std::collections::HashMap::new(),
                precision: std::collections::HashMap::new(),
                values: ProteinValues::Cells(cells),
            },
            sample_ids,
            p1,
            p2,
        ))
    }

    fn insert_batch_row(
        cells: &mut HashMap<CellKey, f64>,
        samples: &[SampleId],
        protein: ProteinId,
        row: &[f64],
    ) {
        for (sample, &value) in samples.iter().zip(row.iter()) {
            cells.insert(
                CellKey {
                    protein,
                    sample: *sample,
                },
                value,
            );
        }
    }

    fn column_batch_sdrf() -> std::io::Result<(tempfile::TempDir, PathBuf)> {
        let dir = tempfile::Builder::new()
            .prefix("mokume-batch-column-wiring-")
            .tempdir()?;
        let sdrf_path = dir.path().join("col.sdrf.tsv");
        fs::write(
            &sdrf_path,
            "source name\tbatch_id\nX1\tg1\nX2\tg1\nX3\tg2\nX4\tg2\n",
        )?;
        Ok((dir, sdrf_path))
    }
}
