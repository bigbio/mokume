use super::*;

mod cases {
    use super::*;

    #[test]
    fn missing_value_check_matches_log2_rows() -> Result<(), Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A1", "B1"]
            .into_iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let protein_ids = ["P1", "P2", "P3"]
            .into_iter()
            .map(|name| proteins.get_or_insert(name).ok_or("protein id"))
            .collect::<Result<Vec<_>, _>>()?;
        // P1 is complete, P2 has a zero in B1, and P3 (filtered out) lacks B1.
        let mut cells = HashMap::new();
        for (protein, sample, value) in [
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 0, 30.0),
            (1, 1, 0.0),
            (2, 0, 40.0),
        ] {
            cells.insert(
                CellKey {
                    protein: protein_ids[protein],
                    sample: sample_ids[sample],
                },
                value,
            );
        }
        let mut matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: protein_ids[..2].iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: HashMap::new(),
            precision: HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        for (columns, missing) in [(&sample_ids[..1], false), (&sample_ids[..], true)] {
            let (_, rows) = matrix.log2_rows(columns)?;
            assert_eq!(rows.iter().flatten().any(|value| value.is_nan()), missing);
            assert_eq!(matrix.has_missing_values(columns), missing);
        }
        matrix.allowed_proteins.remove(&protein_ids[1]);
        assert!(!matrix.has_missing_values(&sample_ids));
        Ok(())
    }

    #[test]
    fn write_csv_missing_fill_matches_python_per_method_convention()
    -> Result<(), Box<dyn std::error::Error>> {
        // Columns sort by sample name: S1=col 1, S2=col 2, S3=col 3. P1 is observed
        // in S1 and S2; P2 only in S1 (missing in the observed sample S2); S3 carries
        // no observation at all.
        let mut samples = StringIdRegistry::<SampleId>::new();
        let s1 = samples.get_or_insert("S1").expect("s1");
        let s2 = samples.get_or_insert("S2").expect("s2");
        let _s3 = samples.get_or_insert("S3").expect("s3");
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let p1 = proteins.get_or_insert("P1").expect("p1");
        let p2 = proteins.get_or_insert("P2").expect("p2");
        let mut cells: std::collections::HashMap<CellKey, f64> = std::collections::HashMap::new();
        cells.insert(
            CellKey {
                protein: p1,
                sample: s1,
            },
            10.0,
        );
        cells.insert(
            CellKey {
                protein: p1,
                sample: s2,
            },
            20.0,
        );
        cells.insert(
            CellKey {
                protein: p2,
                sample: s1,
            },
            30.0,
        );
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: [p1, p2].into_iter().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: std::collections::HashMap::new(),
            precision: std::collections::HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        let cell = |csv: &str, protein: &str, col: usize| -> String {
            csv.lines()
                .find(|line| line.starts_with(&format!("{protein},")))
                .and_then(|line| line.split(',').nth(col))
                .unwrap_or("ABSENT")
                .to_owned()
        };

        // Additive method (`Some(0.0)`): a missing cell in the observed sample S2 is
        // `0`, but the all-empty sample S3 stays blank (Python never densifies it).
        let zero_directory = tempfile::Builder::new()
            .prefix("mokume-missing-zero-")
            .tempdir()?;
        let zero_path = zero_directory.path().join("missing_zero.csv");
        matrix.write_csv(&zero_path, OutputFormat::PythonCompatible, false, Some(0.0))?;
        let zero = fs::read_to_string(&zero_path)?;
        assert_eq!(cell(&zero, "P1", 1), "10");
        assert_eq!(cell(&zero, "P2", 2), "0");
        assert_eq!(cell(&zero, "P2", 3), "");

        // Average/ratio method (`None`): every missing cell is blank.
        let nan_directory = tempfile::Builder::new()
            .prefix("mokume-missing-nan-")
            .tempdir()?;
        let nan_path = nan_directory.path().join("missing_nan.csv");
        matrix.write_csv(&nan_path, OutputFormat::PythonCompatible, false, None)?;
        let nan = fs::read_to_string(&nan_path)?;
        assert_eq!(cell(&nan, "P1", 1), "10");
        assert_eq!(cell(&nan, "P2", 2), "");
        assert_eq!(cell(&nan, "P2", 3), "");
        Ok(())
    }
}
