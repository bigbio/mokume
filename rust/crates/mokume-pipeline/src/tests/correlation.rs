use super::*;

mod cases {
    use super::*;

    #[test]
    fn sample_correlation_filter_excludes_only_the_original_matrix_outlier()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let sample_ids = ["A1", "A2", "A3", "B1", "B2"]
            .into_iter()
            .map(|name| samples.get_or_insert(name).ok_or("sample id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let protein_ids = ["P1", "P2", "P3", "P4"]
            .into_iter()
            .map(|name| proteins.get_or_insert(name).ok_or("protein id"))
            .collect::<Result<Vec<_>, _>>()?;
        let columns = [
            [1.0, 2.0, 4.0, 8.0],
            [2.0, 4.0, 8.0, 16.0],
            [8.0, 4.0, 2.0, 1.0],
            [3.0, 6.0, 12.0, 24.0],
            [6.0, 12.0, 24.0, 48.0],
        ];
        let mut cells = HashMap::new();
        for (sample, values) in sample_ids.iter().zip(columns) {
            for (protein, value) in protein_ids.iter().zip(values) {
                cells.insert(
                    CellKey {
                        protein: *protein,
                        sample: *sample,
                    },
                    value,
                );
            }
        }
        let mut matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: protein_ids.iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: HashMap::new(),
            precision: HashMap::new(),
            values: ProteinValues::Cells(cells),
        };
        let sdrf = SdrfTable::from_reader(
            b"source name\tcomment[data file]\tfactor value[group]\n\
A1\tA1.raw\tA\nA2\tA2.raw\tA\nA3\tA3.raw\tA\n\
B1\tB1.raw\tB\nB2\tB2.raw\tB\n"
                .as_slice(),
        )?;

        matrix.apply_sample_correlation_filter(Some(&sdrf), -0.5, false, false)?;

        let kept = matrix
            .sample_columns(false)
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>();
        assert_eq!(kept, vec!["A1", "A2", "B1", "B2"]);
        assert_eq!(matrix.value(protein_ids[0], sample_ids[0]), Some(1.0));
        Ok(())
    }

    #[test]
    fn sample_correlation_uses_negative_values_for_log2_methods()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut samples = StringIdRegistry::<SampleId>::new();
        let left = samples.get_or_insert("A1").expect("left sample");
        let right = samples.get_or_insert("A2").expect("right sample");
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let protein_ids = ["P1", "P2", "P3"]
            .into_iter()
            .map(|name| proteins.get_or_insert(name).ok_or("protein id"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut cells = HashMap::new();
        for (protein, left_value, right_value) in protein_ids
            .iter()
            .zip([-2.0, 0.0, 2.0])
            .zip([-1.0, 1.0, 3.0])
            .map(|((protein, left_value), right_value)| (protein, left_value, right_value))
        {
            cells.insert(
                CellKey {
                    protein: *protein,
                    sample: left,
                },
                left_value,
            );
            cells.insert(
                CellKey {
                    protein: *protein,
                    sample: right,
                },
                right_value,
            );
        }
        let matrix = ProteinMatrix {
            proteins,
            samples,
            allowed_proteins: protein_ids.iter().copied().collect(),
            excluded_samples: HashSet::new(),
            peptide_counts: HashMap::new(),
            precision: HashMap::new(),
            values: ProteinValues::Cells(cells),
        };

        let (log2_correlation, log2_overlap) =
            matrix.pairwise_sample_correlation(left, right, true);
        let (linear_correlation, linear_overlap) =
            matrix.pairwise_sample_correlation(left, right, false);
        assert_eq!(log2_overlap, 3);
        assert!(log2_correlation.is_some_and(|value| (value - 1.0).abs() < 1e-12));
        assert_eq!((linear_correlation, linear_overlap), (None, 1));
        Ok(())
    }
}
