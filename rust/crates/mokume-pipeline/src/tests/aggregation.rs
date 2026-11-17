use super::*;

mod cases {
    use super::*;

    #[test]
    fn before_sample_normalization_retains_run_transform_and_channel_irs() {
        let factors = IntensityFactors {
            run: HashMap::from([(
                RunCellKey {
                    sample: "sample".to_owned(),
                    run: "run".to_owned(),
                },
                RunNormalizationTransform::new(2.0, 3.0),
            )]),
            irs_scale_by_run: HashMap::from([("run".to_owned(), 3.0)]),
            sample: HashMap::from([("sample".to_owned(), 5.0)]),
        };
        assert_eq!(factors.normalize_run(10.0, "sample", "run"), 69.0);
        assert_eq!(factors.normalize(10.0, "sample", "run"), 345.0);
    }

    #[test]
    fn lfq_min_unique_gate_is_per_cell_for_both_methods() -> mokume_core::Result<()> {
        let fixture = lfq_min_unique_fixture();
        let max_proteins = lfq_fixture_protein_ids(super::QuantMethod::MaxLfq, &fixture)?;
        let direct_proteins = lfq_fixture_protein_ids(super::QuantMethod::DirectLfq, &fixture)?;

        assert!(
            max_proteins.contains(&fixture.dense),
            "MaxLFQ keeps a protein with a two-peptide cell"
        );
        assert!(
            !max_proteins.contains(&fixture.scattered),
            "MaxLFQ drops a protein whose cells never reach two distinct peptides"
        );
        assert!(direct_proteins.contains(&fixture.dense));
        assert!(
            !direct_proteins.contains(&fixture.scattered),
            "DirectLFQ applies the same per-cell unique-peptide gate"
        );
        Ok(())
    }

    // With `cell_precision`, both LFQ methods give a finite precision for
    // exactly the cells they quantify, keyed by the matrix ids.
    #[test]
    fn lfq_cell_precision_covers_quantified_cells() -> mokume_core::Result<()> {
        let fixture = lfq_min_unique_fixture();
        let protein = fixture.dense;
        let mut sums = HashMap::new();
        for (index, sample) in (1..=4).map(SampleId::new).enumerate() {
            for (factor, peptide) in [(1.0, 9), (1.3, 10), (0.8, 11)] {
                let noise = 1.0 + 0.05 * ((index * peptide) % 3) as f64;
                let canonical = PeptideId::new(peptide as u32);
                let key = super::DirectLfqCellKey {
                    protein,
                    canonical,
                    sample,
                };
                sums.insert(key, 1000.0 * factor * noise);
            }
        }
        for method in [super::QuantMethod::MaxLfq, super::QuantMethod::DirectLfq] {
            let mut aggregation = lfq_aggregation(method, sums.clone());
            if let super::FeatureAggregation::Lfq { cell_precision, .. } = &mut aggregation {
                *cell_precision = true;
            }
            let (values, precision) = aggregation.finalize(
                &fixture.allowed_cells,
                &mut StringIdRegistry::<ProteinId>::new(),
                &HashMap::new(),
                &fixture.canonical_peptides,
                &StringIdRegistry::<SampleId>::new(),
            )?;
            let super::ProteinValues::Rows(rows) = values else {
                panic!("{method:?} returns protein rows");
            };
            let cells = rows
                .iter()
                .flat_map(|(protein, values)| {
                    values
                        .iter()
                        .filter(|(_, value)| *value > 0.0)
                        .map(|(sample, _)| super::CellKey {
                            protein: *protein,
                            sample: *sample,
                        })
                })
                .collect::<HashSet<_>>();
            assert_eq!(cells.len(), 4, "{method:?}");
            assert_eq!(
                precision.keys().copied().collect::<HashSet<_>>(),
                cells,
                "{method:?}"
            );
            assert!(
                precision
                    .values()
                    .all(|value| value.is_finite() && *value > 0.0)
            );
        }
        Ok(())
    }
}
