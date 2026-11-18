use super::*;

mod cases {
    use super::*;

    #[test]
    fn peptide_sums_follow_stable_id_order() {
        let expected = HashMap::from([
            (PeptideId::new(0), 1.0),
            (PeptideId::new(1), 1.0),
            (PeptideId::new(2), 1.0e16),
        ]);
        let reordered = HashMap::from([
            (PeptideId::new(2), 1.0e16),
            (PeptideId::new(0), 1.0),
            (PeptideId::new(1), 1.0),
        ]);

        assert_eq!(
            sum_peptide_values(&expected),
            sum_peptide_values(&reordered)
        );
        assert_eq!(sum_peptide_values(&expected), (1.0 + 1.0) + 1.0e16);
    }

    #[test]
    fn canonical_sums_follow_stable_id_order() {
        let canonical = PeptideId::new(9);
        let peptide_to_canonical = (0..3)
            .map(|id| (PeptideId::new(id), canonical))
            .collect::<HashMap<_, _>>();
        // Each map draws its own hash seed, so iteration orders differ.
        for _ in 0..32 {
            let peptides = HashMap::from([
                (PeptideId::new(2), 1.0e16),
                (PeptideId::new(0), 1.0),
                (PeptideId::new(1), 1.0),
            ]);
            let summed = sum_canonical_cell(&peptides, &peptide_to_canonical);
            assert_eq!(summed[&canonical], (1.0 + 1.0) + 1.0e16);
        }
    }
}
