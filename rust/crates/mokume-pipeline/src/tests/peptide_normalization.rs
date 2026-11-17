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
}
