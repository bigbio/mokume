use super::*;

mod cases {
    use super::*;

    #[test]
    fn remove_ids_matches_complete_accessions_not_substrings() {
        let accessions = ["P1", "sp|P10|protein ten"].map(str::to_owned);
        assert!(super::has_removed_accession(
            &accessions,
            &HashSet::from(["P1".to_owned()])
        ));
        assert!(!super::has_removed_accession(
            &["P10".to_owned()],
            &HashSet::from(["P1".to_owned()])
        ));
    }

    #[test]
    fn ups_database_prefix_keeps_each_accession() {
        assert_eq!(
            super::parse_protein_accession("UPS|P02768ups|ALBU_HUMAN_UPS"),
            "P02768ups"
        );
    }
}
