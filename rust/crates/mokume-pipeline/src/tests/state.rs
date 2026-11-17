use super::*;

mod cases {
    use super::*;

    #[test]
    fn null_unique_flag_means_a_one_protein_group() {
        let feature = |proteins: &[&str], unique: Option<bool>| QpxFeatureRecord {
            sequence: "PEPTIDEAK".to_owned(),
            peptidoform: "PEPTIDEAK".to_owned(),
            charge: 2,
            run_file_name: "run1.raw".to_owned(),
            sample_accession: None,
            protein_accessions: proteins.iter().map(|&protein| protein.to_owned()).collect(),
            anchor_protein: None,
            unique,
            is_decoy: Some(false),
            peptide_qvalue: None,
            pg_global_qvalue: None,
            selected_score: None,
            modifications: None,
            transferred: false,
            label: None,
            intensity: 100.0,
            row_start: true,
        };
        let filtering = FilterConfig::default();
        let mut null_decoy = feature(&["P1"], None);
        null_decoy.is_decoy = None;
        assert!(passes_feature_filter(&null_decoy, filtering, false));

        assert!(passes_feature_filter(
            &feature(&["P1"], None),
            filtering,
            false
        ));
        assert!(!passes_feature_filter(
            &feature(&["P1", "P2"], None),
            filtering,
            false
        ));
        assert!(passes_feature_filter(
            &feature(&["P1", "P2"], None),
            filtering,
            true
        ));
        assert!(passes_feature_filter(
            &feature(&["P1", "P2"], Some(true)),
            filtering,
            false
        ));
    }
}
