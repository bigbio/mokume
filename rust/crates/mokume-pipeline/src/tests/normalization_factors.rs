use super::*;

mod cases {
    use super::*;

    #[test]
    fn accepts_normalization_proteins_file_for_supported_subset()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_parquet_guard, parquet) = existing_dummy_path("normalization_proteins")?;
        let (_proteins_guard, proteins) = existing_dummy_path("normalization_proteins_list")?;
        fs::write(&proteins, "P1\nP2\n")?;
        let mut config = base_config(parquet);
        config.normalization.normalization_proteins = Some(proteins);

        validate_features_to_proteins(&config)?;
        validate_implemented_subset(&config)?;

        Ok(())
    }

    #[test]
    fn rejects_empty_normalization_proteins_file_before_loading()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_proteins_guard, proteins) = existing_dummy_path("empty_normalization_proteins_list")?;
        fs::write(&proteins, "\n")?;

        let error = load_normalization_proteins(&proteins).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some(
                format!(
                    "invalid input: normalization proteins file `{}` does not contain protein IDs",
                    proteins.display()
                )
                .as_str()
            )
        );
        Ok(())
    }

    #[test]
    fn rejects_unmatched_normalization_proteins_after_loading()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut collector = NormalizationFactorCollector::new(
            None,
            Some(SampleNormalizationMethod::GlobalMedian),
            None,
            Some(HashSet::from(["P999".to_owned()])),
            &[],
        );

        collector.push(
            QpxFeatureRecord {
                sequence: "PEPTIDEAK".to_owned(),
                peptidoform: "PEPTIDEAK".to_owned(),
                charge: 2,
                run_file_name: "run1.raw".to_owned(),
                sample_accession: None,
                protein_accessions: vec!["P1".to_owned()],
                anchor_protein: None,
                unique: Some(true),
                is_decoy: Some(false),
                peptide_qvalue: None,
                pg_global_qvalue: None,
                selected_score: None,
                modifications: None,
                transferred: false,
                label: None,
                intensity: 100.0,
                row_start: true,
            },
            FilterConfig::default(),
            false,
        )?;

        assert!(
            !collector.has_sample_factor_values(),
            "unmatched normalization proteins must not feed sample factor values"
        );
        Ok(())
    }

    #[test]
    fn rejects_missing_normalization_proteins_file_before_loading()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_parquet_guard, parquet) = existing_dummy_path("missing_normalization_proteins")?;
        let mut config = base_config(parquet);
        let missing_directory = tempfile::Builder::new()
            .prefix(&format!(
                "mokume_missing_normalization_proteins_{}_{}_",
                std::process::id(),
                unique_suffix()
            ))
            .tempdir()?;
        let missing_path = missing_directory.path().join("missing.txt");
        config.normalization.normalization_proteins = Some(missing_path.clone());

        let error = validate_features_to_proteins(&config).err();

        assert_eq!(
            error.map(|error| error.to_string()).as_deref(),
            Some(format!("input file does not exist: {}", missing_path.display()).as_str())
        );
        Ok(())
    }
}
