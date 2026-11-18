use super::*;
use crate::accessions::protein_group_name;

mod cases {
    use super::*;

    #[test]
    fn rejects_limma_de_without_contrasts() -> Result<(), Box<dyn std::error::Error>> {
        let (_parquet_guard, parquet) = existing_dummy_path("limma_de_no_contrasts")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "limma".to_string();

        let error = validate_implemented_subset(&config).err();
        assert!(matches!(error, Some(MokumeError::InvalidInput { .. })));
        Ok(())
    }

    #[test]
    fn resolves_auto_de_method() -> Result<(), Box<dyn std::error::Error>> {
        // `auto` follows Python's quantification-based default: directlfq
        // quantification selects `deqms`, every other quantification selects
        // `limrots` unless the matrix has missing values, which LimROTS rejects.
        // It validates (both targets are ported) and is resolved to the concrete
        // method just before the DE stage runs.
        let (_parquet_guard, parquet) = existing_dummy_path("de_method_auto")?;
        let mut config = base_config(parquet);
        config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "auto".to_string();
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(PathBuf::from("de.csv"));

        config.quantification = QuantMethod::DirectLfq;
        validate_implemented_subset(&config)?;
        assert_eq!(resolve_de_method(&config, || true), "deqms");

        config.quantification = QuantMethod::MaxLfq;
        validate_implemented_subset(&config)?;
        assert_eq!(resolve_de_method(&config, || true), "limrots");
        assert_eq!(resolve_de_method(&config, || false), "deqms");
        Ok(())
    }

    #[test]
    fn expands_de_contrasts_file_appending_to_inline() -> Result<(), Box<dyn std::error::Error>> {
        // Mirror Python (features2proteins.py:768): a TSV with group1/group2
        // columns appends "<g1> vs <g2>" after the repeated --de-contrast entries,
        // empty rows are skipped, and contrasts_file is cleared so downstream sees
        // one resolved list.
        let (_parquet_guard, parquet) = existing_dummy_path("de_contrasts_file")?;
        let (_contrasts_file_guard, contrasts_file) = existing_dummy_path("de_contrasts_file_tsv")?;
        fs::write(&contrasts_file, "group1\tgroup2\nTumor\tNormal\nA\tB\n\t\n")?;

        let mut config = base_config(parquet);
        config.differential_expression.contrasts = Some(vec!["X vs Y".to_string()]);
        config.differential_expression.contrasts_file = Some(contrasts_file);

        let resolved = expand_de_contrasts_file(&config)?;
        assert_eq!(
            resolved.differential_expression.contrasts,
            Some(vec![
                "X vs Y".to_string(),
                "Tumor vs Normal".to_string(),
                "A vs B".to_string(),
            ])
        );
        assert!(resolved.differential_expression.contrasts_file.is_none());
        Ok(())
    }

    #[test]
    fn rejects_de_contrasts_file_missing_group_columns() -> Result<(), Box<dyn std::error::Error>> {
        let (_parquet_guard, parquet) = existing_dummy_path("de_contrasts_bad")?;
        let (_contrasts_file_guard, contrasts_file) = existing_dummy_path("de_contrasts_bad_tsv")?;
        fs::write(&contrasts_file, "groupA\tgroupB\nTumor\tNormal\n")?;

        let mut config = base_config(parquet);
        config.differential_expression.contrasts_file = Some(contrasts_file);

        assert!(matches!(
            expand_de_contrasts_file(&config),
            Err(MokumeError::InvalidInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_invalid_ensemble_members_before_execution() -> Result<(), Box<dyn std::error::Error>>
    {
        let invalid_members = [
            (Vec::<String>::new(), "empty"),
            (vec!["".to_string()], "empty"),
            (vec!["unknown".to_string()], "unknown"),
            (vec!["auto".to_string()], "unknown"),
            (vec!["ensemble".to_string()], "nested"),
            (
                vec!["limma".to_string(), " LIMMA ".to_string()],
                "duplicate",
            ),
        ];

        for (members, expected) in invalid_members {
            let (_parquet_guard, parquet) = existing_dummy_path("invalid_ensemble_member")?;
            let mut config = base_config(parquet);
            config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
            config.differential_expression.enabled = true;
            config.differential_expression.method = "ensemble".to_string();
            config.differential_expression.ensemble_methods = Some(members);
            config.differential_expression.ensemble_min_k = 1;
            config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
            config.differential_expression.output = Some(PathBuf::from("de.csv"));

            let Some(error) = validate_de_ensemble_options(&config).err() else {
                return Err("invalid ensemble member configuration was accepted".into());
            };
            let error = error.to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error, got {error:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn validates_ensemble_min_k_against_configured_members()
    -> Result<(), Box<dyn std::error::Error>> {
        for min_k in [0, 3] {
            let (_parquet_guard, parquet) = existing_dummy_path("invalid_ensemble_min_k")?;
            let mut config = base_config(parquet);
            config.input.sdrf = Some(PathBuf::from("sdrf.tsv"));
            config.differential_expression.enabled = true;
            config.differential_expression.method = "ensemble".to_string();
            config.differential_expression.ensemble_methods =
                Some(vec!["limma".to_string(), "deqms".to_string()]);
            config.differential_expression.ensemble_min_k = min_k;
            config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
            config.differential_expression.output = Some(PathBuf::from("de.csv"));

            let Some(error) = validate_de_ensemble_options(&config).err() else {
                return Err("invalid ensemble min-k was accepted".into());
            };
            let error = error.to_string();
            assert!(error.contains("min-k"), "unexpected error: {error}");
        }
        Ok(())
    }

    #[test]
    fn invalid_ensemble_precedes_missing_input_without_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let tempdir = tempfile::tempdir()?;
        let output = tempdir.path().join("de.csv");
        let mut config = base_config(tempdir.path().join("missing.parquet"));
        config.input.sdrf = Some(tempdir.path().join("missing.sdrf.tsv"));
        config.differential_expression.enabled = true;
        config.differential_expression.method = "ensemble".to_string();
        config.differential_expression.ensemble_methods = Some(vec!["unknown".to_string()]);
        config.differential_expression.ensemble_min_k = 1;
        config.differential_expression.contrasts = Some(vec!["A vs B".to_string()]);
        config.differential_expression.output = Some(output.clone());

        let Some(error) = run_features_to_proteins(&config).err() else {
            return Err("invalid ensemble configuration was accepted".into());
        };
        let error = error.to_string();

        assert!(error.contains("unknown"), "unexpected error: {error}");
        assert!(!output.exists(), "invalid ensemble created DE output");
        Ok(())
    }

    #[test]
    fn protein_fdr_lookup_uses_the_qpx_group_not_a_renamed_one() {
        let accessions = vec!["CONTAM_P1".to_owned(), "P1".to_owned()];
        let passing = HashSet::from([super::protein_group_name(&accessions).unwrap_or_default()]);
        assert!(!super::fails_protein_fdr(Some(&passing), &accessions));
        assert!(super::fails_protein_fdr(
            Some(&HashSet::from(["P1".to_owned()])),
            &accessions
        ));
        assert!(!super::fails_protein_fdr(None, &accessions));
    }
}
