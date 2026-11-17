use super::*;

mod cases {
    use super::*;

    #[test]
    fn match_sdrf_column_prefers_exact_then_substring() {
        let headers: Vec<String> = ["source name", "characteristics[sex]", "batch"]
            .map(String::from)
            .to_vec();
        // Exact lowercased match.
        assert_eq!(match_sdrf_column(&headers, "Characteristics[Sex]"), Some(1));
        // Substring: "sex" is contained in "characteristics[sex]".
        assert_eq!(match_sdrf_column(&headers, "sex"), Some(1));
        assert_eq!(match_sdrf_column(&headers, "missing"), None);
    }

    #[test]
    fn extract_sdrf_covariates_matches_python_oracle() -> Result<(), Box<dyn std::error::Error>> {
        let raw = covariate_fixture()?;

        // Case A: full names, matrix order == SDRF order. Oracle: [[0,0],[1,0],[0,1],[1,1]].
        let names = ["S1", "S2", "S3", "S4"];
        let covs = extract_sdrf_covariates(
            &raw,
            &names,
            &[
                "characteristics[sex]".into(),
                "characteristics[organism part]".into(),
            ],
        )?;
        assert_eq!(
            covs,
            Some(vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ])
        );

        // Case B: substring column match. Oracle: [[0],[1],[0],[1]].
        let covs_b = extract_sdrf_covariates(&raw, &names, &["sex".into()])?;
        assert_eq!(
            covs_b,
            Some(vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]])
        );

        // Case C: matrix sample name is a superstring of the SDRF source name.
        let names_c = ["S1_frac1", "S2_frac1", "S3_frac1", "S4_frac1"];
        let covs_c = extract_sdrf_covariates(&raw, &names_c, &["characteristics[sex]".into()])?;
        assert_eq!(
            covs_c,
            Some(vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]])
        );

        // Explicitly requested covariates reject unmatched samples and constant
        // columns instead of silently changing or dropping the design.
        let names_d = ["S1", "S2", "ZZZ", "S4"];
        assert!(
            extract_sdrf_covariates(&raw, &names_d, &["characteristics[organism part]".into()])
                .is_err()
        );
        assert!(extract_sdrf_covariates(&raw, &names, &["const_col".into()]).is_err());
        assert_eq!(extract_sdrf_covariates(&raw, &names, &[])?, None);
        Ok(())
    }

    #[test]
    fn covariates_keep_numeric_values_and_one_hot_nominal_categories()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = concat!(
            "source name\tage\tsite\n",
            "S1\t20\tnorth\n",
            "S2\t35\tsouth\n",
            "S3\t50\twest\n",
            "S4\t65\twest\n",
        );
        let raw = SdrfRawTable::from_reader(input.as_bytes())?;
        let covariates = extract_sdrf_covariates(
            &raw,
            &["S1", "S2", "S3", "S4"],
            &["age".into(), "site".into()],
        )?;
        assert_eq!(
            covariates,
            Some(vec![
                vec![20.0, 0.0, 0.0],
                vec![35.0, 1.0, 0.0],
                vec![50.0, 0.0, 1.0],
                vec![65.0, 0.0, 1.0],
            ])
        );
        Ok(())
    }

    #[test]
    fn mixed_covariate_columns_are_nominal() -> Result<(), Box<dyn std::error::Error>> {
        // "5637" is a cell line name, not a number, once the column has other labels.
        let raw = SdrfRawTable::from_reader(
            "source name\tcell line\nS1\t5637\nS2\tA549\nS3\t5637\nS4\tK562\n".as_bytes(),
        )?;
        let covariates =
            extract_sdrf_covariates(&raw, &["S1", "S2", "S3", "S4"], &["cell line".into()])?;
        assert_eq!(
            covariates,
            Some(vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 0.0],
                vec![0.0, 1.0],
            ])
        );
        Ok(())
    }

    #[test]
    fn covariates_reject_missing_and_nonfinite_values() -> Result<(), Box<dyn std::error::Error>> {
        for (header, rows, expected) in [
            (
                "missing",
                ["20", "", "50", "65"],
                "contains a missing value",
            ),
            (
                "nonfinite",
                ["20", "NaN", "50", "65"],
                "contains a non-finite value",
            ),
        ] {
            let input = format!(
                "source name\t{header}\nS1\t{}\nS2\t{}\nS3\t{}\nS4\t{}\n",
                rows[0], rows[1], rows[2], rows[3]
            );
            let raw = SdrfRawTable::from_reader(input.as_bytes())?;
            let Err(error) =
                extract_sdrf_covariates(&raw, &["S1", "S2", "S3", "S4"], &[header.into()])
            else {
                panic!("invalid covariate was accepted");
            };
            assert!(error.to_string().contains(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn batch_column_values_require_complete_source_name_mapping()
    -> Result<(), Box<dyn std::error::Error>> {
        let raw = covariate_fixture()?;
        // Oracle: ['b1','b1','b2','b2'].
        assert_eq!(
            batch_column_values_for_samples(&raw, &["S1", "S2", "S3", "S4"], "batch")?,
            vec![
                "b1".to_owned(),
                "b1".to_owned(),
                "b2".to_owned(),
                "b2".to_owned()
            ]
        );
        assert!(batch_column_values_for_samples(&raw, &["S1", "ZZZ"], "BATCH").is_err());
        assert!(batch_column_values_for_samples(&raw, &["S1"], "nonexistent").is_err());
        Ok(())
    }
}
