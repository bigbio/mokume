use super::*;

mod cases {
    use super::*;

    #[test]
    fn reference_detection_uses_real_regex_and_explicit_priority()
    -> Result<(), Box<dyn std::error::Error>> {
        let text = concat!(
            "source name\tcomment[data file]\tfactor value[role]\tcharacteristics[pooled sample]\n",
            "S1\ta.raw\tordinary\tnot pooled\n",
            "S2\tb.raw\tBridge-42\tpooled\n",
            "S3\tc.raw\tbridge-X\tnot pooled\n",
        );
        let raw = SdrfRawTable::from_reader(text.as_bytes())?;
        let typed = SdrfTable::from_reader(text.as_bytes())?;
        assert_eq!(
            detect_reference_samples(&raw, r"^bridge-\d+$")?,
            vec!["S2".to_owned()]
        );
        assert!(detect_reference_samples(&raw, "[").is_err());

        let explicit = IrsConfig {
            reference_samples: Some(vec!["S1".to_owned()]),
            ..IrsConfig::default()
        };
        assert_eq!(
            resolve_irs_reference_samples(&typed, &raw, &explicit)?,
            vec!["S1".to_owned()]
        );

        let custom_regex = IrsConfig {
            reference_regex: "^ordinary$".to_owned(),
            ..IrsConfig::default()
        };
        assert_eq!(
            resolve_irs_reference_samples(&typed, &raw, &custom_regex)?,
            vec!["S1".to_owned()]
        );
        Ok(())
    }
}
