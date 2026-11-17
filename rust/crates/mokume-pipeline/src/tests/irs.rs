use super::*;

mod cases {
    use super::*;

    // Locks the synthetic TMT oracle (also verified against Python's
    // `get_irs_scaling_factors`): reference-channel `irs_value` per run = 200 / 100
    // / 400 -> global_center = median(200,100,400) = 200 ->
    // scale = {M1_1: 1.0, M2_2: 2.0, M3_3: 0.5}.
    #[test]
    fn irs_global_scale_matches_synthetic_tmt_oracle_median() {
        let per_run = vec![
            ("M1_1".to_owned(), vec![150.0_f64, 250.0]), // median 200
            ("M2_2".to_owned(), vec![80.0, 120.0]),      // median 100
            ("M3_3".to_owned(), vec![350.0, 450.0]),     // median 400
        ];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Median);
        assert_eq!(scale.get("M1_1").copied(), Some(1.0));
        assert_eq!(scale.get("M2_2").copied(), Some(2.0));
        assert_eq!(scale.get("M3_3").copied(), Some(0.5));
        assert_eq!(scale.len(), 3);
    }

    #[test]
    fn irs_global_scale_matches_synthetic_tmt_oracle_mean() {
        // Same per-run intensities; mean center = mean(200,100,400) = 233.333...
        let per_run = vec![
            ("M1_1".to_owned(), vec![150.0_f64, 250.0]),
            ("M2_2".to_owned(), vec![80.0, 120.0]),
            ("M3_3".to_owned(), vec![350.0, 450.0]),
        ];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Mean);
        let center = (200.0 + 100.0 + 400.0) / 3.0;
        assert!((scale["M1_1"] - center / 200.0).abs() < 1e-12);
        assert!((scale["M2_2"] - center / 100.0).abs() < 1e-12);
        assert!((scale["M3_3"] - center / 400.0).abs() < 1e-12);
    }

    #[test]
    fn irs_global_scale_drops_nonpositive_and_handles_empty() {
        // A run whose only intensities aggregate to a non-positive center is
        // dropped (Python's `irs_value > 0` filter); empty input yields no scale.
        assert!(irs_global_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        let per_run = vec![
            ("run1".to_owned(), vec![100.0_f64]),
            ("run2".to_owned(), vec![0.0, 0.0]),
        ];
        let scale = irs_global_scale_from_runs(per_run, IrsStat::Median);
        // run2 aggregates to 0 (median drops zeros -> None), so only run1
        // survives; with one surviving run the center equals its own value -> 1.0.
        assert_eq!(scale.get("run1").copied(), Some(1.0));
        assert!(!scale.contains_key("run2"));
    }

    #[test]
    fn irs_by_mixture_scale_matches_python_oracle_median() {
        // mixA center = median(200,150) = 175; mixB center = median(500,120) = 310.
        let scale = irs_by_mixture_scale_from_runs(multi_mixture_runs(), IrsStat::Median);
        assert_eq!(scale.len(), 4);
        assert_close(scale["mixA_1"], 175.0 / 200.0); // 0.875
        assert_close(scale["mixA_2"], 175.0 / 150.0); // 1.16666...
        assert_close(scale["mixB_3"], 310.0 / 500.0); // 0.62
        assert_close(scale["mixB_4"], 310.0 / 120.0); // 2.58333...
    }

    #[test]
    fn irs_by_mixture_scale_matches_python_oracle_mean() {
        // mixA center = mean(200,150) = 175; mixB center = mean(500,200) = 350.
        let scale = irs_by_mixture_scale_from_runs(multi_mixture_runs(), IrsStat::Mean);
        assert_eq!(scale.len(), 4);
        assert_close(scale["mixA_1"], 175.0 / 200.0); // 0.875
        assert_close(scale["mixA_2"], 175.0 / 150.0); // 1.16666...
        assert_close(scale["mixB_3"], 350.0 / 500.0); // 0.7
        assert_close(scale["mixB_4"], 350.0 / 200.0); // 1.75
    }

    #[test]
    fn irs_two_stage_scale_matches_python_oracle_median() {
        // Stage-1 centers: mixA 175, mixB 310. Global center over the distinct
        // mixture centers = median(175,310) = 242.5. Applied scale is the product
        // (stage1 * stage2), which equals global_center / irs_value.
        let scale = irs_two_stage_scale_from_runs(multi_mixture_runs(), IrsStat::Median);
        let global = 242.5;
        assert_eq!(scale.len(), 4);
        assert_close(scale["mixA_1"], (175.0 / 200.0) * (global / 175.0));
        assert_close(scale["mixA_2"], (175.0 / 150.0) * (global / 175.0));
        assert_close(scale["mixB_3"], (310.0 / 500.0) * (global / 310.0));
        assert_close(scale["mixB_4"], (310.0 / 120.0) * (global / 310.0));
    }

    #[test]
    fn irs_two_stage_scale_matches_python_oracle_mean() {
        // Stage-1 centers: mixA 175, mixB 350. Global center = mean(175,350) = 262.5.
        let scale = irs_two_stage_scale_from_runs(multi_mixture_runs(), IrsStat::Mean);
        let global = 262.5;
        assert_eq!(scale.len(), 4);
        assert_close(scale["mixA_1"], (175.0 / 200.0) * (global / 175.0));
        assert_close(scale["mixA_2"], (175.0 / 150.0) * (global / 175.0));
        assert_close(scale["mixB_3"], (350.0 / 500.0) * (global / 350.0));
        assert_close(scale["mixB_4"], (350.0 / 200.0) * (global / 350.0));
    }

    #[test]
    fn irs_scope_functions_handle_empty_and_nonpositive() {
        assert!(irs_by_mixture_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        assert!(irs_two_stage_scale_from_runs(Vec::new(), IrsStat::Median).is_empty());
        // A run aggregating to a non-positive center is dropped before the
        // mixture center is formed (Python's `irs_value > 0`).
        let runs = vec![
            ("run1".to_owned(), "m".to_owned(), vec![100.0]),
            ("run2".to_owned(), "m".to_owned(), vec![0.0, 0.0]),
        ];
        let by_mixture = irs_by_mixture_scale_from_runs(runs.clone(), IrsStat::Median);
        assert_eq!(by_mixture.get("run1").copied(), Some(1.0));
        assert!(!by_mixture.contains_key("run2"));
        let two_stage = irs_two_stage_scale_from_runs(runs, IrsStat::Median);
        assert_eq!(two_stage.get("run1").copied(), Some(1.0));
        assert!(!two_stage.contains_key("run2"));
    }

    #[test]
    fn irs_mixture_first_token_takes_first_underscore_segment() {
        assert_eq!(irs_mixture_first_token("mixA_1"), "mixA");
        assert_eq!(irs_mixture_first_token("plex_2_5"), "plex");
        assert_eq!(irs_mixture_first_token("noUnderscore"), "noUnderscore");
        assert_eq!(irs_mixture_first_token(""), "");
    }

    #[test]
    fn resolve_irs_autodetect_channel_picks_mode_label() -> Result<(), Box<dyn std::error::Error>> {
        // Three pooled rows: TMT131 appears twice, TMT130 once -> mode TMT131.
        // A non-pooled row must not vote.
        let (_sdrf_guard, sdrf) = write_temp_sdrf(
            "pooled",
            "source name\tcomment[label]\tcharacteristics[pooled sample]\n\
             sample_pool_1\tTMT131\tpooled\n\
             sample_pool_2\tTMT131\tpooled\n\
             sample_pool_3\tTMT130\tpooled\n\
             sample_normal\tTMT126\tnot pooled\n",
        )?;
        let channel = resolve_irs_autodetect_channel(&sdrf, "pool")?;
        assert_eq!(channel.as_deref(), Some("TMT131"));
        fs::remove_file(&sdrf)?;
        Ok(())
    }

    #[test]
    fn resolve_irs_autodetect_channel_breaks_ties_lexicographically()
    -> Result<(), Box<dyn std::error::Error>> {
        // TMT127 and TMT131 each match once; pandas `mode().iloc[0]` returns the
        // smallest of the tied labels -> TMT127.
        let (_sdrf_guard, sdrf) = write_temp_sdrf(
            "tie",
            "source name\tcomment[label]\n\
             pool_a\tTMT131\n\
             pool_b\tTMT127\n",
        )?;
        let channel = resolve_irs_autodetect_channel(&sdrf, "pool")?;
        assert_eq!(channel.as_deref(), Some("TMT127"));
        fs::remove_file(&sdrf)?;
        Ok(())
    }

    #[test]
    fn resolve_irs_autodetect_channel_returns_none_on_no_match()
    -> Result<(), Box<dyn std::error::Error>> {
        // No source name matches the regex -> None (Python skips IRS).
        let (_sdrf_guard, sdrf) = write_temp_sdrf(
            "nomatch",
            "source name\tcomment[label]\n\
             ordinary_1\tTMT126\n",
        )?;
        assert!(resolve_irs_autodetect_channel(&sdrf, "pool")?.is_none());
        fs::remove_file(&sdrf)?;
        // Missing comment[label] column -> None.
        let (_sdrf2_guard, sdrf2) = write_temp_sdrf("nolabel", "source name\npool_1\n")?;
        assert!(resolve_irs_autodetect_channel(&sdrf2, "pool")?.is_none());
        fs::remove_file(&sdrf2)?;
        Ok(())
    }

    #[test]
    fn factorize_batch_labels_uses_first_occurrence_order() {
        let values = ["B", "A", "B", "A", "C"].map(String::from);
        assert_eq!(factorize_batch_labels(&values), vec![0, 1, 0, 1, 2]);
    }
}
