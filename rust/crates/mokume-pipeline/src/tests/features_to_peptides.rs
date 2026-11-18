use super::*;

mod cases {
    use super::*;

    #[test]
    fn protein_fdr_report_counts_groups_per_run_and_experiment() {
        let group_ids: HashMap<String, u32> = [("P1", 0), ("P2", 1), ("P3", 2)]
            .into_iter()
            .map(|(name, id)| (name.to_owned(), id))
            .collect();
        // P3 has no protein q-value, so it counts as removed and as missing.
        let min_qvalue: HashMap<String, f64> = [("P1", 0.004), ("P2", 0.03)]
            .into_iter()
            .map(|(name, q)| (name.to_owned(), q))
            .collect();
        let run_groups: std::collections::BTreeMap<String, HashSet<u32>> = [
            ("run1".to_owned(), HashSet::from([0, 1, 2])),
            ("run2".to_owned(), HashSet::from([0])),
        ]
        .into_iter()
        .collect();

        let report = super::ProteinFdrReport::new(&group_ids, &min_qvalue, &run_groups, 0.01);

        assert_eq!(report.groups_before, 3);
        assert_eq!(report.groups_after, 1);
        assert_eq!(report.groups_without_qvalue, 1);
        assert_eq!(
            report.runs,
            vec![
                super::ProteinFdrRunCount {
                    run: "run1".to_owned(),
                    before: 3,
                    after: 1,
                },
                super::ProteinFdrRunCount {
                    run: "run2".to_owned(),
                    before: 1,
                    after: 1,
                },
            ]
        );
    }
}
