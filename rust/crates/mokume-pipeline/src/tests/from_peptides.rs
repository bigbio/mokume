use super::*;
use crate::from_peptides::LfqPeptideObservation;

mod cases {
    use super::*;

    #[test]
    fn run_lfq_from_peptides_estimates_and_filters_zeros() {
        let make = |protein: &str, peptide: &str, sample: &str, intensity: f64| {
            super::LfqPeptideObservation {
                protein: protein.to_owned(),
                peptide: peptide.to_owned(),
                sample: sample.to_owned(),
                intensity,
                weight: 1.0,
            }
        };
        // P1 and P2 each carry two peptides across two samples.
        let observations = vec![
            make("P1", "PEP1", "S1", 100.0),
            make("P1", "PEP2", "S1", 200.0),
            make("P1", "PEP1", "S2", 110.0),
            make("P1", "PEP2", "S2", 220.0),
            make("P2", "PEP3", "S1", 50.0),
            make("P2", "PEP4", "S1", 70.0),
            make("P2", "PEP3", "S2", 55.0),
            make("P2", "PEP4", "S2", 77.0),
        ];

        let result = super::run_lfq_from_peptides(&observations, 1, 50);
        let proteins: HashSet<&str> = result.iter().map(|row| row.protein.as_str()).collect();
        assert_eq!(proteins, ["P1", "P2"].into_iter().collect());
        assert!(result.iter().all(|row| row.intensity > 0.0));
        assert_eq!(result.len(), 4); // 2 proteins x 2 samples, zeros dropped

        // A min_nonan above the available ion count drops every protein.
        assert!(super::run_lfq_from_peptides(&observations, 100, 50).is_empty());
    }

    #[test]
    fn run_lfq_from_peptides_is_invariant_to_observation_order() {
        let make = |peptide: &str, sample: &str, intensity: f64| super::LfqPeptideObservation {
            protein: "A5Z2X5".to_owned(),
            peptide: peptide.to_owned(),
            sample: sample.to_owned(),
            intensity,
            weight: 1.0,
        };
        // Minimal real PXD002099 support pattern that exposed DirectLFQ's
        // encounter-order sensitivity: reversing these rows used to shift all
        // three reported cells by about 2% despite identical observations.
        let observations = vec![
            make("LRTDETLR", "Sample 1", 1_661_357.0),
            make("LRTDETLR", "Sample 2", 2_011_746.0),
            make("LRTDETLR", "Sample 3", 771_094.875),
            make("LTGNPELSSLDEVLAK", "Sample 1", 3_310_899.0),
            make("LTGNPELSSLDEVLAK", "Sample 2", 2_914_053.0),
            make("LTGNPELSSLDEVLAK", "Sample 3", 4_332_763.0),
            make("LTGNPELSSLDEVLAK", "Sample 4", 5_018_000.0),
            make("LTGNPELSSLDEVLAK", "Sample 5", 4_421_422.0),
        ];
        let forward = super::run_lfq_from_peptides(&observations, 2, 50)
            .into_iter()
            .map(|row| ((row.protein, row.sample), row.intensity))
            .collect::<HashMap<_, _>>();
        let mut reversed_observations = observations;
        reversed_observations.reverse();
        let reversed = super::run_lfq_from_peptides(&reversed_observations, 2, 50)
            .into_iter()
            .map(|row| ((row.protein, row.sample), row.intensity))
            .collect::<HashMap<_, _>>();

        assert_eq!(forward.len(), 3);
        assert_eq!(
            forward.keys().collect::<HashSet<_>>(),
            reversed.keys().collect()
        );
        for (cell, expected) in forward {
            let actual = reversed.get(&cell).copied().unwrap_or(f64::NAN);
            let tolerance = expected.abs().max(1.0) * 1e-12;
            assert!(
                (actual - expected).abs() <= tolerance,
                "order changed {cell:?}: forward={expected}, reversed={actual}"
            );
        }
    }
}
