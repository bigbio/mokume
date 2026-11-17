use super::*;
use crate::pibaq::allocation::{finalize_family_allocation, plex_median_factors};
use crate::pibaq::evidence::family_evidence_rows;
use crate::pibaq::fasta::protein_mono_weight;

mod cases {
    use super::*;

    #[test]
    fn pibaq_shared_allocation_is_exact_and_conservative() {
        let family = super::ProteinFamily {
            family_id: "A|B".to_owned(),
            members: vec!["A".to_owned(), "B".to_owned()],
        };
        let sample_anchored = SampleId::new(1);
        let sample_unanchored = SampleId::new(2);
        let observations = vec![
            ("unique_a".to_owned(), sample_anchored, 100.0),
            ("shared".to_owned(), sample_anchored, 300.0),
            ("shared".to_owned(), sample_unanchored, 200.0),
        ];
        let owned_peptides = HashSet::from(["unique_a".to_owned(), "shared".to_owned()]);
        let accession_peptides = HashMap::from([
            (
                "A".to_owned(),
                HashSet::from(["unique_a".to_owned(), "shared".to_owned()]),
            ),
            ("B".to_owned(), HashSet::from(["shared".to_owned()])),
        ]);
        let peptide_accessions = HashMap::from([
            ("unique_a".to_owned(), HashSet::from(["A".to_owned()])),
            (
                "shared".to_owned(),
                HashSet::from(["A".to_owned(), "B".to_owned()]),
            ),
        ]);
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let output = super::finalize_family_allocation(
            &family,
            &observations,
            &owned_peptides,
            &accession_peptides,
            &peptide_accessions,
            false,
            &mut proteins,
        );

        let Some(a) = proteins.get("A") else {
            panic!("A output missing");
        };
        let Some(b) = proteins.get("B") else {
            panic!("B output missing");
        };
        let cell = |protein, sample| output.get(&super::CellKey { protein, sample });

        assert!(cell(b, sample_anchored).is_none());
        assert_eq!(
            cell(a, sample_anchored).map(|value| value.norm_intensity),
            Some(400.0)
        );
        assert_eq!(
            cell(a, sample_unanchored).map(|value| value.norm_intensity),
            Some(100.0)
        );
        assert_eq!(
            cell(b, sample_unanchored).map(|value| value.norm_intensity),
            Some(100.0)
        );
        assert_eq!(
            super::classify_evidence(0, 1, 1, 3),
            super::PibaqEvidence::Medium
        );

        let mut equal_proteins = StringIdRegistry::<ProteinId>::new();
        let equal_output = super::finalize_family_allocation(
            &family,
            &observations,
            &owned_peptides,
            &accession_peptides,
            &peptide_accessions,
            true,
            &mut equal_proteins,
        );
        let Some(a) = equal_proteins.get("A") else {
            panic!("A equal output missing");
        };
        let Some(b) = equal_proteins.get("B") else {
            panic!("B equal output missing");
        };
        let norm_intensity = |protein, sample| {
            equal_output
                .get(&super::CellKey { protein, sample })
                .map(|value| value.norm_intensity)
        };
        assert_eq!(norm_intensity(a, sample_anchored), Some(250.0));
        assert_eq!(norm_intensity(b, sample_anchored), Some(150.0));
    }

    #[test]
    fn pibaq_unique_mode_uses_only_unique_peptides() {
        let values = resolved_values(
            PibaqSharedMode::Unique,
            PibaqFamilyRows::FamilyOnly,
            super::PibaqEvidence::Medium,
            &paralog_observations(),
        );
        assert_eq!(values.get(&("A".to_owned(), 1)), Some(&100.0));
        assert_eq!(values.get(&("A".to_owned(), 2)), Some(&80.0));
        assert_eq!(values.get(&("B".to_owned(), 1)), Some(&50.0));
        assert_eq!(values.get(&("B".to_owned(), 2)), None);
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn pibaq_stable_ratio_keeps_member_missing_without_unique_evidence() {
        let values = resolved_values(
            PibaqSharedMode::StableRatio,
            PibaqFamilyRows::FamilyOnly,
            super::PibaqEvidence::Medium,
            &paralog_observations(),
        );
        let close = |key: (&str, u32), expected: f64| {
            let value = values.get(&(key.0.to_owned(), key.1)).copied();
            assert!(
                value.is_some_and(|value| (value - expected).abs() < 1e-9),
                "{key:?}: {value:?} != {expected}"
            );
        };
        // Ratio 2:1 from sample 1 (the only sample where both members have unique peptides).
        close(("A", 1), (100.0 + 200.0) / 2.0);
        close(("B", 1), (50.0 + 100.0) / 2.0);
        // Sample 2: A keeps only its 2/3 share; B has no unique evidence, so no value.
        close(("A", 2), (80.0 + 200.0 * 2.0 / 3.0) / 2.0);
        assert_eq!(values.get(&("B".to_owned(), 2)), None);
    }

    #[test]
    fn pibaq_family_only_families_get_one_family_row() {
        let observations = vec![
            ("shared".to_owned(), SampleId::new(1), 300.0),
            ("shared".to_owned(), SampleId::new(2), 90.0),
        ];
        let values = resolved_values(
            PibaqSharedMode::Unique,
            PibaqFamilyRows::FamilyOnly,
            super::PibaqEvidence::FamilyOnly,
            &observations,
        );
        assert_eq!(values.get(&("A;B".to_owned(), 1)), Some(&100.0));
        assert_eq!(values.get(&("A;B".to_owned(), 2)), Some(&30.0));
        assert_eq!(values.len(), 2);

        let none = resolved_values(
            PibaqSharedMode::Unique,
            PibaqFamilyRows::None,
            super::PibaqEvidence::FamilyOnly,
            &observations,
        );
        assert!(none.is_empty());

        let all = resolved_values(
            PibaqSharedMode::Unique,
            PibaqFamilyRows::All,
            super::PibaqEvidence::Medium,
            &paralog_observations(),
        );
        assert_eq!(all.get(&("A;B".to_owned(), 1)), Some(&150.0));
        assert_eq!(all.get(&("A".to_owned(), 1)), Some(&100.0));
    }

    #[test]
    fn pibaq_evidence_rows_describe_members_and_family_rows() {
        let (family, owned, accession_peptides, peptide_accessions) = two_member_family_fixture();
        let observations = vec![("shared".to_owned(), SampleId::new(1), 300.0)];
        let counts = super::FamilyCounts::new(
            &family,
            &observations,
            &owned,
            &accession_peptides,
            &peptide_accessions,
        );
        let mut proteins = StringIdRegistry::<ProteinId>::new();
        let output = super::finalize_family_resolved(
            &family,
            &observations,
            &counts,
            super::PibaqEvidence::FamilyOnly,
            PibaqSharedMode::Unique,
            PibaqFamilyRows::FamilyOnly,
            &mut proteins,
        );
        let rows = super::family_evidence_rows(
            &family,
            &counts,
            super::PibaqEvidence::FamilyOnly,
            &output,
            PibaqSharedMode::Unique,
            &proteins,
        );
        let types = rows
            .iter()
            .map(|row| (row.row_id.as_str(), row.row_type))
            .collect::<Vec<_>>();
        assert_eq!(
            types,
            vec![
                ("A", super::PibaqRowType::MemberNotQuantified),
                ("B", super::PibaqRowType::MemberNotQuantified),
                ("A;B", super::PibaqRowType::Family),
            ]
        );
        let family_row = &rows[2];
        assert_eq!(family_row.theoretical_peptides, 3);
        assert_eq!(family_row.theoretical_unique, 2);
        assert_eq!(family_row.observed_shared, 1);
        assert_eq!(family_row.samples_quantified, 1);
        assert_eq!(rows[0].theoretical_unique, 1);
        assert_eq!(rows[0].theoretical_shared, 1);
    }

    #[test]
    fn plex_median_factor_is_geometric_median_per_plex() {
        let factors = HashMap::from([
            (
                ProteinId::new(1),
                HashMap::from([("p1".to_owned(), 2.0), ("p2".to_owned(), 0.5)]),
            ),
            (ProteinId::new(2), HashMap::from([("p1".to_owned(), 8.0)])),
            (ProteinId::new(3), HashMap::from([("p1".to_owned(), 4.0)])),
        ]);
        let medians = super::plex_median_factors(&factors);
        assert!(
            medians
                .get("p1")
                .is_some_and(|value| (value - 4.0).abs() < 1e-12)
        );
        assert!(
            medians
                .get("p2")
                .is_some_and(|value| (value - 0.5).abs() < 1e-12)
        );
    }

    // Lock the monoisotopic molecular weight against the pyOpenMS
    // `AASequence.fromString(seq).getMonoWeight()` values that the Python piBAQ
    // TPA path reads via `digest_fasta_full(..., compute_mw=True)`. Captured
    // with `conda run -n Bigbio python -c "from pyopenms import AASequence;
    // print(AASequence.fromString(SEQ).getMonoWeight())"`.
    #[test]
    fn protein_mono_weight_matches_pyopenms() {
        let cases = [
            ("PEPTIDEAK", 998.492047233),
            ("ALYAAEK", 764.4068587863999),
            ("PEPTIDEAKAPEPTIDECKASHAEDPEPK", 3143.460508429001),
            ("THIDPEAKATHIDPECK", 1903.9098218451002),
            ("ACDEFGHIKLMNPQRSTVWY", 2394.1249175321004),
        ];
        for (sequence, expected) in cases {
            let got = super::protein_mono_weight(sequence);
            assert!(
                (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{sequence}: got {got}, expected {expected}"
            );
        }
    }
}
