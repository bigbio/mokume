use super::*;
use crate::exports::{IonValues, PeptideExportKey};

mod cases {
    use super::*;

    #[test]
    fn peptide_export_ions_merge_max_in_id_order() -> mokume_core::Result<()> {
        let mut export = super::PeptideExport::default();
        for (peptide, intensity) in [(2, 1.0e16), (1, 0.5), (0, 1.0), (1, 1.0), (2, 20.0)] {
            export.push(
                super::IntermediateMeasurement {
                    protein: ProteinId::new(0),
                    sample: SampleId::new(0),
                    peptide: PeptideId::new(peptide),
                    canonical: PeptideId::new(0),
                    intensity,
                    protein_name: "P",
                    peptide_canonical: "PEPTIDE",
                    sample_name: "S",
                    run_file_name: "run",
                    sdrf_record: None,
                },
                intensity * 2.0,
            )?;
        }
        let ions = export.rows[&super::PeptideExportKey {
            protein: ProteinId::new(0),
            sample: SampleId::new(0),
            canonical: PeptideId::new(0),
            context: 0,
        }]
            .iter()
            .collect::<Vec<_>>();
        let expected = |peptide, intensity: f64| {
            (
                PeptideId::new(peptide),
                super::IonValues {
                    intensity,
                    unnormalized: intensity * 2.0,
                },
            )
        };
        assert_eq!(
            ions,
            [expected(0, 1.0), expected(1, 1.0), expected(2, 1.0e16)]
        );
        assert_eq!(
            ions.iter().map(|(_, values)| values.intensity).sum::<f64>(),
            (1.0 + 1.0) + 1.0e16
        );
        assert_eq!(std::mem::size_of::<super::ExportIons>(), 24);
        assert_eq!(export.contexts.len(), 1);
        Ok(())
    }

    #[test]
    fn peptide_export_streams_lexically_sorted_contexts() -> Result<(), Box<dyn std::error::Error>>
    {
        let table = SdrfTable::from_reader(concat!(
            "source name\tcomment[data file]\tcharacteristics[biological replicate]\tfactor value[condition]\tcomment[technical replicate]\n",
            "S\ta\t2\tA\t2\nS\ta\t10\tB\t2\nS\tb\t10\tA\t2\nS\ta\t10\tA\t10\nS\ta\t10\tA\t2\n"
        ).as_bytes())?;
        let names = super::PeptideExportNames {
            proteins: vec!["Z", "A"],
            samples: vec!["z", "a"],
            canonicals: vec!["ZZ", "AA"],
        };
        let export = contextual_peptide_export(&table, &names)?;
        let allowed = [0, 1]
            .into_iter()
            .flat_map(|protein| {
                [0, 1].map(move |sample| CellKey {
                    protein: ProteinId::new(protein),
                    sample: SampleId::new(sample),
                })
            })
            .collect();
        assert_eq!(export.contexts.len(), 5);
        let root = tempfile::tempdir()?;
        let path = root.path().join("peptides.csv");
        let keys = export.ordered_keys(&allowed, &HashSet::new(), &names);
        export.write_csv(
            &path,
            export.exported_rows(&keys, false, true, None, &names),
            false,
            false,
        )?;
        let mut expected = "ProteinName,PeptideCanonical,SampleID,BioReplicate,Condition,Run,TechReplicate,NormIntensity\n".to_owned();
        for protein in ["A", "Z"] {
            for canonical in ["AA", "ZZ"] {
                for sample in ["a", "z"] {
                    for context in ["10,A,a,2", "10,A,a,10", "10,A,b,2", "10,B,a,2", "2,A,a,2"] {
                        expected.push_str(&format!("{protein},{canonical},{sample},{context},3\n"));
                    }
                }
            }
        }
        assert_eq!(std::fs::read(path)?, expected.as_bytes());
        Ok(())
    }
}
