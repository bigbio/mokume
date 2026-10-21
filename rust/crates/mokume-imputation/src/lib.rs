//! Missing-value imputation for protein/sample matrices.
//!
//! The single public entry point [`imputed_values`] dispatches on the
//! configured method name and returns the `(protein, sample, value)` triples to
//! fill in. Values are read through a caller-supplied `value_at` accessor so the
//! caller controls the matrix representation (linear or log space).

mod linalg;
mod methods;
mod stochastic;
mod support;

use mokume_core::{ImputationConfig, MokumeError, ProteinId, Result, SampleId};

use methods::{
    bpca_imputed_values, constant_imputed_values, gms_imputed_values, impseq_imputed_values,
    impseqrob_imputed_values, knn_imputed_values, mindet_imputed_values, minprob_imputed_values,
    qrilc_imputed_values, sample_stat_imputed_values, seqknn_imputed_values, SampleImputationStat,
};

pub fn imputed_values<F>(
    config: &ImputationConfig,
    proteins: &[ProteinId],
    samples: &[SampleId],
    mut value_at: F,
) -> Result<Vec<(ProteinId, SampleId, f64)>>
where
    F: FnMut(ProteinId, SampleId) -> Option<f64>,
{
    let method = config.method.trim().to_ascii_lowercase();
    if matches!(method.as_str(), "minprob" | "qrilc")
        && (!config.tune_sigma.is_finite() || config.tune_sigma <= 0.0)
    {
        return Err(MokumeError::InvalidInput {
            message: "tune_sigma must be finite and positive".to_owned(),
        });
    }

    match method.as_str() {
        "" | "none" => Ok(Vec::new()),
        "mindet" => Ok(mindet_imputed_values(
            proteins,
            samples,
            config.quantile,
            &mut value_at,
        )),
        "minprob" => minprob_imputed_values(
            proteins,
            samples,
            config.quantile,
            config.tune_sigma,
            config.seed,
            &mut value_at,
        ),
        "mean" => Ok(sample_stat_imputed_values(
            proteins,
            samples,
            SampleImputationStat::Mean,
            &mut value_at,
        )),
        "median" => Ok(sample_stat_imputed_values(
            proteins,
            samples,
            SampleImputationStat::Median,
            &mut value_at,
        )),
        "most_frequent" => Ok(sample_stat_imputed_values(
            proteins,
            samples,
            SampleImputationStat::MostFrequent,
            &mut value_at,
        )),
        "constant" | "zero" => Ok(constant_imputed_values(
            proteins,
            samples,
            0.0,
            &mut value_at,
        )),
        "knn" => Ok(knn_imputed_values(
            proteins,
            samples,
            config.n_neighbors,
            &mut value_at,
        )),
        "seqknn" => seqknn_imputed_values(proteins, samples, config.n_neighbors, &mut value_at),
        "impseq" => impseq_imputed_values(proteins, samples, &mut value_at),
        "gms" => Ok(gms_imputed_values(proteins, samples, &mut value_at)),
        "bpca" => bpca_imputed_values(proteins, samples, &mut value_at),
        "impseqrob" => impseqrob_imputed_values(proteins, samples, &mut value_at),
        "qrilc" => qrilc_imputed_values(
            proteins,
            samples,
            config.tune_sigma,
            config.seed,
            &mut value_at,
        ),
        _ => unsupported("imputation"),
    }
}

fn unsupported<T>(stage: &'static str) -> Result<T> {
    Err(MokumeError::NotImplemented { stage })
}

#[cfg(test)]
mod tests {
    use super::imputed_values;
    use mokume_core::{ImputationConfig, ProteinId, SampleId};
    use std::collections::HashMap;

    #[test]
    fn knn_imputation_matches_sklearn_uniform_nan_euclidean_oracle() {
        let proteins = [
            ProteinId::new(30),
            ProteinId::new(31),
            ProteinId::new(32),
            ProteinId::new(33),
        ];
        let samples = [SampleId::new(1), SampleId::new(2), SampleId::new(3)];
        let values = HashMap::from([
            ((ProteinId::new(30), SampleId::new(1)), 10.0),
            ((ProteinId::new(30), SampleId::new(2)), 20.0),
            ((ProteinId::new(30), SampleId::new(3)), 30.0),
            ((ProteinId::new(31), SampleId::new(1)), 100.0),
            ((ProteinId::new(31), SampleId::new(3)), 300.0),
            ((ProteinId::new(32), SampleId::new(2)), 200.0),
            ((ProteinId::new(33), SampleId::new(1)), 1000.0),
            ((ProteinId::new(33), SampleId::new(2)), 1000.0),
            ((ProteinId::new(33), SampleId::new(3)), 1000.0),
        ]);

        let Ok(fills) = imputed_values(
            &ImputationConfig {
                enabled: true,
                method: "knn".to_owned(),
                n_neighbors: 2,
                ..ImputationConfig::default()
            },
            &proteins,
            &samples,
            |protein, sample| values.get(&(protein, sample)).copied(),
        ) else {
            panic!("KNN imputation should be implemented");
        };

        let fills = fills
            .into_iter()
            .map(|(protein, sample, value)| ((protein, sample), value))
            .collect::<HashMap<_, _>>();
        assert_raw_close(
            fill_for(&fills, ProteinId::new(31), SampleId::new(2)),
            510.0,
        );
        assert_raw_close(
            fill_for(&fills, ProteinId::new(32), SampleId::new(1)),
            505.0,
        );
        assert_raw_close(
            fill_for(&fills, ProteinId::new(32), SampleId::new(3)),
            515.0,
        );
    }

    #[test]
    fn impseq_conditional_normal_matches_numpy_oracle() {
        // 6 proteins x 3 samples; rows 1-4 complete so n_complete = 4 >=
        // max(2, p=3) and the conditional-normal branch runs (not the fallback).
        // Row 5 misses samples 2,3; row 6 misses sample 1. Oracle from numpy's
        // impSeq core; tolerance 1e-6 covers Gauss-Jordan vs LAPACK float noise.
        let proteins = [
            ProteinId::new(1),
            ProteinId::new(2),
            ProteinId::new(3),
            ProteinId::new(4),
            ProteinId::new(5),
            ProteinId::new(6),
        ];
        let samples = [SampleId::new(1), SampleId::new(2), SampleId::new(3)];
        let values = HashMap::from([
            ((ProteinId::new(1), SampleId::new(1)), 1.0),
            ((ProteinId::new(1), SampleId::new(2)), 2.0),
            ((ProteinId::new(1), SampleId::new(3)), 3.0),
            ((ProteinId::new(2), SampleId::new(1)), 2.0),
            ((ProteinId::new(2), SampleId::new(2)), 4.0),
            ((ProteinId::new(2), SampleId::new(3)), 6.0),
            ((ProteinId::new(3), SampleId::new(1)), 3.0),
            ((ProteinId::new(3), SampleId::new(2)), 5.0),
            ((ProteinId::new(3), SampleId::new(3)), 8.0),
            ((ProteinId::new(4), SampleId::new(1)), 4.0),
            ((ProteinId::new(4), SampleId::new(2)), 7.0),
            ((ProteinId::new(4), SampleId::new(3)), 11.0),
            ((ProteinId::new(5), SampleId::new(1)), 5.0),
            ((ProteinId::new(6), SampleId::new(2)), 9.0),
            ((ProteinId::new(6), SampleId::new(3)), 14.0),
        ]);

        let Ok(fills) = imputed_values(
            &ImputationConfig {
                enabled: true,
                method: "impseq".to_owned(),
                ..ImputationConfig::default()
            },
            &proteins,
            &samples,
            |protein, sample| values.get(&(protein, sample)).copied(),
        ) else {
            panic!("impSeq imputation should be implemented");
        };

        let fills = fills
            .into_iter()
            .map(|(protein, sample, value)| ((protein, sample), value))
            .collect::<HashMap<_, _>>();
        let tol = 1e-6;
        assert!((fill_for(&fills, ProteinId::new(5), SampleId::new(2)) - 8.8).abs() < tol);
        assert!((fill_for(&fills, ProteinId::new(5), SampleId::new(3)) - 13.8).abs() < tol);
        assert!((fill_for(&fills, ProteinId::new(6), SampleId::new(1)) - 5.0).abs() < tol);
        assert_eq!(fills.len(), 3);
    }

    #[test]
    fn impseqrob_tall_matrix_runs_finite_and_deterministic() {
        // More than 500 complete-row pairs exercises rrcovNA's fixed seeded
        // direction selection, whose output must remain reproducible.
        let samples = [
            SampleId::new(1),
            SampleId::new(2),
            SampleId::new(3),
            SampleId::new(4),
        ];
        let proteins: Vec<ProteinId> = (1..=42).map(ProteinId::new).collect();
        let mut values = HashMap::new();
        for p in 1u32..=42 {
            for (col, s) in [1u32, 2, 3, 4].into_iter().enumerate() {
                // Deterministic structured pattern with cross-dimension variation
                // so the pairwise covariance is non-degenerate (rank 4).
                let value = 10.0
                    + f64::from(p) * 0.1
                    + (col as f64) * 2.0
                    + f64::from((p * 7 + s * 3) % 11) * 0.5;
                // Leave one cell missing in each of proteins 41 and 42.
                let missing = (p == 41 && s == 2) || (p == 42 && s == 4);
                if !missing {
                    values.insert((ProteinId::new(p), SampleId::new(s)), value);
                }
            }
        }

        let config = ImputationConfig {
            enabled: true,
            method: "impseqrob".to_owned(),
            ..ImputationConfig::default()
        };
        let run = || {
            imputed_values(&config, &proteins, &samples, |protein, sample| {
                values.get(&(protein, sample)).copied()
            })
        };

        let Ok(first) = run() else {
            panic!("impseqrob should be implemented");
        };
        let Ok(second) = run() else {
            panic!("impseqrob should be implemented");
        };

        // Exactly the two missing cells are filled, both finite.
        assert_eq!(first.len(), 2);
        for (_, _, value) in &first {
            assert!(
                value.is_finite(),
                "tall-matrix impseqrob fill must be finite"
            );
        }
        // Run-to-run determinism (no RNG): identical fills both times.
        let sorted = |mut v: Vec<(ProteinId, SampleId, f64)>| {
            v.sort_by_key(|x| (x.0, x.1));
            v
        };
        let first_sorted = sorted(first);
        let second_sorted = sorted(second);
        assert_eq!(first_sorted.len(), second_sorted.len());
        for (a, b) in first_sorted.iter().zip(&second_sorted) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!(
                (a.2 - b.2).abs() < 1e-12,
                "tall-matrix impseqrob must be deterministic"
            );
        }
    }

    #[test]
    fn most_frequent_imputation_matches_sklearn_oracle() {
        // 5 proteins x 3 samples. Per-column most-frequent (sklearn
        // SimpleImputer/scipy.stats.mode): s1 -> 9 (count 3); s2 -> 3 (all
        // distinct, smallest wins); s3 -> 1 (tie 1 vs 4 at count 2, smallest
        // wins). Oracle captured from sklearn `SimpleImputer(strategy=
        // "most_frequent")`; statistics_ = [9, 3, 1].
        let proteins = [
            ProteinId::new(1),
            ProteinId::new(2),
            ProteinId::new(3),
            ProteinId::new(4),
            ProteinId::new(5),
        ];
        let samples = [SampleId::new(1), SampleId::new(2), SampleId::new(3)];
        let values = HashMap::from([
            ((ProteinId::new(1), SampleId::new(1)), 9.0),
            ((ProteinId::new(1), SampleId::new(2)), 3.0),
            ((ProteinId::new(1), SampleId::new(3)), 4.0),
            ((ProteinId::new(2), SampleId::new(1)), 9.0),
            ((ProteinId::new(2), SampleId::new(2)), 7.0),
            ((ProteinId::new(2), SampleId::new(3)), 4.0),
            ((ProteinId::new(3), SampleId::new(1)), 9.0),
            ((ProteinId::new(3), SampleId::new(2)), 5.0),
            ((ProteinId::new(3), SampleId::new(3)), 1.0),
            ((ProteinId::new(4), SampleId::new(1)), 2.0),
            ((ProteinId::new(4), SampleId::new(3)), 1.0),
        ]);

        let Ok(fills) = imputed_values(
            &ImputationConfig {
                enabled: true,
                method: "most_frequent".to_owned(),
                ..ImputationConfig::default()
            },
            &proteins,
            &samples,
            |protein, sample| values.get(&(protein, sample)).copied(),
        ) else {
            panic!("most_frequent imputation should be implemented");
        };

        let fills = fills
            .into_iter()
            .map(|(protein, sample, value)| ((protein, sample), value))
            .collect::<HashMap<_, _>>();
        // Only the missing cells are filled (p4/s2, and all of p5).
        assert_eq!(fills.len(), 4);
        assert_raw_close(fill_for(&fills, ProteinId::new(4), SampleId::new(2)), 3.0);
        assert_raw_close(fill_for(&fills, ProteinId::new(5), SampleId::new(1)), 9.0);
        assert_raw_close(fill_for(&fills, ProteinId::new(5), SampleId::new(2)), 3.0);
        assert_raw_close(fill_for(&fills, ProteinId::new(5), SampleId::new(3)), 1.0);
    }

    fn fill_for(
        fills: &HashMap<(ProteinId, SampleId), f64>,
        protein: ProteinId,
        sample: SampleId,
    ) -> f64 {
        if let Some(fill) = fills.get(&(protein, sample)).copied() {
            fill
        } else {
            panic!("missing KNN fill for protein {protein:?}, sample {sample:?}");
        }
    }

    fn assert_raw_close(value: f64, expected: f64) {
        assert!(
            (value - expected).abs() < 1e-12,
            "expected {expected}, got {value}"
        );
    }
}
