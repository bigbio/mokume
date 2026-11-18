mod coverage;
mod irs;
mod math;
mod run;
mod sample;

pub use coverage::coverage_filtered_proteins;
pub use irs::irs_scaling_factors;
pub use mokume_core::stats::{mean_positive, median, median_finite, quantile_linear};
pub use run::{
    RunCellKey, RunNormalizationMethod, RunNormalizationTransform, parse_run_normalization_method,
    run_normalization_factors, run_normalization_transforms,
};
pub use sample::{
    SampleNormalizationMethod, condition_median_sample_factors, global_median_sample_factors,
    parse_sample_normalization_method, tmm_norm_factors,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationScale {
    Linear,
    Log2,
}

#[cfg(test)]
mod tests {
    use super::{
        RunCellKey, RunNormalizationMethod, RunNormalizationTransform, SampleNormalizationMethod,
        parse_run_normalization_method, parse_sample_normalization_method,
        run_normalization_transforms,
    };
    use std::collections::HashMap;

    #[test]
    fn parses_max_min_run_normalization_method() {
        assert_eq!(
            parse_run_normalization_method("max_min").ok(),
            Some(Some(RunNormalizationMethod::MaxMin))
        );
    }

    #[test]
    fn parses_centering_sample_normalization_methods() {
        assert_eq!(
            parse_sample_normalization_method("quantile").ok(),
            Some(Some(SampleNormalizationMethod::Quantile))
        );
        assert_eq!(
            parse_sample_normalization_method("mediancenter").ok(),
            Some(Some(SampleNormalizationMethod::MedianCenter))
        );
        assert_eq!(
            parse_sample_normalization_method("mean_center").ok(),
            Some(Some(SampleNormalizationMethod::MeanCenter))
        );
    }

    #[test]
    fn max_min_run_transform_aligns_run_ranges_within_sample() {
        let transforms = run_normalization_transforms(
            RunNormalizationMethod::MaxMin,
            HashMap::from([
                (
                    RunCellKey {
                        sample: "S1".to_owned(),
                        run: "run-a".to_owned(),
                    },
                    vec![100.0, 300.0],
                ),
                (
                    RunCellKey {
                        sample: "S1".to_owned(),
                        run: "run-b".to_owned(),
                    },
                    vec![1000.0, 3000.0],
                ),
            ]),
        );

        let run_a = transform_for_run(&transforms, "run-a");
        let run_b = transform_for_run(&transforms, "run-b");

        assert_raw_close(run_a.apply(100.0), 550.0);
        assert_raw_close(run_a.apply(300.0), 1650.0);
        assert_raw_close(run_b.apply(1000.0), 550.0);
        assert_raw_close(run_b.apply(3000.0), 1650.0);
    }

    fn transform_for_run(
        transforms: &HashMap<RunCellKey, RunNormalizationTransform>,
        run: &str,
    ) -> RunNormalizationTransform {
        if let Some(transform) = transforms
            .iter()
            .find_map(|(key, transform)| (key.run == run).then_some(*transform))
        {
            transform
        } else {
            panic!("missing transform for run `{run}`");
        }
    }

    fn assert_raw_close(value: f64, expected: f64) {
        assert!(
            (value - expected).abs() < 1e-12,
            "expected {expected}, got {value}"
        );
    }
}
