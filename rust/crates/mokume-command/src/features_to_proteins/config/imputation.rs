use mokume_core::{ImputationConfig, MokumeError};

use super::Features2ProteinsArgs;

pub(super) fn resolve_imputation(
    args: &Features2ProteinsArgs,
) -> mokume_core::Result<ImputationConfig> {
    let tuning_supplied = args.impute_quantile.is_some()
        || args.impute_shift.is_some()
        || args.impute_scale.is_some()
        || args.impute_n_neighbors.is_some()
        || args.impute_seed.is_some()
        || args.impute_tune_sigma.is_some();
    if tuning_supplied && args.impute_method.is_none() {
        return Err(MokumeError::InvalidInput {
            message: "imputation tuning options require --impute-method".to_owned(),
        });
    }
    let method = args
        .impute_method
        .clone()
        .map_or_else(|| "none".to_owned(), |value| value.replace('-', "_"));
    let enabled = args.impute_method.is_some();
    validate_imputation_method(args, &method)?;
    let defaults = ImputationConfig::default();
    Ok(ImputationConfig {
        enabled,
        method,
        quantile: args.impute_quantile.unwrap_or(defaults.quantile),
        n_neighbors: args.impute_n_neighbors.unwrap_or(defaults.n_neighbors),
        seed: args.impute_seed.unwrap_or(defaults.seed),
        tune_sigma: args.impute_tune_sigma.unwrap_or(defaults.tune_sigma),
    })
}

fn validate_imputation_method(
    args: &Features2ProteinsArgs,
    method: &str,
) -> mokume_core::Result<()> {
    let method = method.to_ascii_lowercase();
    if args.impute_quantile.is_some() && !matches!(method.as_str(), "mindet" | "minprob") {
        return Err(MokumeError::InvalidInput {
            message: "--impute-quantile only applies to mindet/minprob".to_owned(),
        });
    }
    if args.impute_shift.is_some() || args.impute_scale.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "MinProb now follows imputeLCMD; use --impute-tune-sigma instead of legacy --impute-shift/--impute-scale".to_owned(),
        });
    }
    if (args.impute_seed.is_some() || args.impute_tune_sigma.is_some())
        && !matches!(method.as_str(), "minprob" | "qrilc")
    {
        return Err(MokumeError::InvalidInput {
            message: "--impute-seed/--impute-tune-sigma only apply to minprob/qrilc".to_owned(),
        });
    }
    if args.impute_tune_sigma.is_some_and(|value| value <= 0.0) {
        return Err(MokumeError::InvalidInput {
            message: "--impute-tune-sigma must be positive".to_owned(),
        });
    }
    if args.impute_n_neighbors.is_some() && !matches!(method.as_str(), "knn" | "seqknn") {
        return Err(MokumeError::InvalidInput {
            message: "--impute-n-neighbors only applies to knn/seqknn".to_owned(),
        });
    }
    Ok(())
}
