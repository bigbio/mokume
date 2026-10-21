use mokume_core::{DifferentialExpressionConfig, MokumeError, QuantMethod};

use super::{invalid_input, Features2ProteinsArgs};
use crate::parsers::DeLog2FcArg;

pub(super) fn resolve_differential_expression(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<DifferentialExpressionConfig> {
    let defaults = DifferentialExpressionConfig::default();
    let enabled = de_options_supplied(args);
    let method = args.de_method.clone().unwrap_or(defaults.method);
    let resolved_method = resolved_de_method(&method, quantification);
    validate_de_method_options(args, &method, resolved_method)?;
    if enabled {
        validate_de_requirements(args, &method)?;
    }
    let (log2fc_threshold, auto_effect_size_gate) = args
        .de_log2fc_threshold
        .unwrap_or(DeLog2FcArg::Fixed(defaults.log2fc_threshold))
        .into_config();
    Ok(DifferentialExpressionConfig {
        enabled,
        contrasts: de_contrasts(args),
        contrasts_file: args.de_contrast_file.clone(),
        method,
        ensemble_methods: (!args.de_ensemble_method.is_empty())
            .then(|| args.de_ensemble_method.clone()),
        ensemble_min_k: args.de_ensemble_min_k.unwrap_or(defaults.ensemble_min_k),
        log2fc_threshold,
        effect_size_gate: args
            .de_effect_size_gate
            .clone()
            .map(|value| value.replace('-', "_"))
            .or(auto_effect_size_gate),
        fdr_threshold: args.de_fdr_threshold.unwrap_or(defaults.fdr_threshold),
        fdr_method: args.de_fdr_method.clone().unwrap_or(defaults.fdr_method),
        output: args.de_output.clone(),
    })
}

fn de_options_supplied(args: &Features2ProteinsArgs) -> bool {
    !args.de_contrast.is_empty()
        || args.de_contrast_file.is_some()
        || args.de_method.is_some()
        || !args.de_ensemble_method.is_empty()
        || args.de_ensemble_min_k.is_some()
        || args.de_log2fc_threshold.is_some()
        || args.de_effect_size_gate.is_some()
        || args.de_fdr_threshold.is_some()
        || args.de_fdr_method.is_some()
        || args.de_output.is_some()
}

fn de_contrasts(args: &Features2ProteinsArgs) -> Option<Vec<String>> {
    (!args.de_contrast.is_empty()).then(|| {
        args.de_contrast
            .chunks_exact(2)
            .map(|groups| format!("{} vs {}", groups[0], groups[1]))
            .collect()
    })
}

fn resolved_de_method(method: &str, quantification: QuantMethod) -> &str {
    if method.eq_ignore_ascii_case("auto") {
        if quantification == QuantMethod::DirectLfq {
            "deqms"
        } else {
            "limrots"
        }
    } else {
        method
    }
}

fn validate_de_method_options(
    args: &Features2ProteinsArgs,
    method: &str,
    resolved_method: &str,
) -> mokume_core::Result<()> {
    if args.de_ensemble_min_k.is_some() && !method.eq_ignore_ascii_case("ensemble") {
        return Err(MokumeError::InvalidInput {
            message: "--de-ensemble-min-k only applies to --de-method ensemble".to_owned(),
        });
    }
    if args.de_fdr_method.is_some()
        && matches!(
            resolved_method.to_ascii_lowercase().as_str(),
            "rots" | "limrots"
        )
    {
        return Err(MokumeError::InvalidInput {
            message: format!(
                "--de-fdr-method does not apply to {resolved_method}, which retains its permutation FDR"
            ),
        });
    }
    Ok(())
}

fn validate_de_requirements(args: &Features2ProteinsArgs, method: &str) -> mokume_core::Result<()> {
    if args.sdrf.is_none() {
        return Err(invalid_input(
            "differential expression requires an SDRF file (--sdrf)",
        ));
    }
    if args.de_output.is_none() {
        return Err(invalid_input(
            "differential expression requires --de-output so results are not discarded",
        ));
    }
    if !args.de_ensemble_method.is_empty() && !method.eq_ignore_ascii_case("ensemble") {
        return Err(invalid_input(
            "--de-ensemble-method only applies to --de-method ensemble",
        ));
    }
    Ok(())
}
