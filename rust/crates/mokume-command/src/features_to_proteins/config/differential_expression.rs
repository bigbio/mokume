use mokume_core::{DifferentialExpressionConfig, MokumeError, QuantMethod};

use super::{invalid_input, Features2ProteinsArgs};
use crate::parsers::DeLog2FcArg;

pub(super) fn resolve_differential_expression(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<DifferentialExpressionConfig> {
    let defaults = DifferentialExpressionConfig::default();
    let enabled = de_options_supplied(args);
    let method = args.de.de_method.clone().unwrap_or(defaults.method);
    let resolved_method = resolved_de_method(&method, quantification, args);
    validate_de_method_options(args, &method, resolved_method)?;
    validate_precision_weights(args, quantification, resolved_method)?;
    if enabled {
        validate_de_requirements(args, &method)?;
    }
    let (log2fc_threshold, auto_effect_size_gate) = args
        .de
        .de_log2fc_threshold
        .unwrap_or(DeLog2FcArg::Fixed(defaults.log2fc_threshold))
        .into_config();
    Ok(DifferentialExpressionConfig {
        enabled,
        contrasts: de_contrasts(args),
        contrasts_file: args.de.de_contrast_file.clone(),
        method,
        ensemble_methods: (!args.de.de_ensemble_method.is_empty())
            .then(|| args.de.de_ensemble_method.clone()),
        ensemble_min_k: args.de.de_ensemble_min_k.unwrap_or(defaults.ensemble_min_k),
        log2fc_threshold,
        effect_size_gate: args
            .de
            .de_effect_size_gate
            .clone()
            .map(|value| value.replace('-', "_"))
            .or(auto_effect_size_gate),
        fdr_threshold: args.de.de_fdr_threshold.unwrap_or(defaults.fdr_threshold),
        fdr_method: args.de.de_fdr_method.clone().unwrap_or(defaults.fdr_method),
        output: args.de.de_output.clone(),
        covariates: args.de.de_covariate.clone(),
        block: args.de.de_block.clone(),
        precision_weights: args.de.de_precision_weights,
    })
}

fn de_options_supplied(args: &Features2ProteinsArgs) -> bool {
    !args.de.de_contrast.is_empty()
        || args.de.de_contrast_file.is_some()
        || args.de.de_method.is_some()
        || !args.de.de_covariate.is_empty()
        || args.de.de_block.is_some()
        || args.de.de_precision_weights
        || !args.de.de_ensemble_method.is_empty()
        || args.de.de_ensemble_min_k.is_some()
        || args.de.de_log2fc_threshold.is_some()
        || args.de.de_effect_size_gate.is_some()
        || args.de.de_fdr_threshold.is_some()
        || args.de.de_fdr_method.is_some()
        || args.de.de_output.is_some()
}

fn de_contrasts(args: &Features2ProteinsArgs) -> Option<Vec<String>> {
    (!args.de.de_contrast.is_empty()).then(|| {
        args.de
            .de_contrast
            .chunks_exact(2)
            .map(|groups| format!("{} vs {}", groups[0], groups[1]))
            .collect()
    })
}

fn resolved_de_method<'a>(
    method: &'a str,
    quantification: QuantMethod,
    args: &Features2ProteinsArgs,
) -> &'a str {
    if method.eq_ignore_ascii_case("auto") {
        if quantification == QuantMethod::DirectLfq
            || !args.de.de_covariate.is_empty()
            || args.de.de_block.is_some()
        {
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
    if args.de.de_ensemble_min_k.is_some() && !method.eq_ignore_ascii_case("ensemble") {
        return Err(MokumeError::InvalidInput {
            message: "--de-ensemble-min-k only applies to --de-method ensemble".to_owned(),
        });
    }
    let design_method = matches!(
        resolved_method.to_ascii_lowercase().as_str(),
        "limma" | "deqms"
    );
    if !args.de.de_covariate.is_empty() && !design_method {
        return Err(invalid_input(
            "--de-covariate requires --de-method limma or deqms",
        ));
    }
    if args.de.de_block.is_some() && !design_method {
        return Err(invalid_input(
            "--de-block requires --de-method limma or deqms",
        ));
    }
    if args.de.de_fdr_method.is_some()
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

fn validate_precision_weights(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
    resolved_method: &str,
) -> mokume_core::Result<()> {
    if !args.de.de_precision_weights {
        return Ok(());
    }
    if !matches!(quantification, QuantMethod::MaxLfq | QuantMethod::DirectLfq) {
        return Err(invalid_input(
            "--de-precision-weights requires --quant-method maxlfq or directlfq",
        ));
    }
    if !resolved_method.eq_ignore_ascii_case("deqms") {
        return Err(invalid_input(
            "--de-precision-weights requires --de-method deqms",
        ));
    }
    if !args.de.de_covariate.is_empty() || args.de.de_block.is_some() {
        return Err(invalid_input(
            "--de-precision-weights does not apply to --de-covariate or --de-block designs",
        ));
    }
    Ok(())
}

fn validate_de_requirements(args: &Features2ProteinsArgs, method: &str) -> mokume_core::Result<()> {
    if args.sdrf.is_none() {
        return Err(invalid_input(
            "differential expression requires an SDRF file (--sdrf)",
        ));
    }
    if args.de.de_output.is_none() {
        return Err(invalid_input(
            "differential expression requires --de-output so results are not discarded",
        ));
    }
    if !args.de.de_ensemble_method.is_empty() && !method.eq_ignore_ascii_case("ensemble") {
        return Err(invalid_input(
            "--de-ensemble-method only applies to --de-method ensemble",
        ));
    }
    Ok(())
}
