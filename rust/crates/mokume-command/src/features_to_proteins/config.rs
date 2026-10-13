use mokume_core::{
    BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, FeatureToProteinsConfig,
    FilterConfig, ImputationConfig, InputConfig, IrsConfig, MaxLfqConfig, MokumeError,
    NormalizationConfig, OutputConfig, OutputFormat, PibaqConfig, QuantMethod, RatioConfig,
    RuntimeConfig,
};

use super::Features2ProteinsArgs;
use crate::parsers::{DeLog2FcArg, QuantMethodArg, DEFAULT_TOPN_PEPTIDES};

struct QuantificationOptions {
    method: QuantMethod,
    topn_peptides: usize,
    run_normalization: String,
    sample_normalization: String,
}

struct ResolvedOptions {
    quantification: QuantificationOptions,
    batch: BatchCorrectionConfig,
    irs: IrsConfig,
    ratio: RatioConfig,
    imputation: ImputationConfig,
    differential_expression: DifferentialExpressionConfig,
}

pub(super) fn into_config(
    args: Features2ProteinsArgs,
) -> mokume_core::Result<FeatureToProteinsConfig> {
    let quantification = resolve_quantification(&args)?;
    let batch = resolve_batch(&args)?;
    let irs = resolve_irs(&args, quantification.method)?;
    let ratio = resolve_ratio(&args, quantification.method)?;
    let imputation = resolve_imputation(&args)?;
    let differential_expression = resolve_differential_expression(&args, quantification.method)?;
    Ok(build_config(
        &args,
        ResolvedOptions {
            quantification,
            batch,
            irs,
            ratio,
            imputation,
            differential_expression,
        },
    ))
}

fn resolve_quantification(
    args: &Features2ProteinsArgs,
) -> mokume_core::Result<QuantificationOptions> {
    let QuantMethodArg { method, topn } = args.quant_method;
    validate_lfq_options(args, method)?;
    if method == QuantMethod::Pibaq && args.min_unique.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "piBAQ defines its denominator independently; do not pass --min-unique"
                .to_owned(),
        });
    }
    validate_input_for_quantification(args, method)?;
    let manages_normalization = matches!(
        method,
        QuantMethod::DirectLfq
            | QuantMethod::Ratio
            | QuantMethod::PeptideCount
            | QuantMethod::SpectralCount
    );
    let run_normalization = args.run_normalization.clone().map_or_else(
        || {
            if manages_normalization {
                "none".to_owned()
            } else {
                "median".to_owned()
            }
        },
        |method| method.replace('-', "_"),
    );
    let sample_normalization = args.sample_normalization.clone().map_or_else(
        || {
            if manages_normalization {
                "none".to_owned()
            } else {
                "globalmedian".to_owned()
            }
        },
        |method| method.replace('-', ""),
    );
    Ok(QuantificationOptions {
        method,
        topn_peptides: topn.unwrap_or(DEFAULT_TOPN_PEPTIDES),
        run_normalization,
        sample_normalization,
    })
}

fn validate_input_for_quantification(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if method == QuantMethod::SpectralCount {
        if args.psm.is_none() || args.parquet.is_none() {
            return Err(MokumeError::InvalidInput {
                message: "spectral_count requires matching QPX inputs via --psm and --parquet"
                    .to_owned(),
            });
        }
    } else if args.psm.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--psm only applies to --quant-method spectral_count".to_owned(),
        });
    }
    Ok(())
}

fn validate_lfq_options(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if method != QuantMethod::DirectLfq && args.directlfq_min_nonan.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-min-nonan requires --quant-method directlfq".to_owned(),
        });
    }
    if !matches!(method, QuantMethod::DirectLfq | QuantMethod::MaxLfq)
        && args.directlfq_num_samples_quadratic.is_some()
    {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-num-samples-quadratic only applies to DirectLFQ/MaxLFQ"
                .to_owned(),
        });
    }
    Ok(())
}

fn resolve_batch(args: &Features2ProteinsArgs) -> mokume_core::Result<BatchCorrectionConfig> {
    let method_supplied = args.batch_method.is_some();
    let method = args.batch_method.clone().map_or_else(
        || "sample_prefix".to_owned(),
        |value| value.replace('-', "_"),
    );
    if !args.batch_correction
        && (method_supplied
            || args.batch_column.is_some()
            || !args.batch_covariate.is_empty()
            || args.batch_nonparametric
            || args.batch_mean_only
            || args.batch_ref.is_some())
    {
        return Err(MokumeError::InvalidInput {
            message: "batch options require --batch-correction".to_owned(),
        });
    }
    Ok(BatchCorrectionConfig {
        enabled: args.batch_correction,
        method,
        column: args.batch_column.clone(),
        covariates: (!args.batch_covariate.is_empty()).then(|| args.batch_covariate.clone()),
        parametric: !args.batch_nonparametric,
        mean_only: args.batch_mean_only,
        ref_batch: args.batch_ref.clone(),
    })
}

fn resolve_irs(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<IrsConfig> {
    let reference_samples =
        (!args.irs_reference_sample.is_empty()).then(|| args.irs_reference_sample.clone());
    let selector_count = validate_irs_selectors(args, reference_samples.is_some())?;
    validate_irs_mode(args, quantification, selector_count)?;
    Ok(IrsConfig {
        enabled: args.irs,
        reference_samples,
        sdrf_column: args.irs_sdrf_column.clone(),
        sdrf_values: (!args.irs_sdrf_value.is_empty()).then(|| args.irs_sdrf_value.clone()),
        reference_regex: args
            .irs_reference_regex
            .clone()
            .unwrap_or_else(|| "pool|powder|ref|reference|bridge".to_owned()),
        stat: args.irs_stat.clone().unwrap_or_else(|| "median".to_owned()),
        remove_reference: args.irs_remove_reference,
    })
}

fn validate_irs_selectors(
    args: &Features2ProteinsArgs,
    has_reference_samples: bool,
) -> mokume_core::Result<usize> {
    if args.irs_sdrf_column.is_some() == args.irs_sdrf_value.is_empty() {
        return Err(MokumeError::InvalidInput {
            message: "--irs-sdrf-column and --irs-sdrf-value must be provided together".to_owned(),
        });
    }
    let selector_count = usize::from(has_reference_samples)
        + usize::from(args.irs_sdrf_column.is_some())
        + usize::from(args.irs_reference_regex.is_some());
    if selector_count > 1 {
        return Err(MokumeError::InvalidInput {
            message: "choose one reference selector: samples, SDRF column+values, or regex"
                .to_owned(),
        });
    }
    Ok(selector_count)
}

fn validate_irs_mode(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
    selector_count: usize,
) -> mokume_core::Result<()> {
    if matches!(
        quantification,
        QuantMethod::PeptideCount | QuantMethod::SpectralCount
    ) && args.irs
    {
        return Err(MokumeError::InvalidInput {
            message: format!("{quantification} quantification cannot apply IRS"),
        });
    }
    if quantification == QuantMethod::Ratio {
        if args.irs {
            return Err(MokumeError::InvalidInput {
                message: "Ratio quantification cannot also apply IRS".to_owned(),
            });
        }
        if args.irs_sdrf_column.is_some()
            || !args.irs_sdrf_value.is_empty()
            || args.irs_stat.is_some()
            || args.irs_remove_reference
        {
            return Err(MokumeError::InvalidInput {
                message: "Ratio accepts --irs-reference-sample or --irs-reference-regex; IRS-only options require --irs"
                    .to_owned(),
            });
        }
    } else if !args.irs
        && (selector_count > 0 || args.irs_stat.is_some() || args.irs_remove_reference)
    {
        return Err(MokumeError::InvalidInput {
            message: "IRS options require --irs".to_owned(),
        });
    }
    Ok(())
}

fn resolve_ratio(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<RatioConfig> {
    if args.ratio_fraction_merge.is_some() && quantification != QuantMethod::Ratio {
        return Err(MokumeError::InvalidInput {
            message: "--ratio-fraction-merge only applies to --quant-method ratio".to_owned(),
        });
    }
    Ok(RatioConfig {
        fraction_merge: args
            .ratio_fraction_merge
            .clone()
            .unwrap_or_else(|| "mean".to_owned()),
    })
}

fn resolve_imputation(args: &Features2ProteinsArgs) -> mokume_core::Result<ImputationConfig> {
    let tuning_supplied = args.impute_quantile.is_some()
        || args.impute_shift.is_some()
        || args.impute_scale.is_some()
        || args.impute_n_neighbors.is_some();
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
    Ok(ImputationConfig {
        enabled,
        method,
        quantile: args.impute_quantile.unwrap_or(0.01),
        shift: args.impute_shift.unwrap_or(1.6),
        scale: args.impute_scale.unwrap_or(0.3),
        n_neighbors: args.impute_n_neighbors.unwrap_or(5),
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
    if (args.impute_shift.is_some() || args.impute_scale.is_some()) && method != "minprob" {
        return Err(MokumeError::InvalidInput {
            message: "--impute-shift/--impute-scale only apply to minprob".to_owned(),
        });
    }
    if args.impute_n_neighbors.is_some() && !matches!(method.as_str(), "knn" | "seqknn") {
        return Err(MokumeError::InvalidInput {
            message: "--impute-n-neighbors only applies to knn/seqknn".to_owned(),
        });
    }
    Ok(())
}

fn resolve_differential_expression(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<DifferentialExpressionConfig> {
    let enabled = de_options_supplied(args);
    let method = args.de_method.clone().unwrap_or_else(|| "auto".to_owned());
    let resolved_method = resolved_de_method(&method, quantification);
    validate_de_method_options(args, &method, resolved_method)?;
    let (log2fc_threshold, auto_effect_size_gate) = args
        .de_log2fc_threshold
        .unwrap_or(DeLog2FcArg::Fixed(0.5))
        .into_config();
    Ok(DifferentialExpressionConfig {
        enabled,
        contrasts: de_contrasts(args),
        contrasts_file: args.de_contrast_file.clone(),
        method,
        ensemble_methods: (!args.de_ensemble_method.is_empty())
            .then(|| args.de_ensemble_method.clone()),
        ensemble_min_k: args.de_ensemble_min_k.unwrap_or(2),
        log2fc_threshold,
        effect_size_gate: args
            .de_effect_size_gate
            .clone()
            .map(|value| value.replace('-', "_"))
            .or(auto_effect_size_gate),
        fdr_threshold: args.de_fdr_threshold.unwrap_or(0.05),
        fdr_method: args
            .de_fdr_method
            .clone()
            .unwrap_or_else(|| "bh".to_owned()),
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

fn build_config(
    args: &Features2ProteinsArgs,
    resolved: ResolvedOptions,
) -> FeatureToProteinsConfig {
    let quantification = resolved.quantification;
    FeatureToProteinsConfig {
        input: input_config(args),
        output: output_config(args),
        filtering: FilterConfig {
            min_aa: args.min_aa,
            min_unique_peptides: args.min_unique.unwrap_or(
                if quantification.method == QuantMethod::Pibaq {
                    0
                } else {
                    2
                },
            ),
            remove_contaminants: !args.keep_contaminants,
        },
        normalization: NormalizationConfig {
            run_method: quantification.run_normalization,
            sample_method: quantification.sample_normalization,
            normalization_proteins: args.normalization_proteins.clone(),
        },
        quantification: quantification.method,
        topn_peptides: quantification.topn_peptides,
        maxlfq: MaxLfqConfig {
            ion_alignment: None,
            force_builtin: false,
        },
        pibaq: pibaq_config(args),
        directlfq: directlfq_config(args),
        batch: resolved.batch,
        irs: resolved.irs,
        coverage_threshold: args.coverage_threshold,
        sample_correlation_threshold: args.min_sample_correlation,
        protein_fdr_threshold: args.filter_protein_fdr,
        ratio: resolved.ratio,
        imputation: resolved.imputation,
        differential_expression: resolved.differential_expression,
        runtime: RuntimeConfig {
            memory: args.memory.clone(),
            threads: args.threads,
        },
    }
}

fn input_config(args: &Features2ProteinsArgs) -> InputConfig {
    InputConfig {
        parquet: args.parquet.clone(),
        msstats: args.msstats.clone(),
        psm: args.psm.clone(),
        sdrf: args.sdrf.clone(),
        fasta: args.fasta.clone(),
    }
}

fn output_config(args: &Features2ProteinsArgs) -> OutputConfig {
    OutputConfig {
        protein_matrix: args.output.clone(),
        export_peptides: args.export_peptides.clone(),
        export_ions: args.export_ions.clone(),
        format: OutputFormat::PythonCompatible,
    }
}

fn pibaq_config(args: &Features2ProteinsArgs) -> PibaqConfig {
    PibaqConfig {
        enzyme: args.pibaq_enzyme.clone(),
        max_aa: args.pibaq_max_aa,
        min_shared: args.pibaq_min_shared,
        families_yaml: args.pibaq_families_yaml.clone(),
        min_anchors: args.pibaq_min_anchors,
        high_anchor_threshold: PibaqConfig::default().high_anchor_threshold,
    }
}

fn directlfq_config(args: &Features2ProteinsArgs) -> DirectLfqConfig {
    DirectLfqConfig {
        cores: None,
        min_nonan: args.directlfq_min_nonan.unwrap_or(1),
        num_samples_quadratic: args.directlfq_num_samples_quadratic.unwrap_or(50),
    }
}
