use mokume_core::{
    BatchCorrectionConfig, DifferentialExpressionConfig, DirectLfqConfig, FeatureToProteinsConfig,
    FilterConfig, ImputationConfig, InputConfig, IrsConfig, MaxLfqConfig, MokumeError,
    NormalizationConfig, OutputConfig, OutputFormat, QuantMethod, RatioConfig, RuntimeConfig,
    TmtConfig, TmtRowMerge, TransferredFeatures,
};

mod differential_expression;
mod imputation;
mod irs;
mod pibaq;

use super::Features2ProteinsArgs;
use crate::parsers::{QuantMethodArg, DEFAULT_TOPN_PEPTIDES};
use differential_expression::resolve_differential_expression;
use imputation::resolve_imputation;
use irs::resolve_irs;
use pibaq::{pibaq_config, validate_pibaq_options};

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

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
    tmt: TmtConfig,
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
    let tmt = resolve_tmt(&args)?;
    let ratio = resolve_ratio(&args, quantification.method)?;
    let imputation = resolve_imputation(&args)?;
    let differential_expression = resolve_differential_expression(&args, quantification.method)?;
    Ok(build_config(
        &args,
        ResolvedOptions {
            quantification,
            batch,
            irs,
            tmt,
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
    validate_export_precision(args, method)?;
    if method == QuantMethod::Pibaq && args.min_unique.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "piBAQ defines its denominator independently; do not pass --min-unique"
                .to_owned(),
        });
    }
    validate_input_for_quantification(args, method)?;
    validate_protein_fdr_request(args, method)?;
    validate_transferred_features_request(args, method)?;
    validate_method_requirements(args, method)?;
    let manages_normalization = matches!(
        method,
        QuantMethod::DirectLfq
            | QuantMethod::Ratio
            | QuantMethod::PeptideCount
            | QuantMethod::SpectralCount
    );
    let (run_default, sample_default) = default_normalization(args, method, manages_normalization);
    let run_normalization = args
        .run_normalization
        .as_deref()
        .map_or(run_default, |method| method.replace('-', "_"));
    let sample_normalization = args
        .sample_normalization
        .as_deref()
        .map_or(sample_default, |method| method.replace('-', ""));
    validate_normalization_options(
        args,
        method,
        manages_normalization,
        &run_normalization,
        &sample_normalization,
    )?;
    Ok(QuantificationOptions {
        method,
        topn_peptides: topn.unwrap_or(DEFAULT_TOPN_PEPTIDES),
        run_normalization,
        sample_normalization,
    })
}

/// Default run and sample normalization for `method`. Global-median takes the
/// median of each sample's detected features, which moves with detection depth
/// and spike-in composition, so MaxLFQ defaults to the hierarchical (DirectLFQ)
/// alignment of shared peptide species unless normalization proteins ask for a
/// median over those proteins.
fn default_normalization(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
    manages_normalization: bool,
) -> (String, String) {
    if manages_normalization {
        return ("none".to_owned(), "none".to_owned());
    }
    let defaults = NormalizationConfig::default();
    if method == QuantMethod::MaxLfq && args.normalization_proteins.is_none() {
        (defaults.run_method, "hierarchical".to_owned())
    } else {
        (defaults.run_method, defaults.sample_method)
    }
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
    if args.sdrf.is_none()
        && args
            .sample_normalization
            .as_deref()
            .is_some_and(|method| method.eq_ignore_ascii_case("condition-median"))
    {
        return Err(MokumeError::InvalidInput {
            message: "conditionmedian sample normalization requires --sdrf option".to_owned(),
        });
    }
    Ok(())
}

/// `--filter-protein-fdr` needs QPX feature input with `pg_global_qvalue`; the
/// MSstats table carries no protein q-values and spectral-count builds its
/// matrix outside the feature ingest, so both are rejected rather than silently
/// ignoring the requested filter.
fn validate_protein_fdr_request(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if args.filter_protein_fdr.is_none() {
        return Ok(());
    }
    if args.msstats.is_some() {
        return Err(invalid_input(
            "--filter-protein-fdr requires QPX --parquet input with a populated `pg_global_qvalue` column; MSstats input carries no protein q-values",
        ));
    }
    if method == QuantMethod::SpectralCount {
        return Err(invalid_input(
            "--filter-protein-fdr is not supported with --quant-method spectral-count",
        ));
    }
    Ok(())
}

/// `--transferred-features` reads `id_run_file_name` from QPX feature input;
/// MSstats input has no such column and spectral-count does not read features.
fn validate_transferred_features_request(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if args
        .transferred_features
        .as_deref()
        .is_some_and(|value| value.eq_ignore_ascii_case("drop"))
        && (args.msstats.is_some() || method == QuantMethod::SpectralCount)
    {
        return Err(invalid_input(
            "--transferred-features requires QPX --parquet feature input and a feature-based quantification method",
        ));
    }
    Ok(())
}

/// Inputs, outputs, and FASTA options that only some quantification methods use.
fn validate_method_requirements(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if method == QuantMethod::Ratio && args.msstats.is_some() {
        return Err(invalid_input(
            "Ratio quantification requires PSM-level QPX input; MSstats feature tables do not contain PSM evidence",
        ));
    }
    validate_pibaq_options(args, method)?;
    if method == QuantMethod::Ratio && args.sdrf.is_none() {
        return Err(invalid_input("Ratio quantification requires --sdrf option"));
    }
    if matches!(
        method,
        QuantMethod::DirectLfq | QuantMethod::Ratio | QuantMethod::SpectralCount
    ) {
        for (name, path) in [
            ("export-peptides", &args.export_peptides),
            (
                "export-unnormalized-peptides",
                &args.export_unnormalized_peptides,
            ),
        ] {
            if path.is_some() {
                return Err(invalid_input(format!(
                    "{name} is not supported by {method} quantification"
                )));
            }
        }
    }
    if args.coverage_threshold.is_some() && args.sdrf.is_none() {
        return Err(invalid_input("coverage-threshold requires --sdrf option"));
    }
    Ok(())
}

fn validate_normalization_options(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
    manages_normalization: bool,
    run_normalization: &str,
    sample_normalization: &str,
) -> mokume_core::Result<()> {
    if args.normalization_proteins.is_some()
        && !matches!(
            sample_normalization.to_ascii_lowercase().as_str(),
            "globalmedian" | "conditionmedian"
        )
    {
        return Err(invalid_input(
            "--normalization-proteins requires globalmedian or conditionmedian sample normalization",
        ));
    }
    if manages_normalization
        && (!run_normalization.eq_ignore_ascii_case("none")
            || !sample_normalization.eq_ignore_ascii_case("none")
            || args.normalization_proteins.is_some())
    {
        let reason = if matches!(
            method,
            QuantMethod::PeptideCount | QuantMethod::SpectralCount
        ) {
            "does not use intensity normalization"
        } else {
            "manages normalization internally"
        };
        return Err(invalid_input(format!(
            "{method} {reason}; use --run-normalization none and --sample-normalization \
             none, and do not pass --normalization-proteins"
        )));
    }
    Ok(())
}

fn validate_lfq_options(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if method != QuantMethod::DirectLfq && args.export_ions.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--export-ions requires --quant-method directlfq".to_owned(),
        });
    }
    if method != QuantMethod::DirectLfq && args.directlfq_min_nonan.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-min-nonan requires --quant-method directlfq".to_owned(),
        });
    }
    if method != QuantMethod::DirectLfq && args.directlfq_num_samples_quadratic.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-num-samples-quadratic only applies to DirectLFQ".to_owned(),
        });
    }
    if method != QuantMethod::DirectLfq && args.directlfq_no_sample_normalization {
        return Err(MokumeError::InvalidInput {
            message: "--directlfq-no-sample-normalization only applies to DirectLFQ".to_owned(),
        });
    }
    if method != QuantMethod::MaxLfq && args.maxlfq_min_ratio_count.is_some() {
        return Err(MokumeError::InvalidInput {
            message: "--maxlfq-min-ratio-count requires --quant-method maxlfq".to_owned(),
        });
    }
    if method != QuantMethod::MaxLfq && args.stabilize {
        return Err(MokumeError::InvalidInput {
            message: "--stabilize requires --quant-method maxlfq".to_owned(),
        });
    }
    Ok(())
}

fn validate_export_precision(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if !matches!(method, QuantMethod::MaxLfq | QuantMethod::DirectLfq)
        && args.export_precision.is_some()
    {
        return Err(MokumeError::InvalidInput {
            message: "--export-precision requires --quant-method maxlfq or directlfq".to_owned(),
        });
    }
    Ok(())
}

fn resolve_batch(args: &Features2ProteinsArgs) -> mokume_core::Result<BatchCorrectionConfig> {
    let method_supplied = args.batch_method.is_some();
    let method = args.batch_method.clone().map_or_else(
        || BatchCorrectionConfig::default().method,
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
    if args.batch_correction {
        validate_batch_columns(args, &method)?;
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

fn validate_batch_columns(args: &Features2ProteinsArgs, method: &str) -> mokume_core::Result<()> {
    let column_method = method.eq_ignore_ascii_case("column");
    if column_method && args.batch_column.is_none() {
        return Err(invalid_input(
            "Batch correction with method 'column' requires --batch-column option",
        ));
    }
    if !column_method && args.batch_column.is_some() {
        return Err(invalid_input(
            "--batch-column requires --batch-method column",
        ));
    }
    if args.sdrf.is_none() && (args.batch_column.is_some() || !args.batch_covariate.is_empty()) {
        return Err(invalid_input(
            "Batch correction with --batch-column or --batch-covariate requires --sdrf option",
        ));
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
            .unwrap_or_else(|| RatioConfig::default().fraction_merge),
    })
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
                    FilterConfig::default().min_unique_peptides
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
            min_ratio_count: args
                .maxlfq_min_ratio_count
                .unwrap_or(MaxLfqConfig::default().min_ratio_count),
            stabilize: args.stabilize,
        },
        pibaq: pibaq_config(args),
        directlfq: directlfq_config(args),
        batch: resolved.batch,
        irs: resolved.irs,
        tmt: resolved.tmt,
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
        transferred_features: if args
            .transferred_features
            .as_deref()
            .is_some_and(|mode| mode.eq_ignore_ascii_case("drop"))
        {
            TransferredFeatures::Drop
        } else {
            TransferredFeatures::Keep
        },
    }
}

fn output_config(args: &Features2ProteinsArgs) -> OutputConfig {
    OutputConfig {
        protein_matrix: args.output.clone(),
        export_peptides: args.export_peptides.clone(),
        export_unnormalized_peptides: args.export_unnormalized_peptides.clone(),
        export_unimputed: args.export_unimputed.clone(),
        export_ions: args.export_ions.clone(),
        export_precision: args.export_precision.clone(),
        format: OutputFormat::PythonCompatible,
    }
}

fn resolve_tmt(args: &Features2ProteinsArgs) -> mokume_core::Result<TmtConfig> {
    let tmt = TmtConfig {
        row_merge: if args
            .tmt_row_merge
            .as_deref()
            .is_some_and(|mode| mode.eq_ignore_ascii_case("best-row"))
        {
            TmtRowMerge::BestRow
        } else {
            TmtRowMerge::MaxPerChannel
        },
        impurity_table: args.tmt_impurity.clone(),
        interference_floor: args.tmt_interference_floor,
        floor_quantile: args
            .tmt_floor_quantile
            .unwrap_or(TmtConfig::default().floor_quantile),
    };
    if args.msstats.is_some() && tmt.is_active() {
        return Err(invalid_input(
            "TMT reporter corrections (--tmt-row-merge best-row, --tmt-impurity, --tmt-interference-floor) require QPX input; MSstats tables hold one reporter channel per row",
        ));
    }
    Ok(tmt)
}

fn directlfq_config(args: &Features2ProteinsArgs) -> DirectLfqConfig {
    let defaults = DirectLfqConfig::default();
    DirectLfqConfig {
        min_nonan: args.directlfq_min_nonan.unwrap_or(defaults.min_nonan),
        num_samples_quadratic: args
            .directlfq_num_samples_quadratic
            .unwrap_or(defaults.num_samples_quadratic),
        normalize_samples: !args.directlfq_no_sample_normalization,
    }
}
