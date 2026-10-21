use mokume_core::{IrsConfig, IrsMissingReference, MokumeError, QuantMethod};

use super::{invalid_input, Features2ProteinsArgs};

pub(super) fn resolve_irs(
    args: &Features2ProteinsArgs,
    quantification: QuantMethod,
) -> mokume_core::Result<IrsConfig> {
    let reference_samples =
        (!args.irs_reference_sample.is_empty()).then(|| args.irs_reference_sample.clone());
    let selector_count = validate_irs_selectors(args, reference_samples.is_some())?;
    validate_irs_mode(args, quantification, selector_count)?;
    let defaults = IrsConfig::default();
    Ok(IrsConfig {
        enabled: args.irs,
        reference_samples,
        sdrf_column: args.irs_sdrf_column.clone(),
        sdrf_values: (!args.irs_sdrf_value.is_empty()).then(|| args.irs_sdrf_value.clone()),
        reference_regex: args
            .irs_reference_regex
            .clone()
            .unwrap_or(defaults.reference_regex),
        stat: args.irs_stat.clone().unwrap_or(defaults.stat),
        remove_reference: args.irs_remove_reference,
        plex_column: args.irs_plex_column.clone(),
        missing_reference: match args
            .irs_missing_reference
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("drop") => IrsMissingReference::Drop,
            Some("plex-median") => IrsMissingReference::PlexMedian,
            _ => IrsMissingReference::Keep,
        },
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
        validate_ratio_reference_options(args)?;
    } else if !args.irs
        && (selector_count > 0
            || args.irs_stat.is_some()
            || args.irs_remove_reference
            || args.irs_plex_column.is_some()
            || args.irs_missing_reference.is_some())
    {
        return Err(invalid_input("IRS options require --irs"));
    }
    if args.irs && args.sdrf.is_none() {
        return Err(invalid_input("IRS options require --sdrf option"));
    }
    Ok(())
}

/// Ratio takes its reference samples from the IRS selectors but runs no IRS.
fn validate_ratio_reference_options(args: &Features2ProteinsArgs) -> mokume_core::Result<()> {
    if args.irs {
        return Err(invalid_input("Ratio quantification cannot also apply IRS"));
    }
    if args.irs_sdrf_column.is_some()
        || !args.irs_sdrf_value.is_empty()
        || args.irs_stat.is_some()
        || args.irs_remove_reference
        || args.irs_plex_column.is_some()
        || args.irs_missing_reference.is_some()
    {
        return Err(invalid_input(
            "Ratio accepts --irs-reference-sample or --irs-reference-regex; IRS-only options require --irs",
        ));
    }
    Ok(())
}
