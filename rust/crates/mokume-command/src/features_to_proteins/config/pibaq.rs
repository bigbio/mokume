use mokume_core::{PibaqConfig, PibaqFamilyRows, PibaqSharedMode, QuantMethod};

use super::{invalid_input, Features2ProteinsArgs};

/// piBAQ needs a FASTA, and its digestion and shared-peptide options need piBAQ.
pub(super) fn validate_pibaq_options(
    args: &Features2ProteinsArgs,
    method: QuantMethod,
) -> mokume_core::Result<()> {
    if method == QuantMethod::Pibaq && args.fasta.is_none() {
        return Err(invalid_input(
            "piBAQ quantification requires --fasta option",
        ));
    }
    if method != QuantMethod::Pibaq && pibaq_options_supplied(args) {
        return Err(invalid_input(
            "piBAQ FASTA/digestion options require --quant-method pibaq",
        ));
    }
    if pibaq_family_rows_with_proportional(args) {
        return Err(invalid_input(
            "--pibaq-family-rows requires --pibaq-shared unique or stable-ratio",
        ));
    }
    Ok(())
}

fn pibaq_options_supplied(args: &Features2ProteinsArgs) -> bool {
    args.fasta.is_some()
        || args.pibaq_enzyme.is_some()
        || args.pibaq_max_aa.is_some()
        || args.pibaq_min_shared.is_some()
        || args.pibaq_families_yaml.is_some()
        || args.pibaq_min_anchors.is_some()
        || args.pibaq_missed_cleavages.is_some()
        || args.pibaq_shared.is_some()
        || args.pibaq_family_rows.is_some()
        || args.pibaq_evidence.is_some()
}

/// Proportional piBAQ has no family rows, so only `none` may be requested.
fn pibaq_family_rows_with_proportional(args: &Features2ProteinsArgs) -> bool {
    args.pibaq_shared
        .as_deref()
        .is_some_and(|mode| mode.eq_ignore_ascii_case("proportional"))
        && args
            .pibaq_family_rows
            .as_deref()
            .is_some_and(|rows| !rows.eq_ignore_ascii_case("none"))
}

pub(super) fn pibaq_config(args: &Features2ProteinsArgs) -> PibaqConfig {
    let defaults = PibaqConfig::default();
    PibaqConfig {
        enzyme: args.pibaq_enzyme.clone().unwrap_or(defaults.enzyme),
        max_aa: args.pibaq_max_aa.unwrap_or(defaults.max_aa),
        min_shared: args.pibaq_min_shared.unwrap_or(defaults.min_shared),
        families_yaml: args.pibaq_families_yaml.clone(),
        min_anchors: args.pibaq_min_anchors.unwrap_or(defaults.min_anchors),
        high_anchor_threshold: defaults.high_anchor_threshold,
        missed_cleavages: args
            .pibaq_missed_cleavages
            .unwrap_or(defaults.missed_cleavages),
        shared_mode: match args
            .pibaq_shared
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("proportional") => PibaqSharedMode::Proportional,
            Some("stable-ratio") => PibaqSharedMode::StableRatio,
            _ => PibaqSharedMode::Unique,
        },
        family_rows: args
            .pibaq_family_rows
            .as_deref()
            .map(str::to_ascii_lowercase)
            .map(|value| match value.as_str() {
                "all" => PibaqFamilyRows::All,
                "none" => PibaqFamilyRows::None,
                _ => PibaqFamilyRows::FamilyOnly,
            }),
        evidence_output: args.pibaq_evidence.clone(),
    }
}
