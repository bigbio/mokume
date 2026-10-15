"""Validated keyword-to-CLI translation for the thin Python wrappers."""

from __future__ import annotations

_ALLOWED = {
    "features2peptides": set(
        "parquet sdrf min_aa min_unique keep_shared_peptides "
        "remove_ids remove_decoy_contaminants "
        "remove_low_frequency_peptides output skip_normalization run_normalization "
        "sample_normalization log2 save_parquet irs_channel irs_autodetect_regex "
        "irs_stat irs_scope aggregation_level filter_config generate_filter_config "
        "filter_min_intensity filter_cv_threshold filter_charge_state "
        "filter_max_missed_cleavages filter_peptide_fdr filter_score "
        "filter_exclude_modification filter_protein_fdr "
        "filter_min_features filter_max_missing_rate".split()
    ),
    "features2proteins": set(
        "parquet msstats psm output sdrf quant_method min_aa "
        "min_unique keep_contaminants "
        "run_normalization sample_normalization normalization_proteins "
        "fasta pibaq_enzyme pibaq_max_aa pibaq_missed_cleavages "
        "pibaq_min_shared pibaq_families pibaq_min_anchors "
        "directlfq_min_nonan directlfq_num_samples_quadratic "
        "export_peptides export_ions batch_correction batch_method batch_column "
        "batch_covariate batch_nonparametric batch_mean_only "
        "batch_ref irs irs_reference_sample irs_sdrf_column "
        "irs_sdrf_value irs_reference_regex irs_stat irs_remove_reference irs_plex_column "
        "irs_missing_reference tmt_row_merge tmt_impurity tmt_interference_floor "
        "tmt_floor_quantile "
        "coverage_threshold min_sample_correlation "
        "ratio_fraction_merge impute_method impute_quantile impute_shift "
        "impute_scale impute_n_neighbors de_contrast "
        "de_contrast_file de_method de_ensemble_method de_ensemble_min_k "
        "de_log2fc de_effect_size_gate de_fdr "
        "de_fdr_method de_output memory threads".split()
    ),
    "peptides2protein": set(
        "fasta peptides quant_method enzyme normalize min_aa max_aa tpa ruler ploidy "
        "organism cpc output qc_report threads directlfq_min_nonan families "
        "min_shared min_anchors high_anchor_threshold".split()
    ),
    "correct-batches": set(
        "input pattern comment sep output sample_id_column protein_id_column "
        "pibaq_raw_column pibaq_corrected_column export_anndata".split()
    ),
    "visualize": {"input", "pattern", "output"},
    "tissuemap": set(
        "input outdir config generate_config tmt_dataset "
        "threads dpi impute_method embedding_method".split()
    ),
    "peptides2protein_qc": {
        "protein_table",
        "qc_report",
        "plot_column",
        "tpa",
        "ruler",
    },
    "peptides2protein_pibaq": set(
        "peptides fasta enzyme output min_aa max_aa ploidy organism cpc qc_report "
        "families families_yaml min_shared min_anchors high_anchor_threshold "
        "normalize tpa ruler verbose".split()
    ),
}

_BOOLEAN = {
    "features2peptides": set(
        "keep_shared_peptides remove_decoy_contaminants "
        "remove_low_frequency_peptides skip_normalization log2 save_parquet".split()
    ),
    "features2proteins": set(
        "keep_contaminants batch_correction batch_nonparametric "
        "batch_mean_only irs irs_remove_reference".split()
    ),
    "peptides2protein": {"normalize", "tpa", "ruler"},
    "correct-batches": {"export_anndata"},
    "visualize": set(),
    "tissuemap": set(),
    "peptides2protein_qc": {"tpa", "ruler"},
    "peptides2protein_pibaq": {"normalize", "tpa", "ruler", "verbose"},
}

_ALIASES = {
    "features2peptides": {},
    "features2proteins": {},
    "peptides2protein": {},
    "tissuemap": {},
    "peptides2protein_pibaq": {"families_yaml": "families"},
}

_REPEAT = {
    "features2peptides": {"filter_charge_state", "filter_exclude_modification"},
    "features2proteins": {
        "batch_covariate",
        "irs_reference_sample",
        "irs_sdrf_value",
        "de_ensemble_method",
    },
    "tissuemap": {"tmt_dataset"},
}

_PAIRED_REPEAT = {"features2proteins": {"de_contrast"}}


def flags_for(command: str, kwargs: dict[str, object]) -> list[str]:
    """Translate validated wrapper kwargs into the command's exact argv."""
    allowed = _ALLOWED[command]
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"{command} got unexpected keyword argument(s): {names}")

    args: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = "--" + _ALIASES.get(command, {}).get(key, key.replace("_", "-"))
        if key in _BOOLEAN[command]:
            _append_boolean(args, command, key, value, flag)
        elif isinstance(value, (list, tuple)):
            _append_sequence(args, command, key, value, flag)
        elif isinstance(value, bool):
            raise TypeError(f"{command}.{key} expects a value, not bool")
        else:
            args.extend((flag, str(value)))
    return args


def _append_boolean(
    args: list[str], command: str, key: str, value: object, flag: str
) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{command}.{key} expects bool")
    if value:
        args.append(flag)


def _append_sequence(
    args: list[str], command: str, key: str, values: list | tuple, flag: str
) -> None:
    if key in _PAIRED_REPEAT.get(command, set()):
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise TypeError(f"{command}.{key} expects pairs of values")
            args.extend((flag, str(value[0]), str(value[1])))
        return
    if key in _REPEAT.get(command, set()):
        for value in values:
            args.extend((flag, str(value)))
        return
    raise TypeError(f"{command}.{key} does not accept a sequence")
