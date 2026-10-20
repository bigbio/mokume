use std::str::FromStr;

use mokume_core::quant::parse_topn_from_method_name;
use mokume_core::{parse_memory_to_bytes, NamedScoreFilterConfig, QuantMethod};

/// The fixed `--quant-method` names, i.e. every method whose name carries no
/// parameter. The TopN family is spelled `top<N>` and is not listed here.
const FIXED_QUANT_METHODS: &str = "directlfq, pibaq, maxlfq, sum, median, ratio, abd, intensity, \
peptide-count, spectral-count";

/// Default `topn_peptides` for methods outside the TopN family. The pipeline
/// only reads the field when the method is [`QuantMethod::TopN`], in which case
/// N always comes from the `top<N>` name, so this value is inert -- it just
/// keeps the config field populated.
pub(crate) const DEFAULT_TOPN_PEPTIDES: usize = 3;

/// A fixed fold-change threshold or Python-compatible `auto` estimation.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DeLog2FcArg {
    Fixed(f64),
    Auto,
}

impl DeLog2FcArg {
    pub(crate) fn into_config(self) -> (f64, Option<String>) {
        match self {
            Self::Fixed(value) => (value, None),
            Self::Auto => (0.5, Some("mixture".to_string())),
        }
    }
}

impl FromStr for DeLog2FcArg {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.trim().eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let threshold = value.parse::<f64>().map_err(|_| {
            format!("invalid log2FC threshold `{value}`: expected `auto` or a non-negative number")
        })?;
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(format!(
                "invalid log2FC threshold `{value}`: expected `auto` or a finite, non-negative number"
            ));
        }
        Ok(Self::Fixed(threshold))
    }
}

pub(crate) fn parse_de_log2fc(value: &str) -> std::result::Result<DeLog2FcArg, String> {
    DeLog2FcArg::from_str(value)
}

/// A validated `--quant-method` value: the parsed [`QuantMethod`] plus, for the
/// `top<N>` family, the N spelled in the name.
///
/// This is a plain `FromStr` newtype rather than a clap `ValueEnum` because
/// `top<N>` is an open-ended family (`top1`, `top3`, `top10`, ...) that no fixed
/// variant list can express. Parsing here (instead of keeping a raw `String` and
/// re-parsing later) means an invalid method is rejected by clap at parse time,
/// with the same exit code as any other bad option.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantMethodArg {
    pub(crate) method: QuantMethod,
    /// `Some(N)` only for the `top<N>` family.
    pub(crate) topn: Option<usize>,
}

impl FromStr for QuantMethodArg {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let lowered = value.trim().to_ascii_lowercase();
        if let Some(topn) = parse_topn_from_method_name(&lowered) {
            return Ok(Self {
                method: QuantMethod::TopN,
                topn: Some(topn),
            });
        }
        // Anything else starting with `top` is a malformed TopN name (`top0`,
        // `topx`, ...); say so rather than reporting a generic unknown method.
        if lowered.starts_with("top") {
            return Err(invalid_topn_message(value));
        }
        let internal = match lowered.as_str() {
            "peptide-count" => "peptide_count",
            "spectral-count" => "spectral_count",
            name if name.contains('_') => return Err(unknown_method_message(value)),
            name => name,
        };
        let method = QuantMethod::from_str(internal).map_err(|_| unknown_method_message(value))?;
        Ok(Self { method, topn: None })
    }
}

/// clap `value_parser` for `--quant-method`. A `fn(&str) -> Result<_, String>`
/// is the form clap accepts directly, and it renders the `String` as the usage
/// error message.
pub(crate) fn parse_quant_method(value: &str) -> std::result::Result<QuantMethodArg, String> {
    QuantMethodArg::from_str(value)
}

/// Methods `peptides2protein` implements. It runs on an already-summarized
/// peptide table, so it offers a smaller set than `features2proteins`.
const PEPTIDES2PROTEIN_METHODS: [&str; 4] = ["pibaq", "maxlfq", "sum", "directlfq"];

/// clap `value_parser` for `quantify peptides2protein --quant-method`.
///
/// Applies the same TopN spelling rules as `--quant-method` over this command's
/// smaller method set.
pub(crate) fn parse_peptides2protein_method(value: &str) -> std::result::Result<String, String> {
    let lowered = value.trim().to_ascii_lowercase();
    if PEPTIDES2PROTEIN_METHODS.contains(&lowered.as_str()) {
        return Ok(lowered);
    }
    if parse_topn_from_method_name(&lowered).is_some() {
        return Ok(lowered);
    }
    // A `top`-prefixed name that carries no usable N is a malformed TopN request,
    // not an unknown method; say which of the two it is.
    if lowered.starts_with("top") {
        return Err(invalid_topn_message(value));
    }
    Err(format!(
        "unknown peptides2protein method `{value}`: expected one of {}, or `top<N>` (e.g. `top3`)",
        PEPTIDES2PROTEIN_METHODS.join(", ")
    ))
}

pub(crate) fn parse_memory(value: &str) -> std::result::Result<String, String> {
    parse_memory_to_bytes(value)
        .map(|_| value.to_owned())
        .map_err(|_| {
            format!("invalid memory value `{value}`: expected a positive size such as 1GB or 512MB")
        })
}

pub(crate) fn parse_positive_i32(value: &str) -> std::result::Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("invalid positive integer `{value}`"))?;
    if parsed <= 0 {
        return Err(format!("expected a positive integer, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_positive_f64(value: &str) -> std::result::Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid positive number `{value}`"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("expected a finite positive number, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_finite_f64(value: &str) -> std::result::Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid finite number `{value}`"))?;
    if !parsed.is_finite() {
        return Err(format!("expected a finite number, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_nonnegative_f64(value: &str) -> std::result::Result<f64, String> {
    let parsed = parse_finite_f64(value)?;
    if parsed < 0.0 {
        return Err(format!("expected a non-negative number, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_fraction(value: &str) -> std::result::Result<f64, String> {
    let parsed = parse_finite_f64(value)?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(format!("expected a number between 0 and 1, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_tmt_floor(
    value: &str,
) -> std::result::Result<mokume_core::TmtInterferenceFloor, String> {
    if value.trim().eq_ignore_ascii_case("auto") {
        return Ok(mokume_core::TmtInterferenceFloor::Auto);
    }
    let parsed = parse_finite_f64(value)?;
    if !(0.0..0.5).contains(&parsed) {
        return Err(format!(
            "expected `auto` or a fraction in [0, 0.5), got `{value}`"
        ));
    }
    Ok(mokume_core::TmtInterferenceFloor::Fixed(parsed))
}

pub(crate) fn parse_correlation(value: &str) -> std::result::Result<f64, String> {
    let parsed = parse_finite_f64(value)?;
    if !(-1.0..=1.0).contains(&parsed) {
        return Err(format!("expected a number between -1 and 1, got `{value}`"));
    }
    Ok(parsed)
}

pub(crate) fn parse_named_score_filter(
    value: &str,
) -> std::result::Result<NamedScoreFilterConfig, String> {
    let (name, threshold) = value
        .rsplit_once('=')
        .ok_or_else(|| format!("invalid score filter `{value}`: expected NAME=THRESHOLD"))?;
    let name = name.trim();
    if name.is_empty() {
        return Err("score filter name cannot be empty".to_owned());
    }
    let threshold = threshold.trim().parse::<f64>().map_err(|_| {
        format!("invalid score filter `{value}`: threshold must be a finite number")
    })?;
    if !threshold.is_finite() {
        return Err(format!(
            "invalid score filter `{value}`: threshold must be a finite number"
        ));
    }
    Ok(NamedScoreFilterConfig {
        name: name.to_owned(),
        threshold,
    })
}

pub(crate) fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer `{value}`"))?;
    if parsed == 0 {
        return Err("invalid positive integer `0`".to_owned());
    }
    Ok(parsed)
}

/// Error for a `top`-prefixed name that is not a valid `top<N>` (`top0`, `topx`).
fn invalid_topn_message(value: &str) -> String {
    format!(
        "invalid quantification method `{value}`: a TopN method is spelled `top<N>` with N >= 1 \
(e.g. `top1`, `top3`, `top5`)"
    )
}

/// Error for a name that is neither a fixed method nor a `top<N>`.
fn unknown_method_message(value: &str) -> String {
    format!("unknown quantification method `{value}`: expected one of {FIXED_QUANT_METHODS}, or `top<N>` (e.g. `top3`)")
}
