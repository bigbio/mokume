use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Parse the `top<N>` method-name spelling, returning N when `value` is `top`
/// followed by a positive integer (`top1`, `top3`, `top5`, `top10`, ...).
///
/// `top<N>` is the only user-facing spelling of a TopN method: N always comes
/// from the name. [`QuantMethod::TopN`] is a unit variant and carries no N, so
/// `from_str` can only recognize that a name belongs to the TopN family; callers
/// that need the N use this function on the same string and store it in
/// `FeatureToProteinsConfig::topn_peptides`. The bare `topn` spelling carries no
/// N and therefore returns `None`. Matching is case-insensitive, and `top0` is
/// rejected because N must be at least 1.
///
/// This mirrors the Python `_TOPN_METHOD_RE = re.compile(r"^top(\d+)$")` guard
/// in `mokume.commands.features2proteins`.
#[must_use]
pub fn parse_topn_from_method_name(value: &str) -> Option<usize> {
    let lowered = value.trim().to_ascii_lowercase();
    let digits = lowered.strip_prefix("top")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let parsed = digits.parse::<usize>().ok()?;
    (parsed >= 1).then_some(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantMethod {
    DirectLfq,
    Pibaq,
    MaxLfq,
    TopN,
    Sum,
    Median,
    Ratio,
    Abd,
    Intensity,
    PeptideCount,
    SpectralCount,
}

impl QuantMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectLfq => "directlfq",
            Self::Pibaq => "pibaq",
            Self::MaxLfq => "maxlfq",
            Self::TopN => "topn",
            Self::Sum => "sum",
            Self::Median => "median",
            Self::Ratio => "ratio",
            Self::Abd => "abd",
            Self::Intensity => "intensity",
            Self::PeptideCount => "peptide_count",
            Self::SpectralCount => "spectral_count",
        }
    }
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for QuantMethod {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "directlfq" => Ok(Self::DirectLfq),
            "pibaq" => Ok(Self::Pibaq),
            "maxlfq" => Ok(Self::MaxLfq),
            // `topn` is not a CLI spelling any more (the CLI rejects it and points
            // at `top<N>`), but it stays accepted here because it is what
            // [`QuantMethod::as_str`] emits: keeping it makes
            // `from_str(method.as_str())` round-trip, so any internal or
            // config-file value written from `as_str` still reads back.
            "topn" => Ok(Self::TopN),
            // `top<N>` (top1/top3/top5/...) carries N in the name and is resolved
            // by [`parse_topn_from_method_name`] -- same unit variant.
            other if parse_topn_from_method_name(other).is_some() => Ok(Self::TopN),
            "sum" | "all" | "allpeptides" => Ok(Self::Sum),
            "median" => Ok(Self::Median),
            "ratio" => Ok(Self::Ratio),
            "abd" | "abundance" | "tmtabundance" => Ok(Self::Abd),
            "intensity" | "reporter" | "tmtreporterintensity" => Ok(Self::Intensity),
            "peptide_count" | "peptidecount" => Ok(Self::PeptideCount),
            "spectral_count" | "spectralcount" | "count" => Ok(Self::SpectralCount),
            other => Err(format!("unknown quantification method: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{QuantMethod, parse_topn_from_method_name};

    #[test]
    fn parses_topn_family_names() {
        // Every `top<N>` spelling resolves to the same unit variant; the N is
        // read from the name separately.
        for name in ["topn", "top1", "top3", "TOP5", "top10"] {
            assert_eq!(
                QuantMethod::from_str(name).ok(),
                Some(QuantMethod::TopN),
                "`{name}` must parse as the TopN family"
            );
        }
        assert_eq!(parse_topn_from_method_name("topn"), None);
        assert_eq!(parse_topn_from_method_name("top1"), Some(1));
        assert_eq!(parse_topn_from_method_name("top3"), Some(3));
        assert_eq!(parse_topn_from_method_name("TOP5"), Some(5));
        assert_eq!(parse_topn_from_method_name("top10"), Some(10));
        // N must be >= 1, and a non-numeric suffix is not a TopN method.
        assert_eq!(parse_topn_from_method_name("top0"), None);
        assert_eq!(parse_topn_from_method_name("topx"), None);
        assert!(QuantMethod::from_str("top0").is_err());
        // `as_str` keeps its canonical spelling: the variant carries no N.
        assert_eq!(QuantMethod::TopN.as_str(), "topn");
    }

    /// Every `as_str` spelling must read back through `from_str`. This is the
    /// reason `from_str` still accepts the bare `topn` the CLI no longer takes:
    /// `as_str` emits it, so a value serialized from `as_str` must parse again.
    #[test]
    fn as_str_round_trips_through_from_str() {
        for method in [
            QuantMethod::DirectLfq,
            QuantMethod::Pibaq,
            QuantMethod::MaxLfq,
            QuantMethod::TopN,
            QuantMethod::Sum,
            QuantMethod::Median,
            QuantMethod::Ratio,
            QuantMethod::Abd,
            QuantMethod::Intensity,
            QuantMethod::PeptideCount,
            QuantMethod::SpectralCount,
        ] {
            assert_eq!(
                QuantMethod::from_str(method.as_str()).ok(),
                Some(method),
                "`{}` must round-trip through from_str",
                method.as_str()
            );
        }
    }

    #[test]
    fn parses_aliases() {
        assert_eq!(
            QuantMethod::from_str("directlfq").ok(),
            Some(QuantMethod::DirectLfq)
        );
        assert_eq!(QuantMethod::from_str("all").ok(), Some(QuantMethod::Sum));
        assert_eq!(
            QuantMethod::from_str("spectralcount").ok(),
            Some(QuantMethod::SpectralCount)
        );
    }
}
