"""
Reproducible draws and standard-normal transforms for the stochastic imputers.

Mirrors the Rust kernel's ``stochastic.rs``: SplitMix64 uniforms, Box-Muller
normals and Acklam's normal quantile with one Halley step, so one seed gives
the same draws in both packages. Draw ``k`` depends only on the seed and
``k``, so a whole stream is computed at once.
"""

import math

import numpy as np

_erfc = np.vectorize(math.erfc, otypes=[float])

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)

_NORMAL_A = (
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.357751867269,
    -30.66479806614716,
    2.506628277459239,
)
_NORMAL_B = (
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
    1.0,
)
_NORMAL_C = (
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
)
_NORMAL_D = (
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
    1.0,
)


def uniforms(seed: int, count: int) -> np.ndarray:
    """The first ``count`` uniforms of the SplitMix64 stream of ``seed``."""
    state = np.uint64(seed) + np.arange(1, count + 1, dtype=np.uint64) * _GAMMA
    z = (state ^ (state >> np.uint64(30))) * _MIX_1
    z = (z ^ (z >> np.uint64(27))) * _MIX_2
    # Midpoints of 2^52 bins lie strictly within (0, 1).
    bins = (z ^ (z >> np.uint64(31))) >> np.uint64(12)
    return (bins.astype(float) + 0.5) / 4503599627370496.0


def normals(seed: int, count: int) -> np.ndarray:
    """The first ``count`` Box-Muller normals; each pair of uniforms gives a
    cosine draw, then its sine partner."""
    pairs = (count + 1) // 2
    uniform = uniforms(seed, 2 * pairs)
    radius = np.sqrt(-2.0 * np.log(uniform[0::2]))
    angle = 2.0 * np.pi * uniform[1::2]
    draws = np.empty(2 * pairs)
    draws[0::2] = radius * np.cos(angle)
    draws[1::2] = radius * np.sin(angle)
    return draws[:count]


def normal_cdf(x):
    """Standard normal distribution function."""
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def _polynomial(coefficients, x):
    value = np.zeros_like(x)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def normal_quantile(p):
    """Acklam rational approximation followed by one Halley refinement."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(-2.0 * np.log(np.minimum(p, 1.0 - p)))
        tail = _polynomial(_NORMAL_C, q) / _polynomial(_NORMAL_D, q)
        tail = np.where(p < 0.5, tail, -tail)
        q = p - 0.5
        central = _polynomial(_NORMAL_A, q * q) * q / _polynomial(_NORMAL_B, q * q)
        x = np.where((p >= 0.02425) & (p <= 0.97575), central, tail)
        # Both branches evaluate the lower tail, at -x for x > 0, else at x.
        tail_cdf = normal_cdf(-np.abs(x))
        error = np.where(x > 0.0, (1.0 - p) - tail_cdf, tail_cdf - p)
        correction = error * np.sqrt(2.0 * np.pi) * np.exp(x * x / 2.0)
        x = x - correction / (1.0 + x * correction / 2.0)
    return np.where(p <= 0.0, -np.inf, np.where(p >= 1.0, np.inf, x))


def check_tune_sigma(tune_sigma: float) -> None:
    """Reject a spread multiplier the draws cannot use."""
    if not (np.isfinite(tune_sigma) and tune_sigma > 0.0):
        raise ValueError("tune_sigma must be finite and positive")
