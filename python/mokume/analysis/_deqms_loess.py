"""One-predictor Gaussian LOESS used by DEqMS::spectraCounteBayes.

R stats loess defaults: degree=2, span=.75, surface="interpolate", cell=.2.
Nodes follow the one-dimensional loess kd tree; fitted values interpolate local
intercepts and derivatives with a cubic Hermite basis. Mirrors the Rust
``mokume_stats`` ``deqms_loess`` module.
"""

import sys

import numpy as np

SPAN = 0.75
CELL = 0.2


def loess_predict(y, x, valid):
    """Fit ``y ~ x`` on the ``valid`` points and predict at every ``x``."""
    order = np.argsort(x[valid], kind="stable")
    points_x = x[valid][order]
    points_y = y[valid][order]
    if len(points_x) < 3:
        return np.full(len(x), np.nan)
    knots = _tree_knots(points_x)
    fits = np.array([_local_quadratic(points_x, points_y, knot) for knot in knots])
    return _interpolate(x, knots, fits)


def _tree_knots(x):
    n = len(x)
    low, high = x[0], x[-1]
    margin = 0.005 * max(high - low, 1e-10 * max(abs(low), abs(high)) + 1e-30)
    knots = [low - margin, high + margin]
    leaf_size = int(np.floor(n * SPAN * CELL))
    pending = [(0, n - 1, knots[0], knots[1])]
    while pending:
        lo, hi, left, right = pending.pop()
        if hi - lo < leaf_size:
            continue
        mid = _split_index(x, lo, hi)
        split = x[mid]
        if split in (left, right):
            continue
        knots.append(split)
        pending.append((lo, mid, left, split))
        if mid < hi:
            pending.append((mid + 1, hi, split, right))
    return np.unique(knots)


def _split_index(x, lo, hi):
    """ehg124 searches offsets 0,+1,-1,+2,-2,... so equal predictors stay together."""
    mid = (lo + hi) // 2
    offset = 0
    while lo <= mid + offset < hi:
        i = mid + offset
        if x[i] != x[i + 1]:
            return i
        offset = 1 - offset if offset <= 0 else -offset
    return mid


def _local_quadratic(x, y, q):
    """Local intercept and slope of the tricube-weighted quadratic fit at ``q``."""
    nf = min(int(np.floor(len(x) * SPAN)), len(x))
    nearest = np.argsort(np.abs(x - q), kind="stable")[:nf]
    delta = x[nearest] - q
    radius = abs(delta[-1])
    if radius == 0.0:
        return np.nan, np.nan
    weight = np.sqrt(np.maximum((1.0 - (np.abs(delta) / radius) ** 3) ** 3, 0.0))
    design = weight[:, None] * np.column_stack([np.ones(nf), delta, delta**2])
    # R equilibrates each column before QR/SVD and truncates at 100*eps*smax.
    scales = np.maximum(np.linalg.norm(design, axis=0), sys.float_info.min)
    beta = np.linalg.lstsq(
        design / scales, weight * y[nearest], rcond=100 * sys.float_info.epsilon
    )[0]
    return beta[0] / scales[0], beta[1] / scales[1]


def _interpolate(query, knots, fits):
    """Cubic Hermite interpolation of the knot fits; NaN outside the knots."""
    upper = np.clip(np.searchsorted(knots, query), 1, len(knots) - 1)
    lower = upper - 1
    width = knots[upper] - knots[lower]
    h = (query - knots[lower]) / width
    phi0 = (1.0 - h) ** 2 * (1.0 + 2.0 * h)
    phi1 = h * h * (3.0 - 2.0 * h)
    psi0 = h * (1.0 - h) ** 2
    psi1 = h * h * (h - 1.0)
    fitted = (
        phi0 * fits[lower, 0]
        + phi1 * fits[upper, 0]
        + (psi0 * fits[lower, 1] + psi1 * fits[upper, 1]) * width
    )
    inside = np.isfinite(query) & (query >= knots[0]) & (query <= knots[-1])
    return np.where(inside, fitted, np.nan)
