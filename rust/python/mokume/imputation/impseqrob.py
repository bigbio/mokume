"""Robust sequential imputation (pure Python).

Reimplements ``rrcovNA::impSeqRob`` as the Rust kernel does, without rpy2 or
R dependencies. The robust start is the Stahel-Donoho covariance of the
complete rows over at most 500 difference directions, chosen as rrcovNA's
fixed generator does; rows are then imputed fewest missing values first and
join the estimate while their outlyingness stays within the support. When
the complete rows are too few, or their robust covariance lacks rank, the
rows with fewest missing values are first filled with draws from the
multivariate normal fitted by the ``norm`` package's EM, as rrcovNA does.

Reference
---------
Béguin C, Hulliger B. The BACON-EEM Algorithm for Multivariate Outlier
Detection in Incomplete Survey Data. *Survey Methodology*. 2008.
"""

from __future__ import annotations

import bisect
import math
import sys

import numpy as np
import pandas as pd

from mokume.core.logger import get_logger

logger = get_logger("mokume.imputation.impseqrob")

_MAX_EXTRADIR_DIRECTIONS = 500
# Floor for "effectively zero" magnitudes.
_TINY = 1e-12
# Consistency constant turning a MAD into a robust SD estimate.
_MAD_TO_SD = 1.4826

# The covariance core is O(p^3) in the number of columns (variables): it
# inverts a p x p (inverse-)covariance per imputed row. Beyond a few thousand
# columns this hangs for hours on wide cohorts, so refuse by default and point
# the caller at column subsetting or a distribution-wise imputer. Callers who
# really mean it can pass ``max_variables=None``.
_MAX_VARIABLES_GUARD = 2000


def _h_alpha_n(alpha: float, n: int, p: int) -> int:
    """Compute h — matches robustbase:::h.alpha.n."""
    n2 = (n + p + 1) // 2
    return max(int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha)), 1)


def _extradir(data: np.ndarray) -> np.ndarray:
    """Difference directions, as rrcovNA selects them: every row pair up to
    500 pairs, else 500 pairs from its fixed 16-bit congruential generator."""
    n = data.shape[0]
    if n * (n - 1) // 2 <= _MAX_EXTRADIR_DIRECTIONS:
        first, second = np.triu_indices(n, k=1)
        return data[first] - data[second]
    seed = 0

    def draw() -> int:
        nonlocal seed
        seed = (seed * 5761 + 999) % 65536
        return seed * n // 65536

    pairs = []
    for _ in range(_MAX_EXTRADIR_DIRECTIONS):
        i = draw()
        j = draw()
        while i == j:
            j = draw()
        pairs.append((i, j))
    first, second = np.array(pairs).T
    return data[first] - data[second]


def _covSD(x: np.ndarray, h: int) -> dict | None:
    """Stahel-Donoho robust covariance — matches rrcovNA:::.covSD.

    Returns ``None`` when no direction survives normalisation or a direction
    has no spread.
    """
    directions = _extradir(x)
    norms = np.sqrt(np.sum(directions * directions, axis=1))
    keep = norms > _TINY
    if not keep.any():
        return None
    directions = directions[keep] / norms[keep, np.newaxis]

    projections = x @ directions.T
    med_y = np.median(projections, axis=0)
    mad_y = np.median(np.abs(projections - med_y), axis=0) * _MAD_TO_SD
    if not np.all(np.isfinite(mad_y) & (mad_y != 0.0)):
        return None
    d = np.max(np.abs(projections - med_y) / mad_y, axis=1)

    support = x[np.argsort(d, kind="stable")[:h]]
    if support.shape[0] < 2:
        covx = np.zeros((x.shape[1], x.shape[1]))
    else:
        covx = np.atleast_2d(np.cov(support, rowvar=False, ddof=1))
    return {
        "cov": covx,
        "mean": support.mean(axis=0),
        "d": d,
        "A": directions,
        "medY": med_y,
        "madY": mad_y,
    }


def _update_invcov(A: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sherman-Morrison rank-one inverse update — matches rrcovNA:::.update_invcov."""
    Af = A @ f
    denom = 1.0 + f @ Af
    if abs(denom) > _TINY:
        return A - np.outer(Af, Af) / denom
    return A


def _has_rank(x: np.ndarray, alpha: float) -> bool:
    """Whether the robust covariance of the complete rows of ``x`` has full
    rank, by rankMM's scale-dependent cutoff on cyclic Jacobi eigenvalues."""
    complete = x[~np.isnan(x).any(axis=1)]
    p = x.shape[1]
    robust = _covSD(complete, _h_alpha_n(alpha, complete.shape[0], p))
    if robust is None:
        return False
    a = robust["cov"].copy()
    norm = math.sqrt(float(np.sum(a * a)))
    upper = np.triu_indices(p, k=1)
    for _ in range(100):
        if math.sqrt(float(np.sum(a[upper] ** 2))) <= norm * sys.float_info.epsilon:
            break
        for q in range(1, p):
            for r in range(q):
                _rotate_rank_pair(a, r, q)
    diagonal = np.abs(np.diag(a))
    maximum = float(diagonal.max(initial=0.0))
    tolerance = p * sys.float_info.epsilon * maximum
    return bool(np.all(diagonal >= tolerance)) and maximum > 0.0


def _rotate_rank_pair(a: np.ndarray, r: int, q: int) -> None:
    """One Jacobi rotation zeroing ``a[r, q]`` of the symmetric ``a``."""
    off = a[r, q]
    if off == 0.0:
        return
    theta = (a[q, q] - a[r, r]) / (2.0 * off)
    t = (
        1.0
        if theta == 0.0
        else math.copysign(1.0, theta) / (abs(theta) + math.hypot(theta, 1.0))
    )
    c = 1.0 / math.hypot(t, 1.0)
    s = t * c
    others = np.ones(a.shape[0], dtype=bool)
    others[[r, q]] = False
    left = c * a[others, r] - s * a[others, q]
    right = s * a[others, r] + c * a[others, q]
    a[others, r] = left
    a[r, others] = left
    a[others, q] = right
    a[q, others] = right
    a[r, r] -= t * off
    a[q, q] += t * off
    a[r, q] = 0.0
    a[q, r] = 0.0


def _initialize_support(x: np.ndarray, alpha: float) -> None:
    """Fill the rows with fewest missing values from norm draws until the
    complete rows give a full-rank robust covariance, as rrcovNA does."""
    p = x.shape[1]
    n_complete = int((~np.isnan(x).any(axis=1)).sum())
    min_complete = math.ceil(p / alpha)
    if n_complete >= min_complete and (n_complete >= 5 * p or _has_rank(x, alpha)):
        return
    draws = _norm_impute(x)
    target = max(min_complete, n_complete + 1)
    while True:
        if target > x.shape[0]:
            raise ValueError(
                "impSeqRob cannot form a full-rank initial robust covariance"
            )
        x[n_complete:target] = draws[n_complete:target]
        if _has_rank(x, alpha):
            return
        target += 1


def _impute_robust_row(row: np.ndarray, mx: np.ndarray, icovx: np.ndarray) -> None:
    """Conditional mean via the inverse covariance blocks:
    ``x_m = mu_m - icov_mm^-1 icov_mo (x_o - mu_o)``."""
    mvar = np.isnan(row)
    if mvar.all():
        row[mvar] = mx[mvar]
        return
    rhs = icovx[np.ix_(mvar, ~mvar)] @ (row[~mvar] - mx[~mvar])
    try:
        adjustment = np.linalg.solve(icovx[np.ix_(mvar, mvar)], rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("impSeqRob conditional covariance is singular") from exc
    row[mvar] = mx[mvar] - adjustment


def _impseqrob_core(
    data: np.ndarray,
    alpha: float = 0.9,
) -> np.ndarray:
    """Robust sequential imputation — matches rrcovNA::impSeqRob."""
    observed = np.isfinite(data)
    sort_order = np.argsort((~observed).sum(axis=1), kind="stable")
    unsort = np.argsort(sort_order, kind="stable")
    x = np.where(observed, data, np.nan)[sort_order]
    if not observed.all():
        _initialize_support(x, alpha)
        _robust_sequential(x, alpha)
    return x[unsort]


class _Support:
    """Outlyingness directions of the robust start and the sorted outlyingness
    of the rows seen so far; a row joins while within the first ``h``."""

    def __init__(self, covm: dict, h: int):
        self.directions = covm["A"]
        self.med = covm["medY"]
        self.mad = covm["madY"]
        self.sorted = sorted(covm["d"].tolist())
        self.h = h

    def outlyingness(self, row: np.ndarray) -> float:
        """Largest robust z-score of ``row`` over the directions."""
        return float(np.max(np.abs(row @ self.directions.T - self.med) / self.mad))

    def admits(self, dx: float) -> bool:
        """Whether a row of outlyingness ``dx`` joins; records ``dx`` either way."""
        accepted = dx <= self.sorted[min(max(self.h - 1, 0), len(self.sorted) - 1)]
        self.h += accepted
        bisect.insort_left(self.sorted, dx)
        return accepted


def _robust_sequential(x: np.ndarray, alpha: float) -> None:
    """Impute the incomplete rows of ``x`` in order; an accepted row updates
    the mean and, before the next row, the inverse covariance."""
    incomplete = np.isnan(x).any(axis=1)
    h = _h_alpha_n(alpha, int((~incomplete).sum()), x.shape[1])
    covm = _covSD(x[~incomplete], h)
    if covm is None:
        raise ValueError("impSeqRob cannot estimate a nondegenerate robust covariance")
    support = _Support(covm, h)
    mx = mxo = covm["mean"]
    nupdate = h
    icovx = accepted = None
    for i in np.flatnonzero(incomplete):
        if icovx is None:
            try:
                icovx = np.linalg.inv(covm["cov"])
            except np.linalg.LinAlgError as exc:
                raise ValueError("impSeqRob robust covariance is singular") from exc
        elif accepted is not None:
            icovx = _update_invcov(
                (nupdate - 1) / max(nupdate - 2, 1) * icovx,
                (accepted - mxo) / math.sqrt(max(nupdate - 1, 1)),
            )

        _impute_robust_row(x[i], mx, icovx)
        mxo = mx
        accepted = None
        if support.admits(support.outlyingness(x[i])):
            nupdate += 1
            mx = ((nupdate - 1) * mxo + x[i]) / nupdate
            accepted = x[i]


def _norm_impute(x: np.ndarray) -> np.ndarray:
    """Fill every missing cell with a draw from the multivariate normal that
    the ``norm`` package's EM fits, as impSeqRob seeds it (1234567)."""
    missing = np.isnan(x)
    center, scale, standardized = _standardize(x, missing)
    theta = _augmented(*_norm_fit(standardized, missing))
    result = x.copy()
    # norm processes missingness patterns in reverse order, keeping row order
    # within a pattern.
    order = sorted(
        range(x.shape[0]), key=lambda row: tuple(missing[row, ::-1]), reverse=True
    )
    noise = iter(_norm_normals(int(missing.sum())))
    for i in order:
        miss, mean, covariance = _conditional(standardized[i], missing[i], theta)
        if miss.size == 0:
            continue
        draws = _conditional_draw(mean, covariance, [next(noise) for _ in miss])
        result[i, miss] = draws * scale[miss] + center[miss]
    return result


def _conditional_draw(
    mean: np.ndarray, covariance: np.ndarray, noise: list[float]
) -> np.ndarray:
    """``mean`` plus the Cholesky factor of ``covariance`` times ``noise``."""
    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "norm conditional covariance is not positive definite"
        ) from exc
    return mean + lower @ np.array(noise)


def _standardize(
    x: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale each column by its observed mean and population SD."""
    if missing.all(axis=0).any():
        raise ValueError("norm initialization requires observations in every column")
    count = (~missing).sum(axis=0)
    total = np.nansum(x, axis=0)
    center = total / count
    variance = (np.nansum(x * x, axis=0) - total * total / count) / count
    scale = np.where(variance > 0.0, np.sqrt(np.maximum(variance, 0.0)), 1.0)
    return center, scale, (x - center) / scale


def _norm_fit(x: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the standardised rows by norm's EM."""
    n, p = x.shape
    order = sorted(range(n), key=lambda row: tuple(missing[row, ::-1]))
    filled = np.where(missing, 0.0, x)
    known_second = filled.T @ filled
    known = (filled.sum(axis=0), np.triu(known_second) + np.triu(known_second, 1).T)
    mean, cov = np.zeros(p), np.eye(p)
    for _ in range(1000):
        new_mean, second = _expected_statistics(
            x, missing, order, _augmented(mean, cov), known
        )
        new_mean /= n
        new_cov = second / n - np.outer(new_mean, new_mean)
        change = max(
            float(np.max(np.abs(new_mean - mean))), float(np.max(np.abs(new_cov - cov)))
        )
        mean, cov = new_mean, new_cov
        if change <= 1e-4:
            break
    return mean, cov


def _expected_statistics(
    x: np.ndarray,
    missing: np.ndarray,
    order: list[int],
    theta: np.ndarray,
    known: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Sums of the rows and of their cross-products, with missing cells at
    their conditional expectations (the E step)."""
    mean, second = known[0].copy(), known[1].copy()
    for i in order:
        miss, cond_mean, cond_cov = _conditional(x[i], missing[i], theta)
        if miss.size == 0:
            continue
        obs = np.flatnonzero(~missing[i])
        mean[miss] += cond_mean
        cross = np.outer(cond_mean, x[i, obs])
        second[np.ix_(miss, obs)] += cross
        second[np.ix_(obs, miss)] += cross.T
        second[np.ix_(miss, miss)] += np.outer(cond_mean, cond_mean) + cond_cov
    return mean, second


def _augmented(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """norm's parameter matrix: -1, then the mean, bordering the covariance."""
    p = mean.size
    theta = np.empty((p + 1, p + 1))
    theta[0, 0] = -1.0
    theta[0, 1:] = mean
    theta[1:, 0] = mean
    theta[1:, 1:] = cov
    return theta


def _conditional(
    row: np.ndarray, missing: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Missing columns of ``row`` with their conditional mean and covariance.

    Sweeps ``theta`` in place on the observed columns, carrying the swept
    matrix between rows as norm::swpobs does.
    """
    for j in range(row.size):
        if missing[j] == (theta[j + 1, j + 1] < 0.0):
            _sweep(theta, j + 1, -1.0 if missing[j] else 1.0)
    miss = np.flatnonzero(missing)
    obs = np.flatnonzero(~missing)
    mean = theta[0, miss + 1] + row[obs] @ theta[np.ix_(obs + 1, miss + 1)]
    return miss, mean, theta[np.ix_(miss + 1, miss + 1)]


def _sweep(theta: np.ndarray, pivot: int, direction: float) -> None:
    """Sweep (``direction`` 1) or reverse-sweep (-1) ``theta`` on ``pivot``."""
    a = theta[pivot, pivot]
    if a == 0.0 or not np.isfinite(a):
        raise ValueError("norm EM encountered a singular covariance pivot")
    column = theta[:, pivot] / a * direction
    column[pivot] = -1.0 / a
    others = np.arange(theta.shape[0]) != pivot
    block = np.outer(a * column[others], column[others])
    theta[np.ix_(others, others)] -= np.triu(block) + np.triu(block, 1).T
    theta[:, pivot] = column
    theta[pivot, :] = column


def _norm_normals(count: int) -> np.ndarray:
    """Draws of norm 1.0-11.1's gauss() after impSeqRob seeds it: Park-Miller
    uniforms and single-precision Box-Muller; the first draw is discarded."""
    state = 1234567
    scale = np.float32(4.656613e-10)
    # norm's literal 3.141593 as a single-precision float.
    pi = np.float32(3.141593)

    def uniform() -> np.float32:
        nonlocal state
        state = state * 16807 % 2147483647
        return np.float32(state) * scale

    uniform()
    pairs = count // 2 + 1
    draws = np.empty(2 * pairs)
    for k in range(pairs):
        radius = np.sqrt(np.float32(-2.0) * np.float32(math.log(uniform())))
        angle = np.float32(2.0) * pi * uniform()
        draws[2 * k] = radius * np.float32(math.cos(angle))
        draws[2 * k + 1] = radius * np.float32(math.sin(angle))
    return draws[1 : count + 1]


def impute_impseqrob(
    data: pd.DataFrame,
    alpha: float = 0.9,
    max_variables: int | None = _MAX_VARIABLES_GUARD,
) -> pd.DataFrame:
    """Impute missing values using robust sequential regression (pure Python).

    Parameters
    ----------
    data : pd.DataFrame
        Matrix (rows = features, columns = variables) with NaN for missing
        values.
    alpha : float
        MCD support fraction (default 0.9).
    max_variables : int or None
        Guard against the O(p^3) blow-up: raise if the matrix has more than
        this many columns. Set to ``None`` to disable and run regardless.

    Returns
    -------
    pd.DataFrame
        Filled matrix with the same shape and index.
    """
    n_vars = data.shape[1]
    if max_variables is not None and n_vars > max_variables:
        raise ValueError(
            f"impSeqRob is covariance-based and scales as O(p^3) in the number "
            f"of columns (here p={n_vars} > max_variables={max_variables}); on a "
            f"wide matrix it can run for hours. Subset the columns to the samples "
            f"being compared (e.g. the two contrast groups), or use a "
            f"distribution-wise imputer such as qrilc/mindet/minprob. Pass "
            f"max_variables=None to override and run anyway."
        )
    n_missing = int(data.isna().sum().sum())
    filled = _impseqrob_core(data.to_numpy(dtype=float), alpha=alpha)
    result = pd.DataFrame(filled, index=data.index, columns=data.columns)
    logger.info("impSeqRob imputation: %d values imputed", n_missing)
    return result
