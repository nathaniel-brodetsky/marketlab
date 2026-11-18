"""The information coefficient: rank correlation of a prediction with what followed.

Within a day the samples overlap when the horizon is longer than the spacing between them,
so the standard error of a day's IC is Newey-West's. Days are taken as independent: the
recorded days here are a month apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


def _z(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else np.zeros_like(r)


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Standard error of the mean of `x`, with Bartlett weights up to `lags`."""
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(v, 0.0) / n))


@dataclass(frozen=True)
class DayIC:
    ic: float
    se: float
    n: int


def day_ic(pred: np.ndarray, target: np.ndarray, lags: int) -> DayIC:
    """Spearman's correlation as the mean product of standardized ranks, and its
    Newey-West standard error. Rows where either is missing are left out."""
    ok = np.isfinite(pred) & np.isfinite(target)
    n = int(ok.sum())
    if n < 3:
        return DayIC(float("nan"), float("nan"), n)
    p = _z(pred[ok]) * _z(target[ok])
    return DayIC(float(p.mean()), newey_west_se(p, lags), n)


@dataclass(frozen=True)
class Summary:
    mean: float
    se: float
    t: float
    #: One-sided: the chance of a mean this high if the true IC were 0.
    p: float
    positive_share: float
    days: int


def combine(days: Sequence[DayIC]) -> Summary:
    """Mean IC over days, each day weighted equally, with the standard error of that mean."""
    d = [x for x in days if np.isfinite(x.ic) and np.isfinite(x.se)]
    if not d:
        nan = float("nan")
        return Summary(nan, nan, nan, nan, nan, 0)
    ic = np.array([x.ic for x in d])
    se = float(np.sqrt(sum(x.se**2 for x in d)) / len(d))
    t = float(ic.mean() / se) if se > 0 else float("nan")
    return Summary(
        mean=float(ic.mean()),
        se=se,
        t=t,
        p=float(stats.norm.sf(t)) if np.isfinite(t) else float("nan"),
        positive_share=float((ic > 0).mean()),
        days=len(d),
    )
