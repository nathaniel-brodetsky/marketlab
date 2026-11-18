"""What the features try to predict: the mid's return and the realized volatility over the
next h seconds after each grid row.

Targets look ahead by definition, so they are kept apart from features (they are not in
the registry and the look-ahead test does not apply to them). Each is computed only
within the row's segment: a horizon that reaches past the end of the segment, into a hole
or past the data, gives no value rather than a return across the hole.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..features.base import Grid, duration_ns, grid, mid_ticks
from ..io.day import Day


def _mid_after(g: Grid, ends: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The grid mid at the last row at or before t + h, and whether that time is still in
    the row's segment."""
    mid = mid_ticks(g.book)
    j = np.searchsorted(g.ts, g.ts + h, side="right") - 1
    inside = g.ts + h < ends[g.segment] if len(g) else np.zeros(0, dtype=bool)
    return mid[np.clip(j, 0, None)] if len(g) else mid, inside


def returns_on(g: Grid, ends: np.ndarray, horizons: Sequence[str]) -> dict[str, np.ndarray]:
    """`ret_<h>` for each horizon on a grid already built; `ends` are the segments' ends."""
    mid = mid_ticks(g.book)
    cols = {}
    for h in horizons:
        later, inside = _mid_after(g, ends, duration_ns(h))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.log(later / mid) * 1e4
        cols[f"ret_{h}"] = np.where(inside, r, np.nan)
    return cols


def future_returns(day: Day, horizons: Sequence[str] = ("1s", "10s", "60s")) -> pl.DataFrame:
    """`ret_<h>`: log(mid(t + h) / mid(t)) in basis points, the mid taken from the last grid
    row at or before t + h; null where t + h leaves the segment."""
    g = grid(day)
    _, ends = day.segment_bounds()
    return g.frame(returns_on(g, ends, horizons))


def future_volatility(day: Day, horizons: Sequence[str] = ("10s", "60s")) -> pl.DataFrame:
    """`vol_<h>`: square root of the sum of squared log changes of the grid mid in
    (t, t + h], in basis points; null where t + h leaves the segment."""
    g = grid(day)
    _, ends = day.segment_bounds()
    mid = mid_ticks(g.book)
    r2 = np.zeros(len(g))
    if len(g) > 1:
        same = g.segment[1:] == g.segment[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.log(mid[1:] / mid[:-1])
        r2[1:] = np.where(same, np.nan_to_num(r * r), 0.0)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    cols = {}
    for h in horizons:
        n = duration_ns(h)
        lo = np.arange(len(g)) + 1  # rows after t
        hi = np.searchsorted(g.ts, g.ts + n, side="right")  # rows up to t + h
        inside = g.ts + n < ends[g.segment] if len(g) else np.zeros(0, dtype=bool)
        v = np.sqrt(np.maximum(c[hi] - c[np.minimum(lo, hi)], 0.0)) * 1e4
        cols[f"vol_{h}"] = np.where(inside, v, np.nan)
    return g.frame(cols)
