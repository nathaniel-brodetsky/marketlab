"""Realized volatility of the mid over a window."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Day
from .base import (
    Grid,
    Method,
    duration_ns,
    grid,
    mid_ticks,
    rolling_extreme,
    same_segment_prev,
    window_sum,
)


class Volatility(Method):
    """- `rv_<w>`: square root of the sum of squared log changes of the mid between grid
      rows (about 100 ms apart) in the window, in basis points;
    - `range_<w>`: (highest - lowest mid in the window) / mid now, in basis points.
    """

    name = "vol"

    def __init__(self, windows: Sequence[str] = ("1s", "10s", "60s")):
        self.windows = tuple(windows)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        return g.frame(self.columns(g))

    def columns(self, g: Grid) -> dict[str, np.ndarray]:
        mid = mid_ticks(g.book)
        r2 = np.zeros(len(g))
        prev = same_segment_prev(g)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.log(mid[1:] / mid[:-1])
        r2[1:] = np.where(prev[1:], r * r, 0.0)
        r2 = np.nan_to_num(r2)
        cols = {}
        for w in self.windows:
            n = duration_ns(w)
            cols[f"rv_{w}"] = np.sqrt(window_sum(g.ts, r2, g, n)) * 1e4
            hi = rolling_extreme(mid, g, n, np.maximum)
            lo = rolling_extreme(mid, g, n, np.minimum)
            cols[f"range_{w}"] = (hi - lo) / mid * 1e4
        return cols
