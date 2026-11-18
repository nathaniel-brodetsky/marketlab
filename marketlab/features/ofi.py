"""Order-flow imbalance at the best level (Cont, Kukanov, Stoikov 2014) and over several
levels (Xu, Gould, Howison 2019).

For consecutive states n-1, n of a level with bid price and quantity P^b, q^b and ask P^a,
q^a:

    e_n = 1{P^b_n >= P^b_{n-1}} q^b_n - 1{P^b_n <= P^b_{n-1}} q^b_{n-1}
        - 1{P^a_n <= P^a_{n-1}} q^a_n + 1{P^a_n >= P^a_{n-1}} q^a_{n-1}

and OFI over (t - w, t] is the sum of e_n over that window. The first state of a segment has
no predecessor and adds nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Book20, Day
from .base import Grid, Method, NotFitted, duration_ns, grid, mid_ticks, window_sum

#: A missing level, so that its appearing and its going count like any other change.
_NO_BID = np.iinfo(np.int64).min
_NO_ASK = np.iinfo(np.int64).max


def flow(bp, bq, ap, aq, same: np.ndarray) -> np.ndarray:
    """e_n for arrays of consecutive states; 0 where the previous state is in another
    segment. Works on 1-d (one level) or 2-d (rows by levels) arrays."""
    e = np.zeros(bp.shape, dtype=np.int64)
    if len(bp) < 2:
        return e
    p_bp, p_bq, p_ap, p_aq = bp[:-1], bq[:-1], ap[:-1], aq[:-1]
    n_bp, n_bq, n_ap, n_aq = bp[1:], bq[1:], ap[1:], aq[1:]
    e[1:] = (
        (n_bp >= p_bp) * n_bq
        - (n_bp <= p_bp) * p_bq
        - (n_ap <= p_ap) * n_aq
        + (n_ap >= p_ap) * p_aq
    )
    keep = same if e.ndim == 1 else same[:, None]
    return np.where(keep, e, 0)


def _same(segment: np.ndarray) -> np.ndarray:
    ok = np.zeros(len(segment), dtype=bool)
    ok[1:] = segment[1:] == segment[:-1]
    return ok


class OrderFlowImbalance(Method):
    """`ofi_<w>`: OFI at the best level over each window, in base units, from every
    best-price update (`book_ticker`)."""

    name = "ofi"

    def __init__(self, windows: Sequence[str] = ("1s", "10s")):
        self.windows = tuple(windows)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        bt = day.valid("book_ticker", ["recv_ts", "bid_px", "bid_qty", "ask_px", "ask_qty"])
        a = {c: bt[c].to_numpy() for c in bt.columns}
        e = flow(a["bid_px"], a["bid_qty"], a["ask_px"], a["ask_qty"], _same(a["segment"]))
        unit = float(day.qty_step)
        return g.frame(
            {
                f"ofi_{w}": window_sum(a["recv_ts"], e, g, duration_ns(w)) * unit
                for w in self.windows
            }
        )


def level_flows(book: Book20, levels: int) -> np.ndarray:
    """e_n per level between consecutive grid rows, rows by levels."""
    k = slice(0, levels)
    bp = np.where(book.bid_qty[:, k] > 0, book.bid_px[:, k], _NO_BID)
    ap = np.where(book.ask_qty[:, k] > 0, book.ask_px[:, k], _NO_ASK)
    return flow(bp, book.bid_qty[:, k], ap, book.ask_qty[:, k], _same(book.segment))


def mid_change(g: Grid, window: int) -> np.ndarray:
    """Mid at t minus mid at the last grid row at or before t - window in the same segment,
    in ticks."""
    mid = mid_ticks(g.book)
    j = np.searchsorted(g.ts, g.ts - window, side="right") - 1
    ok = (j >= 0) & g.full(window)
    jj = np.clip(j, 0, None)
    ok &= g.segment[jj] == g.segment
    return np.where(ok, mid - mid[jj], np.nan)


class MultiLevelOFI(Method):
    """`ofi_l<m>_<w>` for each of the first `levels` levels and `mlofi_<w>`, their
    combination with weights fitted on the training days: least squares of the mid's change
    over the window on the levels' OFI over the same window (Xu, Gould, Howison 2019).

    Levels are compared between consecutive `book_l20` rows, about 100 ms apart, because
    the tables keep the top 20 levels only at those times: changes inside 100 ms that cancel
    out are not seen.
    """

    name = "mlofi"

    def __init__(self, levels: int = 10, window: str = "1s"):
        if not 1 <= levels <= 20:
            raise ValueError("levels must be 1..20")
        self.levels = levels
        self.window = window
        self.weights_: np.ndarray | None = None
        self.r2_: float | None = None

    def _level_windows(self, day: Day, g: Grid) -> np.ndarray:
        e = level_flows(g.book, self.levels)
        w = duration_ns(self.window)
        unit = float(day.qty_step)
        return (
            np.column_stack([window_sum(g.ts, e[:, m], g, w) * unit for m in range(self.levels)])
            if len(g)
            else np.zeros((0, self.levels))
        )

    def fit(self, train: Sequence[Day]) -> MultiLevelOFI:
        xs, ys = [], []
        for day in train:
            g = grid(day)
            x = self._level_windows(day, g)
            y = mid_change(g, duration_ns(self.window))
            ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
            xs.append(x[ok])
            ys.append(y[ok])
        x = np.concatenate(xs) if xs else np.zeros((0, self.levels))
        y = np.concatenate(ys) if ys else np.zeros(0)
        if len(y) <= self.levels:
            raise ValueError(f"{len(y)} training rows are too few for {self.levels} weights")
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ w
        self.weights_ = w
        self.r2_ = float(1 - (resid @ resid) / (y @ y)) if (y @ y) > 0 else 0.0
        return self

    def transform(self, day: Day) -> pl.DataFrame:
        if self.weights_ is None:
            raise NotFitted(f"{self.name}: fit on training days first")
        g = grid(day)
        x = self._level_windows(day, g)
        cols = {f"ofi_l{m + 1}_{self.window}": x[:, m] for m in range(self.levels)}
        # level by level rather than `x @ weights`: a matrix product may sum in another
        # order for another number of rows, and the same row must give the same value
        combined = np.zeros(len(g))
        for m in range(self.levels):
            combined = combined + x[:, m] * self.weights_[m]
        cols[f"mlofi_{self.window}"] = combined
        return g.frame(cols)
