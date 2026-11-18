"""What every method has in common: the interface, the grid it reports on, and windows over
past data that never reach across a hole.

Rule for every method: the value at time t is computed from data received at or before t.
`tests/test_lookahead.py` cuts a day at a random time and checks that nothing before the
cut changes, for every registered method.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import polars as pl

from ..io.day import Book20, Day

_UNITS = {"ms": 1_000_000, "s": 1_000_000_000, "m": 60_000_000_000, "h": 3_600_000_000_000}
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def duration_ns(text: str) -> int:
    """`100ms`, `1s`, `10s`, `5m`, `1h` in nanoseconds."""
    m = _DURATION.fullmatch(text.strip())
    if not m:
        raise ValueError(f"not a duration: {text!r} (examples: 100ms, 1s, 5m)")
    return int(float(m.group(1)) * _UNITS[m.group(2)])


class NotFitted(RuntimeError):
    pass


class Method(ABC):
    """A method turns a day into a table: `ts` plus one column per output, one row per point
    of the day's grid. `fit` sees only training days; `transform` uses what `fit` learned
    and the day's data up to each row's time."""

    name: ClassVar[str]

    def fit(self, train: Sequence[Day]) -> Self:
        return self

    @abstractmethod
    def transform(self, day: Day) -> pl.DataFrame: ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


@dataclass(frozen=True)
class Grid:
    """The times methods report at: every `book_l20` row inside a valid interval, so about
    every 100 ms while the book is valid, and the state of the book there."""

    book: Book20
    #: Start of each row's segment.
    seg_start: np.ndarray

    @property
    def ts(self) -> np.ndarray:
        return self.book.ts

    @property
    def segment(self) -> np.ndarray:
        return self.book.segment

    def __len__(self) -> int:
        return len(self.book)

    def frame(self, columns: dict[str, np.ndarray]) -> pl.DataFrame:
        """A method's output: `ts`, then the columns, with NaN as null."""
        out = {"ts": pl.Series(self.ts, dtype=pl.Int64)}
        for k, v in columns.items():
            v = np.asarray(v)
            out[k] = pl.Series(v, nan_to_null=True) if v.dtype.kind == "f" else pl.Series(v)
        return pl.DataFrame(out)

    def full(self, window: int) -> np.ndarray:
        """Rows whose window `(t - window, t]` lies inside their segment."""
        return self.ts - window >= self.seg_start


def grid(day: Day) -> Grid:
    book = day.book20()
    starts, _ = day.segment_bounds()
    seg_start = starts[book.segment] if len(book) else np.zeros(0, dtype=np.int64)
    return Grid(book=book, seg_start=seg_start)


def mid_ticks(book: Book20) -> np.ndarray:
    """Mid price in price steps (a float: it can fall between two steps)."""
    bid = book.bid_px[:, 0].astype(np.float64)
    ask = book.ask_px[:, 0].astype(np.float64)
    ok = (book.bid_qty[:, 0] > 0) & (book.ask_qty[:, 0] > 0)
    return np.where(ok, (bid + ask) / 2.0, np.nan)


def window_sum(ev_ts: np.ndarray, values: np.ndarray, g: Grid, window: int) -> np.ndarray:
    """For each grid row at t, the sum of `values` of events with time in `(t - window, t]`;
    NaN where that window starts before the row's segment. Events must be sorted by time
    and lie inside valid intervals, so a full window holds only its own segment's events."""
    c = np.concatenate([[0], np.cumsum(values)])
    hi = np.searchsorted(ev_ts, g.ts, side="right")
    lo = np.searchsorted(ev_ts, g.ts - window, side="right")
    out = (c[hi] - c[lo]).astype(np.float64)
    out[~g.full(window)] = np.nan
    return out


def window_last(ev_ts: np.ndarray, values: np.ndarray, g: Grid, ev_seg: np.ndarray) -> np.ndarray:
    """For each grid row, the value of the last event at or before it in the same segment;
    NaN if there is none."""
    i = np.searchsorted(ev_ts, g.ts, side="right") - 1
    ok = i >= 0
    j = np.clip(i, 0, None)
    ok &= ev_seg[j] == g.segment if len(ev_seg) else False
    out = np.full(len(g), np.nan)
    if len(values):
        out[ok] = values[j[ok]]
    return out


def rolling_extreme(values: np.ndarray, g: Grid, window: int, fn: np.ufunc) -> np.ndarray:
    """For each grid row at t, `fn` (np.maximum or np.minimum) over grid values in
    `(t - window, t]`, NaN where the window is not full. A sparse table answers each range
    in two lookups."""
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    lo = np.searchsorted(g.ts, g.ts - window, side="right")
    hi = np.arange(n)  # inclusive
    longest = int((hi - lo + 1).max())
    table = [values.astype(np.float64)]
    k = 1
    while (1 << k) <= longest:
        prev = table[-1]
        half = 1 << (k - 1)
        table.append(fn(prev[:-half], prev[half:]))
        k += 1
    length = hi - lo + 1
    lvl = np.floor(np.log2(np.maximum(length, 1))).astype(np.int64)
    for level in np.unique(lvl):
        rows = lvl == level
        t = table[level]
        a = lo[rows]
        b = hi[rows] - (1 << level) + 1
        out[rows] = fn(t[a], t[b])
    out[~g.full(window)] = np.nan
    return out


def same_segment_prev(g: Grid) -> np.ndarray:
    """Rows whose previous grid row is in the same segment."""
    ok = np.zeros(len(g), dtype=bool)
    ok[1:] = g.segment[1:] == g.segment[:-1]
    return ok


def window_label(window: str) -> str:
    return window.replace(".", "_")
