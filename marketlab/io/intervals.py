"""Where the data can be used: the intervals in which the recorder's book was valid.

The rules are the contract's (`SCHEMA.md`, "Quality kinds"): the book is valid from a
`synced` or `resync` event until the next `gap`, `invalid_book`, `reconnect`, `overflow`,
or `disconnected` with reason `stopped`. A run that goes on over midnight starts the day
with a valid book (`book_at_start` in the manifest), and a run's data ends with the run.

`late` intervals (data more than 1 s behind the exchange) are excluded too unless asked
for: the book is right there, but it shows the market as it was seconds before, and the
receive times bunch up when the backlog arrives. A late interval ends at `on_time`, or
with the connection (`reconnect`, or `disconnected` with reason `stopped`).

Analysis never bridges a hole: each interval is a separate segment, and anything that
looks back over time (a window, a return, a bar) starts again in each segment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from .contract import Run

START = frozenset({"synced", "resync"})
END = frozenset({"gap", "invalid_book", "reconnect", "overflow"})
STOPPED = "stopped"

_REASON = re.compile(r'reason="((?:[^"\\]|\\.)*)"')


def reason(detail: str) -> str | None:
    """The `reason` of a `disconnected` event's detail."""
    m = _REASON.search(detail)
    return m.group(1) if m else None


def valid_intervals(
    events: Iterable[tuple[int, str, str]],
    runs: Sequence[Run],
    include_late: bool = False,
) -> list[tuple[int, int]]:
    """Intervals `[start, end)` of receive time in which the data can be used, in order.

    `events` are the quality journal's (ts, kind, detail) in journal order; `runs` the runs
    that wrote the day, each covering `[first_recv_ts, last_recv_ts]`.
    """
    events = sorted(events, key=lambda e: e[0])  # stable: journal order within a time

    def usable(book: bool, late: bool) -> bool:
        return book and (include_late or not late)

    out: list[tuple[int, int]] = []
    i = 0
    for run in sorted(runs, key=lambda r: r.first_recv_ts):
        lo, hi = run.first_recv_ts, run.last_recv_ts + 1
        while i < len(events) and events[i][0] < lo:
            i += 1  # an event outside every run cannot open or close anything
        book = run.book_at_start is not None
        late = False
        start = lo if usable(book, late) else None
        while i < len(events) and events[i][0] < hi:
            ts, kind, detail = events[i]
            i += 1
            before = usable(book, late)
            if kind in START:
                book = True
            elif kind in END or (kind == "disconnected" and reason(detail) == STOPPED):
                book = False
                if kind in ("reconnect", "disconnected"):
                    late = False  # a late interval ends with its connection
            elif kind == "late":
                late = True
            elif kind == "on_time":
                late = False
            after = usable(book, late)
            if after and not before:
                start = ts
            elif before and not after:
                if start is not None and ts > start:
                    out.append((start, ts))
                start = None
        if usable(book, late) and start is not None and hi > start:
            out.append((start, hi))
    return out


def segment_ids(ts: np.ndarray, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    """For each time, the index of the interval holding it, or -1."""
    ts = np.asarray(ts, dtype=np.int64)
    if not intervals:
        return np.full(ts.shape, -1, dtype=np.int32)
    starts = np.fromiter((s for s, _ in intervals), dtype=np.int64, count=len(intervals))
    ends = np.fromiter((e for _, e in intervals), dtype=np.int64, count=len(intervals))
    i = np.searchsorted(starts, ts, side="right") - 1
    inside = (i >= 0) & (ts < ends[np.clip(i, 0, None)])
    return np.where(inside, i, -1).astype(np.int32)


def intervals_frame(intervals: Sequence[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment": pl.Series(range(len(intervals)), dtype=pl.Int32),
            "start": pl.Series([s for s, _ in intervals], dtype=pl.Int64),
            "end": pl.Series([e for _, e in intervals], dtype=pl.Int64),
        }
    )
