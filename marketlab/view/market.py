"""What the app's Market screen shows, computed here so that it is tested apart from the app:
a stretch of a day with its holes and late parts, the book on a time grid, its lines, and
the frames of a replay.

Prices are in quote currency and quantities in base units, converted with the day's steps.
The stretch reads a little before its start and after its end (`PAD`, `AFTER`), so that the
last second's order flow at its first row and the next second's return at its last row are
whole; nothing from outside the stretch is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features.base import grid, mid_ticks
from ..features.ofi import MultiLevelOFI
from ..io.day import Book20, Day
from ..io.intervals import segment_ids, valid_intervals
from ..targets import returns_on

NS = 1_000_000_000
PAD = 2 * NS  # read before the start: the last second's order flow needs it
AFTER = 2 * NS  # read after the end: the next second's return needs it

Intervals = list[tuple[int, int]]


@dataclass(frozen=True)
class Trades:
    ts: np.ndarray
    price: np.ndarray  # quote currency
    qty: np.ndarray  # base units
    buy: np.ndarray  # the buyer took liquidity

    def __len__(self) -> int:
        return len(self.ts)


@dataclass(frozen=True)
class Stretch:
    """`[lo, hi)` of one day. The book rows are those with a valid book, late ones included
    (they are right but stale, and shaded); `holes` are the parts without a valid book."""

    date: str
    lo: int
    hi: int
    tick: float
    unit: float
    book: Book20
    trades: Trades
    #: Valid intervals of the day, late ones included: what `book.segment` numbers. The
    #: book rows include the last one before `lo` if the book was valid from it to `lo`.
    intervals: Intervals
    holes: Intervals
    late: Intervals
    #: Level-1 order-flow imbalance over the last second at each book row, base units;
    #: NaN where that second reaches before the row's segment.
    ofi: np.ndarray
    #: The mid's log return over the next second at each book row, bp; NaN where the
    #: second leaves the segment.
    next_return: np.ndarray


def _clip(intervals: Intervals, lo: int, hi: int) -> Intervals:
    return [(max(s, lo), min(e, hi)) for s, e in intervals if s < hi and e > lo]


def _complement(intervals: Intervals, lo: int, hi: int) -> Intervals:
    out, at = [], lo
    for s, e in _clip(intervals, lo, hi):
        if s > at:
            out.append((at, s))
        at = max(at, e)
    if at < hi:
        out.append((at, hi))
    return out


def _minus(a: Intervals, b: Intervals) -> Intervals:
    """Parts of `a` outside `b`; both sorted and disjoint."""
    out = []
    for s, e in a:
        out.extend(_complement(b, s, e))
    return out


def stretch(path: str | Path, lo: int, hi: int) -> Stretch:
    if hi <= lo:
        raise ValueError("the stretch must end after it starts")
    day = Day.open(path, include_late=True, between=(lo - PAD, hi + AFTER))
    q = day.raw("quality", ("ts", "kind", "detail"))
    events = list(zip(q["ts"].to_list(), q["kind"].to_list(), q["detail"].to_list(), strict=True))
    strict = valid_intervals(events, day.runs, include_late=False)
    loose = day.intervals
    g = grid(day)
    ofi = MultiLevelOFI(levels=1, window="1s").level_windows(day, g)
    ofi = ofi[:, 0] if len(g) else np.zeros(0)
    _, ends = day.segment_bounds()
    ret = returns_on(g, ends, ["1s"])["ret_1s"] if len(g) else np.zeros(0)
    keep = (g.ts >= lo) & (g.ts < hi)
    # and the last row before the start, when it is in the segment there: the book as it
    # stood at the start (no time bin shows it; a replay from the start does)
    before = int(np.searchsorted(g.ts, lo, side="left")) - 1
    at_start = segment_ids(np.array([lo]), loose)[0]
    if before >= 0 and at_start >= 0 and g.book.segment[before] == at_start:
        keep[before] = True
    b = g.book
    book = Book20(
        ts=b.ts[keep],
        update_id=b.update_id[keep],
        segment=b.segment[keep],
        bid_px=b.bid_px[keep],
        bid_qty=b.bid_qty[keep],
        ask_px=b.ask_px[keep],
        ask_qty=b.ask_qty[keep],
    )
    t = day.valid("trades", ["recv_ts", "price", "qty", "is_buyer_maker"])
    t = t.filter((t["recv_ts"] >= lo) & (t["recv_ts"] < hi))
    tick, unit = float(day.price_step), float(day.qty_step)
    trades = Trades(
        ts=t["recv_ts"].to_numpy(),
        price=t["price"].to_numpy() * tick,
        qty=t["qty"].to_numpy() * unit,
        buy=~t["is_buyer_maker"].to_numpy(),
    )
    return Stretch(
        date=day.date,
        lo=lo,
        hi=hi,
        tick=tick,
        unit=unit,
        book=book,
        trades=trades,
        intervals=loose,
        holes=_complement(loose, lo, hi),
        late=_minus(_clip(loose, lo, hi), strict),
        ofi=ofi[keep],
        next_return=ret[keep],
    )


# --- the time grid ----------------------------------------------------------------------


def bins(
    s: Stretch, columns: int = 1200, finest: int = 100_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """Equal time bins over the stretch, `finest` long at the least and `columns` at the
    most, and the book row shown in each: the last one inside the bin, or -1 where the bin
    has none (a hole, or no update). Bins rather than rows, so that a hole stays empty on
    the chart instead of being bridged, and a long stretch is thinned before drawing."""
    n = int(max(1, min(columns, -(-(s.hi - s.lo) // finest))))
    edges = s.lo + (np.arange(n + 1, dtype=np.int64) * (s.hi - s.lo)) // n
    last = np.searchsorted(s.book.ts, edges[1:], side="left") - 1
    first = np.searchsorted(s.book.ts, edges[:-1], side="left")
    rows = np.where(last >= first, last, -1)
    return edges, rows


def lines(s: Stretch, rows: np.ndarray) -> dict[str, np.ndarray]:
    """At the chosen rows (-1: NaN): mid and size-weighted mid in quote currency, spread in
    ticks, depth of the 20 levels of each side in base units, the last second's level-1
    order-flow imbalance and the next second's return."""
    b = s.book
    ok = rows >= 0
    r = np.where(ok, rows, 0)

    def pick(v: np.ndarray) -> np.ndarray:
        return np.where(ok, v[r], np.nan) if len(v) else np.full(len(rows), np.nan)

    bp, ap = b.bid_px[:, 0].astype(float), b.ask_px[:, 0].astype(float)
    bq, aq = b.bid_qty[:, 0].astype(float), b.ask_qty[:, 0].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = (bp * aq + ap * bq) / (bq + aq)
    return {
        "mid": pick(mid_ticks(b)) * s.tick,
        "weighted_mid": pick(weighted) * s.tick,
        "spread": pick(ap - bp),
        "bid_depth": pick(b.bid_qty.sum(axis=1).astype(float)) * s.unit,
        "ask_depth": pick(b.ask_qty.sum(axis=1).astype(float)) * s.unit,
        "ofi": pick(s.ofi),
        "next_return": pick(s.next_return),
    }


def heat(s: Stretch, rows: np.ndarray) -> np.ndarray:
    """Quantity in base units at levels 20..1 of the bids then 1..20 of the asks (40 rows,
    bottom to top) for each chosen row (columns); NaN where a bin has no row or a level is
    missing."""
    z = np.full((40, len(rows)), np.nan)
    ok = rows >= 0
    if not ok.any():
        return z
    r = rows[ok]
    bid = s.book.bid_qty[r][:, ::-1].T.astype(float)  # level 20 first, at the bottom
    ask = s.book.ask_qty[r].T.astype(float)
    block = np.vstack([bid, ask]) * s.unit
    block[block == 0] = np.nan
    z[:, ok] = block
    return z


def path(s: Stretch, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """The mid in quote currency at every book row in `[lo, hi)`, with a NaN point between
    rows of different segments, so that a line through it breaks at each hole."""
    k = (s.book.ts >= lo) & (s.book.ts < hi)
    ts, seg = s.book.ts[k], s.book.segment[k]
    mid = mid_ticks(s.book)[k] * s.tick
    breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    return np.insert(ts, breaks, ts[breaks - 1] + 1), np.insert(mid, breaks, np.nan)


def thin_trades(t: Trades, at_most: int = 30_000) -> tuple[Trades, int]:
    """The largest `at_most` trades, in time order, and how many there were."""
    if len(t) <= at_most:
        return t, len(t)
    keep = np.sort(np.argsort(t.qty, kind="stable")[-at_most:])
    return Trades(t.ts[keep], t.price[keep], t.qty[keep], t.buy[keep]), len(t)


# --- the replay -------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    ts: int
    #: Book row shown, -1 inside a hole: the book is not known there.
    row: int
    mid: float  # quote currency, NaN in a hole
    #: Levels as ticks from the mid and base units, best first.
    bid_ticks: np.ndarray
    bid_qty: np.ndarray
    ask_ticks: np.ndarray
    ask_qty: np.ndarray
    #: Trades in the frame's last `step`: ticks from the mid, base units, buyer took.
    trade_ticks: np.ndarray
    trade_qty: np.ndarray
    trade_buy: np.ndarray


def replay(s: Stretch, start: int, seconds: float, step: int = 100_000_000) -> list[Frame]:
    """A frame every `step` from `start` for `seconds`, inside the stretch: the book as last
    received at or before the frame's time, unless the time falls in a hole, and the trades
    since the frame before."""
    start = max(start, s.lo)
    end = min(start + int(seconds * NS), s.hi)
    times = np.arange(start, end, step, dtype=np.int64)
    b, t = s.book, s.trades
    idx = np.searchsorted(b.ts, times, side="right") - 1
    seg = segment_ids(times, s.intervals)
    frames = []
    for k, ts in enumerate(times):
        i = int(idx[k])
        valid = i >= 0 and seg[k] >= 0 and b.segment[i] == seg[k]
        lo_t, hi_t = np.searchsorted(t.ts, [ts - step, ts], side="right")
        if not valid:
            empty = np.zeros(0)
            frames.append(
                Frame(int(ts), -1, np.nan, empty, empty, empty, empty, empty, empty, empty)
            )
            continue
        mid = (b.bid_px[i, 0] + b.ask_px[i, 0]) / 2.0
        bl, al = b.bid_qty[i] > 0, b.ask_qty[i] > 0
        frames.append(
            Frame(
                ts=int(ts),
                row=i,
                mid=mid * s.tick,
                bid_ticks=b.bid_px[i][bl] - mid,
                bid_qty=b.bid_qty[i][bl] * s.unit,
                ask_ticks=b.ask_px[i][al] - mid,
                ask_qty=b.ask_qty[i][al] * s.unit,
                trade_ticks=t.price[lo_t:hi_t] / s.tick - mid,
                trade_qty=t.qty[lo_t:hi_t],
                trade_buy=t.buy[lo_t:hi_t],
            )
        )
    return frames
