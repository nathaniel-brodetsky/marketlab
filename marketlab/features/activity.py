"""How busy the market is, and how much of the resting quantity leaves without trading."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Day
from .base import Method, duration_ns, grid, window_sum


class Intensity(Method):
    """Events per second over each window:

    - `trade_rate_<w>`: trades;
    - `ticker_rate_<w>`: changes of the best bid or ask (`book_ticker` rows);
    - `book_rate_<w>`: order book updates at any depth, counted by how far the update ID
      moves, which `book_ticker` carries.
    """

    name = "intensity"

    def __init__(self, windows: Sequence[str] = ("1s", "10s")):
        self.windows = tuple(windows)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        tr = day.valid("trades", ["recv_ts"])
        bt = day.valid("book_ticker", ["recv_ts", "update_id"])
        t_ts = tr["recv_ts"].to_numpy()
        b_ts = bt["recv_ts"].to_numpy()
        uid = bt["update_id"].to_numpy()
        seg = bt["segment"].to_numpy()
        step = np.zeros(len(uid), dtype=np.int64)
        if len(uid) > 1:
            same = seg[1:] == seg[:-1]
            step[1:] = np.where(same, np.maximum(uid[1:] - uid[:-1], 0), 0)
        cols = {}
        for w in self.windows:
            n = duration_ns(w)
            sec = n / 1e9
            cols[f"trade_rate_{w}"] = window_sum(t_ts, np.ones(len(t_ts), np.int64), g, n) / sec
            cols[f"ticker_rate_{w}"] = window_sum(b_ts, np.ones(len(b_ts), np.int64), g, n) / sec
            cols[f"book_rate_{w}"] = window_sum(b_ts, step, g, n) / sec
        return g.frame(cols)


class CancelShare(Method):
    """`cancel_share_<w>`: the share of the quantity that left the recorded levels without
    a trade at that price to explain it. An estimate, for two reasons:

    - Aggregated L2 data cannot tell a cancel from a fill; fills are taken from the trades
      at the level's price between two `book_l20` rows, the rest of a level's decrease is
      counted as cancelled.
    - The rows are about 100 ms apart: an order placed and cancelled between them is not
      seen, and a level that drops out of the top 20 without crossing the best price may
      only have been pushed down, so it is left out.
    """

    name = "cancels"

    def __init__(self, windows: Sequence[str] = ("1s", "10s")):
        self.windows = tuple(windows)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        b = g.book
        n = len(g)
        tr = day.valid("trades", ["recv_ts", "price", "qty", "is_buyer_maker"])
        t_ts = tr["recv_ts"].to_numpy()
        # each trade belongs to the step from the grid row before it to the one at or after
        step = np.searchsorted(g.ts, t_ts, side="left")
        decrease = np.zeros(n, dtype=np.int64)
        explained = np.zeros(n, dtype=np.int64)
        same = np.zeros(n, dtype=bool)
        same[1:] = g.segment[1:] == g.segment[:-1]
        seller = tr["is_buyer_maker"].to_numpy().astype(bool)
        t_px, t_qty = tr["price"].to_numpy(), tr["qty"].to_numpy()
        for bid, px, qty, hit in (
            (True, b.bid_px, b.bid_qty, seller),  # sellers hit bids
            (False, b.ask_px, b.ask_qty, ~seller),  # buyers lift asks
        ):
            d, x = _side(px, qty, same, step[hit], t_px[hit], t_qty[hit], bid=bid)
            decrease += d
            explained += x
        cols = {}
        for w in self.windows:
            ns = duration_ns(w)
            dec = window_sum(g.ts, decrease, g, ns)
            exp = window_sum(g.ts, explained, g, ns)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"cancel_share_{w}"] = (dec - exp) / dec
        return g.frame(cols)


def _side(px, qty, same, t_step, t_px, t_qty, bid: bool, chunk: int = 50_000):
    """Per grid row (the step from the row before), the decrease of the previous row's
    levels and the part of it that trades at those prices explain."""
    n = len(px)
    dec = np.zeros(n, dtype=np.int64)
    exp = np.zeros(n, dtype=np.int64)
    # traded quantity per step and per level of the previous row
    traded = np.zeros((n, px.shape[1]), dtype=np.int64)
    ok = (t_step >= 1) & (t_step < n)
    s, p, q = t_step[ok], t_px[ok], t_qty[ok]
    prev_px = px[s - 1]
    match = (prev_px == p[:, None]) & (qty[s - 1] > 0)
    has = match.any(axis=1)
    lvl = match.argmax(axis=1)
    np.add.at(traded, (s[has], lvl[has]), q[has])
    for a in range(1, n, chunk):
        z = slice(a, min(a + chunk, n))
        zp = slice(a - 1, min(a + chunk, n) - 1)
        p0, q0 = px[zp], qty[zp]
        p1, q1 = px[z], qty[z]
        eq = (p0[:, :, None] == p1[:, None, :]) & (q1[:, None, :] > 0)
        found = eq.any(axis=2)
        q_now = np.where(found, (eq * q1[:, None, :]).sum(axis=2), 0)
        # a level gone from the new row: removed if it lies within the new row's range,
        # unknown if it may only have been pushed below the 20th level
        count = (q1 > 0).sum(axis=1)
        full = count == px.shape[1]
        deepest = p1[np.arange(len(p1)), np.maximum(count - 1, 0)]
        within = (p0 >= deepest[:, None]) if bid else (p0 <= deepest[:, None])
        known = (q0 > 0) & (found | ~full[:, None] | within)
        d = np.where(known, np.maximum(q0 - q_now, 0), 0)
        x = np.minimum(d, traded[z])
        keep = same[z]
        dec[z] = np.where(keep, d.sum(axis=1), 0)
        exp[z] = np.where(keep, x.sum(axis=1), 0)
    return dec, exp
