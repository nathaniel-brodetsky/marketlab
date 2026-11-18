"""Layer 3: what it would have cost to trade. A queue-position fill simulator
(`sim`), probe orders to measure it with, and markouts: how the mid moved after each fill.

A markout is the mid's change from the fill price after a horizon, signed so that it is
positive when the fill was good for us: for a buy, (mid(t + h) - price) / price. A passive
fill starts with half the spread in its favour; a negative markout means the price moved
against it by more than that, which is the cost of adverse selection. It is not computed
when t + h falls outside the fill's segment or after the end of the data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..features.base import duration_ns
from ..io.day import Day
from ..io.intervals import segment_ids
from .market import ASK, BID, Market
from .sim import Mode, Order, Outcome, Result, frames, simulate

HORIZONS = ("100ms", "1s", "10s")

__all__ = [
    "ASK",
    "BID",
    "HORIZONS",
    "Market",
    "Mode",
    "Order",
    "Outcome",
    "Result",
    "frames",
    "markouts",
    "probe_orders",
    "run",
    "simulate",
    "summary",
]


def probe_orders(
    m: Market,
    every: str,
    qty: int,
    cancel_after: str | None,
    passive: bool = True,
    sides: Sequence[int] = (BID, ASK),
) -> list[Order]:
    """One order per side at every multiple of `every` inside a segment: passive ones at
    the best price on their own side, as seen at the decision."""
    step = duration_ns(every)
    wait = duration_ns(cancel_after) if cancel_after else None
    t = m.ticker
    out: list[Order] = []
    for lo, hi in m.day.intervals:
        first = -(-lo // step) * step
        last = min(hi - 1, m.end)
        for when in range(first, last + 1, step):
            i = int(np.searchsorted(t["recv_ts"], when, side="right")) - 1
            if i < 0 or t["recv_ts"][i] < lo:
                continue  # no best price seen in this segment yet
            for side in sides:
                price = int(t["bid_px"][i] if side == BID else t["ask_px"][i])
                out.append(
                    Order(
                        id=len(out),
                        side=side,
                        qty=qty,
                        decided=when,
                        passive=passive,
                        price=price if passive else None,
                        cancel=when + wait if (passive and wait is not None) else None,
                    )
                )
    return out


def markouts(m: Market, fills: pl.DataFrame, horizons: Sequence[str] = HORIZONS) -> pl.DataFrame:
    """The fills with a `markout_<h>` column per horizon, in basis points."""
    ts = fills["ts"].to_numpy()
    side = fills["side"].to_numpy().astype(np.float64)
    price = fills["price"].to_numpy().astype(np.float64)
    seg = segment_ids(ts, m.day.intervals)
    cols = []
    for h in horizons:
        mid = m.mid_at(ts + duration_ns(h), seg)
        with np.errstate(invalid="ignore"):
            mk = side * (mid - price) / price * 1e4
        cols.append(pl.Series(f"markout_{h}", mk, nan_to_null=True))
    return fills.with_columns(cols)


def run(
    day: Day,
    every: str = "10s",
    qty: int = 100,
    cancel_after: str | None = "10s",
    latency: str = "300ms",
    modes: Sequence[Mode] = tuple(Mode),
    aggressive: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Probe orders on a day in every mode: (orders, fills with markouts)."""
    m = Market.of(day)
    lat = duration_ns(latency)
    order_frames, fill_frames = [], []
    passive = probe_orders(m, every, qty, cancel_after)
    for mode in modes:
        o, f = frames(simulate(m, passive, lat, mode), mode)
        order_frames.append(o)
        fill_frames.append(f)
    if aggressive:
        taker = probe_orders(m, every, qty, None, passive=False)
        o, f = frames(simulate(m, taker, lat), "aggressive")
        order_frames.append(o)
        fill_frames.append(f)
    orders = pl.concat(order_frames)
    fills = markouts(m, pl.concat(fill_frames))
    return orders, fills


def summary(
    orders: pl.DataFrame, fills: pl.DataFrame, horizons: Sequence[str] = HORIZONS
) -> pl.DataFrame:
    """Per mode: what became of the orders, how much filled, how long the wait was, and the
    quantity-weighted mean markouts of the fills."""
    decided = [str(o) for o in (Outcome.FILLED, Outcome.PARTIAL, Outcome.CANCELLED)]
    out = orders.group_by("mode", maintain_order=True).agg(
        orders=pl.len(),
        **{str(o): (pl.col("outcome") == str(o)).sum() for o in Outcome},
        fill_rate=(
            pl.col("filled").filter(pl.col("outcome").is_in(decided)).sum()
            / pl.col("qty").filter(pl.col("outcome").is_in(decided)).sum()
        ),
        wait_p50_s=((pl.col("first_fill") - pl.col("arrival")).median() / 1e9),
    )
    w = pl.col("qty").cast(pl.Float64)
    marks = fills.group_by("mode", maintain_order=True).agg(
        fills=pl.len(),
        **{
            f"markout_{h}": (pl.col(f"markout_{h}") * w).sum()
            / w.filter(pl.col(f"markout_{h}").is_not_null()).sum()
            for h in horizons
        },
    )
    return out.join(marks, on="mode", how="left", maintain_order="left")
