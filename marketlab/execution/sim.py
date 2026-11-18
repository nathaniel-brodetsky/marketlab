"""Fills for orders that were never sent, from the recorded market.

Every time is the recorder's receive time. `latency` is how long after a decision the order
takes effect, on that clock: it reaches the exchange half a round trip later, and what the
exchange does then is received one feed lag after that, so latency = RTT / 2 + lag. A
cancel takes the same time, so an order can fill while its cancel is on the way.

**Aggressive orders** take liquidity from the book seen when they arrive, level by level,
as far as the known levels reach.

**Passive orders** are post-only: one that would cross the book when it arrives is
rejected. Otherwise it joins the end of its price level's queue:

1. Ahead of it is the level's quantity as seen when it arrives, less the trades at that
   price already received that the book does not show yet.
2. A trade at its price that hits its side first uses up the quantity ahead, then fills the
   order. A trade at a worse price means the level was taken out: the order fills whole.
3. When a depth update shows the level smaller than the trades since the last update
   explain, the rest was cancelled, and the data cannot say whether ahead of the order or
   behind it. Three modes: `pessimistic` (all behind), `proportional` (the share ahead is
   the share of the level ahead of the order), `optimistic` (all ahead). Whatever the
   mode, no more can be ahead than the level holds. Conclusions are drawn from the
   pessimistic mode; the others show how much the answer depends on it.
4. Quantity added to a level joins behind the order.

Trades are matched to depth updates by exchange time (a trade belongs to the first depth
update at or after its trade time), so that one arriving after the update it is part of is
neither counted twice nor taken for a cancel.

Our orders change nothing in the recorded market: no other trader reacts to them, and the
liquidity an aggressive order takes is still there for the next one. Without our own
fills this cannot be calibrated, so it is conservative by construction: the pessimistic
mode, fills only from trades, and nothing outside the recorded levels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import polars as pl

from .market import ASK, BID, Market, book_at


class Mode(StrEnum):
    PESSIMISTIC = "pessimistic"
    PROPORTIONAL = "proportional"
    OPTIMISTIC = "optimistic"


class Outcome(StrEnum):
    FILLED = "filled"  # whole quantity filled
    PARTIAL = "partial"  # part filled, the rest cancelled
    CANCELLED = "cancelled"  # nothing filled before the cancel took effect
    REJECTED = "rejected"  # post-only and would have crossed
    UNKNOWN = "unknown"  # the queue at its price could not be known
    HOLE = "hole"  # its segment ended while it rested: a gap, a lost connection, a stop
    OPEN = "open"  # still resting when the data ends


@dataclass(frozen=True)
class Order:
    id: int
    side: int  # BID (buy) or ASK (sell)
    qty: int  # quantity steps
    decided: int  # receive time of the decision
    passive: bool = True
    price: int | None = None  # price steps; required for passive orders
    cancel: int | None = None  # receive time of the decision to cancel


@dataclass
class Result:
    order: Order
    outcome: Outcome
    arrival: int
    #: The receive time from which the outcome no longer changes.
    final: int
    fills: list[tuple[int, int, int]] = field(default_factory=list)  # (time, price, qty)
    queue_at_arrival: int | None = None


def simulate(
    m: Market, orders: Sequence[Order], latency: int, mode: Mode = Mode.PESSIMISTIC
) -> list[Result]:
    return [(_passive if o.passive else _aggressive)(m, o, latency, Mode(mode)) for o in orders]


def _aggressive(m: Market, o: Order, latency: int, mode: Mode) -> Result:
    a = o.decided + latency
    seg = m.segment(o.decided)
    if seg < 0 or a > m.segment_end(seg):
        return Result(o, Outcome.HOLE if a <= m.end else Outcome.OPEN, a, min(a, m.end))
    book = book_at(m, a)
    if book is None:
        return Result(o, Outcome.UNKNOWN, a, a)
    taken = book.walk(-o.side, o.qty)  # a buy takes from the asks
    fills = [(a, p, q) for p, q in taken]
    filled = sum(q for _, q in taken)
    if filled == 0:
        return Result(o, Outcome.UNKNOWN, a, a)
    outcome = Outcome.FILLED if filled == o.qty else Outcome.PARTIAL
    return Result(o, outcome, a, a, fills)


def _passive(m: Market, o: Order, latency: int, mode: Mode) -> Result:
    if o.price is None:
        raise ValueError(f"order {o.id}: a passive order needs a price")
    a = o.decided + latency
    seg = m.segment(o.decided)
    if seg < 0:
        return Result(o, Outcome.HOLE, a, o.decided)
    seg_end = m.segment_end(seg)
    if a > seg_end:
        if seg_end < m.end:
            return Result(o, Outcome.HOLE, a, m.segment_close(seg))
        return Result(o, Outcome.OPEN, a, m.end)
    book = book_at(m, a)
    if book is None:
        return Result(o, Outcome.UNKNOWN, a, a)
    other = book.best(-o.side)
    if other is not None and (o.price >= other[0] if o.side == BID else o.price <= other[0]):
        return Result(o, Outcome.REJECTED, a, a)
    level = book.qty(o.side, o.price)
    e0 = m.exchange_time(a)
    if level is None or e0 is None:
        return Result(o, Outcome.UNKNOWN, a, a)

    tr = m.trades
    # trades that hit this side: sellers hit bids, buyers lift asks
    hits_side = tr["is_buyer_maker"] if o.side == BID else ~tr["is_buyer_maker"]
    at_price = hits_side & (tr["price"] == o.price)
    through = hits_side & (tr["price"] < o.price if o.side == BID else tr["price"] > o.price)

    # received by the arrival but newer than the book seen then: already gone from the level
    lo = int(np.searchsorted(tr["recv_ts"], a, side="right"))
    seen = slice(0, lo)
    early = at_price[seen] & (tr["trade_ts"][seen] > e0)
    ahead = max(level - int(tr["qty"][seen][early].sum()), 0)
    queue = ahead

    end = seg_end
    if o.cancel is not None:
        end = min(end, o.cancel + latency)
    hi = int(np.searchsorted(tr["recv_ts"], end, side="right"))
    d = m.depth
    dlo = int(np.searchsorted(d["recv_ts"], a, side="right"))
    dhi = int(np.searchsorted(d["recv_ts"], end, side="right"))
    mine = (d["side"][dlo:dhi] == o.side) & (d["price"][dlo:dhi] == o.price)
    updates = np.flatnonzero(mine) + dlo

    # trades at our price by exchange time, for matching them to depth updates
    pt = np.flatnonzero(at_price)
    p_trade_ts = tr["trade_ts"][pt]
    p_recv = tr["recv_ts"][pt]
    p_qty = tr["qty"][pt]

    events = [(int(tr["recv_ts"][k]), 0, k) for k in range(lo, hi) if at_price[k] or through[k]]
    events += [(int(d["recv_ts"][k]), 1, k) for k in updates]
    events.sort()

    remaining = o.qty
    fills: list[tuple[int, int, int]] = []
    level_prev, e_prev = level, e0
    for ts, kind, k in events:
        if kind == 0:  # a trade
            if tr["trade_ts"][k] <= e0:
                continue  # already in the book the order joined
            if through[k]:
                fills.append((ts, o.price, remaining))
                remaining = 0
            else:
                x = int(tr["qty"][k])
                used = min(x, ahead)
                ahead -= used
                take = min(x - used, remaining)
                if take > 0:
                    fills.append((ts, o.price, take))
                    remaining -= take
        else:  # a depth update at our price
            e = int(d["event_ts"][k])
            new = int(d["qty"][k])
            window = (p_trade_ts > e_prev) & (p_trade_ts <= e)
            traded = int(p_qty[window].sum())
            applied = int(p_qty[window & ((p_recv <= ts) | (p_trade_ts <= e0))].sum())
            pending = traded - applied  # part of this update, not received yet
            before = max(level_prev - traded, 0)
            cancelled = max(before - new, 0)
            now_ahead = max(ahead - pending, 0)
            if mode is Mode.OPTIMISTIC:
                now_ahead -= cancelled
            elif mode is Mode.PROPORTIONAL and before > 0:
                now_ahead -= cancelled * now_ahead / before
            now_ahead = min(max(now_ahead, 0), new)
            ahead = round(now_ahead) + pending
            level_prev, e_prev = new, e
        if remaining == 0:
            return Result(o, Outcome.FILLED, a, ts, fills, queue)

    if end < seg_end or (o.cancel is not None and o.cancel + latency <= seg_end):
        outcome = Outcome.PARTIAL if fills else Outcome.CANCELLED
        return Result(o, outcome, a, end, fills, queue)
    if seg_end < m.end:
        return Result(o, Outcome.HOLE, a, m.segment_close(seg), fills, queue)
    return Result(o, Outcome.OPEN, a, m.end, fills, queue)


def frames(results: Sequence[Result], mode: Mode | str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(orders, fills) as tables."""
    orders = pl.DataFrame(
        {
            "id": [r.order.id for r in results],
            "mode": [str(mode)] * len(results),
            "side": pl.Series([r.order.side for r in results], dtype=pl.Int8),
            "passive": [r.order.passive for r in results],
            "price": [r.order.price for r in results],
            "qty": [r.order.qty for r in results],
            "decided": [r.order.decided for r in results],
            "arrival": [r.arrival for r in results],
            "final": [r.final for r in results],
            "outcome": [str(r.outcome) for r in results],
            "queue_at_arrival": [r.queue_at_arrival for r in results],
            "filled": [sum(q for _, _, q in r.fills) for r in results],
            "first_fill": [r.fills[0][0] if r.fills else None for r in results],
        },
        schema_overrides={"price": pl.Int64, "queue_at_arrival": pl.Int64, "first_fill": pl.Int64},
    )
    rows = [
        (r.order.id, r.order.side, r.order.passive, t, p, q) for r in results for t, p, q in r.fills
    ]
    fills = pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "side": pl.Int8,
            "passive": pl.Boolean,
            "ts": pl.Int64,
            "price": pl.Int64,
            "qty": pl.Int64,
        },
        orient="row",
    ).with_columns(pl.lit(str(mode)).alias("mode"))
    return orders, fills


__all__ = ["ASK", "BID", "Mode", "Order", "Outcome", "Result", "frames", "simulate"]
