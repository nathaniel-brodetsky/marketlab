"""The market as the simulator reads it: the valid rows of the tables it needs, as arrays,
and the book at any receive time rebuilt from the last `book_l20` row and the depth
updates after it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.day import Day
from ..io.intervals import segment_ids

BID, ASK = 1, -1


def _sorted(cols: dict[str, np.ndarray], key: str) -> dict[str, np.ndarray]:
    order = np.argsort(cols[key], kind="stable")
    if (order == np.arange(len(order))).all():
        return cols
    return {k: v[order] for k, v in cols.items()}


@dataclass
class Market:
    day: Day
    l20: object  # Book20
    depth: dict[str, np.ndarray]
    trades: dict[str, np.ndarray]
    ticker: dict[str, np.ndarray]
    #: The last receive time recorded: nothing is known after it.
    end: int

    @classmethod
    def of(cls, day: Day) -> Market:
        d = day.valid(
            "depth_diff",
            ["recv_ts", "event_ts", "final_update_id", "side", "price", "qty"],
        )
        t = day.valid("trades", ["recv_ts", "trade_ts", "price", "qty", "is_buyer_maker"])
        b = day.valid("book_ticker", ["recv_ts", "bid_px", "bid_qty", "ask_px", "ask_qty"])
        depth = _sorted({c: d[c].to_numpy() for c in d.columns}, "recv_ts")
        trades = _sorted({c: t[c].to_numpy() for c in t.columns}, "recv_ts")
        trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
        ticker = _sorted({c: b[c].to_numpy() for c in b.columns}, "recv_ts")
        # the end of the recording, valid or not: a segment that ends before it ends in a hole
        ends = [max((r.last_recv_ts for r in day.runs), default=0)]
        if day.until is not None:
            ends.append(day.until)
        return cls(day, day.book20(), depth, trades, ticker, min(ends))

    def segment(self, ts: int) -> int:
        return int(segment_ids(np.array([ts]), self.day.intervals)[0])

    def segment_end(self, seg: int) -> int:
        """The last receive time of a segment that the data shows: its end, or the end of
        the data if that comes first."""
        return min(self.day.intervals[seg][1] - 1, self.end)

    def segment_close(self, seg: int) -> int:
        """When the data shows that a segment has ended: the time of the event that ended
        it."""
        return self.day.intervals[seg][1]

    def exchange_time(self, ts: int) -> int | None:
        """The exchange time of the last depth update received at or before `ts`: the
        moment at the exchange that the book seen at `ts` shows."""
        i = int(np.searchsorted(self.depth["recv_ts"], ts, side="right")) - 1
        return int(self.depth["event_ts"][i]) if i >= 0 else None

    def mid_at(self, ts: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """Mid in price steps from the last best bid and ask at or before each time, both in
        segment `seg`; NaN where the time lies outside it (after a hole, or after the end
        of the data) or no best price was seen in it yet."""
        r = self.ticker["recv_ts"]
        out = np.full(len(ts), np.nan)
        if not len(r) or not len(ts) or not self.day.intervals:
            return out
        starts, ends = self.day.segment_bounds()
        i = np.searchsorted(r, ts, side="right") - 1
        j = np.clip(i, 0, None)
        s = np.clip(seg, 0, None)
        ok = (seg >= 0) & (i >= 0) & (ts < ends[s]) & (r[j] >= starts[s])
        mid = (self.ticker["bid_px"] + self.ticker["ask_px"]) / 2.0
        out[ok] = mid[j[ok]]
        return out


@dataclass
class Book:
    """The book at one receive time: levels by price for each side, and how far down each
    side is known (the 20th level of the snapshot it was rebuilt from)."""

    bids: dict[int, int]
    asks: dict[int, int]
    deepest_bid: int | None  # levels at or above it are known; None: the whole side is known
    deepest_ask: int | None

    def best(self, side: int) -> tuple[int, int] | None:
        levels = self.bids if side == BID else self.asks
        if not levels:
            return None
        p = max(levels) if side == BID else min(levels)
        return p, levels[p]

    def qty(self, side: int, price: int) -> int | None:
        """Quantity resting at a price, 0 for a known empty price, None if the price lies
        beyond what is known."""
        levels, deepest = (
            (self.bids, self.deepest_bid) if side == BID else (self.asks, self.deepest_ask)
        )
        if price in levels:
            return levels[price]
        if deepest is None:
            return 0
        known = price >= deepest if side == BID else price <= deepest
        return 0 if known else None

    def walk(self, side: int, qty: int) -> list[tuple[int, int]]:
        """Take `qty` from the side's known levels, best first: (price, quantity) per level;
        less than `qty` in all if the known levels hold less."""
        levels = self.bids if side == BID else self.asks
        deepest = self.deepest_bid if side == BID else self.deepest_ask
        prices = sorted(levels, reverse=side == BID)
        out = []
        for p in prices:
            if deepest is not None and (p < deepest if side == BID else p > deepest):
                break
            take = min(qty, levels[p])
            if take > 0:
                out.append((p, take))
                qty -= take
            if qty == 0:
                break
        return out


def book_at(m: Market, ts: int) -> Book | None:
    """The book seen at receive time `ts`: the last `book_l20` row at or before it in the
    same segment, with the depth updates received after that row applied. None if there is
    no such row."""
    seg = m.segment(ts)
    if seg < 0:
        return None
    b = m.l20
    i = int(np.searchsorted(b.ts, ts, side="right")) - 1
    if i < 0 or b.segment[i] != seg:
        return None
    bids = {int(p): int(q) for p, q in zip(b.bid_px[i], b.bid_qty[i], strict=True) if q > 0}
    asks = {int(p): int(q) for p, q in zip(b.ask_px[i], b.ask_qty[i], strict=True) if q > 0}
    deepest_bid = min(bids) if len(bids) == 20 else None
    deepest_ask = max(asks) if len(asks) == 20 else None
    d = m.depth
    lo = int(np.searchsorted(d["recv_ts"], b.ts[i], side="right"))
    hi = int(np.searchsorted(d["recv_ts"], ts, side="right"))
    sample_id = b.update_id[i]
    for k in range(lo, hi):
        if d["final_update_id"][k] <= sample_id or d["side"][k] == 0:
            continue
        levels = bids if d["side"][k] == BID else asks
        p, q = int(d["price"][k]), int(d["qty"][k])
        if q == 0:
            levels.pop(p, None)
        else:
            levels[p] = q
    return Book(bids, asks, deepest_bid, deepest_ask)
