"""The flow of trades: imbalance of aggressive buying and selling, volume, VPIN."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Day
from .base import Method, NotFitted, duration_ns, grid, window_last, window_sum


def _trades(day: Day) -> dict[str, np.ndarray]:
    t = day.valid("trades", ["recv_ts", "qty", "is_buyer_maker"])
    buy = ~t["is_buyer_maker"].to_numpy().astype(bool)  # the buyer took liquidity
    qty = t["qty"].to_numpy()
    return {
        "ts": t["recv_ts"].to_numpy(),
        "segment": t["segment"].to_numpy(),
        "qty": qty,
        "buy": np.where(buy, qty, 0),
        "sell": np.where(buy, 0, qty),
    }


class TradeFlow(Method):
    """`trade_imb_<w>`: (aggressive buy volume - aggressive sell volume) / volume, null
    without trades; `volume_<w>` in base units. The aggressor's side is exact: Binance
    marks whether the buyer's order was resting."""

    name = "trades"

    def __init__(self, windows: Sequence[str] = ("1s", "10s")):
        self.windows = tuple(windows)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        t = _trades(day)
        unit = float(day.qty_step)
        cols = {}
        for w in self.windows:
            n = duration_ns(w)
            b = window_sum(t["ts"], t["buy"], g, n)
            s = window_sum(t["ts"], t["sell"], g, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"trade_imb_{w}"] = (b - s) / (b + s)
            cols[f"volume_{w}"] = (b + s) * unit
        return g.frame(cols)


class VPIN(Method):
    """`vpin`: volume-synchronized probability of informed trading (Easley, Lopez de Prado,
    O'Hara 2012). Trades fill buckets of equal volume V, a trade split across buckets where
    it crosses one; VPIN is the mean of |buy - sell| / V over the last `window` buckets,
    known when the last of them fills. Buckets start again in each segment.

    `fit` sets V from the training days: their volume per hour of valid data, scaled to a
    day, divided by `buckets_per_day`. The default 1000 buckets a day over 50 gives a value
    about an hour into a segment; the paper's 50 over 50 needs a day of unbroken data, and
    on a quieter day than the training ones gives none at all. The aggressor's side is
    exact here, so no bulk classification is needed.

    No presumption that it predicts anything: Andersen and Bondarenko found its forecasting
    power for volatility comes from its link to volume. It goes through the same tests as
    every other feature.
    """

    name = "vpin"

    def __init__(self, buckets_per_day: int = 1000, window: int = 50):
        self.buckets_per_day = buckets_per_day
        self.window = window
        self.bucket_: int | None = None

    def fit(self, train: Sequence[Day]) -> VPIN:
        volume, seconds = 0, 0.0
        for day in train:
            volume += int(_trades(day)["qty"].sum())
            seconds += sum(e - s for s, e in day.intervals) / 1e9
        if volume == 0 or seconds == 0:
            raise ValueError("no trades in the training days")
        self.bucket_ = max(1, round(volume / seconds * 86_400 / self.buckets_per_day))
        return self

    def buckets(self, day: Day) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(close time, segment, |buy - sell|) of every filled bucket, in order."""
        if self.bucket_ is None:
            raise NotFitted(f"{self.name}: fit on training days first")
        v = self.bucket_
        t = _trades(day)
        closes, segs, imbalance = [], [], []
        for seg in np.unique(t["segment"]):
            rows = t["segment"] == seg
            ts, qty, buy = t["ts"][rows], t["qty"][rows], t["buy"][rows]
            ctot = np.concatenate([[0], np.cumsum(qty)])
            cbuy = np.concatenate([[0], np.cumsum(buy)])
            n = int(ctot[-1] // v)
            if n == 0:
                continue
            edges = np.arange(1, n + 1, dtype=np.int64) * v
            # the trade that carries the volume up to each edge: ctot[k] < edge <= ctot[k+1]
            k = np.searchsorted(ctot, edges, side="left") - 1
            is_buy = buy[k] > 0
            buy_at = cbuy[k] + np.where(is_buy, edges - ctot[k], 0)
            per_bucket_buy = np.diff(np.concatenate([[0], buy_at]))
            closes.append(ts[k])
            segs.append(np.full(n, seg, dtype=np.int32))
            imbalance.append(np.abs(2 * per_bucket_buy - v))
        if not closes:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.astype(np.int32), empty
        return np.concatenate(closes), np.concatenate(segs), np.concatenate(imbalance)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        close, seg, imb = self.buckets(day)
        vpin = np.full(len(close), np.nan)
        for s in np.unique(seg):
            rows = np.flatnonzero(seg == s)
            c = np.concatenate([[0], np.cumsum(imb[rows])])
            n = len(rows)
            if n >= self.window:
                vals = (c[self.window :] - c[: n - self.window + 1]) / (self.window * self.bucket_)
                vpin[rows[self.window - 1 :]] = vals
        return g.frame({"vpin": window_last(close, vpin, g, seg)})
