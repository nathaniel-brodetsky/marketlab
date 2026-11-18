"""The shape of the book at each grid point: spread, mid, weighted mid, imbalance, depth."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Day
from .base import Grid, Method, grid, mid_ticks


def _band_name(bps: float) -> str:
    return f"depth_{bps:g}bp".replace(".", "p")


class BookShape(Method):
    """Columns:

    - `mid`: (best bid + best ask) / 2, in quote currency.
    - `spread_ticks`, `spread_bps`.
    - `wmid_dev_bps`: the mid weighted by the opposite side's quantity at the best level,
      (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty), minus the mid. Positive when
      the bid side is heavier.
    - `imb_1`, `imb_5`, `imb_20`: (bid quantity - ask quantity) / (sum) over the first k
      levels.
    - `depth_<x>bp`: quantity on both sides within x basis points of the mid, in base units;
      null where the 20 recorded levels do not reach that far, because then the true
      depth is unknown.
    - `reach_bid_bps`, `reach_ask_bps`: how far from the mid the deepest recorded level is.
      For BTCUSDT 20 levels cover cents, not basis points; this column says how much of the
      book the other columns see.
    """

    name = "book"

    def __init__(
        self, levels: Sequence[int] = (1, 5, 20), bands_bps: Sequence[float] = (0.01, 0.02)
    ):
        self.levels = tuple(levels)
        self.bands_bps = tuple(bands_bps)

    def transform(self, day: Day) -> pl.DataFrame:
        g = grid(day)
        return g.frame(self.columns(g, day))

    def columns(self, g: Grid, day: Day) -> dict[str, np.ndarray]:
        b = g.book
        tick, unit = float(day.price_step), float(day.qty_step)
        mid = mid_ticks(b)
        bid, ask = b.bid_px[:, 0].astype(float), b.ask_px[:, 0].astype(float)
        bq, aq = b.bid_qty[:, 0].astype(float), b.ask_qty[:, 0].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmid = (bid * aq + ask * bq) / (bq + aq)
            out = {
                "mid": mid * tick,
                "spread_ticks": ask - bid,
                "spread_bps": (ask - bid) / mid * 1e4,
                "wmid_dev_bps": (wmid - mid) / mid * 1e4,
            }
            cb = np.cumsum(b.bid_qty, axis=1, dtype=np.float64)
            ca = np.cumsum(b.ask_qty, axis=1, dtype=np.float64)
            for k in self.levels:
                sb, sa = cb[:, k - 1], ca[:, k - 1]
                out[f"imb_{k}"] = (sb - sa) / (sb + sa)
            # deepest recorded level on each side, and whether the side has more beyond it
            nb = (b.bid_qty > 0).sum(axis=1)
            na = (b.ask_qty > 0).sum(axis=1)
            rows = np.arange(len(g))
            last_bid = b.bid_px[rows, np.maximum(nb - 1, 0)].astype(float)
            last_ask = b.ask_px[rows, np.maximum(na - 1, 0)].astype(float)
            out["reach_bid_bps"] = (mid - last_bid) / mid * 1e4
            out["reach_ask_bps"] = (last_ask - mid) / mid * 1e4
            for x in self.bands_bps:
                lo = mid * (1 - x / 1e4)
                hi = mid * (1 + x / 1e4)
                inb = (b.bid_px >= lo[:, None]) & (b.bid_qty > 0)
                ina = (b.ask_px <= hi[:, None]) & (b.ask_qty > 0)
                depth = (b.bid_qty * inb).sum(axis=1) + (b.ask_qty * ina).sum(axis=1)
                # a full side whose deepest level is still inside the band may hold more
                unknown = ((nb == 20) & (last_bid >= lo)) | ((na == 20) & (last_ask <= hi))
                out[_band_name(x)] = np.where(unknown, np.nan, depth * unit)
        return out
