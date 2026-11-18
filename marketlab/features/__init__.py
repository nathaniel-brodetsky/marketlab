"""Layer 1: microstructure features on the grid of valid `book_l20` rows."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..io.day import Day
from .activity import CancelShare, Intensity
from .base import Grid, Method, NotFitted, duration_ns, grid
from .book import BookShape
from .microprice import MicroPrice
from .ofi import MultiLevelOFI, OrderFlowImbalance
from .trades import VPIN, TradeFlow
from .vol import Volatility

__all__ = [
    "VPIN",
    "BookShape",
    "CancelShare",
    "Grid",
    "Intensity",
    "Method",
    "MicroPrice",
    "MultiLevelOFI",
    "NotFitted",
    "OrderFlowImbalance",
    "TradeFlow",
    "Volatility",
    "compute",
    "duration_ns",
    "grid",
]


def compute(day: Day, methods: Sequence[Method]) -> pl.DataFrame:
    """Every method's columns side by side on the day's grid."""
    frames = [m.transform(day) for m in methods]
    if not frames:
        return grid(day).frame({})
    ts = frames[0]["ts"]
    for m, f in zip(methods, frames, strict=True):
        if not f["ts"].equals(ts):
            raise RuntimeError(f"{m!r} reported on another grid")
    return pl.concat([frames[0], *(f.drop("ts") for f in frames[1:])], how="horizontal")
