"""Stoikov's micro-price: the expected mid in the long run given the imbalance and the
spread now (S. Stoikov, "The micro-price: a high-frequency estimator of future prices",
Quantitative Finance, 2018).

The state is x = (imbalance bucket, spread in ticks), with imbalance
I = bid_qty / (bid_qty + ask_qty) cut into equal-width buckets, so that the symmetry
I -> 1 - I maps buckets onto buckets. From consecutive grid rows the fit counts:

- Q: moves from x to x' with the mid unchanged;
- R1: mid changes K = (-1, -1/2, +1/2, +1) tick from x;
- R2: from x to x' with the mid changed.

The data are symmetrized (every observation is added mirrored: I -> 1 - I, mid change
negated), then G1 = (1 - Q)^-1 R1 K is the expected change at the next move of the mid,
B = (1 - Q)^-1 R2 the state after it, and the adjustment g* = sum over i of B^i G1. The
micro-price is mid + g*(x).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..io.day import Day
from .base import Grid, Method, NotFitted, grid, mid_ticks

K = np.array([-1.0, -0.5, 0.5, 1.0])  # mid changes in ticks


def states(g: Grid, n_imb: int, n_spread: int) -> np.ndarray:
    """State index per grid row, -1 outside the model (a spread wider than `n_spread`)."""
    b = g.book
    bq, aq = b.bid_qty[:, 0].astype(float), b.ask_qty[:, 0].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = bq / (bq + aq)
    bucket = np.clip(np.floor(np.nan_to_num(imb) * n_imb), 0, n_imb - 1).astype(np.int64)
    spread = b.ask_px[:, 0] - b.bid_px[:, 0]
    ok = (spread >= 1) & (spread <= n_spread) & np.isfinite(imb)
    return np.where(ok, (spread - 1) * n_imb + bucket, -1).astype(np.int64)


def adjustment(
    x: np.ndarray,
    x_next: np.ndarray,
    dm: np.ndarray,
    n_imb: int,
    n_spread: int,
    iterations: int = 100,
) -> np.ndarray:
    """g* per state from observed transitions (state, next state, mid change in ticks).

    Transitions with a mid change other than 0, +-1/2 or +-1 tick are dropped, as in the
    paper. States never seen get no adjustment.
    """
    n = n_imb * n_spread
    halves = np.round(dm * 2).astype(np.int64)
    keep = (x >= 0) & (x_next >= 0) & (np.abs(halves) <= 2)
    x, x_next, halves = x[keep], x_next[keep], halves[keep]

    # symmetrize: imbalance bucket b -> n_imb - 1 - b within the same spread, mid change negated
    def mirror(s: np.ndarray) -> np.ndarray:
        return (s // n_imb) * n_imb + (n_imb - 1 - s % n_imb)

    x = np.concatenate([x, mirror(x)])
    x_next = np.concatenate([x_next, mirror(x_next)])
    halves = np.concatenate([halves, -halves])
    q = np.zeros((n, n))
    r1 = np.zeros((n, 4))
    r2 = np.zeros((n, n))
    still = halves == 0
    np.add.at(q, (x[still], x_next[still]), 1)
    moved = ~still
    k_index = {-2: 0, -1: 1, 1: 2, 2: 3}
    ki = np.array([k_index[int(h)] for h in halves[moved]], dtype=np.int64)
    np.add.at(r1, (x[moved], ki), 1)
    np.add.at(r2, (x[moved], x_next[moved]), 1)
    total = q.sum(axis=1) + r1.sum(axis=1)
    seen = total > 0
    q[seen] /= total[seen, None]
    r1[seen] /= total[seen, None]
    r2[seen] /= total[seen, None]
    inv = np.linalg.pinv(np.eye(n) - q)
    g1 = inv @ r1 @ K
    b = inv @ r2
    g = g1.copy()
    term = g1
    for _ in range(iterations):
        term = b @ term
        g = g + term
        if np.abs(term).max() < 1e-9:
            break
    g[~seen] = 0.0
    return g


class MicroPrice(Method):
    """`microprice`: Stoikov's micro-price in quote currency, and `microprice_dev_bps`: its
    distance from the mid. The model is fitted on the training days' consecutive grid rows
    (`steps` rows apart, about 100 ms each) and only looks at the state now."""

    name = "microprice"

    def __init__(self, n_imb: int = 10, n_spread: int = 2, steps: int = 1):
        self.n_imb = n_imb
        self.n_spread = n_spread
        self.steps = steps
        self.adjustment_: np.ndarray | None = None

    def fit(self, train: Sequence[Day]) -> MicroPrice:
        xs, xn, dms = [], [], []
        for day in train:
            g = grid(day)
            if len(g) <= self.steps:
                continue
            s = states(g, self.n_imb, self.n_spread)
            mid = mid_ticks(g.book)
            k = self.steps
            # segments are contiguous: equal at both ends means no hole in between
            same = g.segment[k:] == g.segment[:-k]
            xs.append(s[:-k][same])
            xn.append(s[k:][same])
            dms.append((mid[k:] - mid[:-k])[same])
        if not xs:
            raise ValueError("no training rows")
        self.adjustment_ = adjustment(
            np.concatenate(xs), np.concatenate(xn), np.concatenate(dms), self.n_imb, self.n_spread
        )
        return self

    def transform(self, day: Day) -> pl.DataFrame:
        if self.adjustment_ is None:
            raise NotFitted(f"{self.name}: fit on training days first")
        g = grid(day)
        s = states(g, self.n_imb, self.n_spread)
        mid = mid_ticks(g.book)
        adj = np.where(s >= 0, self.adjustment_[np.clip(s, 0, None)], np.nan)
        tick = float(day.price_step)
        return g.frame(
            {
                "microprice": (mid + adj) * tick,
                "microprice_dev_bps": adj / mid * 1e4,
            }
        )
