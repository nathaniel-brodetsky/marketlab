"""A trend on a few named perpetuals, a day at a time: each coin long or short by the sign of
its move over each of several lookbacks, their mean its direction (with 20, 60 and 120 days:
-1, -1/3, 1/3 or 1), the coins in equal shares of the capital.

Sized `whole`, a coin's position is its direction times its share (gross 1 when all agree);
`coin`, that times `target` over the coin's volatility over the last `window` days (its
daily log returns, annualized), at most 1: each coin held to about that volatility, never on
more than its share. Decided at a daily close from the bars up to it, held over the next
day; a coin without a bar at the close, or without the days a lookback or the volatility
needs, holds nothing (a lookback it lacks counts as no direction).
"""

from __future__ import annotations

import numpy as np

from . import xsection as xs

SIZES = ("whole", "coin")


def volatility(p: xs.Panel, window: int) -> np.ndarray:
    """Days x symbols: the annualized volatility of the daily log returns over the last
    `window` days, that day's included; NaN unless all of them are there."""
    lr = np.log(p.close / xs._lag(p.close, 1))
    return np.sqrt(xs._trailing_mean(lr**2, window) * 365)


def decide(p: xs.Panel, cols: list[int], lookbacks: tuple[int, ...], size: str, target: float,
           vol: np.ndarray, i: int) -> np.ndarray:  # fmt: skip
    """The weights over every symbol for the day after the close of day `i`."""
    if size not in SIZES:
        raise ValueError(f"size is one of {SIZES}, not {size!r}")
    w = np.zeros(len(p.symbols))
    lc = np.log(p.close[: i + 1, cols])
    for c, j in enumerate(cols):
        if not np.isfinite(lc[i, c]):
            continue
        s = 0.0
        for n in lookbacks:
            if i - n >= 0 and np.isfinite(lc[i - n, c]):
                s += np.sign(lc[i, c] - lc[i - n, c])
        x = s / len(lookbacks) / len(cols)
        if size == "coin":
            v = vol[i, j]
            x = x * min(target / v, 1.0) if np.isfinite(v) and v > 0 else 0.0
        w[j] = x
    return w


def holdings(p: xs.Panel, coins: tuple[str, ...], lookbacks: tuple[int, ...], size: str,
             target: float, window: int, start: int) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The weights held over each day, decided at the close before from the close of `start`
    on (as `xsection.holdings`: nothing held over `start` itself), and the weight traded
    each day, at the close it was decided at."""
    missing = [c for c in coins if c not in p.symbols]
    if missing:
        raise ValueError(f"no daily bars for {', '.join(missing)}")
    cols = [p.symbols.index(c) for c in coins]
    vol = volatility(p, window)
    n = len(p.dates)
    held = np.zeros((n, len(p.symbols)))
    traded = np.zeros(n)
    for i in range(start, n - 1):
        held[i + 1] = decide(p, cols, lookbacks, size, target, vol, i)
        traded[i] = np.abs(held[i + 1] - held[i]).sum()
    return held, traded
