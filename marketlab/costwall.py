"""The cost wall: how strong a signal must be to pay for a round trip, horizon by horizon.

A rule that trades when its signal is in the top or bottom `q` of its range and holds for
a horizon `h` earns, before costs, about `rho * sigma_h * phi(z_q) / q` a trade, where
`rho` is the signal's correlation with the return over `h`, `sigma_h` the standard
deviation of that return, and `z_q` the normal quantile that leaves `q` above it (the mean
of a standard normal beyond `z_q` is `phi(z_q) / q`). Setting that equal to the round-trip
cost `c` gives the correlation the signal needs:

    rho_needed(h, c) = c / (sigma_h * phi(z_q) / q)

Above 1 no signal can pay the cost at that horizon, however good. The returns and the
signal are taken to be jointly normal and the rank IC to stand in for `rho`: a rough wall,
for deciding where to look, not a verdict on any signal.

`sigma_h` is measured on the data: from the trade archives (the last trade price on a grid
of step `h` within each UTC day) for horizons under an hour, from hourly klines (closes at
multiples of `h`) for an hour and over.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from . import archive
from .carry import read_klines
from .features import duration_ns as _short

NS_DAY = 86_400 * 1_000_000_000
HOUR = 3_600 * 1_000_000_000
HORIZONS = ("1s", "5s", "30s", "1m", "5m", "15m", "1h", "4h", "1d", "1w")


def duration_ns(text: str) -> int:
    """A horizon: `1d` and `1w` too, beside the durations features take (`1s`, `5m`, `1h`)."""
    t = text.strip()
    if t[:-1].isdigit() and t[-1] in "dw":
        return int(t[:-1]) * NS_DAY * (7 if t[-1] == "w" else 1)
    return _short(t)


def kappa(q: float) -> float:
    """The mean of a standard normal beyond the point that leaves `q` above it."""
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")
    z = stats.norm.isf(q)
    return float(stats.norm.pdf(z) / q)


def needed(cost_bp: float, sigma_bp: float, q: float) -> float:
    """The correlation a signal needs at a horizon of standard deviation `sigma_bp` to earn
    `cost_bp` a round trip trading its top and bottom `q`."""
    return cost_bp / (sigma_bp * kappa(q)) if sigma_bp > 0 else float("inf")


def grid_returns(ts: np.ndarray, price: np.ndarray, start: int, step: int) -> np.ndarray:
    """Log returns, in bp, of the last trade price on a grid of `step` from `start`, inside
    one day: the grid begins at the first point after the first trade and ends at the last
    point at or before the last trade."""
    if len(ts) < 2:
        return np.zeros(0)
    first = start + -(-(int(ts[0]) - start) // step) * step
    grid = np.arange(first, int(ts[-1]) + 1, step, dtype=np.int64)
    if len(grid) < 2:
        return np.zeros(0)
    p = np.log(price[np.searchsorted(ts, grid, side="right") - 1])
    return np.diff(p) * 1e4


def bar_returns(open_time: np.ndarray, close: np.ndarray, step: int) -> np.ndarray:
    """Log returns, in bp, between hourly closes `step` apart: bars whose open is a multiple
    of `step` (from 1970, UTC), next to one another; a gap in the bars drops its return."""
    if not len(open_time):
        return np.zeros(0)
    end = open_time + HOUR  # a close is the price at its bar's end
    keep = end % step == 0
    t, p = end[keep], np.log(close[keep])
    ok = np.diff(t) == step
    return (np.diff(p) * 1e4)[ok]


@dataclass(frozen=True)
class Row:
    horizon: str
    source: str
    n: int
    sigma_bp: float
    mean_abs_bp: float

    def needed(self, cost_bp: float, q: float) -> float:
        return needed(cost_bp, self.sigma_bp, q)


def _row(h: str, source: str, r: np.ndarray) -> Row:
    n = len(r)
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    return Row(h, source, n, sd, float(np.abs(r).mean()) if n else float("nan"))


def from_trades(days: Sequence[archive.ArchiveDay], horizons: Sequence[str]) -> list[Row]:
    """Each horizon under an hour, over the days given."""
    short = [h for h in horizons if duration_ns(h) < HOUR]
    parts: dict[str, list[np.ndarray]] = {h: [] for h in short}
    for day in days:
        trades, _ = archive.read(day)
        order = np.argsort(trades["ts"].to_numpy(), kind="stable")
        ts = trades["ts"].to_numpy()[order]
        price = trades["price"].to_numpy()[order].astype(float)
        start = int(dt.datetime.fromisoformat(day.date).replace(tzinfo=dt.UTC).timestamp()) * 10**9
        inside = (ts >= start) & (ts < start + NS_DAY)
        for h in short:
            parts[h].append(grid_returns(ts[inside], price[inside], start, duration_ns(h)))
    label = f"trades, {len(days)} days"
    return [_row(h, label, np.concatenate(parts[h]) if parts[h] else np.zeros(0)) for h in short]


def from_klines(
    root: str | Path, symbol: str, market: str, horizons: Sequence[str], since: int | None
) -> list[Row]:
    """Each horizon of an hour and over, from the hourly klines at or after `since` (ns)."""
    k = read_klines(root, market, symbol, "1h")
    ot, close = k["open_time"].to_numpy(), k["close"].to_numpy()
    if since is not None:
        keep = ot >= since
        ot, close = ot[keep], close[keep]
    out = []
    for h in horizons:
        step = duration_ns(h)
        if step < HOUR:
            continue
        if step % HOUR:
            raise ValueError(f"{h}: horizons of an hour and over must be whole hours")
        days = (ot[-1] - ot[0]) // NS_DAY if len(ot) else 0
        out.append(_row(h, f"{market} 1h klines, {days} days", bar_returns(ot, close, step)))
    return out


def wall(
    root: str | Path,
    symbol: str = "BTCUSDT",
    horizons: Sequence[str] = HORIZONS,
    last_days: int | None = None,
    years: float = 1.0,
    market: str = "spot",
) -> list[Row]:
    """The standard deviation of the return at every horizon: trades for the short ones (the
    last `last_days` days in the archive, or all), klines for the long ones (the last
    `years` of them)."""
    days = archive.find(root, symbol, "aggTrades", "spot")
    if last_days is not None:
        days = days[-last_days:]
    rows = from_trades(days, horizons) if days else []
    k = read_klines(root, market, symbol, "1h")
    if len(k):
        since = int(k["open_time"].to_numpy()[-1] - years * 365 * NS_DAY)
        rows += from_klines(root, symbol, market, horizons, since)
    order = {h: i for i, h in enumerate(horizons)}
    return sorted(rows, key=lambda r: order[r.horizon])
