"""A first, rough look at cross-sectional signals on Binance's USD-M perpetuals, a day at a
time, on the development years only.

Not a hypothesis: nothing here is written down before it is run, and nothing here can read
a day after `DEV_END` (2023-12-31): the later years are kept for the hypothesis written
about whatever this finds, if anything. Its job is to say, cheaply, whether a signal
clears its costs at all.

Each UTC day `d`, at its close:

- the universe is the `top` perpetuals by mean quote volume over the 30 days to `d`, among
  those with a close on `d` and on `d - 30`;
- signals: `mom7`, `mom30` (log change of the close over 7 or 30 days), `rev1` (minus the
  last day's log change), `carry` (minus the mean of the funding rates settled over the last
  7 days: a short in a coin whose longs pay earns that);
- the portfolio, rebalanced every `hold` days: long the perpetuals strictly above the
  signal's `1 - q` quantile in the universe, short those strictly below its `q` quantile
  (so that coins tied at the same value, as funding rates often are, are not ranked by
  their names), equal weights, half the capital a side; held at those
  weights. A day's return of a position is the close-to-close change less the funding
  settled that day (for a long; plus it, for a short); a perpetual with no bar that day
  returns nothing. Each rebalance pays `cost_bp / 2` on every unit of weight traded, so a
  round trip costs `cost_bp`.

Reported: the daily rank IC of each signal against the next `hold` days' return (Newey-West
over `hold` days), and the portfolio's gross, costs, net, Sharpe from months, drawdown.
The universe is the perpetuals Binance still lists: coins delisted and gone from its list
are missing, which flatters anything that buys losers.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from .validation import metrics

DEV_END = "2023-12-31"
SIGNALS = ("mom7", "mom30", "rev1", "carry")


class HeldOut(ValueError):
    pass


@dataclass
class Panel:
    dates: list[str]  # UTC days
    symbols: list[str]
    close: np.ndarray  # days x symbols, NaN without a bar
    qvol: np.ndarray  # quote volume, NaN without a bar
    funding: np.ndarray  # sum of the rates settled that day; NaN without a funding history


def _day(ms: np.ndarray) -> np.ndarray:
    return ms // 86_400_000


def load(root: str | Path, first: str, last: str) -> Panel:
    """Every perpetual's daily bars and funding up to `last`; the days from `first` on,
    with 40 days before it for the signals. `last` after `DEV_END` is refused."""
    if last > DEV_END:
        raise HeldOut(f"{last} is after {DEV_END}: the years after it are held out")
    d0 = (dt.date.fromisoformat(first) - dt.date(1970, 1, 1)).days - 40
    d1 = (dt.date.fromisoformat(last) - dt.date(1970, 1, 1)).days
    days = np.arange(d0, d1 + 1)
    folder = Path(root) / "um" / "klines1d"
    files = sorted(folder.glob("*.csv"))
    symbols = [f.stem for f in files]
    close = np.full((len(days), len(files)), np.nan)
    qvol = np.full_like(close, np.nan)
    funding = np.full_like(close, np.nan)
    for j, f in enumerate(files):
        k = pl.read_csv(f, schema_overrides={"open_time": pl.Int64})
        d = _day(k["open_time"].to_numpy())
        ok = (d >= d0) & (d <= d1)
        close[d[ok] - d0, j] = k["close"].to_numpy().astype(float)[ok]
        qvol[d[ok] - d0, j] = k["quote_volume"].to_numpy().astype(float)[ok]
        fr = Path(root) / "um" / "fundingRate" / f.stem / f"{f.stem}-fundingRate.csv"
        if fr.exists():
            r = pl.read_csv(fr, schema_overrides={"funding_time": pl.Int64,
                                                  "funding_rate": pl.Float64})  # fmt: skip
            fd = _day(r["funding_time"].to_numpy())
            keep = (fd >= d0) & (fd <= d1)
            col = np.zeros(len(days))
            np.add.at(col, fd[keep] - d0, r["funding_rate"].to_numpy()[keep])
            funding[:, j] = col
    dates = [(dt.date(1970, 1, 1) + dt.timedelta(days=int(x))).isoformat() for x in days]
    return Panel(dates, symbols, close, qvol, funding)


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k] if k else a
    return out


def _trailing_mean(a: np.ndarray, k: int) -> np.ndarray:
    """The mean of the last `k` rows, this one included; NaN unless all are there."""
    c = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(np.nan_to_num(a), axis=0)])
    n = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(np.isfinite(a), axis=0)])
    out = np.full_like(a, np.nan)
    out[k - 1 :] = (c[k:] - c[:-k]) / k
    full = np.full_like(a, False, dtype=bool)
    full[k - 1 :] = (n[k:] - n[:-k]) == k
    return np.where(full, out, np.nan)


def signals(p: Panel) -> dict[str, np.ndarray]:
    lc = np.log(p.close)
    return {
        "mom7": lc - _lag(lc, 7),
        "mom30": lc - _lag(lc, 30),
        "rev1": -(lc - _lag(lc, 1)),
        "carry": -_trailing_mean(p.funding, 7),
    }


def daily_returns(p: Panel) -> np.ndarray:
    """A long's return each day: close to close, less the funding settled that day."""
    r = p.close / _lag(p.close, 1) - 1
    return r - np.nan_to_num(p.funding)


def universe(p: Panel, top: int) -> np.ndarray:
    """Days x symbols: in the universe at that day's close."""
    liq = _trailing_mean(p.qvol, 30)
    ok = np.isfinite(p.close) & np.isfinite(_lag(p.close, 30)) & np.isfinite(liq)
    out = np.zeros_like(ok)
    for i in range(len(p.dates)):
        idx = np.flatnonzero(ok[i])
        if len(idx):
            best = idx[np.argsort(-liq[i, idx], kind="stable")[:top]]
            out[i, best] = True
    return out


@dataclass
class Result:
    signal: str
    hold: int
    days: int
    ic: float
    ic_t: float
    gross_pct: float  # a year
    cost_pct: float
    net_pct: float
    sharpe_months: float
    max_drawdown_pct: float
    positive_months: float
    turnover: float  # weight traded at a rebalance, of a gross of 1
    names: float  # mean positions a side


def forward(r: np.ndarray, hold: int) -> np.ndarray:
    """The log return over the next `hold` days after each day; NaN unless every day is there."""
    lr = np.log1p(r)
    out = np.full_like(r, np.nan)
    for i in range(len(r) - hold):
        span = lr[i + 1 : i + 1 + hold]
        out[i] = np.where(np.isfinite(span).all(axis=0), span.sum(axis=0), np.nan)
    return out


def portfolio(x: np.ndarray, inu: np.ndarray, r: np.ndarray, start: int, hold: int, q: float,
              cost_bp: float) -> tuple[np.ndarray, list[float], list[int]]:  # fmt: skip
    """The long-short portfolio's return each day (costs on the day of the rebalance), the
    weight traded at each rebalance, and the positions a side."""
    n_days, n_sym = x.shape
    w = np.zeros(n_sym)
    pnl = np.zeros(n_days)
    turns: list[float] = []
    names: list[int] = []
    for i in range(start, n_days - 1):
        if (i - start) % hold == 0:
            idx = np.flatnonzero(inu[i] & np.isfinite(x[i]))
            new = np.zeros(n_sym)
            if len(idx):
                v = x[i, idx]
                hi, lo = np.quantile(v, 1 - q), np.quantile(v, q)
                longs, shorts = idx[v > hi], idx[v < lo]  # strictly: ties are not ranked
                if len(longs) and len(shorts):
                    new[longs] = 0.5 / len(longs)
                    new[shorts] = -0.5 / len(shorts)
                    names.append((len(longs) + len(shorts)) // 2)
            turns.append(float(np.abs(new - w).sum()))
            pnl[i] -= turns[-1] * cost_bp / 2 / 1e4
            w = new
        pnl[i + 1] += float((w * np.nan_to_num(r[i + 1])).sum())
    return pnl, turns, names


def _sharpe(monthly: np.ndarray) -> float:
    if len(monthly) < 2 or not monthly.std(ddof=1) > 0:
        return float("nan")
    return float(monthly.mean() / monthly.std(ddof=1) * np.sqrt(12))


def evaluate(
    p: Panel,
    name: str,
    x: np.ndarray,
    hold: int,
    top: int,
    q: float,
    cost_bp: float,
    first: str,
    r: np.ndarray | None = None,
    inu: np.ndarray | None = None,
) -> Result:  # fmt: skip
    r = daily_returns(p) if r is None else r
    inu = universe(p, top) if inu is None else inu
    start = p.dates.index(first) if first in p.dates else 0
    fwd = forward(r, hold)
    ics = []
    for i in range(start, len(p.dates) - hold):
        m = inu[i] & np.isfinite(x[i]) & np.isfinite(fwd[i])
        if m.sum() >= 10 and np.ptp(x[i, m]) > 0 and np.ptp(fwd[i, m]) > 0:
            ics.append(stats.spearmanr(x[i, m], fwd[i, m]).statistic)
    ic_days = np.array([v for v in ics if np.isfinite(v)])
    ic = float(ic_days.mean()) if len(ic_days) else float("nan")
    se = metrics.newey_west_se(ic_days, hold) if len(ic_days) > 2 else float("nan")
    pnl, turns, names = portfolio(x, inu, r, start, hold, q, cost_bp)
    days = p.dates[start:]
    net = pnl[start:]
    month = np.array([d[:7] for d in days])
    sums = np.array([net[month == mo].sum() for mo in dict.fromkeys(month)])
    cum = np.cumsum(net)
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max())
    years = len(days) / 365.0
    cost = sum(turns) * cost_bp / 2 / 1e4
    return Result(
        signal=name,
        hold=hold,
        days=len(days),
        ic=ic,
        ic_t=ic / se if se > 0 else float("nan"),
        gross_pct=(net.sum() + cost) / years * 100,
        cost_pct=cost / years * 100,
        net_pct=net.sum() / years * 100,
        sharpe_months=_sharpe(sums),
        max_drawdown_pct=dd * 100,
        positive_months=float((sums > 0).mean()) if len(sums) else float("nan"),
        turnover=float(np.mean(turns)) if turns else float("nan"),
        names=float(np.mean(names)) if names else float("nan"),
    )


def report(
    root: str | Path,
    first: str = "2020-01-01",
    last: str = DEV_END,
    top: int = 50,
    q: float = 0.2,
    cost_bp: float = 9.0,
    holds: tuple[int, ...] = (1, 7),
) -> tuple[Panel, list[Result]]:
    p = load(root, first, last)
    sig = signals(p)
    r, inu = daily_returns(p), universe(p, top)
    out = []
    for name in SIGNALS:
        for h in holds:
            if name == "rev1" and h != 1:
                continue
            out.append(evaluate(p, name, sig[name], h, top, q, cost_bp, first, r, inu))
    return p, out
