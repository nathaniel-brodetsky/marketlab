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
are missing, which flatters anything that buys losers. Seven contracts are not a coin's own
price and are left out (`NOT_COINS`). A perpetual that settled funding on a day with no
daily bar (often an earlier listing whose bars Binance no longer serves) is missing from
the universe on that day; `Panel.unbarred` counts those days.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from .validation import metrics

DEV_END = "2023-12-31"
SIGNALS = ("mom7", "mom30", "rev1", "carry")
# A stablecoin, four indices (two gone from Binance's list, in the archive) and two gold
# tokens (listed after the development years): none of them a coin's own price.
NOT_COINS = frozenset({"USDCUSDT", "BTCDOMUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT",
                       "PAXGUSDT", "XAUTUSDT"})  # fmt: skip
KLINE_TYPES = {"open_time": pl.Int64, "close": pl.Float64, "quote_volume": pl.Float64}
FUNDING_TYPES = {"funding_time": pl.Int64, "funding_rate": pl.Float64}


class HeldOut(ValueError):
    pass


@dataclass
class Panel:
    dates: list[str]  # UTC days
    symbols: list[str]
    close: np.ndarray  # days x symbols, NaN without a bar
    qvol: np.ndarray  # quote volume, NaN without a bar
    funding: np.ndarray  # sum of the rates settled that day; NaN without a funding history
    unbarred: dict[str, int] = field(default_factory=dict)  # days with funding, no bar


def read_columns(path: Path, types: dict[str, pl.DataType]) -> pl.DataFrame:
    """The columns named, with their types: read as text and then cast, so that a column
    whose first rows look whole (a volume of 100, then 1.5) cannot fail the read."""
    raw = pl.read_csv(path, infer_schema=False, columns=list(types))
    return raw.select(pl.col(c).cast(t) for c, t in types.items())


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
    files = sorted(f for f in folder.glob("*.csv") if f.stem not in NOT_COINS)
    symbols = [f.stem for f in files]
    close = np.full((len(days), len(files)), np.nan)
    qvol = np.full_like(close, np.nan)
    funding = np.full_like(close, np.nan)
    unbarred: dict[str, int] = {}
    for j, f in enumerate(files):
        k = read_columns(f, KLINE_TYPES)
        d = _day(k["open_time"].to_numpy())
        ok = (d >= d0) & (d <= d1)
        close[d[ok] - d0, j] = k["close"].to_numpy()[ok]
        qvol[d[ok] - d0, j] = k["quote_volume"].to_numpy()[ok]
        fr = Path(root) / "um" / "fundingRate" / f.stem / f"{f.stem}-fundingRate.csv"
        if fr.exists():
            r = read_columns(fr, FUNDING_TYPES)
            fd = _day(r["funding_time"].to_numpy())
            keep = (fd >= d0) & (fd <= d1)
            col = np.zeros(len(days))
            np.add.at(col, fd[keep] - d0, r["funding_rate"].to_numpy()[keep])
            funding[:, j] = col
            settled = np.zeros(len(days), dtype=bool)
            settled[fd[keep] - d0] = True
            n = int((settled & ~np.isfinite(close[:, j])).sum())
            if n:
                unbarred[f.stem] = n
    dates = [(dt.date(1970, 1, 1) + dt.timedelta(days=int(x))).isoformat() for x in days]
    return Panel(dates, symbols, close, qvol, funding, unbarred)


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


def holdings(x: np.ndarray, inu: np.ndarray, start: int, hold: int,
             q: float) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:  # fmt: skip
    """The weights held over each day (decided at the close of the day before), the weight
    traded each day, the weight traded at each rebalance, and the positions a side."""
    n_days, n_sym = x.shape
    w = np.zeros(n_sym)
    held = np.zeros((n_days, n_sym))
    traded = np.zeros(n_days)
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
            traded[i] = turns[-1]
            w = new
        held[i + 1] = w
    return held, traded, turns, names


def portfolio(x: np.ndarray, inu: np.ndarray, r: np.ndarray, start: int, hold: int, q: float,
              cost_bp: float) -> tuple[np.ndarray, list[float], list[int]]:  # fmt: skip
    """The long-short portfolio's return each day (costs on the day of the rebalance), the
    weight traded at each rebalance, and the positions a side."""
    held, traded, turns, names = holdings(x, inu, start, hold, q)
    pnl = (held * np.nan_to_num(r)).sum(axis=1) - traded * cost_bp / 2 / 1e4
    return pnl, turns, names


def daily_ic(x: np.ndarray, fwd: np.ndarray, inu: np.ndarray, start: int,
             hold: int) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The days with a rank IC (10 coins or more, neither side all tied) and their ICs."""
    at, ics = [], []
    for i in range(start, len(x) - hold):
        m = inu[i] & np.isfinite(x[i]) & np.isfinite(fwd[i])
        if m.sum() >= 10 and np.ptp(x[i, m]) > 0 and np.ptp(fwd[i, m]) > 0:
            v = stats.spearmanr(x[i, m], fwd[i, m]).statistic
            if np.isfinite(v):
                at.append(i)
                ics.append(v)
    return np.array(at, dtype=int), np.array(ics)


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
    ic_days = daily_ic(x, fwd, inu, start, hold)[1]
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


@dataclass
class Explained:
    """Where a portfolio's money came from, all in % a year of the capital unless said."""

    signal: str
    hold: int
    years: list[dict]  # year: ic, gross, cost, net
    legs: dict[str, float]  # long and short, price and funding; the costs
    quintiles: list[dict]  # the signal's fifths: mean and median next-day return, bp
    coin_days: list[dict]  # the largest contributions of one coin on one day, either way
    best_share: float  # of the gross, the 10 best coin-days
    net_without_best: float  # the net without the 10 best coin-days


def explain(
    p: Panel,
    name: str,
    hold: int,
    top: int = 50,
    q: float = 0.2,
    cost_bp: float = 9.0,
    first: str = "2020-01-01",
    n_days: int = 10,
) -> Explained:  # fmt: skip
    """Breaks one portfolio's P&L down by year, by leg, into price and funding, and by coin
    and day; and the next day's return by fifth of the signal, mean and median, which shows
    where a rank IC and a P&L of opposite signs part."""
    x = signals(p)[name]
    r, inu = daily_returns(p), universe(p, top)
    start = p.dates.index(first) if first in p.dates else 0
    held, traded, _, _ = holdings(x, inu, start, hold, q)
    price = np.nan_to_num(p.close / _lag(p.close, 1) - 1)
    fund = np.nan_to_num(p.funding)
    c_price, c_fund = held * price, -held * fund
    contrib = c_price + c_fund
    cost = traded * cost_bp / 2 / 1e4
    pnl = contrib.sum(axis=1) - cost
    years_n = (len(p.dates) - start) / 365.0
    pct = 100 / years_n
    long, short = held > 0, held < 0
    legs = {
        "long_price": float(c_price[long].sum() * pct),
        "long_funding": float(c_fund[long].sum() * pct),
        "short_price": float(c_price[short].sum() * pct),
        "short_funding": float(c_fund[short].sum() * pct),
        "costs": float(-cost.sum() * pct),
    }
    fwd = forward(r, hold)
    at, ic = daily_ic(x, fwd, inu, start, hold)
    year = np.array([d[:4] for d in p.dates])
    years = []
    for y in dict.fromkeys(year[start:]):
        k = year == y
        k[:start] = False
        share = k.sum() / 365.0
        years.append({
            "year": y,
            "ic": float(ic[year[at] == y].mean()) if (year[at] == y).any() else float("nan"),
            "gross_pct": float((pnl[k].sum() + cost[k].sum()) / share * 100),
            "cost_pct": float(cost[k].sum() / share * 100),
            "net_pct": float(pnl[k].sum() / share * 100),
        })  # fmt: skip
    nxt = np.vstack([r[1:], np.full((1, r.shape[1]), np.nan)])  # the next day's return
    buckets: list[list[np.ndarray]] = [[] for _ in range(5)]
    for i in range(start, len(p.dates) - 1):
        m = inu[i] & np.isfinite(x[i]) & np.isfinite(nxt[i])
        if m.sum() < 10:
            continue
        rank = stats.rankdata(x[i, m]) / m.sum()  # (0, 1]
        b = np.minimum((rank * 5 - 1e-9).astype(int), 4)
        for j in range(5):
            buckets[j].append(nxt[i, m][b == j])
    quint = []
    for j in range(5):
        v = np.concatenate(buckets[j]) if buckets[j] else np.zeros(0)
        mean, median = (v.mean(), np.median(v)) if len(v) else (np.nan, np.nan)
        quint.append({"fifth": j + 1, "coin_days": len(v), "mean_bp": float(mean * 1e4),
                      "median_bp": float(median * 1e4)})  # fmt: skip
    flat = contrib[start:].ravel()
    order = np.argsort(flat, kind="stable")
    pick = [*order[::-1][:n_days], *order[: n_days // 2]]
    coin_days = []
    for f in pick:
        i, j = divmod(int(f), contrib.shape[1])
        i += start
        coin_days.append({"date": p.dates[i], "symbol": p.symbols[j],
                          "weight": float(held[i, j]), "price_pct": float(price[i, j] * 100),
                          "funding_pct": float(fund[i, j] * 100),
                          "contribution_pct": float(contrib[i, j] * 100)})  # fmt: skip
    gross = contrib[start:].sum()
    best = flat[order[::-1][:n_days]].sum()
    return Explained(
        signal=name,
        hold=hold,
        years=years,
        legs=legs,
        quintiles=quint,
        coin_days=coin_days,
        best_share=float(best / gross) if gross else float("nan"),
        net_without_best=float((pnl[start:].sum() - best) * pct),
    )
