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
The universe is the perpetuals in um/universe.csv (those Binance lists, on an underlying it
calls a coin) and in um/gone.csv (those gone from its list, from its archive); without
gone.csv, coins delisted are missing, which flatters anything that buys losers. Any other
file under um/klines1d (a contract on shares, a fund or a metal, say) is left out, and so
are seven contracts that are not a coin's own price (`NOT_COINS`). A perpetual that settled
funding on a day with no daily bar (often an earlier listing whose bars Binance no longer
serves) is missing from the universe on that day; `Panel.unbarred` counts those days.
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
    settlements: np.ndarray | None = None  # days x symbols: how many settled that day
    # left out, with um/universe.csv there: perpetuals on an underlying Binance does not call
    # a coin (symbol: its type), and files of contracts neither listed as perpetuals nor gone
    other_type: dict[str, str] = field(default_factory=dict)
    unlisted: list[str] = field(default_factory=list)
    listed: bool = False  # um/universe.csv was there to say which files are perpetuals


def read_columns(path: Path, types: dict[str, pl.DataType]) -> pl.DataFrame:
    """The columns named, with their types: read as text and then cast, so that a column
    whose first rows look whole (a volume of 100, then 1.5) cannot fail the read."""
    raw = pl.read_csv(path, infer_schema=False, columns=list(types))
    return raw.select(pl.col(c).cast(t) for c, t in types.items())


def _day(ms: np.ndarray) -> np.ndarray:
    return ms // 86_400_000


def perpetuals(root: str | Path) -> tuple[set[str] | None, dict[str, str]]:
    """The symbols that are perpetuals on coins, by um/universe.csv (those Binance lists) and
    um/gone.csv (those gone from its list, from its archive), or None without universe.csv;
    and the listed perpetuals on an underlying of another type than a coin, with the type."""
    um = Path(root) / "um"
    if not (um / "universe.csv").exists():
        return None, {}
    listed = pl.read_csv(um / "universe.csv", infer_schema=False)
    if "underlying_type" not in listed.columns:
        listed = listed.with_columns(underlying_type=pl.lit(""))
    kind = listed["underlying_type"].fill_null("").str.to_uppercase()
    coin = kind.is_in(["", "COIN"])
    names, kinds = listed["symbol"].filter(~coin), listed["underlying_type"].filter(~coin)
    other = dict(zip(names, kinds, strict=True))
    keep = set(listed["symbol"].filter(coin))
    if (um / "gone.csv").exists():
        keep |= set(pl.read_csv(um / "gone.csv", infer_schema=False)["symbol"])
    return keep, other


def load(root: str | Path, first: str, last: str, held_out: bool = False) -> Panel:
    """Every perpetual's daily bars and funding up to `last`; the days from `first` on,
    with 40 days before it for the signals. `last` after `DEV_END` is refused, unless
    `held_out`: only a hypothesis's held-out run (`xsection_kind.final`) asks for that."""
    if last > DEV_END and not held_out:
        raise HeldOut(f"{last} is after {DEV_END}: the years after it are held out")
    d0 = (dt.date.fromisoformat(first) - dt.date(1970, 1, 1)).days - 40
    d1 = (dt.date.fromisoformat(last) - dt.date(1970, 1, 1)).days
    days = np.arange(d0, d1 + 1)
    folder = Path(root) / "um" / "klines1d"
    keep, other = perpetuals(root)
    every = sorted(f for f in folder.glob("*.csv") if f.stem not in NOT_COINS)
    files = [f for f in every if keep is None or f.stem in keep]
    unlisted = [f.stem for f in every if keep is not None and f.stem not in keep
                and f.stem not in other]  # fmt: skip
    symbols = [f.stem for f in files]
    close = np.full((len(days), len(files)), np.nan)
    qvol = np.full_like(close, np.nan)
    funding = np.full_like(close, np.nan)
    settlements = np.zeros(close.shape, dtype=np.int64)
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
            t = r["funding_time"].to_numpy()
            if len(np.unique(t)) != len(t):  # summed twice, it would double the day's rate
                raise ValueError(f"{fr}: a settlement time given twice")
            fd = _day(t)
            inside = (fd >= d0) & (fd <= d1)
            col = np.zeros(len(days))
            np.add.at(col, fd[inside] - d0, r["funding_rate"].to_numpy()[inside])
            funding[:, j] = col
            np.add.at(settlements[:, j], fd[inside] - d0, 1)
            n = int(((settlements[:, j] > 0) & ~np.isfinite(close[:, j])).sum())
            if n:
                unbarred[f.stem] = n
    dates = [(dt.date(1970, 1, 1) + dt.timedelta(days=int(x))).isoformat() for x in days]
    return Panel(dates, symbols, close, qvol, funding, unbarred, settlements, other, unlisted,
                 keep is not None)  # fmt: skip


def largest_funding(p: Panel, inu: np.ndarray, start: int, n: int = 5) -> list[dict]:
    """The `n` largest rates settled in one day by a perpetual in the universe that day,
    either way, from `start` on: symbol, date, % and how many settlements made it."""
    f = np.where(inu, np.abs(np.nan_to_num(p.funding)), 0.0)[start:]
    count = p.settlements if p.settlements is not None else np.zeros(p.close.shape, int)
    out = []
    for k in np.argsort(-f.ravel(), kind="stable")[:n]:
        i, j = divmod(int(k), f.shape[1])
        if f[i, j] == 0:
            break
        i += start
        out.append({"symbol": p.symbols[j], "date": p.dates[i],
                    "pct": float(p.funding[i, j] * 100),
                    "settlements": int(count[i, j])})  # fmt: skip
    return out


HOUR_MS = 3_600_000


Bars = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _hourly_bars(root: str | Path, sym: str, held_out: bool) -> Bars:
    """(hours since the epoch, and the open, high and low of the hour bar starting then) from
    the monthly files under um/klines/<sym>/1h; a file of a month after `DEV_END`'s is not
    opened, unless `held_out`."""
    import io
    import zipfile

    folder = Path(root) / "um" / "klines" / sym / "1h"
    last = DEV_END[:7]
    hours, opens, highs, lows = [], [], [], []
    for f in sorted(folder.glob(f"{sym}-1h-*.zip")):
        if not held_out and f.stem[-7:] > last:
            continue
        with zipfile.ZipFile(f) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"{f}: expected one CSV inside, found {names}")
            data = z.read(names[0])
        if not data.strip():
            continue
        header = not data[:1].isdigit()
        raw = pl.read_csv(io.BytesIO(data), has_header=header, infer_schema=False,
                          new_columns=None if header else [f"c{k}" for k in range(12)])  # fmt: skip
        t = raw[raw.columns[0]].cast(pl.Int64).to_numpy()
        t = np.where(t > 10**14, t // 1000, t)  # µs from 2025 on, ms before
        hours.append(t // HOUR_MS)
        opens.append(raw[raw.columns[1]].cast(pl.Float64).to_numpy())
        highs.append(raw[raw.columns[2]].cast(pl.Float64).to_numpy())
        lows.append(raw[raw.columns[3]].cast(pl.Float64).to_numpy())
    if not hours:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
    h = np.concatenate(hours)
    order = np.argsort(h, kind="stable")
    return (h[order], np.concatenate(opens)[order], np.concatenate(highs)[order],
            np.concatenate(lows)[order])  # fmt: skip


@dataclass
class Lagged:
    """A long's return each day as if traded `hours` after each daily close: from the open
    of the hour bar `hours` after one close to that after the next, less the funding settled
    between (the hour itself included, as a close includes its own settlement)."""

    hours: int
    r: np.ndarray  # days x symbols, NaN where an hour bar is missing at either end
    priced: np.ndarray  # days x symbols: an hour bar at that day's trading time
    # over the same span, the 24 hour bars from its first price: the highest high over that
    # price, less 1, and 1 less the lowest low over it; NaN without the price or any bar
    up: np.ndarray | None = None
    down: np.ndarray | None = None


def lagged(p: Panel, root: str | Path, hours: list[int],
           held_out: bool = False) -> dict[int, Lagged]:  # fmt: skip
    """`p`'s returns traded each of `hours` (0 .. 23) after each daily close. Day `D`'s
    close is the start of day `D + 1`; traded `hours` later, at its hour bar's open; and how
    far each day's span went each way, from its hour bars. Nothing from `DEV_END`'s next day
    on is read, bars nor funding, unless `held_out`."""
    if any(not 0 <= h < 24 for h in hours):
        raise ValueError("hours: from 0 to 23 after the close")
    if p.dates[-1] > DEV_END and not held_out:
        raise HeldOut(f"{p.dates[-1]} is after {DEV_END}: the years after it are held out")
    day0 = (dt.date.fromisoformat(p.dates[0]) - dt.date(1970, 1, 1)).days
    end_ms = ((dt.date.fromisoformat(DEV_END) - dt.date(1970, 1, 1)).days + 1) * 86_400_000
    n, m = len(p.dates), len(p.symbols)
    price = {L: np.full((n, m), np.nan) for L in hours}
    fund = {L: np.zeros((n, m)) for L in hours}
    high = {L: np.full((n, m), np.nan) for L in hours}
    low = {L: np.full((n, m), np.nan) for L in hours}
    for j, sym in enumerate(p.symbols):
        h, o, hi, lo = _hourly_bars(root, sym, held_out)
        fr = Path(root) / "um" / "fundingRate" / sym / f"{sym}-fundingRate.csv"
        rf = read_columns(fr, FUNDING_TYPES) if fr.exists() else None
        if rf is not None and not held_out:  # the last day's window reaches past DEV_END
            rf = rf.filter(pl.col("funding_time") < end_ms)
        # the hour bars laid out from the panel's first day: hour 24 i + k is day i's k-th
        at = h - day0 * 24
        ok = (at >= 0) & (at < (n + 1) * 24)
        bar_hi, bar_lo = np.full((n + 1) * 24, -np.inf), np.full((n + 1) * 24, np.inf)
        bar_hi[at[ok]], bar_lo[at[ok]] = hi[ok], lo[ok]
        for L in hours:
            if len(h):
                want = (day0 + 1 + np.arange(n)) * 24 + L  # day D's trading hour
                k = np.clip(np.searchsorted(h, want), 0, len(h) - 1)
                price[L][:, j] = np.where(h[k] == want, o[k], np.nan)
            # day D's span, from day D - 1's trading hour to day D's: its 24 hour bars
            top = bar_hi[L : L + n * 24].reshape(n, 24).max(axis=1)
            bottom = bar_lo[L : L + n * 24].reshape(n, 24).min(axis=1)
            high[L][:, j] = np.where(np.isfinite(top), top, np.nan)
            low[L][:, j] = np.where(np.isfinite(bottom), bottom, np.nan)
            if rf is not None:
                t = rf["funding_time"].to_numpy()
                d = (t - L * HOUR_MS) // 86_400_000 - day0  # [D + L, D + 1 + L)
                ok = (d >= 0) & (d < n)
                np.add.at(fund[L][:, j], d[ok], rf["funding_rate"].to_numpy()[ok])
    out = {}
    for L in hours:
        before = _lag(price[L], 1)  # the span's first price
        out[L] = Lagged(L, price[L] / before - 1 - fund[L], np.isfinite(price[L]),
                        high[L] / before - 1, 1 - low[L] / before)  # fmt: skip
    return out


def at_hours(p: Panel, root: str | Path, rules: list[tuple[str, int]], hours: list[int],
             top: int = 50, q: float = 0.2, cost_bp: float = 9.0,
             first: str = "2020-01-01", staggered: bool = False) -> list[dict]:  # fmt: skip
    """Each rule (signal, hold) decided at the daily close as ever, traded `hours` later:
    its net a year, Sharpe from months, and the share of its positions with an hour bar to
    trade at (a position without one returns nothing that day)."""
    x, inu = signals(p), universe(p, top)
    start = p.dates.index(first) if first in p.dates else 0
    month = np.array([d[:7] for d in p.dates[start:]])
    years = (len(p.dates) - start) / 365.0
    lag = lagged(p, root, hours)
    out = []
    for name, hold in rules:
        held, _, _, _ = holdings(x[name], inu, start, hold, q, staggered)
        for L in hours:
            pnl, _, _ = portfolio(x[name], inu, lag[L].r, start, hold, q, cost_bp, staggered)
            net = pnl[start:]
            sums = np.array([net[month == mo].sum() for mo in dict.fromkeys(month)])
            on = held != 0
            before = np.zeros_like(lag[L].priced)
            before[1:] = lag[L].priced[:-1]
            ends = lag[L].priced & before  # a price at both ends of the day
            priced = float(ends[on].mean()) if on.any() else float("nan")
            out.append({"rule": f"{name}:{hold}", "hours": L, "priced": priced,
                        "net_pct": float(net.sum() / years * 100),
                        "sharpe_months": _sharpe(sums)})  # fmt: skip
    return out


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k] if k else a
    return out


def _trailing_mean(a: np.ndarray, k: int) -> np.ndarray:
    """The mean of the last `k` rows, this one included; NaN unless all are there (a NaN
    in the window carries through the sum). Each window is summed on its own, in order: a
    day's mean is the same to the last bit wherever the data begin. (A running sum's
    difference was not: its rounding carried what came before, and coins tied on a rate
    were ranked by it.)"""
    out = np.full(a.shape, np.nan)
    if len(a) >= k:
        last = len(a) - k + 1
        s = a[:last].astype(float)
        for t in range(1, k):
            s = s + a[t : t + last]
        out[k - 1 :] = s / k
    return out


def signals(p: Panel) -> dict[str, np.ndarray]:
    lc = np.log(p.close)
    return {
        "mom7": lc - _lag(lc, 7),
        "mom30": lc - _lag(lc, 30),
        "rev1": -(lc - _lag(lc, 1)),
        # rounded where no two rates differ (they are given to 1e-8): coins whose windows
        # hold the same rates in another order tie, as they should, not by rounding
        "carry": -np.round(_trailing_mean(p.funding, 7), 12),
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


Held = tuple[np.ndarray, np.ndarray, list[float], list[int]]


def _tranches(start: int, hold: int, staggered: bool) -> list[int]:
    """The days a rule's portfolios start on: `start`; or, staggered and held more than a
    day, `hold` of them a day apart from it, one rebalanced each day."""
    return [start + k for k in range(hold)] if staggered and hold > 1 else [start]


def _pick(x: np.ndarray, inu: np.ndarray, q: float) -> np.ndarray:
    """A rebalance's weights from a day's signal `x` over the coins in the universe `inu`:
    long those strictly above its `1 - q` quantile, short those strictly below its `q`
    quantile, half the capital a side in equal weights; nothing unless both sides have a
    coin."""
    new = np.zeros(len(x))
    idx = np.flatnonzero(inu & np.isfinite(x))
    if len(idx):
        v = x[idx]
        hi, lo = np.quantile(v, 1 - q), np.quantile(v, q)
        longs, shorts = idx[v > hi], idx[v < lo]  # strictly: ties are not ranked
        if len(longs) and len(shorts):
            new[longs] = 0.5 / len(longs)
            new[shorts] = -0.5 / len(shorts)
    return new


def holdings(x: np.ndarray, inu: np.ndarray, start: int, hold: int, q: float,
             staggered: bool = False) -> Held:  # fmt: skip
    """The weights held over each day (decided at the close of the day before), the weight
    traded each day, the weight traded at each rebalance, and the positions a side.

    `staggered`: `hold` portfolios with a `hold`-th of the capital each, the k-th started `k`
    days after `start`, so that one is rebalanced every day and no start day decides the
    result (a weekly rule started on a Monday or a Sunday holds the same coins once all
    seven are in). Each trades on its own: what two of them trade in one coin the same day is
    not netted, so the costs are an upper bound."""
    starts = _tranches(start, hold, staggered)
    if len(starts) > 1:
        parts = [holdings(x, inu, s, hold, q) for s in starts]
        held = sum(h for h, _, _, _ in parts) / len(starts)
        traded = sum(t for _, t, _, _ in parts) / len(starts)
        turns = [u / len(starts) for _, _, us, _ in parts for u in us]
        names = [n for _, _, _, ns in parts for n in ns]
        return held, traded, turns, names
    n_days, n_sym = x.shape
    w = np.zeros(n_sym)
    held = np.zeros((n_days, n_sym))
    traded = np.zeros(n_days)
    turns: list[float] = []
    names: list[int] = []
    for i in range(start, n_days - 1):
        if (i - start) % hold == 0:
            new = _pick(x[i], inu[i], q)
            if new.any():
                names.append(int(np.count_nonzero(new)) // 2)
            turns.append(float(np.abs(new - w).sum()))
            traded[i] = turns[-1]
            w = new
        held[i + 1] = w
    return held, traded, turns, names


@dataclass
class Margin:
    """Each position on isolated margin of its own size, as it is opened at a rebalance: a
    loss takes from that margin and nothing else, and a position whose margin falls to the
    `maintenance` rate of its size at the day's worst price (from the hour bars) is
    liquidated, its whole margin lost, and holds nothing until its portfolio's next
    rebalance. Between rebalances a position is held at its weight, as without. The day's
    worst price is its hour bars' extreme, or its close where that is worse (or there are
    no hour bars)."""

    up: np.ndarray  # days x symbols: as Lagged.up, over the day each return spans
    down: np.ndarray  # as Lagged.down
    maintenance: float  # of the position's size at that price


def _isolated(held: np.ndarray, r: np.ndarray, start: int, hold: int,
              margin: Margin) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One portfolio's `held` (rebalanced every `hold` days from `start`) on isolated margin:
    its return each day and the weight it traded."""
    n, m = held.shape
    pnl, traded = np.zeros(n), np.zeros(n)
    w, alive, lost = np.zeros(m), np.ones(m, dtype=bool), np.zeros(m)
    for i in range(start, n - 1):
        if (i - start) % hold == 0:
            new = held[i + 1]
            traded[i] = np.abs(new - w * alive).sum()
            w, alive, lost = new, np.ones(m, dtype=bool), np.zeros(m)
        size = np.abs(w)
        on = alive & (w != 0)
        day = np.where(on, w * r[i + 1], 0.0)
        # the day's worst move against the position: its hour bars' extreme, or its close's
        hours = np.where(w > 0, margin.down[i + 1], margin.up[i + 1])
        close = np.maximum(np.where(w > 0, -r[i + 1], r[i + 1]), 0.0)
        against = np.fmax(hours, close)
        at_worst = size * (1 + np.where(w > 0, -against, against))  # the position's size then
        out = on & (size + lost - size * against <= margin.maintenance * at_worst)
        day[out] = -(size[out] + lost[out])  # what was left of its margin
        alive &= ~out
        lost += day
        pnl[i + 1] = day.sum()
    return pnl, traded


@dataclass
class Stop:
    """A position whose loss since its rebalance, its funding in, reaches `weights` times its
    own weight at the end of a day (the day's return ends at its trading hour) is sold there,
    at that hour's price, paying for the sale, and holds nothing until its portfolio's next
    rebalance."""

    weights: float


def _stopped(held: np.ndarray, r: np.ndarray, start: int, hold: int,
             stop: Stop) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One portfolio's `held` (rebalanced every `hold` days from `start`) with a stop: its
    return each day and the weight it traded."""
    n, m = held.shape
    pnl, traded = np.zeros(n), np.zeros(n)
    w, alive, lost = np.zeros(m), np.ones(m, dtype=bool), np.zeros(m)
    for i in range(start, n - 1):
        if (i - start) % hold == 0:
            new = held[i + 1]
            traded[i] += np.abs(new - w * alive).sum()
            w, alive, lost = new, np.ones(m, dtype=bool), np.zeros(m)
        size = np.abs(w)
        on = alive & (w != 0)
        day = np.where(on, w * r[i + 1], 0.0)
        out = on & (lost + day <= -stop.weights * size)
        traded[i + 1] += size[out].sum()  # sold at the day's trading hour
        alive &= ~out
        lost += day
        pnl[i + 1] = day.sum()
    return pnl, traded


Pnl = tuple[np.ndarray, list[float], list[int]]


def portfolio(x: np.ndarray, inu: np.ndarray, r: np.ndarray, start: int, hold: int, q: float,
              cost_bp: float, staggered: bool = False, margin: Margin | None = None,
              stop: Stop | None = None) -> Pnl:  # fmt: skip
    """The long-short portfolio's return each day (costs on the day they are paid), the
    weight traded at each rebalance, and the positions a side; with `margin`, each position
    on isolated margin of its size (`Margin`), or with `stop`, sold at a loss (`Stop`); each
    staggered portfolio on its own."""
    if margin is not None and stop is not None:
        raise ValueError("a margin or a stop, not both")
    if margin is not None or stop is not None:
        rr = np.nan_to_num(r)
        starts = _tranches(start, hold, staggered)  # each on its own
        _, _, turns, names = holdings(x, inu, start, hold, q, staggered)
        pnl, traded = np.zeros(len(r)), np.zeros(len(r))
        for s in starts:
            held_s, _, _, _ = holdings(x, inu, s, hold, q)
            if margin is not None:
                ps, ts = _isolated(held_s, rr, s, hold, margin)
            else:
                ps, ts = _stopped(held_s, rr, s, hold, stop)
            pnl, traded = pnl + ps / len(starts), traded + ts / len(starts)
        return pnl - traded * cost_bp / 2 / 1e4, turns, names
    held, traded, turns, names = holdings(x, inu, start, hold, q, staggered)
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
    staggered: bool = False,
) -> Result:  # fmt: skip
    r = daily_returns(p) if r is None else r
    inu = universe(p, top) if inu is None else inu
    start = p.dates.index(first) if first in p.dates else 0
    fwd = forward(r, hold)
    ic_days = daily_ic(x, fwd, inu, start, hold)[1]
    ic = float(ic_days.mean()) if len(ic_days) else float("nan")
    se = metrics.newey_west_se(ic_days, hold) if len(ic_days) > 2 else float("nan")
    pnl, turns, names = portfolio(x, inu, r, start, hold, q, cost_bp, staggered)
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
    staggered: bool = False,
) -> tuple[Panel, list[Result]]:
    p = load(root, first, last)
    sig = signals(p)
    r, inu = daily_returns(p), universe(p, top)
    out = []
    for name in SIGNALS:
        for h in holds:
            if name == "rev1" and h != 1:
                continue
            out.append(evaluate(p, name, sig[name], h, top, q, cost_bp, first, r, inu,
                                staggered))  # fmt: skip
    return p, out


def market(inu: np.ndarray, r: np.ndarray, start: int) -> np.ndarray:
    """The universe's return each day, equal weights: those in it at the day before's
    close, as a portfolio decided then holds them; nothing before `start + 1`."""
    mkt = np.zeros(len(r))
    for i in range(start + 1, len(r)):
        m = inu[i - 1] & np.isfinite(r[i])
        if m.any():
            mkt[i] = r[i, m].mean()
    return mkt


def slope(y: np.ndarray, x: np.ndarray) -> float:
    """The least-squares slope of `y` on `x`: a beta."""
    return float(np.cov(y, x)[0, 1] / x.var(ddof=1)) if len(x) > 1 and x.var() > 0 else float("nan")


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
    market_pct: float  # the universe, equal weights, held as the portfolio's day is
    beta: float  # of the portfolio's gross day on the market's
    net_less_beta: float  # the net less beta times the market
    late_net: float  # the same portfolio decided a day late: on the day before's signal
    late_sharpe: float


def explain(
    p: Panel,
    name: str,
    hold: int,
    top: int = 50,
    q: float = 0.2,
    cost_bp: float = 9.0,
    first: str = "2020-01-01",
    n_days: int = 10,
    staggered: bool = False,
) -> Explained:  # fmt: skip
    """Breaks one portfolio's P&L down by year, by leg, into price and funding, and by coin
    and day; and the next day's return by fifth of the signal, mean and median, which shows
    where a rank IC and a P&L of opposite signs part. Then two checks: the portfolio's beta
    to the universe (equal weights), so that a long leg that only rides a rising market is
    seen; and the same portfolio decided a day late, which shows how much of it depends on
    trading at the very close its signal was read at."""
    x = signals(p)[name]
    r, inu = daily_returns(p), universe(p, top)
    start = p.dates.index(first) if first in p.dates else 0
    held, traded, _, _ = holdings(x, inu, start, hold, q, staggered)
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
    # the market a day: the universe the portfolio was decided in, equal weights, over the
    # day the portfolio is held
    mkt = market(inu, r, start)
    g, mk = contrib.sum(axis=1)[start + 1 :], mkt[start + 1 :]
    beta = slope(g, mk)
    # decided a day late: the signal and the universe of the day before
    inu_late = np.zeros_like(inu)
    inu_late[1:] = inu[:-1]
    late, _, _ = portfolio(_lag(x, 1), inu_late, r, start, hold, q, cost_bp, staggered)
    month = np.array([d[:7] for d in p.dates[start:]])
    late_months = np.array([late[start:][month == mo].sum() for mo in dict.fromkeys(month)])
    return Explained(
        signal=name,
        hold=hold,
        years=years,
        legs=legs,
        quintiles=quint,
        coin_days=coin_days,
        best_share=float(best / gross) if gross else float("nan"),
        net_without_best=float((pnl[start:].sum() - best) * pct),
        market_pct=float(mkt[start:].sum() * pct),
        beta=beta,
        net_less_beta=float((pnl[start:] - beta * mkt[start:]).sum() * pct),
        late_net=float(late[start:].sum() * pct),
        late_sharpe=_sharpe(late_months),
    )
