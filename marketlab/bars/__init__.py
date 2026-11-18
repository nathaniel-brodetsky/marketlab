"""Bars: by time, and by activity (a number of trades, of base units, of quote currency).

Every bar lies inside one segment: time bars that a hole touches are dropped, activity bars
start again after each hole, and the unfinished bar at a segment's end is dropped. A bar's
price is the mid at its close, taken from the best bid and ask at or before the close, so
that the trade price's bounce between bid and ask does not count as a return.

`compare` puts the bar types side by side. The claim that activity bars give returns closer
to normal and less dependent on each other (Lopez de Prado, "Advances in Financial Machine
Learning", ch. 2) is not taken on faith: it is measured on the recorded days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from ..features.base import duration_ns
from ..io.day import Day

KINDS = ("tick", "volume", "dollar")


def _mid_at(day: Day, ts: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Mid (in price steps) from the last best bid and ask at or before each time, in the
    same segment."""
    bt = day.valid("book_ticker", ["recv_ts", "bid_px", "ask_px"])
    b_ts = bt["recv_ts"].to_numpy()
    mid = (bt["bid_px"].to_numpy() + bt["ask_px"].to_numpy()) / 2.0
    b_seg = bt["segment"].to_numpy()
    i = np.searchsorted(b_ts, ts, side="right") - 1
    ok = i >= 0
    j = np.clip(i, 0, None)
    ok &= (b_seg[j] == seg) if len(b_seg) else False
    out = np.full(len(ts), np.nan)
    if len(mid):
        out[ok] = mid[j[ok]]
    return out


def _trades(day: Day) -> pl.DataFrame:
    t = day.valid("trades", ["recv_ts", "price", "qty", "is_buyer_maker"])
    return t.with_columns(
        notional=pl.col("price") * pl.col("qty"),  # in price step x qty step: exact
        buy_qty=pl.when(~pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0),
    )


def _finish(day: Day, df: pl.DataFrame) -> pl.DataFrame:
    """Add the mid at the close and convert to base and quote units."""
    tick, unit = float(day.price_step), float(day.qty_step)
    close_mid = _mid_at(day, df["end"].to_numpy() - 1, df["segment"].to_numpy())
    return df.with_columns(
        pl.Series("mid", close_mid * tick, nan_to_null=True),
        (pl.col("open", "high", "low", "close") * tick),
        (pl.col("volume", "buy_volume") * unit),
        (pl.col("notional") * tick * unit),
    )


def time_bars(day: Day, every: str = "1s") -> pl.DataFrame:
    """Bars `[start, end)` on a clock aligned to multiples of `every`, only where the whole
    bar lies in one segment. Bars without trades are kept (no open/high/low/close, volume
    0): a quiet second is information too."""
    n = duration_ns(every)
    rows = []
    for seg, (lo, hi) in enumerate(day.intervals):
        first = -(-lo // n) * n  # the first boundary at or after the segment's start
        count = (hi - first) // n
        if count > 0:
            starts = first + n * np.arange(count, dtype=np.int64)
            rows.append(pl.DataFrame({"start": starts, "segment": np.full(count, seg, np.int32)}))
    bars = (
        pl.concat(rows) if rows else pl.DataFrame(schema={"start": pl.Int64, "segment": pl.Int32})
    ).with_columns(end=pl.col("start") + n)
    t = _trades(day).with_columns(start=(pl.col("recv_ts") // n) * n)
    agg = t.group_by("start", maintain_order=True).agg(
        open=pl.col("price").first(),
        high=pl.col("price").max(),
        low=pl.col("price").min(),
        close=pl.col("price").last(),
        volume=pl.col("qty").sum(),
        buy_volume=pl.col("buy_qty").sum(),
        notional=pl.col("notional").sum(),
        trades=pl.len(),
    )
    df = bars.join(agg, on="start", how="left").with_columns(
        pl.col("volume", "buy_volume", "notional", "trades").fill_null(0)
    )
    return _finish(day, df.select(_COLUMNS))


_COLUMNS = [
    "start",
    "end",
    "segment",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "buy_volume",
    "notional",
    "trades",
]


def activity_bars(day: Day, kind: str, threshold: int) -> pl.DataFrame:
    """A bar closes at the trade that brings its count of trades (`tick`), base quantity in
    steps (`volume`) or notional in price step x quantity step (`dollar`) to `threshold` or
    beyond; trades are not split and the next bar starts from zero. Bars start again in
    each segment and the unfinished last one is dropped. `start` is the first trade's
    receive time, `end` one nanosecond after the closing trade's."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    t = _trades(day)
    measure = {
        "tick": pl.lit(1, dtype=pl.Int64),
        "volume": pl.col("qty"),
        "dollar": pl.col("notional"),
    }
    t = t.with_columns(m=measure[kind])
    pieces = []
    for _, part in t.group_by("segment", maintain_order=True):
        bar = _close_at(part["m"].to_list(), threshold)
        keep = bar >= 0
        if keep.any():
            pieces.append(part.filter(pl.Series(keep)).with_columns(bar=pl.Series(bar[keep])))
    if not pieces:
        empty = pl.DataFrame(schema={c: pl.Int64 for c in _COLUMNS}).with_columns(
            pl.col("segment").cast(pl.Int32)
        )
        return _finish(day, empty)
    t = pl.concat(pieces)
    df = t.group_by("segment", "bar", maintain_order=True).agg(
        start=pl.col("recv_ts").first(),
        end=pl.col("recv_ts").last() + 1,
        open=pl.col("price").first(),
        high=pl.col("price").max(),
        low=pl.col("price").min(),
        close=pl.col("price").last(),
        volume=pl.col("qty").sum(),
        buy_volume=pl.col("buy_qty").sum(),
        notional=pl.col("notional").sum(),
        trades=pl.len(),
    )
    return _finish(day, df.select(_COLUMNS))


def _close_at(measure: list[int], threshold: int) -> np.ndarray:
    """Bar number of each trade, -1 for the trades of an unfinished last bar."""
    out = np.empty(len(measure), dtype=np.int64)
    k = total = 0
    for i, x in enumerate(measure):
        out[i] = k
        total += x
        if total >= threshold:
            k += 1
            total = 0
    out[out == k] = -1
    return out


@dataclass(frozen=True)
class Thresholds:
    tick: int
    volume: int  # quantity steps
    dollar: int  # price step x quantity step


def calibrate(days: list[Day], bars_per_hour: float) -> Thresholds:
    """Thresholds that give about `bars_per_hour` bars of each kind per hour of valid data
    on these days. Fit them on training days when the bars feed a model."""
    trades = volume = notional = 0
    seconds = 0.0
    for d in days:
        t = _trades(d)
        trades += t.height
        volume += int(t["qty"].sum()) if t.height else 0
        notional += int(t["notional"].sum()) if t.height else 0
        seconds += sum(e - s for s, e in d.intervals) / 1e9
    bars = bars_per_hour * seconds / 3600
    if bars <= 0 or trades == 0:
        raise ValueError("no trades in valid data")
    return Thresholds(
        tick=max(1, round(trades / bars)),
        volume=max(1, round(volume / bars)),
        dollar=max(1, round(notional / bars)),
    )


def returns(bars: pl.DataFrame) -> np.ndarray:
    """Log returns of the close mid between consecutive bars of the same segment."""
    mid = bars["mid"].to_numpy().astype(float)
    seg = bars["segment"].to_numpy()
    if len(mid) < 2:
        return np.zeros(0)
    same = seg[1:] == seg[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(mid[1:] / mid[:-1])
    r = r[same]
    return r[np.isfinite(r)]


def properties(r: np.ndarray, lags: int = 10) -> dict[str, float]:
    """Shape and dependence of a return series."""
    n = len(r)
    if n < lags + 3 or np.std(r) == 0:
        return {"n": n}
    z = r - r.mean()
    acf = np.array([np.dot(z[:-k], z[k:]) / np.dot(z, z) for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    jb = stats.jarque_bera(r)
    return {
        "n": n,
        "std_bps": float(np.std(r) * 1e4),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jarque_bera": float(jb.statistic),
        "acf1": float(acf[0]),
        "ljung_box_p": float(stats.chi2.sf(q, lags)),
        "zero_share": float(np.mean(r == 0)),
    }


def compare(day: Day, every: str = "1s", thresholds: Thresholds | None = None) -> pl.DataFrame:
    """Time bars of `every` against tick, volume and dollar bars. Without thresholds they
    are calibrated on this day to the number of time bars, which describes the day but
    must not feed a model."""
    tb = time_bars(day, every)
    if thresholds is None:
        per_hour = 3600 / (duration_ns(every) / 1e9)
        thresholds = calibrate([day], per_hour)
    tick, unit = day.price_step, day.qty_step
    shown = {
        "tick": f"{thresholds.tick} trades",
        "volume": f"{thresholds.volume * unit:f}".rstrip("0").rstrip(".") + " base",
        "dollar": f"{float(thresholds.dollar * tick * unit):,.0f} quote",
    }
    rows = [{"bars": f"time {every}", "threshold": every, **properties(returns(tb))}]
    for kind in KINDS:
        b = activity_bars(day, kind, getattr(thresholds, kind))
        rows.append({"bars": kind, "threshold": shown[kind], **properties(returns(b))})
    return pl.DataFrame(rows)
