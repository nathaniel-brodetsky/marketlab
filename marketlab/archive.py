"""Binance's public trade archives (data.binance.vision) as `scripts/fetch_binance.py` lays
them out: `<root>/<market>/<kind>/<SYMBOL>/<SYMBOL>-<kind>-<date>.zip`, one CSV per zip.
Reading them, and checking them before research touches them.

Columns, in Binance's order (some archives carry a header line, most do not):

- aggTrades: aggregate id, price, quantity, first trade id, last trade id, time, buyer was
  the maker, best price match. One row is one taker order's fills at one price.
- trades: trade id, price, quantity, quote quantity, time, buyer was the maker, best match.

Spot times are in microseconds from 2025-01-01 and milliseconds before; read times are
nanoseconds. Prices and quantities are read as floats.

The check looks at the data's integrity only, never at returns: ids that skip (missing
rows), raw trade ids an aggregate range does not reach, times that go back or fall outside
their day, empty quantities, and the longest silence. Across days: dates missing from the
range, and ids that do not continue from one day to the next.
"""

from __future__ import annotations

import datetime as dt
import io
import itertools
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

KINDS = {
    "aggTrades": [
        "agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker",
        "is_best_match",
    ],
    "trades": ["trade_id", "price", "qty", "quote_qty", "ts", "is_buyer_maker", "is_best_match"],
}  # fmt: skip
_TYPES = {"price": pl.Float64, "qty": pl.Float64, "quote_qty": pl.Float64}
NS_PER_DAY = 86_400 * 1_000_000_000
#: Times at or above this are microseconds (a millisecond time this large is in year 5138).
MICROS_FROM = 10**14


class ArchiveError(Exception):
    pass


@dataclass(frozen=True)
class ArchiveDay:
    path: Path
    date: str
    kind: str


def find(
    root: str | Path, symbol: str, kind: str = "aggTrades", market: str = "spot"
) -> list[ArchiveDay]:
    """The day files of a symbol, by date."""
    if kind not in KINDS:
        raise ArchiveError(f"kind {kind!r}: one of {', '.join(KINDS)}")
    folder = Path(root) / market / kind / symbol
    pattern = re.compile(rf"^{re.escape(symbol)}-{re.escape(kind)}-(\d{{4}}-\d{{2}}-\d{{2}})\.zip$")
    days = []
    if folder.is_dir():
        for p in folder.iterdir():
            m = pattern.match(p.name)
            if m:
                days.append(ArchiveDay(p, m.group(1), kind))
    return sorted(days, key=lambda d: d.date)


def read(day: ArchiveDay) -> tuple[pl.DataFrame, str]:
    """The day's rows with `ts` in nanoseconds, and the unit the file had ("us" or "ms")."""
    with zipfile.ZipFile(day.path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ArchiveError(f"{day.path}: expected one CSV inside, found {names}")
        data = z.read(names[0])
    columns = KINDS[day.kind]
    header = bool(data) and not data[:1].isdigit()
    df = pl.read_csv(
        io.BytesIO(data),
        has_header=header,
        new_columns=columns,
        schema_overrides={
            c: _TYPES.get(c, pl.String if c.startswith("is_") else pl.Int64) for c in columns
        },
    )  # fmt: skip
    df = df.with_columns(
        pl.col(c).str.to_lowercase() == "true" for c in columns if c.startswith("is_")
    )
    if df.is_empty():
        return df, "us"
    unit = "us" if int(df["ts"].max()) >= MICROS_FROM else "ms"
    scale = 1_000 if unit == "us" else 1_000_000
    return df.with_columns(pl.col("ts") * scale), unit


@dataclass
class DayCheck:
    date: str
    unit: str
    rows: int
    trades: int  # raw trades: the rows themselves, or the aggregates' id ranges
    volume: float  # base units
    buy_share: float  # of the volume, taker buys
    first_id: int
    last_id: int
    first_trade_id: int
    last_trade_id: int
    id_gaps: int  # places where the id skips
    ids_missing: int  # ids skipped in all
    trade_gaps: int  # aggTrades: an aggregate whose first trade id does not follow the last one's
    trades_missing: int
    backwards: int  # times earlier than the row before
    outside: int  # times outside the file's day
    empty: int  # rows with a price or quantity not above zero
    longest_silence_s: float
    problems: list[str] = field(default_factory=list)


def check_day(day: ArchiveDay) -> DayCheck:
    df, unit = read(day)
    start = int(dt.datetime.fromisoformat(day.date).replace(tzinfo=dt.UTC).timestamp()) * 10**9
    agg = day.kind == "aggTrades"
    idcol = "agg_id" if agg else "trade_id"
    ids = df[idcol].to_numpy()
    ts = df["ts"].to_numpy()
    qty = df["qty"].to_numpy()
    di = np.diff(ids)
    gaps = di != 1
    if agg:
        first, last = df["first_id"].to_numpy(), df["last_id"].to_numpy()
        trades = int((last - first + 1).sum())
        tgap = first[1:] != last[:-1] + 1
        t_missing = int(np.clip(first[1:] - last[:-1] - 1, 0, None).sum())
        first_trade, last_trade = (int(first[0]), int(last[-1])) if len(df) else (0, 0)
    else:
        trades = len(df)
        tgap = np.zeros(max(len(df) - 1, 0), dtype=bool)
        t_missing = 0
        first_trade, last_trade = (int(ids[0]), int(ids[-1])) if len(df) else (0, 0)
    edges = (
        np.concatenate([[start], ts, [start + NS_PER_DAY]])
        if len(df)
        else np.array([start, start + NS_PER_DAY])
    )
    volume = float(qty.sum())
    buys = float(qty[~df["is_buyer_maker"].to_numpy()].sum()) if len(df) else 0.0
    c = DayCheck(
        date=day.date,
        unit=unit,
        rows=len(df),
        trades=trades,
        volume=volume,
        buy_share=buys / volume if volume > 0 else float("nan"),
        first_id=int(ids[0]) if len(df) else 0,
        last_id=int(ids[-1]) if len(df) else 0,
        first_trade_id=first_trade,
        last_trade_id=last_trade,
        id_gaps=int(gaps.sum()),
        ids_missing=int(np.clip(di - 1, 0, None).sum()),
        trade_gaps=int(tgap.sum()),
        trades_missing=t_missing,
        backwards=int((np.diff(ts) < 0).sum()),
        outside=int(((ts < start) | (ts >= start + NS_PER_DAY)).sum()),
        empty=int(((df["price"].to_numpy() <= 0) | (qty <= 0)).sum()),
        longest_silence_s=float(np.diff(np.sort(edges)).max() / 1e9),
    )  # fmt: skip
    if c.rows == 0:
        c.problems.append("no rows")
    if c.id_gaps:
        c.problems.append(f"{c.ids_missing} ids missing in {c.id_gaps} places")
    if c.trade_gaps:
        c.problems.append(f"{c.trades_missing} raw trades missing in {c.trade_gaps} places")
    if c.backwards:
        c.problems.append(f"{c.backwards} times go back")
    if c.outside:
        c.problems.append(f"{c.outside} times outside the day")
    if c.empty:
        c.problems.append(f"{c.empty} rows with an empty price or quantity")
    return c


@dataclass
class ArchiveCheck:
    days: list[DayCheck]
    missing_dates: list[str]
    breaks: list[str]  # consecutive days whose ids do not continue

    @property
    def ok(self) -> bool:
        return not self.missing_dates and not self.breaks and not any(d.problems for d in self.days)


def check(days: list[ArchiveDay]) -> ArchiveCheck:
    checks = [check_day(d) for d in days]
    missing: list[str] = []
    breaks: list[str] = []
    for a, b in itertools.pairwise(checks):
        da, db = dt.date.fromisoformat(a.date), dt.date.fromisoformat(b.date)
        gap = (db - da).days
        missing += [(da + dt.timedelta(days=k)).isoformat() for k in range(1, gap)]
        if gap == 1 and a.rows and b.rows and b.first_id != a.last_id + 1:
            breaks.append(f"{a.date} -> {b.date}: ids {a.last_id} then {b.first_id}")
    return ArchiveCheck(checks, missing, breaks)


def table(result: ArchiveCheck) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": d.date,
                "unit": d.unit,
                "rows": d.rows,
                "trades": d.trades,
                "volume": round(d.volume, 2),
                "buy_share": round(d.buy_share, 3),
                "silence_s": round(d.longest_silence_s, 1),
                "problems": "; ".join(d.problems) or "-",
            }
            for d in result.days
        ]
    )
