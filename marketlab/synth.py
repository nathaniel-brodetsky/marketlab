"""Synthetic days in the contract's shape, for tests and for trying the code without a
recording.

The market is a toy: limit orders arrive and leave at random levels near the top, market
orders eat the best level, and the best price moves when a level is used up. It only has to
be consistent the way the recorder's output is: `book_ticker` and `book_l20` show the same
book as the depth updates, trades take quantity from the levels they hit, update IDs run on,
and the quality journal says where the book was not valid. Nothing here says anything about
real markets.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .io.contract import SCHEMA_VERSION, TABLES, TIME_COLUMN, Run
from .io.day import Day

MS = 1_000_000
SEC = 1_000_000_000
#: 2026-11-21 00:00 UTC, the date the marketrec fixtures use.
DAY0 = 1_795_219_200 * SEC
PRICE_STEP = "0.01000000"
QTY_STEP = "0.00001000"


@dataclass
class Scenario:
    seed: int = 1
    start: int = DAY0 + 10 * 3600 * SEC
    seconds: float = 60.0
    #: Mean events per second: changes of resting quantity, and trades.
    book_rate: float = 40.0
    trade_rate: float = 6.0
    #: Holes: (second, length) of a `gap` followed by a `resync`.
    gaps: Sequence[tuple[float, float]] = ()
    #: Lost connections: (second, length) of a `reconnect` followed by `synced`.
    reconnects: Sequence[tuple[float, float]] = ()
    #: Late data: (second, length) between `late` and `on_time`.
    late: Sequence[tuple[float, float]] = ()
    #: Normal lag of the data behind the exchange.
    lag_ms: float = 140.0
    levels: int = 40
    mid_ticks: int = 9_700_000  # 97000.00 in steps of 0.01


@dataclass
class _Tables:
    depth_diff: list[tuple] = field(default_factory=list)
    book_ticker: list[tuple] = field(default_factory=list)
    trades: list[tuple] = field(default_factory=list)
    book_l20: list[tuple] = field(default_factory=list)
    quality: list[tuple] = field(default_factory=list)


def generate(s: Scenario) -> dict[str, pl.DataFrame]:
    """The tables of one synthetic run, every row from the `json` feed."""
    rng = random.Random(s.seed)
    out = _Tables()
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    best_bid = s.mid_ticks - 1
    for k in range(s.levels):
        bids[best_bid - k] = rng.randint(1, 200_000)
        asks[best_bid + 1 + k] = rng.randint(1, 200_000)
    update_id = 50_000_000_000
    trade_id = 3_000_000_000
    end = s.start + int(s.seconds * SEC)

    def window(spans: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
        return [(s.start + int(a * SEC), s.start + int((a + b) * SEC)) for a, b in spans]

    gaps, reconnects, lates = window(s.gaps), window(s.reconnects), window(s.late)

    def inside(t: int, spans: list[tuple[int, int]]) -> bool:
        return any(a <= t < b for a, b in spans)

    # the journal: when the book stops and starts being valid
    q = out.quality
    q.append((s.start, "connected", "conn=1"))
    synced_at = s.start + 3 * MS
    q.append((synced_at, "synced", f"update_id={update_id}"))
    for a, b in gaps:
        q.append((a, "gap", f"expected={update_id} got={update_id + 2}"))
        q.append((b, "resync", f"update_id={update_id}"))
    for n, (a, b) in enumerate(reconnects, start=2):
        q.append((a, "disconnected", f'conn={n - 1} reason="connection reset by peer"'))
        q.append((a, "reconnect", ""))
        q.append((a + (b - a) // 2, "connected", f"conn={n}"))
        q.append((b, "synced", f"update_id={update_id}"))
    for a, b in lates:
        q.append((a, "late", "lag_ms=2000"))
        q.append((b, "on_time", f"lag_ms=140 late_for_ms={(b - a) // MS} max_lag_ms=2000"))
    q.sort(key=lambda r: r[0])

    def lag(t: int) -> int:
        return 2_000 * MS if inside(t, lates) else int(s.lag_ms * MS)

    def book_valid(t: int) -> bool:
        return t >= synced_at and not inside(t, gaps) and not inside(t, reconnects)

    def connected(t: int) -> bool:
        return not any(a <= t < a + (b - a) // 2 for a, b in reconnects)

    def best() -> tuple[int, int, int, int]:
        b = max(bids)
        a = min(asks)
        return b, bids[b], a, asks[a]

    last_ticker = None
    changed: dict[tuple[int, int], int] = {}
    first_id_in_batch = update_id + 1
    next_diff = s.start + 100 * MS
    next_sample = s.start
    t = s.start
    rate = s.book_rate + s.trade_rate
    while True:
        t += max(1, int(rng.expovariate(rate) * SEC))
        if t >= end:
            break
        # depth updates go out every 100 ms with every level changed since the last one
        while next_diff <= t:
            if changed and connected(next_diff):
                ev = next_diff - lag(next_diff)
                for (side, price), qty in sorted(changed.items()):
                    out.depth_diff.append(
                        (next_diff, ev, first_id_in_batch, update_id, side, price, qty)
                    )
                changed.clear()
                first_id_in_batch = update_id + 1
            next_diff += 100 * MS
        if not connected(t):
            continue
        update_id += 1
        if rng.random() < s.trade_rate / rate:
            buy = rng.random() < 0.5
            book, price = (asks, min(asks)) if buy else (bids, max(bids))
            qty = min(book[price], rng.randint(1, 60_000))
            trade_id += 1
            out.trades.append((t, t - lag(t), t - lag(t) - MS, trade_id, price, qty, not buy))
            book[price] -= qty
            side = -1 if buy else 1
            if book[price] == 0:
                del book[price]
            changed[(side, price)] = book.get(price, 0)
        else:
            side = 1 if rng.random() < 0.5 else -1
            book = bids if side == 1 else asks
            top = max(bids) if side == 1 else min(asks)
            depth = min(int(rng.expovariate(0.25)), s.levels - 1)
            price = top - depth if side == 1 else top + depth
            b, _, a, _ = best()
            if a - b > 1 and rng.random() < 0.7:
                # a wide spread is closed quickly by a new best level inside it
                price = b + 1 if side == 1 else a - 1
                book[price] = rng.randint(1, 80_000)
            elif rng.random() < 0.55:
                book[price] = book.get(price, 0) + rng.randint(1, 80_000)
            elif price in book:
                book[price] = max(0, book[price] - rng.randint(1, 80_000))
                if book[price] == 0:
                    del book[price]
            changed[(side, price)] = book.get(price, 0)
        # keep both sides deep and never empty
        while len(bids) < s.levels:
            p = min(bids) - 1 if bids else min(asks) - 1
            bids[p] = rng.randint(1, 200_000)
            changed[(1, p)] = bids[p]
        while len(asks) < s.levels:
            p = max(asks) + 1 if asks else max(bids) + 1
            asks[p] = rng.randint(1, 200_000)
            changed[(-1, p)] = asks[p]
        ticker = best()
        if ticker != last_ticker:
            out.book_ticker.append((t, update_id, *ticker))
            last_ticker = ticker
        if t >= next_sample:
            if book_valid(t):
                bp = sorted(bids, reverse=True)[:20]
                ap = sorted(asks)[:20]
                out.book_l20.append(
                    (
                        t,
                        update_id,
                        _pad(bp),
                        _pad([bids[p] for p in bp]),
                        _pad(ap),
                        _pad([asks[p] for p in ap]),
                    )
                )
            next_sample = (t // (100 * MS) + 1) * (100 * MS)
    stop = end
    q.append((stop, "disconnected", 'conn=1 reason="stopped"'))
    return _frames(out)


def tables(**rows: list[tuple]) -> dict[str, pl.DataFrame]:
    """Tables from rows given by hand, in the contract's column order without `source`."""
    return _frames(_Tables(**rows))


def memory_day(
    frames: dict[str, pl.DataFrame],
    price_step: str = "1",
    qty_step: str = "1",
    runs: Sequence[Run] | None = None,
    include_late: bool = False,
) -> Day:
    """A day from tables made by hand; one run over all rows unless runs are given."""
    return Day.from_tables(
        frames,
        symbol="TEST",
        date="2026-11-21",
        price_step=price_step,
        qty_step=qty_step,
        runs=runs or [run_of(frames)],
        include_late=include_late,
    )


def _pad(v: list[int]) -> list[int]:
    return v + [0] * (20 - len(v))


def _frames(t: _Tables) -> dict[str, pl.DataFrame]:
    rows = {
        "depth_diff": t.depth_diff,
        "book_ticker": t.book_ticker,
        "trades": t.trades,
        "book_l20": t.book_l20,
        "quality": t.quality,
    }
    out = {}
    for name, data in rows.items():
        schema = TABLES[name]
        cols = [c for c in schema.names if c != "source"]
        columns = list(zip(*data, strict=True)) if data else [[] for _ in cols]
        arrays = [
            pa.array(list(col), type=schema.field(c).type)
            for c, col in zip(cols, columns, strict=True)
        ]
        arrays.append(pa.array(["json"] * len(data), type=pa.string()))
        out[name] = pl.from_arrow(pa.Table.from_arrays(arrays, schema=schema))
    return out


def run_of(tables: dict[str, pl.DataFrame]) -> Run:
    lo = min(int(df[TIME_COLUMN[n]].min()) for n, df in tables.items() if df.height)
    hi = max(int(df[TIME_COLUMN[n]].max()) for n, df in tables.items() if df.height)
    return Run("synthetic", lo, hi, None, 1)


def day(s: Scenario | None = None, include_late: bool = False) -> Day:
    """A synthetic day in memory."""
    s = s or Scenario()
    tables = generate(s)
    return Day.from_tables(
        tables,
        symbol="BTCUSDT",
        date=_date(s.start),
        price_step=PRICE_STEP,
        qty_step=QTY_STEP,
        runs=[run_of(tables)],
        include_late=include_late,
    )


def write(s: Scenario, root: Path) -> Path:
    """Write a synthetic run as marketrec would: one part per table, and the manifest."""
    tables = generate(s)
    date = _date(s.start)
    d = Path(root) / "binance" / "BTCUSDT" / date
    files: dict[str, list[str]] = {}
    for name, df in tables.items():
        if df.height == 0:
            continue
        (d / name).mkdir(parents=True, exist_ok=True)
        t = df.to_arrow().cast(TABLES[name])
        t = t.replace_schema_metadata(
            {
                "marketrec.schema_version": str(SCHEMA_VERSION),
                "marketrec.table": name,
                "marketrec.symbol": "BTCUSDT",
                "marketrec.price_step": PRICE_STEP,
                "marketrec.qty_step": QTY_STEP,
            }
        )
        pq.write_table(t, d / name / "part-000.parquet")
        files[name] = [f"{name}/part-000.parquet"]
    run = run_of(tables)
    q = tables["quality"]["kind"].value_counts()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "symbol": "BTCUSDT",
        "date": date,
        "price_step": PRICE_STEP,
        "qty_step": QTY_STEP,
        "host": "synthetic",
        "runs": [
            {
                "recorder": run.recorder,
                "first_recv_ts": run.first_recv_ts,
                "last_recv_ts": run.last_recv_ts,
                "book_at_start": None,
                "derivation": 1,
            }
        ],
        "rows": {n: df.height for n, df in tables.items()},
        "files": files,
        "quality": dict(zip(q["kind"].to_list(), q["count"].to_list(), strict=True)),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return d


def _date(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts / SEC, tz=dt.UTC).strftime("%Y-%m-%d")
