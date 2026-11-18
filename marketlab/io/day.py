"""One recorded day of one symbol, as the research code sees it.

`Day.open` reads a day directory written by marketrec. Every table can be read whole
(`raw`) or only where the data can be used (`valid`), with the segment each row falls in.
`truncated(t)` gives the same day as it was known at time `t`; the tests use it to show that
no feature looks ahead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from .contract import (
    TABLES,
    TIME_COLUMN,
    ContractError,
    PartMeta,
    Run,
    part_files,
    part_meta,
    read_manifest,
    read_parts,
)
from .intervals import segment_ids, valid_intervals

Loader = Callable[[str, tuple[str, ...]], pl.DataFrame]


@dataclass(frozen=True)
class Book20:
    """The `book_l20` rows inside valid intervals, as arrays. Levels are best first; a missing
    level has price and quantity 0."""

    ts: np.ndarray  # int64, receive time
    update_id: np.ndarray  # int64
    segment: np.ndarray  # int32
    bid_px: np.ndarray  # int64 (n, 20), steps
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)


class Day:
    def __init__(
        self,
        *,
        symbol: str,
        date: str,
        price_step: Decimal,
        qty_step: Decimal,
        runs: Sequence[Run],
        loader: Loader,
        source: str = "json",
        include_late: bool = False,
        until: int | None = None,
        path: Path | None = None,
    ) -> None:
        self.symbol = symbol
        self.date = date
        self.price_step = Decimal(price_step)
        self.qty_step = Decimal(qty_step)
        self.runs = tuple(runs)
        self.source = source
        self.include_late = include_late
        self.until = until
        self.path = path
        self._loader = loader
        self._cache: dict[tuple[str, tuple[str, ...]], pl.DataFrame] = {}

    # construction

    @classmethod
    def open(cls, path: str | Path, source: str = "json", include_late: bool = False) -> Day:
        """A day directory `<root>/binance/<symbol>/<date>`."""
        path = Path(path)
        manifest = read_manifest(path)
        if manifest is not None:
            meta = PartMeta("", manifest.symbol, manifest.price_step, manifest.qty_step)
            runs: Sequence[Run] = manifest.runs
            missing = [
                f for files in manifest.files.values() for f in files if not (path / f).exists()
            ]
            if missing:
                raise ContractError(f"{path}: the manifest lists missing parts {missing}")
        else:
            meta = _meta_from_parts(path)
            runs = ()
        day_meta = meta

        def load(table: str, columns: tuple[str, ...]) -> pl.DataFrame:
            return pl.from_arrow(read_parts(path, table, list(columns), expect=day_meta))

        day = cls(
            symbol=meta.symbol,
            date=manifest.date if manifest else path.name,
            price_step=meta.price_step,
            qty_step=meta.qty_step,
            runs=runs,
            loader=load,
            source=source,
            include_late=include_late,
            path=path,
        )
        if not runs:
            day.runs = (day._pseudo_run(),)
        return day

    @classmethod
    def from_tables(
        cls,
        tables: Mapping[str, pl.DataFrame],
        *,
        symbol: str,
        date: str,
        price_step: Decimal | str,
        qty_step: Decimal | str,
        runs: Sequence[Run],
        source: str = "json",
        include_late: bool = False,
    ) -> Day:
        """A day held in memory, for tests and synthetic data."""
        frames = dict(tables)

        def load(table: str, columns: tuple[str, ...]) -> pl.DataFrame:
            if table in frames:
                return frames[table].select(columns)
            return pl.DataFrame(schema={c: _polars_type(table, c) for c in columns})

        return cls(
            symbol=symbol,
            date=date,
            price_step=Decimal(price_step),
            qty_step=Decimal(qty_step),
            runs=runs,
            loader=load,
            source=source,
            include_late=include_late,
        )

    def truncated(self, until: int) -> Day:
        """The same day as known at receive time `until`: rows after it are gone."""
        bound = until if self.until is None else min(until, self.until)
        return Day(
            symbol=self.symbol,
            date=self.date,
            price_step=self.price_step,
            qty_step=self.qty_step,
            runs=self.runs,
            loader=self._loader,
            source=self.source,
            include_late=self.include_late,
            until=bound,
            path=self.path,
        )

    # reading

    def raw(self, table: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
        """All rows of the table from this day's feed, in receive order."""
        if table not in TABLES:
            raise ContractError(f"no table {table}")
        cols = tuple(columns) if columns else tuple(TABLES[table].names)
        key = (table, cols)
        if key not in self._cache:
            t = TIME_COLUMN[table]
            need = tuple(dict.fromkeys((*cols, t, "source")))
            df = self._loader(table, need)
            df = df.filter(pl.col("source") == self.source)
            if self.until is not None:
                df = df.filter(pl.col(t) <= self.until)
            self._cache[key] = df.select(cols)
        return self._cache[key]

    @cached_property
    def intervals(self) -> list[tuple[int, int]]:
        """Intervals `[start, end)` of receive time where the data can be used."""
        q = self.raw("quality", ("ts", "kind", "detail"))
        runs = self.runs
        if self.until is not None:
            runs = tuple(
                Run(r.recorder, r.first_recv_ts, min(r.last_recv_ts, self.until), r.book_at_start)
                for r in runs
                if r.first_recv_ts <= self.until
            )
        events = zip(q["ts"].to_list(), q["kind"].to_list(), q["detail"].to_list(), strict=True)
        return valid_intervals(events, runs, include_late=self.include_late)

    def valid(self, table: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
        """The rows inside valid intervals, with the `segment` they fall in."""
        t = TIME_COLUMN[table]
        cols = tuple(columns) if columns else tuple(TABLES[table].names)
        df = self.raw(table, tuple(dict.fromkeys((*cols, t))))
        seg = segment_ids(df[t].to_numpy(), self.intervals)
        return (
            df.with_columns(pl.Series("segment", seg, dtype=pl.Int32))
            .filter(pl.col("segment") >= 0)
            .select(*cols, "segment")
        )

    def book20(self) -> Book20:
        df = self.valid("book_l20")
        return Book20(
            ts=df["recv_ts"].to_numpy(),
            update_id=df["last_update_id"].to_numpy(),
            segment=df["segment"].to_numpy(),
            bid_px=_levels(df, "bid_px"),
            bid_qty=_levels(df, "bid_qty"),
            ask_px=_levels(df, "ask_px"),
            ask_qty=_levels(df, "ask_qty"),
        )

    def segment_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        iv = self.intervals
        return (
            np.array([s for s, _ in iv], dtype=np.int64),
            np.array([e for _, e in iv], dtype=np.int64),
        )

    @property
    def tick(self) -> float:
        return float(self.price_step)

    def __repr__(self) -> str:
        return f"Day({self.symbol} {self.date}, feed {self.source})"

    # helpers

    def _pseudo_run(self) -> Run:
        """Without a manifest (a day still being written): one run over all the data, with
        the book unknown at its start."""
        lo, hi = None, None
        for table in ("quality", "book_ticker", "trades", "book_l20", "depth_diff"):
            t = TIME_COLUMN[table]
            col = self.raw(table, (t,))[t]
            if len(col):
                lo = col.min() if lo is None else min(lo, col.min())
                hi = col.max() if hi is None else max(hi, col.max())
        if lo is None:
            return Run("unknown", 0, -1, None)
        return Run("unknown", int(lo), int(hi), None)


def _levels(df: pl.DataFrame, col: str) -> np.ndarray:
    a = df[col].to_numpy()
    return a.reshape(len(df), 20) if a.size else np.zeros((0, 20), dtype=np.int64)


def _polars_type(table: str, column: str) -> pl.DataType:
    field = TABLES[table].field(column)
    return pl.from_arrow(pa.Table.from_batches([], schema=pa.schema([field]))).schema[column]


def _meta_from_parts(path: Path) -> PartMeta:
    for table in TABLES:
        files = part_files(path, table)
        if files:
            m = part_meta(files[0], table)
            return PartMeta("", m.symbol, m.price_step, m.qty_step)
    raise ContractError(f"{path}: no manifest and no parts")
