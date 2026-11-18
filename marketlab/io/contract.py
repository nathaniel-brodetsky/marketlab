"""The data contract of marketrec, version 1 (`SCHEMA.md`): what a day directory holds.

marketrec only writes these files and this package only reads them. Everything here checks
what it reads against the contract and raises `ContractError` instead of guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1

_L20 = pa.list_(pa.int64(), 20)

TABLES: dict[str, pa.Schema] = {
    "depth_diff": pa.schema(
        [
            ("recv_ts", pa.int64()),
            ("event_ts", pa.int64()),
            ("first_update_id", pa.int64()),
            ("final_update_id", pa.int64()),
            ("side", pa.int8()),
            ("price", pa.int64()),
            ("qty", pa.int64()),
            ("source", pa.string()),
        ]
    ),
    "book_ticker": pa.schema(
        [
            ("recv_ts", pa.int64()),
            ("update_id", pa.int64()),
            ("bid_px", pa.int64()),
            ("bid_qty", pa.int64()),
            ("ask_px", pa.int64()),
            ("ask_qty", pa.int64()),
            ("source", pa.string()),
        ]
    ),
    "trades": pa.schema(
        [
            ("recv_ts", pa.int64()),
            ("event_ts", pa.int64()),
            ("trade_ts", pa.int64()),
            ("trade_id", pa.int64()),
            ("price", pa.int64()),
            ("qty", pa.int64()),
            ("is_buyer_maker", pa.bool_()),
            ("source", pa.string()),
        ]
    ),
    "book_l20": pa.schema(
        [
            ("recv_ts", pa.int64()),
            ("last_update_id", pa.int64()),
            ("bid_px", _L20),
            ("bid_qty", _L20),
            ("ask_px", _L20),
            ("ask_qty", _L20),
            ("source", pa.string()),
        ]
    ),
    "quality": pa.schema(
        [
            ("ts", pa.int64()),
            ("kind", pa.string()),
            ("detail", pa.string()),
            ("source", pa.string()),
        ]
    ),
    "link_rtt": pa.schema(
        [
            ("recv_ts", pa.int64()),
            ("conn", pa.int64()),
            ("rtt_ns", pa.int64()),
            ("source", pa.string()),
        ]
    ),
}

#: The column with each table's time: the receive time, except in `quality`.
TIME_COLUMN = {name: ("ts" if name == "quality" else "recv_ts") for name in TABLES}

QUALITY_KINDS = frozenset(
    {
        "connected",
        "disconnected",
        "gap",
        "snapshot_too_old",
        "invalid_book",
        "buffer_overflow",
        "reconnect",
        "overflow",
        "synced",
        "resync",
        "bbo_mismatch",
        "undecodable",
        "snapshot_failed",
        "server_shutdown",
        "late",
        "on_time",
    }
)

_PART = re.compile(r"part-(\d+)\.parquet")


class ContractError(Exception):
    """The files do not follow the data contract this package reads."""


@dataclass(frozen=True)
class Run:
    """One recorder run that wrote to a day (`runs` in `manifest.json`)."""

    recorder: str
    first_recv_ts: int
    last_recv_ts: int
    #: The book's update ID when the run's data for this day began, if the book was valid
    #: then (a run that goes on over midnight); else None.
    book_at_start: int | None
    derivation: int = 0


@dataclass(frozen=True)
class Manifest:
    symbol: str
    date: str
    price_step: Decimal
    qty_step: Decimal
    runs: tuple[Run, ...]
    rows: dict[str, int] = field(default_factory=dict)
    files: dict[str, list[str]] = field(default_factory=dict)
    quality: dict[str, int] = field(default_factory=dict)
    synced_share: float | None = None
    raw: dict = field(default_factory=dict, repr=False)


def read_manifest(day: Path) -> Manifest | None:
    """The day's manifest, or None while a run that started this day has not written one."""
    path = Path(day) / "manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    version = m.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ContractError(
            f"{path}: schema version {version}, this package reads {SCHEMA_VERSION}"
        )
    try:
        runs = tuple(
            Run(
                recorder=r["recorder"],
                first_recv_ts=int(r["first_recv_ts"]),
                last_recv_ts=int(r["last_recv_ts"]),
                book_at_start=None if r.get("book_at_start") is None else int(r["book_at_start"]),
                derivation=int(r.get("derivation", 0)),
            )
            for r in m["runs"]
        )
        return Manifest(
            symbol=m["symbol"],
            date=m["date"],
            price_step=Decimal(m["price_step"]),
            qty_step=Decimal(m["qty_step"]),
            runs=tuple(sorted(runs, key=lambda r: r.first_recv_ts)),
            rows={k: int(v) for k, v in m.get("rows", {}).items()},
            files={k: list(v) for k, v in m.get("files", {}).items()},
            quality={k: int(v) for k, v in m.get("quality", {}).items()},
            synced_share=m.get("synced_share"),
            raw=m,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ContractError(f"{path}: {e!r}") from e


@dataclass(frozen=True)
class PartMeta:
    table: str
    symbol: str
    price_step: Decimal
    qty_step: Decimal


def part_files(day: Path, table: str) -> list[Path]:
    """The finished parts of a table, in order. Unfinished parts are hidden `.tmp` files and
    never match."""
    d = Path(day) / table
    if not d.is_dir():
        return []
    parts = [(int(m.group(1)), p) for p in d.iterdir() if (m := _PART.fullmatch(p.name))]
    return [p for _, p in sorted(parts)]


def _same_columns(a: pa.Schema, b: pa.Schema) -> bool:
    def norm(t: pa.DataType) -> object:
        if pa.types.is_fixed_size_list(t):
            return ("fixed_size_list", t.list_size, norm(t.value_type))
        return t

    return [(f.name, norm(f.type)) for f in a] == [(f.name, norm(f.type)) for f in b]


def part_meta(path: Path, table: str) -> PartMeta:
    """Check one part's schema and key-value metadata against the contract."""
    f = pq.ParquetFile(path)
    if not _same_columns(f.schema_arrow, TABLES[table]):
        raise ContractError(f"{path}: columns {f.schema_arrow.names} differ from the contract")
    kv = {k.decode(): v.decode() for k, v in (f.metadata.metadata or {}).items()}
    try:
        version = int(kv["marketrec.schema_version"])
        meta = PartMeta(
            table=kv["marketrec.table"],
            symbol=kv["marketrec.symbol"],
            price_step=Decimal(kv["marketrec.price_step"]),
            qty_step=Decimal(kv["marketrec.qty_step"]),
        )
    except (KeyError, ValueError) as e:
        raise ContractError(f"{path}: metadata {e!r} missing or malformed") from e
    if version != SCHEMA_VERSION:
        raise ContractError(
            f"{path}: schema version {version}, this package reads {SCHEMA_VERSION}"
        )
    if meta.table != table:
        raise ContractError(f"{path}: holds table {meta.table}, expected {table}")
    return meta


def _may_overlap(meta: pq.FileMetaData, column: str, between: tuple[int, int]) -> bool:
    """Whether a part can hold rows with `column` in `[lo, hi)`, by its row groups'
    statistics; True when they are missing."""
    lo, hi = between
    names = meta.schema.names
    if column not in names:
        return True
    i = names.index(column)
    for g in range(meta.num_row_groups):
        st = meta.row_group(g).column(i).statistics
        if st is None or not st.has_min_max or (st.min < hi and st.max >= lo):
            return True
    return False


def read_parts(
    day: Path,
    table: str,
    columns: list[str] | None = None,
    expect: PartMeta | None = None,
    between: tuple[int, int] | None = None,
) -> pa.Table:
    """All finished parts of a table, checked against the contract and against `expect`
    (symbol and steps); an empty table with the contract's columns if there are none.
    With `between=(lo, hi)`, only the rows with time in `[lo, hi)`: parts and row groups
    outside are skipped by their statistics, so minutes of a day read in a fraction of the
    whole."""
    schema = TABLES[table]
    cols = columns or schema.names
    unknown = set(cols) - set(schema.names)
    if unknown:
        raise ContractError(f"{table} has no columns {sorted(unknown)}")
    pieces = []
    for p in part_files(day, table):
        meta = part_meta(p, table)
        if expect is not None and (meta.symbol, meta.price_step, meta.qty_step) != (
            expect.symbol,
            expect.price_step,
            expect.qty_step,
        ):
            raise ContractError(
                f"{p}: {meta.symbol} with steps {meta.price_step}/{meta.qty_step}, "
                f"expected {expect.symbol} with {expect.price_step}/{expect.qty_step}"
            )
        if between is None:
            pieces.append(pq.read_table(p, columns=cols))
            continue
        t = TIME_COLUMN[table]
        if not _may_overlap(pq.ParquetFile(p).metadata, t, between):
            continue
        lo, hi = between
        pieces.append(pq.read_table(p, columns=cols, filters=[(t, ">=", lo), (t, "<", hi)]))
    if not pieces:
        return pa.Table.from_batches([], schema=pa.schema([schema.field(c) for c in cols]))
    return pa.concat_tables(pieces)
