"""`marketlab`: look at recorded days from the command line."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path

import polars as pl

from . import bars, synth
from .features import compute
from .io import ContractError, Day, read_manifest
from .registry import FITTED, METHODS, default_methods
from .report import describe

SEC = 1_000_000_000


def _clock(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / SEC, tz=dt.UTC).strftime("%H:%M:%S")


def _seconds(intervals: list[tuple[int, int]]) -> float:
    return sum(e - s for s, e in intervals) / SEC


def cmd_day(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    print(f"day         {a.day}")
    print(
        f"symbol      {day.symbol}, price step {day.price_step.normalize():f}, "
        f"qty step {day.qty_step.normalize():f}, feed {day.source}"
    )
    for r in day.runs:
        start = (
            "book carried over from the day before"
            if r.book_at_start is not None
            else "book synced after the start"
        )
        print(
            f"run         {r.recorder}, {_clock(r.first_recv_ts)}-{_clock(r.last_recv_ts)} UTC, "
            f"{start}, derivation {r.derivation}"
        )
    rows = ", ".join(
        f"{t} {day.raw(t, [c]).height}"
        for t, c in (
            ("depth_diff", "recv_ts"),
            ("book_ticker", "recv_ts"),
            ("trades", "recv_ts"),
            ("book_l20", "recv_ts"),
            ("quality", "ts"),
        )
    )
    print(f"rows        {rows}")
    manifest = read_manifest(Path(a.day))
    if manifest is not None and manifest.raw.get("covered_ns"):
        # time with a stream connection open: not the hours between two runs
        covered = manifest.raw["covered_ns"] / SEC
        what = "with a connection open"
    else:
        covered = sum(r.last_recv_ts - r.first_recv_ts + 1 for r in day.runs) / SEC
        what = "from the first row to the last"
    usable = _seconds(day.intervals)
    with_late = Day.open(a.day, include_late=True)
    late = _seconds(with_late.intervals) - _seconds(Day.open(a.day).intervals)
    holes = max(covered - _seconds(with_late.intervals), 0.0)
    share = usable / covered if covered > 0 else 0.0
    print(
        f"usable      {usable:.1f} s of {covered:.1f} s {what} ({share:.2%}), in "
        f"{len(day.intervals)} segments"
    )
    print(
        f"excluded    {holes:.1f} s without a valid book, "
        f"{late:.1f} s late{'' if not a.late else ' (kept: --late)'}"
    )
    if day.intervals:
        longest = max(day.intervals, key=lambda iv: iv[1] - iv[0])
        print(f"longest     {(longest[1] - longest[0]) / SEC:.1f} s from {_clock(longest[0])} UTC")
    return 0


def cmd_bars(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    table = bars.compare(day, a.every)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(table)
    print(
        "thresholds are calibrated on this day to the number of time bars: a description "
        "of the day, not an input for a model"
    )
    return 0


def cmd_features(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    train = [Day.open(p, include_late=a.late) for p in a.train]
    methods = []
    for m in default_methods():
        if m.name in FITTED:
            if not train:
                print(f"skip        {m.name}: needs --train days to fit on")
                continue
            m.fit(train)
        methods.append(m)
    table = compute(day, methods)
    print(f"grid        {table.height} rows")
    print(describe(table))
    if a.out:
        table.write_parquet(a.out)
        print(f"written     {a.out}")
    return 0


def _latency(a: argparse.Namespace) -> tuple[str, str]:
    """The latency to use and where it came from: given, or from the day's manifest as
    half the median round trip plus the median lag of the data."""
    if a.latency:
        return a.latency, "given"
    m = read_manifest(Path(a.day))
    rtt = m.raw.get("rtt_p50_ms") if m else None
    lag = m.raw.get("lag_p50_ms") if m else None
    if rtt is None or lag is None:
        raise ContractError(
            "the manifest has no round trip or lag to derive the latency from: give --latency"
        )
    ms = rtt / 2 + lag
    return f"{ms:g}ms", f"RTT p50 {rtt} ms / 2 + lag p50 {lag} ms, from the manifest"


def cmd_execution(a: argparse.Namespace) -> int:
    from .execution import run, summary

    day = Day.open(a.day, include_late=a.late)
    latency, source = _latency(a)
    qty = int(Decimal(a.size) / day.qty_step)
    if qty <= 0:
        raise ContractError(f"--size {a.size} is less than one quantity step {day.qty_step}")
    print(f"latency     {latency} ({source})")
    print(
        f"orders      at the best bid and the best ask every {a.every}, {a.size} each, "
        f"post-only, cancelled after {a.cancel_after}"
        + (f"; market orders of {a.size} as well" if a.aggressive else "")
    )
    orders, fills = run(
        day,
        every=a.every,
        qty=qty,
        cancel_after=a.cancel_after,
        latency=latency,
        aggressive=a.aggressive,
    )
    table = summary(orders, fills)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(table)
    print(
        "markouts in basis points, positive when the fill was good for us; conclusions "
        "from the pessimistic row only"
    )
    if a.out:
        fills.write_parquet(a.out)
        print(f"written     {a.out}")
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    s = synth.Scenario(
        seed=a.seed,
        seconds=a.seconds,
        gaps=[(a.seconds * 0.3, 5.0)],
        late=[(a.seconds * 0.6, 3.0)],
    )
    path = synth.write(s, Path(a.out))
    print(f"written     {path} (synthetic: a toy market, not real data)")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marketlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def day_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("day", help="day directory, e.g. data/binance/BTCUSDT/2026-11-21")
        sp.add_argument(
            "--late", action="store_true", help="keep intervals where the data was late"
        )

    d = sub.add_parser("day", help="what a day holds and how much of it is usable")
    day_arg(d)
    d.set_defaults(run=cmd_day)

    b = sub.add_parser("bars", help="time bars against tick, volume and dollar bars")
    day_arg(b)
    b.add_argument("--every", default="1s", help="time bar length (default 1s)")
    b.set_defaults(run=cmd_bars)

    f = sub.add_parser("features", help="layer 1 features on a day, summarized")
    day_arg(f)
    f.add_argument("--train", nargs="*", default=[], help="days to fit fitted methods on")
    f.add_argument("--out", help="write the features to this Parquet file")
    f.set_defaults(run=cmd_features)

    e = sub.add_parser("execution", help="probe orders through the fill simulator: fills, markouts")
    day_arg(e)
    e.add_argument("--every", default="10s", help="one order per side this often (default 10s)")
    e.add_argument("--size", default="0.001", help="order size in base units (default 0.001)")
    e.add_argument("--cancel-after", default="10s", help="passive orders' life (default 10s)")
    e.add_argument("--latency", help="decision to effect, e.g. 300ms (default: from the manifest)")
    e.add_argument("--aggressive", action="store_true", help="probe market orders as well")
    e.add_argument("--out", help="write the fills with their markouts to this Parquet file")
    e.set_defaults(run=cmd_execution)

    s = sub.add_parser("synth", help="write a synthetic day to try the commands")
    s.add_argument("out", help="data root, e.g. /tmp/synth")
    s.add_argument("--seconds", type=float, default=600.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(run=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.run(a)
    except ContractError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


assert set(METHODS) >= FITTED

if __name__ == "__main__":
    sys.exit(main())
