#!/usr/bin/env python3
"""Download, from Binance's public archive (data.binance.vision), the hourly klines of the
USD-M perpetuals named in a file, for the months between two dates in which each has daily
bars: to price a daily rule traded some hours after the daily close. Standard library only,
Python 3.9 or newer, any OS. Run it after fetch_universe.py (and fetch_archive_universe.py)
on their folder:

    python3 fetch_hourly.py --symbols-file runs/universe-dev.txt \\
        --start 2020-01-01 --end 2023-12-31

Each monthly file the archive lists for a symbol (an S3 listing, as fetch_archive_universe.py
reads it) is checked against its checksum and kept under OUT/um/klines/<SYMBOL>/1h/. Which
months: those of the dates that the symbol's daily bars in OUT/um/klines1d/<SYMBOL>.csv
touch, within --start .. --end (only the bars' days are read there, not their prices).
Running it again downloads nothing twice. Exit code 1 if anything failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fetch_archive_universe",
                                               HERE / "fetch_archive_universe.py")  # fmt: skip
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def bar_months(out: Path, sym: str, wanted: set[str]) -> set[str]:
    """The months, among `wanted`, in which `sym` has a daily bar: its first column only."""
    path = out / "um" / "klines1d" / f"{sym}.csv"
    if not path.exists():
        return set()
    have = set()
    with path.open() as f:
        next(f, None)
        for line in f:
            t = line.split(",", 1)[0]
            if t.strip():
                have.add(fa._date(t)[:7])
    return have & wanted


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--symbols-file", required=True, type=Path, help="one symbol a line")
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD, included")
    p.add_argument("--out", default="binance-data", type=Path, help="default ./binance-data")
    p.add_argument("--jobs", default=4, type=int, help="files downloaded at once (default 4)")
    p.add_argument("--tries", default=5, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--base-url", default=fa.fb.BASE, help=argparse.SUPPRESS)
    p.add_argument("--list-url", default=fa.LIST, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.end < a.start:
        raise SystemExit(f"--end {a.end} is before --start {a.start}")
    syms = [s.strip() for s in a.symbols_file.read_text().splitlines() if s.strip()]
    wanted = set(fa.fb.months(a.start, a.end))
    failed: list[str] = []
    files = 0
    for n, sym in enumerate(syms, 1):
        months = bar_months(a.out, sym, wanted)
        if not months:
            print(f"  [{n}/{len(syms)}] {sym:<16} no daily bars in these months", flush=True)
            continue
        try:
            paths, bad = fa.fetch_months(a, "klines", sym, months, "1h")
        except (RuntimeError, OSError, fa.fb.Missing) as e:
            failed.append(sym)
            print(f"  [{n}/{len(syms)}] {sym:<16} FAILED: {e}", file=sys.stderr, flush=True)
            continue
        failed += [f"{sym} {b}" for b in bad]
        files += len(paths)
        gap = len(months) - len(paths)
        print(f"  [{n}/{len(syms)}] {sym:<16} {len(paths)} of {len(months)} months"
              + (f" ({gap} not in the archive)" if gap else ""), flush=True)  # fmt: skip
    print(f"done: {files} monthly files of hourly bars for {len(syms)} perpetuals "
          f"-> {a.out / 'um' / 'klines'}")  # fmt: skip
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
