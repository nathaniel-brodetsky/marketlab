#!/usr/bin/env python3
"""Download a series from FRED, the St. Louis Fed's public data (no key), as CSV. Standard
library only, Python 3.9 or newer, any OS.

    python3 fetch_fred.py --series GS3M

GS3M: the market yield of three-month US Treasury bills, % a year, monthly. The file is
OUT/fred/<SERIES>.csv as FRED gives it (a date and a value a line, "." where there is
none), replaced only once the new one has been read whole. If FRED's download address does
not answer, the same CSV can be saved by hand from the series' page (Download, CSV).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://fred.stlouisfed.org"
AGENT = {"User-Agent": "fetch_fred/1.0"}


def rows(text: str) -> list[tuple[str, float]]:
    """The dated values in FRED's CSV, header and missing values left out."""
    out = []
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 2:
            continue
        try:
            out.append((dt.date.fromisoformat(r[0].strip()[:10]).isoformat(), float(r[1])))
        except ValueError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--series", default="GS3M")
    p.add_argument(
        "--out", default="binance-data", type=Path, help="folder (default ./binance-data)"
    )
    p.add_argument("--tries", default=4, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    url = f"{a.base_url.rstrip('/')}/graph/fredgraph.csv?id={a.series}"
    text = None
    for k in range(a.tries):
        try:
            req = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                text = r.read().decode()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if k == a.tries - 1:
                print(f"FAILED: {url}: {e}", file=sys.stderr)
                return 1
            time.sleep(2 ** (k + 1))
    got = rows(text or "")
    if not got:
        print(f"FAILED: {url} gave no dated values: {(text or '')[:200]!r}", file=sys.stderr)
        return 1
    path = a.out / "fred" / f"{a.series}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(text or "")
    tmp.replace(path)
    print(f"{a.series}: {len(got)} values, {got[0][0]} .. {got[-1][0]} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
