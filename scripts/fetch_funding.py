#!/usr/bin/env python3
"""Download the funding-rate history of a Binance USD-M perpetual from Binance's public
market-data API (GET /fapi/v1/fundingRate: no key, no account), into one CSV. Standard
library only, Python 3.9 or newer, any OS.

    python3 fetch_funding.py --start 2019-09-01 --end 2026-10-31

The file is OUT/um/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate.csv with the columns
funding_time (ms since 1970, UTC), funding_rate (a fraction: 0.0001 is 0.01%) and
mark_price (empty where Binance gives none). Each settlement once, in time order. Running
it again keeps what is there and asks only for what comes after its last row, or before
its first. A thousand settlements a request, a quarter of a second apart: about eight
requests for seven years of settlements every eight hours, far inside the endpoint's limit
of 500 requests in five minutes. Exit code 1 if a request failed after its retries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://fapi.binance.com"
PATH = "/fapi/v1/fundingRate"
LIMIT = 1000
AGENT = {"User-Agent": "fetch_funding/1.0"}
COLUMNS = ("funding_time", "funding_rate", "mark_price")
UTC = dt.timezone.utc  # noqa: UP017  (dt.UTC needs Python 3.11)


def ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1000


def request(base: str, symbol: str, start: int, end: int, tries: int, timeout: float) -> list:
    """One page, oldest first; again after a network error or a rate limit, waiting 2, 4,
    8 ... s, or as long as Binance asks."""
    q = urllib.parse.urlencode({"symbol": symbol, "startTime": start, "endTime": end,
                                "limit": LIMIT})  # fmt: skip
    url = f"{base}{PATH}?{q}"
    for k in range(tries):
        wait = 2 ** (k + 1)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT),
                                        timeout=timeout) as r:  # fmt: skip
                rows = json.loads(r.read().decode())
            if not isinstance(rows, list):
                raise RuntimeError(f"{url}: unexpected answer {str(rows)[:200]}")
            return rows
        except urllib.error.HTTPError as e:
            if e.code not in (418, 429) and e.code < 500:
                raise RuntimeError(f"{url}: HTTP {e.code} {e.read()[:200]!r}") from None
            wait = max(wait, int(e.headers.get("Retry-After") or 0))
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        if k == tries - 1:
            raise RuntimeError(f"{url}: {err}")
        time.sleep(wait)
    raise AssertionError("unreachable")


def fetch(base: str, symbol: str, start: int, end: int, tries: int, timeout: float,
          pause: float) -> list[tuple[int, str, str]]:  # fmt: skip
    """Every settlement in [start, end], a page at a time."""
    out: list[tuple[int, str, str]] = []
    while start <= end:
        rows = request(base, symbol, start, end, tries, timeout)
        page = []
        for r in rows:
            t = int(r["fundingTime"])
            if (out and t <= out[-1][0]) or (page and t <= page[-1][0]):
                raise RuntimeError(f"settlements out of order at {t}")
            page.append((t, str(r["fundingRate"]), str(r.get("markPrice") or "")))
        out += [x for x in page if start <= x[0] <= end]
        if len(rows) < LIMIT:
            break
        start = page[-1][0] + 1
        time.sleep(pause)
    return out


def read(path: Path) -> list[tuple[int, str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.reader(f)
        if tuple(next(r, ())) != COLUMNS:
            raise SystemExit(f"{path}: not a file this script wrote (columns {COLUMNS})")
        return [(int(t), rate, mark) for t, rate, mark in r]


def write(path: Path, rows: list[tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD, included")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--out", default="binance-data", type=Path, help="folder (default ./binance-data)"
    )
    p.add_argument("--tries", default=5, type=int, help="attempts per request")
    p.add_argument("--timeout", default=30.0, type=float, help="seconds a request may take")
    p.add_argument("--pause", default=0.25, type=float, help="seconds between requests")
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.end < a.start:
        raise SystemExit(f"--end {a.end} is before --start {a.start}")
    symbol = a.symbol.upper()
    path = a.out / "um" / "fundingRate" / symbol / f"{symbol}-fundingRate.csv"
    have = read(path)
    lo, hi = ms(a.start), ms(a.end + dt.timedelta(days=1)) - 1
    spans = [(lo, hi)]
    if have:  # only what is not there yet
        spans = [(lo, min(hi, have[0][0] - 1)), (max(lo, have[-1][0] + 1), hi)]
    try:
        new = []
        for s, e in spans:
            if s <= e:
                new += fetch(a.base_url.rstrip("/"), symbol, s, e, a.tries, a.timeout, a.pause)
    except RuntimeError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    rows = sorted({r[0]: r for r in [*have, *new]}.values())
    write(path, rows)
    if not rows:
        print(f"{symbol}: no settlements between {a.start} and {a.end}")
        return 0
    gaps: dict[float, int] = {}
    for k in range(1, len(rows)):  # (itertools.pairwise needs Python 3.10)
        h = round((rows[k][0] - rows[k - 1][0]) / 3_600_000, 2)
        gaps[h] = gaps.get(h, 0) + 1
    first = dt.datetime.fromtimestamp(rows[0][0] / 1000, UTC)
    last = dt.datetime.fromtimestamp(rows[-1][0] / 1000, UTC)
    spacing = ", ".join(f"{h:g} h x{n}" for h, n in sorted(gaps.items(), key=lambda x: -x[1])[:5])
    print(f"{symbol}: {len(rows)} settlements ({len(new)} new), {first:%Y-%m-%d %H:%M} .. "
          f"{last:%Y-%m-%d %H:%M} UTC -> {path}")  # fmt: skip
    print(f"spacing     {spacing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
