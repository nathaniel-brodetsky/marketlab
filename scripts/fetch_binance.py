#!/usr/bin/env python3
"""Download Binance's public archives (data.binance.vision), one file a day or a month,
each checked against the SHA-256 checksum Binance publishes next to it. Standard library
only, Python 3.9 or newer, any OS.

    python3 fetch_binance.py --start 2026-09-01 --end 2026-11-18
    python3 fetch_binance.py --kind trades --start 2026-11-01 --end 2026-11-07
    python3 fetch_binance.py --market um --symbol ETHUSDT --start 2026-11-01 --end 2026-11-03
    python3 fetch_binance.py --market um --kind klines --interval 1h --monthly \
        --start 2019-09-01 --end 2026-10-31

Files land in OUT/<market>/<kind>/<SYMBOL>/<SYMBOL>-<kind>-<date>.zip, klines in
OUT/<market>/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-<date>.zip, each next to its
.CHECKSUM; with --monthly, one file per month (<date> is YYYY-MM) for every month the
dates touch. Running it again fetches only what is missing or failed its check, so an
interrupted run is resumed by starting it again. A file Binance has not published yet
(usually today's, or this month's) is reported as missing and skipped. Exit code 1 if any
file failed.

Spot archives from 2025-01-01 on carry timestamps in microseconds, earlier ones in
milliseconds; the files are kept as Binance publishes them.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.binance.vision"
MARKETS = {"spot": "data/spot", "um": "data/futures/um", "cm": "data/futures/cm"}
KINDS = ("aggTrades", "trades", "klines")
INTERVALS = ("1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
CHUNK = 1 << 20
AGENT = {"User-Agent": "fetch_binance/1.0"}


class Missing(Exception):
    """The file is not published (HTTP 404)."""


def dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    return [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]


def months(start: dt.date, end: dt.date) -> list[str]:
    """Every month from `start`'s to `end`'s, as YYYY-MM."""
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def target(
    market: str, kind: str, symbol: str, period: str, interval: str | None, monthly: bool
) -> tuple[str, str, str]:
    """(the folder under the base URL, the folder under OUT, the file name)."""
    tag = interval if kind == "klines" else kind
    sub = f"{kind}/{symbol}/{interval}" if kind == "klines" else f"{kind}/{symbol}"
    freq = "monthly" if monthly else "daily"
    return f"{MARKETS[market]}/{freq}/{sub}", f"{market}/{sub}", f"{symbol}-{tag}-{period}.zip"


def _open(url: str, timeout: float):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Missing(url) from None
        raise


def retrying(fn, url: str, tries: int):
    """`fn(url)`, again after a network error, waiting 2, 4, 8 ... s; never after a 404."""
    for k in range(tries):
        try:
            return fn(url)
        except Missing:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 ** (k + 1))
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def expected_sha256(text: str, name: str) -> str:
    """The digest in a .CHECKSUM file: "<sha256>  <file name>"."""
    parts = text.split()
    if len(parts) < 1 or len(parts[0]) != 64:
        raise RuntimeError(f"{name}.CHECKSUM: unexpected content {text[:80]!r}")
    if len(parts) > 1 and parts[1].lstrip("*") != name:
        raise RuntimeError(f"{name}.CHECKSUM names {parts[1]!r}")
    return parts[0].lower()


def fetch_file(
    base: str,
    remote: str,
    local: str,
    name: str,
    label: str,
    out: Path,
    tries: int,
    timeout: float,
) -> tuple[str, str, int]:
    """One file and its checksum: (label, status, size)."""
    url = f"{base}/{remote}/{name}"
    folder = out / local
    folder.mkdir(parents=True, exist_ok=True)
    zpath, spath = folder / name, folder / f"{name}.CHECKSUM"

    if zpath.exists() and spath.exists():
        want = expected_sha256(spath.read_text(), name)
        if file_sha256(zpath) == want:
            return label, "have", zpath.stat().st_size

    try:
        text = retrying(lambda u: _open(u, timeout).read().decode(), url + ".CHECKSUM", tries)
    except Missing:
        return label, "missing", 0
    want = expected_sha256(text, name)

    part = folder / f"{name}.part"
    for attempt in range(2):  # a second try if the bytes do not match the checksum

        def download(u: str) -> str:
            h = hashlib.sha256()
            with _open(u, timeout) as r, part.open("wb") as f:
                while block := r.read(CHUNK):
                    f.write(block)
                    h.update(block)
            return h.hexdigest()

        try:
            got = retrying(download, url, tries)
        except Missing:
            part.unlink(missing_ok=True)
            return label, "missing", 0
        if got == want:
            break
        part.unlink(missing_ok=True)
        if attempt == 1:
            return label, "bad checksum", 0
    with zipfile.ZipFile(part) as z:
        if z.testzip() is not None:
            part.unlink(missing_ok=True)
            return label, "bad zip", 0
    part.replace(zpath)
    spath.write_text(text)
    return label, "ok", zpath.stat().st_size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--start", required=True, type=dt.date.fromisoformat, help="first day, YYYY-MM-DD"
    )
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="last day, YYYY-MM-DD")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--kind", default="aggTrades", choices=KINDS)
    p.add_argument("--interval", choices=INTERVALS, help="for --kind klines, e.g. 1h")
    p.add_argument("--monthly", action="store_true", help="one file a month instead of a day")
    p.add_argument(
        "--market", default="spot", choices=sorted(MARKETS), help="spot, um or cm futures"
    )
    p.add_argument(
        "--out", default="binance-data", type=Path, help="folder (default ./binance-data)"
    )
    p.add_argument("--jobs", default=4, type=int, help="files downloaded at once (default 4)")
    p.add_argument("--tries", default=5, type=int, help="attempts per file on network errors")
    p.add_argument(
        "--timeout", default=60.0, type=float, help="seconds without data before retrying"
    )
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help="only print the URLs")
    a = p.parse_args(argv)
    symbol = a.symbol.upper()
    if (a.kind == "klines") != (a.interval is not None):
        p.error("--interval goes with --kind klines, and klines need it")
    todo = months(a.start, a.end) if a.monthly else [d.isoformat() for d in dates(a.start, a.end)]
    base = a.base_url.rstrip("/")
    files = {d: target(a.market, a.kind, symbol, d, a.interval, a.monthly) for d in todo}

    if a.dry_run:
        for remote, _, name in files.values():
            print(f"{base}/{remote}/{name}")
        return 0

    what = f"{a.kind} {a.interval}" if a.interval else a.kind
    unit = "months" if a.monthly else "days"
    print(f"{symbol} {a.market} {what}: {len(todo)} {unit}, {todo[0]} .. {todo[-1]} -> {a.out}")
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {
            pool.submit(fetch_file, base, *files[d], d, a.out, a.tries, a.timeout): d for d in todo
        }
        for f in cf.as_completed(futures):
            d = futures[f]
            try:
                day, status, size = f.result()
            except Exception as e:  # report the file, keep the others going
                day, status, size = d, f"error: {e}", 0
            results.append((day, status, size))
            print(f"  {day}  {status:<12} {size / 1e6:8.1f} MB", flush=True)

    results.sort()
    good = [r for r in results if r[1] in ("ok", "have")]
    missing = [r[0] for r in results if r[1] == "missing"]
    failed = [r for r in results if r[1] not in ("ok", "have", "missing")]
    total = sum(r[2] for r in good) / 1e6
    print(f"done: {len(good)} of {len(todo)} {unit} on disk, {total:,.1f} MB")
    if missing:
        print(f"not published: {', '.join(missing)}")
    for day, status, _ in failed:
        print(f"FAILED {day}: {status}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
