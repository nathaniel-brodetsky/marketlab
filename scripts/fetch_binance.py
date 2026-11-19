#!/usr/bin/env python3
"""Download Binance's public trade archives (data.binance.vision), one file a day, each
checked against the SHA-256 checksum Binance publishes next to it. Standard library only,
Python 3.9 or newer, any OS.

    python3 fetch_binance.py --start 2026-09-01 --end 2026-11-18
    python3 fetch_binance.py --kind trades --start 2026-11-01 --end 2026-11-07
    python3 fetch_binance.py --market um --symbol ETHUSDT --start 2026-11-01 --end 2026-11-03

Files land in OUT/<market>/<kind>/<SYMBOL>/<SYMBOL>-<kind>-<date>.zip, each next to its
.CHECKSUM. Running it again fetches only what is missing or failed its check, so an
interrupted run is resumed by starting it again. A day Binance has not published yet
(usually today) is reported as missing and skipped. Exit code 1 if any day failed.

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
MARKETS = {"spot": "data/spot/daily", "um": "data/futures/um/daily", "cm": "data/futures/cm/daily"}
KINDS = ("aggTrades", "trades")
CHUNK = 1 << 20
AGENT = {"User-Agent": "fetch_binance/1.0"}


class Missing(Exception):
    """The file is not published (HTTP 404)."""


def dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    return [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]


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


def fetch_day(
    base: str,
    market: str,
    kind: str,
    symbol: str,
    day: dt.date,
    out: Path,
    tries: int,
    timeout: float,
) -> tuple[str, str, int]:
    name = f"{symbol}-{kind}-{day.isoformat()}.zip"
    url = f"{base}/{MARKETS[market]}/{kind}/{symbol}/{name}"
    folder = out / market / kind / symbol
    folder.mkdir(parents=True, exist_ok=True)
    zpath, spath = folder / name, folder / f"{name}.CHECKSUM"

    if zpath.exists() and spath.exists():
        want = expected_sha256(spath.read_text(), name)
        if file_sha256(zpath) == want:
            return day.isoformat(), "have", zpath.stat().st_size

    try:
        text = retrying(lambda u: _open(u, timeout).read().decode(), url + ".CHECKSUM", tries)
    except Missing:
        return day.isoformat(), "missing", 0
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
            return day.isoformat(), "missing", 0
        if got == want:
            break
        part.unlink(missing_ok=True)
        if attempt == 1:
            return day.isoformat(), "bad checksum", 0
    with zipfile.ZipFile(part) as z:
        if z.testzip() is not None:
            part.unlink(missing_ok=True)
            return day.isoformat(), "bad zip", 0
    part.replace(zpath)
    spath.write_text(text)
    return day.isoformat(), "ok", zpath.stat().st_size


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
    p.add_argument(
        "--market", default="spot", choices=sorted(MARKETS), help="spot, um or cm futures"
    )
    p.add_argument(
        "--out", default="binance-data", type=Path, help="folder (default ./binance-data)"
    )
    p.add_argument("--jobs", default=4, type=int, help="days downloaded at once (default 4)")
    p.add_argument("--tries", default=5, type=int, help="attempts per file on network errors")
    p.add_argument(
        "--timeout", default=60.0, type=float, help="seconds without data before retrying"
    )
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help="only print the URLs")
    a = p.parse_args(argv)
    symbol = a.symbol.upper()
    todo = dates(a.start, a.end)

    if a.dry_run:
        for d in todo:
            print(f"{a.base_url}/{MARKETS[a.market]}/{a.kind}/{symbol}/{symbol}-{a.kind}-{d}.zip")
        return 0

    print(f"{symbol} {a.market} {a.kind}: {len(todo)} days, {todo[0]} .. {todo[-1]} -> {a.out}")
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {
            pool.submit(
                fetch_day,
                a.base_url.rstrip("/"),
                a.market,
                a.kind,
                symbol,
                d,
                a.out,
                a.tries,
                a.timeout,
            ): d
            for d in todo
        }
        for f in cf.as_completed(futures):
            d = futures[f]
            try:
                day, status, size = f.result()
            except Exception as e:  # report the day, keep the others going
                day, status, size = d.isoformat(), f"error: {e}", 0
            results.append((day, status, size))
            print(f"  {day}  {status:<12} {size / 1e6:8.1f} MB", flush=True)

    results.sort()
    good = [r for r in results if r[1] in ("ok", "have")]
    missing = [r[0] for r in results if r[1] == "missing"]
    failed = [r for r in results if r[1] not in ("ok", "have", "missing")]
    total = sum(r[2] for r in good) / 1e6
    print(f"done: {len(good)} of {len(todo)} days on disk, {total:,.1f} MB")
    if missing:
        print(f"not published: {', '.join(missing)}")
    for day, status, _ in failed:
        print(f"FAILED {day}: {status}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
