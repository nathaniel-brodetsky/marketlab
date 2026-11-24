#!/usr/bin/env python3
"""Download Binance's daily positioning files of USD-M perpetuals (open interest and the
long/short ratios: data/futures/um/daily/metrics on data.binance.vision), each checked
against its checksum, for the (symbol, day) pairs in a file. Standard library only,
Python 3.9 or newer, any OS.

    marketlab metrics days --data binance-data --from 2021-12-01 --to 2026-10-31 \\
        --out runs/metrics-days.csv
    python3 fetch_metrics.py --days-file runs/metrics-days.csv --out binance-data

Files land in OUT/um/metrics/<SYMBOL>/<SYMBOL>-metrics-<DAY>.zip with their .CHECKSUM;
a file the archive does not have (a coin before Binance kept them, a day not published yet)
is counted as missing. Running it again fetches nothing twice.

--live --close DAY: for the pairs of that day and the two before it the archive does not
have yet, all accounts' long/short ratio from Binance's public API (GET
/futures/data/globalLongShortAccountRatio, 5-minute periods: the last one before the close),
into OUT/um/metrics/<SYMBOL>/live.csv (day, ratio, when it was taken): what the shadow
decides from at a close whose file is not out. A hypothesis's runs read the archive only.
Exit code 1 if anything failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fetch_binance", HERE / "fetch_binance.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

REMOTE = "data/futures/um/daily/metrics"
API = "https://fapi.binance.com"
DAY = 86_400_000
LIVE_COLUMNS = ("day", "count_long_short_ratio", "taken_at")
UTC = dt.timezone.utc  # noqa: UP017  (dt.UTC needs Python 3.11)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as f:
        r = csv.reader(f)
        if tuple(next(r, ())) != ("symbol", "day"):
            raise SystemExit(f"{path}: not a days file (symbol,day)")
        return [(s, d) for s, d in r]


def fetch_one(base: str, out: Path, sym: str, day: str, tries: int, timeout: float) -> str:
    name = f"{sym}-metrics-{day}.zip"
    return fb.fetch_file(base, f"{REMOTE}/{sym}", f"um/metrics/{sym}", name, name, out, tries,
                         timeout)[1]  # fmt: skip


def close_ms(day: str) -> int:
    d = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) * 1000


def live_ratio(api: str, sym: str, day: str, tries: int, timeout: float) -> float | None:
    """All accounts' long/short ratio of the last 5-minute period before `day`'s close."""
    end = close_ms(day)
    q = urllib.parse.urlencode({"symbol": sym, "period": "5m", "startTime": end - 30 * 60_000,
                                "endTime": end - 1, "limit": 30})  # fmt: skip
    url = f"{api}/futures/data/globalLongShortAccountRatio?{q}"
    try:
        rows = json.loads(fb.retrying(lambda u: fb._open(u, timeout).read().decode(), url, tries))
    except fb.Missing:
        return None
    if not isinstance(rows, list):
        raise RuntimeError(f"{url}: unexpected answer {str(rows)[:200]}")
    rows = [r for r in rows if int(r["timestamp"]) < end]
    return float(max(rows, key=lambda r: int(r["timestamp"]))["longShortRatio"]) if rows else None


def write_live(out: Path, sym: str, day: str, ratio: float) -> None:
    f = out / "um" / "metrics" / sym / "live.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if f.exists():
        with f.open(newline="") as fh:
            r = csv.reader(fh)
            if tuple(next(r, ())) != LIVE_COLUMNS:
                raise SystemExit(f"{f}: not a file this script wrote (columns {LIVE_COLUMNS})")
            rows = {row[0]: row for row in r}
    now = dt.datetime.now(UTC).isoformat(timespec="seconds")
    rows.setdefault(day, [day, repr(ratio), now])  # a close taken once stays as it was
    tmp = f.with_suffix(".part")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LIVE_COLUMNS)
        w.writerows(rows[k] for k in sorted(rows))
    tmp.replace(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--days-file", required=True, type=Path, help="symbol,day a line")
    p.add_argument("--out", default="binance-data", type=Path, help="default ./binance-data")
    p.add_argument("--jobs", default=8, type=int, help="files downloaded at once (default 8)")
    p.add_argument("--tries", default=5, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--live", action="store_true",
                   help="the API's ratio where the archive has no file yet")  # fmt: skip
    p.add_argument("--close", type=dt.date.fromisoformat, help="with --live: the close, YYYY-MM-DD")
    p.add_argument("--base-url", default=fb.BASE, help=argparse.SUPPRESS)
    p.add_argument("--api-url", default=API, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.live and not a.close:
        raise SystemExit("--live asks for --close")
    pairs = read_pairs(a.days_file)
    base = a.base_url.rstrip("/")
    status: dict[str, int] = {}
    failed: list[str] = []
    missing: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(fetch_one, base, a.out, s, d, a.tries, a.timeout): (s, d)
                   for s, d in pairs}  # fmt: skip
        for k, fut in enumerate(cf.as_completed(futures), 1):
            s, d = futures[fut]
            try:
                st = fut.result()
            except (RuntimeError, OSError) as e:
                st = "failed"
                failed.append(f"{s} {d}: {e}")
            status[st] = status.get(st, 0) + 1
            if st == "missing":
                missing.append((s, d))
            elif st not in ("ok", "have"):
                failed.append(f"{s} {d}: {st}")
            if k % 2000 == 0 or k == len(pairs):
                print(f"  {k}/{len(pairs)}: {dict(sorted(status.items()))}", flush=True)
    if a.live:
        recent = {(a.close - dt.timedelta(days=k)).isoformat() for k in range(3)}
        taken = 0
        for s, d in sorted(missing):
            if d not in recent:
                continue
            try:
                ratio = live_ratio(a.api_url.rstrip("/"), s, d, a.tries, a.timeout)
            except (RuntimeError, OSError) as e:
                failed.append(f"{s} {d} live: {e}")
                continue
            if ratio is not None:
                write_live(a.out, s, d, ratio)
                taken += 1
        print(f"  live: the API's ratio at the close for {taken} pairs the archive does not have")
    print(f"done: {len(pairs)} pairs -> {a.out / 'um' / 'metrics'}")
    if failed:
        print("FAILED: " + "; ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""),
              file=sys.stderr)  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
