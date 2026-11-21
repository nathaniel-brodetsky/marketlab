#!/usr/bin/env python3
"""Download every USD-M perpetual's daily bars and funding history from Binance's public
market-data API (no key, no account). Standard library only, Python 3.9 or newer, any OS.

    python3 fetch_universe.py --start 2020-01-01 --end 2026-10-31

1. GET /fapi/v1/exchangeInfo: every perpetual quoted in USDT, whatever its status now (one
   being settled or closed is kept while Binance still lists it, so the universe is less
   biased towards the coins that survived), into OUT/um/universe.csv.
2. GET /fapi/v1/klines, interval 1d, 1500 days a request: OUT/um/klines1d/<SYMBOL>.csv
   with open_time (ms, UTC), open, high, low, close, volume, quote_volume, trades.
3. GET /fapi/v1/fundingRate, as scripts/fetch_funding.py does:
   OUT/um/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate.csv.

Running it again asks only for what is missing. Requests are spaced to stay inside
Binance's limits (klines by weight, funding 500 requests in five minutes); a few hundred
perpetuals take some twenty minutes, most of it the funding. A symbol that fails is
reported and the others go on; exit code 1 if any failed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fetch_funding", HERE / "fetch_funding.py")
ff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ff)

BASE = "https://fapi.binance.com"
AGENT = {"User-Agent": "fetch_universe/1.0"}
DAY = 86_400_000
KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades")
UNIVERSE_COLUMNS = ("symbol", "status", "onboard_date", "base", "quote")


def get(url: str, tries: int, timeout: float):
    """JSON from `url`; again after a network error or a rate limit, waiting 2, 4, 8 ... s,
    or as long as Binance asks."""
    for k in range(tries):
        wait = 2 ** (k + 1)
        try:
            req = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
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


def universe(base: str, quote: str, tries: int, timeout: float) -> list[dict]:
    info = get(f"{base}/fapi/v1/exchangeInfo", tries, timeout)
    out = []
    for s in info.get("symbols", []):
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == quote:
            out.append({"symbol": s["symbol"], "status": s.get("status", ""),
                        "onboard_date": int(s.get("onboardDate") or 0),
                        "base": s.get("baseAsset", ""), "quote": quote})  # fmt: skip
    return sorted(out, key=lambda x: x["symbol"])


def read_rows(path: Path, columns: tuple[str, ...]) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.reader(f)
        if tuple(next(r, ())) != columns:
            raise SystemExit(f"{path}: not a file this script wrote (columns {columns})")
        return [row for row in r]


def write_rows(path: Path, columns: tuple[str, ...], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    tmp.replace(path)


def daily_bars(base: str, symbol: str, start: int, end: int, tries: int, timeout: float,
               pause: float) -> list[list[str]]:  # fmt: skip
    """Every daily bar opening in [start, end], 1500 a request."""
    out: list[list[str]] = []
    while start <= end:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": "1d", "startTime": start,
                                    "endTime": end, "limit": 1500})  # fmt: skip
        rows = get(f"{base}/fapi/v1/klines?{q}", tries, timeout)
        if not isinstance(rows, list):
            raise RuntimeError(f"{symbol}: unexpected answer {str(rows)[:200]}")
        for r in rows:
            t = int(r[0])
            if out and t <= int(out[-1][0]):
                raise RuntimeError(f"{symbol}: bars out of order at {t}")
            out.append([str(t), r[1], r[2], r[3], r[4], r[5], r[7], str(r[8])])
        if len(rows) < 1500:
            break
        start = int(rows[-1][0]) + DAY
        time.sleep(pause)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD, included")
    p.add_argument("--quote", default="USDT")
    p.add_argument(
        "--out", default="binance-data", type=Path, help="folder (default ./binance-data)"
    )
    p.add_argument("--symbols", nargs="*", help="only these (default: every perpetual)")
    p.add_argument("--no-funding", action="store_true", help="daily bars only")
    p.add_argument("--tries", default=5, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--pause", default=0.3, type=float, help="seconds between bar requests")
    p.add_argument(
        "--funding-pause", default=0.65, type=float, help="seconds between funding requests"
    )
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.end < a.start:
        raise SystemExit(f"--end {a.end} is before --start {a.start}")
    base = a.base_url.rstrip("/")
    lo, hi = ff.ms(a.start), ff.ms(a.end + dt.timedelta(days=1)) - 1
    try:
        syms = universe(base, a.quote, a.tries, a.timeout)
    except RuntimeError as e:
        print(f"FAILED: exchangeInfo: {e}", file=sys.stderr)
        return 1
    if a.symbols:
        wanted = {s.upper() for s in a.symbols}
        syms = [s for s in syms if s["symbol"] in wanted]
    write_rows(a.out / "um" / "universe.csv", UNIVERSE_COLUMNS,
               [[s[c] for c in UNIVERSE_COLUMNS] for s in syms])  # fmt: skip
    statuses: dict[str, int] = {}
    for s in syms:
        statuses[s["status"]] = statuses.get(s["status"], 0) + 1
    print(f"{len(syms)} {a.quote} perpetuals: "
          + ", ".join(f"{k or '?'} {v}" for k, v in sorted(statuses.items())))  # fmt: skip
    failed = []
    for n, s in enumerate(syms, 1):
        sym = s["symbol"]
        try:
            path = a.out / "um" / "klines1d" / f"{sym}.csv"
            have = read_rows(path, KLINE_COLUMNS)
            start = max(lo, int(have[-1][0]) + DAY) if have else lo
            new = []
            if start <= hi:
                new = daily_bars(base, sym, start, hi, a.tries, a.timeout, a.pause)
            rows = sorted({int(r[0]): r for r in [*have, *new]}.values(), key=lambda r: int(r[0]))
            write_rows(path, KLINE_COLUMNS, rows)
            got = f"{len(rows)} days"
            if not a.no_funding:
                fpath = a.out / "um" / "fundingRate" / sym / f"{sym}-fundingRate.csv"
                fhave = ff.read(fpath)
                fstart = max(lo, fhave[-1][0] + 1) if fhave else lo
                fnew = []
                if fstart <= hi:
                    time.sleep(a.funding_pause)  # the funding limit counts requests, not weight
                    fnew = ff.fetch(base, sym, fstart, hi, a.tries, a.timeout, a.funding_pause)
                frows = sorted({r[0]: r for r in [*fhave, *fnew]}.values())
                ff.write(fpath, frows)
                got += f", {len(frows)} settlements"
            print(f"  [{n}/{len(syms)}] {sym:<16} {s['status']:<10} {got}", flush=True)
            time.sleep(a.pause)
        except (RuntimeError, OSError) as e:
            if "-1121" in str(e) or "Invalid symbol" in str(e):  # listed, but no data served
                print(f"  [{n}/{len(syms)}] {sym:<16} {s['status']:<10} no data served", flush=True)
                continue
            failed.append(sym)
            print(f"  [{n}/{len(syms)}] {sym:<16} FAILED: {e}", file=sys.stderr, flush=True)
    print(f"done: {len(syms) - len(failed)} of {len(syms)} perpetuals -> {a.out / 'um'}")
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
