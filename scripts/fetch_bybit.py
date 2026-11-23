#!/usr/bin/env python3
"""Download every USDT perpetual Bybit lists, trading or delisted, from its public v5 market
API (no key, no account): daily bars and funding history, in the files fetch_universe.py
writes for Binance, so that marketlab.xsection reads the folder as it reads binance-data.
Standard library only, Python 3.9 or newer, any OS.

    python3 fetch_bybit.py --start 2020-01-01 --end 2026-10-31 --out bybit-data

1. GET /v5/market/instruments-info, category linear, status Trading and then Closed, a
   cursor a page: every perpetual quoted in USDT, of either status, into OUT/um/universe.csv
   with the type of its underlying, and the closed ones on coins again into OUT/um/gone.csv,
   so that the coins Bybit has delisted stay in the universe. The type is Bybit's
   symbolType: none, or "innovation" (its zone for new coins), is a coin; the others it
   gives (stock, ETF, commodity, forex) are not. Stablecoins and the ETH/BTC ratio, which
   it types as coins, are named in NOT_COINS here.
2. GET /v5/market/kline, interval D, newest first, 1000 a request:
   OUT/um/klines1d/<SYMBOL>.csv with Binance's columns; quote_volume is Bybit's turnover,
   and trades, which Bybit does not give, is left empty. A perpetual Bybit serves no bars
   of is reported and left without a file. Bybit serves a delisted perpetual's bars from
   2024-02-05 on only (none before, as downloaded on 2026-11-23): its earlier years hold
   the coins that survived them.
3. GET /v5/market/funding/history, newest first, 200 a request:
   OUT/um/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate.csv (mark_price left empty).

Running it again asks only for what comes after the last row of each file. Requests are
--pause apart, far inside Bybit's 600 a five seconds; some 1,200 perpetuals take about an
hour, most of it the funding (some coins settle every hour). A symbol that fails is
reported and the others go on; exit code 1 if any failed.
"""

from __future__ import annotations

import argparse
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
_spec = importlib.util.spec_from_file_location("fetch_universe", HERE / "fetch_universe.py")
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)
ff = fu.ff

BASE = "https://api.bybit.com"
AGENT = {"User-Agent": "fetch_bybit/1.0"}
DAY = 86_400_000
BAR_PAGE, FUNDING_PAGE = 1000, 200
GONE_COLUMNS = ("symbol", "status", "onboard_date", "delivery_date", "base", "quote")
RETRY = {10002, 10006, 10016}  # request expired, too many visits, server error
INVALID = 10001  # params error: what Bybit says of a symbol it does not serve
COINS = ("", "innovation")  # Bybit's symbolType of a coin's perpetual
# what Bybit types as a coin and is not a coin's own price (named as of 2026-11-23)
NOT_COINS = {"USDCUSDT": "STABLE", "USDEUSDT": "STABLE", "USD1USDT": "STABLE",
             "RLUSDUSDT": "STABLE", "FDUSDUSDT": "STABLE", "BUSDUSDT": "STABLE",
             "USTUSDT": "STABLE", "ETHBTCUSDT": "INDEX", "PAXGUSDT": "COMMODITY",
             "XAUTUSDT": "COMMODITY"}  # fmt: skip


class NotServed(RuntimeError):
    pass


def call(base: str, path: str, tries: int, timeout: float, **q) -> dict:
    """The `result` of a v5 GET; again after a network error, a rate limit or a server error,
    waiting 2, 4, 8 ... s. NotServed for a symbol Bybit does not know."""
    url = f"{base}{path}?{urllib.parse.urlencode(q)}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            code = d.get("retCode")
            if code == 0:
                return d["result"]
            msg = f"{url}: retCode {code} {d.get('retMsg')}"
            if code == INVALID and "symbol" in str(d.get("retMsg")).lower():
                raise NotServed(msg)
            if code not in RETRY:
                raise RuntimeError(msg)
            err: Exception = RuntimeError(msg)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429) and e.code < 500:
                raise RuntimeError(f"{url}: HTTP {e.code} {e.read()[:200]!r}") from None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        if k == tries - 1:
            raise RuntimeError(f"{url}: {err}")
        time.sleep(2 ** (k + 1))
    raise AssertionError("unreachable")


def underlying(s: dict) -> str:
    """COIN, or the type of what else a perpetual is on: Bybit's symbolType, in capitals, or
    NOT_COINS's."""
    if s["symbol"] in NOT_COINS:
        return NOT_COINS[s["symbol"]]
    kind = s.get("symbolType") or ""
    return "COIN" if kind in COINS else kind.upper()


def instruments(base: str, quote: str, tries: int, timeout: float) -> list[dict]:
    """Every perpetual quoted in `quote` Bybit lists as trading or closed; one trading under
    a name also closed earlier is kept as trading."""
    out: dict[str, dict] = {}
    for status in ("Closed", "Trading"):  # the trading one written over the closed
        cursor = ""
        while True:
            q = {"category": "linear", "status": status, "limit": 1000}
            if cursor:
                q["cursor"] = cursor
            res = call(base, "/v5/market/instruments-info", tries, timeout, **q)
            for s in res.get("list", []):
                if (s.get("contractType") == "LinearPerpetual" and s.get("quoteCoin") == quote
                        and s.get("status") == status):  # fmt: skip
                    out[s["symbol"]] = {
                        "symbol": s["symbol"], "status": status,
                        "onboard_date": int(s.get("launchTime") or 0),
                        "delivery_date": int(s.get("deliveryTime") or 0),
                        "base": s.get("baseCoin", ""), "quote": quote,
                        "underlying_type": underlying(s),
                    }  # fmt: skip
            cursor = res.get("nextPageCursor") or ""
            if not cursor or not res.get("list"):
                break
    return [out[k] for k in sorted(out)]


def daily_bars(base: str, symbol: str, start: int, end: int, tries: int, timeout: float,
               pause: float) -> list[list[str]]:  # fmt: skip
    """Every daily bar opening in [start, end], oldest first."""
    got: dict[int, list[str]] = {}
    hi = end
    while hi >= start:
        res = call(base, "/v5/market/kline", tries, timeout, category="linear", symbol=symbol,
                   interval="D", start=start, end=hi, limit=BAR_PAGE)  # fmt: skip
        rows = res.get("list", [])
        for r in rows:
            t = int(r[0])
            if start <= t <= end:
                got[t] = [str(t), r[1], r[2], r[3], r[4], r[5], r[6], ""]
        if len(rows) < BAR_PAGE:
            break
        oldest = min(int(r[0]) for r in rows)
        if max(int(r[0]) for r in rows) > hi:
            raise RuntimeError(f"{symbol}: a page of bars after its end {hi}")
        hi = oldest - 1
        time.sleep(pause)
    return [got[t] for t in sorted(got)]


def funding(base: str, symbol: str, start: int, end: int, tries: int, timeout: float,
            pause: float) -> list[tuple[int, str, str]]:  # fmt: skip
    """Every settlement in [start, end], oldest first: pages back from `end` until one is
    empty or reaches `start`."""
    got: dict[int, tuple[int, str, str]] = {}
    hi = end
    while True:
        res = call(base, "/v5/market/funding/history", tries, timeout, category="linear",
                   symbol=symbol, endTime=hi, limit=FUNDING_PAGE)  # fmt: skip
        rows = res.get("list", [])
        if not rows:
            break
        oldest = hi
        for r in rows:
            t = int(r["fundingRateTimestamp"])
            oldest = min(oldest, t)
            if start <= t <= end:
                got[t] = (t, str(r["fundingRate"]), "")
        if max(int(r["fundingRateTimestamp"]) for r in rows) > hi:
            raise RuntimeError(f"{symbol}: a page of settlements after its end {hi}")
        if oldest <= start:
            break
        hi = oldest - 1
        time.sleep(pause)
    return [got[t] for t in sorted(got)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD, included")
    p.add_argument("--quote", default="USDT")
    p.add_argument("--out", default="bybit-data", type=Path, help="folder (default ./bybit-data)")
    p.add_argument("--symbols", nargs="*", help="only these (default: every perpetual)")
    p.add_argument("--list-only", action="store_true",
                   help="write universe.csv and gone.csv, download nothing else")  # fmt: skip
    p.add_argument("--tries", default=5, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--pause", default=0.1, type=float, help="seconds between requests")
    p.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.end < a.start:
        raise SystemExit(f"--end {a.end} is before --start {a.start}")
    today = dt.datetime.now(ff.UTC).date()
    if a.end >= today:  # Bybit serves the day's bar while it is open: it would stay so
        raise SystemExit(f"--end {a.end}: the daily bar of {today} (UTC) is not closed yet")
    base = a.base_url.rstrip("/")
    lo, hi = ff.ms(a.start), ff.ms(a.end + dt.timedelta(days=1)) - 1
    try:
        syms = instruments(base, a.quote, a.tries, a.timeout)
    except RuntimeError as e:
        print(f"FAILED: instruments-info: {e}", file=sys.stderr)
        return 1
    if a.symbols:
        wanted = {s.upper() for s in a.symbols}
        syms = [s for s in syms if s["symbol"] in wanted]
    um = a.out / "um"
    closed = [s for s in syms if s["status"] == "Closed" and s["underlying_type"] == "COIN"]
    fu.write_rows(um / "universe.csv", fu.UNIVERSE_COLUMNS,
                  [[s[c] for c in fu.UNIVERSE_COLUMNS] for s in syms])  # fmt: skip
    fu.write_rows(um / "gone.csv", GONE_COLUMNS, [[s[c] for c in GONE_COLUMNS] for s in closed])
    print(f"{len(syms)} {a.quote} perpetuals on Bybit: {fu._counts(syms, 'status')}; "
          f"underlying {fu._counts(syms, 'underlying_type')}", flush=True)  # fmt: skip
    if a.list_only:
        print(f"the lists -> {um}")
        return 0
    failed, unserved = [], []
    for n, s in enumerate(syms, 1):
        sym, tag = s["symbol"], f"  [{n}/{len(syms)}] {s['symbol']:<20} {s['status']:<8}"
        try:
            path = um / "klines1d" / f"{sym}.csv"
            have = fu.read_rows(path, fu.KLINE_COLUMNS)
            start = max(lo, int(have[-1][0]) + DAY) if have else lo
            new = []
            if start <= hi:
                new = daily_bars(base, sym, start, hi, a.tries, a.timeout, a.pause)
            rows = sorted({int(r[0]): r for r in [*have, *new]}.values(), key=lambda r: int(r[0]))
            if not rows:
                unserved.append(sym)
                print(f"{tag} no bars served", flush=True)
                time.sleep(a.pause)
                continue
            fu.write_rows(path, fu.KLINE_COLUMNS, rows)
            fpath = um / "fundingRate" / sym / f"{sym}-fundingRate.csv"
            fhave = ff.read(fpath)
            fstart = max(lo, fhave[-1][0] + 1) if fhave else lo
            time.sleep(a.pause)
            fnew = []
            if fstart <= hi:
                fnew = funding(base, sym, fstart, hi, a.tries, a.timeout, a.pause)
            frows = sorted({r[0]: r for r in [*fhave, *fnew]}.values())
            ff.write(fpath, frows)
            first = dt.datetime.fromtimestamp(int(rows[0][0]) / 1000, ff.UTC).date()
            last = dt.datetime.fromtimestamp(int(rows[-1][0]) / 1000, ff.UTC).date()
            print(f"{tag} {len(rows)} days {first} .. {last}, {len(frows)} settlements",
                  flush=True)  # fmt: skip
            time.sleep(a.pause)
        except NotServed:
            unserved.append(sym)
            print(f"{tag} no data served", flush=True)
        except (RuntimeError, OSError) as e:
            failed.append(sym)
            print(f"{tag} FAILED: {e}", file=sys.stderr, flush=True)
    gone = {s["symbol"] for s in closed}
    print(f"done: {len(syms) - len(failed) - len(unserved)} of {len(syms)} perpetuals -> {um}; "
          f"no bars served for {len(unserved)} ({len(gone & set(unserved))} of the "
          f"{len(gone)} delisted coins)")  # fmt: skip
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
