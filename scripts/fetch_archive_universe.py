#!/usr/bin/env python3
"""Fill in, from Binance's public archive (data.binance.vision), what its API no longer
serves: the daily bars of the USD-M perpetuals gone from its list, and of listed ones whose
earlier listing's bars it no longer serves; and the funding of the gone ones. Standard
library only, Python 3.9 or newer, any OS. Run it after fetch_universe.py, on its folder:

    python3 fetch_archive_universe.py --start 2020-01-01 --end 2026-10-31

1. The archive's perpetuals: an S3 listing of data/futures/um/monthly/klines/, those
   quoted in USDT and without a delivery date.
2. Which to fill in: those in neither OUT/um/universe.csv nor OUT/um/contracts.csv (gone
   from Binance's list; one it lists as a contract of another type is left out), and
   those in universe.csv that settled funding on a day with no bar in
   OUT/um/klines1d/<SYMBOL>.csv. One not in the list with a bar on the listed perpetuals'
   last day is still trading, so not gone, wherever Binance lists it: left out too.
3. For each, the monthly 1d klines the archive has in the dates (each checked against its
   checksum, kept under OUT/um/klines/<SYMBOL>/1d/), merged into klines1d/<SYMBOL>.csv: a
   day already there is kept as it is.
4. For a gone one with no funding file: the API's funding history; where it serves none,
   the archive's monthly fundingRate files (calc_time, funding_interval_hours,
   last_funding_rate). The gone ones are listed in OUT/um/gone.csv.

The S3 listing's address and the funding files' columns are as Binance published them when
this was written; a listing or a file that does not match is reported, not guessed at.
Running it again downloads nothing twice. Exit code 1 if anything failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import importlib.util
import io
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fb = _load("fetch_binance")
ff = _load("fetch_funding")
fu = _load("fetch_universe")

LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINES = "data/futures/um/monthly/klines/"
FUNDING = "data/futures/um/monthly/fundingRate/"
DAY = 86_400_000
GONE_COLUMNS = ("symbol", "first_day", "last_day", "days", "settlements", "funding_from")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def listing(list_url: str, prefix: str, delimiter: str, tries: int,
            timeout: float) -> tuple[list[str], list[str]]:  # fmt: skip
    """(the common prefixes, the keys) under `prefix`, every page of the listing."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        q = {"prefix": prefix, "marker": marker}
        if delimiter:
            q["delimiter"] = delimiter
        url = f"{list_url}?{urllib.parse.urlencode(q)}"
        body = fb.retrying(lambda u: fb._open(u, timeout).read(), url, tries)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as e:
            raise RuntimeError(f"{url}: not an S3 listing ({e})") from None
        if _local(root.tag) != "ListBucketResult":
            raise RuntimeError(f"{url}: not an S3 listing ({_local(root.tag)})")
        page_p = [c.text or "" for el in root if _local(el.tag) == "CommonPrefixes"
                  for c in el if _local(c.tag) == "Prefix"]  # fmt: skip
        page_k = [c.text or "" for el in root if _local(el.tag) == "Contents"
                  for c in el if _local(c.tag) == "Key"]  # fmt: skip
        prefixes += page_p
        keys += page_k
        flags = {_local(el.tag): (el.text or "") for el in root}
        if flags.get("IsTruncated", "false").lower() != "true":
            return prefixes, keys
        nxt = flags.get("NextMarker") or max(page_p + page_k, default="")
        if nxt <= marker:
            raise RuntimeError(f"{url}: the listing does not advance past {marker!r}")
        marker = nxt


def perpetuals(prefixes: list[str]) -> list[str]:
    syms = [p[len(KLINES) :].rstrip("/") for p in prefixes if p.startswith(KLINES)]
    return sorted(s for s in syms if s.endswith("USDT"))  # BTCUSDT_230331 is a delivery


def _day(t: int) -> int:
    return t // DAY


def _date(t: str) -> str:
    return (dt.date(1970, 1, 1) + dt.timedelta(days=_day(int(t)))).isoformat()


def _last_day(out: Path, sym: str) -> str:
    """The day of the last bar in klines1d/<sym>.csv, or "" without one."""
    path = out / "um" / "klines1d" / f"{sym}.csv"
    if not path.exists():
        return ""
    lines = [x for x in path.read_text().splitlines()[1:] if x.strip()]
    return _date(lines[-1].split(",")[0]) if lines else ""


def unbarred(out: Path, sym: str, lo: int, hi: int) -> int:
    """The days in [lo, hi] (ms) on which `sym` settled funding and has no daily bar."""
    bars = fu.read_rows(out / "um" / "klines1d" / f"{sym}.csv", fu.KLINE_COLUMNS)
    have = {_day(int(r[0])) for r in bars}
    paid = ff.read(out / "um" / "fundingRate" / sym / f"{sym}-fundingRate.csv")
    return len({_day(t) for t, _, _ in paid if lo <= t <= hi} - have)


def _csv_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"{path}: expected one CSV inside, found {names}")
        text = z.read(names[0]).decode()
    return [r for r in csv.reader(io.StringIO(text)) if r]


def _ms(t: str) -> int:
    v = int(t)
    return v // 1000 if v > 10**14 else v  # microseconds, should a file carry them


def kline_rows(path: Path, lo: int, hi: int) -> list[list[str]]:
    """A monthly 1d kline file's bars in [lo, hi], as klines1d/<SYMBOL>.csv keeps them."""
    out = []
    for r in _csv_rows(path):
        if not r[0].isdigit():
            continue  # the header, where there is one
        if len(r) < 9:
            raise RuntimeError(f"{path}: a row of {len(r)} columns, not 12")
        t = _ms(r[0])
        if lo <= t <= hi:
            out.append([str(t), r[1], r[2], r[3], r[4], r[5], r[7], r[8]])
    return out


def funding_rows(path: Path, lo: int, hi: int) -> list[tuple[int, str, str]]:
    """A monthly fundingRate file's settlements in [lo, hi], as fetch_funding keeps them."""
    rows = _csv_rows(path)
    head = [c.strip() for c in rows[0]] if rows else []
    if "calc_time" in head and "last_funding_rate" in head:
        ti, ri = head.index("calc_time"), head.index("last_funding_rate")
    elif "funding_time" in head and "funding_rate" in head:
        ti, ri = head.index("funding_time"), head.index("funding_rate")
    else:
        raise RuntimeError(f"{path}: unknown columns {head}")
    out = []
    for r in rows[1:]:
        t = _ms(r[ti])
        if lo <= t <= hi:
            out.append((t, r[ri], ""))
    return out


def fetch_months(a, kind: str, sym: str, months: set[str]) -> tuple[list[Path], list[str]]:
    """The monthly files of `kind` (klines/1d or fundingRate) the archive has for `sym` in
    the months given: their paths, and what failed."""
    sub = f"klines/{sym}/1d" if kind == "klines" else f"fundingRate/{sym}"
    tag = "1d" if kind == "klines" else "fundingRate"
    _, keys = listing(a.list_url, f"data/futures/um/monthly/{sub}/", "", a.tries, a.timeout)
    names = sorted({k.rsplit("/", 1)[1] for k in keys if k.endswith(".zip")})
    names = [n for n in names if n.startswith(f"{sym}-{tag}-") and n[-11:-4] in months]
    local = f"um/{sub}" if kind == "klines" else f"um/fundingRate/{sym}/monthly"
    got, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        jobs = [pool.submit(fb.fetch_file, a.base_url.rstrip("/"),
                            f"data/futures/um/monthly/{sub}", local, n, n, a.out, a.tries,
                            a.timeout) for n in names]  # fmt: skip
        for j in jobs:
            name, status, _ = j.result()
            if status in ("ok", "have"):
                got.append(a.out / local / name)
            elif status != "missing":
                failed.append(f"{name}: {status}")
    return sorted(got), failed


def fill_bars(a, sym: str, months: set[str], lo: int, hi: int) -> tuple[int, int, list[str]]:
    """(days added, days now, what failed)."""
    paths, failed = fetch_months(a, "klines", sym, months)
    path = a.out / "um" / "klines1d" / f"{sym}.csv"
    have = fu.read_rows(path, fu.KLINE_COLUMNS)
    rows = {int(r[0]): r for p in paths for r in kline_rows(p, lo, hi)}
    rows.update({int(r[0]): r for r in have})  # a day already there is kept as it is
    if rows:
        fu.write_rows(path, fu.KLINE_COLUMNS, [rows[t] for t in sorted(rows)])
    return len(rows) - len(have), len(rows), failed


def fill_funding(a, sym: str, months: set[str], lo: int, hi: int) -> tuple[int, str, list[str]]:
    """(settlements, where they came from, what failed), for a perpetual gone from the list."""
    path = a.out / "um" / "fundingRate" / sym / f"{sym}-fundingRate.csv"
    have = ff.read(path)
    if have:
        return len(have), "file", []
    try:
        time.sleep(a.funding_pause)
        rows = ff.fetch(a.api_url.rstrip("/"), sym, lo, hi, a.tries, a.timeout, a.funding_pause)
    except RuntimeError as e:
        if "-1121" not in str(e) and "Invalid symbol" not in str(e):
            raise
        rows = []
    source = "api"
    if not rows:
        paths, failed = fetch_months(a, "fundingRate", sym, months)
        got = {t: (t, rate, mark) for p in paths for t, rate, mark in funding_rows(p, lo, hi)}
        rows, source = [got[t] for t in sorted(got)], "archive"
        if failed:
            return 0, "archive", failed
    if not rows:
        return 0, "none", []
    ff.write(path, rows)
    return len(rows), source, []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD, included")
    p.add_argument("--out", default="binance-data", type=Path, help="default ./binance-data")
    p.add_argument("--jobs", default=4, type=int, help="files downloaded at once (default 4)")
    p.add_argument("--tries", default=5, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--funding-pause", default=0.65, type=float)
    p.add_argument("--base-url", default=fb.BASE, help=argparse.SUPPRESS)
    p.add_argument("--api-url", default=fu.BASE, help=argparse.SUPPRESS)
    p.add_argument("--list-url", default=LIST, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.end < a.start:
        raise SystemExit(f"--end {a.end} is before --start {a.start}")
    listed_file = a.out / "um" / "universe.csv"
    contracts_file = a.out / "um" / "contracts.csv"
    for path in (listed_file, contracts_file):
        if not path.exists():
            raise SystemExit(f"no {path}: run fetch_universe.py --list-only on this folder first")
    with listed_file.open(newline="") as f:
        listed = {row["symbol"] for row in csv.DictReader(f)}
    with contracts_file.open(newline="") as f:
        known = {row["symbol"]: row["contract_type"] for row in csv.DictReader(f)}
    lo, hi = ff.ms(a.start), ff.ms(a.end + dt.timedelta(days=1)) - 1
    months = set(fb.months(a.start, a.end))
    try:
        prefixes, _ = listing(a.list_url, KLINES, "/", a.tries, a.timeout)
    except (RuntimeError, fb.Missing) as e:
        print(f"FAILED: the archive's list of perpetuals: {e}", file=sys.stderr)
        return 1
    archive = perpetuals(prefixes)
    # gone: in the archive, not in Binance's list of contracts at all; a contract of another
    # type there (shares, funds, metals, whatever Binance calls them) is no coin's perpetual
    other = {s: known[s] for s in archive if s not in listed and s in known}
    gone = [s for s in archive if s not in listed and s not in known]
    gaps = {s: n for s in archive if s in listed and (n := unbarred(a.out, s, lo, hi))}
    print(f"{len(archive)} USDT perpetuals in the archive: {len(gone)} gone from Binance's "
          f"list, {len(gaps)} listed with days of funding and no bar")  # fmt: skip
    if other:
        types: dict[str, int] = {}
        for t in other.values():
            types[t] = types.get(t, 0) + 1
        print(f"left out, {len(other)} Binance lists as contracts of other types: "
              + ", ".join(f"{t or '?'} {n}" for t, n in sorted(types.items())))  # fmt: skip
    # the last day the listed ones have a bar (to --end at most): a perpetual that has one
    # then is still trading, gone from the list or not
    newest = min(max((_last_day(a.out, s) for s in listed), default=""), a.end.isoformat())
    trading: list[str] = []
    failed: list[str] = []
    gone_rows = []
    gone_file = a.out / "um" / "gone.csv"
    before = {r[0]: r[5] for r in fu.read_rows(gone_file, GONE_COLUMNS)}
    targets = [*gone, *sorted(gaps)]
    for n, sym in enumerate(targets, 1):
        try:
            added, days, bad = fill_bars(a, sym, months, lo, hi)
            failed += [f"{sym} {b}" for b in bad]
            line = f"  [{n}/{len(targets)}] {sym:<16} "
            bars = fu.read_rows(a.out / "um" / "klines1d" / f"{sym}.csv", fu.KLINE_COLUMNS)
            first, last = (_date(bars[0][0]), _date(bars[-1][0])) if bars else ("", "")
            if sym in gaps:
                line += f"listed   +{added} days of {gaps[sym]} without a bar"
            elif newest and last >= newest:
                trading.append(sym)
                line += f"trading  {days} days {first} .. {last}: not gone, left out"
            else:
                settled, source, bad = fill_funding(a, sym, months, lo, hi)
                failed += [f"{sym} {b}" for b in bad]
                if source == "file":  # fetched before: where from, as recorded then
                    source = before.get(sym, source)
                gone_rows.append([sym, first, last, days, settled, source])
                line += f"gone     {days} days {first} .. {last}, {settled} settlements ({source})"
            print(line, flush=True)
        except (RuntimeError, OSError, fb.Missing) as e:
            failed.append(sym)
            print(f"  [{n}/{len(targets)}] {sym:<16} FAILED: {e}", file=sys.stderr, flush=True)
    fu.write_rows(gone_file, GONE_COLUMNS, gone_rows)
    with_bars = sum(1 for r in gone_rows if r[3])
    if trading:
        print(f"left out, {len(trading)} not in Binance's list yet trading on {newest}: "
              + ", ".join(trading[:12]) + (" ..." if len(trading) > 12 else ""))  # fmt: skip
    print(f"done: {with_bars} of {len(gone_rows)} gone perpetuals with bars, "
          f"{sum(1 for r in gone_rows if r[4])} with funding -> {a.out / 'um'}")  # fmt: skip
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
