"""Binance's positioning on each USD-M perpetual, from its public archive's daily metrics
files (data/futures/um/daily/metrics, fetched by scripts/fetch_metrics.py into
um/metrics/<SYMBOL>/<SYMBOL>-metrics-<DAY>.zip): a row every 5 minutes of open interest, the
top traders' and all accounts' long/short ratios and the takers' buy/sell volume ratio.

A day's value is its last row's (the takers' ratio: the day's mean), so that it is known at
that day's close. Where a day's file is not out yet, the shadow may read what
scripts/fetch_metrics.py --live took from Binance's API at the close (um/metrics/<SYMBOL>/
live.csv); a hypothesis's runs read the archive only. Nothing of a day after
`xsection.DEV_END` is read unless `held_out`.

The signals, each so that the highest is held long, as the others are: the crowd the other
way (contrarian).
  crowd       minus all accounts' long/short ratio
  crowd_top   minus the top traders' long/short ratio of positions
  takers      minus the takers' buy/sell volume ratio
  oi7         minus the change in open interest (in coins) over 7 days, in log
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from . import xsection as xs

FIELDS = ("count_long_short_ratio", "sum_toptrader_long_short_ratio",
          "sum_taker_long_short_vol_ratio", "sum_open_interest")  # fmt: skip
MEAN = "sum_taker_long_short_vol_ratio"  # the day's mean; the others its last row
SIGNALS = ("crowd", "crowd_top", "takers", "oi7")
LIVE_COLUMNS = ("day", "count_long_short_ratio", "taken_at")


def path(root: str | Path, sym: str, day: str) -> Path:
    return Path(root) / "um" / "metrics" / sym / f"{sym}-metrics-{day}.zip"


def day_values(file: Path) -> dict[str, float]:
    """A day's file: each field's last value (the takers' ratio, its mean)."""
    with zipfile.ZipFile(file) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{file}: expected one CSV inside, found {names}")
        text = z.read(names[0]).decode()
    rows = sorted(csv.DictReader(io.StringIO(text)), key=lambda r: r.get("create_time", ""))
    out = {}
    for f in FIELDS:
        vals = [float(r[f]) for r in rows if r.get(f) not in (None, "")]
        if vals:
            out[f] = float(np.mean(vals)) if f == MEAN else vals[-1]
    return out


def read_live(root: str | Path, sym: str) -> dict[str, float]:
    """The all-accounts ratio taken from the API at each close, by day."""
    f = Path(root) / "um" / "metrics" / sym / "live.csv"
    if not f.exists():
        return {}
    with f.open(newline="") as fh:
        return {r["day"]: float(r["count_long_short_ratio"]) for r in csv.DictReader(fh)}


def read(root: str | Path, p: xs.Panel, held_out: bool = False,
         live: bool = False) -> dict[str, np.ndarray]:  # fmt: skip
    """Each field, days x symbols of the panel, NaN without a file; `live`: the all-accounts
    ratio from live.csv where the archive has no file."""
    if p.dates and p.dates[-1] > xs.DEV_END and not held_out:
        raise xs.HeldOut(f"{p.dates[-1]} is after {xs.DEV_END}: the years after it are held out")
    n, m = len(p.dates), len(p.symbols)
    out = {f: np.full((n, m), np.nan) for f in FIELDS}
    for j, sym in enumerate(p.symbols):
        folder = Path(root) / "um" / "metrics" / sym
        if not folder.is_dir():
            continue
        lv = read_live(root, sym) if live else {}
        for i, day in enumerate(p.dates):
            f = path(root, sym, day)
            if f.exists():
                for k, v in day_values(f).items():
                    out[k][i, j] = v
            elif day in lv:
                out["count_long_short_ratio"][i, j] = lv[day]
    return out


def signals(root: str | Path, p: xs.Panel, held_out: bool = False,
            live: bool = False) -> dict[str, np.ndarray]:  # fmt: skip
    v = read(root, p, held_out, live)
    oi = v["sum_open_interest"]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.log(oi / xs._lag(oi, 7))
    return {
        "crowd": -v["count_long_short_ratio"],
        "crowd_top": -v["sum_toptrader_long_short_ratio"],
        "takers": -v["sum_taker_long_short_vol_ratio"],
        "oi7": -np.where(np.isfinite(change), change, np.nan),
    }


def wanted(p: xs.Panel, first: str, top: int = 50, before: int = 7) -> list[tuple[str, str]]:
    """(symbol, day) for each coin's days in the universe of `top` from `first` on, and the
    `before` days before each: what a rule on these signals reads."""
    inu = xs.universe(p, top)
    s0 = p.dates.index(first) if first in p.dates else 0
    need = np.zeros(inu.shape, dtype=bool)
    for k in range(before + 1):
        need[s0 : len(p.dates) - k] |= inu[s0 + k :]
    need[:s0] = False
    return [(p.symbols[j], p.dates[i]) for i, j in zip(*np.nonzero(need), strict=True)]


def write_days(pairs: list[tuple[str, str]], out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("symbol", "day"))
        w.writerows(pairs)
