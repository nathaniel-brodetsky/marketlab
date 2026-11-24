"""The shadow: what a cross-sectional rule written down as a hypothesis would hold, decided
each day at the daily close from the data there was then, and written down before the next
day begins. No key, no account, no order: a list of coins and weights a day.

It records decisions only, never what they made: a rule tested forward is judged once, by
`marketlab experiment run --final` on the data of its whole held-out period, and a running
score would invite stopping or changing it on the way. What the record is for: that every
decision was made in time, from the coins listed then (one delisted later is in it), and
that the data were there to make it; and, should the rule ever trade, it is what an order
would have been placed for.

`record` refuses a close that is not over yet (today's, in UTC) and data that are not up to
date: the close's daily bar missing for BTCUSDT or ETHUSDT, or for more than `MAX_MISSING`
of the perpetuals that had one the day before. A close already recorded is left as it was.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import positioning as pos
from . import trend as tr
from . import xsection as xs
from . import xsection_kind as xk

COLUMNS = ("close_date", "decided_at", "trade_at", "symbol", "weight")
MAX_MISSING = 0.05  # of the perpetuals with a bar the day before
MUST_HAVE = ("BTCUSDT", "ETHUSDT")


class NotReady(ValueError):
    pass


@dataclass
class Decision:
    close_date: str
    trade_at: str  # UTC, ISO
    weights: dict[str, float]  # the coins held over the next day, by weight
    universe: int  # perpetuals in the universe at the close


def decide(root: str | Path, spec: xk.Spec, close: str, today: str | None = None) -> Decision:
    """What `spec`'s primary rule holds over the day after `close`, decided at its close
    from the data up to it and nothing later."""
    today = today or dt.datetime.now(dt.UTC).date().isoformat()
    if close >= today:
        raise NotReady(f"the close of {close} is not over yet (today is {today}, UTC)")
    rule = spec.primary
    back = max(4 * rule.hold + 10, xk.history([rule]))
    first = (dt.date.fromisoformat(close) - dt.timedelta(days=back)).isoformat()
    p = xs.load(root, first, close, held_out=True)  # decisions only: no return is computed
    last = len(p.dates) - 1
    had, has = np.isfinite(p.close[last - 1]), np.isfinite(p.close[last])
    lacking = [s for s in MUST_HAVE if s in p.symbols and not has[p.symbols.index(s)]]
    missing = float((had & ~has).sum() / max(had.sum(), 1))
    if lacking or not has.any() or missing > MAX_MISSING:
        raise NotReady(
            f"the daily bars of {close} are not all there: "
            + (f"none for {', '.join(lacking)}; " if lacking else "")
            + f"{missing:.0%} of the perpetuals with one the day before have none "
            f"(scripts/fetch_universe.py --end {close})"
        )
    inu = xs.universe(p, rule.top)
    trade = dt.datetime.fromisoformat(close).replace(tzinfo=dt.UTC) + dt.timedelta(
        days=1, hours=rule.hours
    )
    if rule.signal == "trend":  # a few coins, every day: its weights at this close
        missing = [c for c in rule.coins if c not in p.symbols]
        if missing:
            raise NotReady(f"no daily bars for {', '.join(missing)}")
        cols = [p.symbols.index(c) for c in rule.coins]
        w = tr.decide(p, cols, rule.lookbacks, rule.size, rule.target,
                      tr.volatility(p, rule.window), last)  # fmt: skip
        return Decision(close, trade.isoformat(timespec="minutes"),
                        {p.symbols[j]: float(w[j]) for j in np.flatnonzero(w)},
                        int(inu[last].sum()))  # fmt: skip
    if rule.signal in pos.SIGNALS:  # the archive's files, and the API's at a close not out yet
        x = pos.signals(root, p, held_out=True, live=True)[rule.signal]
    else:
        x = xs.signals(p)[rule.signal]
    # one more row, so that the close of the last day decides the day after it
    x = np.vstack([x, np.full((1, x.shape[1]), np.nan)])
    inu = np.vstack([inu, np.zeros((1, inu.shape[1]), dtype=bool)])
    # a rule held for several days and not staggered rebalances on the days its held-out
    # period's first day sets; staggered, any start gives the same once all are in
    anchor = (dt.date.fromisoformat(spec.holdout_first) - dt.date.fromisoformat(p.dates[0])).days
    start = anchor % rule.hold
    held, _, _, _ = xs.holdings(x, inu, start, rule.hold, rule.q, rule.staggered, rule.long)
    w = held[last + 1]
    return Decision(
        close_date=close,
        trade_at=trade.isoformat(timespec="minutes"),
        weights={p.symbols[j]: float(w[j]) for j in np.flatnonzero(w)},
        universe=int(inu[last].sum()),
    )


def pending(spec: xk.Spec, close: str, out: str | Path, most: int = 31) -> list[str]:
    """The closes to record up to `close`: those after the last one recorded (a machine off
    for some days catches up, each decision from the data up to its own close), `most` at
    most; just `close` if none is recorded yet."""
    path = Path(out) / f"{spec.slug}.csv"
    last = ""
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.reader(f))[1:]
        last = max((r[0] for r in rows if r), default="")
    if not last or last >= close:
        return [close] if last != close else []
    first = dt.date.fromisoformat(last) + dt.timedelta(days=1)
    end = dt.date.fromisoformat(close)
    days = [(first + dt.timedelta(days=k)).isoformat() for k in range((end - first).days + 1)]
    return days[-most:]


def record(root: str | Path, spec: xk.Spec, close: str, out: str | Path,
           today: str | None = None) -> tuple[Decision | None, Path]:  # fmt: skip
    """Appends `spec`'s decision at `close` to OUT/<id-name>.csv, unless that close is there
    already: (the decision, or None if it was there; the file)."""
    path = Path(out) / f"{spec.slug}.csv"
    if path.exists():
        with path.open(newline="") as f:
            r = csv.reader(f)
            if tuple(next(r, ())) != COLUMNS:
                raise ValueError(f"{path}: not a shadow record (columns {COLUMNS})")
            if any(row and row[0] == close for row in r):
                return None, path
    d = decide(root, spec, close, today)
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        rows = sorted(d.weights.items()) or [("", 0.0)]  # a day without positions, recorded
        for sym, weight in rows:
            w.writerow([close, now, d.trade_at, sym, f"{weight:.6f}"])
    return d, path
