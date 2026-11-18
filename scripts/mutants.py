"""Inject deliberate faults one at a time and check that the tests catch each of them.

    python scripts/mutants.py

Each fault is a one-line change: a window that reads ahead, an interval that includes its
end, a flow that crosses a hole, a bar that keeps its unfinished tail and so on. A fault
the tests miss is printed as SURVIVED and the script exits with 1. The source is restored
after every run.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

M = [
    (
        "marketlab/features/base.py",
        'hi = np.searchsorted(ev_ts, g.ts, side="right")',
        'hi = np.searchsorted(ev_ts, g.ts + 200_000_000, side="right")',
        "window reads 200 ms ahead",
    ),
    (
        "marketlab/features/base.py",
        'lo = np.searchsorted(ev_ts, g.ts - window, side="right")',
        'lo = np.searchsorted(ev_ts, g.ts - window, side="left")',
        "window includes t-w",
    ),
    (
        "marketlab/features/base.py",
        "    out[~g.full(window)] = np.nan\n    return out\n\n\ndef window_last",
        "    return out\n\n\ndef window_last",
        "partial windows kept",
    ),
    (
        "marketlab/features/base.py",
        "hi = np.arange(n)  # inclusive",
        "hi = np.minimum(np.arange(n) + 1, n - 1)",
        "extreme reads next row",
    ),
    (
        "marketlab/features/base.py",
        "ok &= ev_seg[j] == g.segment if len(ev_seg) else False",
        "pass",
        "last value across segments",
    ),
    (
        "marketlab/io/intervals.py",
        "    inside = (i >= 0) & (ts < ends[np.clip(i, 0, None)])",
        "    inside = (i >= 0) & (ts <= ends[np.clip(i, 0, None)])",
        "end inclusive",
    ),
    (
        "marketlab/io/intervals.py",
        '            elif kind == "late":\n                late = True',
        '            elif kind == "late":\n                pass',
        "late ignored",
    ),
    (
        "marketlab/io/intervals.py",
        '                if kind in ("reconnect", "disconnected"):\n                    late = False',
        "                if False:\n                    late = False",
        "late survives reconnect",
    ),
    (
        "marketlab/io/intervals.py",
        "        book = run.book_at_start is not None",
        "        book = False",
        "midnight carry ignored",
    ),
    (
        "marketlab/features/ofi.py",
        "        - (n_ap <= p_ap) * n_aq",
        "        - (n_ap < p_ap) * n_aq",
        "ofi ask term",
    ),
    (
        "marketlab/features/ofi.py",
        "    keep = same if e.ndim == 1 else same[:, None]",
        "    keep = np.ones_like(same) if e.ndim == 1 else np.ones_like(same)[:, None]",
        "ofi across holes",
    ),
    (
        "marketlab/features/ofi.py",
        "_NO_ASK = np.iinfo(np.int64).max",
        "_NO_ASK = 0",
        "missing ask as 0",
    ),
    (
        "marketlab/features/book.py",
        "                sb, sa = cb[:, k - 1], ca[:, k - 1]",
        "                sb, sa = cb[:, min(k, 19)], ca[:, min(k, 19)]",
        "imbalance one level too deep",
    ),
    (
        "marketlab/features/book.py",
        "                unknown = ((nb == 20) & (last_bid >= lo)) | ((na == 20) & (last_ask <= hi))",
        "                unknown = np.zeros(len(nb), bool)",
        "depth never unknown",
    ),
    (
        "marketlab/features/trades.py",
        '    buy = ~t["is_buyer_maker"].to_numpy().astype(bool)',
        '    buy = t["is_buyer_maker"].to_numpy().astype(bool)',
        "aggressor flipped",
    ),
    (
        "marketlab/features/trades.py",
        "            buy_at = cbuy[k] + np.where(is_buy, edges - ctot[k], 0)",
        "            buy_at = cbuy[k + 1]",
        "vpin no split",
    ),
    (
        "marketlab/features/trades.py",
        "            closes.append(ts[k])",
        "            closes.append(ts[np.minimum(k + 1, len(ts) - 1)])",
        "vpin closes late",
    ),
    (
        "marketlab/features/activity.py",
        '        step = np.searchsorted(g.ts, t_ts, side="left")',
        '        step = np.searchsorted(g.ts, t_ts, side="left") + 1',
        "cancel trades to next step",
    ),
    (
        "marketlab/features/activity.py",
        "        known = (q0 > 0) & (found | ~full[:, None] | within)",
        "        known = q0 > 0",
        "pushed-out counted",
    ),
    (
        "marketlab/features/activity.py",
        "            step[1:] = np.where(same, np.maximum(uid[1:] - uid[:-1], 0), 0)",
        "            step[1:] = np.maximum(uid[1:] - uid[:-1], 0)",
        "book rate across holes",
    ),
    (
        "marketlab/features/vol.py",
        "        r2[1:] = np.where(prev[1:], r * r, 0.0)",
        "        r2[1:] = r * r",
        "rv across holes",
    ),
    (
        "marketlab/features/microprice.py",
        "    x = np.concatenate([x, mirror(x)])",
        "    x = np.concatenate([x, x])",
        "no symmetrization",
    ),
    (
        "marketlab/features/microprice.py",
        "    keep = (x >= 0) & (x_next >= 0) & (np.abs(halves) <= 2)",
        "    keep = (x >= 0) & (x_next >= 0)",
        "big moves kept",
    ),
    (
        "marketlab/bars/__init__.py",
        "        if total >= threshold:",
        "        if total > threshold:",
        "bar closes late",
    ),
    ("marketlab/bars/__init__.py", "    out[out == k] = -1", "    pass", "unfinished bar kept"),
    (
        "marketlab/bars/__init__.py",
        "        first = -(-lo // n) * n  # the first boundary at or after the segment's start",
        "        first = lo // n * n",
        "time bar starts before segment",
    ),
    (
        "marketlab/bars/__init__.py",
        '    close_mid = _mid_at(day, df["end"].to_numpy() - 1, df["segment"].to_numpy())',
        '    close_mid = _mid_at(day, df["end"].to_numpy() + 500_000_000, df["segment"].to_numpy())',
        "bar mid from the future",
    ),
    (
        "marketlab/io/day.py",
        "                df = df.filter(pl.col(t) <= self.until)",
        "                df = df.filter(pl.col(t) <= self.until + 1_000_000_000)",
        "truncation leaks",
    ),
    (
        "marketlab/io/contract.py",
        "    if meta.table != table:",
        "    if False:",
        "table metadata unchecked",
    ),
]
survived = []
for f, old, new, label in M:
    p = ROOT / f
    src = p.read_text()
    if src.count(old) != 1:
        print(f"cannot place: {label}")
        survived.append(label)
        continue
    p.write_text(src.replace(old, new))
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        p.write_text(src)
    killed = r.returncode != 0
    print(("killed  " if killed else "SURVIVED"), label)
    if not killed:
        survived.append(label)
print(f"{len(M) - len(survived)} of {len(M)} faults caught")
sys.exit(1 if survived else 0)
