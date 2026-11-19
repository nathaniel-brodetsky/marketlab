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
        "        out[i] = k\n        total += x\n        if total >= threshold:",
        "        out[i] = k\n        total += x\n        if total > threshold:",
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
    (
        "marketlab/execution/sim.py",
        "    seg = m.segment(o.decided)\n    if seg < 0:\n        return Result(o, Outcome.HOLE, a, o.decided)\n    seg_end = m.segment_end(seg)\n    if a > seg_end:",
        "    a = o.decided\n    seg = m.segment(o.decided)\n    if seg < 0:\n        return Result(o, Outcome.HOLE, a, o.decided)\n    seg_end = m.segment_end(seg)\n    if a > seg_end:",
        "passive order without arrival latency",
    ),
    (
        "marketlab/execution/sim.py",
        "        end = min(end, o.cancel + latency)",
        "        end = min(end, o.cancel)",
        "cancel without latency",
    ),
    (
        "marketlab/execution/sim.py",
        '    ahead = max(level - int(tr["qty"][seen][early].sum()), 0)',
        "    ahead = level",
        "queue ignores trades seen by the arrival",
    ),
    (
        "marketlab/execution/sim.py",
        '            if tr["trade_ts"][k] <= e0:\n                continue',
        "            if False:\n                continue",
        "trades already in the book count again",
    ),
    (
        "marketlab/execution/sim.py",
        "            pending = traded - applied  # part of this update, not received yet",
        "            pending = 0",
        "late trades taken for cancels",
    ),
    (
        "marketlab/execution/sim.py",
        "            now_ahead = min(max(now_ahead, 0), new)",
        "            now_ahead = max(now_ahead, 0)",
        "more ahead than the level",
    ),
    (
        "marketlab/execution/sim.py",
        "                now_ahead -= cancelled * now_ahead / before",
        "                now_ahead -= cancelled / 2",
        "proportional share wrong",
    ),
    (
        "marketlab/execution/sim.py",
        "    if other is not None and (o.price >= other[0] if o.side == BID else o.price <= other[0]):",
        "    if False:",
        "crossing orders accepted",
    ),
    (
        "marketlab/execution/sim.py",
        '    hits_side = tr["is_buyer_maker"] if o.side == BID else ~tr["is_buyer_maker"]',
        '    hits_side = ~tr["is_buyer_maker"] if o.side == BID else tr["is_buyer_maker"]',
        "wrong aggressor fills",
    ),
    (
        "marketlab/execution/sim.py",
        "            if through[k]:",
        "            if False:",
        "trade-through ignored",
    ),
    (
        "marketlab/execution/sim.py",
        "    taken = book.walk(-o.side, o.qty)  # a buy takes from the asks",
        "    taken = book.walk(o.side, o.qty)",
        "market order takes its own side",
    ),
    (
        "marketlab/execution/__init__.py",
        "            mk = side * (mid - price) / price * 1e4",
        "            mk = (mid - price) / price * 1e4",
        "markout unsigned",
    ),
    (
        "marketlab/execution/__init__.py",
        '            i = int(np.searchsorted(t["recv_ts"], when, side="right")) - 1',
        '            i = int(np.searchsorted(t["recv_ts"], when + 300_000_000, side="right")) - 1',
        "probe price from the future",
    ),
    (
        "marketlab/execution/market.py",
        "        ok = (seg >= 0) & (i >= 0) & (ts < ends[s]) & (r[j] >= starts[s])",
        "        ok = (seg >= 0) & (i >= 0) & (r[j] >= starts[s])",
        "markout past its segment",
    ),
    (
        "marketlab/execution/market.py",
        "        ok = (seg >= 0) & (i >= 0) & (ts < ends[s]) & (r[j] >= starts[s])",
        "        ok = (seg >= 0) & (i >= 0) & (ts < ends[s])",
        "markout from a price before the hole",
    ),
    (
        "marketlab/execution/market.py",
        '    hi = int(np.searchsorted(d["recv_ts"], ts, side="right"))',
        "    hi = lo",
        "book without later depth updates",
    ),
    (
        "marketlab/validation/splits.py",
        "        train_mask &= ~((t1 >= lo) & (t0 <= hi))",
        "        pass",
        "no purging",
    ),
    (
        "marketlab/validation/splits.py",
        "        lo, hi = t0[mine].min(), t1[mine].max() + embargo",
        "        lo, hi = t0[mine].min(), t1[mine].max()",
        "no embargo",
    ),
    (
        "marketlab/validation/splits.py",
        "    return [{g: v[j] for g, v in by_group.items()} for j in range(n)]",
        "    return [{g: v[0] for g, v in by_group.items()} for j in range(n)]",
        "every path the same split",
    ),
    (
        "marketlab/validation/metrics.py",
        "        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n",
        "        v += 0 * (e[k:] @ e[:-k]) / n",
        "Newey-West without the lags",
    ),
    (
        "marketlab/validation/metrics.py",
        "    r = stats.rankdata(x)",
        "    r = np.asarray(x, dtype=float)",
        "IC on values, not ranks",
    ),
    (
        "marketlab/validation/metrics.py",
        "    se = float(np.sqrt(sum(x.se**2 for x in d)) / len(d))",
        "    se = float(np.sqrt(sum(x.se**2 for x in d)) / len(d) ** 2)",
        "error of the mean too small",
    ),
    (
        "marketlab/validation/pbo.py",
        "        best = int(np.argmax(is_perf))",
        "        best = int(np.argmax(oos))",
        "PBO picks with hindsight",
    ),
    (
        "marketlab/validation/multiple.py",
        "    ranked = p[order] * n / np.arange(1, n + 1)",
        "    ranked = p[order] * n",
        "Bonferroni instead of BH",
    ),
    (
        "marketlab/validation/multiple.py",
        "    return float(np.sqrt(trials_var) * z)",
        "    return 0.0",
        "trials not deflated",
    ),
    (
        "marketlab/validation/surrogates.py",
        "    return np.roll(x, int(rng.integers(min_lag, n - min_lag + 1)))",
        "    return np.roll(x, 0)",
        "shift surrogate not shifted",
    ),
    (
        "marketlab/experiment.py",
        "    every = survey(data_root, spec.symbol, exclude=spec.holdout)",
        "    every = survey(data_root, spec.symbol)",
        "held-out days used in the run",
    ),
    (
        "marketlab/experiment.py",
        '    dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], root).stdout.strip())',
        "    dirty = False",
        "uncommitted code not recorded",
    ),
    (
        "marketlab/targets/__init__.py",
        '        cols[f"ret_{h}"] = np.where(inside, r, np.nan)',
        '        cols[f"ret_{h}"] = r',
        "returns across a hole",
    ),
    (
        "marketlab/bars/__init__.py",
        "    return min((lo, hi), key=lambda t: abs(_count(segments, t) - target))",
        "    return max(1, round(total / target))",
        "thresholds from totals, not counts",
    ),
    (
        "marketlab/experiment.py",
        "    infos = [d for d in every if d.usable_hours >= spec.min_hours]",
        "    infos = every",
        "days short of usable hours used",
    ),
    (
        "marketlab/experiment.py",
        "    skipped = [d for d in every if d.usable_hours < spec.min_hours]",
        "    skipped = []",
        "skipped days not named",
    ),
    (
        "marketlab/experiment.py",
        "    dev = [d for d in every if d.usable_hours >= spec.min_hours and d.date not in spec.holdout]",
        "    dev = [d for d in every if d.usable_hours >= spec.min_hours]",
        "held-out days trained on by final",
    ),
    (
        "marketlab/bars/__init__.py",
        "    if len(steps) > 1:",
        "    if False:",
        "bars calibrated across different steps",
    ),
    (
        "marketlab/cli.py",
        "((n, k) for k, n in quality.items() if k not in _ROUTINE and n)",
        "((n, k) for k, n in quality.items() if n)",
        "routine events reported as problems",
    ),
    (
        "marketlab/io/contract.py",
        '        pieces.append(pq.read_table(p, columns=cols, filters=[(t, ">=", lo), (t, "<", hi)]))',
        '        pieces.append(pq.read_table(p, columns=cols, filters=[(t, ">=", lo)]))',
        "a window read past its end",
    ),
    (
        "marketlab/io/contract.py",
        "        if st is None or not st.has_min_max or (st.min < hi and st.max >= lo):",
        "        if st is None or not st.has_min_max:",
        "parts inside a window skipped",
    ),
    (
        "marketlab/io/day.py",
        '            window = None if table == "quality" else between',
        "            window = between",
        "the journal cut to the window",
    ),
    (
        "marketlab/view/market.py",
        "PAD = 2 * NS  # read before the start: the last second's order flow needs it",
        "PAD = 0",
        "no data read before a stretch",
    ),
    (
        "marketlab/view/market.py",
        "AFTER = 2 * NS  # read after the end: the next second's return needs it",
        "AFTER = 0",
        "no data read after a stretch",
    ),
    (
        "marketlab/view/market.py",
        "        keep[before] = True",
        "        pass",
        "the book at a stretch's start unknown",
    ),
    (
        "marketlab/view/market.py",
        "        late=_minus(_clip(loose, lo, hi), strict),",
        "        late=[],",
        "late parts not shaded",
    ),
    (
        "marketlab/view/market.py",
        "    rows = np.where(last >= first, last, -1)",
        "    rows = last",
        "a bin in a hole shows the row before it",
    ),
    (
        "marketlab/view/market.py",
        "    bid = s.book.bid_qty[r][:, ::-1].T.astype(float)  # level 20 first, at the bottom",
        "    bid = s.book.bid_qty[r].T.astype(float)",
        "bid levels upside down in the heat map",
    ),
    (
        "marketlab/view/market.py",
        "        weighted = (bp * aq + ap * bq) / (bq + aq)",
        "        weighted = (bp * bq + ap * aq) / (bq + aq)",
        "weighted mid with the sizes swapped",
    ),
    (
        "marketlab/view/market.py",
        "        valid = i >= 0 and seg[k] >= 0 and b.segment[i] == seg[k]",
        "        valid = i >= 0",
        "a replay shows a stale book in a hole",
    ),
    (
        "marketlab/view/market.py",
        "    breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1",
        "    breaks = np.zeros(0, dtype=np.int64)",
        "the replay's mid bridges a hole",
    ),
    (
        "marketlab/maker.py",
        '    return (w < 0) | (fills["ts"].to_numpy() <= w + latency)',
        '    return (w < 0) | (fills["ts"].to_numpy() <= w)',
        "a withdrawal takes effect at once",
    ),
    (
        "marketlab/maker.py",
        "        for s, against in ((1, z < -threshold), (-1, z > threshold)):",
        "        for s, against in ((1, z > threshold), (-1, z < -threshold)):",
        "orders withdrawn when the signal is for them",
    ),
    (
        "marketlab/maker.py",
        "    stale = (first < 0) | (decided >= seg_end[np.clip(first, 0, None)])",
        "    stale = first < 0",
        "a row from before a hole read after it",
    ),
    (
        "marketlab/maker.py",
        "            when = np.maximum(ts[np.clip(j, 0, n - 1)], decided[k])",
        "            when = ts[np.clip(j, 0, n - 1)]",
        "an order withdrawn before it was decided",
    ),
    (
        "marketlab/maker.py",
        "        return self.rule - self.base",
        "        return self.base - self.rule",
        "the difference the wrong way round",
    ),
    (
        "marketlab/maker.py",
        "            deltas.append(score(x, spec, v, spec.queue, np.roll(z, lag)).delta)",
        "            deltas.append(score(x, spec, v, spec.queue, z).delta)",
        "surrogates not shifted",
    ),
    (
        "marketlab/maker.py",
        '    lo = np.searchsorted(ts, ts - span, side="left")',
        "    lo = np.zeros(len(ts), dtype=np.int64)",
        "the signal scaled by the whole past, not the hour",
    ),
    (
        "marketlab/maker.py",
        "    p = float(stats.t.sf(mean / se, ok.sum() - 1)) if se > 0 else (0.0 if mean > 0 else 1.0)",
        "    p = float(stats.t.cdf(mean / se, ok.sum() - 1)) if se > 0 else (0.0 if mean > 0 else 1.0)",
        "p from the wrong tail",
    ),
    (
        "marketlab/maker.py",
        "BEFORE = 60 * 1_000_000_000",
        "BEFORE = 0",
        "a piece read from its first decision, without the book before it",
    ),
    (
        "marketlab/maker.py",
        "                if lo <= o.decided < hi",
        "                if lo <= o.decided <= hi",
        "an order decided at a piece's end simulated twice",
    ),
    (
        "marketlab/events/hawkes.py",
        "            grad[6] += (a00 * (s0 - beta * d0) + a01 * (s1 - beta * d1)) * inv",
        "            grad[6] += (a00 * s0 + a01 * s1) * inv",
        "the gradient in beta forgets the kernel's own decay",
    ),
    (
        "marketlab/events/hawkes.py",
        "        d0 = e * (d0 + dt * s0)",
        "        d0 = e * d0",
        "the derivative sum not carried forward",
    ),
    (
        "marketlab/events/hawkes.py",
        "    ll -= mu0 * horizon + a00 * c0 + a01 * c1 + mu1 * horizon + a10 * c0 + a11 * c1",
        "    ll -= mu0 * horizon + a00 * c0 + mu1 * horizon + a10 * c0 + a11 * c1",
        "a kernel left out of the compensator",
    ),
    (
        "marketlab/events/hawkes.py",
        "        (trade_ts[1:] != trade_ts[:-1]) | (buy[1:] != buy[:-1]) | (segment[1:] != segment[:-1])",
        "        (trade_ts[1:] != trade_ts[:-1]) | (segment[1:] != segment[:-1])",
        "a buy and a sell at one time taken for one order",
    ),
    (
        "marketlab/events/hawkes.py",
        "        first = -(-(lo + window) // step) * step",
        "        first = -(-lo // step) * step",
        "a window that starts before its interval",
    ),
    (
        "marketlab/events/hawkes.py",
        '    k = np.searchsorted(r.at, query, side="right") - 1',
        '    k = np.searchsorted(r.at, query, side="left") - 1',
        "a fit not used at its own time",
    ),
    (
        "marketlab/volatility.py",
        '        return 1.0 - self.candidate / self.har if self.har > 0 else float("nan")',
        '        return 1.0 - self.har / self.candidate if self.har > 0 else float("nan")',
        "the gain the wrong way round",
    ),
    (
        "marketlab/volatility.py",
        "        out[r] = np.roll(f[r], lag, axis=0)",
        "        out[r] = f[r]",
        "volatility surrogates not shifted",
    ),
    (
        "marketlab/volatility.py",
        "    dev = [d for d in every if d.usable_hours >= spec.min_hours and d.date not in spec.holdout]",
        "    dev = [d for d in every if d.usable_hours >= spec.min_hours]",
        "the held-out days in the final fit",
    ),
    (
        "marketlab/volatility.py",
        "    lam_before, _ = hk.at_times(r, ts, side, query - minute, step)",
        "    lam_before, _ = hk.at_times(r, ts, side, query, step)",
        "the intensity's change measured against itself",
    ),
    (
        "marketlab/experiment.py",
        "            out.setdefault(slug, 1.0)",
        "            pass",
        "a hypothesis run first is adjusted only for those run before it",
    ),
    (
        "marketlab/strategy.py",
        "while pending and pending[0][0] <= t:",
        "while pending:",
        "the position limit counts fills not received yet",
    ),
    (
        "marketlab/strategy.py",
        "if side * position + qty > q.max_position * qty:",
        "if side * position > q.max_position * qty:",
        "an order that takes the position past its limit",
    ),
    (
        "marketlab/strategy.py",
        "        i = self.row(t)\n        if i < 0 or not self.against[side][i]:",
        "        i = self.row(t)\n        i = min(i + 1, len(self.ts) - 1) if i >= 0 else i\n"
        "        if i < 0 or not self.against[side][i]:",
        "the strategy reads the signal one row ahead",
    ),
    (
        "marketlab/strategy.py",
        "                nxt[side] = r.final\n",
        "                nxt[side] = r.final if r.outcome not in (Outcome.REJECTED, Outcome.UNKNOWN)"
        " else t + RETRY\n",
        "a rejection known before the order arrives",
    ),
    (
        "marketlab/strategy.py",
        "reach = life + 2 * lat + 2 * SEC",
        "reach = life",
        "a piece read too short for the orders decided at its end",
    ),
    (
        "marketlab/strategy.py",
        'captured=float((signed * (fills["mid"] - fills["price"])).sum())',
        'captured=float((signed * (fills["price"] - fills["mid"])).sum())',
        "captured spread with the wrong sign",
    ),
    (
        "marketlab/execution/sim.py",
        "            elif hidden:\n                continue  # behind all of a level not yet seen\n",
        "",
        "trades reach an order at a level not yet seen",
    ),
    (
        "marketlab/execution/sim.py",
        "                ahead, hidden = new + pending, False",
        "                ahead, hidden = new, False",
        "trades of the first update at an unseen level counted twice",
    ),
    (
        "marketlab/strategy.py",
        "return min(bid, int(np.floor(mid * (1 - offset_bp / 1e4))))",
        "return min(bid, int(np.ceil(mid * (1 - offset_bp / 1e4))))",
        "a bid away from the mid rounded towards it",
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
