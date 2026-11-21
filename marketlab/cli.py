"""`marketlab`: look at recorded days from the command line."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from . import bars, synth
from .features import compute
from .io import ContractError, Day, read_manifest
from .registry import FITTED, METHODS, default_methods
from .report import describe

SEC = 1_000_000_000
HORIZONS_COSTWALL = ("1s", "5s", "30s", "1m", "5m", "15m", "1h", "4h", "1d", "1w")


def _clock(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / SEC, tz=dt.UTC).strftime("%H:%M:%S")


def _seconds(intervals: list[tuple[int, int]]) -> float:
    return sum(e - s for s, e in intervals) / SEC


#: Quality events of a healthy day; `late` time is shown under `excluded`.
_ROUTINE = frozenset({"connected", "disconnected", "synced", "late", "on_time"})


def _problems(quality: dict[str, int]) -> str:
    """The day's quality events other than routine ones, most frequent first."""
    found = sorted(
        ((n, k) for k, n in quality.items() if k not in _ROUTINE and n), key=lambda x: (-x[0], x[1])
    )
    text = ", ".join(f"{n} {k}" for n, k in found) or "none"
    if quality.get("undecodable"):
        text += (
            "; undecodable messages are missing from the tables and break the book: "
            "`marketrec verify` names the first"
        )
    return text


def cmd_day(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    print(f"day         {a.day}")
    print(
        f"symbol      {day.symbol}, price step {day.price_step.normalize():f}, "
        f"qty step {day.qty_step.normalize():f}, feed {day.source}"
    )
    for r in day.runs:
        start = (
            "book carried over from the day before"
            if r.book_at_start is not None
            else "book synced after the start"
        )
        print(
            f"run         {r.recorder}, {_clock(r.first_recv_ts)}-{_clock(r.last_recv_ts)} UTC, "
            f"{start}, derivation {r.derivation}"
        )
    rows = ", ".join(
        f"{t} {day.raw(t, [c]).height}"
        for t, c in (
            ("depth_diff", "recv_ts"),
            ("book_ticker", "recv_ts"),
            ("trades", "recv_ts"),
            ("book_l20", "recv_ts"),
            ("quality", "ts"),
        )
    )
    print(f"rows        {rows}")
    manifest = read_manifest(Path(a.day))
    if manifest is not None:
        print(f"problems    {_problems(manifest.quality)}")
    if manifest is not None and manifest.raw.get("covered_ns"):
        # time with a stream connection open: not the hours between two runs
        covered = manifest.raw["covered_ns"] / SEC
        what = "with a connection open"
    else:
        covered = sum(r.last_recv_ts - r.first_recv_ts + 1 for r in day.runs) / SEC
        what = "from the first row to the last"
    usable = _seconds(day.intervals)
    with_late = Day.open(a.day, include_late=True)
    late = _seconds(with_late.intervals) - _seconds(Day.open(a.day).intervals)
    holes = max(covered - _seconds(with_late.intervals), 0.0)
    share = usable / covered if covered > 0 else 0.0
    print(
        f"usable      {usable:.1f} s of {covered:.1f} s {what} ({share:.2%}), in "
        f"{len(day.intervals)} segments"
    )
    print(
        f"excluded    {holes:.1f} s without a valid book, "
        f"{late:.1f} s late{'' if not a.late else ' (kept: --late)'}"
    )
    if day.intervals:
        longest = max(day.intervals, key=lambda iv: iv[1] - iv[0])
        print(f"longest     {(longest[1] - longest[0]) / SEC:.1f} s from {_clock(longest[0])} UTC")
    return 0


def cmd_bars(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    table = bars.compare(day, a.every)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(table)
    print(
        "thresholds are calibrated on this day to the number of time bars: a description "
        "of the day, not an input for a model"
    )
    return 0


def cmd_features(a: argparse.Namespace) -> int:
    day = Day.open(a.day, include_late=a.late)
    train = [Day.open(p, include_late=a.late) for p in a.train]
    methods = []
    for m in default_methods():
        if m.name in FITTED:
            if not train:
                print(f"skip        {m.name}: needs --train days to fit on")
                continue
            m.fit(train)
        methods.append(m)
    table = compute(day, methods)
    print(f"grid        {table.height} rows")
    print(describe(table))
    if a.out:
        table.write_parquet(a.out)
        print(f"written     {a.out}")
    return 0


def _latency(a: argparse.Namespace) -> tuple[str, str]:
    """The latency to use and where it came from: given, or from the day's manifest as
    half the median round trip plus the median lag of the data."""
    if a.latency:
        return a.latency, "given"
    m = read_manifest(Path(a.day))
    rtt = m.raw.get("rtt_p50_ms") if m else None
    lag = m.raw.get("lag_p50_ms") if m else None
    if rtt is None or lag is None:
        raise ContractError(
            "the manifest has no round trip or lag to derive the latency from: give --latency"
        )
    ms = rtt / 2 + lag
    return f"{ms:g}ms", f"RTT p50 {rtt} ms / 2 + lag p50 {lag} ms, from the manifest"


def cmd_execution(a: argparse.Namespace) -> int:
    from .execution import run, summary

    day = Day.open(a.day, include_late=a.late)
    latency, source = _latency(a)
    qty = int(Decimal(a.size) / day.qty_step)
    if qty <= 0:
        raise ContractError(f"--size {a.size} is less than one quantity step {day.qty_step}")
    print(f"latency     {latency} ({source})")
    print(
        f"orders      at the best bid and the best ask every {a.every}, {a.size} each, "
        f"post-only, cancelled after {a.cancel_after}"
        + (f"; market orders of {a.size} as well" if a.aggressive else "")
    )
    orders, fills = run(
        day,
        every=a.every,
        qty=qty,
        cancel_after=a.cancel_after,
        latency=latency,
        aggressive=a.aggressive,
    )
    table = summary(orders, fills)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(table)
    print(
        "markouts in basis points, positive when the fill was good for us; conclusions "
        "from the pessimistic row only"
    )
    if a.out:
        fills.write_parquet(a.out)
        print(f"written     {a.out}")
    return 0


def _repo_dir(spec: Path, name: str, given: str | None) -> Path:
    """`given`, or `name` at the top of the git repository holding the spec."""
    if given:
        return Path(given)
    import subprocess

    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=spec.resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(top or ".") / name


def print_report(r) -> None:
    j = r.to_json()
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    print(f"experiment  {r.spec.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    days = [d.date for d in r.days]
    span = f"{days[0]} .. {days[-1]}" if days else "none"
    print(f"days        {len(days)}, {span}; held out: {', '.join(r.spec.holdout) or 'none'}")
    if r.skipped:
        short = ", ".join(f"{d.date} {d.usable_hours:.1f} h" for d in r.skipped)
        print(f"skipped     under {r.spec.min_hours:g} usable hours: {short}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    k = r.kfold
    print(
        f"k-fold      IC {k.mean:.4f} +- {k.se:.4f} (Newey-West, {r.lags} lags), "
        f"p {k.p:.2g}, positive on {k.positive_share:.0%} of {k.days} days"
    )
    c = j["cpcv"]
    print(
        f"cpcv        {c['paths']} paths: mean {c['mean']:.4f}, std {c['std']:.4f}, "
        f"min {c['min']:.4f}, positive {c['positive_share']:.0%}"
    )
    for kind, s in r.surrogate.items():
        print(
            f"surrogates  {kind:<6} real {s['real']:.4f}, {s['percentile']:g}th percentile "
            f"of {s['n']}: {s['threshold']:.4f}, p {s['p']:.3g}"
        )
    if r.pbo:
        print(
            f"pbo         {r.pbo.pbo:.2f} over {len(r.spec.variants)} variants "
            f"({r.pbo.combinations} combinations)"
        )
    for name, ic in r.variants.items():
        print(f"variant     {name}: IC {ic:.4f}" if ic is not None else f"variant     {name}: -")
    st = r.stability

    def fmt(d: dict) -> str:
        return ", ".join(f"{k} {v:.4f}" if v is not None else f"{k} -" for k, v in d.items())

    print(f"by time     {fmt(st['time_of_day'])}")
    print(f"by vol      {fmt(st['volatility_tercile'])}")
    print(f"p, BH       {r.p_bh:.3g} with every hypothesis in the registry")
    reasons = f": {'; '.join(r.reasons)}" if r.reasons else ""
    print(f"verdict     {r.verdict}{reasons}")


def kind_of(spec_path: Path) -> str:
    """A hypothesis's kind: `forecast` (the default), `maker`, `volatility`, `tradeflow`,
    `screen`, `edge` or `carry`."""
    import yaml

    raw = yaml.safe_load(spec_path.read_text())
    return str(raw.get("kind", "forecast")) if isinstance(raw, dict) else "forecast"


def print_screen_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    s = r.spec
    print(f"experiment  {s.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    span = f"{r.days[0]} .. {r.days[-1]}" if r.days else "none"
    print(
        f"days        {len(r.days)} development, {span}; "
        f"held out: {s.holdout[0]} .. {s.holdout[-1]} ({len(s.holdout)})"
    )
    if r.missing:
        print(f"missing     {', '.join(r.missing)}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    print(f"grid        {len(s.combos)} combinations, every {s.sampling}; rank IC by day")
    t = pl.DataFrame(r.table).sort("mean_ic", descending=True, nulls_last=True)
    with _wide():
        print(t.with_columns(pl.col(pl.Float64).round(4)))
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over the whole grid")
    for c in r.passed_over:
        print(f"refused     {c['combination']}: not above its surrogates")
    for k, c in enumerate(r.chosen, 1):
        sur = ", ".join(f"{kind} {x['threshold']:.4f}" for kind, x in c["surrogates"].items())
        print(
            f"chosen {k}    {c['combination']}: IC {c['mean_ic']:.4f}, positive on "
            f"{c['positive_days']:.0%} of days; surrogates' {s.surrogate_percentile:g}th: {sur}"
        )
    print(f"verdict     {r.verdict}" + (f": {'; '.join(r.reasons)}" if r.reasons else ""))


def print_carry_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    s = r.spec
    print(f"experiment  {s.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    print(
        f"period      {s.first} .. {s.last}: {r.settlements} funding settlements; held out: "
        f"{s.holdout_first} .. {s.holdout_last}"
    )
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    if r.filled:
        print(f"prices      {r.filled} taken from the bar before (no bar at the settlement)")
    print(
        f"position    long spot, short the perpetual, while the mean funding rate of the last "
        f"settlements is above the threshold; costs {s.spot_bp:g} + {s.perp_bp:g} bp a leg, "
        f"in and out"
    )
    cols = ["rule", "mean_bp", "annual_pct", "sharpe_months", "max_drawdown_bp", "held", "entries",
            "funding_bp_year", "basis_bp_year", "fees_bp_year", "positive_months", "p"]  # fmt: skip
    rows = [{k: x[k] for k in cols} for x in r.table]
    rows.append({k: r.always.get(k) for k in cols} | {"rule": "always held"})
    t = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3))
    with _wide():
        print(t)
    print(f"primary     {s.primary.name} (first row); mean_bp a day, net of costs")
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over the {len(s.variants)} rules, by month")
    print(f"p (BH)      {r.p_bh:.3g} over every hypothesis written")
    print(f"verdict     {r.verdict}" + (f": {'; '.join(r.reasons)}" if r.reasons else ""))


def cmd_carry_report(a: argparse.Namespace) -> int:
    from . import carry

    spec_path = Path(a.spec)
    spec = carry.load_spec(spec_path)
    results = _repo_dir(spec_path, "results", a.results)
    rf = carry.read_riskfree(a.riskfree) if a.riskfree else None
    rep = carry.report(spec, a.data, results, a.leverage, rf, a.mmr)
    for name, r in rep.items():
        print(
            f"\n{name:<11} {r['first']} .. {r['last']}, {r['rule']}; capital "
            f"{r['capital']:g} x the notional (the spot leg and the short's margin at leverage "
            f"{r['leverage']:g})"
        )
        t = pl.DataFrame([*r["years"], r["all"]], infer_schema_length=None)
        with _wide():
            print(t.with_columns(pl.col(pl.Float64).round(2)))
        over = r["sharpe_months_over_bills"]
        print(
            f"sharpe      from months: {r['sharpe_months']:.2f}"
            + (f"; on the capital, over the bills: {over:.2f}" if over is not None else "")
        )
        worst = ", ".join(f"{w['month']} {w['net_bp']:+.0f} bp" for w in r["worst_months"])
        print(f"worst       months: {worst}")
        rises = ", ".join(f"{k} {v:.1%}" for k, v in r["rises"].items())
        lev = ", ".join(f"{k} {v:.2f}x" for k, v in r["survives"].items())
        print(f"rises       the perpetual's largest rise while held, within: {rises}")
        print(
            f"leverage    the short leg survives that with its margin topped up every: {lev} "
            f"(maintenance margin {r['mmr_pct']:g}%)"
        )
    if "held out" not in rep:
        print("\nheld out    not opened yet: `marketlab experiment run SPEC --final` first")
    print(
        "\ncarry_pct, always_pct: % a year of the notional, the primary rule and holding always; "
        "bills_pct: the risk-free rate given (--riskfree); on_capital_pct: the primary's return "
        "on the capital it needs; over_bills_pct: that less the bills. Not modelled: the "
        "exchange's own risk, auto-deleveraging, slippage, moving margin between the legs"
    )
    return 0


def cmd_xsection(a: argparse.Namespace) -> int:
    import polars.selectors as cs

    from . import xsection as xs

    try:
        p, results = xs.report(a.data, a.first, a.until, a.top, a.q, a.cost, tuple(a.holds))
    except xs.HeldOut as e:
        raise ContractError(str(e)) from None
    if not p.symbols:
        raise ContractError(f"no daily bars under {a.data}/um/klines1d: scripts/fetch_universe.py")
    print(
        f"{len(p.symbols)} perpetuals under {a.data}, {a.first} .. {a.until}; each day the "
        f"{a.top} with the most quote volume over 30 days; long the top {a.q:.0%} by the "
        f"signal, short the bottom {a.q:.0%}; a round trip {a.cost:g} bp; funding in the P&L; "
        f"left out, not coins: {', '.join(sorted(xs.NOT_COINS))}"
    )
    if p.unbarred:
        most = sorted(p.unbarred.items(), key=lambda kv: (-kv[1], kv[0]))
        print(
            f"{len(most)} perpetuals settled funding on {sum(p.unbarred.values())} days with no "
            f"daily bar: Binance serves no bars for those days (an earlier listing of a coin "
            f"listed again, for one), so they are missing from the universe; most: "
            + ", ".join(f"{s} {n}" for s, n in most[:8])
        )
    rows = [
        {
            "signal": r.signal,
            "hold_days": r.hold,
            "ic": round(r.ic, 4),
            "ic_t": round(r.ic_t, 2),
            "gross_pct": round(r.gross_pct, 2),
            "cost_pct": round(r.cost_pct, 2),
            "net_pct": round(r.net_pct, 2),
            "sharpe_months": round(r.sharpe_months, 2),
            "max_dd_pct": round(r.max_drawdown_pct, 2),
            "pos_months": round(r.positive_months, 2),
            "turnover": round(r.turnover, 2),
            "names_side": round(r.names, 1),
        }
        for r in results
    ]
    with _wide():
        print(pl.DataFrame(rows).sort("net_pct", descending=True))
    print(
        "ic: mean daily rank IC of the signal against the next hold_days' return of a long "
        "(funding included), ic_t its t (Newey-West); gross/cost/net_pct: % a year of the "
        "capital, half long, half short; turnover: weight traded at a rebalance; names_side: "
        "positions a side. A rough look on the development years, not a hypothesis: nothing "
        f"after {xs.DEV_END} is read. "
        + (
            "The perpetuals gone from Binance's list are in, from its archive (um/gone.csv)"
            if (Path(a.data) / "um" / "gone.csv").exists()
            else "The perpetuals are those Binance lists now: coins gone from its list are "
            "missing (scripts/fetch_archive_universe.py fills them in)"
        )
    )
    for item in a.explain or []:
        name, _, hold = item.partition(":")
        if name not in xs.SIGNALS or not hold.isdigit() or int(hold) < 1:
            raise ContractError(f"--explain {item}: SIGNAL:HOLD, a signal of {xs.SIGNALS}")
        e = xs.explain(p, name, int(hold), a.top, a.q, a.cost, a.first)
        g = e.legs
        print(
            f"\n{name}, rebalanced every {hold} days, % a year: long {g['long_price']:+.1f} "
            f"price, {g['long_funding']:+.1f} funding; short {g['short_price']:+.1f} price, "
            f"{g['short_funding']:+.1f} funding; costs {g['costs']:+.1f}"
        )
        with _wide():
            print(pl.DataFrame(e.years).with_columns(cs.float().round(3)))
            print(pl.DataFrame(e.quintiles).with_columns(cs.float().round(1)))
            print(pl.DataFrame(e.coin_days).with_columns(cs.float().round(3)))
        print(
            f"fifths: the universe each day cut into fifths by the signal (5: highest), the "
            f"next day's return of a long, mean and median, bp. The {len(e.coin_days)} coin-days "
            f"with the largest contributions, % of the capital, best first, then the worst. "
            f"The 10 best: {e.best_share:.0%} of the gross; the net without them "
            f"{e.net_without_best:+.1f}% a year"
        )
    return 0


def cmd_costwall(a: argparse.Namespace) -> int:
    from . import costwall as cw

    rows = cw.wall(a.data, a.symbol, a.horizons, a.last_days, a.years, a.market)
    if not rows:
        raise ContractError(
            f"no {a.symbol} aggTrades days or 1h klines under {a.data}: scripts/fetch_binance.py"
        )
    k = cw.kappa(a.q)
    table = []
    for r in rows:
        row = {
            "horizon": r.horizon,
            "from": r.source,
            "returns": r.n,
            "sigma_bp": round(r.sigma_bp, 2),
            "mean_abs_bp": round(r.mean_abs_bp, 2),
        }
        for c in a.costs:
            need = r.needed(c, a.q)
            row[f"ic_for_{c:g}bp"] = f"{need:.3f}" if need <= 1 else ">1"
        table.append(row)
    print(
        f"{a.symbol}: the correlation a signal needs with the return over each horizon to pay "
        f"a round trip, trading its top and bottom {a.q:.0%} (a trade then earns "
        f"{k:.2f} x IC x sigma before costs)"
    )
    with _wide():
        print(pl.DataFrame(table))
    print(
        "sigma_bp: standard deviation of the log return over the horizon, bp; mean_abs_bp: its "
        "mean size. ic_for_Nbp: the IC needed against a round trip of N bp; >1: none can pay it. "
        "Returns and signal taken as jointly normal: a rough wall for where to look, not a "
        "verdict on any signal"
    )
    return 0


def print_edge_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    s = r.spec
    print(f"experiment  {s.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    span = f"{r.days[0]} .. {r.days[-1]}" if r.days else "none"
    print(
        f"days        {len(r.days)} development, {span}; "
        f"held out: {s.holdout[0]} .. {s.holdout[-1]} ({len(s.holdout)})"
    )
    if r.missing:
        print(f"missing     {len(r.missing)} days: {', '.join(r.missing)}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    print(
        f"rule        buy above the {s.quantile:g} quantile of the signal over the "
        f"{s.lookback} before, sell below the {1 - s.quantile:.3g}; filled {s.latency} later "
        f"at the last trade price, held for the horizon; every {s.sampling}"
    )
    print(f"cost        {s.cost_bp:g} bp a round trip; be_bp: the cost at which it breaks even")
    t = pl.DataFrame(r.table).sort("be_bp", descending=True, nulls_last=True)
    with _wide():
        print(
            t.select(
                "combination",
                pl.col("trades_day").round(1),
                pl.col("be_bp").round(3),
                pl.col("se").round(3),
                pl.col("p_cost").round(4),
                pl.col("positive_days").round(2),
                pl.col("hit_rate").round(3),
            )
        )
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over the whole grid")
    for c in r.passed_over:
        print(f"refused     {c['combination']}: not above its surrogates")
    for k, c in enumerate(r.chosen, 1):
        th = c["surrogates"]["shift"]["threshold"]
        print(
            f"chosen {k}    {c['combination']}: {c['be_bp']:.3f} bp a trade, above the cost "
            f"on {c['positive_days']:.0%} of days; surrogates' {s.surrogate_percentile:g}th: "
            f"{th:.3f}"
        )
    print(f"verdict     {r.verdict}" + (f": {'; '.join(r.reasons)}" if r.reasons else ""))


def print_tradeflow_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    s = r.spec
    print(f"experiment  {s.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    span = f"{r.days[0]} .. {r.days[-1]}" if r.days else "none"
    print(
        f"days        {len(r.days)} development, {span}; "
        f"held out: {s.holdout[0]} .. {s.holdout[-1]} ({len(s.holdout)})"
        if s.holdout
        else f"days        {len(r.days)}, {span}"
    )
    if r.missing:
        print(f"missing     {', '.join(r.missing)}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    print(
        f"feature     signed volume over {s.primary.text}; target: return over {s.horizon}, "
        f"every {s.sampling}"
    )
    m = r.summary
    print(
        f"IC          {m.mean:.4f} +- {m.se:.4f} (mean of days), p {m.p:.2g}, "
        f"positive on {m.positive_share:.0%} of {m.days} days"
    )
    for kind, x in r.surrogate.items():
        print(
            f"surrogates  {kind:<6} {x['percentile']:g}th percentile of {x['n']}: "
            f"{x['threshold']:.4f}, p {x['p']:.3g}"
        )
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over {len(s.variants)} variants")
    for name, v in r.variants.items():
        print(f"variant     {name}: {v:.4f}" if v is not None else f"variant     {name}: -")
    for label, part in r.stability.items():
        cells = ", ".join(f"{k} {v:.3f}" if v is not None else f"{k} -" for k, v in part.items())
        print(f"stability   {label.replace('_', ' ')}: {cells}")
    print(f"p (BH)      {r.p_bh:.3g} over every hypothesis written")
    print(f"verdict     {r.verdict}" + (f": {'; '.join(r.reasons)}" if r.reasons else ""))


def print_maker_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    print(f"experiment  {r.spec.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    days = [d.date for d in r.days]
    span = f"{days[0]} .. {days[-1]}" if days else "none"
    print(f"days        {len(days)}, {span}; held out: {', '.join(r.spec.holdout) or 'none'}")
    if r.skipped:
        short = ", ".join(f"{d.date} {d.usable_hours:.1f} h" for d in r.skipped)
        print(f"skipped     under {r.spec.min_hours:g} usable hours: {short}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    s, h = r.spec, r.spec.horizon
    base = np.nanmean([x.base for x in r.by_day.values()])
    rule = np.nanmean([x.rule for x in r.by_day.values()])
    kept = np.nanmean([x.fills_kept for x in r.by_day.values()])
    print(
        f"orders      every {s.every} each side, {s.size} base, cancelled after "
        f"{s.cancel_after}, latency {s.latency}, {s.queue} queue"
    )
    print(
        f"markout {h:<4}without the rule {base:+.3f} bp, with it {rule:+.3f} bp "
        f"(mean of days); the rule keeps {kept:.0%} of the filled quantity"
    )
    print(
        f"difference  {r.mean_delta:+.4f} +- {r.se:.4f} bp, p {r.p:.2g}, "
        f"positive on {r.positive_share:.0%} of {len(r.by_day)} days"
    )
    sh = r.surrogate["shift"]
    if sh["threshold"] is None:
        print("surrogates  shift  none: no day long enough to shift the signal")
    else:
        print(
            f"surrogates  shift  real {sh['real']:+.4f}, {sh['percentile']:g}th percentile "
            f"of {sh['n']}: {sh['threshold']:+.4f}, p {sh['p']:.3g}"
        )
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over {len(s.variants)} variants")
    for name, v in r.variants.items():
        print(f"variant     {name}: {v:+.4f} bp" if v is not None else f"variant     {name}: -")
    print("queue       " + ", ".join(f"{m} {v:+.4f}" for m, v in r.modes.items() if v is not None))
    st = r.stability

    def fmt(d: dict) -> str:
        return ", ".join(f"{k} {v:+.4f}" if v is not None else f"{k} -" for k, v in d.items())

    print(f"by time     {fmt(st['time_of_day'])}")
    print(f"by vol      {fmt(st['volatility_tercile'])}")
    print(
        f"break-even  a maker fee below {rule:+.3f} bp per fill (negative: a rebate of that "
        f"size) at {h}, before any other cost"
    )
    print(f"p, BH       {r.p_bh:.3g} with every hypothesis in the registry")
    reasons = f": {'; '.join(r.reasons)}" if r.reasons else ""
    print(f"verdict     {r.verdict}{reasons}")


def print_volatility_report(r) -> None:
    p = r.provenance
    code = p.head[:12] + (", with uncommitted changes" if p.dirty else "")
    print(f"experiment  {r.spec.slug}: spec from commit {p.spec_commit[:12]}, code {code}")
    days = [d.date for d in r.days]
    span = f"{days[0]} .. {days[-1]}" if days else "none"
    print(f"days        {len(days)}, {span}; held out: {', '.join(r.spec.holdout) or 'none'}")
    if r.skipped:
        short = ", ".join(f"{d.date} {d.usable_hours:.1f} h" for d in r.skipped)
        print(f"skipped     under {r.spec.min_hours:g} usable hours: {short}")
    if r.verdict == "insufficient data":
        print(f"verdict     {r.verdict}: {'; '.join(r.reasons)}")
        return
    s = r.spec
    print(
        f"target      log({s.floor_bp:g} bp + realized volatility over the next {s.horizon}), "
        f"every {s.sampling}; feature {s.primary.key}"
    )
    loss = [x for x in r.by_day.values() if np.isfinite(x.har)]
    print(
        "error       mean squared, mean of days: persistence "
        f"{np.mean([x.persistence for x in loss]):.4f}, HAR {np.mean([x.har for x in loss]):.4f}, "
        f"HAR with the feature {np.mean([x.candidate for x in loss]):.4f}"
    )
    print(
        f"gain        over HAR {r.mean_gain:+.4f} +- {r.se:.4f}, p {r.p:.2g}, positive on "
        f"{r.positive_share:.0%} of {len(loss)} days; "
        f"over persistence {r.mean_gain_persistence:+.4f}"
    )
    sh = r.surrogate["shift"]
    if sh["threshold"] is None:
        print("surrogates  shift  none: no day long enough to shift the feature")
    else:
        print(
            f"surrogates  shift  real {sh['real']:+.4f}, {sh['percentile']:g}th percentile "
            f"of {sh['n']}: {sh['threshold']:+.4f}, p {sh['p']:.3g}"
        )
    if r.pbo:
        print(f"pbo         {r.pbo.pbo:.2f} over {len(s.variants)} variants")
    for name, v in r.variants.items():
        print(f"variant     {name}: {v:+.4f}" if v is not None else f"variant     {name}: -")
    st = r.stability

    def fmt(d: dict) -> str:
        return ", ".join(f"{k} {v:+.4f}" if v is not None else f"{k} -" for k, v in d.items())

    print(f"by time     {fmt(st['time_of_day'])}")
    print(f"by vol      {fmt(st['volatility_tercile'])}")
    diag = [d.get(s.primary.key) for d in r.diagnostics.values() if d.get(s.primary.key)]
    if diag:
        ks = [d["ks_pass_share"] for d in diag if d["ks_pass_share"] is not None]
        br = [d["branching_median"] for d in diag if d["branching_median"] is not None]
        print(
            f"hawkes      {sum(d['fits'] for d in diag)} fits; the time-rescaling check accepts "
            f"{np.mean(ks):.0%} of them; branching ratio, median of days {np.median(br):.2f}"
        )
    print(f"p, BH       {r.p_bh:.3g} with every hypothesis in the registry")
    reasons = f": {'; '.join(r.reasons)}" if r.reasons else ""
    print(f"verdict     {r.verdict}{reasons}")


def cmd_experiment_run(a: argparse.Namespace) -> int:
    from . import experiment as ex

    spec_path = Path(a.spec)
    kind = kind_of(spec_path)
    if kind == "volatility":
        from . import volatility

        spec = volatility.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        cache = _repo_dir(spec_path, ".cache", a.cache)
        if a.final:
            rec = volatility.final(spec, a.data, results, cache)
            print(f"held out    {', '.join(rec['days'])}: opened once, now recorded")
            g = rec["mean_gain"]
            print(
                f"gain        over HAR {g:+.4f} (mean of days)"
                if g is not None
                else "gain        -"
            )
            for d, x in rec["by_day"].items():
                print(
                    f"            {d}: {x['gain']:+.4f}"
                    if x and x["gain"] is not None
                    else f"            {d}: -"
                )
            return 0
        report = volatility.run(spec, a.data, results, cache, workers=a.workers)
        print_volatility_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    if kind == "screen":
        from . import screen

        spec = screen.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        cache = _repo_dir(spec_path, ".cache", a.cache)
        if a.final:
            rec = screen.final(spec, a.data, results, cache, workers=a.workers)
            print(f"held out    {', '.join(rec['days'])}: opened once, now recorded")
            for c in rec["chosen"]:
                ic = c["mean_ic"]
                print(
                    f"            {c['combination']}: IC {ic:.4f} +- {c['se']:.4f}, "
                    f"p (BH over {len(rec['chosen'])}) {c['p_bh']:.3g}, positive on "
                    f"{c['positive_days']:.0%} of days: "
                    + ("confirmed" if c["confirmed"] else "not confirmed")
                    if ic is not None
                    else f"            {c['combination']}: -"
                )
            return 0
        report = screen.run(spec, a.data, results, cache, workers=a.workers)
        print_screen_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    if kind == "carry":
        from . import carry

        spec = carry.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        if a.final:
            rec = carry.final(spec, a.data, results)
            x = rec["summary"]
            first, last = rec["period"]
            print(f"held out    {first} .. {last}: opened once, now recorded")
            print(
                f"            {rec['rule']}: {x['mean_bp']:.3f} +- {x['se']:.3f} bp a day, "
                f"{x['annual_pct']:.2f}% a year, Sharpe {x['sharpe_months']:.2f} from months, "
                f"max drawdown "
                f"{x['max_drawdown_bp']:.0f} bp, held {x['held']:.0%}, p {x['p']:.3g}: "
                + ("confirmed" if rec["confirmed"] else "not confirmed")
            )
            a_ = rec["always_held"]
            print(f"            always held: {a_['annual_pct']:.2f}% a year, max drawdown "
                  f"{a_['max_drawdown_bp']:.0f} bp")  # fmt: skip
            return 0
        report = carry.run(spec, a.data, results)
        print_carry_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    if kind == "edge":
        from . import edge

        spec = edge.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        cache = _repo_dir(spec_path, ".cache", a.cache)
        if a.final:
            rec = edge.final(spec, a.data, results, cache, workers=a.workers)
            print(f"held out    {len(rec['days'])} days: opened once, now recorded")
            if rec["missing"]:
                print(f"missing     {len(rec['missing'])} days: {', '.join(rec['missing'])}")
            for c in rec["chosen"]:
                be = c["be_bp"]
                print(
                    f"            {c['combination']}: {be:.3f} +- {c['se']:.3f} bp a trade "
                    f"against a cost of {rec['cost_bp']:g}, p (BH over {len(rec['chosen'])}) "
                    f"{c['p_bh']:.3g}, above the cost on {c['positive_days']:.0%} of days: "
                    + ("confirmed" if c["confirmed"] else "not confirmed")
                    if be is not None and c["se"] is not None
                    else f"            {c['combination']}: -"
                )
            return 0
        report = edge.run(spec, a.data, results, cache, workers=a.workers)
        print_edge_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    if kind == "tradeflow":
        from . import tradeflow

        spec = tradeflow.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        cache = _repo_dir(spec_path, ".cache", a.cache)
        if a.final:
            rec = tradeflow.final(spec, a.data, results, cache)
            m = rec["summary"]
            print(f"held out    {', '.join(rec['days'])}: opened once, now recorded")
            if rec["missing"]:
                print(f"missing     {', '.join(rec['missing'])}")
            print(
                f"IC          {m['mean']:.4f} +- {m['se']:.4f}, p {m['p']:.2g}, "
                f"positive on {m['positive_share']:.0%} of days"
                if m["mean"] is not None
                else "IC          -"
            )
            for d, ic in rec["ic_by_day"].items():
                print(f"            {d}: {ic:.4f}" if ic is not None else f"            {d}: -")
            return 0
        report = tradeflow.run(spec, a.data, results, cache, workers=a.workers)
        print_tradeflow_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    if kind == "maker":
        from . import maker

        spec = maker.load_spec(spec_path)
        results = _repo_dir(spec_path, "results", a.results)
        cache = _repo_dir(spec_path, ".cache", a.cache)
        if a.final:
            rec = maker.final(spec, a.data, results, cache)
            print(f"held out    {', '.join(rec['days'])}: opened once, now recorded")
            print(f"difference  {rec['mean_delta_bp']:+.4f} bp (mean of days)")
            for d, x in rec["by_day"].items():
                print(
                    f"            {d}: {x['delta']:+.4f}"
                    if x["delta"] is not None
                    else f"            {d}: -"
                )
            return 0
        report = maker.run(spec, a.data, results, cache, workers=a.workers)
        print_maker_report(report)
        print(f"written     {results / (spec.slug + '.json')}")
        return 0
    spec = ex.load_spec(spec_path)
    results = _repo_dir(spec_path, "results", a.results)
    cache = _repo_dir(spec_path, ".cache", a.cache)
    if a.final:
        rec = ex.final(spec, a.data, results, cache)
        s = rec["summary"]
        print(f"held out    {', '.join(rec['days'])}: opened once, now recorded")
        print(f"IC          {s['mean']:.4f} +- {s['se']:.4f}, p {s['p']:.2g}")
        for d, ic in rec["ic_by_day"].items():
            print(f"            {d}: {ic:.4f}" if ic is not None else f"            {d}: -")
        return 0
    report = ex.run(spec, a.data, results, cache, workers=a.workers)
    print_report(report)
    print(f"written     {results / (spec.slug + '.json')}")
    return 0


def cmd_experiment_list(a: argparse.Namespace) -> int:
    from . import experiment as ex

    table = ex.registry_table(Path(a.results), Path(a.experiments))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(table)
    return 0


def _fee(text: str) -> float:
    return float(text)


def cmd_strategy(a: argparse.Namespace) -> int:
    from dataclasses import replace

    from . import strategy as st
    from .execution import Mode
    from .io.days import survey

    base = st.Quoting(
        size=Decimal(a.size),
        life=a.life,
        max_position=a.max_position,
        latency=a.latency,
        window=None if a.signal == "none" else a.signal,
        threshold=a.threshold,
        vol_lookback=a.vol_lookback,
    )
    gates: list[float | None] = []
    for text in a.vol_quantiles or (["0.8"] if a.vol_window else ["none"]):
        gates.append(None if text == "none" else float(text))
    if any(g is not None for g in gates) and not a.vol_window:
        raise ContractError("--vol-quantiles needs --vol-window, e.g. --vol-window 1m")
    rules = [
        replace(base, offset_bp=o)
        if g is None
        else replace(base, offset_bp=o, vol_window=a.vol_window, vol_quantile=g)
        for o in a.offsets
        for g in gates
    ]
    found = survey(a.data, a.symbol, exclude=a.exclude)
    if a.days:
        found = [d for d in found if d.date in set(a.days)]
        missing = sorted(set(a.days) - {d.date for d in found})
        if missing:
            raise ContractError(f"no finished days {', '.join(missing)} under {a.data}")
    days = [d for d in found if d.usable_hours >= a.min_hours]
    short = [d for d in found if d.usable_hours < a.min_hours]
    if not days:
        raise ContractError(f"no {a.symbol} day under {a.data} with {a.min_hours:g} usable hours")
    for q in rules:
        print(f"rule        {q.name}")
    print(f"days        {len(days)}, {days[0].date} .. {days[-1].date}")
    if short:
        text = ", ".join(f"{d.date} {d.usable_hours:.1f} h" for d in short)
        print(f"skipped     under {a.min_hours:g} usable hours: {text}")
    rows, summary, nets = [], [], []
    for q in rules:
        for name in a.modes:
            mode = Mode(name)
            results = st.run([d.path for d in days], q, mode, workers=a.workers)
            t = st.table(results).with_columns(
                hours=pl.Series([d.usable_hours for d in days]),
                offset_bp=pl.lit(q.offset_bp),
                vol_quantile=pl.lit(q.vol_quantile if q.vol_window else None, pl.Float64),
                rule=pl.lit(q.name),
            )
            rows.append(t)
            s = st.summarize(results)
            gate = f"{q.vol_quantile:g}" if q.vol_window else "none"
            summary.append(
                {
                    "offset_bp": q.offset_bp,
                    "gate": gate,
                    "mode": str(mode),
                    "fills_day": round(s.fills_per_day),
                    "traded_day": round(s.notional_per_day),
                    "gross_day": round(s.gross_per_day, 2),
                    "captured_bp": round(s.captured_bp, 3),
                    "inventory_bp": round(s.inventory_bp, 3),
                    "be_bp": round(s.break_even_bp, 3),
                    "be_se": round(s.break_even_se, 3),
                    "gross_pos": round(s.positive_days, 2),
                    "unseen": round(float(np.mean([r.unseen for r in results])), 2),
                    "quoted": round(float(t.select(
                        (pl.col("quoted_bid") + pl.col("quoted_ask")) / 2).mean().item()), 2),
                }
            )  # fmt: skip
            nets.append(
                {"offset_bp": q.offset_bp, "gate": gate, "mode": str(mode)}
                | {f"fee {f:g}": round(s.net_per_day(f), 2) for f in a.fees}
            )
            if a.by_day:
                gated = f", volatility gate {gate}" if q.vol_window else ""
                print(f"\n{q.offset_bp:g} bp from the mid{gated}, {mode} queue")
                shown = t.select(
                    "date",
                    pl.col("hours").round(1),
                    "fills",
                    pl.col("notional").round(0),
                    pl.col("gross").round(2),
                    pl.col("captured").round(2),
                    pl.col("inventory").round(2),
                    pl.col("break_even_bp").round(3).alias("be_bp"),
                    pl.col("end_position").alias("end_pos"),
                    pl.col("max_position").alias("max_pos"),
                    ((pl.col("quoted_bid") + pl.col("quoted_ask")) / 2).round(2).alias("quoted"),
                    pl.col("unseen").round(2),
                )
                with _wide():
                    print(shown)
    with _wide():
        print("\nper day, mean over days (quote currency, before fees); per traded notional, bp")
        print(pl.DataFrame(summary))
        print(
            "\nnet a day (quote currency) at each maker fee, bp of traded notional; "
            "negative: a rebate"
        )
        print(pl.DataFrame(nets))
    print(
        "\nbe_bp: the maker fee at which it breaks even, mean of days, be_se its standard "
        "error; gross_pos: share of days with a positive gross P&L; gate: the quantile of "
        "the volatility above which no order is out (none: no gate); unseen: share of orders "
        "at a level the book did not show (rested behind all of it); quoted: share of the "
        "time with an order in the book. Conclusions from the pessimistic queue; every order "
        "goes through the simulator alone (the market does not react to it)"
    )
    if a.out:
        out = pl.concat(rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        if a.out.endswith(".csv"):
            out.write_csv(a.out)
        else:
            out.write_parquet(a.out)
        print(f"written     {a.out}")
    return 0


def _wide() -> pl.Config:
    return pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    )  # fmt: skip


def cmd_archive_check(a: argparse.Namespace) -> int:
    from . import archive

    days = archive.find(a.root, a.symbol.upper(), a.kind, a.market)
    if not days:
        raise ContractError(
            f"no {a.symbol.upper()} {a.kind} files under {Path(a.root) / a.market / a.kind}"
        )
    result = archive.check(days)
    units = sorted({d.unit for d in result.days})
    print(
        f"archive     {a.symbol.upper()} {a.market} {a.kind}: {len(days)} days, "
        f"{days[0].date} .. {days[-1].date}; times in {', '.join(units)}"
    )
    with _wide():
        print(archive.table(result))
    rows = sum(d.rows for d in result.days)
    trades = sum(d.trades for d in result.days)
    print(f"rows        {rows:,} ({trades:,} raw trades)")
    if result.missing_dates:
        print(f"MISSING     {', '.join(result.missing_dates)}")
    for b in result.breaks:
        print(f"BREAK       {b}")
    bad = [d for d in result.days if d.problems]
    for d in bad:
        print(f"PROBLEM     {d.date}: {'; '.join(d.problems)}")
    print("verdict     " + ("ok" if result.ok else "problems above: fix or leave those days out"))
    return 0 if result.ok else 1


def cmd_synth(a: argparse.Namespace) -> int:
    s = synth.Scenario(
        seed=a.seed,
        seconds=a.seconds,
        gaps=[(a.seconds * 0.3, 5.0)],
        late=[(a.seconds * 0.6, 3.0)],
    )
    path = synth.write(s, Path(a.out))
    print(f"written     {path} (synthetic: a toy market, not real data)")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marketlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def day_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("day", help="day directory, e.g. data/binance/BTCUSDT/2026-11-21")
        sp.add_argument(
            "--late", action="store_true", help="keep intervals where the data was late"
        )

    d = sub.add_parser("day", help="what a day holds and how much of it is usable")
    day_arg(d)
    d.set_defaults(run=cmd_day)

    b = sub.add_parser("bars", help="time bars against tick, volume and dollar bars")
    day_arg(b)
    b.add_argument("--every", default="1s", help="time bar length (default 1s)")
    b.set_defaults(run=cmd_bars)

    f = sub.add_parser("features", help="layer 1 features on a day, summarized")
    day_arg(f)
    f.add_argument("--train", nargs="*", default=[], help="days to fit fitted methods on")
    f.add_argument("--out", help="write the features to this Parquet file")
    f.set_defaults(run=cmd_features)

    e = sub.add_parser("execution", help="probe orders through the fill simulator: fills, markouts")
    day_arg(e)
    e.add_argument("--every", default="10s", help="one order per side this often (default 10s)")
    e.add_argument("--size", default="0.001", help="order size in base units (default 0.001)")
    e.add_argument("--cancel-after", default="10s", help="passive orders' life (default 10s)")
    e.add_argument("--latency", help="decision to effect, e.g. 300ms (default: from the manifest)")
    e.add_argument("--aggressive", action="store_true", help="probe market orders as well")
    e.add_argument("--out", help="write the fills with their markouts to this Parquet file")
    e.set_defaults(run=cmd_execution)

    x = sub.add_parser("experiment", help="run a hypothesis through the protocol")
    xs = x.add_subparsers(dest="action", required=True)
    xr = xs.add_parser("run", help="run a committed experiments/NNN-name.yaml")
    xr.add_argument("spec")
    xr.add_argument("--data", required=True, help="data root of marketrec, e.g. ../marketrec/data")
    xr.add_argument("--results", help="default: results/ next to experiments/")
    xr.add_argument("--cache", help="per-day inputs; default: .cache/ next to experiments/")
    xr.add_argument(
        "--final", action="store_true", help="open the held-out days, once, after the run"
    )
    xr.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="processes for the surrogates (default: all cores but one)",
    )
    xr.set_defaults(run=cmd_experiment_run)
    xl = xs.add_parser("list", help="every recorded hypothesis, with p adjusted for all")
    xl.add_argument("--results", default="results")
    xl.add_argument("--experiments", default="experiments")
    xl.set_defaults(run=cmd_experiment_list)

    g = sub.add_parser(
        "strategy", help="quote both sides through the fill simulator: P&L and break-even fee"
    )
    g.add_argument("--data", required=True, help="data root of marketrec, e.g. ../marketrec/data")
    g.add_argument("--symbol", default="BTCUSDT")
    g.add_argument("--days", nargs="*", default=[], help="only these dates (default: all)")
    g.add_argument("--exclude", nargs="*", default=[], help="leave these dates out")
    g.add_argument("--min-hours", type=float, default=20.0, help="usable hours a day (default 20)")
    g.add_argument("--size", default="0.001", help="order size in base units (default 0.001)")
    g.add_argument("--life", default="10s", help="an order rests at most this long (default 10s)")
    g.add_argument("--max-position", type=int, default=5, help="in orders, either side (default 5)")
    g.add_argument("--latency", default="300ms", help="decision to effect (default 300ms)")
    g.add_argument(
        "--signal",
        default="none",
        help="OFI window to withdraw orders on, e.g. 1s; none: no signal (default)",
    )
    g.add_argument("--threshold", type=float, default=1.0, help="the signal's (default 1.0)")
    g.add_argument(
        "--modes",
        nargs="+",
        default=["pessimistic"],
        choices=["pessimistic", "proportional", "optimistic"],
        help="queue modes (default: pessimistic)",
    )
    g.add_argument(
        "--fees",
        nargs="+",
        type=_fee,
        default=[-1.0, 0.0, 1.0, 2.0, 7.5],
        help="maker fees to show, bp; negative: rebate (default -1 0 1 2 7.5)",
    )
    g.add_argument(
        "--offsets",
        nargs="+",
        type=float,
        default=[0.0],
        help="distances of the orders from the mid, bp, one run each; 0: at the best prices",
    )
    g.add_argument(
        "--vol-window",
        help="volatility gate: realized volatility of the trade price over this, e.g. 1m; "
        "no orders while it is above its quantile (default: no gate)",
    )
    g.add_argument(
        "--vol-lookback", default="60m", help="the gate's quantile over this before (default 60m)"
    )
    g.add_argument(
        "--vol-quantiles",
        nargs="+",
        help="the gate's quantiles, one run each; none: a run without the gate "
        "(default 0.8 with --vol-window)",
    )
    g.add_argument("--by-day", action="store_true", help="a table of the days for each run")
    g.add_argument("--workers", type=int, default=1, help="days at a time (default 1)")
    g.add_argument("--out", help="per-day table to this .csv or .parquet file")
    g.set_defaults(run=cmd_strategy)

    xs_ = sub.add_parser(
        "xsection", help="cross-sectional signals on the perpetuals, development years only"
    )
    xs_.add_argument("--data", default="binance-data", help="default ./binance-data")
    xs_.add_argument("--first", default="2020-01-01")
    xs_.add_argument("--until", default="2023-12-31", help="no later than 2023-12-31")
    xs_.add_argument("--top", type=int, default=50, help="the universe each day (default 50)")
    xs_.add_argument("--q", type=float, default=0.2, help="share long and short (default 0.2)")
    xs_.add_argument("--cost", type=float, default=9.0, help="a round trip, bp (default 9)")
    xs_.add_argument("--holds", type=int, nargs="+", default=[1, 7], help="days between rebalances")
    xs_.add_argument(
        "--explain",
        nargs="*",
        metavar="SIGNAL:HOLD",
        help="where the money came from, e.g. carry:1",
    )
    xs_.set_defaults(run=cmd_xsection)

    w = sub.add_parser(
        "costwall", help="how strong a signal must be to pay a round trip, horizon by horizon"
    )
    w.add_argument("--data", default="binance-data", help="default ./binance-data")
    w.add_argument("--symbol", default="BTCUSDT")
    w.add_argument("--horizons", nargs="+", default=list(HORIZONS_COSTWALL))
    w.add_argument("--last-days", type=int, help="only the most recent aggTrades days")
    w.add_argument("--years", type=float, default=1.0, help="of hourly klines (default 1)")
    w.add_argument("--market", default="spot", choices=["spot", "um"], help="klines' market")
    w.add_argument(
        "--costs",
        nargs="+",
        type=float,
        default=[15.0, 9.0, 3.6],
        help="round trips, bp (default 15: spot taker, 9: USD-M taker, 3.6: USD-M maker fees "
        "alone; regular account paying in BNB)",
    )
    w.add_argument("--q", type=float, default=0.05, help="share traded at each end (default 0.05)")
    w.set_defaults(run=cmd_costwall)

    c = sub.add_parser(
        "carry-report",
        help="a carry hypothesis after its verdict: by year, against bills, on capital, margin",
    )
    c.add_argument("spec", help="a committed experiments/NNN-name.yaml of kind carry")
    c.add_argument("--data", required=True, help="e.g. binance-data")
    c.add_argument("--results", help="default: results/ next to experiments/")
    c.add_argument(
        "--leverage", type=float, default=2.0, help="of the short leg, for the capital (default 2)"
    )
    c.add_argument(
        "--riskfree",
        help="CSV of date and rate in %% a year, e.g. binance-data/fred/GS3M.csv from "
        "scripts/fetch_fred.py",
    )
    c.add_argument(
        "--mmr", type=float, default=0.4, help="maintenance margin, %% of notional (default 0.4)"
    )
    c.set_defaults(run=cmd_carry_report)

    r = sub.add_parser(
        "archive", help="Binance public trade archives from scripts/fetch_binance.py"
    )
    rs = r.add_subparsers(dest="action", required=True)
    rc = rs.add_parser("check", help="integrity of the downloaded days, before any research")
    rc.add_argument("root", nargs="?", default="binance-data", help="default ./binance-data")
    rc.add_argument("--symbol", default="BTCUSDT")
    rc.add_argument("--kind", default="aggTrades", choices=["aggTrades", "trades"])
    rc.add_argument("--market", default="spot", choices=["spot", "um", "cm"])
    rc.set_defaults(run=cmd_archive_check)

    s = sub.add_parser("synth", help="write a synthetic day to try the commands")
    s.add_argument("out", help="data root, e.g. /tmp/synth")
    s.add_argument("--seconds", type=float, default=600.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(run=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.run(a)
    except ContractError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .experiment import ExperimentError

        if isinstance(e, ExperimentError):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


assert set(METHODS) >= FITTED

if __name__ == "__main__":
    sys.exit(main())
