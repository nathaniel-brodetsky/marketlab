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
    """A hypothesis's kind: `forecast` (the default), `maker` or `volatility`."""
    import yaml

    raw = yaml.safe_load(spec_path.read_text())
    return str(raw.get("kind", "forecast")) if isinstance(raw, dict) else "forecast"


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
