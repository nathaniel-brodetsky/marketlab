"""Hypotheses of the `screen` kind: many trade-only signals at once, counted as one.

A screen names a grid of signals and targets on Binance's trade archives, a rule for
choosing among them, and how many to choose. The development run scores every combination
(rank IC by day, as in `tradeflow`), measures how overfit the choice of the best is (PBO
over the whole grid), and applies the rule blindly: the best by mean IC that are positive
on enough days and beat their surrogates, in that order, until it has taken as many as it
should. Only those are scored on the held-out days, once, with Benjamini-Hochberg over
them. Every combination tried is in the PBO, and the screen is one hypothesis in the
family, so trying many does not buy anything.

Signals, at each time of the grid, from trades at or before it only:

- direction (against the return over the horizon): `signed_volume` (buys minus sells),
  `signed_count` (buy orders minus sell orders), `size_skew` (mean buy size minus mean sell
  size), over a count of aggregate trades (`100t`) or a time (`30s`);
- activity (against the size of the return): `intensity` (aggregate trades per second),
  `volume_rate` (base units per second), `realized_vol` (of the trade price sampled each
  second), over a time; `hawkes` (log total intensity of taker orders, from two-sided
  Hawkes fits every `step` on the `window` before, `events.hawkes`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import __version__, archive
from . import experiment as ex
from .events import hawkes as hk
from .features import duration_ns
from .tradeflow import NS_DAY, Window, _dates
from .validation import metrics, pbo
from .validation.multiple import benjamini_hochberg

CACHE_VERSION = 1
SEC = 1_000_000_000
DIRECTION = ("signed_volume", "signed_count", "size_skew")
ACTIVITY = ("intensity", "volume_rate", "realized_vol", "hawkes")
COUNTABLE = ("signed_volume", "signed_count", "size_skew")  # may use a count of trades
_TARGET = re.compile(r"^(abs_)?return_(.+)$")


@dataclass(frozen=True)
class Signal:
    method: str
    window: str = ""  # a Window's text; for hawkes, its fitting window
    step: str = ""  # hawkes only

    @property
    def key(self) -> str:
        return (
            f"hawkes_{self.window}_{self.step}"
            if self.method == "hawkes"
            else f"{self.method}_{self.window}"
        )


@dataclass(frozen=True)
class Combo:
    signal: Signal
    target: str  # return_<h> or abs_return_<h>

    @property
    def name(self) -> str:
        return f"{self.signal.key} -> {self.target}"

    @property
    def horizon(self) -> str:
        m = _TARGET.match(self.target)
        assert m
        return m.group(2)


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    root_kind: str
    market: str
    symbol: str
    first: str
    last: str
    holdout: tuple[str, ...]
    min_days: int
    sampling: str
    combos: tuple[Combo, ...]
    take: int
    min_positive_days: float
    surrogate_percentile: float
    surrogates: int
    seed: int
    max_pbo: float
    alpha: float
    path: Path = field(compare=False)

    @property
    def slug(self) -> str:
        return f"{self.id}-{self.name}"

    @property
    def development(self) -> tuple[str, ...]:
        a, b = dt.date.fromisoformat(self.first), dt.date.fromisoformat(self.last)
        days = (a + dt.timedelta(days=k) for k in range((b - a).days + 1))
        return tuple(d.isoformat() for d in days if d.isoformat() not in self.holdout)

    @property
    def signals(self) -> tuple[Signal, ...]:
        return tuple(dict.fromkeys(c.signal for c in self.combos))

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.target for c in self.combos))


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ex.ExperimentError(f"{path}: {e}") from e
    if not isinstance(raw, dict) or raw.get("kind") != "screen":
        raise ex.ExperimentError(f"{path}: not a `screen` hypothesis")
    known = {
        "id",
        "name",
        "kind",
        "hypothesis",
        "data",
        "sampling",
        "grid",
        "selection",
        "protocol",
        "success",
    }
    if set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw) - known)}")
    data, sel = raw.get("data", {}), raw.get("selection", {})
    proto, ok, grid = raw.get("protocol", {}), raw.get("success", {}), raw.get("grid", {})
    days = _dates(ex._get(data, "days", "data"), "data.days")
    holdout = _dates(ex._get(data, "holdout", "data"), "data.holdout")
    if set(days) & set(holdout):
        raise ex.ExperimentError(f"{path}: held-out days inside the development days")
    combos: list[Combo] = []
    for group, allowed in (("direction", DIRECTION), ("activity", ACTIVITY)):
        part = grid.get(group)
        if not part:
            continue
        targets = [str(t) for t in ex._get(part, "targets", f"grid.{group}")]
        for t in targets:
            m = _TARGET.match(t)
            if not m or bool(m.group(1)) != (group == "activity"):
                raise ex.ExperimentError(f"{path}: target {t!r} in grid.{group}")
            duration_ns(m.group(2))
        for s in ex._get(part, "signals", f"grid.{group}"):
            method = str(ex._get(s, "method", f"grid.{group}"))
            if method not in allowed:
                raise ex.ExperimentError(f"{path}: {method!r} is not a {group} signal")
            if method == "hawkes":
                prm = ex._get(s, "params", "hawkes")
                sigs = [
                    Signal(
                        "hawkes",
                        str(ex._get(prm, "window", "hawkes")),
                        str(ex._get(prm, "step", "hawkes")),
                    )
                ]
                duration_ns(sigs[0].window), duration_ns(sigs[0].step)
            else:
                sigs = [Signal(method, str(w)) for w in ex._get(s, "windows", method)]
                for sg in sigs:
                    w = Window(sg.window)
                    if w.trades is None:
                        duration_ns(w.text)
                    elif method not in COUNTABLE or w.trades < 1:
                        raise ex.ExperimentError(f"{path}: {method} over {w.text}")
            combos += [Combo(sg, t) for sg in sigs for t in targets]
    if not combos:
        raise ex.ExperimentError(f"{path}: the grid is empty")
    if len(set(combos)) != len(combos):
        raise ex.ExperimentError(f"{path}: a combination is listed twice")
    req = sel.get("require", {})
    if sel.get("rank_by", "mean_ic") != "mean_ic":
        raise ex.ExperimentError(f"{path}: selection ranks by `mean_ic`")
    spec = Spec(
        id=str(ex._get(raw, "id", "top")),
        name=str(ex._get(raw, "name", "top")),
        hypothesis=" ".join(str(ex._get(raw, "hypothesis", "top")).split()),
        root_kind=str(data.get("kind", "aggTrades")),
        market=str(data.get("market", "spot")),
        symbol=str(ex._get(data, "symbol", "data")),
        first=days[0],
        last=days[-1],
        holdout=holdout,
        min_days=int(data.get("min_days", 20)),
        sampling=str(raw.get("sampling", "5s")),
        combos=tuple(combos),
        take=int(sel.get("take", 3)),
        min_positive_days=float(req.get("positive_days", 0.6)),
        surrogate_percentile=float(req.get("surrogates_above", 95)),
        surrogates=int(proto.get("surrogates", 200)),
        seed=int(proto.get("seed", 1)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        alpha=float(ok.get("alpha", 0.05)),
        path=path,
    )
    duration_ns(spec.sampling)
    if spec.surrogates < 200:
        raise ex.ExperimentError(
            f"{path}: the protocol asks for at least 200 surrogates of each kind"
        )
    if not 1 <= spec.take <= len(combos):
        raise ex.ExperimentError(f"{path}: selection.take must be from 1 to {len(combos)}")
    return spec


# --- one day's inputs --------------------------------------------------------------------


def _window_sum(
    cum: np.ndarray, ts: np.ndarray, grid: np.ndarray, upto: np.ndarray, w: Window, lo: int
) -> np.ndarray:
    """The sum of the values whose prefix sums are `cum` over the window ending at each grid
    time; NaN where the window starts before the day's first trade."""
    if w.trades is not None:
        first = upto - w.trades
        return np.where(first >= 0, cum[upto] - cum[np.clip(first, 0, None)], np.nan)
    since = np.searchsorted(ts, grid - w.ns, side="right")
    return np.where(grid - w.ns >= lo, cum[upto] - cum[since], np.nan)


def _prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x)])


def day_inputs(trades: pl.DataFrame, date: str, spec: Spec) -> tuple[pl.DataFrame, dict]:
    """On the day's grid: every signal and every target; and the Hawkes fits' diagnostics."""
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * SEC
    order = np.argsort(trades["ts"].to_numpy(), kind="stable")
    ts = trades["ts"].to_numpy()[order]
    price = trades["price"].to_numpy()[order]
    qty = trades["qty"].to_numpy()[order]
    buy = ~trades["is_buyer_maker"].to_numpy().astype(bool)[order]
    every = duration_ns(spec.sampling)
    if len(ts) == 0:
        cols = {"ts": np.zeros(0, np.int64)}
        cols |= {s.key: np.zeros(0) for s in spec.signals} | {t: np.zeros(0) for t in spec.targets}
        return pl.DataFrame(cols), {}
    lo, hi = max(int(ts[0]), start), min(int(ts[-1]), start + NS_DAY - 1)
    grid = np.arange(-(-lo // every) * every, hi + 1, every, dtype=np.int64)
    upto = np.searchsorted(ts, grid, side="right")
    sign = np.where(buy, 1.0, -1.0)
    sums = {
        "signed_volume": _prefix(sign * qty),
        "signed_count": _prefix(sign),
        "count": _prefix(np.ones(len(ts))),
        "volume": _prefix(qty),
        "buy_qty": _prefix(np.where(buy, qty, 0.0)),
        "buy_n": _prefix(buy.astype(float)),
        "sell_qty": _prefix(np.where(buy, 0.0, qty)),
        "sell_n": _prefix((~buy).astype(float)),
    }
    logp = np.log(price)
    # the trade price each second, for realized volatility
    seconds = np.arange(-(-lo // SEC) * SEC, hi + 1, SEC, dtype=np.int64)
    p_sec = logp[np.searchsorted(ts, seconds, side="right") - 1]
    r2 = _prefix(np.diff(p_sec, prepend=p_sec[:1]) ** 2)
    sec_upto = np.searchsorted(seconds, grid, side="right")
    cols: dict[str, np.ndarray] = {"ts": grid}
    diag: dict = {}
    for s in spec.signals:
        if s.method == "hawkes":
            cols[s.key], diag[s.key] = _hawkes(ts, buy, lo, hi, grid, s)
            continue
        w = Window(s.window)
        if s.method in ("signed_volume", "signed_count"):
            x = _window_sum(sums[s.method], ts, grid, upto, w, lo)
        elif s.method == "size_skew":
            part = {
                k: _window_sum(sums[k], ts, grid, upto, w, lo)
                for k in ("buy_qty", "buy_n", "sell_qty", "sell_n")
            }
            with np.errstate(invalid="ignore", divide="ignore"):
                x = part["buy_qty"] / part["buy_n"] - part["sell_qty"] / part["sell_n"]
            x = np.where(np.isfinite(x), x, np.nan)
        elif s.method in ("intensity", "volume_rate"):
            total = _window_sum(
                sums["count" if s.method == "intensity" else "volume"], ts, grid, upto, w, lo
            )
            x = total / (w.ns / SEC)
        else:  # realized_vol: squared one-second log returns in (t - window, t]
            since = np.searchsorted(seconds, grid - w.ns, side="right")
            x = np.where(grid - w.ns >= lo, np.sqrt(r2[sec_upto] - r2[since]), np.nan)
        cols[s.key] = x
    now = logp[upto - 1]
    for t in spec.targets:
        h = duration_ns(_TARGET.match(t).group(2))
        later = logp[np.searchsorted(ts, grid + h, side="right") - 1]
        ret = np.where(grid + h <= hi, later - now, np.nan)
        cols[t] = np.abs(ret) if t.startswith("abs_") else ret
    return pl.DataFrame(cols), diag


def _hawkes(ts, buy, lo, hi, grid, s: Signal) -> tuple[np.ndarray, dict]:
    """Log total intensity of taker orders at each grid time, from the fits before it. One
    taker order sweeping several prices is several aggregate trades at one time: one event."""
    t, side, _ = hk.market_orders(ts, ts, buy, np.zeros(len(ts), dtype=np.int64))
    step = duration_ns(s.step)
    r = hk.rolling(t, side, [(lo, hi + 1)], duration_ns(s.window), step)
    lam, _ = hk.at_times(r, t, side, grid, step)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(lam.sum(axis=1))
    fitted = np.isfinite(r.params[:, 6])
    diag = {
        "fits": int(fitted.sum()),
        "windows": len(r.at),
        "branching_median": ex._f(np.median(r.branching[fitted])) if fitted.any() else None,
    }
    return np.where(np.isfinite(x), x, np.nan), diag


def _cache_file(cache: Path, day: archive.ArchiveDay, spec: Spec) -> Path:
    st = day.path.stat()
    parts = [
        str(CACHE_VERSION),
        __version__,
        day.path.name,
        str(st.st_size),
        str(st.st_mtime_ns),
        spec.sampling,
        ",".join(s.key for s in spec.signals),
        ",".join(spec.targets),
    ]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache / spec.symbol / day.date / f"screen-{key}.parquet"


def load_inputs(
    day: archive.ArchiveDay, spec: Spec, cache: Path | None
) -> tuple[pl.DataFrame, dict]:
    f = _cache_file(cache, day, spec) if cache is not None else None
    if f is not None and f.exists():
        side = f.with_suffix(".json")
        return pl.read_parquet(f), json.loads(side.read_text()) if side.exists() else {}
    trades, _ = archive.read(day)
    df, diag = day_inputs(trades, day.date, spec)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(f)
        f.with_suffix(".json").write_text(json.dumps(diag))
    return df, diag


def _load(task: tuple[archive.ArchiveDay, Spec, Path | None]) -> tuple[pl.DataFrame, dict]:
    return load_inputs(*task)


def load_all(
    days: Sequence[archive.ArchiveDay], spec: Spec, cache: Path | None, workers: int
) -> list[tuple[pl.DataFrame, dict]]:
    tasks = [(d, spec, cache) for d in days]
    if workers <= 1 or len(tasks) <= 1:
        return [_load(t) for t in tasks]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    with ProcessPoolExecutor(max_workers=min(workers, 4), mp_context=context) as pool:
        return list(pool.map(_load, tasks))


class Samples:
    """Every day's grid rows end to end; one target at a time for `experiment._surrogates`."""

    def __init__(self, inputs: Sequence[pl.DataFrame], target: str | None = None):
        self.inputs = list(inputs)
        self.df = pl.concat(self.inputs) if self.inputs else pl.DataFrame()
        self.day = np.concatenate([np.full(len(d), i) for i, d in enumerate(self.inputs)])
        self.target = self.df[target].to_numpy() if target and len(self.df) else np.zeros(0)

    def with_target(self, target: str) -> Samples:
        s = Samples.__new__(Samples)
        s.inputs, s.df, s.day = self.inputs, self.df, self.day
        s.target = self.df[target].to_numpy()
        return s

    def rows_of(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.day == day)


def _lags(spec: Spec, c: Combo) -> int:
    return max(1, ceil(duration_ns(c.horizon) / duration_ns(spec.sampling)))


def combo_ics(s: Samples, spec: Spec, c: Combo) -> list[metrics.DayIC]:
    t = s.with_target(c.target)
    x = s.df[c.signal.key].to_numpy()
    return [
        metrics.day_ic(x[t.rows_of(i)], t.target[t.rows_of(i)], _lags(spec, c))
        for i in range(len(s.inputs))
    ]


# --- the development run ----------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    days: list[str]
    missing: list[str]
    table: list[dict]  # every combination: name, mean IC, se, p, positive share
    pbo: pbo.PBO | None
    chosen: list[dict]  # the combinations the rule took, with their surrogates
    passed_over: list[dict]  # tried by the rule and refused by its surrogates
    diagnostics: dict
    verdict: str
    reasons: list[str]

    def to_json(self) -> dict:
        s = self.spec
        return {
            "id": s.id,
            "name": s.name,
            "kind": "screen",
            "hypothesis": s.hypothesis,
            "spec": s.path.name,
            "spec_commit": self.provenance.spec_commit,
            "head": self.provenance.head,
            "dirty": self.provenance.dirty,
            "marketlab": __version__,
            "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data": {
                "symbol": s.symbol,
                "days": self.days,
                "missing": self.missing,
                "holdout": list(s.holdout),
            },
            "combinations": len(s.combos),
            "table": self.table,
            "pbo": self.pbo.__dict__ if self.pbo else None,
            "chosen": self.chosen,
            "passed_over": self.passed_over,
            "hawkes": self.diagnostics,
            # the screen's claim is decided on the held-out days: until then it counts at 1
            "p_raw": None,
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _days(root, spec: Spec, dates: Sequence[str]) -> tuple[list[archive.ArchiveDay], list[str]]:
    found = {d.date: d for d in archive.find(root, spec.symbol, spec.root_kind, spec.market)}
    return [found[d] for d in dates if d in found], [d for d in dates if d not in found]


def run(
    spec: Spec,
    root,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
    workers: int = 1,
) -> Report:
    prov = prov or ex.provenance(spec.path)
    days, missing = _days(root, spec, spec.development)
    assert not {d.date for d in days} & set(spec.holdout), "a held-out day slipped in"
    if len(days) < spec.min_days:
        reason = (
            f"{len(days)} development days in the data, the hypothesis asks for {spec.min_days}"
        )
        report = Report(
            spec,
            prov,
            [d.date for d in days],
            missing,
            [],
            None,
            [],
            [],
            {},
            "insufficient data",
            [reason],
        )
        return _finish(report, results)
    loaded = load_all(days, spec, cache, workers)
    s = Samples([df for df, _ in loaded])
    diagnostics = {d.date: diag for d, (_, diag) in zip(days, loaded, strict=True) if diag}
    table, perf = [], np.full((len(days), len(spec.combos)), np.nan)
    for j, c in enumerate(spec.combos):
        ics = combo_ics(s, spec, c)
        m = metrics.combine(ics)
        perf[:, j] = [x.ic for x in ics]
        table.append(
            {
                "combination": c.name,
                "mean_ic": ex._f(m.mean),
                "se": ex._f(m.se),
                "p": ex._f(m.p),
                "positive_days": ex._f(m.positive_share),
            }
        )
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.combos) > 1 else None
    # the rule, applied blindly: best mean IC first
    ranked = sorted(range(len(spec.combos)), key=lambda j: -(table[j]["mean_ic"] or -np.inf))
    chosen, passed_over = [], []
    for j in ranked:
        if len(chosen) == spec.take:
            break
        row, c = table[j], spec.combos[j]
        if row["mean_ic"] is None or row["mean_ic"] <= 0:
            break
        if (row["positive_days"] or 0) < spec.min_positive_days:
            continue
        t = s.with_target(c.target)
        shape = types.SimpleNamespace(
            sampling=spec.sampling,
            horizon=c.horizon,
            seed=spec.seed,
            surrogates=spec.surrogates,
            surrogate_percentile=spec.surrogate_percentile,
        )
        sur = ex._surrogates(t, s.df[c.signal.key].to_numpy(), row["mean_ic"], shape, workers)
        entry = {**row, "surrogates": sur}
        if all(x["real"] is not None and x["real"] > x["threshold"] for x in sur.values()):
            chosen.append(entry)
        else:
            passed_over.append(entry)
    reasons = []
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} over the grid is above {spec.max_pbo}")
    if len(chosen) < spec.take:
        reasons.append(f"the rule found {len(chosen)} of the {spec.take} it should take")
    verdict = "screened" if not reasons else "failed"
    report = Report(
        spec,
        prov,
        [d.date for d in days],
        missing,
        table,
        result_pbo,
        chosen,
        passed_over,
        diagnostics,
        verdict,
        reasons,
    )
    return _finish(report, results)


def _finish(report: Report, results: Path) -> Report:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{report.spec.slug}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return report


# --- the held-out days ---------------------------------------------------------------------


def final(
    spec: Spec,
    root,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
    workers: int = 1,
) -> dict:
    """The chosen combinations, and only they, on the held-out days, once; Benjamini-Hochberg
    over them. A second call is refused."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    dev = results / f"{spec.slug}.json"
    if not dev.exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    record = json.loads(dev.read_text())
    if record["verdict"] != "screened":
        raise ex.ExperimentError(
            f"{spec.slug} did not pass its screen: nothing to open the held-out days for"
        )
    names = [c["combination"] for c in record["chosen"]]
    combos = [c for c in spec.combos if c.name in names]
    prov = prov or ex.provenance(spec.path)
    days, missing = _days(root, spec, spec.holdout)
    if not days:
        raise ex.ExperimentError("none of the held-out days is in the data")
    s = Samples([df for df, _ in load_all(days, spec, cache, workers)])
    rows = []
    for c in combos:
        ics = combo_ics(s, spec, c)
        m = metrics.combine(ics)
        rows.append(
            {
                "combination": c.name,
                "mean_ic": ex._f(m.mean),
                "se": ex._f(m.se),
                "p": ex._f(m.p),
                "positive_days": ex._f(m.positive_share),
                "ic_by_day": {d.date: ex._f(x.ic) for d, x in zip(days, ics, strict=True)},
            }
        )
    p = np.array([r["p"] if r["p"] is not None else 1.0 for r in rows])
    adj = benjamini_hochberg(p) if len(p) else p
    for r, a in zip(rows, adj, strict=True):
        r["p_bh"] = float(a)
        r["confirmed"] = bool(a < spec.alpha and (r["mean_ic"] or 0) > 0)
    result = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": [d.date for d in days],
        "missing": missing,
        "chosen": rows,
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result
