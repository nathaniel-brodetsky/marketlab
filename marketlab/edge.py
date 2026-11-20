"""Hypotheses of the `edge` kind: whether a trade-flow signal points to moves larger than
what a taker pays to trade them.

The rule, on Binance's trade archives, a day at a time. At each time of the grid (every
`sampling`), with no position open:

- buy if the signal is above its `quantile` over the `lookback` before (the grid times
  before, not this one; none with less than half of `lookback` of the signal), sell if it
  is below its `1 - quantile`;
- the order reaches the market `latency` after the decision and fills at the last trade
  price then; the position is closed at the last trade price `horizon` later, and the next
  decision comes once it is closed. The spread of BTCUSDT spot, a cent, is a hundredth of a
  basis point and is left out;
- no position is opened that would close after the end of the day, by the clock.

The gross of a trade is its signed log return, in basis points; the mean over days of each
day's mean gross is the round-trip cost at which the rule breaks even (`be_bp`). The
hypothesis names the cost it must beat (`cost_bp`: both legs, taker).

Development: every combination of signal and horizon is scored, PBO is measured over the
whole grid, and the rule written in the hypothesis is applied blindly: the strongest
evidence of beating the cost first (the smallest one-sided p of the mean against it, so
that a long horizon's few noisy trades do not win on a lucky mean), required to beat the
cost, on enough of the days, and to be above its surrogates (the signal moved in time
within its day by a random offset, the rule run again on it). Only the combinations taken
go to the held-out days, once, with a one-sided test of the mean against the cost and
Benjamini-Hochberg over them. Every combination tried is in the PBO, and the hypothesis is
one in the family, so trying many buys nothing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from . import __version__, archive, screen
from . import experiment as ex
from .features import duration_ns
from .tradeflow import NS_DAY, Window, _dates
from .validation import pbo
from .validation.multiple import benjamini_hochberg

CACHE_VERSION = 1
SEC = 1_000_000_000


@dataclass(frozen=True)
class Combo:
    signal: screen.Signal
    horizon: str

    @property
    def name(self) -> str:
        return f"{self.signal.key} -> {self.horizon}"


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
    quantile: float
    lookback: str
    latency: str
    cost_bp: float
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
    def signals(self) -> tuple[screen.Signal, ...]:
        return tuple(dict.fromkeys(c.signal for c in self.combos))

    @property
    def horizons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.horizon for c in self.combos))

    @property
    def back(self) -> int:
        """The lookback in grid times."""
        return duration_ns(self.lookback) // duration_ns(self.sampling)


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ex.ExperimentError(f"{path}: {e}") from e
    if not isinstance(raw, dict) or raw.get("kind") != "edge":
        raise ex.ExperimentError(f"{path}: not an `edge` hypothesis")
    known = {"id", "name", "kind", "hypothesis", "data", "sampling", "signals", "horizons",
             "rule", "cost_bp", "selection", "protocol", "success"}  # fmt: skip
    if set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw) - known)}")
    data, sel, rule = raw.get("data", {}), raw.get("selection", {}), raw.get("rule", {})
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    days = _dates(ex._get(data, "days", "data"), "data.days")
    held = ex._get(data, "holdout", "data")
    skip: set[str] = set()
    if isinstance(held, dict) and "except" in held:  # days seen before, for other questions
        skip = set(_dates(held["except"], "data.holdout.except"))
        held = {k: v for k, v in held.items() if k != "except"}
    holdout = tuple(d for d in _dates(held, "data.holdout") if d not in skip)
    if set(days) & set(holdout):
        raise ex.ExperimentError(f"{path}: held-out days inside the development days")
    signals: list[screen.Signal] = []
    for s in ex._get(raw, "signals", "top"):
        method = str(ex._get(s, "method", "signals"))
        if method not in screen.DIRECTION:
            raise ex.ExperimentError(f"{path}: {method!r} is not a direction signal")
        for w in ex._get(s, "windows", method):
            win = Window(str(w))
            if win.trades is None:
                duration_ns(win.text)
            elif win.trades < 1:
                raise ex.ExperimentError(f"{path}: {method} over {win.text}")
            signals.append(screen.Signal(method, win.text))
    horizons = [str(h) for h in ex._get(raw, "horizons", "top")]
    for h in horizons:
        if duration_ns(h) <= 0:
            raise ex.ExperimentError(f"{path}: horizon {h!r}")
    combos = [Combo(s, h) for s in signals for h in horizons]
    if not combos:
        raise ex.ExperimentError(f"{path}: the grid is empty")
    if len(set(combos)) != len(combos):
        raise ex.ExperimentError(f"{path}: a combination is listed twice")
    req = sel.get("require", {})
    if sel.get("rank_by", "p_cost") != "p_cost":
        raise ex.ExperimentError(f"{path}: selection ranks by `p_cost`")
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
        quantile=float(ex._get(rule, "quantile", "rule")),
        lookback=str(ex._get(rule, "lookback", "rule")),
        latency=str(ex._get(rule, "latency", "rule")),
        cost_bp=float(ex._get(raw, "cost_bp", "top")),
        take=int(sel.get("take", 1)),
        min_positive_days=float(req.get("positive_days", 0.6)),
        surrogate_percentile=float(req.get("surrogates_above", 95)),
        surrogates=int(proto.get("surrogates", 200)),
        seed=int(proto.get("seed", 1)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        alpha=float(ok.get("alpha", 0.05)),
        path=path,
    )
    every = duration_ns(spec.sampling)
    if not 0.5 < spec.quantile < 1:
        raise ex.ExperimentError(f"{path}: rule.quantile must be above 0.5 and below 1")
    if spec.back < 2 or duration_ns(spec.lookback) % every:
        raise ex.ExperimentError(f"{path}: rule.lookback must be whole grid steps, at least two")
    if duration_ns(spec.latency) < 0:
        raise ex.ExperimentError(f"{path}: rule.latency must not be negative")
    if spec.cost_bp < 0:
        raise ex.ExperimentError(f"{path}: cost_bp must not be negative")
    if spec.surrogates < 200:
        raise ex.ExperimentError(
            f"{path}: the protocol asks for at least 200 surrogates of each kind"
        )
    if not 1 <= spec.take <= len(combos):
        raise ex.ExperimentError(f"{path}: selection.take must be from 1 to {len(combos)}")
    return spec


# --- one day's inputs --------------------------------------------------------------------


def day_inputs(trades: pl.DataFrame, date: str, spec: Spec) -> pl.DataFrame:
    """On the day's grid (`screen.day_inputs`): every signal; `gross_<h>`, the log return in
    bp from the last trade price at `latency` after each time to the last one `horizon`
    after that; and `open_<h>`, whether a position taken then closes within the day."""
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * SEC
    shape = types.SimpleNamespace(sampling=spec.sampling, signals=spec.signals, targets=())
    df, _ = screen.day_inputs(trades, date, shape)
    order = np.argsort(trades["ts"].to_numpy(), kind="stable")
    ts = trades["ts"].to_numpy()[order]
    logp = np.log(trades["price"].to_numpy()[order]) if len(ts) else np.zeros(0)
    grid = df["ts"].to_numpy()
    at = grid + duration_ns(spec.latency)
    cols = {}
    for h in spec.horizons:
        out = at + duration_ns(h)
        if len(ts):
            entry = logp[np.searchsorted(ts, at, side="right") - 1]
            exit_ = logp[np.searchsorted(ts, out, side="right") - 1]
            cols[f"gross_{h}"] = (exit_ - entry) * 1e4
        else:
            cols[f"gross_{h}"] = np.zeros(0)
        cols[f"open_{h}"] = out <= start + NS_DAY
    return df.with_columns(**{k: pl.Series(v) for k, v in cols.items()})


def _cache_file(cache: Path, day: archive.ArchiveDay, spec: Spec) -> Path:
    st = day.path.stat()
    parts = [
        str(CACHE_VERSION),
        __version__,
        day.path.name,
        str(st.st_size),
        str(st.st_mtime_ns),
        spec.sampling,
        spec.latency,
        ",".join(s.key for s in spec.signals),
        ",".join(spec.horizons),
    ]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache / spec.symbol / day.date / f"edge-{key}.parquet"


def load_inputs(day: archive.ArchiveDay, spec: Spec, cache: Path | None) -> pl.DataFrame:
    f = _cache_file(cache, day, spec) if cache is not None else None
    if f is not None and f.exists():
        return pl.read_parquet(f)
    trades, _ = archive.read(day)
    df = day_inputs(trades, day.date, spec)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(f)
    return df


def _load(task: tuple[archive.ArchiveDay, Spec, Path | None]) -> pl.DataFrame:
    return load_inputs(*task)


def load_all(
    days: Sequence[archive.ArchiveDay], spec: Spec, cache: Path | None, workers: int
) -> list[pl.DataFrame]:
    tasks = [(d, spec, cache) for d in days]
    if workers <= 1 or len(tasks) <= 1:
        return [_load(t) for t in tasks]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    with ProcessPoolExecutor(max_workers=min(workers, 4), mp_context=context) as pool:
        return list(pool.map(_load, tasks))


# --- the rule ----------------------------------------------------------------------------


def sides(x: np.ndarray, quantile: float, back: int) -> np.ndarray:
    """+1 where the signal is above its `quantile` over the `back` grid times before (not
    counting this one), -1 where it is below its `1 - quantile`, else 0; 0 with less than
    half of `back` of the signal before."""
    s = pl.Series(np.asarray(x, dtype=np.float64)).fill_nan(None).shift(1)
    kw = {"interpolation": "linear", "window_size": back, "min_samples": max(back // 2, 1)}
    hi = s.rolling_quantile(quantile, **kw).to_numpy()
    lo = s.rolling_quantile(1 - quantile, **kw).to_numpy()
    with np.errstate(invalid="ignore"):
        return np.where(x > hi, 1, np.where(x < lo, -1, 0)).astype(np.int8)


def take(ts: np.ndarray, side: np.ndarray, can_open: np.ndarray, hold: int) -> np.ndarray:
    """The grid times a position is opened at: the first time with a side and room in the
    day, then the first such time once the position is closed, `hold` after its decision."""
    cand = np.flatnonzero((side != 0) & can_open)
    at = ts[cand]
    taken = []
    k = 0
    while k < len(cand):
        taken.append(cand[k])
        k = int(np.searchsorted(at, at[k] + hold, side="left"))
    return np.array(taken, dtype=np.int64)


@dataclass(frozen=True)
class DayTrades:
    n: int
    mean: float  # bp a trade; NaN without trades
    hits: int  # trades with a positive gross


def day_trades(
    df: pl.DataFrame,
    spec: Spec,
    c: Combo,
    x: np.ndarray | None = None,
    side: np.ndarray | None = None,
) -> DayTrades:
    """The rule on a day for one combination; `x` in place of the signal (for surrogates),
    or `side`, its `sides` worked out already."""
    if side is None:
        x = df[c.signal.key].to_numpy() if x is None else x
        side = sides(x, spec.quantile, spec.back)
    hold = duration_ns(spec.latency) + duration_ns(c.horizon)
    rows = take(df["ts"].to_numpy(), side, df[f"open_{c.horizon}"].to_numpy(), hold)
    g = side[rows] * df[f"gross_{c.horizon}"].to_numpy()[rows]
    return DayTrades(len(g), float(g.mean()) if len(g) else float("nan"), int((g > 0).sum()))


@dataclass(frozen=True)
class Score:
    days: int  # with trades
    trades_day: float
    be_bp: float  # mean over days of each day's mean gross a trade
    se: float
    p_cost: float  # one-sided: the mean above the cost
    positive_days: float  # share of days with trades whose mean beats the cost
    hit_rate: float

    def row(self) -> dict:
        return {k: ex._f(v) for k, v in self.__dict__.items()}


def score(by_day: Sequence[DayTrades], cost: float) -> Score:
    m = np.array([d.mean for d in by_day if d.n])
    n = len(m)
    trades = sum(d.n for d in by_day)
    nan = float("nan")
    se = float(m.std(ddof=1) / np.sqrt(n)) if n > 1 else nan
    mean = float(m.mean()) if n else nan
    p = float(stats.t.sf((mean - cost) / se, n - 1)) if n > 1 and se > 0 else nan
    return Score(
        days=n,
        trades_day=trades / len(by_day) if by_day else nan,
        be_bp=mean,
        se=se,
        p_cost=p,
        positive_days=float((m > cost).mean()) if n else nan,
        hit_rate=sum(d.hits for d in by_day) / trades if trades else nan,
    )


def surrogate_means(inputs: Sequence[pl.DataFrame], spec: Spec, c: Combo, seed: int) -> np.ndarray:
    """The rule's break-even with each day's signal moved in time, round its day, by a random
    offset of a tenth to nine tenths of the day: as often in each direction, at times
    unrelated to the returns."""
    rng = np.random.default_rng(seed)
    out = np.empty(spec.surrogates)
    signals = [df[c.signal.key].to_numpy() for df in inputs]
    for k in range(spec.surrogates):
        days = []
        for df, x in zip(inputs, signals, strict=True):
            n = len(x)
            shift = int(rng.integers(max(n // 10, 1), max(9 * n // 10, 2))) if n > 1 else 0
            days.append(day_trades(df, spec, c, np.roll(x, shift)))
        out[k] = score(days, spec.cost_bp).be_bp
    return out


# --- the development run ----------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    days: list[str]
    missing: list[str]
    table: list[dict]  # every combination: its score
    pbo: pbo.PBO | None
    chosen: list[dict]
    passed_over: list[dict]  # beat the cost and were refused by their surrogates
    verdict: str
    reasons: list[str]

    def to_json(self) -> dict:
        s = self.spec
        return {
            "id": s.id,
            "name": s.name,
            "kind": "edge",
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
            "cost_bp": s.cost_bp,
            "combinations": len(s.combos),
            "table": self.table,
            "pbo": self.pbo.__dict__ if self.pbo else None,
            "chosen": self.chosen,
            "passed_over": self.passed_over,
            # decided on the held-out days: until then it counts at 1
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
    dates = [d.date for d in days]
    if len(days) < spec.min_days:
        reason = (
            f"{len(days)} development days in the data, the hypothesis asks for {spec.min_days}"
        )
        report = Report(spec, prov, dates, missing, [], None, [], [], "insufficient data", [reason])
        return _finish(report, results)
    inputs = load_all(days, spec, cache, workers)
    table, perf = [], np.full((len(days), len(spec.combos)), np.nan)
    by_signal = {  # the sides depend on the signal only, not on the horizon
        sg.key: [sides(df[sg.key].to_numpy(), spec.quantile, spec.back) for df in inputs]
        for sg in spec.signals
    }
    for j, c in enumerate(spec.combos):
        sd = by_signal[c.signal.key]
        by_day = [day_trades(df, spec, c, side=x) for df, x in zip(inputs, sd, strict=True)]
        perf[:, j] = [d.mean for d in by_day]
        table.append({"combination": c.name, **score(by_day, spec.cost_bp).row()})
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.combos) > 1 else None
    # the rule, applied blindly: the strongest evidence of beating the cost first
    ranked = sorted(
        range(len(spec.combos)),
        key=lambda j: table[j]["p_cost"] if table[j]["p_cost"] is not None else np.inf,
    )
    chosen, passed_over = [], []
    for j in ranked:
        row, c = table[j], spec.combos[j]
        if len(chosen) == spec.take:
            break
        if row["be_bp"] is None or row["be_bp"] <= spec.cost_bp:
            continue
        if (row["positive_days"] or 0) < spec.min_positive_days:
            continue
        sur = surrogate_means(inputs, spec, c, spec.seed + j)
        finite = sur[np.isfinite(sur)]
        threshold = float(np.percentile(finite, spec.surrogate_percentile)) if len(finite) else None
        entry = {**row, "surrogates": {"shift": {"threshold": threshold, "n": len(finite)}}}
        if threshold is not None and row["be_bp"] > threshold:
            chosen.append(entry)
        else:
            passed_over.append(entry)
    reasons = []
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} over the grid is above {spec.max_pbo}")
    if len(chosen) < spec.take:
        reasons.append(
            f"the rule found {len(chosen)} of the {spec.take} it should take beating "
            f"{spec.cost_bp:g} bp"
        )
    verdict = "screened" if not reasons else "failed"
    report = Report(
        spec, prov, dates, missing, table, result_pbo, chosen, passed_over, verdict, reasons
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
    """The chosen combinations, and only they, on the held-out days, once: the mean against
    the cost, one-sided, Benjamini-Hochberg over them. A second call is refused."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    dev = results / f"{spec.slug}.json"
    if not dev.exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    record = json.loads(dev.read_text())
    if record["verdict"] != "screened":
        raise ex.ExperimentError(
            f"{spec.slug} did not pass on its development days: nothing to open the held-out "
            "days for"
        )
    names = [c["combination"] for c in record["chosen"]]
    combos = [c for c in spec.combos if c.name in names]
    prov = prov or ex.provenance(spec.path)
    days, missing = _days(root, spec, spec.holdout)
    if not days:
        raise ex.ExperimentError("none of the held-out days is in the data")
    inputs = load_all(days, spec, cache, workers)
    rows = []
    for c in combos:
        by_day = [day_trades(df, spec, c) for df in inputs]
        rows.append(
            {
                "combination": c.name,
                **score(by_day, spec.cost_bp).row(),
                "by_day": {
                    d.date: {"trades": x.n, "mean": ex._f(x.mean)}
                    for d, x in zip(days, by_day, strict=True)
                },
            }
        )
    p = np.array([r["p_cost"] if r["p_cost"] is not None else 1.0 for r in rows])
    adj = benjamini_hochberg(p) if len(p) else p
    for r, a in zip(rows, adj, strict=True):
        r["p_bh"] = float(a)
        r["confirmed"] = bool(a < spec.alpha and (r["be_bp"] or 0) > spec.cost_bp)
    result = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "cost_bp": spec.cost_bp,
        "days": [d.date for d in days],
        "missing": missing,
        "chosen": rows,
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result
