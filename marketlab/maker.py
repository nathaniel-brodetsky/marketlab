"""Hypotheses of the `maker` kind: does a signal cut what passive orders lose to adverse
selection?

Probe orders rest at the best price on each side, decided at every multiple of `every`,
and are cancelled after `cancel_after` (`marketlab.execution`). The rule withdraws an order
whenever the signal points against it beyond a threshold: a bid when the level-1
order-flow imbalance, scaled by its own spread over the hour before, falls below minus the
threshold; an ask when it rises above it. A signal already against the order at its
decision withdraws it at once, so that it never fills.

What is measured is the fills' quantity-weighted mean markout at the hypothesis's horizon:
with the rule, minus without it, day by day. Withdrawing an order changes nothing before
the withdrawal takes effect (the queue ahead of it evolves the same), so every rule and
every surrogate follows from one simulation per day: an order withdrawn at `t` keeps its
fills received by `t + latency` and loses the rest. `tests/test_maker.py` checks this
against the simulator itself.

The rule has thresholds but nothing fitted, so the days are scored directly, without
folds. Against chance: the same rule with the signal shifted in time within its day, by at
least `min_shift`, which keeps how often and for how long orders are withdrawn and destroys
when. Against choosing: PBO over the variants. Against many tests: Benjamini-Hochberg over
the registry. The conclusion is drawn from the pessimistic queue; the other two modes are
reported for sensitivity.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from . import __version__
from . import experiment as ex
from .execution import Market, Mode, frames, markouts, probe_orders, simulate
from .features import MultiLevelOFI, Volatility, duration_ns, grid
from .io.day import Day
from .io.days import DayInfo, survey
from .validation import pbo

CACHE_VERSION = 2
HORIZONS = ("100ms", "1s", "10s", "60s")
NS_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class Variant:
    window: str
    threshold: float


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    symbol: str
    min_hours: float
    holdout: tuple[str, ...]
    min_days: int
    every: str
    size: Decimal
    cancel_after: str
    latency: str
    queue: Mode
    primary: Variant
    variants: tuple[Variant, ...]
    horizon: str
    surrogates: int
    min_shift: str
    seed: int
    min_delta_bp: float
    alpha: float
    surrogate_percentile: float
    max_pbo: float
    min_positive_days: float
    path: Path = field(compare=False)

    @property
    def slug(self) -> str:
        return f"{self.id}-{self.name}"


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    known = {
        "id",
        "name",
        "kind",
        "hypothesis",
        "data",
        "orders",
        "signal",
        "variants",
        "target",
        "protocol",
        "success",
    }
    if not isinstance(raw, dict) or set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw or {}) - known)}")
    if raw.get("kind") != "maker":
        raise ex.ExperimentError(f"{path}: not a `maker` hypothesis")
    g = ex._get
    data, orders, sig = raw.get("data", {}), g(raw, "orders", "top"), g(raw, "signal", "top")
    target, proto, ok = g(raw, "target", "top"), raw.get("protocol", {}), raw.get("success", {})
    if g(sig, "method", "signal") != "ofi_z":
        raise ex.ExperimentError(f"{path}: only `ofi_z` is supported as a signal so far")

    def variant(v: dict) -> Variant:
        return Variant(str(g(v, "window", "variant")), float(g(v, "threshold", "variant")))

    primary = variant(g(sig, "params", "signal"))
    variants = tuple(variant(v) for v in raw.get("variants", [])) or (primary,)
    if primary not in variants:
        variants = (primary, *variants)
    spec = Spec(
        id=str(g(raw, "id", "top")),
        name=str(g(raw, "name", "top")),
        hypothesis=" ".join(str(g(raw, "hypothesis", "top")).split()),
        symbol=str(g(data, "symbol", "data")),
        min_hours=float(data.get("min_hours", 20)),
        holdout=tuple(str(d) for d in data.get("holdout", [])),
        min_days=int(data.get("min_days", 10)),
        every=str(g(orders, "every", "orders")),
        size=Decimal(str(g(orders, "size", "orders"))),
        cancel_after=str(g(orders, "cancel_after", "orders")),
        latency=str(g(orders, "latency", "orders")),
        queue=Mode(str(orders.get("queue", "pessimistic"))),
        primary=primary,
        variants=variants,
        horizon=str(g(target, "markout", "target")),
        surrogates=int(proto.get("surrogates", 200)),
        min_shift=str(proto.get("min_shift", "10m")),
        seed=int(proto.get("seed", 1)),
        min_delta_bp=float(ok.get("min_delta_bp", 0.0)),
        alpha=float(ok.get("alpha", 0.05)),
        surrogate_percentile=float(ok.get("surrogate_percentile", 95)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_days=float(ok.get("min_positive_days", 0.6)),
        path=path,
    )
    if spec.horizon not in HORIZONS:
        raise ex.ExperimentError(f"{path}: the markout horizon must be one of {HORIZONS}")
    for d in (spec.every, spec.cancel_after, spec.latency, spec.min_shift):
        duration_ns(d)
    for v in variants:
        duration_ns(v.window)
    if spec.surrogates < 200:
        raise ex.ExperimentError(f"{path}: the protocol asks for at least 200 surrogates")
    return spec


# --- the signal --------------------------------------------------------------------------


def trailing_std(ts: np.ndarray, x: np.ndarray, span: int, least: int) -> np.ndarray:
    """Standard deviation of the finite values in `[t - span, t)` before each row (not the
    row itself), NaN with fewer than `least` of them."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo = np.searchsorted(ts, ts - span, side="left")
    m = cn[i] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c1[i] - c1[lo]) / m
        var = (c2[i] - c2[lo]) / m - mean**2
    return np.where(m >= least, np.sqrt(np.maximum(var, 0.0)), np.nan)


def signal(day: Day, windows: Sequence[str]) -> pl.DataFrame:
    """On the day's grid: `z_<w>`, level-1 order-flow imbalance over `w` divided by its
    standard deviation over the hour before; `seg_end`, the end of each row's segment; and
    `rv_60s` for the stability report."""
    g = grid(day)
    _, ends = day.segment_bounds()
    cols: dict[str, np.ndarray] = {"seg_end": ends[g.segment] if len(g) else np.zeros(0, np.int64)}
    for w in windows:
        x = MultiLevelOFI(levels=1, window=w).level_windows(day, g)
        x = x[:, 0] if len(g) else np.zeros(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"z_{w}"] = x / trailing_std(g.ts, x, NS_HOUR, 600)
    cols["rv_60s"] = Volatility(windows=("60s",)).columns(g)["rv_60s"] if len(g) else np.zeros(0)
    return pl.DataFrame({"ts": g.ts, "segment": g.segment, **cols}).with_columns(
        pl.all().exclude("ts", "segment").fill_nan(None)
    )


# --- one day's inputs --------------------------------------------------------------------


@dataclass
class DayInputs:
    orders: pl.DataFrame  # id, side, decided, cancel (pessimistic mode's outcome too)
    fills: pl.DataFrame  # mode, id, side, ts, qty, markout_<h>
    signal: pl.DataFrame


#: Simulated an hour at a time: a whole busy day's depth updates (some 50 million) are
#: several GB, an hour's a few hundred MB.
CHUNK = "1h"
#: Data read before a piece: the best prices and the book its first decisions need.
BEFORE = 60 * 1_000_000_000


def simulate_day(
    path: Path, spec: Spec, chunk: str | None = CHUNK
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The probe orders of a day in every queue mode: (orders of `spec.queue`, fills with
    markouts). With `chunk`, the day goes through the simulator a piece at a time: the
    orders decided in a piece, with the data from `BEFORE` ahead of it to as far as their
    cancel, latency and longest markout reach. An order sees exactly what it sees in the
    whole day (`tests/test_maker.py` compares the two), in a fraction of the memory."""
    whole = Day.open(path)
    qty = int(spec.size / whole.qty_step)
    if qty < 1 or Decimal(qty) * whole.qty_step != spec.size:
        raise ex.ExperimentError(f"{whole.date}: size {spec.size} is not whole quantity steps")
    lat = duration_ns(spec.latency)
    reach = duration_ns(spec.cancel_after) + lat + max(duration_ns(h) for h in HORIZONS)
    if chunk is None:
        pieces = [(None, None)]
    else:
        step = duration_ns(chunk)
        first = min(r.first_recv_ts for r in whole.runs)
        last = max(r.last_recv_ts for r in whole.runs)
        pieces = [(lo, lo + step) for lo in range(first // step * step, last + 1, step)]
    order_frames, fill_frames, next_id = [], [], 0
    for lo, hi in pieces:
        if lo is None:
            m = Market.of(whole)
            orders = probe_orders(m, spec.every, qty, spec.cancel_after)
        else:
            m = Market.of(Day.open(path, between=(lo - BEFORE, hi + reach + 1_000_000_000)))
            orders = [
                o
                for o in probe_orders(m, spec.every, qty, spec.cancel_after)
                if lo <= o.decided < hi
            ]
        orders = [replace(o, id=next_id + k) for k, o in enumerate(orders)]
        next_id += len(orders)
        if not orders:
            continue
        piece = []
        for mode in Mode:
            o, f = frames(simulate(m, orders, lat, mode), mode)
            piece.append(f)
            if mode == spec.queue:
                order_frames.append(o)
        fill_frames.append(markouts(m, pl.concat(piece), HORIZONS))
    cols = ("mode", "id", "side", "ts", "qty", *(f"markout_{h}" for h in HORIZONS))
    orders = (
        pl.concat(order_frames).select("id", "side", "decided", "outcome", "filled", "qty")
        if order_frames
        else pl.DataFrame(schema={"id": pl.Int64, "side": pl.Int8, "decided": pl.Int64,
                                  "outcome": pl.String, "filled": pl.Int64, "qty": pl.Int64})
    )  # fmt: skip
    fills = pl.concat(fill_frames).select(cols) if fill_frames else None
    if fills is None:
        fills = pl.DataFrame(
            schema={"mode": pl.String, "id": pl.Int64, "side": pl.Int8, "ts": pl.Int64,
                    "qty": pl.Int64, **{f"markout_{h}": pl.Float64 for h in HORIZONS}}
        )  # fmt: skip
    return orders, fills


def day_inputs(path: Path, spec: Spec) -> DayInputs:
    orders, fills = simulate_day(path, spec)
    return DayInputs(
        orders=orders.sort("id"),
        fills=fills,
        signal=signal(Day.open(path), sorted({v.window for v in spec.variants})),
    )


def _cache_dir(cache: Path, info: DayInfo, spec: Spec) -> Path:
    manifest = (info.path / "manifest.json").read_bytes()
    parts = [
        str(CACHE_VERSION),
        __version__,
        hashlib.sha256(manifest).hexdigest(),
        spec.every,
        str(spec.size),
        spec.cancel_after,
        spec.latency,
        ",".join(sorted({v.window for v in spec.variants})),
    ]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache / spec.symbol / info.date / f"maker-{key}"


def load_inputs(info: DayInfo, spec: Spec, cache: Path | None) -> DayInputs:
    names = ("orders", "fills", "signal")
    if cache is not None:
        d = _cache_dir(cache, info, spec)
        if all((d / f"{n}.parquet").exists() for n in names):
            return DayInputs(*(pl.read_parquet(d / f"{n}.parquet") for n in names))
    x = day_inputs(info.path, spec)
    if cache is not None:
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            tmp = d / f"{n}.tmp"
            getattr(x, n).write_parquet(tmp)
            tmp.replace(d / f"{n}.parquet")
    return x


def _load_one(task: tuple[DayInfo, Spec, Path | None]) -> DayInputs:
    return load_inputs(*task)


def load_all(infos: Sequence[DayInfo], spec: Spec, cache: Path | None, workers: int) -> list:
    tasks = [(i, spec, cache) for i in infos]
    if workers <= 1 or len(tasks) <= 1:
        return [_load_one(t) for t in tasks]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    from concurrent.futures.process import BrokenProcessPool

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(_load_one, tasks))
    except BrokenProcessPool as e:
        raise ex.ExperimentError(
            f"a worker was stopped from outside, most likely for want of memory: run again "
            f"with fewer --workers than {workers} (days already done are cached)"
        ) from e


# --- the rule ----------------------------------------------------------------------------


def withdrawals(
    orders: pl.DataFrame,
    sig: pl.DataFrame,
    z: np.ndarray,
    threshold: float,
    cancel_after: int,
) -> np.ndarray:
    """For each order, the receive time at which the rule withdraws it, or -1. The signal
    is read at the grid rows from the last one at the decision (if the decision is still in
    that row's segment) to the last one at the planned cancel; a bid goes when it is below
    `-threshold`, an ask when above. `z` replaces the signal's values (a surrogate)."""
    ts = sig["ts"].to_numpy()
    seg_end = sig["seg_end"].to_numpy()
    n = len(ts)
    out = np.full(len(orders), -1, dtype=np.int64)
    if n == 0 or len(orders) == 0:
        return out
    decided = orders["decided"].to_numpy()
    side = orders["side"].to_numpy()
    first = np.searchsorted(ts, decided, side="right") - 1
    last = np.searchsorted(ts, decided + cancel_after, side="right") - 1
    stale = (first < 0) | (decided >= seg_end[np.clip(first, 0, None)])
    start = np.where(stale, first + 1, first)
    idx = np.arange(n)
    with np.errstate(invalid="ignore"):
        for s, against in ((1, z < -threshold), (-1, z > threshold)):
            pos = np.where(against, idx, n)
            nxt = np.minimum.accumulate(pos[::-1])[::-1]
            nxt = np.concatenate([nxt, [n]])
            k = side == s
            j = nxt[np.clip(start[k], 0, n)]
            hit = (j <= last[k]) & (j < n)
            when = np.maximum(ts[np.clip(j, 0, n - 1)], decided[k])
            out[np.flatnonzero(k)[hit]] = when[hit]
    return out


def _order_of(orders: pl.DataFrame, ids: np.ndarray) -> np.ndarray:
    """Each id's row in `orders` (sorted by id)."""
    return np.searchsorted(orders["id"].to_numpy(), ids)


def kept(fills: pl.DataFrame, orders: pl.DataFrame, pulled: np.ndarray, latency: int) -> np.ndarray:
    """Which fills survive the withdrawals: those received by the time it takes effect."""
    w = pulled[_order_of(orders, fills["id"].to_numpy())]
    return (w < 0) | (fills["ts"].to_numpy() <= w + latency)


def mean_markout(fills: pl.DataFrame, mask: np.ndarray, horizon: str) -> float:
    m = fills[f"markout_{horizon}"].to_numpy().astype(float)
    q = fills["qty"].to_numpy().astype(float)
    ok = mask & np.isfinite(m)
    return float((m[ok] * q[ok]).sum() / q[ok].sum()) if q[ok].sum() > 0 else float("nan")


@dataclass
class DayScore:
    base: float  # mean markout without the rule, bp
    rule: float  # with it
    fills_kept: float  # share of the filled quantity the rule keeps

    @property
    def delta(self) -> float:
        return self.rule - self.base


def score(
    x: DayInputs, spec: Spec, v: Variant, mode: Mode, z: np.ndarray | None = None
) -> DayScore:
    f = x.fills.filter(pl.col("mode") == str(mode))
    zz = x.signal[f"z_{v.window}"].to_numpy().astype(float) if z is None else z
    pulled = withdrawals(x.orders, x.signal, zz, v.threshold, duration_ns(spec.cancel_after))
    keep = kept(f, x.orders, pulled, duration_ns(spec.latency))
    all_ = np.ones(len(f), dtype=bool)
    q = f["qty"].to_numpy().astype(float)
    return DayScore(
        base=mean_markout(f, all_, spec.horizon),
        rule=mean_markout(f, keep, spec.horizon),
        fills_kept=float(q[keep].sum() / q.sum()) if q.sum() > 0 else float("nan"),
    )


# --- the run -----------------------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    days: list[DayInfo]
    by_day: dict[str, DayScore]
    mean_delta: float
    se: float
    p: float
    positive_share: float
    surrogate: dict
    pbo: pbo.PBO | None
    variants: dict[str, float]
    modes: dict[str, float]
    stability: dict
    p_bh: float
    verdict: str
    reasons: list[str]
    skipped: list[DayInfo] = field(default_factory=list)

    def to_json(self) -> dict:
        s = self.spec
        f = ex._f
        return {
            "id": s.id,
            "name": s.name,
            "kind": "maker",
            "hypothesis": s.hypothesis,
            "spec": s.path.name,
            "spec_commit": self.provenance.spec_commit,
            "head": self.provenance.head,
            "dirty": self.provenance.dirty,
            "marketlab": __version__,
            "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data": {
                "symbol": s.symbol,
                "days": [
                    {"date": d.date, "usable_hours": round(d.usable_hours, 3)} for d in self.days
                ],
                "skipped": [
                    {"date": d.date, "usable_hours": round(d.usable_hours, 3)} for d in self.skipped
                ],
                "holdout": list(s.holdout),
            },
            "orders": {
                "every": s.every,
                "size": str(s.size),
                "cancel_after": s.cancel_after,
                "latency": s.latency,
                "queue": str(s.queue),
            },
            "signal": asdict(s.primary),
            "markout": s.horizon,
            "by_day": {
                d: {
                    "base": f(x.base),
                    "rule": f(x.rule),
                    "delta": f(x.delta),
                    "fills_kept": f(x.fills_kept),
                }
                for d, x in self.by_day.items()
            },
            "mean_delta_bp": f(self.mean_delta),
            "se_bp": f(self.se),
            "positive_share": f(self.positive_share),
            "surrogates": self.surrogate,
            "pbo": asdict(self.pbo) if self.pbo else None,
            "variants": self.variants,
            "modes": self.modes,
            "stability": self.stability,
            "p_raw": f(self.p),
            "p_bh": f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _name(v: Variant) -> str:
    return f"window {v.window}, threshold {v.threshold:g}"


def _surrogates(inputs: list[DayInputs], spec: Spec, real: float) -> dict:
    """The primary rule with the signal rolled within each day by a random number of grid
    rows covering at least `min_shift`; the random numbers depend only on the seed."""
    v = spec.primary
    min_shift = duration_ns(spec.min_shift)
    values = []
    for k in range(spec.surrogates):
        rng = np.random.default_rng([spec.seed, k])
        deltas = []
        for x in inputs:
            z = x.signal[f"z_{v.window}"].to_numpy().astype(float)
            ts = x.signal["ts"].to_numpy()
            n = len(z)
            span = ts[-1] - ts[0] if n else 0
            if n < 3 or span <= 2 * min_shift:
                continue
            rows = int(np.searchsorted(ts, ts[0] + min_shift))
            lag = int(rng.integers(rows, n - rows + 1))
            deltas.append(score(x, spec, v, spec.queue, np.roll(z, lag)).delta)
        values.append(float(np.nanmean(deltas)) if deltas else float("nan"))
    a = np.array(values)
    a = a[np.isfinite(a)]
    if len(a) == 0:  # no day long enough to shift by min_shift: the control cannot be run
        return {
            "shift": {
                "n": 0,
                "percentile": spec.surrogate_percentile,
                "threshold": None,
                "real": ex._f(real),
                "p": None,
                "mean": None,
            }
        }
    return {
        "shift": {
            "n": len(a),
            "percentile": spec.surrogate_percentile,
            "threshold": float(np.percentile(a, spec.surrogate_percentile)),
            "real": ex._f(real),
            "p": float((1 + np.sum(a >= real)) / (1 + len(a))),
            "mean": float(a.mean()),
        }
    }


def _stability(inputs: list[DayInputs], spec: Spec) -> dict:
    """The difference pooled over the days, by time of day and by the volatility of the
    minute before the decision."""
    v = spec.primary
    lat, wait = duration_ns(spec.latency), duration_ns(spec.cancel_after)
    rows = []
    for x in inputs:
        f = x.fills.filter(pl.col("mode") == str(spec.queue))
        z = x.signal[f"z_{v.window}"].to_numpy().astype(float)
        pulled = withdrawals(x.orders, x.signal, z, v.threshold, wait)
        keep = kept(f, x.orders, pulled, lat)
        ts = x.signal["ts"].to_numpy()
        rv = x.signal["rv_60s"].to_numpy().astype(float)
        d = x.orders["decided"].to_numpy()[_order_of(x.orders, f["id"].to_numpy())]
        j = np.clip(np.searchsorted(ts, d, side="right") - 1, 0, None)
        rows.append(
            pl.DataFrame(
                {
                    "hour": (d // NS_HOUR) % 24,
                    "rv": rv[j] if len(rv) else np.full(len(d), np.nan),
                    "m": f[f"markout_{spec.horizon}"].to_numpy().astype(float),
                    "q": f["qty"].to_numpy().astype(float),
                    "keep": keep,
                }
            )
        )
    df = pl.concat(rows) if rows else pl.DataFrame()

    def delta(mask: np.ndarray) -> float | None:
        m, q, k = df["m"].to_numpy(), df["q"].to_numpy(), df["keep"].to_numpy()
        ok = mask & np.isfinite(m)
        if ok.sum() < 30 or (ok & k).sum() < 10:
            return None
        base = (m[ok] * q[ok]).sum() / q[ok].sum()
        rule = (m[ok & k] * q[ok & k]).sum() / q[ok & k].sum()
        return ex._f(rule - base)

    if df.is_empty():
        return {"time_of_day": {}, "volatility_tercile": {}}
    hour = df["hour"].to_numpy()
    times = {
        f"{lo:02d}-{lo + 6:02d} UTC": delta((hour >= lo) & (hour < lo + 6))
        for lo in range(0, 24, 6)
    }
    rv = df["rv"].to_numpy()
    vol = {}
    if np.isfinite(rv).any():
        q1, q2 = np.nanpercentile(rv, [100 / 3, 200 / 3])
        vol = {
            "low": delta(rv <= q1),
            "middle": delta((rv > q1) & (rv <= q2)),
            "high": delta(rv > q2),
        }
    return {"time_of_day": times, "volatility_tercile": vol}


def run(
    spec: Spec,
    data_root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
    workers: int = 1,
) -> Report:
    prov = prov or ex.provenance(spec.path)
    every = survey(data_root, spec.symbol, exclude=spec.holdout)
    infos = [d for d in every if d.usable_hours >= spec.min_hours]
    skipped = [d for d in every if d.usable_hours < spec.min_hours]
    nan = float("nan")
    if len(infos) < max(spec.min_days, 4):
        reason = (
            f"{len(infos)} days with {spec.min_hours:g} usable hours, "
            f"the hypothesis asks for {max(spec.min_days, 4)}"
        )
        r = Report(
            spec,
            prov,
            infos,
            {},
            nan,
            nan,
            nan,
            nan,
            {},
            None,
            {},
            {},
            {},
            nan,
            "insufficient data",
            [reason],
            skipped,
        )
        return _finish(r, results)
    inputs = load_all(infos, spec, cache, workers)
    scores = [score(x, spec, spec.primary, spec.queue) for x in inputs]
    d = np.array([s.delta for s in scores])
    ok = np.isfinite(d)
    mean = float(d[ok].mean())
    se = float(d[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else nan
    p = float(stats.t.sf(mean / se, ok.sum() - 1)) if se > 0 else (0.0 if mean > 0 else 1.0)
    positive = float((d[ok] > 0).mean())

    perf = np.full((len(inputs), len(spec.variants)), np.nan)
    variant_delta = {}
    for j, v in enumerate(spec.variants):
        vd = (
            d
            if v == spec.primary
            else np.array([score(x, spec, v, spec.queue).delta for x in inputs])
        )
        perf[:, j] = vd
        variant_delta[_name(v)] = ex._f(np.nanmean(vd))
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.variants) > 1 else None
    modes = {
        str(mode): ex._f(np.nanmean([score(x, spec, spec.primary, mode).delta for x in inputs]))
        for mode in Mode
    }
    surrogate = _surrogates(inputs, spec, mean)
    stability = _stability(inputs, spec)
    p_bh = ex._bh_with_registry(results, spec, p)

    reasons = []
    if not mean > spec.min_delta_bp:
        reasons.append(f"mean difference {mean:.4f} bp is not above {spec.min_delta_bp}")
    if not p_bh < spec.alpha:
        reasons.append(f"p after Benjamini-Hochberg {p_bh:.3g} is not below {spec.alpha}")
    s = surrogate["shift"]
    if s["threshold"] is None:
        reasons.append(f"no day is long enough to shift the signal by {spec.min_shift}")
    elif not mean > s["threshold"]:
        reasons.append(
            f"not above the {spec.surrogate_percentile:g}th percentile of shifted signals"
        )
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} is above {spec.max_pbo}")
    if not positive >= spec.min_positive_days:
        reasons.append(
            f"positive on {positive:.0%} of days, "
            f"the hypothesis asks for {spec.min_positive_days:.0%}"
        )
    r = Report(
        spec,
        prov,
        infos,
        {i.date: sc for i, sc in zip(infos, scores, strict=True)},
        mean,
        se,
        p,
        positive,
        surrogate,
        result_pbo,
        variant_delta,
        modes,
        stability,
        p_bh,
        "failed" if reasons else "passed",
        reasons,
        skipped,
    )
    return _finish(r, results)


def _finish(r: Report, results: Path) -> Report:
    import json

    results.mkdir(parents=True, exist_ok=True)
    (results / f"{r.spec.slug}.json").write_text(json.dumps(r.to_json(), indent=1) + "\n")
    return r


def final(
    spec: Spec,
    data_root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
) -> dict:
    """The primary rule on the held-out days, once."""
    import json

    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or ex.provenance(spec.path)
    held = [
        d
        for d in survey(data_root, spec.symbol)
        if d.usable_hours >= spec.min_hours and d.date in spec.holdout
    ]
    if not held:
        raise ex.ExperimentError("none of the held-out days is in the data")
    scores = [score(load_inputs(i, spec, cache), spec, spec.primary, spec.queue) for i in held]
    d = np.array([s.delta for s in scores])
    record = {
        "id": spec.id,
        "name": spec.name,
        "kind": "maker",
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": [i.date for i in held],
        "by_day": {
            i.date: {"base": ex._f(s.base), "rule": ex._f(s.rule), "delta": ex._f(s.delta)}
            for i, s in zip(held, scores, strict=True)
        },
        "mean_delta_bp": ex._f(np.nanmean(d)),
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
