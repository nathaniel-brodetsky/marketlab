"""Hypotheses of the `volatility` kind: does a feature forecast the next seconds' volatility
better than volatility itself does?

The target is log(floor + realized volatility of the mid over the next `horizon`), sampled
at the first grid row of each `sampling` interval. Two baselines, as the specification asks:
the next window as the current one (`persistence`), and HAR, a linear model on the log
realized volatility of the last 10 s, 60 s and 300 s. The candidate is HAR with the
feature's columns added. Both are fitted by least squares on the training days of a purged
k-fold over days and scored on the test days.

The metric is the gain over HAR by day: one minus the candidate's mean squared error over
HAR's. Against chance: the feature's columns rolled within each day by at least
`min_shift`, which keeps their values and their link to each other and destroys their
timing, then fitted and scored again. Against choosing: PBO over the variants. Against
many tests: Benjamini-Hochberg over the registry.

Features:

- `hawkes`: from rolling fits of the two-sided Hawkes process of market orders
  (`events.hawkes`), the log of the total intensity, its change over the last minute, and
  the branching ratio. The share of fits the time-rescaling check accepts is reported.
- `vpin`: VPIN with its bucket volume set from the development days' volume per valid hour
  (a unit, not fitted to the target; the held-out days use the same).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from . import __version__
from . import experiment as ex
from .events import hawkes as hk
from .features import VPIN, Volatility, duration_ns, grid
from .features.trades import _trades
from .io.day import Day
from .io.days import DayInfo, survey
from .targets import future_volatility
from .validation import pbo, splits

CACHE_VERSION = 1
HAR = ("10s", "60s", "300s")
METHODS = ("hawkes", "vpin")


@dataclass(frozen=True)
class Variant:
    method: str
    params: tuple[tuple[str, str], ...]

    @property
    def key(self) -> str:
        return self.method + "(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"

    def get(self, name: str) -> str:
        return dict(self.params)[name]


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    symbol: str
    min_hours: float
    holdout: tuple[str, ...]
    min_days: int
    sampling: str
    horizon: str
    floor_bp: float
    primary: Variant
    variants: tuple[Variant, ...]
    folds: int
    embargo: str
    surrogates: int
    min_shift: str
    seed: int
    min_gain: float
    alpha: float
    surrogate_percentile: float
    max_pbo: float
    min_positive_days: float
    path: Path = field(compare=False)

    @property
    def slug(self) -> str:
        return f"{self.id}-{self.name}"


REQUIRED = {"hawkes": ("window", "step"), "vpin": ("buckets_per_day", "window")}


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    known = {
        "id",
        "name",
        "kind",
        "hypothesis",
        "data",
        "sampling",
        "target",
        "feature",
        "variants",
        "protocol",
        "success",
    }
    if not isinstance(raw, dict) or set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw or {}) - known)}")
    if raw.get("kind") != "volatility":
        raise ex.ExperimentError(f"{path}: not a `volatility` hypothesis")
    g = ex._get
    data, target, feat = raw.get("data", {}), g(raw, "target", "top"), g(raw, "feature", "top")
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    method = str(g(feat, "method", "feature"))
    if method not in METHODS:
        raise ex.ExperimentError(f"{path}: the feature must be one of {METHODS}")

    def variant(v: dict) -> Variant:
        missing = [k for k in REQUIRED[method] if k not in v]
        if missing:
            raise ex.ExperimentError(f"{path}: a {method} variant needs {missing}")
        return Variant(method, tuple((k, str(v[k])) for k in REQUIRED[method]))

    primary = variant(g(feat, "params", "feature"))
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
        sampling=str(raw.get("sampling", "10s")),
        horizon=str(g(target, "horizon", "target")),
        floor_bp=float(target.get("floor_bp", 0.1)),
        primary=primary,
        variants=variants,
        folds=int(proto.get("folds", 5)),
        embargo=str(proto.get("embargo", "300s")),
        surrogates=int(proto.get("surrogates", 200)),
        min_shift=str(proto.get("min_shift", "10m")),
        seed=int(proto.get("seed", 1)),
        min_gain=float(ok.get("min_gain", 0.0)),
        alpha=float(ok.get("alpha", 0.05)),
        surrogate_percentile=float(ok.get("surrogate_percentile", 95)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_days=float(ok.get("min_positive_days", 0.6)),
        path=path,
    )
    for d in (spec.sampling, spec.horizon, spec.embargo, spec.min_shift):
        duration_ns(d)
    for v in variants:
        if method == "hawkes":
            duration_ns(v.get("window")), duration_ns(v.get("step"))
        else:
            int(v.get("buckets_per_day")), int(v.get("window"))
    if spec.surrogates < 200:
        raise ex.ExperimentError(f"{path}: the protocol asks for at least 200 surrogates")
    return spec


# --- one day's inputs --------------------------------------------------------------------


def _log(spec: Spec, rv: np.ndarray) -> np.ndarray:
    return np.log(spec.floor_bp + rv)


def hawkes_columns(day: Day, query: np.ndarray, v: Variant) -> tuple[dict, dict]:
    """At each query time: log total intensity, its change over the minute before, and the
    branching ratio, from the fits made before it; and the fits' diagnostics."""
    t = day.valid("trades", ["recv_ts", "trade_ts", "is_buyer_maker"])
    ts, side, _ = hk.market_orders(
        t["recv_ts"].to_numpy(),
        t["trade_ts"].to_numpy(),
        ~t["is_buyer_maker"].to_numpy().astype(bool),
        t["segment"].to_numpy(),
    )
    step = duration_ns(v.get("step"))
    r = hk.rolling(ts, side, day.intervals, duration_ns(v.get("window")), step)
    minute = 60 * 1_000_000_000
    lam, br = hk.at_times(r, ts, side, query, step)
    lam_before, _ = hk.at_times(r, ts, side, query - minute, step)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_now = np.log(lam.sum(axis=1))
        log_before = np.log(lam_before.sum(axis=1))
    fitted = np.isfinite(r.params[:, 6])
    diag = {
        "fits": int(fitted.sum()),
        "windows": len(r.at),
        "ks_pass_share": ex._f(np.mean((r.ks_p[fitted] > 0.05).all(axis=1)))
        if fitted.any()
        else None,
        "branching_median": ex._f(np.median(r.branching[fitted])) if fitted.any() else None,
        "beta_median": ex._f(np.median(r.params[fitted, 6])) if fitted.any() else None,
    }
    cols = {
        f"{v.key}:loglam": log_now,
        f"{v.key}:dloglam": log_now - log_before,
        f"{v.key}:branching": br,
    }
    return cols, diag


def vpin_columns(day: Day, rows: np.ndarray, v: Variant, unit_base: float) -> dict:
    """VPIN at the sampled grid rows, buckets of `unit_base` base units over `buckets_per_day`."""
    vp = VPIN(buckets_per_day=int(v.get("buckets_per_day")), window=int(v.get("window")))
    vp.bucket_ = max(1, round(unit_base / float(day.qty_step)))
    x = vp.transform(day)["vpin"].to_numpy()
    return {f"{v.key}:vpin": x[rows]}


def bucket_units(infos: Sequence[DayInfo], spec: Spec) -> dict[str, float]:
    """For each VPIN variant: base volume per valid second of these days, scaled to a day,
    over its buckets per day."""
    out = {}
    if spec.primary.method != "vpin":
        return out
    volume, seconds = 0.0, 0.0
    for info in infos:
        d = Day.open(info.path)
        volume += float(_trades(d)["qty"].sum()) * float(d.qty_step)
        seconds += sum(e - s for s, e in d.intervals) / 1e9
    for v in spec.variants:
        out[v.key] = volume / seconds * 86_400 / int(v.get("buckets_per_day"))
    return out


def day_inputs(path: Path, spec: Spec, units: dict[str, float]) -> tuple[pl.DataFrame, dict]:
    day = Day.open(path)
    g = grid(day)
    rows = ex._sample_rows(g.ts, g.segment, duration_ns(spec.sampling))
    ts = g.ts[rows]
    vol = Volatility(windows=tuple(dict.fromkeys((*HAR, spec.horizon)))).columns(g)
    fut = future_volatility(day, [spec.horizon])[f"vol_{spec.horizon}"].to_numpy()
    cols: dict[str, np.ndarray] = {"ts": ts, "segment": g.segment[rows]}
    for w in HAR:
        cols[f"har_{w}"] = _log(spec, vol[f"rv_{w}"][rows])
    cols["persistence"] = _log(spec, vol[f"rv_{spec.horizon}"][rows])
    cols["target"] = _log(spec, fut[rows])
    diag = {}
    for v in spec.variants:
        if v.method == "hawkes":
            c, diag[v.key] = hawkes_columns(day, ts, v)
        else:
            c = vpin_columns(day, rows, v, units[v.key])
        cols.update(c)
    df = pl.DataFrame(cols).with_columns(pl.all().exclude("ts", "segment").fill_nan(None))
    return df, diag


def _cache_file(cache: Path, info: DayInfo, spec: Spec, units: dict[str, float]) -> Path:
    manifest = (info.path / "manifest.json").read_bytes()
    parts = [
        str(CACHE_VERSION),
        __version__,
        hashlib.sha256(manifest).hexdigest(),
        spec.sampling,
        spec.horizon,
        repr(spec.floor_bp),
        "|".join(v.key for v in spec.variants),
        repr(sorted(units.items())),
    ]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache / spec.symbol / info.date / f"vol-{key}.parquet"


def load_inputs(
    info: DayInfo, spec: Spec, cache: Path | None, units: dict[str, float]
) -> tuple[pl.DataFrame, dict]:
    if cache is not None:
        f = _cache_file(cache, info, spec, units)
        if f.exists() and f.with_suffix(".json").exists():
            return pl.read_parquet(f), json.loads(f.with_suffix(".json").read_text())
    df, diag = day_inputs(info.path, spec, units)
    if cache is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(f.with_suffix(".tmp"))
        f.with_suffix(".tmp").replace(f)
        f.with_suffix(".json").write_text(json.dumps(diag))
    return df, diag


def _load_one(task):
    return load_inputs(*task)


def load_all(infos, spec: Spec, cache: Path | None, units: dict, workers: int) -> list:
    tasks = [(i, spec, cache, units) for i in infos]
    if workers <= 1 or len(tasks) <= 1:
        return [_load_one(t) for t in tasks]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(_load_one, tasks))
    except BrokenProcessPool as e:
        raise ex.ExperimentError(
            f"a worker was stopped from outside, most likely for want of memory: run again "
            f"with fewer --workers than {workers} (days already done are cached)"
        ) from e


# --- fitting and scoring -----------------------------------------------------------------


class Samples:
    def __init__(self, frames: Sequence[pl.DataFrame]):
        self.df = pl.concat(frames) if frames else pl.DataFrame()
        self.day = np.concatenate([np.full(len(f), i) for i, f in enumerate(frames)])
        self.n_days = len(frames)

    def matrix(self, cols: Sequence[str]) -> np.ndarray:
        return np.column_stack([self.df[c].to_numpy().astype(float) for c in cols])


def feature_columns(v: Variant) -> list[str]:
    if v.method == "hawkes":
        return [f"{v.key}:loglam", f"{v.key}:dloglam", f"{v.key}:branching"]
    return [f"{v.key}:vpin"]


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), x])
    w, *_ = np.linalg.lstsq(a, y, rcond=None)
    return w


def _apply(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return w[0] + x @ w[1:]


@dataclass
class DayLoss:
    persistence: float
    har: float
    candidate: float
    rows: int

    @property
    def gain(self) -> float:
        """1 - MSE(candidate) / MSE(HAR): the share of HAR's error the feature removes."""
        return 1.0 - self.candidate / self.har if self.har > 0 else float("nan")

    @property
    def gain_persistence(self) -> float:
        return 1.0 - self.candidate / self.persistence if self.persistence > 0 else float("nan")


def evaluate(
    s: Samples, spec: Spec, v: Variant, feats: np.ndarray | None = None
) -> tuple[list[DayLoss], np.ndarray]:
    """Out-of-sample losses by day over a purged k-fold, on the rows where everything is
    known; `feats` replaces the feature's columns (a surrogate). Also the test rows' squared
    errors of HAR and the candidate (for the stability report), NaN elsewhere."""
    har = s.matrix([f"har_{w}" for w in HAR])
    f = s.matrix(feature_columns(v)) if feats is None else feats
    y = s.df["target"].to_numpy().astype(float)
    p = s.df["persistence"].to_numpy().astype(float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(har).all(axis=1) & np.isfinite(f).all(axis=1)
    idx = np.flatnonzero(ok)
    nan_errors = np.full((len(y), 2), np.nan)
    if len(np.unique(s.day[idx])) < max(2, spec.folds):
        # the feature is known on too few days to fold: nothing to score
        return [DayLoss(np.nan, np.nan, np.nan, 0) for _ in range(s.n_days)], nan_errors
    ts = s.df["ts"].to_numpy()
    t0 = ts[idx] - duration_ns(HAR[-1])
    t1 = ts[idx] + duration_ns(spec.horizon)
    folds = splits.purged_kfold(s.day[idx], t0, t1, spec.folds, duration_ns(spec.embargo))
    e_har = np.full(len(y), np.nan)
    e_cand = np.full(len(y), np.nan)
    for fo in folds:
        tr, te = idx[fo.train], idx[fo.test]
        wh = _ols(har[tr], y[tr])
        wc = _ols(np.column_stack([har[tr], f[tr]]), y[tr])
        e_har[te] = (y[te] - _apply(wh, har[te])) ** 2
        e_cand[te] = (y[te] - _apply(wc, np.column_stack([har[te], f[te]]))) ** 2
    e_per = np.where(ok, (y - p) ** 2, np.nan)
    out = []
    for d in range(s.n_days):
        m = (s.day == d) & np.isfinite(e_cand)
        if m.sum() < 30:
            out.append(DayLoss(np.nan, np.nan, np.nan, int(m.sum())))
            continue
        out.append(DayLoss(e_per[m].mean(), e_har[m].mean(), e_cand[m].mean(), int(m.sum())))
    return out, np.column_stack([e_har, e_cand])


def _rolled(s: Samples, v: Variant, rng: np.random.Generator, min_rows: int) -> np.ndarray:
    """The feature's columns, each day's block rolled by one random number of rows."""
    f = s.matrix(feature_columns(v))
    out = np.empty_like(f)
    for d in range(s.n_days):
        r = np.flatnonzero(s.day == d)
        n = len(r)
        lag = int(rng.integers(min_rows, n - min_rows + 1)) if n > 2 * min_rows else 0
        out[r] = np.roll(f[r], lag, axis=0)
    return out


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    days: list[DayInfo]
    by_day: dict[str, DayLoss]
    mean_gain: float
    se: float
    p: float
    positive_share: float
    mean_gain_persistence: float
    surrogate: dict
    pbo: pbo.PBO | None
    variants: dict[str, float | None]
    stability: dict
    diagnostics: dict
    units: dict
    p_bh: float
    verdict: str
    reasons: list[str]
    skipped: list[DayInfo] = field(default_factory=list)

    def to_json(self) -> dict:
        s, f = self.spec, ex._f
        return {
            "id": s.id,
            "name": s.name,
            "kind": "volatility",
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
            "feature": s.primary.key,
            "target": {"horizon": s.horizon, "floor_bp": s.floor_bp, "sampling": s.sampling},
            "by_day": {
                d: {
                    "mse_persistence": f(x.persistence),
                    "mse_har": f(x.har),
                    "mse_candidate": f(x.candidate),
                    "gain": f(x.gain),
                    "rows": x.rows,
                }
                for d, x in self.by_day.items()
            },
            "mean_gain": f(self.mean_gain),
            "se": f(self.se),
            "positive_share": f(self.positive_share),
            "mean_gain_over_persistence": f(self.mean_gain_persistence),
            "surrogates": self.surrogate,
            "pbo": asdict(self.pbo) if self.pbo else None,
            "variants": self.variants,
            "stability": self.stability,
            "diagnostics": self.diagnostics,
            "vpin_bucket_base_units": self.units,
            "p_raw": f(self.p),
            "p_bh": f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _summary(losses: list[DayLoss]) -> tuple[float, float, float, float]:
    g = np.array([x.gain for x in losses])
    g = g[np.isfinite(g)]
    if len(g) < 2:
        return float("nan"), float("nan"), 1.0, float("nan")
    mean, se = float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g)))
    p = float(stats.t.sf(mean / se, len(g) - 1)) if se > 0 else (0.0 if mean > 0 else 1.0)
    return mean, se, p, float((g > 0).mean())


def _stability(s: Samples, errors: np.ndarray) -> dict:
    """The gain pooled over the days, by time of day and by the volatility of the minute
    before."""
    hour = (s.df["ts"].to_numpy() // 3_600_000_000_000) % 24
    vol = s.df["har_60s"].to_numpy().astype(float)
    eh, ec = errors[:, 0], errors[:, 1]

    def gain(mask: np.ndarray) -> float | None:
        m = mask & np.isfinite(ec)
        return ex._f(1 - ec[m].mean() / eh[m].mean()) if m.sum() >= 100 else None

    times = {
        f"{lo:02d}-{lo + 6:02d} UTC": gain((hour >= lo) & (hour < lo + 6)) for lo in range(0, 24, 6)
    }
    ok = np.isfinite(vol) & np.isfinite(ec)
    out = {"time_of_day": times, "volatility_tercile": {}}
    if ok.any():
        q1, q2 = np.percentile(vol[ok], [100 / 3, 200 / 3])
        out["volatility_tercile"] = {
            "low": gain(vol <= q1),
            "middle": gain((vol > q1) & (vol <= q2)),
            "high": gain(vol > q2),
        }
    return out


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
    needed = max(spec.min_days, spec.folds)
    if len(infos) < needed:
        reason = (
            f"{len(infos)} days with {spec.min_hours:g} usable hours, "
            f"the hypothesis asks for {needed}"
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
            nan,
            {},
            None,
            {},
            {},
            {},
            {},
            nan,
            "insufficient data",
            [reason],
            skipped,
        )
        return _finish(r, results)
    units = bucket_units(infos, spec)
    loaded = load_all(infos, spec, cache, units, workers)
    s = Samples([df for df, _ in loaded])
    losses, errors = evaluate(s, spec, spec.primary)
    scored = sum(np.isfinite(x.gain) for x in losses)
    if scored < needed:
        reason = (
            f"the feature and the target are known together on {scored} days, "
            f"the hypothesis asks for {needed}"
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
            nan,
            {},
            None,
            {},
            {},
            {},
            units,
            nan,
            "insufficient data",
            [reason],
            skipped,
        )
        return _finish(r, results)
    mean, se, p, positive = _summary(losses)
    gp = np.array([x.gain_persistence for x in losses])
    mean_p = float(np.nanmean(gp)) if np.isfinite(gp).any() else nan

    perf = np.full((len(infos), len(spec.variants)), np.nan)
    variant_gain = {}
    for j, v in enumerate(spec.variants):
        lv = losses if v == spec.primary else evaluate(s, spec, v)[0]
        perf[:, j] = [x.gain for x in lv]
        variant_gain[v.key] = _summary(lv)[0]
        variant_gain[v.key] = ex._f(variant_gain[v.key])
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.variants) > 1 else None

    min_rows = max(1, int(np.ceil(duration_ns(spec.min_shift) / duration_ns(spec.sampling))))
    values = []
    for k in range(spec.surrogates):
        rng = np.random.default_rng([spec.seed, k])
        sl, _ = evaluate(s, spec, spec.primary, _rolled(s, spec.primary, rng, min_rows))
        values.append(_summary(sl)[0])
    a = np.array(values)
    a = a[np.isfinite(a)]
    surrogate = {
        "shift": {
            "n": len(a),
            "percentile": spec.surrogate_percentile,
            "threshold": ex._f(np.percentile(a, spec.surrogate_percentile)) if len(a) else None,
            "real": ex._f(mean),
            "p": float((1 + np.sum(a >= mean)) / (1 + len(a))) if len(a) else None,
            "mean": ex._f(a.mean()) if len(a) else None,
        }
    }
    stability = _stability(s, errors)
    diagnostics = {i.date: diag for i, (_, diag) in zip(infos, loaded, strict=True) if diag}
    p_bh = ex._bh_with_registry(results, spec, p)

    reasons = []
    if not mean > spec.min_gain:
        reasons.append(f"mean gain over HAR {mean:.4f} is not above {spec.min_gain}")
    if not p_bh < spec.alpha:
        reasons.append(f"p after Benjamini-Hochberg {p_bh:.3g} is not below {spec.alpha}")
    sh = surrogate["shift"]
    if sh["threshold"] is None or not mean > sh["threshold"]:
        reasons.append(
            f"not above the {spec.surrogate_percentile:g}th percentile of shifted features"
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
        {i.date: x for i, x in zip(infos, losses, strict=True)},
        mean,
        se,
        p,
        positive,
        mean_p,
        surrogate,
        result_pbo,
        variant_gain,
        stability,
        diagnostics,
        units,
        p_bh,
        "failed" if reasons else "passed",
        reasons,
        skipped,
    )
    return _finish(r, results)


def _finish(r: Report, results: Path) -> Report:
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
    """HAR and the candidate fitted on every development day, scored on the held-out days,
    once."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or ex.provenance(spec.path)
    every = survey(data_root, spec.symbol)
    dev = [d for d in every if d.usable_hours >= spec.min_hours and d.date not in spec.holdout]
    held = [d for d in every if d.usable_hours >= spec.min_hours and d.date in spec.holdout]
    if not held:
        raise ex.ExperimentError("none of the held-out days is in the data")
    units = bucket_units(dev, spec)
    v = spec.primary
    train = Samples([load_inputs(i, spec, cache, units)[0] for i in dev])
    test = Samples([load_inputs(i, spec, cache, units)[0] for i in held])

    def parts(s: Samples):
        har = s.matrix([f"har_{w}" for w in HAR])
        f = s.matrix(feature_columns(v))
        y = s.df["target"].to_numpy().astype(float)
        p = s.df["persistence"].to_numpy().astype(float)
        ok = (
            np.isfinite(y)
            & np.isfinite(p)
            & np.isfinite(har).all(axis=1)
            & np.isfinite(f).all(axis=1)
        )
        return har, f, y, p, ok

    har, f, y, _, ok = parts(train)
    wh, wc = _ols(har[ok], y[ok]), _ols(np.column_stack([har[ok], f[ok]]), y[ok])
    har, f, y, p, ok = parts(test)
    by_day = {}
    for d, info in enumerate(held):
        m = ok & (test.day == d)
        if m.sum() < 30:
            by_day[info.date] = None
            continue
        x = DayLoss(
            float(((y[m] - p[m]) ** 2).mean()),
            float(((y[m] - _apply(wh, har[m])) ** 2).mean()),
            float(((y[m] - _apply(wc, np.column_stack([har[m], f[m]]))) ** 2).mean()),
            int(m.sum()),
        )
        by_day[info.date] = {
            "gain": ex._f(x.gain),
            "gain_over_persistence": ex._f(x.gain_persistence),
        }
    gains = [x["gain"] for x in by_day.values() if x and x["gain"] is not None]
    record = {
        "id": spec.id,
        "name": spec.name,
        "kind": "volatility",
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": [i.date for i in held],
        "by_day": by_day,
        "mean_gain": ex._f(np.mean(gains)) if gains else None,
        "weights_har": [float(x) for x in wh],
        "weights_candidate": [float(x) for x in wc],
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
