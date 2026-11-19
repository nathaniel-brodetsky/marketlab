"""Running a hypothesis through the protocol, from a file committed before the first run.

An experiment is `experiments/NNN-name.yaml`: data, features, the target and horizon, the
metric and the bar it has to clear, written down before anything is computed. The runner
refuses a file that is not committed or has changed since, and records the commit, so a
result can always be traced to the hypothesis as it stood.

The protocol, in the order of the specification:

1. The hypothesis before the run (the file and its commit).
2. Splits by time: purged k-fold over days for the out-of-sample predictions, and CPCV for
   the final estimate as a distribution over backtest paths, not one number. The last
   days are held out and opened once, at the end (`--final`).
3. Surrogate data: the out-of-sample prediction against the target, with their alignment
   destroyed in three ways, at least 200 times each; the real result has to beat the 95th
   percentile of each.
4. Multiple tests: Benjamini-Hochberg over every hypothesis in `results/`, failed ones
   included; PBO over the variants of this one.
5. Costs: for trading rules, through the fill simulator (no trading rule yet).
6. Stability: by day, by time of day and by volatility.

Each run writes `results/NNN-name.json` with one of three verdicts: `passed`, `failed`,
`insufficient data`. Failed ones stay.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import __version__
from .features import MultiLevelOFI, Volatility, duration_ns, grid
from .features.ofi import mid_change
from .io.day import Day
from .io.days import DayInfo, survey
from .targets import returns_on
from .validation import metrics, pbo, splits, surrogates
from .validation.multiple import benjamini_hochberg

#: Raised when cached inputs would differ: bump it whenever their computation changes.
CACHE_VERSION = 1

VERDICTS = ("passed", "failed", "insufficient data")


class ExperimentError(Exception):
    pass


# --- the specification -----------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    levels: int
    window: str


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
    feature: str
    primary: Variant
    variants: tuple[Variant, ...]
    horizon: str
    folds: int
    cpcv_test_groups: int
    embargo: str
    surrogates: int
    seed: int
    min_ic: float
    alpha: float
    surrogate_percentile: float
    max_pbo: float
    min_positive_days: float
    path: Path = field(compare=False)

    @property
    def slug(self) -> str:
        return f"{self.id}-{self.name}"


def _get(d: dict, key: str, where: str):
    if key not in d:
        raise ExperimentError(f"{where}: `{key}` is missing")
    return d[key]


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ExperimentError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise ExperimentError(f"{path}: not a mapping")
    known = {
        "id",
        "name",
        "kind",
        "hypothesis",
        "data",
        "sampling",
        "feature",
        "variants",
        "target",
        "protocol",
        "success",
    }
    if raw.get("kind", "forecast") != "forecast":
        raise ExperimentError(f"{path}: a `{raw['kind']}` hypothesis, not a forecast")
    unknown = set(raw) - known
    if unknown:
        raise ExperimentError(f"{path}: unknown keys {sorted(unknown)}")
    data, feat, target = raw.get("data", {}), raw.get("feature", {}), raw.get("target", {})
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    if _get(feat, "method", "feature") != "mlofi":
        raise ExperimentError(f"{path}: only `mlofi` is supported as a feature so far")
    if _get(target, "kind", "target") != "return":
        raise ExperimentError(f"{path}: only `return` is supported as a target so far")

    def variant(v: dict) -> Variant:
        return Variant(int(_get(v, "levels", "variant")), str(_get(v, "window", "variant")))

    primary = variant(_get(feat, "params", "feature"))
    variants = tuple(variant(v) for v in raw.get("variants", [])) or (primary,)
    if primary not in variants:
        variants = (primary, *variants)
    spec = Spec(
        id=str(_get(raw, "id", "top")),
        name=str(_get(raw, "name", "top")),
        hypothesis=" ".join(str(_get(raw, "hypothesis", "top")).split()),
        symbol=str(_get(data, "symbol", "data")),
        min_hours=float(data.get("min_hours", 20)),
        holdout=tuple(str(d) for d in data.get("holdout", [])),
        min_days=int(data.get("min_days", 10)),
        sampling=str(raw.get("sampling", "1s")),
        feature="mlofi",
        primary=primary,
        variants=variants,
        horizon=str(_get(target, "horizon", "target")),
        folds=int(proto.get("folds", 5)),
        cpcv_test_groups=int(proto.get("cpcv_test_groups", 2)),
        embargo=str(proto.get("embargo", "60s")),
        surrogates=int(proto.get("surrogates", 200)),
        seed=int(proto.get("seed", 1)),
        min_ic=float(ok.get("min_ic", 0.0)),
        alpha=float(ok.get("alpha", 0.05)),
        surrogate_percentile=float(ok.get("surrogate_percentile", 95)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_days=float(ok.get("min_positive_days", 0.6)),
        path=path,
    )
    for d in (spec.sampling, spec.horizon, spec.embargo, *(v.window for v in variants)):
        duration_ns(d)  # raises on a bad duration
    if spec.surrogates < 200:
        raise ExperimentError(f"{path}: the protocol asks for at least 200 surrogates of each kind")
    return spec


# --- the commit the hypothesis was written in ------------------------------------------


@dataclass(frozen=True)
class Provenance:
    spec_commit: str
    head: str
    dirty: bool


def _git(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def provenance(spec_path: Path) -> Provenance:
    """The spec must be committed and unchanged since. Uncommitted changes elsewhere in the
    code are allowed but recorded: a result from such a tree is marked dirty."""
    where = spec_path.resolve().parent
    top = _git(["rev-parse", "--show-toplevel"], where)
    if top.returncode != 0:
        raise ExperimentError(f"{spec_path}: not in a git repository; commit it before running")
    root = Path(top.stdout.strip())
    rel = str(spec_path.resolve().relative_to(root))
    if _git(["ls-files", "--error-unmatch", rel], root).returncode != 0:
        raise ExperimentError(f"{rel} is not committed: commit the hypothesis before running it")
    changed = _git(["status", "--porcelain", "--", rel], root).stdout.strip()
    if changed:
        raise ExperimentError(f"{rel} has changed since its commit: commit it, as a new hypothesis")
    commit = _git(["log", "-1", "--format=%H", "--", rel], root).stdout.strip()
    head = _git(["rev-parse", "HEAD"], root).stdout.strip()
    dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], root).stdout.strip())
    return Provenance(commit, head, dirty)


# --- per-day inputs, cached --------------------------------------------------------------


def _sample_rows(ts: np.ndarray, seg: np.ndarray, every: int) -> np.ndarray:
    """The first grid row of each `every` ns of each segment."""
    if len(ts) == 0:
        return np.zeros(0, dtype=np.int64)
    bucket = ts // every
    first = np.ones(len(ts), dtype=bool)
    first[1:] = (bucket[1:] != bucket[:-1]) | (seg[1:] != seg[:-1])
    return np.flatnonzero(first)


def day_inputs(day: Day, spec: Spec) -> pl.DataFrame:
    """For the sampled rows of a day: each window's per-level OFI and contemporaneous mid
    change (what the weights are fitted on), the target, and the volatility of the last
    minute (for the stability report)."""
    g = grid(day)
    rows = _sample_rows(g.ts, g.segment, duration_ns(spec.sampling))
    levels = max(v.levels for v in spec.variants)
    cols: dict[str, np.ndarray] = {"ts": g.ts[rows], "segment": g.segment[rows]}
    for w in sorted({v.window for v in spec.variants}):
        x = MultiLevelOFI(levels=levels, window=w).level_windows(day, g)
        for m in range(levels):
            cols[f"ofi_l{m + 1}_{w}"] = x[rows, m]
        cols[f"dmid_{w}"] = mid_change(g, duration_ns(w))[rows]
    _, ends = day.segment_bounds()
    cols["target"] = returns_on(g, ends, [spec.horizon])[f"ret_{spec.horizon}"][rows]
    cols["rv_60s"] = Volatility(windows=("60s",)).columns(g)["rv_60s"][rows]
    return pl.DataFrame(cols).with_columns(pl.all().exclude("ts", "segment").fill_null(np.nan))


def _cache_key(info: DayInfo, spec: Spec) -> str:
    manifest = (info.path / "manifest.json").read_bytes()
    parts = [
        str(CACHE_VERSION),
        __version__,
        hashlib.sha256(manifest).hexdigest(),
        spec.sampling,
        spec.horizon,
        str(max(v.levels for v in spec.variants)),
        ",".join(sorted({v.window for v in spec.variants})),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_inputs(info: DayInfo, spec: Spec, cache: Path | None) -> pl.DataFrame:
    if cache is not None:
        f = cache / spec.symbol / info.date / f"{_cache_key(info, spec)}.parquet"
        if f.exists():
            return pl.read_parquet(f)
    df = day_inputs(Day.open(info.path), spec)
    if cache is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.replace(f)
    return df


# --- fitting and predicting -----------------------------------------------------------


class Samples:
    """Every day's sampled rows end to end, with the day each row belongs to."""

    def __init__(self, inputs: Sequence[pl.DataFrame]):
        self.inputs = list(inputs)
        self.df = pl.concat(self.inputs) if self.inputs else pl.DataFrame()
        self.day = np.concatenate([np.full(len(d), i) for i, d in enumerate(self.inputs)])
        self.target = self.df["target"].to_numpy() if len(self.df) else np.zeros(0)
        self._x: dict[Variant, np.ndarray] = {}

    def x(self, v: Variant) -> np.ndarray:
        if v not in self._x:
            self._x[v] = np.column_stack(
                [self.df[f"ofi_l{m + 1}_{v.window}"].to_numpy() for m in range(v.levels)]
            )
        return self._x[v]

    def rows_of(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.day == day)


def fit(s: Samples, rows: np.ndarray, v: Variant) -> np.ndarray:
    """Least squares of the mid's change over the window on the levels' OFI over it, as in
    Xu, Gould, Howison (2019), on the given rows."""
    x = s.x(v)[rows]
    y = s.df[f"dmid_{v.window}"].to_numpy()[rows]
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() <= v.levels:
        raise ExperimentError("too few training rows to fit the weights")
    w, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    return w


def predict(s: Samples, rows: np.ndarray, v: Variant, w: np.ndarray) -> np.ndarray:
    x = s.x(v)[rows]
    out = np.zeros(len(rows))
    for m in range(v.levels):  # level by level: the same row gives the same value
        out = out + x[:, m] * w[m]
    return out


def oos_predictions(s: Samples, folds: Sequence[splits.Split], v: Variant) -> np.ndarray:
    """Out-of-sample predictions: every test row from the fit on its fold's purged
    training rows."""
    pred = np.full(len(s.target), np.nan)
    for f in folds:
        pred[f.test] = predict(s, f.test, v, fit(s, f.train, v))
    return pred


def day_ics(s: Samples, pred: np.ndarray, lags: int) -> list[metrics.DayIC]:
    out = []
    for i in range(len(s.inputs)):
        r = s.rows_of(i)
        out.append(metrics.day_ic(pred[r], s.target[r], lags))
    return out


# --- the run ----------------------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: Provenance
    days: list[DayInfo]
    lags: int
    kfold: metrics.Summary
    by_day: dict[str, float | None]
    cpcv_paths: list[float]
    surrogate: dict[str, dict[str, float | None]]
    pbo: pbo.PBO | None
    variants: dict[str, float | None]
    stability: dict[str, dict[str, float | None]]
    p_bh: float
    verdict: str
    reasons: list[str]
    #: Days left out for having fewer usable hours than the hypothesis asks.
    skipped: list[DayInfo] = field(default_factory=list)

    def to_json(self) -> dict:
        s = self.spec
        paths = np.array(self.cpcv_paths)
        return {
            "id": s.id,
            "name": s.name,
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
            "feature": {"method": s.feature, "primary": asdict(s.primary)},
            "target": {"kind": "return", "horizon": s.horizon},
            "sampling": s.sampling,
            "newey_west_lags": self.lags,
            "kfold": {k: _f(v) for k, v in asdict(self.kfold).items()},
            "ic_by_day": self.by_day,
            "cpcv": {
                "paths": len(paths),
                "mean": _f(paths.mean()) if len(paths) else None,
                "std": _f(paths.std()) if len(paths) else None,
                "min": _f(paths.min()) if len(paths) else None,
                "positive_share": _f((paths > 0).mean()) if len(paths) else None,
                "path_ics": [_f(x) for x in paths],
            },
            "surrogates": self.surrogate,
            "pbo": asdict(self.pbo) if self.pbo else None,
            "variants": self.variants,
            "stability": self.stability,
            "p_raw": _f(self.kfold.p),
            "p_bh": _f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _f(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _variant_name(v: Variant) -> str:
    return f"levels {v.levels}, window {v.window}"


def run(
    spec: Spec,
    data_root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: Provenance | None = None,
    workers: int = 1,
) -> Report:
    prov = prov or provenance(spec.path)
    every = survey(data_root, spec.symbol, exclude=spec.holdout)
    infos = [d for d in every if d.usable_hours >= spec.min_hours]
    skipped = [d for d in every if d.usable_hours < spec.min_hours]
    lags = max(1, ceil(duration_ns(spec.horizon) / duration_ns(spec.sampling)))
    nan = float("nan")
    needed = max(spec.min_days, spec.folds, spec.cpcv_test_groups + 1)
    if len(infos) < needed:
        empty = metrics.Summary(nan, nan, nan, nan, nan, 0)
        reason = (
            f"{len(infos)} days with {spec.min_hours:g} usable hours, "
            f"the hypothesis asks for {needed}"
        )
        report = Report(
            spec,
            prov,
            infos,
            lags,
            empty,
            {},
            [],
            {},
            None,
            {},
            {},
            nan,
            "insufficient data",
            [reason],
            skipped,
        )
        return _finish(report, results)
    s = Samples([load_inputs(i, spec, cache) for i in infos])
    # each row's span: from the start of its feature window to the end of its target
    ts = s.df["ts"].to_numpy()
    window = max(duration_ns(v.window) for v in spec.variants)
    t0, t1 = ts - window, ts + duration_ns(spec.horizon)
    embargo = duration_ns(spec.embargo)

    kf = splits.purged_kfold(s.day, t0, t1, spec.folds, embargo)
    pred = oos_predictions(s, kf, spec.primary)
    ics = day_ics(s, pred, lags)
    kfold = metrics.combine(ics)
    by_day = {i.date: _f(d.ic) for i, d in zip(infos, ics, strict=True)}

    # CPCV: the primary variant over every combination of test days, then the paths
    cv = splits.cpcv(s.day, t0, t1, spec.cpcv_test_groups, embargo)
    split_ics: list[dict[int, float]] = []
    for f in cv:
        w = fit(s, f.train, spec.primary)
        p = np.full(len(s.target), np.nan)
        p[f.test] = predict(s, f.test, spec.primary, w)
        split_ics.append(
            {
                g: metrics.day_ic(p[s.rows_of(g)], s.target[s.rows_of(g)], lags).ic
                for g in f.test_groups
            }
        )
    paths = [
        float(np.nanmean([split_ics[k][g] for g, k in path.items()]))
        for path in splits.cpcv_paths(cv)
    ]

    surrogate = _surrogates(s, pred, kfold.mean, spec, workers)

    # every variant's out-of-sample IC by day: the variants table and PBO
    perf = np.full((len(infos), len(spec.variants)), np.nan)
    variant_ic: dict[str, float | None] = {}
    for j, v in enumerate(spec.variants):
        vi = ics if v == spec.primary else day_ics(s, oos_predictions(s, kf, v), lags)
        perf[:, j] = [d.ic for d in vi]
        variant_ic[_variant_name(v)] = _f(metrics.combine(vi).mean)
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.variants) > 1 else None

    stability = _stability(s, pred, lags, kfold)
    p_bh = _bh_with_registry(results, spec, kfold.p)
    verdict, reasons = _verdict(spec, kfold, paths, surrogate, result_pbo, p_bh)
    report = Report(
        spec,
        prov,
        infos,
        lags,
        kfold,
        by_day,
        paths,
        surrogate,
        result_pbo,
        variant_ic,
        stability,
        p_bh,
        verdict,
        reasons,
        skipped,
    )
    return _finish(report, results)


_DAYS: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
_SHAPE: tuple[int, int] = (2, 2)


def _init_workers(days, block: int, min_lag: int) -> None:
    global _DAYS, _SHAPE
    _DAYS, _SHAPE = days, (block, min_lag)


def _surrogate_ic(task: tuple[str, int, int]) -> float:
    """One surrogate's mean IC over the days; its random numbers depend only on the seed,
    the kind and its number, so any number of workers gives the same values."""
    kind, k, seed = task
    rng = np.random.default_rng([seed, surrogates.KINDS.index(kind), k])
    block, min_lag = _SHAPE
    ics = []
    for zp, zt, raw in _DAYS:
        if kind == "block":
            a, b = surrogates.block_shuffle(zp, block, rng), zt
        elif kind == "iaaft":
            a, b = metrics._z(surrogates.iaaft(raw, rng)), zt
        else:
            a, b = zp, surrogates.shift(zt, min_lag, rng)
        ics.append(float(np.mean(a * b)))
    return float(np.mean(ics))


def _surrogates(s: Samples, pred: np.ndarray, real: float, spec: Spec, workers: int) -> dict:
    """The real mean IC against the same computation with the prediction and the target
    misaligned, day by day: blocks of a minute of the prediction shuffled, the prediction
    replaced by an IAAFT surrogate of itself, the target shifted by at least ten horizons.
    All three keep the values, so the IC is computed from the same ranks in a new order."""
    block = max(2, ceil(duration_ns("60s") / duration_ns(spec.sampling)))
    min_lag = max(block, ceil(10 * duration_ns(spec.horizon) / duration_ns(spec.sampling)))
    days = []
    for i in range(len(s.inputs)):
        r = s.rows_of(i)
        p, t = pred[r], s.target[r]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() >= 3:
            days.append((metrics._z(p[ok]), metrics._z(t[ok]), p[ok]))
    tasks = [(kind, k, spec.seed) for kind in surrogates.KINDS for k in range(spec.surrogates)]
    if workers <= 1:
        _init_workers(days, block, min_lag)
        values = [_surrogate_ic(t) for t in tasks]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # not fork: this process runs threads (polars, BLAS), and a forked copy of a
        # threaded process can deadlock
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_workers,
            initargs=(days, block, min_lag),
        ) as pool:
            values = list(pool.map(_surrogate_ic, tasks, chunksize=4))
    out = {}
    for j, kind in enumerate(surrogates.KINDS):
        v = np.array(values[j * spec.surrogates : (j + 1) * spec.surrogates])
        out[kind] = {
            "n": spec.surrogates,
            "percentile": spec.surrogate_percentile,
            "threshold": float(np.percentile(v, spec.surrogate_percentile)),
            "real": _f(real),
            "p": float((1 + np.sum(v >= real)) / (1 + spec.surrogates)),
            "mean": float(v.mean()),
        }
    return out


def _stability(s: Samples, pred: np.ndarray, lags: int, kfold: metrics.Summary) -> dict:
    """The out-of-sample IC by time of day and by the volatility of the minute before."""
    hour = (s.df["ts"].to_numpy() // 3_600_000_000_000) % 24
    rv = s.df["rv_60s"].to_numpy()

    def ic_where(mask: np.ndarray) -> float | None:
        ics = []
        for i in range(len(s.inputs)):
            r = s.rows_of(i)
            m = mask[r]
            if m.sum() >= 30:
                ics.append(metrics.day_ic(pred[r][m], s.target[r][m], lags))
        return _f(metrics.combine(ics).mean) if ics else None

    times = {
        f"{lo:02d}-{lo + 6:02d} UTC": ic_where((hour >= lo) & (hour < lo + 6))
        for lo in range(0, 24, 6)
    }
    if np.isfinite(rv).any():
        q1, q2 = np.nanpercentile(rv, [100 / 3, 200 / 3])
        vol = {
            "low": ic_where(rv <= q1),
            "middle": ic_where((rv > q1) & (rv <= q2)),
            "high": ic_where(rv > q2),
        }
    else:
        vol = {}
    return {
        "positive_days": {"share": _f(kfold.positive_share)},
        "time_of_day": times,
        "volatility_tercile": vol,
    }


def _verdict(spec, kfold, paths, surrogate, result_pbo, p_bh) -> tuple[str, list[str]]:
    reasons = []
    mean_path = float(np.mean(paths)) if paths else float("nan")
    if not mean_path > spec.min_ic:
        reasons.append(f"CPCV mean IC {mean_path:.4f} is not above {spec.min_ic}")
    if not p_bh < spec.alpha:
        reasons.append(f"p after Benjamini-Hochberg {p_bh:.3g} is not below {spec.alpha}")
    for kind, s in surrogate.items():
        if not (s["real"] is not None and s["real"] > s["threshold"]):
            reasons.append(
                f"not above the {spec.surrogate_percentile:g}th percentile of {kind} surrogates"
            )
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} is above {spec.max_pbo}")
    if not kfold.positive_share >= spec.min_positive_days:
        reasons.append(
            f"positive on {kfold.positive_share:.0%} of days, "
            f"the hypothesis asks for {spec.min_positive_days:.0%}"
        )
    return ("failed", reasons) if reasons else ("passed", [])


def _registry(results: Path) -> list[dict]:
    if not results.is_dir():
        return []
    out = []
    for f in sorted(results.glob("*.json")):
        if f.name.endswith(".holdout.json"):
            continue
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def written(experiments: Path) -> dict[str, str]:
    """Every hypothesis file in `experiments/`: its `id-name` and file name."""
    out = {}
    if experiments.is_dir():
        for f in sorted(experiments.glob("*.yaml")):
            try:
                raw = yaml.safe_load(f.read_text())
            except yaml.YAMLError:
                continue
            if isinstance(raw, dict) and "id" in raw and "name" in raw:
                out[f"{raw['id']}-{raw['name']}"] = f.name
    return out


def family(results: Path, experiments: Path | None) -> dict[str, float]:
    """The p-value of every hypothesis: those recorded in `results/`, and those written in
    `experiments/` but not run yet, at 1. Counting the unrun ones keeps a verdict from
    depending on the order the hypotheses are run in: running first never helps."""
    out = {
        f"{r['id']}-{r['name']}": (r["p_raw"] if r.get("p_raw") is not None else 1.0)
        for r in _registry(results)
    }
    if experiments is not None:
        for slug in written(experiments):
            out.setdefault(slug, 1.0)
    return out


def _bh_with_registry(results: Path, spec, p: float) -> float:
    """This hypothesis's p-value adjusted with every other one written or recorded."""
    others = [q for slug, q in family(results, spec.path.parent).items() if slug != spec.slug]
    ps = np.array([*others, p if np.isfinite(p) else 1.0])
    return float(benjamini_hochberg(ps)[-1])


def _finish(report: Report, results: Path) -> Report:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{report.spec.slug}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return report


def registry_table(results: Path, experiments: Path | None = None) -> pl.DataFrame:
    """Every recorded hypothesis with its verdict and its p-value adjusted for all of them;
    with `experiments`, the ones written but not run too, as `not run` at p = 1."""
    rows = _registry(results)
    if experiments is not None:
        seen = {f"{r['id']}-{r['name']}" for r in rows}
        for slug in written(experiments):
            if slug not in seen:
                id_, name = slug.split("-", 1)
                rows.append({"id": id_, "name": name, "verdict": "not run", "p_raw": None})
    if not rows:
        return pl.DataFrame(
            schema={
                "id": pl.String,
                "name": pl.String,
                "verdict": pl.String,
                "p_raw": pl.Float64,
                "p_bh": pl.Float64,
                "dirty": pl.Boolean,
            }
        )
    p = np.array([r.get("p_raw") if r.get("p_raw") is not None else 1.0 for r in rows])
    adj = benjamini_hochberg(p)
    return pl.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "name": [r["name"] for r in rows],
            "verdict": [r["verdict"] for r in rows],
            "p_raw": [r.get("p_raw") for r in rows],
            "p_bh": adj.tolist(),
            "dirty": [bool(r.get("dirty")) for r in rows],
        }
    )


# --- the held-out days -------------------------------------------------------------------


def final(
    spec: Spec,
    data_root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: Provenance | None = None,
) -> dict:
    """Fit the primary variant on every development day and score the held-out days, once.
    A second call is refused: the held-out days have been seen."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or provenance(spec.path)
    every = survey(data_root, spec.symbol)
    dev = [d for d in every if d.usable_hours >= spec.min_hours and d.date not in spec.holdout]
    held = [d for d in every if d.usable_hours >= spec.min_hours and d.date in spec.holdout]
    if not held:
        raise ExperimentError("none of the held-out days is in the data")
    lags = max(1, ceil(duration_ns(spec.horizon) / duration_ns(spec.sampling)))
    train = Samples([load_inputs(i, spec, cache) for i in dev])
    w = fit(train, np.arange(len(train.target)), spec.primary)
    test = Samples([load_inputs(i, spec, cache) for i in held])
    pred = predict(test, np.arange(len(test.target)), spec.primary, w)
    ics = day_ics(test, pred, lags)
    s = metrics.combine(ics)
    record = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": [i.date for i in held],
        "ic_by_day": {i.date: _f(d.ic) for i, d in zip(held, ics, strict=True)},
        "summary": {k: _f(v) for k, v in asdict(s).items()},
        "weights": [float(x) for x in w],
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
