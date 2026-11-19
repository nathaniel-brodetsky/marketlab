"""Hypotheses of the `tradeflow` kind: does the flow of taker orders in Binance's public
trade archives (`marketlab.archive`) forecast the trade price?

The feature is signed taker volume, buys minus sells, over a window that is either a
number of aggregate trades (`100t`: the last hundred taker orders' fills at a price) or a
time (`5s`). The target is the log return of the last trade price over the next `horizon`.
Both are taken on a grid every `sampling`, inside each day, between its first and last
trade. The feature reads only trades at or before its time; the target only the price at
its time and at the end of its horizon.

The feature has no fitted parameters and its sign is fixed by the hypothesis (more buying,
higher price), so every development day is out of sample for it; what selection there is,
is the choice between variants, and that is what the probability of backtest overfitting
measures. The protocol is otherwise that of `experiment`:

1. The specification, committed before the run; days for development and held-out days
   named in it, apart; the held-out days opened once, with `--final`.
2. The rank IC by day and its mean, with a Newey-West standard error inside each day; the
   one-sided p of the mean.
3. Surrogates: the same IC with the feature and the target misaligned three ways
   (`experiment._surrogates`), 200 times or more each; the real one above the percentile.
4. PBO across the variants; Benjamini-Hochberg over every hypothesis written.
5. Stability by time of day and by the size of the last minute's move.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import __version__, archive
from . import experiment as ex
from .features import duration_ns
from .validation import metrics, pbo

CACHE_VERSION = 1
NS_DAY = 86_400 * 1_000_000_000
_TRADES = re.compile(r"^(\d+)t$")


@dataclass(frozen=True)
class Window:
    text: str

    @property
    def trades(self) -> int | None:
        m = _TRADES.match(self.text)
        return int(m.group(1)) if m else None

    @property
    def ns(self) -> int | None:
        return None if self.trades is not None else duration_ns(self.text)

    @property
    def column(self) -> str:
        return f"flow_{self.text}"


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    root_kind: str  # aggTrades or trades
    market: str
    symbol: str
    first: str
    last: str
    holdout: tuple[str, ...]
    min_days: int
    sampling: str
    primary: Window
    variants: tuple[Window, ...]
    horizon: str
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

    @property
    def development(self) -> tuple[str, ...]:
        a, b = dt.date.fromisoformat(self.first), dt.date.fromisoformat(self.last)
        days = (a + dt.timedelta(days=k) for k in range((b - a).days + 1))
        return tuple(d.isoformat() for d in days if d.isoformat() not in self.holdout)


def _dates(raw, where: str) -> tuple[str, ...]:
    """A list of dates, or {from, to}: every date between, both included."""
    if isinstance(raw, dict):
        a = dt.date.fromisoformat(str(ex._get(raw, "from", where)))
        b = dt.date.fromisoformat(str(ex._get(raw, "to", where)))
        if b < a:
            raise ex.ExperimentError(f"{where}: `to` is before `from`")
        return tuple((a + dt.timedelta(days=k)).isoformat() for k in range((b - a).days + 1))
    return tuple(str(dt.date.fromisoformat(str(d))) for d in raw or [])


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ex.ExperimentError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise ex.ExperimentError(f"{path}: not a mapping")
    if raw.get("kind") != "tradeflow":
        raise ex.ExperimentError(f"{path}: not a `tradeflow` hypothesis")
    known = {"id", "name", "kind", "hypothesis", "data", "sampling", "feature", "variants",
             "target", "protocol", "success"}  # fmt: skip
    unknown = set(raw) - known
    if unknown:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(unknown)}")
    data, feat, target = raw.get("data", {}), raw.get("feature", {}), raw.get("target", {})
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    if ex._get(feat, "method", "feature") != "signed_volume":
        raise ex.ExperimentError(f"{path}: the feature is `signed_volume`")
    if ex._get(target, "kind", "target") != "return":
        raise ex.ExperimentError(f"{path}: the target is `return`")
    days = _dates(ex._get(data, "days", "data"), "data.days")
    holdout = _dates(ex._get(data, "holdout", "data"), "data.holdout")
    if set(days) & set(holdout):
        raise ex.ExperimentError(f"{path}: held-out days inside the development days")
    primary = Window(str(ex._get(ex._get(feat, "params", "feature"), "window", "feature")))
    variants = tuple(Window(str(ex._get(v, "window", "variant"))) for v in raw.get("variants", []))
    variants = variants or (primary,)
    if primary not in variants:
        variants = (primary, *variants)
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
        primary=primary,
        variants=variants,
        horizon=str(ex._get(target, "horizon", "target")),
        surrogates=int(proto.get("surrogates", 200)),
        seed=int(proto.get("seed", 1)),
        min_ic=float(ok.get("min_ic", 0.0)),
        alpha=float(ok.get("alpha", 0.05)),
        surrogate_percentile=float(ok.get("surrogate_percentile", 95)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_days=float(ok.get("min_positive_days", 0.6)),
        path=path,
    )
    for d in (spec.sampling, spec.horizon):
        duration_ns(d)
    for w in variants:
        if w.trades is None:
            duration_ns(w.text)
        elif w.trades < 1:
            raise ex.ExperimentError(f"{path}: window {w.text} holds no trades")
    if spec.surrogates < 200:
        raise ex.ExperimentError(
            f"{path}: the protocol asks for at least 200 surrogates of each kind"
        )
    if spec.root_kind not in archive.KINDS:
        raise ex.ExperimentError(f"{path}: data.kind is one of {', '.join(archive.KINDS)}")
    return spec


# --- one day's inputs --------------------------------------------------------------------


def day_inputs(trades: pl.DataFrame, date: str, spec: Spec) -> pl.DataFrame:
    """On the day's grid: each variant's signed volume, the target, and the size of the
    last minute's move (for the stability report)."""
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * 10**9
    ts = trades["ts"].to_numpy()
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    price = trades["price"].to_numpy()[order]
    signed = np.where(trades["is_buyer_maker"].to_numpy()[order], -1.0, 1.0)
    signed = signed * trades["qty"].to_numpy()[order]
    every = duration_ns(spec.sampling)
    horizon = duration_ns(spec.horizon)
    nan = np.full(0, np.nan)
    if len(ts) == 0:
        cols = {"ts": np.zeros(0, np.int64), "target": nan, "move_60s": nan}
        cols |= {w.column: nan for w in spec.variants}
        return pl.DataFrame(cols)
    lo, hi = max(int(ts[0]), start), min(int(ts[-1]), start + NS_DAY - 1)
    grid = np.arange(-(-lo // every) * every, hi + 1, every, dtype=np.int64)
    upto = np.searchsorted(ts, grid, side="right")  # trades at or before each time
    cum = np.concatenate([[0.0], np.cumsum(signed)])
    cols: dict[str, np.ndarray] = {"ts": grid}
    for w in spec.variants:
        if w.trades is not None:
            first = upto - w.trades
            x = np.where(first >= 0, cum[upto] - cum[np.clip(first, 0, None)], np.nan)
        else:
            since = np.searchsorted(ts, grid - w.ns, side="right")
            x = np.where(grid - w.ns >= lo, cum[upto] - cum[since], np.nan)
        cols[w.column] = x
    last = np.log(price)
    now = np.where(upto > 0, last[np.clip(upto - 1, 0, None)], np.nan)
    later_upto = np.searchsorted(ts, grid + horizon, side="right")
    later = last[np.clip(later_upto - 1, 0, None)]
    cols["target"] = np.where(grid + horizon <= hi, later - now, np.nan)
    before = np.searchsorted(ts, grid - 60 * 10**9, side="right")
    then = np.where(before > 0, last[np.clip(before - 1, 0, None)], np.nan)
    cols["move_60s"] = np.where(grid - 60 * 10**9 >= lo, np.abs(now - then), np.nan)
    return pl.DataFrame(cols)


def _cache_file(cache: Path, day: archive.ArchiveDay, spec: Spec) -> Path:
    st = day.path.stat()
    parts = [
        str(CACHE_VERSION), __version__, day.path.name, str(st.st_size), str(st.st_mtime_ns),
        spec.sampling, spec.horizon, ",".join(sorted(w.text for w in spec.variants)),
    ]  # fmt: skip
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache / spec.symbol / day.date / f"tradeflow-{key}.parquet"


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


class Samples:
    """Every day's grid rows end to end, in the shape `experiment._surrogates` reads."""

    def __init__(self, inputs: Sequence[pl.DataFrame]):
        self.inputs = list(inputs)
        self.df = pl.concat(self.inputs) if self.inputs else pl.DataFrame()
        self.day = np.concatenate([np.full(len(d), i) for i, d in enumerate(self.inputs)])
        self.target = self.df["target"].to_numpy() if len(self.df) else np.zeros(0)

    def x(self, w: Window) -> np.ndarray:
        return self.df[w.column].to_numpy()

    def rows_of(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.day == day)


def day_ics(s: Samples, pred: np.ndarray, lags: int) -> list[metrics.DayIC]:
    return [
        metrics.day_ic(pred[s.rows_of(i)], s.target[s.rows_of(i)], lags)
        for i in range(len(s.inputs))
    ]


# --- the run ----------------------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    days: list[str]
    missing: list[str]
    lags: int
    summary: metrics.Summary
    by_day: dict[str, float | None]
    surrogate: dict
    pbo: pbo.PBO | None
    variants: dict[str, float | None]
    stability: dict
    p_bh: float
    verdict: str
    reasons: list[str]

    def to_json(self) -> dict:
        s = self.spec
        return {
            "id": s.id,
            "name": s.name,
            "kind": "tradeflow",
            "hypothesis": s.hypothesis,
            "spec": s.path.name,
            "spec_commit": self.provenance.spec_commit,
            "head": self.provenance.head,
            "dirty": self.provenance.dirty,
            "marketlab": __version__,
            "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data": {
                "source": f"binance {s.market} {s.root_kind}",
                "symbol": s.symbol,
                "days": self.days,
                "missing": self.missing,
                "holdout": list(s.holdout),
            },
            "feature": {"method": "signed_volume", "primary": s.primary.text},
            "target": {"kind": "return", "horizon": s.horizon},
            "sampling": s.sampling,
            "newey_west_lags": self.lags,
            "summary": {k: ex._f(v) for k, v in self.summary.__dict__.items()},
            "ic_by_day": self.by_day,
            "surrogates": self.surrogate,
            "pbo": self.pbo.__dict__ if self.pbo else None,
            "variants": self.variants,
            "stability": self.stability,
            "p_raw": ex._f(self.summary.p),
            "p_bh": ex._f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _days(
    root: str | Path, spec: Spec, dates: Sequence[str]
) -> tuple[list[archive.ArchiveDay], list[str]]:
    found = {d.date: d for d in archive.find(root, spec.symbol, spec.root_kind, spec.market)}
    return [found[d] for d in dates if d in found], [d for d in dates if d not in found]


def _stability(s: Samples, pred: np.ndarray, lags: int) -> dict:
    hour = (s.df["ts"].to_numpy() // 3_600_000_000_000) % 24
    move = s.df["move_60s"].to_numpy()

    def ic_where(mask: np.ndarray) -> float | None:
        ics = []
        for i in range(len(s.inputs)):
            r = s.rows_of(i)
            if mask[r].sum() >= 30:
                ics.append(metrics.day_ic(pred[r][mask[r]], s.target[r][mask[r]], lags))
        return ex._f(metrics.combine(ics).mean) if ics else None

    times = {
        f"{a:02d}-{a + 6:02d} UTC": ic_where((hour >= a) & (hour < a + 6)) for a in range(0, 24, 6)
    }
    out: dict = {"time_of_day": times}
    if np.isfinite(move).any():
        q1, q2 = np.nanpercentile(move, [100 / 3, 200 / 3])
        out["last_minute_move"] = {
            "small": ic_where(move <= q1),
            "middle": ic_where((move > q1) & (move <= q2)),
            "large": ic_where(move > q2),
        }
    return out


def run(
    spec: Spec,
    root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
    workers: int = 1,
) -> Report:
    prov = prov or ex.provenance(spec.path)
    days, missing = _days(root, spec, spec.development)
    assert not {d.date for d in days} & set(spec.holdout), "a held-out day slipped in"
    lags = max(1, ceil(duration_ns(spec.horizon) / duration_ns(spec.sampling)))
    nan = float("nan")
    if len(days) < spec.min_days:
        empty = metrics.Summary(nan, nan, nan, nan, nan, 0)
        reason = (
            f"{len(days)} development days in the data, the hypothesis asks for {spec.min_days}"
        )
        report = Report(spec, prov, [d.date for d in days], missing, lags, empty, {}, {}, None,
                        {}, {}, nan, "insufficient data", [reason])  # fmt: skip
        return _finish(report, results)
    s = Samples([load_inputs(d, spec, cache) for d in days])
    pred = s.x(spec.primary)
    ics = day_ics(s, pred, lags)
    summary = metrics.combine(ics)
    by_day = {d.date: ex._f(x.ic) for d, x in zip(days, ics, strict=True)}
    surrogate = ex._surrogates(s, pred, summary.mean, spec, workers)
    perf = np.full((len(days), len(spec.variants)), np.nan)
    variant_ic: dict[str, float | None] = {}
    for j, w in enumerate(spec.variants):
        vi = ics if w == spec.primary else day_ics(s, s.x(w), lags)
        perf[:, j] = [x.ic for x in vi]
        variant_ic[w.text] = ex._f(metrics.combine(vi).mean)
    result_pbo = pbo.pbo(np.nan_to_num(perf), groups=16) if len(spec.variants) > 1 else None
    stability = _stability(s, pred, lags)
    p_bh = ex._bh_with_registry(results, spec, summary.p)
    reasons = []
    if not summary.mean > spec.min_ic:
        reasons.append(f"mean IC {summary.mean:.4f} is not above {spec.min_ic}")
    if not p_bh < spec.alpha:
        reasons.append(f"p after Benjamini-Hochberg {p_bh:.3g} is not below {spec.alpha}")
    for kind, x in surrogate.items():
        if not (x["real"] is not None and x["real"] > x["threshold"]):
            reasons.append(
                f"not above the {spec.surrogate_percentile:g}th percentile of {kind} surrogates"
            )
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} is above {spec.max_pbo}")
    if not summary.positive_share >= spec.min_positive_days:
        reasons.append(
            f"positive on {summary.positive_share:.0%} of days, "
            f"the hypothesis asks for {spec.min_positive_days:.0%}"
        )
    verdict = "failed" if reasons else "passed"
    report = Report(spec, prov, [d.date for d in days], missing, lags, summary, by_day, surrogate,
                    result_pbo, variant_ic, stability, p_bh, verdict, reasons)  # fmt: skip
    return _finish(report, results)


def _finish(report: Report, results: Path) -> Report:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{report.spec.slug}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return report


def final(
    spec: Spec,
    root: str | Path,
    results: Path,
    cache: Path | None = None,
    prov: ex.Provenance | None = None,
) -> dict:
    """The primary variant's IC on the held-out days, once. A second call is refused."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or ex.provenance(spec.path)
    days, missing = _days(root, spec, spec.holdout)
    if not days:
        raise ex.ExperimentError("none of the held-out days is in the data")
    lags = max(1, ceil(duration_ns(spec.horizon) / duration_ns(spec.sampling)))
    s = Samples([load_inputs(d, spec, cache) for d in days])
    ics = day_ics(s, s.x(spec.primary), lags)
    summary = metrics.combine(ics)
    record = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": [d.date for d in days],
        "missing": missing,
        "ic_by_day": {d.date: ex._f(x.ic) for d, x in zip(days, ics, strict=True)},
        "summary": {k: ex._f(v) for k, v in summary.__dict__.items()},
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
