"""Hypotheses of the `xsection` kind: a cross-sectional rule on Binance's USD-M perpetuals
on coins, a day at a time, as `marketlab.xsection` computes it, written down before its
held-out period is read.

The rule: each UTC day, the `top` perpetuals by mean quote volume over 30 days (among those
with a close on that day and 30 days before); long those strictly above the signal's
`1 - q` quantile, short those strictly below its `q` quantile, half the capital a side,
equal weights, rebalanced every `hold` days at the day's close; `round_trip_bp` on every
unit of weight traded, the funding paid or received in the P&L.

The development run scores the primary and its variants on the development days: the
daily net, in bp of the capital, tested for a mean above zero with Newey-West errors and
adjusted with Benjamini-Hochberg over every hypothesis written; PBO over the variants by
month; the share of months up; and the deflated Sharpe ratio of the primary against the
`trials`, the rules looked at on these days before the hypothesis was written (a bar only
if `min_dsr` is given). Its beta to the universe (equal weights) is reported.

The held-out period is opened once, for the primary only: it is confirmed if its mean is
above zero at `alpha` (one test, written down before), it was up in at least
`min_positive_years` of its calendar years, and its beta to the universe is within
`max_abs_beta`. Nothing before `final` reads a day after `xsection.DEV_END`.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from . import __version__
from . import experiment as ex
from . import xsection as xs
from .tradeflow import _dates
from .validation import metrics, multiple, pbo


@dataclass(frozen=True)
class Rule:
    signal: str
    hold: int  # days between rebalances
    top: int  # the universe's size
    q: float  # the share of the universe a side
    hours: int = 0  # traded this long after the daily close, from hourly bars

    @property
    def name(self) -> str:
        late = f", traded {self.hours} h after the close" if self.hours else ""
        return (f"{self.signal}, every {self.hold} days, top {self.top}, {self.q:.0%} a side"
                + late)  # fmt: skip


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    first: str
    last: str
    holdout_first: str
    holdout_last: str
    primary: Rule
    variants: tuple[Rule, ...]  # the primary first
    trials: tuple[Rule, ...]  # looked at before the hypothesis was written, the primary in
    round_trip_bp: float
    nw_lags: int
    alpha: float
    max_pbo: float
    min_positive_months: float
    min_dsr: float | None  # None: reported, not a bar
    min_positive_years: int
    max_abs_beta: float
    path: Path = field(compare=False)

    @property
    def slug(self) -> str:
        return f"{self.id}-{self.name}"


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ex.ExperimentError(f"{path}: {e}") from e
    if not isinstance(raw, dict) or raw.get("kind") != "xsection":
        raise ex.ExperimentError(f"{path}: not an `xsection` hypothesis")
    known = {"id", "name", "kind", "hypothesis", "data", "rule", "costs", "variants", "trials",
             "protocol", "success"}  # fmt: skip
    if set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw) - known)}")
    data, costs = ex._get(raw, "data", "top"), ex._get(raw, "costs", "top")
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    days = _dates(ex._get(data, "days", "data"), "data.days")
    held = _dates(ex._get(data, "holdout", "data"), "data.holdout")
    if not days or not held:
        raise ex.ExperimentError(f"{path}: data.days and data.holdout must not be empty")
    if days[-1] > xs.DEV_END or held[0] <= xs.DEV_END:
        raise ex.ExperimentError(
            f"{path}: the development days must end by {xs.DEV_END} and the held-out ones "
            f"begin after it: nothing else reads past it"
        )

    def rule(d: dict, base: Rule | None) -> Rule:
        def get(key: str, default=None):
            if key in d:
                return d[key]
            if base is not None:
                return getattr(base, key)
            if default is not None:
                return default
            return ex._get(d, key, "rule")

        r = Rule(str(get("signal")), int(get("hold")), int(get("top", 50)), float(get("q", 0.2)),
                 int(get("hours", 0)))  # fmt: skip
        if r.signal not in xs.SIGNALS:
            raise ex.ExperimentError(f"{path}: signal {r.signal!r} is none of {xs.SIGNALS}")
        if r.hold < 1 or r.top < 10 or not 0 < r.q < 0.5 or not 0 <= r.hours < 24:
            raise ex.ExperimentError(
                f"{path}: {r.name}: hold at least 1 day, top at least 10, q between 0 and 0.5, "
                f"hours from 0 to 23"
            )
        return r

    primary = rule(ex._get(raw, "rule", "top"), None)
    variants = list(
        dict.fromkeys([primary, *(rule(v, primary) for v in raw.get("variants") or [])])
    )
    trials = list(dict.fromkeys([primary, *(rule(v, primary) for v in raw.get("trials") or [])]))
    spec = Spec(
        id=str(ex._get(raw, "id", "top")),
        name=str(ex._get(raw, "name", "top")),
        hypothesis=" ".join(str(ex._get(raw, "hypothesis", "top")).split()),
        first=days[0],
        last=days[-1],
        holdout_first=held[0],
        holdout_last=held[-1],
        primary=primary,
        variants=tuple(variants),
        trials=tuple(trials),
        round_trip_bp=float(ex._get(costs, "round_trip_bp", "costs")),
        nw_lags=int(proto.get("nw_lags", 10)),
        alpha=float(ok.get("alpha", 0.05)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_months=float(ok.get("min_positive_months", 0.5)),
        min_dsr=float(ok["min_dsr"]) if ok.get("min_dsr") is not None else None,
        min_positive_years=int(ok.get("min_positive_years", 1)),
        max_abs_beta=float(ok.get("max_abs_beta", 1.0)),
        path=path,
    )
    if spec.round_trip_bp < 0:
        raise ex.ExperimentError(f"{path}: costs must not be negative")
    return spec


# --- a rule's days ---------------------------------------------------------------------------


@dataclass
class Days:
    """A rule over a period: each day's net and the market's, as fractions of the capital."""

    dates: list[str]
    net: np.ndarray
    gross: np.ndarray
    market: np.ndarray
    names: float  # positions a side, on average
    priced: float  # the share of the positions with a return: a bar at both ends of the day


#: the least share of the held-out universe's coin-days with an hour bar at both ends for
#: a rule traded hours after the close; below it, nothing is computed
MIN_PRICED = 0.98


def follow(p: xs.Panel, rule: Rule, first: str, cost_bp: float, cache: dict,
           root: str | Path | None = None, held_out: bool = False) -> Days:  # fmt: skip
    """The rule from `first` to the panel's last day; `cache` keeps what rules share. A rule
    traded hours after the close reads the hour bars under `root`."""
    if rule.top not in cache:
        cache[rule.top] = xs.universe(p, rule.top)
    if "x" not in cache:
        cache["x"] = xs.signals(p)
    key = ("r", rule.hours)
    if key not in cache:
        if rule.hours == 0:
            cache[key] = xs.daily_returns(p)
        else:
            cache[key] = xs.lagged(p, root, [rule.hours], held_out)[rule.hours].r
    inu, r, x = cache[rule.top], cache[key], cache["x"][rule.signal]
    start = p.dates.index(first)
    pnl, _, names = xs.portfolio(x, inu, r, start, rule.hold, rule.q, cost_bp)
    gross, _, _ = xs.portfolio(x, inu, r, start, rule.hold, rule.q, 0.0)
    held, _, _, _ = xs.holdings(x, inu, start, rule.hold, rule.q)
    on = held[start:] != 0
    mkt = xs.market(inu, r, start)
    return Days(p.dates[start:], pnl[start:], gross[start:], mkt[start:],
                float(np.mean(names)) if names else float("nan"),
                float(np.isfinite(r[start:])[on].mean()) if on.any() else float("nan"))  # fmt: skip


def score(d: Days, lags: int) -> dict:
    """The daily net in bp: its mean tested above zero (Newey-West), a year, Sharpe from
    months, drawdown, months and years up, beta to the market."""
    x = d.net * 1e4
    n = len(x)
    se = metrics.newey_west_se(x, lags) if n > 2 else float("nan")
    mean = float(x.mean()) if n else float("nan")
    month = np.array([s[:7] for s in d.dates])
    months = np.array([x[month == m].sum() for m in dict.fromkeys(month)])
    year = np.array([s[:4] for s in d.dates])
    years = {y: float(x[year == y].sum() / 100) for y in dict.fromkeys(year)}
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return {
        "days": n,
        "mean_bp": ex._f(mean),
        "se": ex._f(se),
        "p": ex._f(stats.norm.sf(mean / se)) if se > 0 else None,
        "annual_pct": ex._f(mean * 365 / 100),
        "gross_pct": ex._f(d.gross.mean() * 365 * 100) if n else None,
        "sharpe_months": ex._f(xs._sharpe(months)),
        "max_drawdown_pct": ex._f((peak - cum).max() / 100) if n else None,
        "positive_months": ex._f((months > 0).mean()) if len(months) else None,
        "beta": ex._f(xs.slope(d.gross[1:], d.market[1:])),
        "names_side": ex._f(d.names),
        "priced": ex._f(d.priced),
        "by_year_pct": years,
    }


def deflated(d: Days, trials: list[Days]) -> float:
    """The primary's deflated Sharpe ratio, daily, against the trials' daily Sharpe ratios."""
    sr = [t.net.mean() / t.net.std(ddof=1) for t in trials if t.net.std(ddof=1) > 0]
    x = d.net
    if len(x) < 3 or not x.std(ddof=1) > 0 or len(sr) < 1:
        return float("nan")
    return multiple.deflated_sharpe(
        float(x.mean() / x.std(ddof=1)), len(x), float(stats.skew(x)),
        float(stats.kurtosis(x, fisher=False)), float(np.var(sr, ddof=1)) if len(sr) > 1 else 0.0,
        len(sr),
    )  # fmt: skip


# --- the development run -----------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    perpetuals: int
    table: list[dict]  # every variant, the primary first
    trials: list[dict]  # rule and Sharpe from months
    dsr: float
    by_month: list[dict]  # the primary's
    pbo: pbo.PBO | None
    p_bh: float
    verdict: str
    reasons: list[str]

    def to_json(self) -> dict:
        s = self.spec
        primary = self.table[0] if self.table else {}
        return {
            "id": s.id,
            "name": s.name,
            "kind": "xsection",
            "hypothesis": s.hypothesis,
            "spec": s.path.name,
            "spec_commit": self.provenance.spec_commit,
            "head": self.provenance.head,
            "dirty": self.provenance.dirty,
            "marketlab": __version__,
            "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data": {
                "days": [s.first, s.last],
                "holdout": [s.holdout_first, s.holdout_last],
                "perpetuals": self.perpetuals,
            },
            "costs": {"round_trip_bp": s.round_trip_bp},
            "primary": s.primary.name,
            "table": self.table,
            "trials": self.trials,
            "dsr": ex._f(self.dsr),
            "by_month": self.by_month,
            "pbo": self.pbo.__dict__ if self.pbo else None,
            "p_raw": primary.get("p"),
            "p_bh": ex._f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _finish(report: Report, results: Path) -> Report:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{report.spec.slug}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return report


def run(spec: Spec, root: str | Path, results: Path, prov: ex.Provenance | None = None) -> Report:
    prov = prov or ex.provenance(spec.path)
    p = xs.load(root, spec.first, spec.last)
    if not p.symbols:
        why = f"no daily bars under {root}/um/klines1d"
        report = Report(spec, prov, 0, [], [], float("nan"), [], None, float("nan"),
                        "insufficient data", [why])  # fmt: skip
        return _finish(report, results)
    cache: dict = {}
    runs = {r: follow(p, r, spec.first, spec.round_trip_bp, cache, root) for r in
            dict.fromkeys([*spec.variants, *spec.trials])}  # fmt: skip
    if not any(np.any(d.net != 0) for d in runs.values()):
        report = Report(spec, prov, len(p.symbols), [], [], float("nan"), [], None, float("nan"),
                        "insufficient data", ["no position on any day"])  # fmt: skip
        return _finish(report, results)
    table = [{"rule": r.name, "signal": r.signal, "hold": r.hold, "top": r.top, "q": r.q,
              **score(runs[r], spec.nw_lags)} for r in spec.variants]  # fmt: skip
    trials = [{"rule": r.name, "sharpe_months": score(runs[r], spec.nw_lags)["sharpe_months"]}
              for r in spec.trials]  # fmt: skip
    dsr = deflated(runs[spec.primary], [runs[r] for r in spec.trials])
    d = runs[spec.primary]
    month = np.array([s[:7] for s in d.dates])
    by_month = [{"month": m, "net_bp": float(d.net[month == m].sum() * 1e4)}
                for m in dict.fromkeys(month)]  # fmt: skip
    result_pbo = None
    if len(spec.variants) > 1:
        perf = np.stack([[runs[r].net[month == m].mean() for m in dict.fromkeys(month)]
                         for r in spec.variants], axis=1)  # fmt: skip
        result_pbo = pbo.pbo(perf, groups=16)
    primary = table[0]
    p_bh = ex._bh_with_registry(results, spec, primary["p"] if primary["p"] is not None else 1.0)
    reasons = []
    if not (primary["mean_bp"] or 0) > 0:
        reasons.append(f"mean {primary['mean_bp']} bp a day is not above zero")
    if not p_bh < spec.alpha:
        reasons.append(f"p after Benjamini-Hochberg {p_bh:.3g} is not below {spec.alpha}")
    if result_pbo is not None and not result_pbo.pbo <= spec.max_pbo:
        reasons.append(f"PBO {result_pbo.pbo:.2f} over the variants is above {spec.max_pbo}")
    if not (primary["positive_months"] or 0) >= spec.min_positive_months:
        reasons.append(f"up in {primary['positive_months']:.0%} of months, the hypothesis asks "
                       f"for {spec.min_positive_months:.0%}")  # fmt: skip
    if spec.min_dsr is not None and not dsr >= spec.min_dsr:
        reasons.append(f"deflated Sharpe ratio {dsr:.2f} against {len(spec.trials)} trials is "
                       f"below {spec.min_dsr}")  # fmt: skip
    report = Report(spec, prov, len(p.symbols), table, trials, dsr, by_month, result_pbo, p_bh,
                    "failed" if reasons else "passed", reasons)  # fmt: skip
    return _finish(report, results)


# --- the held-out period -------------------------------------------------------------------


def final(spec: Spec, root: str | Path, results: Path, prov: ex.Provenance | None = None) -> dict:
    """The primary rule over the held-out period, once. A second call is refused."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or ex.provenance(spec.path)
    p = xs.load(root, spec.holdout_first, spec.holdout_last, held_out=True)
    if not p.symbols or not np.isfinite(p.close[p.dates.index(spec.holdout_first) :]).any():
        raise ex.ExperimentError("no daily bars in the held-out period")
    cache: dict = {}
    if spec.primary.hours:  # the hour bars there first, or nothing is computed
        h, start = spec.primary.hours, p.dates.index(spec.holdout_first)
        inu = cache[spec.primary.top] = xs.universe(p, spec.primary.top)
        lag = xs.lagged(p, root, [h], held_out=True)[h]
        cache[("r", h)] = lag.r
        need = inu[start:-1]
        got = lag.priced[start:-1] & lag.priced[start + 1 :]
        share = float(got[need].mean()) if need.any() else 0.0
        if share < MIN_PRICED:
            short = (need & ~got).sum(axis=0)
            names = [p.symbols[j] for j in np.argsort(-short, kind="stable") if short[j]]
            raise ex.ExperimentError(
                f"hour bars at both ends for {share:.1%} of the held-out universe's coin-days "
                f"(at least {MIN_PRICED:.0%} asked): fetch them with scripts/fetch_hourly.py "
                f"for {', '.join(names)}; nothing was computed or recorded"
            )
    d = follow(p, spec.primary, spec.holdout_first, spec.round_trip_bp, cache, root, True)
    s = score(d, spec.nw_lags)
    up = sum(v > 0 for v in s["by_year_pct"].values())
    beta = s["beta"] if s["beta"] is not None else float("nan")
    reasons = []
    if not ((s["mean_bp"] or 0) > 0 and s["p"] is not None and s["p"] < spec.alpha):
        reasons.append(f"mean {s['mean_bp']} bp a day, p {s['p']}: not above zero at {spec.alpha}")
    if up < spec.min_positive_years:
        reasons.append(f"up in {up} of its {len(s['by_year_pct'])} years, the hypothesis asks "
                       f"for {spec.min_positive_years}")  # fmt: skip
    if not abs(beta) <= spec.max_abs_beta:
        reasons.append(f"beta to the universe {beta:.2f} is beyond {spec.max_abs_beta}")
    month = np.array([x[:7] for x in d.dates])
    record = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "period": [spec.holdout_first, spec.holdout_last],
        "rule": spec.primary.name,
        "perpetuals": len(p.symbols),
        "summary": s,
        "confirmed": not reasons,
        "reasons": reasons,
        "by_month": [
            {"month": m, "net_bp": float(d.net[month == m].sum() * 1e4)}
            for m in dict.fromkeys(month)
        ],
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
