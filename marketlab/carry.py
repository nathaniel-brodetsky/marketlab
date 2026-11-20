"""Hypotheses of the `carry` kind: holding spot against a short perpetual for its funding,
after what it costs to get in and out.

The position: long `q` of the coin on spot and short `q` of the USD-M perpetual, opened
and closed together. While it is held, the short side receives each settlement's funding
(`q` x mark price x rate; it pays when the rate is negative), and the two legs together
gain or lose the change of the basis, perpetual less spot. Each opening and each closing
pays a taker fee on each leg. Everything is in basis points of the notional at the
opening, `q` x spot price then.

The rule, at each funding settlement and only then: hold the position over the next
period if the mean rate of the last `lookback` settlements, this one included, is above
`threshold_bp`; open or close it, right after the settlement, at the open of the price bar
that starts then (or, if Binance has no such bar, the close of the last bar before it).
A decision reads the rates settled by its time and the prices at it, nothing later. A
period starts flat and ends flat: what is held at its last settlement is closed then.

Daily P&L (UTC days, the funding and the basis on the day of the settlement that ends a
period, fees on the day they are paid; nothing on days without a position) is tested for
a mean above zero with Newey-West errors; PBO over the variants by month; the held-out
period is opened once, for the primary rule only.

Data, from `scripts/fetch_funding.py` and `scripts/fetch_binance.py --kind klines
--monthly`: ROOT/um/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate.csv and
ROOT/{spot,um}/klines/<SYMBOL>/<interval>/<SYMBOL>-<interval>-YYYY-MM.zip.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from . import __version__
from . import experiment as ex
from .features import duration_ns
from .tradeflow import _dates
from .validation import metrics, pbo

NS_DAY = 86_400 * 1_000_000_000
MS = 1_000_000
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
                 "ignore"]  # fmt: skip
FALLBACK = 6 * 3_600 * 1_000_000_000  # how far back a missing bar may take its price from


@dataclass(frozen=True)
class Rule:
    lookback: int  # settlements in the mean
    threshold_bp: float  # per settlement

    @property
    def name(self) -> str:
        return f"mean of {self.lookback} above {self.threshold_bp:g} bp"


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    hypothesis: str
    symbol: str
    interval: str
    first: str
    last: str
    holdout_first: str
    holdout_last: str
    primary: Rule
    variants: tuple[Rule, ...]
    spot_bp: float
    perp_bp: float
    nw_lags: int
    alpha: float
    max_pbo: float
    min_positive_months: float
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
    if not isinstance(raw, dict) or raw.get("kind") != "carry":
        raise ex.ExperimentError(f"{path}: not a `carry` hypothesis")
    known = {"id", "name", "kind", "hypothesis", "data", "rule", "costs", "variants",
             "protocol", "success"}  # fmt: skip
    if set(raw) - known:
        raise ex.ExperimentError(f"{path}: unknown keys {sorted(set(raw) - known)}")
    data, costs = raw.get("data", {}), ex._get(raw, "costs", "top")
    proto, ok = raw.get("protocol", {}), raw.get("success", {})
    days = _dates(ex._get(data, "days", "data"), "data.days")
    held = _dates(ex._get(data, "holdout", "data"), "data.holdout")
    if not days or not held:
        raise ex.ExperimentError(f"{path}: data.days and data.holdout must not be empty")
    if not (days[-1] < held[0] or held[-1] < days[0]):
        raise ex.ExperimentError(f"{path}: the held-out period overlaps the development days")

    def rule(d: dict, base: Rule | None) -> Rule:
        r = Rule(
            int(d.get("lookback", base.lookback if base else ex._get(d, "lookback", "rule"))),
            float(d.get("threshold_bp", base.threshold_bp if base else 0.0)),
        )
        if r.lookback < 1:
            raise ex.ExperimentError(f"{path}: lookback must be at least one settlement")
        return r

    primary = rule(ex._get(raw, "rule", "top"), None)
    variants = [primary] + [rule(v, primary) for v in raw.get("variants") or []]
    variants = list(dict.fromkeys(variants))
    spec = Spec(
        id=str(ex._get(raw, "id", "top")),
        name=str(ex._get(raw, "name", "top")),
        hypothesis=" ".join(str(ex._get(raw, "hypothesis", "top")).split()),
        symbol=str(ex._get(data, "symbol", "data")),
        interval=str(data.get("interval", "1h")),
        first=days[0],
        last=days[-1],
        holdout_first=held[0],
        holdout_last=held[-1],
        primary=primary,
        variants=tuple(variants),
        spot_bp=float(ex._get(costs, "spot_bp", "costs")),
        perp_bp=float(ex._get(costs, "perp_bp", "costs")),
        nw_lags=int(proto.get("nw_lags", 10)),
        alpha=float(ok.get("alpha", 0.05)),
        max_pbo=float(ok.get("max_pbo", 0.5)),
        min_positive_months=float(ok.get("min_positive_months", 0.5)),
        path=path,
    )
    duration_ns(spec.interval)
    if spec.spot_bp < 0 or spec.perp_bp < 0:
        raise ex.ExperimentError(f"{path}: costs must not be negative")
    return spec


# --- data --------------------------------------------------------------------------------


def _ns(t: np.ndarray) -> np.ndarray:
    """Binance's times in ms (13 digits) or, on spot from 2025, in µs (16 digits), as ns."""
    t = t.astype(np.int64)
    return np.where(t > 10**14, t * 1_000, t * MS)


def read_klines(root: str | Path, market: str, symbol: str, interval: str) -> pl.DataFrame:
    """Every monthly file there is: open_time (ns), open and close, by time, once each."""
    folder = Path(root) / market / "klines" / symbol / interval
    parts = []
    for f in sorted(folder.glob(f"{symbol}-{interval}-*.zip")):
        with zipfile.ZipFile(f) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ex.ExperimentError(f"{f}: expected one CSV inside, found {names}")
            data = z.read(names[0])
        header = bool(data) and not data[:1].isdigit()
        df = pl.read_csv(io.BytesIO(data), has_header=header, new_columns=KLINE_COLUMNS)
        parts.append(
            df.select(
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
            )
        )
    if not parts:
        return pl.DataFrame(schema={"open_time": pl.Int64, "open": pl.Float64, "close": pl.Float64})
    df = pl.concat(parts)
    return (
        df.with_columns(pl.Series("open_time", _ns(df["open_time"].to_numpy())))
        .unique("open_time", keep="first")
        .sort("open_time")
    )


def read_funding(root: str | Path, symbol: str) -> pl.DataFrame:
    """time (ns), rate (a fraction), mark (NaN where Binance gave none), by time."""
    f = Path(root) / "um" / "fundingRate" / symbol / f"{symbol}-fundingRate.csv"
    if not f.exists():
        raise ex.ExperimentError(f"no funding history at {f}: scripts/fetch_funding.py writes it")
    df = pl.read_csv(f, schema_overrides={"funding_time": pl.Int64, "funding_rate": pl.Float64,
                                          "mark_price": pl.Float64})  # fmt: skip
    return pl.DataFrame(
        {
            "time": _ns(df["funding_time"].to_numpy()),
            "rate": df["funding_rate"].to_numpy(),
            "mark": df["mark_price"].fill_null(np.nan).to_numpy(),
        }
    ).sort("time")


def price_at(bars: pl.DataFrame, t: np.ndarray, interval: str) -> tuple[np.ndarray, int]:
    """The open of the bar starting at each time (rounded down to the interval); where there
    is none, the close of the last bar before, if it is less than six hours old. And how many
    times needed that."""
    step = duration_ns(interval)
    start = t // step * step
    ot = bars["open_time"].to_numpy()
    out = np.full(len(t), np.nan)
    if not len(ot):
        return out, 0
    i = np.searchsorted(ot, start)  # the first bar starting at or after
    at = np.minimum(i, len(ot) - 1)
    exact = ot[at] == start
    out[exact] = bars["open"].to_numpy()[at[exact]]
    j = i - 1  # the last bar starting before
    ok = ~exact & (j >= 0)
    ok[ok] &= start[ok] - ot[j[ok]] <= FALLBACK
    out[ok] = bars["close"].to_numpy()[j[ok]]
    return out, int(ok.sum())


@dataclass
class Market:
    """Every settlement with the rate, and the spot and perpetual prices right after it."""

    time: np.ndarray  # ns
    rate: np.ndarray
    spot: np.ndarray
    perp: np.ndarray
    mark: np.ndarray  # for the funding payment: Binance's mark price, else the perpetual's
    filled: int = 0  # prices taken from the bar before

    @classmethod
    def load(cls, root: str | Path, spec: Spec) -> Market:
        f = read_funding(root, spec.symbol)
        t = f["time"].to_numpy()
        spot, a = price_at(read_klines(root, "spot", spec.symbol, spec.interval), t, spec.interval)
        perp, b = price_at(read_klines(root, "um", spec.symbol, spec.interval), t, spec.interval)
        mark = f["mark"].to_numpy()
        mark = np.where(np.isfinite(mark) & (mark > 0), mark, perp)
        return cls(t, f["rate"].to_numpy(), spot, perp, mark, a + b)


# --- the rule ----------------------------------------------------------------------------


def trailing_mean(rate: np.ndarray, lookback: int) -> np.ndarray:
    """The mean of the last `lookback` rates, this one included; NaN before there are that
    many."""
    c = np.concatenate([[0.0], np.cumsum(rate)])
    out = np.full(len(rate), np.nan)
    if len(rate) >= lookback:
        out[lookback - 1 :] = (c[lookback:] - c[:-lookback]) / lookback
    return out


@dataclass
class Path_:
    """One rule over one period, by settlement: what each brought, in fractions of the
    notional at the opening."""

    rows: np.ndarray  # the settlements of the period, indices into the market
    funding: np.ndarray
    basis: np.ndarray
    fees: np.ndarray
    held: np.ndarray  # the position over the period after each settlement
    entries: int


def follow(m: Market, rule: Rule, lo: int, hi: int, spot_bp: float, perp_bp: float) -> Path_:
    """The rule over the settlements at times in [lo, hi)."""
    rows = np.flatnonzero((m.time >= lo) & (m.time < hi))
    bad = rows[~(np.isfinite(m.spot[rows]) & np.isfinite(m.perp[rows]))]
    if len(bad):
        when = ", ".join(_stamp(m.time[i]) for i in bad[:5])
        raise ex.ExperimentError(f"no price at {len(bad)} settlements, the first at {when}")
    sig = trailing_mean(m.rate, rule.lookback)
    n = len(rows)
    funding, basis, fees = np.zeros(n), np.zeros(n), np.zeros(n)
    held = np.zeros(n, dtype=bool)
    fs, fp = spot_bp / 1e4, perp_bp / 1e4
    state, entry, entries = False, np.nan, 0
    with np.errstate(invalid="ignore"):
        want = sig[rows] > rule.threshold_bp / 1e4
    for k, i in enumerate(rows):
        w = bool(want[k]) and k < n - 1  # flat at the end of the period
        if w != state:
            if w:
                entry = m.spot[i]
                entries += 1
            fees[k] += (fs * m.spot[i] + fp * m.perp[i]) / entry  # in and out alike
            state = w
        held[k] = state
        if state:
            j = rows[k + 1]
            funding[k + 1] += m.rate[j] * m.mark[j] / entry
            basis[k + 1] += ((m.perp[i] - m.spot[i]) - (m.perp[j] - m.spot[j])) / entry
    return Path_(rows, funding, basis, fees, held, entries)


def _stamp(t: int) -> str:
    return dt.datetime.fromtimestamp(t / 1e9, dt.UTC).strftime("%Y-%m-%d %H:%M")


def _day0(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * 10**9


def daily(m: Market, p: Path_, first: str, last: str) -> pl.DataFrame:
    """The P&L of each UTC day from `first` to `last`, in bp: funding, basis, fees, net."""
    lo, hi = _day0(first), _day0(last) + NS_DAY
    days = (hi - lo) // NS_DAY
    d = (m.time[p.rows] - lo) // NS_DAY
    out = {}
    for name, v in (("funding", p.funding), ("basis", p.basis), ("fees", -p.fees)):
        out[name] = np.bincount(d, weights=v * 1e4, minlength=days)[:days]
    dates = [(dt.date.fromisoformat(first) + dt.timedelta(days=k)).isoformat() for k in range(days)]
    df = pl.DataFrame({"date": dates, **out})
    return df.with_columns(net=pl.col("funding") + pl.col("basis") + pl.col("fees"))


@dataclass(frozen=True)
class Score:
    days: int
    mean_bp: float  # a day, net
    se: float  # Newey-West
    p: float  # one-sided: the mean above zero
    annual_pct: float
    sharpe: float  # a year
    max_drawdown_bp: float
    held: float  # share of the settlement periods with the position
    entries: int
    funding_bp_year: float
    basis_bp_year: float
    fees_bp_year: float
    positive_months: float

    def row(self) -> dict:
        return {k: (ex._f(v) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score(d: pl.DataFrame, p: Path_, lags: int) -> Score:
    x = d["net"].to_numpy()
    n = len(x)
    years = n / 365.0
    se = metrics.newey_west_se(x, lags)
    mean = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    months = monthly(d)
    return Score(
        days=n,
        mean_bp=mean,
        se=se,
        p=float(stats.norm.sf(mean / se)) if se > 0 else float("nan"),
        annual_pct=mean * 365 / 100,
        sharpe=mean / sd * np.sqrt(365) if sd > 0 else float("nan"),
        max_drawdown_bp=float((peak - cum).max()) if n else float("nan"),
        held=float(p.held[:-1].mean()) if len(p.held) > 1 else float("nan"),
        entries=p.entries,
        funding_bp_year=float(d["funding"].sum()) / years if n else float("nan"),
        basis_bp_year=float(d["basis"].sum()) / years if n else float("nan"),
        fees_bp_year=float(d["fees"].sum()) / years if n else float("nan"),
        positive_months=float((months["net"] > 0).mean()) if len(months) else float("nan"),
    )


def monthly(d: pl.DataFrame) -> pl.DataFrame:
    """Net P&L by month: its sum and its mean a day."""
    return (
        d.group_by(pl.col("date").str.slice(0, 7).alias("month"), maintain_order=True)
        .agg(pl.col("net").sum(), pl.col("net").mean().alias("mean"))
        .sort("month")
    )


ALWAYS = Rule(1, -1e12)  # held from the first settlement to the last: for comparison


# --- the development run -----------------------------------------------------------------


@dataclass
class Report:
    spec: Spec
    provenance: ex.Provenance
    settlements: int
    filled: int
    table: list[dict]  # every variant, the primary first
    always: dict
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
            "kind": "carry",
            "hypothesis": s.hypothesis,
            "spec": s.path.name,
            "spec_commit": self.provenance.spec_commit,
            "head": self.provenance.head,
            "dirty": self.provenance.dirty,
            "marketlab": __version__,
            "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data": {
                "symbol": s.symbol,
                "days": [s.first, s.last],
                "holdout": [s.holdout_first, s.holdout_last],
                "settlements": self.settlements,
                "prices_from_the_bar_before": self.filled,
            },
            "costs": {"spot_bp": s.spot_bp, "perp_bp": s.perp_bp},
            "primary": s.primary.name,
            "table": self.table,
            "always_held": self.always,
            "by_month": self.by_month,
            "pbo": self.pbo.__dict__ if self.pbo else None,
            "p_raw": primary.get("p"),
            "p_bh": ex._f(self.p_bh),
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _period(m: Market, spec: Spec, rule: Rule, first: str, last: str):
    p = follow(m, rule, _day0(first), _day0(last) + NS_DAY, spec.spot_bp, spec.perp_bp)
    d = daily(m, p, first, last)
    return p, d


def run(
    spec: Spec,
    root: str | Path,
    results: Path,
    prov: ex.Provenance | None = None,
    market: Market | None = None,
) -> Report:
    prov = prov or ex.provenance(spec.path)
    m = market or Market.load(root, spec)
    lo, hi = _day0(spec.first), _day0(spec.last) + NS_DAY
    inside = (m.time >= lo) & (m.time < hi)
    if not inside.any():
        reason = f"no funding settlement between {spec.first} and {spec.last}"
        report = Report(spec, prov, 0, m.filled, [], {}, [], None, float("nan"),
                        "insufficient data", [reason])  # fmt: skip
        return _finish(report, results)
    table, months = [], []
    for r in spec.variants:
        p, d = _period(m, spec, r, spec.first, spec.last)
        table.append({"rule": r.name, "lookback": r.lookback, "threshold_bp": r.threshold_bp,
                      **score(d, p, spec.nw_lags).row()})  # fmt: skip
        months.append(monthly(d))
        if r == spec.primary:
            by_month = months[-1].to_dicts()
    pa, da = _period(m, spec, ALWAYS, spec.first, spec.last)
    always = score(da, pa, spec.nw_lags).row()
    perf = np.stack([x["mean"].to_numpy() for x in months], axis=1)
    result_pbo = pbo.pbo(perf, groups=16) if len(spec.variants) > 1 else None
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
        reasons.append(
            f"positive in {primary['positive_months']:.0%} of months, the hypothesis asks "
            f"for {spec.min_positive_months:.0%}"
        )
    report = Report(spec, prov, int(inside.sum()), m.filled, table, always, by_month,
                    result_pbo, p_bh, "failed" if reasons else "passed", reasons)  # fmt: skip
    return _finish(report, results)


def _finish(report: Report, results: Path) -> Report:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{report.spec.slug}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return report


# --- the held-out period -------------------------------------------------------------------


def final(
    spec: Spec,
    root: str | Path,
    results: Path,
    prov: ex.Provenance | None = None,
    market: Market | None = None,
) -> dict:
    """The primary rule over the held-out period, once. A second call is refused."""
    out = results / f"{spec.slug}.holdout.json"
    if out.exists():
        raise ex.ExperimentError(f"the held-out days of {spec.slug} were opened already ({out})")
    if not (results / f"{spec.slug}.json").exists():
        raise ex.ExperimentError(f"run {spec.slug} on the development days first")
    prov = prov or ex.provenance(spec.path)
    m = market or Market.load(root, spec)
    lo, hi = _day0(spec.holdout_first), _day0(spec.holdout_last) + NS_DAY
    if not ((m.time >= lo) & (m.time < hi)).any():
        raise ex.ExperimentError("no funding settlement in the held-out period")
    p, d = _period(m, spec, spec.primary, spec.holdout_first, spec.holdout_last)
    s = score(d, p, spec.nw_lags)
    pa, da = _period(m, spec, ALWAYS, spec.holdout_first, spec.holdout_last)
    record = {
        "id": spec.id,
        "name": spec.name,
        "spec_commit": prov.spec_commit,
        "head": prov.head,
        "dirty": prov.dirty,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "period": [spec.holdout_first, spec.holdout_last],
        "rule": spec.primary.name,
        "summary": s.row(),
        "confirmed": bool(s.mean_bp > 0 and s.p < spec.alpha),
        "always_held": score(da, pa, spec.nw_lags).row(),
        "by_month": monthly(d).to_dicts(),
    }
    results.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
