"""Hypotheses of the `maker` kind: a withdrawal keeps exactly the fills the simulator gives
an order cancelled then, the rule reads the signal as it was known, and the verdicts on
inputs with a planted effect and without one."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from marketlab import experiment as ex
from marketlab import maker, synth
from marketlab.execution import Market, Mode, Order, frames, probe_orders, simulate
from marketlab.io.day import Day
from marketlab.io.days import DayInfo

SEC = 1_000_000_000
MS = 1_000_000


@pytest.fixture(scope="module")
def day(tmp_path_factory) -> Day:
    s = synth.Scenario(seed=7, seconds=600.0, trade_rate=12.0, gaps=((300.0, 4.0),))
    return Day.open(synth.write(s, tmp_path_factory.mktemp("maker")))


def test_a_withdrawal_keeps_the_fills_of_an_order_cancelled_then(day):
    # the rule's shortcut: cancelling earlier changes nothing before the cancel takes effect
    m = Market.of(day)
    lat = 300 * MS
    base = probe_orders(m, "2s", 100, "10s")
    rng = np.random.default_rng(3)
    pulls = [o.decided + int(rng.integers(0, 10 * SEC)) for o in base]
    early = [
        Order(o.id, o.side, o.qty, o.decided, o.passive, o.price, cancel=p)
        for o, p in zip(base, pulls, strict=True)
    ]
    checked = 0
    for mode in Mode:
        orders, f_base = frames(simulate(m, base, lat, mode), mode)
        _, f_early = frames(simulate(m, early, lat, mode), mode)
        keep = maker.kept(f_base, orders.sort("id"), np.array(pulls), lat)
        a = f_base.filter(pl.Series(keep)).select("id", "ts", "qty")
        b = f_early.select("id", "ts", "qty")
        assert a.equals(b), mode
        checked += len(f_base) - len(a)
    assert checked > 50, "the test must drop fills to mean anything"


def test_the_signal_is_known_at_its_time(day):
    full = maker.signal(day, ["1s", "5s"])
    ts = full["ts"].to_numpy()
    cut = int(ts[len(ts) * 2 // 3])
    part = maker.signal(day.truncated(cut), ["1s", "5s"])
    # seg_end names the end of a segment, which a truncated day does not know yet; the rule
    # only asks whether a decision is still before it, which the decision's time tells
    cols = ["ts", "segment", "z_1s", "z_5s", "rv_60s"]
    assert part.select(cols).equals(full.filter(pl.col("ts") <= cut).select(cols))
    assert (full["seg_end"] > full["ts"]).all()
    z = full["z_1s"].to_numpy()
    assert np.isfinite(z).mean() > 0.5 and np.nanstd(z) == pytest.approx(1, abs=0.3)


def test_trailing_std_is_the_hour_before():
    rng = np.random.default_rng(1)
    ts = np.cumsum(rng.integers(50, 150, 4000)) * MS
    x = rng.standard_normal(4000)
    x[[10, 500, 2000]] = np.nan
    got = maker.trailing_std(ts, x, 60 * SEC, 30)
    for i in (0, 29, 30, 31, 700, 1500, 3999):
        prev = x[(ts >= ts[i] - 60 * SEC) & (ts < ts[i])]
        prev = prev[np.isfinite(prev)]
        want = np.std(prev) if len(prev) >= 30 else np.nan
        assert got[i] == pytest.approx(want, nan_ok=True), i


def signal_frame(z, step=100 * MS):
    """A signal on a grid every `step` from 0, all in one segment."""
    n = len(z)
    ts = step * np.arange(n, dtype=np.int64)
    return pl.DataFrame(
        {
            "ts": ts,
            "segment": np.zeros(n, np.int32),
            "seg_end": np.full(n, ts[-1] + step),
            "z_1s": np.asarray(z, float),
        }
    )


def orders_frame(decided, side):
    return pl.DataFrame(
        {"id": np.arange(len(decided)), "decided": np.asarray(decided), "side": np.asarray(side)}
    )


def test_withdrawals_follow_the_signal_as_it_was_known():
    z = np.zeros(200)
    z[50] = -1.5  # against a bid from 5.0 s
    z[120] = 1.5  # against an ask from 12.0 s
    sig = signal_frame(z)
    o = orders_frame(
        [4 * SEC + 50 * MS, 5 * SEC, 4 * SEC, 1 * SEC, 4 * SEC, 13 * SEC, -1 * SEC],
        [1, 1, -1, -1, 1, -1, 1],
    )
    got = maker.withdrawals(o, sig, z, 1.0, 10 * SEC)
    assert got[0] == 5 * SEC, "the first row against the bid after its decision"
    assert got[1] == 5 * SEC, "against already at the decision: withdrawn at once"
    assert got[2] == 12 * SEC, "an ask waits for the signal above the threshold"
    assert got[3] == -1, "the crossing comes after the planned cancel"
    assert maker.withdrawals(o, sig, z, 2.0, 10 * SEC)[0] == -1, "not beyond the threshold"
    assert got[5] == -1, "the row at 12 s is before this ask's decision"
    assert got[6] == 5 * SEC, "decided before the first row: from the first row on"
    late = orders_frame([5 * SEC + 50 * MS], [1])
    assert maker.withdrawals(late, sig, z, 1.0, 10 * SEC)[0] == 5 * SEC + 50 * MS, (
        "against since the row before the decision: withdrawn at the decision, not before it"
    )
    # segment 0 is valid until 10 s, segment 1 from 20 s: a decision in the hole does not
    # read the last row before it, one just after the last row of its segment does
    z2 = np.zeros(200)
    z2[99] = -3.0  # the last row of segment 0, at 9.9 s
    sig2 = pl.DataFrame(
        {
            "ts": np.r_[np.arange(100) * 100 * MS, 20 * SEC + np.arange(100) * 100 * MS],
            "segment": np.r_[np.zeros(100), np.ones(100)].astype(np.int32),
            "seg_end": np.r_[np.full(100, 10 * SEC), np.full(100, 30 * SEC)],
            "z_1s": z2,
        }
    )
    o2 = orders_frame([15 * SEC, 9 * SEC + 950 * MS], [1, 1])
    w2 = maker.withdrawals(o2, sig2, z2, 1.0, 10 * SEC)
    assert w2[0] == -1 and w2[1] == 9 * SEC + 950 * MS


SPEC = """
id: "009"
name: planted-maker
kind: maker
hypothesis: A planted effect is found.
data: {symbol: BTCUSDT, min_hours: 0.01, min_days: 12, holdout: [2026-01-30]}
orders: {every: 10s, size: 0.001, cancel_after: 10s, latency: 300ms, queue: pessimistic}
signal: {method: ofi_z, params: {window: 1s, threshold: 1.0}}
variants:
  - {window: 1s, threshold: 0.5}
  - {window: 1s, threshold: 1.0}
  - {window: 1s, threshold: 2.0}
target: {markout: 10s}
protocol: {surrogates: 200, min_shift: 10m, seed: 4}
success:
  {min_delta_bp: 0.0, alpha: 0.05, surrogate_percentile: 95, max_pbo: 0.5, min_positive_days: 0.6}
"""


def fake_day(seed: int, planted: bool, hours: float = 2.0) -> maker.DayInputs:
    """A day of a signal that wanders (AR(1) every 100 ms), an order per side every 10 s
    and one fill each 2 s after its decision. Planted: a fill whose signal was beyond one
    unit against it at its decision marks out at -5 bp, the others at +1 bp."""
    rng = np.random.default_rng(seed)
    n = int(hours * 36000)
    e = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = e[0]
    for i in range(1, n):
        z[i] = 0.995 * z[i - 1] + 0.1 * e[i]
    sig = signal_frame(z).with_columns(rv_60s=pl.lit(1.0))
    decided = np.repeat(np.arange(1, int(hours * 360)) * 10 * SEC, 2)
    side = np.tile([1, -1], len(decided) // 2)
    o = orders_frame(decided, side).with_columns(qty=pl.lit(100))
    zd = z[decided // (100 * MS)]
    against = np.where(side == 1, zd < -1, zd > 1)
    mk = (
        np.where(against, -5.0, 1.0)
        if planted
        else np.where(rng.random(len(side)) < 0.2, -5.0, 1.0)
    )
    mk = mk + rng.normal(0, 0.5, len(mk))
    fills = pl.DataFrame(
        {
            "id": np.arange(len(decided)),
            "side": side,
            "ts": decided + 2 * SEC,
            "qty": np.full(len(decided), 100),
            **{f"markout_{h}": mk for h in maker.HORIZONS},
        }
    )
    fills = pl.concat([fills.with_columns(mode=pl.lit(str(m))) for m in Mode])
    return maker.DayInputs(orders=o, fills=fills, signal=sig)


@pytest.fixture
def planted(monkeypatch):
    def use(planted_effect: bool, n_days: int = 13):
        days = {f"2026-01-{d + 1:02d}": fake_day(d, planted_effect) for d in range(n_days)}
        if "2026-01-30" not in days:
            days["2026-01-30"] = fake_day(99, planted_effect)
        infos = [DayInfo(Path("/nowhere") / d, d, 24.0) for d in sorted(days)]
        monkeypatch.setattr(
            maker,
            "survey",
            lambda root, symbol, exclude=(): [i for i in infos if i.date not in exclude],
        )
        monkeypatch.setattr(maker, "load_inputs", lambda info, spec, cache: days[info.date])
        monkeypatch.setattr(
            maker, "load_all", lambda infos, spec, cache, workers: [days[i.date] for i in infos]
        )

    return use


def spec_file(tmp_path, text=SPEC):
    p = tmp_path / "experiments" / "009-planted-maker.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


PROV = ex.Provenance("a" * 40, "b" * 40, False)


def test_a_planted_effect_passes_and_its_absence_fails(tmp_path, planted, capsys):
    planted(True)
    spec = maker.load_spec(spec_file(tmp_path))
    r = maker.run(spec, "unused", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    assert r.mean_delta > 0.5 and r.positive_share == 1.0
    assert r.surrogate["shift"]["real"] > r.surrogate["shift"]["threshold"]
    assert r.pbo is not None and r.pbo.pbo <= 0.5
    assert "2026-01-30" not in r.by_day, "the held-out day stays closed"
    j = json.loads((tmp_path / "results" / "009-planted-maker.json").read_text())
    assert j["kind"] == "maker" and j["verdict"] == "passed" and j["p_raw"] < 0.05
    from marketlab.cli import print_maker_report

    print_maker_report(r)
    assert "break-even" in capsys.readouterr().out
    rec = maker.final(spec, "unused", tmp_path / "results", prov=PROV)
    assert rec["days"] == ["2026-01-30"] and rec["mean_delta_bp"] > 0.5
    with pytest.raises(ex.ExperimentError, match="opened already"):
        maker.final(spec, "unused", tmp_path / "results", prov=PROV)

    planted(False)
    r = maker.run(spec, "unused", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed"
    assert any("shifted signals" in x for x in r.reasons), r.reasons


def test_too_few_days_and_wrong_kinds_are_refused(tmp_path, planted):
    planted(True, n_days=5)
    spec = maker.load_spec(spec_file(tmp_path))
    assert maker.run(spec, "unused", tmp_path / "r", prov=PROV).verdict == "insufficient data"
    with pytest.raises(ex.ExperimentError, match="not a forecast"):
        ex.load_spec(spec_file(tmp_path))
    with pytest.raises(ex.ExperimentError, match="not a `maker`"):
        maker.load_spec(spec_file(tmp_path, SPEC.replace("kind: maker", "kind: forecast")))


def test_the_committed_hypothesis_loads():
    p = Path(__file__).resolve().parents[1] / "experiments" / "002-maker-ofi-pull.yaml"
    s = maker.load_spec(p)
    assert s.slug == "002-maker-ofi-pull" and s.queue is Mode.PESSIMISTIC
    assert s.primary == maker.Variant("1s", 1.0) and len(s.variants) == 6
    assert s.holdout == ex.load_spec(p.with_name("001-mlofi.yaml")).holdout


def test_end_to_end_on_synthetic_days(tmp_path):
    from test_experiment import git

    data = tmp_path / "data"
    for k in range(5):
        start = synth.DAY0 + k * 86_400_000_000_000 + 36_000_000_000_000
        synth.write(synth.Scenario(seed=60 + k, seconds=900, start=start, trade_rate=10.0), data)
    git(tmp_path, "init", "-q", "-b", "main")
    text = SPEC.replace("min_days: 12", "min_days: 4").replace("[2026-01-30]", "[2026-11-25]")
    p = spec_file(tmp_path, text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "hypothesis")
    spec = maker.load_spec(p)
    r = maker.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache", workers=2)
    assert list(r.by_day) == [f"2026-11-{d}" for d in range(21, 25)]
    assert r.verdict in ex.VERDICTS and np.isfinite(r.mean_delta)
    assert all(0 < s.fills_kept <= 1 for s in r.by_day.values())
    assert set(r.modes) == {"pessimistic", "proportional", "optimistic"}
    assert len(list((tmp_path / ".cache").rglob("*.parquet"))) == 3 * 4
    again = maker.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert again.mean_delta == r.mean_delta, "from the cache, the same numbers"
    rec = maker.final(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert rec["days"] == ["2026-11-25"]


def test_a_day_simulated_in_pieces_equals_the_whole_day(day, tmp_path):
    # pieces of two minutes cross the gap, segment ends and the day's end; every order sees
    # what it sees in the whole day
    spec = maker.load_spec(spec_file(tmp_path))
    spec = maker.replace(spec, every="3s")
    whole_o, whole_f = maker.simulate_day(day.path, spec, chunk=None)
    part_o, part_f = maker.simulate_day(day.path, spec, chunk="2m")
    assert len(whole_o) > 300 and len(whole_f) > 100
    assert part_o.equals(whole_o)
    key = ["mode", "id", "ts"]
    assert part_f.sort(key).equals(whole_f.sort(key))
