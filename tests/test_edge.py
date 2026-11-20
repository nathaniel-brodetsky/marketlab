"""Hypotheses of the `edge` kind: the returns and the rule worked out by hand, no decision
reading a trade after its time, a planted edge found and its absence not, surrogates that
refuse, the held-out days opened once and shown in the list, the specification checked,
and the committed hypothesis."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_tradeflow import PROV, SEC, frame, start_of, write_day

from marketlab import edge
from marketlab import experiment as ex

SPEC = """
id: "023"
name: planted-edge
kind: edge
hypothesis: The planted flow pays more than it costs.
data:
  kind: aggTrades
  market: spot
  symbol: BTCUSDT
  days: {from: 2026-01-01, to: 2026-01-12}
  holdout: {from: 2026-01-13, to: 2026-01-15}
  min_days: 8
sampling: 5s
signals:
  - {method: signed_volume, windows: [10s, 30s]}
  - {method: signed_count, windows: [100t]}
horizons: [1m, 5m]
rule: {quantile: 0.95, lookback: 10m, latency: 300ms}
cost_bp: 1.0
selection: {rank_by: p_cost, require: {positive_days: 0.6, surrogates_above: 95}, take: 1}
protocol: {surrogates: 200, seed: 5}
success: {max_pbo: 0.5, alpha: 0.05}
"""

DAY = start_of("2026-01-01")


def spec_file(tmp_path, text=SPEC, name="023-planted-edge.yaml"):
    p = tmp_path / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_returns_by_hand(tmp_path):
    spec = edge.load_spec(spec_file(tmp_path))
    assert len(spec.combos) == 6 and spec.back == 120
    t = np.arange(1, 601)  # a trade each second for ten minutes
    price = 100 * np.exp(np.cumsum(np.where(t % 7 == 0, 3e-4, -2e-5)))
    df = edge.day_inputs(frame(t, price, t / 100, t % 2 == 0), "2026-01-01", spec)
    row = df.filter(pl.col("ts") == DAY + 100 * SEC).row(0, named=True)
    # filled 300 ms after 100 s: the last trade then is at 100 s; closed at 160.3 s: 160 s
    assert row["gross_1m"] == pytest.approx(np.log(price[159] / price[99]) * 1e4)
    assert row["signed_volume_10s"] == pytest.approx(
        (np.where(t % 2 == 0, 1, -1) * t / 100)[90:100].sum()
    )
    assert row["open_1m"] and row["open_5m"]
    late = df.filter(pl.col("ts") == DAY + 580 * SEC).row(0, named=True)
    assert late["gross_1m"] == pytest.approx(np.log(price[599] / price[579]) * 1e4)  # last trade
    # the latency counts: decided at 95 s, filled at 95.3 s, after the trade at 95.2 s
    shifted = edge.day_inputs(frame(t + 0.2, price, t / 100, t % 2 == 0), "2026-01-01", spec)
    r2 = shifted.filter(pl.col("ts") == DAY + 95 * SEC).row(0, named=True)
    assert r2["gross_1m"] == pytest.approx(np.log(price[154] / price[94]) * 1e4)
    # no position past midnight, by the clock
    end = frame([86_000.0, 86_390.0, 86_399.0], [100.0, 101.0, 102.0], [1.0] * 3, [True] * 3)
    e = edge.day_inputs(end, "2026-01-01", spec)
    opened = e.filter(pl.col("open_5m"))["ts"].max()
    assert opened <= DAY + 86_400 * SEC - 300_000_000 - 300 * SEC


def test_the_rule_by_hand():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    x[:30] = np.nan
    side = edge.sides(x, 0.9, 40)
    for i in [*rng.choice(np.arange(30, 400), 60, replace=False).tolist(), 49, 50, 51]:
        before = x[max(i - 40, 0) : i]
        before = before[np.isfinite(before)]
        if len(before) < 20:
            assert side[i] == 0
            continue
        hi, lo = np.quantile(before, 0.9), np.quantile(before, 0.1)
        assert side[i] == (1 if x[i] > hi else -1 if x[i] < lo else 0)
    assert (side == 1).sum() > 15 and (side == -1).sum() > 15
    # one position at a time: the next decision once the last is closed
    ts = np.arange(len(x)) * 5 * SEC
    can = np.ones(len(x), bool)
    can[350:] = False
    rows = edge.take(ts, side, can, 61 * SEC)
    expect, free = [], -1
    for i in range(len(x)):
        if side[i] != 0 and can[i] and ts[i] >= free:
            expect.append(i)
            free = ts[i] + 61 * SEC
    assert rows.tolist() == expect and len(expect) > 10
    assert np.diff(ts[rows]).min() >= 61 * SEC and rows.max() < 350


def test_no_decision_reads_a_trade_after_its_time(tmp_path):
    spec = edge.load_spec(spec_file(tmp_path))
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 3600, 14_000))
    trades = frame(t, 97000 * np.exp(np.cumsum(rng.normal(0, 1e-5, len(t)))),
                   rng.exponential(0.01, len(t)), rng.random(len(t)) < 0.5)  # fmt: skip
    full = edge.day_inputs(trades, "2026-01-01", spec)
    for c in spec.combos:
        hold = 300_000_000 + int(c.horizon[:-1]) * 60 * SEC
        side = edge.sides(full[c.signal.key].to_numpy(), spec.quantile, spec.back)
        rows = edge.take(full["ts"].to_numpy(), side, full[f"open_{c.horizon}"].to_numpy(), hold)
        assert len(rows) > 5
        for cut in (1500.0, 2702.5):
            part = trades.filter(pl.col("ts") <= DAY + int(cut * SEC))
            short = edge.day_inputs(part, "2026-01-01", spec)
            s2 = edge.sides(short[c.signal.key].to_numpy(), spec.quantile, spec.back)
            r2 = edge.take(short["ts"].to_numpy(), s2, short[f"open_{c.horizon}"].to_numpy(), hold)
            n = len(short)
            assert np.array_equal(side[:n], s2), (c.name, cut)
            assert rows[rows < n].tolist() == r2.tolist(), (c.name, cut)


def test_a_planted_edge_is_found_and_its_absence_is_not(tmp_path, capsys):
    for beta, sub in ((1.5, "planted"), (0.0, "none")):
        for k in range(15):
            write_day(tmp_path / sub, f"2026-01-{k + 1:02d}", beta, seed=30 * k + (beta > 0))
    spec = edge.load_spec(spec_file(tmp_path))
    r = edge.run(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)
    assert r.verdict == "screened", r.reasons
    best = r.chosen[0]
    assert best["combination"].startswith("signed_volume") and best["be_bp"] > 2.0
    # taken for the strength of its evidence, not for the largest mean: a few noisy trades
    # at a long horizon may have a larger one
    assert best["p_cost"] == min(row["p_cost"] for row in r.table)
    assert best["be_bp"] > best["surrogates"]["shift"]["threshold"]
    j = json.loads((tmp_path / "results" / "023-planted-edge.json").read_text())
    assert j["kind"] == "edge" and j["p_raw"] is None and j["combinations"] == 6
    from marketlab.cli import print_edge_report

    print_edge_report(r)
    out = capsys.readouterr().out
    assert "chosen 1" in out and "1 bp a round trip" in out
    rec = edge.final(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)
    assert rec["days"] == ["2026-01-13", "2026-01-14", "2026-01-15"]
    assert rec["chosen"][0]["combination"] == best["combination"]
    assert rec["chosen"][0]["confirmed"], rec["chosen"][0]
    with pytest.raises(ex.ExperimentError, match="opened already"):
        edge.final(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)
    table = ex.registry_table(tmp_path / "results", tmp_path / "experiments")
    assert table["held_out"].to_list() == ["confirmed 1/1"]

    r = edge.run(spec, tmp_path / "none", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed" and not r.chosen, r.chosen
    top = max(r.table, key=lambda row: row["be_bp"])
    assert abs(top["be_bp"]) < 2.0
    with pytest.raises(ex.ExperimentError, match="did not pass"):
        edge.final(spec, tmp_path / "none", tmp_path / "results2", prov=PROV)
    # the same planted days at a cost above what the flow pays: nothing is taken
    # (and with no share of days asked for, so that the cost alone keeps them out)
    text = SPEC.replace("cost_bp: 1.0", "cost_bp: 100").replace(
        "positive_days: 0.6", "positive_days: 0"
    )
    dear = edge.load_spec(spec_file(tmp_path, text))
    r = edge.run(dear, tmp_path / "planted", tmp_path / "results3", prov=PROV)
    assert r.verdict == "failed" and not r.chosen and not r.passed_over
    assert any("beating 100 bp" in x for x in r.reasons)


def test_the_rule_passes_over_a_combination_its_surrogates_refuse(tmp_path, monkeypatch):
    for k in range(12):
        write_day(tmp_path / "d", f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    spec = edge.load_spec(spec_file(tmp_path))
    real, calls = edge.surrogate_means, []

    def refuse_the_first(inputs, spec, c, seed):
        out = real(inputs, spec, c, seed)
        calls.append(c.name)
        return out + 1e3 if len(calls) == 1 else out

    monkeypatch.setattr(edge, "surrogate_means", refuse_the_first)
    r = edge.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    best = min(r.table, key=lambda row: row["p_cost"])["combination"]
    assert [c["combination"] for c in r.passed_over] == [best]
    assert r.chosen and r.chosen[0]["combination"] != best


def test_scores():
    days = [edge.DayTrades(4, 3.0, 3), edge.DayTrades(2, 1.0, 1), edge.DayTrades(0, np.nan, 0)]
    s = edge.score(days, 0.5)
    assert s.days == 2 and s.trades_day == pytest.approx(2.0)
    assert s.be_bp == pytest.approx(2.0) and s.se == pytest.approx(np.std([3, 1], ddof=1) / 2**0.5)
    assert s.positive_days == 1.0 and s.hit_rate == pytest.approx(4 / 6)
    from scipy import stats

    assert s.p_cost == pytest.approx(stats.t.sf((2.0 - 0.5) / s.se, 1))
    assert edge.score(days, 2.5).p_cost > 0.5  # below the cost: no evidence it beats it


def test_bad_specifications_are_refused(tmp_path):
    bad = {
        "not an `edge`": SPEC.replace("kind: edge", "kind: screen"),
        "held-out days inside": SPEC.replace("to: 2026-01-12", "to: 2026-01-14"),
        "not a direction signal": SPEC.replace("method: signed_count", "method: intensity"),
        "rule.quantile": SPEC.replace("quantile: 0.95", "quantile: 0.4"),
        "rule.lookback": SPEC.replace("lookback: 10m", "lookback: 7s"),
        "cost_bp must not": SPEC.replace("cost_bp: 1.0", "cost_bp: -1"),
        "take must be": SPEC.replace("take: 1", "take: 7"),
        "ranks by `p_cost`": SPEC.replace("rank_by: p_cost", "rank_by: be_bp"),
        "at least 200": SPEC.replace("surrogates: 200", "surrogates: 20"),
        "`cost_bp` is missing": SPEC.replace("cost_bp: 1.0\n", ""),
    }
    for message, text in bad.items():
        with pytest.raises(ex.ExperimentError, match=message):
            edge.load_spec(spec_file(tmp_path, text))
    spec = edge.load_spec(spec_file(tmp_path))
    assert edge.run(spec, tmp_path / "empty", tmp_path / "r", prov=PROV).verdict == (
        "insufficient data"
    )


def test_end_to_end_through_the_command(tmp_path, capsys):
    from test_experiment import git

    from marketlab.cli import main

    # six held-out days: three hours of trades are too few for a test with two degrees of freedom
    for k in range(18):
        write_day(tmp_path / "data", f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path, SPEC.replace("to: 2026-01-15", "to: 2026-01-18"))
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "edge")
    args = ["experiment", "run", str(p), "--data", str(tmp_path / "data")]
    assert main([*args, "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "cost        1 bp a round trip" in out and "verdict     screened" in out
    assert main([*args, "--final"]) == 0
    assert ": confirmed" in capsys.readouterr().out
    assert len(list((tmp_path / ".cache").rglob("edge-*.parquet"))) == 18
    assert main(["experiment", "list", "--results", str(tmp_path / "results"),
                 "--experiments", str(tmp_path / "experiments")]) == 0  # fmt: skip
    assert "confirmed 1/1" in capsys.readouterr().out


def test_the_committed_hypothesis_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = edge.load_spec(root / "007-flow-beats-cost.yaml")
    assert len(spec.combos) == 32 and spec.take == 1 and spec.cost_bp == 9.0
    assert len(spec.development) == 79 and len(spec.holdout) == 243 - 8
    assert spec.development[0] == "2026-09-01" and spec.development[-1] == "2026-11-18"
    assert spec.holdout[0] == "2026-01-02" and spec.holdout[-1] == "2026-08-31"
    # the first of each month was read before, as a day of Tardis.dev's futures recording
    assert not {f"2026-{m:02d}-01" for m in range(1, 9)} & set(spec.holdout)
    from marketlab.cli import kind_of

    assert kind_of(root / "007-flow-beats-cost.yaml") == "edge"
