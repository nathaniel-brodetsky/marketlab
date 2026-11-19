"""Hypotheses of the `screen` kind: the signals worked out by hand, none of them reading a
trade after its time, the rule taking a planted signal and refusing noise, the held-out
days opened once for the chosen only, and the committed screen."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_tradeflow import PROV, SEC, frame, start_of, write_day

from marketlab import experiment as ex
from marketlab import screen as sc

SPEC = """
id: "021"
name: planted-screen
kind: screen
hypothesis: The planted flow is chosen.
data:
  kind: aggTrades
  market: spot
  symbol: BTCUSDT
  days: {from: 2026-01-01, to: 2026-01-12}
  holdout: {from: 2026-01-13, to: 2026-01-15}
  min_days: 8
sampling: 5s
grid:
  direction:
    targets: [return_30s]
    signals:
      - {method: signed_volume, windows: [100t, 30s]}
      - {method: size_skew, windows: [100t]}
  activity:
    targets: [abs_return_30s]
    signals:
      - {method: intensity, windows: [30s]}
      - {method: realized_vol, windows: [1m]}
selection: {rank_by: mean_ic, require: {positive_days: 0.6, surrogates_above: 95}, take: 2}
protocol: {surrogates: 200, seed: 4}
success: {max_pbo: 0.5, alpha: 0.05}
"""

ALL = """
id: "022"
name: every-signal
kind: screen
hypothesis: Every signal computes.
data:
  symbol: BTCUSDT
  days: {from: 2026-01-01, to: 2026-01-01}
  holdout: [2026-01-02]
  min_days: 1
sampling: 5s
grid:
  direction:
    targets: [return_5s, return_30s]
    signals:
      - {method: signed_volume, windows: [3t, 10s]}
      - {method: signed_count, windows: [3t, 10s]}
      - {method: size_skew, windows: [4t, 10s]}
  activity:
    targets: [abs_return_30s]
    signals:
      - {method: intensity, windows: [10s]}
      - {method: volume_rate, windows: [10s]}
      - {method: realized_vol, windows: [10s]}
      - {method: hawkes, params: {window: 10m, step: 5m}}
selection: {take: 1}
protocol: {surrogates: 200}
"""


def spec_file(tmp_path, text=SPEC, name="021-planted-screen.yaml"):
    p = tmp_path / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_every_signal_by_hand(tmp_path):
    spec = sc.load_spec(spec_file(tmp_path, ALL, "022-every-signal.yaml"))
    assert len(spec.combos) == 6 * 2 + 4 * 1
    # a trade every second, 1 .. 60 s: buys at even seconds, qty = second / 100
    t = np.arange(1, 61)
    price = 100 * np.exp(np.cumsum(np.where(t % 3 == 0, 1e-4, -5e-5)))
    df, _ = sc.day_inputs(frame(t, price, t / 100, t % 2 == 0), "2026-01-01", spec)
    row = df.filter(pl.col("ts") == start_of("2026-01-01") + 40 * SEC).row(0, named=True)
    sign, q = np.where(t % 2 == 0, 1.0, -1.0), t / 100
    last3, last10 = slice(37, 40), slice(30, 40)  # trades 38..40 s and 31..40 s
    assert row["signed_volume_3t"] == pytest.approx((sign * q)[last3].sum())
    assert row["signed_count_10s"] == pytest.approx(sign[last10].sum())
    b, s = (sign > 0)[last10], (sign < 0)[last10]
    assert row["size_skew_10s"] == pytest.approx(q[last10][b].mean() - q[last10][s].mean())
    assert row["intensity_10s"] == pytest.approx(1.0)
    assert row["volume_rate_10s"] == pytest.approx(q[last10].sum() / 10)
    r = np.diff(np.log(price))[29:39]  # one-second returns ending at 31 .. 40 s
    assert row["realized_vol_10s"] == pytest.approx(np.sqrt((r**2).sum()))
    assert row["return_5s"] == pytest.approx(np.log(price[44] / price[39]))
    assert np.isnan(row["abs_return_30s"])  # 40 + 30 s is past the last trade, at 60 s
    row30 = df.filter(pl.col("ts") == start_of("2026-01-01") + 30 * SEC).row(0, named=True)
    assert row30["abs_return_30s"] == pytest.approx(abs(np.log(price[59] / price[29])))
    assert np.isnan(df["hawkes_10m_5m"].to_numpy()).all()  # a minute of trades: no fit yet


def test_no_signal_reads_a_trade_after_its_time(tmp_path):
    spec = sc.load_spec(spec_file(tmp_path, ALL, "022-every-signal.yaml"))
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 2400, 12_000))
    trades = frame(t, 97000 * np.exp(np.cumsum(rng.normal(0, 1e-5, len(t)))),
                   rng.exponential(0.01, len(t)), rng.random(len(t)) < 0.5)  # fmt: skip
    full, diag = sc.day_inputs(trades, "2026-01-01", spec)
    assert diag["hawkes_10m_5m"]["fits"] > 3
    assert np.isfinite(full["hawkes_10m_5m"].to_numpy()).sum() > 100
    signals = [s.key for s in spec.signals]
    for cut in (900.0, 1530.5, 2100.0):
        part = trades.filter(pl.col("ts") <= start_of("2026-01-01") + int(cut * SEC))
        short, _ = sc.day_inputs(part, "2026-01-01", spec)
        a = full.filter(pl.col("ts") <= short["ts"].max()).select("ts", *signals)
        b = short.select("ts", *signals)
        for c in signals:
            assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True, rtol=1e-9), (
                cut,
                c,
            )


def test_the_rule_takes_the_planted_signal_and_refuses_noise(tmp_path, capsys):
    for beta, sub in ((1.5, "planted"), (0.0, "none")):
        for k in range(15):
            write_day(
                tmp_path / sub, f"2026-01-{k + 1:02d}", beta, seed=20 * k + (beta > 0), hours=1.0
            )
    spec = sc.load_spec(spec_file(tmp_path))
    r = sc.run(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)
    assert r.verdict == "screened", r.reasons
    names = [c["combination"] for c in r.chosen]
    assert len(names) == 2 and all(n.startswith("signed_volume") for n in names)
    assert r.pbo is not None and r.pbo.pbo <= 0.5
    j = json.loads((tmp_path / "results" / "021-planted-screen.json").read_text())
    assert j["kind"] == "screen" and j["p_raw"] is None and j["combinations"] == 5
    from marketlab.cli import print_screen_report

    print_screen_report(r)
    assert "chosen 1" in capsys.readouterr().out
    rec = sc.final(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)
    assert rec["days"] == ["2026-01-13", "2026-01-14", "2026-01-15"]
    assert [c["combination"] for c in rec["chosen"]] == [c for c in names if c in rec_names(rec)]
    assert all(c["confirmed"] for c in rec["chosen"])
    with pytest.raises(ex.ExperimentError, match="opened already"):
        sc.final(spec, tmp_path / "planted", tmp_path / "results", prov=PROV)

    r = sc.run(spec, tmp_path / "none", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed" and len(r.chosen) < 2, r.chosen
    with pytest.raises(ex.ExperimentError, match="did not pass"):
        sc.final(spec, tmp_path / "none", tmp_path / "results2", prov=PROV)


def rec_names(rec) -> set[str]:
    return {c["combination"] for c in rec["chosen"]}


def test_bad_screens_are_refused(tmp_path):
    bad = {
        "not a `screen`": SPEC.replace("kind: screen", "kind: tradeflow"),
        "held-out days inside": SPEC.replace("to: 2026-01-12", "to: 2026-01-14"),
        "not a direction signal": SPEC.replace("method: size_skew", "method: intensity"),
        "intensity over 100t": SPEC.replace(
            "{method: intensity, windows: [30s]}", "{method: intensity, windows: [100t]}"
        ),
        "in grid.activity": SPEC.replace("targets: [abs_return_30s]", "targets: [return_30s]"),
        "take must be": SPEC.replace("take: 2", "take: 9"),
        "at least 200": SPEC.replace("surrogates: 200", "surrogates: 10"),
    }
    for message, text in bad.items():
        with pytest.raises(ex.ExperimentError, match=message):
            sc.load_spec(spec_file(tmp_path, text))
    assert (
        sc.run(
            sc.load_spec(spec_file(tmp_path)), tmp_path / "empty", tmp_path / "r", prov=PROV
        ).verdict
        == "insufficient data"
    )


def test_the_committed_screen_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = sc.load_spec(root / "006-screen-trades.yaml")
    assert len(spec.combos) == 44 and spec.take == 3
    assert len(spec.development) == 58 and len(spec.holdout) == 21
    assert spec.development[-1] == "2026-10-28" and spec.holdout[0] == "2026-10-29"
    keys = {s.key for s in spec.signals}
    assert {"signed_volume_100t", "hawkes_10m_5m", "realized_vol_1m"} <= keys
    from marketlab.cli import kind_of

    assert kind_of(root / "006-screen-trades.yaml") == "screen"


def test_end_to_end_through_the_command(tmp_path, capsys):
    from test_experiment import git

    from marketlab.cli import main

    for k in range(15):
        write_day(tmp_path / "data", f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "screen")
    assert (
        main(["experiment", "run", str(p), "--data", str(tmp_path / "data"), "--workers", "2"]) == 0
    )
    out = capsys.readouterr().out
    assert "grid        5 combinations" in out and "verdict     screened" in out
    assert main(["experiment", "run", str(p), "--data", str(tmp_path / "data"), "--final"]) == 0
    assert "confirmed" in capsys.readouterr().out
    assert len(list((tmp_path / ".cache").rglob("screen-*.parquet"))) == 15


def test_the_rule_passes_over_a_signal_its_surrogates_refuse(tmp_path, monkeypatch):
    for k in range(15):
        write_day(tmp_path / "d", f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    spec = sc.load_spec(spec_file(tmp_path))
    real, calls = ex._surrogates, []

    def refuse_the_first(s, pred, mean, shape, workers):
        out = real(s, pred, mean, shape, workers)
        calls.append(mean)
        if len(calls) == 1:
            out = {k: {**v, "threshold": mean + 1.0} for k, v in out.items()}
        return out

    monkeypatch.setattr(ex, "_surrogates", refuse_the_first)
    r = sc.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    best = max(r.table, key=lambda row: row["mean_ic"])["combination"]
    assert [c["combination"] for c in r.passed_over] == [best]
    assert best not in [c["combination"] for c in r.chosen]
    assert r.chosen and r.chosen[0]["combination"].startswith("signed_volume")
