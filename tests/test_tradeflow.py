"""Hypotheses of the `tradeflow` kind: the signed volume and the target worked out by hand,
the feature reading nothing after its time, a planted effect found and its absence not,
the held-out days opened once, the specification checked, and the committed one."""

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from marketlab import archive
from marketlab import experiment as ex
from marketlab import tradeflow as tf

SEC = 1_000_000_000
PROV = ex.Provenance("a" * 40, "b" * 40, False)

SPEC = """
id: "019"
name: planted-flow
kind: tradeflow
hypothesis: Planted flow is found.
data:
  kind: aggTrades
  market: spot
  symbol: BTCUSDT
  days: {from: 2026-01-01, to: 2026-01-12}
  holdout: {from: 2026-01-13, to: 2026-01-15}
  min_days: 8
sampling: 5s
feature: {method: signed_volume, params: {window: 100t}}
variants:
  - {window: 25t}
  - {window: 100t}
  - {window: 5s}
  - {window: 30s}
target: {kind: return, horizon: 30s}
protocol: {surrogates: 200, seed: 3}
success: {min_ic: 0.0, alpha: 0.05, surrogate_percentile: 95, max_pbo: 0.5, min_positive_days: 0.6}
"""


def start_of(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * SEC


def frame(ts_s, price, qty, buy) -> pl.DataFrame:
    """A day's trades in the shape `archive.read` gives, times in seconds from midnight."""
    t0 = start_of("2026-01-01")
    return pl.DataFrame(
        {
            "ts": [t0 + int(t * SEC) for t in ts_s],
            "price": [float(p) for p in price],
            "qty": [float(q) for q in qty],
            "is_buyer_maker": [not b for b in buy],
        }
    )


def spec_file(tmp_path, text=SPEC):
    p = tmp_path / "experiments" / "019-planted-flow.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_signed_volume_and_target_by_hand(tmp_path):
    spec = tf.load_spec(spec_file(tmp_path))
    # trades at 1, 2, 3 ... 100 s; buys at even seconds, sells at odd; qty = second / 100
    t = np.arange(1, 101)
    trades = frame(t, 100 + t * 0.01, t / 100, t % 2 == 0)
    df = tf.day_inputs(trades, "2026-01-01", spec)
    assert df["ts"].to_list() == [start_of("2026-01-01") + k * 5 * SEC for k in range(1, 21)]
    row = df.filter(pl.col("ts") == start_of("2026-01-01") + 50 * SEC).row(0, named=True)
    signed = np.where(t % 2 == 0, 1, -1) * t / 100
    assert row["flow_25t"] == pytest.approx(signed[25:50].sum())  # trades 26 .. 50 s
    assert np.isnan(row["flow_100t"])  # only 50 trades so far
    assert row["flow_5s"] == pytest.approx(signed[45:50].sum())  # (45, 50]
    assert row["flow_30s"] == pytest.approx(signed[20:50].sum())
    assert row["target"] == pytest.approx(np.log(100.8) - np.log(100.5))
    assert np.isnan(df["flow_30s"][4])  # at 25 s the 30 s window starts before the first trade
    assert np.isnan(df["target"][-1]) and np.isfinite(df["target"][-7])  # 30 s left at 70 s


def test_the_feature_reads_nothing_after_its_time(tmp_path):
    spec = tf.load_spec(spec_file(tmp_path))
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 3600, 20_000))
    trades = frame(
        t, 97000 + rng.normal(0, 5, len(t)), rng.exponential(0.01, len(t)), rng.random(len(t)) < 0.5
    )
    full = tf.day_inputs(trades, "2026-01-01", spec)
    for cut in (600.0, 1777.7, 3000.0):
        short = tf.day_inputs(
            trades.filter(pl.col("ts") <= start_of("2026-01-01") + int(cut * SEC)),
            "2026-01-01",
            spec,
        )
        cols = [w.column for w in spec.variants]
        a = full.filter(pl.col("ts") <= short["ts"].max()).select("ts", *cols)
        assert a.equals(short.select("ts", *cols)), cut


def write_day(root: Path, date: str, beta: float, seed: int, hours: float = 2.0) -> None:
    """Trades over `hours` from midnight, about 4 a second; each second's price move is
    `beta` bp per unit of the signed flow of the ten seconds before, plus noise."""
    rng = np.random.default_rng(seed)
    seconds = int(hours * 3600)
    n = rng.poisson(4, seconds)
    sec = np.repeat(np.arange(seconds), n)
    buy = rng.random(len(sec)) < 0.5
    qty = np.round(rng.exponential(0.01, len(sec)) + 0.00001, 5)
    flow = np.bincount(sec, weights=np.where(buy, qty, -qty), minlength=seconds) / 0.03
    drive = np.convolve(flow, np.ones(10) / 10)[:seconds]
    drive = np.concatenate([[0.0], drive[:-1]])
    logp = np.log(97000) + np.cumsum(beta * drive + rng.normal(0, 0.5, seconds)) * 1e-4
    price = np.round(np.exp(logp[sec]), 2)
    ts_us = start_of(date) // 1000 + sec * 1_000_000 + np.sort(rng.integers(0, 999_999, len(sec)))
    order = np.lexsort((ts_us,))
    k = np.arange(len(sec))
    df = pl.DataFrame(
        {
            "agg_id": 1000 + k,
            "price": price[order],
            "qty": qty[order],
            "first_id": 5000 + 2 * k,
            "last_id": 5001 + 2 * k,
            "ts": np.sort(ts_us),
            "is_buyer_maker": np.where(buy[order], "False", "True"),
            "is_best_match": "True",
        }
    )
    folder = root / "spot" / "aggTrades" / "BTCUSDT"
    folder.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"BTCUSDT-aggTrades-{date}.csv", df.write_csv(include_header=False))
    (folder / f"BTCUSDT-aggTrades-{date}.zip").write_bytes(buf.getvalue())


def test_a_planted_effect_is_found_and_its_absence_is_not(tmp_path, capsys):
    for beta, sub in ((1.5, "planted"), (0.0, "none")):
        root = tmp_path / sub / "data"
        for k in range(15):
            write_day(root, f"2026-01-{k + 1:02d}", beta, seed=10 * k + (beta > 0))
        (root / "spot" / "aggTrades" / "BTCUSDT" / "BTCUSDT-aggTrades-2026-01-05.zip").unlink()
    spec = tf.load_spec(spec_file(tmp_path))
    r = tf.run(spec, tmp_path / "planted" / "data", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    assert r.missing == ["2026-01-05"] and len(r.days) == 11
    assert not set(r.days) & set(spec.holdout)
    assert r.summary.mean > 0.05 and r.summary.positive_share == 1.0
    assert all(x["real"] > x["threshold"] for x in r.surrogate.values())
    j = json.loads((tmp_path / "results" / "019-planted-flow.json").read_text())
    assert j["kind"] == "tradeflow" and j["verdict"] == "passed" and j["p_raw"] < 0.05
    from marketlab.cli import print_tradeflow_report

    print_tradeflow_report(r)
    assert "signed volume" in capsys.readouterr().out
    rec = tf.final(spec, tmp_path / "planted" / "data", tmp_path / "results", prov=PROV)
    assert rec["days"] == ["2026-01-13", "2026-01-14", "2026-01-15"]
    assert rec["summary"]["mean"] > 0.05
    with pytest.raises(ex.ExperimentError, match="opened already"):
        tf.final(spec, tmp_path / "planted" / "data", tmp_path / "results", prov=PROV)

    r = tf.run(spec, tmp_path / "none" / "data", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed" and abs(r.summary.mean) < 0.03
    assert any("surrogates" in x for x in r.reasons), r.reasons


def test_the_family_counts_hypotheses_of_every_kind(tmp_path):
    spec = tf.load_spec(spec_file(tmp_path))
    (tmp_path / "experiments" / "020-other.yaml").write_text('id: "020"\nname: other\n')
    root = tmp_path / "data"
    for k in range(12):
        write_day(root, f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    r = tf.run(spec, root, tmp_path / "results", prov=PROV)
    assert r.p_bh == pytest.approx(min(1.0, r.summary.p * 2))  # two written, this one best
    (tmp_path / "results" / "notes.json").write_text("{}")  # not a hypothesis's record
    table = ex.registry_table(tmp_path / "results", tmp_path / "experiments")
    assert table["id"].to_list() == ["019", "020"]


def test_too_little_data_and_bad_specifications(tmp_path):
    spec = tf.load_spec(spec_file(tmp_path))
    assert (
        tf.run(spec, tmp_path / "empty", tmp_path / "r", prov=PROV).verdict == "insufficient data"
    )
    bad = {
        "not a `tradeflow`": SPEC.replace("kind: tradeflow", "kind: maker"),
        "held-out days inside": SPEC.replace("to: 2026-01-12", "to: 2026-01-13"),
        "holds no trades": SPEC.replace("{window: 25t}", "{window: 0t}"),
        "at least 200": SPEC.replace("surrogates: 200", "surrogates: 50"),
        "signed_volume": SPEC.replace("method: signed_volume", "method: mlofi"),
    }
    for message, text in bad.items():
        with pytest.raises(ex.ExperimentError, match=message):
            tf.load_spec(spec_file(tmp_path, text))
    with pytest.raises(ValueError):
        tf.load_spec(spec_file(tmp_path, SPEC.replace("{window: 5s}", "{window: 5x}")))


def test_the_committed_hypothesis_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = tf.load_spec(root / "005-tradeflow-100t.yaml")
    assert len(spec.development) == 58 and len(spec.holdout) == 21
    assert spec.development[0] == "2026-09-01" and spec.development[-1] == "2026-10-28"
    assert spec.holdout[0] == "2026-10-29" and spec.holdout[-1] == "2026-11-18"
    assert spec.primary.trades == 100 and len(spec.variants) == 7
    from marketlab.cli import kind_of

    assert kind_of(root / "005-tradeflow-100t.yaml") == "tradeflow"
    assert archive.KINDS[spec.root_kind][0] == "agg_id"


def test_end_to_end_through_the_command(tmp_path, capsys):
    from test_experiment import git

    from marketlab.cli import main

    data = tmp_path / "data"
    for k in range(15):
        write_day(data, f"2026-01-{k + 1:02d}", 1.5, seed=k, hours=1.0)
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "hypothesis")
    assert main(["experiment", "run", str(p), "--data", str(data), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "019-planted-flow" in out and "verdict" in out
    assert main(["experiment", "run", str(p), "--data", str(data), "--final"]) == 0
    out = capsys.readouterr().out
    assert "held out    2026-01-13, 2026-01-14, 2026-01-15" in out
    cached = list((tmp_path / ".cache").rglob("tradeflow-*.parquet"))
    assert len(cached) == 15
