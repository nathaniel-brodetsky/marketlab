"""Hypotheses of the `carry` kind: the archives read whatever unit their times are in, the
rule's money worked out by hand, no decision reading a rate or a price after its time, a
planted carry found and none where there is none, the held-out period opened once, the
specification checked, and the committed hypothesis."""

import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_tradeflow import PROV

from marketlab import carry
from marketlab import experiment as ex

H = 3_600_000  # ms
SPEC = """
id: "024"
name: planted-carry
kind: carry
hypothesis: The planted funding pays more than it costs.
data:
  symbol: BTCUSDT
  interval: 1h
  days: {from: 2021-01-01, to: 2021-12-31}
  holdout: {from: 2022-01-01, to: 2022-06-30}
rule: {lookback: 21, threshold_bp: 0.0}
costs: {spot_bp: 7.5, perp_bp: 4.5}
variants:
  - {lookback: 3}
  - {lookback: 63}
  - {threshold_bp: 1.0}
protocol: {nw_lags: 10}
success: {alpha: 0.05, max_pbo: 0.5, min_positive_months: 0.6}
"""


def ms(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp()) * 1000


def spec_file(tmp_path, text=SPEC, name="024-planted-carry.yaml"):
    p = tmp_path / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def write_klines(root, market, open_ms, open_, close, header=False, us_from=None):
    """Monthly 1h kline files as Binance publishes them; times in µs from `us_from` (ms)."""
    folder = root / market / "klines" / "BTCUSDT" / "1h"
    folder.mkdir(parents=True, exist_ok=True)
    months = np.array([dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m")
                       for t in open_ms])  # fmt: skip
    for mo in dict.fromkeys(months):
        k = months == mo
        t = open_ms[k] * (1000 if us_from is not None and open_ms[k][0] >= us_from else 1)
        lines = [",".join(carry.KLINE_COLUMNS)] if header else []
        lines += [f"{a},{o},{max(o, c)},{min(o, c)},{c},1.0,{a + 1},1.0,3,0.5,0.5,0"
                  for a, o, c in zip(t, open_[k], close[k], strict=True)]  # fmt: skip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(f"BTCUSDT-1h-{mo}.csv", "\n".join(lines) + "\n")
        (folder / f"BTCUSDT-1h-{mo}.zip").write_bytes(buf.getvalue())


def write_funding(root, t_ms, rate, mark=None):
    f = root / "um" / "fundingRate" / "BTCUSDT" / "BTCUSDT-fundingRate.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    marks = ["" if mark is None or not np.isfinite(x) else f"{x}" for x in
             (mark if mark is not None else np.full(len(t_ms), np.nan))]  # fmt: skip
    rows = [f"{t},{r:.10f},{m}" for t, r, m in zip(t_ms, rate, marks, strict=True)]
    f.write_text("funding_time,funding_rate,mark_price\n" + "\n".join(rows) + "\n")


def planted(root, mean_bp: float, seed: int, first="2020-12-01", last="2022-07-01"):
    """Hourly prices with a basis of a few bp around zero, and a settlement every 8 h with a
    rate of `mean_bp` give or take one bp."""
    rng = np.random.default_rng(seed)
    t = np.arange(ms(first), ms(last), H)
    spot = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(t))))  # a few % a year
    perp = spot * (1 + rng.normal(0, 2e-4, len(t)))
    write_klines(root, "spot", t, spot, spot * (1 + rng.normal(0, 1e-3, len(t))))
    write_klines(root, "um", t, perp, perp * (1 + rng.normal(0, 1e-3, len(t))))
    ft = t[::8] + 3  # Binance's settlement times are a few ms past the hour at times
    write_funding(root, ft, rng.normal(mean_bp, 1.0, len(ft)) / 1e4)


def test_the_archives_are_read_in_any_unit(tmp_path):
    t = np.arange(ms("2024-12-31T20:00"), ms("2025-01-01T04:00"), H)
    write_klines(tmp_path, "spot", t, np.arange(len(t)) + 100.0, np.arange(len(t)) + 100.5,
                 header=True, us_from=ms("2025-01-01"))  # fmt: skip
    k = carry.read_klines(tmp_path, "spot", "BTCUSDT", "1h")
    assert k["open_time"].to_list() == [x * 1_000_000 for x in t]
    # the bar that starts at the settlement; where there is none, the close of the one before
    write_klines(tmp_path, "um", np.delete(t, 3), np.arange(7) + 200.0, np.arange(7) + 200.5)
    u = carry.read_klines(tmp_path, "um", "BTCUSDT", "1h")
    at = np.array([t[1] + 5, t[3] + 2, t[5]]) * 1_000_000
    price, filled = carry.price_at(u, at, "1h")
    assert price.tolist() == [201.0, 202.5, 204.0] and filled == 1
    far, _ = carry.price_at(u, np.array([t[-1] + 7 * H]) * 1_000_000, "1h")
    assert np.isnan(far[0])  # more than six hours since the last bar
    write_funding(tmp_path, [t[0], t[4]], [1e-4, -2e-4], np.array([np.nan, 41000.0]))
    f = carry.read_funding(tmp_path, "BTCUSDT")
    assert f["time"].to_list() == [t[0] * 1_000_000, t[4] * 1_000_000]
    assert f["rate"].to_list() == [1e-4, -2e-4] and np.isnan(f["mark"][0])
    with pytest.raises(ex.ExperimentError, match="fetch_funding"):
        carry.read_funding(tmp_path / "nothing", "BTCUSDT")


def test_the_money_by_hand():
    # six settlements; the mean of the last two rates decides
    rate = np.array([-1, 3, 1, 2, -6, 1]) * 1e-4
    spot = np.array([100.0, 101, 102, 103, 104, 105])
    perp = np.array([100.1, 101.0, 102.3, 103.1, 103.9, 105.2])
    mark = perp + 0.05
    t = np.arange(6) * 8 * 3_600 * 10**9 + carry._day0("2021-03-01")
    m = carry.Market(t, rate, spot, perp, mark)
    p = carry.follow(m, carry.Rule(2, 0.0), int(t[0]), int(t[-1]) + 1, 7.5, 4.5)
    # means: -, 1, 2, 1.5, -2, -2.5 bp: in at 1, out at 4
    assert p.held.tolist() == [False, True, True, True, False, False] and p.entries == 1
    # each period against the notional at its start, q x the spot price then
    assert p.fees[1] == pytest.approx((7.5e-4 * 101 + 4.5e-4 * 101.0) / 101)
    assert p.fees[4] == pytest.approx((7.5e-4 * 104 + 4.5e-4 * 103.9) / 104)
    assert p.funding.tolist() == pytest.approx([0, 0, 1e-4 * mark[2] / 101, 2e-4 * mark[3] / 102,
                                                -6e-4 * mark[4] / 103, 0])  # fmt: skip
    basis = perp - spot
    assert p.basis.tolist() == pytest.approx([0, 0, (basis[1] - basis[2]) / 101,
                                              (basis[2] - basis[3]) / 102,
                                              (basis[3] - basis[4]) / 103, 0])  # fmt: skip
    d = carry.daily(m, p, "2021-03-01", "2021-03-03")
    assert d["date"].to_list() == ["2021-03-01", "2021-03-02", "2021-03-03"]
    net = (p.funding + p.basis - p.fees) * 1e4
    assert d["net"].to_list() == pytest.approx([net[:3].sum(), net[3:].sum(), 0.0])
    # held to the end of a period: closed at its last settlement
    q = carry.follow(m, carry.Rule(2, 0.0), int(t[0]), int(t[3]) + 1, 7.5, 4.5)
    assert q.held.tolist() == [False, True, True, False] and q.fees[3] > 0
    s = carry.score(d, p, 1)
    assert s.entries == 1 and s.held == pytest.approx(3 / 5)
    assert s.annual_pct == pytest.approx(d["net"].mean() * 365 / 100)
    cum = np.cumsum(d["net"].to_numpy())
    assert s.max_drawdown_bp == pytest.approx(max(0, (np.maximum.accumulate(cum) - cum).max(),
                                                  -cum.min()))  # fmt: skip


def test_a_year_held_while_the_price_triples_earns_its_rate():
    # one bp a settlement for a year, the price going from 100 to 300: each period's return
    # is against the notional then, so the year earns its 1095 bp, not three times as much
    n = 3 * 365 + 1
    t = np.arange(n) * 8 * 3_600 * 10**9 + carry._day0("2021-01-01")
    spot = np.linspace(100.0, 300.0, n)
    m = carry.Market(t, np.full(n, 1e-4), spot, spot.copy(), spot.copy())
    p = carry.follow(m, carry.Rule(1, 0.0), int(t[0]), int(t[-1]) + 1, 0.0, 0.0)
    assert p.entries == 1
    # paid on the notional at the settlement, eight hours of price rise later: a hair over
    assert p.funding.sum() * 1e4 == pytest.approx((spot[1:] / spot[:-1]).sum(), rel=1e-9)
    assert p.funding.sum() * 1e4 == pytest.approx(1095, rel=2e-3)


def test_no_decision_reads_after_its_time():
    rng = np.random.default_rng(4)
    n = 300
    t = np.arange(n) * 8 * 3_600 * 10**9 + carry._day0("2021-01-01")
    base = carry.Market(t, rng.normal(0.5, 1, n) / 1e4, 100 + rng.normal(0, 1, n).cumsum(),
                        np.zeros(n), np.zeros(n))  # fmt: skip
    base.perp = base.spot * (1 + rng.normal(0, 3e-4, n))
    base.mark = base.perp.copy()
    cut = 200
    later = carry.Market(t, base.rate.copy(), base.spot.copy(), base.perp.copy(), base.mark.copy())
    later.rate[cut:] = rng.normal(-3, 1, n - cut) / 1e4
    later.spot[cut:] *= 1.1
    for rule in (carry.Rule(3, 0.0), carry.Rule(21, 0.5)):
        a = carry.follow(base, rule, int(t[0]), int(t[-1]) + 1, 7.5, 4.5)
        b = carry.follow(later, rule, int(t[0]), int(t[-1]) + 1, 7.5, 4.5)
        assert np.array_equal(a.held[:cut], b.held[:cut])
        assert np.allclose(a.fees[:cut], b.fees[:cut])
        assert np.allclose(a.funding[:cut], b.funding[:cut])
        assert a.entries > 3


def test_a_planted_carry_is_found_and_none_where_there_is_none(tmp_path, capsys):
    planted(tmp_path / "paid", 1.0, seed=1)
    planted(tmp_path / "none", 0.0, seed=2)
    spec = carry.load_spec(spec_file(tmp_path))
    r = carry.run(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    top = r.table[0]
    assert top["rule"] == "mean of 21 above 0 bp" and top["entries"] <= 3
    assert top["funding_bp_year"] == pytest.approx(3 * 365, rel=0.1)  # 1 bp, three a day
    assert abs(top["basis_bp_year"]) < 30 and top["fees_bp_year"] < 0
    assert r.always["held"] > 0.99
    assert r.settlements == 365 * 3
    from marketlab.cli import print_carry_report

    print_carry_report(r)
    out = capsys.readouterr().out
    assert "always held" in out and "verdict     passed" in out
    rec = carry.final(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    assert rec["confirmed"] and rec["summary"]["days"] == 181
    with pytest.raises(ex.ExperimentError, match="opened already"):
        carry.final(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    table = ex.registry_table(tmp_path / "results", tmp_path / "experiments")
    assert table["held_out"].to_list() == ["confirmed"]

    r = carry.run(spec, tmp_path / "none", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed" and r.table[0]["mean_bp"] < 0.5
    assert r.table[0]["entries"] > 5  # in and out as the mean crosses zero, paying each time


def test_bad_specifications_are_refused(tmp_path):
    bad = {
        "not a `carry`": SPEC.replace("kind: carry", "kind: edge"),
        "overlaps": SPEC.replace("from: 2022-01-01", "from: 2021-12-01"),
        "at least one settlement": SPEC.replace("lookback: 3}", "lookback: 0}"),
        "costs must not": SPEC.replace("spot_bp: 7.5", "spot_bp: -1"),
        "`costs` is missing": SPEC.replace("costs: {spot_bp: 7.5, perp_bp: 4.5}\n", ""),
    }
    for message, text in bad.items():
        with pytest.raises(ex.ExperimentError, match=message):
            carry.load_spec(spec_file(tmp_path, text))
    spec = carry.load_spec(spec_file(tmp_path))
    assert [v.lookback for v in spec.variants] == [21, 3, 63, 21]
    assert spec.variants[3].threshold_bp == 1.0
    planted(tmp_path / "early", 1.0, seed=3, first="2019-01-01", last="2019-03-01")
    assert carry.run(spec, tmp_path / "early", tmp_path / "r", prov=PROV).verdict == (
        "insufficient data"
    )


def test_end_to_end_through_the_command(tmp_path, capsys):
    from test_experiment import git

    from marketlab.cli import main

    planted(tmp_path / "data", 1.0, seed=5)
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "carry")
    args = ["experiment", "run", str(p), "--data", str(tmp_path / "data")]
    assert main(args) == 0
    assert "verdict     passed" in capsys.readouterr().out
    assert main([*args, "--final"]) == 0
    assert ": confirmed" in capsys.readouterr().out
    assert main(["experiment", "list", "--results", str(tmp_path / "results"),
                 "--experiments", str(tmp_path / "experiments")]) == 0  # fmt: skip
    assert "confirmed" in capsys.readouterr().out


def test_the_committed_hypothesis_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = carry.load_spec(root / "008-funding-carry.yaml")
    assert spec.primary == carry.Rule(21, 0.0) and len(spec.variants) == 12
    assert (spec.first, spec.last) == ("2020-01-01", "2023-12-31")
    assert (spec.holdout_first, spec.holdout_last) == ("2024-01-01", "2026-10-31")
    assert (spec.spot_bp, spec.perp_bp) == (7.5, 4.5)
    from marketlab.cli import kind_of

    assert kind_of(root / "008-funding-carry.yaml") == "carry"


def test_monthly_by_hand():
    d = pl.DataFrame({"date": ["2021-01-30", "2021-01-31", "2021-02-01"],
                      "net": [1.0, -3.0, 2.0]})  # fmt: skip
    m = carry.monthly(d)
    assert m.to_dicts() == [{"month": "2021-01", "net": -2.0, "mean": -1.0},
                            {"month": "2021-02", "net": 2.0, "mean": 2.0}]  # fmt: skip
