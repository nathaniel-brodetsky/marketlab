"""A trend on named coins: its direction and size worked out by hand, a coin without what it
needs holding nothing; as a hypothesis, the specification checked, the held-out run reading
the days its lookbacks need before its first, the shadow deciding what that run holds, and
the committed 016 and 017."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_tradeflow import PROV
from test_xsection import write
from test_xsection_kind import SPEC, planted, spec_file

from marketlab import experiment as ex
from marketlab import shadow
from marketlab import trend as tr
from marketlab import xsection as xs
from marketlab import xsection_kind as xk

TREND = (
    SPEC.replace(
        "rule: {signal: carry, hold: 7, top: 20, q: 0.2}",
        "rule: {signal: trend, hold: 1, top: 20, coins: [C00USDT, C01USDT], lookbacks: [20, 60],"
        " size: coin, target: 0.3, window: 10}",
    )
    .replace("variants:\n  - {hold: 1}", "variants:\n  - {coins: [C00USDT]}")
    .replace("trials:\n  - {hold: 1}\n  - {hold: 3}", "trials:\n  - {size: whole}")
)


def two(tmp_path, days=40, step=0.02):
    """Two coins from 2022-01-01: one up `step` a day in log, one down as much."""
    t = np.arange(days)
    write(tmp_path, "AAAUSDT", "2022-01-01", 100 * np.exp(step * t), np.full(days, 1e6))
    write(tmp_path, "BBBUSDT", "2022-01-01", 100 * np.exp(-step * t), np.full(days, 1e6))
    return xs.load(tmp_path, "2022-01-01", "2022-02-09")


def test_the_direction_and_size_by_hand(tmp_path):
    p = two(tmp_path)
    a, b = p.symbols.index("AAAUSDT"), p.symbols.index("BBBUSDT")
    start = p.dates.index("2022-01-01")
    held, traded = tr.holdings(p, ("AAAUSDT", "BBBUSDT"), (2, 4), "whole", 0.2, 3, start)
    d = p.dates.index("2022-01-03")  # 2 days of history: one lookback of two has a sign
    assert held[d + 1, a] == pytest.approx(0.25) and held[d + 1, b] == pytest.approx(-0.25)
    d = p.dates.index("2022-01-05")  # both: the whole half each
    assert held[d + 1, a] == pytest.approx(0.5) and held[d + 1, b] == pytest.approx(-0.5)
    assert traded[d] == pytest.approx(0.5) and traded[d + 1] == pytest.approx(0.0)
    assert not held[: p.dates.index("2022-01-03") + 1].any()  # nothing before a sign
    # each coin to a volatility: 2% a day is 0.02 * sqrt(365) = 38.2% a year; to 20%, scaled
    # by 0.2 / 0.382; to 50%, not above its half
    sized, _ = tr.holdings(p, ("AAAUSDT", "BBBUSDT"), (2, 4), "coin", 0.2, 3, start)
    assert sized[d + 1, a] == pytest.approx(0.5 * 0.2 / (0.02 * np.sqrt(365)))
    assert sized[d + 1, b] == pytest.approx(-0.5 * 0.2 / (0.02 * np.sqrt(365)))
    capped, _ = tr.holdings(p, ("AAAUSDT", "BBBUSDT"), (2, 4), "coin", 0.5, 3, start)
    assert capped[d + 1, a] == pytest.approx(0.5)
    vol = tr.volatility(p, 3)
    assert vol[p.dates.index("2022-01-04"), a] == pytest.approx(0.02 * np.sqrt(365))
    assert np.isnan(vol[p.dates.index("2022-01-03"), a])  # 3 returns not there yet
    with pytest.raises(ValueError, match="no daily bars for CCCUSDT"):
        tr.holdings(p, ("AAAUSDT", "CCCUSDT"), (2,), "whole", 0.2, 3, start)
    with pytest.raises(ValueError, match="size is one of"):
        tr.decide(p, [a], (2,), "book", 0.2, vol, d)
    # bars from the panel's first row: a lookback reaching before it has no sign
    q = xs.Panel(p.dates[40:], p.symbols, p.close[40:], p.qvol[40:], p.funding[40:])
    h, _ = tr.holdings(q, ("AAAUSDT", "BBBUSDT"), (2, 4), "whole", 0.2, 3, 0)
    assert not h[:3].any() and h[3, a] == pytest.approx(0.25) and h[5, a] == pytest.approx(0.5)


def test_a_coin_without_a_bar_at_the_close_holds_nothing(tmp_path):
    t = np.arange(40)
    close = 100 * np.exp(0.02 * t)
    close[20] = np.nan
    write(tmp_path, "AAAUSDT", "2022-01-01", close, np.full(40, 1e6))
    p = xs.load(tmp_path, "2022-01-01", "2022-02-09")
    held, _ = tr.holdings(p, ("AAAUSDT",), (2, 4), "coin", 0.2, 3, 0)
    d = p.dates.index("2022-01-01") + 20
    assert held[d + 1, 0] == 0.0 and held[d, 0] > 0
    # the day after, its 2-day lookback reads the missing day: one sign of two; and its
    # volatility, missing a return, holds nothing until 3 are there again
    assert held[d + 2, 0] == 0.0 and held[d + 5, 0] > 0


def test_a_trend_as_a_hypothesis(tmp_path):
    planted(tmp_path / "d", True, seed=1)
    spec = xk.load_spec(spec_file(tmp_path, TREND))
    assert spec.primary == xk.Rule("trend", 1, 20, 0.2, 0, False, None, None, (), "picked",
                                   ("C00USDT", "C01USDT"), (20, 60), "coin", 0.3, 10)  # fmt: skip
    assert spec.primary.name == ("trend on C00, C01, the sign of the last 20 and 60 days, each "
                                 "coin to 30% volatility over 10 days")  # fmt: skip
    assert spec.variants[1].coins == ("C00USDT",) and spec.variants[1].lookbacks == (20, 60)
    assert spec.trials[1].size == "whole" and spec.trials[1].coins == ("C00USDT", "C01USDT")
    assert xk.history([spec.primary]) == 60 + 10 + 10
    assert xk.history([xk.Rule("carry", 7, 50, 0.2)]) == 0
    for old, new, why in (
        ("coins: [C00USDT, C01USDT]", "coins: []", "coins are a list of symbols"),
        ("coins: [C00USDT, C01USDT]", "coins: [C00USDT, C00USDT]", "coins are a list"),
        ("lookbacks: [20, 60]", "lookbacks: [0, 60]", "lookbacks are different whole days"),
        ("lookbacks: [20, 60]", "lookbacks: [20, 20]", "lookbacks are different whole days"),
        ("size: coin", "size: book", "size is one of"),
        ("target: 0.3", "target: 0", "target is above 0"),
        ("window: 10", "window: 1", "window 2 days or more"),
        ("hold: 1, top: 20", "hold: 7, top: 20", "a trend is held 1 day"),
        ("hold: 1, top: 20", "hold: 1, stop: 1, top: 20", "a trend is held 1 day"),
        ("signal: trend", "signal: carry", "for signal trend only"),
    ):
        with pytest.raises(ex.ExperimentError, match=why):
            xk.load_spec(spec_file(tmp_path, TREND.replace(old, new)))
    # followed: its weights at the day's returns, less the cost of every unit traded
    p = xs.load(tmp_path / "d", spec.first, spec.last)
    start = p.dates.index(spec.first)
    got = xk.follow(p, spec.primary, spec.first, 9.0, {})
    held, traded = tr.holdings(p, spec.primary.coins, (20, 60), "coin", 0.3, 10, start)
    want = (held * np.nan_to_num(xs.daily_returns(p))).sum(axis=1) - traded * 9.0 / 2 / 1e4
    assert np.allclose(got.net, want[start:], rtol=0, atol=1e-15)
    assert got.names == pytest.approx((held[start:] != 0).sum(axis=1).mean())
    # run, then opened once: the held-out run reads the days before its first its lookbacks
    # need, so it holds what a run on all the data holds
    report = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert report.table and report.table[0]["rule"] == spec.primary.name
    xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    rec = json.loads((tmp_path / "results" / f"{spec.slug}.holdout.json").read_text())
    every = xs.load(tmp_path / "d", "2021-11-01", spec.holdout_last, held_out=True)
    whole = xk.follow(every, spec.primary, spec.holdout_first, 9.0, {}, held_out=True)
    assert rec["summary"]["mean_bp"] == pytest.approx(whole.net.mean() * 1e4, abs=1e-9)
    # the shadow decides, at a held-out close, what that run holds the day after
    close = "2024-03-10"
    d = shadow.decide(tmp_path / "d", spec, close, today="2026-11-23")
    i = every.dates.index(close)
    h, _ = tr.holdings(every, spec.primary.coins, (20, 60), "coin", 0.3, 10,
                       every.dates.index(spec.holdout_first))  # fmt: skip
    assert d.weights == pytest.approx({every.symbols[j]: h[i + 1, j]
                                       for j in np.flatnonzero(h[i + 1])})  # fmt: skip
    assert set(d.weights) == {"C00USDT", "C01USDT"} and d.trade_at == "2024-03-11T00:00+00:00"


def test_the_committed_trends_load():
    root = Path(__file__).resolve().parents[1] / "experiments"
    a = xk.load_spec(root / "016-trend-btc-eth.yaml")
    b = xk.load_spec(root / "017-trend-btc-eth-forward.yaml")
    rule = xk.Rule("trend", 1, 50, 0.2, 1, False, None, None, (), "picked",
                   ("BTCUSDT", "ETHUSDT"), (20, 60, 120), "coin", 0.2, 30)  # fmt: skip
    assert a.primary == b.primary == rule and a.trials == b.trials and a.variants == b.variants
    assert (a.holdout_first, a.holdout_last) == ("2024-01-01", "2026-10-31")
    assert (b.holdout_first, b.holdout_last) == ("2026-12-01", "2028-11-30")
    assert len(a.trials) == 12 and sum(t.hours == 0 for t in a.trials) == 8
    assert [v.coins for v in a.variants] == [("BTCUSDT", "ETHUSDT"), ("BTCUSDT",)]
    assert a.max_abs_beta == 1.0 and a.min_positive_years == 2 and a.min_dsr is None
