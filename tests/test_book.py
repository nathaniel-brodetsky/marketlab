"""A book of rules (`signal: book` in an `xsection` hypothesis): its parts' days weighted by
their shares of the capital, in a run, in the held-out run and in the shadow."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_tradeflow import PROV
from test_xsection import zigzag
from test_xsection_kind import DAYS, SPEC, planted, spec_file

from marketlab import experiment as ex
from marketlab import shadow
from marketlab import xsection as xs
from marketlab import xsection_kind as xk

PARTS = """rule:
  signal: book
  parts:
    - {weight: 1, signal: carry, hold: 7, top: 20, q: 0.2}
    - {weight: 3, signal: trend, coins: [C00USDT, C01USDT], lookbacks: [20, 60], size: coin,
       target: 0.3, window: 10, hold: 1, top: 20}
"""
BOOK = (
    SPEC.replace("rule: {signal: carry, hold: 7, top: 20, q: 0.2}\n", PARTS)
    .replace("variants:\n  - {hold: 1}",
             "variants:\n  - parts:\n      - {weight: 1, signal: carry, hold: 1, top: 20, q: 0.2}"
             "\n      - {weight: 1, signal: mom7, hold: 1, top: 20, q: 0.2}")
    .replace("trials:\n  - {hold: 1}\n  - {hold: 3}", "trials: []")
)  # fmt: skip
CARRY = xk.Rule("carry", 7, 20, 0.2)
TREND = xk.Rule("trend", 1, 20, 0.2, 0, False, None, None, (), "picked", ("C00USDT", "C01USDT"),
                (20, 60), "coin", 0.3, 10)  # fmt: skip


def test_a_book_of_rules(tmp_path):
    planted(tmp_path / "d", True, seed=1)
    spec = xk.load_spec(spec_file(tmp_path, BOOK))
    # its weights its shares of the capital: divided by their sum
    assert spec.primary.signal == "book" and spec.primary.parts == ((CARRY, 0.25), (TREND, 0.75))
    assert spec.primary.name == (
        "a book: 25% in carry, every 7 days, top 20, 20% a side; 75% in trend on C00, C01, the "
        "sign of the last 20 and 60 days, each coin to 30% volatility over 10 days")  # fmt: skip
    assert [w for _, w in spec.variants[1].parts] == [0.5, 0.5] and spec.trials == (spec.primary,)
    assert xk.history([spec.primary]) == 60 + 10 + 10 == xk.history([CARRY, TREND])
    # followed: each part on its own share, the book's day the parts' weighted
    p = xs.load(tmp_path / "d", spec.first, spec.last)
    got = xk.follow(p, spec.primary, spec.first, 9.0, {})
    a = xk.follow(p, CARRY, spec.first, 9.0, {})
    b = xk.follow(p, TREND, spec.first, 9.0, {})
    for k in ("net", "gross", "market"):
        assert np.allclose(getattr(got, k), 0.25 * getattr(a, k) + 0.75 * getattr(b, k),
                           rtol=0, atol=1e-15)  # fmt: skip
    assert got.dates == a.dates and got.names == pytest.approx(a.names + b.names)
    assert np.any(a.net != 0) and np.any(b.net != 0)
    # run, then opened once: the held-out run reads the days its trend part's lookbacks need,
    # so it holds what a run on all the data holds
    report = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert [r["rule"] for r in report.table] == [v.name for v in spec.variants]
    rec = xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    every = xs.load(tmp_path / "d", "2021-11-01", spec.holdout_last, held_out=True)
    whole = xk.follow(every, spec.primary, spec.holdout_first, 9.0, {}, held_out=True)
    assert rec["summary"]["mean_bp"] == pytest.approx(whole.net.mean() * 1e4, abs=1e-9)
    assert rec["rule"] == spec.primary.name
    # the shadow: each part's decision at its share, a coin in both netted
    close = "2024-03-10"
    d = shadow.decide(tmp_path / "d", spec, close, today="2026-11-23")
    one = shadow.decide(tmp_path / "d", replace(spec, primary=CARRY), close, today="2026-11-23")
    two = shadow.decide(tmp_path / "d", replace(spec, primary=TREND), close, today="2026-11-23")
    want: dict[str, float] = {}
    for part, w in ((one, 0.25), (two, 0.75)):
        for sym, x in part.weights.items():
            want[sym] = want.get(sym, 0.0) + w * x
    assert one.weights and two.weights and set(two.weights) == {"C00USDT", "C01USDT"}
    assert d.weights == pytest.approx({s: x for s, x in want.items() if x})
    assert (d.trade_at, d.universe) == (one.trade_at, one.universe)


def test_bad_books_are_refused(tmp_path):
    two = ("    - {weight: 1, signal: carry, hold: 7, top: 20, q: 0.2}\n"
           "    - {weight: 3, signal: trend")  # fmt: skip
    for old, new, why in (
        (PARTS, "rule: {signal: book}\n", "a book's parts are a list of two or more rules"),
        (two, "    - {weight: 3, signal: trend", "a book's parts are a list of two or more"),
        ("weight: 3", "weight: 0", "a part's weight is above 0, not 0"),
        ("weight: 3", "weight: true", "a part's weight is above 0, not True"),
        ("weight: 3", "weight: -1", "a part's weight is above 0"),
        ("  signal: book\n", "  signal: book\n  hold: 7\n", "a book is its parts and nothing "
         "else, not hold"),
        ("q: 0.2}\n    - {weight: 3", "q: 0.2, hours: 1}\n    - {weight: 3",
         "a book's parts trade at the same hour"),
        ("    - {weight: 1, signal: carry, hold: 7, top: 20, q: 0.2}\n",
         "    - {weight: 1, signal: carry, hold: 7, top: 20, q: 0.2, split: [1, 2]}\n",
         "none split among hours"),
        ("    - {weight: 3, signal: trend, coins: [C00USDT, C01USDT], lookbacks: [20, 60], "
         "size: coin,\n       target: 0.3, window: 10, hold: 1, top: 20}\n",
         "    - {weight: 3, signal: carry, hold: 7, top: 20, q: 0.2}\n",
         "a book's parts are different rules"),
        ("    - {weight: 1, signal: carry, hold: 7, top: 20, q: 0.2}\n",
         "    - {weight: 1, signal: book, parts: [{weight: 1, signal: carry, hold: 7, top: 20},"
         " {weight: 1, signal: carry, hold: 1, top: 20}]}\n",
         "a book's part is not a book"),
        ("coins: [C00USDT, C01USDT]", "coins: []", "a trend's coins are a list of symbols"),
        ("variants:\n  - parts:", "variants:\n  - {hold: 1}\n  - parts:",
         "a book is its parts and nothing else, not hold"),
    ):  # fmt: skip
        assert BOOK.count(old) == 1, old
        with pytest.raises(ex.ExperimentError, match=why):
            xk.load_spec(spec_file(tmp_path, BOOK.replace(old, new)))
    with pytest.raises(ex.ExperimentError, match="parts: for signal book only"):
        xk.load_spec(spec_file(tmp_path, SPEC.replace(
            "rule: {signal: carry, hold: 7, top: 20, q: 0.2}",
            "rule: {signal: carry, hold: 7, top: 20, q: 0.2, parts: []}")))  # fmt: skip


def test_a_book_s_held_out_run_checks_every_part_s_hour_bars(tmp_path):
    # twelve coins whose volume rises with their number: C00 is never among the top ten the
    # reversal part trades, only in the trend part, four hours after the close
    zigzag(tmp_path / "d", days=DAYS, first="2021-11-01")
    text = SPEC.replace("rule: {signal: carry, hold: 7, top: 20, q: 0.2}\n", (
        "rule:\n  signal: book\n  parts:\n"
        "    - {weight: 1, signal: rev1, hold: 1, top: 10, q: 0.2, hours: 4}\n"
        "    - {weight: 1, signal: trend, coins: [C00USDT], lookbacks: [20], window: 10, "
        "hold: 1, top: 10, hours: 4}\n")).replace("variants:\n  - {hold: 1}\n", "").replace(
        "trials:\n  - {hold: 1}\n  - {hold: 3}", "trials: []")  # fmt: skip
    spec = xk.load_spec(spec_file(tmp_path, text))
    xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    folder = tmp_path / "d" / "um" / "klines" / "C00USDT" / "1h"
    year = {f: f.read_bytes() for f in folder.glob("C00USDT-1h-2024-*.zip")}
    assert year
    for f in year:
        f.unlink()
    with pytest.raises(ex.ExperimentError, match="for C00USDT; nothing was computed or recorded"):
        xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert not (tmp_path / "results" / f"{spec.slug}.holdout.json").exists()
    for f, data in year.items():
        f.write_bytes(data)
    rec = xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert rec["rule"].startswith("a book: 50% in rev1, every 1 days, top 10")
    assert json.loads((tmp_path / "results" / f"{spec.slug}.holdout.json").read_text())


def test_the_committed_book_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "020-book-crowd-trend.yaml")
    crowd = xk.load_spec(root / "019-xsection-crowd-forward.yaml").primary
    trend = xk.load_spec(root / "017-trend-btc-eth-forward.yaml").primary
    eleven = xk.load_spec(root / "011-xsection-carry-forward.yaml")
    assert spec.primary.parts == ((crowd, 0.39), (trend, 0.61))
    assert [tuple(r for r, _ in v.parts) for v in spec.variants] == [
        (crowd, trend), (crowd, eleven.primary, trend)]  # fmt: skip
    assert len(spec.trials) == 4 and all(t.signal == "book" for t in spec.trials)
    assert [round(w, 2) for _, w in spec.trials[2].parts] == [0.27, 0.31, 0.42]
    assert [round(w, 2) for _, w in spec.trials[3].parts] == [0.24, 0.30, 0.46]
    assert (spec.first, spec.last) == ("2022-01-01", "2023-12-31")
    assert (spec.holdout_first, spec.holdout_last) == (eleven.holdout_first, eleven.holdout_last)
    assert xk.history([spec.primary]) == 120 + 30 + 10
    assert spec.min_dsr is None and spec.min_positive_years == 2 and spec.max_abs_beta == 1.0
