"""The shadow: a decision is what the held-out run would hold the next day, made from the
data up to its close and nothing later; recorded once; refused before the close is over or
while its bars are missing."""

import datetime as dt
import shutil

import numpy as np
import pytest
from test_xsection import write
from test_xsection_kind import DAYS, SPEC, planted, spec_file

from marketlab import shadow
from marketlab import xsection as xs
from marketlab import xsection_kind as xk
from marketlab.cli import main

CLOSE = "2024-03-10"


def cut(src, dst, last: str):
    """A copy of the data with nothing after `last`: as it was at that close."""
    shutil.copytree(src, dst)
    end = (dt.date.fromisoformat(last) - dt.date(1970, 1, 1)).days * 86_400_000 + 86_400_000
    for f in list((dst / "um" / "klines1d").glob("*.csv")) + list(
        (dst / "um" / "fundingRate").glob("*/*.csv")
    ):
        lines = f.read_text().splitlines()
        f.write_text("\n".join([lines[0], *(x for x in lines[1:] if int(x.split(",")[0]) < end)])
                     + "\n")  # fmt: skip


def test_a_decision_is_what_the_held_out_run_holds_and_reads_nothing_later(tmp_path):
    planted(tmp_path / "all", True, seed=1)
    spec = xk.load_spec(spec_file(tmp_path))
    d = shadow.decide(tmp_path / "all", spec, CLOSE, today="2026-11-23")
    longs = {s: w for s, w in d.weights.items() if w > 0}
    shorts = {s: w for s, w in d.weights.items() if w < 0}
    assert len(longs) == len(shorts) == 4 and d.universe == 20
    assert sum(longs.values()) == pytest.approx(0.5) and sum(shorts.values()) == pytest.approx(-0.5)
    assert d.trade_at == "2024-03-11T00:00+00:00"
    # the held-out run, from its first day, holds the same over the day after the close
    p = xs.load(tmp_path / "all", spec.holdout_first, "2024-03-20", held_out=True)
    h = p.dates.index(spec.holdout_first)
    held, _, _, _ = xs.holdings(xs.signals(p)["carry"], xs.universe(p, 20), h, 7, 0.2)
    i = p.dates.index(CLOSE)
    want = {p.symbols[j]: held[i + 1, j] for j in np.flatnonzero(held[i + 1])}
    assert d.weights == pytest.approx(want)
    for k in range(1, 7):  # the days between its rebalances too
        day = (dt.date.fromisoformat(CLOSE) + dt.timedelta(days=k)).isoformat()
        dk = shadow.decide(tmp_path / "all", spec, day, today="2026-11-23")
        j = p.dates.index(day)
        assert dk.weights == pytest.approx(
            {p.symbols[c]: held[j + 1, c] for c in np.flatnonzero(held[j + 1])})  # fmt: skip
    # with nothing after the close on disk, the same decision
    cut(tmp_path / "all", tmp_path / "then", CLOSE)
    assert shadow.decide(tmp_path / "then", spec, CLOSE, today="2026-11-23") == d


def test_recorded_once_and_refused_before_its_time(tmp_path, capsys):
    planted(tmp_path / "d", True, seed=2)
    # BTCUSDT and ETHUSDT, whose bars end a day before the planted coins'
    for sym in ("BTCUSDT", "ETHUSDT"):
        write(tmp_path / "d", sym, "2021-11-01", np.ones(DAYS - 1), np.full(DAYS - 1, 1.0))
    spec = xk.load_spec(spec_file(tmp_path))
    out = tmp_path / "shadow"
    d, path = shadow.record(tmp_path / "d", spec, CLOSE, out, today="2026-11-23")
    rows = path.read_text().splitlines()
    assert rows[0] == ",".join(shadow.COLUMNS) and len(rows) == 1 + len(d.weights)
    assert all(r.startswith(f"{CLOSE},") for r in rows[1:])
    again, _ = shadow.record(tmp_path / "d", spec, CLOSE, out, today="2026-11-23")
    assert again is None and path.read_text().splitlines() == rows
    shadow.record(tmp_path / "d", spec, "2024-03-11", out, today="2026-11-23")
    assert len(path.read_text().splitlines()) > len(rows)
    with pytest.raises(shadow.NotReady, match="not over yet"):
        shadow.decide(tmp_path / "d", spec, "2024-06-30", today="2024-06-30")
    with pytest.raises(shadow.NotReady, match="none for BTCUSDT, ETHUSDT"):
        shadow.decide(tmp_path / "d", spec, "2024-06-30", today="2024-07-01")
    with pytest.raises(shadow.NotReady, match="not all there"):  # no bar at all that day
        shadow.decide(tmp_path / "d", spec, "2024-07-02", today="2024-07-05")
    # the command
    args = ["shadow", str(spec.path), "--data", str(tmp_path / "d"), "--out", str(out)]
    assert main([*args, "--close", "2024-03-12"]) == 0
    assert "4 long and 4 short of a universe of 20" in capsys.readouterr().out
    assert main([*args, "--close", "2024-03-12"]) == 0
    assert "recorded already" in capsys.readouterr().out
    assert main([*args, "--close", "2024-06-30"]) == 2
    assert "none for BTCUSDT" in capsys.readouterr().err


def test_a_staggered_rule_decides_the_same_whatever_its_period_began_on(tmp_path):
    planted(tmp_path / "d", True, seed=3)
    mom = SPEC.replace("signal: carry, hold: 7", "signal: mom7, hold: 7")  # a signal a day
    text = mom.replace("q: 0.2}", "q: 0.2, staggered: true}")
    a = shadow.decide(tmp_path / "d", xk.load_spec(spec_file(tmp_path, text)), CLOSE,
                      today="2026-11-23")  # fmt: skip
    later = text.replace("from: 2024-01-01", "from: 2024-01-04")
    b = shadow.decide(tmp_path / "d", xk.load_spec(spec_file(tmp_path, later, "033-b.yaml")),
                      CLOSE, today="2026-11-23")  # fmt: skip
    assert a.weights == pytest.approx(b.weights) and len(a.weights) > 8
    plain = mom.replace("from: 2024-01-01", "from: 2024-01-04")
    c = shadow.decide(tmp_path / "d", xk.load_spec(spec_file(tmp_path, plain, "034-c.yaml")),
                      CLOSE, today="2026-11-23")  # fmt: skip
    d = shadow.decide(tmp_path / "d", xk.load_spec(spec_file(tmp_path, mom, "035-d.yaml")), CLOSE,
                      today="2026-11-23")  # fmt: skip
    assert c.weights != pytest.approx(d.weights)  # one portfolio: its start day decides


def test_too_many_bars_missing(tmp_path):
    planted(tmp_path / "d", True, seed=4)
    for j in range(3):  # three coins of 25 without their last bar
        f = tmp_path / "d" / "um" / "klines1d" / f"C{j:02d}USDT.csv"
        f.write_text("\n".join(f.read_text().splitlines()[:-1]) + "\n")
    spec = xk.load_spec(spec_file(tmp_path))
    with pytest.raises(shadow.NotReady, match="12% of the perpetuals with one the day before"):
        shadow.decide(tmp_path / "d", spec, "2024-06-30", today="2024-07-01")
    assert shadow.decide(tmp_path / "d", spec, "2024-06-29", today="2024-07-01").weights


def test_the_universe_fetch_refuses_a_day_not_closed(tmp_path):
    from test_fetch import fu

    today = dt.datetime.now(dt.UTC).date().isoformat()
    with pytest.raises(SystemExit, match="is not closed yet"):
        fu.main(["--start", "2026-01-01", "--end", today, "--out", str(tmp_path), "--tries",
                 "1", "--base-url", "http://127.0.0.1:9"])  # fmt: skip
