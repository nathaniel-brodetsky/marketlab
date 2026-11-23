"""Hypotheses of the `xsection` kind: a planted carry passes and is confirmed once, none
where there is none; the development run reads no held-out day; the specification checked;
the committed hypothesis 009; the whole of it through the command."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
from test_tradeflow import PROV
from test_xsection import write, zigzag

from marketlab import experiment as ex
from marketlab import xsection_kind as xk

SPEC = """
id: "031"
name: planted-xsection
kind: xsection
hypothesis: The planted funding pays.
data:
  days: {from: 2022-03-01, to: 2023-06-30}
  holdout: {from: 2024-01-01, to: 2024-06-30}
rule: {signal: carry, hold: 7, top: 20, q: 0.2}
variants:
  - {hold: 1}
trials:
  - {hold: 1}
  - {hold: 3}
costs: {round_trip_bp: 9.0}
protocol: {nw_lags: 10}
success: {alpha: 0.05, max_pbo: 0.5, min_positive_months: 0.5, min_dsr: 0.95,
          min_positive_years: 1, max_abs_beta: 0.3}
"""
DAYS = 973  # 2021-11-01 .. 2024-06-30


def spec_file(tmp_path, text=SPEC, name="031-planted-xsection.yaml"):
    p = tmp_path / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def planted(root, persist: bool, seed: int, n_sym=25):
    """Funding a day that lasts 60 days (`persist`) or changes every day, prices that walk:
    with lasting funding, the coins whose shorts pay keep paying their longs."""
    rng = np.random.default_rng(seed)
    for j in range(n_sym):
        if persist:
            f = np.repeat(rng.normal(0, 3e-3, DAYS // 60 + 1), 60)[:DAYS]
        else:
            f = rng.normal(0, 3e-3, DAYS)
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, DAYS)))
        write(root, f"C{j:02d}USDT", "2021-11-01", close, rng.uniform(1e6, 2e6, DAYS), f)


def test_a_planted_carry_passes_and_is_confirmed_once(tmp_path, capsys):
    planted(tmp_path / "paid", True, seed=1)
    spec = xk.load_spec(spec_file(tmp_path))
    r = xk.run(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    top = r.table[0]
    assert top["rule"] == "carry, every 7 days, top 20, 20% a side" and top["names_side"] == 4
    assert top["annual_pct"] > 30 and abs(top["beta"]) < 0.3 and r.dsr >= 0.95
    assert 0 < top["gross_pct"] - top["annual_pct"] < 10  # the costs, a few % a year
    assert [t["rule"] for t in r.trials] == [f"carry, every {h} days, top 20, 20% a side"
                                             for h in (7, 1, 3)]  # fmt: skip
    assert r.pbo is not None and r.perpetuals == 25
    assert list(top["by_year_pct"]) == ["2022", "2023"]
    from marketlab.cli import print_xsection_report

    print_xsection_report(r)
    out = capsys.readouterr().out
    assert "verdict     passed" in out and "dsr" in out and "trials      carry every 7 days" in out
    with pytest.raises(ex.ExperimentError, match="development days first"):
        xk.final(spec, tmp_path / "paid", tmp_path / "elsewhere", prov=PROV)
    rec = xk.final(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    assert rec["confirmed"], rec["reasons"]
    assert rec["summary"]["days"] == 182 and list(rec["summary"]["by_year_pct"]) == ["2024"]
    with pytest.raises(ex.ExperimentError, match="opened already"):
        xk.final(spec, tmp_path / "paid", tmp_path / "results", prov=PROV)
    table = ex.registry_table(tmp_path / "results", tmp_path / "experiments")
    assert table["held_out"].to_list() == ["confirmed"]
    # trials that differ more (a reversal that pays its costs every day, a momentum) make
    # the same Sharpe ratio worth less: the best of them by luck would reach further
    wide = SPEC.replace("  - {hold: 3}\n", "  - {hold: 3}\n  - {signal: rev1, hold: 1}\n"
                        "  - {signal: mom7, hold: 1}\n")  # fmt: skip
    w = xk.run(xk.load_spec(spec_file(tmp_path, wide, "032-wide.yaml")), tmp_path / "paid",
               tmp_path / "results-wide", prov=PROV)  # fmt: skip
    assert len(w.trials) == 5 and w.dsr < r.dsr and w.table == r.table
    # the held-out bars: a year up asked of a holdout of one, and no beta at all allowed
    strict = SPEC.replace("min_positive_years: 1, max_abs_beta: 0.3",
                          "min_positive_years: 2, max_abs_beta: 0.0001")  # fmt: skip
    st = xk.load_spec(spec_file(tmp_path, strict, "034-strict.yaml"))
    xk.run(st, tmp_path / "paid", tmp_path / "results-strict", prov=PROV)
    rec = xk.final(st, tmp_path / "paid", tmp_path / "results-strict", prov=PROV)
    assert not rec["confirmed"] and len(rec["reasons"]) == 2
    assert "up in 1 of its 1 years" in rec["reasons"][0] and "beta" in rec["reasons"][1]
    no_bar = SPEC.replace(" min_dsr: 0.95,", "")
    n = xk.load_spec(spec_file(tmp_path, no_bar, "033-no-bar.yaml"))
    assert n.min_dsr is None
    hard = SPEC.replace("min_dsr: 0.95", "min_dsr: 1.01")  # a bar no ratio clears
    h = xk.run(xk.load_spec(spec_file(tmp_path, hard, "035-hard.yaml")), tmp_path / "paid",
               tmp_path / "results-hard", prov=PROV)  # fmt: skip
    assert h.verdict == "failed" and h.reasons == [
        f"deflated Sharpe ratio {h.dsr:.2f} against 3 trials is below 1.01"]  # fmt: skip


def test_none_where_there_is_none(tmp_path):
    planted(tmp_path / "none", False, seed=2)
    spec = xk.load_spec(spec_file(tmp_path))
    r = xk.run(spec, tmp_path / "none", tmp_path / "results", prov=PROV)
    assert r.verdict == "failed" and r.table[0]["p"] > 0.05
    rec = xk.final(spec, tmp_path / "none", tmp_path / "results", prov=PROV)
    assert not rec["confirmed"] and "not above zero" in rec["reasons"][0]


def test_the_development_run_reads_no_held_out_day(tmp_path):
    planted(tmp_path / "a", True, seed=3)
    planted(tmp_path / "b", True, seed=3)
    # the same files, but every close from 2024-01-01 on ten times what it was
    cut = DAYS - 182
    for f in sorted((tmp_path / "b" / "um" / "klines1d").glob("*.csv")):
        lines = f.read_text().splitlines()
        rows = [x.split(",") for x in lines[1:]]
        for k, row in enumerate(rows[cut:]):
            row[4] = str(float(row[4]) * (10 if k % 2 else 0.1))
        f.write_text("\n".join([lines[0], *(",".join(x) for x in rows)]) + "\n")
    spec = xk.load_spec(spec_file(tmp_path))
    a = xk.run(spec, tmp_path / "a", tmp_path / "ra", prov=PROV)
    b = xk.run(spec, tmp_path / "b", tmp_path / "rb", prov=PROV)
    assert a.table == b.table and a.dsr == b.dsr
    fa = xk.final(spec, tmp_path / "a", tmp_path / "ra", prov=PROV)
    fb = xk.final(spec, tmp_path / "b", tmp_path / "rb", prov=PROV)
    assert fa["summary"] != fb["summary"]  # the held-out run does read them


def test_bad_specifications_are_refused(tmp_path):
    bad = {
        "not an `xsection`": SPEC.replace("kind: xsection", "kind: carry"),
        "unknown keys": SPEC + "extra: 1\n",
        "must end by 2023-12-31": SPEC.replace("to: 2023-06-30", "to: 2024-01-01"),
        "begin after it": SPEC.replace("from: 2024-01-01", "from: 2023-12-31"),
        "is none of": SPEC.replace("- {hold: 3}", "- {signal: mom8, hold: 3}"),
        "hold at least 1 day": SPEC.replace("- {hold: 1}\ntrials", "- {hold: 0}\ntrials"),
        "q between 0 and 0.5": SPEC.replace("q: 0.2", "q: 0.5"),
        "top at least 10": SPEC.replace("top: 20", "top: 5"),
        "hours from 0 to 23": SPEC.replace("q: 0.2}", "q: 0.2, hours: 24}"),
        "costs must not": SPEC.replace("round_trip_bp: 9.0", "round_trip_bp: -1"),
        "`costs` is missing": SPEC.replace("costs: {round_trip_bp: 9.0}\n", ""),
    }
    for message, text in bad.items():
        with pytest.raises(ex.ExperimentError, match=message):
            xk.load_spec(spec_file(tmp_path, text))
    spec = xk.load_spec(spec_file(tmp_path))
    assert [v.hold for v in spec.variants] == [7, 1]
    assert [(t.signal, t.hold) for t in spec.trials] == [("carry", 7), ("carry", 1), ("carry", 3)]
    assert all(t.top == 20 and t.q == 0.2 for t in spec.trials)
    r = xk.run(spec, tmp_path / "nothing", tmp_path / "r", prov=PROV)
    assert r.verdict == "insufficient data"


def test_the_committed_hypothesis_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "009-xsection-carry.yaml")
    assert spec.primary == xk.Rule("carry", 7, 50, 0.2)
    assert [v.hold for v in spec.variants] == [7, 1] and len(spec.trials) == 7
    assert (spec.first, spec.last) == ("2020-01-01", "2023-12-31")
    assert (spec.holdout_first, spec.holdout_last) == ("2024-01-01", "2026-10-31")
    assert spec.round_trip_bp == 9.0 and spec.min_positive_years == 2
    from marketlab.cli import kind_of

    assert kind_of(root / "009-xsection-carry.yaml") == "xsection"


def test_end_to_end_through_the_command(tmp_path, capsys):
    from test_experiment import git

    from marketlab.cli import main

    planted(tmp_path / "data", True, seed=5)
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path)
    git(tmp_path, "add", "experiments")
    git(tmp_path, "commit", "-q", "-m", "xsection")
    args = ["experiment", "run", str(p), "--data", str(tmp_path / "data")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "verdict     passed" in out and "25 perpetuals on coins" in out
    assert main([*args, "--final"]) == 0
    out = capsys.readouterr().out
    assert "opened once, now recorded" in out and "by year, %: 2024 +" in out
    assert out.rstrip().endswith("confirmed")
    assert main([*args, "--final"]) == 2  # once only
    assert main(["experiment", "list", "--results", str(tmp_path / "results"),
                 "--experiments", str(tmp_path / "experiments")]) == 0  # fmt: skip
    assert "confirmed" in capsys.readouterr().out


def test_a_rule_traded_hours_after_the_close(tmp_path):
    # every day's move made in its first two hours: a reversal pays traded at the close and
    # loses traded four hours later; the held-out run checks its hour bars before anything
    zigzag(tmp_path / "d", days=DAYS, first="2021-11-01")
    late = SPEC.replace("rule: {signal: carry, hold: 7, top: 20, q: 0.2}",
                        "rule: {signal: rev1, hold: 1, top: 12, q: 0.2, hours: 4}").replace(
        "variants:\n  - {hold: 1}", "variants:\n  - {hours: 0}")  # fmt: skip
    spec = xk.load_spec(spec_file(tmp_path, late))
    assert spec.primary == xk.Rule("rev1", 1, 12, 0.2, 4) and spec.variants[1].hours == 0
    assert spec.primary.name.endswith(", traded 4 h after the close")
    one = tmp_path / "d" / "um" / "klines" / "C00USDT" / "1h" / "C00USDT-1h-2024-02.zip"
    good = one.read_bytes()
    one.write_text("not a zip")  # a held-out month: the development run never opens it
    r = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert r.table[0]["annual_pct"] < -1000 < 1000 < r.table[1]["annual_pct"]
    assert r.table[0]["priced"] > 0.99
    with pytest.raises(zipfile.BadZipFile):  # the held-out run reads it
        xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    one.write_bytes(good)
    year = {f: f.read_bytes() for f in one.parent.glob("C00USDT-1h-2024-*.zip")}
    for f in year:  # a coin of twelve without its held-out hour bars: 92% of the coin-days
        f.unlink()
    with pytest.raises(ex.ExperimentError, match="for C00USDT; nothing was computed or recorded"):
        xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert not (tmp_path / "results" / "031-planted-xsection.holdout.json").exists()
    for f, data in year.items():
        f.write_bytes(data)
    rec = xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert not rec["confirmed"] and rec["summary"]["annual_pct"] < -1000
    assert rec["rule"].endswith(", traded 4 h after the close")


def test_the_committed_momentum_hypothesis_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "010-xsection-momentum.yaml")
    assert spec.primary == xk.Rule("mom7", 1, 50, 0.2, 1)
    assert [(v.signal, v.hold, v.hours) for v in spec.variants] == [
        ("mom7", 1, 1), ("mom7", 7, 1), ("mom30", 1, 1)]  # fmt: skip
    assert len(spec.trials) == 7 and all(t.hours == 1 for t in spec.trials)
    assert (spec.holdout_first, spec.holdout_last) == ("2024-01-01", "2026-10-31")
    assert spec.min_dsr is None and spec.min_positive_years == 2


def test_a_staggered_rule():
    text = SPEC.replace("q: 0.2}", "q: 0.2, staggered: true}")
    import tempfile

    d = Path(tempfile.mkdtemp())
    spec = xk.load_spec(spec_file(d, text))
    assert spec.primary.staggered and spec.primary.name.endswith("as 7 staggered portfolios")
    assert not spec.variants[1].staggered  # held a day: one portfolio
    with pytest.raises(ex.ExperimentError, match="staggered is true or false"):
        xk.load_spec(spec_file(d, text.replace("staggered: true", "staggered: 1")))
    planted(d / "paid", True, seed=1)
    r = xk.run(spec, d / "paid", d / "results", prov=PROV)
    plain = xk.run(xk.load_spec(spec_file(d, SPEC, "032-plain.yaml")), d / "paid",
                   d / "results-plain", prov=PROV)  # fmt: skip
    assert r.table[0]["annual_pct"] != plain.table[0]["annual_pct"]
    assert r.table[1] == plain.table[1]  # the daily variant is the same


def test_the_committed_forward_carry_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "011-xsection-carry-forward.yaml")
    assert spec.primary == xk.Rule("carry", 7, 50, 0.2, 1, True)
    assert spec.primary.name == ("carry, every 7 days, top 50, 20% a side, as 7 staggered "
                                 "portfolios, traded 1 h after the close")  # fmt: skip
    assert [(v.hold, v.staggered) for v in spec.variants] == [(7, True), (1, False)]
    assert len(spec.trials) == 7 and all(t.hours == 1 for t in spec.trials)
    assert (spec.holdout_first, spec.holdout_last) == ("2026-12-01", "2028-11-30")
    assert spec.min_dsr is None and spec.min_positive_years == 2


def test_a_rule_on_isolated_margin(tmp_path):
    zigzag(tmp_path / "d", days=DAYS, first="2021-11-01")
    text = SPEC.replace("rule: {signal: carry, hold: 7, top: 20, q: 0.2}",
                        "rule: {signal: rev1, hold: 1, top: 12, q: 0.2, isolated: 0.05}").replace(
        "variants:\n  - {hold: 1}", "variants:\n  - {isolated: null}")  # fmt: skip
    spec = xk.load_spec(spec_file(tmp_path, text))
    assert spec.primary == xk.Rule("rev1", 1, 12, 0.2, 0, False, 0.05)
    assert spec.variants[1].isolated is None and spec.trials[1].isolated == 0.05
    assert spec.primary.name.endswith(", each position on isolated margin (maintenance 5.0%)")
    for bad in ("true", "false", "1", "-0.1", "5%"):
        with pytest.raises(ex.ExperimentError, match="isolated is a maintenance rate"):
            xk.load_spec(spec_file(tmp_path, text.replace("isolated: 0.05", f"isolated: {bad}")))
    # swings of a few % a day liquidate nothing: the same as on the whole capital
    r = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert r.table[0]["annual_pct"] == pytest.approx(r.table[1]["annual_pct"], abs=1e-9)
    # the coin of the widest swings tripling within its hours for ten days: a short in it is
    # liquidated, its margin lost, where on the whole capital it loses its day's move
    from test_xsection import hourly, ms

    a, h0 = 0.12, ms("2022-06-01") // 3_600_000

    def price(h):
        return 10 * np.exp(a * (-1.0) ** (h // 24 + (1 if h % 24 >= 2 else 0)))

    hourly(tmp_path / "d", "C11USDT", "2021-11-01", DAYS + 1, price,
           high=lambda h: price(h) * (3 if h0 <= h < h0 + 240 else 1))  # fmt: skip
    r = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    assert r.table[0]["annual_pct"] < r.table[1]["annual_pct"] - 1
    # the held-out run checks its hour bars first, traded at the close as it is
    for f in (tmp_path / "d" / "um" / "klines" / "C00USDT" / "1h").glob("C00USDT-1h-2024-*.zip"):
        f.unlink()
    with pytest.raises(ex.ExperimentError, match="for C00USDT; nothing was computed or recorded"):
        xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)


def test_the_committed_isolated_carry_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "012-xsection-carry-isolated.yaml")
    assert spec.primary == xk.Rule("carry", 7, 50, 0.2, 1, True, 0.05)
    assert spec.primary.name == ("carry, every 7 days, top 50, 20% a side, as 7 staggered "
                                 "portfolios, traded 1 h after the close, each position on "
                                 "isolated margin (maintenance 5.0%)")  # fmt: skip
    assert [(v.hold, v.staggered, v.isolated) for v in spec.variants] == [
        (7, True, 0.05), (1, False, 0.05)]  # fmt: skip
    eleven = xk.load_spec(root / "011-xsection-carry-forward.yaml")
    assert spec.trials[1] == eleven.primary  # 011 itself among the trials
    assert len(spec.trials) == 10 and sum(t.isolated is None for t in spec.trials) == 8
    assert (spec.holdout_first, spec.holdout_last) == (eleven.holdout_first, eleven.holdout_last)
    assert spec.min_dsr is None and spec.min_positive_years == 2 and spec.max_abs_beta == 0.3


def test_a_rule_with_a_stop(tmp_path):
    text = SPEC.replace("q: 0.2}", "q: 0.2, stop: 1}").replace(
        "variants:\n  - {hold: 1}", "variants:\n  - {stop: null}")  # fmt: skip
    spec = xk.load_spec(spec_file(tmp_path, text))
    assert spec.primary == xk.Rule("carry", 7, 20, 0.2, 0, False, None, 1.0)
    assert spec.variants[1].stop is None and spec.trials[1].stop == 1.0
    assert spec.primary.name.endswith(", each position sold once it has lost its weight")
    half = xk.load_spec(spec_file(tmp_path, text.replace("stop: 1}", "stop: 0.5}"), "033-h.yaml"))
    assert half.primary.name.endswith("sold once it has lost 0.5 times its weight")
    for bad in ("0", "-1", "true", "x"):
        with pytest.raises(ex.ExperimentError, match="stop is a loss above 0"):
            xk.load_spec(spec_file(tmp_path, text.replace("stop: 1}", f"stop: {bad}}}")))
    with pytest.raises(ex.ExperimentError, match="isolated or stop, not both"):
        xk.load_spec(spec_file(tmp_path, text.replace("stop: 1}", "stop: 1, isolated: 0.05}")))
    planted(tmp_path / "d", True, seed=6)
    never = xk.load_spec(spec_file(tmp_path, text.replace("stop: 1}", "stop: 100}"), "034-n.yaml"))
    r = xk.run(never, tmp_path / "d", tmp_path / "r1", prov=PROV)
    assert r.table[0]["annual_pct"] == pytest.approx(r.table[1]["annual_pct"], abs=1e-9)
    tight = xk.load_spec(spec_file(tmp_path, text.replace("stop: 1}", "stop: 0.01}"), "035-t.yaml"))
    r = xk.run(tight, tmp_path / "d", tmp_path / "r2", prov=PROV)
    assert abs(r.table[0]["annual_pct"] - r.table[1]["annual_pct"]) > 1


def test_the_committed_carry_with_a_stop_loads():
    root = Path(__file__).resolve().parents[1] / "experiments"
    spec = xk.load_spec(root / "013-xsection-carry-stop.yaml")
    assert spec.primary == xk.Rule("carry", 7, 50, 0.2, 1, True, None, 1.0)
    assert spec.primary.name.endswith("traded 1 h after the close, each position sold once it "
                                      "has lost its weight")  # fmt: skip
    assert [(v.hold, v.staggered, v.stop) for v in spec.variants] == [(7, True, 1.0),
                                                                      (1, False, 1.0)]  # fmt: skip
    eleven = xk.load_spec(root / "011-xsection-carry-forward.yaml")
    twelve = xk.load_spec(root / "012-xsection-carry-isolated.yaml")
    assert spec.trials[1] == eleven.primary and twelve.primary in spec.trials
    assert len(spec.trials) == 12 and all(t.hours in (0, 1) for t in spec.trials)
    assert (spec.holdout_first, spec.holdout_last) == (eleven.holdout_first, eleven.holdout_last)
    assert spec.min_dsr is None and spec.min_positive_years == 2 and spec.max_abs_beta == 0.3
