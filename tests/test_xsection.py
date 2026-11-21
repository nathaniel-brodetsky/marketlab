"""The cross-sectional look: the universe, the portfolio, its costs and its funding worked
out by hand; a planted momentum found and none where there is none; the held-out years
refused and never read."""

import datetime as dt

import numpy as np
import pytest

from marketlab import xsection as xs
from marketlab.cli import main

DAY_MS = 86_400_000


def ms(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * 1000


def write(root, sym, first, close, qvol, funding=None):
    """Daily bars from `first`, and settlements every 8 h with the day's rate a third each."""
    k = root / "um" / "klines1d" / f"{sym}.csv"
    k.parent.mkdir(parents=True, exist_ok=True)
    t0 = ms(first)
    lines = ["open_time,open,high,low,close,volume,quote_volume,trades"]
    lines += [
        f"{t0 + i * DAY_MS},{c},{c},{c},{c},1.0,{v},10"
        for i, (c, v) in enumerate(zip(close, qvol, strict=True))
    ]
    k.write_text("\n".join(lines) + "\n")
    if funding is not None:
        f = root / "um" / "fundingRate" / sym / f"{sym}-fundingRate.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = ["funding_time,funding_rate,mark_price"]
        for i, rate in enumerate(funding):
            rows += [f"{t0 + i * DAY_MS + h * 3_600_000},{rate / 3:.10f}," for h in (0, 8, 16)]
        f.write_text("\n".join(rows) + "\n")


def planted(root, drift: float, seed: int, n_sym=40, days=700):
    """Each coin drifts at a rate that persists for 60 days, plus noise: momentum pays."""
    rng = np.random.default_rng(seed)
    for j in range(n_sym):
        mu = np.repeat(rng.normal(0, drift, days // 60 + 1), 60)[:days]
        lr = mu + rng.normal(0, 0.03, days)
        close = 10 * np.exp(np.cumsum(lr))
        write(root, f"C{j:02d}USDT", "2021-01-01", close, rng.uniform(1e6, 2e6, days),
              rng.normal(1e-4, 5e-5, days))  # fmt: skip


def test_the_portfolio_by_hand(tmp_path):
    # 10 coins over 45 days; coin j's volume is j, so the top 5 are coins 5 .. 9
    days = 45
    for j in range(10):
        close = 100 * (1 + 0.01 * j) ** np.arange(days)  # coin j gains j % a day
        write(tmp_path, f"C{j}USDT", "2022-01-01", close, np.full(days, float(j + 1)),
              np.full(days, 3e-4 * j))  # fmt: skip
    p = xs.load(tmp_path, "2022-02-01", "2022-02-14")
    assert p.dates[-1] == "2022-02-14" and p.dates[0] == "2021-12-23"
    inu = xs.universe(p, top=5)
    i = p.dates.index("2022-02-01")
    assert sorted(np.flatnonzero(inu[i])) == [5, 6, 7, 8, 9]
    assert not inu[p.dates.index("2022-01-15")].any()  # less than 30 days of bars
    r = xs.daily_returns(p)
    assert r[i, 7] == pytest.approx(0.07 - 3e-4 * 7)  # a long pays the funding
    x = xs.signals(p)["mom7"]
    pnl, turns, names = xs.portfolio(x, inu, r, i, 7, 0.2, 9.0)
    # top 20% of 5 coins = 1 a side: long coin 9 (strongest), short coin 5
    assert names == [1, 1] and turns[0] == pytest.approx(1.0) and turns[1] == pytest.approx(0.0)
    assert pnl[i] == pytest.approx(-1.0 * 9 / 2 / 1e4)
    assert pnl[i + 1] == pytest.approx(0.5 * r[i + 1, 9] - 0.5 * r[i + 1, 5])
    carry = xs.signals(p)["carry"]
    assert carry[i, 3] == pytest.approx(-3e-4 * 3)  # minus the mean daily rate over 7 days


def test_a_planted_momentum_is_found_and_none_where_there_is_none(tmp_path):
    planted(tmp_path / "trend", 0.01, seed=1)
    planted(tmp_path / "none", 0.0, seed=2)
    _, found = xs.report(tmp_path / "trend", "2021-03-01", "2022-11-30", top=30, cost_bp=9.0)
    mom30 = next(r for r in found if r.signal == "mom30" and r.hold == 7)
    assert mom30.ic > 0.1 and mom30.ic_t > 5 and mom30.net_pct > 20
    assert mom30.cost_pct == pytest.approx(mom30.gross_pct - mom30.net_pct)
    _, none = xs.report(tmp_path / "none", "2021-03-01", "2022-11-30", top=30, cost_bp=9.0)
    for r in none:
        assert abs(r.ic) < 0.03, r
    rev = next(r for r in none if r.signal == "rev1")
    assert rev.net_pct < 0 and rev.cost_pct > 5  # trading every day for nothing pays the costs


def test_the_held_out_years_are_neither_asked_for_nor_read(tmp_path, capsys):
    write(tmp_path, "AUSDT", "2023-12-01", np.arange(1.0, 61.0), np.ones(60))
    with pytest.raises(xs.HeldOut):
        xs.load(tmp_path, "2023-12-01", "2024-01-01")
    p = xs.load(tmp_path, "2023-12-10", "2023-12-31")
    assert p.dates[-1] == "2023-12-31" and np.isfinite(p.close[-1, 0])
    assert len(p.dates) == 22 + 40  # 2024 bars in the file are not read
    assert main(["xsection", "--data", str(tmp_path), "--until", "2024-06-30"]) == 2
    assert "held out" in capsys.readouterr().err


def test_the_command(tmp_path, capsys):
    planted(tmp_path, 0.01, seed=3, n_sym=25, days=400)
    assert main(["xsection", "--data", str(tmp_path), "--first", "2021-03-01",
                 "--until", "2021-12-31", "--top", "20"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "25 perpetuals" in out and "mom30" in out and "carry" in out and "net_pct" in out
    assert "not coins: BLUEBIRDUSDT, BTCDOMUSDT" in out and "no daily bar" not in out
    assert "coins gone from its list are missing" in out
    (tmp_path / "um" / "gone.csv").write_text("symbol\n")
    assert main(["xsection", "--data", str(tmp_path), "--first", "2021-03-01",
                 "--until", "2021-12-31", "--top", "20"]) == 0  # fmt: skip
    assert "are in, from its archive" in capsys.readouterr().out
    assert main(["xsection", "--data", str(tmp_path / "empty")]) == 2


def test_coins_tied_on_the_signal_are_not_ranked_by_name():
    x = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0]])
    x = np.vstack([x, x])
    inu = np.ones_like(x, dtype=bool)
    r = np.zeros_like(x)
    _, turns, names = xs.portfolio(x, inu, r, 0, 1, 0.2, 9.0)
    assert names == [1] and turns == [pytest.approx(1.0)]  # long coin 5, short coin 6 only
    tied = np.ones_like(x)
    _, turns, names = xs.portfolio(tied, inu, r, 0, 1, 0.2, 9.0)
    assert names == [] and turns == [0.0]  # all tied: no position


def test_columns_whose_first_rows_look_whole_are_read(tmp_path):
    # polars guesses a column's type from its first rows: whole volumes, then a fraction
    k = tmp_path / "um" / "klines1d" / "AUSDT.csv"
    k.parent.mkdir(parents=True)
    t0 = ms("2021-01-01")
    lines = ["open_time,open,high,low,close,volume,quote_volume,trades"]
    lines += [f"{t0 + i * DAY_MS},1,1,1,{1 + i},{100 if i < 150 else 1.5},"
              f"{200 if i < 150 else 2.5},10" for i in range(200)]  # fmt: skip
    k.write_text("\n".join(lines) + "\n")
    p = xs.load(tmp_path, "2021-03-01", "2021-07-19")
    assert p.qvol[p.dates.index("2021-05-30"), 0] == 200.0
    assert p.qvol[p.dates.index("2021-05-31"), 0] == 2.5 and p.close[-1, 0] == 200.0


def test_contracts_that_are_not_coins_are_left_out_and_days_without_bars_counted(tmp_path):
    write(tmp_path, "AUSDT", "2022-01-01", np.ones(60), np.ones(60), np.zeros(60))
    write(tmp_path, "USDCUSDT", "2022-01-01", np.ones(60), np.full(60, 1e9), np.zeros(60))
    # funding settled from 2022-01-01, bars only from 2022-01-21
    write(tmp_path, "BUSDT", "2022-01-01", np.ones(60), np.ones(60), np.zeros(60))
    k = tmp_path / "um" / "klines1d" / "BUSDT.csv"
    lines = k.read_text().splitlines()
    k.write_text("\n".join([lines[0], *lines[21:]]) + "\n")
    p = xs.load(tmp_path, "2022-02-10", "2022-03-01")
    assert p.symbols == ["AUSDT", "BUSDT"] and p.unbarred == {"BUSDT": 20}


def test_where_the_money_came_from(tmp_path):
    # 12 coins over 120 days, random walks with funding; coin 3 gains 400% on 2022-03-10
    rng = np.random.default_rng(7)
    days = 120
    for j in range(12):
        lr = rng.normal(0, 0.03, days)
        if j == 3:  # the strongest for ten days, so held long, then up 400% on 2022-03-10
            lr[58:68] += 0.05
            lr[68] = np.log(5.0)
        write(tmp_path, f"C{j:02d}USDT", "2022-01-01", 10 * np.exp(np.cumsum(lr)),
              rng.uniform(1e6, 2e6, days), rng.normal(2e-4, 3e-4, days))  # fmt: skip
    p = xs.load(tmp_path, "2022-02-15", "2022-04-30")
    whole = xs.evaluate(p, "mom7", xs.signals(p)["mom7"], 1, 12, 0.2, 9.0, "2022-02-15")
    e = xs.explain(p, "mom7", 1, top=12, q=0.2, cost_bp=9.0, first="2022-02-15")
    assert sum(e.legs.values()) == pytest.approx(whole.net_pct)  # the legs add up to the net
    assert e.legs["costs"] == pytest.approx(-whole.cost_pct)
    assert [y["year"] for y in e.years] == ["2022"]
    assert e.years[0]["net_pct"] == pytest.approx(whole.net_pct)
    held, _, _, _ = xs.holdings(xs.signals(p)["mom7"], xs.universe(p, 12),
                                p.dates.index("2022-02-15"), 1, 0.2)  # fmt: skip
    i = p.dates.index("2022-03-10")
    assert held[i, 3] == pytest.approx(0.5 / 3)  # 3 coins long, half the capital
    top = e.coin_days[0]
    assert (top["date"], top["symbol"]) == ("2022-03-10", "C03USDT")
    assert top["price_pct"] == pytest.approx(400.0) and top["weight"] == pytest.approx(0.5 / 3)
    assert top["contribution_pct"] == pytest.approx(100 / 6 * (4.0 - top["funding_pct"] / 100))
    assert [d["contribution_pct"] for d in e.coin_days[:10]] == sorted(
        [d["contribution_pct"] for d in e.coin_days[:10]], reverse=True)  # fmt: skip
    # that day alone is 2/3 of the capital over 75 days: some 325% a year
    assert whole.net_pct - e.net_without_best > 300
    assert sum(qq["coin_days"] for qq in e.quintiles) > 0


def test_the_fifths_show_a_planted_momentum(tmp_path):
    planted(tmp_path, 0.01, seed=4, n_sym=30, days=400)
    p = xs.load(tmp_path, "2021-03-01", "2021-12-31")
    e = xs.explain(p, "mom30", 1, top=30, first="2021-03-01")
    means = [qq["mean_bp"] for qq in e.quintiles]
    assert means[4] > means[2] > means[0] and means[4] - means[0] > 50
    assert e.quintiles[4]["median_bp"] > e.quintiles[0]["median_bp"]
    planted(tmp_path / "none", 0.0, seed=6, n_sym=30, days=400)
    flat = xs.explain(xs.load(tmp_path / "none", "2021-03-01", "2021-12-31"), "mom7", 1, top=30,
                      first="2021-03-01")  # fmt: skip
    means = [qq["mean_bp"] for qq in flat.quintiles]
    assert abs(means[4] - means[0]) < 40  # the next day's, not the signal's own


def test_the_command_explains(tmp_path, capsys):
    planted(tmp_path, 0.01, seed=5, n_sym=20, days=300)
    assert main(["xsection", "--data", str(tmp_path), "--first", "2021-03-01", "--until",
                 "2021-10-31", "--top", "15", "--explain", "carry:1", "mom30:7"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "carry, rebalanced every 1 days" in out and "mom30, rebalanced every 7 days" in out
    assert "median_bp" in out and "contribution_pct" in out and "the net without them" in out
    assert main(["xsection", "--data", str(tmp_path), "--first", "2021-03-01", "--until",
                 "2021-10-31", "--explain", "nothing:1"]) == 2  # fmt: skip
