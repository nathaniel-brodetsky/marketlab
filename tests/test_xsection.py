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
    assert "um/universe.csv: every file under um/klines1d taken for a perpetual" in out
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


def test_a_day_ranks_the_same_wherever_the_data_begin_and_rates_tie_in_any_order(tmp_path):
    # most coins on Binance's default rate most days, as in the market: ties at the cut
    rng = np.random.default_rng(0)
    days = 200
    for j in range(30):
        f = np.where(rng.random(days) < 0.8, 3e-4, np.round(rng.normal(3e-4, 4e-4, days), 8))
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
        write(tmp_path, f"C{j:02d}USDT", "2023-06-01", close, rng.uniform(1e6, 2e6, days), f)
    # pairs of coins whose last week holds the same rates, one in the other's reverse order
    # (ordinary rates, and a squeeze's), and one a day's rate off by the smallest step
    week = np.array([1e-4, 3e-4, 1.2e-3, 6.25e-5, -4e-4, 4.5e-4, 6.25e-5])
    squeeze = np.array([-0.0464, -0.173, 0.6646, 0.0452, -0.0705, -0.0563, -0.1336])
    step = week + np.array([1e-8, 0, 0, 0, 0, 0, 0])
    for sym, w in (("AAAUSDT", week), ("BBBUSDT", week[::-1]), ("DDDUSDT", squeeze),
                   ("EEEUSDT", squeeze[::-1]), ("CCCUSDT", step)):  # fmt: skip
        f = np.full(days, 3e-4)
        f[-7:] = w
        write(tmp_path, sym, "2023-06-01", np.full(days, 10.0), np.full(days, 3e6), f)
    last = "2023-12-17"  # the 200th day
    a = xs.load(tmp_path, "2023-07-15", last)
    b = xs.load(tmp_path, "2023-09-01", last)
    ia, ib = a.dates.index("2023-09-01"), b.dates.index("2023-09-01")
    sa, sb = xs.signals(a)["carry"], xs.signals(b)["carry"]
    assert np.array_equal(sa[ia:], sb[ib:], equal_nan=True)
    ua, ub = xs.universe(a, 20), xs.universe(b, 20)
    assert np.array_equal(ua[ia:], ub[ib:])
    ha = xs.holdings(sa, ua, ia, 7, 0.2, True)[0]
    hb = xs.holdings(sb, ub, ib, 7, 0.2, True)[0]
    assert np.array_equal(ha[ia:], hb[ib:])
    assert (ha[ia:] != 0).any()
    c = {s: sb[-1, b.symbols.index(s)] for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT",
                                                  "EEEUSDT")}  # fmt: skip
    # tied, so that neither is ranked above the other by rounding; a step apart, not tied
    assert c["AAAUSDT"] == c["BBBUSDT"] and c["DDDUSDT"] == c["EEEUSDT"]
    assert c["CCCUSDT"] < c["AAAUSDT"]
    assert np.isnan(xs._trailing_mean(np.array([[1.0], [np.nan], [1.0]]), 2)).all()
    assert xs._trailing_mean(np.array([[1.0], [3.0]]), 2)[-1, 0] == 2.0  # as long as the window


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


def test_only_the_perpetuals_on_coins_listed_or_gone_are_in(tmp_path, capsys):
    for sym in ("AUSDT", "BUSDT", "IDXUSDT", "GONEUSDT", "SHRUSDT"):
        write(tmp_path, sym, "2022-01-01", np.ones(60), np.ones(60), np.zeros(60))
    p = xs.load(tmp_path, "2022-02-10", "2022-03-01")  # no universe.csv: every file
    assert len(p.symbols) == 5 and not p.listed
    um = tmp_path / "um"
    # an old universe.csv, before the underlying's type was written: every one a coin
    (um / "universe.csv").write_text("symbol,status\nAUSDT,TRADING\nBUSDT,SETTLING\n"
                                     "IDXUSDT,TRADING\n")  # fmt: skip
    p = xs.load(tmp_path, "2022-02-10", "2022-03-01")
    assert p.symbols == ["AUSDT", "BUSDT", "IDXUSDT"] and p.listed
    assert p.unlisted == ["GONEUSDT", "SHRUSDT"]
    (um / "universe.csv").write_text("symbol,status,underlying_type\nAUSDT,TRADING,COIN\n"
                                     "BUSDT,SETTLING,\nIDXUSDT,TRADING,INDEX\n")  # fmt: skip
    (um / "gone.csv").write_text("symbol,first_day\nGONEUSDT,2022-01-01\n")
    p = xs.load(tmp_path, "2022-02-10", "2022-03-01")
    assert p.symbols == ["AUSDT", "BUSDT", "GONEUSDT"]
    assert p.other_type == {"IDXUSDT": "INDEX"} and p.unlisted == ["SHRUSDT"]
    assert main(["xsection", "--data", str(tmp_path), "--first", "2022-02-10",
                 "--until", "2022-03-01", "--top", "3"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "3 perpetuals under" in out
    assert "left out, 1 perpetuals on an underlying Binance does not call a coin: INDEX 1" in out
    assert "left out, 1 files of contracts neither listed as perpetuals nor gone" in out
    assert "SHRUSDT" in out and "no " + str(tmp_path) not in out


def test_the_funding_a_day_its_settlements_and_a_settlement_given_twice(tmp_path):
    funding = np.full(60, 3e-4)
    funding[45] = 0.15  # 5% at each of the day's three settlements
    write(tmp_path, "AUSDT", "2022-01-01", np.ones(60), np.full(60, 9.0), funding)
    loud = np.zeros(60)
    loud[50] = -0.6  # larger, but out of the universe: the least volume
    write(tmp_path, "BUSDT", "2022-01-01", np.ones(60), np.full(60, 1.0), loud)
    write(tmp_path, "CUSDT", "2022-01-01", np.ones(60), np.full(60, 5.0), np.zeros(60))
    p = xs.load(tmp_path, "2022-02-10", "2022-03-01")
    i = p.dates.index("2022-02-15")
    assert p.settlements[i, 0] == 3 and p.funding[i, 0] == pytest.approx(0.15)
    big = xs.largest_funding(p, xs.universe(p, top=2), p.dates.index("2022-02-10"), n=2)
    assert big[0] == {"symbol": "AUSDT", "date": "2022-02-15", "pct": pytest.approx(15.0),
                      "settlements": 3}  # fmt: skip
    assert big[1]["symbol"] == "AUSDT" and big[1]["pct"] == pytest.approx(0.03)
    f = tmp_path / "um" / "fundingRate" / "CUSDT" / "CUSDT-fundingRate.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join([*lines, lines[-1]]) + "\n")
    with pytest.raises(ValueError, match=r"CUSDT-fundingRate\.csv: a settlement time given twice"):
        xs.load(tmp_path, "2022-02-10", "2022-03-01")


def test_a_portfolio_that_only_rides_the_market_has_its_beta_and_nothing_else(tmp_path):
    # coin j's day is b_j times the market's, b_j from 0 to 2; the signal ranks the coins
    # by b_j, so the portfolio is long the high betas and short the low ones
    rng = np.random.default_rng(11)
    days = 300
    m = rng.normal(0.003, 0.03, days)
    b = np.arange(12) / 5.5
    for j in range(12):
        r = b[j] * m + rng.normal(0, 0.0005, days)  # simple returns: linear in the market's
        write(tmp_path, f"C{j:02d}USDT", "2021-01-01", 10 * np.cumprod(1 + r),
              rng.uniform(1e6, 2e6, days), np.full(days, -b[j] * 1e-6))  # fmt: skip
    p = xs.load(tmp_path, "2021-03-01", "2021-10-27")
    e = xs.explain(p, "carry", 7, top=12, q=0.2, first="2021-03-01")
    # long the 3 highest betas, short the 3 lowest, half the capital a side
    assert e.beta == pytest.approx(0.5 * (b[9:].mean() - b[:3].mean()), abs=0.05)
    whole = xs.evaluate(p, "carry", xs.signals(p)["carry"], 7, 12, 0.2, 9.0, "2021-03-01")
    assert e.market_pct > 50 and whole.net_pct > 30 and abs(e.net_less_beta) < 1


def test_a_one_day_reversal_pays_on_time_and_costs_a_day_late(tmp_path):
    # every coin up a_j one day and down a_j the next: yesterday's losers win today
    days = 120
    for j in range(12):
        close = 10 * np.exp(0.01 * (j + 1) * (-1.0) ** np.arange(days))
        write(tmp_path, f"C{j:02d}USDT", "2021-01-01", close, np.full(days, 1e6 + j))
    p = xs.load(tmp_path, "2021-03-01", "2021-04-30")
    e = xs.explain(p, "rev1", 1, top=12, q=0.2, first="2021-03-01")
    whole = xs.evaluate(p, "rev1", xs.signals(p)["rev1"], 1, 12, 0.2, 9.0, "2021-03-01")
    assert whole.net_pct > 1000 and e.late_net < -1000 and e.late_sharpe < 0


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
    assert out.count("beta to the universe (equal weights") == 2 and "Decided a day late" in out
    assert "the largest funding in a day, in the universe: C" in out and "(3 settlements)" in out
    assert main(["xsection", "--data", str(tmp_path), "--first", "2021-03-01", "--until",
                 "2021-10-31", "--explain", "nothing:1"]) == 2  # fmt: skip


def hourly(root, sym, first_day: str, days: int, price, high=None, low=None) -> None:
    """Monthly hour-bar archives, as Binance publishes them (no header): `price(hour)` is
    each bar's open and close, `high(hour)` and `low(hour)` its extremes (the open without
    them), `hour` counted from the epoch."""
    import io
    import zipfile

    h0 = ms(first_day) // 3_600_000
    by_month: dict[str, list[str]] = {}
    for h in range(h0, h0 + days * 24):
        t = h * 3_600_000
        month = dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m")
        o = price(h)
        hi, lo = (high or price)(h), (low or price)(h)
        by_month.setdefault(month, []).append(
            f"{t},{o},{hi},{lo},{o},1,{t + 3_599_999},1,1,1,1,0")  # fmt: skip
    folder = root / "um" / "klines" / sym / "1h"
    folder.mkdir(parents=True, exist_ok=True)
    for month, rows in by_month.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(f"{sym}-1h-{month}.csv", "\n".join(rows) + "\n")
        (folder / f"{sym}-1h-{month}.zip").write_bytes(buf.getvalue())


def test_traded_hours_after_the_close_by_hand(tmp_path):
    # one coin, its price at hour h is 100 + h / 1000; funding at 0, 8 and 16 h, a distinct
    # rate each: 1e-6 times the hour of the day plus one, times the day's number plus one
    days = 60
    day0 = ms("2023-11-15") // 86_400_000
    close = [100 + ((day0 + i + 1) * 24) / 1000 for i in range(days)]  # the next 00:00's open
    write(tmp_path, "AUSDT", "2023-11-15", close, np.ones(days))
    f = tmp_path / "um" / "fundingRate" / "AUSDT" / "AUSDT-fundingRate.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = ["funding_time,funding_rate,mark_price"]

    def rate(i, h):
        return 1e-6 * (h + 1) * (i + 1)

    for i in range(days + 1):
        for h in (0, 8, 16):
            rows.append(f"{(day0 + i) * 86_400_000 + h * 3_600_000},{rate(i, h)},")
    f.write_text("\n".join(rows) + "\n")
    hourly(tmp_path, "AUSDT", "2023-11-15", days + 1, lambda h: 100 + h / 1000)
    # a month after the development years that would not even open
    (tmp_path / "um" / "klines" / "AUSDT" / "1h" / "AUSDT-1h-2024-02.zip").write_text("no zip")
    p = xs.load(tmp_path, "2023-12-26", "2023-12-31")
    lag = xs.lagged(p, tmp_path, [0, 5])
    daily = xs.daily_returns(p)
    i = p.dates.index("2023-12-28")
    d = ms("2023-12-28") // 3_600_000  # that day's first hour
    assert lag[0].r[i, 0] == pytest.approx(daily[i, 0])  # at the close: as the daily bars
    # five hours later: from 05:00 to the next 05:00, paying the 08:00 and 16:00 settlements
    # of that day and the next 00:00's
    k = d // 24 - day0
    paid = rate(k, 8) + rate(k, 16) + rate(k + 1, 0)
    want = (100 + (d + 24 + 5) / 1000) / (100 + (d + 5) / 1000) - 1 - paid
    assert lag[5].r[i, 0] == pytest.approx(want)
    last = p.dates.index("2023-12-31")  # its trading hour is in 2024: not read
    assert np.isnan(lag[0].r[last, 0]) and np.isnan(lag[5].r[last, 0])
    assert lag[5].priced[last - 1, 0] and not lag[5].priced[last, 0]
    with pytest.raises(ValueError, match="from 0 to 23"):
        xs.lagged(p, tmp_path, [24])
    held = xs.load(tmp_path, "2024-01-01", "2024-01-05", held_out=True)
    with pytest.raises(xs.HeldOut):
        xs.lagged(held, tmp_path, [0])


def zigzag(root, days=120, first="2021-01-01", coins=None):
    """Twelve coins whose daily close swings up and down by a_j, the hour bars making the
    day's whole move in its first two hours: traded at the close, a reversal gets it all;
    four hours later, it is already gone and the next day's move is the wrong way."""
    day0 = ms(first) // 86_400_000
    for j in range(12) if coins is None else coins:
        a = 0.01 * (j + 1)

        def level(b, a=a):
            return 10 * np.exp(a * (-1.0) ** b)

        close = [level(day0 + i + 1) for i in range(days)]
        write(root, f"C{j:02d}USDT", first, close, np.full(days, 1e6 + j))
        hourly(root, f"C{j:02d}USDT", first, days + 1,
               lambda h, level=level: level(h // 24 + (1 if h % 24 >= 2 else 0)))  # fmt: skip


def test_a_move_made_in_the_first_hours_pays_at_the_close_and_not_four_hours_later(tmp_path):
    zigzag(tmp_path)
    p = xs.load(tmp_path, "2021-03-01", "2021-04-30")
    rows = xs.at_hours(p, tmp_path, [("rev1", 1)], [0, 4], top=12, first="2021-03-01")
    by = {r["hours"]: r for r in rows}
    whole = xs.evaluate(p, "rev1", xs.signals(p)["rev1"], 1, 12, 0.2, 9.0, "2021-03-01")
    assert by[0]["net_pct"] == pytest.approx(whole.net_pct, rel=1e-6)
    assert by[4]["net_pct"] < -1000 and by[4]["sharpe_months"] < 0
    assert by[0]["priced"] == 1.0 and by[4]["priced"] == 1.0


def test_the_command_trades_hours_after_the_close(tmp_path, capsys):
    zigzag(tmp_path)
    args = ["xsection", "--data", str(tmp_path), "--first", "2021-03-01", "--until",
            "2021-04-30", "--top", "12"]  # fmt: skip
    out_file = tmp_path / "universe.txt"
    assert main([*args, "--hours", "0", "4", "--hour-rules", "rev1:1",
                 "--list-universe", str(out_file)]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "traded H hours after the daily close" in out and "rev1:1" in out
    assert "12 perpetuals ever in the universe" in out
    assert out_file.read_text().split() == [f"C{j:02d}USDT" for j in range(12)]
    assert main([*args, "--hours", "24"]) == 2
    assert main([*args, "--hours", "1", "--hour-rules", "rev2:1"]) == 2


def test_staggered_portfolios_are_one_a_day_and_no_start_day_decides():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 15))
    inu = np.ones_like(x, dtype=bool)
    plain = [xs.holdings(x, inu, 10 + k, 7, 0.2) for k in range(7)]
    held, traded, turns, names = xs.holdings(x, inu, 10, 7, 0.2, staggered=True)
    assert np.allclose(held, sum(h for h, _, _, _ in plain) / 7)  # a seventh each
    assert np.allclose(traded, sum(t for _, t, _, _ in plain) / 7)
    assert sum(turns) == pytest.approx(traded.sum()) and names and max(names) <= 3
    gross = np.abs(held[17:]).sum(axis=1)  # all seven in from day 17; a coin one holds long
    assert (gross > 0.2).all() and (gross <= 1 + 1e-9).all()  # and another short nets out
    # started on another day of the week: the same coins once all seven are in
    for k in range(1, 7):
        other, _, _, _ = xs.holdings(x, inu, 10 + k, 7, 0.2, staggered=True)
        assert np.allclose(other[10 + k + 7 :], held[10 + k + 7 :])
        once, _, _, _ = xs.holdings(x, inu, 10 + k, 7, 0.2)
        assert not np.allclose(once[17 + k :], plain[0][0][17 + k :])  # one portfolio: not so
    one, _, _, _ = xs.holdings(x, inu, 10, 1, 0.2, staggered=True)
    assert np.array_equal(one, xs.holdings(x, inu, 10, 1, 0.2)[0])


def test_the_command_runs_staggered(tmp_path, capsys):
    planted(tmp_path, 0.01, seed=3, n_sym=25, days=400)
    args = ["xsection", "--data", str(tmp_path), "--first", "2021-03-01", "--until",
            "2021-12-31", "--top", "20"]  # fmt: skip
    assert main(args) == 0
    plain = capsys.readouterr().out
    assert main([*args, "--staggered"]) == 0
    out = capsys.readouterr().out
    assert "N staggered portfolios, one rebalanced each day" in out
    row = [x for x in out.splitlines() if "│ mom30" in x and "┆ 7 " in x]
    assert row and row[0] not in plain  # the weekly rows change
    same = [x for x in out.splitlines() if "│ rev1" in x]
    assert same and same[0] in plain  # the daily ones do not


def test_the_hour_bars_give_each_days_highest_and_lowest(tmp_path):
    # flat at 10 but for a spike to 25 at 05:00 on Jan 4 and a dip to 4 at 00:00 on Jan 6
    write(tmp_path, "AAAUSDT", "2021-01-01", np.full(10, 10.0), np.full(10, 1e6))
    spike, dip = ms("2021-01-04") // 3_600_000 + 5, ms("2021-01-06") // 3_600_000
    hourly(tmp_path, "AAAUSDT", "2021-01-01", 11, lambda h: 10.0,
           high=lambda h: 25.0 if h == spike else 10.0,
           low=lambda h: 4.0 if h == dip else 10.0)  # fmt: skip
    p = xs.load(tmp_path, "2021-01-05", "2021-01-08")
    lag = xs.lagged(p, tmp_path, [0, 1])
    day = {d: p.dates.index(d) for d in ("2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06")}
    for L in (0, 1):
        assert lag[L].up[day["2021-01-04"], 0] == pytest.approx(1.5)  # 25 over 10, less 1
        assert lag[L].up[day["2021-01-03"], 0] == 0 and lag[L].up[day["2021-01-05"], 0] == 0
    # the dip's hour opens Jan 6: traded at the close, Jan 6's span; an hour after, Jan 5's
    assert lag[0].down[day["2021-01-06"], 0] == pytest.approx(0.6)
    assert lag[0].down[day["2021-01-05"], 0] == 0
    assert lag[1].down[day["2021-01-05"], 0] == pytest.approx(0.6)
    assert lag[1].down[day["2021-01-06"], 0] == 0
    assert np.isnan(lag[1].up[: p.dates.index("2021-01-01"), 0]).all()  # no bars, no price


def margin_case(n=12):
    """Ten coins ranked 0 .. 9 every day: long 8 and 9, short 0 and 1, a quarter each."""
    x = np.tile(np.arange(10.0), (n, 1))
    z = np.zeros((n, 10))
    return x, np.ones_like(x, dtype=bool), z.copy(), z.copy(), z.copy()


def test_on_isolated_margin_a_position_loses_its_margin_and_no_more():
    x, inu, r, up, down = margin_case()
    r[2, 0], up[2, 0] = 1.5, 1.5  # the short 0 up 150% on day 2: its margin gone
    r[3, 0] = 0.5  # and on up the next day, held no more
    r[4, 9], down[4, 9] = -0.5, 0.5  # the long 9 halved: half its margin left
    r[5, 9], down[5, 9] = -0.4, 0.97  # then 97% down within day 5, 40% at its end
    r[5, 1], up[5, 1] = 0.1, 0.6  # the short 1 60% up within day 5: a margin left
    r[8, 0] = 0.2  # rebalanced at day 7's close: the short 0 held again
    r[9, 1], up[9, 1] = 1.2, np.nan  # no hour bars for the short 1 on day 9: its close, +120%,
    # after a rebalance that gave it its whole margin back
    m = xs.Margin(up, down, 0.05)
    capped, turns, names = xs.portfolio(x, inu, r, 0, 7, 0.2, 0.0, margin=m)
    plain, turns0, names0 = xs.portfolio(x, inu, r, 0, 7, 0.2, 0.0)
    want = np.zeros(12)
    want[[2, 3, 4, 5, 8, 9]] = [-0.25, 0.0, -0.125, -0.125 - 0.025, -0.05, -0.25]
    assert capped == pytest.approx(want)
    assert plain[[2, 3, 5]] == pytest.approx([-0.375, -0.125, -0.125])
    assert turns == turns0 and names == names0
    # the two positions liquidated are bought back at the rebalance, and pay for it
    paid, _, _ = xs.portfolio(x, inu, r, 0, 7, 0.2, 10.0, margin=m)
    assert capped - paid == pytest.approx(np.where(np.arange(12) == 0, 5e-4, 0)
                                          + np.where(np.arange(12) == 7, 2.5e-4, 0))  # fmt: skip
    # a margin rate of 50%: the short 1's 60% within the day takes it too
    half, _, _ = xs.portfolio(x, inu, r, 0, 7, 0.2, 0.0, margin=xs.Margin(up, down, 0.5))
    assert half[5] == pytest.approx(-0.125 - 0.25)


def test_on_isolated_margin_without_a_liquidation_nothing_changes():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 20))
    inu = np.ones_like(x, dtype=bool)
    r = rng.normal(0, 0.03, x.shape)
    up, down = np.abs(r) + 0.05, np.abs(r) + 0.05
    m = xs.Margin(up, down, 0.05)
    for hold, stag in ((1, False), (7, False), (7, True)):
        a = xs.portfolio(x, inu, r, 3, hold, 0.2, 9.0, stag)
        b = xs.portfolio(x, inu, r, 3, hold, 0.2, 9.0, stag, m)
        assert np.allclose(a[0], b[0], rtol=0, atol=1e-15) and a[1:] == b[1:]


def test_staggered_portfolios_are_liquidated_each_on_its_own():
    x, inu, r, up, down = margin_case(30)
    r[10, 0], up[10, 0] = 1.5, 1.5  # the short 0 up 150% on day 10: all seven lose it
    r[11:18, 0] = 0.1  # and it goes on up: back in one portfolio a day, at its rebalance
    capped, _, _ = xs.portfolio(x, inu, r, 0, 7, 0.2, 0.0, True, xs.Margin(up, down, 0.05))
    plain, _, _ = xs.portfolio(x, inu, r, 0, 7, 0.2, 0.0, True)
    assert capped[10] == pytest.approx(-0.25) and plain[10] == pytest.approx(-0.375)
    assert capped[11:18] == pytest.approx(-np.arange(1, 8) / 7 * 0.25 * 0.1)
    assert plain[11:18] == pytest.approx(np.full(7, -0.025))
