"""Bars never reach across a hole, and activity bars close where they should."""

import numpy as np
import pytest

from marketlab import bars, synth
from marketlab.io import Run

S = 1_000_000_000


def trades_day(qty, gap=None, qty_step="1"):
    """One trade a second from 1 s on, with best prices around them."""
    trades = [((i + 1) * S, 0, 0, i + 1, 100, q, i % 2 == 0) for i, q in enumerate(qty)]
    tickers = [((i + 1) * S - 1, i + 1, 99 + i % 3, 1, 101 + i % 3, 1) for i in range(len(qty))]
    quality = [(0, "connected", "conn=1"), (1, "synced", "")]
    if gap:
        quality += [(gap[0], "gap", ""), (gap[1], "resync", "")]
    frames = synth.tables(trades=trades, book_ticker=tickers, quality=quality)
    runs = [Run("t", 0, (len(qty) + 1) * S, None)]
    return synth.memory_day(frames, qty_step=qty_step, runs=runs)


def test_volume_bars_close_at_the_trade_that_reaches_the_threshold():
    day = trades_day([3, 25, 4, 10, 2])
    b = bars.activity_bars(day, "volume", 10)
    assert b["volume"].to_list() == [28, 14]  # 3 + 25 closes, then 4 + 10; 2 is unfinished
    assert b["trades"].to_list() == [2, 2]
    assert b["start"].to_list() == [1 * S, 3 * S]
    assert b["end"].to_list() == [2 * S + 1, 4 * S + 1]


def test_tick_bars_hold_exactly_n_trades_and_start_again_after_a_hole():
    day = trades_day([1] * 10, gap=(int(4.5 * S), int(5.5 * S)))
    b = bars.activity_bars(day, "tick", 3)
    # segment 0 holds trades 1..4 (one bar of 3, then one left over); segment 1 trades 6..10
    assert b["trades"].to_list() == [3, 3]
    assert b["segment"].to_list() == [0, 1]
    assert b["start"].to_list() == [1 * S, 6 * S]


def test_dollar_bars_use_exact_notional():
    day = trades_day([5, 5, 5, 5])  # 500 each at price 100
    b = bars.activity_bars(day, "dollar", 1000)
    assert b["notional"].to_list() == [1000, 1000]


def test_time_bars_lie_inside_one_segment(test_day):
    b = bars.time_bars(test_day, "1s")
    starts, ends = test_day.segment_bounds()
    seg = b["segment"].to_numpy()
    assert (b["start"].to_numpy() >= starts[seg]).all()
    assert (b["end"].to_numpy() <= ends[seg]).all()
    assert (b["start"].to_numpy() % S == 0).all()
    # every valid trade inside a reported bar is counted once
    t = test_day.valid("trades", ["recv_ts"])["recv_ts"].to_numpy()
    inside = np.isin(t // S * S, b["start"].to_numpy())
    assert int(b["trades"].sum()) == int(inside.sum())


def test_quiet_time_bars_are_kept():
    day = trades_day([1, 1, 0, 0, 1])
    b = bars.time_bars(day, "1s")
    assert 0 in b["volume"].to_list()


def test_returns_stay_inside_segments():
    day = trades_day([1] * 10, gap=(int(4.5 * S), int(5.5 * S)))
    b = bars.activity_bars(day, "tick", 1)
    r = bars.returns(b)
    seg = b["segment"].to_numpy()
    assert len(r) == len(b) - 1 - int((seg[1:] != seg[:-1]).sum())


def test_calibrated_bars_come_in_similar_numbers(test_day):
    table = bars.compare(test_day, "1s")
    n = table["n"].to_list()
    assert table["bars"].to_list() == ["time 1s", "tick", "volume", "dollar"]
    assert max(n) / min(n) < 1.2


def test_calibration_counts_the_bars_when_trades_overshoot():
    # heavy-tailed sizes: a big trade passes the threshold by far and the excess is lost, so
    # total / target would give far fewer volume bars than asked for
    rng = np.random.default_rng(11)
    qty = np.minimum((rng.pareto(1.2, 4000) + 1) * 10, 50_000).astype(int).tolist()
    day = trades_day(qty)
    th = bars.calibrate([day], bars_per_hour=1800)  # one bar every two seconds
    target = 1800 * sum(e - s for s, e in day.intervals) / 1e9 / 3600
    for kind in ("volume", "dollar"):
        n = bars.activity_bars(day, kind, getattr(th, kind)).height
        assert abs(n - target) <= 0.02 * target, (kind, n, target)
    naive = sum(qty) / target
    assert bars.activity_bars(day, "volume", round(naive)).height < 0.8 * target


def test_bad_arguments():
    day = trades_day([1, 2])
    with pytest.raises(ValueError):
        bars.activity_bars(day, "seconds", 3)
    with pytest.raises(ValueError):
        bars.activity_bars(day, "tick", 0)


def test_days_with_different_steps_are_not_calibrated_together():
    a = trades_day([3, 25, 4, 10, 2])
    finer = trades_day([30, 250, 40, 100, 20], qty_step="0.1")  # the same trades
    with pytest.raises(ValueError, match="different steps"):
        bars.calibrate([a, finer], 3600)
    assert bars.calibrate([a, a], 3600).volume > 0
