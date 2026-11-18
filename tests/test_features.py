"""Features against values worked out by hand on small books."""

import numpy as np
import pytest

from marketlab import synth
from marketlab.features import (
    VPIN,
    BookShape,
    CancelShare,
    Intensity,
    MicroPrice,
    MultiLevelOFI,
    NotFitted,
    OrderFlowImbalance,
    TradeFlow,
    Volatility,
    compute,
    duration_ns,
    grid,
)
from marketlab.features.base import Grid, rolling_extreme, window_sum
from marketlab.features.microprice import adjustment
from marketlab.features.ofi import flow
from marketlab.io import Book20, Run
from marketlab.registry import FITTED, METHODS, default_methods

S = 1_000_000_000


def l20(ts, uid, bids, asks):
    """A book_l20 row from [(price, qty), ...] best first."""

    def pad(levels, i):
        v = [lv[i] for lv in levels]
        return v + [0] * (20 - len(v))

    return (ts, uid, pad(bids, 0), pad(bids, 1), pad(asks, 0), pad(asks, 1))


def book_day(rows, trades=(), tickers=(), quality=None, runs=None):
    quality = quality or [(0, "connected", "conn=1"), (1, "synced", "update_id=1")]
    frames = synth.tables(
        book_l20=list(rows), trades=list(trades), book_ticker=list(tickers), quality=quality
    )
    return synth.memory_day(frames, runs=runs or [Run("t", 0, 100 * S, None)])


def test_duration():
    assert duration_ns("100ms") == 100_000_000
    assert duration_ns("10s") == 10 * S
    assert duration_ns("1.5m") == 90 * S
    with pytest.raises(ValueError):
        duration_ns("10 parsecs")


def test_ofi_follows_cont_kukanov_stoikov():
    #             bid  qty  ask  qty
    s = np.array(
        [
            [100, 5, 101, 7],  # start
            [100, 8, 101, 7],  # more on the bid: +3
            [100, 8, 102, 4],  # the ask moves away: its old quantity counts, +7
            [99, 6, 102, 4],  # the bid falls: its old quantity goes, -8
            [99, 6, 101, 3],
        ]
    )  # a better ask: its new quantity, -3
    same = np.array([False, True, True, True, True])
    e = flow(s[:, 0], s[:, 1], s[:, 2], s[:, 3], same)
    assert e.tolist() == [0, 3, 7, -8, -3]
    same[2] = False  # a hole before the third state: no flow across it
    assert flow(s[:, 0], s[:, 1], s[:, 2], s[:, 3], same).tolist() == [0, 3, 0, -8, -3]


def test_ofi_over_a_window_from_best_prices():
    states = [
        (100, 5, 101, 7),
        (100, 8, 101, 7),
        (100, 8, 102, 4),
        (99, 6, 102, 4),
        (99, 6, 101, 3),
    ]
    tickers = [(int((1.1 + 0.2 * i) * S), i + 1, *st) for i, st in enumerate(states)]
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3)]
    frames = synth.tables(
        book_l20=rows,
        book_ticker=tickers,
        quality=[(0, "connected", "conn=1"), (1, "synced", "")],
    )
    day = synth.memory_day(frames, qty_step="0.5", runs=[Run("t", 0, 10 * S, None)])
    f = OrderFlowImbalance(windows=("2s",)).transform(day)["ofi_2s"].to_list()
    assert f[-1] == pytest.approx((3 + 7 - 8 - 3) * 0.5)  # (1 s, 3 s]: all five states


def _grid(ts, seg, seg_start):
    n = len(ts)
    z = np.zeros((n, 20), dtype=np.int64)
    book = Book20(np.array(ts), np.zeros(n, np.int64), np.array(seg, np.int32), z, z, z, z)
    return Grid(book=book, seg_start=np.array(seg_start, dtype=np.int64))


def test_windows_look_back_from_t_inclusive_and_only_when_full():
    g = _grid([10, 20, 30, 40], [0, 0, 0, 0], [5, 5, 5, 5])
    out = window_sum(np.array([10, 15, 25, 40]), np.array([1, 2, 4, 8]), g, 20)
    assert np.isnan(out[:2]).all()  # (t - 20, t] starts before the segment
    assert out[2:].tolist() == [6, 12]  # (10, 30]: 15 and 25; (20, 40]: 25 and 40
    hi = rolling_extreme(np.array([3.0, 1.0, 2.0, 5.0]), g, 20, np.maximum)
    assert np.isnan(hi[:2]).all() and hi[2:].tolist() == [2.0, 5.0]  # rows 1..2, rows 2..3
    lo = rolling_extreme(np.array([3.0, 1.0, 2.0, 0.5]), g, 20, np.minimum)
    assert lo[2:].tolist() == [1.0, 0.5]


def test_book_shape_on_a_small_book():
    day = book_day([l20(2 * S, 5, [(100, 3), (99, 1)], [(101, 1), (103, 2)])])
    f = BookShape(levels=(1, 5), bands_bps=(100,)).transform(day).row(0, named=True)
    mid = 100.5
    assert f["mid"] == mid
    assert f["spread_ticks"] == 1
    assert f["wmid_dev_bps"] == pytest.approx((100.75 - mid) / mid * 1e4)
    assert f["imb_1"] == pytest.approx(0.5)
    assert f["imb_5"] == pytest.approx(1 / 7)
    assert f["depth_100bp"] == 4  # within 1 % of 100.5: the bid at 100 and the ask at 101
    assert f["reach_bid_bps"] == pytest.approx(1.5 / mid * 1e4)
    assert f["reach_ask_bps"] == pytest.approx(2.5 / mid * 1e4)


def test_depth_beyond_the_twenty_recorded_levels_is_unknown():
    bids = [(1000 - i, 1) for i in range(20)]  # a full side, all within the band
    day = book_day([l20(2 * S, 5, bids, [(1001, 1)])])
    f = BookShape(bands_bps=(1000,)).transform(day).row(0, named=True)
    assert f["depth_1000bp"] is None


def test_trade_imbalance_uses_the_aggressor():
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3, 4)]
    trades = [  # recv, event, trade time, id, price, qty, buyer was maker
        (int(1.5 * S), 0, 0, 1, 101, 5, False),
        (int(2.5 * S), 0, 0, 2, 100, 3, True),
        (int(3.5 * S), 0, 0, 3, 101, 2, False),
    ]
    f = TradeFlow(windows=("3s",)).transform(book_day(rows, trades))
    assert f["trade_imb_3s"].to_list()[-1] == pytest.approx((7 - 3) / 10)
    assert f["volume_3s"].to_list()[-1] == 10
    assert f["trade_imb_3s"].to_list()[0] is None  # the window starts before the sync


def test_vpin_splits_trades_across_buckets():
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3, 4, 5)]
    trades = [
        (int(1.5 * S), 0, 0, 1, 101, 6, False),  # buy 6
        (int(2.5 * S), 0, 0, 2, 100, 6, True),  # sell 6: 4 in the first bucket, 2 in the second
        (int(3.5 * S), 0, 0, 3, 101, 8, False),  # buy 8
    ]
    v = VPIN(window=2)
    v.bucket_ = 10
    day = book_day(rows, trades)
    close, _, imbalance = v.buckets(day)
    assert close.tolist() == [int(2.5 * S), int(3.5 * S)]
    assert imbalance.tolist() == [2, 6]  # |6 - 4| and |8 - 2|
    out = v.transform(day)["vpin"].to_list()
    assert out[:3] == [None, None, None]
    assert out[3:] == [pytest.approx(0.4)] * 2  # (2 + 6) / (2 * 10), known from 3.5 s on


def test_vpin_splits_a_buy_that_crosses_a_bucket_edge():
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3, 4)]
    trades = [
        (int(1.5 * S), 0, 0, 1, 101, 6, False),  # buy 6
        (int(2.5 * S), 0, 0, 2, 101, 6, False),  # buy 6: 4 fill the first bucket, 2 the next
        (int(3.5 * S), 0, 0, 3, 100, 8, True),  # sell 8
    ]
    v = VPIN(window=2)
    v.bucket_ = 10
    _, _, imbalance = v.buckets(book_day(rows, trades))
    assert imbalance.tolist() == [10, 6]  # |10 - 0| and |2 - 8|


def test_vpin_starts_again_after_a_hole():
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3, 4, 6, 7)]
    trades = [(int((i + 1.5) * S), 0, 0, i, 101, 10, False) for i in range(3)]
    trades += [(int(6.5 * S), 0, 0, 9, 101, 10, False)]
    quality = [
        (0, "connected", "conn=1"),
        (1, "synced", ""),
        (int(4.5 * S), "gap", ""),
        (int(5.5 * S), "resync", ""),
    ]
    v = VPIN(window=2)
    v.bucket_ = 10
    out = v.transform(book_day(rows, trades, quality=quality))["vpin"].to_list()
    assert out[2:4] == [pytest.approx(1.0)] * 2  # two one-sided buckets before the hole
    assert out[4:] == [None, None]  # after it, one bucket so far: not enough


def test_vpin_needs_fit():
    with pytest.raises(NotFitted):
        VPIN().transform(book_day([l20(2 * S, 5, [(100, 1)], [(101, 1)])]))


def test_cancel_share_counts_what_trades_do_not_explain():
    rows = [
        l20(2 * S, 5, [(100, 10), (99, 5)], [(101, 3), (102, 4)]),
        l20(3 * S, 9, [(100, 4), (99, 5)], [(102, 4)]),
    ]
    trades = [(int(2.5 * S), 0, 0, 1, 100, 2, True)]  # a seller takes 2 at 100
    f = CancelShare(windows=("1s",)).transform(book_day(rows, trades))
    # bid 100: 6 left, 2 traded; ask 101: 3 gone with no trade; 7 of 9 unexplained
    assert f["cancel_share_1s"].to_list()[-1] == pytest.approx(7 / 9)


def test_a_level_pushed_below_the_twentieth_is_not_counted_as_cancelled():
    asks = [(200, 1)]
    before = [(100 - i, 1) for i in range(20)]
    after = [(101, 5)] + [(100 - i, 1) for i in range(19)]  # 81 fell out of view
    rows = [l20(2 * S, 5, before, asks), l20(3 * S, 6, after, asks)]
    f = CancelShare(windows=("1s",)).transform(book_day(rows))
    assert f["cancel_share_1s"].to_list()[-1] is None  # nothing known to have decreased


def test_book_rate_counts_update_ids():
    rows = [l20(t * S, 5, [(100, 1)], [(101, 1)]) for t in (1, 2, 3)]
    tickers = [
        (int(1.5 * S), 100, 100, 1, 101, 1),
        (int(1.8 * S), 103, 100, 2, 101, 1),
        (int(2.2 * S), 103, 100, 2, 101, 1),
        (int(2.9 * S), 110, 100, 3, 101, 1),
    ]
    f = Intensity(windows=("2s",)).transform(book_day(rows, tickers=tickers))
    assert f["book_rate_2s"].to_list()[-1] == pytest.approx((3 + 0 + 7) / 2)
    assert f["ticker_rate_2s"].to_list()[-1] == pytest.approx(4 / 2)


def test_no_rate_or_return_reaches_across_a_hole():
    quality = [
        (0, "connected", "conn=1"),
        (1, "synced", ""),
        (int(2.5 * S), "gap", ""),
        (3 * S, "resync", ""),
    ]
    rows = [
        l20(1 * S, 5, [(100, 1)], [(101, 1)]),
        l20(2 * S, 6, [(100, 1)], [(101, 1)]),
        l20(int(3.1 * S), 900, [(200, 1)], [(201, 1)]),  # the mid jumped during the hole
        l20(int(3.6 * S), 901, [(200, 1)], [(201, 1)]),
        # (3.02 s, 4.02 s] is the first full window after the resync at 3 s, and the
        # first update after the hole lies in it
        l20(int(4.02 * S), 902, [(200, 1)], [(201, 1)]),
    ]
    tickers = [
        (int(1.5 * S), 5, 100, 1, 101, 1),
        (int(3.05 * S), 900, 200, 1, 201, 1),  # 895 updates later, across the hole
        (int(3.5 * S), 901, 200, 1, 201, 1),
    ]
    day = book_day(rows, tickers=tickers, quality=quality)
    rate = Intensity(windows=("1s",)).transform(day)["book_rate_1s"].to_list()
    assert rate[-1] == pytest.approx(1.0)  # one update in the window, not 896
    rv = Volatility(windows=("1s",)).transform(day)["rv_1s"].to_list()
    assert rv[-1] == 0.0  # the jump lies across the hole, not in the window


def test_a_missing_level_counts_like_one_that_moved_away():
    from marketlab.features.ofi import level_flows

    rows = np.zeros((2, 20), dtype=np.int64)
    bid_px, bid_qty, ask_px, ask_qty = rows.copy(), rows.copy(), rows.copy(), rows.copy()
    bid_px[:, 0], bid_qty[:, 0] = 100, 5
    ask_px[:, 0], ask_qty[:, 0] = 101, 5
    ask_px[0, 1], ask_qty[0, 1] = 102, 7  # the second ask level is gone in the next row
    bid_px[1, 1], bid_qty[1, 1] = 99, 4  # a second bid level appears
    book = Book20(
        np.array([1, 2]), np.array([1, 2]), np.zeros(2, np.int32), bid_px, bid_qty, ask_px, ask_qty
    )
    e = level_flows(book, 2)
    assert e[1].tolist() == [0, 4 + 7]  # level 2: +4 on the bid, +7 for the ask that left


def test_realized_volatility_of_the_mid():
    mids = [(100, 101), (101, 102), (100, 101), (100, 101)]  # mid 100.5, 101.5, 100.5, 100.5
    rows = [l20((t + 1) * S, 5, [(b, 1)], [(a, 1)]) for t, (b, a) in enumerate(mids)]
    f = Volatility(windows=("3s",)).transform(book_day(rows))
    r = np.log([101.5 / 100.5, 100.5 / 101.5, 1.0])
    assert f["rv_3s"].to_list()[-1] == pytest.approx(np.sqrt((r**2).sum()) * 1e4)
    assert f["range_3s"].to_list()[-1] == pytest.approx(1 / 100.5 * 1e4)


def test_microprice_recovers_a_known_answer():
    # imbalance bucket 1 (bid heavy): up a tick with p 0.3, down with 0.1, else no change;
    # after a move the state is drawn at random. Then E[next move] = (0.3 - 0.1) / 0.4 tick
    # and, the states being symmetric, the micro-price is mid +- 1/2 tick.
    rng = np.random.default_rng(3)
    n = 200_000
    x = rng.integers(0, 2, n)
    u = rng.random(n)
    up = np.where(x == 1, u < 0.3, u < 0.1)
    down = np.where(x == 1, (u >= 0.3) & (u < 0.4), (u >= 0.1) & (u < 0.4))
    dm = up * 1.0 - down * 1.0
    nxt = np.where(dm == 0, x, rng.integers(0, 2, n))
    g = adjustment(x, nxt, dm, n_imb=2, n_spread=1)
    assert g == pytest.approx([-0.5, 0.5], abs=0.02)
    assert g[0] == -g[1]  # symmetrized


def test_microprice_rises_with_the_bid_side(train_day):
    m = MicroPrice().fit([train_day])
    tight = m.adjustment_[:10]  # spread of one tick, imbalance buckets 0..9
    assert (np.diff(tight) > 0).all()
    assert np.abs(tight).max() <= 0.5 + 1e-9  # inside the spread


def test_more_levels_explain_at_least_as_much(train_day):
    r2 = [MultiLevelOFI(levels=k).fit([train_day]).r2_ for k in (1, 3, 10)]
    assert 0 < r2[0] <= r2[1] <= r2[2]


def test_every_method_reports_on_the_grid(train_day, test_day):
    methods = default_methods()
    for m in methods:
        if m.name in FITTED:
            with pytest.raises(NotFitted):
                m.transform(test_day)
            m.fit([train_day])
    table = compute(test_day, methods)
    g = grid(test_day)
    assert table["ts"].to_list() == g.ts.tolist()
    assert len(set(table.columns)) == len(table.columns)
    assert set(METHODS) == {m.name for m in methods}
    # every column has values somewhere
    empty = [c for c in table.columns if table[c].null_count() == table.height]
    # VPIN needs 50 buckets of 1/1000 of a day, more than 5 minutes; and the synthetic book's
    # 20 levels lie one tick apart, so they never reach 0.1 bp: the depth there is unknown
    assert sorted(empty) == ["depth_0p1bp", "depth_0p25bp", "vpin"]


def test_vpin_with_short_buckets_on_synthetic_data(train_day, test_day):
    v = VPIN(buckets_per_day=20_000, window=10).fit([train_day])
    out = v.transform(test_day)["vpin"].to_numpy()
    ok = out[np.isfinite(out)]
    assert len(ok) > 0 and ((ok >= 0) & (ok <= 1)).all()
