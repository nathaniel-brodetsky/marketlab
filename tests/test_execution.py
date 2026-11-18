"""The fill simulator on queues built by hand, where the answer is known.

Common setup, in price and quantity steps of 1: the best bid is 100 with 10 resting, the
best ask 101 with 10. A buy order for 5 at 100 is decided at 1.0 s and, with 100 ms of
latency, joins the queue at 1.1 s behind those 10. The book it joins is the one the depth
update with exchange time 0.4 s shows.
"""

import numpy as np
import polars as pl
import pytest

from marketlab import synth
from marketlab.execution import (
    ASK,
    BID,
    Market,
    Mode,
    Order,
    Outcome,
    frames,
    markouts,
    probe_orders,
    run,
    simulate,
)
from marketlab.execution.market import book_at
from marketlab.io import Run

MS = 1_000_000
S = 1_000_000_000
LAT = 100 * MS


def l20(ts, uid, bids, asks):
    def pad(levels, i):
        v = [lv[i] for lv in levels]
        return v + [0] * (20 - len(v))

    return (ts, uid, pad(bids, 0), pad(bids, 1), pad(asks, 0), pad(asks, 1))


def market(trades=(), depth=(), quality=None, extra_tickers=()):
    """`trades`: (recv ms, trade ms, price, qty, seller_aggressor);
    `depth`: (recv ms, event ms, side, price, qty, update id)."""
    quality = quality or [(0, "connected", "conn=1"), (1, "synced", "update_id=1")]
    depth_rows = [(500 * MS, 400 * MS, 1, 1, 1, 100, 10)]
    depth_rows += [(r * MS, e * MS, u, u, s, p, q) for r, e, s, p, q, u in depth]
    trade_rows = [
        (r * MS, r * MS, t * MS, i + 1, p, q, seller)
        for i, (r, t, p, q, seller) in enumerate(trades)
    ]
    frames_ = synth.tables(
        depth_diff=depth_rows,
        trades=trade_rows,
        book_l20=[l20(1 * S, 1, [(100, 10), (99, 5)], [(101, 10), (102, 5)])],
        book_ticker=[(900 * MS, 1, 100, 10, 101, 10), *extra_tickers],
        quality=quality,
    )
    day = synth.memory_day(frames_, runs=[Run("t", 0, 10 * S, None)])
    return Market.of(day)


def buy(**kw):
    args = {"id": 1, "side": BID, "qty": 5, "decided": 1 * S, "price": 100, "cancel": 5 * S}
    args.update(kw)
    return Order(**args)


def one(m, order, mode=Mode.PESSIMISTIC):
    return simulate(m, [order], LAT, mode)[0]


def test_trades_at_the_price_use_up_the_queue_then_fill():
    m = market(
        trades=[
            (1200, 1150, 100, 4, True),
            (1300, 1250, 100, 4, True),
            (1400, 1350, 100, 5, True),
            (1500, 1450, 100, 2, True),
        ]
    )
    for mode in Mode:
        r = one(m, buy(), mode)
        assert r.queue_at_arrival == 10
        assert r.fills == [(1400 * MS, 100, 3), (1500 * MS, 100, 2)], mode
        assert (r.outcome, r.final) == (Outcome.FILLED, 1500 * MS)


def test_trades_on_the_other_side_or_at_better_prices_do_not_fill():
    m = market(trades=[(1200, 1150, 100, 50, False), (1300, 1250, 101, 50, True)])
    r = one(m, buy())
    assert r.fills == []
    assert (r.outcome, r.final) == (Outcome.CANCELLED, 5 * S + LAT)


def test_cancels_count_ahead_or_behind_by_mode():
    # 10 ahead; 20 more join behind (30); then 12 are cancelled (18); then a trade of 5
    m = market(
        depth=[(1250, 1200, 1, 100, 30, 2), (1450, 1400, 1, 100, 18, 3)],
        trades=[(1600, 1550, 100, 5, True)],
    )
    o = buy(cancel=1700 * MS)  # takes effect at 1.8 s
    pess, prop, opt = (one(m, o, mode) for mode in Mode)
    # all behind: still 10 ahead, 5 after the trade
    assert (pess.outcome, pess.fills, pess.final) == (Outcome.CANCELLED, [], 1800 * MS)
    # a third of the level was ahead: 10 - 12 * 10/30 = 6, then 1
    assert (prop.outcome, prop.fills, prop.final) == (Outcome.CANCELLED, [], 1800 * MS)
    # all ahead: nothing left ahead, the trade fills the order
    assert (opt.outcome, opt.fills, opt.final) == (
        Outcome.FILLED,
        [(1600 * MS, 100, 5)],
        1600 * MS,
    )


def test_no_more_can_be_ahead_than_the_level_holds():
    m = market(depth=[(1250, 1200, 1, 100, 4, 2)], trades=[(1300, 1250, 100, 5, True)])
    for mode in Mode:
        assert one(m, buy(), mode).fills == [(1300 * MS, 100, 1)], mode


def test_a_trade_through_the_price_fills_the_whole_order():
    m = market(trades=[(1300, 1250, 99, 1, True)])
    r = one(m, buy())
    assert r.fills == [(1300 * MS, 100, 5)]
    assert r.outcome == Outcome.FILLED


def test_an_order_can_fill_while_its_cancel_is_on_the_way():
    early = market(trades=[(1250, 1200, 100, 15, True)])
    assert one(early, buy(cancel=1200 * MS)).outcome == Outcome.FILLED  # cancel works at 1.3 s
    late = market(trades=[(1350, 1300, 100, 15, True)])
    r = one(late, buy(cancel=1200 * MS))
    assert (r.outcome, r.final, r.fills) == (Outcome.CANCELLED, 1300 * MS, [])


def test_the_queue_counts_what_was_seen_by_the_arrival_and_nothing_twice():
    m = market(
        trades=[
            (1050, 600, 100, 3, True),  # received before the arrival, newer than the book: gone
            (1200, 300, 100, 9, True),  # received after, but older than the book: already in it
            (1300, 1250, 100, 8, True),
        ]
    )
    r = one(m, buy())
    assert r.queue_at_arrival == 7
    assert r.fills == [(1300 * MS, 100, 1)]


def test_trades_and_cancels_are_told_apart_by_exchange_time():
    # the update showing 10 -> 6 arrives before the trade of 4 that caused it
    m = market(
        depth=[(1300, 1250, 1, 100, 6, 2)],
        trades=[(1350, 1200, 100, 4, True), (1400, 1380, 100, 7, True)],
    )
    r = one(m, buy(cancel=1500 * MS), Mode.OPTIMISTIC)
    # no cancel: 10 ahead, 6 after the first trade, so the second fills 1
    assert r.fills == [(1400 * MS, 100, 1)]


def test_an_order_that_would_cross_is_rejected():
    m = market()
    assert one(m, buy(price=101)).outcome == Outcome.REJECTED
    assert one(m, buy(side=ASK, price=100)).outcome == Outcome.REJECTED


def test_a_hole_ends_what_can_be_known():
    q = [(0, "connected", "conn=1"), (1, "synced", ""), (1500 * MS, "gap", "")]
    m = market(quality=q, trades=[(1600, 1550, 100, 50, True)])
    r = one(m, buy())
    # known to be in a hole from the gap event on
    assert (r.outcome, r.final, r.fills) == (Outcome.HOLE, 1500 * MS, [])


def test_an_order_still_resting_at_the_end_of_the_data_is_open():
    m = Market.of(market().day.truncated(2 * S))
    r = one(m, buy())
    assert (r.outcome, r.final) == (Outcome.OPEN, 2 * S)


def test_aggressive_orders_walk_the_book():
    m = market()
    r = one(m, buy(passive=False, price=None, qty=15, cancel=None))
    assert r.fills == [(1100 * MS, 101, 10), (1100 * MS, 102, 5)]
    assert r.outcome == Outcome.FILLED
    r = one(m, buy(passive=False, price=None, qty=20, cancel=None))
    assert sum(q for _, _, q in r.fills) == 15
    assert r.outcome == Outcome.PARTIAL


def test_the_book_follows_depth_updates_after_the_snapshot():
    m = market(depth=[(1050, 1000, -1, 101, 0, 2), (1060, 1010, 1, 100, 3, 3)])
    b = book_at(m, 1100 * MS)
    assert b.best(ASK) == (102, 5)
    assert b.best(BID) == (100, 3)
    assert b.qty(BID, 50) == 0  # a short side is known all the way down


def test_markouts_are_signed_and_stop_at_the_data():
    ticks = [(1150 * MS, 2, 101, 1, 102, 1), (2 * S, 3, 99, 1, 100, 1)]
    m = market(extra_tickers=ticks)
    buy_ = buy(passive=False, price=None, qty=1, cancel=None)
    sell = Order(2, ASK, 1, 1 * S, passive=False)
    _, fills = frames(simulate(m, [buy_, sell], LAT), "aggressive")
    f = markouts(m, fills).sort("side", descending=True)
    b, s = f.row(0, named=True), f.row(1, named=True)
    assert b["price"] == 101 and s["price"] == 100
    assert b["markout_100ms"] == pytest.approx((101.5 - 101) / 101 * 1e4)
    assert b["markout_1s"] == pytest.approx((99.5 - 101) / 101 * 1e4)
    assert s["markout_1s"] == pytest.approx(-(99.5 - 100) / 100 * 1e4)
    assert b["markout_10s"] is None  # 11.1 s is after the end of the data


def test_markouts_never_reach_across_a_hole():
    q = [
        (0, "connected", "conn=1"),
        (1, "synced", ""),
        (1500 * MS, "gap", ""),
        (2000 * MS, "resync", ""),
    ]
    m = market(quality=q)  # one best price, at 0.9 s, before the hole
    mid = m.mid_at(np.array([1200 * MS, 1600 * MS, 2100 * MS, 2100 * MS]), np.array([0, 0, 0, 1]))
    assert mid[0] == 100.5
    assert np.isnan(mid[1:]).all()  # in the hole; past its segment; no price seen in segment 1


def test_probes_and_the_summary_on_a_synthetic_day():
    day = synth.day(synth.Scenario(seed=7, seconds=120, gaps=[(40, 3)]))
    m = Market.of(day)
    probes = probe_orders(m, "5s", 2000, "5s")
    assert probes and all(o.cancel == o.decided + 5 * S for o in probes)
    assert {o.side for o in probes} == {BID, ASK}
    orders, fills = run(day, every="5s", qty=2000, cancel_after="5s", latency="200ms")
    by_mode = {mo: orders.filter(pl.col("mode") == mo)["filled"].sum() for mo in Mode}
    # cancels ahead can only help: more fills from pessimistic to optimistic
    assert by_mode[Mode.PESSIMISTIC] <= by_mode[Mode.PROPORTIONAL] <= by_mode[Mode.OPTIMISTIC]
    assert fills.height > 0
    assert set(orders["outcome"].unique()) <= {str(o) for o in Outcome}


def test_the_command_takes_its_latency_from_the_manifest(tmp_path, capsys):
    import json

    from marketlab.cli import main

    d = synth.write(synth.Scenario(seconds=60), tmp_path)
    assert main(["execution", str(d)]) == 2  # no round trips recorded: say so
    assert "give --latency" in capsys.readouterr().err
    m = json.loads((d / "manifest.json").read_text())
    m.update(rtt_p50_ms=331, lag_p50_ms=138)
    (d / "manifest.json").write_text(json.dumps(m))
    assert main(["execution", str(d), "--every", "5s"]) == 0
    out = capsys.readouterr().out
    assert "latency     303.5ms (RTT p50 331 ms / 2 + lag p50 138 ms" in out
    assert "pessimistic" in out and "optimistic" in out
    assert main(["execution", str(d), "--latency", "50ms", "--aggressive"]) == 0
    assert "aggressive" in capsys.readouterr().out
