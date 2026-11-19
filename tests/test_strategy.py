"""The market-making rule through the simulator: each order fills as the simulator gives it
alone, the position stays within its limit, the accounts add up, a day in pieces gives what
the whole day gives, and no decision depends on what came after it."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from marketlab import maker, synth
from marketlab import strategy as st
from marketlab.cli import main
from marketlab.execution import Market, Order, simulate
from marketlab.execution.market import ASK, BID
from marketlab.features import duration_ns
from marketlab.io.day import Day
from marketlab.io.intervals import segment_ids

SEC = 1_000_000_000
SIZE = Decimal("0.001")
PLAIN = st.Quoting(size=SIZE, max_position=2)
SIGNAL = st.Quoting(size=SIZE, max_position=2, window="1s", threshold=0.5)


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    s = synth.Scenario(seed=7, seconds=600.0, trade_rate=12.0, gaps=((300.0, 4.0),))
    return synth.write(s, tmp_path_factory.mktemp("strategy"))


@pytest.fixture(scope="module")
def results(path):
    return {q: st.simulate_day(path, q, chunk=None) for q in (PLAIN, SIGNAL)}


def steps(r: st.DayResult, day: Day) -> np.ndarray:
    return np.rint(r.fills["qty"].to_numpy() / float(day.qty_step)).astype(np.int64)


@pytest.mark.parametrize("q", [PLAIN, SIGNAL], ids=["plain", "signal"])
def test_each_order_fills_as_the_simulator_gives_it_alone(path, results, q):
    r, day = results[q], Day.open(path)
    m = Market.of(day)
    qty = int(q.size / day.qty_step)
    lat, life = duration_ns(q.latency), duration_ns(q.life)
    assert len(r.orders) > 100 and len(r.fills) > 50
    by_id = {k[0]: g for k, g in r.fills.group_by("id")}
    tick = float(day.price_step)
    for row in r.orders.iter_rows(named=True):
        o = Order(row["id"], row["side"], qty, row["decided"], True, row["price"], row["cancel"])
        alone = simulate(m, [o], lat)[0]
        assert str(alone.outcome) == row["outcome"]
        got = by_id.get(row["id"], pl.DataFrame(schema=st.FILL_SCHEMA))
        assert got["ts"].to_list() == [f[0] for f in alone.fills]
        assert np.allclose(got["price"].to_numpy(), [f[1] * tick for f in alone.fills])
        assert row["filled"] == sum(f[2] for f in alone.fills)
        # placed at the best price on its side as last seen, and out within its life
        t = m.ticker
        i = int(np.searchsorted(t["recv_ts"], row["decided"], side="right")) - 1
        assert row["price"] == (t["bid_px"][i] if row["side"] == BID else t["ask_px"][i])
        assert row["decided"] < row["cancel"] <= row["decided"] + life
    # one order a side at a time: the next is decided once the last is done
    for side in (BID, ASK):
        o = r.orders.filter(pl.col("side") == side).sort("decided")
        assert o["decided"].is_unique().all()


def test_the_position_stays_within_its_limit_and_each_decision_knew_it(path, results):
    day = Day.open(path)
    qty = int(SIZE / day.qty_step)
    at_limit = 0
    for q, r in results.items():
        limit = q.max_position * qty
        assert r.fills["ts"].is_sorted()
        ts = r.fills["ts"].to_numpy()
        signed = r.fills["side"].to_numpy().astype(np.int64) * steps(r, day)
        held = np.cumsum(signed)
        assert np.abs(held).max() <= limit
        assert r.max_position == pytest.approx(np.abs(held).max() * float(day.qty_step))
        at_limit += int((np.abs(held) == limit).sum())
        for row in r.orders.iter_rows(named=True):
            known = int(signed[ts <= row["decided"]].sum())
            assert row["side"] * known + qty <= limit, "an order that could pass the limit"
    assert at_limit > 0, "the limit never bound: the test shows nothing"


def test_the_accounts_add_up(path, results):
    day = Day.open(path)
    b = day.valid("book_ticker", ["recv_ts", "bid_px", "ask_px"]).sort("recv_ts").tail(1)
    last_mid = float((b["bid_px"][0] + b["ask_px"][0]) / 2) * float(day.price_step)
    m = Market.of(day)
    for r in results.values():
        f = r.fills
        side, price, qty = (f[c].to_numpy().astype(float) for c in ("side", "price", "qty"))
        held = float((side * qty).sum())
        assert r.last_mid == pytest.approx(last_mid)
        assert r.end_position == pytest.approx(held)
        assert r.gross == pytest.approx(-(side * price * qty).sum() + held * last_mid)
        assert r.notional == pytest.approx((price * qty).sum())
        # captured: each fill against the mid seen at its time
        seg = segment_ids(f["ts"].to_numpy(), day.intervals)
        mid = m.mid_at(f["ts"].to_numpy(), seg) * float(day.price_step)
        assert np.allclose(f["mid"].to_numpy(), mid)
        assert r.captured == pytest.approx((side * (mid - price) * qty).sum())
        assert r.captured + r.inventory == pytest.approx(r.gross)
        assert r.net(r.per_notional_bp) == pytest.approx(0.0, abs=1e-9)
        assert r.net(-1.0) == pytest.approx(r.gross + r.notional * 1e-4)
        assert 0 < r.quoted["bid"] <= 1 and 0 < r.quoted["ask"] <= 1


@pytest.mark.parametrize("q", [PLAIN, SIGNAL], ids=["plain", "signal"])
def test_a_day_in_pieces_gives_what_the_whole_day_gives(path, results, q):
    # pieces of two minutes cross the gap, the segment ends and the day's end
    whole = results[q]
    part = st.simulate_day(path, q, chunk="2m")
    assert part.orders.equals(whole.orders)
    assert part.fills.equals(whole.fills)
    assert (part.gross, part.last_mid, part.quoted) == (whole.gross, whole.last_mid, whole.quoted)
    opened = st.simulate_in(Day.open(path), q)
    assert opened.orders.equals(whole.orders) and opened.gross == whole.gross


def test_pieces_read_far_enough_for_orders_that_rest_long(tmp_path):
    # a quiet market and long-lived orders: many rest across the end of their piece
    s = synth.Scenario(seed=11, seconds=600.0, trade_rate=0.5)
    path = synth.write(s, tmp_path)
    q = st.Quoting(size=SIZE, life="40s", max_position=2)
    whole = st.simulate_day(path, q, chunk=None)
    part = st.simulate_day(path, q, chunk="15s")
    late = whole.fills.join(whole.orders, on=["id", "side"]).filter(
        pl.col("ts") > pl.col("decided") + 20 * SEC
    )
    assert len(late) >= 2, "no order filled long after its decision: the test shows nothing"
    assert part.orders.equals(whole.orders)
    assert part.fills.equals(whole.fills)


@pytest.mark.parametrize("q", [PLAIN, SIGNAL], ids=["plain", "signal"])
@pytest.mark.parametrize("cut_s", [150.3, 301.0, 452.7])
def test_no_decision_depends_on_what_came_after_it(path, results, q, cut_s):
    day = Day.open(path)
    cut = day.intervals[0][0] + int(cut_s * SEC)
    full = results[q]
    short = st.simulate_in(day.truncated(cut), q)
    keys = ["id", "side", "decided", "price"]
    a = full.orders.filter(pl.col("decided") < cut)
    b = short.orders.filter(pl.col("decided") < cut)
    assert len(a) > 20
    assert a.select(keys).equals(b.select(keys)), "a decision changed with the data after it"
    # orders over before the cut: everything the same, fills too
    done = pl.col("decided") + duration_ns(q.life) + 2 * duration_ns(q.latency) < cut
    assert full.orders.filter(done).equals(short.orders.filter(done))
    early = pl.col("ts") <= cut
    assert full.fills.filter(early).equals(short.fills.filter(early))


def test_the_signal_keeps_orders_out_while_it_is_against_them(path, results):
    r = results[SIGNAL]
    s = st._Signal(maker.signal(Day.open(path), ["1s"]), "1s", SIGNAL.threshold)
    life = duration_ns(SIGNAL.life)
    pulled = 0
    for row in r.orders.iter_rows(named=True):
        side, t, cancel = row["side"], row["decided"], row["cancel"]
        i = s.row(t)
        assert i < 0 or not s.against[side][i], "placed while the signal was against it"
        rows = np.flatnonzero((s.ts > t) & (s.ts <= cancel))
        if cancel < t + life:
            pulled += 1
            assert s.ts[rows[-1]] == cancel and s.against[side][rows[-1]]
            assert not s.against[side][rows[:-1]].any()
        else:
            assert not s.against[side][rows].any()
    assert pulled > 10
    # a threshold no value reaches is no signal at all
    never = replace(SIGNAL, threshold=1e12)
    assert st.simulate_day(path, never, chunk=None).orders.equals(results[PLAIN].orders)


def test_quoting_is_checked(path):
    with pytest.raises(ValueError, match="max_position"):
        st.Quoting(max_position=0)
    with pytest.raises(ValueError, match="threshold"):
        st.Quoting(threshold=-1)
    with pytest.raises(ValueError, match="whole quantity steps"):
        st.simulate_day(path, st.Quoting(size=Decimal("0.000001")))


def test_summary_and_many_days(tmp_path, results):
    paths = []
    for k in range(2):
        start = synth.DAY0 + k * 86_400 * SEC + 36_000 * SEC
        paths.append(synth.write(synth.Scenario(seed=40 + k, seconds=240, start=start), tmp_path))
    q = replace(PLAIN, max_position=3)
    alone = st.run(paths, q)
    pooled = st.run(paths, q, workers=2)
    assert [x.date for x in pooled] == [x.date for x in alone]
    assert all(a.fills.equals(b.fills) for a, b in zip(alone, pooled, strict=True))
    s = st.summarize(alone)
    be = [x.per_notional_bp for x in alone]
    assert s.days == 2 and s.break_even_bp == pytest.approx(np.mean(be))
    assert s.break_even_se == pytest.approx(np.std(be, ddof=1) / np.sqrt(2))
    assert s.net_per_day(s.break_even_bp) == pytest.approx(
        np.mean([x.gross for x in alone]) - s.break_even_bp / 1e4 * s.notional_per_day
    )
    assert s.gross_per_day == pytest.approx(np.mean([x.gross for x in alone]))
    assert st.summarize([]).days == 0


def test_the_command(tmp_path, capsys):
    data = tmp_path / "data"
    synth.write(synth.Scenario(seed=9, seconds=240), data)
    out = tmp_path / "days.csv"
    code = main(
        ["strategy", "--data", str(data), "--min-hours", "0", "--signal", "1s",
         "--modes", "pessimistic", "optimistic", "--fees", "-1", "0", "2", "--out", str(out)]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    assert "break-even  maker fee" in text and "fee  -1.00 bp" in text
    assert "optimistic queue" in text
    t = pl.read_csv(out)
    assert t["mode"].to_list() == ["pessimistic", "optimistic"]
    assert main(["strategy", "--data", str(data), "--min-hours", "30"]) == 2
