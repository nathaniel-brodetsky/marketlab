"""The market-making rule through the simulator: each order is priced by the rule and fills
as the simulator gives it alone, the position stays within its limit, the accounts add up,
a day in pieces gives what the whole day gives, and no decision depends on what came after
it. At the best prices and away from them, where some orders rest at levels the book does
not show; with the order-flow signal, the volatility gate, or both."""

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
#: The volatility gate over 5 s against its median over the minute before: high about half
#: of the time where it is defined, from some 35 s into each segment.
GATE = st.Quoting(size=SIZE, max_position=2, vol_window="5s", vol_lookback="60s", vol_quantile=0.5)
BOTH = replace(SIGNAL, vol_window="5s", vol_lookback="60s", vol_quantile=0.5)
#: About 6 steps from the mid at 97000.00: away from the best prices, filled at times.
AWAY = st.Quoting(size=SIZE, max_position=2, offset_bp=0.006)
#: About 22 steps: past the 20 levels the book shows, often. Unseen levels filling are
#: tested on queues built by hand in `test_execution.py`.
DEEP = st.Quoting(size=SIZE, max_position=2, offset_bp=0.023)


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    s = synth.Scenario(seed=7, seconds=600.0, trade_rate=12.0, gaps=((300.0, 4.0),))
    return synth.write(s, tmp_path_factory.mktemp("strategy"))


@pytest.fixture(scope="module")
def busy(tmp_path_factory):
    """As many trades as book changes: the price moves enough to reach orders away from it."""
    s = synth.Scenario(seed=7, seconds=600.0, trade_rate=20.0, book_rate=20.0, gaps=((300.0, 4.0),))
    return synth.write(s, tmp_path_factory.mktemp("busy"))


@pytest.fixture(scope="module")
def results(path):
    return {q: st.simulate_day(path, q, chunk=None) for q in (PLAIN, SIGNAL, GATE, BOTH)}


@pytest.fixture(scope="module")
def away(busy):
    return {q: st.simulate_day(busy, q, chunk=None) for q in (AWAY, DEEP)}


def steps(r: st.DayResult, day: Day) -> np.ndarray:
    return np.rint(r.fills["qty"].to_numpy() / float(day.qty_step)).astype(np.int64)


def check_orders(r: st.DayResult, day: Day, q: st.Quoting, min_fills: int = 50) -> None:
    m = Market.of(day)
    qty = int(q.size / day.qty_step)
    lat, life = duration_ns(q.latency), duration_ns(q.life)
    assert len(r.orders) > 80 and len(r.fills) >= min_fills
    by_id = {k[0]: g for k, g in r.fills.group_by("id")}
    tick = float(day.price_step)
    for row in r.orders.iter_rows(named=True):
        o = Order(row["id"], row["side"], qty, row["decided"], True, row["price"], row["cancel"])
        alone = simulate(m, [o], lat, unseen="behind")[0]
        assert str(alone.outcome) == row["outcome"]
        rested = alone.final > alone.arrival and row["outcome"] not in ("rejected", "unknown")
        assert row["unseen"] == (rested and alone.queue_at_arrival is None)
        got = by_id.get(row["id"], pl.DataFrame(schema=st.FILL_SCHEMA))
        assert got["ts"].to_list() == [f[0] for f in alone.fills]
        assert np.allclose(got["price"].to_numpy(), [f[1] * tick for f in alone.fills])
        assert row["filled"] == sum(f[2] for f in alone.fills)
        # priced from the best prices last seen, and out within its life
        t = m.ticker
        i = int(np.searchsorted(t["recv_ts"], row["decided"], side="right")) - 1
        bid, ask = int(t["bid_px"][i]), int(t["ask_px"][i])
        assert row["price"] == st.quote_price(row["side"], bid, ask, q.offset_bp)
        if q.offset_bp == 0:
            assert row["price"] == (bid if row["side"] == BID else ask)
        assert row["decided"] < row["cancel"] <= row["decided"] + life
    # one order a side at a time: the next is decided once the last is done
    for side in (BID, ASK):
        o = r.orders.filter(pl.col("side") == side).sort("decided")
        assert o["decided"].is_unique().all()


RULES = pytest.mark.parametrize(
    "q", [PLAIN, SIGNAL, GATE, BOTH], ids=["plain", "signal", "gate", "both"]
)


@RULES
def test_each_order_fills_as_the_simulator_gives_it_alone(path, results, q):
    check_orders(results[q], Day.open(path), q)


@pytest.mark.parametrize("q", [AWAY, DEEP], ids=["away", "deep"])
def test_orders_away_from_the_best_prices(busy, away, q):
    r, day = away[q], Day.open(busy)
    check_orders(r, day, q, min_fills=20 if q is AWAY else 0)
    t = Market.of(day).ticker
    i = np.searchsorted(t["recv_ts"], r.orders["decided"].to_numpy(), side="right") - 1
    best = np.where(r.orders["side"].to_numpy() == BID, t["bid_px"][i], t["ask_px"][i])
    assert (r.orders["price"].to_numpy() != best).mean() > 0.9
    at_best = st.simulate_day(busy, replace(q, offset_bp=0.0), chunk=None)
    assert at_best.unseen == 0 and r.fills.height < at_best.fills.height
    if q is DEEP:
        hidden = r.orders.filter(pl.col("unseen"))
        assert r.unseen > 0.3 and r.unseen == pytest.approx(len(hidden) / len(r.orders))
    # in pieces and cut short, as at the best prices
    part = st.simulate_day(busy, q, chunk="2m")
    assert part.orders.equals(r.orders) and part.fills.equals(r.fills)
    cut = day.intervals[0][0] + 200 * SEC
    short = st.simulate_in(day.truncated(cut), q)
    done = pl.col("decided") + duration_ns(q.life) + 2 * duration_ns(q.latency) < cut
    assert r.orders.filter(done).equals(short.orders.filter(done))
    early = pl.col("decided") < cut
    keys = ["id", "side", "decided", "price"]
    assert r.orders.filter(early).select(keys).equals(short.orders.filter(early).select(keys))


def test_quote_prices():
    assert [st.quote_price(s, 100, 101, 0) for s in (BID, ASK)] == [100, 101]
    assert [st.quote_price(s, 100, 101, 100) for s in (BID, ASK)] == [99, 102]  # 1% of 100.5
    assert [st.quote_price(s, 90, 110, 1) for s in (BID, ASK)] == [90, 110]  # never inside
    assert [st.quote_price(s, 9_700_000, 9_700_001, 1) for s in (BID, ASK)] == [
        9_699_030,  # floor(9_700_000.5 * 0.9999) = floor(9_699_030.45)
        9_700_971,  # ceil(9_700_000.5 * 1.0001) = ceil(9_700_970.55)
    ]


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


@RULES
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


def test_pieces_read_far_enough_for_fills_while_a_cancel_is_on_its_way(tmp_path):
    # a slow line: an order decided near the end of its piece is still in the book for a
    # latency after its life, so a piece read for `life` past its end misses its fills
    s = synth.Scenario(seed=12, seconds=600.0, trade_rate=2.0)
    path = synth.write(s, tmp_path)
    q = st.Quoting(size=SIZE, life="10s", latency="5s", max_position=2)
    whole = st.simulate_day(path, q, chunk=None)
    part = st.simulate_day(path, q, chunk="5s")
    step, life = 5 * SEC, 10 * SEC
    past = whole.fills.join(whole.orders, on=["id", "side"]).filter(
        pl.col("ts") > (pl.col("decided") // step + 1) * step + life
    )
    assert len(past) >= 2, "no fill later than its piece's end and a life: the test shows nothing"
    assert part.orders.equals(whole.orders)
    assert part.fills.equals(whole.fills)


@RULES
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


def _blockers(path, q: st.Quoting) -> list[st._Signal]:
    day, out = Day.open(path), []
    if q.window:
        out.append(st._Signal.ofi(maker.signal(day, [q.window]), q.window, q.threshold))
    if q.vol_window:
        out.append(st._Signal.gate(st.vol_gate(day, q.vol_window, q.vol_lookback, q.vol_quantile)))
    return out


def _against_at(s: st._Signal, side: int, t: int) -> bool:
    i = s.row(t)
    return i >= 0 and bool(s.against[side][i])


@pytest.mark.parametrize("q", [SIGNAL, GATE, BOTH], ids=["signal", "gate", "both"])
def test_orders_stay_out_while_a_signal_or_the_gate_is_against_them(path, results, q):
    r = results[q]
    blockers = _blockers(path, q)
    life = duration_ns(q.life)
    pulled = {id(s): 0 for s in blockers}
    for row in r.orders.iter_rows(named=True):
        side, t, cancel = row["side"], row["decided"], row["cancel"]
        for s in blockers:
            assert not _against_at(s, side, t), "placed while a signal or the gate was against"
        # withdrawn at the first row of any of them against it before its life is out
        first = [
            int(s.ts[i])
            for s in blockers
            for i in np.flatnonzero((s.ts > t) & (s.ts <= t + life) & s.against[side])[:1]
        ]
        assert cancel == min([*first, t + life])
        for s in blockers:
            if cancel < t + life and _against_at(s, side, cancel):
                pulled[id(s)] += 1
    assert all(n > 10 for n in pulled.values()), pulled
    # a threshold no value reaches is no signal at all
    never = replace(SIGNAL, threshold=1e12)
    assert st.simulate_day(path, never, chunk=None).orders.equals(results[PLAIN].orders)


def test_the_gate_by_hand(path):
    day = Day.open(path)
    g = st.vol_gate(day, "5s", "60s", 0.5)
    tr = day.valid("trades", ["recv_ts", "price"])
    tts, px, seg = (tr[c].to_numpy() for c in ("recv_ts", "price", "segment"))
    assert len(day.intervals) == 2
    for k, (s, e) in enumerate(day.intervals):
        rows = g.filter(pl.col("segment") == k)
        assert rows["ts"].to_list() == list(range(s, e, SEC))
        assert (rows["seg_end"] == e).all()

    def last_price(t, k):
        mine = (seg == k) & (tts <= t)
        return float(px[mine][-1]) if mine.any() else np.nan

    def rv(t, k):
        p = [last_price(t - j * SEC, k) for j in range(5, -1, -1)]
        r = np.log(np.array(p[1:]) / np.array(p[:-1]))
        return float(np.sqrt(np.nansum(r * r)) * 1e4)

    rng = np.random.default_rng(3)
    for i in [*rng.choice(len(g), 40, replace=False).tolist(), 0, 4, 5, 34, 35, 36]:
        row = g.row(i, named=True)
        t, k = row["ts"], row["segment"]
        s = day.intervals[k][0]
        if t - 5 * SEC < s:
            assert row["rv"] is None
        else:
            assert row["rv"] == pytest.approx(rv(t, k), rel=1e-9, abs=1e-12)
        before = [rv(u, k) for u in range(max(t - 60 * SEC, s + 5 * SEC), t, SEC)]
        if len(before) < 30:
            assert row["limit"] is None and not row["high"]
        else:
            assert row["limit"] == pytest.approx(np.quantile(before, 0.5), rel=1e-9)
            assert row["high"] == (row["rv"] > row["limit"])
    defined = g.filter(pl.col("limit").is_not_null())
    assert 0.3 < defined["high"].mean() < 0.7 and len(defined) > 400


def test_the_gate_is_high_only_above_its_limit(tmp_path):
    # a quiet market: the price often holds for five seconds, and its volatility, nil, is
    # then at a limit of nil too, which is not above it
    quiet = synth.write(synth.Scenario(seed=11, seconds=600.0, trade_rate=0.5), tmp_path)
    g = st.vol_gate(Day.open(quiet), "5s", "60s", 0.5).filter(pl.col("limit").is_not_null())
    at = g.filter(pl.col("rv") == pl.col("limit"))
    assert len(at) > 20 and not at["high"].any()
    assert g["high"].to_list() == (g["rv"] > g["limit"]).to_list()


def test_the_gate_reads_nothing_after_its_time(path):
    day = Day.open(path)
    full = st.vol_gate(day, "5s", "60s", 0.8)
    for cut_s in (61.5, 302.0, 451.0):
        short = st.vol_gate(day.truncated(day.intervals[0][0] + int(cut_s * SEC)), "5s", "60s", 0.8)
        assert len(short) > 50
        assert (
            full.filter(pl.col("ts") <= short["ts"].max())
            .drop("seg_end")
            .equals(short.drop("seg_end"))
        )


def test_a_gate_never_defined_is_no_gate(path, results):
    # half of an hour's lookback never fits in a segment of five minutes
    never = replace(GATE, vol_lookback="1h")
    g = st.vol_gate(Day.open(path), "5s", "1h", 0.5)
    assert g["limit"].null_count() == len(g) and not g["high"].any()
    assert st.simulate_day(path, never, chunk=None).orders.equals(results[PLAIN].orders)
    assert results[GATE].quoted["bid"] < results[PLAIN].quoted["bid"]


def test_quoting_is_checked(path):
    with pytest.raises(ValueError, match="max_position"):
        st.Quoting(max_position=0)
    with pytest.raises(ValueError, match="threshold"):
        st.Quoting(threshold=-1)
    with pytest.raises(ValueError, match="offset_bp"):
        st.Quoting(offset_bp=-0.5)
    with pytest.raises(ValueError, match="vol_quantile"):
        st.Quoting(vol_window="1m", vol_quantile=1.5)
    with pytest.raises(ValueError, match="vol_window must be whole seconds"):
        st.Quoting(vol_window="1500ms")
    with pytest.raises(ValueError, match="vol_lookback must be whole seconds"):
        st.Quoting(vol_window="1m", vol_lookback="1s")
    st.Quoting(vol_lookback="1s")  # no gate: its lookback is not read
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
    out = tmp_path / "runs" / "days.csv"
    code = main(
        ["strategy", "--data", str(data), "--min-hours", "0", "--signal", "1s",
         "--modes", "pessimistic", "optimistic", "--offsets", "0", "0.01",
         "--fees", "-1", "0", "2", "--by-day", "--out", str(out)]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    assert "0.01 bp from the mid, optimistic queue" in text and "at the best prices" in text
    assert "be_bp" in text and "fee -1" in text and "fee 2" in text
    t = pl.read_csv(out)
    assert t["mode"].to_list() == ["pessimistic", "optimistic"] * 2
    assert t["offset_bp"].to_list() == [0.0, 0.0, 0.01, 0.01]
    assert main(["strategy", "--data", str(data), "--min-hours", "0"]) == 0
    assert "from the mid, pessimistic" not in capsys.readouterr().out, "days only with --by-day"
    assert main(["strategy", "--data", str(data), "--min-hours", "30"]) == 2
    gated = tmp_path / "gated.csv"
    code = main(
        ["strategy", "--data", str(data), "--min-hours", "0", "--vol-window", "5s",
         "--vol-lookback", "60s", "--vol-quantiles", "none", "0.5", "--by-day",
         "--out", str(gated)]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    assert "above its 0.5 quantile over the 60s before" in text
    assert "volatility gate 0.5, pessimistic queue" in text
    t = pl.read_csv(gated)
    assert t["vol_quantile"].to_list() == [None, 0.5]
    assert t["quoted_bid"][1] < t["quoted_bid"][0]
    alone = ["strategy", "--data", str(data), "--min-hours", "0", "--vol-quantiles", "0.5"]
    assert main(alone) == 2, "a quantile without --vol-window"
