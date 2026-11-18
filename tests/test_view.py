"""The Market screen's data: a stretch read alone equals the same rows of the whole day, holes
and late parts are where the journal puts them, nothing bridges a hole, and the replay
shows the book as it stood at each moment."""

from pathlib import Path

import numpy as np
import pytest

from marketlab import synth
from marketlab.features.base import grid
from marketlab.features.ofi import MultiLevelOFI
from marketlab.io.contract import TABLES
from marketlab.io.day import Day
from marketlab.targets import returns_on
from marketlab.view import market as mk

SEC = 1_000_000_000
SCENARIO = synth.Scenario(seed=5, seconds=420.0, gaps=((100.0, 5.0),), late=((300.0, 3.0),))
T0 = SCENARIO.start


@pytest.fixture(scope="module")
def path(tmp_path_factory) -> Path:
    return synth.write(SCENARIO, tmp_path_factory.mktemp("view"))


@pytest.fixture(scope="module")
def s(path) -> mk.Stretch:
    return mk.stretch(path, T0 + 60 * SEC, T0 + 360 * SEC)


def test_a_window_reads_the_same_rows_as_the_whole_day(path):
    lo, hi = T0 + 90 * SEC, T0 + 200 * SEC
    whole, part = Day.open(path), Day.open(path, between=(lo, hi))
    for table in TABLES:
        a, b = whole.raw(table), part.raw(table)
        if table == "quality":
            assert a.equals(b), "the journal is read whole"
            continue
        assert b.equals(a.filter((a["recv_ts"] >= lo) & (a["recv_ts"] < hi))), table
        assert len(b) < len(a) or len(a) == 0, table
    assert part.intervals == whole.intervals
    assert Day.open(path, between=(0, 1)).raw("trades").is_empty()


def test_holes_and_late_parts_are_where_the_journal_puts_them(s):
    assert s.holes == [(T0 + 100 * SEC, T0 + 105 * SEC)]
    assert s.late == [(T0 + 300 * SEC, T0 + 303 * SEC)]
    shown = s.book.ts[s.book.ts >= s.lo]
    assert not ((shown >= T0 + 100 * SEC) & (shown < T0 + 105 * SEC)).any()
    assert ((shown >= T0 + 300 * SEC) & (shown < T0 + 303 * SEC)).any(), "late rows are shown"
    assert (s.book.ts < s.hi).all() and (s.trades.ts >= s.lo).all()


def test_the_window_computes_what_the_whole_day_computes(path, s):
    # the order flow of the last second and the next second's return need data from outside
    # the stretch: read with its pads, they equal the whole day's values at the same rows
    day = Day.open(path, include_late=True)
    g = grid(day)
    ofi = MultiLevelOFI(levels=1, window="1s").level_windows(day, g)[:, 0]
    _, ends = day.segment_bounds()
    ret = returns_on(g, ends, ["1s"])["ret_1s"]
    rows = np.searchsorted(g.ts, s.book.ts)
    assert np.array_equal(g.ts[rows], s.book.ts)
    assert np.allclose(ofi[rows], s.ofi, equal_nan=True)
    assert np.allclose(ret[rows], s.next_return, equal_nan=True)
    first = int(np.searchsorted(s.book.ts, s.lo))
    assert np.isfinite(s.ofi[first]) and np.isfinite(s.next_return[-1])


def test_the_next_second_reaches_past_the_end_of_the_stretch(path):
    # a stretch that ends right after a row whose next second moves the mid: that return
    # needs rows after the end, which the stretch reads but does not show
    day = Day.open(path, include_late=True)
    g = grid(day)
    _, ends = day.segment_bounds()
    ret = returns_on(g, ends, ["1s"])["ret_1s"]
    j = int(
        np.flatnonzero((ret[:-1] != 0) & np.isfinite(ret[:-1]) & (g.ts[:-1] > T0 + 200 * SEC))[0]
    )
    s = mk.stretch(path, T0 + 150 * SEC, int(g.ts[j + 1]))
    assert s.book.ts[-1] == g.ts[j]
    assert s.next_return[-1] == pytest.approx(ret[j]) and ret[j] != 0


def test_each_bin_shows_its_last_row_and_a_hole_shows_nothing(s):
    edges, rows = mk.bins(s, columns=600)
    assert len(rows) == 600 and edges[0] == s.lo and edges[-1] == s.hi
    for k in np.flatnonzero(rows >= 0):
        inside = np.flatnonzero((s.book.ts >= edges[k]) & (s.book.ts < edges[k + 1]))
        assert rows[k] == inside[-1]
    hole = (edges[:-1] >= T0 + 100 * SEC) & (edges[1:] <= T0 + 105 * SEC)
    assert hole.sum() >= 9 and (rows[hole] == -1).all()
    _, fine = mk.bins(s, columns=10**6)
    assert len(fine) == (s.hi - s.lo) // (100 * 1_000_000), "no finer than 100 ms"


def test_lines_and_heat_are_the_rows_values_in_units(s):
    _, rows = mk.bins(s, columns=300)
    ln, z = mk.lines(s, rows), mk.heat(s, rows)
    b = s.book
    for k in np.flatnonzero(rows >= 0)[:50]:
        r = rows[k]
        mid = (b.bid_px[r, 0] + b.ask_px[r, 0]) / 2 * s.tick
        assert ln["mid"][k] == pytest.approx(mid, abs=1e-6)  # a tick is 0.01
        assert ln["spread"][k] == b.ask_px[r, 0] - b.bid_px[r, 0]
        bq, aq = b.bid_qty[r, 0], b.ask_qty[r, 0]
        weighted = (b.bid_px[r, 0] * aq + b.ask_px[r, 0] * bq) / (bq + aq) * s.tick
        assert ln["weighted_mid"][k] == pytest.approx(weighted, abs=1e-6)
        assert ln["bid_depth"][k] == pytest.approx(b.bid_qty[r].sum() * s.unit)
        assert ln["ask_depth"][k] == pytest.approx(b.ask_qty[r].sum() * s.unit)
        q = np.concatenate([b.bid_qty[r][::-1], b.ask_qty[r]]) * s.unit
        assert np.array_equal(np.where(q == 0, np.nan, q), z[:, k], equal_nan=True)
        assert z[19, k] == b.bid_qty[r, 0] * s.unit and z[20, k] == b.ask_qty[r, 0] * s.unit
    empty = rows < 0
    assert np.isnan(ln["mid"][empty]).all() and np.isnan(z[:, empty]).all()


def test_the_replay_shows_the_book_as_it_stood(s):
    step = 100_000_000
    frames = mk.replay(s, T0 + 95 * SEC, 15, step)
    assert len(frames) == 150
    b, t = s.book, s.trades
    back = b.ts[b.ts >= T0 + 105 * SEC][0]  # the book is known again from its first row
    for f in frames:
        hole = T0 + 100 * SEC <= f.ts < back
        assert (f.row == -1) == hole, f.ts
        if hole:
            assert len(f.bid_qty) == 0 and np.isnan(f.mid)
            continue
        assert b.ts[f.row] <= f.ts and (f.row + 1 == len(b) or b.ts[f.row + 1] > f.ts)
        mid = (b.bid_px[f.row, 0] + b.ask_px[f.row, 0]) / 2
        assert f.bid_ticks[0] == b.bid_px[f.row, 0] - mid and f.bid_ticks[0] < 0 < f.ask_ticks[0]
        k = (t.ts > f.ts - step) & (t.ts <= f.ts)
        assert np.allclose(f.trade_qty, t.qty[k]) and np.array_equal(f.trade_buy, t.buy[k])


def test_a_replay_from_the_start_knows_the_book_already(s):
    f = mk.replay(s, s.lo, 1)[0]
    assert f.row == 0 and s.book.ts[0] < s.lo, "the last row before the stretch"
    _, rows = mk.bins(s, columns=300)
    assert 0 not in rows, "but no bin shows it"


def test_a_line_through_the_path_breaks_at_the_hole(s):
    ts, mid = mk.path(s, T0 + 95 * SEC, T0 + 115 * SEC)
    gap = np.flatnonzero(np.isnan(mid))
    assert len(gap) == 1
    assert ts[gap[0] - 1] < T0 + 100 * SEC and ts[gap[0] + 1] >= T0 + 105 * SEC
    assert np.all(np.diff(ts) > 0)


def test_thinning_keeps_the_largest_trades_in_time_order():
    t = mk.Trades(
        ts=np.arange(6), price=np.ones(6), qty=np.array([5.0, 1, 9, 2, 7, 3]), buy=np.ones(6, bool)
    )
    kept, total = mk.thin_trades(t, at_most=3)
    assert total == 6 and kept.ts.tolist() == [0, 2, 4]
    assert mk.thin_trades(t, at_most=6)[0] is t


def test_the_app_opens_on_recorded_days(path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("plotly")
    monkeypatch.setenv("MARKETLAB_DATA", str(path.parents[2]))
    app = Path(__file__).resolve().parents[1] / "app" / "main.py"
    at = testing.AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception, at.exception
    assert "Рынок" in at.title[0].value
    assert "без валидного стакана" in at.caption[0].value
