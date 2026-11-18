"""The rules for where data can be used, one by one."""

import numpy as np

from marketlab.io import Run, segment_ids, valid_intervals


def run(lo=0, hi=999, book=None):
    return Run("test", lo, hi, book)


def iv(events, runs=None, late=False):
    return valid_intervals(events, runs or [run()], include_late=late)


def test_valid_from_sync_until_a_gap_and_again_from_the_resync():
    ev = [(0, "connected", "conn=1"), (10, "synced", ""), (50, "gap", ""), (70, "resync", "")]
    assert iv(ev) == [(10, 50), (70, 1000)]


def test_every_loss_ends_the_interval():
    for kind in ("gap", "invalid_book", "reconnect", "overflow"):
        assert iv([(10, "synced", ""), (40, kind, "")]) == [(10, 40)], kind


def test_a_stop_ends_it_and_a_handover_does_not():
    stop = [(10, "synced", ""), (40, "disconnected", 'conn=1 reason="stopped"')]
    assert iv(stop) == [(10, 40)]
    handover = [
        (10, "synced", ""),
        (30, "connected", "conn=2"),
        (40, "disconnected", 'conn=1 reason="handover"'),
    ]
    assert iv(handover) == [(10, 1000)]
    failed = [(10, "synced", ""), (40, "disconnected", 'conn=2 reason="connect timed out"')]
    assert iv(failed) == [(10, 1000)]  # an attempt that never delivered data


def test_events_that_leave_the_book_as_it_is_change_nothing():
    for kind in ("bbo_mismatch", "undecodable", "snapshot_failed", "server_shutdown", "connected"):
        assert iv([(10, "synced", ""), (40, kind, "")]) == [(10, 1000)], kind


def test_late_data_is_left_out_unless_asked_for():
    ev = [(10, "synced", ""), (40, "late", "lag_ms=2000"), (60, "on_time", "")]
    assert iv(ev) == [(10, 40), (60, 1000)]
    assert iv(ev, late=True) == [(10, 1000)]


def test_a_late_interval_ends_with_its_connection():
    ev = [
        (10, "synced", ""),
        (40, "late", ""),
        (50, "reconnect", ""),
        (80, "synced", ""),  # no on_time: the new connection is not late
    ]
    assert iv(ev) == [(10, 40), (80, 1000)]


def test_a_gap_while_late_stays_out_until_both_are_over():
    ev = [
        (10, "synced", ""),
        (40, "late", ""),
        (50, "gap", ""),
        (60, "resync", ""),
        (90, "on_time", ""),
    ]
    assert iv(ev) == [(10, 40), (90, 1000)]
    assert iv(ev, late=True) == [(10, 50), (60, 1000)]


def test_a_run_over_midnight_starts_valid_and_each_run_ends_its_data():
    runs = [run(0, 99, book=12345), run(200, 299)]
    ev = [(150, "synced", ""), (210, "connected", ""), (220, "synced", "")]
    # the event at 150 lies between the runs and opens nothing
    assert iv(ev, runs) == [(0, 100), (220, 300)]


def test_without_a_sync_nothing_is_valid():
    assert iv([(0, "connected", ""), (5, "snapshot_too_old", "")]) == []


def test_segments_include_their_start_and_exclude_their_end():
    intervals = [(10, 50), (70, 100)]
    ts = np.array([0, 9, 10, 49, 50, 69, 70, 99, 100])
    assert segment_ids(ts, intervals).tolist() == [-1, -1, 0, 0, -1, -1, 1, 1, -1]
    assert segment_ids(ts, []).tolist() == [-1] * len(ts)
