"""Reading what marketrec writes, and refusing what does not follow the contract."""

import json
import shutil

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import fixture_day

from marketlab import synth
from marketlab.io import TABLES, ContractError, Day, segment_ids
from marketlab.io.contract import _same_columns, part_files, read_manifest

MS = 1_000_000
T0 = synth.DAY0


def test_a_marketrec_day_with_a_gap_has_two_segments():
    # tests/fixtures/gap.jsonl in marketrec: synced at 180 ms, a gap at 305 ms, resync at 360 ms
    day = Day.open(fixture_day("gap"))
    assert day.symbol == "BTCUSDT"
    assert str(day.price_step) == "0.01000000"
    assert day.intervals == [(T0 + 180 * MS, T0 + 305 * MS), (T0 + 360 * MS, T0 + 405 * MS + 1)]


def test_a_marketrec_day_with_a_reconnect_restarts_at_the_new_sync():
    day = Day.open(fixture_day("reconnect"))
    assert day.intervals == [(T0 + 180 * MS, T0 + 230 * MS), (T0 + 1360 * MS, T0 + 1405 * MS + 1)]


@pytest.mark.parametrize("name", ["short", "gap", "reconnect"])
def test_book_rows_lie_in_valid_intervals_as_the_contract_says(name):
    # "book_l20 has no rows there": sampled only while the book is valid, late or not
    day = Day.open(fixture_day(name), include_late=True)
    ts = day.raw("book_l20")["recv_ts"].to_numpy()
    assert len(ts) > 0
    assert (segment_ids(ts, day.intervals) >= 0).all()


def test_every_table_reads_with_the_contract_columns():
    day = Day.open(fixture_day("short"))
    for table, schema in TABLES.items():
        df = day.raw(table)
        assert df.columns == schema.names, table
    assert day.raw("trades").height == 210
    assert day.raw("link_rtt").height == 0  # no directory: an empty table, not an error


def test_synthetic_parts_have_the_columns_marketrec_writes(tmp_path):
    d = synth.write(synth.Scenario(seconds=5), tmp_path)
    real = fixture_day("short")
    for table in ("depth_diff", "book_ticker", "trades", "book_l20", "quality"):
        mine = pq.ParquetFile(part_files(d, table)[0])
        theirs = pq.ParquetFile(part_files(real, table)[0])
        assert _same_columns(mine.schema_arrow, theirs.schema_arrow), table
        assert _same_columns(theirs.schema_arrow, TABLES[table]), table
        keys = {k for k in theirs.metadata.metadata if k.startswith(b"marketrec.")}
        assert keys <= set(mine.metadata.metadata), table


def _copy(tmp_path, name="short"):
    dst = tmp_path / "BTCUSDT" / "2026-11-21"
    shutil.copytree(fixture_day(name), dst)
    return dst


def test_another_schema_version_is_refused(tmp_path):
    d = _copy(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["schema_version"] = 2
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ContractError, match="schema version 2"):
        Day.open(d)


def test_a_part_the_manifest_lists_must_exist(tmp_path):
    d = _copy(tmp_path)
    (d / "trades" / "part-000.parquet").unlink()
    with pytest.raises(ContractError, match="missing parts"):
        Day.open(d)


def test_a_part_of_another_table_is_refused(tmp_path):
    d = _copy(tmp_path)
    shutil.copy(d / "trades" / "part-000.parquet", d / "quality" / "part-001.parquet")
    with pytest.raises(ContractError):
        Day.open(d).raw("quality")


def test_a_part_that_says_it_holds_another_table_is_refused(tmp_path):
    d = _copy(tmp_path)
    p = d / "quality" / "part-000.parquet"
    t = pq.read_table(p)
    meta = dict(pq.ParquetFile(p).metadata.metadata)
    meta[b"marketrec.table"] = b"trades"
    pq.write_table(t.replace_schema_metadata(meta), p)
    with pytest.raises(ContractError, match="holds table trades"):
        Day.open(d).raw("quality")


def test_a_part_with_other_steps_is_refused(tmp_path):
    d = _copy(tmp_path)
    p = d / "trades" / "part-000.parquet"
    t = pq.read_table(p)
    meta = dict(t.schema.metadata or {})
    meta.update(pq.ParquetFile(p).metadata.metadata)
    meta[b"marketrec.price_step"] = b"0.10000000"
    pq.write_table(t.replace_schema_metadata(meta), p)
    with pytest.raises(ContractError, match="steps"):
        Day.open(d).raw("trades")


def test_unfinished_parts_are_not_read(tmp_path):
    d = _copy(tmp_path)
    shutil.copy(d / "trades" / "part-000.parquet", d / "trades" / ".part-001.parquet.tmp")
    assert Day.open(d).raw("trades").height == 210


def test_a_day_without_manifest_is_read_with_the_book_unknown_at_its_start(tmp_path):
    d = _copy(tmp_path, "gap")
    with_manifest = Day.open(d).intervals
    (d / "manifest.json").unlink()
    assert read_manifest(d) is None
    day = Day.open(d)
    assert day.symbol == "BTCUSDT"
    assert day.intervals[0][0] == with_manifest[0][0]  # valid from the sync, not before
    assert day.intervals[:1] == with_manifest[:1]


def test_valid_rows_carry_their_segment():
    day = Day.open(fixture_day("gap"))
    v = day.valid("depth_diff", ["recv_ts"])
    seg = v["segment"].to_numpy()
    assert set(seg.tolist()) <= {0, 1}
    for s, (lo, hi) in enumerate(day.intervals):
        ts = v.filter(v["segment"] == s)["recv_ts"].to_numpy()
        assert ((ts >= lo) & (ts < hi)).all()
    raw = day.raw("depth_diff")["recv_ts"].to_numpy()
    assert v.height == int((segment_ids(raw, day.intervals) >= 0).sum()) < len(raw)


def test_truncation_keeps_only_what_was_known():
    day = Day.open(fixture_day("gap"))
    cut = T0 + 330 * MS  # after the gap, before the resync
    early = day.truncated(cut)
    assert early.intervals == [(T0 + 180 * MS, T0 + 305 * MS)]
    for table in TABLES:
        t = "ts" if table == "quality" else "recv_ts"
        assert (early.raw(table)[t].to_numpy() <= cut).all()
    assert np.array_equal(
        early.raw("depth_diff")["recv_ts"].to_numpy(),
        day.raw("depth_diff").filter(day.raw("depth_diff")["recv_ts"] <= cut)["recv_ts"].to_numpy(),
    )


def test_the_day_command_counts_time_with_a_connection_open(capsys):
    from marketlab.cli import main

    assert main(["day", str(fixture_day("reconnect"))]) == 0
    out = capsys.readouterr().out
    # the manifest's covered time, not the run's span: 0.230 s + 0.175 s of 1.405 s
    assert "of 0.4 s with a connection open" in out
    assert "in 2 segments" in out
    assert "problems    1 reconnect\n" in out


def test_problems_are_the_events_of_a_day_that_is_not_well():
    from marketlab.cli import _problems

    assert _problems({"connected": 1, "synced": 1, "late": 3, "on_time": 3}) == "none"
    text = _problems({"synced": 1, "gap": 1, "undecodable": 830, "buffer_overflow": 85})
    assert text.startswith("830 undecodable, 85 buffer_overflow, 1 gap; undecodable messages")


@pytest.mark.parametrize("name", ["short", "gap", "reconnect"])
def test_usable_time_agrees_with_the_recorder(name):
    # two implementations of the same rules: marketrec counts synced_ns while recording
    day = Day.open(fixture_day(name), include_late=True)
    m = read_manifest(fixture_day(name))
    usable = sum(e - s for s, e in day.intervals)
    assert abs(usable - m.raw["synced_ns"]) <= len(day.intervals)  # our ends are last row + 1 ns
