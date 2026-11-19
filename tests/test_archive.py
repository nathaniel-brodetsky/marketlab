"""Binance's public trade archives: read with or without a header line, in microseconds or
milliseconds, and checked for missing ids, missing raw trades, times out of order or out of
their day, empty rows, missing dates and ids that do not continue from one day to the
next."""

import datetime as dt
import io
import zipfile

import numpy as np
import pytest

from marketlab import archive
from marketlab.cli import main

DAY_NS = 86_400 * 10**9


def start_of(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.UTC).timestamp()) * 10**9


def rows(date: str, n: int = 50, first_agg: int = 1000, first_trade: int = 5000, unit="us"):
    """Clean aggTrades rows spread over the day: each aggregate holds two raw trades."""
    t0 = start_of(date)
    step = DAY_NS // (n + 1)
    out = []
    for k in range(n):
        ts = (t0 + (k + 1) * step) // (1_000 if unit == "us" else 1_000_000)
        f = first_trade + 2 * k
        maker = "True" if k % 3 == 0 else "False"
        out.append([first_agg + k, 97000.0 + k * 0.01, 0.002, f, f + 1, ts, maker, "True"])
    return out


def write(root, date, data, header=False, kind="aggTrades", symbol="BTCUSDT"):
    folder = root / "spot" / kind / symbol
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}-{kind}-{date}"
    lines = [",".join(archive.KINDS[kind])] if header else []
    lines += [",".join(str(x) for x in r) for r in data]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{name}.csv", "\n".join(lines) + "\n")
    (folder / f"{name}.zip").write_bytes(buf.getvalue())


def test_read_with_or_without_header_in_either_unit(tmp_path):
    write(tmp_path / "a", "2026-11-01", rows("2026-11-01"))
    write(tmp_path / "b", "2026-11-01", rows("2026-11-01"), header=True)
    write(tmp_path / "c", "2026-11-01", rows("2026-11-01", unit="ms"))
    frames = []
    for sub, unit in (("a", "us"), ("b", "us"), ("c", "ms")):
        (day,) = archive.find(tmp_path / sub, "BTCUSDT")
        df, got = archive.read(day)
        assert got == unit and df.columns == archive.KINDS["aggTrades"]
        frames.append(df)
    assert frames[0].equals(frames[1])
    # milliseconds lose the microseconds, and nothing else
    assert np.array_equal(frames[2]["ts"].to_numpy(), frames[0]["ts"].to_numpy() // 10**6 * 10**6)
    df = frames[0]
    assert df["is_buyer_maker"].dtype.is_(df["is_best_match"].dtype)
    assert df["is_buyer_maker"].sum() == 17 and df["is_best_match"].all()
    assert int(df["ts"].min()) >= start_of("2026-11-01")


def test_a_clean_range_passes_and_its_totals_add_up(tmp_path, capsys):
    write(tmp_path, "2026-11-01", rows("2026-11-01", first_agg=1000, first_trade=5000))
    write(tmp_path, "2026-11-02", rows("2026-11-02", first_agg=1050, first_trade=5100))
    r = archive.check(archive.find(tmp_path, "BTCUSDT"))
    assert r.ok, [d.problems for d in r.days]
    d = r.days[0]
    assert (d.rows, d.trades, d.unit) == (50, 100, "us")
    assert d.volume == pytest.approx(0.1) and d.buy_share == pytest.approx(33 / 50)
    assert d.longest_silence_s == pytest.approx(86_400 / 51, abs=1)
    assert main(["archive", "check", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 days, 2026-11-01 .. 2026-11-02" in out and "verdict     ok" in out


def test_every_kind_of_damage_is_named(tmp_path, capsys):
    data = rows("2026-11-01")
    data[10][0] += 3  # aggregate ids skip 3...
    for r in data[11:]:
        r[0] += 3  # ...and go on from there
    data[20][3] += 4  # 4 raw trades missing before this aggregate
    data[30][5], data[31][5] = data[31][5], data[30][5]  # two times out of order
    data[40][5] = data[40][5] + 2 * DAY_NS // 1000  # a time in another day
    data[45][2] = 0.0  # an empty quantity
    write(tmp_path, "2026-11-01", data)
    write(tmp_path, "2026-11-02", rows("2026-11-02", first_agg=2000, first_trade=9000))
    write(tmp_path, "2026-11-04", rows("2026-11-04", first_agg=3000, first_trade=9500))
    r = archive.check(archive.find(tmp_path, "BTCUSDT"))
    d = r.days[0]
    assert (d.id_gaps, d.ids_missing) == (1, 3)
    assert (d.trade_gaps, d.trades_missing) == (1, 4)
    assert d.backwards == 2  # out of order, and back from the other day
    assert (d.outside, d.empty) == (1, 1)
    assert r.missing_dates == ["2026-11-03"]
    assert r.breaks == ["2026-11-01 -> 2026-11-02: ids 1052 then 2000"]
    assert not r.ok and not r.days[1].problems
    assert main(["archive", "check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    for text in ("3 ids missing", "4 raw trades missing", "times go back", "outside the day",
                 "empty price or quantity", "MISSING     2026-11-03", "BREAK"):  # fmt: skip
        assert text in out, text


def test_trades_files_and_wrong_inputs(tmp_path):
    t0 = start_of("2026-11-01") // 1000
    data = [[700 + k, 97000.0, 0.001, 97.0, t0 + 1000 * (k + 1), "False", "True"] for k in range(5)]
    data[3][0] += 1
    for r in data[4:]:
        r[0] += 1
    write(tmp_path, "2026-11-01", data, kind="trades")
    (day,) = archive.find(tmp_path, "BTCUSDT", kind="trades")
    c = archive.check_day(day)
    assert (c.rows, c.trades, c.id_gaps, c.ids_missing, c.trade_gaps) == (5, 5, 1, 1, 0)
    assert archive.find(tmp_path, "ETHUSDT") == []
    with pytest.raises(archive.ArchiveError, match="kind"):
        archive.find(tmp_path, "BTCUSDT", kind="klines")
    assert main(["archive", "check", str(tmp_path / "nowhere")]) == 2
