"""The two download scripts against a local server: monthly klines where the archive keeps
them, each file checked against its checksum and fetched once; the funding history a page
at a time, extended without asking again for what is there."""

import datetime as dt
import hashlib
import http.server
import importlib.util
import io
import json
import threading
import types
import urllib.parse
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fb = _load("fetch_binance")
ff = _load("fetch_funding")


@pytest.fixture
def server(monkeypatch):
    for k in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(k, "*")
    state = types.SimpleNamespace(files={}, calls=[], funding=[], klines={}, invalid=set())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            state.calls.append(self.path)
            if u.path == "/fapi/v1/fundingRate":
                q = {k: int(v[0]) if k != "symbol" else v[0]
                     for k, v in urllib.parse.parse_qs(u.query).items()}  # fmt: skip
                lo, hi = q["startTime"], q["endTime"]
                rows = [r for r in state.funding if lo <= r["fundingTime"] <= hi
                        and r.get("symbol", q["symbol"]) == q["symbol"]]  # fmt: skip
                body = json.dumps(rows[: q["limit"]]).encode()
            elif u.path == "/fapi/v1/klines":
                q = dict(urllib.parse.parse_qsl(u.query))
                if q["symbol"] in state.invalid:
                    body = b'{"code":-1121,"msg":"Invalid symbol."}'
                    self.send_response(400)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                lo, hi = int(q["startTime"]), int(q["endTime"])
                rows = [r for r in state.klines.get(q["symbol"], []) if lo <= r[0] <= hi]
                body = json.dumps(rows[: int(q["limit"])]).encode()
            elif u.path in state.files:
                body = state.files[u.path]
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), text)
    return buf.getvalue()


def _publish(state, path: str, data: bytes, checksum: bytes | None = None) -> None:
    name = path.rsplit("/", 1)[1]
    state.files[path] = data
    digest = hashlib.sha256(checksum if checksum is not None else data).hexdigest()
    state.files[path + ".CHECKSUM"] = f"{digest}  {name}\n".encode()


def test_monthly_klines_are_fetched_once_and_checked(server, tmp_path, capsys):
    folder = "/data/futures/um/monthly/klines/BTCUSDT/1h"
    for month in ("2024-01", "2024-02"):
        name = f"BTCUSDT-1h-{month}.zip"
        _publish(server, f"{folder}/{name}", _zip(name, f"1704067200000,1,2,0.5,1.5,{month}\n"))
    args = ["--market", "um", "--kind", "klines", "--interval", "1h", "--monthly",
            "--start", "2024-01-15", "--end", "2024-03-02", "--out", str(tmp_path),
            "--base-url", server.url]  # fmt: skip
    assert fb.main(args) == 0
    out = capsys.readouterr().out
    assert "3 months, 2024-01 .. 2024-03" in out and "not published: 2024-03" in out
    got = tmp_path / "um" / "klines" / "BTCUSDT" / "1h"
    assert sorted(p.name for p in got.glob("*.zip")) == ["BTCUSDT-1h-2024-01.zip",
                                                          "BTCUSDT-1h-2024-02.zip"]  # fmt: skip
    zips = [c for c in server.calls if c.endswith(".zip")]
    assert fb.main(args) == 0  # again: nothing downloaded twice
    assert [c for c in server.calls if c.endswith(".zip")] == zips
    assert "have" in capsys.readouterr().out
    # a file that does not match its checksum is refused, twice, and the run fails
    name = "BTCUSDT-1h-2024-03.zip"
    _publish(server, f"{folder}/{name}", _zip(name, "x\n"), checksum=b"something else")
    assert fb.main(args) == 1
    assert "bad checksum" in capsys.readouterr().err
    assert not (got / name).exists()


def test_the_urls(capsys):
    fb.main(["--kind", "klines", "--interval", "1h", "--monthly", "--start", "2019-12-31",
             "--end", "2020-01-01", "--dry-run"])  # fmt: skip
    assert capsys.readouterr().out.split() == [
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2019-12.zip",
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip",
    ]
    fb.main(["--start", "2026-11-01", "--end", "2026-11-01", "--dry-run"])
    assert capsys.readouterr().out.split() == [
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2026-11-01.zip"
    ]
    with pytest.raises(SystemExit):
        fb.main(["--kind", "klines", "--start", "2026-11-01", "--end", "2026-11-01"])
    assert fb.months(dt.date(2023, 11, 30), dt.date(2024, 2, 1)) == [
        "2023-11", "2023-12", "2024-01", "2024-02"]  # fmt: skip


def _settlements(first: dt.datetime, n: int) -> list[dict]:
    t0 = int(first.timestamp() * 1000)
    return [{"symbol": "BTCUSDT", "fundingTime": t0 + k * 8 * 3_600_000 + k % 3,
             "fundingRate": f"{0.0001 + k * 1e-8:.8f}",
             "markPrice": "" if k < 5 else f"{40000 + k}.5"} for k in range(n)]  # fmt: skip


def test_the_funding_history_a_page_at_a_time(server, tmp_path, capsys):
    server.funding += _settlements(dt.datetime(2020, 1, 1, tzinfo=dt.UTC), 2500)
    args = ["--start", "2020-01-01", "--end", "2022-04-30", "--out", str(tmp_path),
            "--base-url", server.url, "--pause", "0"]  # fmt: skip
    assert ff.main(args) == 0
    f = tmp_path / "um" / "fundingRate" / "BTCUSDT" / "BTCUSDT-fundingRate.csv"
    rows = f.read_text().splitlines()
    assert rows[0] == "funding_time,funding_rate,mark_price"
    assert len(rows) == 1 + 2500 and len(server.calls) == 3  # 1000, 1000, 500
    assert rows[1].endswith(",0.00010000,") and rows[6].endswith(",40005.5")
    times = [int(r.split(",")[0]) for r in rows[1:]]
    assert times == sorted(set(times)) == [r["fundingTime"] for r in server.funding]
    assert "spacing     8 h x2499" in capsys.readouterr().out
    # later settlements, and a later end: only what comes after the last row is asked for
    server.funding += _settlements(dt.datetime(2022, 4, 15, tzinfo=dt.UTC), 3)[1:]
    calls = len(server.calls)
    assert ff.main([*args[:3], "2022-05-31", *args[4:]]) == 0
    assert len(server.calls) == calls + 1
    q = urllib.parse.parse_qs(urllib.parse.urlparse(server.calls[-1]).query)
    assert int(q["startTime"][0]) == times[-1] + 1
    assert len(f.read_text().splitlines()) == 1 + 2500 + 2
    with pytest.raises(SystemExit, match="not a file this script wrote"):
        f.write_text("time,rate\n1,2\n")
        ff.main(args)


ff_red = _load("fetch_fred")


def test_a_fred_series(server, tmp_path, capsys):
    text = "observation_date,GS3M\n2024-01-01,5.37\n2024-02-01,.\n2024-03-01,5.25\n"
    server.files["/graph/fredgraph.csv"] = text.encode()
    assert ff_red.main(["--out", str(tmp_path), "--base-url", server.url]) == 0
    f = tmp_path / "fred" / "GS3M.csv"
    assert f.read_text() == text
    assert "GS3M: 2 values, 2024-01-01 .. 2024-03-01" in capsys.readouterr().out
    assert server.calls[-1] == "/graph/fredgraph.csv?id=GS3M"
    server.files["/graph/fredgraph.csv"] = b"<html>not a csv</html>"
    assert ff_red.main(["--out", str(tmp_path), "--base-url", server.url]) == 1
    assert f.read_text() == text  # a bad answer does not replace a good file


fu = _load("fetch_universe")
DAY_MS = 86_400_000


def _bars(first: str, n: int) -> list:
    t0 = int(dt.datetime.fromisoformat(first).replace(tzinfo=dt.UTC).timestamp() * 1000)
    out = []
    for k in range(n):
        t, close = t0 + k * DAY_MS, f"{1 + k / 1000:.3f}"
        out.append([t, "1.0", "1.2", "0.9", close, "10.0", t + DAY_MS - 1, f"{1000 + k}.5", 7,
                    "5.0", "5.0", "0"])  # fmt: skip
    return out


def test_the_universe_of_perpetuals(server, tmp_path, capsys):
    info = {"symbols": [
        {"symbol": "AAAUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "status": "TRADING", "onboardDate": 1577836800000, "baseAsset": "AAA"},
        {"symbol": "BBBUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "status": "SETTLING", "onboardDate": 1600000000000, "baseAsset": "BBB"},
        {"symbol": "CCCUSDT_261225", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT"},
        {"symbol": "DDDBUSD", "contractType": "PERPETUAL", "quoteAsset": "BUSD"},
        {"symbol": "EEEUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "status": "CLOSE", "onboardDate": 1600000000000, "baseAsset": "EEE"},
    ]}  # fmt: skip
    server.files["/fapi/v1/exchangeInfo"] = json.dumps(info).encode()
    server.klines["AAAUSDT"] = _bars("2020-01-01", 1600)
    server.klines["BBBUSDT"] = _bars("2021-03-01", 30)
    server.invalid.add("EEEUSDT")
    server.funding += [dict(r, symbol="AAAUSDT") for r in
                       _settlements(dt.datetime(2020, 1, 1, tzinfo=dt.UTC), 30)]  # fmt: skip
    args = ["--start", "2020-01-01", "--end", "2024-12-31", "--out", str(tmp_path),
            "--base-url", server.url, "--pause", "0", "--funding-pause", "0"]  # fmt: skip
    assert fu.main(args) == 0
    out = capsys.readouterr().out
    assert "3 USDT perpetuals: CLOSE 1, SETTLING 1, TRADING 1" in out
    assert "EEEUSDT" in out and "no data served" in out
    rows = (tmp_path / "um" / "universe.csv").read_text().splitlines()
    assert rows[0] == "symbol,status,onboard_date,base,quote" and len(rows) == 4
    a = (tmp_path / "um" / "klines1d" / "AAAUSDT.csv").read_text().splitlines()
    assert a[0] == "open_time,open,high,low,close,volume,quote_volume,trades"
    assert len(a) == 1 + 1600 and a[1].endswith(",1000.5,7")
    assert len([c for c in server.calls if "klines" in c and "AAAUSDT" in c]) == 2  # 1500 + 100
    f = tmp_path / "um" / "fundingRate" / "AAAUSDT" / "AAAUSDT-fundingRate.csv"
    assert len(f.read_text().splitlines()) == 1 + 30
    # again, a day later: only the new bar is asked for
    server.klines["AAAUSDT"] = _bars("2020-01-01", 1601)
    calls = len(server.calls)
    assert fu.main(["--start", "2020-01-01", "--end", "2025-01-01", *args[4:]]) == 0
    asked = [c for c in server.calls[calls:] if "klines" in c and "AAAUSDT" in c]
    q = urllib.parse.parse_qs(urllib.parse.urlparse(asked[0]).query)
    assert len(asked) == 1 and int(q["startTime"][0]) == server.klines["AAAUSDT"][1599][0] + DAY_MS
    assert len((tmp_path / "um" / "klines1d" / "AAAUSDT.csv").read_text().splitlines()) == 1602
