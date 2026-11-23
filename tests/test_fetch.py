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
    state = types.SimpleNamespace(files={}, calls=[], funding=[], klines={}, invalid=set(),
                                  bucket=set(), page=1000)  # fmt: skip

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
            elif u.path == "/bucket":
                body = _s3_listing(state, dict(urllib.parse.parse_qsl(u.query)))
            elif urllib.parse.unquote(u.path) in state.files:  # as a server reads an escape
                body = state.files[urllib.parse.unquote(u.path)]
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


def _s3_listing(state, q: dict) -> bytes:
    """A page of an S3 listing (version 1) of `state.bucket`, as the archive's bucket gives."""
    prefix, delim, marker = q.get("prefix", ""), q.get("delimiter", ""), q.get("marker", "")
    items: list[tuple[str, str]] = []
    for k in sorted(state.bucket):
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        item = ("p", prefix + rest.split(delim)[0] + delim) if delim and delim in rest else ("k", k)
        if item not in items:
            items.append(item)
    items = [it for it in items if it[1] > marker]
    page, more = items[: state.page], len(items) > state.page
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
           f"<Prefix>{prefix}</Prefix><IsTruncated>{str(more).lower()}</IsTruncated>"]  # fmt: skip
    if more and delim:
        xml.append(f"<NextMarker>{page[-1][1]}</NextMarker>")
    for kind, v in page:
        xml.append(f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>" if kind == "p"
                   else f"<Contents><Key>{v}</Key><Size>1</Size></Contents>")  # fmt: skip
    return "".join([*xml, "</ListBucketResult>"]).encode()


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
         "status": "TRADING", "onboardDate": 1577836800000, "baseAsset": "AAA",
         "underlyingType": "COIN"},
        {"symbol": "BBBUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "status": "SETTLING", "onboardDate": 1600000000000, "baseAsset": "BBB",
         "underlyingType": "INDEX"},
        {"symbol": "CCCUSDT_261225", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT"},
        {"symbol": "DDDBUSD", "contractType": "PERPETUAL", "quoteAsset": "BUSD"},
        {"symbol": "EEEUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "status": "CLOSE", "onboardDate": 1600000000000, "baseAsset": "EEE"},
        {"symbol": "SHRUSDT", "contractType": "SHARE_PERPETUAL", "quoteAsset": "USDT",
         "status": "TRADING", "underlyingType": "EQUITY"},
    ]}  # fmt: skip
    server.files["/fapi/v1/exchangeInfo"] = json.dumps(info).encode()
    server.klines["AAAUSDT"] = _bars("2020-01-01", 1600)
    server.klines["BBBUSDT"] = _bars("2021-03-01", 30)
    server.invalid.add("EEEUSDT")
    server.funding += [dict(r, symbol="AAAUSDT") for r in
                       _settlements(dt.datetime(2020, 1, 1, tzinfo=dt.UTC), 30)]  # fmt: skip
    args = ["--start", "2020-01-01", "--end", "2024-12-31", "--out", str(tmp_path),
            "--base-url", server.url, "--pause", "0", "--funding-pause", "0"]  # fmt: skip
    # the lists only: nothing downloaded
    lists = tmp_path / "lists"
    assert fu.main([*args[:5], str(lists), *args[6:], "--list-only"]) == 0
    assert not [c for c in server.calls if "klines" in c or "fundingRate" in c]
    assert not (lists / "um" / "klines1d").exists()
    out = capsys.readouterr().out
    assert "TRADING 1; underlying ? 1, COIN 1, INDEX 1" in out
    assert "2 other USDT contracts: CURRENT_QUARTER 1, SHARE_PERPETUAL 1" in out
    assert (lists / "um" / "contracts.csv").read_text().splitlines() == [
        "symbol,contract_type,underlying_type,status",
        "AAAUSDT,PERPETUAL,COIN,TRADING",
        "BBBUSDT,PERPETUAL,INDEX,SETTLING",
        "CCCUSDT_261225,CURRENT_QUARTER,,",
        "EEEUSDT,PERPETUAL,,CLOSE",
        "SHRUSDT,SHARE_PERPETUAL,EQUITY,TRADING",
    ]
    assert fu.main(args) == 0
    out = capsys.readouterr().out
    assert "3 USDT perpetuals: CLOSE 1, SETTLING 1, TRADING 1" in out
    assert "EEEUSDT" in out and "no data served" in out and "SHRUSDT" not in out
    rows = (tmp_path / "um" / "universe.csv").read_text().splitlines()
    assert rows[0] == "symbol,status,onboard_date,base,quote,underlying_type" and len(rows) == 4
    assert rows[2] == "BBBUSDT,SETTLING,1600000000000,BBB,USDT,INDEX"
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


fa = _load("fetch_archive_universe")


def _archive(state, sym: str, kind: str, month: str, text: str) -> None:
    """A monthly archive file, its checksum, and its key in the bucket's listing."""
    sub, tag = (f"klines/{sym}/1d", "1d") if kind == "klines" else (f"fundingRate/{sym}", kind)
    name = f"{sym}-{tag}-{month}.zip"
    key = f"data/futures/um/monthly/{sub}/{name}"
    _publish(state, "/" + key, _zip(name, text))
    state.bucket |= {key, key + ".CHECKSUM"}


def _month_bars(month: str, close: float, header: bool = False) -> str:
    first = dt.datetime.fromisoformat(f"{month}-01").replace(tzinfo=dt.UTC)
    t0 = int(first.timestamp() * 1000)
    n = ((first.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - first).days
    rows = [",".join(fu.KLINE_COLUMNS)] if header else []
    rows += [f"{t0 + k * DAY_MS},1,1,1,{close},10,{t0 + (k + 1) * DAY_MS - 1},{500 + k}.5,9,5,5,0"
             for k in range(n)]  # fmt: skip
    return "\n".join(rows) + "\n"


def test_the_archive_fills_in_the_perpetuals_gone_and_the_bars_not_served(server, tmp_path,
                                                                          capsys):  # fmt: skip
    # listed: AAAUSDT, its funding from 2021-01-01 but its bars from the API only from June
    fu.write_rows(tmp_path / "um" / "universe.csv", fu.UNIVERSE_COLUMNS,
                  [["AAAUSDT", "TRADING", 0, "AAA", "USDT"],
                   ["BBBUSDT", "TRADING", 0, "BBB", "USDT"]])  # fmt: skip
    api = _bars("2021-06-01", 61)
    fu.write_rows(tmp_path / "um" / "klines1d" / "AAAUSDT.csv", fu.KLINE_COLUMNS,
                  [[str(r[0]), *r[1:6], r[7], str(r[8])] for r in api])  # fmt: skip
    fu.write_rows(
        tmp_path / "um" / "klines1d" / "BBBUSDT.csv",
        fu.KLINE_COLUMNS,
        [[str(r[0]), *r[1:6], r[7], str(r[8])] for r in _bars("2021-01-01", 200)],
    )
    t0 = int(dt.datetime(2021, 1, 1, tzinfo=dt.UTC).timestamp() * 1000)
    ff.write(tmp_path / "um" / "fundingRate" / "AAAUSDT" / "AAAUSDT-fundingRate.csv",
             [(t0 + k * DAY_MS, "0.0001", "") for k in range(212)])  # fmt: skip
    ff.write(tmp_path / "um" / "fundingRate" / "BBBUSDT" / "BBBUSDT-fundingRate.csv",
             [(t0 + k * DAY_MS, "0.0001", "") for k in range(200)])  # fmt: skip
    for month in ("2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06"):
        _archive(server, "AAAUSDT", "klines", month, _month_bars(month, 7.0, header=True))
    _archive(server, "BBBUSDT", "klines", "2021-01", _month_bars("2021-01", 3.0))
    # gone: GONEUSDT (funding from the API), GTWOUSDT (funding from the archive only)
    for month in ("2020-11", "2020-12", "2021-01"):
        _archive(server, "GONEUSDT", "klines", month, _month_bars(month, 2.0))
    _archive(server, "GTWOUSDT", "klines", "2021-02", _month_bars("2021-02", 4.0))
    feb = int(dt.datetime(2021, 2, 1, tzinfo=dt.UTC).timestamp() * 1000)
    _archive(
        server,
        "GTWOUSDT",
        "fundingRate",
        "2021-02",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        + "".join(f"{feb + k * 8 * 3_600_000 + 5},8,-0.0002\n" for k in range(84)),
    )
    server.funding += [
        {
            "symbol": "GONEUSDT",
            "fundingTime": t0 + k * 8 * 3_600_000,
            "fundingRate": "0.0003",
            "markPrice": "2",
        }
        for k in range(93)
    ]
    # not perpetuals in USDT: a delivery contract and a BUSD one
    _archive(server, "AAAUSDT_210625", "klines", "2021-01", _month_bars("2021-01", 1.0))
    _archive(server, "CCCBUSD", "klines", "2021-01", _month_bars("2021-01", 1.0))
    # not gone: SHRUSDT a contract of another type in Binance's list, TRDUSDT in no list
    # but trading on the last day
    _archive(server, "SHRUSDT", "klines", "2021-03", _month_bars("2021-03", 9.0))
    for month in ("2021-06", "2021-07"):
        _archive(server, "TRDUSDT", "klines", month, _month_bars(month, 8.0))
    server.page = 2  # the listing a page of two at a time
    args = ["--start", "2021-01-01", "--end", "2021-07-31", "--out", str(tmp_path),
            "--base-url", server.url, "--api-url", server.url, "--list-url",
            server.url + "/bucket", "--funding-pause", "0"]  # fmt: skip
    with pytest.raises(SystemExit, match=r"contracts\.csv: run fetch_universe\.py --list-only"):
        fa.main(args)
    fu.write_rows(tmp_path / "um" / "contracts.csv", fu.CONTRACT_COLUMNS,
                  [["AAAUSDT", "PERPETUAL", "COIN", "TRADING"],
                   ["BBBUSDT", "PERPETUAL", "COIN", "TRADING"],
                   ["SHRUSDT", "SHARE_PERPETUAL", "EQUITY", "TRADING"]])  # fmt: skip
    assert fa.main(args) == 0
    out = capsys.readouterr().out
    assert "6 USDT perpetuals in the archive: 3 gone from Binance's list, 1 listed" in out
    assert "left out, 1 Binance lists as contracts of other types: SHARE_PERPETUAL 1" in out
    assert "left out, 1 not in Binance's list yet trading on 2021-07-31: TRDUSDT" in out
    assert "done: 2 of 2 gone perpetuals with bars, 2 with funding" in out
    assert not any("SHRUSDT" in c for c in server.calls if c.endswith(".zip"))
    assert not any("TRDUSDT" in c and "fundingRate" in c for c in server.calls)
    assert not (tmp_path / "um" / "fundingRate" / "TRDUSDT").exists()
    a = fu.read_rows(tmp_path / "um" / "klines1d" / "AAAUSDT.csv", fu.KLINE_COLUMNS)
    days = [int(r[0]) for r in a]
    assert days == sorted(set(days)) and days[0] == t0 and len(a) == 151 + 61
    june = int(dt.datetime(2021, 6, 1, tzinfo=dt.UTC).timestamp() * 1000)
    by_day = {int(r[0]): r for r in a}
    assert by_day[t0][4] == "7.0" and by_day[t0][6] == "500.5"  # from the archive
    assert by_day[june][4] == "1.000"  # the API's bar kept, not the archive's
    assert "+151 days of 151 without a bar" in out
    g = fu.read_rows(tmp_path / "um" / "klines1d" / "GONEUSDT.csv", fu.KLINE_COLUMNS)
    assert len(g) == 31 and int(g[0][0]) == t0  # November and December are before --start
    gf = ff.read(tmp_path / "um" / "fundingRate" / "GONEUSDT" / "GONEUSDT-fundingRate.csv")
    assert len(gf) == 93 and gf[0][1] == "0.0003"
    tf = ff.read(tmp_path / "um" / "fundingRate" / "GTWOUSDT" / "GTWOUSDT-fundingRate.csv")
    assert len(tf) == 84 and tf[0] == (feb + 5, "-0.0002", "")
    gone = (tmp_path / "um" / "gone.csv").read_text().splitlines()
    assert gone[0] == "symbol,first_day,last_day,days,settlements,funding_from"
    assert gone[1:] == ["GONEUSDT,2021-01-01,2021-01-31,31,93,api",
                        "GTWOUSDT,2021-02-01,2021-02-28,28,84,archive"]  # fmt: skip
    assert not any("BBBUSDT" in c and c.endswith(".zip") for c in server.calls)
    zips = [c for c in server.calls if c.endswith(".zip")]
    assert fa.main(args) == 0  # again: nothing downloaded twice
    assert [c for c in server.calls if c.endswith(".zip")] == zips
    assert "0 listed with days of funding and no bar" in capsys.readouterr().out
    assert (tmp_path / "um" / "gone.csv").read_text().splitlines()[1:] == gone[1:]
    # a listing that is not one fails the run
    assert fa.main([*args[:-3], server.url + "/nothing", "--funding-pause", "0"]) == 1
    assert "the archive's list of perpetuals" in capsys.readouterr().err


fh = _load("fetch_hourly")


def test_the_hourly_bars_of_the_months_with_daily_ones(server, tmp_path, capsys):
    # AAAUSDT has daily bars from 2021-02-10 to 2021-04-20; the archive has its hour bars for
    # January .. May, but only February .. April are its months, and May is past --end
    daily = [[str(r[0]), *r[1:6], r[7], str(r[8])] for r in _bars("2021-02-10", 70)]
    fu.write_rows(tmp_path / "um" / "klines1d" / "AAAUSDT.csv", fu.KLINE_COLUMNS, daily)
    for month in ("2021-01", "2021-02", "2021-03", "2021-04", "2021-05"):
        name = f"AAAUSDT-1h-{month}.zip"
        key = f"data/futures/um/monthly/klines/AAAUSDT/1h/{name}"
        _publish(server, "/" + key, _zip(name, "1,1,1,1,1,1,1,1,1,1,1,0\n"))
        server.bucket |= {key, key + ".CHECKSUM"}
    names = tmp_path / "symbols.txt"
    names.write_text("AAAUSDT\nNOBARSUSDT\n")
    args = ["--symbols-file", str(names), "--start", "2021-01-01", "--end", "2021-04-30",
            "--out", str(tmp_path), "--base-url", server.url, "--list-url",
            server.url + "/bucket"]  # fmt: skip
    assert fh.main(args) == 0
    out = capsys.readouterr().out
    got = sorted(f.name for f in (tmp_path / "um" / "klines" / "AAAUSDT" / "1h").glob("*.zip"))
    assert got == [f"AAAUSDT-1h-2021-0{m}.zip" for m in (2, 3, 4)]
    assert "AAAUSDT          3 of 3 months" in out and "NOBARSUSDT       no daily bars" in out
    zips = [c for c in server.calls if c.endswith(".zip")]
    assert fh.main(args) == 0  # again: nothing downloaded twice
    assert [c for c in server.calls if c.endswith(".zip")] == zips
    # a symbol in Chinese letters, as Binance lists some: its files asked for in UTF-8 escapes
    sym = "龙虾USDT"
    fu.write_rows(tmp_path / "um" / "klines1d" / f"{sym}.csv", fu.KLINE_COLUMNS, daily)
    name = f"{sym}-1h-2021-03.zip"
    key = f"data/futures/um/monthly/klines/{sym}/1h/{name}"
    _publish(server, "/" + key, _zip(name, "1,1,1,1,1,1,1,1,1,1,1,0\n"))
    server.bucket |= {key, key + ".CHECKSUM"}
    names.write_text(f"{sym}\n")
    assert fh.main(args) == 0
    assert (tmp_path / "um" / "klines" / sym / "1h" / name).exists()
    assert f"{sym:<16} 1 of 3 months (2 not in the archive)" in capsys.readouterr().out
