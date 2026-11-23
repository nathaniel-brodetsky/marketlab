"""scripts/fetch_bybit.py against a local server that answers as Bybit's v5 API does: newest
first, a page at a time, retCode in a 200; the folder it writes read by marketlab.xsection as
binance-data is."""

import datetime as dt
import http.server
import importlib.util
import json
import threading
import types
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from marketlab import xsection as xs

ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000
_spec = importlib.util.spec_from_file_location("fetch_bybit", ROOT / "scripts" / "fetch_bybit.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


def _ms(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp() * 1000)


def _bars(first: str, n: int) -> list[list[str]]:
    """n daily bars from `first`, closing at 100, 101, ...; turnover 1000.5 + k."""
    t0 = _ms(first)
    return [[str(t0 + k * DAY_MS), "1", "2", "0.5", str(100 + k), "10", f"{1000.5 + k}"]
            for k in range(n)]  # fmt: skip


def _settlements(first: str, n: int, every_h: int = 8, rate: float = 1e-4) -> list[tuple]:
    t0 = _ms(first)
    return [(t0 + k * every_h * 3_600_000, f"{rate * (1 + k % 3):.8f}") for k in range(n)]


def _perp(symbol: str, status: str, launch: str = "2020-03-25", **kw) -> dict:
    return {"symbol": symbol, "contractType": "LinearPerpetual", "status": status,
            "baseCoin": symbol.removesuffix("USDT"), "quoteCoin": "USDT",
            "launchTime": str(_ms(launch)), "deliveryTime": "0", **kw}  # fmt: skip


@pytest.fixture
def bybit(monkeypatch):
    for k in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(k, "*")
    monkeypatch.setattr(fb, "BAR_PAGE", 4)
    monkeypatch.setattr(fb, "FUNDING_PAGE", 3)
    monkeypatch.setattr(fb.time, "sleep", lambda s: None)
    state = types.SimpleNamespace(instruments=[], klines={}, funding={}, invalid=set(),
                                  busy=set(), calls=[], page=2)  # fmt: skip

    def answer(path: str, q: dict) -> dict:
        if path in state.busy:  # once
            state.busy.discard(path)
            return {"retCode": 10006, "retMsg": "Too many visits!"}
        if path == "/v5/market/instruments-info":
            # asked for Closed, Bybit gives the contracts pending their opening too
            also = {"Closed": {"PendingOpen"}}.get(q["status"], set())
            rows = [s for s in state.instruments if s["status"] in {q["status"], *also}]
            at = int(q.get("cursor") or 0)
            more = at + state.page < len(rows)
            return {"retCode": 0, "result": {"category": "linear",
                    "list": rows[at : at + state.page],
                    "nextPageCursor": str(at + state.page) if more else ""}}  # fmt: skip
        if q.get("symbol") in state.invalid:
            return {"retCode": 10001, "retMsg": "params error: Symbol Is Invalid"}
        if path == "/v5/market/kline":
            lo, hi = int(q["start"]), int(q["end"])
            rows = [r for r in state.klines.get(q["symbol"], []) if lo <= int(r[0]) <= hi]
            rows = sorted(rows, key=lambda r: -int(r[0]))[: int(q["limit"])]
            return {"retCode": 0, "result": {"symbol": q["symbol"], "list": rows}}
        if path == "/v5/market/funding/history":
            assert "startTime" not in q  # only endTime is given, as the probe found it works
            rows = [r for r in state.funding.get(q["symbol"], []) if r[0] <= int(q["endTime"])]
            rows = sorted(rows, key=lambda r: -r[0])[: int(q["limit"])]
            return {"retCode": 0, "result": {"list": [
                {"symbol": q["symbol"], "fundingRate": rate, "fundingRateTimestamp": str(t)}
                for t, rate in rows]}}  # fmt: skip
        raise AssertionError(path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            state.calls.append((u.path, dict(urllib.parse.parse_qsl(u.query))))
            body = json.dumps(answer(u.path, state.calls[-1][1])).encode()
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


def _asked(state, path: str, symbol: str, since: int = 0) -> list[dict]:
    return [q for p, q in state.calls[since:] if p == path and q.get("symbol") == symbol]


def test_every_perpetual_trading_or_delisted_and_their_days(bybit, tmp_path, capsys):
    bybit.instruments = [
        _perp("AAAUSDT", "Trading"),
        _perp("LUNAUSDT", "Closed", deliveryTime=str(_ms("2020-05-01"))),
        _perp("OLDUSDT", "Closed"),  # delisted, Bybit serves none of its bars
        _perp("RELUSDT", "Closed"),  # delisted, then listed again under the same name
        _perp("RELUSDT", "Trading"),
        _perp("BADUSDT", "Trading"),  # listed, but its symbol refused
        _perp("NEWUSDT", "PendingOpen"),
        _perp("PREUSDT", "PreLaunch"),
        _perp("BTCPERP", "Trading", quoteCoin="USDC"),
        _perp("AAAUSDT-26DEC25", "Trading", contractType="LinearFutures"),
    ]
    bybit.klines = {"AAAUSDT": _bars("2020-03-25", 10), "LUNAUSDT": _bars("2020-03-28", 5),
                    "RELUSDT": _bars("2020-04-01", 3)}  # fmt: skip
    bybit.funding = {"AAAUSDT": _settlements("2020-03-25", 30),
                     "LUNAUSDT": _settlements("2020-03-28", 13, every_h=4)}  # fmt: skip
    bybit.invalid.add("BADUSDT")
    bybit.busy.add("/v5/market/funding/history")  # the first funding request is refused once
    args = ["--start", "2020-03-01", "--end", "2020-04-03", "--out", str(tmp_path),
            "--base-url", bybit.url, "--pause", "0"]  # fmt: skip
    assert fb.main(args) == 0
    out = capsys.readouterr().out
    assert "5 USDT perpetuals on Bybit: Closed 2, Trading 3" in out
    assert "BADUSDT" in out and "no data served" in out
    assert "OLDUSDT" in out and "no bars served" in out
    assert "no bars served for 2 (1 of the 2 delisted)" in out
    um = tmp_path / "um"
    assert (um / "universe.csv").read_text().splitlines() == [
        "symbol,status,onboard_date,base,quote,underlying_type",
        f"AAAUSDT,Trading,{_ms('2020-03-25')},AAA,USDT,",
        f"BADUSDT,Trading,{_ms('2020-03-25')},BAD,USDT,",
        f"RELUSDT,Trading,{_ms('2020-03-25')},REL,USDT,",
    ]
    assert (um / "gone.csv").read_text().splitlines() == [
        "symbol,status,onboard_date,delivery_date,base,quote",
        f"LUNAUSDT,Closed,{_ms('2020-03-25')},{_ms('2020-05-01')},LUNA,USDT",
        f"OLDUSDT,Closed,{_ms('2020-03-25')},0,OLD,USDT",
    ]
    assert sorted(f.name for f in (um / "klines1d").iterdir()) == [
        "AAAUSDT.csv", "LUNAUSDT.csv", "RELUSDT.csv"]  # fmt: skip
    a = (um / "klines1d" / "AAAUSDT.csv").read_text().splitlines()
    assert a[0] == "open_time,open,high,low,close,volume,quote_volume,trades"
    assert a[1] == f"{_ms('2020-03-25')},1,2,0.5,100,10,1000.5,"  # oldest first, turnover
    assert len(a) == 1 + 10 and a[-1].startswith(str(_ms("2020-04-03")))
    assert len(_asked(bybit, "/v5/market/kline", "AAAUSDT")) == 3  # 4 + 4 + 2, newest first
    f = (um / "fundingRate" / "AAAUSDT" / "AAAUSDT-fundingRate.csv").read_text().splitlines()
    assert f[0] == "funding_time,funding_rate,mark_price" and len(f) == 1 + 30
    assert f[1] == f"{_ms('2020-03-25')},0.00010000," and f[3].split(",")[1] == "0.00030000"
    # the paging went back past the first settlement, to an empty page
    assert len(_asked(bybit, "/v5/market/funding/history", "AAAUSDT")) == 1 + 30 // 3 + 1
    # read as binance-data is: every day's funding summed, the delisted coin in the panel
    p = xs.load(tmp_path, "2020-04-01", "2020-04-03", held_out=True)
    assert p.symbols == ["AAAUSDT", "LUNAUSDT", "RELUSDT"]
    i, j = p.dates.index("2020-03-26"), p.symbols.index("AAAUSDT")
    assert p.close[i, j] == 101 and p.qvol[i, j] == 1001.5
    assert p.funding[i, j] == pytest.approx(1e-4 * (1 + 2 + 3))  # settlements 3, 4, 5
    k = p.symbols.index("LUNAUSDT")
    assert p.funding[p.dates.index("2020-03-29"), k] == pytest.approx(1e-4 * (1 + 2 + 3) * 2)
    assert int(p.settlements[p.dates.index("2020-03-29"), k]) == 6  # every four hours
    assert np.isnan(p.close[p.dates.index("2020-04-03"), k])  # delisted by then


def test_again_later_asks_only_for_what_comes_after(bybit, tmp_path, capsys):
    bybit.instruments = [_perp("AAAUSDT", "Trading")]
    bybit.klines = {"AAAUSDT": _bars("2020-03-25", 20)}
    bybit.funding = {"AAAUSDT": _settlements("2020-03-25", 60)}
    args = ["--out", str(tmp_path), "--base-url", bybit.url, "--pause", "0"]
    assert fb.main(["--start", "2020-03-25", "--end", "2020-04-03", *args]) == 0
    seen = len(bybit.calls)
    assert fb.main(["--start", "2020-03-25", "--end", "2020-04-05", *args]) == 0
    bars = _asked(bybit, "/v5/market/kline", "AAAUSDT", seen)
    assert len(bars) == 1 and int(bars[0]["start"]) == _ms("2020-04-04")
    rates = _asked(bybit, "/v5/market/funding/history", "AAAUSDT", seen)
    # 6 new, 3 a page, and the page that reaches the last one kept
    assert len(rates) == 3 and int(rates[0]["endTime"]) == _ms("2020-04-06") - 1
    um = tmp_path / "um"
    assert len((um / "klines1d" / "AAAUSDT.csv").read_text().splitlines()) == 1 + 12
    f = um / "fundingRate" / "AAAUSDT" / "AAAUSDT-fundingRate.csv"
    t = [int(r.split(",")[0]) for r in f.read_text().splitlines()[1:]]
    assert t == sorted(set(t)) and len(t) == 36  # 12 days, three a day, each once
    capsys.readouterr()


def test_a_day_not_closed_yet_and_a_failure_reported(bybit, tmp_path, capsys):
    today = dt.datetime.now(dt.UTC).date().isoformat()
    with pytest.raises(SystemExit, match="not closed yet"):
        fb.main(["--start", "2020-03-25", "--end", today, "--out", str(tmp_path)])
    bybit.instruments = [_perp("AAAUSDT", "Trading"), _perp("BBBUSDT", "Trading")]
    bybit.klines = {"AAAUSDT": _bars("2020-03-25", 3), "BBBUSDT": _bars("2020-03-25", 3)}
    bybit.funding = {"BBBUSDT": [(_ms("2020-03-25"), "not a rate")]}
    args = ["--start", "2020-03-25", "--end", "2020-03-27", "--out", str(tmp_path),
            "--base-url", bybit.url, "--pause", "0", "--tries", "2"]  # fmt: skip
    assert fb.main(args) == 0  # a coin with no funding at all is no failure
    assert (tmp_path / "um" / "fundingRate" / "AAAUSDT" / "AAAUSDT-fundingRate.csv").exists()
    bybit.busy |= {"/v5/market/kline"}
    bybit.klines["CCCUSDT"] = _bars("2020-03-25", 3)
    bybit.instruments.append(_perp("CCCUSDT", "Trading"))
    capsys.readouterr()
    assert fb.main([*args, "--tries", "1"]) == 1  # refused once, no second try: failed
    err = capsys.readouterr().err
    assert "FAILED" in err and "retCode 10006" in err and err.count("FAILED") == 2
