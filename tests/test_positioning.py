"""Binance's positioning files: a day's value known at its close, nothing held out read on
the development days, the archive's files before the API's; their download, checked, and
the API's ratio at a close not out yet; the days to fetch; a contrarian rule on them as a
hypothesis and in the shadow; the committed 018 and 019."""

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

import numpy as np
import pytest
from test_tradeflow import PROV
from test_xsection import write

from marketlab import experiment as ex
from marketlab import positioning as pos
from marketlab import shadow
from marketlab import xsection as xs
from marketlab import xsection_kind as xk
from marketlab.cli import main

ROOT = Path(__file__).resolve().parents[1]
HEAD = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,"
        "sum_taker_long_short_vol_ratio")  # fmt: skip


def metrics_zip(sym: str, day: str, rows: list[tuple]) -> bytes:
    """A day's file: rows of (time, open interest, top ratio, all ratio, takers' ratio)."""
    lines = [HEAD] + [f"{day} {t},{sym},{oi},{oi * 10},1.0,{top},{al},{tk}"
                      for t, oi, top, al, tk in rows]  # fmt: skip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{sym}-metrics-{day}.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def put(root, sym, day, rows):
    f = pos.path(root, sym, day)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(metrics_zip(sym, day, rows))


def test_a_days_value_and_what_is_read(tmp_path):
    put(tmp_path, "AAAUSDT", "2023-12-30", [("23:55:00", 120, 1.5, 2.5, 1.2),
                                             ("00:00:00", 100, 1.1, 2.0, 0.8),
                                             ("12:00:00", 110, 1.3, 2.2, 1.0)])  # fmt: skip
    v = pos.day_values(pos.path(tmp_path, "AAAUSDT", "2023-12-30"))
    assert v["sum_open_interest"] == 120 and v["count_long_short_ratio"] == 2.5  # its last row
    assert v["sum_toptrader_long_short_ratio"] == 1.5
    assert v["sum_taker_long_short_vol_ratio"] == pytest.approx(1.0)  # the day's mean
    days = 12
    for s in ("AAAUSDT", "BBBUSDT"):
        write(tmp_path, s, "2023-12-25", np.full(days, 10.0), np.full(days, 1e6))
    p = xs.load(tmp_path, "2023-12-25", "2023-12-31")
    got = pos.read(tmp_path, p)
    i, j = p.dates.index("2023-12-30"), p.symbols.index("AAAUSDT")
    assert got["count_long_short_ratio"][i, j] == 2.5
    assert np.isnan(got["count_long_short_ratio"][i - 1, j])  # no file that day
    assert np.isnan(got["count_long_short_ratio"][:, p.symbols.index("BBBUSDT")]).all()
    q = xs.load(tmp_path, "2023-12-25", "2024-01-05", held_out=True)
    with pytest.raises(xs.HeldOut):
        pos.read(tmp_path, q)
    # the API's ratio at a close, where the archive has no file, for the shadow only
    live = tmp_path / "um" / "metrics" / "AAAUSDT" / "live.csv"
    live.write_text("day,count_long_short_ratio,taken_at\n2024-01-04,3.0,x\n2023-12-30,9.9,x\n")
    k = q.dates.index("2024-01-04")
    assert np.isnan(pos.read(tmp_path, q, held_out=True)["count_long_short_ratio"][k, j])
    both = pos.read(tmp_path, q, held_out=True, live=True)["count_long_short_ratio"]
    assert both[k, j] == 3.0 and both[q.dates.index("2023-12-30"), j] == 2.5  # the archive's first
    sig = pos.signals(tmp_path, p)
    assert sig["crowd"][i, j] == -2.5 and sig["crowd_top"][i, j] == -1.5
    assert sig["takers"][i, j] == pytest.approx(-1.0)


def test_open_interest_over_7_days_and_the_days_to_fetch(tmp_path):
    for s in ("AAAUSDT", "BBBUSDT"):
        write(tmp_path, s, "2023-01-01", np.full(40, 10.0), np.full(40, 1e6))
    for k, day in enumerate(("2023-01-20", "2023-01-27")):
        put(tmp_path, "AAAUSDT", day, [("23:55:00", 100 * (k + 1), 1, 1, 1)])
    p = xs.load(tmp_path, "2023-01-01", "2023-02-09")
    oi7 = pos.signals(tmp_path, p)["oi7"]
    j = p.symbols.index("AAAUSDT")
    assert oi7[p.dates.index("2023-01-27"), j] == pytest.approx(-np.log(2.0))
    assert np.isnan(oi7[p.dates.index("2023-01-26"), j])
    # in the universe from the day with 30 days of bars before it: those days, and 7 before
    pairs = pos.wanted(p, "2023-01-01", top=10, before=7)
    first = min(d for s, d in pairs if s == "AAAUSDT")
    assert first == (dt.date(2023, 1, 31) - dt.timedelta(days=7)).isoformat()
    assert max(d for s, d in pairs) == "2023-02-09" and len({s for s, _ in pairs}) == 2
    assert pos.wanted(p, "2023-02-05", top=10)[0][1] == "2023-02-05"  # nothing before `first`
    out = tmp_path / "runs" / "days.csv"
    assert main(["metrics", "days", "--data", str(tmp_path), "--from", "2023-01-01", "--to",
                 "2023-02-09", "--top", "10", "--out", str(out)]) == 0  # fmt: skip
    assert out.read_text().splitlines()[0] == "symbol,day"
    assert len(out.read_text().splitlines()) == 1 + len(pairs)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fm = _load("fetch_metrics")


@pytest.fixture
def archive(monkeypatch):
    for k in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(k, "*")
    state = types.SimpleNamespace(files={}, api={}, calls=[])

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            state.calls.append(self.path)
            if u.path == "/futures/data/globalLongShortAccountRatio":
                q = dict(urllib.parse.parse_qsl(u.query))
                # from startTime on, and, as a lax server might, past endTime too
                rows = [r for r in state.api.get(q["symbol"], [])
                        if int(q["startTime"]) <= r["timestamp"]]  # fmt: skip
                body = json.dumps(rows).encode()
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


def test_the_files_fetched_checked_and_the_apis_ratio_at_a_close(archive, tmp_path, capsys):
    data = metrics_zip("AAAUSDT", "2026-11-20", [("23:55:00", 1, 1, 1.7, 1)])
    name = "AAAUSDT-metrics-2026-11-20.zip"
    folder = "/data/futures/um/daily/metrics/AAAUSDT/"
    archive.files[folder + name] = data
    archive.files[folder + name + ".CHECKSUM"] = (
        f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode())  # fmt: skip
    close = int(dt.datetime(2026, 11, 22, tzinfo=dt.UTC).timestamp() * 1000)  # 21st's close
    archive.api["AAAUSDT"] = [{"symbol": "AAAUSDT", "longShortRatio": str(v), "timestamp": t}
                              for t, v in ((close - 600_000, 2.1), (close - 300_000, 2.2),
                                           (close, 9.9))]  # fmt: skip
    days = tmp_path / "days.csv"
    days.write_text("symbol,day\nAAAUSDT,2026-11-20\nAAAUSDT,2026-11-21\nAAAUSDT,2026-10-01\n")
    args = ["--days-file", str(days), "--out", str(tmp_path / "d"), "--base-url", archive.url,
            "--api-url", archive.url, "--tries", "1"]  # fmt: skip
    assert fm.main([*args, "--live", "--close", "2026-11-21"]) == 0
    out = capsys.readouterr().out
    assert "'have'" not in out and "'missing': 2" in out and "'ok': 1" in out
    assert pos.path(tmp_path / "d", "AAAUSDT", "2026-11-20").read_bytes() == data
    # the 21st's file not out: the API's last 5 minutes before its close; the 1st of October,
    # too old for the API, nothing
    live = (tmp_path / "d" / "um" / "metrics" / "AAAUSDT" / "live.csv").read_text().splitlines()
    assert live[0] == "day,count_long_short_ratio,taken_at" and len(live) == 2
    assert live[1].startswith("2026-11-21,2.2,")
    # again: nothing fetched twice, and a close taken stays as it was
    archive.api["AAAUSDT"][1]["longShortRatio"] = "5.0"
    seen = len(archive.calls)
    assert fm.main([*args, "--live", "--close", "2026-11-21"]) == 0
    assert "'have': 1" in capsys.readouterr().out
    assert not [c for c in archive.calls[seen:] if c.endswith(name)]
    assert (tmp_path / "d" / "um" / "metrics" / "AAAUSDT" / "live.csv").read_text().splitlines()[
        1].startswith("2026-11-21,2.2,")  # fmt: skip
    # a file whose bytes do not match its checksum: failed
    archive.files[folder + name] = data + b"x"
    pos.path(tmp_path / "d", "AAAUSDT", "2026-11-20").unlink()
    assert fm.main(args) == 1
    assert "bad checksum" in capsys.readouterr().err


SPEC = """
id: "041"
name: planted-crowd
kind: xsection
hypothesis: The crowd is wrong.
data:
  days: {from: 2023-08-01, to: 2023-12-31}
  holdout: {from: 2024-01-01, to: 2024-02-15}
rule: {signal: crowd, hold: 7, staggered: true, top: 12, q: 0.2}
variants:
  - {hold: 1}
  - {q: 0.1}
trials:
  - {hold: 1}
  - {signal: takers}
costs: {round_trip_bp: 9.0}
protocol: {nw_lags: 10}
success: {alpha: 0.05, max_pbo: 0.5, min_positive_months: 0.5, min_positive_years: 1,
          max_abs_beta: 0.3}
"""


def planted_crowd(root, n_sym=15, first="2023-06-01", days=260, seed=3):
    """The crowd's ratio a random walk a coin; each day the coins it is longest on fall."""
    rng = np.random.default_rng(seed)
    d0 = dt.date.fromisoformat(first)
    ratio = 1.5 + np.cumsum(rng.normal(0, 0.05, (days, n_sym)), axis=0)
    move = rng.normal(0, 0.02, (days, n_sym))
    move[1:] -= 0.02 * (ratio[:-1] - ratio[:-1].mean(axis=1, keepdims=True))
    close = 10 * np.exp(np.cumsum(move, axis=0))
    for j in range(n_sym):
        sym = f"C{j:02d}USDT"
        write(root, sym, first, close[:, j], np.full(days, 1e6 * (1 + j)))
        for i in range(days):
            day = (d0 + dt.timedelta(days=i)).isoformat()
            put(root, sym, day, [("23:55:00", 100 + i, 1.0, round(ratio[i, j], 6), 1.0)])
    return ratio


def test_a_contrarian_rule_on_the_crowd(tmp_path):
    planted_crowd(tmp_path / "d")
    f = tmp_path / "experiments" / "041-planted-crowd.yaml"
    f.parent.mkdir(parents=True)
    f.write_text(SPEC)
    spec = xk.load_spec(f)
    assert spec.primary.signal == "crowd" and spec.trials[2].signal == "takers"
    p = xs.load(tmp_path / "d", spec.first, spec.last)
    got = xk.follow(p, spec.primary, spec.first, 9.0, {}, tmp_path / "d")
    start = p.dates.index(spec.first)
    want, _, _ = xs.portfolio(pos.signals(tmp_path / "d", p)["crowd"], xs.universe(p, 12),
                              xs.daily_returns(p), start, 7, 0.2, 9.0, True)  # fmt: skip
    assert np.allclose(got.net, want[start:], rtol=0, atol=1e-15) and got.net.mean() > 0
    with pytest.raises(ex.ExperimentError, match="positioning is read from a data root"):
        xk.follow(p, spec.primary, spec.first, 9.0, {})
    # a run reads the archive only: an API ratio where a file is missing is not its
    pos.path(tmp_path / "d", "C05USDT", "2023-09-01").unlink()
    (tmp_path / "d" / "um" / "metrics" / "C05USDT" / "live.csv").write_text(
        "day,count_long_short_ratio,taken_at\n2023-09-01,99.0,x\n")  # fmt: skip
    gap = xk.follow(p, spec.primary, spec.first, 9.0, {}, tmp_path / "d")
    want, _, _ = xs.portfolio(pos.signals(tmp_path / "d", p)["crowd"], xs.universe(p, 12),
                              xs.daily_returns(p), start, 7, 0.2, 9.0, True)  # fmt: skip
    assert np.allclose(gap.net, want[start:], rtol=0, atol=1e-15)
    i, j = p.dates.index("2023-09-01"), p.symbols.index("C05USDT")
    assert not np.isfinite(pos.signals(tmp_path / "d", p)["crowd"][i, j])
    # the development run reads no held-out day's file: one made nonsense changes nothing
    report = xk.run(spec, tmp_path / "d", tmp_path / "results", prov=PROV)
    for k in range(40):
        day = (dt.date(2024, 1, 1) + dt.timedelta(days=k)).isoformat()
        pos.path(tmp_path / "d", "C00USDT", day).write_bytes(b"not a zip")
    again = xk.run(spec, tmp_path / "d", tmp_path / "results2", prov=PROV)
    assert again.table == report.table and report.verdict == "passed"
    # the held-out run reads them
    with pytest.raises(Exception, match="zip"):
        xk.final(spec, tmp_path / "d", tmp_path / "results", prov=PROV)


def test_the_shadow_decides_from_the_archive_and_the_apis_ratio(tmp_path):
    ratio = planted_crowd(tmp_path / "d")
    f = tmp_path / "experiments" / "041-planted-crowd.yaml"
    f.parent.mkdir(parents=True)
    f.write_text(SPEC)
    spec = xk.load_spec(f)
    close = "2024-01-20"
    d = shadow.decide(tmp_path / "d", spec, close, today="2026-11-24")
    p = xs.load(tmp_path / "d", spec.holdout_first, "2024-02-15", held_out=True)
    held, _, _, _ = xs.holdings(pos.signals(tmp_path / "d", p, held_out=True)["crowd"],
                                xs.universe(p, 12), p.dates.index(spec.holdout_first), 7, 0.2,
                                True)  # fmt: skip
    i = p.dates.index(close)
    assert d.weights == pytest.approx({p.symbols[j]: held[i + 1, j]
                                       for j in np.flatnonzero(held[i + 1])})  # fmt: skip
    # the close's file not out: the API's ratio stands in for it, for the shadow
    j = 3
    sym = f"C{j:02d}USDT"
    pos.path(tmp_path / "d", sym, close).unlink()
    gone = shadow.decide(tmp_path / "d", spec, close, today="2026-11-24")
    (tmp_path / "d" / "um" / "metrics" / sym / "live.csv").write_text(
        f"day,count_long_short_ratio,taken_at\n{close},{round(ratio[233, j], 6)},x\n")  # fmt: skip
    assert dt.date(2023, 6, 1) + dt.timedelta(days=233) == dt.date(2024, 1, 20)
    back = shadow.decide(tmp_path / "d", spec, close, today="2026-11-24")
    assert back.weights == pytest.approx(d.weights) and gone.weights != back.weights


def test_the_committed_crowd_hypotheses_load():
    a = xk.load_spec(ROOT / "experiments" / "018-xsection-crowd.yaml")
    b = xk.load_spec(ROOT / "experiments" / "019-xsection-crowd-forward.yaml")
    rule = xk.Rule("crowd", 7, 50, 0.2, 1, True)
    assert a.primary == b.primary == rule and a.trials == b.trials and a.variants == b.variants
    assert (a.first, a.last) == ("2022-01-01", "2023-12-31")
    assert (a.holdout_first, a.holdout_last) == ("2024-01-01", "2026-10-31")
    assert (b.holdout_first, b.holdout_last) == ("2026-12-01", "2028-11-30")
    assert [(v.hold, v.q, v.top) for v in a.variants] == [(7, 0.2, 50), (1, 0.2, 50),
                                                          (7, 0.1, 50), (7, 0.2, 30)]  # fmt: skip
    assert len(a.trials) == 9 and {t.signal for t in a.trials} == set(pos.SIGNALS)
    assert sum(t.hours == 0 for t in a.trials) == 8
