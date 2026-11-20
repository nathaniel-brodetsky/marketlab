"""The cost wall: the needed correlation worked out by hand, the returns on a grid of trades
and of hourly closes worked out by hand, and the command on a recorded-like archive."""

import numpy as np
import pytest
from scipy import stats
from test_carry import H, ms, write_klines
from test_tradeflow import write_day

from marketlab import costwall as cw
from marketlab.cli import HORIZONS_COSTWALL, main

NS = 1_000_000_000


def test_the_needed_correlation():
    # the mean of a standard normal beyond its top 5%: phi(1.645) / 0.05
    assert cw.kappa(0.05) == pytest.approx(stats.norm.pdf(stats.norm.isf(0.05)) / 0.05)
    assert cw.kappa(0.05) == pytest.approx(2.0627, abs=1e-3)
    assert cw.kappa(0.5) == pytest.approx(np.sqrt(2 / np.pi), rel=1e-9)  # trading always
    # 9 bp against a horizon whose return has a standard deviation of 100 bp
    assert cw.needed(9.0, 100.0, 0.05) == pytest.approx(9 / (100 * cw.kappa(0.05)))
    assert cw.needed(9.0, 2.0, 0.05) > 1  # no signal can pay that
    # simulated: a signal with correlation rho, its top and bottom 5% traded
    rng = np.random.default_rng(1)
    rho, sigma = 0.1, 40.0
    x = rng.normal(size=400_000)
    r = sigma * (rho * x + np.sqrt(1 - rho**2) * rng.normal(size=len(x)))
    lo, hi = np.quantile(x, [0.05, 0.95])
    gross = np.concatenate([r[x > hi], -r[x < lo]]).mean()
    assert gross == pytest.approx(rho * sigma * cw.kappa(0.05), rel=0.03)
    with pytest.raises(ValueError):
        cw.kappa(1.0)
    assert cw.duration_ns("1w") == 7 * cw.duration_ns("1d") == 7 * 24 * cw.duration_ns("1h")


def test_returns_on_a_grid_of_trades_by_hand():
    start = 1_000 * NS
    ts = (start + np.array([0.5, 1.2, 1.9, 3.4, 4.0, 6.1]) * NS).astype(np.int64)
    price = np.array([100.0, 101.0, 102.0, 99.0, 100.0, 104.0])
    r = cw.grid_returns(ts, price, start, 2 * NS)
    # grid at 2, 4, 6 s: the last prices then are 102, 100, 100
    assert r == pytest.approx(np.diff(np.log([102.0, 100.0, 100.0])) * 1e4)
    assert len(cw.grid_returns(ts[:1], price[:1], start, NS)) == 0


def test_returns_between_hourly_closes_by_hand():
    t = (np.arange(10) * H + ms("2024-01-01")) * 1_000_000  # bars opening at 0 .. 9 h
    t = np.delete(t, 5)  # no bar from 5 to 6 h
    close = 100.0 + np.arange(9)  # 100 .. 104 for bars 0 .. 4, 105 .. 108 for bars 6 .. 9
    r = cw.bar_returns(t, close, 2 * 3_600 * NS)
    # closes at 2, 4, 8 and 10 h: 101, 103, 106, 108; the 4 h to 8 h step is not 2 h
    assert r == pytest.approx(np.log([103 / 101, 108 / 106]) * 1e4)


def test_the_command(tmp_path, capsys):
    for k in range(3):
        write_day(tmp_path, f"2026-01-{k + 1:02d}", 0.0, seed=k, hours=1.0)
    rng = np.random.default_rng(2)
    t = np.arange(ms("2025-01-01"), ms("2026-01-04"), H)
    close = 97000 * np.exp(np.cumsum(rng.normal(0, 0.005, len(t))))
    write_klines(tmp_path, "spot", t, close, close)
    rows = cw.wall(tmp_path, horizons=HORIZONS_COSTWALL)
    assert [r.horizon for r in rows] == list(HORIZONS_COSTWALL)
    hourly = next(r for r in rows if r.horizon == "1h")
    assert hourly.sigma_bp == pytest.approx(50, rel=0.1)  # 0.5% an hour
    day = next(r for r in rows if r.horizon == "1d")
    assert day.sigma_bp == pytest.approx(50 * np.sqrt(24), rel=0.2)
    assert rows[0].source == "trades, 3 days" and rows[0].n > 3 * 3000
    assert main(["costwall", "--data", str(tmp_path), "--costs", "9", "1000"]) == 0
    out = capsys.readouterr().out
    assert "ic_for_9bp" in out and ">1" in out and "ic_for_1000bp" in out
    assert main(["costwall", "--data", str(tmp_path / "none")]) == 2
