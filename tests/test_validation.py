"""The statistics of the protocol against answers known in advance."""

from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from marketlab import synth
from marketlab.io import Run
from marketlab.targets import future_returns, future_volatility
from marketlab.validation import metrics, splits, surrogates
from marketlab.validation.multiple import (
    benjamini_hochberg,
    deflated_sharpe,
    expected_max_sharpe,
)
from marketlab.validation.pbo import pbo

S = 1_000_000_000


def l20(ts, bid, ask):
    pad = [0] * 19
    return (ts, 1, [bid, *pad], [1, *pad], [ask, *pad], [1, *pad])


def test_future_returns_stop_at_the_end_of_the_segment():
    quality = [(0, "connected", ""), (1, "synced", ""), (int(4.5 * S), "gap", "")]
    mids = [(100, 102), (102, 104), (98, 100), (100, 102)]  # 101, 103, 99, 101
    rows = [l20((t + 1) * S, b, a) for t, (b, a) in enumerate(mids)]
    day = synth.memory_day(
        synth.tables(book_l20=rows, quality=quality), runs=[Run("t", 0, 10 * S, None)]
    )
    r = future_returns(day, ["1s", "2s"])
    assert r["ret_1s"].to_list()[:3] == pytest.approx(
        [np.log(103 / 101) * 1e4, np.log(99 / 103) * 1e4, np.log(101 / 99) * 1e4]
    )
    assert r["ret_1s"].to_list()[3] is None  # 5 s lies in the hole
    assert r["ret_2s"].to_list()[2:] == [None, None]
    v = future_volatility(day, ["2s"])["vol_2s"].to_list()
    expect = np.sqrt(np.log(103 / 101) ** 2 + np.log(99 / 103) ** 2) * 1e4
    assert v[0] == pytest.approx(expect)


def spans(groups, per=10, step=10, reach=0):
    """Samples of consecutive groups in time; each depends on `reach` beyond its time."""
    g = np.repeat(np.arange(groups), per)
    t = np.arange(len(g)) * step
    return g, t, t + reach


def test_purged_kfold_tests_each_group_once_and_purges_overlaps():
    g, t0, t1 = spans(5)
    folds = splits.purged_kfold(g, t0, t1, 5, embargo=0)
    assert sorted(i for f in folds for i in f.test) == list(range(50))
    for f in folds:
        assert not set(f.train) & set(f.test)
    # labels reaching 25 ahead overlap the next group's first samples: purged when it is tested
    g, t0, t1 = spans(5, reach=25)
    f = splits.purged_kfold(g, t0, t1, 5, embargo=0)[2]  # tests samples 20..29 (t 200..290)
    # their labels reach to 315: training samples whose span meets 200..315 go
    assert {18, 19, 30, 31}.isdisjoint(f.train)  # t1 205, 215; t0 300, 310
    assert {17, 32} <= set(f.train)  # t1 195 ends before; t0 320 starts after
    e = splits.purged_kfold(g, t0, t1, 5, embargo=30)[2]
    assert set(range(30, 35)).isdisjoint(e.train)  # starting within 30 after: 300..340
    assert 35 in e.train


def test_cpcv_builds_every_combination_and_the_paths_through_them():
    g, t0, t1 = spans(6)
    cv = splits.cpcv(g, t0, t1, 2, embargo=0)
    assert len(cv) == 15
    assert {f.test_groups for f in cv} == set(combinations(range(6), 2))
    paths = splits.cpcv_paths(cv)
    assert len(paths) == splits.n_paths(6, 2) == 5
    for p in paths:
        assert sorted(p) == list(range(6))  # every group once per path
        assert all(g in cv[i].test_groups for g, i in p.items())
    used = [(g, i) for p in paths for g, i in p.items()]
    assert len(used) == len(set(used)) == 30  # every prediction used once


def test_newey_west_widens_the_error_of_dependent_samples():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000)
    assert metrics.newey_west_se(x, 0) == pytest.approx(x.std() / np.sqrt(len(x)))
    y = np.convolve(x, np.ones(10) / 10, mode="same")  # strongly autocorrelated
    assert metrics.newey_west_se(y, 20) > 2 * metrics.newey_west_se(y, 0)


def test_day_ic_is_spearman_and_combines_over_days():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(1000)
    b = 0.3 * a + rng.standard_normal(1000)
    d = metrics.day_ic(a, b, lags=1)
    assert d.ic == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-12)
    a[::7] = np.nan
    assert metrics.day_ic(a, b, 1).n == 1000 - len(range(0, 1000, 7))
    s = metrics.combine([metrics.DayIC(0.1, 0.02, 10), metrics.DayIC(0.3, 0.02, 10)])
    assert s.mean == pytest.approx(0.2)
    assert s.se == pytest.approx(np.sqrt(2 * 0.02**2) / 2)
    assert s.p == pytest.approx(stats.norm.sf(0.2 / s.se))
    assert s.positive_share == 1.0


def test_surrogates_keep_the_values_and_lose_the_order():
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.standard_normal(4096))
    b = surrogates.block_shuffle(x, 64, rng)
    assert np.array_equal(np.sort(b), np.sort(x))
    starts = {round(v, 9) for v in x[::64]}
    assert {round(v, 9) for v in b[::64]} == starts  # whole blocks moved
    s = surrogates.iaaft(x, rng)
    assert np.array_equal(np.sort(s), np.sort(x))
    a, c = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    assert np.linalg.norm(a - c) / np.linalg.norm(a) < 0.05  # the spectrum, nearly
    assert abs(np.corrcoef(np.diff(x), np.diff(s))[0, 1]) < 0.1  # not the series itself
    r = surrogates.shift(x, 100, rng)
    lag = int(np.flatnonzero(r == x[0])[0])
    assert 100 <= lag <= len(x) - 100
    assert np.array_equal(np.roll(x, lag), r)


def test_pbo_is_low_for_a_real_edge_and_high_for_a_fit_to_noise():
    rng = np.random.default_rng(8)
    real = rng.standard_normal((32, 10)) * 0.1
    real[:, 3] += 1.0  # one variant better in every period
    assert pbo(real).pbo == 0.0
    noise = rng.standard_normal((32, 10))
    assert 0.25 < pbo(noise).pbo < 0.75  # no skill: the best in sample is a coin flip
    # two variants, each the best in one half and the worst in the other: whichever looks
    # best in sample is the worst out of sample
    trap = rng.standard_normal((32, 10)) * 0.01
    trap[:16, 0] += 5
    trap[16:, 0] -= 5
    trap[:16, 1] -= 5
    trap[16:, 1] += 5
    assert pbo(trap, groups=2).pbo == 1.0
    with pytest.raises(ValueError):
        pbo(np.zeros((1, 3)))


def test_benjamini_hochberg_by_hand():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005 0.01 0.03 0.04 -> times 4/rank: 0.02 0.02 0.04 0.04
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg(np.array([0.9])) == pytest.approx([0.9])


def test_deflated_sharpe_pays_for_the_trials():
    assert expected_max_sharpe(1.0, 1) == 0.0
    assert 0 < expected_max_sharpe(0.01, 10) < expected_max_sharpe(0.01, 1000)
    # one trial: the probabilistic Sharpe ratio; a zero Sharpe ratio is a coin flip
    assert deflated_sharpe(0.0, 250, 0.0, 3.0, 0.01, 1) == pytest.approx(0.5)
    one = deflated_sharpe(0.1, 250, 0.0, 3.0, 0.01, 1)
    many = deflated_sharpe(0.1, 250, 0.0, 3.0, 0.01, 100)
    assert one > many
    fat = deflated_sharpe(0.1, 250, -1.0, 10.0, 0.01, 1)
    assert fat < one  # negative skew and fat tails make the same ratio less sure
