"""The Hawkes process of market orders: the likelihood and its gradient against direct sums,
parameters recovered from simulated data, the time-rescaling check that accepts the right
model and rejects a wrong one, intensities, and rolling fits that use nothing later."""

import numpy as np
import pytest
from scipy import stats

from marketlab.events import hawkes as hk

TRUE = np.array([0.5, 0.4, 0.3, 0.1, 0.15, 0.25, 5.0])


def brute_loglik(t, side, horizon, p):
    mu = p[:2]
    a = np.array([[p[2], p[3]], [p[4], p[5]]])
    b = p[6]
    ll = 0.0
    for k in range(len(t)):
        i = side[k]
        lam = mu[i] + sum(a[i, side[j]] * b * np.exp(-b * (t[k] - t[j])) for j in range(k))
        ll += np.log(lam)
    comp = mu.sum() * horizon
    comp += sum(
        (a[0, side[j]] + a[1, side[j]]) * (1 - np.exp(-b * (horizon - t[j]))) for j in range(len(t))
    )
    return ll - comp


def test_the_likelihood_and_its_gradient_are_the_direct_sums():
    t, side = hk.simulate(TRUE, 40.0, 2)
    assert 20 < len(t) < 200
    grad = np.empty(7)
    for p in (TRUE, np.array([0.2, 0.9, 0.05, 0.6, 0.4, 0.01, 0.7])):
        assert hk.loglik(t, side, 40.0, p, grad) == pytest.approx(brute_loglik(t, side, 40.0, p))
        for k in range(7):
            e = np.zeros(7)
            e[k] = 1e-6 * p[k]
            num = (brute_loglik(t, side, 40.0, p + e) - brute_loglik(t, side, 40.0, p - e)) / (
                2 * e[k]
            )
            assert grad[k] == pytest.approx(num, rel=1e-4, abs=1e-6), (k, p)


def test_maximum_likelihood_finds_the_parameters_of_simulated_data():
    t, side = hk.simulate(TRUE, 20_000.0, 1)
    f = hk.fit(t, side, 20_000.0)
    assert f.converged
    assert f.params == pytest.approx(TRUE, rel=0.15)
    assert f.branching == pytest.approx(hk.branching_ratio(TRUE), rel=0.1)
    assert hk.branching_ratio(TRUE) == pytest.approx(
        max(abs(np.linalg.eigvals([[0.3, 0.1], [0.15, 0.25]])))
    )
    assert min(f.ks_p) > 0.01, "the right model passes the time-rescaling check"


def test_the_time_rescaling_check_rejects_a_poisson_model_of_clustered_events():
    t, side = hk.simulate(TRUE, 5_000.0, 3)
    rate = [(side == s).sum() / 5_000.0 for s in (0, 1)]
    poisson = np.array([rate[0], rate[1], 0, 0, 0, 0, 1.0])
    comp = hk.compensators(t, side, poisson)
    gaps = np.diff(np.concatenate([[0.0], comp[side == 0]]))
    assert stats.kstest(gaps, "expon").pvalue < 1e-6
    right = hk.compensators(t, side, TRUE)
    gaps = np.diff(np.concatenate([[0.0], right[side == 0]]))
    assert stats.kstest(gaps, "expon").pvalue > 0.01


def test_intensities_are_the_kernel_sums_before_each_query():
    t, side = hk.simulate(TRUE, 60.0, 4)
    query = np.array([0.5, 10.0, 10.0, 33.3, 59.9])
    params = np.tile(TRUE, (5, 1))
    params[2, 6] = 2.0  # another decay for one query
    params[4] = np.nan
    lam = hk.intensities(t, side, query, params, np.zeros(5, dtype=np.int64))
    for q in range(4):
        p = params[q]
        before = t < query[q]
        x = np.exp(-p[6] * (query[q] - t[before]))
        s0, s1 = x[side[before] == 0].sum(), x[side[before] == 1].sum()
        assert lam[q, 0] == pytest.approx(p[0] + p[6] * (p[2] * s0 + p[3] * s1))
        assert lam[q, 1] == pytest.approx(p[1] + p[6] * (p[4] * s0 + p[5] * s1))
    assert np.isnan(lam[4]).all()


def test_trades_of_one_market_order_are_one_event():
    recv = np.array([10, 11, 12, 20, 21, 30, 31])
    trade = np.array([5, 5, 5, 15, 15, 25, 25])
    buy = np.array([True, True, True, True, False, False, False])
    seg = np.array([0, 0, 0, 0, 0, 0, 1])
    ts, side, s = hk.market_orders(recv, trade, buy, seg)
    assert ts.tolist() == [10, 20, 21, 30, 31]
    assert side.tolist() == [0, 0, 1, 1, 1]
    assert s.tolist() == [0, 0, 0, 0, 1]


NS = 1_000_000_000


def day_events(seconds=3600.0, seed=5):
    t, side = hk.simulate(np.array([2.0, 1.8, 0.3, 0.1, 0.1, 0.3, 4.0]), seconds, seed)
    return (t * NS).astype(np.int64), side


def test_rolling_fits_stay_inside_intervals_and_use_nothing_later():
    ts, side = day_events()
    intervals = [(0, 1500 * NS), (1600 * NS, 3600 * NS)]
    window, step = 600 * NS, 300 * NS
    r = hk.rolling(ts, side, intervals, window, step)
    for at in r.at:
        assert any(lo <= at - window and at <= hi for lo, hi in intervals), at
    assert np.isfinite(r.params[:, 6]).sum() >= 8
    cut = 2500 * NS
    early = hk.rolling(ts[ts < cut], side[ts < cut], intervals, window, step)
    k = np.flatnonzero(r.at <= cut)
    assert np.array_equal(early.at[k], r.at[k])
    assert np.allclose(early.params[k], r.params[k], equal_nan=True)
    assert np.isfinite(r.branching[np.isfinite(r.params[:, 6])]).all()


def test_a_query_uses_the_last_fit_before_it_and_no_older_than_a_step():
    ts, side = day_events()
    step = 300 * NS
    r = hk.rolling(ts, side, [(0, 3600 * NS)], 600 * NS, step)
    q = np.array([599 * NS, 600 * NS, 899 * NS, 900 * NS, 3600 * NS + step + 1])
    lam, br = hk.at_times(r, ts, side, q, step)
    assert np.isnan(lam[0]).all(), "before the first fit"
    assert br[1] == r.branching[0] and br[2] == r.branching[0], "the fit made at 600 s"
    assert br[3] == r.branching[1]
    assert np.isnan(br[4]), "the last fit is more than a step old"
    assert (lam[1:4] > 0).all()
