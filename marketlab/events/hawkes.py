"""Layer 2: aggressive buying and selling as a two-dimensional Hawkes process.

Each market order raises, for a while, the chance of the next ones on either side:

    lambda_i(t) = mu_i + sum_j alpha_ij * beta * sum_{t_k^j < t} exp(-beta (t - t_k^j))

with i, j in {buy, sell}, one decay `beta` for all four kernels. `alpha_ij` is how many
events of side i one event of side j causes on average, so the spectral radius of `alpha`
is the branching ratio: the share of events caused by earlier ones (below 1 for a process
that does not explode).

**Events** are market orders, not trades: the trades of one aggressive order share their
exchange time and side, and one order sweeping five levels is one event, not five that
arrive together. Their time is when the first of them was received, the clock everything
else runs on.

**Estimation** is by maximum likelihood on a window of events, with the gradient written
out, in loops compiled by numba: the log-likelihood and its gradient in one pass over the
events, carrying for each side the decayed sum S_j(t) = sum exp(-beta (t - t_k^j)) and its
derivative in beta, D_j(t) = sum (t - t_k^j) exp(-beta (t - t_k^j)). Parameters are
optimised on the log scale, so they stay positive. History before a window is left out.

**Check of the model** by the time-rescaling theorem: between consecutive events of a
side, the compensator's increase is exponential with mean 1 if the model is right. Each
window gets a Kolmogorov-Smirnov p-value per side; windows where the model fails are
marked, not hidden.

**Rolling**: a fit every `step` on the `window` of events before it, inside one valid
segment; it describes the time from its fit to the next one and uses nothing later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from scipy import optimize, stats

NS = 1_000_000_000
#: Parameter order everywhere: mu_buy, mu_sell, alpha_bb, alpha_bs, alpha_sb, alpha_ss, beta.
NAMES = ("mu_buy", "mu_sell", "a_bb", "a_bs", "a_sb", "a_ss", "beta")
_LO = np.log(np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-3]))
_HI = np.log(np.array([1e6, 1e6, 10.0, 10.0, 10.0, 10.0, 1e5]))


@numba.njit(cache=True)
def loglik(
    t: np.ndarray, side: np.ndarray, horizon: float, p: np.ndarray, grad: np.ndarray
) -> float:
    """Log-likelihood of events at times `t` (seconds from the window's start, sorted) of
    sides `side` (0 buy, 1 sell) on [0, horizon]; the gradient in the parameters is
    written into `grad`."""
    mu0, mu1, a00, a01, a10, a11, beta = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    for k in range(7):
        grad[k] = 0.0
    s0 = s1 = d0 = d1 = 0.0
    last = 0.0
    ll = 0.0
    c0 = c1 = cd0 = cd1 = 0.0
    for k in range(len(t)):
        dt = t[k] - last
        e = np.exp(-beta * dt)
        d0 = e * (d0 + dt * s0)
        d1 = e * (d1 + dt * s1)
        s0 *= e
        s1 *= e
        last = t[k]
        if side[k] == 0:
            lam = mu0 + beta * (a00 * s0 + a01 * s1)
            inv = 1.0 / lam
            grad[0] += inv
            grad[2] += beta * s0 * inv
            grad[3] += beta * s1 * inv
            grad[6] += (a00 * (s0 - beta * d0) + a01 * (s1 - beta * d1)) * inv
            s0 += 1.0
        else:
            lam = mu1 + beta * (a10 * s0 + a11 * s1)
            inv = 1.0 / lam
            grad[1] += inv
            grad[4] += beta * s0 * inv
            grad[5] += beta * s1 * inv
            grad[6] += (a10 * (s0 - beta * d0) + a11 * (s1 - beta * d1)) * inv
            s1 += 1.0
        ll += np.log(lam)
        tau = horizon - t[k]
        x = np.exp(-beta * tau)
        if side[k] == 0:
            c0 += 1.0 - x
            cd0 += tau * x
        else:
            c1 += 1.0 - x
            cd1 += tau * x
    ll -= mu0 * horizon + a00 * c0 + a01 * c1 + mu1 * horizon + a10 * c0 + a11 * c1
    grad[0] -= horizon
    grad[1] -= horizon
    grad[2] -= c0
    grad[3] -= c1
    grad[4] -= c0
    grad[5] -= c1
    grad[6] -= (a00 + a10) * cd0 + (a01 + a11) * cd1
    return ll


@numba.njit(cache=True)
def compensators(t: np.ndarray, side: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The compensator of each event's own side at the event: Lambda_i(t_k) = mu_i t_k +
    sum_j alpha_ij (N_j(t_k) - S_j(t_k)), with N_j the events of side j before t_k."""
    mu0, mu1, a00, a01, a10, a11, beta = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    out = np.empty(len(t))
    s0 = s1 = 0.0
    n0 = n1 = 0.0
    last = 0.0
    for k in range(len(t)):
        e = np.exp(-beta * (t[k] - last))
        s0 *= e
        s1 *= e
        last = t[k]
        if side[k] == 0:
            out[k] = mu0 * t[k] + a00 * (n0 - s0) + a01 * (n1 - s1)
            s0 += 1.0
            n0 += 1.0
        else:
            out[k] = mu1 * t[k] + a10 * (n0 - s0) + a11 * (n1 - s1)
            s1 += 1.0
            n1 += 1.0
    return out


@numba.njit(cache=True)
def intensities(
    t: np.ndarray, side: np.ndarray, query: np.ndarray, params: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """lambda_buy and lambda_sell just before each query time, with that query's parameters
    (a row of `params`), from the events `start[q]` on that come before it."""
    out = np.full((len(query), 2), np.nan)
    for q in range(len(query)):
        p = params[q]
        if not np.isfinite(p[6]):
            continue
        beta = p[6]
        s0 = s1 = 0.0
        k = start[q]
        while k < len(t) and t[k] < query[q]:
            x = np.exp(-beta * (query[q] - t[k]))
            if side[k] == 0:
                s0 += x
            else:
                s1 += x
            k += 1
        out[q, 0] = p[0] + beta * (p[2] * s0 + p[3] * s1)
        out[q, 1] = p[1] + beta * (p[4] * s0 + p[5] * s1)
    return out


@numba.njit(cache=True)
def simulate(p: np.ndarray, horizon: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Events of the process on [0, horizon] by Ogata's thinning, for tests."""
    np.random.seed(seed)  # noqa: NPY002 (numba compiles the legacy generator)
    mu0, mu1, a00, a01, a10, a11, beta = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    times = []
    sides = []
    s0 = s1 = 0.0
    t = 0.0
    while True:
        bound = mu0 + mu1 + beta * ((a00 + a10) * s0 + (a01 + a11) * s1)
        dt = np.random.exponential(1.0 / bound)  # noqa: NPY002
        t += dt
        if t > horizon:
            break
        e = np.exp(-beta * dt)
        s0 *= e
        s1 *= e
        l0 = mu0 + beta * (a00 * s0 + a01 * s1)
        l1 = mu1 + beta * (a10 * s0 + a11 * s1)
        u = np.random.random() * bound  # noqa: NPY002
        if u < l0:
            times.append(t)
            sides.append(0)
            s0 += 1.0
        elif u < l0 + l1:
            times.append(t)
            sides.append(1)
            s1 += 1.0
    return np.array(times), np.array(sides, dtype=np.int8)


def branching_ratio(p: np.ndarray) -> float:
    """Spectral radius of the 2x2 matrix alpha."""
    a = np.array([[p[2], p[3]], [p[4], p[5]]])
    return float(np.max(np.abs(np.linalg.eigvals(a))))


@dataclass(frozen=True)
class Fit:
    params: np.ndarray  # NAMES order
    loglik: float
    events: tuple[int, int]  # buys, sells
    ks_p: tuple[float, float]  # time-rescaling check per side
    converged: bool

    @property
    def branching(self) -> float:
        return branching_ratio(self.params)


def start_values(t: np.ndarray, side: np.ndarray, horizon: float) -> np.ndarray:
    """Half of each side's rate as background, a quarter of an event caused per event and
    kernel, and a decay of the mean gap between events."""
    n0, n1 = int((side == 0).sum()), int((side == 1).sum())
    gap = horizon / max(len(t), 1)
    return np.array(
        [0.5 * max(n0, 1) / horizon, 0.5 * max(n1, 1) / horizon, 0.25, 0.25, 0.25, 0.25, 1.0 / gap]
    )


def fit(t: np.ndarray, side: np.ndarray, horizon: float, start: np.ndarray | None = None) -> Fit:
    """Maximum likelihood on one window; `t` in seconds from its start."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    side = np.ascontiguousarray(side, dtype=np.int8)
    p0 = start_values(t, side, horizon) if start is None else np.asarray(start, float)
    theta0 = np.clip(np.log(p0), _LO, _HI)
    grad = np.empty(7)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        p = np.exp(theta)
        ll = loglik(t, side, horizon, p, grad)
        if not np.isfinite(ll):
            return 1e300, np.zeros(7)
        return -ll, -grad * p

    res = optimize.minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=list(zip(_LO, _HI, strict=True))
    )
    p = np.exp(res.x)
    comp = compensators(t, side, p)
    ks = []
    for s in (0, 1):
        c = comp[side == s]
        gaps = np.diff(np.concatenate([[0.0], c]))
        ks.append(float(stats.kstest(gaps, "expon").pvalue) if len(gaps) >= 5 else float("nan"))
    return Fit(
        params=p,
        loglik=-float(res.fun),
        events=(int((side == 0).sum()), int((side == 1).sum())),
        ks_p=(ks[0], ks[1]),
        converged=bool(res.success),
    )


# --- a day's events and rolling fits ---------------------------------------------------


def market_orders(
    recv_ts: np.ndarray, trade_ts: np.ndarray, buy: np.ndarray, segment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trades (in receive order) grouped into market orders: consecutive trades of one side
    with one exchange time and segment. (receive time of the first, side 0 buy / 1 sell,
    segment)."""
    if len(recv_ts) == 0:
        return recv_ts, np.zeros(0, np.int8), segment
    new = np.ones(len(recv_ts), dtype=bool)
    new[1:] = (
        (trade_ts[1:] != trade_ts[:-1]) | (buy[1:] != buy[:-1]) | (segment[1:] != segment[:-1])
    )
    k = np.flatnonzero(new)
    return recv_ts[k], np.where(buy[k], 0, 1).astype(np.int8), segment[k]


@dataclass(frozen=True)
class Rolling:
    """Fits every `step`, each on the `window` before its time."""

    at: np.ndarray  # fit times, ns
    params: np.ndarray  # (fits, 7), NaN where no fit
    branching: np.ndarray
    ks_p: np.ndarray  # (fits, 2)
    events: np.ndarray  # (fits,)


def rolling(
    ts: np.ndarray,
    side: np.ndarray,
    intervals: list[tuple[int, int]],
    window: int,
    step: int,
    min_events: int = 50,
) -> Rolling:
    """A fit at every multiple of `step` whose window `[at - window, at)` lies inside one
    valid interval and holds at least `min_events` of each side; each starts from the fit
    before it in the same interval."""
    at_all, rows = [], []
    for lo, hi in intervals:
        first = -(-(lo + window) // step) * step
        prev = None
        for at in range(first, hi + 1, step):
            a, b = np.searchsorted(ts, [at - window, at], side="left")
            s = side[a:b]
            at_all.append(at)
            if (s == 0).sum() < min_events or (s == 1).sum() < min_events:
                rows.append(None)
                prev = None
                continue
            f = fit((ts[a:b] - (at - window)) / NS, s, window / NS, prev)
            rows.append(f)
            prev = f.params
    n = len(at_all)
    params = np.full((n, 7), np.nan)
    ks = np.full((n, 2), np.nan)
    br = np.full(n, np.nan)
    ev = np.zeros(n, dtype=np.int64)
    for k, f in enumerate(rows):
        if f is not None:
            params[k], ks[k], br[k], ev[k] = f.params, f.ks_p, f.branching, sum(f.events)
    return Rolling(np.array(at_all, dtype=np.int64), params, br, ks, ev)


def at_times(
    r: Rolling, ts: np.ndarray, side: np.ndarray, query: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each query time: lambda_buy and lambda_sell (events per second) from the fit
    made at or before it, at most `step` before; and that fit's branching ratio. NaN where
    there is none."""
    k = np.searchsorted(r.at, query, side="right") - 1
    ok = (
        (k >= 0) & (query - r.at[np.clip(k, 0, None)] < step)
        if len(r.at)
        else np.zeros(len(query), bool)
    )
    params = np.full((len(query), 7), np.nan)
    br = np.full(len(query), np.nan)
    params[ok] = r.params[k[ok]]
    br[ok] = r.branching[k[ok]]
    beta = params[:, 6]
    reach = np.where(np.isfinite(beta), 30.0 / np.where(np.isfinite(beta), beta, 1.0), 0.0)
    start = np.searchsorted(ts, query - (reach * NS).astype(np.int64), side="left")
    t = (ts - ts[0]) / NS if len(ts) else np.zeros(0)
    q = (query - (ts[0] if len(ts) else 0)) / NS
    lam = intensities(t, side, q.astype(np.float64), params, start.astype(np.int64))
    return lam, br
