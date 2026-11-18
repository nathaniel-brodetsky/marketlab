"""Corrections for having tried many things.

- Benjamini-Hochberg over every hypothesis in the registry, failed ones included: the
  share of false discoveries among those called discoveries is kept at alpha.
- The deflated Sharpe ratio (Bailey, Lopez de Prado, "The deflated Sharpe ratio", 2014)
  for trading rules: the chance that the true Sharpe ratio is above what the best of N
  unskilled trials would show, given the skew and kurtosis of the returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values: a hypothesis is a discovery at FDR alpha if its value is <= alpha."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def expected_max_sharpe(trials_var: float, n_trials: int) -> float:
    """The Sharpe ratio the best of `n_trials` skill-less trials is expected to reach."""
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    z = (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(trials_var) * z)


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurtosis: float, trials_var: float, n_trials: int
) -> float:
    """`sr` per period (not annualized), `kurtosis` not in excess (3 for a normal)."""
    sr0 = expected_max_sharpe(trials_var, n_trials)
    denom = np.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))
