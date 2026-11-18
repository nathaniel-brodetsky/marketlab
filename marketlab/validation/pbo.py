"""Probability of backtest overfitting by combinatorially symmetric cross-validation
(Bailey, Borwein, Lopez de Prado, Zhu, "The probability of backtest overfitting", 2017).

The input is the out-of-sample performance of N variants of a hypothesis in T periods
(days). The periods are cut into S groups; for every way of taking half of them as "in
sample", the best variant in sample is ranked among all variants on the other half. PBO is
the share of those choices in which it falls at or below the median: the chance that
picking the best-looking variant picks one that does no better than the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PBO:
    pbo: float
    combinations: int
    #: Median of the logit of the chosen variant's relative rank out of sample.
    median_logit: float


def pbo(perf: np.ndarray, groups: int = 16) -> PBO:
    """`perf`: periods by variants. `groups` is lowered to the even number of periods
    available; it must leave at least two groups."""
    t, n = perf.shape
    s = min(groups, t - t % 2)
    if s < 2 or n < 2:
        raise ValueError("PBO needs at least two periods and two variants")
    parts = np.array_split(np.arange(t), s)
    sums = np.stack([perf[p].sum(axis=0) for p in parts])  # groups by variants
    sizes = np.array([len(p) for p in parts], dtype=float)
    logits = []
    for chosen in combinations(range(s), s // 2):
        mask = np.zeros(s, dtype=bool)
        mask[list(chosen)] = True
        is_perf = sums[mask].sum(axis=0) / sizes[mask].sum()
        oos = sums[~mask].sum(axis=0) / sizes[~mask].sum()
        best = int(np.argmax(is_perf))
        rank = 1 + int((oos < oos[best]).sum()) + 0.5 * int((oos == oos[best]).sum() - 1)
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
    lam = np.array(logits)
    return PBO(float((lam <= 0).mean()), len(lam), float(np.median(lam)))
