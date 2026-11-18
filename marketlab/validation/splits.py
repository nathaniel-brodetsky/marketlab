"""Splitting by time without leaking: purged k-fold with an embargo, and combinatorial
purged cross-validation (CPCV), after Lopez de Prado, "Advances in Financial Machine
Learning", ch. 7 and 12.

Every sample carries the span of time it depends on: from the start of its feature window
`t0` to the end of its target `t1`. Samples are grouped in time order (by day here). A
training sample is dropped ("purged") when its span overlaps the span of any test group,
widened by the embargo after it, so that no training label overlaps what is being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray  # sample indices
    test: np.ndarray
    test_groups: tuple[int, ...]


def _purge(
    group: np.ndarray, t0: np.ndarray, t1: np.ndarray, test_groups: tuple[int, ...], embargo: int
) -> Split:
    test_mask = np.isin(group, test_groups)
    train_mask = ~test_mask
    for g in test_groups:
        mine = group == g
        if not mine.any():
            continue
        lo, hi = t0[mine].min(), t1[mine].max() + embargo
        train_mask &= ~((t1 >= lo) & (t0 <= hi))
    return Split(np.flatnonzero(train_mask), np.flatnonzero(test_mask), tuple(test_groups))


def purged_kfold(
    group: np.ndarray, t0: np.ndarray, t1: np.ndarray, k: int, embargo: int
) -> list[Split]:
    """K folds of contiguous groups, each tested once, trained on the rest after purging."""
    groups = np.unique(group)
    if not 2 <= k <= len(groups):
        raise ValueError(f"{k} folds need between 2 and {len(groups)} groups")
    folds = np.array_split(groups, k)
    return [_purge(group, t0, t1, tuple(int(g) for g in f), embargo) for f in folds]


def cpcv(
    group: np.ndarray, t0: np.ndarray, t1: np.ndarray, n_test: int, embargo: int
) -> list[Split]:
    """Every combination of `n_test` groups as the test set."""
    groups = [int(g) for g in np.unique(group)]
    if not 1 <= n_test < len(groups):
        raise ValueError(f"{n_test} test groups need more than that many groups")
    return [_purge(group, t0, t1, c, embargo) for c in combinations(groups, n_test)]


def cpcv_paths(splits: list[Split]) -> list[dict[int, int]]:
    """Backtest paths through the CPCV splits: each path takes every group's predictions
    from one split in which that group was tested, the j-th path from the j-th such split.
    With N groups and k test groups there are C(N - 1, k - 1) paths."""
    by_group: dict[int, list[int]] = {}
    for i, s in enumerate(splits):
        for g in s.test_groups:
            by_group.setdefault(g, []).append(i)
    counts = {len(v) for v in by_group.values()}
    if len(counts) != 1:
        raise ValueError("every group must be tested equally often")
    n = counts.pop()
    return [{g: v[j] for g, v in by_group.items()} for j in range(n)]


def n_paths(n_groups: int, n_test: int) -> int:
    return comb(n_groups - 1, n_test - 1)
