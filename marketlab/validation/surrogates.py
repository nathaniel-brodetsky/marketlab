"""Surrogate data: series that keep some properties of the real one and lose the relation
being tested. A result counts only if the real data beat the 95th percentile of the
surrogates.

- `block_shuffle` keeps the values and their dependence within blocks, and destroys the
  order of the blocks.
- `iaaft` (Schreiber, Schmitz 1996) keeps the distribution of values and, closely, the
  power spectrum, and destroys nonlinear structure.
- `shift` moves the target against the features by a random lag, keeping both series whole.
"""

from __future__ import annotations

import numpy as np

KINDS = ("block", "iaaft", "shift")


def block_shuffle(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x.copy()
    starts = np.arange(0, n, max(block, 1))
    order = rng.permutation(len(starts))
    return np.concatenate([x[s : s + block] for s in starts[order]])


def iaaft(x: np.ndarray, rng: np.random.Generator, iterations: int = 20) -> np.ndarray:
    """Iterate between imposing the power spectrum and the values of `x`, from a random
    permutation, until the order of the values stops changing or `iterations` runs out.
    The result holds exactly the values of `x`."""
    n = len(x)
    if n < 4:
        return rng.permutation(x)
    sorted_x = np.sort(x)
    amplitude = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    ranks = np.empty(n, dtype=np.int64)
    positions = np.arange(n)
    prev = None
    for _ in range(iterations):
        phases = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amplitude * np.exp(1j * phases), n)
        order = np.argsort(s, kind="stable")
        ranks[order] = positions
        s = sorted_x[ranks]
        if prev is not None and np.array_equal(order, prev):
            break
        prev = order
    return s


def shift(x: np.ndarray, min_lag: int, rng: np.random.Generator) -> np.ndarray:
    """Circular shift by a lag of at least `min_lag` from either end."""
    n = len(x)
    if n <= 2 * min_lag:
        return rng.permutation(x)
    return np.roll(x, int(rng.integers(min_lag, n - min_lag + 1)))
