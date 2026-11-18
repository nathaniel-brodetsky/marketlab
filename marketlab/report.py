"""Plain-text summaries for the command line."""

from __future__ import annotations

import numpy as np
import polars as pl


def describe(df: pl.DataFrame, skip: tuple[str, ...] = ("ts",)) -> str:
    """One line per column: share of rows with a value, then mean, std and percentiles."""
    head = f"{'column':<24}{'filled':>8}{'mean':>12}{'std':>12}{'p1':>12}{'p50':>12}{'p99':>12}"
    lines = [head]
    for c in df.columns:
        if c in skip:
            continue
        v = df[c].cast(pl.Float64).to_numpy()
        ok = np.isfinite(v)
        filled = ok.mean() if len(v) else 0.0
        if ok.any():
            x = v[ok]
            p1, p50, p99 = np.percentile(x, [1, 50, 99])
            stats = [x.mean(), x.std(), p1, p50, p99]
            nums = "".join(f"{s:>12.4g}" for s in stats)
        else:
            nums = f"{'-':>12}" * 5
        lines.append(f"{c:<24}{filled:>8.1%}{nums}")
    return "\n".join(lines)
