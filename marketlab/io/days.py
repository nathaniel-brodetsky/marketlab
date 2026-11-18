"""Finding the recorded days of a symbol, and the ones fit for research."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .contract import read_manifest
from .day import Day

NS_PER_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class DayInfo:
    path: Path
    date: str
    #: Hours with a valid book that is not late.
    usable_hours: float


def find_days(root: str | Path, symbol: str) -> list[Path]:
    """Day directories with a manifest under `<root>/binance/<symbol>`, by date."""
    base = Path(root) / "binance" / symbol
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "manifest.json").is_file())


def survey(root: str | Path, symbol: str, exclude: Sequence[str] = ()) -> list[DayInfo]:
    """Every finished day with its usable hours, by date, leaving out the dates in
    `exclude`."""
    out = []
    for p in find_days(root, symbol):
        if p.name in exclude or read_manifest(p) is None:
            continue
        hours = sum(e - s for s, e in Day.open(p).intervals) / NS_PER_HOUR
        out.append(DayInfo(p, p.name, hours))
    return out


def select_days(
    root: str | Path,
    symbol: str,
    min_hours: float,
    exclude: Sequence[str] = (),
) -> list[DayInfo]:
    """The days with at least `min_hours` of usable data, by date, leaving out the dates in
    `exclude`."""
    return [d for d in survey(root, symbol, exclude) if d.usable_hours >= min_hours]
