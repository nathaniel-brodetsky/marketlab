from pathlib import Path

import pytest

from marketlab import synth

DATA = Path(__file__).parent / "data"
CONTRACT = DATA / "contract"


def fixture_day(name: str) -> Path:
    """A day directory written by marketrec (see tests/data/README.md)."""
    return CONTRACT / name / "BTCUSDT" / "2026-11-21"


@pytest.fixture(scope="session")
def train_day():
    return synth.day(synth.Scenario(seed=101, seconds=900))


@pytest.fixture(scope="session")
def test_day():
    return synth.day(
        synth.Scenario(
            seed=102, seconds=300, gaps=[(60, 5)], reconnects=[(150, 4)], late=[(220, 3)]
        )
    )
