"""No method looks ahead: cut a day at a random time, and nothing before the cut changes.

This covers every method in the registry, so a new method is tested by registering it.
Fitted methods are fitted once on a separate training day; what they learned is fixed, and
the test is about what `transform` reads.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from marketlab import bars, synth
from marketlab.registry import FITTED, METHODS

S = 1_000_000_000

_DAYS: dict[int, object] = {}


def scenario_day(seed: int):
    if seed not in _DAYS:
        _DAYS[seed] = synth.day(
            synth.Scenario(
                seed=seed,
                seconds=40,
                gaps=[(8 + seed % 5, 2)],
                reconnects=[(20, 3)],
                late=[(30 + seed % 3, 2)],
            )
        )
    return _DAYS[seed]


@pytest.fixture(scope="module")
def fitted(train_day):
    out = {}
    for name, cls in METHODS.items():
        m = cls(windows=("1s", "5s")) if "windows" in cls.__init__.__code__.co_varnames else cls()
        if name == "vpin":
            m = cls(buckets_per_day=50_000, window=5)
        if name in FITTED:
            m.fit([train_day])
        out[name] = m
    return out


def same_frames(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    if a.columns != b.columns or a.height != b.height:
        return False
    for c in a.columns:
        x = a[c].cast(pl.Float64).to_numpy()
        y = b[c].cast(pl.Float64).to_numpy()
        if not np.array_equal(x, y, equal_nan=True):
            return False
    return True


cut_settings = settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


cuts = st.lists(st.floats(0.0, 1.0), min_size=1, max_size=8)


@cut_settings
@given(seed=st.integers(0, 3), at=cuts)
def test_no_feature_changes_when_the_future_is_cut_off(fitted, seed, at):
    day = scenario_day(seed)
    whole = {name: m.transform(day) for name, m in fitted.items()}
    start, end = day.runs[0].first_recv_ts, day.runs[0].last_recv_ts
    for a in at:
        cut = start + int((end - start) * a)
        early = day.truncated(cut)
        for name, m in fitted.items():
            full = whole[name].filter(pl.col("ts") <= cut)
            assert same_frames(full, m.transform(early)), f"{name} changed before {cut}"


@cut_settings
@given(seed=st.integers(0, 3), at=cuts)
def test_no_bar_changes_when_the_future_is_cut_off(seed, at):
    day = scenario_day(seed)
    kinds = (("tick", 7), ("volume", 200_000), ("dollar", 2 * 10**12))
    whole = {k: bars.activity_bars(day, k, th) for k, th in kinds}
    timed = bars.time_bars(day, "1s")
    start, end = day.runs[0].first_recv_ts, day.runs[0].last_recv_ts
    for a in at:
        cut = start + int((end - start) * a)
        early = day.truncated(cut)
        assert same_frames(timed.filter(pl.col("end") <= cut + 1), bars.time_bars(early, "1s"))
        for kind, th in kinds:
            full = whole[kind].filter(pl.col("end") <= cut + 1)
            assert same_frames(full, bars.activity_bars(early, kind, th)), kind


def test_the_check_catches_a_method_that_looks_ahead(fitted):
    """The test above is only worth something if it fails on a method that peeks."""

    class Peeking(METHODS["trades"]):
        def transform(self, day):
            out = super().transform(day)
            # the next row's value: information from the future
            return out.with_columns(pl.col("volume_1s").shift(-1))

    day = scenario_day(0)
    cut = day.runs[0].first_recv_ts + 15 * S
    m = Peeking(windows=("1s",))
    full = m.transform(day).filter(pl.col("ts") <= cut)
    assert not same_frames(full, m.transform(day.truncated(cut)))


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 3), at=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=4))
@example(seed=0, at=[0.9999999999999999])  # one nanosecond before the stop: not a hole yet
def test_no_fill_changes_when_the_future_is_cut_off(seed, at):
    """An order's outcome, once final, and a markout, once its horizon has passed, are the
    same whether or not the data goes on."""
    from marketlab.execution import Market, Mode, frames, markouts, probe_orders, simulate

    day = scenario_day(seed)
    m = Market.of(day)
    lat = 200 * 1_000_000
    probes = probe_orders(m, "2s", 30_000, "3s") + probe_orders(m, "2s", 30_000, None, False)
    full = {
        mode: {
            r.order.id + 10_000 * (not r.order.passive): r for r in simulate(m, probes, lat, mode)
        }
        for mode in Mode
    }
    _, fills = frames(list(full[Mode.PESSIMISTIC].values()), "p")
    marks = markouts(m, fills)
    start, end = day.runs[0].first_recv_ts, day.runs[0].last_recv_ts
    for a in at:
        cut = start + int((end - start) * a)
        mc = Market.of(day.truncated(cut))
        early = probe_orders(mc, "2s", 30_000, "3s") + probe_orders(mc, "2s", 30_000, None, False)
        # the decisions themselves: made from what was known when they were made
        assert [o for o in probes if o.decided <= cut] == [o for o in early if o.decided <= cut]
        for mode in Mode:
            part = {
                r.order.id + 10_000 * (not r.order.passive): r
                for r in simulate(mc, early, lat, mode)
            }
            for key, r in full[mode].items():
                if r.final <= cut and r.outcome != "open":
                    p = part[key]
                    assert (p.order, p.outcome, p.final, p.fills, p.queue_at_arrival) == (
                        r.order,
                        r.outcome,
                        r.final,
                        r.fills,
                        r.queue_at_arrival,
                    ), (mode, key, cut)
        _, f = frames([part_r for part_r in simulate(mc, early, lat, Mode.PESSIMISTIC)], "p")
        cut_marks = markouts(mc, f)
        for h, n in (("100ms", 100 * 1_000_000), ("1s", S), ("10s", 10 * S)):
            done = marks.filter(pl.col("ts") + n <= cut).select(
                "id", "ts", "passive", f"markout_{h}"
            )
            again = cut_marks.select("id", "ts", "passive", f"markout_{h}").join(
                done.select("id", "ts", "passive"), on=["id", "ts", "passive"], how="semi"
            )
            assert same_frames(
                done.sort("id", "passive", "ts"), again.sort("id", "passive", "ts")
            ), h


def test_each_probe_decision_uses_only_what_was_known_at_it():
    from marketlab.execution import Market, probe_orders

    for seed in range(4):
        day = scenario_day(seed)
        probes = probe_orders(Market.of(day), "1s", 100, "1s")
        times = sorted({o.decided for o in probes})
        assert len(times) > 20
        for t in times:
            known = probe_orders(Market.of(day.truncated(t)), "1s", 100, "1s")
            assert [o for o in known if o.decided == t] == [o for o in probes if o.decided == t]
