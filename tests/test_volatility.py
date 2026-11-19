"""Hypotheses of the `volatility` kind: verdicts on inputs where a feature adds to HAR and
where it does not, too little data, the committed files, and a run on synthetic days."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from marketlab import experiment as ex
from marketlab import synth
from marketlab import volatility as vol
from marketlab.io.days import DayInfo

SEC = 1_000_000_000
PROV = ex.Provenance("a" * 40, "b" * 40, False)

SPEC = """
id: "019"
name: planted-vol
kind: volatility
hypothesis: A planted feature is found.
data: {symbol: BTCUSDT, min_hours: 0.01, min_days: 8, holdout: [2026-01-30]}
sampling: 10s
target: {horizon: 10s, floor_bp: 0.1}
feature: {method: hawkes, params: {window: 10m, step: 5m}}
variants:
  - {window: 5m, step: 5m}
  - {window: 10m, step: 5m}
  - {window: 30m, step: 5m}
protocol: {folds: 4, embargo: 300s, surrogates: 200, min_shift: 10m, seed: 5}
success:
  {min_gain: 0.0, alpha: 0.05, surrogate_percentile: 95, max_pbo: 0.5, min_positive_days: 0.6}
"""


def fake_day(seed: int, planted: bool, rows: int = 3000) -> pl.DataFrame:
    """HAR columns that wander, a target that follows them, and the hawkes variants'
    columns: the primary's log intensity adds to the target when planted."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.05, rows))
    cols = {
        "ts": 1_767_225_600 * SEC + seed * 86_400 * SEC + np.arange(rows) * 10 * SEC,
        "segment": np.zeros(rows, dtype=np.int32),
    }
    for w in vol.HAR:
        cols[f"har_{w}"] = base + rng.normal(0, 0.3, rows)
    cols["persistence"] = cols["har_10s"]
    signal = rng.normal(0, 1, rows)
    cols["target"] = base + (0.6 * signal if planted else 0) + rng.normal(0, 0.5, rows)
    for window in ("5m", "10m", "30m"):
        key = f"hawkes(window={window},step=5m)"
        x = signal if window == "10m" else rng.normal(0, 1, rows)
        cols[f"{key}:loglam"] = x
        cols[f"{key}:dloglam"] = rng.normal(0, 1, rows)
        cols[f"{key}:branching"] = rng.uniform(0, 1, rows)
    return pl.DataFrame(cols)


@pytest.fixture
def planted(monkeypatch):
    def use(effect: bool, n_days: int = 10):
        days = {f"2026-01-{d + 1:02d}": fake_day(d, effect) for d in range(n_days)}
        days["2026-01-30"] = fake_day(29, effect)
        infos = [DayInfo(Path("/nowhere") / d, d, 24.0) for d in sorted(days)]
        monkeypatch.setattr(
            vol,
            "survey",
            lambda root, symbol, exclude=(): [i for i in infos if i.date not in exclude],
        )
        monkeypatch.setattr(vol, "bucket_units", lambda infos, spec: {})
        monkeypatch.setattr(
            vol, "load_inputs", lambda info, spec, cache, units: (days[info.date], {})
        )
        monkeypatch.setattr(
            vol,
            "load_all",
            lambda infos, spec, cache, units, workers: [(days[i.date], {}) for i in infos],
        )

    return use


def spec_file(tmp_path, text=SPEC):
    p = tmp_path / "experiments" / "019-planted-vol.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_a_planted_feature_passes_and_noise_fails(tmp_path, planted, capsys):
    planted(True)
    spec = vol.load_spec(spec_file(tmp_path))
    r = vol.run(spec, "unused", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    assert r.mean_gain > 0.1 and r.positive_share == 1.0 and r.mean_gain_persistence > 0
    assert r.surrogate["shift"]["real"] > r.surrogate["shift"]["threshold"]
    assert r.pbo is not None and r.pbo.pbo <= 0.5
    assert "2026-01-30" not in r.by_day
    j = json.loads((tmp_path / "results" / "019-planted-vol.json").read_text())
    assert j["kind"] == "volatility" and j["p_raw"] < 0.05
    from marketlab.cli import print_volatility_report

    print_volatility_report(r)
    assert "over HAR" in capsys.readouterr().out
    loaded = []
    load = vol.load_inputs

    def spy(info, spec, cache, units):
        loaded.append(info.date)
        return load(info, spec, cache, units)

    vol.load_inputs = spy
    try:
        rec = vol.final(spec, "unused", tmp_path / "results", prov=PROV)
    finally:
        vol.load_inputs = load
    assert rec["days"] == ["2026-01-30"] and rec["mean_gain"] > 0.1
    assert loaded == [d for d in sorted(r.by_day)] + ["2026-01-30"], "fit on the others only"
    with pytest.raises(ex.ExperimentError, match="opened already"):
        vol.final(spec, "unused", tmp_path / "results", prov=PROV)

    planted(False)
    r = vol.run(spec, "unused", tmp_path / "results2", prov=PROV)
    assert r.verdict == "failed" and r.mean_gain < 0.01
    assert any("shifted features" in x for x in r.reasons), r.reasons


def test_too_little_data_and_wrong_kinds(tmp_path, planted):
    planted(True, n_days=3)
    spec = vol.load_spec(spec_file(tmp_path))
    assert vol.run(spec, "unused", tmp_path / "r", prov=PROV).verdict == "insufficient data"
    with pytest.raises(ex.ExperimentError, match="not a forecast"):
        ex.load_spec(spec_file(tmp_path))
    with pytest.raises(ex.ExperimentError, match="not a `volatility`"):
        vol.load_spec(spec_file(tmp_path, SPEC.replace("kind: volatility", "kind: maker")))
    bad = SPEC.replace("{window: 5m, step: 5m}", "{window: 5m}")
    with pytest.raises(ex.ExperimentError, match="needs"):
        vol.load_spec(spec_file(tmp_path, bad))


def test_the_committed_hypotheses_load():
    root = Path(__file__).resolve().parents[1] / "experiments"
    h, v = vol.load_spec(root / "003-hawkes-vol.yaml"), vol.load_spec(root / "004-vpin-vol.yaml")
    assert h.primary.key == "hawkes(window=10m,step=5m)" and len(h.variants) == 3
    assert v.primary.key == "vpin(buckets_per_day=1000,window=50)" and len(v.variants) == 3
    holdout = ex.load_spec(root / "001-mlofi.yaml").holdout
    assert h.holdout == v.holdout == holdout


def test_end_to_end_on_synthetic_days(tmp_path):
    from test_experiment import git

    data = tmp_path / "data"
    for k in range(5):
        start = synth.DAY0 + k * 86_400_000_000_000 + 36_000_000_000_000
        synth.write(synth.Scenario(seed=70 + k, seconds=2400, start=start), data)
    git(tmp_path, "init", "-q", "-b", "main")
    text = (
        SPEC.replace("min_days: 8", "min_days: 4")
        .replace("[2026-01-30]", "[2026-11-25]")
        .replace("window: 30m", "window: 15m")
        .replace("folds: 4", "folds: 2")
    )
    p = spec_file(tmp_path, text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "hypothesis")
    spec = vol.load_spec(p)
    r = vol.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache", workers=2)
    assert list(r.by_day) == [f"2026-11-{d}" for d in range(21, 25)]
    assert r.verdict in ex.VERDICTS and np.isfinite(r.mean_gain)
    d = r.diagnostics["2026-11-21"][spec.primary.key]
    assert d["fits"] > 0 and 0 <= d["ks_pass_share"] <= 1
    cached = list((tmp_path / ".cache").rglob("vol-*.parquet"))
    assert len(cached) == 4
    frame = pl.read_parquet(cached[0])
    change = frame[f"{spec.primary.key}:dloglam"].to_numpy().astype(float)
    assert np.nanstd(change) > 0, "the intensity's change over a minute is not all zero"
    again = vol.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert again.mean_gain == r.mean_gain, "from the cache, the same numbers"
    rec = vol.final(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert rec["days"] == ["2026-11-25"]


def test_vpin_buckets_are_the_development_days_volume_per_day(tmp_path):
    data = tmp_path / "data"
    infos = []
    for k in range(2):
        start = synth.DAY0 + k * 86_400_000_000_000 + 36_000_000_000_000
        path = synth.write(synth.Scenario(seed=80 + k, seconds=300, start=start), data)
        infos.append(DayInfo(path, path.name, 0.08))
    spec = vol.load_spec(Path(__file__).resolve().parents[1] / "experiments" / "004-vpin-vol.yaml")
    units = vol.bucket_units(infos, spec)
    from marketlab.features.trades import _trades
    from marketlab.io.day import Day

    days = [Day.open(i.path) for i in infos]
    volume = sum(float(_trades(d)["qty"].sum()) * float(d.qty_step) for d in days)
    seconds = sum(sum(e - s for s, e in d.intervals) / 1e9 for d in days)
    for v in spec.variants:
        want = volume / seconds * 86_400 / int(v.get("buckets_per_day"))
        assert units[v.key] == pytest.approx(want)
