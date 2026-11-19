"""The experiment runner: the hypothesis file and its commit, the verdicts on data with a
planted signal and on data without one, the registry and the held-out days."""

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from marketlab import experiment as ex
from marketlab import synth
from marketlab.io.days import DayInfo

SPEC = """
id: "007"
name: planted
hypothesis: A planted signal is found.
data: {symbol: BTCUSDT, min_hours: 0.01, min_days: 6, holdout: [2026-01-30]}
sampling: 1s
feature: {method: mlofi, params: {levels: 3, window: 1s}}
variants:
  - {levels: 1, window: 1s}
  - {levels: 3, window: 1s}
  - {levels: 3, window: 5s}
target: {kind: return, horizon: 1s}
protocol: {folds: 4, cpcv_test_groups: 2, embargo: 5s, surrogates: 200, seed: 3}
success: {min_ic: 0.0, alpha: 0.05, surrogate_percentile: 95, max_pbo: 0.5, min_positive_days: 0.6}
"""

PROV = ex.Provenance("a" * 40, "b" * 40, False)


def spec_file(tmp_path, text=SPEC, name="007-planted.yaml"):
    p = tmp_path / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def fake_days(n, signal, seed=0, rows=600):
    """Inputs as `day_inputs` makes them, with the target a noisy function of the OFI
    (`signal` > 0) or independent of it (`signal` = 0)."""
    rng = np.random.default_rng(seed)
    w_true = np.array([1.0, 0.5, 0.25])
    out = {}
    for d in range(n):
        date = f"2026-01-{d + 1:02d}"
        ts = 1_767_225_600_000_000_000 + d * 86_400_000_000_000 + np.arange(rows) * 10**9
        cols = {"ts": ts, "segment": np.zeros(rows, dtype=np.int32)}
        for w in ("1s", "5s"):
            x = rng.standard_normal((rows, 3))
            for m in range(3):
                cols[f"ofi_l{m + 1}_{w}"] = x[:, m]
            cols[f"dmid_{w}"] = x @ w_true + 0.5 * rng.standard_normal(rows)
            if w == "1s":
                flow = x @ w_true
        cols["target"] = signal * flow + rng.standard_normal(rows)
        cols["rv_60s"] = rng.uniform(0.5, 2.0, rows)
        out[date] = pl.DataFrame(cols)
    return out


@pytest.fixture
def planted(monkeypatch):
    def use(frames, short=()):
        """`short`: dates with less usable time than the hypothesis asks."""
        infos = [
            DayInfo(Path("/nowhere") / d, d, 0.001 if d in short else 24.0) for d in sorted(frames)
        ]

        def survey(root, symbol, exclude=()):
            return [i for i in infos if i.date not in exclude]

        monkeypatch.setattr(ex, "survey", survey)
        monkeypatch.setattr(ex, "load_inputs", lambda info, spec, cache: frames[info.date])

    return use


def test_a_planted_signal_passes_every_step(tmp_path, planted):
    planted(fake_days(10, signal=0.3))
    spec = ex.load_spec(spec_file(tmp_path))
    assert spec.primary in spec.variants
    r = ex.run(spec, "unused", tmp_path / "results", prov=PROV)
    assert r.verdict == "passed", r.reasons
    assert r.kfold.mean > 0.2 and r.kfold.positive_share == 1.0
    assert len(r.cpcv_paths) == 9  # C(10 - 1, 2 - 1)
    assert all(s["real"] > s["threshold"] for s in r.surrogate.values())
    assert r.pbo is not None and r.pbo.pbo <= 0.5
    j = json.loads((tmp_path / "results" / "007-planted.json").read_text())
    assert j["verdict"] == "passed" and j["spec_commit"] == "a" * 40
    assert [d["date"] for d in j["data"]["days"]] == [f"2026-01-{d:02d}" for d in range(1, 11)]
    assert set(j["stability"]) == {"positive_days", "time_of_day", "volatility_tercile"}


def test_no_signal_fails_on_the_surrogates(tmp_path, planted):
    planted(fake_days(10, signal=0.0, seed=1))
    r = ex.run(ex.load_spec(spec_file(tmp_path)), "unused", tmp_path / "results", prov=PROV)
    assert r.verdict == "failed"
    assert abs(r.kfold.mean) < 0.05
    assert any("surrogates" in reason for reason in r.reasons)


def test_too_few_days_is_insufficient_data(tmp_path, planted):
    planted(fake_days(4, signal=0.3))
    r = ex.run(ex.load_spec(spec_file(tmp_path)), "unused", tmp_path / "results", prov=PROV)
    assert r.verdict == "insufficient data"
    assert json.loads((tmp_path / "results" / "007-planted.json").read_text())["verdict"] == (
        "insufficient data"
    )


def test_days_short_of_usable_hours_are_named(tmp_path, planted, capsys):
    planted(fake_days(10, signal=0.3), short={"2026-01-03", "2026-01-07"})
    r = ex.run(ex.load_spec(spec_file(tmp_path)), "unused", tmp_path / "results", prov=PROV)
    assert [d.date for d in r.days] == [f"2026-01-{d:02d}" for d in (1, 2, 4, 5, 6, 8, 9, 10)], (
        "the short days take no part"
    )
    assert [d.date for d in r.skipped] == ["2026-01-03", "2026-01-07"]
    j = json.loads((tmp_path / "results" / "007-planted.json").read_text())
    assert [d["date"] for d in j["data"]["skipped"]] == ["2026-01-03", "2026-01-07"]
    from marketlab.cli import print_report

    print_report(r)
    out = capsys.readouterr().out
    assert "skipped     under 0.01 usable hours: 2026-01-03 0.0 h, 2026-01-07 0.0 h" in out


def test_the_registry_adjusts_every_hypothesis_together(tmp_path, planted):
    planted(fake_days(10, signal=0.3))
    results = tmp_path / "results"
    ex.run(ex.load_spec(spec_file(tmp_path)), "unused", results, prov=PROV)
    for i, p in enumerate([0.04, 0.5]):
        (results / f"10{i}-old.json").write_text(
            json.dumps({"id": f"10{i}", "name": "old", "verdict": "failed", "p_raw": p})
        )
    table = ex.registry_table(results)
    assert table.height == 3
    old = table.filter(pl.col("id") == "100")
    assert old["p_bh"][0] == pytest.approx(min(0.04 * 3 / 2, 1.0))


def test_a_hypothesis_written_but_not_run_counts_at_p_one(tmp_path):
    # two hypotheses written before either runs: whichever runs first is adjusted for the
    # other at p = 1, so running first never helps; the list shows the unrun one too
    exp, results = tmp_path / "experiments", tmp_path / "results"
    first = spec_file(tmp_path)
    spec_file(tmp_path, SPEC.replace('id: "007"', 'id: "008"'), name="008-planted.yaml")
    spec = ex.load_spec(first)
    assert ex._bh_with_registry(results, spec, 0.03) == pytest.approx(0.03 * 2)
    results.mkdir()
    (results / "008-planted.json").write_text(
        json.dumps({"id": "008", "name": "planted", "verdict": "failed", "p_raw": 0.5})
    )
    assert ex._bh_with_registry(results, spec, 0.03) == pytest.approx(0.03 * 2 / 1)
    (results / "008-planted.json").write_text(
        json.dumps({"id": "008", "name": "planted", "verdict": "passed", "p_raw": 0.01})
    )
    assert ex._bh_with_registry(results, spec, 0.03) == pytest.approx(0.03)
    (results / "008-planted.json").unlink()
    table = ex.registry_table(results, exp)
    assert table.sort("id")["verdict"].to_list() == ["not run", "not run"]
    assert ex.written(exp) == {"007-planted": "007-planted.yaml", "008-planted": "008-planted.yaml"}


def test_surrogates_do_not_depend_on_the_number_of_workers(tmp_path, planted):
    planted(fake_days(8, signal=0.1, seed=4))
    spec = ex.load_spec(spec_file(tmp_path))
    one = ex.run(spec, "unused", tmp_path / "r1", prov=PROV, workers=1)
    two = ex.run(spec, "unused", tmp_path / "r2", prov=PROV, workers=2)
    assert one.surrogate == two.surrogate


def test_the_held_out_days_open_once(tmp_path, planted, monkeypatch):
    planted(fake_days(10, signal=0.3) | {"2026-01-30": fake_days(1, 0.3, seed=9)["2026-01-01"]})
    spec = ex.load_spec(spec_file(tmp_path))
    results = tmp_path / "results"
    with pytest.raises(ex.ExperimentError, match="first"):
        ex.final(spec, "unused", results, prov=PROV)
    r = ex.run(spec, "unused", results, prov=PROV)
    assert "2026-01-30" not in [d.date for d in r.days]
    loaded = []
    load = ex.load_inputs
    monkeypatch.setattr(ex, "load_inputs", lambda i, s, c: loaded.append(i.date) or load(i, s, c))
    rec = ex.final(spec, "unused", results, prov=PROV)
    assert rec["days"] == ["2026-01-30"]
    assert loaded == [d.date for d in r.days] + ["2026-01-30"], "fit on the others, scored on it"
    assert rec["summary"]["mean"] > 0.1
    with pytest.raises(ex.ExperimentError, match="opened already"):
        ex.final(spec, "unused", results, prov=PROV)


def test_a_spec_is_checked_before_anything_runs(tmp_path):
    with pytest.raises(ex.ExperimentError, match="unknown keys"):
        ex.load_spec(spec_file(tmp_path, SPEC + "\nextra: 1\n"))
    with pytest.raises(ex.ExperimentError, match="200 surrogates"):
        ex.load_spec(spec_file(tmp_path, SPEC.replace("surrogates: 200", "surrogates: 20")))
    with pytest.raises(ValueError, match="duration"):
        ex.load_spec(spec_file(tmp_path, SPEC.replace("horizon: 1s", "horizon: soon")))
    with pytest.raises(ex.ExperimentError, match="feature"):
        ex.load_spec(spec_file(tmp_path, SPEC.replace("method: mlofi", "method: vpin")))


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_the_hypothesis_must_be_committed_and_unchanged(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    p = spec_file(tmp_path)
    with pytest.raises(ex.ExperimentError, match="not committed"):
        ex.provenance(p)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "hypothesis")
    prov = ex.provenance(p)
    assert len(prov.spec_commit) == 40 and prov.spec_commit == prov.head and not prov.dirty
    (tmp_path / "code.py").write_text("x = 1\n")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "commit", "-q", "-m", "code")
    (tmp_path / "code.py").write_text("x = 2\n")
    later = ex.provenance(p)
    assert later.spec_commit == prov.spec_commit != later.head
    assert later.dirty  # code changed since the last commit: recorded
    p.write_text(SPEC.replace("min_ic: 0.0", "min_ic: -1.0"))
    with pytest.raises(ex.ExperimentError, match="changed since its commit"):
        ex.provenance(p)


def test_end_to_end_on_synthetic_days(tmp_path):
    data = tmp_path / "data"
    for k in range(7):
        start = synth.DAY0 + k * 86_400_000_000_000 + 36_000_000_000_000
        synth.write(synth.Scenario(seed=40 + k, seconds=120, start=start), data)
    git(tmp_path, "init", "-q", "-b", "main")
    text = SPEC.replace("holdout: [2026-01-30]", "holdout: [2026-11-27]")
    p = spec_file(tmp_path, text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "hypothesis")
    spec = ex.load_spec(p)
    r = ex.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert [d.date for d in r.days] == [f"2026-11-{d}" for d in range(21, 27)]
    assert r.verdict in ex.VERDICTS
    cached = list((tmp_path / ".cache").rglob("*.parquet"))
    assert len(cached) == 6
    again = ex.run(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert again.kfold == r.kfold  # from the cache, the same numbers
    rec = ex.final(spec, data, tmp_path / "results", cache=tmp_path / ".cache")
    assert rec["days"] == ["2026-11-27"]
