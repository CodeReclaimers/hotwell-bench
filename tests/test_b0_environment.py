"""B0 gates: the installed environment is the pinned one, and it reproduces.

Three independent checks:
1. Provenance: hotwell is the non-editable v0.4.0 tag commit; numpy is pinned.
2. Upstream reproduction: the six golden runs pinned in hotwell's own
   tests/test_golden_hashes.py (at v0.4.0) reproduce from the installed package.
   These pins were not produced by this repo, so agreement is an independent check.
3. Noise path: the goldens run with sigma = 0, where draws never reach a value,
   so they cannot see a numpy change in normal(). hotwell's raw/normal draw pins
   (tests/test_rng.py at v0.4.0) and a short noisy run cover that path.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from hotwell import data
from hotwell.plant import load_config
from hotwell.rng import StreamBank
from hotwell.scenario import Scenario, ScenarioEvent, load_scenario
from hotwell.simulator import Simulator

from hotwell_bench.provenance import (
    PINNED_HOTWELL_COMMIT,
    PINNED_NUMPY,
    check_environment,
    environment_record,
)

FIDELITIES = {"two_state_1e": ("two_state", False), "astrom_bell_3e": ("astrom_bell", True)}

# Copied verbatim from hotwell tests/test_golden_hashes.py at v0.4.0 (5f57636).
# (fidelity, scenario) -> (steps, state_history_hash, observation_hash)
UPSTREAM_GOLDEN = {
    ("two_state_1e", "cp1_trip_unrecovered"): (
        8550,
        "2f2c4a18fc03617cb06f40ec25d18cc063c88879e8bacae942a68fd858c9b66f",
        "f4c4cccdb0288ad6c06a1e0d6cedda598c15cf89c414d98308ddc80ac169bef7"),
    ("two_state_1e", "cp1_trip_recovered"): (
        18000,
        "dcd1b3a5290214940b80fa1ad34aea220d996637d35c4b1b8803b0ff3911439a",
        "a6c98287cc8056b4c2f5aa906e9548e4fab33efdd8fbb996f678d08f2d097921"),
    ("two_state_1e", "load_step"): (
        18000,
        "ac2199cc8b0b51a603464933c78cff344a18a6ff042c2949161f9d40ff5352d4",
        "028653f2de48e94ca50f6500ea6706f5096a78bcbf415648d8855b0e40aac12a"),
    ("astrom_bell_3e", "cp1_trip_unrecovered"): (
        10468,
        "4079cb8c94c05e9e39588a6292007db68918c4d0e2013d76703bf55ddc0ec5d9",
        "96dea09f42f7ccaa56b593257c64be7874c7e86e6ce497da5de4d3e11315f3fb"),
    ("astrom_bell_3e", "cp1_trip_recovered"): (
        18000,
        "fc970dbee00475701add99d62730cce6f714918ee8ed5a5bb4e593a6abb43d31",
        "9a300b5425692c51c3afe597bb9310010b79ffd29bef52007e1db590035821eb"),
    ("astrom_bell_3e", "load_step"): (
        18000,
        "35aae16396f4cf023be94f07029346b35e87466cc977900833b3273b317b31df",
        "d6386717605138b006af1e8780707828de4fd257b82a113972287271d6916dcb"),
}


def _config(fidelity: str) -> dict:
    model, three_element = FIDELITIES[fidelity]
    cfg = load_config()
    cfg["drum"]["model"] = model
    cfg["control"]["drum_level"]["three_element"] = three_element
    return cfg


def test_environment_matches_pins():
    rec = environment_record()
    print(f"\nenvironment: {rec}")
    assert check_environment() == rec
    assert rec["hotwell"]["version"] == "0.4.0"
    assert rec["hotwell"]["requested_revision"] == "v0.4.0"
    assert rec["hotwell"]["commit_id"] == PINNED_HOTWELL_COMMIT
    assert rec["numpy"] == PINNED_NUMPY == np.__version__


def test_check_environment_rejects_a_drifted_environment(monkeypatch):
    """The gate must fail closed: a wrong commit, an editable install, or another
    numpy each raise."""
    import hotwell_bench.provenance as prov
    good = prov.environment_record()
    for mutate in (lambda r: r["hotwell"].update(commit_id="0" * 40),
                   lambda r: r["hotwell"].update(editable=True),
                   lambda r: r.update(numpy="2.6.0")):
        bad = copy.deepcopy(good)
        mutate(bad)
        monkeypatch.setattr(prov, "environment_record", lambda b=bad: b)
        with pytest.raises(RuntimeError) as exc:
            prov.check_environment()
        print(f"\nrejected: {exc.value}")


@pytest.mark.parametrize(("fidelity", "scenario"), sorted(UPSTREAM_GOLDEN))
def test_upstream_golden_runs_reproduce(fidelity: str, scenario: str):
    steps, state_h, obs_h = UPSTREAM_GOLDEN[(fidelity, scenario)]
    sc = load_scenario(data.builtin_scenarios()[scenario])
    sim = Simulator(_config(fidelity), scenario=sc)
    sim.initialize_steady_state(sc.load_mw)
    sim.run()
    n = int(round(sim.t / sim.dt))
    print(f"\n{fidelity:15s} {scenario:22s} steps {n} (upstream {steps})\n"
          f"  state {sim.state_history_hash}\n  obs   {sim.observation_hash}")
    assert n == steps
    assert sim.state_history_hash == state_h
    assert sim.observation_hash == obs_h


def test_upstream_numpy_draw_pins():
    """Copied from hotwell tests/test_rng.py::test_pinned_raw_draws_numpy_version_gate
    at v0.4.0 (pinned there under numpy 2.5.1)."""
    raw = [int(x) for x in
           StreamBank(42).stream("sensor.drum_level").bit_generator.random_raw(3)]
    nrm = StreamBank(42).stream("sensor.drum_level").normal(size=3).tolist()
    print(f"\nnumpy {np.__version__}: raw {raw}\nnormal {nrm}")
    assert raw == [2491596885235603570, 16407104577408639165, 14955272865311411715]
    assert nrm == [0.11610006701218922, 0.23067914034706685, 0.1631065687488972]


# A short noisy AB + 3E run with a hidden fault: exercises sensor noise, load noise
# and a plant fault, the paths an item uses and the goldens do not.
SMOKE_SENSOR_NOISE = {
    "drum_level": {"sigma": 3.0e-3, "tau": 1.0},
    "feed_flow": {"sigma": 0.3, "tau": 0.5},
    "frequency": {"sigma": 0.005, "tau": 0.0},
}
SMOKE_LOAD_NOISE = {"nonvital": {"sigma_w": 3.0e5, "tau": 120.0}}
# Pinned at B0 (2026-11-07) under hotwell v0.4.0 / numpy 2.5.1 by this test's own
# first run. Self-pinned, unlike the upstream goldens: it detects change, it does
# not validate the result.
SMOKE_PIN = (
    2400,
    "a2db40a76850f87071693c2eff85e303702a2ee593a03ac4ab4a2576cb02756f",
    "5cb581daa21cfef6312b6be4bee9f0191721cce988c50969a7c9c11957b5088c",
)


def _smoke_run(noisy: bool = True) -> Simulator:
    cfg = _config("astrom_bell_3e")
    if noisy:
        cfg["sensors"]["noise"] = SMOKE_SENSOR_NOISE
        cfg["loads"]["noise"] = SMOKE_LOAD_NOISE
    sc = Scenario("b0_smoke", 7, 60.0, "islanded",
                  (ScenarioEvent(60.0, "pump_degrade", {"target": "fp1", "head_factor": 0.97}),),
                  120.0)
    sim = Simulator(cfg, scenario=sc)
    sim.initialize_steady_state(60.0)
    sim.run()
    return sim


def test_noisy_faulted_run_is_deterministic_and_pinned():
    a, b = _smoke_run(), _smoke_run()
    n = int(round(a.t / a.dt))
    print(f"\nsmoke: steps {n}\n  state {a.state_history_hash}\n  obs   {a.observation_hash}"
          f"\n  drum_level {a.last_obs.values['drum_level']:+.4f} m, "
          f"fw_valve_pos {a.last_obs.values['fw_valve_pos']:.4f}")
    assert (a.state_history_hash, a.observation_hash) == (b.state_history_hash,
                                                          b.observation_hash)
    assert (n, a.state_history_hash, a.observation_hash) == SMOKE_PIN
    # the noise config must actually act: the noise-free twin differs in both hashes
    q = _smoke_run(noisy=False)
    print(f"  noise-free twin: state {q.state_history_hash[:16]}… obs {q.observation_hash[:16]}…")
    assert q.state_history_hash != a.state_history_hash
    assert q.observation_hash != a.observation_hash
