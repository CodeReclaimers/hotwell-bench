"""B2 gates: runner capture, twin identity, events and shift log, cache, secrecy,
parallel = serial, and (slow) reproduction of hotwell's own quiet-day hashes."""
from __future__ import annotations

import numpy as np
import pytest

from hotwell_bench.build import build
from hotwell_bench.items import SCHEMA, DayProfile, Fault, ItemRecord, OperatorPlan
from hotwell_bench.run import COLUMNS, run_built, run_cached, run_items
from hotwell_bench.secrecy import REPO_ROOT, check_output_path

FLAT = DayProfile(True, 48.0, 3600.0, 1800.0, 14400.0, 1800.0)


def _item(index=9000, fault=None, cp_swap=None, cadence=100, sigma=0.3, seed=7) -> ItemRecord:
    return ItemRecord(SCHEMA, "dev", index, 0, seed, FLAT,
                      OperatorPlan(cadence, cp_swap, None, 5.0, sigma), fault)


FP1_DEGRADE = Fault(category="pump_performance_loss", component="FP-1",
                    mechanism="pump_degrade", target="fp1", t_inj=60, shape="step",
                    params={"head_factor": 0.97})
DRUM_LEVEL_BIAS = Fault(category="instrument_fault", component="drum_level",
                        mechanism="sensor_fault", target="drum_level", t_inj=60,
                        shape="step", params={"kind": "bias", "bias": 0.2},
                        sensor_kind="bias", magnitude=0.2)


def test_capture_is_one_row_per_second_in_spec_order():
    r = run_built(build(_item()), "faulted", until=120)
    print(f"\nrows {r.values.shape}; t {r.t[:3]}…{r.t[-1]}; wall {r.summary['wall_s']:.2f} s")
    assert r.values.shape == (121, len(COLUMNS))
    assert np.array_equal(r.t, np.arange(121))
    assert r.summary["steps"] == 2400
    assert not r.summary["rejected"]


def test_null_item_twin_is_identical():
    b = build(_item())
    a, t = run_built(b, "faulted", until=180), run_built(b, "twin", until=180)
    assert np.array_equal(a.values, t.values)
    assert a.events == t.events and a.shift_log == t.shift_log
    assert a.summary["state_hash"] == t.summary["state_hash"]


def test_twins_agree_before_onset_and_diverge_after():
    b = build(_item(fault=FP1_DEGRADE))
    a, t = run_built(b, "faulted", until=180), run_built(b, "twin", until=180)
    diff = np.any(a.values != t.values, axis=1)
    first = int(np.argmax(diff))
    cols = [COLUMNS[j] for j in np.nonzero(a.values[first] != t.values[first])[0]]
    print(f"\nfirst differing second {first} (onset 60): {cols}; "
          f"fw_valve_pos at 180 s faulted {a.column('fw_valve_pos')[-1]:.4f} "
          f"twin {t.column('fw_valve_pos')[-1]:.4f}")
    assert not diff[:61].any()          # row 60 is the observation at t = 60 s, before the event
    assert diff[61:].any()
    assert a.column("fw_valve_pos")[-1] > t.column("fw_valve_pos")[-1]   # FRV opens (M5)


def test_events_and_shift_log_capture_operations_and_alarms():
    b = build(_item(fault=DRUM_LEVEL_BIAS, cp_swap=30))
    r = run_built(b, "faulted", until=400)
    for e in r.events:
        print(f"  {e['t']:8.2f}  {e['kind']:10s} {e['id']:20s} {e['value']}")
    for e in r.shift_log:
        if e["proc_id"] != "NOP-03":
            print(f"  log {e['t']:8.2f} {e['proc_id']} {e['step_id']} {e['outcome']}")
    ind = {(e["id"], e["value"]) for e in r.events if e["kind"] == "indication"}
    assert ("cp2_running", True) in ind and ("cp1_running", False) in ind
    alarms = [(e["id"], e["value"]) for e in r.events if e["kind"] == "alarm"]
    assert ("DRUM_LVL_HI", "active") in alarms
    nop01 = [e["outcome"] for e in r.shift_log if e["proc_id"] == "NOP-01"]
    assert nop01[0] == "started" and nop01[-1] == "completed"
    assert all(e["t"] >= 30 for e in r.shift_log if e["proc_id"] == "NOP-01")


def test_cache_hit_returns_the_stored_run(tmp_path):
    it = _item(fault=FP1_DEGRADE)
    a = run_cached(it, "faulted", tmp_path, until=90)
    files = sorted(p.name for p in tmp_path.iterdir())
    b = run_cached(it, "faulted", tmp_path, until=90)
    print(f"\ncache files {files}")
    assert len(files) == 2
    assert np.array_equal(a.values, b.values) and a.events == b.events
    assert b.summary["wall_s"] == a.summary["wall_s"]       # loaded, not re-run
    run_cached(it, "faulted", tmp_path, until=91)            # different key -> new entry
    assert len(list(tmp_path.iterdir())) == 4


def test_secrecy_guard_refuses_test_split_output_inside_the_repo(tmp_path):
    with pytest.raises(PermissionError):
        check_output_path(REPO_ROOT / "cache", "test")
    with pytest.raises(PermissionError):
        run_cached(ItemRecord(**{**_item().__dict__, "split": "test"}), "faulted",
                   REPO_ROOT / "cache", until=1)
    assert check_output_path(tmp_path, "test") == tmp_path.resolve()
    assert check_output_path(REPO_ROOT / "cache", "dev") == (REPO_ROOT / "cache").resolve()


def test_parallel_runs_equal_serial_runs(tmp_path):
    items = [_item(index=9100), _item(index=9101, fault=FP1_DEGRADE)]
    par = run_items(items, tmp_path, workers=2, until=75)
    for it in items:
        for v in ("faulted", "twin"):
            ser = run_built(build(it), v, until=75).summary
            assert par[(it.item_id, v)]["state_hash"] == ser["state_hash"]
            assert par[(it.item_id, v)]["observation_hash"] == ser["observation_hash"]


@pytest.mark.slow
def test_quiet_day_mirror_reproduces_hotwells_released_hashes():
    """A record mirroring hotwell's scripts/quiet_day.py (AB + 3E, seed 42, profile
    60 -> 36 MW 03:00-04:00, back 10:00-11:00, NOP-03 every 300 s, CP swap 06:00, FP
    swap 18:00, fixed 5 s reaction) must reproduce the state and observation hashes
    that `scripts/quiet_day.py --fidelity astrom_bell_3e` printed at v0.4.0. They
    were produced by hotwell, not by this repo."""
    it = ItemRecord(SCHEMA, "dev", 9999, 0, 42,
                    DayProfile(False, 36.0, 3 * 3600.0, 3600.0, 6 * 3600.0, 3600.0),
                    OperatorPlan(300, 6 * 3600, 18 * 3600, 5.0, 0.0), None)
    s = run_built(build(it), "faulted", progress=True).summary
    print(f"\nstate {s['state_hash']}\nobs   {s['observation_hash']}")
    assert s["state_hash"] == "e05dc79db6be78ca52116121bb1c52616433f2ec4b22a0bc9ec6c4f69c201ee1"
    assert s["observation_hash"] == ("1323b1701670bd2631beff6119023be41ea47dc40c85900bf07e8ee5f"
                                     "7e05828")


def test_snapshot_continues_bit_identically_and_leaves_the_original_alone():
    from hotwell_bench.run import Drive
    b = build(_item(fault=FP1_DEGRADE, cp_swap=40))
    whole = Drive(b, "faulted")
    whole.advance(150)
    part = Drive(b, "faulted")
    part.advance(50)
    t_before, k_before = part.sim.t, part.k
    branch = part.snapshot()
    branch.advance(150)
    print(f"\nwhole {whole.sim.state_history_hash[:16]}… branch "
          f"{branch.sim.state_history_hash[:16]}…; original still at k={part.k}")
    assert branch.sim.state_history_hash == whole.sim.state_history_hash
    assert branch.sim.observation_hash == whole.sim.observation_hash
    assert np.array_equal(branch.result().values, whole.result().values)
    assert branch.result().shift_log == whole.result().shift_log
    assert (part.sim.t, part.k) == (t_before, k_before)
    assert branch.runner.streams is branch.sim.streams       # identity kept inside the copy


def test_event_injected_into_a_branch_equals_a_run_with_it_from_the_start():
    import dataclasses

    from hotwell.scenario import ScenarioEvent

    from hotwell_bench.run import Drive
    b = build(_item())
    ev = ScenarioEvent(80.0, "pump_degrade", {"target": "cp1", "head_factor": 0.98})
    sc = dataclasses.replace(b.twin, events=tuple(sorted(b.twin.events + (ev,),
                                                         key=lambda e: e.t)))
    ref = Drive(b, "twin")
    ref.sim.scenario = sc                     # swapped at t = 0
    ref.advance(150)
    base = Drive(b, "twin")
    base.advance(60)
    br = base.snapshot()
    br.sim.scenario = sc                      # swapped at t = 60, before the event
    br.advance(150)
    plain = Drive(b, "twin")
    plain.advance(150)
    assert br.sim.state_history_hash == ref.sim.state_history_hash
    assert br.sim.state_history_hash != plain.sim.state_history_hash
