"""B1 gates: taxonomy closure, sampler determinism and distributions, builder."""
from __future__ import annotations

import math
from collections import Counter

import pytest
from hotwell.faults import FACTOR_FAULTS, PLANT_FAULT_TYPES, STUCK_TARGETS
from hotwell.sensors import OBSERVATION_SPEC

from hotwell_bench import presets as P
from hotwell_bench.build import build, fault_event, nonvital_p0_at_rated, scenario_dict
from hotwell_bench.items import (
    SCHEMA,
    ItemRecord,
    answer_key,
    detection_scale,
    sample_item,
)
from hotwell_bench.taxonomy import CATEGORIES, FAULT_CATEGORIES, SENSOR_KINDS, SENSOR_MECHANISM

MASTER = 20261107          # a dev-only seed; test-split seeds never appear in this repo
N_DIST = 4000


def _allowed_targets(mechanism: str) -> set[str]:
    if mechanism == "actuator_stuck":
        return set(STUCK_TARGETS)
    targets = FACTOR_FAULTS[mechanism][1]
    return {""} if targets is None else set(targets)


# ------------------------------------------------------------------ taxonomy

def test_taxonomy_is_closed_against_hotwell():
    """Every hotwell plant-fault mechanism and every target it accepts is reachable
    from exactly the taxonomy, and nothing in the taxonomy is unknown to hotwell."""
    mechs = {m for m, _ in CATEGORIES.values()}
    assert mechs - {SENSOR_MECHANISM} == set(PLANT_FAULT_TYPES), mechs ^ set(PLANT_FAULT_TYPES)
    for cat, (mech, comps) in CATEGORIES.items():
        if mech == SENSOR_MECHANISM:
            assert set(comps.values()) == {s.name for s in OBSERVATION_SPEC}
            continue
        print(f"{cat:30s} {mech:18s} {sorted(comps.values())}")
        assert set(comps.values()) == _allowed_targets(mech), cat
    # thresholds exist for every thresholded (mechanism, target) the sampler can draw
    for mech in ("pump_degrade", "valve_degrade", "firing_degrade", "condenser_fouling",
                 "tube_leak"):
        for t in _allowed_targets(mech):
            assert (mech, t) in P.THRESHOLDS, (mech, t)


def test_answer_vocabulary_contains_no_hotwell_event_names():
    """Answer-key hygiene: categories and components are operator words."""
    from hotwell.scenario import EVENT_TYPES
    words = set(FAULT_CATEGORIES) | {c for _, comps in CATEGORIES.values()
                                     for c in comps if comps is not None}
    instruments = {s.name for s in OBSERVATION_SPEC}
    leaked = {w for w in words - instruments if w in EVENT_TYPES}
    print(f"\nvocabulary {len(words)} words; leaked {leaked}")
    assert not leaked


# ------------------------------------------------------------------ sampler

def test_sampler_is_deterministic_and_streams_are_independent():
    a = sample_item(MASTER, "dev", 7)
    assert a == sample_item(MASTER, "dev", 7)
    others = [sample_item(MASTER, "dev", 8), sample_item(MASTER, "test", 7),
              sample_item(MASTER + 1, "dev", 7)]
    for o in others:
        assert o.sim_seed != a.sim_seed and o.profile != a.profile


def test_attempt_keeps_category_and_component_but_resamples_details():
    """B-Q5 regeneration must not bias the category mix."""
    changed = 0
    for i in range(200):
        a0, a1 = sample_item(MASTER, "dev", i, 0), sample_item(MASTER, "dev", i, 1)
        assert (a0.fault is None) == (a1.fault is None)
        if a0.fault is None:
            continue
        assert (a0.fault.category, a0.fault.component) == (a1.fault.category,
                                                           a1.fault.component)
        assert a0.sim_seed != a1.sim_seed
        changed += a0.fault.t_inj != a1.fault.t_inj
    print(f"\nonset changed under a new attempt in {changed} fault items of 200")
    assert changed > 100


def test_json_round_trip_and_schema_gate():
    for i in range(50):
        it = sample_item(MASTER, "dev", i)
        back = ItemRecord.from_json(it.to_json())
        assert back == it and back.content_hash() == it.content_hash()
    bad = sample_item(MASTER, "dev", 0).to_dict()
    bad["schema"] = "hotwell-bench/item/0"
    with pytest.raises(ValueError):
        ItemRecord.from_dict(bad)


def test_sampler_rejects_bad_inputs():
    for args in ((-1, "dev", 0), (1.5, "dev", 0), (1, "train", 0), (1, "dev", -1)):
        with pytest.raises(ValueError):
            sample_item(*args)


# Self-pin (B1, 2026-11-07): the record for (MASTER, dev, 0). A frozen item set
# is only reproducible if the sampler is, so any sampler change must show here
# and be a deliberate re-pin.
PIN_DEV0 = "c39f2fe7d08b55636910aee3bbf6b45667886563a23843564a9be617154eac6d"


def test_sampler_pin():
    it = sample_item(MASTER, "dev", 0)
    print(f"\n{it.to_json()}\nhash {it.content_hash()}")
    assert it.schema == SCHEMA
    assert it.content_hash() == PIN_DEV0


def _binom_ok(k: int, n: int, p: float, z: float = 4.0) -> bool:
    return abs(k - n * p) <= z * math.sqrt(n * p * (1 - p))


def test_sampled_distributions_match_the_plan():
    items = [sample_item(MASTER, "dev", i) for i in range(N_DIST)]
    faults = [it.fault for it in items if it.fault is not None]
    n_null = N_DIST - len(faults)
    cats = Counter(f.category for f in faults)
    kinds = Counter(f.sensor_kind for f in faults if f.sensor_kind)
    flat = sum(it.profile.flat for it in items)
    cp = sum(it.operator.cp_swap_s is not None for it in items)
    fp = sum(it.operator.fp_swap_s is not None for it in items)
    ramp = [f.shape == "ramp" for f in faults if f.category not in
            ("stuck_actuator", "actuator_sluggish", "instrument_fault")]
    print(f"\nN {N_DIST}: null {n_null} ({n_null / N_DIST:.3f}, plan {P.P_NULL}); "
          f"flat days {flat / N_DIST:.3f} (plan {P.P_FLAT_DAY}); CP swap {cp / N_DIST:.3f}, "
          f"FP swap {fp / N_DIST:.3f} (plan {P.P_SWAP}); ramped factor faults "
          f"{sum(ramp) / len(ramp):.3f} (plan {P.P_RAMP})")
    for c in FAULT_CATEGORIES:
        print(f"  {c:30s} {cats[c]:5d}")
    print(f"  sensor kinds {dict(kinds)}")
    assert _binom_ok(n_null, N_DIST, P.P_NULL)
    assert _binom_ok(flat, N_DIST, P.P_FLAT_DAY)
    assert _binom_ok(cp, N_DIST, P.P_SWAP) and _binom_ok(fp, N_DIST, P.P_SWAP)
    assert _binom_ok(sum(ramp), len(ramp), P.P_RAMP)
    for c in FAULT_CATEGORIES:
        assert _binom_ok(cats[c], len(faults), 1 / len(FAULT_CATEGORIES)), c
    for k in SENSOR_KINDS:
        assert _binom_ok(kinds[k], sum(kinds.values()), 1 / len(SENSOR_KINDS)), k

    lo, hi = P.MAG_RANGE
    for it in items:
        f = it.fault
        bp = it.profile.breakpoints()
        assert all(t1 > t0 for (t0, _), (t1, _) in zip(bp, bp[1:], strict=False))
        assert bp[-2][0] < P.DAY_S
        assert P.TROUGH_MW[0] <= it.profile.trough_mw <= P.TROUGH_MW[1]
        if f is None:
            continue
        assert P.ONSET_S[0] <= f.t_inj <= P.ONSET_S[1]
        if f.magnitude_ratio is not None:
            assert lo * (1 - 1e-12) <= f.magnitude_ratio <= hi * (1 + 1e-12), f
        if f.mechanism in ("pump_degrade", "valve_degrade", "firing_degrade"):
            (factor,) = [v for k, v in f.params.items() if k.endswith("_factor")]
            assert 0.0 < factor < 1.0
        if f.sensor_kind in ("bias", "drift"):
            assert f.threshold == detection_scale(f.target)


# ------------------------------------------------------------------ builder

def test_every_sampled_item_builds_and_validates_in_hotwell():
    nv0 = nonvital_p0_at_rated()
    n = 0
    for split in ("dev", "test"):
        for i in range(150):
            it = sample_item(MASTER, split, i)
            b = build(it)                      # load_validated raises on any bad event
            n += 1
            ev, tw = b.scenario.events, b.twin.events
            assert b.scenario.seed == b.twin.seed == it.sim_seed
            assert b.scenario.max_t == b.twin.max_t == P.DAY_S
            if it.fault is None:
                assert ev == tw
            else:
                extra = [e for e in ev if e not in tw]
                assert len(ev) == len(tw) + 1 and len(extra) == 1
                assert extra[0].t == it.fault.t_inj and extra[0].type == it.fault.mechanism
            loads = [e for e in tw if e.type == "set_load"]
            assert len(loads) == (0 if it.profile.flat else 2)
            if loads:
                assert loads[0].params["p0"] == nv0 + (it.profile.trough_mw - 60.0) * 1e6
                assert loads[-1].params["p0"] == nv0           # back to rated exactly
            ids = Counter(pid for _, pid, _ in b.schedule)
            assert ids["NOP-01"] == (it.operator.cp_swap_s is not None)
            assert ids["NOP-02"] == (it.operator.fp_swap_s is not None)
            assert b.schedule[0][0] == it.operator.nop03_cadence_s
            assert all(s0[0] <= s1[0] for s0, s1 in zip(b.schedule, b.schedule[1:],
                                                       strict=False))
    print(f"\nbuilt and hotwell-validated {n} items; nv0 {nv0 / 1e6:.4f} MW")


def test_fault_event_carries_only_hotwell_params():
    it = next(sample_item(MASTER, "dev", i) for i in range(100)
              if sample_item(MASTER, "dev", i).fault is not None)
    row = fault_event(it)
    print(f"\n{it.fault.category}/{it.fault.component}: {row}")
    assert row["t"] == it.fault.t_inj
    assert "category" not in row and "component" not in row


def test_answer_key_shape():
    keys = [answer_key(sample_item(MASTER, "dev", i)) for i in range(200)]
    for k in keys:
        if not k["fault_present"]:
            assert k == {"fault_present": False, "category": "none", "component": None,
                         "onset_s": None}
        else:
            assert k["category"] in FAULT_CATEGORIES
            assert (k["component"] is None) == (k["category"] in (
                "combustion_efficiency_loss", "condenser_vacuum_degradation",
                "boiler_tube_leak"))


def test_scenario_name_is_split_and_index_only():
    """The name carries no fault information (it may surface in logs)."""
    it = sample_item(MASTER, "dev", 3)
    assert scenario_dict(it)["name"] == "dev-0003-a0"
    assert scenario_dict(it, twin=True)["name"] == "dev-0003-a0-twin"
