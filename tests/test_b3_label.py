"""B3 labelling gates: rule arithmetic on synthetic data, seeded simulator cases for
each rule, and (slow) G-B2: reproduction of hotwell's 0.4.0 detectability sweep."""
from __future__ import annotations

import numpy as np
import pytest

from hotwell_bench import label as L
from hotwell_bench.build import build
from hotwell_bench.items import SCHEMA, DayProfile, Fault, ItemRecord, OperatorPlan
from hotwell_bench.run import COLUMNS, run_built, run_items
from hotwell_bench.taxonomy import CATEGORIES

FLAT = DayProfile(True, 48.0, 3600.0, 1800.0, 14400.0, 1800.0)
NO_OPERATOR = OperatorPlan(86400, None, None, 5.0, 0.0)   # empty schedule


def fault_for(mechanism: str, target: str, t_inj: int, params: dict, kind=None) -> Fault:
    """A Fault whose category/component come from the taxonomy (reverse lookup)."""
    for cat, (mech, comps) in CATEGORIES.items():
        if mech == mechanism and target in comps.values():
            comp = next(c for c, t in comps.items() if t == target)
            comp = None if target == "" else comp
            return Fault(category=cat, component=comp, mechanism=mechanism, target=target,
                         t_inj=t_inj, shape="step", params=params, sensor_kind=kind)
    raise KeyError((mechanism, target))


def _item(fault=None, op=NO_OPERATOR, seed=42, index=9500) -> ItemRecord:
    return ItemRecord(SCHEMA, "dev", index, 0, seed, FLAT, op, fault)


# ------------------------------------------------------------- synthetic arithmetic

def _noise(n=3000, seed=0):
    """White noise with sigma = D_i / 3 per instrument (D_i = 3 sigma when noise-dominated)."""
    g = np.random.default_rng(seed)
    return g.normal(size=(n, len(COLUMNS))) * (L.SCALE / 3.0)


def test_rules_on_synthetic_signals_report_the_start_of_the_evidence():
    T = _noise()
    j = COLUMNS.index("drum_level")
    # level: +2 D_i from sample 300 on
    F = T.copy()
    F[300:, j] += 2.0 * L.SCALE[j]
    ev = L.evidence(F, T)
    print(f"\nlevel: {ev['level'][j]}, stuck {ev['stuck'][j]}, variance {ev['variance'][j]}")
    assert ev["level"][j] == 300 and ev["first_divergence"][j] == 300
    # stuck: frozen from sample 400
    F = T.copy()
    F[400:, j] = F[400, j]
    ev = L.evidence(F, T)
    print(f"stuck: level {ev['level'][j]}, stuck {ev['stuck'][j]}, variance {ev['variance'][j]}")
    assert ev["stuck"][j] == 401          # first sample that differs from the twin
    assert ev["variance"][j] is not None
    # noise x3 from sample 1500: variance fires, level may not
    F = T.copy()
    F[1500:, j] = 3.0 * T[1500:, j]
    ev = L.evidence(F, T)
    print(f"noise x3: level {ev['level'][j]}, variance {ev['variance'][j]}")
    assert ev["variance"][j] is not None and 1500 <= ev["variance"][j] < 1500 + L.VAR_WINDOW
    # a chance-length run (twin max + 5) is not stuck evidence
    F = T.copy()
    _, lt = L._runs(T[:, j])
    k = int(lt.max()) + 5
    F[2000:2000 + k, j] = F[2000, j]
    assert L.evidence(F, T)["stuck"][j] is None
    # identical: nothing anywhere
    ev = L.evidence(T, T.copy())
    assert all(x is None for r in ("level", "stuck", "variance") for x in ev[r])


def test_level_rule_needs_ten_consecutive_samples():
    T = np.zeros((100, len(COLUMNS)))
    F = T.copy()
    j = 0
    F[20:29, j] = 2 * L.SCALE[j]           # 9 samples: no
    F[50:60, j] = 2 * L.SCALE[j]           # 10 samples: yes, at 50
    assert L.level_rule(F, T)[j] == 50


# ------------------------------------------------------------- seeded simulator cases

def _label(fault, until, op=NO_OPERATOR):
    b = build(_item(fault, op))
    fa, tw = run_built(b, "faulted", until=until), run_built(b, "twin", until=until)
    return L.label_runs(b.item, fa, tw)


def test_null_item_has_no_evidence():
    lab = _label(None, 200)
    assert lab.t_det is None and lab.observable is None and not any(lab.per_rule.values())


@pytest.mark.parametrize(("name", "fault", "rule", "inst"), [
    ("FP-1 head 0.97", fault_for("pump_degrade", "fp1", 60, {"head_factor": 0.97}),
     "level", None),
    ("stuck drum_pressure sensor", fault_for("sensor_fault", "drum_pressure", 60,
                                             {"kind": "stuck"}, "stuck"),
     "stuck", "drum_pressure"),
    ("noise x2.66 on fw_valve_pos", fault_for("sensor_fault", "fw_valve_pos", 60,
                                              {"kind": "noise_increase", "factor": 2.66},
                                              "noise_increase"),
     "variance", "fw_valve_pos"),
])
def test_each_rule_catches_its_seeded_fault_and_nothing_precedes_onset(name, fault, rule, inst):
    lab = _label(fault, 1500)
    early = {r: dict(sorted(v.items(), key=lambda kv: kv[1])[:3])
             for r, v in lab.per_rule.items()}
    print(f"\n{name}: t_det {lab.t_det} by {lab.rule}/{lab.instrument}; per rule {early}")
    assert lab.observable and lab.t_det >= 60
    assert all(t >= 60 for v in lab.per_rule.values() for t in v.values())
    assert rule in lab.per_rule and lab.per_rule[rule]
    if inst is not None:
        assert inst in lab.per_rule[rule]


def test_level_rule_alone_misses_the_stuck_and_noise_cases():
    """The B2 finding, as a regression guard: without the new rules these items
    would be labelled unobservable."""
    for fault in (fault_for("sensor_fault", "drum_pressure", 60, {"kind": "stuck"}, "stuck"),
                  fault_for("sensor_fault", "fw_valve_pos", 60,
                            {"kind": "noise_increase", "factor": 2.66}, "noise_increase")):
        lab = _label(fault, 1500)
        print(f"\n{fault.target} {fault.sensor_kind}: level-rule hits {lab.per_rule['level']}")
        assert fault.target not in lab.per_rule["level"]


# ------------------------------------------------------------- G-B2 (slow)

# hotwell scripts/fault_detectability.py --fidelity astrom_bell_3e at v0.4.0 (re-run for
# the 0.4.0 release and recorded in hotwell HANDOFF): (mechanism, target, param, value)
# -> delay after onset [s] or None, first instrument.
SWEEP_AB = [
    ("pump_degrade", "fp1", "head_factor", 0.995, 2, "fw_valve_pos"),
    ("pump_degrade", "fp1", "head_factor", 0.99, 2, "fw_valve_pos"),
    ("pump_degrade", "fp1", "head_factor", 0.98, 1, "fp1_current"),
    ("pump_degrade", "fp1", "head_factor", 0.97, 1, "fp1_current"),
    ("pump_degrade", "cp1", "head_factor", 0.99, 23, "da_lcv_pos"),
    ("pump_degrade", "cp1", "head_factor", 0.98, 1, "condensate_flow"),
    ("pump_degrade", "cp1", "head_factor", 0.95, 1, "condensate_flow"),
    ("pump_degrade", "cp1", "head_factor", 0.9, 1, "condensate_flow"),
    ("valve_degrade", "frv", "cv_factor", 0.99, None, None),
    ("valve_degrade", "frv", "cv_factor", 0.98, 2, "fw_valve_pos"),
    ("valve_degrade", "frv", "cv_factor", 0.95, 2, "fw_valve_pos"),
    ("valve_degrade", "frv", "cv_factor", 0.9, 1, "fw_valve_pos"),
    ("valve_degrade", "lcv", "cv_factor", 0.99, None, None),
    ("valve_degrade", "lcv", "cv_factor", 0.98, 28, "da_lcv_pos"),
    ("valve_degrade", "lcv", "cv_factor", 0.95, 1, "condensate_flow"),
    ("valve_degrade", "lcv", "cv_factor", 0.9, 1, "condensate_flow"),
    ("firing_degrade", "", "eta_factor", 0.998, None, None),
    ("firing_degrade", "", "eta_factor", 0.995, None, None),
    ("firing_degrade", "", "eta_factor", 0.99, 99, "firing_rate"),
    ("firing_degrade", "", "eta_factor", 0.98, 21, "fw_valve_pos"),
    ("condenser_fouling", "", "slope_factor", 1.05, None, None),
    ("condenser_fouling", "", "slope_factor", 1.1, None, None),
    ("condenser_fouling", "", "slope_factor", 1.25, 1, "cond_pressure"),
    ("condenser_fouling", "", "slope_factor", 1.5, 1, "cond_pressure"),
    ("actuator_slow", "fw_valve", "rate_factor", 0.5, None, None),
    ("actuator_slow", "fw_valve", "rate_factor", 0.2, None, None),
    ("actuator_slow", "fw_valve", "rate_factor", 0.05, None, None),
    ("actuator_slow", "throttle", "rate_factor", 0.5, None, None),
    ("actuator_slow", "throttle", "rate_factor", 0.2, None, None),
    ("actuator_slow", "throttle", "rate_factor", 0.05, None, None),
    ("actuator_slow", "firing", "rate_factor", 0.5, None, None),
    ("actuator_slow", "firing", "rate_factor", 0.2, None, None),
    ("actuator_slow", "firing", "rate_factor", 0.05, None, None),
    ("tube_leak", "", "m_dot_design", 0.1, 689, "hotwell_level"),
    ("tube_leak", "", "m_dot_design", 0.3, 388, "hotwell_level"),
    ("tube_leak", "", "m_dot_design", 1.0, 123, "drum_level"),
    ("tube_leak", "", "m_dot_design", 3.0, 44, "drum_level"),
    ("actuator_stuck", "fw_valve", None, None, 4, "fw_valve_pos"),
    ("actuator_stuck", "throttle", None, None, 6, "frequency"),
    ("actuator_stuck", "firing", None, None, 115, "fw_valve_pos"),
    ("actuator_stuck", "da_lcv", None, None, 36, "da_lcv_pos"),
]


def gb2_items() -> list[ItemRecord]:
    items = []
    for k, (mech, tgt, pname, val, _, _) in enumerate(SWEEP_AB):
        params = {} if pname is None else {pname: val}
        items.append(_item(fault_for(mech, tgt, 300, params), index=9600 + k))
    return items


@pytest.mark.slow
def test_gb2_level_rule_reproduces_the_hotwell_sweep(tmp_path):
    """G-B2: on the sweep's setup (AB + 3E, flat 60 MW, no operator, seed 42, step at
    300 s, 1 h), the level rule's delay and first instrument equal hotwell's within
    1 s; no rule fires on a row hotwell finds undetectable (guards the stuck and
    variance rules against quantization-decorrelation artifacts); and the twin's
    state hash equals a standalone fault-free run's."""
    items = gb2_items()
    sums = run_items(items, tmp_path, until=3600)
    from hotwell_bench.run import run_cached
    bad = []
    for it, (mech, tgt, _pname, val, delay, inst) in zip(items, SWEEP_AB, strict=True):
        fa = run_cached(it, "faulted", tmp_path, until=3600)
        tw = run_cached(it, "twin", tmp_path, until=3600)
        lv = L.evidence(fa.values, tw.values)["level"]
        hits = [(t, COLUMNS[i]) for i, t in enumerate(lv) if t is not None]
        got = min(hits) if hits else None
        got_delay = None if got is None else max(0, got[0] - 300)
        ok = (got_delay is None and delay is None) or (
            got_delay is not None and delay is not None and abs(got_delay - delay) <= 1
            and got[1] == inst)
        lab = L.label_runs(it, fa, tw)
        print(f"{mech:18s} {tgt:9s} {val!s:>6s}: hotwell {delay!s:>4s} {inst or '-':16s} "
              f"bench level {got_delay!s:>4s} {got[1] if got else '-':16s} | all rules "
              f"t_det-300 {None if lab.t_det is None else lab.t_det - 300} ({lab.rule}/"
              f"{lab.instrument}) {'OK' if ok else 'MISMATCH'}")
        if delay is None and lab.t_det is not None:
            ok = False                      # no rule may fire where hotwell finds nothing
        if not ok:
            bad.append((mech, tgt, val))
    twins = {sums[(it.item_id, 'twin')]['state_hash'] for it in items}
    null = run_built(build(_item(None, index=9599)), "faulted", until=3600).summary
    print(f"twin state hashes distinct: {len(twins)}; standalone fault-free "
          f"{null['state_hash'][:16]}…")
    assert not bad, bad
    assert twins == {null["state_hash"]}
