"""Turn an ItemRecord into hotwell inputs (B1; PLAN-bench-0.1.md §3).

`build(item)` returns:
- the plant config: AB drum + three-element feedwater, provisional noise;
- the scenario and its fault-free twin, as YAML text and as Scenario objects
  parsed by hotwell's own `load_scenario`, so every event passes hotwell's
  load-time validation;
- the operator schedule, each entry validated with `hotwell.procedures.validate`.

No simulation happens here. The one exception is a single noise-free steady-state
initialisation at rated load, which finds the non-vital block's initial demand.
The profile is applied to that block the same way hotwell's quiet day does:
p0 = nv0 + (MW − 60)·1e6.
"""
from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from hotwell.plant import load_config
from hotwell.procedures import load_procedures, validate
from hotwell.scenario import Scenario, load_scenario
from hotwell.simulator import Simulator

from hotwell_bench import presets as P
from hotwell_bench.items import ItemRecord
from hotwell_bench.taxonomy import SENSOR_MECHANISM

#: initial pumps (hotwell baseline): CP-1 and FP-1 run, CP-2 and FP-2 stand by
CP_SWAP = ("NOP-01", {"from": "cp1", "to": "cp2"})
FP_SWAP = ("NOP-02", {"from": "fp1", "to": "fp2"})
TRIM = "NOP-03"


def build_config() -> dict:
    cfg = load_config()
    cfg["drum"]["model"] = P.DRUM_MODEL
    cfg["control"]["drum_level"]["three_element"] = P.THREE_ELEMENT
    cfg["sensors"]["noise"] = copy.deepcopy(P.SENSOR_NOISE)
    cfg["loads"]["noise"] = copy.deepcopy(P.LOAD_NOISE)
    return cfg


@lru_cache(maxsize=1)
def nonvital_p0_at_rated() -> float:
    """Non-vital block demand after steady init at rated load [W]. It is item-
    independent: the init is noise-free and draws nothing."""
    sim = Simulator(build_config(), seed=0)
    sim.initialize_steady_state(P.RATED_MW)
    return float(sim.plant.loads.nonvital.p0)


def _profile_events(item: ItemRecord, nv0: float) -> list[dict]:
    ev = []
    bp = item.profile.breakpoints()
    for (t0, m0), (t1, m1) in zip(bp, bp[1:], strict=False):
        if m1 != m0:
            ev.append({"t": t0, "type": "set_load", "target": "nonvital",
                       "p0": nv0 + (m1 - P.RATED_MW) * 1e6, "ramp_s": t1 - t0})
    return ev


def fault_event(item: ItemRecord) -> dict | None:
    f = item.fault
    if f is None:
        return None
    row = {"t": float(f.t_inj), "type": f.mechanism}
    if f.target:
        row["target"] = f.target
    row.update(f.params)
    assert f.mechanism != SENSOR_MECHANISM or "kind" in row
    return row


def scenario_dict(item: ItemRecord, twin: bool = False) -> dict:
    """The scenario as a YAML-shaped dict. twin=True drops the fault event, and
    nothing else changes (same seed, profile, length)."""
    events = _profile_events(item, nonvital_p0_at_rated())
    fe = fault_event(item)
    if fe is not None and not twin:
        events.append(fe)
    events.sort(key=lambda e: e["t"])
    return {"name": f"{item.item_id}-a{item.attempt}" + ("-twin" if twin else ""),
            "seed": item.sim_seed,
            "init": {"load_mw": P.RATED_MW, "mode": "islanded"},
            "events": events,
            "end": {"max_t": P.DAY_S}}


def scenario_yaml(item: ItemRecord, twin: bool = False) -> str:
    return yaml.safe_dump(scenario_dict(item, twin), sort_keys=False)


def load_validated(text: str) -> Scenario:
    """Parse through hotwell's own loader (which validates every event at load)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "scenario.yaml"
        path.write_text(text)
        return load_scenario(path)


def operator_schedule(item: ItemRecord) -> list[tuple[float, str, dict]]:
    """(t [s], procedure id, params), sorted by time; ties keep the order NOP-03,
    NOP-01, NOP-02. Each entry is validated against the packaged procedures."""
    op = item.operator
    sched = [(float(t), TRIM, {}) for t in range(op.nop03_cadence_s, int(P.DAY_S),
                                                  op.nop03_cadence_s)]
    if op.cp_swap_s is not None:
        sched.append((float(op.cp_swap_s), *CP_SWAP))
    if op.fp_swap_s is not None:
        sched.append((float(op.fp_swap_s), *FP_SWAP))
    sched.sort(key=lambda s: s[0])          # stable sort keeps the tie order above
    procs = load_procedures()
    for _, pid, params in sched:
        validate(procs[pid], params)
    return sched


@dataclass(frozen=True)
class BuiltItem:
    item: ItemRecord
    config: dict
    scenario: Scenario
    twin: Scenario
    scenario_yaml: str
    twin_yaml: str
    schedule: list


def build(item: ItemRecord) -> BuiltItem:
    sy, ty = scenario_yaml(item), scenario_yaml(item, twin=True)
    return BuiltItem(item=item, config=build_config(), scenario=load_validated(sy),
                     twin=load_validated(ty), scenario_yaml=sy, twin_yaml=ty,
                     schedule=operator_schedule(item))
