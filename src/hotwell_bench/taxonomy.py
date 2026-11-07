"""Answer taxonomy (operator vocabulary) and its mapping onto hotwell mechanisms.

PLAN-bench-0.1.md §2.1. The answer key uses CATEGORY and COMPONENT names only;
hotwell event names never reach a packaged item (B4 leak scan). The tests check
this table against hotwell's own fault tables in both directions, so a mechanism
hotwell adds or renames cannot be silently missed.

One naming fix against the plan's table: the plan lists "LCV" under valve
capacity loss and "DA LCV" under stuck actuators. They are the same valve
(hotwell `lcv` / `da_lcv`, the deaerator level-control valve on the condensate
line), so both categories use "DA LCV".

A second deviation, found by the answer-hygiene test: the plan's category
`actuator_stuck` is identical to the hotwell event name, which B4's leak scan
forbids in packaged files (task.md lists the categories). It is renamed
`stuck_actuator`.
"""
from __future__ import annotations

from hotwell.sensors import OBSERVATION_SPEC

NULL_CATEGORY = "none"
SENSOR_MECHANISM = "sensor_fault"

#: category -> (hotwell mechanism, {component name -> hotwell target})
#: target "" = the mechanism takes no target (plant-wide).
CATEGORIES: dict[str, tuple[str, dict[str, str]]] = {
    "pump_performance_loss": ("pump_degrade", {
        "CP-1": "cp1", "CP-2": "cp2", "FP-1": "fp1", "FP-2": "fp2"}),
    "valve_capacity_loss": ("valve_degrade", {
        "FRV": "frv", "DA LCV": "lcv", "makeup valve": "makeup_valve"}),
    "stuck_actuator": ("actuator_stuck", {
        "FRV": "fw_valve", "throttle": "throttle", "firing": "firing",
        "DA LCV": "da_lcv", "makeup valve": "makeup_valve"}),
    "actuator_sluggish": ("actuator_slow", {
        "FRV": "fw_valve", "throttle": "throttle", "firing": "firing"}),
    "combustion_efficiency_loss": ("firing_degrade", {"—": ""}),
    "condenser_vacuum_degradation": ("condenser_fouling", {"—": ""}),
    "boiler_tube_leak": ("tube_leak", {"—": ""}),
    "instrument_fault": (SENSOR_MECHANISM, {s.name: s.name for s in OBSERVATION_SPEC}),
}
FAULT_CATEGORIES = tuple(CATEGORIES)          # sampling order is this order
SENSOR_KINDS = ("bias", "drift", "stuck", "lag", "noise_increase")


def plantwide(category: str) -> bool:
    """True when the category has no component to name (the answer's component is null)."""
    return set(CATEGORIES[category][1].values()) == {""}
