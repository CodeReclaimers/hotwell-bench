"""Item records and the item sampler (B1; PLAN-bench-0.1.md §3).

An item is a frozen parameter record. It holds everything needed to rebuild the
item's hotwell config, scenario and operator schedule, and nothing that depends on
running the simulator. Simulation, labelling and packaging consume records.

Seeding. Every random choice comes from numpy SeedSequence trees keyed by
(master_seed, split, index, ...), never from the simulator's own streams:
- STRUCTURE stream (master, split, index, 0): fault present?, category, component.
  Independent of `attempt`, so regenerating an unobservable item (B-Q5) keeps
  its category and component and the category mix is not biased toward easy faults.
- DETAIL stream (master, split, index, 1, attempt): simulator seed, day profile,
  operator plan, fault onset/shape/magnitude. A new attempt resamples all of
  these. This matters because e.g. a degraded standby pump is only observable if
  the day contains a swap onto it.
Each sampler draws a fixed number of values whatever the branch taken, so adding
a branch-dependent field later does not shift unrelated draws.

The master seed is a run-time input. For the test split it must never be
committed (PLAN §1).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from hotwell.sensors import OBSERVATION_SPEC

from hotwell_bench import presets as P
from hotwell_bench.taxonomy import (
    CATEGORIES,
    NULL_CATEGORY,
    SAMPLED_CATEGORIES,
    SENSOR_KINDS,
    SENSOR_MECHANISM,
    plantwide,
    sampled_components,
)

SCHEMA = "hotwell-bench/item/1"
SPLIT_IDS = {"dev": 1, "test": 2}
_TAG_STRUCTURE, _TAG_DETAIL = 0, 1
_SPEC = {s.name: s for s in OBSERVATION_SPEC}


# ----------------------------------------------------------------- records

@dataclass(frozen=True)
class DayProfile:
    """Gross-MW day profile: 60 MW, optional ramp down to a trough and back."""
    flat: bool
    trough_mw: float
    ramp_down_start_s: float
    ramp_down_s: float
    trough_len_s: float
    ramp_up_s: float

    def breakpoints(self) -> list[tuple[float, float]]:
        """(t [s], MW) piecewise-linear breakpoints over [0, DAY_S]."""
        if self.flat:
            return [(0.0, P.RATED_MW), (P.DAY_S, P.RATED_MW)]
        t1 = self.ramp_down_start_s
        t2 = t1 + self.ramp_down_s
        t3 = t2 + self.trough_len_s
        t4 = t3 + self.ramp_up_s
        return [(0.0, P.RATED_MW), (t1, P.RATED_MW), (t2, self.trough_mw),
                (t3, self.trough_mw), (t4, P.RATED_MW), (P.DAY_S, P.RATED_MW)]


@dataclass(frozen=True)
class OperatorPlan:
    nop03_cadence_s: int
    cp_swap_s: int | None          # None: no condensate pump swap today
    fp_swap_s: int | None
    reaction_s: float
    reaction_sigma: float


@dataclass(frozen=True)
class Fault:
    category: str                  # answer-key category (taxonomy)
    component: str | None          # answer-key component; None if plant-wide
    mechanism: str                 # hotwell event type (never packaged)
    target: str                    # hotwell target ("" if none)
    t_inj: int                     # onset [s]
    shape: str                     # "step" | "ramp"
    params: dict = field(default_factory=dict)   # hotwell event params besides target
    sensor_kind: str | None = None  # recorded, not graded for 0.1 (B-Q3)
    magnitude: float | None = None  # deviation from healthy (see presets.THRESHOLDS)
    threshold: float | None = None
    threshold_source: str | None = None

    @property
    def magnitude_ratio(self) -> float | None:
        if self.magnitude is None or not self.threshold:
            return None
        return abs(self.magnitude) / self.threshold


@dataclass(frozen=True)
class ItemRecord:
    schema: str
    split: str
    index: int
    attempt: int
    sim_seed: int
    profile: DayProfile
    operator: OperatorPlan
    fault: Fault | None

    @property
    def item_id(self) -> str:
        return f"{self.split}-{self.index:04d}"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys): the byte form hashed and stored in keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> ItemRecord:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unknown item schema {d.get('schema')!r} (expected {SCHEMA})")
        f = d["fault"]
        return cls(schema=d["schema"], split=d["split"], index=d["index"],
                   attempt=d["attempt"], sim_seed=d["sim_seed"],
                   profile=DayProfile(**d["profile"]), operator=OperatorPlan(**d["operator"]),
                   fault=None if f is None else Fault(**f))

    @classmethod
    def from_json(cls, s: str) -> ItemRecord:
        return cls.from_dict(json.loads(s))


def answer_key(item: ItemRecord) -> dict:
    """What the grader compares against (t_det and confusers are added by B3)."""
    f = item.fault
    if f is None:
        return {"fault_present": False, "category": NULL_CATEGORY, "component": None,
                "onset_s": None}
    return {"fault_present": True, "category": f.category, "component": f.component,
            "onset_s": f.t_inj, "sensor_kind": f.sensor_kind}


# ----------------------------------------------------------------- sampling

def _rng(*entropy: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(entropy))))


def detection_scale(instrument: str) -> float:
    """D_i = max(3 σ_i, 2 LSB_i): the twin criterion's per-instrument bound."""
    sigma = P.SENSOR_NOISE.get(instrument, {}).get("sigma", 0.0)
    return max(3.0 * sigma, 2.0 * _SPEC[instrument].resolution)


def _sample_profile(g: np.random.Generator) -> DayProfile:
    # fixed draw count: 6 values, whether or not the day is flat
    flat = bool(g.random() < P.P_FLAT_DAY)
    trough = round(float(g.uniform(*P.TROUGH_MW)), 1)
    start = float(round(g.uniform(*P.RAMP_DOWN_START_S)))
    down = float(round(g.uniform(*P.RAMP_S)))
    length = float(round(g.uniform(*P.TROUGH_LEN_S)))
    up = float(round(g.uniform(*P.RAMP_S)))
    return DayProfile(flat, trough, start, down, length, up)


def _sample_operator(g: np.random.Generator) -> OperatorPlan:
    # fixed draw count: 5 values
    cadence = int(g.integers(P.NOP03_CADENCE_S[0], P.NOP03_CADENCE_S[1] + 1))
    cp_on = bool(g.random() < P.P_SWAP)
    cp_t = int(g.integers(P.SWAP_WINDOW_S[0], P.SWAP_WINDOW_S[1] + 1))
    fp_on = bool(g.random() < P.P_SWAP)
    fp_t = int(g.integers(P.SWAP_WINDOW_S[0], P.SWAP_WINDOW_S[1] + 1))
    return OperatorPlan(cadence, cp_t if cp_on else None, fp_t if fp_on else None,
                        P.REACTION_S, P.REACTION_SIGMA)


_FACTOR_PARAM = {"pump_degrade": "head_factor", "valve_degrade": "cv_factor",
                 "firing_degrade": "eta_factor"}


def _sample_fault(g: np.random.Generator, category: str, component: str) -> Fault:
    """Fixed draw count: 8 uniforms from g, consumed in the same order whatever
    the mechanism, then mapped through the mechanism's ranges."""
    u = g.random(8)
    mechanism, comps = CATEGORIES[category]
    target = comps[component]
    comp_out = None if plantwide(category) else component
    t_inj = int(P.ONSET_S[0] + math.floor(u[0] * (P.ONSET_S[1] - P.ONSET_S[0] + 1)))

    def logu(x: float, lo: float, hi: float) -> float:
        return float(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))))

    ratio = logu(u[1], *P.MAG_RANGE)
    ramp = bool(u[2] < P.P_RAMP)
    ramp_s = float(round(logu(u[3], *P.ONSET_RAMP_S)))
    common = dict(category=category, component=comp_out, mechanism=mechanism,
                  target=target, t_inj=t_inj)

    if mechanism in ("pump_degrade", "valve_degrade", "firing_degrade",
                     "condenser_fouling", "tube_leak"):
        thr, src = P.THRESHOLDS[(mechanism, target)]
        dev = ratio * thr
        if mechanism in _FACTOR_PARAM:
            params = {_FACTOR_PARAM[mechanism]: 1.0 - dev}
        elif mechanism == "condenser_fouling":
            params = {"slope_factor": 1.0 + dev}
        else:
            params = {"m_dot_design": dev}
        if ramp:
            params["ramp_s"] = ramp_s
        return Fault(**common, shape="ramp" if ramp else "step", params=params,
                     magnitude=dev, threshold=thr, threshold_source=src)

    if mechanism == "actuator_slow":
        rf = logu(u[4], *P.SLOW_RATE_FACTOR)
        return Fault(**common, shape="step", params={"rate_factor": rf}, magnitude=rf,
                     threshold=None, threshold_source="never detected at constant load")

    if mechanism == "actuator_stuck":
        return Fault(**common, shape="step", params={})

    assert mechanism == SENSOR_MECHANISM
    kind = SENSOR_KINDS[min(int(u[5] * len(SENSOR_KINDS)), len(SENSOR_KINDS) - 1)]
    d_i = detection_scale(target)
    sign = 1.0 if u[6] < 0.5 else -1.0
    if kind == "bias":
        b = sign * ratio * d_i
        return Fault(**common, shape="step", params={"kind": "bias", "bias": b},
                     sensor_kind=kind, magnitude=b, threshold=d_i,
                     threshold_source="D_i = max(3 sigma, 2 LSB)")
    if kind == "drift":
        b = sign * ratio * d_i
        t_reach = logu(u[7], *P.SENSOR_DRIFT_TIME_S)
        return Fault(**common, shape="ramp",
                     params={"kind": "drift", "rate": b / t_reach, "max_bias": abs(b)},
                     sensor_kind=kind, magnitude=b, threshold=d_i,
                     threshold_source="D_i = max(3 sigma, 2 LSB)")
    if kind == "stuck":
        return Fault(**common, shape="step", params={"kind": "stuck"}, sensor_kind=kind)
    if kind == "lag":
        tau = logu(u[4], *P.SENSOR_LAG_TAU_S)
        return Fault(**common, shape="step", params={"kind": "lag", "tau_s": tau},
                     sensor_kind=kind, magnitude=tau)
    factor = logu(u[4], *P.SENSOR_NOISE_FACTOR)
    return Fault(**common, shape="step", params={"kind": "noise_increase", "factor": factor},
                 sensor_kind=kind, magnitude=factor)


def sample_item(master_seed: int, split: str, index: int, attempt: int = 0) -> ItemRecord:
    if not (isinstance(master_seed, int) and master_seed >= 0):
        raise ValueError(f"master_seed must be a non-negative int, got {master_seed!r}")
    if split not in SPLIT_IDS:
        raise ValueError(f"split must be one of {sorted(SPLIT_IDS)}, got {split!r}")
    if index < 0 or attempt < 0:
        raise ValueError("index and attempt must be >= 0")
    sid = SPLIT_IDS[split]

    s = _rng(master_seed, sid, index, _TAG_STRUCTURE)
    u_null, u_cat, u_comp = s.random(3)          # fixed: 3 draws
    null = bool(u_null < P.P_NULL)
    category = SAMPLED_CATEGORIES[min(int(u_cat * len(SAMPLED_CATEGORIES)),
                                      len(SAMPLED_CATEGORIES) - 1)]
    names = sampled_components(category)
    component = names[min(int(u_comp * len(names)), len(names) - 1)]

    d = _rng(master_seed, sid, index, _TAG_DETAIL, attempt)
    sim_seed = int(d.integers(0, 2**31))
    profile = _sample_profile(d)
    operator = _sample_operator(d)
    fault = _sample_fault(d, category, component)  # drawn even for null items (fixed count)
    return ItemRecord(SCHEMA, split, index, attempt, sim_seed, profile, operator,
                      None if null else fault)
