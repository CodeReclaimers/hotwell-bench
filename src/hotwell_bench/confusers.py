"""Confusable-fault check (B3; PLAN-bench-0.1.md §4.3).

For each observable fault item, a short a-priori table proposes CONFUSERS:
faults of another category or component that could produce the same sensed
signature. Each confuser's magnitude is fitted to match the primary's
signature. If no labelling rule (label.evidence) can then tell the confuser's
run from the primary's, the confuser is accepted as an equally correct answer.

Mechanics:
- Twin snapshot: the item's twin, which replays the primary's recorded operator
  actions, is driven to just before onset (t_inj − 1 s) and snapshotted
  (run.Drive.snapshot, exact). Branches inherit the replay, so every confuser
  run sees exactly the primary's operator actions. Every trial branches from that
  snapshot with a scenario that contains the confuser's event, built and
  validated through the normal builder.
- Signature: over a fit window starting SETTLE_S after the primary's ramp ends
  (FIT_WINDOW_S long, cut at the first scheduled pump swap inside it if at least
  MIN_FIT_WINDOW_S remains), the instrument with the largest |mean(F − T)| / D_i,
  together with that mean.
- Fit: secant iteration on the confuser's deviation until its mean shift on the
  signature instrument matches within FIT_TOL (relative) or FIT_TOL_D × D_i.
  A deviation that cannot produce the signature's sign is "cannot reproduce".
- Verdict: the fitted confuser runs to the primary's end and is compared with the
  primary over the whole day using the same rules as onset labelling. The
  primary is the reference, so both runs share every noise draw. If either run
  ends early (plant death) and the other does not, they are separable.

Pairs NOT in the table (recorded, deliberately out of 0.1 scope):
- tube_leak vs hotwell_level sensor drift: a two-parameter fit (rate and cap);
- actuator_sluggish vs sensor lag: the signature exists only in transients, so a
  settled-window fit does not apply.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass

import numpy as np

from hotwell_bench import presets as P
from hotwell_bench.build import build
from hotwell_bench.items import Fault, ItemRecord, detection_scale
from hotwell_bench.label import evidence
from hotwell_bench.run import COLUMNS, Drive, RunResult
from hotwell_bench.taxonomy import CATEGORIES

SETTLE_S = 600.0
FIT_WINDOW_S = 1800.0
MIN_FIT_WINDOW_S = 300.0
FIT_TOL, FIT_TOL_D = 0.05, 0.25
MAX_EVALS = 6
POSITION_INSTRUMENT = {"fw_valve": "fw_valve_pos", "throttle": "throttle_pos",
                       "firing": "firing_rate", "da_lcv": "da_lcv_pos",
                       "makeup_valve": "makeup_valve_pos"}
_FACTOR = {"pump_degrade": "head_factor", "valve_degrade": "cv_factor",
           "firing_degrade": "eta_factor"}


@dataclass(frozen=True)
class Candidate:
    mechanism: str
    target: str
    sensor_kind: str | None = None     # for sensor_fault confusers
    fitted: bool = True                # False: no magnitude (stuck kinds)


@dataclass
class ConfuserResult:
    category: str
    component: str | None
    mechanism: str
    target: str
    params: dict | None
    confusable: bool
    reason: str
    signature: str | None = None
    target_shift: float | None = None
    achieved_shift: float | None = None
    evaluations: int = 0
    separated_by: list | None = None   # [t, rule, instrument]

    def to_dict(self) -> dict:
        return asdict(self)


def running_at(run: RunResult, t: float, prefix: str) -> str:
    """Which of <prefix>1 / <prefix>2 is running at time t (from indication events)."""
    state = {f"{prefix}1": True, f"{prefix}2": False}     # hotwell baseline
    for e in run.events:
        if e["t"] > t:
            break
        if e["kind"] == "indication" and e["id"] in (f"{prefix}1_running", f"{prefix}2_running"):
            state[e["id"][:3]] = bool(e["value"])
    on = [k for k, v in state.items() if v]
    return on[0] if len(on) == 1 else f"{prefix}1"


def ramp_s(f: Fault) -> float:
    if f.mechanism == "sensor_fault" and f.params.get("kind") == "drift":
        return abs(f.params["max_bias"] / f.params["rate"])
    return float(f.params.get("ramp_s", 0.0))


def candidates(item: ItemRecord, faulted: RunResult) -> list[Candidate]:
    f = item.fault
    if f is None:
        return []
    fp = running_at(faulted, f.t_inj, "fp")
    cp = running_at(faulted, f.t_inj, "cp")
    m, t, kind = f.mechanism, f.target, f.sensor_kind
    out: list[Candidate] = []
    if m == "pump_degrade" and t == fp:
        out.append(Candidate("valve_degrade", "frv"))
    elif m == "pump_degrade" and t == cp:
        out.append(Candidate("valve_degrade", "lcv"))
    elif m == "valve_degrade" and t == "frv":
        out.append(Candidate("pump_degrade", fp))
    elif m == "valve_degrade" and t == "lcv":
        out.append(Candidate("pump_degrade", cp))
    elif m == "firing_degrade":
        out.append(Candidate("sensor_fault", "firing_rate", "bias"))
    elif m == "condenser_fouling":
        out.append(Candidate("sensor_fault", "cond_pressure", "bias"))
    elif m == "sensor_fault" and kind in ("bias", "drift") and t == "firing_rate":
        out.append(Candidate("firing_degrade", ""))
    elif m == "sensor_fault" and kind in ("bias", "drift") and t == "cond_pressure":
        out.append(Candidate("condenser_fouling", ""))
    elif m == "actuator_stuck":
        out.append(Candidate("sensor_fault", POSITION_INSTRUMENT[t], "stuck", fitted=False))
    elif m == "sensor_fault" and kind == "stuck" and t in POSITION_INSTRUMENT.values():
        act = next(a for a, i in POSITION_INSTRUMENT.items() if i == t)
        out.append(Candidate("actuator_stuck", act, fitted=False))
    return out


def _category(mechanism: str, target: str) -> tuple[str, str | None]:
    for cat, (mech, comps) in CATEGORIES.items():
        if mech == mechanism and target in comps.values():
            comp = next(c for c, tg in comps.items() if tg == target)
            return cat, (None if target == "" else comp)
    raise KeyError((mechanism, target))


def confuser_fault(primary: Fault, c: Candidate, dev: float | None) -> Fault:
    """The confuser as a Fault at the primary's onset and shape. `dev` is the
    deviation (signed for sensor bias); None for unfitted kinds."""
    cat, comp = _category(c.mechanism, c.target)
    R = ramp_s(primary)
    common = dict(category=cat, component=comp, mechanism=c.mechanism, target=c.target,
                  t_inj=primary.t_inj)
    if c.mechanism == "sensor_fault":
        if c.sensor_kind == "stuck":
            return Fault(**common, shape="step", params={"kind": "stuck"}, sensor_kind="stuck")
        if R > 0:
            return Fault(**common, shape="ramp", sensor_kind="drift", magnitude=dev,
                         params={"kind": "drift", "rate": dev / R, "max_bias": abs(dev)})
        return Fault(**common, shape="step", sensor_kind="bias", magnitude=dev,
                     params={"kind": "bias", "bias": dev})
    if c.mechanism == "actuator_stuck":
        return Fault(**common, shape="step", params={})
    if c.mechanism in _FACTOR:
        params = {_FACTOR[c.mechanism]: 1.0 - dev}
    elif c.mechanism == "condenser_fouling":
        params = {"slope_factor": 1.0 + dev}
    else:
        raise ValueError(c.mechanism)
    if R > 0:
        params["ramp_s"] = R
    return Fault(**common, shape="ramp" if R > 0 else "step", params=params, magnitude=dev)


def _initial_dev(c: Candidate, sig: str, shift: float) -> float:
    if c.mechanism == "sensor_fault":
        return shift if c.target == sig else detection_scale(c.target)
    return P.THRESHOLDS.get((c.mechanism, c.target), (0.01, ""))[0]


def _valid(c: Candidate, dev: float) -> bool:
    if c.mechanism == "sensor_fault":
        return True
    return 0.0 < dev < 0.9


class _Brancher:
    """Twin snapshot at t_inj − 1 s and branch runs with a given confuser fault."""

    def __init__(self, item: ItemRecord, actions: list):
        self.item = item
        twin = Drive(build(item), "twin", replay=actions)
        twin.advance(item.fault.t_inj - 1.0)
        self.snap = twin

    def run(self, fault: Fault, until: float) -> RunResult:
        conf_item = dataclasses.replace(self.item, fault=fault)
        sc = build(conf_item).scenario
        d = self.snap.snapshot()
        d.sim.scenario = sc                     # same prefix; the new event lies ahead
        d.variant = "faulted"
        d.advance(until)
        return d.result()


def check_item(item: ItemRecord, fa: RunResult, tw: RunResult) -> list[ConfuserResult]:
    f = item.fault
    cands = candidates(item, fa)
    if not cands:
        return []
    n = min(len(fa.t), len(tw.t))
    t_end = float(fa.t[n - 1])
    w0 = f.t_inj + ramp_s(f) + SETTLE_S
    w1 = min(w0 + FIT_WINDOW_S, t_end)
    # a pump swap inside the window changes the signature mid-fit; stop the window at
    # the first scheduled swap after it opens if enough window remains
    swaps = [t for t in (item.operator.cp_swap_s, item.operator.fp_swap_s)
             if t is not None and w0 < t < w1]
    if swaps and min(swaps) - w0 >= MIN_FIT_WINDOW_S:
        w1 = float(min(swaps))
    br = _Brancher(item, fa.actions)
    results = []
    for c in cands:
        cat, comp = _category(c.mechanism, c.target)
        base = dict(category=cat, component=comp, mechanism=c.mechanism, target=c.target)
        sig = shift = None
        dev = None
        evals = 0
        achieved = None
        if c.fitted:
            if w1 - w0 < MIN_FIT_WINDOW_S:
                results.append(ConfuserResult(**base, params=None, confusable=False,
                                              reason="no settled fit window before the end"))
                continue
            sl = slice(int(w0), int(w1) + 1)
            d = (fa.values[sl] - tw.values[sl]).mean(axis=0)
            rel = np.abs(d) / np.array([detection_scale(k) for k in COLUMNS])
            j = int(np.argmax(rel))
            sig, shift = COLUMNS[j], float(d[j])
            tol = max(FIT_TOL * abs(shift), FIT_TOL_D * detection_scale(sig))

            def measure(x: float, _j=j, _sl=sl, _c=c) -> float:
                r = br.run(confuser_fault(f, _c, x), until=w1 + 1)
                return float((r.values[_sl, _j] - tw.values[_sl, _j]).mean())

            x0 = _initial_dev(c, sig, shift)
            m0 = measure(x0)
            evals = 1
            pts = [(x0, m0)]
            if m0 == 0.0 or np.sign(m0) != np.sign(shift):
                results.append(ConfuserResult(**base, params=None, confusable=False,
                                              reason="cannot reproduce the signature's sign",
                                              signature=sig, target_shift=shift,
                                              achieved_shift=m0, evaluations=evals))
                continue
            x = x0 * shift / m0
            while evals < MAX_EVALS:
                if not _valid(c, x):
                    break
                m = measure(x)
                evals += 1
                pts.append((x, m))
                if abs(m - shift) <= tol:
                    break
                (xa, ma), (xb, mb) = pts[-2], pts[-1]
                x = xb + (shift - mb) * (xb - xa) / (mb - ma) if mb != ma else xb * shift / mb
            dev, achieved = min(pts, key=lambda p: abs(p[1] - shift))
            if abs(achieved - shift) > tol or not _valid(c, dev):
                results.append(ConfuserResult(**base, params=None, confusable=False,
                                              reason="fit did not converge", signature=sig,
                                              target_shift=shift, achieved_shift=achieved,
                                              evaluations=evals))
                continue
        cf = confuser_fault(f, c, dev)
        cr = br.run(cf, until=t_end)            # same sample count as the primary
        same_end = (len(cr.t) == len(fa.t)) and (cr.summary["plant_dead"]
                                                  == fa.summary["plant_dead"])
        ev = evidence(cr.values, fa.values)
        hits = [(t, r, COLUMNS[i]) for r in ("level", "stuck", "variance")
                for i, t in enumerate(ev[r]) if t is not None]
        sep = min(hits) if hits else None
        confusable = same_end and sep is None
        reason = ("indistinguishable by every rule" if confusable else
                  "different end (plant death)" if not same_end else "separated")
        results.append(ConfuserResult(
            **base, params=cf.params, confusable=confusable, reason=reason, signature=sig,
            target_shift=shift, achieved_shift=achieved, evaluations=evals,
            separated_by=None if sep is None else [int(sep[0]), sep[1], sep[2]]))
    return results
