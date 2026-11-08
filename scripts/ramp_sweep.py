#!/usr/bin/env python3
"""Ramped-onset detectability sweep (B3; PLAN-bench-0.1.md §3, §8).

Usage: python scripts/ramp_sweep.py [--workers N] [--cache DIR]

The 0.4.0 sweep measured step onsets only. Here every thresholded fault
(presets.THRESHOLDS) is run at 1×, 2× and 4× its step threshold, as a step
(ramp 0) and with ramp_s ∈ {600, 3600, 21600}. Setup as in G-B2: AB + 3E, flat
60 MW, no operator, seed 42, onset at 1800 s, run until onset + ramp + 3600 s.
Labelled with the bench rules (label.label_runs). Reported per row:
- t_det − t_inj;
- t_det − (t_inj + ramp_s): negative means detected before the ramp completed;
- rule and first instrument.

Standby-pump and makeup-valve rows are expected to be unobservable here (no swap,
no makeup demand); they are kept to document that.
"""
from __future__ import annotations

import argparse
import sys
import tempfile

from hotwell_bench import presets as P
from hotwell_bench.items import SCHEMA, DayProfile, Fault, ItemRecord, OperatorPlan
from hotwell_bench.label import label_runs
from hotwell_bench.run import run_cached, run_items
from hotwell_bench.taxonomy import CATEGORIES

ONSET = 1800
RAMPS = (0.0, 600.0, 3600.0, 21600.0)
RATIOS = (1.0, 2.0, 4.0)
FLAT = DayProfile(True, 48.0, 3600.0, 1800.0, 14400.0, 1800.0)
NO_OPERATOR = OperatorPlan(86400, None, None, 5.0, 0.0)
FACTOR = {"pump_degrade": "head_factor", "valve_degrade": "cv_factor",
          "firing_degrade": "eta_factor"}


def make_item(k: int, mech: str, target: str, ratio: float, ramp: float) -> ItemRecord:
    thr = P.THRESHOLDS[(mech, target)][0]
    dev = ratio * thr
    if mech in FACTOR:
        params = {FACTOR[mech]: 1.0 - dev}
    elif mech == "condenser_fouling":
        params = {"slope_factor": 1.0 + dev}
    else:
        params = {"m_dot_design": dev}
    if ramp > 0:
        params["ramp_s"] = ramp
    cat, comp = next((c, next(n for n, t in comps.items() if t == target))
                     for c, (m, comps) in CATEGORIES.items()
                     if m == mech and target in comps.values())
    f = Fault(category=cat, component=None if target == "" else comp, mechanism=mech,
              target=target, t_inj=ONSET, shape="ramp" if ramp > 0 else "step",
              params=params, magnitude=dev, threshold=thr)
    return ItemRecord(SCHEMA, "dev", 9800 + k, 0, 42, FLAT, NO_OPERATOR, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    cache = a.cache or tempfile.mkdtemp(prefix="ramp-")
    rows = []
    k = 0
    for (mech, target) in P.THRESHOLDS:
        for ramp in RAMPS:
            for ratio in RATIOS:
                rows.append((make_item(k, mech, target, ratio, ramp), mech, target, ratio, ramp))
                k += 1
    by_until: dict[float, list] = {}
    for it, *_ in rows:
        by_until.setdefault(ONSET + it.fault.params.get("ramp_s", 0.0) + 3600.0, []).append(it)
    for until, items in sorted(by_until.items()):
        run_items(items, cache, workers=a.workers, until=until)
    print(f"\nramped-onset sweep (AB + 3E, flat 60 MW, no operator, seed 42, onset {ONSET} s)")
    print(f"{'fault':28s} {'x thr':>5s} {'ramp_s':>7s} {'t_det-t_inj':>11s} "
          f"{'t_det-ramp_end':>14s}  rule/instrument")
    for it, mech, target, ratio, ramp in rows:
        until = ONSET + ramp + 3600.0
        fa = run_cached(it, "faulted", cache, until=until)
        tw = run_cached(it, "twin", cache, until=until)
        lab = label_runs(it, fa, tw)
        d = None if lab.t_det is None else lab.t_det - ONSET
        de = None if d is None else d - int(ramp)
        print(f"{mech + ' ' + target:28s} {ratio:5.0f} {ramp:7.0f} "
              f"{'n/d' if d is None else d:>11} {'-' if de is None else de:>14}  "
              f"{'-' if lab.rule is None else f'{lab.rule}/{lab.instrument}'}", flush=True)
    print(f"cache: {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
