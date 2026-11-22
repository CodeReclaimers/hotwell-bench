#!/usr/bin/env python3
"""Windowed detectability sweep: step thresholds under the P02 rules (PLAN-bench-0.2.md §3.1, C2).

Setup, as in G-B2 (tests/test_b3_label.py, scripts/gb2_sweep.py): AB + 3E, flat
60 MW, no operator, seed 42, step onset at 300 s, run to 3600 s. Every item is
labelled with `label.label_runs(..., preset=P02)`. Those rules are level, stuck,
variance and the windowed rule. A grid point counts as detected when the label is
observable (t_det ≤ 3600 s by construction). The strict stratum's verdict is recorded
alongside, as a diagnostic.

Rows:
- mechanism rows: every `P01.thresholds` row whose source starts with "swept".
  - Deviations are the 0.1 threshold × 2^(−k/2), k = −2 … 10, i.e. 2× down to 1/32×.
  - The "assumed: X" rows take X's new value, as in 0.1.
- sensor bias: every instrument, both signs, bias = D_i × 2^(−k/2), k = −2 … 10.
  - D_i = max(3σ_i, 2 LSB_i), the 0.1 sensor scale (items.detection_scale).
  - The instrument's threshold is the larger of the two signs' thresholds, floored at
    1 LSB (Alan, 2026-11-22): below 1 LSB only the variance rule fires, on instruments
    that sit at a clipped zero (standby pump currents, makeup flow and valve).
  - An instrument with no threshold (frequency: the control loop cancels the reading's
    bias) is not to be sampled for bias or drift (Alan, 2026-11-22; wired in C3).
- sensor noise_increase: every instrument, factor in NOISE_FACTORS.
- drift takes the bias threshold of its instrument ("assumed: bias"), because a
  drift ends at its maximum bias. stuck has no magnitude.

Threshold of a row: the smallest grid value g such that g and every larger grid value
are detected. Detected grid values below the threshold are reported as non-monotone.
A row whose largest grid value is not detected gets no threshold.

Horizon: 1 h after a 300 s onset, the 0.1 definition (Alan, 2026-11-22). It binds on
tube_leak, whose threshold point is detected only at 2964 s, so the table errs toward
larger faults there.

Output:
- the raw results as JSON (default out/window_sweep/results.json);
- a generated module (default src/hotwell_bench/window_detectability.py) holding
  THRESHOLDS_V020, SENSOR_BIAS_THRESHOLD and SENSOR_NOISE_THRESHOLD, with
  provenance strings in the 0.1 style.

Usage: python scripts/window_sweep.py [--workers N] [--cache DIR] [--out-json F] [--out-module F]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from hotwell.sensors import OBSERVATION_SPEC

from hotwell_bench import window_thresholds as WT
from hotwell_bench.items import SCHEMA, DayProfile, Fault, ItemRecord, OperatorPlan, detection_scale
from hotwell_bench.label import label_runs
from hotwell_bench.presets import P01, P02
from hotwell_bench.provenance import check_environment
from hotwell_bench.run import COLUMNS, run_cached, run_items
from hotwell_bench.secrecy import REPO_ROOT
from hotwell_bench.taxonomy import CATEGORIES, SENSOR_MECHANISM

ONSET = 300
UNTIL = 3600
SEED = 42
GRID_K = tuple(range(-2, 11))                       # multiplier 2^(-k/2)
NOISE_FACTORS = (1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
FLAT = DayProfile(True, 48.0, 3600.0, 1800.0, 14400.0, 1800.0)   # as tests/test_b3_label.py
NO_OPERATOR = OperatorPlan(86400, None, None, 5.0, 0.0)
FACTOR = {"pump_degrade": "head_factor", "valve_degrade": "cv_factor",
          "firing_degrade": "eta_factor"}
INDEX0 = 7000
LSB = {x.name: x.resolution for x in OBSERVATION_SPEC}


def mult(k: int) -> float:
    return 2.0 ** (-k / 2.0)


def _fault(mech: str, target: str, params: dict, magnitude, kind=None) -> Fault:
    for cat, (m, comps) in CATEGORIES.items():
        if m == mech and target in comps.values():
            comp = next(c for c, t in comps.items() if t == target)
            return Fault(category=cat, component=None if target == "" else comp,
                         mechanism=mech, target=target, t_inj=ONSET, shape="step",
                         params=params, magnitude=magnitude, sensor_kind=kind)
    raise KeyError((mech, target))


def mechanism_params(mech: str, dev: float) -> dict:
    if mech in FACTOR:
        return {FACTOR[mech]: 1.0 - dev}
    if mech == "condenser_fouling":
        return {"slope_factor": 1.0 + dev}
    assert mech == "tube_leak", mech
    return {"m_dot_design": dev}


def build_rows() -> list[dict]:
    rows = []
    for (mech, tgt), (thr, src) in P01.thresholds.items():
        if not src.startswith("swept"):
            continue
        for k in GRID_K:
            dev = thr * mult(k)
            rows.append(dict(group="mechanism", key=[mech, tgt], value=dev,
                             fault=_fault(mech, tgt, mechanism_params(mech, dev), dev)))
    for c in COLUMNS:
        d = detection_scale(c)
        for sign in (1.0, -1.0):
            for k in GRID_K:
                b = sign * d * mult(k)
                rows.append(dict(group="bias", key=[c, sign], value=abs(b),
                                 fault=_fault(SENSOR_MECHANISM, c, {"kind": "bias", "bias": b},
                                              b, "bias")))
        for f in NOISE_FACTORS:
            rows.append(dict(group="noise", key=[c], value=f,
                             fault=_fault(SENSOR_MECHANISM, c,
                                          {"kind": "noise_increase", "factor": f}, f,
                                          "noise_increase")))
    for n, r in enumerate(rows):
        r["item"] = ItemRecord(SCHEMA, "dev", INDEX0 + n, 0, SEED, FLAT, NO_OPERATOR, r["fault"])
    return rows


def threshold(points: list[tuple[float, bool]]) -> tuple[float | None, float | None, list]:
    """(threshold, largest undetected value below it, non-monotone detected values)."""
    pts = sorted(points)
    thr = None
    for v, det in reversed(pts):
        if not det:
            break
        thr = v
    below = [v for v, det in pts if thr is not None and v < thr]
    miss = max((v for v, det in pts if not det and (thr is None or v < thr)), default=None)
    nonmono = [v for v, det in pts if det and v in below]
    return thr, miss, nonmono


def _src(miss, nonmono) -> str:
    s = "swept, P02 rules" + (f" ({miss:.6g} not detected)" if miss is not None
                              else " (smallest swept)")
    return s + (f"; also detected below: {[float(f'{v:.6g}') for v in nonmono]}"
                if nonmono else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=str(REPO_ROOT / "cache" / "window_sweep"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "out" / "window_sweep" / "results.json"))
    ap.add_argument("--out-module", default=str(REPO_ROOT / "src" / "hotwell_bench"
                                                / "window_detectability.py"))
    a = ap.parse_args()
    env = check_environment()
    rows = build_rows()
    print(f"{len(rows)} items: " + str({g: sum(r['group'] == g for r in rows)
                                        for g in ('mechanism', 'bias', 'noise')}), flush=True)
    t0 = time.time()
    run_items([r["item"] for r in rows], a.cache, workers=a.workers, until=UNTIL)
    print(f"runs done in {time.time() - t0:.0f} s", flush=True)

    res = []
    for r in rows:
        it = r["item"]
        lab = label_runs(it, run_cached(it, "faulted", a.cache, until=UNTIL),
                         run_cached(it, "twin", a.cache, until=UNTIL), preset=P02)
        rec = dict(group=r["group"], key=r["key"], value=r["value"], item_id=it.item_id,
                   observable=bool(lab.observable),
                   delay=None if lab.t_det is None else lab.t_det - ONSET,
                   rule=lab.rule, instrument=lab.instrument,
                   rules_fired=sorted(k for k, v in lab.per_rule.items() if v),
                   strict_observable=bool(lab.strict["observable"]),
                   strict_delay=(None if lab.strict["t_det"] is None
                                 else lab.strict["t_det"] - ONSET))
        res.append(rec)

    def table(group, keyf):
        out = {}
        for rec in res:
            if rec["group"] == group:
                out.setdefault(keyf(rec), []).append(rec)
        return out

    print(f"\n{'row':34s} {'value':>10s} {'det':>4s} {'delay':>6s} {'rule/instrument':32s} "
          f"{'strict':>6s}")
    for rec in res:
        print(f"{rec['group'] + ' ' + ' '.join(map(str, rec['key'])):34s} {rec['value']:10.5g} "
              f"{'yes' if rec['observable'] else 'no':>4s} "
              f"{'-' if rec['delay'] is None else rec['delay']:>6} "
              f"{(rec['rule'] or '-') + '/' + (rec['instrument'] or '-'):32s} "
              f"{'-' if rec['strict_delay'] is None else rec['strict_delay']:>6}")

    # mechanism table
    new: dict[tuple[str, str], tuple[float | None, str]] = {}
    print("\nmechanism thresholds (0.1 -> P02):")
    for (mech, tgt), recs in table("mechanism", lambda r: tuple(r["key"])).items():
        thr, miss, nm = threshold([(r["value"], r["observable"]) for r in recs])
        sthr, _, _ = threshold([(r["value"], r["strict_observable"]) for r in recs])
        new[(mech, tgt)] = (thr, _src(miss, nm))
        print(f"  {mech:18s} {tgt:5s} {P01.thresholds[(mech, tgt)][0]:8.4g} -> "
              f"{'none' if thr is None else f'{thr:.4g}'}   strict "
              f"{'none' if sthr is None else f'{sthr:.4g}'}   {new[(mech, tgt)][1]}")
    out_thr = {}
    for (mech, tgt), (_thr0, src0) in P01.thresholds.items():
        if (mech, tgt) in new:
            out_thr[(mech, tgt)] = new[(mech, tgt)]
        elif src0.startswith("assumed: "):
            ref = src0.removeprefix("assumed: ").split(";")[0].split("/")[0].strip()
            refkey = next((k for k in new if k[0] == mech and k[1] == ref), None)
            if refkey is None:
                raise SystemExit(f"cannot resolve {src0!r} for {(mech, tgt)}")
            out_thr[(mech, tgt)] = (new[refkey][0], src0)
        else:
            raise SystemExit(f"unhandled 0.1 source {src0!r} for {(mech, tgt)}")

    bias, noise = {}, {}
    print("\nsensor thresholds (bias in D_i units; noise factor):")
    bt = table("bias", lambda r: r["key"][0])
    nt = table("noise", lambda r: r["key"][0])
    for c in COLUMNS:
        per_sign = []
        for sign in (1.0, -1.0):
            per_sign.append(threshold([(r["value"], r["observable"]) for r in bt[c]
                                       if r["key"][1] == sign]))
        thrs = [p[0] for p in per_sign]
        b = None if None in thrs else max(thrs)
        bsrc = "both signs: " + "; ".join(
            f"{s} {_src(p[1], p[2])}" for s, p in zip("+-", per_sign, strict=True))
        if b is not None and b < LSB[c]:
            # Alan, 2026-11-22: sub-LSB biases on idle, clipped instruments are caught only
            # by the variance rule through the quantized distribution; floor at 1 LSB.
            bsrc = f"floored at 1 LSB ({LSB[c]:g}; swept {b:.6g}); " + bsrc
            b = LSB[c]
        bias[c] = (b, bsrc)
        nthr, nmiss, nnm = threshold([(r["value"], r["observable"]) for r in nt[c]])
        noise[c] = (nthr, _src(nmiss, nnm))
        d = detection_scale(c)
        print(f"  {c:18s} D_i {d:9.4g}  bias {'none' if b is None else f'{b / d:.3g}':>6s}"
              f"  noise {'none' if nthr is None else f'{nthr:g}':>5s}  theta600/D_i "
              f"{WT.WINDOW_THETA[c] / d:.3g}")

    raw = json.dumps({"preset": P02.name, "window_stats_sha256": WT.STATS_SHA256,
                      "setup": {"onset": ONSET, "until": UNTIL, "seed": SEED,
                                "grid_k": list(GRID_K), "noise_factors": list(NOISE_FACTORS)},
                      "environment": env, "results": res}, indent=1, sort_keys=True)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(raw)
    sha = hashlib.sha256(raw.encode()).hexdigest()

    def fmt(v):
        return "None" if v is None else repr(float(v))

    def entry(key, v, s):
        parts = [s[i:i + 80] for i in range(0, len(s), 80)]   # implicit concatenation
        return ([f"    {key!r}: (", f"        {fmt(v)},"]
                + [f"        {x!r}" for x in parts[:-1]] + [f"        {parts[-1]!r},", "    ),"])

    lines = [
        '"""Step detectability thresholds under the P02 rules (GENERATED; do not edit by hand).',
        "",
        "Written by scripts/window_sweep.py; that script's docstring defines the setup,",
        "the grid and the threshold definition. Values are deviations from healthy in",
        "the units of presets.THRESHOLDS_V040 (mechanisms), engineering units (sensor",
        "bias) and noise-sigma factors (noise_increase).",
        '"""',
        "from __future__ import annotations",
        "",
        "from types import MappingProxyType",
        "",
        f"RESULTS_SHA256 = {sha!r}",
        f"PRESET = {P02.name!r}",
        f"WINDOW_STATS_SHA256 = {WT.STATS_SHA256!r}",
        "",
        "THRESHOLDS_V020 = MappingProxyType({",
        *[x for k, (v, s) in out_thr.items() for x in entry(k, v, s)],
        "})",
        "",
        "SENSOR_BIAS_THRESHOLD = MappingProxyType({",
        *[x for c, (v, s) in bias.items() for x in entry(c, v, s)],
        "})",
        "",
        "SENSOR_NOISE_THRESHOLD = MappingProxyType({",
        *[x for c, (v, s) in noise.items() for x in entry(c, v, s)],
        "})",
        "",
    ]
    Path(a.out_module).write_text("\n".join(lines))
    print(f"\nwrote {a.out_json} (sha256 {sha[:16]}…) and {a.out_module}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
