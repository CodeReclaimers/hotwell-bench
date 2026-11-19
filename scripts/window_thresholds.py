#!/usr/bin/env python3
"""Per-instrument thresholds θ_i for the windowed level rule, from seed-pair null days.

Input: the statistics file written by scripts/null_pairs.py. For pair p and instrument i,
M[p, i] is the largest θ the windowed rule (window W, sustain S) would exceed somewhere
in that null day. So the rule fires on pair p iff M[p, i] > θ_i for some i.

Estimator (fixed before the null-pair data was seen, commit of this script):
1. Scale: s_i = median over calibration pairs of M[p, i], the typical daily maximum.
   Normalizing by the daily maximum rather than the bulk spread balances instruments
   with heavy tails, such as pump currents around swaps, against light-tailed ones.
2. Multiplier: z = max over calibration pairs p and instruments i of M[p, i] / s_i.
3. θ_i = max(z · s_i, LSB_i), where LSB_i is the instrument's resolution. A sustained
   mean difference below one LSB is not taken as evidence even on an instrument that
   never varies.

Guarantee: if null days are exchangeable, a new null day's largest normalized
statistic exceeds the maximum of n calibration values with probability 1/(n + 1). With
n = 100 that is about 0.99%, inside the plan's ≤ 1% per-day target. The LSB floor only
raises θ, which lowers the probability further.

Validation (pre-registered): the odd pairs, never used above, are counted.
- PASS if at most 3 of 100 fire. Under a true rate of 1%, P(X ≥ 4) ≈ 1.8%, so 4 or more
  indicates the calibration days are not representative.

Output:
- a Python module (default src/hotwell_bench/window_thresholds.py) holding θ_i and the
  provenance;
- a report on stdout.

Usage: python scripts/window_thresholds.py STATS_JSON [--window 600] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from hotwell.sensors import OBSERVATION_SPEC

from hotwell_bench.run import COLUMNS
from hotwell_bench.secrecy import REPO_ROOT

MAX_VALIDATION_FIRES = 3          # of 100 (see the module docstring)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--out", default=str(REPO_ROOT / "src" / "hotwell_bench"
                                         / "window_thresholds.py"))
    a = ap.parse_args()
    raw = Path(a.stats).read_bytes()
    rec = json.loads(raw)
    w, s = str(a.window), rec["sustain_s"]
    lsb = np.array([{x.name: x.resolution for x in OBSERVATION_SPEC}[c] for c in COLUMNS])
    pairs = rec["pairs"]
    cal = sorted((k for k in pairs if pairs[k]["role"] == "calibration"), key=int)
    val = sorted((k for k in pairs if pairs[k]["role"] == "validation"), key=int)

    def M(keys):
        return np.array([[pairs[k][w]["M"][c] for c in COLUMNS] for k in keys])

    Mc, Mv = M(cal), M(val)
    scale = np.median(Mc, axis=0)
    if np.any(scale <= 0):
        zero = [c for c, x in zip(COLUMNS, scale, strict=True) if x <= 0]
        print(f"instruments with zero typical daily maximum (LSB floor applies): {zero}")
    with np.errstate(divide="ignore", invalid="ignore"):
        Zc = np.where(scale > 0, Mc / scale, 0.0)
    z = float(Zc.max())
    pz, iz = np.unravel_index(int(Zc.argmax()), Zc.shape)
    theta = np.maximum(z * scale, lsb)
    floored = [c for c, t, f in zip(COLUMNS, z * scale, lsb, strict=True) if t < f]

    fires = [(k, [c for c, m, t in zip(COLUMNS, Mv[n], theta, strict=True) if m > t])
             for n, k in enumerate(val)]
    fires = [(k, cs) for k, cs in fires if cs]
    cal_fires = sum(bool(np.any(Mc[n] > theta)) for n in range(len(cal)))
    ok = len(fires) <= MAX_VALIDATION_FIRES

    print(f"window {w} s, sustain {s} s; calibration pairs {len(cal)}, validation {len(val)}")
    print(f"z = {z:.3f}, set by pair {cal[pz]} instrument {COLUMNS[iz]}")
    print("per-instrument normalized daily maximum, calibration: median / p90 / max")
    q = np.quantile(Zc, [0.5, 0.9, 1.0], axis=0)
    for n, c in enumerate(COLUMNS):
        print(f"  {c:18s} s_i {scale[n]:11.4g}  theta {theta[n]:11.4g}  LSB {lsb[n]:9.3g}  "
              f"{q[0, n]:5.2f} / {q[1, n]:5.2f} / {q[2, n]:5.2f}")
    print(f"LSB floor applied: {floored}")
    print(f"calibration pairs firing: {cal_fires}/{len(cal)}")
    print(f"validation pairs firing: {len(fires)}/{len(val)} {fires}")
    print(f"VALIDATION {'PASS' if ok else 'FAIL'} (at most {MAX_VALIDATION_FIRES} allowed)")

    lines = [
        '"""Windowed level rule thresholds θ_i (GENERATED; do not edit by hand).',
        "",
        "Written by scripts/window_thresholds.py from the scripts/null_pairs.py statistics",
        "file below; the script's docstring defines the estimator.",
        '"""',
        "from __future__ import annotations",
        "",
        f"STATS_SHA256 = {hashlib.sha256(raw).hexdigest()!r}",
        f"CAL_MASTER = {rec['cal_master']}",
        f"BACKGROUND_PRESET = {rec['background_preset']!r}",
        f"WINDOW_S = {int(w)}",
        f"SUSTAIN_S = {s}",
        f"N_CALIBRATION = {len(cal)}",
        f"N_VALIDATION = {len(val)}",
        f"Z = {z!r}",
        f"VALIDATION_FIRES = {len(fires)}",
        f"LSB_FLOORED = {tuple(floored)!r}",
        "THETA = {",
        *[f"    {c!r}: {float(t)!r}," for c, t in zip(COLUMNS, theta, strict=True)],
        "}",
        "",
    ]
    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
