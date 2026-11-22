#!/usr/bin/env python3
"""Empirical thresholds for the windowed level rules, from replay-mode seed-pair null days.

Input: the statistics file written by `scripts/null_pairs.py --mode replay`. For pair p,
instrument i and window W, M[p, W, i] is the largest θ that the windowed rule (window W,
sustain S) would exceed somewhere in that null day, computed on A − B.

Reference variability. An analyst without a twin must tell the fault effect apart from
one day's deviation from its expected behaviour. The difference of two independent
days has √2 times that spread, so the statistic used is M / √2 (REFERENCE_SCALE).
z is scale-invariant, so false-positive counts do not change, except where the
LSB floor (which is not scaled) binds. There it can only remove fires: cp2_current
at W = 600 is floored, which removes the 3 old-validation fires seen without the
1/√2 scale.

Two threshold sets (PLAN-bench-0.2.md §2.1; Alan's labelling decision of 2026-11-21):
- WINDOW: the lenient rule, which takes part in observability and t_det.
  - It uses one window, WINDOW_S = 600.
  - s_i = median over calibration pairs of M/√2.
  - z = max over calibration pairs and instruments of (M/√2)/s_i.
  - θ_i = max(z·s_i, LSB_i).
- STRICT: the strict per-instrument stratum (recorded, never used to drop items).
  - It uses windows STRICT_WINDOWS_S = (10, 60, 600, 3600), with s_{W,i} per
    (window, instrument).
  - z is joint, the maximum over calibration pairs, windows and instruments.
  - θ_{W,i} = max(z·s_{W,i}, LSB_i).

Guarantee: if null days are exchangeable, a new null day fires a rule set with
probability ≤ 1/(n + 1), where n = 100 calibration pairs. That is 0.99%. The LSB floor
only lowers it.

Validation sets, both reported:
- "validation" (odd k < 200): looked at during the 2026-11-19/21 diagnoses.
- "fresh_validation" (k ≥ 200): generated after this script was committed.

GATE (pre-registered): each rule set may fire on at most 3 of the fresh validation
pairs. Under a true rate of 1%, P(X ≥ 4 of 100) ≈ 1.8%.

Output: a Python module (default src/hotwell_bench/window_thresholds.py) and a report.

Usage: python scripts/window_thresholds.py STATS_JSON [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from hotwell.sensors import OBSERVATION_SPEC

from hotwell_bench.run import COLUMNS
from hotwell_bench.secrecy import REPO_ROOT

REFERENCE_SCALE = 1.0 / math.sqrt(2.0)
WINDOW_S = 600
STRICT_WINDOWS_S = (10, 60, 600, 3600)
MAX_FRESH_FIRES = 3
LSB = np.array([{x.name: x.resolution for x in OBSERVATION_SPEC}[c] for c in COLUMNS])


def _M(pairs: dict, keys: list[str], windows) -> np.ndarray:
    """(pairs, windows, instruments) reference statistic M / √2."""
    return REFERENCE_SCALE * np.array([[[pairs[k][str(w)]["M"][c] for c in COLUMNS]
                                        for w in windows] for k in keys])


def fit(cal: np.ndarray) -> tuple[np.ndarray, float, tuple, np.ndarray]:
    """Scale, joint z, argmax (pair, window, instrument) and floored θ (windows × inst)."""
    scale = np.median(cal, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(scale > 0, cal / scale, 0.0)
    z = float(Z.max())
    arg = np.unravel_index(int(Z.argmax()), Z.shape)
    return scale, z, arg, np.maximum(z * scale, LSB)


def fires(M: np.ndarray, theta: np.ndarray, keys: list[str], windows) -> list:
    out = []
    for n, k in enumerate(keys):
        hit = [(w, c) for j, w in enumerate(windows) for i, c in enumerate(COLUMNS)
               if M[n, j, i] > theta[j, i]]
        if hit:
            out.append((k, hit))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--out", default=str(REPO_ROOT / "src" / "hotwell_bench"
                                         / "window_thresholds.py"))
    a = ap.parse_args()
    raw = Path(a.stats).read_bytes()
    rec = json.loads(raw)
    if rec.get("mode") != "replay":
        raise SystemExit("needs replay-mode statistics (scripts/null_pairs.py --mode replay)")
    pairs = rec["pairs"]
    by_role = {r: sorted((k for k in pairs if pairs[k]["role"] == r), key=int)
               for r in ("calibration", "validation", "fresh_validation")}
    print({r: len(v) for r, v in by_role.items()})
    out_vals = {}
    ok = True
    for name, windows in (("WINDOW", (WINDOW_S,)), ("STRICT", STRICT_WINDOWS_S)):
        cal = _M(pairs, by_role["calibration"], windows)
        scale, z, (p, j, i), theta = fit(cal)
        print(f"\n{name}: windows {windows}, sustain {rec['sustain_s']} s; z = {z:.3f} "
              f"(pair {by_role['calibration'][p]}, W {windows[j]}, {COLUMNS[i]})")
        for jw, w in enumerate(windows):
            fl = [c for ii, c in enumerate(COLUMNS) if z * scale[jw, ii] < LSB[ii]]
            print(f"  W {w}: LSB floor on {fl}")
        counts = {}
        for role in ("calibration", "validation", "fresh_validation"):
            keys = by_role[role]
            f = fires(_M(pairs, keys, windows), theta, keys, windows) if keys else []
            counts[role] = len(f)
            print(f"  {role}: {len(f)}/{len(keys)} pairs fire {f}")
        gate = by_role["fresh_validation"] and counts["fresh_validation"] <= MAX_FRESH_FIRES
        ok &= bool(gate)
        print(f"  GATE {'PASS' if gate else 'FAIL'} (fresh validation, at most "
              f"{MAX_FRESH_FIRES} of {len(by_role['fresh_validation'])})")
        out_vals[name] = (windows, z, theta, counts)

    wz, zw, thw, cw = out_vals["WINDOW"]
    sz, zs, ths, cs = out_vals["STRICT"]
    lines = [
        '"""Empirical windowed-rule thresholds (GENERATED; do not edit by hand).',
        "",
        "Written by scripts/window_thresholds.py from the replay-mode scripts/null_pairs.py",
        "statistics below; that script's docstring defines the estimators.",
        '"""',
        "from __future__ import annotations",
        "",
        f"STATS_SHA256 = {hashlib.sha256(raw).hexdigest()!r}",
        f"CAL_MASTER = {rec['cal_master']}",
        f"BACKGROUND_PRESET = {rec['background_preset']!r}",
        f"SUSTAIN_S = {rec['sustain_s']}",
        f"REFERENCE_SCALE = {REFERENCE_SCALE!r}  # 1/sqrt(2): single-day reference",
        f"N_PAIRS = {dict((r, len(v)) for r, v in by_role.items())!r}",
        "",
        f"WINDOW_S = {wz[0]}",
        f"WINDOW_Z = {zw!r}",
        f"WINDOW_FIRES = {cw!r}",
        "WINDOW_THETA = {",
        *[f"    {c!r}: {float(t)!r}," for c, t in zip(COLUMNS, thw[0], strict=True)],
        "}",
        "",
        f"STRICT_WINDOWS_S = {tuple(sz)!r}",
        f"STRICT_Z = {zs!r}",
        f"STRICT_FIRES = {cs!r}",
        "STRICT_THETA = {",
        *[f"    {w}: {{" + ", ".join(f"{c!r}: {float(t)!r}" for c, t in
                                     zip(COLUMNS, ths[j], strict=True)) + "},"
          for j, w in enumerate(sz)],
        "}",
        "",
    ]
    Path(a.out).write_text("\n".join(lines))
    print(f"\nwrote {a.out}; overall GATE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
